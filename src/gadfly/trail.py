"""Per-session conceptual breadcrumb trail of the main agent.

The journal is *state* — where the agent is, what workstreams it owns,
what its drift looks like. The trail is *path* — the ordered, conceptual
sequence of steps the agent has taken. Different lens.

The trail exists because a class of laziness is invisible from a single
action and only visible across multiple actions:

  - hardcoded_instance : N consecutive instance-level patches when a
                         class slot exists
  - premature_ceiling  : climbed instance→class but stopped before
                         the architecture slot
  - wrong_layer        : fix lands in the wrong architectural bucket
  - rule_skip          : repo has a rule visible in CLAUDE.md / active
                         plan / surrounding code, the agent bypassed it
  - incomplete_coverage: closed one branch of N obviously-equivalent
  - recon_as_work      : N consecutive reads / greps with no commit
  - rationalization    : re-explains a prior wrong-level fix instead of
                         correcting it

Each PostToolUse triggers one LLM call. The model decides:

  (a) does this action ADVANCE the trail (new breadcrumb) or REPEAT the
      last one (no change)?
  (b) is a longitudinal LAZINESS PATTERN visible across the last few
      breadcrumbs (Einstein-three-level violation)?

If (a) → append breadcrumb. If (b) → emit a DriftFlag. The reasoning the
model writes is AUDIT ONLY — on flag delivery, the hook substitutes a
canonical Socratic question from `prompts.TRAIL_DRIFT_QUESTIONS` so the
supervised (smarter) agent gets a fixed, well-crafted prompt instead of
the maintainer's possibly-shaky prose.

Persistence mirrors `journal.py`:
  - current state:    ~/.claude/gadfly/trail/<session_id>.json
  - snapshots (sha):  ~/.claude/gadfly/trails/<sha>.json
  - audit log:        log/<session_id>.jsonl, type="trail_update"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ThinkingConfigDisabled,
    create_sdk_mcp_server,
    query,
    tool,
)

from . import log as audit_log
from .prompts import (
    EVALUATE_STOP_DESCRIPTION,
    EVALUATE_STOP_INPUT_SCHEMA,
    EVALUATE_STOP_JSON_SCHEMA,
    STOP_RUBRIC_SYSTEM_PROMPT,
    TRAIL_DRIFT_QUESTIONS,
    TRAIL_STOP_QUESTION,
    TRAIL_UPDATE_SYSTEM_PROMPT,
    UPDATE_TRAIL_DESCRIPTION,
    UPDATE_TRAIL_INPUT_SCHEMA,
    UPDATE_TRAIL_JSON_SCHEMA,  # noqa: F401  — re-exported for backends


)

# --- Schema -----------------------------------------------------------------

SCHEMA_VERSION = 1

# Hard caps to keep trail JSON bounded over long sessions.
MAX_BREADCRUMBS = 30
MAX_DRIFT_FLAGS = 40
PROMPT_WINDOW = 10  # how many recent breadcrumbs render into the prompt
DRIFT_PROMPT_WINDOW = 5  # how many recent drift_flags render into the prompt
DRIFT_SUPPRESS_WINDOW = 3  # K-consecutive-same-kind suppression rule
STALL_THRESHOLD = 5  # consecutive non-advancing actions → force unclear crumb
MAX_BREADCRUMB_TEXT = 200
MAX_DRIFT_REASONING = 600
MAX_CITED = 6

# Abstraction levels and drift kinds. Mirror the prompt taxonomy.
_VALID_LEVELS = frozenset({
    "instance", "class", "architecture", "rationalization", "unclear",
})
_VALID_DRIFT_KINDS = frozenset({
    "hardcoded_instance", "premature_ceiling", "wrong_layer", "rule_skip",
    "incomplete_coverage", "recon_as_work", "rationalization", "other",
})


def _coerce_level(value: Any) -> str:
    """Map raw model output to a valid AbstractionLevel; default 'unclear'."""
    if isinstance(value, str) and value in _VALID_LEVELS:
        return value
    return "unclear"


def _coerce_drift_kind(value: Any) -> str:
    """Map raw model output to a valid DriftKind; default 'other'."""
    if isinstance(value, str) and value in _VALID_DRIFT_KINDS:
        return value
    return "other"


def _clip(text: Any, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[+{len(s) - limit}c]"


@dataclass(frozen=True)
class Breadcrumb:
    action_index: int
    breadcrumb_text: str
    abstraction_level: str
    action_summary: str
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Breadcrumb":
        return cls(
            action_index=int(d.get("action_index") or 0),
            breadcrumb_text=str(d.get("breadcrumb_text") or ""),
            abstraction_level=_coerce_level(d.get("abstraction_level")),
            action_summary=str(d.get("action_summary") or ""),
            ts=float(d.get("ts") or 0.0),
        )


@dataclass(frozen=True)
class DriftFlag:
    action_index: int
    drift_kind: str
    drift_reasoning: str
    cited_action_indexes: list[int]
    suppressed: bool
    delivered_to_agent: bool
    ts: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cited_action_indexes"] = list(self.cited_action_indexes)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DriftFlag":
        cited_raw = d.get("cited_action_indexes") or []
        cited = []
        for x in cited_raw:
            try:
                cited.append(int(x))
            except (TypeError, ValueError):
                continue
        return cls(
            action_index=int(d.get("action_index") or 0),
            drift_kind=_coerce_drift_kind(d.get("drift_kind")),
            drift_reasoning=str(d.get("drift_reasoning") or ""),
            cited_action_indexes=cited,
            suppressed=bool(d.get("suppressed") or False),
            delivered_to_agent=bool(d.get("delivered_to_agent") or False),
            ts=float(d.get("ts") or 0.0),
        )


@dataclass
class Trail:
    breadcrumbs: list[Breadcrumb] = field(default_factory=list)
    drift_flags: list[DriftFlag] = field(default_factory=list)
    action_index: int = 0
    prompt_sha: str = ""
    schema_version: int = SCHEMA_VERSION
    # Consecutive PostToolUse events where the model said advances_trail=False.
    # Resets to 0 on every actual advance. When it hits STALL_THRESHOLD the
    # trail forces an `unclear` breadcrumb so detection doesn't go blind.
    non_advance_streak: int = 0
    # Snapshot of journal.root_goal at last update — used to detect a user
    # redirect, which resets the K-window suppression rule. Empty when the
    # journal was unavailable.
    last_root_goal: str = ""
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "breadcrumbs": [b.to_dict() for b in self.breadcrumbs],
            "drift_flags": [f.to_dict() for f in self.drift_flags],
            "action_index": self.action_index,
            "prompt_sha": self.prompt_sha,
            "schema_version": self.schema_version,
            "non_advance_streak": self.non_advance_streak,
            "last_root_goal": self.last_root_goal,
            "ts": self.ts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trail":
        bc_raw = d.get("breadcrumbs") or []
        breadcrumbs = [
            Breadcrumb.from_dict(b) for b in bc_raw if isinstance(b, dict)
        ]
        df_raw = d.get("drift_flags") or []
        drift_flags = [
            DriftFlag.from_dict(f) for f in df_raw if isinstance(f, dict)
        ]
        return cls(
            breadcrumbs=breadcrumbs,
            drift_flags=drift_flags,
            action_index=int(d.get("action_index") or 0),
            prompt_sha=str(d.get("prompt_sha") or ""),
            schema_version=int(d.get("schema_version") or SCHEMA_VERSION),
            non_advance_streak=int(d.get("non_advance_streak") or 0),
            last_root_goal=str(d.get("last_root_goal") or ""),
            ts=float(d.get("ts") or 0.0),
        )


def _system_prompt_sha() -> str:
    return hashlib.sha256(
        TRAIL_UPDATE_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()[:16]


def empty_trail() -> Trail:
    return Trail(prompt_sha=_system_prompt_sha())


# --- Persistence ------------------------------------------------------------


def _current_dir() -> Path:
    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "trail"
    return Path.home() / ".claude" / "gadfly" / "trail"


def _current_path(session_id: str) -> Path:
    safe = (session_id or "unknown").replace("/", "_")
    return _current_dir() / f"{safe}.json"


def load_current(session_id: str) -> Trail | None:
    """Load the per-session current trail. Returns None when missing or
    invalidated by a schema/prompt change."""
    try:
        p = _current_path(session_id)
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        t = Trail.from_dict(data)
    except Exception:
        return None
    if t.schema_version != SCHEMA_VERSION:
        return None
    if t.prompt_sha and t.prompt_sha != _system_prompt_sha():
        # Prompt evolved — fall back to a fresh trail so the new rubric
        # applies from scratch (same invariant as journal).
        return None
    return t


def save_current(session_id: str, trail: Trail) -> None:
    try:
        d = _current_dir()
        d.mkdir(parents=True, exist_ok=True)
        _current_path(session_id).write_text(trail.to_json(), encoding="utf-8")
    except Exception:
        pass


# --- User-message builder ---------------------------------------------------


def _render_trail_block(t: Trail) -> str:
    lines: list[str] = []
    lines.append(f"## Current trail (action_index={t.action_index})")
    if not t.breadcrumbs:
        lines.append("breadcrumbs: (empty — this is the first action)")
    else:
        lines.append(
            f"breadcrumbs (last {min(PROMPT_WINDOW, len(t.breadcrumbs))} of "
            f"{len(t.breadcrumbs)}, chronological):"
        )
        for b in t.breadcrumbs[-PROMPT_WINDOW:]:
            lines.append(
                f"  [{b.abstraction_level}] {_clip(b.breadcrumb_text, 160)} "
                f"← #{b.action_index}"
            )
            if b.action_summary:
                lines.append(
                    f"    action: {_clip(b.action_summary, 200)}"
                )
    if t.drift_flags:
        lines.append(
            f"\ndrift history (last {min(DRIFT_PROMPT_WINDOW, len(t.drift_flags))} "
            f"of {len(t.drift_flags)}, chronological):"
        )
        for f in t.drift_flags[-DRIFT_PROMPT_WINDOW:]:
            sup = " [suppressed]" if f.suppressed else ""
            deliv = " [delivered]" if f.delivered_to_agent else ""
            cited = ", ".join(f"#{i}" for i in f.cited_action_indexes[:MAX_CITED])
            lines.append(
                f"  #{f.action_index} {f.drift_kind}{sup}{deliv}  "
                f"cited: {cited}  reason: {_clip(f.drift_reasoning, 200)}"
            )
    if t.non_advance_streak:
        lines.append(
            f"\nnon_advance_streak: {t.non_advance_streak} "
            f"(STALL_THRESHOLD={STALL_THRESHOLD})"
        )
    return "\n".join(lines)


def build_update_user_message(
    *,
    trail: Trail,
    action_summary: str | None,
    assistant_reasoning: str | None,
    latest_user_message: str | None,
    journal_root_goal: str | None,
    json_mode: bool = False,
) -> str:
    """Assemble the maintainer prompt.

    `json_mode=True` appends the OpenAI-JSON output-format suffix (the
    openai_json backend uses response_format=json_object and the model
    needs explicit shape instructions). Tool-call backends ignore it.
    """
    from .prompts import _TRAIL_OUTPUT_FORMAT_SUFFIX

    parts: list[str] = [_render_trail_block(trail)]

    if journal_root_goal:
        parts.append(
            "## Root goal (from journal)\n"
            + _clip(journal_root_goal, 800)
        )

    if latest_user_message:
        parts.append(
            "## Most recent user message (verbatim)\n"
            + _clip(latest_user_message, 2000)
        )

    if assistant_reasoning:
        parts.append(
            "## Agent's reasoning immediately before this action\n"
            + _clip(assistant_reasoning, 1500)
        )

    if action_summary:
        parts.append(
            f"## New action (#{trail.action_index + 1})\n"
            + _clip(action_summary, 1500)
        )

    parts.append(
        "## Task\n"
        "Apply the rubric in the system prompt. Decide:\n"
        " (1) `advances_trail` — did the conceptual stance change "
        "(new file area / new abstraction level / new sub-goal)? If the "
        "new action is a mechanical re-edit at the same level / locus "
        "as the last breadcrumb, set false.\n"
        " (2) `drift_detected` — does the trail (not the action alone) "
        "show one of the 7 longitudinal laziness patterns? If yes, "
        "pick the SINGLE most-applicable `drift_kind`, set "
        "`drift_reasoning` (audit-only English text ≤600c), and "
        "populate `cited_action_indexes` with the specific breadcrumb "
        "indexes that constitute the pattern (≥1 entry).\n"
        "Call `update_trail` exactly once. Default both booleans to "
        "false when uncertain."
    )
    msg = "\n\n".join(parts)
    if json_mode:
        msg += _TRAIL_OUTPUT_FORMAT_SUFFIX
    return msg


# --- LLM call ---------------------------------------------------------------


@dataclass
class _Captured:
    payload: dict[str, Any] | None = None


def _build_update_tool(captured: _Captured):
    @tool("update_trail", UPDATE_TRAIL_DESCRIPTION, UPDATE_TRAIL_INPUT_SCHEMA)
    async def update_trail(args: dict[str, Any]) -> dict[str, Any]:
        captured.payload = args
        return {"content": [{"type": "text", "text": "trail recorded"}]}

    return update_trail


def _build_options(captured: _Captured, model: str) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly_trail",
        "1.0.0",
        [_build_update_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=TRAIL_UPDATE_SYSTEM_PROMPT,
        mcp_servers={"gadfly_trail": server},
        allowed_tools=["mcp__gadfly_trail__update_trail"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        # Recursion guard: empty settings string blocks the inner CLI from
        # inheriting hooks. Same load-bearing pattern as journal/watchdog.
        settings="{}",
        thinking=ThinkingConfigDisabled(type="disabled"),
        max_turns=2,
        env={"GADFLY_INTERNAL": "1"},
    )


RunQuery = Callable[[str, ClaudeAgentOptions], Awaitable[None]]


async def _default_run_query(prompt: str, options: ClaudeAgentOptions) -> None:
    async for _ in query(prompt=prompt, options=options):
        pass


DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_TIMEOUT_S = 30.0


# --- Backend selection (env-driven, mirrors watchdog._default_backend_from_env) -


def _default_backend_from_env():
    """Build an OpenAI-compat backend when env vars say so, else None.

    Trail respects the same GADFLY_BACKEND / GADFLY_BASE_URL /
    GADFLY_MODEL / GADFLY_API_KEY_ENV / GADFLY_EXTRA_HEADERS shaped
    config as the watchdog. Returns None when env config is incomplete
    — caller falls back to ClaudeSDKBackend (subscription Haiku).
    """
    name = os.environ.get("GADFLY_BACKEND", "").strip()
    if name not in ("openai_compat", "openai_json"):
        return None
    base_url = os.environ.get("GADFLY_BASE_URL", "").strip()
    if not base_url:
        return None
    key_env = os.environ.get("GADFLY_API_KEY_ENV", "OPENROUTER_API_KEY").strip()
    api_key = (
        os.environ.get(key_env, "")
        if key_env and key_env != "NONE"
        else ""
    )
    extra: dict[str, str] | None = None
    raw = os.environ.get("GADFLY_EXTRA_HEADERS", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                extra = parsed
        except Exception:
            extra = None
    if name == "openai_json":
        from .backends.openai_json import OpenAIJsonBackend
        return OpenAIJsonBackend(
            base_url=base_url, api_key=api_key, extra_headers=extra
        )
    from .backends.openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(
        base_url=base_url, api_key=api_key, extra_headers=extra
    )


# --- Update logic -----------------------------------------------------------


@dataclass
class TrailUpdateResult:
    trail: Trail
    drift_flag: DriftFlag | None
    error: str | None
    skipped_reason: str | None
    latency_ms: float
    diff_summary: list[str]


async def _update_async_sdk(
    *,
    user_message: str,
    model: str,
    timeout_s: float,
    run_query: RunQuery,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the Claude-SDK backend with a forced tool call."""
    captured = _Captured()
    options = _build_options(captured, model)
    try:
        await asyncio.wait_for(
            run_query(user_message, options), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return None, f"timeout after {timeout_s}s"
    except FileNotFoundError as exc:
        return None, f"claude CLI not found: {exc!s}"
    except Exception as exc:
        return None, f"agent-sdk error: {exc!r}"
    if captured.payload is None:
        return None, "Haiku did not call update_trail"
    return captured.payload, None


async def _update_async_oai(
    *,
    backend,
    user_message: str,
    model: str,
    timeout_s: float,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call an OpenAI-compatible backend (tool-call or JSON mode)."""
    try:
        br = await backend.evaluate(
            system_prompt=TRAIL_UPDATE_SYSTEM_PROMPT,
            user_message=user_message,
            model=model,
            tool_name="update_trail",
            tool_description=UPDATE_TRAIL_DESCRIPTION,
            tool_parameters=UPDATE_TRAIL_JSON_SCHEMA,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return None, f"backend error: {exc!r}"
    if br.verdict_args is None:
        return None, br.error or "no payload"
    return dict(br.verdict_args), None


def _drift_was_recent_for_kind(trail: Trail, kind: str, window: int) -> bool:
    """K-consecutive-same-kind rule: scan the most recent `window` drift
    flags. If ANY of them matches `kind`, suppress the current one. The
    rule explicitly defends against the v1 echo chamber.
    """
    recent = trail.drift_flags[-window:]
    return any(f.drift_kind == kind for f in recent)


def _diff_summary(old: Trail, new: Trail) -> list[str]:
    out: list[str] = []
    added_b = len(new.breadcrumbs) - len(old.breadcrumbs)
    if added_b > 0:
        last = new.breadcrumbs[-1]
        out.append(
            f"breadcrumb +1 [{last.abstraction_level}] "
            f"#{last.action_index}"
        )
    added_f = len(new.drift_flags) - len(old.drift_flags)
    if added_f > 0:
        last_f = new.drift_flags[-1]
        sup = " [suppressed]" if last_f.suppressed else ""
        deliv = " [delivered]" if last_f.delivered_to_agent else ""
        out.append(
            f"drift +1 {last_f.drift_kind}{sup}{deliv} #{last_f.action_index}"
        )
    if new.non_advance_streak != old.non_advance_streak:
        out.append(
            f"non_advance_streak {old.non_advance_streak}→{new.non_advance_streak}"
        )
    if new.last_root_goal != old.last_root_goal:
        out.append("root_goal changed (suppression window reset)")
    return out


async def update_for_action_async(
    *,
    session_id: str,
    action_index: int,
    action_summary: str,
    assistant_reasoning: str | None,
    latest_user_message: str | None,
    journal_root_goal: str | None,
    cwd: str | None = None,
    model: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    backend: Any = None,
    run_query: RunQuery | None = None,
) -> TrailUpdateResult:
    """Async core. Callers already in an event loop (run_corpus.py,
    pytest-asyncio tests) use this directly via `await`. The sync wrapper
    below (`update_for_action`) is for the production hook which runs
    sync top-level.

    Never raises — every failure ends in an audit-log entry, the trail
    being saved unchanged, and the caller moving on.
    """
    t0 = time.perf_counter()
    base = load_current(session_id) or empty_trail()
    prior_sha_str: str | None = None
    try:
        prior_sha_str = audit_log.ensure_trail_snapshot(base.to_json())
    except Exception:
        prior_sha_str = None

    # Window reset: when the journal root_goal has changed since last
    # update, the user redirected. Past drift_flags become moot for the
    # suppression rule — we still keep them for audit, but signal the
    # window has been refreshed by re-stamping last_root_goal BEFORE
    # the K-lookup.
    redirect = (
        journal_root_goal is not None
        and bool(journal_root_goal)
        and base.last_root_goal
        and journal_root_goal.strip() != base.last_root_goal.strip()
    )

    payload, error = await _call_model_async(
        trail=base,
        action_summary=action_summary,
        assistant_reasoning=assistant_reasoning,
        latest_user_message=latest_user_message,
        journal_root_goal=journal_root_goal,
        model=model,
        timeout_s=timeout_s,
        backend=backend,
        run_query=run_query,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    new_trail, drift, skipped_reason = _apply_payload(
        base=base,
        payload=payload,
        action_index=action_index,
        action_summary=action_summary,
        journal_root_goal=journal_root_goal,
        redirect=redirect,
    )

    # Persist + write audit event before returning. Trail save is
    # best-effort: if it fails the next hook iteration starts from
    # an older state, no crash.
    try:
        save_current(session_id, new_trail)
    except Exception:
        pass

    try:
        new_sha = audit_log.ensure_trail_snapshot(new_trail.to_json())
    except Exception:
        new_sha = None

    drift_kind_log = drift.drift_kind if drift is not None else None
    suppressed = bool(drift.suppressed) if drift is not None else False
    delivered = bool(drift.delivered_to_agent) if drift is not None else False
    advances = (
        len(new_trail.breadcrumbs) > len(base.breadcrumbs)
        and (
            not new_trail.breadcrumbs
            or new_trail.breadcrumbs[-1].action_index == action_index
        )
    )
    try:
        audit_log.append_trail_event(
            session_id=session_id,
            action_index=action_index,
            prior_trail_sha=prior_sha_str,
            new_trail_sha=new_sha,
            advances_trail=advances,
            drift_detected=drift is not None,
            drift_kind=drift_kind_log,
            suppressed=suppressed,
            delivered_to_agent=delivered,
            diff_summary=_diff_summary(base, new_trail),
            latency_ms=latency_ms,
            error=error,
            skipped_reason=skipped_reason,
        )
    except Exception:
        pass

    return TrailUpdateResult(
        trail=new_trail,
        drift_flag=drift,
        error=error,
        skipped_reason=skipped_reason,
        latency_ms=latency_ms,
        diff_summary=_diff_summary(base, new_trail),
    )


async def _call_model_async(
    *,
    trail: Trail,
    action_summary: str,
    assistant_reasoning: str | None,
    latest_user_message: str | None,
    journal_root_goal: str | None,
    model: str | None,
    timeout_s: float,
    backend: Any,
    run_query: RunQuery | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the maintainer LLM call. Returns (payload, error). Selects
    backend via env vars unless one was passed explicitly. JSON-mode
    backend gets the format-suffix; tool-call backends do not.
    """
    json_mode = False
    eff_backend = backend
    # Auto-discover from env ONLY when neither backend nor run_query was
    # explicitly passed. The run_query parameter is the test-side hook
    # used to mock the SDK call — env-driven backend selection would
    # bypass it and silently route to a live OpenAI-compat endpoint.
    if eff_backend is None and run_query is None:
        eff_backend = _default_backend_from_env()
    if eff_backend is not None and type(eff_backend).__name__.startswith(
        "OpenAIJson"
    ):
        json_mode = True

    user_message = build_update_user_message(
        trail=trail,
        action_summary=action_summary,
        assistant_reasoning=assistant_reasoning,
        latest_user_message=latest_user_message,
        journal_root_goal=journal_root_goal,
        json_mode=json_mode,
    )

    eff_model = model or os.environ.get("GADFLY_MODEL") or DEFAULT_MODEL

    try:
        if eff_backend is not None:
            return await _update_async_oai(
                backend=eff_backend,
                user_message=user_message,
                model=eff_model,
                timeout_s=timeout_s,
            )
        rq = run_query or _default_run_query
        return await _update_async_sdk(
            user_message=user_message,
            model=eff_model,
            timeout_s=timeout_s,
            run_query=rq,
        )
    except Exception as exc:
        return None, f"call error: {exc!r}"


def update_for_action(
    *,
    session_id: str,
    action_index: int,
    action_summary: str,
    assistant_reasoning: str | None,
    latest_user_message: str | None,
    journal_root_goal: str | None,
    cwd: str | None = None,
    model: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    backend: Any = None,
    run_query: RunQuery | None = None,
) -> TrailUpdateResult:
    """Sync entry point for the production hook (not inside an event loop).

    Async callers (run_corpus.py, pytest-asyncio tests) use
    `update_for_action_async` directly instead — this wrapper would call
    asyncio.run() and fail with `cannot be called from a running event loop`.
    """
    try:
        return asyncio.run(
            update_for_action_async(
                session_id=session_id,
                action_index=action_index,
                action_summary=action_summary,
                assistant_reasoning=assistant_reasoning,
                latest_user_message=latest_user_message,
                journal_root_goal=journal_root_goal,
                cwd=cwd,
                model=model,
                timeout_s=timeout_s,
                backend=backend,
                run_query=run_query,
            )
        )
    except Exception as exc:
        # Last-resort: return a no-op result so the hook never crashes.
        return TrailUpdateResult(
            trail=empty_trail(),
            drift_flag=None,
            error=f"asyncio.run failed: {exc!r}",
            skipped_reason=None,
            latency_ms=0.0,
            diff_summary=[],
        )


def _apply_payload(
    *,
    base: Trail,
    payload: dict[str, Any] | None,
    action_index: int,
    action_summary: str,
    journal_root_goal: str | None,
    redirect: bool,
) -> tuple[Trail, DriftFlag | None, str | None]:
    """Code-side validation + suppression. NEVER trusts the model's bool
    outputs blindly — applies stall guard, K-window suppression, and
    drift-citation hygiene.

    Returns (new_trail, drift_flag_to_record, skipped_reason).
    `drift_flag_to_record.delivered_to_agent` is the gate that controls
    whether the hook will forward a canonical question; the flag itself
    is always persisted into trail.drift_flags for the viewer.
    """
    skipped_reason: str | None = None
    new_root_goal = (
        (journal_root_goal or "").strip()
        if journal_root_goal is not None
        else base.last_root_goal
    )
    new_trail = Trail(
        breadcrumbs=list(base.breadcrumbs),
        drift_flags=list(base.drift_flags),
        action_index=action_index,
        prompt_sha=_system_prompt_sha(),
        schema_version=SCHEMA_VERSION,
        non_advance_streak=base.non_advance_streak,
        last_root_goal=new_root_goal,
        ts=time.time(),
    )

    if payload is None:
        # Model failed — stall guard still ticks because no advance happened.
        new_trail.non_advance_streak = base.non_advance_streak + 1
        skipped_reason = skipped_reason or "no payload"
        _maybe_stall_breadcrumb(new_trail, action_index, action_summary)
        return new_trail, None, skipped_reason

    advances = bool(payload.get("advances_trail"))
    drift_detected = bool(payload.get("drift_detected"))

    if advances:
        bc_text = _clip(payload.get("breadcrumb_text"), MAX_BREADCRUMB_TEXT)
        if not bc_text:
            # Model said "advances" but gave no text — synthesize from action.
            bc_text = _clip(action_summary, MAX_BREADCRUMB_TEXT)
        level = _coerce_level(payload.get("abstraction_level"))
        new_trail.breadcrumbs.append(
            Breadcrumb(
                action_index=action_index,
                breadcrumb_text=bc_text,
                abstraction_level=level,
                action_summary=_clip(action_summary, 240),
                ts=time.time(),
            )
        )
        new_trail.non_advance_streak = 0
    else:
        new_trail.non_advance_streak = base.non_advance_streak + 1
        _maybe_stall_breadcrumb(new_trail, action_index, action_summary)

    # Enforce breadcrumb cap.
    if len(new_trail.breadcrumbs) > MAX_BREADCRUMBS:
        new_trail.breadcrumbs = new_trail.breadcrumbs[-MAX_BREADCRUMBS:]

    drift_flag_obj: DriftFlag | None = None
    if drift_detected:
        kind = _coerce_drift_kind(payload.get("drift_kind"))
        reasoning = _clip(payload.get("drift_reasoning"), MAX_DRIFT_REASONING)
        cited_raw = payload.get("cited_action_indexes") or []
        cited: list[int] = []
        if isinstance(cited_raw, list):
            for x in cited_raw[:MAX_CITED]:
                try:
                    cited.append(int(x))
                except (TypeError, ValueError):
                    continue
        suppressed = False
        # Citation hygiene: empty / missing cites → suppress, drift was
        # likely a hallucination per the prompt's anti-echo-chamber clause.
        if not cited:
            suppressed = True
            if not skipped_reason:
                skipped_reason = "drift without citations — suppressed"
        # Window reset on user redirect: pretend the prior K events had
        # no same-kind flag, because the user changed direction.
        if not suppressed and not redirect:
            if _drift_was_recent_for_kind(
                new_trail, kind, DRIFT_SUPPRESS_WINDOW
            ):
                suppressed = True
                if not skipped_reason:
                    skipped_reason = (
                        f"same kind {kind} in last "
                        f"{DRIFT_SUPPRESS_WINDOW} drifts"
                    )
        delivered = (not suppressed)
        drift_flag_obj = DriftFlag(
            action_index=action_index,
            drift_kind=kind,
            drift_reasoning=reasoning,
            cited_action_indexes=cited,
            suppressed=suppressed,
            delivered_to_agent=delivered,
            ts=time.time(),
        )
        new_trail.drift_flags.append(drift_flag_obj)
        if len(new_trail.drift_flags) > MAX_DRIFT_FLAGS:
            new_trail.drift_flags = new_trail.drift_flags[-MAX_DRIFT_FLAGS:]

    return new_trail, drift_flag_obj, skipped_reason


def _maybe_stall_breadcrumb(
    trail: Trail, action_index: int, action_summary: str
) -> None:
    """Stall guard: if non_advance_streak hits STALL_THRESHOLD, force an
    `unclear`-level breadcrumb so the trail keeps moving. Without this
    a long Bash/Read recon could freeze the timeline and blind drift
    detection.
    """
    if trail.non_advance_streak < STALL_THRESHOLD:
        return
    trail.breadcrumbs.append(
        Breadcrumb(
            action_index=action_index,
            breadcrumb_text=_clip(
                f"stall-guard breadcrumb after "
                f"{trail.non_advance_streak} non-advancing actions",
                MAX_BREADCRUMB_TEXT,
            ),
            abstraction_level="unclear",
            action_summary=_clip(action_summary, 240),
            ts=time.time(),
        )
    )
    trail.non_advance_streak = 0
    if len(trail.breadcrumbs) > MAX_BREADCRUMBS:
        trail.breadcrumbs = trail.breadcrumbs[-MAX_BREADCRUMBS:]


def question_for_kind(kind: str) -> str:
    """Return the canonical Socratic question for a drift_kind. Falls
    back to the 'other' question for unknown kinds.
    """
    return TRAIL_DRIFT_QUESTIONS.get(kind) or TRAIL_DRIFT_QUESTIONS["other"]


# --- Stop-event rubric ------------------------------------------------------


@dataclass(frozen=True)
class StopVerdict:
    stop_appropriate: bool
    reasoning: str
    missing_pieces: list[str]
    delivered_to_agent: bool  # true ⇒ Stop hook emitted decision=block
    error: str | None
    latency_ms: float
    raw_payload: dict[str, Any] | None


def _render_trail_for_stop(t: Trail, window: int = 30) -> str:
    """Render the full breadcrumb trail for the stop rubric. Larger
    window than per-tick rendering because the rubric needs the whole
    journey, not just the recent suffix."""
    if not t.breadcrumbs:
        return "(empty trail — agent did no recorded tool actions this session)"
    lines: list[str] = []
    crumbs = t.breadcrumbs[-window:]
    if len(t.breadcrumbs) > window:
        lines.append(
            f"(showing last {window} of {len(t.breadcrumbs)} breadcrumbs)"
        )
    for b in crumbs:
        lines.append(
            f"  [{b.abstraction_level}] #{b.action_index}  "
            f"{_clip(b.breadcrumb_text, 160)}"
        )
        if b.action_summary:
            lines.append(f"      action: {_clip(b.action_summary, 200)}")
    return "\n".join(lines)


def build_stop_user_message(
    *,
    trail: Trail,
    latest_user_message: str,
    final_assistant_text: str | None,
    json_mode: bool = False,
) -> str:
    """Compose the user message for the stop rubric. `json_mode=True`
    appends the openai_json output-format suffix."""
    from .prompts import _STOP_OUTPUT_FORMAT_SUFFIX

    parts: list[str] = []
    parts.append(
        "## Most recent USER REQUEST (verbatim)\n"
        + _clip(latest_user_message, 3000)
    )
    parts.append("## Breadcrumb trail\n" + _render_trail_for_stop(trail))
    if final_assistant_text:
        parts.append(
            "## Agent's FINAL TEXT (the message ending the turn)\n"
            + _clip(final_assistant_text, 4000)
        )
    else:
        parts.append("## Agent's FINAL TEXT\n(none recorded)")
    parts.append(
        "## Task\n"
        "Apply the rubric in the system prompt. Decide:\n"
        " (1) Decompose the USER REQUEST into concrete asks.\n"
        " (2) For each ask: find evidence in the trail (committed work) or "
        "in the FINAL TEXT (legitimate scoping / blocker).\n"
        " (3) If everything is addressed → stop_appropriate=true. If any "
        "ask is open without a legitimate reason → stop_appropriate=false "
        "with `missing_pieces` naming what's open.\n"
        "Default stop_appropriate=true when uncertain. Call "
        "`evaluate_stop` exactly once."
    )
    msg = "\n\n".join(parts)
    if json_mode:
        msg += _STOP_OUTPUT_FORMAT_SUFFIX
    return msg


def _build_stop_tool(captured: _Captured):
    @tool("evaluate_stop", EVALUATE_STOP_DESCRIPTION, EVALUATE_STOP_INPUT_SCHEMA)
    async def evaluate_stop(args: dict[str, Any]) -> dict[str, Any]:
        captured.payload = args
        return {"content": [{"type": "text", "text": "stop recorded"}]}

    return evaluate_stop


def _build_stop_options(captured: _Captured, model: str) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly_stop",
        "1.0.0",
        [_build_stop_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=STOP_RUBRIC_SYSTEM_PROMPT,
        mcp_servers={"gadfly_stop": server},
        allowed_tools=["mcp__gadfly_stop__evaluate_stop"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        settings="{}",
        thinking=ThinkingConfigDisabled(type="disabled"),
        max_turns=2,
        env={"GADFLY_INTERNAL": "1"},
    )


async def _evaluate_stop_sdk(
    *,
    user_message: str,
    model: str,
    timeout_s: float,
    run_query: RunQuery,
) -> tuple[dict[str, Any] | None, str | None]:
    captured = _Captured()
    options = _build_stop_options(captured, model)
    try:
        await asyncio.wait_for(
            run_query(user_message, options), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return None, f"timeout after {timeout_s}s"
    except FileNotFoundError as exc:
        return None, f"claude CLI not found: {exc!s}"
    except Exception as exc:
        return None, f"agent-sdk error: {exc!r}"
    if captured.payload is None:
        return None, "model did not call evaluate_stop"
    return captured.payload, None


async def _evaluate_stop_oai(
    *,
    backend,
    user_message: str,
    model: str,
    timeout_s: float,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        br = await backend.evaluate(
            system_prompt=STOP_RUBRIC_SYSTEM_PROMPT,
            user_message=user_message,
            model=model,
            tool_name="evaluate_stop",
            tool_description=EVALUATE_STOP_DESCRIPTION,
            tool_parameters=EVALUATE_STOP_JSON_SCHEMA,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return None, f"backend error: {exc!r}"
    if br.verdict_args is None:
        return None, br.error or "no payload"
    return dict(br.verdict_args), None


async def evaluate_stop_async(
    *,
    session_id: str,
    latest_user_message: str,
    final_assistant_text: str | None,
    model: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    backend: Any = None,
    run_query: RunQuery | None = None,
) -> StopVerdict:
    """Run the stop rubric on the current session's trail. Returns a
    StopVerdict — never raises. The trail is loaded from disk via
    load_current(session_id); when missing, the rubric is asked
    against an empty trail (still valid — the rubric works on the
    user_request alone in that case).
    """
    t0 = time.perf_counter()
    trail_state = load_current(session_id) or empty_trail()

    json_mode = False
    eff_backend = backend
    if eff_backend is None and run_query is None:
        eff_backend = _default_backend_from_env()
    if eff_backend is not None and type(eff_backend).__name__.startswith(
        "OpenAIJson"
    ):
        json_mode = True

    user_message = build_stop_user_message(
        trail=trail_state,
        latest_user_message=latest_user_message,
        final_assistant_text=final_assistant_text,
        json_mode=json_mode,
    )
    eff_model = model or os.environ.get("GADFLY_MODEL") or DEFAULT_MODEL

    if eff_backend is not None:
        payload, error = await _evaluate_stop_oai(
            backend=eff_backend,
            user_message=user_message,
            model=eff_model,
            timeout_s=timeout_s,
        )
    else:
        rq = run_query or _default_run_query
        payload, error = await _evaluate_stop_sdk(
            user_message=user_message,
            model=eff_model,
            timeout_s=timeout_s,
            run_query=rq,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if payload is None:
        # Conservative default: when the rubric fails, treat the stop as
        # appropriate — better to let the agent stop than to force it
        # into a confused continuation loop on a model error.
        return StopVerdict(
            stop_appropriate=True,
            reasoning="(rubric failed — defaulting to appropriate)",
            missing_pieces=[],
            delivered_to_agent=False,
            error=error,
            latency_ms=latency_ms,
            raw_payload=None,
        )

    stop_ok = bool(payload.get("stop_appropriate", True))
    reasoning = str(payload.get("reasoning") or "")[:1000]
    missing_raw = payload.get("missing_pieces") or []
    missing: list[str] = []
    if isinstance(missing_raw, list):
        for m in missing_raw[:10]:
            if isinstance(m, str) and m.strip():
                missing.append(m.strip()[:200])
    return StopVerdict(
        stop_appropriate=stop_ok,
        reasoning=reasoning,
        missing_pieces=missing,
        # delivered_to_agent is set by the hook layer based on
        # GADFLY_STOP_FEEDBACK; trail layer just reports the verdict.
        delivered_to_agent=False,
        error=None,
        latency_ms=latency_ms,
        raw_payload=dict(payload),
    )


def evaluate_stop(
    *,
    session_id: str,
    latest_user_message: str,
    final_assistant_text: str | None,
    model: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    backend: Any = None,
    run_query: RunQuery | None = None,
) -> StopVerdict:
    """Sync entry point for the Stop hook (not inside an event loop)."""
    try:
        return asyncio.run(
            evaluate_stop_async(
                session_id=session_id,
                latest_user_message=latest_user_message,
                final_assistant_text=final_assistant_text,
                model=model,
                timeout_s=timeout_s,
                backend=backend,
                run_query=run_query,
            )
        )
    except Exception as exc:
        return StopVerdict(
            stop_appropriate=True,
            reasoning=f"(asyncio.run failed: {exc!r})",
            missing_pieces=[],
            delivered_to_agent=False,
            error=f"asyncio.run failed: {exc!r}",
            latency_ms=0.0,
            raw_payload=None,
        )


def stop_question() -> str:
    """The canonical Socratic question delivered to the agent when the
    stop rubric flags premature_stop. Single fixed string — the model's
    own reasoning never reaches the agent."""
    return TRAIL_STOP_QUESTION
