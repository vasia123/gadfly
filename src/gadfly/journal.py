"""Per-session journal — Gadfly's model of what the agent is working on.

The watchdog used to be stateless per hook call: each PostToolUse spawned
a fresh Haiku evaluation with only the last 5 prior actions and recent
user requests. Two failure modes emerged in real sessions:

  1. **Edit-window blindness.** Long Edit series interleaved with grep /
     Bash pushed real code changes out of the 5-action window, so Haiku
     flagged a subsequent Write(MEMORY.md) as "claimed-as-landed but no
     Edits" — incorrectly.
  2. **Verdict echo chamber.** Haiku had no memory of its own prior
     verdicts, so the same "Symptom fix" flag fired 14 times on the same
     topic, long after the agent had heard and pushed back.

The journal solves both by maintaining persistent state across hook
calls. It is *the agent's worklist as Haiku understands it*: a small set
of workstreams, each with status, notes, and watchdog flag history. The
maintainer (this module) is a tiny Haiku call that takes (prior_journal,
new_action_or_user_msg) → updated_journal. The verdict call then reads
the journal as its primary context.

Persistence:
  - current state:    ~/.claude/gadfly/journal/<session_id>.json
  - snapshots (sha):  ~/.claude/gadfly/journals/<sha>.json
  - audit log:        log/<session_id>.jsonl, type="journal_update"

Phase-1 (shadow): this module runs on every PostToolUse but the verdict
prompt does not yet consume the journal. Phase-2 cutover swaps the
verdict prompt to journal-based composition. Phase-3 deletes the
legacy goal.py distillation path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ThinkingConfigDisabled,
    create_sdk_mcp_server,
    query,
    tool,
)

from . import log as audit_log
from .pairs import Pair


# --- Schema ------------------------------------------------------------------

SCHEMA_VERSION = 1

WorkstreamStatus = Literal["open", "in-progress", "blocked", "done", "abandoned"]
FlagMarker = Literal["symptom", "rationalization", "other"]

# Caps to keep the journal bounded over long sessions.
MAX_FLAG_HISTORY_PER_WS = 8
MAX_NOTES_LEN = 1500
MAX_DONE_NOTES_LEN = 200  # truncated for done/abandoned after a while
MAX_WORKSTREAMS = 20


@dataclass(frozen=True)
class FlagEvent:
    action_index: int
    reason: str
    marker: FlagMarker
    agent_pushed_back: bool = False
    pushback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlagEvent":
        return cls(
            action_index=int(d.get("action_index") or 0),
            reason=str(d.get("reason") or ""),
            marker=_coerce_marker(d.get("marker")),
            agent_pushed_back=bool(d.get("agent_pushed_back") or False),
            pushback=d.get("pushback") if isinstance(d.get("pushback"), str) else None,
        )


def _coerce_marker(value: Any) -> FlagMarker:
    if value in ("symptom", "rationalization", "other"):
        return value  # type: ignore[return-value]
    return "other"


def _coerce_status(value: Any) -> WorkstreamStatus:
    if value in ("open", "in-progress", "blocked", "done", "abandoned"):
        return value  # type: ignore[return-value]
    return "open"


@dataclass
class Workstream:
    id: str
    title: str
    status: WorkstreamStatus = "open"
    origin: str = ""
    notes: str = ""
    watchdog_flags: int = 0
    flag_history: list[FlagEvent] = field(default_factory=list)
    last_touched: int = 0
    # Phase B/outcomes-feedback: provenance-tagged ids of priors shown to
    # this workstream's maintainer prompt. Format: "<kind>:<id>"
    # (e.g. "correction:c-abc123"). Used at workstream-close time to
    # increment/decrement usefulness_score of the project-state entries.
    priors_consulted: list[str] = field(default_factory=list)
    # Set when outcomes-feedback has fired for this workstream — guards
    # against double-counting if the journal is re-saved / aggregator
    # re-runs.
    priors_scored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "origin": self.origin,
            "notes": self.notes,
            "watchdog_flags": self.watchdog_flags,
            "flag_history": [f.to_dict() for f in self.flag_history],
            "last_touched": self.last_touched,
            "priors_consulted": list(self.priors_consulted),
            "priors_scored": self.priors_scored,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Workstream":
        fh_raw = d.get("flag_history") or []
        flag_history = [FlagEvent.from_dict(f) for f in fh_raw if isinstance(f, dict)]
        return cls(
            id=str(d.get("id") or ""),
            title=str(d.get("title") or ""),
            status=_coerce_status(d.get("status")),
            origin=str(d.get("origin") or ""),
            notes=str(d.get("notes") or ""),
            watchdog_flags=int(d.get("watchdog_flags") or 0),
            flag_history=flag_history,
            last_touched=int(d.get("last_touched") or 0),
            priors_consulted=[
                str(p) for p in (d.get("priors_consulted") or []) if isinstance(p, str)
            ],
            priors_scored=bool(d.get("priors_scored") or False),
        )


@dataclass
class Drift:
    initial_workstream_ids: list[str] = field(default_factory=list)
    observations: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Drift":
        ids = d.get("initial_workstream_ids") or []
        return cls(
            initial_workstream_ids=[str(x) for x in ids if isinstance(x, str)],
            observations=str(d.get("observations") or ""),
        )


@dataclass
class Journal:
    root_goal: str = ""
    workstreams: list[Workstream] = field(default_factory=list)
    drift: Drift = field(default_factory=Drift)
    action_index: int = 0
    prompt_sha: str = ""
    schema_version: int = SCHEMA_VERSION
    # Hashes of pairs already consumed (so we don't re-feed the maintainer
    # the same user messages). Same idea as goal.py's pair_hashes.
    consumed_pair_hashes: list[str] = field(default_factory=list)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_goal": self.root_goal,
            "workstreams": [w.to_dict() for w in self.workstreams],
            "drift": self.drift.to_dict(),
            "action_index": self.action_index,
            "prompt_sha": self.prompt_sha,
            "schema_version": self.schema_version,
            "consumed_pair_hashes": list(self.consumed_pair_hashes),
            "ts": self.ts,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Journal":
        ws_raw = d.get("workstreams") or []
        workstreams = [Workstream.from_dict(w) for w in ws_raw if isinstance(w, dict)]
        drift_raw = d.get("drift") or {}
        drift = Drift.from_dict(drift_raw if isinstance(drift_raw, dict) else {})
        return cls(
            root_goal=str(d.get("root_goal") or ""),
            workstreams=workstreams,
            drift=drift,
            action_index=int(d.get("action_index") or 0),
            prompt_sha=str(d.get("prompt_sha") or ""),
            schema_version=int(d.get("schema_version") or SCHEMA_VERSION),
            consumed_pair_hashes=[
                str(h) for h in (d.get("consumed_pair_hashes") or []) if isinstance(h, str)
            ],
            ts=float(d.get("ts") or 0.0),
        )

    def workstream_by_id(self, wid: str) -> Workstream | None:
        for w in self.workstreams:
            if w.id == wid:
                return w
        return None


def empty_journal() -> Journal:
    return Journal(prompt_sha=_system_prompt_sha())


# --- Persistence -------------------------------------------------------------


def _current_dir() -> Path:
    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "journal"
    return Path.home() / ".claude" / "gadfly" / "journal"


def _current_path(session_id: str) -> Path:
    safe = session_id.replace("/", "_") or "unknown"
    return _current_dir() / f"{safe}.json"


def load_current(session_id: str) -> Journal | None:
    """Load the per-session current journal. Returns None when missing or
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
        j = Journal.from_dict(data)
    except Exception:
        return None
    if j.schema_version != SCHEMA_VERSION:
        return None
    if j.prompt_sha and j.prompt_sha != _system_prompt_sha():
        # Prompt was tightened — fall back to a fresh rebuild so the new
        # rubric is applied from scratch.
        return None
    return j


def save_current(session_id: str, journal: Journal) -> None:
    try:
        d = _current_dir()
        d.mkdir(parents=True, exist_ok=True)
        _current_path(session_id).write_text(journal.to_json(), encoding="utf-8")
    except Exception:
        pass


# --- The journal maintainer prompt ------------------------------------------


UPDATE_SYSTEM_PROMPT = """\
You are the journal maintainer for a code-supervision watchdog called
gadfly. The journal models *what an autonomous coding agent is working
on right now*. You will be called once per agent action (tool use or
new user message) with the current journal and the new evidence. Your
job: return an updated journal via the `update_journal` tool. Always
call it exactly once. Output nothing outside the tool call.

The journal has fields:

  root_goal : one to three sentences capturing the user's current
              objective for the whole session. Re-state as the user
              redirects.
  workstreams : list of concrete work items the agent is pursuing.
                Each has id, title, status, origin, notes,
                watchdog_flags (count), flag_history (last 8), and
                last_touched (action_index).
                status ∈ {open, in-progress, blocked, done, abandoned}.
  drift : initial_workstream_ids (set once, after the first user msg
          has been processed) and free-form observations.
  action_index : monotonically increasing counter; you receive the
                 new index in the input.

Rules for updates:

(R1) Continuity. Do NOT delete workstreams you previously created
     unless they are demonstrably resolved (status: done) or
     explicitly abandoned by user redirection. When in doubt, keep.

(R2) Don't invent workstreams from a single grep / Read. A workstream
     represents work the agent has *committed to* — a thread of
     intent. A Read or Bash(ls) is exploration; do not spawn a
     workstream from it unless the agent's reasoning explicitly
     declares "I'm going to do X". Edit / Write / MultiEdit always
     belong to *some* workstream — assign them; create a new one
     only if no existing one fits.

(R3) Status transitions:
     - open → in-progress when first Edit/Write/Bash lands for it
     - any → blocked when agent explicitly says "I cannot proceed
       because of X" or hits a definitive error
     - any → done when agent's reasoning says "done", "fixed",
       "complete", "landed" AND the relevant tool call corroborates
     - any → abandoned when the user redirects away from it
       explicitly

(R4) Notes are free-form but tight. ≤1500 chars. Append the latest
     evidence: file paths edited, conclusions reached, contested
     points. Drop stale early-attempt scaffolding once the workstream
     advances. Don't restate the title.

(R5) Drift snapshot. After you have processed the FIRST user message
     of the session, populate drift.initial_workstream_ids with the
     IDs of every workstream that currently exists. Never touch it
     again (it is the historical baseline). drift.observations is
     free-form — note when the agent has not touched a priority-1
     workstream for a while, when a new workstream emerged unplanned,
     or when the user redirected significantly.

(R6) Flag history is APPEND-ONLY. You receive new flag_history
     entries via the input field `new_flag_events` (a list of
     FlagEvents to append to the relevant workstream). You do not
     invent flags yourself — those come from the verdict call.

(R7) Reject implausibly large changes. If the input would have you
     delete more than half of existing workstreams, OR rewrite the
     root_goal radically without a corresponding new user message,
     prefer the conservative update (carry forward). Note your
     concern in drift.observations.

(R8) Stable IDs. Workstream ids are short kebab-case slugs derived
     from the title (e.g. "marlin-exl3-regression", "tiered-bench").
     Once assigned, never change. New workstreams get fresh ids.

(R9) Cap flag_history at the 8 most recent entries per workstream.
     Older entries silently drop.

Output protocol: call `update_journal` exactly once with the full
new state. Don't emit partial diffs — return the complete journal.
"""


UPDATE_JOURNAL_DESCRIPTION = (
    "Submit the updated journal. Call this exactly once. Pass the FULL "
    "new state (root_goal, workstreams, drift). Do not output text "
    "outside this tool call."
)


# JSON schema for the update_journal tool. We accept JSON-serialisable
# primitives only; the SDK's @tool decorator wires this into the inner
# CLI's tool schema.
UPDATE_JOURNAL_INPUT_SCHEMA: dict[str, Any] = {
    "root_goal": str,
    "workstreams": list,
    "drift": dict,
}


def _system_prompt_sha() -> str:
    return hashlib.sha256(UPDATE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


# --- Haiku call --------------------------------------------------------------


@dataclass
class _Captured:
    payload: dict[str, Any] | None = None


def _build_update_tool(captured: _Captured):
    @tool("update_journal", UPDATE_JOURNAL_DESCRIPTION, UPDATE_JOURNAL_INPUT_SCHEMA)
    async def update_journal(args: dict[str, Any]) -> dict[str, Any]:
        captured.payload = args
        return {"content": [{"type": "text", "text": "journal recorded"}]}

    return update_journal


def _build_options(captured: _Captured, model: str) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly_journal",
        "1.0.0",
        [_build_update_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=UPDATE_SYSTEM_PROMPT,
        mcp_servers={"gadfly_journal": server},
        allowed_tools=["mcp__gadfly_journal__update_journal"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        # See watchdog._build_options — empty settings is the recursion
        # guard (`hooks={}` does NOT propagate through subprocess_cli.py).
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


# --- User-message builder ----------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit}c]"


def _render_journal_block(j: Journal) -> str:
    """Compact rendering of the current journal for the maintainer prompt."""
    lines: list[str] = []
    lines.append(f"## Current journal (action_index={j.action_index})")
    lines.append(f"root_goal: {j.root_goal or '(empty)'}")
    if not j.workstreams:
        lines.append("workstreams: (none yet)")
    else:
        lines.append("workstreams:")
        for w in j.workstreams:
            lines.append(
                f"  - id={w.id} status={w.status} flags={w.watchdog_flags} "
                f"last_touched={w.last_touched}"
            )
            lines.append(f"    title: {w.title}")
            if w.origin:
                lines.append(f"    origin: {w.origin}")
            if w.notes:
                lines.append(f"    notes: {_truncate(w.notes, 600)}")
            if w.flag_history:
                lines.append(f"    flag_history (last {len(w.flag_history)}):")
                for f in w.flag_history[-3:]:
                    pb = " [agent pushed back]" if f.agent_pushed_back else ""
                    lines.append(f"      #{f.action_index} [{f.marker}]{pb} {f.reason[:140]}")
    lines.append(
        f"drift.initial_workstream_ids: {j.drift.initial_workstream_ids or '(unset)'}"
    )
    if j.drift.observations:
        lines.append(f"drift.observations: {_truncate(j.drift.observations, 400)}")
    return "\n".join(lines)


def build_update_user_message(
    *,
    journal: Journal,
    new_pairs: list[Pair],
    action_summary: str | None,
    assistant_reasoning: str | None,
    new_flag_events: list[FlagEvent] | None = None,
    project_priors_block: str = "",
) -> str:
    """Assemble the maintainer prompt.

    Inputs:
      - journal : the current state
      - new_pairs : (assistant, user) pairs the maintainer has not yet
        seen. The maintainer should consume them when forming root_goal /
        workstream titles, and update consumed_pair_hashes accordingly
        (we do that on our side after the call — Haiku just sees the
        new content here).
      - action_summary : short string describing the tool call that
        just fired (e.g. "Edit(normalization.rs)" or "Bash: cargo test").
        None when the trigger was a new user message rather than a tool.
      - assistant_reasoning : the assistant text immediately before the
        action, when present.
      - new_flag_events : append these to the relevant workstream's
        flag_history. The maintainer assigns them by id.
    """
    parts: list[str] = [_render_journal_block(journal)]

    # Phase B: durable findings from past sessions in this cwd, retrieved
    # by historian.find_relevant_priors. Goes ABOVE new pairs so the
    # maintainer can use them when titling new workstreams (avoiding
    # title-drift between sessions on the same topic).
    if project_priors_block:
        parts.append(project_priors_block)

    if new_pairs:
        parts.append("## New conversation pairs since last update")
        for i, p in enumerate(new_pairs, start=1):
            block = [f"### new pair {i}"]
            if p.assistant_text:
                block.append("agent said:\n" + _truncate(p.assistant_text, 1500))
            block.append("user replied:\n" + _truncate(p.user_text, 2000))
            parts.append("\n\n".join(block))

    if assistant_reasoning:
        parts.append(
            "## Agent's reasoning immediately before the new action\n"
            + _truncate(assistant_reasoning, 1500)
        )

    if action_summary:
        parts.append("## New action\n" + _truncate(action_summary, 1500))

    if new_flag_events:
        parts.append(
            "## New flag events to append (do not invent more)\n"
            + json.dumps([f.to_dict() for f in new_flag_events], ensure_ascii=False)
        )

    parts.append(
        "## Task\n"
        "Update the journal. Apply rules R1-R9 from the system prompt. "
        "Call `update_journal` exactly once with the FULL new state — "
        "root_goal (string), workstreams (list of objects with id, "
        "title, status, origin, notes, watchdog_flags, flag_history, "
        "last_touched), drift ({initial_workstream_ids, observations}). "
        "Do not emit a partial diff."
    )
    return "\n\n".join(parts)


# --- Update logic ------------------------------------------------------------


def _diff_summary(old: Journal | None, new: Journal) -> list[str]:
    """Human-readable list of changes between old and new — for audit log."""
    out: list[str] = []
    old_ws = {w.id: w for w in (old.workstreams if old else [])}
    new_ws = {w.id: w for w in new.workstreams}

    for wid in new_ws.keys() - old_ws.keys():
        out.append(f"created {wid}: {new_ws[wid].title!r}")
    for wid in old_ws.keys() - new_ws.keys():
        out.append(f"dropped {wid}")
    for wid in new_ws.keys() & old_ws.keys():
        o, n = old_ws[wid], new_ws[wid]
        if o.status != n.status:
            out.append(f"{wid}: {o.status}→{n.status}")
        if o.watchdog_flags != n.watchdog_flags:
            out.append(f"{wid}.flags {o.watchdog_flags}→{n.watchdog_flags}")
    if (old.root_goal if old else "") != new.root_goal:
        out.append("root_goal updated")
    return out


def _is_poisonous_update(old: Journal, candidate: Journal, had_new_user_msg: bool) -> str | None:
    """Refuse implausibly large mutations. Returns a reason string when
    the update should be rejected (carry forward old).

    A new user message legitimises scope-shifting changes (the user
    redirected) — the guard only fires for mutations that happen
    without any new user input to justify them.
    """
    if had_new_user_msg:
        return None
    if not old.workstreams:
        return None  # nothing to defend yet
    drops = len(
        {w.id for w in old.workstreams} - {w.id for w in candidate.workstreams}
    )
    if drops > 0 and drops > len(old.workstreams) // 2:
        return f"refused: would drop {drops}/{len(old.workstreams)} workstreams"
    if old.root_goal and _root_goal_changed_radically(old.root_goal, candidate.root_goal):
        return "refused: radical root_goal rewrite without new user message"
    return None


def _root_goal_changed_radically(old: str, new: str) -> bool:
    if not new:
        return True
    a = set(old.lower().split())
    b = set(new.lower().split())
    if not a:
        return False
    overlap = len(a & b) / max(1, len(a))
    return overlap < 0.2


def _coerce_workstream_payload(d: dict[str, Any]) -> Workstream:
    """Lenient coercion of Haiku-supplied workstream JSON into our dataclass."""
    return Workstream.from_dict(d)


def _journal_from_payload(payload: dict[str, Any], *, base: Journal, action_index: int) -> Journal:
    """Build a new Journal from the maintainer's `update_journal` payload,
    carrying over base.consumed_pair_hashes (updated by caller) and prompt_sha."""
    root_goal = str(payload.get("root_goal") or "").strip() or base.root_goal
    ws_raw = payload.get("workstreams")
    if not isinstance(ws_raw, list):
        ws_raw = []
    workstreams: list[Workstream] = []
    for item in ws_raw:
        if isinstance(item, dict):
            try:
                w = _coerce_workstream_payload(item)
                # Trim notes / flag_history per caps.
                w.notes = _truncate(w.notes, MAX_NOTES_LEN)
                if len(w.flag_history) > MAX_FLAG_HISTORY_PER_WS:
                    w.flag_history = w.flag_history[-MAX_FLAG_HISTORY_PER_WS:]
                workstreams.append(w)
            except Exception:
                continue
    if len(workstreams) > MAX_WORKSTREAMS:
        # Prefer the most recently touched — done/abandoned drop first.
        workstreams.sort(
            key=lambda w: (
                0 if w.status in ("done", "abandoned") else 1,
                w.last_touched,
            )
        )
        workstreams = workstreams[-MAX_WORKSTREAMS:]

    drift_raw = payload.get("drift")
    drift = Drift.from_dict(drift_raw if isinstance(drift_raw, dict) else {})

    # Drift initial_workstream_ids is set once and never changes — preserve
    # the base value if it has been set, otherwise accept Haiku's choice.
    if base.drift.initial_workstream_ids:
        drift.initial_workstream_ids = base.drift.initial_workstream_ids

    return Journal(
        root_goal=root_goal,
        workstreams=workstreams,
        drift=drift,
        action_index=action_index,
        prompt_sha=_system_prompt_sha(),
        schema_version=SCHEMA_VERSION,
        consumed_pair_hashes=list(base.consumed_pair_hashes),  # caller patches
        ts=time.time(),
    )


@dataclass
class JournalUpdateResult:
    journal: Journal
    error: str | None
    skipped_reason: str | None
    latency_ms: float
    diff: list[str]
    # Phase B/C: priors that were retrieved and shown to the maintainer.
    # The hook reads this to decide whether to also surface them as
    # additionalContext to the AGENT (Phase C) when a new workstream
    # was created in this update. Empty list when priors weren't
    # consulted (Phase B off, no cwd, or no relevant findings).
    priors_consulted: list[Any] = field(default_factory=list)


async def _update_async(
    *,
    journal: Journal,
    user_message: str,
    model: str,
    timeout_s: float,
    run_query: RunQuery,
) -> tuple[dict[str, Any] | None, str | None]:
    captured = _Captured()
    options = _build_options(captured, model)
    try:
        await asyncio.wait_for(run_query(user_message, options), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None, f"timeout after {timeout_s}s"
    except FileNotFoundError as exc:
        return None, f"claude CLI not found: {exc!s}"
    except Exception as exc:
        return None, f"agent-sdk error: {exc!r}"
    if captured.payload is None:
        return None, "Haiku did not call update_journal"
    return captured.payload, None


_CLOSED_STATUSES = {"done", "abandoned"}


def _apply_outcomes_feedback(*, cwd: str, base: Journal, candidate: Journal) -> None:
    """When a workstream just closed and had priors_consulted, score the
    referenced project_state entries against the workstream's final
    notes. Mentioned priors get +1 usefulness_score; ignored ones get -1.

    MemoryArena lesson: the hard part of memory is the retrieval-vs-use
    gap. The score nudges Phase B retrieval toward priors that the
    agent actually leans on, away from priors that surface and get
    ignored. Single signal per workstream-close — bounded growth.
    """
    from . import project_state as ps

    base_status = {w.id: w.status for w in base.workstreams}
    state_loaded = False
    state: ps.ProjectState | None = None

    for w in candidate.workstreams:
        if w.priors_scored:
            continue
        if w.status not in _CLOSED_STATUSES:
            continue
        was = base_status.get(w.id)
        if was in _CLOSED_STATUSES:
            # Already closed before this update — don't re-score.
            continue
        if not w.priors_consulted:
            w.priors_scored = True
            continue
        # Lazy-load project state — we only need it if there's actual work.
        if not state_loaded:
            state = ps.load_state(cwd)
            state_loaded = True
        if state is None:
            return
        notes_lower = (w.notes or "").lower()
        for tagged_id in w.priors_consulted:
            kind, _, fid = tagged_id.partition(":")
            if not fid:
                continue
            entry = None
            container_label = ""
            if kind == "promise":
                entry = state.promises.get(fid)
                container_label = "promise"
            elif kind == "correction":
                entry = state.corrections.get(fid)
                container_label = "correction"
            elif kind == "subsystem":
                entry = state.subsystems.get(fid)
                container_label = "subsystem"
            if entry is None:
                continue
            # Match by title token-overlap rather than substring — titles
            # often paraphrase, and substring would miss "auth" vs.
            # "authentication". A single shared salient token is enough
            # to count as "mentioned".
            title = getattr(entry, "title", None) or getattr(entry, "rule", "")
            mentioned = _title_appears_in_notes(title, notes_lower)
            delta = 1 if mentioned else -1
            current = getattr(entry, "usefulness_score", 0)
            new = max(-5, min(10, current + delta))
            setattr(entry, "usefulness_score", new)
            _ = container_label  # currently unused, kept for future audit
        w.priors_scored = True

    if state is not None:
        ps.save_state(state)


def _title_appears_in_notes(title: str, notes_lower: str) -> bool:
    """Token-overlap match. Used to detect whether a prior's title is
    referenced in the workstream's final notes. Stop-words ignored."""
    if not title or not notes_lower:
        return False
    stop = {
        "the", "a", "an", "and", "or", "of", "in", "to", "for", "on",
        "with", "is", "are", "was", "were", "be", "been", "by", "as",
        "at", "from", "this", "that", "these", "those", "it", "its",
    }
    import re as _re

    tokens = {
        t for t in _re.split(r"[^A-Za-z0-9_]+", title.lower())
        if len(t) >= 3 and t not in stop
    }
    if not tokens:
        return False
    # At least 2 distinct salient tokens (or 1 if title only has 1)
    # must appear in notes.
    hits = sum(1 for t in tokens if t in notes_lower)
    return hits >= min(2, len(tokens))


def update_for_action(
    *,
    session_id: str,
    action_index: int,
    action_summary: str | None,
    assistant_reasoning: str | None,
    pairs: list[Pair],
    new_flag_events: list[FlagEvent] | None = None,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery = _default_run_query,
    cwd: str | None = None,
    workstream_hint_files: list[str] | None = None,
) -> JournalUpdateResult:
    """Run one journal update for the given action / new user pairs.

    Synchronous, swallows all exceptions, logs an audit event. Returns
    the journal as it stands after the update (or the prior one when the
    update failed / was rejected).
    """
    base = load_current(session_id) or empty_journal()

    # Identify which pairs are new vs already consumed.
    seen = set(base.consumed_pair_hashes)
    new_pairs = [p for p in pairs if p.to_cache_hash() not in seen]
    had_new_user_msg = len(new_pairs) > 0

    # Skip the Haiku call entirely when nothing new happened (no new
    # pairs, no action, no flag events). This is the common steady-state
    # case for a single hook call without a fresh user message.
    if not new_pairs and not action_summary and not new_flag_events:
        audit_log.append_journal_event(
            session_id=session_id,
            action_index=action_index,
            prior_journal_sha=audit_log.ensure_journal_snapshot(base.to_json()) if base.workstreams or base.root_goal else None,
            new_journal_sha=None,
            diff_summary=[],
            latency_ms=0.0,
            error=None,
            skipped_reason="no_change",
        )
        return JournalUpdateResult(journal=base, error=None, skipped_reason="no_change", latency_ms=0.0, diff=[])

    # Phase B (default-on): retrieve project priors from the historian
    # corpus and inject. Rollback to Phase A (no priors) by setting
    # GADFLY_HISTORIAN_PRIORS=0. Late import to avoid loading historian
    # transitively in environments where the feature is fully off.
    priors_block = ""
    priors_hits: list[Any] = []
    if (
        cwd
        and os.environ.get("GADFLY_HISTORIAN_PRIORS", "1") != "0"
        and (new_pairs or new_flag_events or action_summary)
    ):
        try:
            from . import historian as _historian

            # Best-effort title for retrieval: the latest user message
            # we just received, or the latest assistant reasoning.
            title = ""
            if new_pairs:
                title = new_pairs[-1].user_text[:200]
            elif assistant_reasoning:
                title = assistant_reasoning[:200]
            priors_hits = _historian.find_relevant_priors(
                cwd,
                workstream_title=title,
                file_paths=workstream_hint_files or [],
            )
            priors_block = _historian.render_priors_block(priors_hits)
        except Exception:
            # Priors are a quality boost, not a requirement. Stay silent.
            priors_block = ""
            priors_hits = []

    prompt = build_update_user_message(
        journal=base,
        new_pairs=new_pairs,
        action_summary=action_summary,
        assistant_reasoning=assistant_reasoning,
        new_flag_events=new_flag_events,
        project_priors_block=priors_block,
    )

    t0 = time.perf_counter()
    try:
        payload, err = asyncio.run(
            _update_async(
                journal=base,
                user_message=prompt,
                model=model,
                timeout_s=timeout_s,
                run_query=run_query,
            )
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        prior_sha = audit_log.ensure_journal_snapshot(base.to_json())
        audit_log.append_journal_event(
            session_id=session_id,
            action_index=action_index,
            prior_journal_sha=prior_sha,
            new_journal_sha=None,
            diff_summary=[],
            latency_ms=latency_ms,
            error=f"asyncio.run failed: {exc!r}",
        )
        return JournalUpdateResult(journal=base, error=f"asyncio.run failed: {exc!r}", skipped_reason=None, latency_ms=latency_ms, diff=[])

    latency_ms = (time.perf_counter() - t0) * 1000
    prior_sha = audit_log.ensure_journal_snapshot(base.to_json())

    if payload is None:
        audit_log.append_journal_event(
            session_id=session_id,
            action_index=action_index,
            prior_journal_sha=prior_sha,
            new_journal_sha=None,
            diff_summary=[],
            latency_ms=latency_ms,
            error=err,
        )
        return JournalUpdateResult(journal=base, error=err, skipped_reason=None, latency_ms=latency_ms, diff=[])

    candidate = _journal_from_payload(payload, base=base, action_index=action_index)
    candidate.consumed_pair_hashes = list(base.consumed_pair_hashes) + [
        p.to_cache_hash() for p in new_pairs
    ]
    # Set drift.initial_workstream_ids the first time we have workstreams
    # AND at least one user pair has been consumed.
    if (
        not candidate.drift.initial_workstream_ids
        and candidate.workstreams
        and candidate.consumed_pair_hashes
    ):
        candidate.drift.initial_workstream_ids = [w.id for w in candidate.workstreams]

    # Outcomes-feedback prep: stamp newly-created workstreams with the
    # priors that were retrieved this turn. Carry forward existing
    # priors_consulted / priors_scored on workstreams that already had
    # them (Haiku's payload doesn't repeat these fields).
    base_ws_by_id = {w.id: w for w in base.workstreams}
    if priors_hits:
        for w in candidate.workstreams:
            existing = base_ws_by_id.get(w.id)
            if existing is None:
                # Brand new workstream → it's the one we showed priors to.
                w.priors_consulted = [
                    f"{getattr(h, 'kind', '?')}:{getattr(h, 'id', '?')}"
                    for h in priors_hits
                ]
            else:
                w.priors_consulted = list(existing.priors_consulted)
                w.priors_scored = existing.priors_scored
    else:
        for w in candidate.workstreams:
            existing = base_ws_by_id.get(w.id)
            if existing is not None:
                w.priors_consulted = list(existing.priors_consulted)
                w.priors_scored = existing.priors_scored

    # Poisoning guard.
    poison = _is_poisonous_update(base, candidate, had_new_user_msg=had_new_user_msg)
    if poison:
        # Append the concern to drift.observations on the BASE so it shows
        # up in the viewer, but don't accept the candidate.
        carried = Journal.from_dict(base.to_dict())
        carried.drift.observations = _truncate(
            (carried.drift.observations + "\n" if carried.drift.observations else "") + poison,
            1500,
        )
        carried.ts = time.time()
        save_current(session_id, carried)
        carried_sha = audit_log.ensure_journal_snapshot(carried.to_json())
        audit_log.append_journal_event(
            session_id=session_id,
            action_index=action_index,
            prior_journal_sha=prior_sha,
            new_journal_sha=carried_sha,
            diff_summary=[f"poison: {poison}"],
            latency_ms=latency_ms,
            error=None,
            skipped_reason=poison,
        )
        return JournalUpdateResult(journal=carried, error=None, skipped_reason=poison, latency_ms=latency_ms, diff=[f"poison: {poison}"])

    # Outcomes-feedback: detect workstreams that just closed (status →
    # done/abandoned) with priors_consulted and not yet scored. For
    # each, check whether the priors appear in the final notes and
    # bump their project_state.usefulness_score accordingly.
    if cwd and os.environ.get("GADFLY_HISTORIAN_PRIORS", "1") != "0":
        try:
            _apply_outcomes_feedback(cwd=cwd, base=base, candidate=candidate)
        except Exception:
            # Outcomes feedback is a quality nudge, not a requirement.
            pass

    save_current(session_id, candidate)
    new_sha = audit_log.ensure_journal_snapshot(candidate.to_json())
    diff = _diff_summary(base, candidate)
    audit_log.append_journal_event(
        session_id=session_id,
        action_index=action_index,
        prior_journal_sha=prior_sha,
        new_journal_sha=new_sha,
        diff_summary=diff,
        latency_ms=latency_ms,
        error=None,
    )
    return JournalUpdateResult(
        journal=candidate,
        error=None,
        skipped_reason=None,
        latency_ms=latency_ms,
        diff=diff,
        priors_consulted=list(priors_hits),
    )
