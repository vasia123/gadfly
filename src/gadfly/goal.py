"""Distil the agent's *current goal* from the conversation trail.

Pair extraction and service-tag scrubbing live in `pairs.py`; this module
is now just the Haiku-distillation layer + per-session cache.

A single raw user message rarely captures the goal of the work the agent
is doing. People say "actually use criterion not bench harness" — that
line alone is meaningless. To understand it you need the assistant's
previous proposal it was replying to, plus the earlier user message that
set the original objective. So `pairs.extract_pairs` builds chronological
(prev-assistant, user) pairs from the transcript, and `load_or_distill`
asks Haiku once: "given the conversation so far, in one or two sentences
— what is the agent supposed to be doing right now?". The answer is
cached per session, keyed by the hash of the pairs already distilled,
and grown incrementally as new user messages appear.

Failure semantics — same as the rest of gadfly: any error returns None
and is logged via the regular audit-log error field. Distillation is a
quality boost, not a requirement.

NOTE: Phase 3 of the journal migration deletes this module entirely
(the journal carries `root_goal` natively). During Phase 1 it runs in
parallel with the journal updater so we can compare outputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
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
# Back-compat: existing callers (session.py, tests) import pair utilities
# and the `Pair` dataclass from `gadfly.goal`. Keep those names alive here.
from .pairs import (  # noqa: F401
    Pair,
    clean_user_text,
    extract_pairs,
    _JUNK_PATTERNS,
    _STRIPPABLE_TAGS,
)


@dataclass
class GoalState:
    """What we know about the agent's goal in this session."""

    goal: str | None = None
    raw_pairs: list[Pair] | None = None
    error: str | None = None


# --- Cache -------------------------------------------------------------------


def _cache_dir() -> Path:
    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "goals"
    return Path.home() / ".claude" / "gadfly" / "goals"


def _cache_path(session_id: str) -> Path:
    safe = session_id.replace("/", "_") or "unknown"
    return _cache_dir() / f"{safe}.json"


@dataclass
class _CacheEntry:
    goal: str
    pair_hashes: list[str]
    ts: float
    prompt_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _system_prompt_sha() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


def _load_cache(session_id: str) -> _CacheEntry | None:
    try:
        p = _cache_path(session_id)
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        goal = data.get("goal")
        hashes = data.get("pair_hashes")
        ts = data.get("ts")
        prompt_sha = data.get("prompt_sha") or ""
        if not isinstance(goal, str) or not isinstance(hashes, list):
            return None
        if prompt_sha and prompt_sha != _system_prompt_sha():
            return None
        return _CacheEntry(
            goal=goal,
            pair_hashes=[str(h) for h in hashes],
            ts=float(ts or 0),
            prompt_sha=str(prompt_sha),
        )
    except Exception:
        return None


def _save_cache(session_id: str, entry: _CacheEntry) -> None:
    try:
        d = _cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        _cache_path(session_id).write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


# --- Distillation via Haiku --------------------------------------------------


SYSTEM_PROMPT = """\
You are a goal-distillation assistant for a code-supervision system.

Given a conversation between a user and a coding agent (as a series of
(assistant_text, user_text) pairs), and optionally a previous goal
summary, produce ONE to THREE sentences describing what the agent should
be working on RIGHT NOW.

Rules:

(1) Capture the *current* objective. If the user has redirected or
    narrowed the task in later messages, follow the latest redirection.

(2) Preserve concrete nouns (file names, function names, library names,
    metric names). Drop chit-chat.

(3) Do not include the agent's proposed implementation steps. Just the
    goal as the user has framed it.

(4) If a previous goal is given, treat it as the agent's understanding so
    far. Update it with the new pairs.

(5) **Unresolved-problem priority.** This is the most important rule.
    When a new user message reports that an EARLIER problem is still not
    fixed — phrasings like "the bug is still there", "doesn't work",
    "didn't help", "this hasn't gone away", "проблема осталась",
    "баг никуда не делся", "не сработало" — that earlier problem is now
    the TOP priority of the goal, even if the agent has been working on
    something else in between. Do NOT drop it from the goal just because
    later pairs went off on tangents. The newer work, if any, becomes
    secondary or is on hold until the original bug is fixed.

    Concrete example:
      pair N-3: user → "fix bug X"
      pair N-2: agent → "fixed (here is patch)"  +  user → "now do feature Y"
      pair N-1: agent → "Y done"               +  user → "bug X is still there"
    Distilled goal: "Fix bug X — it is still reproducing after the first
    attempt. Feature Y (which was added on top) is on hold until X is
    actually resolved."

(6) When in doubt about whether two threads are part of the same goal or
    two separate goals, list BOTH in the distilled summary. Losing
    information is worse than slightly more verbose output.

(7) Output your answer by calling the `submit_goal` tool exactly once.
    Do not write any text outside the tool call.
"""


SUBMIT_GOAL_DESCRIPTION = (
    "Submit the distilled goal. Call this exactly once. Field `goal` "
    "is one or two short sentences describing what the agent should be "
    "working on right now, preserving concrete names from the conversation."
)


def _build_user_prompt(prior_goal: str | None, pairs: list[Pair]) -> str:
    parts: list[str] = []
    if prior_goal:
        parts.append("## Previous goal (the agent's current understanding)\n" + prior_goal)
    parts.append("## Conversation pairs (oldest first)")
    for i, pair in enumerate(pairs, start=1):
        block = [f"### pair {i}"]
        if pair.assistant_text:
            block.append("agent said:\n" + pair.assistant_text[:1500])
        block.append("user replied:\n" + pair.user_text[:2000])
        parts.append("\n\n".join(block))
    parts.append(
        "## Task\n"
        "Distil the agent's CURRENT goal from the conversation above. "
        "Call submit_goal exactly once."
    )
    return "\n\n".join(parts)


@dataclass
class _Captured:
    goal: str | None = None


def _build_submit_goal_tool(captured: _Captured):
    @tool(
        "submit_goal",
        SUBMIT_GOAL_DESCRIPTION,
        {"goal": str},
    )
    async def submit_goal(args: dict[str, Any]) -> dict[str, Any]:
        g = args.get("goal")
        if isinstance(g, str):
            captured.goal = g.strip()
        return {"content": [{"type": "text", "text": "goal recorded"}]}

    return submit_goal


def _build_options(captured: _Captured, model: str) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly_goal",
        "1.0.0",
        [_build_submit_goal_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"gadfly_goal": server},
        allowed_tools=["mcp__gadfly_goal__submit_goal"],
        permission_mode="bypassPermissions",
        setting_sources=[],
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


async def _distill_async(
    *,
    prior_goal: str | None,
    pairs: list[Pair],
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery = _default_run_query,
) -> tuple[str | None, str | None]:
    captured = _Captured()
    options = _build_options(captured, model)
    prompt = _build_user_prompt(prior_goal, pairs)
    try:
        await asyncio.wait_for(run_query(prompt, options), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None, f"timeout after {timeout_s}s"
    except FileNotFoundError as exc:
        return None, f"claude CLI not found: {exc!s}"
    except Exception as exc:
        return None, f"agent-sdk error: {exc!r}"
    if not captured.goal:
        return None, "Haiku did not call submit_goal"
    return captured.goal, None


# --- Public entry point ------------------------------------------------------


def load_or_distill(
    *,
    session_id: str,
    pairs: list[Pair],
    run_query: RunQuery = _default_run_query,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> GoalState:
    """Return the current goal state for this session."""
    if not pairs:
        return GoalState(goal=None, raw_pairs=[], error=None)

    cache = _load_cache(session_id)
    pair_hashes = [p.to_cache_hash() for p in pairs]

    consumed = 0
    if cache:
        prefix = cache.pair_hashes
        for i, h in enumerate(prefix):
            if i >= len(pair_hashes) or pair_hashes[i] != h:
                break
            consumed += 1
        if consumed != len(prefix):
            cache = None
            consumed = 0

    new_pairs = pairs[consumed:]
    if cache and not new_pairs:
        return GoalState(goal=cache.goal, raw_pairs=[], error=None)

    prior_goal = cache.goal if cache else None
    pairs_cached = consumed
    pairs_new = len(new_pairs)

    t0 = time.perf_counter()
    try:
        goal, err = asyncio.run(
            _distill_async(
                prior_goal=prior_goal,
                pairs=new_pairs,
                model=model,
                timeout_s=timeout_s,
                run_query=run_query,
            )
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        audit_log.append_goal_event(
            session_id=session_id,
            pairs_total=len(pairs),
            pairs_new=pairs_new,
            pairs_cached=pairs_cached,
            prior_goal=prior_goal,
            goal=None,
            latency_ms=latency_ms,
            error=f"asyncio.run failed: {exc!r}",
            cache_hit=False,
        )
        return GoalState(goal=prior_goal, raw_pairs=new_pairs, error=f"asyncio.run failed: {exc!r}")

    latency_ms = (time.perf_counter() - t0) * 1000

    if goal is None:
        audit_log.append_goal_event(
            session_id=session_id,
            pairs_total=len(pairs),
            pairs_new=pairs_new,
            pairs_cached=pairs_cached,
            prior_goal=prior_goal,
            goal=None,
            latency_ms=latency_ms,
            error=err,
            cache_hit=False,
        )
        return GoalState(goal=prior_goal, raw_pairs=new_pairs, error=err)

    _save_cache(
        session_id,
        _CacheEntry(
            goal=goal,
            pair_hashes=pair_hashes,
            ts=time.time(),
            prompt_sha=_system_prompt_sha(),
        ),
    )
    audit_log.append_goal_event(
        session_id=session_id,
        pairs_total=len(pairs),
        pairs_new=pairs_new,
        pairs_cached=pairs_cached,
        prior_goal=prior_goal,
        goal=goal,
        latency_ms=latency_ms,
        error=None,
        cache_hit=False,
    )
    return GoalState(goal=goal, raw_pairs=[], error=None)
