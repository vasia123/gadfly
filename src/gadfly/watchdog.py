"""Grade a single Claude Code tool action by calling a model and parsing
its forced tool-call back into a Verdict.

This module owns:
  - selection of the system prompt (legacy vs journal-aware),
  - assembly of the user message (delegated to prompts.build_user_message),
  - the EvaluationResult wrapper used by the hook / audit log.

It does NOT own the wire protocol. That lives in `backends/`. The
default backend is `ClaudeSDKBackend` (claude-agent-sdk + the user's
Claude Code CLI subscription); callers can pass any other backend for
corpus benchmarking or self-hosted endpoints.

Two safety properties we preserve at all costs (regardless of backend):
  1. Never break the parent Claude Code session. Any failure
     (timeout, transport error, CLI missing, no verdict) becomes a
     silent_ok verdict plus a logged error.
  2. The Claude-SDK backend never recurses (see CLAUDE.md §3 — those
     guards live in backends/claude_sdk.py).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from . import log as audit_log
from .backends import Backend, BackendResult, ClaudeSDKBackend
from .backends.claude_sdk import DEFAULT_MAX_TURNS
from .prompts import (
    SUBMIT_VERDICT_DESCRIPTION,
    SUBMIT_VERDICT_JSON_SCHEMA,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_JOURNAL,
    build_user_message,
)
from .session import SessionContext
from .verdict import Verdict


def _select_system_prompt(use_journal: bool) -> str:
    """Phase 2 default. `GADFLY_JOURNAL_VERDICT=0` rolls back to the
    legacy verdict prompt (Phase 1 shadow mode)."""
    if use_journal and os.environ.get("GADFLY_JOURNAL_VERDICT", "1") != "0":
        return SYSTEM_PROMPT_JOURNAL
    return SYSTEM_PROMPT


DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_TIMEOUT_S = 60.0  # CLI cold start ~10s first time, ~2-3s warm; model adds 1-3s


@dataclass(frozen=True)
class EvaluationResult:
    verdict: Verdict
    error: str | None  # human-readable error or None on success
    user_message: str = ""  # the exact prompt we sent the model (audit log)
    system_prompt_sha: str = ""  # sha of the SYSTEM_PROMPT at evaluation time


# Back-compat type alias: tests that pass `run_query=...` keep working
# because evaluate_async wraps the callable in a ClaudeSDKBackend
# constructed with that run_query.
RunQuery = Callable[[str, Any], Awaitable[None]]


async def evaluate_async(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: Any,
    context: SessionContext,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    backend: Backend | None = None,
) -> EvaluationResult:
    use_journal = (
        context.journal is not None
        and os.environ.get("GADFLY_JOURNAL_VERDICT", "1") != "0"
    )
    system_prompt = _select_system_prompt(use_journal)

    verdict_patterns: dict[str, Any] | None = None
    if use_journal and context.cwd:
        try:
            from . import project_state as _ps

            state = _ps.load_state(context.cwd)
            if state.verdict_patterns:
                verdict_patterns = state.verdict_patterns
        except Exception:
            verdict_patterns = None
    latest_user_msg = (
        context.recent_user_requests[-1] if context.recent_user_requests else None
    )
    user_message = build_user_message(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_response=tool_response,
        recent_user_requests=context.recent_user_requests,
        last_assistant_plan=context.last_assistant_plan,
        recent_actions=context.recent_actions,
        distilled_goal=context.distilled_goal,
        journal=context.journal if use_journal else None,
        verdict_patterns=verdict_patterns,
        per_file_snapshots=context.per_file_snapshots,
        file_touch_trajectory=context.file_touch_trajectory,
        latest_user_message_verbatim=latest_user_msg,
        active_plan=context.active_plan,
        recent_dialogue_pairs=context.recent_dialogue_pairs,
        recent_bash_actions=context.recent_bash_actions,
        cwd=context.cwd,
    )
    system_prompt_sha = audit_log.ensure_system_prompt(system_prompt)

    def _result(verdict: Verdict, error: str | None) -> EvaluationResult:
        return EvaluationResult(
            verdict=verdict,
            error=error,
            user_message=user_message,
            system_prompt_sha=system_prompt_sha,
        )

    if backend is None:
        # Back-compat with the run_query injection used by unit tests.
        if run_query is not None:
            backend = ClaudeSDKBackend(run_query=run_query, max_turns=max_turns)
        else:
            backend = ClaudeSDKBackend(max_turns=max_turns)

    try:
        br: BackendResult = await backend.evaluate(
            system_prompt=system_prompt,
            user_message=user_message,
            model=model,
            tool_name="submit_verdict",
            tool_description=SUBMIT_VERDICT_DESCRIPTION,
            tool_parameters=SUBMIT_VERDICT_JSON_SCHEMA,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        # Backend itself crashed before returning a BackendResult — stay silent.
        return _result(Verdict.silent_ok(), f"backend crashed: {exc!r}")

    if br.verdict_args is None:
        return _result(Verdict.silent_ok(), br.error or "no verdict")
    return _result(Verdict.from_tool_input(br.verdict_args), None)


def evaluate(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: Any,
    context: SessionContext,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    backend: Backend | None = None,
) -> EvaluationResult:
    """Sync entry point used by the hook."""
    try:
        return asyncio.run(
            evaluate_async(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_response=tool_response,
                context=context,
                model=model,
                timeout_s=timeout_s,
                run_query=run_query,
                max_turns=max_turns,
                backend=backend,
            )
        )
    except Exception as exc:
        return EvaluationResult(
            verdict=Verdict.silent_ok(),
            error=f"asyncio.run failed: {exc!r}",
            user_message="",
            system_prompt_sha="",
        )
