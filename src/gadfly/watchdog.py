"""Call Haiku to grade a single Claude Code tool action.

This is a one-shot, non-agentic call. We hand Haiku:
  - the gadfly system prompt (the "professional or not" rubric),
  - a user-message describing the action and its context, and
  - exactly one tool: `submit_verdict(professional, reason, suggestion)`.

Haiku must call that tool exactly once; we capture its arguments and turn
them into a Verdict.

Backend: `claude-agent-sdk` running on top of the user's authenticated
Claude Code CLI. This bills against the user's existing Claude subscription
instead of requiring a separate ANTHROPIC_API_KEY.

Two safety properties we must preserve at all costs:
  1. Never break the parent Claude Code session. Any failure (timeout,
     transport error, CLI missing, no verdict) becomes a silent_ok verdict
     plus a logged error.
  2. Never recurse. The inner CLI must not pick up the same PostToolUse
     hook. Enforced two ways:
        - `setting_sources=[]` so the inner CLI ignores
          ~/.claude/settings.json (where our hook is wired in),
        - `env={"GADFLY_INTERNAL": "1"}` as a belt-and-braces check that
          hook.py reads on startup and exits early when set.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
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
    SUBMIT_VERDICT_DESCRIPTION,
    SUBMIT_VERDICT_INPUT_SCHEMA,
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
DEFAULT_TIMEOUT_S = 60.0  # CLI cold start ~10s first time, ~2-3s warm; Haiku adds 1-3s
DEFAULT_MAX_TURNS = 2  # turn 1: submit_verdict tool_use; turn 2: Haiku must
# close out after receiving the tool_result. max_turns=1 trips
# "Reached maximum number of turns" — empirically verified.


@dataclass(frozen=True)
class EvaluationResult:
    verdict: Verdict
    error: str | None  # human-readable error or None on success
    user_message: str = ""  # the exact prompt we sent Haiku (for the audit log)
    system_prompt_sha: str = ""  # sha of the SYSTEM_PROMPT at evaluation time


@dataclass
class _Captured:
    """Closed-over container that the submit_verdict tool writes into."""

    verdict_args: dict[str, Any] | None = None


def _build_submit_verdict_tool(captured: _Captured):
    @tool("submit_verdict", SUBMIT_VERDICT_DESCRIPTION, SUBMIT_VERDICT_INPUT_SCHEMA)
    async def submit_verdict(args: dict[str, Any]) -> dict[str, Any]:
        captured.verdict_args = args
        return {"content": [{"type": "text", "text": "verdict recorded"}]}

    return submit_verdict


def _build_options(captured: _Captured, model: str, system_prompt: str = SYSTEM_PROMPT) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly",
        "1.0.0",
        [_build_submit_verdict_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        mcp_servers={"gadfly": server},
        allowed_tools=["mcp__gadfly__submit_verdict"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        # CRITICAL recursion guard: pass an EMPTY settings object via
        # `--settings '{}'`. Without this the inner CLI inherits hooks from
        # the user's ~/.claude/settings.json (cc-telegram-notify Stop /
        # Notification, our own gadfly hook, etc.) and fires phantom
        # notifications when its own turn ends. The `hooks={}` option in
        # ClaudeAgentOptions sounds like the right knob but it is for SDK-
        # internal Python hook callbacks; subprocess_cli.py does NOT
        # translate it to a CLI flag. Verified by reading SDK source.
        settings="{}",
        # Extended thinking is wasted latency for one-shot classification
        # with a forced tool call — disable it explicitly. CAUTION:
        # `ThinkingConfigDisabled` is a TypedDict, not a dataclass, so
        # calling it with no arguments silently produces `{}` and the SDK
        # then explodes with `KeyError('type')`. The `type=` kwarg below
        # is mandatory. Covered by tests/test_live.py::test_evaluate_*.
        thinking=ThinkingConfigDisabled(type="disabled"),
        max_turns=DEFAULT_MAX_TURNS,
        env={"GADFLY_INTERNAL": "1"},
    )


# Default query-runner. Tests substitute this via the `run_query` parameter
# to avoid spawning a real CLI.
async def _default_run_query(prompt: str, options: ClaudeAgentOptions) -> None:
    async for _ in query(prompt=prompt, options=options):
        # We don't care about the message stream — we only need the iterator
        # to drive the SDK forward so the submit_verdict callback fires.
        pass


RunQuery = Callable[[str, ClaudeAgentOptions], Awaitable[None]]


async def evaluate_async(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: Any,
    context: SessionContext,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery = _default_run_query,
) -> EvaluationResult:
    captured = _Captured()
    # Phase 2 is the default — verdict reads the journal when one is
    # supplied. Rollback to Phase 1 (legacy prompt) by setting
    # GADFLY_JOURNAL_VERDICT=0.
    use_journal = (
        context.journal is not None
        and os.environ.get("GADFLY_JOURNAL_VERDICT", "1") != "0"
    )
    system_prompt = _select_system_prompt(use_journal)
    options = _build_options(captured, model, system_prompt=system_prompt)
    # E2: pull verdict_patterns from project_state for this cwd, so the
    # journal-mode prompt can render them. Cheap (one JSON read).
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

    try:
        await asyncio.wait_for(run_query(user_message, options), timeout=timeout_s)
    except asyncio.TimeoutError:
        return _result(Verdict.silent_ok(), f"timeout after {timeout_s}s")
    except FileNotFoundError as exc:
        # claude CLI not on PATH
        return _result(Verdict.silent_ok(), f"claude CLI not found: {exc!s}")
    except Exception as exc:
        return _result(Verdict.silent_ok(), f"agent-sdk error: {exc!r}")

    if captured.verdict_args is None:
        return _result(Verdict.silent_ok(), "Haiku did not call submit_verdict")
    return _result(Verdict.from_tool_input(captured.verdict_args), None)


def evaluate(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: Any,
    context: SessionContext,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery = _default_run_query,
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
            )
        )
    except Exception as exc:
        # asyncio.run itself can fail (e.g., already-running loop in some
        # bizarre embedding). Stay silent.
        return EvaluationResult(
            verdict=Verdict.silent_ok(),
            error=f"asyncio.run failed: {exc!r}",
            user_message="",
            system_prompt_sha="",
        )
