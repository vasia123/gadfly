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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    query,
    tool,
)

from . import log as audit_log
from .prompts import (
    SUBMIT_VERDICT_DESCRIPTION,
    SUBMIT_VERDICT_INPUT_SCHEMA,
    SYSTEM_PROMPT,
    build_user_message,
)
from .session import SessionContext
from .verdict import Verdict

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_TIMEOUT_S = 60.0  # CLI cold start ~10s first time, ~2-3s warm; Haiku adds 1-3s
DEFAULT_MAX_TURNS = 2


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


def _build_options(captured: _Captured, model: str) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly",
        "1.0.0",
        [_build_submit_verdict_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"gadfly": server},
        allowed_tools=["mcp__gadfly__submit_verdict"],
        permission_mode="bypassPermissions",
        setting_sources=[],
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
    options = _build_options(captured, model)
    user_message = build_user_message(
        tool_name=tool_name,
        tool_input=tool_input,
        tool_response=tool_response,
        last_user_request=context.last_user_request,
        last_assistant_plan=context.last_assistant_plan,
        recent_actions=context.recent_actions,
    )
    system_prompt_sha = audit_log.ensure_system_prompt(SYSTEM_PROMPT)

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
