"""claude-agent-sdk backend.

Spawns the user's authenticated Claude Code CLI under the hood. Bills
against subscription instead of an ANTHROPIC_API_KEY.

All three recursion guards from CLAUDE.md (section 3) are preserved
here verbatim — `setting_sources=[]`, `settings="{}"`, and
`env={"GADFLY_INTERNAL": "1"}`. Removing any of them re-enables the
inner CLI's PostToolUse loop and the watchdog recurses on itself.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ThinkingConfigDisabled,
    create_sdk_mcp_server,
    query,
    tool,
)

from .base import BackendResult


# turn 1: forced submit_verdict tool_use; turn 2: model receives the
# tool_result and closes out. max_turns=1 trips "Reached maximum
# number of turns" — empirically verified.
DEFAULT_MAX_TURNS = 2


@dataclass
class _Captured:
    """Closed-over container the submit_verdict tool writes into."""

    verdict_args: dict[str, Any] | None = None


def _build_submit_verdict_tool(
    captured: _Captured,
    *,
    tool_name: str,
    tool_description: str,
    mcp_input_schema: dict[str, Any],
):
    """Inline @tool factory. Translates the dict-of-classes schema the
    SDK wants from the canonical JSON Schema by reading the type names
    out of `mcp_input_schema` keys/types."""

    @tool(tool_name, tool_description, mcp_input_schema)
    async def submit_verdict(args: dict[str, Any]) -> dict[str, Any]:
        captured.verdict_args = args
        return {"content": [{"type": "text", "text": "verdict recorded"}]}

    return submit_verdict


def _json_schema_to_mcp(parameters: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style JSON Schema to the dict-of-classes shape
    `claude_agent_sdk.tool` expects: `{"prop_name": bool|str|int|...}`.

    Only top-level scalar types matter for the verdict tool (boolean +
    two strings) — defensive on the type field, falls back to `str`.
    """
    out: dict[str, Any] = {}
    props = parameters.get("properties", {}) or {}
    for name, spec in props.items():
        t = (spec or {}).get("type")
        if t == "boolean":
            out[name] = bool
        elif t == "integer":
            out[name] = int
        elif t == "number":
            out[name] = float
        else:
            out[name] = str
    return out


RunQuery = Callable[[str, ClaudeAgentOptions], Awaitable[None]]


async def _default_run_query(prompt: str, options: ClaudeAgentOptions) -> None:
    async for _ in query(prompt=prompt, options=options):
        # Drive the iterator to dispatch the submit_verdict callback.
        # We don't care about message content — verdict is captured
        # via the tool's side channel.
        pass


@dataclass
class ClaudeSDKBackend:
    """Backend running on claude-agent-sdk + user's CLI subscription."""

    run_query: RunQuery = _default_run_query
    max_turns: int = DEFAULT_MAX_TURNS
    name: str = field(default="claude_sdk", init=False)

    def _build_options(
        self,
        *,
        captured: _Captured,
        model: str,
        system_prompt: str,
        tool_name: str,
        tool_description: str,
        mcp_input_schema: dict[str, Any],
    ) -> ClaudeAgentOptions:
        server = create_sdk_mcp_server(
            "gadfly",
            "1.0.0",
            [
                _build_submit_verdict_tool(
                    captured,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    mcp_input_schema=mcp_input_schema,
                )
            ],
        )
        return ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            mcp_servers={"gadfly": server},
            allowed_tools=[f"mcp__gadfly__{tool_name}"],
            permission_mode="bypassPermissions",
            # Recursion guards — see CLAUDE.md section 3.
            setting_sources=[],
            settings="{}",
            thinking=ThinkingConfigDisabled(type="disabled"),
            max_turns=self.max_turns,
            env={"GADFLY_INTERNAL": "1"},
        )

    async def evaluate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        model: str,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict[str, Any],
        timeout_s: float,
    ) -> BackendResult:
        captured = _Captured()
        mcp_input_schema = _json_schema_to_mcp(tool_parameters)
        options = self._build_options(
            captured=captured,
            model=model,
            system_prompt=system_prompt,
            tool_name=tool_name,
            tool_description=tool_description,
            mcp_input_schema=mcp_input_schema,
        )
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                self.run_query(user_message, options), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            return BackendResult(
                verdict_args=None,
                error=f"timeout after {timeout_s}s",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except FileNotFoundError as exc:
            return BackendResult(
                verdict_args=None,
                error=f"claude CLI not found: {exc!s}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            return BackendResult(
                verdict_args=None,
                error=f"agent-sdk error: {exc!r}",
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        dt = (time.monotonic() - t0) * 1000
        if captured.verdict_args is None:
            return BackendResult(
                verdict_args=None,
                error=f"model did not call {tool_name}",
                latency_ms=dt,
            )
        return BackendResult(
            verdict_args=captured.verdict_args, error=None, latency_ms=dt
        )
