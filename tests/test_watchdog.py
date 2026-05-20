"""Tests for watchdog.evaluate — the claude-agent-sdk call is mocked via the
`run_query` dependency-injection parameter so no CLI ever spawns."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from gadfly import watchdog
from gadfly.backends import claude_sdk as sdk_backend
from gadfly.session import SessionContext


async def _runner_that_emits(verdict_args: dict[str, Any]):
    """Return a run_query that simulates Haiku invoking the submit_verdict tool.

    The trick: the mcp_servers config is built by `_build_options` and the
    submit_verdict tool's handler closes over `captured`. To reach that
    handler from a fake `run_query`, we pull the SdkMcpServer instance out of
    options and call the registered tool handler directly through the MCP
    Server's internal registry.
    """

    async def runner(prompt: str, options):
        # Get the MCP server instance the watchdog built. The fake will
        # invoke the submit_verdict handler with the given args, exactly as
        # Haiku would have via the CLI.
        server_config = options.mcp_servers["gadfly"]
        server = server_config["instance"]
        # mcp.server.lowlevel.Server keeps its registered tool handlers in
        # `request_handlers`; the easier path is to walk our own list_tools
        # / call_tool. But we don't have direct access to the handler list,
        # so we monkeypatch from the watchdog's own internals: re-attach the
        # capture via _Captured -- the handler we built was given a closure
        # over it, but we can't reach `captured` from here either.
        #
        # Pragmatic solution: keep a side channel — the test runner accepts
        # the captured ref via a marker. See _runner_that_captures below.
        raise NotImplementedError("use _runner_that_captures instead")

    return runner


def _runner_that_captures(verdict_args: dict[str, Any] | None):
    """Build a `run_query` that writes `verdict_args` into the watchdog's
    `_Captured` container, simulating Haiku's submit_verdict call.

    Trick: we patch `_build_submit_verdict_tool` so we can grab a reference
    to the `captured` container at construction time. Implemented in the
    fixture below.
    """

    async def runner(prompt: str, options):
        # The captured container has already been written by the patched
        # _build_submit_verdict_tool below; nothing to do here. If
        # verdict_args is None we simulate "Haiku failed to call the tool".
        pass

    return runner


@pytest.fixture
def captured_ref(monkeypatch):
    """Expose the `_Captured` container that ClaudeSDKBackend created.

    The backend's _build_submit_verdict_tool builds an MCP tool closing
    over the _Captured; we monkeypatch that builder to side-channel the
    container into the test so the fake `run_query` can write a verdict
    into it the way the real CLI would."""
    holder: dict[str, sdk_backend._Captured] = {}
    original = sdk_backend._build_submit_verdict_tool

    def patched(captured, **kwargs):
        holder["c"] = captured
        return original(captured, **kwargs)

    monkeypatch.setattr(sdk_backend, "_build_submit_verdict_tool", patched)
    return holder


def _make_runner(captured_ref, verdict_args: dict[str, Any] | None):
    """run_query that sets the captured verdict (or doesn't, simulating
    a Haiku response that omits the tool call)."""

    async def runner(prompt: str, options):
        if verdict_args is not None:
            captured_ref["c"].verdict_args = verdict_args

    return runner


def test_happy_path_professional_true(captured_ref):
    runner = _make_runner(captured_ref, {"professional": True, "reason": "", "suggestion": ""})
    res = watchdog.evaluate(
        tool_name="Edit",
        tool_input={"file_path": "x.py", "old_string": "a", "new_string": "b"},
        tool_response={"success": True},
        context=SessionContext(),
        run_query=runner,
    )
    assert res.error is None
    assert res.verdict.professional is True


def test_happy_path_professional_false(captured_ref):
    runner = _make_runner(
        captured_ref,
        {"professional": False, "reason": "stub instead of impl", "suggestion": "implement it"},
    )
    res = watchdog.evaluate(
        tool_name="Edit",
        tool_input={"file_path": "x.py", "old_string": "a", "new_string": "TODO"},
        tool_response={"success": True},
        context=SessionContext(recent_user_requests=["implement validate_token"]),
        run_query=runner,
    )
    assert res.error is None
    assert res.verdict.professional is False
    assert "stub" in res.verdict.reason
    assert res.verdict.suggestion == "implement it"


def test_haiku_did_not_call_tool(captured_ref):
    runner = _make_runner(captured_ref, None)  # nothing captured
    res = watchdog.evaluate(
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response={},
        context=SessionContext(),
        run_query=runner,
    )
    assert res.verdict.professional is True  # silent_ok
    assert res.error and "did not call" in res.error


def test_runner_raises_filenotfound(captured_ref):
    async def runner(prompt, options):
        raise FileNotFoundError("claude")

    res = watchdog.evaluate(
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response={},
        context=SessionContext(),
        run_query=runner,
    )
    assert res.verdict.professional is True
    assert res.error and "claude CLI not found" in res.error


def test_runner_raises_arbitrary_exception(captured_ref):
    async def runner(prompt, options):
        raise RuntimeError("boom")

    res = watchdog.evaluate(
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response={},
        context=SessionContext(),
        run_query=runner,
    )
    assert res.verdict.professional is True
    assert res.error and "boom" in res.error


def test_timeout(captured_ref):
    async def slow_runner(prompt, options):
        await asyncio.sleep(5)

    res = watchdog.evaluate(
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response={},
        context=SessionContext(),
        run_query=slow_runner,
        timeout_s=0.05,
    )
    assert res.verdict.professional is True
    assert res.error and "timeout" in res.error


def test_options_block_recursion(captured_ref):
    """The watchdog must wire setting_sources=[] and GADFLY_INTERNAL=1 so the
    inner CLI does not pick up gadfly's own hook."""
    captured_options: dict = {}

    async def runner(prompt, options):
        captured_options["opts"] = options
        captured_ref["c"].verdict_args = {"professional": True, "reason": "", "suggestion": ""}

    watchdog.evaluate(
        tool_name="Bash",
        tool_input={"command": "ls"},
        tool_response={},
        context=SessionContext(),
        run_query=runner,
    )
    opts = captured_options["opts"]
    assert opts.setting_sources == []
    # The actual recursion guard: --settings '{}' on the inner CLI
    # overrides whatever hooks would otherwise be inherited from
    # ~/.claude/settings.json. The SDK's `hooks` option is a no-op for
    # this — verified by reading subprocess_cli.py source.
    assert opts.settings == "{}"
    assert opts.env.get("GADFLY_INTERNAL") == "1"
    assert opts.permission_mode == "bypassPermissions"
    assert opts.allowed_tools == ["mcp__gadfly__submit_verdict"]
    assert opts.model == watchdog.DEFAULT_MODEL
