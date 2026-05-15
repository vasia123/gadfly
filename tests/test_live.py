"""End-to-end live tests against the real Claude Code CLI + Haiku.

These tests are SKIPPED by default. To run:

    GADFLY_LIVE=1 .venv/bin/python -m pytest -m live

They bill against the user's Claude subscription, so they are gated. Each
test is designed to catch a specific class of regression that pure unit
tests cannot, because the bug only manifests once the SDK actually talks
to the CLI:

  * ThinkingConfigDisabled() called without `type=` produces `{}` (it's
    a TypedDict, not a dataclass) and the SDK explodes with
    `KeyError('type')` only at the moment of the inner-CLI handshake.
    A unit test cannot see this — it has to be a live call.
  * Likewise for any other config that looks fine at construction time
    but fails on the wire.
"""

from __future__ import annotations

import os

import pytest

from gadfly.session import SessionContext
from gadfly.watchdog import evaluate
from gadfly import goal as goal_mod


live = pytest.mark.live

pytestmark = pytest.mark.skipif(
    os.environ.get("GADFLY_LIVE") != "1",
    reason="set GADFLY_LIVE=1 to run live tests (bills the real subscription)",
)


@live
def test_evaluate_completes_against_real_haiku_without_sdk_errors():
    """Smoke: evaluate() on a trivial professional action must come back
    with a structured verdict, error=None. If the SDK is misconfigured
    (e.g. ThinkingConfigDisabled missing `type`), error will be a non-None
    string mentioning KeyError or TypeError and this test fails fast.
    """
    res = evaluate(
        tool_name="Bash",
        tool_input={"command": "git status", "description": "show status"},
        tool_response={"exit_code": 0, "stdout": "On branch main", "stderr": ""},
        context=SessionContext(
            recent_user_requests=["check current state of the repo"],
            last_assistant_plan=None,
            recent_actions=[],
        ),
    )
    # The verdict object always exists, but if the SDK crashed before
    # Haiku could call submit_verdict, error will hold the failure.
    assert res.error is None, f"SDK failed before Haiku could reply: {res.error}"
    # And running `git status` is not corner-cutting.
    assert res.verdict.professional is True, (
        f"expected professional=True for `git status`, got {res.verdict!r}"
    )
    # The prompt we sent must have been recorded for auditing.
    assert res.user_message
    assert res.system_prompt_sha


@live
def test_evaluate_thinking_disabled_does_not_leak_key_error():
    """Regression: when ThinkingConfigDisabled() was constructed without
    `type=`, it serialized as `{}` and the SDK raised
    `KeyError('type')` ~100ms into the call. This test would have caught
    that — error must not mention KeyError."""
    res = evaluate(
        tool_name="Bash",
        tool_input={"command": "echo ok"},
        tool_response={"exit_code": 0, "stdout": "ok"},
        context=SessionContext(recent_user_requests=["sanity check"]),
    )
    assert res.error is None or "KeyError" not in res.error, (
        f"SDK config regression — KeyError leaked: {res.error}"
    )
    assert res.error is None or "TypeError" not in res.error, (
        f"SDK config regression — TypeError leaked: {res.error}"
    )


@live
def test_goal_distillation_preserves_unresolved_earlier_bug(tmp_path, monkeypatch):
    """Regression: in session 7131d98b the user reported "bug is still there"
    several pairs after an earlier bug-fix request was acknowledged, and the
    agent had moved on to a different feature. Haiku collapsed the goal to
    the new feature and dropped the unresolved bug entirely. The unresolved-
    problem priority rule in SYSTEM_PROMPT must keep both visible."""
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    pairs = [
        goal_mod.Pair(
            assistant_text=None,
            user_text="phantom Telegram notification arrives 30 seconds after the agent stops — fix it",
        ),
        goal_mod.Pair(
            assistant_text="I added hooks={} to ClaudeAgentOptions, that should block inner CLI hooks.",
            user_text="ok, проверь что работает",
        ),
        goal_mod.Pair(
            assistant_text="all tests green, restarted viewer",
            user_text="теперь страница показывает ВСЕ события, нужна пагинация",
        ),
        goal_mod.Pair(
            assistant_text="pagination implemented, /api/session?limit=N, 63 tests passing",
            user_text="фантомное уведомление никуда не делось, баг ещё там",
        ),
    ]
    state = goal_mod.load_or_distill(session_id="live-unresolved", pairs=pairs)
    assert state.error is None, f"distillation failed: {state.error}"
    assert state.goal, "expected a non-empty distilled goal"
    g = state.goal.lower()
    # The unresolved bug must be back at the top.
    assert any(marker in g for marker in (
        "phantom", "notification", "telegram", "bug", "баг", "уведомлен",
    )), f"goal lost the unresolved bug entirely: {state.goal!r}"


@live
def test_goal_distillation_completes_against_real_haiku(tmp_path, monkeypatch):
    """Smoke: goal.load_or_distill against a small two-pair conversation
    must return a non-empty goal string with error=None. Catches the same
    class of SDK-config regressions as the evaluate() test, but for the
    goal module."""
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    pairs = [
        goal_mod.Pair(
            assistant_text=None,
            user_text="design a small Rust benchmark crate that catches performance regressions",
        ),
        goal_mod.Pair(
            assistant_text="I'll start with criterion benchmarks for the hot path.",
            user_text="yes, criterion is fine",
        ),
    ]
    state = goal_mod.load_or_distill(session_id="live-smoke", pairs=pairs)
    assert state.error is None, f"distillation failed: {state.error}"
    assert state.goal, "expected a non-empty distilled goal"
    # And it should mention something specific from the conversation.
    goal_lower = state.goal.lower()
    assert any(
        marker in goal_lower
        for marker in ("benchmark", "criterion", "regression", "rust")
    ), f"goal lost key nouns from the conversation: {state.goal!r}"
