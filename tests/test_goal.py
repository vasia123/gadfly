"""Tests for the goal-distillation pipeline.

We test the pair extraction and cache logic exhaustively, and use a fake
run_query to exercise the distillation flow without spawning Claude CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gadfly import goal as g


# --- extract_pairs ----------------------------------------------------------


def _entries_from(messages: list[dict]) -> list[dict]:
    """Wrap raw message dicts into transcript-style entries."""
    return [{"message": m} for m in messages]


def test_extract_pairs_pairs_user_with_preceding_assistant():
    entries = _entries_from(
        [
            {"role": "user", "content": "design a benchmark plan"},
            {"role": "assistant", "content": [{"type": "text", "text": "I'll use the cargo bench harness"}]},
            {"role": "user", "content": "actually use criterion"},
        ]
    )
    pairs = g.extract_pairs(entries)
    assert len(pairs) == 2
    assert pairs[0].assistant_text is None  # nothing before the first user msg
    assert pairs[0].user_text == "design a benchmark plan"
    assert pairs[1].assistant_text == "I'll use the cargo bench harness"
    assert pairs[1].user_text == "actually use criterion"


def test_extract_pairs_skips_tool_result_only_user_messages():
    entries = _entries_from(
        [
            {"role": "user", "content": "implement X"},
            {"role": "assistant", "content": [{"type": "text", "text": "running tests"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "done"}]},
            {"role": "user", "content": "also handle edge case Y"},
        ]
    )
    pairs = g.extract_pairs(entries)
    assert [p.user_text for p in pairs] == ["implement X", "also handle edge case Y"]


def test_extract_pairs_skips_interrupted_marker():
    entries = _entries_from(
        [
            {"role": "user", "content": "do thing"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "[Request interrupted by user for tool use]"},
            {"role": "user", "content": "actually different thing"},
        ]
    )
    pairs = g.extract_pairs(entries)
    assert [p.user_text for p in pairs] == ["do thing", "actually different thing"]


def test_extract_pairs_filters_command_artifacts():
    """Claude Code injects synthetic user-role messages for slash commands
    and background-task notifications — they would otherwise dominate the
    pair list."""
    entries = _entries_from(
        [
            {"role": "user", "content": "do the thing"},
            {"role": "user", "content": "<command-name>/exit</command-name>\n<command-message>exit</command-message>"},
            {"role": "user", "content": "<local-command-stdout>Goodbye!</local-command-stdout>"},
            {"role": "user", "content": "<task-notification>...</task-notification>"},
            {"role": "user", "content": "<local-command-caveat>...</local-command-caveat>"},
            {"role": "user", "content": "real follow-up"},
        ]
    )
    pairs = g.extract_pairs(entries)
    assert [p.user_text for p in pairs] == ["do the thing", "real follow-up"]


def test_extract_pairs_strips_system_reminder_blocks():
    """System-reminders embedded inside user-role text blocks (e.g. the
    'A session-scoped Stop hook is now active …' Claude Code injects after
    /goal) must not leak into the pair list — otherwise the watchdog grades
    the agent against the reminder instead of real user intent."""
    entries = _entries_from(
        [
            {"role": "user", "content": "<system-reminder>\nA session-scoped Stop hook is now active with condition: …\n</system-reminder>"},
            {
                "role": "user",
                "content": (
                    "<system-reminder>some hook note</system-reminder>\n"
                    "actually use criterion"
                ),
            },
        ]
    )
    pairs = g.extract_pairs(entries)
    assert [p.user_text for p in pairs] == ["actually use criterion"]


def test_clean_user_text_drops_all_service_tags():
    """Unit-level: every kind of injected service tag must be stripped."""
    cases = [
        ("<command-name>/compact</command-name>", None),
        ("<local-command-stdout>...</local-command-stdout>", None),
        ("<local-command-caveat>foo</local-command-caveat>", None),
        ("<task-notification>tn</task-notification>", None),
        ("<system-reminder>multi\nline</system-reminder>", None),
        ("   \n  ", None),
        ("[Request interrupted by user]", None),
        ("real text", "real text"),
        ("<system-reminder>x</system-reminder>\nreal text", "real text"),
    ]
    for raw, expected in cases:
        assert g.clean_user_text(raw) == expected, f"input: {raw!r}"


def test_extract_pairs_deduplicates_consecutive_resends():
    entries = _entries_from(
        [
            {"role": "user", "content": "implement X"},
            {"role": "user", "content": "implement X"},  # exact repeat — skip
            {"role": "user", "content": "implement Y"},
        ]
    )
    pairs = g.extract_pairs(entries)
    assert [p.user_text for p in pairs] == ["implement X", "implement Y"]


def test_extract_pairs_assistant_text_belongs_to_one_pair_only():
    """An assistant message preceding two user messages should only attach
    to the first — not be reused for the second."""
    entries = _entries_from(
        [
            {"role": "assistant", "content": [{"type": "text", "text": "ask first"}]},
            {"role": "user", "content": "reply 1"},
            {"role": "user", "content": "reply 2"},
        ]
    )
    pairs = g.extract_pairs(entries)
    assert pairs[0].assistant_text == "ask first"
    assert pairs[1].assistant_text is None


# --- Cache + load_or_distill ------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect cache to a tmp dir so tests don't touch the user's real cache."""
    log_dir = tmp_path / "log"
    monkeypatch.setenv("GADFLY_LOG_DIR", str(log_dir))
    # `_cache_dir()` is computed at call time from env, so this is enough.
    return tmp_path / "goals"


def _fake_runner(captured_set_to: str | None):
    """Return a run_query that, when invoked, simulates Haiku calling
    submit_goal with `captured_set_to` (or doesn't, simulating a miss)."""

    async def runner(prompt, options):
        if captured_set_to is None:
            return  # Haiku failed to call the tool
        # Reach into the in-process MCP server we built and invoke the tool
        # handler directly with the value we want captured. We rely on the
        # SdkMcpTool exposing .handler.
        srv_cfg = options.mcp_servers["gadfly_goal"]
        # The handler closes over the captured container we want to write to;
        # easiest is to patch via the goal module's _build_submit_goal_tool —
        # see captured_ref fixture below.
        raise NotImplementedError("use captured_ref fixture instead")

    return runner


@pytest.fixture
def captured_ref(monkeypatch):
    """Expose the goal-distillation `_Captured` instance to the test so the
    fake runner can write into it."""
    holder: dict[str, g._Captured] = {}
    original = g._build_submit_goal_tool

    def patched(captured):
        holder["c"] = captured
        return original(captured)

    monkeypatch.setattr(g, "_build_submit_goal_tool", patched)
    return holder


def _runner_writing(captured_ref, goal_text: str | None):
    async def runner(prompt, options):
        if goal_text is not None:
            captured_ref["c"].goal = goal_text
    return runner


def test_load_or_distill_returns_none_for_empty_pairs(isolated_cache):
    state = g.load_or_distill(session_id="s1", pairs=[])
    assert state.goal is None
    assert state.raw_pairs == []


def test_load_or_distill_distils_and_caches(isolated_cache, captured_ref):
    pairs = [
        g.Pair(assistant_text=None, user_text="goal: implement watchdog"),
        g.Pair(assistant_text="I'll start with Edit", user_text="actually do prompts.py first"),
    ]
    state = g.load_or_distill(
        session_id="s1",
        pairs=pairs,
        run_query=_runner_writing(captured_ref, "implement watchdog, starting with prompts.py"),
    )
    assert state.error is None
    assert state.goal == "implement watchdog, starting with prompts.py"

    # Second call with the same pairs — must come from cache (no runner invocation).
    def must_not_run(prompt, options):
        raise AssertionError("cache miss: distillation re-ran")

    state2 = g.load_or_distill(session_id="s1", pairs=pairs, run_query=must_not_run)
    assert state2.goal == state.goal


def test_load_or_distill_incrementally_passes_prior_goal(isolated_cache, captured_ref):
    """When a new pair appears after a cached goal, the prior goal must be
    sent into the next distillation call."""
    pairs1 = [g.Pair(assistant_text=None, user_text="first goal")]
    g.load_or_distill(
        session_id="s1",
        pairs=pairs1,
        run_query=_runner_writing(captured_ref, "goal A"),
    )

    # Capture the prompt the second call receives.
    seen_prompts: list[str] = []

    async def capture(prompt, options):
        seen_prompts.append(prompt)
        captured_ref["c"].goal = "goal B"

    pairs2 = pairs1 + [g.Pair(assistant_text="ok", user_text="redirect: focus on Y")]
    state = g.load_or_distill(session_id="s1", pairs=pairs2, run_query=capture)
    assert state.goal == "goal B"
    assert len(seen_prompts) == 1
    # The prompt must mention the prior goal "goal A".
    assert "goal A" in seen_prompts[0]
    # And only the *new* pair (not pair 1) should be sent.
    assert "redirect: focus on Y" in seen_prompts[0]
    assert "first goal" not in seen_prompts[0]


def test_load_or_distill_invalidates_on_system_prompt_change(isolated_cache, captured_ref, monkeypatch):
    """When SYSTEM_PROMPT is edited, cached goals must be regenerated from
    scratch so users see distillations under the current rubric."""
    pairs = [g.Pair(assistant_text=None, user_text="initial goal")]
    g.load_or_distill(
        session_id="s-prompt",
        pairs=pairs,
        run_query=_runner_writing(captured_ref, "old rubric goal"),
    )

    # Now pretend SYSTEM_PROMPT has been edited (different sha).
    monkeypatch.setattr(g, "SYSTEM_PROMPT", g.SYSTEM_PROMPT + "\n# rubric tweaked\n")

    captured_prompts: list[str] = []

    async def capture(prompt, options):
        captured_prompts.append(prompt)
        captured_ref["c"].goal = "new rubric goal"

    state = g.load_or_distill(session_id="s-prompt", pairs=pairs, run_query=capture)
    assert state.goal == "new rubric goal"
    # Critically: this is treated as cold (no prior_goal), so the
    # earlier "old rubric goal" was NOT carried forward.
    assert "old rubric goal" not in captured_prompts[0]


def test_load_or_distill_invalidates_when_prefix_changes(isolated_cache, captured_ref):
    """If the transcript was rewritten (different earlier pair), the cache
    must be invalidated and the new pair set distilled from scratch."""
    pairs_old = [g.Pair(assistant_text=None, user_text="original goal")]
    g.load_or_distill(
        session_id="s1",
        pairs=pairs_old,
        run_query=_runner_writing(captured_ref, "old goal"),
    )

    pairs_new = [g.Pair(assistant_text=None, user_text="different first message")]
    seen_prompts: list[str] = []

    async def capture(prompt, options):
        seen_prompts.append(prompt)
        captured_ref["c"].goal = "new goal"

    state = g.load_or_distill(session_id="s1", pairs=pairs_new, run_query=capture)
    assert state.goal == "new goal"
    # The new distillation must not have been told about "old goal" — the
    # cache was invalidated.
    assert "old goal" not in seen_prompts[0]


def test_load_or_distill_logs_audit_event_on_success(isolated_cache, captured_ref, tmp_path):
    """Each successful distillation must write a goal_distill record to the
    session's audit log so the viewer can show it."""
    pairs = [g.Pair(assistant_text=None, user_text="goal one")]
    g.load_or_distill(
        session_id="audit-1",
        pairs=pairs,
        run_query=_runner_writing(captured_ref, "distilled goal"),
    )
    # Audit log location is rooted at GADFLY_LOG_DIR (set by isolated_cache).
    log_path = tmp_path / "log" / "audit-1.jsonl"
    assert log_path.is_file(), "audit log was not written"
    import json
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    distill_records = [r for r in records if r.get("type") == "goal_distill"]
    assert len(distill_records) == 1
    rec = distill_records[0]
    assert rec["goal"] == "distilled goal"
    assert rec["error"] is None
    assert rec["pairs_new"] == 1
    assert rec["pairs_cached"] == 0


def test_load_or_distill_logs_audit_event_on_error(isolated_cache, captured_ref, tmp_path):
    pairs = [g.Pair(assistant_text=None, user_text="goal one")]

    async def boom(prompt, options):
        raise RuntimeError("network down")

    g.load_or_distill(session_id="audit-2", pairs=pairs, run_query=boom)
    import json
    log_path = tmp_path / "log" / "audit-2.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    distill_records = [r for r in records if r.get("type") == "goal_distill"]
    assert len(distill_records) == 1
    assert distill_records[0]["goal"] is None
    assert "network down" in distill_records[0]["error"]


def test_load_or_distill_no_audit_event_on_pure_cache_hit(isolated_cache, captured_ref, tmp_path):
    """When nothing new is happening (cache hit), do not flood the log."""
    pairs = [g.Pair(assistant_text=None, user_text="goal one")]
    g.load_or_distill(
        session_id="audit-3",
        pairs=pairs,
        run_query=_runner_writing(captured_ref, "first goal"),
    )
    # Second call with identical pairs — cache hit, no new event expected.
    g.load_or_distill(
        session_id="audit-3",
        pairs=pairs,
        run_query=_runner_writing(captured_ref, "should not run"),
    )
    import json
    log_path = tmp_path / "log" / "audit-3.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    distill_records = [r for r in records if r.get("type") == "goal_distill"]
    assert len(distill_records) == 1


def test_load_or_distill_failure_falls_back_to_raw_pairs(isolated_cache, captured_ref):
    pairs = [g.Pair(assistant_text=None, user_text="implement X")]

    async def failing(prompt, options):
        # Don't write captured.goal — simulate Haiku not calling submit_goal.
        return

    state = g.load_or_distill(session_id="s1", pairs=pairs, run_query=failing)
    assert state.goal is None
    assert state.raw_pairs == pairs
    assert state.error and "submit_goal" in state.error


def test_load_or_distill_runner_exception_is_swallowed(isolated_cache, captured_ref):
    pairs = [g.Pair(assistant_text=None, user_text="implement X")]

    async def boom(prompt, options):
        raise RuntimeError("network down")

    state = g.load_or_distill(session_id="s1", pairs=pairs, run_query=boom)
    assert state.goal is None
    assert state.error and "network down" in state.error


# --- Junk filters in pair extraction ----------------------------------------


def test_pair_to_cache_hash_is_stable_and_distinguishes_content():
    a = g.Pair(assistant_text="hello", user_text="world")
    b = g.Pair(assistant_text="hello", user_text="world")
    c = g.Pair(assistant_text=None, user_text="world")
    assert a.to_cache_hash() == b.to_cache_hash()
    assert a.to_cache_hash() != c.to_cache_hash()
