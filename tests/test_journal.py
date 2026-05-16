"""Tests for journal.py.

The Haiku call is mocked via the `run_query` DI parameter — same pattern
as test_watchdog.py and test_goal.py. We patch `_build_update_tool` to
expose the `_Captured` container so our fake runner can drop a payload
into it as if Haiku had invoked the `update_journal` MCP tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gadfly import journal as j
from gadfly.pairs import Pair


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect both the audit log and the journal-current dirs into tmp_path."""
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    return tmp_path


@pytest.fixture
def captured_ref(monkeypatch: pytest.MonkeyPatch):
    """Expose the `_Captured` instance the maintainer created."""
    holder: dict[str, j._Captured] = {}
    original = j._build_update_tool

    def patched(captured: j._Captured):
        holder["c"] = captured
        return original(captured)

    monkeypatch.setattr(j, "_build_update_tool", patched)
    return holder


def _runner_writing(captured_ref, payload: dict[str, Any] | None):
    """run_query that simulates Haiku invoking update_journal with payload.

    Pass payload=None to simulate Haiku failing to call the tool.
    """

    async def runner(prompt: str, options) -> None:
        if payload is not None:
            captured_ref["c"].payload = payload

    return runner


def _sample_payload(
    *,
    root_goal: str = "investigate marlin, fix inefficiencies",
    workstreams: list[dict[str, Any]] | None = None,
    drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "root_goal": root_goal,
        "workstreams": workstreams
        or [
            {
                "id": "marlin-exl3",
                "title": "investigate marlin EXL3 regression",
                "status": "open",
                "origin": "user msg #1",
                "notes": "not yet started",
                "watchdog_flags": 0,
                "flag_history": [],
                "last_touched": 1,
            }
        ],
        "drift": drift or {"initial_workstream_ids": [], "observations": ""},
    }


# --- Dataclass round-trip ----------------------------------------------------


def test_journal_to_dict_from_dict_round_trip():
    src = j.Journal(
        root_goal="x",
        workstreams=[
            j.Workstream(
                id="ws1",
                title="t",
                status="in-progress",
                origin="user msg",
                notes="some notes",
                watchdog_flags=2,
                flag_history=[
                    j.FlagEvent(action_index=5, reason="r", marker="symptom"),
                    j.FlagEvent(
                        action_index=7,
                        reason="r2",
                        marker="rationalization",
                        agent_pushed_back=True,
                        pushback="no",
                    ),
                ],
                last_touched=7,
            )
        ],
        drift=j.Drift(initial_workstream_ids=["ws1"], observations="obs"),
        action_index=7,
        prompt_sha="abc",
        consumed_pair_hashes=["h1", "h2"],
    )
    restored = j.Journal.from_dict(src.to_dict())
    assert restored.root_goal == "x"
    assert len(restored.workstreams) == 1
    w = restored.workstreams[0]
    assert w.id == "ws1"
    assert w.status == "in-progress"
    assert len(w.flag_history) == 2
    assert w.flag_history[1].agent_pushed_back is True
    assert restored.drift.initial_workstream_ids == ["ws1"]
    assert restored.consumed_pair_hashes == ["h1", "h2"]


def test_journal_status_coercion_is_safe():
    """Unknown statuses fall back to 'open' rather than raising."""
    w = j.Workstream.from_dict({"id": "x", "title": "t", "status": "totally-bogus"})
    assert w.status == "open"


def test_flag_marker_coercion_is_safe():
    f = j.FlagEvent.from_dict(
        {"action_index": 1, "reason": "r", "marker": "wat", "agent_pushed_back": False}
    )
    assert f.marker == "other"


# --- Persistence -------------------------------------------------------------


def test_load_current_returns_none_when_missing(isolated_paths):
    assert j.load_current("nonexistent-session") is None


def test_save_and_load_round_trip(isolated_paths):
    journal = j.empty_journal()
    journal.root_goal = "hello"
    j.save_current("s1", journal)
    loaded = j.load_current("s1")
    assert loaded is not None
    assert loaded.root_goal == "hello"
    assert loaded.prompt_sha == j._system_prompt_sha()


def test_load_current_invalidates_on_prompt_sha_mismatch(isolated_paths, monkeypatch):
    j.save_current("s1", j.empty_journal())
    # Forge the on-disk file with a stale prompt_sha — load must return None.
    p = j._current_path("s1")
    data = json.loads(p.read_text())
    data["prompt_sha"] = "stalestalestale"
    p.write_text(json.dumps(data))
    assert j.load_current("s1") is None


def test_load_current_invalidates_on_schema_version_mismatch(isolated_paths):
    j.save_current("s1", j.empty_journal())
    p = j._current_path("s1")
    data = json.loads(p.read_text())
    data["schema_version"] = 999
    p.write_text(json.dumps(data))
    assert j.load_current("s1") is None


def test_load_current_returns_none_for_garbage(isolated_paths):
    j._current_dir().mkdir(parents=True, exist_ok=True)
    j._current_path("s1").write_text("not json at all{{{")
    assert j.load_current("s1") is None


# --- update_for_action: happy path ------------------------------------------


def test_update_creates_journal_from_scratch(isolated_paths, captured_ref):
    pairs = [Pair(assistant_text=None, user_text="investigate marlin regression")]
    runner = _runner_writing(captured_ref, _sample_payload())
    res = j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=pairs,
        run_query=runner,
    )
    assert res.error is None
    assert res.skipped_reason is None
    assert res.journal.root_goal == "investigate marlin, fix inefficiencies"
    assert len(res.journal.workstreams) == 1
    assert res.journal.workstreams[0].id == "marlin-exl3"
    # Pair consumed.
    assert res.journal.consumed_pair_hashes == [pairs[0].to_cache_hash()]
    # Drift snapshot was taken (first time we have workstreams + a pair).
    assert res.journal.drift.initial_workstream_ids == ["marlin-exl3"]
    # diff_summary contains the creation event.
    assert any("created marlin-exl3" in d for d in res.diff)


def test_update_persists_to_disk(isolated_paths, captured_ref):
    pairs = [Pair(assistant_text=None, user_text="do X")]
    runner = _runner_writing(captured_ref, _sample_payload())
    j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=pairs,
        run_query=runner,
    )
    loaded = j.load_current("s1")
    assert loaded is not None
    assert loaded.root_goal == "investigate marlin, fix inefficiencies"


def test_update_no_change_skips_haiku_call(isolated_paths, captured_ref):
    """No new pairs, no action, no flag events → don't even call Haiku."""
    base = j.empty_journal()
    base.consumed_pair_hashes = ["preexisting"]
    j.save_current("s1", base)

    call_count = {"n": 0}

    async def runner(prompt, options):
        call_count["n"] += 1
        captured_ref["c"].payload = _sample_payload()

    res = j.update_for_action(
        session_id="s1",
        action_index=2,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[Pair(assistant_text=None, user_text="X")],  # but already consumed
        run_query=runner,
    )
    # The pair we passed has a different hash than "preexisting", so it
    # IS new — fix test by reusing the same hash. Easier: pass empty pairs.
    # (kept asserting on call_count for the empty-pairs path below.)
    # Reset and run the actual no-change case:
    call_count["n"] = 0
    res = j.update_for_action(
        session_id="s1",
        action_index=3,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[],
        run_query=runner,
    )
    assert call_count["n"] == 0
    assert res.skipped_reason == "no_change"


def test_update_appends_new_flag_event(isolated_paths, captured_ref):
    """When new_flag_events is supplied, the maintainer is told about it.
    We assert that Haiku's payload (which the test supplies) is honored —
    i.e. that the rendering does include the new_flag_events block."""
    pairs = [Pair(assistant_text=None, user_text="do X")]

    seen_prompts: list[str] = []

    async def runner(prompt, options):
        seen_prompts.append(prompt)
        captured_ref["c"].payload = _sample_payload()

    j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary="Edit(x.rs)",
        assistant_reasoning="I'll patch x.rs",
        pairs=pairs,
        new_flag_events=[
            j.FlagEvent(
                action_index=1,
                reason="symptom fix",
                marker="symptom",
                agent_pushed_back=False,
            )
        ],
        run_query=runner,
    )
    assert len(seen_prompts) == 1
    assert "New flag events to append" in seen_prompts[0]
    assert "symptom fix" in seen_prompts[0]


def test_update_carries_consumed_pair_hashes_forward(isolated_paths, captured_ref):
    """Across two updates with overlapping pair lists, only NEW pairs are
    advertised to Haiku."""
    p1 = Pair(assistant_text=None, user_text="goal 1")
    p2 = Pair(assistant_text="ok", user_text="and goal 2")

    seen_prompts: list[str] = []

    async def runner(prompt, options):
        seen_prompts.append(prompt)
        captured_ref["c"].payload = _sample_payload()

    j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[p1],
        run_query=runner,
    )
    j.update_for_action(
        session_id="s1",
        action_index=2,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[p1, p2],  # p1 already consumed
        run_query=runner,
    )
    assert "goal 1" in seen_prompts[0]
    # Second prompt only sees p2 in the "new pairs" block.
    second = seen_prompts[1]
    assert "and goal 2" in second
    # The "## New conversation pairs" section should contain only p2.
    if "## New conversation pairs since last update" in second:
        new_section = second.split("## New conversation pairs since last update")[1]
        new_section = new_section.split("## ")[0]  # next ## section ends the block
        assert "goal 1" not in new_section


# --- Failure modes -----------------------------------------------------------


def test_update_haiku_did_not_call_tool_returns_prior(isolated_paths, captured_ref):
    pairs = [Pair(assistant_text=None, user_text="X")]
    runner = _runner_writing(captured_ref, None)  # nothing captured
    res = j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=pairs,
        run_query=runner,
    )
    assert res.error and "did not call" in res.error
    # We keep the empty prior journal.
    assert res.journal.root_goal == ""


def test_update_runner_raises_returns_prior(isolated_paths, captured_ref):
    async def runner(prompt, options):
        raise RuntimeError("boom")

    res = j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[Pair(assistant_text=None, user_text="X")],
        run_query=runner,
    )
    assert res.error and "boom" in res.error


def test_update_filenotfound_handled(isolated_paths, captured_ref):
    async def runner(prompt, options):
        raise FileNotFoundError("claude")

    res = j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[Pair(assistant_text=None, user_text="X")],
        run_query=runner,
    )
    assert res.error and "claude CLI not found" in res.error


# --- Poisoning guard ---------------------------------------------------------


def test_update_refuses_dropping_more_than_half_workstreams(isolated_paths, captured_ref):
    base = j.empty_journal()
    base.workstreams = [
        j.Workstream(id=f"ws{i}", title=f"t{i}", status="open") for i in range(4)
    ]
    base.consumed_pair_hashes = ["seed"]
    j.save_current("s1", base)

    # Haiku tries to drop all 4 → 1, which is > 50%.
    payload = _sample_payload(
        workstreams=[
            {"id": "ws0", "title": "t0", "status": "open", "origin": "", "notes": "",
             "watchdog_flags": 0, "flag_history": [], "last_touched": 0}
        ],
    )
    runner = _runner_writing(captured_ref, payload)
    # Trigger an update — needs SOMETHING new to not hit the no_change path.
    res = j.update_for_action(
        session_id="s1",
        action_index=2,
        action_summary="Bash: ls",
        assistant_reasoning=None,
        pairs=[],
        run_query=runner,
    )
    assert res.skipped_reason and "refused" in res.skipped_reason
    # All 4 workstreams remain.
    assert len(res.journal.workstreams) == 4


def test_update_refuses_radical_root_goal_rewrite_without_new_user_msg(isolated_paths, captured_ref):
    base = j.empty_journal()
    base.root_goal = "investigate the marlin EXL3 regression and fix benchmarks"
    base.workstreams = [j.Workstream(id="ws1", title="t")]
    j.save_current("s1", base)

    payload = _sample_payload(
        root_goal="implement a new chat UI",  # totally different
        workstreams=[
            {"id": "ws1", "title": "t", "status": "open", "origin": "", "notes": "",
             "watchdog_flags": 0, "flag_history": [], "last_touched": 0}
        ],
    )
    runner = _runner_writing(captured_ref, payload)
    res = j.update_for_action(
        session_id="s1",
        action_index=2,
        action_summary="Bash: ls",
        assistant_reasoning=None,
        pairs=[],  # NO new pairs → no new user message
        run_query=runner,
    )
    assert res.skipped_reason and "radical" in res.skipped_reason
    assert res.journal.root_goal.startswith("investigate the marlin")


def test_update_accepts_radical_root_goal_when_new_user_msg(isolated_paths, captured_ref):
    base = j.empty_journal()
    base.root_goal = "investigate the marlin EXL3 regression and fix benchmarks"
    base.workstreams = [j.Workstream(id="ws1", title="t")]
    j.save_current("s1", base)

    payload = _sample_payload(
        root_goal="implement a new chat UI",
        workstreams=[
            {"id": "ws-new", "title": "chat UI", "status": "open", "origin": "", "notes": "",
             "watchdog_flags": 0, "flag_history": [], "last_touched": 2}
        ],
    )
    runner = _runner_writing(captured_ref, payload)
    new_pair = Pair(assistant_text=None, user_text="now do something completely different")
    res = j.update_for_action(
        session_id="s1",
        action_index=2,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[new_pair],
        run_query=runner,
    )
    assert res.skipped_reason is None  # accepted because had_new_user_msg=True
    assert res.journal.root_goal == "implement a new chat UI"


# --- Caps --------------------------------------------------------------------


def test_flag_history_trimmed_to_cap(isolated_paths, captured_ref):
    long_history = [
        {
            "action_index": i,
            "reason": f"r{i}",
            "marker": "symptom",
            "agent_pushed_back": False,
        }
        for i in range(20)
    ]
    payload = _sample_payload(
        workstreams=[
            {
                "id": "ws1",
                "title": "t",
                "status": "open",
                "origin": "",
                "notes": "",
                "watchdog_flags": 20,
                "flag_history": long_history,
                "last_touched": 20,
            }
        ],
    )
    runner = _runner_writing(captured_ref, payload)
    res = j.update_for_action(
        session_id="s1",
        action_index=20,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[Pair(assistant_text=None, user_text="x")],
        run_query=runner,
    )
    assert len(res.journal.workstreams[0].flag_history) == j.MAX_FLAG_HISTORY_PER_WS


# --- Audit log ---------------------------------------------------------------


def test_update_writes_journal_event_to_audit_log(isolated_paths, captured_ref):
    runner = _runner_writing(captured_ref, _sample_payload())
    j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[Pair(assistant_text=None, user_text="x")],
        run_query=runner,
    )
    log_path = Path(isolated_paths) / "log" / "s1.jsonl"
    assert log_path.is_file()
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    j_events = [r for r in records if r.get("type") == "journal_update"]
    assert len(j_events) == 1
    ev = j_events[0]
    assert ev["action_index"] == 1
    assert ev["error"] is None
    assert ev["new_journal_sha"]
    # The snapshot file actually exists.
    snap_dir = Path(isolated_paths) / "journals"
    assert (snap_dir / f"{ev['new_journal_sha']}.json").is_file()


def test_no_change_skip_still_writes_audit_event(isolated_paths, captured_ref):
    """We log even the no-op skips so viewer can show 'observed but unchanged'."""

    async def runner(prompt, options):
        raise AssertionError("should not be called")

    j.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=[],
        run_query=runner,
    )
    log_path = Path(isolated_paths) / "log" / "s1.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert any(r.get("type") == "journal_update" and r.get("skipped_reason") == "no_change" for r in records)
