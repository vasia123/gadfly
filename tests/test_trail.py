"""Tests for trail.py.

Mirrors test_journal.py: the LLM call is mocked via the `run_query` DI
parameter. We patch `_build_update_tool` to expose the `_Captured`
container so our fake runner can drop a payload into it as if Haiku had
invoked the `update_trail` MCP tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gadfly import session as session_mod
from gadfly import trail as t
from gadfly.trail import Breadcrumb, DriftFlag, Trail


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    return tmp_path


@pytest.fixture
def captured_ref(monkeypatch: pytest.MonkeyPatch):
    """Expose the `_Captured` instance the maintainer created."""
    holder: dict[str, t._Captured] = {}
    original = t._build_update_tool

    def patched(captured: t._Captured):
        holder["c"] = captured
        return original(captured)

    monkeypatch.setattr(t, "_build_update_tool", patched)
    return holder


def _runner_writing(captured_ref, payload: dict[str, Any] | None):
    """run_query that simulates Haiku invoking update_trail with payload.

    Pass payload=None to simulate Haiku failing to call the tool.
    """

    async def runner(prompt: str, options) -> None:
        if payload is not None:
            captured_ref["c"].payload = payload

    return runner


# --- Dataclass round-trips --------------------------------------------------


def test_breadcrumb_roundtrip():
    b = Breadcrumb(
        action_index=3,
        breadcrumb_text="introduced HazardCatalog",
        abstraction_level="class",
        action_summary="Edit(hazards.go)",
        ts=12345.0,
    )
    b2 = Breadcrumb.from_dict(b.to_dict())
    assert b2 == b


def test_breadcrumb_level_coercion():
    """Unknown levels fall back to 'unclear' rather than raising."""
    b = Breadcrumb.from_dict({
        "action_index": 1,
        "breadcrumb_text": "x",
        "abstraction_level": "totally-bogus",
        "action_summary": "Edit(a.go)",
    })
    assert b.abstraction_level == "unclear"


def test_drift_flag_roundtrip():
    f = DriftFlag(
        action_index=4,
        drift_kind="hardcoded_instance",
        drift_reasoning="agent patched fire literal",
        cited_action_indexes=[1, 2, 3],
        suppressed=False,
        delivered_to_agent=True,
        ts=9.0,
    )
    f2 = DriftFlag.from_dict(f.to_dict())
    assert f2 == f


def test_drift_kind_coercion():
    f = DriftFlag.from_dict({
        "drift_kind": "wat",
        "cited_action_indexes": [],
    })
    assert f.drift_kind == "other"


def test_drift_cited_int_coercion_skips_garbage():
    f = DriftFlag.from_dict({
        "drift_kind": "rule_skip",
        "cited_action_indexes": [1, "2", "abc", None, 3.7],
    })
    # Valid ints + str-castables kept; None and floats coerce-or-skip.
    assert 1 in f.cited_action_indexes
    assert 2 in f.cited_action_indexes


def test_trail_roundtrip():
    trail = Trail(
        breadcrumbs=[
            Breadcrumb(1, "x", "instance", "Edit(a.go)", 1.0),
        ],
        drift_flags=[
            DriftFlag(2, "hardcoded_instance", "r", [1], False, True, 2.0),
        ],
        action_index=2,
        prompt_sha="sha",
        non_advance_streak=3,
        last_root_goal="goal x",
        ts=2.5,
    )
    restored = Trail.from_dict(json.loads(trail.to_json()))
    assert restored.breadcrumbs == trail.breadcrumbs
    assert restored.drift_flags == trail.drift_flags
    assert restored.action_index == 2
    assert restored.non_advance_streak == 3
    assert restored.last_root_goal == "goal x"


# --- Persistence ------------------------------------------------------------


def test_load_current_returns_none_when_missing(isolated_paths):
    assert t.load_current("nonexistent") is None


def test_save_load_roundtrip(isolated_paths):
    trail = t.empty_trail()
    trail.breadcrumbs.append(
        Breadcrumb(1, "x", "class", "Edit(a.go)", 1.0)
    )
    t.save_current("s1", trail)
    loaded = t.load_current("s1")
    assert loaded is not None
    assert len(loaded.breadcrumbs) == 1
    assert loaded.breadcrumbs[0].abstraction_level == "class"


def test_load_current_invalidates_on_prompt_sha_mismatch(isolated_paths):
    t.save_current("s1", t.empty_trail())
    p = t._current_path("s1")
    data = json.loads(p.read_text())
    data["prompt_sha"] = "stalestalestale"
    p.write_text(json.dumps(data))
    assert t.load_current("s1") is None


def test_load_current_invalidates_on_schema_version_mismatch(isolated_paths):
    t.save_current("s1", t.empty_trail())
    p = t._current_path("s1")
    data = json.loads(p.read_text())
    data["schema_version"] = 999
    p.write_text(json.dumps(data))
    assert t.load_current("s1") is None


def test_load_current_handles_garbage(isolated_paths):
    t._current_dir().mkdir(parents=True, exist_ok=True)
    t._current_path("s1").write_text("not json at all{{{")
    assert t.load_current("s1") is None


# --- _apply_payload — advance vs repeat -------------------------------------


def test_advance_appends_breadcrumb(isolated_paths):
    base = t.empty_trail()
    new, drift, sr = t._apply_payload(
        base=base,
        payload={
            "advances_trail": True,
            "breadcrumb_text": "introduced HazardCatalog",
            "abstraction_level": "class",
            "drift_detected": False,
        },
        action_index=1,
        action_summary="Edit(hazards.go)",
        journal_root_goal="goal",
        redirect=False,
    )
    assert len(new.breadcrumbs) == 1
    assert new.breadcrumbs[0].abstraction_level == "class"
    assert new.non_advance_streak == 0
    assert drift is None
    assert sr is None


def test_repeat_does_not_append(isolated_paths):
    base = t.empty_trail()
    new, _, _ = t._apply_payload(
        base=base,
        payload={"advances_trail": False, "drift_detected": False},
        action_index=1,
        action_summary="Edit(a.go)",
        journal_root_goal="goal",
        redirect=False,
    )
    assert len(new.breadcrumbs) == 0
    assert new.non_advance_streak == 1


def test_advance_resets_non_advance_streak():
    base = t.empty_trail()
    base.non_advance_streak = 3
    new, _, _ = t._apply_payload(
        base=base,
        payload={
            "advances_trail": True,
            "breadcrumb_text": "moved up",
            "abstraction_level": "architecture",
            "drift_detected": False,
        },
        action_index=10,
        action_summary="Edit(arch.go)",
        journal_root_goal="goal",
        redirect=False,
    )
    assert new.non_advance_streak == 0


def test_advance_with_no_text_synthesizes_from_action():
    base = t.empty_trail()
    new, _, _ = t._apply_payload(
        base=base,
        payload={
            "advances_trail": True,
            "breadcrumb_text": None,
            "drift_detected": False,
        },
        action_index=1,
        action_summary="Edit(fallback.go)",
        journal_root_goal=None,
        redirect=False,
    )
    assert len(new.breadcrumbs) == 1
    assert "Edit(fallback.go)" in new.breadcrumbs[0].breadcrumb_text


# --- Drift handling ---------------------------------------------------------


def test_drift_without_citations_is_suppressed():
    base = t.empty_trail()
    base.breadcrumbs.append(Breadcrumb(1, "x", "instance", "Edit(a)", 1.0))
    new, drift, sr = t._apply_payload(
        base=base,
        payload={
            "advances_trail": False,
            "drift_detected": True,
            "drift_kind": "hardcoded_instance",
            "drift_reasoning": "vague claim",
            "cited_action_indexes": [],
        },
        action_index=2,
        action_summary="Edit(b)",
        journal_root_goal="g",
        redirect=False,
    )
    assert drift is not None
    assert drift.suppressed is True
    assert drift.delivered_to_agent is False
    assert "without citations" in (sr or "")
    # The flag is still persisted into the trail for audit visibility.
    assert len(new.drift_flags) == 1


def test_drift_with_citations_is_delivered():
    base = t.empty_trail()
    for i in range(3):
        base.breadcrumbs.append(Breadcrumb(i + 1, "x", "instance", "Edit", float(i)))
    new, drift, _ = t._apply_payload(
        base=base,
        payload={
            "advances_trail": False,
            "drift_detected": True,
            "drift_kind": "hardcoded_instance",
            "drift_reasoning": "3 consecutive instance patches",
            "cited_action_indexes": [1, 2, 3],
        },
        action_index=4,
        action_summary="Edit(d.go)",
        journal_root_goal="g",
        redirect=False,
    )
    assert drift is not None
    assert drift.suppressed is False
    assert drift.delivered_to_agent is True
    assert drift.cited_action_indexes == [1, 2, 3]


def test_drift_same_kind_in_window_is_suppressed():
    """K-window repetition rule: same drift_kind in last K → suppress."""
    base = t.empty_trail()
    base.drift_flags.append(
        DriftFlag(1, "hardcoded_instance", "first", [1], False, True, 1.0)
    )
    new, drift, sr = t._apply_payload(
        base=base,
        payload={
            "advances_trail": False,
            "drift_detected": True,
            "drift_kind": "hardcoded_instance",
            "drift_reasoning": "another instance patch",
            "cited_action_indexes": [2, 3],
        },
        action_index=4,
        action_summary="Edit(e.go)",
        journal_root_goal="g",
        redirect=False,
    )
    assert drift is not None
    assert drift.suppressed is True
    assert "same kind hardcoded_instance" in (sr or "")


def test_drift_different_kind_in_window_is_not_suppressed():
    base = t.empty_trail()
    base.drift_flags.append(
        DriftFlag(1, "hardcoded_instance", "first", [1], False, True, 1.0)
    )
    new, drift, _ = t._apply_payload(
        base=base,
        payload={
            "advances_trail": False,
            "drift_detected": True,
            "drift_kind": "rule_skip",  # different kind
            "drift_reasoning": "ignored CLAUDE.md",
            "cited_action_indexes": [3],
        },
        action_index=4,
        action_summary="Edit(f.go)",
        journal_root_goal="g",
        redirect=False,
    )
    assert drift is not None
    assert drift.suppressed is False
    assert drift.delivered_to_agent is True


def test_drift_window_reset_on_root_goal_redirect():
    """User redirect (root_goal changed) → K-window doesn't apply."""
    base = t.empty_trail()
    base.drift_flags.append(
        DriftFlag(1, "hardcoded_instance", "first", [1], False, True, 1.0)
    )
    # `redirect=True` mimics what update_for_action computes when
    # journal.root_goal differs from base.last_root_goal.
    new, drift, _ = t._apply_payload(
        base=base,
        payload={
            "advances_trail": False,
            "drift_detected": True,
            "drift_kind": "hardcoded_instance",
            "drift_reasoning": "fresh start, still pattern",
            "cited_action_indexes": [2, 3],
        },
        action_index=4,
        action_summary="Edit(g.go)",
        journal_root_goal="completely new goal",
        redirect=True,
    )
    # Despite same drift_kind in window, redirect bypasses suppression.
    assert drift is not None
    assert drift.suppressed is False
    assert drift.delivered_to_agent is True


# --- Stall guard ------------------------------------------------------------


def test_stall_guard_fires_after_5_non_advancing(isolated_paths):
    base = t.empty_trail()
    for i in range(5):
        base, _, _ = t._apply_payload(
            base=base,
            payload={"advances_trail": False, "drift_detected": False},
            action_index=10 + i,
            action_summary=f"Bash: ls {i}",
            journal_root_goal="g",
            redirect=False,
        )
    # On the 5th non-advance, stall guard forces an `unclear` breadcrumb.
    assert any(
        b.abstraction_level == "unclear" and "stall-guard" in b.breadcrumb_text
        for b in base.breadcrumbs
    )
    # Streak reset after forced breadcrumb.
    assert base.non_advance_streak == 0


def test_stall_guard_does_not_fire_below_threshold():
    base = t.empty_trail()
    for i in range(4):  # below threshold
        base, _, _ = t._apply_payload(
            base=base,
            payload={"advances_trail": False, "drift_detected": False},
            action_index=i + 1,
            action_summary=f"Read({i})",
            journal_root_goal="g",
            redirect=False,
        )
    assert len(base.breadcrumbs) == 0
    assert base.non_advance_streak == 4


# --- Caps -------------------------------------------------------------------


def test_breadcrumbs_capped_at_max():
    base = t.empty_trail()
    for i in range(t.MAX_BREADCRUMBS + 5):
        base, _, _ = t._apply_payload(
            base=base,
            payload={
                "advances_trail": True,
                "breadcrumb_text": f"crumb {i}",
                "abstraction_level": "class",
                "drift_detected": False,
            },
            action_index=i + 1,
            action_summary=f"Edit({i})",
            journal_root_goal="g",
            redirect=False,
        )
    assert len(base.breadcrumbs) == t.MAX_BREADCRUMBS
    # Most recent kept; oldest dropped.
    assert base.breadcrumbs[-1].breadcrumb_text == f"crumb {t.MAX_BREADCRUMBS + 4}"


# --- Failure paths ----------------------------------------------------------


def test_payload_none_increments_streak_and_logs_skip():
    base = t.empty_trail()
    new, drift, sr = t._apply_payload(
        base=base,
        payload=None,
        action_index=1,
        action_summary="Edit(a)",
        journal_root_goal="g",
        redirect=False,
    )
    assert drift is None
    assert new.non_advance_streak == 1
    assert sr == "no payload"


# --- update_for_action — full path with mocked runner -----------------------


def test_update_for_action_happy_path_advance(isolated_paths, captured_ref):
    runner = _runner_writing(
        captured_ref,
        {
            "advances_trail": True,
            "breadcrumb_text": "added HazardCatalog",
            "abstraction_level": "class",
            "drift_detected": False,
        },
    )
    res = t.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary="Edit(hazards.go)",
        assistant_reasoning="introducing the catalog",
        latest_user_message="add hazard system",
        journal_root_goal="add hazard system",
        run_query=runner,
    )
    assert res.error is None
    assert len(res.trail.breadcrumbs) == 1
    assert res.trail.breadcrumbs[0].abstraction_level == "class"
    assert res.drift_flag is None


def test_update_for_action_happy_path_drift(isolated_paths, captured_ref):
    # Seed prior trail with 3 instance breadcrumbs so drift is plausible.
    seed = t.empty_trail()
    for i in range(3):
        seed.breadcrumbs.append(
            Breadcrumb(i + 1, f"patch {i}", "instance", f"Edit({i})", 1.0)
        )
    t.save_current("s1", seed)

    runner = _runner_writing(
        captured_ref,
        {
            "advances_trail": False,
            "drift_detected": True,
            "drift_kind": "hardcoded_instance",
            "drift_reasoning": "3 instance patches with class slot visible",
            "cited_action_indexes": [1, 2, 3],
        },
    )
    res = t.update_for_action(
        session_id="s1",
        action_index=4,
        action_summary="Edit(another_instance.go)",
        assistant_reasoning="adding another case",
        latest_user_message="fix it",
        journal_root_goal="fix hazard",
        run_query=runner,
    )
    assert res.error is None
    assert res.drift_flag is not None
    assert res.drift_flag.drift_kind == "hardcoded_instance"
    assert res.drift_flag.delivered_to_agent is True


def test_update_for_action_handles_runner_returning_none(isolated_paths, captured_ref):
    runner = _runner_writing(captured_ref, None)
    res = t.update_for_action(
        session_id="s1",
        action_index=1,
        action_summary="Edit(x)",
        assistant_reasoning=None,
        latest_user_message=None,
        journal_root_goal=None,
        run_query=runner,
    )
    # We never raise — fail silently. trail saved with non_advance_streak=1.
    assert res.error is not None
    assert "did not call" in res.error
    loaded = t.load_current("s1")
    assert loaded is not None
    assert loaded.non_advance_streak == 1


def test_update_for_action_writes_audit_event(isolated_paths, captured_ref):
    runner = _runner_writing(
        captured_ref,
        {
            "advances_trail": True,
            "breadcrumb_text": "x",
            "abstraction_level": "class",
            "drift_detected": False,
        },
    )
    t.update_for_action(
        session_id="audit-sess",
        action_index=1,
        action_summary="Edit(a)",
        assistant_reasoning=None,
        latest_user_message=None,
        journal_root_goal=None,
        run_query=runner,
    )
    log_path = Path(isolated_paths) / "log" / "audit-sess.jsonl"
    assert log_path.is_file()
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    trail_events = [r for r in records if r.get("type") == "trail_update"]
    assert len(trail_events) == 1
    e = trail_events[0]
    assert e["advances_trail"] is True
    assert e["drift_detected"] is False
    assert e["new_trail_sha"]
    assert e["prior_trail_sha"]


# --- Snapshot SHA stability -------------------------------------------------


def test_snapshot_sha_stable_across_runs():
    trail_a = Trail(
        breadcrumbs=[Breadcrumb(1, "x", "class", "Edit(a)", 1.0)],
        drift_flags=[],
        action_index=1,
        prompt_sha="sha",
        ts=1.5,
    )
    trail_b = Trail.from_dict(json.loads(trail_a.to_json()))
    assert trail_a.to_json() == trail_b.to_json()


# --- question_for_kind ------------------------------------------------------


def test_question_for_kind_returns_canonical_for_each_kind():
    expected = {
        "hardcoded_instance", "premature_ceiling", "wrong_layer", "rule_skip",
        "incomplete_coverage", "recon_as_work", "rationalization", "other",
    }
    for k in expected:
        q = t.question_for_kind(k)
        assert isinstance(q, str) and len(q) > 50, k


def test_question_for_kind_falls_back_to_other_for_unknown():
    q1 = t.question_for_kind("totally-unknown")
    q2 = t.question_for_kind("other")
    assert q1 == q2


# --- Sidechain filter at session.load level --------------------------------


def test_session_load_skips_sidechain_entries(tmp_path: Path):
    """Verifies the load-time isSidechain filter doesn't pollute the
    SessionContext that trail.update_for_action will eventually consume.
    The fix lives in session.py but the trail rubric depends on it
    holding — main-agent breadcrumbs only.
    """
    transcript = tmp_path / "t.jsonl"
    # Main agent says hello; sidechain (subagent) edits a file; main agent
    # then runs grep. Only the main agent's entries should reach load().
    entries = [
        {"type": "user", "message": {"role": "user", "content": "build it"}},
        {
            "type": "assistant",
            "isSidechain": True,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Edit", "id": "x",
                     "input": {"file_path": "sub.go", "old_string": "a", "new_string": "b"}},
                ],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "Bash", "id": "y",
                     "input": {"command": "grep foo ."}},
                ],
            },
        },
    ]
    with transcript.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    ctx = session_mod.load(str(transcript), distill=False)
    # action_index counts main-agent tool_uses only.
    assert ctx.action_index == 1
    assert all("sub.go" not in a for a in ctx.recent_actions)
    assert any("grep foo" in a for a in ctx.recent_actions)
