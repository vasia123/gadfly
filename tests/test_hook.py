"""End-to-end hook tests: feed stdin, capture stdout, verify exit + log."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gadfly import hook
from gadfly.verdict import Verdict
from gadfly.watchdog import EvaluationResult


def _run_hook(monkeypatch, payload: dict | str) -> tuple[int, str]:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    rc = hook.main()
    return rc, buf.getvalue()


def test_hook_skips_when_disabled(monkeypatch, tmp_log_dir: Path):
    monkeypatch.setenv("GADFLY_DISABLE", "1")
    rc, out = _run_hook(monkeypatch, {"hook_event_name": "PostToolUse", "tool_name": "Bash"})
    assert rc == 0
    assert out == ""
    assert not list(tmp_log_dir.glob("*.jsonl"))


def test_hook_skips_empty_stdin(monkeypatch, tmp_log_dir: Path):
    rc, out = _run_hook(monkeypatch, "")
    assert rc == 0
    assert out == ""


def test_hook_skips_invalid_json(monkeypatch, tmp_log_dir: Path):
    rc, out = _run_hook(monkeypatch, "not json")
    assert rc == 0
    assert out == ""


def test_hook_skips_wrong_event(monkeypatch, tmp_log_dir: Path):
    rc, out = _run_hook(
        monkeypatch,
        {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {}},
    )
    assert rc == 0
    assert out == ""


def test_hook_skips_unwatched_tool(monkeypatch, tmp_log_dir: Path):
    rc, out = _run_hook(
        monkeypatch,
        {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/x"},
        },
    )
    assert rc == 0
    assert out == ""


def test_hook_silent_when_verdict_professional(monkeypatch, tmp_log_dir: Path):
    with patch.object(
        hook.watchdog,
        "evaluate",
        return_value=EvaluationResult(Verdict.silent_ok(), None),
    ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py", "old_string": "x", "new_string": "y"},
                "tool_response": {"success": True},
                "session_id": "s1",
                "transcript_path": "",
            },
        )
    assert rc == 0
    assert out == ""
    # Log file should still record the silent-ok verdict.
    log_files = list(tmp_log_dir.glob("s1.jsonl"))
    assert len(log_files) == 1
    record = json.loads(log_files[0].read_text().strip())
    assert record["verdict"]["professional"] is True
    assert record["tool_name"] == "Edit"


def test_hook_emits_additional_context_when_unprofessional(monkeypatch, tmp_log_dir: Path):
    verdict = Verdict(professional=False, reason="adds TODO", suggestion="implement it")
    with patch.object(
        hook.watchdog,
        "evaluate",
        return_value=EvaluationResult(verdict, None),
    ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py", "old_string": "x", "new_string": "TODO"},
                "tool_response": {"success": True},
                "session_id": "s2",
                "transcript_path": "",
            },
        )
    assert rc == 0
    payload = json.loads(out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "TODO" in payload["hookSpecificOutput"]["additionalContext"]
    log_files = list(tmp_log_dir.glob("s2.jsonl"))
    assert len(log_files) == 1
    rec = json.loads(log_files[0].read_text().strip())
    assert rec["verdict"]["professional"] is False


def test_hook_shadow_mode_swallows_additional_context(monkeypatch, tmp_log_dir: Path):
    """GADFLY_SHADOW=1: hook runs everything, writes audit log, but the
    additionalContext stays out of the agent's view. Used to validate
    the trail rubric on real sessions before flipping FEEDBACK on."""
    monkeypatch.setenv("GADFLY_SHADOW", "1")
    verdict = Verdict(professional=False, reason="symptom fix", suggestion="x")
    with patch.object(
        hook.watchdog,
        "evaluate",
        return_value=EvaluationResult(verdict, None),
    ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py", "old_string": "x", "new_string": "y"},
                "tool_response": {"success": True},
                "session_id": "shadow_sid",
                "transcript_path": "",
            },
        )
    assert rc == 0
    # Hook produced no agent-facing output despite an unprofessional verdict.
    assert out == ""
    # But the verdict was still written to the audit log for the viewer.
    log_files = list(tmp_log_dir.glob("shadow_sid.jsonl"))
    assert len(log_files) == 1
    rec = json.loads(log_files[0].read_text().strip())
    assert rec["verdict"]["professional"] is False


def test_hook_writes_heartbeat_for_historian(monkeypatch, tmp_log_dir: Path):
    """H2: every PostToolUse on a watched tool drops a heartbeat tick
    keyed by encoded cwd. The daemon picks these up to know which
    sessions to digest."""
    monkeypatch.setenv("GADFLY_HISTORIAN", "1")
    with patch.object(
        hook.watchdog,
        "evaluate",
        return_value=EvaluationResult(Verdict.silent_ok(), None),
    ):
        rc, _ = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py", "old_string": "x", "new_string": "y"},
                "tool_response": {"success": True},
                "session_id": "sH1",
                "transcript_path": "",
                "cwd": "/home/vasis/projects_hobby/gadfly",
            },
        )
    assert rc == 0
    hb_dir = tmp_log_dir.parent / "heartbeat"
    expected = hb_dir / "-home-vasis-projects-hobby-gadfly.tick"
    assert expected.is_file()
    payload = json.loads(expected.read_text())
    assert payload["session_id"] == "sH1"
    assert "ts" in payload
    assert "last_action_index" in payload


def test_hook_skips_heartbeat_when_historian_disabled(monkeypatch, tmp_log_dir: Path):
    monkeypatch.setenv("GADFLY_HISTORIAN", "0")
    with patch.object(
        hook.watchdog,
        "evaluate",
        return_value=EvaluationResult(Verdict.silent_ok(), None),
    ):
        rc, _ = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py"},
                "tool_response": {},
                "session_id": "sH2",
                "transcript_path": "",
                "cwd": "/some/cwd",
            },
        )
    assert rc == 0
    hb_dir = tmp_log_dir.parent / "heartbeat"
    # No heartbeat file should have been written.
    assert not hb_dir.exists() or not list(hb_dir.glob("*.tick"))


def test_hook_phase_c_injects_priors_on_workstream_creation(monkeypatch, tmp_log_dir: Path):
    """H9/Phase C: when journal opens a new workstream AND priors were
    consulted, the hook surfaces them to the AGENT via additionalContext."""
    from gadfly.journal import JournalUpdateResult, Journal
    from gadfly.historian import PriorHit

    fake_priors = [
        PriorHit(
            kind="correction",
            id="c1",
            title="don't mock the database in integration tests",
            body="reason: prior incident",
            evidence_quote="we got burned mocking the db",
            source_session="abc-12-345",
            score=0.7,
        )
    ]
    fake_update = JournalUpdateResult(
        journal=Journal(),
        error=None,
        skipped_reason=None,
        latency_ms=0.0,
        diff=["created ws_test: 'investigate db tests'"],
        priors_consulted=fake_priors,
    )

    monkeypatch.setenv("GADFLY_JOURNAL", "1")
    monkeypatch.setenv("GADFLY_PHASE_C", "1")

    with patch.object(hook.journal, "update_for_action", return_value=fake_update), \
         patch.object(
             hook.watchdog,
             "evaluate",
             return_value=EvaluationResult(Verdict.silent_ok(), None),
         ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py", "old_string": "x", "new_string": "y"},
                "tool_response": {},
                "session_id": "sC1",
                "transcript_path": "",
                "cwd": "/some/cwd",
            },
        )
    assert rc == 0
    # Phase C injects context even though verdict was silent.
    assert out, f"expected additionalContext from Phase C, got nothing"
    payload = json.loads(out)
    ctx_text = payload["hookSpecificOutput"]["additionalContext"]
    assert "gadfly historian" in ctx_text
    assert "don't mock the database in integration tests" in ctx_text
    assert "we got burned mocking the db" in ctx_text


def test_hook_phase_c_silent_when_no_workstream_created(monkeypatch, tmp_log_dir: Path):
    """No new workstream → no Phase C injection, even when priors exist."""
    from gadfly.journal import JournalUpdateResult, Journal
    from gadfly.historian import PriorHit

    fake_update = JournalUpdateResult(
        journal=Journal(),
        error=None,
        skipped_reason=None,
        latency_ms=0.0,
        diff=["ws_existing.flags 2→3"],  # no "created " line
        priors_consulted=[PriorHit(kind="correction", id="c1", title="t",
                                    body="b", evidence_quote="e",
                                    source_session="s", score=0.5)],
    )
    monkeypatch.setenv("GADFLY_JOURNAL", "1")
    monkeypatch.setenv("GADFLY_PHASE_C", "1")

    with patch.object(hook.journal, "update_for_action", return_value=fake_update), \
         patch.object(
             hook.watchdog,
             "evaluate",
             return_value=EvaluationResult(Verdict.silent_ok(), None),
         ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py"},
                "tool_response": {},
                "session_id": "sC2",
                "transcript_path": "",
                "cwd": "/some/cwd",
            },
        )
    assert rc == 0
    assert out == ""


def test_hook_phase_c_disabled_via_env(monkeypatch, tmp_log_dir: Path):
    from gadfly.journal import JournalUpdateResult, Journal
    from gadfly.historian import PriorHit

    fake_update = JournalUpdateResult(
        journal=Journal(), error=None, skipped_reason=None,
        latency_ms=0.0, diff=["created ws_x: 'thing'"],
        priors_consulted=[PriorHit(kind="promise", id="p1", title="t",
                                    body="b", evidence_quote="e",
                                    source_session="s", score=0.5)],
    )
    monkeypatch.setenv("GADFLY_JOURNAL", "1")
    monkeypatch.setenv("GADFLY_PHASE_C", "0")

    with patch.object(hook.journal, "update_for_action", return_value=fake_update), \
         patch.object(
             hook.watchdog,
             "evaluate",
             return_value=EvaluationResult(Verdict.silent_ok(), None),
         ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py"},
                "tool_response": {},
                "session_id": "sC3",
                "transcript_path": "",
                "cwd": "/some/cwd",
            },
        )
    assert rc == 0
    assert out == ""


def test_hook_phase_c_appends_to_existing_verdict_context(monkeypatch, tmp_log_dir: Path):
    """When verdict has its own additionalContext AND Phase C fires, both
    are joined."""
    from gadfly.journal import JournalUpdateResult, Journal
    from gadfly.historian import PriorHit

    fake_update = JournalUpdateResult(
        journal=Journal(), error=None, skipped_reason=None,
        latency_ms=0.0, diff=["created ws_y: 'feature'"],
        priors_consulted=[PriorHit(kind="correction", id="c1",
                                    title="use real db",
                                    body="b", evidence_quote="db",
                                    source_session="s", score=0.5)],
    )
    bad_verdict = Verdict(professional=False, reason="adds TODO",
                          suggestion="implement it")
    monkeypatch.setenv("GADFLY_JOURNAL", "1")
    monkeypatch.setenv("GADFLY_PHASE_C", "1")

    with patch.object(hook.journal, "update_for_action", return_value=fake_update), \
         patch.object(
             hook.watchdog,
             "evaluate",
             return_value=EvaluationResult(bad_verdict, None),
         ):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py", "old_string": "x", "new_string": "TODO"},
                "tool_response": {},
                "session_id": "sC4",
                "transcript_path": "",
                "cwd": "/some/cwd",
            },
        )
    assert rc == 0
    payload = json.loads(out)
    ctx_text = payload["hookSpecificOutput"]["additionalContext"]
    # Both verdict and Phase C content appear.
    assert "TODO" in ctx_text  # from verdict
    assert "gadfly historian" in ctx_text  # from Phase C
    assert "use real db" in ctx_text


def test_hook_swallows_internal_crashes(monkeypatch, tmp_log_dir: Path):
    with patch.object(hook.watchdog, "evaluate", side_effect=RuntimeError("boom")):
        rc, out = _run_hook(
            monkeypatch,
            {
                "hook_event_name": "PostToolUse",
                "tool_name": "Edit",
                "tool_input": {"file_path": "a.py"},
                "tool_response": {},
                "session_id": "s3",
                "transcript_path": "",
            },
        )
    # Must exit 0 even on crash.
    assert rc == 0
    # Log should record the crash.
    log_files = list(tmp_log_dir.glob("*.jsonl"))
    assert log_files
    rec = json.loads(log_files[0].read_text().strip())
    assert rec["error"] and "boom" in rec["error"]
