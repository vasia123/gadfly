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
