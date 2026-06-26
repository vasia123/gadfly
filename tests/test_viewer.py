"""Tests for the audit log enrichment and the web viewer's API endpoints.

We don't render the HTML; we verify the JSON contracts the page relies on."""

from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

from gadfly import log as audit_log
from gadfly import viewer
from gadfly.verdict import Verdict


# --- log enrichment ----------------------------------------------------------


def test_log_append_records_payload_and_user_message(tmp_log_dir: Path):
    sha = audit_log.ensure_system_prompt("hello system prompt")
    audit_log.append(
        session_id="s-rich",
        tool_name="Edit",
        tool_input={"file_path": "x.py", "old_string": "a", "new_string": "b"},
        verdict=Verdict(professional=False, reason="stub", suggestion="impl it"),
        latency_ms=1234.5,
        error=None,
        payload={"hook_event_name": "PostToolUse", "tool_name": "Edit"},
        user_message="full prompt text",
        system_prompt_sha=sha,
    )
    rec = json.loads((tmp_log_dir / "s-rich.jsonl").read_text().strip())
    assert rec["payload"]["hook_event_name"] == "PostToolUse"
    assert rec["user_message"] == "full prompt text"
    assert rec["system_prompt_sha"] == sha
    # system prompt was content-addressed:
    stored = audit_log.read_system_prompt(sha)
    assert stored == "hello system prompt"


def test_ensure_system_prompt_is_idempotent(tmp_log_dir: Path):
    sha1 = audit_log.ensure_system_prompt("same text")
    sha2 = audit_log.ensure_system_prompt("same text")
    sha3 = audit_log.ensure_system_prompt("different text")
    assert sha1 == sha2
    assert sha1 != sha3


# --- viewer HTTP API ---------------------------------------------------------


@pytest.fixture
def viewer_server(tmp_log_dir: Path):
    """Start a viewer server on an ephemeral port, yield base URL, tear down."""
    viewer._Handler.log_dir = tmp_log_dir
    srv = ThreadingHTTPServer(("127.0.0.1", 0), viewer._Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _seed(tmp_log_dir: Path, session_id: str, records: list[dict]) -> None:
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_log_dir / f"{session_id}.jsonl"
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _get_json(url: str):
    with urlopen(url, timeout=2) as r:
        return json.loads(r.read())


def _get_text(url: str) -> tuple[int, str]:
    try:
        with urlopen(url, timeout=2) as r:
            return r.status, r.read().decode()
    except Exception as e:
        # urlopen raises HTTPError on 4xx; pull code off it.
        code = getattr(e, "code", 0)
        return code, ""


def test_api_sessions_counts_verdicts_and_goal_events_separately(tmp_log_dir: Path, viewer_server: str):
    """Goal-distill records share the JSONL file with verdicts but must NOT
    inflate the verdict count or the flagged count."""
    _seed(
        tmp_log_dir,
        "mixed",
        [
            {"type": "verdict", "ts": 1.0, "tool_name": "Edit", "verdict": {"professional": True}},
            {"type": "goal_distill", "ts": 2.0, "goal": "first goal", "prior_goal": None},
            {"type": "verdict", "ts": 3.0, "tool_name": "Bash", "verdict": {"professional": False}},
            {"type": "goal_distill", "ts": 4.0, "goal": "updated", "prior_goal": "first goal"},
            {"type": "goal_distill", "ts": 5.0, "goal": None, "error": "timeout"},
        ],
    )
    data = _get_json(viewer_server + "/api/sessions")
    s = next(s for s in data if s["id"] == "mixed")
    assert s["count"] == 2
    assert s["flagged"] == 1
    assert s["goal_events"] == 3
    # The most-recent record was a goal event — last_tool reflects that.
    assert s["last_tool"] == "goal"


def test_api_sessions_lists_files_with_summary(tmp_log_dir: Path, viewer_server: str):
    _seed(
        tmp_log_dir,
        "alpha",
        [
            {"ts": 1.0, "tool_name": "Edit", "verdict": {"professional": True}},
            {"ts": 2.0, "tool_name": "Bash", "verdict": {"professional": False}},
        ],
    )
    _seed(
        tmp_log_dir,
        "beta",
        [{"ts": 100.0, "tool_name": "Write", "verdict": {"professional": True}}],
    )
    data = _get_json(viewer_server + "/api/sessions")
    assert isinstance(data, list)
    by_id = {s["id"]: s for s in data}
    assert by_id["alpha"]["count"] == 2
    assert by_id["alpha"]["flagged"] == 1
    assert by_id["alpha"]["latest_ts"] == 2.0
    assert by_id["alpha"]["last_tool"] == "Bash"
    assert by_id["beta"]["count"] == 1
    assert by_id["beta"]["flagged"] == 0
    # Sorted newest-first.
    assert data[0]["id"] == "beta"


def test_api_session_returns_total_and_records_in_order(tmp_log_dir: Path, viewer_server: str):
    recs = [
        {"ts": 1.0, "tool_name": "Edit", "verdict": {"professional": True}},
        {"ts": 2.0, "tool_name": "Bash", "verdict": {"professional": True}},
    ]
    _seed(tmp_log_dir, "gamma", recs)
    data = _get_json(viewer_server + "/api/session/gamma")
    assert data["total"] == 2
    assert [r["tool_name"] for r in data["records"]] == ["Edit", "Bash"]


def test_api_session_paginates_with_limit(tmp_log_dir: Path, viewer_server: str):
    """`?limit=N` must return the N MOST RECENT records (tail of the file),
    not the first N — newest belongs on top of the viewer."""
    recs = [
        {"ts": float(i), "tool_name": f"T{i}", "verdict": {"professional": True}}
        for i in range(10)
    ]
    _seed(tmp_log_dir, "page", recs)
    data = _get_json(viewer_server + "/api/session/page?limit=3")
    assert data["total"] == 10
    assert [r["tool_name"] for r in data["records"]] == ["T7", "T8", "T9"]


def test_api_session_limit_zero_returns_empty(tmp_log_dir: Path, viewer_server: str):
    _seed(tmp_log_dir, "z", [{"ts": 1.0, "tool_name": "X", "verdict": {"professional": True}}])
    data = _get_json(viewer_server + "/api/session/z?limit=0")
    assert data["total"] == 1
    assert data["records"] == []


def test_api_session_unknown_returns_empty(viewer_server: str):
    data = _get_json(viewer_server + "/api/session/does-not-exist")
    assert data == {"total": 0, "records": []}


def test_api_system_prompt_roundtrip(tmp_log_dir: Path, viewer_server: str):
    sha = audit_log.ensure_system_prompt("the rubric goes here")
    code, body = _get_text(viewer_server + f"/api/system_prompt/{sha}")
    assert code == 200
    assert body == "the rubric goes here"


def test_api_system_prompt_unknown_is_404(viewer_server: str):
    code, _ = _get_text(viewer_server + "/api/system_prompt/deadbeef")
    assert code == 404


def test_index_html_served_on_root(viewer_server: str):
    code, body = _get_text(viewer_server + "/")
    assert code == 200
    assert "<!doctype html>" in body.lower()
    assert "gadfly" in body


# --- trail viewer features --------------------------------------------------


def test_api_sessions_counts_trail_events_and_drifts(tmp_log_dir: Path, viewer_server: str):
    """Trail records are counted separately from verdicts. The sidebar
    badge `N trail · M drift` reads `trail_events` and `trail_drifts`."""
    _seed(
        tmp_log_dir,
        "trail-sess",
        [
            {"type": "verdict", "ts": 1.0, "tool_name": "Edit",
             "verdict": {"professional": True}},
            {"type": "trail_update", "ts": 2.0, "action_index": 1,
             "advances_trail": True, "drift_detected": False,
             "drift_kind": None, "suppressed": False,
             "delivered_to_agent": False},
            {"type": "trail_update", "ts": 3.0, "action_index": 2,
             "advances_trail": False, "drift_detected": True,
             "drift_kind": "hardcoded_instance", "suppressed": False,
             "delivered_to_agent": True},
            {"type": "trail_update", "ts": 4.0, "action_index": 3,
             "advances_trail": False, "drift_detected": True,
             "drift_kind": "hardcoded_instance", "suppressed": True,
             "delivered_to_agent": False},
        ],
    )
    data = _get_json(viewer_server + "/api/sessions")
    s = next(s for s in data if s["id"] == "trail-sess")
    # 1 verdict, 3 trail events, 2 of which are drift (one delivered, one suppressed)
    assert s["count"] == 1
    assert s["trail_events"] == 3
    assert s["trail_drifts"] == 2
    # Newest record was a trail event → last_tool reflects that.
    assert s["last_tool"] == "trail"


def test_api_sessions_zero_trail_keys_present_when_absent(tmp_log_dir: Path, viewer_server: str):
    """Sessions with no trail events still expose `trail_events: 0` so the
    JS template doesn't see `undefined` and render `undefined trail`."""
    _seed(
        tmp_log_dir,
        "no-trail",
        [{"ts": 1.0, "tool_name": "Edit",
          "verdict": {"professional": True}}],
    )
    data = _get_json(viewer_server + "/api/sessions")
    s = next(s for s in data if s["id"] == "no-trail")
    assert s["trail_events"] == 0
    assert s["trail_drifts"] == 0


def test_api_trail_snapshot_roundtrip(tmp_log_dir: Path, viewer_server: str):
    """The viewer reads trail snapshots from a SHA-content-addressed store
    (mirroring the journal). `/api/trail_snapshot/<sha>` returns the
    snapshot pretty-printed."""
    snap_json = json.dumps({
        "breadcrumbs": [
            {"action_index": 1, "breadcrumb_text": "added catalog",
             "abstraction_level": "class", "action_summary": "Edit(hazards.go)",
             "ts": 1.0},
        ],
        "drift_flags": [],
        "action_index": 1,
        "prompt_sha": "abc",
        "schema_version": 1,
        "non_advance_streak": 0,
        "last_root_goal": "build hazard",
        "ts": 1.5,
    }, sort_keys=True, ensure_ascii=False)
    sha = audit_log.ensure_trail_snapshot(snap_json)
    code, body = _get_text(viewer_server + f"/api/trail_snapshot/{sha}")
    assert code == 200
    parsed = json.loads(body)
    assert parsed["breadcrumbs"][0]["abstraction_level"] == "class"
    assert parsed["action_index"] == 1


def test_api_trail_snapshot_unknown_is_404(viewer_server: str):
    code, _ = _get_text(viewer_server + "/api/trail_snapshot/deadbeef")
    assert code == 404


def test_index_html_includes_trail_ui(viewer_server: str):
    """Smoke-test that the trail UI shipped: tabs, timeline panel,
    drift-summary container, and the drift-question table the
    timeline renderer reads from."""
    code, body = _get_text(viewer_server + "/")
    assert code == 200
    # Main-pane tabs
    assert 'id="main-tabs"' in body
    assert 'data-tab="trail"' in body
    assert 'data-tab="drift"' in body
    # Timeline panel container
    assert 'id="timeline"' in body
    # Drift-summary chips container
    assert 'id="drift-summary"' in body
    # JS renderer entry points
    assert "function renderTimeline" in body
    assert "function renderTimelineDrift" in body
    assert "function renderDriftSummary" in body
    assert "function toggleDriftKindFilter" in body
    # Canonical-question lookup table for the timeline
    assert "TRAIL_DRIFT_QUESTIONS" in body
    # Suppression-explainer (so suppressed drifts don't read as opaque)
    assert "function suppressionExplain" in body
