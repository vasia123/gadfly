"""Tests for historian.py.

The Haiku call is mocked via the `run_query` DI parameter — same
pattern as test_watchdog.py / test_journal.py. We patch
`_build_extract_tool` to expose the `_Captured` container.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gadfly import historian as h
from gadfly import project_state as ps


# --- Compaction --------------------------------------------------------------


def _msg(role: str, content: Any) -> dict[str, Any]:
    return {"message": {"role": role, "content": content}}


def _text_block(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def _tool_use(name: str, inp: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "name": name, "input": inp}


def _tool_result(content: str) -> dict[str, Any]:
    return {"type": "tool_result", "content": content}


def test_compact_drops_tool_results_and_keeps_user_text():
    entries = [
        _msg("user", "investigate the marlin EXL3 regression"),
        _msg("user", [_tool_result("some output")]),  # dropped
        _msg("assistant", [_text_block("I'll start by reading benches.")]),
    ]
    events = h.compact_transcript(entries)
    kinds = [e.kind for e in events]
    assert "user" in kinds
    assert "assistant_text" in kinds
    # No event has tool_result content.
    assert all("some output" not in e.text for e in events)


def test_compact_summarizes_tool_use():
    entries = [
        _msg("assistant", [
            _tool_use("Edit", {"file_path": "src/a.py", "old_string": "x", "new_string": "y"}),
            _tool_use("Bash", {"command": "cargo test --quiet"}),
        ]),
    ]
    events = h.compact_transcript(entries)
    texts = [e.text for e in events]
    assert any("Edit(src/a.py)" in t for t in texts)
    assert any("Bash:" in t and "cargo test" in t for t in texts)


def test_compact_action_index_monotonic():
    entries = [
        _msg("assistant", [_tool_use("Read", {"file_path": "a"})]),
        _msg("assistant", [_tool_use("Edit", {"file_path": "b"})]),
    ]
    events = h.compact_transcript(entries)
    tool_events = [e for e in events if e.kind == "assistant_tool"]
    indices = [e.action_index for e in tool_events]
    assert indices == sorted(indices)
    assert indices[-1] > 0


def test_compact_strips_service_tags_from_user_messages():
    entries = [
        _msg("user", "<command-name>/compact</command-name>"),
        _msg("user", "<system-reminder>noise</system-reminder>"),
        _msg("user", "real instruction"),
    ]
    events = h.compact_transcript(entries)
    user_events = [e for e in events if e.kind == "user"]
    assert len(user_events) == 1
    assert user_events[0].text == "real instruction"


def test_compact_truncates_long_blocks():
    long_text = "x" * 5000
    entries = [_msg("user", long_text)]
    events = h.compact_transcript(entries)
    assert "+" in events[0].text  # truncation marker


# --- Distill with mocked Haiku ---------------------------------------------


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    return tmp_path


@pytest.fixture
def captured_ref(monkeypatch: pytest.MonkeyPatch):
    holder: dict[str, h._Captured] = {}
    original = h._build_extract_tool

    def patched(captured: h._Captured):
        holder["c"] = captured
        return original(captured)

    monkeypatch.setattr(h, "_build_extract_tool", patched)
    return holder


def _runner_returning(captured_ref, payload: dict[str, Any] | None):
    async def runner(prompt: str, options) -> None:
        if payload is not None:
            captured_ref["c"].payload = payload
    return runner


def _write_transcript(path: Path, entries: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_distill_writes_raw_digest_with_findings(isolated, captured_ref):
    cwd = "/proj/x"
    tpath = isolated / "session1.jsonl"
    _write_transcript(tpath, [
        _msg("user", "implement validator function"),
        _msg("assistant", [_text_block("I'll add validator. TODO: error handling later.")]),
        _msg("assistant", [_tool_use("Edit", {"file_path": "v.py", "old_string": "a",
                                              "new_string": "b"})]),
    ])
    payload = {
        "new_promises": [
            {"title": "add error handling for validator",
             "evidence_quote": "TODO: error handling later"}
        ],
        "fulfilled_promises": [],
        "new_corrections": [],
        "knowledge_updates": [],
    }
    res = h.distill_session(
        cwd=cwd,
        session_id="session1",
        transcript_path=tpath,
        transcript_mtime=tpath.stat().st_mtime,
        run_query=_runner_returning(captured_ref, payload),
    )
    assert res.error is None
    assert res.chunks == 1
    # Raw digest persisted.
    raw = ps.read_raw_digest(cwd, "session1")
    assert raw is not None
    assert raw["new_promises"][0]["title"] == "add error handling for validator"
    assert raw["new_promises"][0]["evidence_quote"] == "TODO: error handling later"


def test_distill_empty_transcript_writes_empty_digest(isolated, captured_ref):
    cwd = "/proj/x"
    tpath = isolated / "empty.jsonl"
    tpath.write_text("")  # no entries

    # Haiku should NOT be called for empty transcripts.
    called = {"n": 0}

    async def runner(prompt, options):
        called["n"] += 1
        captured_ref["c"].payload = {"new_promises": [], "fulfilled_promises": [],
                                     "new_corrections": [], "knowledge_updates": []}

    res = h.distill_session(
        cwd=cwd,
        session_id="empty",
        transcript_path=tpath,
        transcript_mtime=tpath.stat().st_mtime,
        run_query=runner,
    )
    assert res.error is None
    assert res.chunks == 0
    assert called["n"] == 0
    # An empty digest must still land on disk so we don't keep re-trying.
    assert ps.read_raw_digest(cwd, "empty") is not None


def test_distill_error_does_not_write_raw_digest(isolated, captured_ref):
    """Root-cause fix: on Haiku error, do NOT write an empty raw digest.
    Otherwise the session gets marked 'digested' and we lose its data
    forever even though it's actually retryable."""
    cwd = "/proj/error-no-write"
    tpath = isolated / "s.jsonl"
    _write_transcript(tpath, [_msg("user", "real content"),
                              _msg("assistant", [_text_block("yes")])])

    async def runner(prompt, options):
        raise RuntimeError("transient haiku error")

    res = h.distill_session(
        cwd=cwd, session_id="error_sid",
        transcript_path=tpath, transcript_mtime=tpath.stat().st_mtime,
        run_query=runner,
    )
    assert res.error and "transient" in res.error
    # Raw digest MUST NOT exist — that's the regression we're guarding.
    assert ps.read_raw_digest(cwd, "error_sid") is None
    # Failure marker WAS written.
    assert ps.get_distill_attempts(cwd, "error_sid") == 1


def test_distill_success_after_failure_clears_marker(isolated, captured_ref):
    """Transient errors should not leak forever — a successful retry
    clears the attempt counter."""
    cwd = "/proj/recover"
    tpath = isolated / "s.jsonl"
    _write_transcript(tpath, [_msg("user", "x"),
                              _msg("assistant", [_text_block("y")])])

    # First attempt fails.
    async def fail(prompt, options):
        raise RuntimeError("flaky")
    h.distill_session(cwd=cwd, session_id="r1", transcript_path=tpath,
                      transcript_mtime=tpath.stat().st_mtime, run_query=fail)
    assert ps.get_distill_attempts(cwd, "r1") == 1

    # Second attempt succeeds.
    runner = _runner_returning(captured_ref, {
        "new_promises": [], "fulfilled_promises": [],
        "new_corrections": [], "knowledge_updates": [],
    })
    res = h.distill_session(cwd=cwd, session_id="r1", transcript_path=tpath,
                            transcript_mtime=tpath.stat().st_mtime, run_query=runner)
    assert res.error is None
    # Marker cleared.
    assert ps.get_distill_attempts(cwd, "r1") == 0
    # Digest now exists.
    assert ps.read_raw_digest(cwd, "r1") is not None


def test_exhausted_sessions_are_skipped_by_discovery(isolated, monkeypatch):
    """After MAX_DISTILL_ATTEMPTS failures, the daemon stops retrying.
    Manual reset is `rm <project>/failed/<sid>.json`."""
    cwd = "/proj/exhaust"
    cwd_encoded = ps.encode_cwd(cwd)
    fake = isolated / "projects" / cwd_encoded
    fake.mkdir(parents=True, exist_ok=True)
    (fake / "boom.jsonl").write_text(json.dumps(_msg("user", "x")) + "\n")
    monkeypatch.setattr(h, "transcript_dir_for", lambda enc: fake)

    # Record MAX_DISTILL_ATTEMPTS failures.
    for _ in range(ps.MAX_DISTILL_ATTEMPTS):
        ps.record_distill_failure(cwd, "boom", "broken")
    assert ps.session_exhausted(cwd, "boom")

    found = h.find_undigested_sessions(cwd, cwd_encoded=cwd_encoded)
    assert all(sid != "boom" for sid, _, _ in found)


def test_distill_haiku_failure_returns_error_no_raise(isolated, captured_ref):
    cwd = "/proj/x"
    tpath = isolated / "s.jsonl"
    _write_transcript(tpath, [_msg("user", "hello"),
                              _msg("assistant", [_text_block("hi")])])

    async def runner(prompt, options):
        raise RuntimeError("boom")

    res = h.distill_session(
        cwd=cwd,
        session_id="s",
        transcript_path=tpath,
        transcript_mtime=tpath.stat().st_mtime,
        run_query=runner,
    )
    assert res.error and "boom" in res.error


def test_distill_haiku_did_not_call_tool_returns_error(isolated, captured_ref):
    cwd = "/proj/x"
    tpath = isolated / "s.jsonl"
    _write_transcript(tpath, [_msg("user", "hello"),
                              _msg("assistant", [_text_block("hi")])])
    res = h.distill_session(
        cwd=cwd,
        session_id="s",
        transcript_path=tpath,
        transcript_mtime=tpath.stat().st_mtime,
        run_query=_runner_returning(captured_ref, None),
    )
    assert res.error and "did not call" in res.error


def test_distill_chunks_large_transcript(isolated, captured_ref, monkeypatch):
    """When events > MAX_EVENTS_PER_CHUNK, multiple Haiku calls happen
    and findings accumulate."""
    monkeypatch.setattr(h, "MAX_EVENTS_PER_CHUNK", 3)
    cwd = "/proj/x"
    tpath = isolated / "big.jsonl"
    # 10 user/assistant events → 4 chunks of ≤3 each.
    entries = []
    for i in range(5):
        entries.append(_msg("user", f"step {i}"))
        entries.append(_msg("assistant", [_text_block(f"working on {i}")]))
    _write_transcript(tpath, entries)

    call_count = {"n": 0}

    async def runner(prompt, options):
        call_count["n"] += 1
        captured_ref["c"].payload = {
            "new_promises": [{"title": f"p{call_count['n']}",
                              "evidence_quote": "ev"}],
            "fulfilled_promises": [], "new_corrections": [],
            "knowledge_updates": [],
        }

    res = h.distill_session(
        cwd=cwd,
        session_id="big",
        transcript_path=tpath,
        transcript_mtime=tpath.stat().st_mtime,
        run_query=runner,
    )
    assert res.error is None
    assert res.chunks >= 2
    # Findings accumulate across chunks.
    raw = ps.read_raw_digest(cwd, "big")
    assert raw is not None
    assert len(raw["new_promises"]) == res.chunks  # one promise per chunk


# --- Audit log integration --------------------------------------------------


def test_distill_emits_historian_event_in_audit_log(isolated, captured_ref):
    cwd = "/proj/x"
    tpath = isolated / "s.jsonl"
    _write_transcript(tpath, [_msg("user", "hi"),
                              _msg("assistant", [_text_block("ok")])])
    payload = {"new_promises": [], "fulfilled_promises": [],
               "new_corrections": [], "knowledge_updates": []}
    h.distill_session(
        cwd=cwd,
        session_id="s",
        transcript_path=tpath,
        transcript_mtime=tpath.stat().st_mtime,
        run_query=_runner_returning(captured_ref, payload),
    )
    log_path = Path(isolated) / "log" / "s.jsonl"
    assert log_path.is_file()
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    events = [r for r in records if r.get("type") == "historian_digest"]
    assert len(events) == 1
    assert events[0]["cwd"] == cwd
    assert events[0]["error"] is None


# --- Heartbeat reading ------------------------------------------------------


def test_list_heartbeats_returns_payloads(isolated):
    d = isolated / "heartbeat"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "-home-vasis-projects-hobby-gadfly.tick"
    p.write_text(json.dumps({"ts": 1000.0, "session_id": "abc",
                             "last_action_index": 5}))
    found = h.list_heartbeats()
    assert len(found) == 1
    cwd_encoded, payload, mtime = found[0]
    assert cwd_encoded == "-home-vasis-projects-hobby-gadfly"
    assert payload["session_id"] == "abc"


def test_session_is_done_threshold(isolated):
    import time as _time
    now = _time.time()
    # Active session with matching heartbeat fresh → not done.
    assert h.session_is_done(now, {"ts": now, "session_id": "s1"},
                              session_id="s1") is False
    # Active session with matching heartbeat stale → done.
    old = now - h.DONE_AFTER_S - 1
    assert h.session_is_done(now, {"ts": old, "session_id": "s1"},
                              session_id="s1") is True
    # No heartbeat, transcript ancient → done.
    assert h.session_is_done(0.0, None) is True


def test_session_is_done_historical_session_in_active_cwd(isolated):
    """Crucial property: heartbeat belongs to the CURRENT session in the
    cwd. Older sessions in the same cwd are by definition done, even
    when the cwd-level heartbeat is fresh."""
    import time as _time
    now = _time.time()
    fresh_hb = {"ts": now, "session_id": "current-active-session"}
    # Same cwd, different session, its own transcript untouched for >5min
    # → done despite the fresh heartbeat.
    ancient_transcript_mtime = now - h.DONE_AFTER_S - 100
    assert h.session_is_done(
        ancient_transcript_mtime, fresh_hb, session_id="historical-session"
    ) is True
    # Same cwd, different session, but THIS session's transcript was
    # touched recently (maybe being worked on in parallel? edge case)
    # → not done.
    recent = now - 10
    assert h.session_is_done(
        recent, fresh_hb, session_id="another-recent-session"
    ) is False


# --- find_undigested_sessions ------------------------------------------------


def _setup_fake_projects_dir(isolated: Path, cwd: str, sessions: dict[str, str]) -> Path:
    """Create a fake ~/.claude/projects/<encoded>/ with session files.
    `sessions` is {sid: jsonl_content}. Returns the encoded-dir path.
    """
    cwd_encoded = ps.encode_cwd(cwd)
    fake = isolated / "projects" / cwd_encoded
    fake.mkdir(parents=True, exist_ok=True)
    for sid, content in sessions.items():
        (fake / f"{sid}.jsonl").write_text(content)
    return fake


def test_sweep_once_processes_done_sessions_end_to_end(isolated, captured_ref, monkeypatch):
    """Force-done sweep digests one session and persists state.json."""
    cwd = "/proj/sweep"
    transcript = (
        json.dumps(_msg("user", "investigate X")) + "\n"
        + json.dumps(_msg("assistant", [_text_block("I'll defer details for later.")])) + "\n"
    )
    fake = _setup_fake_projects_dir(isolated, cwd, {"sw1": transcript})

    # Point historian at our fake projects dir.
    monkeypatch.setattr(h, "transcript_dir_for", lambda enc: fake)

    # Drop a stale heartbeat so discover_work knows about this cwd. The
    # ts is old enough that force_done isn't strictly needed, but we set
    # both so the test isn't time-flaky.
    hb_dir = isolated / "heartbeat"
    hb_dir.mkdir(parents=True, exist_ok=True)
    (hb_dir / f"{ps.encode_cwd(cwd)}.tick").write_text(
        json.dumps({"ts": 0.0, "session_id": "sw1", "cwd": cwd})
    )

    payload = {
        "new_promises": [{"title": "do X details",
                          "evidence_quote": "I'll defer details for later."}],
        "fulfilled_promises": [], "new_corrections": [], "knowledge_updates": [],
    }
    runner = _runner_returning(captured_ref, payload)

    summary = h.sweep_once(force_done=True, run_query=runner)
    assert summary["considered"] == 1
    assert summary["processed"] == 1
    assert summary["errors"] == 0

    # state.json was persisted and the promise graduated.
    state = ps.load_state(cwd)
    assert len(state.promises) == 1
    assert "sw1" in state.digested_sessions


def test_sweep_skips_active_sessions(isolated, captured_ref, monkeypatch):
    """Without force_done, a fresh heartbeat means the session is still
    active and should be skipped."""
    cwd = "/proj/active"
    transcript = (
        json.dumps(_msg("user", "x")) + "\n"
        + json.dumps(_msg("assistant", [_text_block("y")])) + "\n"
    )
    fake = _setup_fake_projects_dir(isolated, cwd, {"act1": transcript})
    monkeypatch.setattr(h, "transcript_dir_for", lambda enc: fake)

    # Fresh heartbeat — session is "active".
    hb_dir = isolated / "heartbeat"
    hb_dir.mkdir(parents=True, exist_ok=True)
    (hb_dir / f"{ps.encode_cwd(cwd)}.tick").write_text(
        json.dumps({"ts": __import__("time").time(), "session_id": "act1", "cwd": cwd})
    )

    async def runner(prompt, options):
        raise AssertionError("should not be called for active sessions")

    summary = h.sweep_once(force_done=False, run_query=runner)
    assert summary["considered"] == 0


def test_rebuild_recovers_state_from_raw(isolated, monkeypatch):
    """Drift recovery: scribble random nonsense into state.json, run
    rebuild, verify state is reconstructed from raw/ digests."""
    cwd = "/proj/rebuild"
    # Two raw digests on disk.
    for sid, ts in [("r1", 100.0), ("r2", 200.0)]:
        ps.write_raw_digest(cwd, sid, {
            "session_id": sid,
            "ts": ts,
            "new_promises": [{"title": f"do {sid}", "evidence_quote": "ev"}],
            "fulfilled_promises": [], "new_corrections": [], "knowledge_updates": [],
        })
    # Vandalise state.json.
    bad = ps.ProjectState(cwd=cwd)
    bad.promises["spurious"] = ps.Promise(id="spurious", title="?",
                                          provenance=_prov_for_test())
    ps.save_state(bad)

    rc = h.main(["rebuild", cwd])
    assert rc == 0
    state = ps.load_state(cwd)
    assert "spurious" not in state.promises
    titles = {p.title for p in state.promises.values()}
    assert titles == {"do r1", "do r2"}


def _prov_for_test() -> ps.Provenance:
    return ps.Provenance(source_session="s", source_action_index=1,
                         evidence_quote="q", first_seen_ts=0.0)


def test_propose_claudemd_renders_corrections_and_subsystems(isolated):
    """H8: propose-claudemd outputs only PROMOTED findings, never quarantine."""
    cwd = "/proj/propose"
    # Seed a corpus where one correction passes promotion (≥2 sessions)
    # and one subsystem passes (≥3 file mentions).
    digests = [
        _digest_for_retrieval("s1", 100.0, corrections=[
            {"rule": "no mocks in integration tests",
             "why": "burned last quarter",
             "how_to_apply": "use a real db",
             "evidence_quote": "don't mock the database"}
        ]),
        _digest_for_retrieval("s2", 200.0, corrections=[
            {"rule": "no mocks in integration tests",
             "why": "burned",
             "how_to_apply": "use real",
             "evidence_quote": "no mocks please"}
        ]),
        _digest_for_retrieval("s3", 300.0, knowledge=[
            {"subsystem_id": "auth", "title": "Auth subsystem",
             "purpose": "validate users",
             "files": ["auth.py", "session.py", "tokens.py"],
             "evidence_quote": "auth files"}
        ]),
    ]
    for d in digests:
        ps.write_raw_digest(cwd, d["session_id"], d)
    ps.save_state(ps.rebuild_from_raw(cwd))

    out = h.propose_claudemd(cwd, min_confidence=2)
    assert "Project conventions" in out
    assert "no mocks in integration tests" in out
    assert "Codebase map" in out
    assert "Auth subsystem" in out
    # Provenance markers must appear (anti-hallucination).
    assert "evidence:" in out


def test_propose_claudemd_empty_when_nothing_promoted(isolated):
    """When nothing has reached threshold, output is a helpful empty message."""
    cwd = "/proj/empty-propose"
    # Single-session correction — stays in quarantine.
    ps.write_raw_digest(cwd, "s1", _digest_for_retrieval("s1", 100.0, corrections=[
        {"rule": "single shot", "why": "w", "how_to_apply": "h",
         "evidence_quote": "single"}
    ]))
    ps.save_state(ps.rebuild_from_raw(cwd))

    out = h.propose_claudemd(cwd)
    assert "nothing high-confidence" in out
    assert "single shot" not in out  # not promoted, not included


def test_status_cli_runs_without_crashing(isolated, capsys):
    """The status subcommand must not crash even on an empty corpus."""
    rc = h.main(["status"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gadfly historian" in out


def test_status_exits_nonzero_when_daemon_stale_and_backlog(isolated, capsys, monkeypatch):
    """H14: backlog exists AND state.json hasn't been touched recently →
    status exits 1 with a warning. systemd / cron can monitor this."""
    import os as _os
    import time as _time

    cwd = "/proj/stale-test"
    cwd_encoded = ps.encode_cwd(cwd)
    # Create a fake transcript needing digest.
    fake = isolated / "projects" / cwd_encoded
    fake.mkdir(parents=True, exist_ok=True)
    tp = fake / "ses1.jsonl"
    tp.write_text("{}\n")
    # Mark it as ancient so session_is_done returns True.
    ancient = _time.time() - 24 * 3600
    _os.utime(tp, (ancient, ancient))
    monkeypatch.setattr(h, "transcript_dir_for", lambda enc: fake)
    # Drop a fresh-enough heartbeat for an OTHER session in the cwd so
    # discover_work knows about it; the active session is "ses-active",
    # NOT "ses1", which is the ancient one we want digested.
    hb_dir = isolated / "heartbeat"
    hb_dir.mkdir(parents=True, exist_ok=True)
    (hb_dir / f"{cwd_encoded}.tick").write_text(
        json.dumps({"ts": _time.time(), "session_id": "ses-active", "cwd": cwd})
    )

    # No state.json anywhere → daemon stale; backlog of 1 → exit 1.
    rc = h.main(["status"])
    out = capsys.readouterr().out
    assert rc == 1, f"expected non-zero exit, got {rc}: {out}"
    assert "WARNING" in out
    assert "stalled" in out.lower() or "daemon" in out.lower()


def test_status_exits_zero_when_no_backlog(isolated, capsys):
    """No undigested sessions = nothing to do = OK regardless of daemon state."""
    rc = h.main(["status"])
    assert rc == 0


def test_find_relevant_priors_keyword_overlap(isolated):
    """H6: keyword-overlap retrieves the right promise."""
    cwd = "/proj/retrieval"
    digests = [
        _digest_for_retrieval(
            "s1", 100.0,
            promises=[{"title": "implement marlin EXL3 benchmark",
                       "evidence_quote": "I'll add the marlin bench later"}]
        ),
        _digest_for_retrieval(
            "s2", 200.0,
            promises=[{"title": "rewrite documentation",
                       "evidence_quote": "TODO docs"}]
        ),
    ]
    for d in digests:
        ps.write_raw_digest(cwd, d["session_id"], d)
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)

    hits = h.find_relevant_priors(
        cwd,
        workstream_title="investigate marlin regression",
        file_paths=[],
    )
    titles = [h_.title for h_ in hits]
    # Marlin promise should rank above the docs promise.
    assert titles
    assert "marlin" in titles[0].lower()


def test_find_relevant_priors_file_path_overlap(isolated):
    """File-path overlap surfaces relevant subsystems even without keyword match."""
    cwd = "/proj/files"
    # 3 file mentions to graduate from quarantine.
    digests = [
        _digest_for_retrieval("s1", 100.0, knowledge=[{
            "subsystem_id": "auth-flow", "title": "Auth flow", "purpose": "login",
            "files": ["auth.py", "session.py", "tokens.py"],
            "evidence_quote": "auth subsystem"
        }]),
    ]
    for d in digests:
        ps.write_raw_digest(cwd, d["session_id"], d)
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)

    hits = h.find_relevant_priors(
        cwd,
        workstream_title="completely unrelated topic",
        file_paths=["auth.py"],
    )
    assert any(hit.kind == "subsystem" and "auth" in hit.title.lower() for hit in hits)


def test_find_relevant_priors_respects_max():
    cwd = "/proj/many"
    digests = []
    for i in range(20):
        digests.append(_digest_for_retrieval(
            f"s{i}", float(i),
            promises=[{"title": f"do thing {i}",
                       "evidence_quote": f"I'll do thing {i}"}]
        ))
    for d in digests:
        ps.write_raw_digest(cwd, d["session_id"], d)
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)

    hits = h.find_relevant_priors(cwd, workstream_title="thing", max_results=3)
    assert len(hits) <= 3


def test_render_priors_block_includes_evidence(isolated):
    hits = [h.PriorHit(
        kind="correction", id="c1", title="no mocks",
        body="burned last quarter",
        evidence_quote="don't mock the database",
        source_session="abc12345-67",
        score=0.5,
    )]
    rendered = h.render_priors_block(hits)
    assert "Project priors" in rendered
    assert "no mocks" in rendered
    assert "don't mock the database" in rendered
    assert "session abc12345-67" in rendered


def test_render_priors_block_empty_returns_empty():
    assert h.render_priors_block([]) == ""


def _digest_for_retrieval(sid, ts, *, promises=None, corrections=None, knowledge=None):
    return {
        "session_id": sid, "ts": ts,
        "new_promises": promises or [],
        "fulfilled_promises": [],
        "new_corrections": corrections or [],
        "knowledge_updates": knowledge or [],
    }


def test_find_undigested_skips_already_digested(isolated, tmp_path):
    """Sessions whose mtime <= recorded mtime are skipped; newer-on-disk
    ones are surfaced again."""
    cwd = "/proj/test"
    cwd_encoded = ps.encode_cwd(cwd)
    # Simulate the Claude Code transcript dir.
    fake_projects = isolated / "projects" / cwd_encoded
    fake_projects.mkdir(parents=True, exist_ok=True)
    tp = fake_projects / "s1.jsonl"
    tp.write_text("{}\n")
    mtime = tp.stat().st_mtime

    # Patch the home dir to point at our fake. This avoids touching the
    # user's real ~/.claude.
    import gadfly.historian as historian_mod

    def fake_dir(_e):
        return fake_projects
    historian_mod.transcript_dir_for = fake_dir  # type: ignore[assignment]

    # No state yet → 1 undigested.
    found = h.find_undigested_sessions(cwd, cwd_encoded=cwd_encoded)
    assert len(found) == 1
    assert found[0][0] == "s1"

    # Mark digested.
    state = ps.load_state(cwd)
    state.digested_sessions["s1"] = {"mtime": mtime, "sha": None, "ts": 0.0}
    ps.save_state(state)
    assert h.find_undigested_sessions(cwd, cwd_encoded=cwd_encoded) == []

    # Touch transcript → re-surfaces.
    import os as _os
    new_mtime = mtime + 100
    _os.utime(tp, (new_mtime, new_mtime))
    assert len(h.find_undigested_sessions(cwd, cwd_encoded=cwd_encoded)) == 1
