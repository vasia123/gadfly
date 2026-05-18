"""Tests for project_state.py — the historian's data model.

Pure plumbing: no SDK calls. Focus on round-trips, cwd encoding, and
the byte-identical re-aggregation contract that's central to drift
defence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gadfly import project_state as ps


# --- cwd encoding -----------------------------------------------------------


def test_encode_cwd_mirrors_claude_codes_layout():
    """Verified against actual ~/.claude/projects/ directory names."""
    assert ps.encode_cwd("/home/vasis/projects_hobby/gadfly") == (
        "-home-vasis-projects-hobby-gadfly"
    )
    assert ps.encode_cwd("/home/vasis/projects_hobby/8mart-games") == (
        "-home-vasis-projects-hobby-8mart-games"
    )


def test_encode_cwd_defensive():
    assert ps.encode_cwd("") == "unknown"
    # Multi-slash garbage collapses to a single dash run.
    assert ps.encode_cwd("/a//b") == "-a-b"
    # Non-alphanumeric chars normalised to dashes.
    assert ps.encode_cwd("/a/b!c@d") == "-a-b-c-d"


# --- Provenance round-trip --------------------------------------------------


def test_provenance_round_trip():
    p = ps.Provenance(
        source_session="abc-def",
        source_action_index=7,
        evidence_quote="don't use mocks in integration tests",
        first_seen_ts=1700.0,
    )
    assert ps.Provenance.from_dict(p.to_dict()) == p


def test_provenance_clips_long_evidence_quote():
    long_quote = "x" * (ps.MAX_EVIDENCE_QUOTE + 50)
    p = ps.Provenance.from_dict(
        {"source_session": "s", "source_action_index": 1,
         "evidence_quote": long_quote, "first_seen_ts": 0.0}
    )
    assert len(p.evidence_quote) == ps.MAX_EVIDENCE_QUOTE


# --- Dataclass round-trips --------------------------------------------------


def _prov(sid="s1", ai=1, q="quote", ts=1000.0) -> ps.Provenance:
    return ps.Provenance(
        source_session=sid,
        source_action_index=ai,
        evidence_quote=q,
        first_seen_ts=ts,
    )


def test_promise_round_trip():
    p = ps.Promise(id="x", title="X", provenance=_prov(), status="open",
                   fulfilled_in_session=None, last_seen_ts=1100.0)
    assert ps.Promise.from_dict(p.to_dict()) == p


def test_promise_status_coercion_is_safe():
    p = ps.Promise.from_dict({"id": "x", "title": "X",
                              "provenance": _prov().to_dict(),
                              "status": "bogus"})
    assert p.status == "open"


def test_correction_confidence_bounded():
    c = ps.Correction(id="c", rule="r", why="w", how_to_apply="h",
                      provenance=_prov(),
                      seen_in_sessions=["s1", "s2", "s3", "s4", "s5"])
    assert c.confidence == 3
    c2 = ps.Correction(id="c", rule="r", why="w", how_to_apply="h",
                       provenance=_prov(), seen_in_sessions=["s1"])
    assert c2.confidence == 1
    c3 = ps.Correction(id="c", rule="r", why="w", how_to_apply="h",
                       provenance=_prov(), seen_in_sessions=[])
    # Even with empty history we floor at 1 (an entry exists).
    assert c3.confidence == 1


def test_subsystem_files_sorted_and_unique():
    s = ps.Subsystem.from_dict({
        "id": "s", "title": "T", "purpose": "p",
        "files": ["b.py", "a.py", "b.py", "c.py"],
        "provenance": _prov().to_dict(),
    })
    assert s.files == ["a.py", "b.py", "c.py"]


# --- Persistence ------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    return tmp_path


def test_load_state_returns_fresh_when_missing(isolated):
    s = ps.load_state("/some/cwd")
    assert s.cwd == "/some/cwd"
    assert s.promises == {}


def test_save_then_load_round_trip(isolated):
    state = ps.ProjectState(cwd="/abc")
    state.promises["p1"] = ps.Promise(id="p1", title="t", provenance=_prov())
    ps.save_state(state)
    loaded = ps.load_state("/abc")
    assert "p1" in loaded.promises
    assert loaded.promises["p1"].title == "t"


def test_load_state_invalidates_on_schema_mismatch(isolated):
    ps.save_state(ps.ProjectState(cwd="/x"))
    p = ps.state_path("/x")
    data = json.loads(p.read_text())
    data["schema_version"] = 999
    p.write_text(json.dumps(data))
    loaded = ps.load_state("/x")
    assert loaded.schema_version == ps.SCHEMA_VERSION
    assert loaded.promises == {}  # reset to empty


def test_load_state_invalidates_on_prompt_sha_mismatch(isolated):
    state = ps.ProjectState(cwd="/x", prompt_sha="old")
    state.promises["p"] = ps.Promise(id="p", title="t", provenance=_prov())
    ps.save_state(state)
    loaded = ps.load_state("/x", expected_prompt_sha="new")
    # Reset to fresh state because prompt has changed.
    assert loaded.promises == {}
    assert loaded.prompt_sha == "new"


def test_write_raw_digest_never_overwrites(isolated):
    ps.write_raw_digest("/cwd", "sid1", {"a": 1})
    ps.write_raw_digest("/cwd", "sid1", {"a": 2})  # ignored
    loaded = ps.read_raw_digest("/cwd", "sid1")
    assert loaded == {"a": 1}


def test_raw_digest_safe_sid(isolated):
    ps.write_raw_digest("/cwd", "abc/def..\\xx", {"k": "v"})
    assert ps.read_raw_digest("/cwd", "abc/def..\\xx") == {"k": "v"}


# --- Deterministic re-aggregator (the core invariant) -----------------------


def _digest(
    sid: str, ts: float, *,
    promises: list[dict] | None = None,
    corrections: list[dict] | None = None,
    knowledge: list[dict] | None = None,
    fulfilled: list[dict] | None = None,
    fulfilled_ids: list[str] | None = None,
) -> dict:
    return {
        "session_id": sid,
        "ts": ts,
        "new_promises": promises or [],
        "fulfilled_promises": fulfilled or [],
        "new_corrections": corrections or [],
        "knowledge_updates": knowledge or [],
        "fulfilled_promise_ids": fulfilled_ids or [],
    }


def test_aggregate_single_session_promise_in_active_store():
    """Promises graduate immediately — single instance is the point."""
    digests = [_digest("s1", 100.0, promises=[
        {"title": "implement validator", "evidence_quote": "I'll do it later",
         "action_index": 4}
    ])]
    s = ps.aggregate(digests, cwd="/x")
    assert len(s.promises) == 1
    p = next(iter(s.promises.values()))
    assert p.title == "implement validator"
    assert p.status == "open"


def test_aggregate_correction_quarantine_then_promote():
    """One-shot correction stays in quarantine; ≥2 sessions promotes."""
    d1 = _digest("s1", 100.0, corrections=[
        {"rule": "no mocks in integration tests",
         "why": "burned last quarter",
         "how_to_apply": "use real db",
         "evidence_quote": "don't mock the database"}
    ])
    d2 = _digest("s2", 200.0, corrections=[
        {"rule": "no mocks in integration tests",
         "why": "burned last quarter",
         "how_to_apply": "use real db",
         "evidence_quote": "no mocks please"}
    ])
    # After 1 session: still in quarantine.
    s1 = ps.aggregate([d1], cwd="/x")
    assert s1.corrections == {}
    assert len(s1.quarantine) == 1
    assert s1.quarantine[0].kind == "correction"
    # After 2: promoted.
    s2 = ps.aggregate([d1, d2], cwd="/x")
    assert len(s2.corrections) == 1
    corr = next(iter(s2.corrections.values()))
    assert sorted(corr.seen_in_sessions) == ["s1", "s2"]
    assert corr.confidence == 2


def test_aggregate_subsystem_requires_three_file_mentions():
    d = _digest("s1", 100.0, knowledge=[
        {"subsystem_id": "auth", "title": "Auth", "purpose": "validate",
         "files": ["a.py", "b.py"], "evidence_quote": "auth code"}
    ])
    s = ps.aggregate([d], cwd="/x")
    assert s.subsystems == {}
    assert len(s.quarantine) == 1
    assert s.quarantine[0].kind == "subsystem"

    # Add a 3rd file from a second session — should promote.
    d2 = _digest("s2", 200.0, knowledge=[
        {"subsystem_id": "auth", "title": "Auth", "purpose": "",
         "files": ["c.py"], "evidence_quote": "more auth code"}
    ])
    s2 = ps.aggregate([d, d2], cwd="/x")
    assert len(s2.subsystems) == 1
    sub = next(iter(s2.subsystems.values()))
    assert sub.files == ["a.py", "b.py", "c.py"]
    assert sub.last_touched_session == "s2"


def test_aggregate_fulfilled_marks_existing_promise():
    d1 = _digest("s1", 100.0, promises=[
        {"title": "implement X", "evidence_quote": "I'll do X", "action_index": 1}
    ])
    d2 = _digest("s2", 200.0, fulfilled=[
        {"id_hint": "implement X", "evidence_quote": "X done"}
    ])
    s = ps.aggregate([d1, d2], cwd="/x")
    assert len(s.promises) == 1
    p = next(iter(s.promises.values()))
    assert p.status == "fulfilled"
    assert p.fulfilled_in_session == "s2"


def test_aggregate_cross_session_fulfillment_by_id():
    """A later digest carrying `fulfilled_promise_ids` closes the
    referenced open promise — the language-agnostic cross-session
    fulfillment signal."""
    d1 = _digest("s1", 100.0, promises=[
        {"title": "wire daemon telemetry", "evidence_quote": "TODO next session",
         "action_index": 2}
    ])
    s_after_d1 = ps.aggregate([d1], cwd="/x")
    pid = next(iter(s_after_d1.promises.keys()))

    d2 = _digest("s2", 200.0, fulfilled_ids=[pid])
    s = ps.aggregate([d1, d2], cwd="/x")
    p = s.promises[pid]
    assert p.status == "fulfilled"
    assert p.fulfilled_in_session == "s2"


def test_cross_session_fulfillment_id_unknown_is_noop():
    """An id that doesn't match any current promise is silently ignored —
    Haiku may have referenced a revoked promise from a stale candidate
    list. Must not raise."""
    d1 = _digest("s1", 100.0, promises=[
        {"title": "A", "evidence_quote": "do A", "action_index": 1}
    ])
    d2 = _digest("s2", 200.0, fulfilled_ids=["bogus-id-not-in-state"])
    s = ps.aggregate([d1, d2], cwd="/x")
    p = next(iter(s.promises.values()))
    assert p.status == "open"


def test_cross_session_fulfillment_already_fulfilled_is_noop():
    """First writer wins — a second session claiming to fulfil an
    already-fulfilled promise leaves the original attribution intact."""
    d1 = _digest("s1", 100.0, promises=[
        {"title": "A", "evidence_quote": "do A", "action_index": 1}
    ])
    s_after = ps.aggregate([d1], cwd="/x")
    pid = next(iter(s_after.promises.keys()))
    d2 = _digest("s2", 200.0, fulfilled_ids=[pid])
    d3 = _digest("s3", 300.0, fulfilled_ids=[pid])
    s = ps.aggregate([d1, d2, d3], cwd="/x")
    assert s.promises[pid].status == "fulfilled"
    assert s.promises[pid].fulfilled_in_session == "s2"


def test_cross_session_fulfillment_rebuild_deterministic(isolated):
    """Same raw digests → same fulfilled state across rebuilds — the
    drift-resistance invariant for the new signal."""
    d1 = _digest("s1", 100.0, promises=[
        {"title": "build retrieval", "evidence_quote": "later", "action_index": 1}
    ])
    s_after_d1 = ps.aggregate([d1], cwd="/x")
    pid = next(iter(s_after_d1.promises.keys()))
    d2 = _digest("s2", 200.0, fulfilled_ids=[pid])
    ps.write_raw_digest("/x", "s1", d1)
    ps.write_raw_digest("/x", "s2", d2)

    s_a = ps.rebuild_from_raw("/x")
    s_b = ps.rebuild_from_raw("/x")
    assert s_a.promises[pid].status == "fulfilled"
    assert s_b.promises[pid].status == "fulfilled"
    assert s_a.to_dict() == s_b.to_dict()


def test_cross_session_fulfillment_protects_against_age_out():
    """Fulfillment runs BEFORE age-out — a stale promise that's just
    been closed by Haiku must be `fulfilled`, not `aged-out`."""
    age = ps.PROMISE_AGE_OUT_S
    d1 = _digest("s1", 100.0, promises=[
        {"title": "stale work", "evidence_quote": "later", "action_index": 1}
    ])
    s_after = ps.aggregate([d1], cwd="/x")
    pid = next(iter(s_after.promises.keys()))
    # d2 is far enough in the future that d1's promise would age out
    # if not fulfilled.
    d2 = _digest("s2", 100.0 + age + 1000.0, fulfilled_ids=[pid])
    s = ps.aggregate([d1, d2], cwd="/x")
    assert s.promises[pid].status == "fulfilled"


def test_aggregate_idempotent_byte_identical():
    """The core invariant: rebuild from the same raw inputs must produce
    byte-identical JSON across runs."""
    digests = [
        _digest("s1", 100.0, corrections=[
            {"rule": "no mocks", "why": "burned", "how_to_apply": "use real",
             "evidence_quote": "do not mock"}
        ]),
        _digest("s2", 200.0, corrections=[
            {"rule": "no mocks", "why": "burned", "how_to_apply": "use real",
             "evidence_quote": "no mocks please"}
        ]),
        _digest("s3", 50.0, knowledge=[
            {"subsystem_id": "core", "title": "Core", "purpose": "main",
             "files": ["a.py", "b.py", "c.py"], "evidence_quote": "core files"}
        ]),
    ]
    s1 = ps.aggregate(digests, cwd="/x", prompt_sha="abc")
    s2 = ps.aggregate(digests, cwd="/x", prompt_sha="abc")
    j1 = json.dumps(s1.to_dict(), sort_keys=True)
    j2 = json.dumps(s2.to_dict(), sort_keys=True)
    assert j1 == j2


def test_aggregate_order_independent():
    """Input order must not change output (sort key is (ts, sid))."""
    d_a = _digest("sA", 100.0, corrections=[
        {"rule": "rule X", "why": "w", "how_to_apply": "h",
         "evidence_quote": "X1"}
    ])
    d_b = _digest("sB", 200.0, corrections=[
        {"rule": "rule X", "why": "w", "how_to_apply": "h",
         "evidence_quote": "X2"}
    ])
    s_ab = ps.aggregate([d_a, d_b], cwd="/x")
    s_ba = ps.aggregate([d_b, d_a], cwd="/x")
    # Same content, but provenance points to the EARLIEST session — that's
    # determined by sort order in aggregate(). So this is now reliable.
    assert json.dumps(s_ab.to_dict(), sort_keys=True) == json.dumps(
        s_ba.to_dict(), sort_keys=True
    )


def test_aggregate_respects_revocation():
    d = _digest("s1", 100.0, promises=[
        {"title": "implement X", "evidence_quote": "I'll do X", "action_index": 1}
    ])
    pid = ps.derive_id("promise", "implement X")
    revoked = [ps.RevokedEntry(kind="promise", id=pid, title="implement X",
                                reason="not actually a promise", ts=200.0)]
    s = ps.aggregate([d], cwd="/x", revoked=revoked)
    assert s.promises == {}
    # And the revocation is preserved in the rebuilt state.
    assert len(s.revoked) == 1


def test_rebuild_from_raw_matches_in_memory(isolated):
    """Persist raw digests, rebuild, verify identical to the in-memory
    aggregate. This is what `gadfly historian --rebuild` does."""
    digests = [
        _digest("s1", 100.0, promises=[
            {"title": "Implement X", "evidence_quote": "I'll do X",
             "action_index": 1}
        ]),
        _digest("s2", 200.0, corrections=[
            {"rule": "no mocks", "why": "w", "how_to_apply": "h",
             "evidence_quote": "no mocks"}
        ]),
    ]
    cwd = "/projects/foo"
    for d in digests:
        ps.write_raw_digest(cwd, d["session_id"], d)
    rebuilt = ps.rebuild_from_raw(cwd)
    direct = ps.aggregate(digests, cwd=cwd)
    assert json.dumps(rebuilt.to_dict(), sort_keys=True) == json.dumps(
        direct.to_dict(), sort_keys=True
    )


def test_rebuild_preserves_usefulness_score(isolated):
    """Rebuild must carry forward the outcomes-feedback usefulness scores
    that aren't re-derivable from raw episodic alone."""
    cwd = "/projects/bar"
    # Seed raw digests producing one promoted correction.
    for sid, ts in [("s1", 100.0), ("s2", 200.0)]:
        ps.write_raw_digest(cwd, sid, _digest(sid, ts, corrections=[
            {"rule": "be careful", "why": "reasons", "how_to_apply": "ok",
             "evidence_quote": "careful"}
        ]))
    # Initial rebuild: corrections active, usefulness=0.
    state = ps.rebuild_from_raw(cwd)
    assert len(state.corrections) == 1
    cid = next(iter(state.corrections))
    # Bump usefulness score, save.
    state.corrections[cid].usefulness_score = 5
    ps.save_state(state)
    # Rebuild again — score must survive.
    state2 = ps.rebuild_from_raw(cwd)
    assert state2.corrections[cid].usefulness_score == 5


# --- derive_id stability ---------------------------------------------------


def test_derive_id_is_deterministic():
    assert ps.derive_id("a", "b") == ps.derive_id("a", "b")


def test_derive_id_differs_for_different_input():
    assert ps.derive_id("a", "b") != ps.derive_id("a", "c")


def test_slugify_handles_unicode_and_special():
    assert ps.slugify("Investigate marlin EXL3 regression!") == (
        "investigate-marlin-exl3-regression"
    )
    assert ps.slugify("") == "untitled"


# --- Promote / revoke ------------------------------------------------------


def test_promote_correction_from_quarantine(isolated):
    cwd = "/proj/promote"
    # Single-session correction — sits in quarantine.
    ps.write_raw_digest(cwd, "s1", _digest("s1", 100.0, corrections=[
        {"rule": "always run cargo fmt", "why": "consistency",
         "how_to_apply": "cargo fmt", "evidence_quote": "cargo fmt"}
    ]))
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)
    assert state.corrections == {}
    assert len(state.quarantine) == 1
    cid = state.quarantine[0].payload["id"]

    ok, msg = ps.promote_finding(cwd, cid)
    assert ok, msg
    state2 = ps.load_state(cwd)
    assert cid in state2.corrections
    assert state2.quarantine == []  # removed from quarantine


def test_revoke_active_correction_records_audit(isolated):
    cwd = "/proj/revoke"
    # ≥2 sessions to promote.
    for sid in ("s1", "s2"):
        ps.write_raw_digest(cwd, sid, _digest(sid, 100.0, corrections=[
            {"rule": "use real db", "why": "w", "how_to_apply": "h",
             "evidence_quote": "real db"}
        ]))
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)
    assert len(state.corrections) == 1
    cid = next(iter(state.corrections))

    ok, msg = ps.revoke_finding(cwd, cid, reason="superseded")
    assert ok, msg

    state2 = ps.load_state(cwd)
    assert cid not in state2.corrections
    assert any(r.id == cid and r.reason == "superseded" for r in state2.revoked)


def test_revoke_unknown_id_returns_false(isolated):
    cwd = "/proj/revoke-missing"
    ok, msg = ps.revoke_finding(cwd, "does-not-exist")
    assert not ok
    assert "not found" in msg.lower()


def test_aggregate_skips_revoked_on_rebuild(isolated):
    """After revoke, rebuild from raw must NOT resurrect the entry."""
    cwd = "/proj/revoke-rebuild"
    for sid in ("s1", "s2"):
        ps.write_raw_digest(cwd, sid, _digest(sid, 100.0, corrections=[
            {"rule": "this rule will be revoked", "why": "w",
             "how_to_apply": "h", "evidence_quote": "ev"}
        ]))
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)
    cid = next(iter(state.corrections))
    ps.revoke_finding(cwd, cid, reason="not actually a rule")

    # Rebuild — should respect the revocation.
    rebuilt = ps.rebuild_from_raw(cwd)
    assert cid not in rebuilt.corrections
    assert any(r.id == cid for r in rebuilt.revoked)


def test_aggregate_ages_out_old_open_promises(isolated):
    """Open promises whose last_seen_ts is > PROMISE_AGE_OUT_DAYS older
    than the corpus's max timestamp transition to status=aged-out.
    Prevents the corpus from accumulating stale promises that surface
    in retrieval forever (the bizprofit-style 'watchdog' false positive).
    """
    cwd = "/proj/age-out"
    # Old promise (15 days ago) — should age out.
    import time as _t
    now = _t.time()
    old_ts = now - 15 * 24 * 3600
    fresh_ts = now - 1 * 24 * 3600

    ps.write_raw_digest(cwd, "old", _digest("old", old_ts, promises=[
        {"title": "Continue Step 4 of audit journal", "evidence_quote": "I'll do it later"}
    ]))
    ps.write_raw_digest(cwd, "recent", _digest("recent", fresh_ts, promises=[
        {"title": "implement reflection mode", "evidence_quote": "I'll add reflection next"}
    ]))
    state = ps.rebuild_from_raw(cwd)

    # Two promises total. Old one should be aged-out; fresh one stays open.
    by_status: dict[str, list[ps.Promise]] = {"open": [], "aged-out": [], "fulfilled": []}
    for p in state.promises.values():
        by_status[p.status].append(p)
    assert len(by_status["open"]) == 1, f"expected 1 open, got {len(by_status['open'])}"
    assert len(by_status["aged-out"]) == 1, f"expected 1 aged-out, got {len(by_status['aged-out'])}"
    # The aged-out one is the audit-journal promise.
    aged = by_status["aged-out"][0]
    assert "audit" in aged.title.lower() or "step" in aged.title.lower()


def test_aggregate_age_out_threshold_respected(isolated):
    """A promise just under the threshold stays open."""
    import time as _t
    cwd = "/proj/age-threshold"
    now = _t.time()
    # 6 days old — under the 7-day cutoff. Stays open.
    fresh_ts = now - 6 * 24 * 3600
    # Add a digest with a newer reference so max_ts is now.
    ps.write_raw_digest(cwd, "old", _digest("old", fresh_ts, promises=[
        {"title": "still relevant", "evidence_quote": "I'll do X"}
    ]))
    ps.write_raw_digest(cwd, "new", _digest("new", now, promises=[]))
    state = ps.rebuild_from_raw(cwd)
    p = next(iter(state.promises.values()))
    assert p.status == "open", f"expected open at 6 days, got {p.status}"


def test_verdict_pattern_fingerprint_groups_similar_reasons(isolated):
    """E2: phrasings that share underlying topic hash to the same key.

    'Symptom fix: composable does not exist yet'
    'Symptom fix: useMail composable not yet defined'
    should both bucket together (token-based, stop-words removed)."""
    fp1 = ps.derive_pattern_fingerprint(
        "symptom", "Symptom fix: composable does not exist yet"
    )
    fp2 = ps.derive_pattern_fingerprint(
        "symptom", "Symptom fix: useMail composable not yet defined"
    )
    # Both should contain 'composable' in their fingerprint
    assert "composable" in fp1
    assert "composable" in fp2
    # And both share the marker prefix
    assert fp1.startswith("symptom-")
    assert fp2.startswith("symptom-")


def test_verdict_pattern_fingerprint_distinguishes_markers():
    assert (
        ps.derive_pattern_fingerprint("symptom", "fix sleep race")
        != ps.derive_pattern_fingerprint("rationalization", "fix sleep race")
    )


def test_update_verdict_pattern_creates_then_bumps(isolated):
    cwd = "/proj/patterns"
    ps.update_verdict_pattern(
        cwd, marker="symptom", reason="Symptom fix: composable not yet defined",
        delta=1, session_id="s1", workstream_id="w1",
    )
    state = ps.load_state(cwd)
    assert len(state.verdict_patterns) == 1
    pattern = next(iter(state.verdict_patterns.values()))
    assert pattern.value_score == 1
    assert pattern.total_flags == 1

    # Bump again — different phrasing, same topic.
    ps.update_verdict_pattern(
        cwd, marker="symptom", reason="Symptom fix: composable doesn't exist",
        delta=1, session_id="s2", workstream_id="w2",
    )
    state2 = ps.load_state(cwd)
    # Should be same fingerprint (just one pattern)
    assert len(state2.verdict_patterns) == 1
    pattern2 = next(iter(state2.verdict_patterns.values()))
    assert pattern2.total_flags == 2
    assert pattern2.value_score == 2
    assert "w1" in pattern2.sample_workstream_ids
    assert "w2" in pattern2.sample_workstream_ids


def test_update_verdict_pattern_score_bounded(isolated):
    cwd = "/proj/cap"
    for _ in range(20):
        ps.update_verdict_pattern(cwd, marker="symptom",
                                   reason="symptom fix: foo",
                                   delta=1, session_id="s", workstream_id="w")
    state = ps.load_state(cwd)
    p = next(iter(state.verdict_patterns.values()))
    assert p.value_score == 10  # capped at +10


def test_update_verdict_pattern_negative_score_bounded(isolated):
    cwd = "/proj/cap-neg"
    for _ in range(20):
        ps.update_verdict_pattern(cwd, marker="symptom",
                                   reason="symptom fix: bar",
                                   delta=-1, session_id="s", workstream_id="w")
    state = ps.load_state(cwd)
    p = next(iter(state.verdict_patterns.values()))
    assert p.value_score == -10


def test_verdict_patterns_round_trip(isolated):
    cwd = "/proj/rt"
    ps.update_verdict_pattern(cwd, marker="rationalization",
                               reason="Rationalization: composable not yet defined",
                               delta=1, session_id="s", workstream_id="w")
    state = ps.load_state(cwd)
    j = json.dumps(state.to_dict(), sort_keys=True)
    restored = ps.ProjectState.from_dict(json.loads(j))
    assert len(restored.verdict_patterns) == 1


def test_find_finding_locates_across_stores(isolated):
    cwd = "/proj/find"
    ps.write_raw_digest(cwd, "s1", _digest("s1", 100.0, corrections=[
        {"rule": "single sighting", "why": "w", "how_to_apply": "h",
         "evidence_quote": "single"}
    ]))
    state = ps.rebuild_from_raw(cwd)
    ps.save_state(state)
    # Single-session correction is in quarantine.
    cid = state.quarantine[0].payload["id"]
    found = ps.find_finding(state, cid)
    assert found is not None
    kind, location, _ = found
    assert kind == "correction"
    assert location == "quarantine"
