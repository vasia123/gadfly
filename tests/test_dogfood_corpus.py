"""Structural + replay tests for the dogfood corpus.

The dogfood corpus collects laziness committed by gadfly's own author
*during* the development of gadfly — see
tests/fixtures/dogfood_corpus/README.md.

These tests:
  - validate every case loads and the schema matches the wrong_level
    corpus shape (so run_corpus.py --rubric trail consumes it),
  - validate every drift_kind reference resolves to a real entry in
    prompts.TRAIL_DRIFT_QUESTIONS,
  - GADFLY_LIVE=1 — replay each case against the configured model and
    assert the trail rubric catches the laziness.

Replay test is OFF by default (uses a real model = real cost). Live
benchmark results live in scratchpad / commit messages, not here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "dogfood_corpus" / "cases.json"


@pytest.fixture(scope="module")
def cases() -> list[dict[str, Any]]:
    assert CORPUS.is_file(), f"dogfood corpus missing: {CORPUS}"
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data, "dogfood corpus must be non-empty"
    return data


# --- Schema ---------------------------------------------------------------


_REQUIRED_TOP = {
    "case_id", "cwd", "session_id", "tool_name", "tool_input",
    "tool_response", "expected", "mode", "ack_excerpt", "session_context",
}
_REQUIRED_EXPECTED = {
    "professional", "original_reason", "category", "confidence",
    "drift_kind_primary", "drift_kind_secondary_acceptable",
    "lazy_pattern_summary", "why_this_action",
}
_REQUIRED_CONTEXT = {
    "recent_user_requests", "last_assistant_plan", "recent_actions",
    "action_index", "cwd",
}


def test_every_case_has_required_top_level_fields(cases):
    for c in cases:
        missing = _REQUIRED_TOP - set(c.keys())
        assert not missing, (
            f"{c.get('case_id', '?')} missing top fields: {missing}"
        )


def test_every_case_has_required_expected_fields(cases):
    for c in cases:
        cid = c["case_id"]
        missing = _REQUIRED_EXPECTED - set(c["expected"].keys())
        assert not missing, f"{cid} missing expected fields: {missing}"


def test_every_case_has_required_context_fields(cases):
    for c in cases:
        cid = c["case_id"]
        missing = _REQUIRED_CONTEXT - set(c["session_context"].keys())
        assert not missing, f"{cid} missing context fields: {missing}"


def test_every_case_marks_professional_false(cases):
    """Dogfood cases are by construction *positives* — the agent was lazy."""
    for c in cases:
        assert c["expected"]["professional"] is False, c["case_id"]


def test_every_drift_kind_is_real(cases):
    """drift_kind_primary AND every secondary must resolve to a real entry
    in TRAIL_DRIFT_QUESTIONS — otherwise the canonical question lookup at
    hook-emit time would silently fall back to 'other'."""
    from gadfly.prompts import TRAIL_DRIFT_QUESTIONS

    valid = set(TRAIL_DRIFT_QUESTIONS.keys())
    for c in cases:
        cid = c["case_id"]
        primary = c["expected"]["drift_kind_primary"]
        assert primary in valid, f"{cid} primary {primary!r} not in {valid}"
        for sec in c["expected"]["drift_kind_secondary_acceptable"]:
            assert sec in valid, f"{cid} secondary {sec!r} not in {valid}"


def test_lazy_message_carries_the_pushback_signal(cases):
    """The verbatim lazy assistant text MUST be present in
    last_assistant_plan — that is the load-bearing signal the trail
    rubric reads. A case without it is a fixture bug."""
    for c in cases:
        plan = c["session_context"]["last_assistant_plan"]
        assert plan and len(plan) > 100, (
            f"{c['case_id']} last_assistant_plan must hold the verbatim lazy text"
        )


def test_ack_excerpt_is_a_real_user_pushback(cases):
    """ack_excerpt holds the user's verbatim Russian pushback that
    proved the lazy interpretation."""
    for c in cases:
        ex = c["ack_excerpt"]
        assert isinstance(ex, str) and ex.strip(), c["case_id"]


def test_recent_actions_present(cases):
    """Trail rubric needs ≥3 recent_actions to detect longitudinal
    patterns. Cases with shorter trails are unfair to the rubric."""
    for c in cases:
        n = len(c["session_context"]["recent_actions"])
        assert n >= 3, (
            f"{c['case_id']} recent_actions={n} too short for trail rubric"
        )


# --- Live replay (gated, expensive) ---------------------------------------


@pytest.mark.skipif(
    os.environ.get("GADFLY_LIVE") != "1",
    reason="live replay needs GADFLY_LIVE=1 + a configured backend",
)
def test_dogfood_caught_by_trail_rubric_live(cases):
    """End-to-end: every dogfood case must trigger drift_detected=true
    on the configured production model. drift_kind must match primary
    OR fall into the secondary-acceptable set.
    """
    import asyncio
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_corpus import _seed_trail_from_actions, _summarize_case_action  # noqa: E402

    from gadfly import session as session_mod  # noqa: E402
    from gadfly import trail as trail_mod  # noqa: E402

    failures: list[str] = []
    for c in cases:
        cid = c["case_id"]
        ctx_d = c["session_context"]
        ctx = session_mod.SessionContext()
        ctx.recent_user_requests = list(ctx_d.get("recent_user_requests") or [])
        ctx.last_assistant_plan = ctx_d.get("last_assistant_plan")
        ctx.recent_actions = list(ctx_d.get("recent_actions") or [])
        ctx.action_index = int(ctx_d.get("action_index") or 0)

        seed = _seed_trail_from_actions(ctx.recent_actions)
        sid = f"dogfood_test_{cid}"
        trail_mod.save_current(sid, seed)

        res = asyncio.run(
            trail_mod.update_for_action_async(
                session_id=sid,
                action_index=ctx.action_index + 1,
                action_summary=_summarize_case_action(c),
                assistant_reasoning=ctx.last_assistant_plan,
                latest_user_message=(
                    ctx.recent_user_requests[-1]
                    if ctx.recent_user_requests else None
                ),
                journal_root_goal=None,
                cwd=c.get("cwd"),
            )
        )

        if res.drift_flag is None:
            failures.append(f"{cid}: no drift detected")
            continue
        kind = res.drift_flag.drift_kind
        expected_primary = c["expected"]["drift_kind_primary"]
        allowed = {expected_primary, *c["expected"]["drift_kind_secondary_acceptable"]}
        if kind not in allowed:
            failures.append(
                f"{cid}: drift kind={kind!r} not in expected {sorted(allowed)}"
            )

    assert not failures, "dogfood replay failures:\n  " + "\n  ".join(failures)
