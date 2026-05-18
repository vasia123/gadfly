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
from gadfly import historian as historian_mod
from gadfly import journal as journal_mod
from gadfly import project_state as project_state_mod
from gadfly.pairs import Pair


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
def test_journal_update_against_real_haiku_creates_workstreams(tmp_path, monkeypatch):
    """Smoke: journal.update_for_action against a small two-pair
    conversation produces a journal with ≥1 workstream and a non-empty
    root_goal. Catches SDK-config regressions for the journal module
    (parallel to test_evaluate_completes_against_real_haiku_without_sdk_errors)."""
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    pairs = [
        Pair(
            assistant_text=None,
            user_text=(
                "investigate why marlin feature slows down EXL3 path and fix it"
            ),
        ),
        Pair(
            assistant_text="I'll start by benchmarking marlin-vs-base on Qwen3.",
            user_text="yes, start with the kernel-level bench",
        ),
    ]
    res = journal_mod.update_for_action(
        session_id="live-journal-smoke",
        action_index=1,
        action_summary=None,
        assistant_reasoning=None,
        pairs=pairs,
    )
    assert res.error is None, f"journal update failed: {res.error}"
    j = res.journal
    assert j.root_goal, "expected a non-empty root_goal"
    assert j.workstreams, "expected at least one workstream"
    g = j.root_goal.lower()
    assert any(m in g for m in ("marlin", "exl3", "regression", "bench")), (
        f"root_goal lost key nouns: {j.root_goal!r}"
    )


@live
def test_journal_aware_verdict_silence_on_repeated_pushed_back_flags(tmp_path, monkeypatch):
    """Phase-2 repetition rule: a workstream with 5 prior symptom flags
    where the agent pushed back must NOT receive a 6th echo from Haiku.

    Activates the journal-aware system prompt via GADFLY_JOURNAL_VERDICT=1.
    """
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    monkeypatch.setenv("GADFLY_JOURNAL_VERDICT", "1")

    flag_history = [
        journal_mod.FlagEvent(
            action_index=40 + i,
            reason="Symptom fix: dispatch workaround instead of fixing the 4.3us overhead at root cause",
            marker="symptom",
            agent_pushed_back=True,
            pushback="capture-state dispatch IS the root-cause fix; counter 4144→0 proves it",
        )
        for i in range(5)
    ]
    journal = journal_mod.Journal(
        root_goal=(
            "Investigate marlin EXL3 regression, find and fix all "
            "inefficiencies, build tiered benchmark coverage."
        ),
        workstreams=[
            journal_mod.Workstream(
                id="cuda-layernorm",
                title="cuda-layernorm 4.3us regression",
                status="in-progress",
                origin="action #34",
                notes=(
                    "Capture-state dispatch landed in normalization.rs:92 and "
                    "cuda_graph.rs:521,543. Counter 4144→0 on Qwen3 eager. "
                    "Agent argues this IS the root-cause fix."
                ),
                watchdog_flags=5,
                flag_history=flag_history,
                last_touched=45,
            )
        ],
        drift=journal_mod.Drift(
            initial_workstream_ids=["marlin", "tiered-bench"],
            observations="cuda-layernorm emerged unplanned.",
        ),
        action_index=46,
    )
    ctx = SessionContext(
        recent_user_requests=["профайлинг cuda-layernorm — root cause через capture state"],
        last_assistant_plan=(
            "Capture-state-aware dispatch landed. Verify counter=0 на Qwen3 "
            "eager, сделаю короткий bench."
        ),
        recent_actions=[],
        journal=journal,
    )
    res = evaluate(
        tool_name="Edit",
        tool_input={
            "file_path": "normalization.rs",
            "old_string": "        let xs = self.weight.broadcast_mul(xs)?;",
            "new_string": "        let xs = self.weight.broadcast_mul(xs)?;\n        // verified",
        },
        tool_response={"success": True},
        context=ctx,
    )
    assert res.error is None, f"verdict call failed: {res.error}"
    assert res.verdict.professional is True, (
        f"Repetition rule failed — 6th symptom flag was emitted despite 5 prior "
        f"pushed-back flags. verdict={res.verdict!r}"
    )


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


# --- Historian (Phase A + B) -----------------------------------------------


_HISTORIAN_FIXTURE_TRANSCRIPT = [
    # User asks for a real feature that produces durable findings.
    {"message": {"role": "user", "content": (
        "we should add a tiered benchmark framework that catches "
        "performance regressions at kernel, layer, and chain levels"
    )}},
    {"message": {"role": "assistant", "content": [
        {"type": "text", "text":
            "I'll plan it out. I'll start with criterion-based kernel "
            "benches (Tier 1), then layer-level (Tier 2), then chain "
            "(Tier 3). TODO: documentation of how to read the bench "
            "outputs will be a follow-up."},
    ]}},
    {"message": {"role": "user", "content": (
        "good. always run cargo fmt before any commit in this repo — "
        "we got bit last time by an unformatted commit"
    )}},
    {"message": {"role": "assistant", "content": [
        {"type": "text", "text":
            "Acknowledged: cargo fmt before each commit. Adding kernel "
            "bench scaffolding now."},
        {"type": "tool_use", "name": "Edit", "input": {
            "file_path": "benches/kernel_bench.rs",
            "old_string": "", "new_string": "// stub kernel bench"
        }},
        {"type": "tool_use", "name": "Edit", "input": {
            "file_path": "benches/layer_bench.rs",
            "old_string": "", "new_string": "// stub layer bench"
        }},
        {"type": "tool_use", "name": "Edit", "input": {
            "file_path": "benches/chain_bench.rs",
            "old_string": "", "new_string": "// stub chain bench"
        }},
    ]}},
]


def _write_fixture_transcript(path):
    import json as _json
    with path.open("w") as f:
        for e in _HISTORIAN_FIXTURE_TRANSCRIPT:
            f.write(_json.dumps(e) + "\n")


@live
def test_historian_distill_extracts_findings_from_fixture(tmp_path, monkeypatch):
    """Smoke: a single small but content-rich transcript produces at
    least one finding. If the SDK or prompt regresses (e.g. KeyError on
    ThinkingConfigDisabled, prompt too aggressive about empty lists),
    this catches it."""
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    tp = tmp_path / "session-fixture.jsonl"
    _write_fixture_transcript(tp)

    res = historian_mod.distill_session(
        cwd="/proj/live-test",
        session_id="session-fixture",
        transcript_path=tp,
        transcript_mtime=tp.stat().st_mtime,
    )
    assert res.error is None, f"distill failed: {res.error}"
    d = res.raw_digest
    total = (
        len(d["new_promises"])
        + len(d["new_corrections"])
        + len(d["knowledge_updates"])
    )
    assert total >= 1, (
        f"expected ≥1 finding from a content-rich transcript, got: "
        f"promises={len(d['new_promises'])}, "
        f"corrections={len(d['new_corrections'])}, "
        f"knowledge={len(d['knowledge_updates'])}"
    )
    # Every finding must carry an evidence_quote (anti-hallucination).
    for key in ("new_promises", "new_corrections", "knowledge_updates"):
        for item in d[key]:
            assert item.get("evidence_quote"), (
                f"{key} item missing evidence_quote: {item}"
            )


@live
def test_historian_retrieval_surfaces_seeded_finding(tmp_path, monkeypatch):
    """Phase B retrieval: after digesting the fixture, query for a topic
    we KNOW is in the corpus and verify the top hit matches.

    This is the MemoryArena-style active-use validation: storage isn't
    enough, retrieval must surface the right thing at the right moment.
    """
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    tp = tmp_path / "session-fixture.jsonl"
    _write_fixture_transcript(tp)

    cwd = "/proj/retrieval-live"
    res = historian_mod.distill_session(
        cwd=cwd,
        session_id="session-fixture",
        transcript_path=tp,
        transcript_mtime=tp.stat().st_mtime,
    )
    assert res.error is None
    # Re-aggregate so retrieval sees a fresh state.
    state = project_state_mod.rebuild_from_raw(cwd)
    project_state_mod.save_state(state)

    hits = historian_mod.find_relevant_priors(
        cwd,
        workstream_title="add benchmark coverage for kernel performance",
        file_paths=["benches/kernel_bench.rs"],
    )
    assert hits, "retrieval returned nothing for a query that should match"
    # The top hit should be related to benchmarking / kernels.
    top = hits[0]
    haystack = (top.title + " " + top.body).lower()
    assert any(
        m in haystack
        for m in ("bench", "kernel", "tier", "criterion", "performance")
    ), f"top hit unrelated to the query: {top!r}"


_MULTILINGUAL_FULFILL_TRANSCRIPT = [
    {"message": {"role": "user", "content": (
        "напиши хелпер для парсинга rate-limit заголовков "
        "в src/api/ratelimit.py — нужно прочитать X-RateLimit-* "
        "и вернуть структуру"
    )}},
    {"message": {"role": "assistant", "content": [
        {"type": "text", "text":
            "реализовал parse_ratelimit_headers в src/api/ratelimit.py: "
            "читает X-RateLimit-Limit/Remaining/Reset и возвращает "
            "RateLimitInfo. добавил тесты в tests/test_ratelimit.py — "
            "всё готово, тесты зелёные."},
        {"type": "tool_use", "name": "Edit", "input": {
            "file_path": "src/api/ratelimit.py",
            "old_string": "", "new_string": "def parse_ratelimit_headers(): ..."
        }},
        {"type": "tool_use", "name": "Edit", "input": {
            "file_path": "tests/test_ratelimit.py",
            "old_string": "", "new_string": "def test_parse(): ..."
        }},
    ]}},
]


@live
def test_historian_cross_session_fulfillment_russian(tmp_path, monkeypatch):
    """Language-agnostic fulfillment via real Haiku.

    Seed an open promise about parsing rate-limit headers (an English
    promise title). Run distill on a transcript written in Russian
    where the agent says "всё готово, тесты зелёные" while editing the
    matching files. Haiku must return the promise's id in
    `fulfilled_promise_ids` — proving the deterministic English-verb
    matcher is NOT load-bearing: a real model handles cross-language
    semantics.
    """
    monkeypatch.setenv("GADFLY_LOG_DIR", str(tmp_path / "log"))
    cwd = "/proj/multilingual-fulfill"

    # Seed an open promise via a synthetic raw digest.
    seed = {
        "session_id": "seed-session",
        "ts": 100.0,
        "new_promises": [{
            "title": "implement parse_ratelimit_headers helper in src/api/ratelimit.py",
            "evidence_quote": "TODO: parse X-RateLimit-* headers in a helper",
            "action_index": 3,
        }],
        "fulfilled_promises": [], "new_corrections": [],
        "knowledge_updates": [],
    }
    project_state_mod.write_raw_digest(cwd, "seed-session", seed)
    state = project_state_mod.rebuild_from_raw(cwd)
    project_state_mod.save_state(state)
    assert len(state.promises) == 1
    pid = next(iter(state.promises.keys()))
    assert state.promises[pid].status == "open"

    # Write the Russian-language fulfillment transcript.
    tp = tmp_path / "ru-session.jsonl"
    import json as _json
    with tp.open("w") as f:
        for e in _MULTILINGUAL_FULFILL_TRANSCRIPT:
            f.write(_json.dumps(e, ensure_ascii=False) + "\n")

    res = historian_mod.distill_session(
        cwd=cwd,
        session_id="ru-session",
        transcript_path=tp,
        transcript_mtime=tp.stat().st_mtime,
    )
    assert res.error is None, f"distill failed: {res.error}"
    fulfilled_ids = res.raw_digest.get("fulfilled_promise_ids") or []
    assert pid in fulfilled_ids, (
        f"Haiku did NOT recognise the Russian-language fulfillment.\n"
        f"  expected pid: {pid}\n"
        f"  fulfilled_promise_ids: {fulfilled_ids}\n"
        f"  This is the load-bearing multilingual case — if it fails, "
        f"the deterministic English-verb matcher would have been the "
        f"only fallback, and that fails silently for non-English users."
    )
    # Aggregator applies it.
    rebuilt = project_state_mod.rebuild_from_raw(cwd)
    assert rebuilt.promises[pid].status == "fulfilled"
    assert rebuilt.promises[pid].fulfilled_in_session == "ru-session"


@live
def test_evaluate_does_not_flag_use_of_earlier_defined_symbol():
    """Edit-window-blindness regression. SessionContext carries a
    `per_file_snapshots` block — the verbatim on-disk content of
    `src/util.py` showing `def helper(): ...`. Haiku reads the snapshot
    and recognises `helper` is defined, even though no `recent_action`
    contains the defining edit."""
    snapshots = {
        "src/util.py": (
            "def helper(x):\n"
            "    return x * 2\n"
            "\n"
            "def caller_a():\n"
            "    return helper(1)\n"
            "\n"
            "def caller_b():\n"
            "    return helper(2)\n"
            "\n"
            "def caller_c():\n"
            "    return helper(3)\n"
            "\n"
            "def caller_d():\n"
            "    pass  # callsite\n"
        ),
    }
    res = evaluate(
        tool_name="Edit",
        tool_input={
            "file_path": "src/util.py",
            "old_string": "pass  # callsite",
            "new_string": "return helper(42)  # uses helper defined 11 edits ago",
        },
        tool_response={"success": True},
        context=SessionContext(
            distilled_goal="ship util.py with multiple call sites of helper()",
            recent_user_requests=["wire up callers of helper"],
            last_assistant_plan="adding another caller of helper",
            recent_actions=[
                "Edit(other.py)\n  -: a\n  +: b",
                "Edit(other.py)\n  -: c\n  +: d",
                "Edit(other.py)\n  -: e\n  +: f",
                "Edit(other.py)\n  -: g\n  +: h",
                "Edit(other.py)\n  -: i\n  +: j",
            ],
            per_file_snapshots=snapshots,
            file_touch_trajectory=[(i, "Edit", "src/util.py") for i in (1, 3, 5, 10)],
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    # The key assertion: even if Haiku flags something, the reason MUST
    # NOT claim `helper` is undefined. The snapshot proves it exists.
    reason = (res.verdict.reason or "").lower()
    assert "helper" not in reason or "not defined" not in reason, (
        f"watchdog still claims helper is undefined despite snapshot:\n"
        f"  reason: {res.verdict.reason}\n"
        f"  suggestion: {res.verdict.suggestion}"
    )


@live
def test_evaluate_understands_snapshot_in_exotic_language():
    """The deleted regex hardcoded a list of definition keywords
    (fn|def|class|struct|impl|enum|trait|function|interface|type|
    const|let|var|func) — it would have missed any language whose
    definition keyword doesn't appear in that list. Snapshot
    approach hands Haiku the raw bytes; Haiku reads Lean / Erlang /
    Lisp / Elixir natively. This test passes a Lean-style snapshot
    where `def` is replaced with `theorem` (Lean's proof syntax).
    """
    snapshots = {
        "lib/Group.lean": (
            "namespace Group\n"
            "\n"
            "theorem mul_assoc (a b c : α) : (a * b) * c = a * (b * c) :=\n"
            "  by rw [← mul_assoc]\n"
            "\n"
            "theorem one_mul (a : α) : 1 * a = a :=\n"
            "  by rw [Group.one_mul]\n"
            "\n"
            "-- usage site below ↓\n"
            "example (x : α) : 1 * x = x := one_mul x\n"
        ),
    }
    res = evaluate(
        tool_name="Edit",
        tool_input={
            "file_path": "lib/Group.lean",
            "old_string": "example (x : α) : 1 * x = x := one_mul x",
            "new_string": (
                "example (x y : α) : 1 * (x * y) = x * y :=\n"
                "  by rw [one_mul]"
            ),
        },
        tool_response={"success": True},
        context=SessionContext(
            recent_user_requests=["прорабатываем Group.lean — добавь ещё example"],
            last_assistant_plan="extending the one_mul example with associativity",
            recent_actions=[],
            per_file_snapshots=snapshots,
            file_touch_trajectory=[(1, "Edit", "lib/Group.lean")],
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    reason = (res.verdict.reason or "").lower()
    # `one_mul` IS in the snapshot. Watchdog must not claim it's undefined.
    assert "one_mul" not in reason or "not defined" not in reason, (
        f"watchdog flagged `one_mul` as undefined despite Lean snapshot:\n"
        f"  reason: {res.verdict.reason}"
    )


@live
def test_evaluate_silences_drift_on_explicit_user_redirect():
    """Failure mode 2 regression. Build a journal with a 'finish historian'
    workstream, then evaluate a Bash action exploring a completely
    different project — but with a recent user message that explicitly
    redirects ("глянь на vllm-rust сессию"). The EXPLICIT USER REDIRECT
    rule should keep watchdog silent on this first action."""
    from gadfly.journal import Journal, Workstream, Drift
    j = Journal(
        root_goal="finish historian phase A/B/C",
        workstreams=[
            Workstream(
                id="historian-system",
                title="historian system, phase A/B/C completion",
                status="in-progress",
                last_touched=20,
                notes="phase B retrieval validated against gadfly corpus",
            )
        ],
        drift=Drift(initial_workstream_ids=["historian-system"]),
        action_index=21,
    )
    res = evaluate(
        tool_name="Bash",
        tool_input={
            "command": "ls -lt ~/.claude/projects/ | grep -i vllm",
            "description": "Find vllm-rust project dir",
        },
        tool_response={"exit_code": 0, "stdout": "-home-vasis-projects-hobby-vllm-rust"},
        context=SessionContext(
            recent_user_requests=["глянь на сессию vllm-rust"],
            last_assistant_plan="user asked to look at vllm-rust — finding the dir",
            recent_actions=[],
            journal=j,
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    # Must stay silent — user explicitly redirected.
    assert res.verdict.professional is True, (
        f"watchdog flagged drift despite explicit user redirect.\n"
        f"  reason: {res.verdict.reason}\n"
        f"  suggestion: {res.verdict.suggestion}"
    )


@live
def test_evaluate_silences_flag_when_plan_prescribes_action():
    """Round 3 regression: an active plan explicitly prescribing
    'REMOVE test_foo_*' must keep watchdog silent on the corresponding
    deletion, even if SYSTEM_PROMPT's symptom-fix pattern would
    otherwise fire on test deletion."""
    snapshot = (
        "def helper_kept(): return 1\n"
        "# helper_removed used to live here; deleted earlier in session\n"
    )
    test_snapshot = (
        "import pytest\n"
        "from src.util import helper_kept\n"
        "\n"
        "def test_helper_kept(): assert helper_kept() == 1\n"
    )
    res = evaluate(
        tool_name="Edit",
        tool_input={
            "file_path": "tests/test_util.py",
            "old_string": (
                "def test_helper_removed():\n"
                "    assert helper_removed() == 2\n"
            ),
            "new_string": "",
        },
        tool_response={"success": True},
        context=SessionContext(
            recent_user_requests=["execute the plan"],
            last_assistant_plan="removing the test for the deleted helper_removed",
            recent_actions=[],
            active_plan=(
                "## Approved Plan: cleanup of removed helper\n"
                "Step 1: Remove `helper_removed()` from src/util.py.\n"
                "Step 2: REMOVE test_helper_removed from tests/test_util.py — \n"
                "        no unit-under-test left.\n"
                "Step 3: Leave helper_kept and its test alone.\n"
            ),
            per_file_snapshots={
                "src/util.py": snapshot,
                "tests/test_util.py": test_snapshot,
            },
            file_touch_trajectory=[(1, "Edit", "src/util.py")],
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    # The plan explicitly authorized the deletion → silence expected.
    assert res.verdict.professional is True, (
        f"Watchdog flagged plan-prescribed deletion.\n"
        f"  reason: {res.verdict.reason}\n"
        f"  suggestion: {res.verdict.suggestion}"
    )


@live
def test_evaluate_does_not_flag_user_approved_action_against_stale_proposal():
    """Reconstruct the bizprofit volta dialogue. The user's most recent
    reply ('ставим') authorizes a Volta install proposed several turns
    earlier — the agent's MOST RECENT text said the opposite (use nvm
    instead). Without the dialogue block, Haiku misreads 'ставим' as
    approving the nvm pivot and flags the volta install as
    rationalization. With the block, Haiku traces the dialogue and
    classifies correctly."""
    pairs = [
        ("Поставлю Volta — она держит pinned-node на проекте, "
         "потребует мутацию ~/.bashrc", "погоди"),
        ("Ок, рассмотрю альтернативы без правки env",
         "что предлагаешь?"),
        ("Разблокируюсь без правки user-env: использую префикс "
         "`source ~/.nvm/nvm.sh && nvm use 22 &&` в Bash-вызовах. "
         "Volta — открытый вопрос для твоего workflow.",
         "ставим"),
    ]
    res = evaluate(
        tool_name="Bash",
        tool_input={
            "command": "curl https://get.volta.sh | bash",
            "description": "install Volta (canonical installer; mutates ~/.bashrc)",
        },
        tool_response={
            "exit_code": 0,
            "stdout": (
                "Installing latest version of Volta (2.0.2)\n"
                "Fetching... ✓\n"
                "Installing to /home/user/.volta\n"
                "Adding Volta to PATH in /home/user/.bashrc\n"
                "Volta is installed!\n"
            ),
        },
        context=SessionContext(
            recent_user_requests=["ставим"],
            last_assistant_plan=(
                "Ставлю Volta — это потребует мутации ~/.bashrc, но "
                "пользователь явно одобрил."
            ),
            recent_actions=[],
            recent_dialogue_pairs=pairs,
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    # Core regression: the original false positive flagged this as
    # "Rationalization: agent committed to no-env-mutation then mutated
    # env". With the dialogue block, Haiku must recognise that the
    # user reinstated the original Volta path via "ставим" referring
    # back to the earlier proposal. The reason MUST NOT claim the
    # action contradicts the agent's stated pivot.
    reason = (res.verdict.reason or "").lower()
    rationalization_signals = (
        "contradicts the stated pivot",
        "committed to nvm",
        "agent committed to",
        "agent explicitly said",
        "stated pivot",
        "no-env-mutation",
    )
    assert not any(s in reason for s in rationalization_signals), (
        f"Watchdog still misreads the multi-turn authorization.\n"
        f"  reason: {res.verdict.reason}\n"
        f"  suggestion: {res.verdict.suggestion}"
    )


@live
def test_evaluate_does_not_confuse_same_named_functions_across_files():
    """v-calc regression: agent edits MaterialsImport.vue; Haiku used
    to cite a definition in CoatingsImport.vue (different scope, same
    name) and flag "duplicate". The CURRENT EDIT TARGET marker on the
    snapshot must steer Haiku to the right file."""
    snapshots = {
        "frontend/components/admin/CoatingsImport.vue": (
            "<script setup>\n"
            "function downloadDictionary() {\n"
            "  // download coatings dictionary\n"
            "}\n"
            "</script>\n"
        ),
        "frontend/components/admin/MaterialsImport.vue": (
            "<script setup>\n"
            "function downloadTemplate() { /* materials template */ }\n"
            "function downloadDictionary() {\n"
            "  // download materials dictionary — defined once, line 162\n"
            "}\n"
            "</script>\n"
            "<template>\n"
            "  <a @click=\"downloadTemplate\">Скачать шаблон</a>\n"
            "</template>\n"
        ),
    }
    # Agent edits MaterialsImport.vue to ADD a template binding for
    # the already-defined downloadDictionary. Not a redefinition.
    res = evaluate(
        tool_name="Edit",
        tool_input={
            "file_path": "frontend/components/admin/MaterialsImport.vue",
            "old_string": (
                "  <a @click=\"downloadTemplate\">Скачать шаблон</a>\n"
            ),
            "new_string": (
                "  <a @click=\"downloadTemplate\">Скачать шаблон</a>\n"
                "  <a @click=\"downloadDictionary\">Скачать справочник</a>\n"
            ),
        },
        tool_response={"success": True},
        context=SessionContext(
            recent_user_requests=["добавь ссылку на скачивание справочника"],
            last_assistant_plan="wiring up the template binding for the existing downloadDictionary",
            recent_actions=[],
            per_file_snapshots=snapshots,
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    reason = (res.verdict.reason or "").lower()
    # The original FP had "duplicate function declaration" — must not
    # repeat with the target marker in place.
    assert "duplicate" not in reason, (
        f"Watchdog still confused same-named functions across files.\n"
        f"  reason: {res.verdict.reason}"
    )
    assert "coatingsimport" not in reason, (
        f"Watchdog cited wrong file (CoatingsImport.vue) when target was "
        f"MaterialsImport.vue.\n  reason: {res.verdict.reason}"
    )


@live
def test_evaluate_finds_symbol_added_in_middle_of_large_file():
    """Regression for the prompts.py case (gadfly session 7131d98b):
    file grew to 26KB, `_shorten_path` was added in the middle. The old
    head+tail truncation dropped 18KB / ~447 lines from the middle,
    hiding the function. Haiku then falsely claimed it wasn't defined.

    The fix gives the CURRENT EDIT TARGET a 60KB budget AND, when over,
    slices around the anchor (old_string) instead of head+tail.
    """
    # Build a 70KB synthetic file. The freshly-added utility lives in
    # the middle, near the edit anchor.
    head_pad = "// header padding\n" * 1500   # ~30KB
    tail_pad = "// tail padding\n" * 1500     # ~24KB
    target_body = (
        head_pad
        + "function _shorten_path(path, cwd) {\n"
        + "  return path.startsWith(cwd) ? path.slice(cwd.length + 1) : path;\n"
        + "}\n\n"
        + "// EDIT ANCHOR — call site referencing _shorten_path\n"
        + "const out = _shorten_path('/abs/path', '/abs');\n"
        + tail_pad
    )
    snapshots = {
        "src/utils/path.js": target_body,
    }
    res = evaluate(
        tool_name="Edit",
        tool_input={
            "file_path": "src/utils/path.js",
            "old_string": "// EDIT ANCHOR — call site referencing _shorten_path",
            "new_string": (
                "// EDIT ANCHOR — call site referencing _shorten_path\n"
                "// using helper added a few edits ago"
            ),
        },
        tool_response={"success": True},
        context=SessionContext(
            recent_user_requests=["wire up the call site"],
            last_assistant_plan="adding the comment near the call site",
            recent_actions=[],
            per_file_snapshots=snapshots,
            file_touch_trajectory=[(1, "Edit", "src/utils/path.js")],
        ),
    )
    assert res.error is None, f"SDK failed: {res.error}"
    reason = (res.verdict.reason or "").lower()
    # Critical: Haiku must NOT claim _shorten_path is undefined. The
    # function is in the snapshot, near the anchor — the new
    # edit-centered slicing must surface it.
    assert "_shorten_path" not in reason or "not defined" not in reason, (
        f"Watchdog still claims _shorten_path is undefined despite being "
        f"present in the snapshot near the edit anchor.\n"
        f"  reason: {res.verdict.reason}"
    )
    assert "does not exist" not in reason or "_shorten_path" not in reason, (
        f"Same as above with 'does not exist' phrasing.\n"
        f"  reason: {res.verdict.reason}"
    )
