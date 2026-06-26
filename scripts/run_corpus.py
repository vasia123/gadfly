"""Replay the watchdog-corpus through evaluate() and print a report.

Use case 1 — regression check on current Haiku stack:
    python scripts/run_corpus.py

Use case 2 — comparative eval against another model:
    python scripts/run_corpus.py --model claude-sonnet-4-6
    python scripts/run_corpus.py --model claude-opus-4-7

Use case 3 — single case for debugging:
    python scripts/run_corpus.py --only case_011_edit

Output: per-case verdict (✓/✗) + reason; aggregate catch rate at the end.

A case PASSES when evaluate() returns professional=False (the watchdog
catches the same thing it caught originally). It FAILS when the model
returns professional=True (regression — the catch was lost).

Cost note: ~10-15s per case via subscription billing. 24 cases ≈ 4-6 min.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gadfly import session as session_mod  # noqa: E402
from gadfly import trail as trail_mod  # noqa: E402
from gadfly.backends import select_backend  # noqa: E402
from gadfly.watchdog import evaluate_async, DEFAULT_MODEL  # noqa: E402

DEFAULT_POSITIVE = ROOT / "tests" / "fixtures" / "watchdog_corpus" / "cases.json"
DEFAULT_NEGATIVE = ROOT / "tests" / "fixtures" / "watchdog_corpus" / "cases_negative.json"
DEFAULT_WRONG_LEVEL = (
    ROOT / "tests" / "fixtures" / "wrong_level_corpus" / "cases.json"
)


_RECON_BASH_TOKENS = (
    "grep ", "rg ", "ls ", "find ", "cat ", "head ", "tail ", "wc ",
    "git status", "git log", "git diff", "git blame", "git show",
)


def _seed_abstraction_level(action_summary: str) -> str:
    """Heuristic classifier for synthesized seed breadcrumbs.

    Edits and Writes are concrete commits — `instance` is the right
    starting level (the rubric only escalates to drift when MULTIPLE
    instance-level breadcrumbs accumulate, which is precisely what
    `hardcoded_instance` looks for).

    Reconnaissance tool calls (Read, grep/ls/find via Bash, git
    status/log) are `unclear` — they advance no commit. The trail's
    `recon_as_work` pattern fires when ≥5 of these stack up.

    Without this classifier the seed defaulted to all-unclear which
    triggered recon_as_work on every case regardless of the actual
    action shape.
    """
    if not action_summary:
        return "unclear"
    s = action_summary.strip()
    if s.startswith(("Edit(", "Write(", "MultiEdit(")):
        return "instance"
    if s.startswith("Read("):
        return "unclear"
    if s.startswith("Bash:"):
        rest = s[len("Bash:"):].strip().lower()
        if any(rest.startswith(tok) for tok in _RECON_BASH_TOKENS):
            return "unclear"
        # Build / test / migrate / install — treat as instance commit.
        return "instance"
    return "unclear"


def _seed_trail_from_actions(recent_actions: list[str]) -> trail_mod.Trail:
    """Manufacture a synthetic trail from a case's `recent_actions` list.

    The corpus cases predate the trail module — they have no real
    breadcrumb history. To exercise the trail rubric on them we synthesize
    a seed: one breadcrumb per prior action, with abstraction_level
    inferred from the action shape (see `_seed_abstraction_level`).

    The level is load-bearing for the rubric's longitudinal pattern
    rules — flat-unclear seeds trigger recon_as_work on every case.
    """
    seed = trail_mod.empty_trail()
    for i, a in enumerate(recent_actions[-trail_mod.PROMPT_WINDOW:], start=1):
        seed.breadcrumbs.append(
            trail_mod.Breadcrumb(
                action_index=i,
                breadcrumb_text=(a or "")[:160],
                abstraction_level=_seed_abstraction_level(a or ""),
                action_summary=(a or "")[:240],
                ts=0.0,
            )
        )
    seed.action_index = len(recent_actions)
    return seed


def _ctx_from_dict(d: dict) -> session_mod.SessionContext:
    ctx = session_mod.SessionContext()
    ctx.recent_user_requests = list(d.get("recent_user_requests") or [])
    ctx.last_assistant_plan = d.get("last_assistant_plan")
    ctx.recent_actions = list(d.get("recent_actions") or [])
    ctx.action_index = int(d.get("action_index") or 0)
    ctx.cwd = d.get("cwd") or ""
    ctx.per_file_snapshots = dict(d.get("per_file_snapshots") or {})
    ctx.file_touch_trajectory = [
        tuple(t) for t in (d.get("file_touch_trajectory") or [])
    ]
    ctx.active_plan = d.get("active_plan")
    ctx.recent_dialogue_pairs = [
        tuple(p) for p in (d.get("recent_dialogue_pairs") or [])
    ]
    ctx.recent_bash_actions = [
        tuple(b) for b in (d.get("recent_bash_actions") or [])
    ]
    # journal stays None — corpus replay does not exercise journal-mode
    return ctx


async def _run_case(case: dict, model: str, timeout_s: float, max_turns: int,
                    backend=None):
    ctx = _ctx_from_dict(case["session_context"])
    t0 = time.monotonic()
    res = await evaluate_async(
        tool_name=case["tool_name"],
        tool_input=case["tool_input"],
        tool_response=case["tool_response"],
        context=ctx,
        model=model,
        timeout_s=timeout_s,
        max_turns=max_turns,
        backend=backend,
    )
    dt = time.monotonic() - t0
    return res, dt


def _summarize_case_action(case: dict) -> str:
    """One-line summary of the case's trigger action, for the trail prompt."""
    tn = case.get("tool_name") or "?"
    ti = case.get("tool_input") or {}
    if tn == "Edit":
        fp = ti.get("file_path", "?")
        return (
            f"Edit({fp})\n  -: {(ti.get('old_string') or '')[:180]}"
            f"\n  +: {(ti.get('new_string') or '')[:180]}"
        )
    if tn == "Write":
        return f"Write({ti.get('file_path', '?')})"
    if tn == "MultiEdit":
        return f"MultiEdit({ti.get('file_path', '?')}, {len(ti.get('edits') or [])} edits)"
    if tn == "Bash":
        return f"Bash: {(ti.get('command') or '')[:240]}"
    return tn


async def _run_case_trail(
    case: dict, model: str, timeout_s: float, backend
):
    """Trail-mode replay. Seeds a synthetic trail from recent_actions,
    saves it under the case_id as session_id, then calls
    trail.update_for_action so the model sees the prior path + the
    current action and decides drift.

    Returns (TrailUpdateResult, elapsed_seconds).
    """
    ctx = _ctx_from_dict(case["session_context"])
    sid = f"corpus_{case['case_id']}"
    # Seed: synthesize prior breadcrumbs from recent_actions.
    seed = _seed_trail_from_actions(ctx.recent_actions)
    trail_mod.save_current(sid, seed)
    t0 = time.monotonic()
    res = await trail_mod.update_for_action_async(
        session_id=sid,
        action_index=(ctx.action_index or 0) + 1,
        action_summary=_summarize_case_action(case),
        assistant_reasoning=ctx.last_assistant_plan,
        latest_user_message=(
            ctx.recent_user_requests[-1]
            if ctx.recent_user_requests else None
        ),
        journal_root_goal=None,  # corpus has no journal state
        cwd=case.get("cwd") or None,
        model=model,
        timeout_s=timeout_s,
        backend=backend,
    )
    dt = time.monotonic() - t0
    return res, dt


def _short(s: str, n: int = 140) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


async def main_async(args):
    if args.cases:
        cases_path = Path(args.cases)
    elif args.wrong_level:
        cases_path = DEFAULT_WRONG_LEVEL
    elif args.negative:
        cases_path = DEFAULT_NEGATIVE
    else:
        cases_path = DEFAULT_POSITIVE
    if not cases_path.exists():
        print(f"ERROR: cases file not found: {cases_path}", file=sys.stderr)
        sys.exit(2)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    # Auto-detect mode from first case if not forced by --negative.
    # wrong_level corpus cases carry mode="wrong_level_positive"; their
    # success criterion matches the positive corpus (model must flag).
    if args.negative or (cases and cases[0].get("mode") == "negative"):
        corpus_mode = "negative"
    else:
        corpus_mode = "positive"
    if args.only:
        cases = [c for c in cases if c["case_id"] == args.only]
        if not cases:
            print(f"No case with id={args.only}", file=sys.stderr)
            sys.exit(2)

    # Ensure journal-mode is OFF for replay (corpus cases predate journal
    # capture or don't carry journal state). GADFLY_JOURNAL_VERDICT=0 forces
    # legacy SYSTEM_PROMPT. Override via --journal-mode if desired.
    if not args.journal_mode:
        os.environ["GADFLY_JOURNAL_VERDICT"] = "0"

    backend = None
    if args.backend in ("openai_compat", "openai_json"):
        api_key = ""
        if args.api_key_env and args.api_key_env != "NONE":
            api_key = os.environ.get(args.api_key_env, "")
            if not api_key:
                print(f"WARNING: env var {args.api_key_env} is empty",
                      file=sys.stderr)
        if not args.base_url:
            print(f"ERROR: --base-url is required for --backend {args.backend}",
                  file=sys.stderr)
            sys.exit(2)
        extra: dict[str, str] = {}
        if args.extra_headers:
            try:
                extra = json.loads(args.extra_headers)
            except json.JSONDecodeError as exc:
                print(f"ERROR: --extra-headers is not valid JSON: {exc}",
                      file=sys.stderr)
                sys.exit(2)
        backend = select_backend(
            args.backend,
            base_url=args.base_url,
            api_key=api_key,
            extra_headers=extra or None,
        )

    print(f"Backend: {args.backend}  | model: {args.model}  | "
          f"cases: {len(cases)} ({corpus_mode})  | "
          f"journal_mode: {bool(args.journal_mode)}")
    if args.backend in ("openai_compat", "openai_json"):
        masked = (api_key[:4] + "…") if api_key else "(none)"
        print(f"Endpoint: {args.base_url}  | api_key: {masked}")
    print(f"Timeout: {args.timeout}s per case")
    print("=" * 100)

    n_pass = 0
    n_fail = 0
    n_error = 0
    results = []

    for i, case in enumerate(cases):
        cid = case["case_id"]
        try:
            if args.rubric == "trail":
                tres, dt = await _run_case_trail(
                    case, args.model, args.timeout, backend=backend,
                )
            else:
                res, dt = await _run_case(
                    case, args.model, args.timeout, args.max_turns,
                    backend=backend,
                )
        except Exception as exc:
            print(f"[{i:02d}] {cid:25s} ERROR {exc!r}")
            n_error += 1
            results.append({"case_id": cid, "status": "error", "error": repr(exc)})
            continue

        # Reduce the rubric to a single "flagged?" boolean so the
        # success/failure logic is shared between rubrics.
        if args.rubric == "trail":
            flagged = (
                tres.drift_flag is not None
                and tres.drift_flag.delivered_to_agent
            )
            verdict_reason_text = (
                tres.drift_flag.drift_reasoning
                if tres.drift_flag else "(silent)"
            )
            verdict_suggestion_text = (
                trail_mod.question_for_kind(tres.drift_flag.drift_kind)
                if tres.drift_flag else ""
            )
            verdict_professional = not flagged
            raw_args = {
                "advances_trail": (
                    len(tres.trail.breadcrumbs) >
                    len(_seed_trail_from_actions(
                        _ctx_from_dict(case["session_context"]).recent_actions
                    ).breadcrumbs)
                ),
                "drift_detected": tres.drift_flag is not None,
                "drift_kind": (
                    tres.drift_flag.drift_kind if tres.drift_flag else None
                ),
                "cited_action_indexes": (
                    list(tres.drift_flag.cited_action_indexes)
                    if tres.drift_flag else []
                ),
                "suppressed": (
                    bool(tres.drift_flag.suppressed)
                    if tres.drift_flag else False
                ),
            }
            err = tres.error
        else:
            flagged = res.verdict.professional is False
            verdict_reason_text = res.verdict.reason or "(silent)"
            verdict_suggestion_text = res.verdict.suggestion
            verdict_professional = res.verdict.professional
            raw_args = res.raw_verdict_args
            err = res.error

        # Positive corpus: success = model FIRES (flagged=True).
        # Negative corpus: success = model STAYS SILENT (flagged=False).
        if corpus_mode == "negative":
            ok = (not flagged)
            marker = "✓ SILENT" if ok else "✗ FLAGGED"
            status = "correct_silence" if ok else "false_flag"
        else:
            ok = flagged
            marker = "✓ CAUGHT" if ok else "✗ MISSED"
            status = "caught" if ok else "missed"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        cwd_tail = (case.get("cwd") or "?").split("/")[-1]
        print(f"[{i:02d}] {cid:25s} {marker}  {cwd_tail:14s} {dt:5.1f}s")
        orig = case["expected"]["original_reason"]
        print(f"     orig: {_short(orig, 130)}")
        print(f"     new : {_short(verdict_reason_text, 130)}")
        if err:
            print(f"     ERR : {err}")
        results.append({
            "case_id": cid,
            "status": status,
            "verdict_professional": verdict_professional,
            "verdict_reason": verdict_reason_text,
            "verdict_suggestion": verdict_suggestion_text,
            "raw_verdict_args": raw_args,
            "original_reason": orig,
            "latency_s": round(dt, 2),
            "error": err,
        })

    print("=" * 100)
    total_eval = n_pass + n_fail
    rate = (n_pass / total_eval * 100) if total_eval else 0.0
    metric = "Correct silence" if corpus_mode == "negative" else "Caught"
    print(f"{metric}: {n_pass}/{total_eval}  ({rate:.1f}%)   "
          f"missed: {n_fail}   errored: {n_error}")

    if args.out:
        out = {
            "model": args.model,
            "mode": corpus_mode,
            "journal_mode": bool(args.journal_mode),
            "passed": n_pass,
            "failed": n_fail,
            "errored": n_error,
            "results": results,
        }
        Path(args.out).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Report → {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--max-turns", type=int, default=4,
                   help="SDK max_turns. Haiku needs 2; Sonnet/Opus often "
                        "need 4+. Default 4 works across models. "
                        "Ignored for --backend openai_compat.")
    p.add_argument("--backend",
                   choices=("claude_sdk", "openai_compat", "openai_json"),
                   default="claude_sdk",
                   help="Backend transport. Default: claude_sdk (subscription). "
                        "openai_compat — forced tool-call. "
                        "openai_json — response_format=json_object (experimental).")
    p.add_argument("--base-url", default=None,
                   help="Required for --backend openai_compat. "
                        "Examples: https://api.openai.com/v1, "
                        "https://api.anthropic.com/v1, "
                        "https://openrouter.ai/api/v1, "
                        "http://localhost:8000/v1.")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                   help="Env var holding the API key (default: OPENAI_API_KEY). "
                        "Pass NONE for unauthenticated local endpoints.")
    p.add_argument("--extra-headers", default=None,
                   help='Extra request headers as JSON, e.g. '
                        '\'{"anthropic-version":"2023-06-01"}\'.')
    p.add_argument("--only", help="Run only the case with this id")
    p.add_argument("--out", help="Write JSON report to this path")
    p.add_argument(
        "--negative", action="store_true",
        help="Run negative-corpus mode: success = model stays silent "
             "(matches the agent's pushback that the original flag was wrong). "
             "Loads cases_negative.json by default; overridable via --cases.",
    )
    p.add_argument(
        "--wrong-level", action="store_true",
        help="Run wrong_level corpus (dnd-llm pushbacks). Positive-mode "
             "scoring (success = model flags). Loads "
             "tests/fixtures/wrong_level_corpus/cases.json; overridable "
             "via --cases.",
    )
    p.add_argument(
        "--rubric", choices=("verdict", "trail"), default="verdict",
        help="Which rubric to exercise. `verdict` (default) = the watchdog "
             "professional/reason verdict. `trail` = trail.update_for_action "
             "synthesizing a seed trail from recent_actions and asking for "
             "drift on the trigger action. For trail mode, success = "
             "drift_detected (positive/wrong_level corpora) or no drift "
             "(negative corpus).",
    )
    p.add_argument("--cases", default=None,
                   help="Override cases file path. Default depends on --negative.")
    p.add_argument(
        "--journal-mode", action="store_true",
        help="Use SYSTEM_PROMPT_JOURNAL (default: legacy SYSTEM_PROMPT, "
             "since corpus cases do not carry journal state)",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
