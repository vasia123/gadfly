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
from gadfly.watchdog import evaluate_async, DEFAULT_MODEL  # noqa: E402

CASES_FILE = ROOT / "tests" / "fixtures" / "watchdog_corpus" / "cases.json"


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


async def _run_case(case: dict, model: str, timeout_s: float, max_turns: int):
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
    )
    dt = time.monotonic() - t0
    return res, dt


def _short(s: str, n: int = 140) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


async def main_async(args):
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
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

    print(f"Model: {args.model}  | cases: {len(cases)}  | "
          f"journal_mode: {bool(args.journal_mode)}")
    print(f"Timeout: {args.timeout}s per case")
    print("=" * 100)

    n_pass = 0
    n_fail = 0
    n_error = 0
    results = []

    for i, case in enumerate(cases):
        cid = case["case_id"]
        try:
            res, dt = await _run_case(case, args.model, args.timeout, args.max_turns)
        except Exception as exc:
            print(f"[{i:02d}] {cid:25s} ERROR {exc!r}")
            n_error += 1
            results.append({"case_id": cid, "status": "error", "error": repr(exc)})
            continue
        flagged = res.verdict.professional is False
        marker = "✓ CAUGHT" if flagged else "✗ MISSED"
        if flagged:
            n_pass += 1
        else:
            n_fail += 1
        cwd_tail = (case.get("cwd") or "?").split("/")[-1]
        print(f"[{i:02d}] {cid:25s} {marker}  {cwd_tail:14s} {dt:5.1f}s")
        orig = case["expected"]["original_reason"]
        print(f"     orig: {_short(orig, 130)}")
        print(f"     new : {_short(res.verdict.reason or '(silent)', 130)}")
        if res.error:
            print(f"     ERR : {res.error}")
        results.append({
            "case_id": cid,
            "status": "caught" if flagged else "missed",
            "verdict_professional": res.verdict.professional,
            "verdict_reason": res.verdict.reason,
            "verdict_suggestion": res.verdict.suggestion,
            "original_reason": orig,
            "latency_s": round(dt, 2),
            "error": res.error,
        })

    print("=" * 100)
    total_eval = n_pass + n_fail
    rate = (n_pass / total_eval * 100) if total_eval else 0.0
    print(f"Caught: {n_pass}/{total_eval}  ({rate:.1f}%)   "
          f"missed: {n_fail}   errored: {n_error}")

    if args.out:
        out = {
            "model": args.model,
            "journal_mode": bool(args.journal_mode),
            "caught": n_pass,
            "missed": n_fail,
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
                        "need 4+. Default 4 works across models.")
    p.add_argument("--only", help="Run only the case with this id")
    p.add_argument("--out", help="Write JSON report to this path")
    p.add_argument(
        "--journal-mode", action="store_true",
        help="Use SYSTEM_PROMPT_JOURNAL (default: legacy SYSTEM_PROMPT, "
             "since corpus cases do not carry journal state)",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
