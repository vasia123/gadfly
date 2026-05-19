"""Compare two watchdog-corpus reports produced by `run_corpus.py --out …`.

Use cases:
  - Detect prompt regression: today's Haiku run vs baseline.
  - Compare models: Haiku report vs Sonnet/Opus report.
  - Sanity-check a prompt experiment: report on the legacy prompt vs the
    updated one (run both, write to separate JSON files, diff).

Usage:
    python scripts/compare_corpus.py BASELINE.json CURRENT.json

Prints a table of cases where status differs (caught↔missed), plus
aggregate deltas. Exits non-zero when CURRENT has strictly more misses
than BASELINE — handy in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--allow-regressions", type=int, default=0,
                    help="exit 0 when num_regressions <= this (default: 0)")
    args = ap.parse_args()

    b = load(args.baseline)
    c = load(args.current)
    b_by = {r["case_id"]: r for r in b["results"]}
    c_by = {r["case_id"]: r for r in c["results"]}

    common = sorted(set(b_by) & set(c_by))
    regressions = []  # baseline=caught, current=missed
    improvements = []  # baseline=missed, current=caught
    for cid in common:
        bs, cs = b_by[cid]["status"], c_by[cid]["status"]
        if bs == "caught" and cs == "missed":
            regressions.append(cid)
        elif bs == "missed" and cs == "caught":
            improvements.append(cid)

    def _rate(rep):
        n_caught = sum(1 for r in rep["results"] if r["status"] == "caught")
        total = sum(1 for r in rep["results"] if r["status"] in ("caught", "missed"))
        return n_caught, total

    bn, bt = _rate(b)
    cn, ct = _rate(c)
    print(f"Baseline ({b.get('model','?')}): {bn}/{bt}  ({bn/bt:.1%})")
    print(f"Current  ({c.get('model','?')}): {cn}/{ct}  ({cn/ct:.1%})")
    print()

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}) — baseline caught, current missed:")
        for cid in regressions:
            cr = c_by[cid].get("verdict_reason") or "(silent)"
            print(f"  - {cid}: {cr[:160]}")
    else:
        print("REGRESSIONS: none")
    print()

    if improvements:
        print(f"IMPROVEMENTS ({len(improvements)}) — baseline missed, current caught:")
        for cid in improvements:
            print(f"  + {cid}")
    else:
        print("IMPROVEMENTS: none")

    only_b = sorted(set(b_by) - set(c_by))
    only_c = sorted(set(c_by) - set(b_by))
    if only_b:
        print(f"\nOnly in baseline: {only_b}")
    if only_c:
        print(f"\nOnly in current: {only_c}")

    if len(regressions) > args.allow_regressions:
        print(f"\nFAIL: {len(regressions)} regression(s) > allowed "
              f"{args.allow_regressions}", file=sys.stderr)
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()
