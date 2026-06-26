"""Build the words-benchmark from vetted_24h_words.json.

Inputs:
  - candidates_24h_words.json — (agent_text, user_text, ...) pairs
  - vetted_24h_words.json     — hand-curated ground truth labels

Output:
  - cases_24h_words.json          — positives (LAZY_WORDS)
  - cases_24h_words_negative.json — negatives (NEUTRAL)

Case schema is intentionally minimal — the WORDS rubric reads just
`agent_text` and answers "did this text show laziness?". `user_text`
stays as `ack_excerpt` for cross-validation but is NOT fed to the
rubric at evaluation time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "wrong_level_corpus"
CAND = FIX / "candidates_24h_words.json"
VET = FIX / "vetted_24h_words.json"
OUT_POS = FIX / "cases_24h_words.json"
OUT_NEG = FIX / "cases_24h_words_negative.json"


def main() -> None:
    cands = {c["msg_id"]: c for c in json.loads(CAND.read_text(encoding="utf-8"))}
    vetted = json.loads(VET.read_text(encoding="utf-8"))

    pos: list[dict] = []
    neg: list[dict] = []
    for v in vetted:
        mid = v["msg_id"]
        c = cands.get(mid)
        if c is None:
            print(f"warning: msg_id={mid} missing from candidates", file=sys.stderr)
            continue
        is_lazy = v["label"] == "LAZY_WORDS"
        case = {
            "case_id": (
                f"wl_words_{mid:03d}" if is_lazy
                else f"wl_words_neg_{mid:03d}"
            ),
            "cwd": "/home/vasis/projects_hobby/dnd-llm",
            "session_id": c["file"].replace(".jsonl", ""),
            "mode": (
                "words_positive" if is_lazy else "words_negative"
            ),
            "agent_text": c["agent_text"],
            "user_text": c["user_text"],
            "preceding_action_summary": c.get("preceding_action_summary", ""),
            "expected": {
                "professional": (not is_lazy),
                "label": v["label"],
                "lazy_kind": v.get("lazy_kind"),
                "lazy_markers": list(v.get("lazy_markers") or []),
                "category": v.get("note", "")[:200],
                "confidence": "high",
                "why_this_action": v.get("note", "")[:600],
            },
            "ack_excerpt": c["user_text"][:400],
        }
        (pos if is_lazy else neg).append(case)

    OUT_POS.write_text(
        json.dumps(pos, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_NEG.write_text(
        json.dumps(neg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"positives: {len(pos)} → {OUT_POS}")
    print(f"negatives: {len(neg)} → {OUT_NEG}")


if __name__ == "__main__":
    main()
