"""Build wrong_level corpus 24h supplement: positives + negatives.

Inputs:
  - candidates_24h.json (61 messages w/ metadata)
  - vetted_24h.json — hand-curated ground truth. Manually inspected
    every candidate, dropped auto-injected noise (session-continuation,
    Stop hook reactivations, "Continue from where you left off"
    artifacts), overrode classifier mislabels (e.g. msg 23 — design
    question not pushback), kept only confidence-high or carefully-
    judged borderline cases.

  candidates_24h_classified.json is kept for audit (what the agent
  classifier produced before manual review) but NOT consumed here.

Output:
  - cases_24h.json — positive cases (PUSHBACK label)
  - cases_24h_negative.json — negative cases (CLEAN label)

Both files use the same schema as `cases.json` so run_corpus.py replays
them as-is. Negative cases get mode="wrong_level_negative" so the
test runner knows: success = silence (no drift_detected).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gadfly import session as session_mod  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "wrong_level_corpus"
CANDIDATES = FIX / "candidates_24h.json"
VETTED = FIX / "vetted_24h.json"
OUT_POS = FIX / "cases_24h.json"
OUT_NEG = FIX / "cases_24h_negative.json"
DNDDIR = Path(os.path.expanduser(
    "~/.claude/projects/-home-vasis-projects-hobby-dnd-llm"
))


def _index_transcript(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _serialize_context(ctx) -> dict:
    return {
        "recent_user_requests": list(ctx.recent_user_requests),
        "last_assistant_plan": ctx.last_assistant_plan,
        "recent_actions": list(ctx.recent_actions),
        "action_index": ctx.action_index,
        "cwd": ctx.cwd,
        "per_file_snapshots": dict(ctx.per_file_snapshots),
        "file_touch_trajectory": [list(t) for t in ctx.file_touch_trajectory],
        "active_plan": ctx.active_plan,
        "recent_dialogue_pairs": [list(p) for p in ctx.recent_dialogue_pairs],
        "recent_bash_actions": [list(b) for b in ctx.recent_bash_actions],
    }


def _slice_to_tmp(entries: list[dict], last_idx: int) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in entries[: last_idx + 1]:
        tmp.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


def _tool_result_payload_for(
    entries: list[dict], assistant_idx: int, tu_id: str
) -> dict:
    """Find the tool_result block matching `tu_id` and shape it like
    the watchdog's tool_response."""
    for j in range(assistant_idx + 1, min(assistant_idx + 6, len(entries))):
        e = entries[j]
        if e.get("type") != "user":
            continue
        for blk in (e.get("message") or {}).get("content") or []:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_result":
                continue
            if blk.get("tool_use_id") != tu_id:
                continue
            content = blk.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for x in content:
                    if isinstance(x, dict) and x.get("type") == "text":
                        text += (x.get("text") or "")
            return {
                "is_error": bool(blk.get("is_error", False)),
                "content": text[:8000],
            }
    return {"is_error": False, "content": "(no matching tool_result captured)"}


def _build_case(
    cand: dict,
    cls: dict,
    transcript: Path,
    entries: list[dict],
    mode: str,
) -> dict | None:
    assistant_idx = cand["prev_assistant_idx"]
    # Find the specific tool_use we attached to so we can pull its input
    # and matching tool_result.
    msg = entries[assistant_idx].get("message") or {}
    tu_block = None
    for blk in reversed(msg.get("content") or []):
        if (
            isinstance(blk, dict)
            and blk.get("type") == "tool_use"
            and blk.get("name") in {"Edit", "Write", "MultiEdit", "Bash"}
        ):
            tu_block = blk
            break
    if tu_block is None:
        return None
    tool_name = tu_block.get("name", "?")
    tool_input = tu_block.get("input") or {}
    tu_id = tu_block.get("id", "")
    tool_response = _tool_result_payload_for(entries, assistant_idx, tu_id)

    # Slice the transcript at the assistant's tool_result so
    # session.load() sees the same state the watchdog would have seen.
    # The slice end is the user message that holds the tool_result.
    result_idx = assistant_idx
    for j in range(assistant_idx + 1, min(assistant_idx + 6, len(entries))):
        e = entries[j]
        if e.get("type") != "user":
            continue
        if any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            and b.get("tool_use_id") == tu_id
            for b in (e.get("message") or {}).get("content") or []
        ):
            result_idx = j
            break

    slice_path = _slice_to_tmp(entries, result_idx)
    try:
        ctx = session_mod.load(
            slice_path,
            distill=False,
            current_tool_input=tool_input,
            cwd="/home/vasis/projects_hobby/dnd-llm",
        )
    except Exception as exc:
        print(f"  session.load failed for {cand['msg_id']}: {exc!r}")
        return None
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass

    prefix = "wl_24h" if mode == "wrong_level_positive" else "wl_24h_neg"
    expected: dict = {
        "professional": (mode != "wrong_level_positive"),
        "original_reason": cand["user_text"][:600],
        "category": cls.get("note", "")[:200],
        "confidence": "high",  # vetted = manually-reviewed = high
        "why_this_action": cls.get("note", "")[:600],
    }
    if mode == "wrong_level_positive":
        expected["drift_kind_primary"] = cls.get("drift_kind_hint") or "other"
        # Allow the model to pick a closely-related kind without failing
        # the case — this matches dogfood_corpus's approach.
        expected["drift_kind_secondary_acceptable"] = [
            k for k in {
                "incomplete_coverage", "rationalization",
                "premature_ceiling", "wrong_layer", "rule_skip",
                "hardcoded_instance", "recon_as_work", "other",
            }
            if k != expected["drift_kind_primary"]
        ]

    return {
        "case_id": f"{prefix}_{cand['msg_id']:03d}_{tool_name.lower()}",
        "cwd": "/home/vasis/projects_hobby/dnd-llm",
        "session_id": transcript.stem,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_response": tool_response,
        "expected": expected,
        "mode": mode,
        "ack_excerpt": cand["user_text"][:400],
        "session_context": _serialize_context(ctx),
    }


def main() -> None:
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    vetted = json.loads(VETTED.read_text(encoding="utf-8"))
    cand_by_id = {c["msg_id"]: c for c in candidates}
    vet_by_id = {c["msg_id"]: c for c in vetted}

    # vetted is the authoritative subset — everything not in vetted is
    # excluded as auto-injected noise (session-continuation, Stop hook
    # reactivations, "Continue from where you left off") or as a
    # low-confidence case dropped after manual review.
    msg_ids = sorted(vet_by_id.keys())

    by_file: dict[str, list[int]] = {}
    for mid in msg_ids:
        c = cand_by_id.get(mid)
        if not c:
            print(f"vetted msg_id={mid} missing from candidates_24h.json — skipping")
            continue
        by_file.setdefault(c["file"], []).append(mid)

    positives: list[dict] = []
    negatives: list[dict] = []
    skipped_build = 0

    for fname, ids in by_file.items():
        path = DNDDIR / fname
        if not path.is_file():
            print(f"missing transcript: {fname}")
            continue
        entries = _index_transcript(path)
        for mid in ids:
            cand = cand_by_id[mid]
            vet = vet_by_id[mid]
            label = vet["label"]
            if label == "PUSHBACK":
                mode = "wrong_level_positive"
                bucket = positives
            elif label == "CLEAN":
                mode = "wrong_level_negative"
                bucket = negatives
            else:
                print(f"unknown label for msg_id={mid}: {label!r}")
                continue
            case = _build_case(cand, vet, path, entries, mode)
            if case is None:
                skipped_build += 1
                continue
            bucket.append(case)

    OUT_POS.write_text(
        json.dumps(positives, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUT_NEG.write_text(
        json.dumps(negatives, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    n_excluded = len(candidates) - len(vetted)
    print(f"positives (PUSHBACK): {len(positives)} → {OUT_POS}")
    print(f"negatives (CLEAN):    {len(negatives)} → {OUT_NEG}")
    print(f"excluded from vetting: {n_excluded} (auto-injected noise + low conf)")
    print(f"skipped (build failed): {skipped_build}")


if __name__ == "__main__":
    main()
