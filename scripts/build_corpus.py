"""Build a replayable corpus of watchdog flags the agent acknowledged.

For each candidate verdict in ~/.claude/gadfly/log/*.jsonl that the agent
later acknowledged in transcript text:

  1. Locate the target tool_use in the source transcript.
  2. Truncate the transcript at (and including) the matching tool_result.
  3. Run the CURRENT session.load() on the truncated slice — captures all
     SessionContext fields the watchdog would have seen at that moment.
  4. Serialize SessionContext + tool_name/tool_input/tool_response + the
     original verdict reason + ack excerpt into one JSON fixture.

The resulting corpus is replayable forever, decoupled from transcript
schema drift and from the user's current filesystem state (snapshots
are frozen inline). Tests load these fixtures and re-run evaluate()
through current prompts against any chosen model.

Run: python scripts/build_corpus.py
Output: tests/fixtures/watchdog_corpus/cases.json
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gadfly import session as session_mod  # noqa: E402

LOG_GLOB = os.path.expanduser("~/.claude/gadfly/log/*.jsonl")
OUT_DIR = ROOT / "tests" / "fixtures" / "watchdog_corpus"

ACK_NEGATIVE = re.compile(
    r"watchdog\s+(ошиб|вр[еёe]т|галлюц|паник|прома|"
    r"перепутал|игнорир|снова промахнулся)",
    re.IGNORECASE,
)
ACK_POSITIVE = re.compile(
    r"(watchdog\s+(прав|справедлив|снова прав|снова заметил)|"
    r"critical\s+watchdog\s+catch|"
    r"\bcritical\s+catch|"
    r"справедливо|согласен|учту|учитыва|"
    r"я поторопил|поспешил|переделаю|перепиш(?:у|ем)|"
    r"откач(?:у|иваю)|откатываю)",
    re.IGNORECASE,
)


def find_flagged_acks() -> list[dict]:
    """Walk all audit logs, return verdict records where the agent later
    acknowledged the flag in their next assistant message."""
    out: list[dict] = []
    for fp in sorted(glob.glob(LOG_GLOB)):
        flagged = []
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") != "verdict":
                    continue
                v = r.get("verdict") or {}
                if v.get("professional") is False:
                    flagged.append(r)
        if not flagged:
            continue
        tp = (flagged[0].get("payload") or {}).get("transcript_path")
        if not tp or not os.path.exists(tp):
            continue
        # Load transcript entries
        entries: list[dict] = []
        try:
            with open(tp) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
        except OSError:
            continue
        for rec in flagged:
            ack = _find_ack(entries, rec)
            if not ack:
                continue
            out.append({
                "verdict_rec": rec,
                "transcript_path": tp,
                "entries": entries,
                "target_idx": ack["target_idx"],
                "result_idx": ack["result_idx"],
                "ack_excerpt": ack["ack_text"][:600],
            })
    return out


def _find_ack(entries: list[dict], verdict_rec: dict) -> dict | None:
    payload = verdict_rec.get("payload") or {}
    target_input = payload.get("tool_input") or {}
    target_tool = payload.get("tool_name")
    target_use_id = payload.get("tool_use_id")
    target_input_sig = json.dumps(target_input, sort_keys=True)[:400]

    # 1. find assistant message with matching tool_use
    target_idx = None
    for i, m in enumerate(entries):
        if m.get("type") != "assistant":
            continue
        msg = m.get("message") or {}
        for blk in msg.get("content") or []:
            if blk.get("type") != "tool_use":
                continue
            if target_use_id and blk.get("id") == target_use_id:
                target_idx = i
                break
            if (
                blk.get("name") == target_tool
                and json.dumps(blk.get("input") or {}, sort_keys=True)[:400] == target_input_sig
            ):
                target_idx = i
                break
        if target_idx is not None:
            break
    if target_idx is None:
        return None

    # 2. find the next user message containing tool_result for this tool_use
    result_idx = None
    for j in range(target_idx + 1, min(target_idx + 5, len(entries))):
        m = entries[j]
        if m.get("type") != "user":
            continue
        msg = m.get("message") or {}
        for blk in msg.get("content") or []:
            if blk.get("type") == "tool_result":
                if target_use_id and blk.get("tool_use_id") == target_use_id:
                    result_idx = j
                    break
                # fallback: any tool_result in the next user message
                result_idx = j
                break
        if result_idx is not None:
            break
    if result_idx is None:
        return None

    # 3. find next assistant text message with ack language
    ack_text = None
    for k in range(result_idx + 1, min(result_idx + 6, len(entries))):
        m = entries[k]
        if m.get("type") != "assistant":
            continue
        msg = m.get("message") or {}
        chunks = [
            b.get("text", "")
            for b in (msg.get("content") or [])
            if b.get("type") == "text"
        ]
        txt = "\n".join(chunks).strip()
        if not txt:
            continue
        if ACK_NEGATIVE.search(txt):
            return None
        if ACK_POSITIVE.search(txt):
            ack_text = txt
        break
    if ack_text is None:
        return None
    return {"target_idx": target_idx, "result_idx": result_idx, "ack_text": ack_text}


def _slice_transcript(entries: list[dict], last_idx_inclusive: int) -> str:
    """Write entries[0:last_idx_inclusive+1] to a temp jsonl file. Returns path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in entries[: last_idx_inclusive + 1]:
        tmp.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


def _serialize_context(ctx) -> dict:
    """SessionContext → JSON-safe dict. Drops pairs (only journal uses them)
    and journal (not used in replay — most ack cases predate journal-mode)."""
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


def extract_case(cand: dict, idx: int) -> dict | None:
    rec = cand["verdict_rec"]
    payload = rec.get("payload") or {}
    entries = cand["entries"]
    target_idx = cand["target_idx"]
    result_idx = cand["result_idx"]

    slice_path = _slice_transcript(entries, result_idx)
    try:
        ctx = session_mod.load(
            slice_path,
            distill=False,  # no Haiku call during extraction
            current_tool_input=payload.get("tool_input"),
            cwd=payload.get("cwd"),
        )
    finally:
        try:
            os.unlink(slice_path)
        except OSError:
            pass

    verdict = rec.get("verdict") or {}
    case = {
        "case_id": f"case_{idx:03d}_{payload.get('tool_name','?').lower()}",
        "cwd": payload.get("cwd"),
        "session_id": payload.get("session_id"),
        "tool_name": payload.get("tool_name"),
        "tool_input": payload.get("tool_input") or {},
        "tool_response": payload.get("tool_response"),
        "expected": {
            "professional": False,
            "original_reason": verdict.get("reason", ""),
            "original_suggestion": verdict.get("suggestion", ""),
        },
        "ack_excerpt": cand["ack_excerpt"],
        "session_context": _serialize_context(ctx),
    }
    return case


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = find_flagged_acks()
    print(f"Found {len(cands)} ack candidates after transcript walk")

    cases = []
    for i, c in enumerate(cands):
        try:
            case = extract_case(c, i)
        except Exception as exc:
            print(f"  [{i}] FAILED to extract: {exc!r}")
            continue
        if not case:
            continue
        cases.append(case)
        print(f"  [{i}] {case['case_id']} ({case['cwd'].split('/')[-1]})  "
              f"snapshots={len(case['session_context']['per_file_snapshots'])} "
              f"actions={len(case['session_context']['recent_actions'])}")

    out_file = OUT_DIR / "cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(cases)} cases → {out_file}")


if __name__ == "__main__":
    main()
