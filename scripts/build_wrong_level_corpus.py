"""Build the wrong-level corpus from dnd-llm user pushbacks.

Input: tests/fixtures/wrong_level_corpus/raw_pushback_pairs.json — the
56 (msg_uuid, user_quote, category) tuples a classifier agent extracted
from a week of dnd-llm sessions.

For each pushback:
  1. Find the user message by uuid in the source transcript.
  2. Walk backwards (within ~20 prior records) to find the nearest
     assistant tool_use of {Edit, Write, MultiEdit, Bash} — that is the
     "trigger code" the critique attached to. If the trigger action
     was text-only, this lifts the case to a concrete tool action
     (the user said: "привязываюсь к тексту, но можно найти
     конкретные кусочки кода в предыдущих сообщениях").
  3. Slice the transcript at the tool_result for that tool_use.
  4. session.load() the slice — same SessionContext build the watchdog
     would have seen at hook firing time.
  5. Emit a case in the same schema as watchdog_corpus/cases.json:
     {case_id, cwd, session_id, tool_name, tool_input, tool_response,
      expected={professional: false, original_reason: user_quote,
                category, confidence},
      ack_excerpt (= user_quote),
      session_context}.

The corpus is replayable forever, decoupled from transcript schema
drift. evaluate() replays each case under any chosen model/prompt.

Run: .venv/bin/python scripts/build_wrong_level_corpus.py
Output: tests/fixtures/wrong_level_corpus/cases.json
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

FIX_DIR = ROOT / "tests" / "fixtures" / "wrong_level_corpus"
RAW_FILE = FIX_DIR / "raw_pushback_pairs.json"
USERS_FILE = FIX_DIR / "raw_user_messages.json"  # msg_id → uuid lookup
OUT_FILE = FIX_DIR / "cases.json"
DND_DIR = Path(os.path.expanduser(
    "~/.claude/projects/-home-vasis-projects-hobby-dnd-llm"
))
TRIGGER_TOOLS = {"Edit", "Write", "MultiEdit", "Bash"}
WALK_BACK_LIMIT = 25  # look up to ~25 records back for the trigger tool_use


def _index_transcript(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _find_uuid(entries: list[dict], uuid: str) -> int | None:
    for i, e in enumerate(entries):
        if e.get("uuid") == uuid:
            return i
    return None


def _find_trigger_tool_use(
    entries: list[dict], user_idx: int, limit: int = WALK_BACK_LIMIT
) -> tuple[int, dict] | None:
    """Walk backwards from user_idx to find the nearest assistant message
    with a tool_use in TRIGGER_TOOLS. Returns (assistant_idx, block) or None.
    """
    lo = max(0, user_idx - limit)
    for i in range(user_idx - 1, lo - 1, -1):
        e = entries[i]
        if e.get("type") != "assistant":
            continue
        msg = e.get("message") or {}
        blocks = msg.get("content") or []
        # Prefer the LATEST tool_use in this assistant message (assistants
        # often emit text + tool_use; we want the action).
        for blk in reversed(blocks):
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "tool_use" and blk.get("name") in TRIGGER_TOOLS:
                return (i, blk)
    return None


def _find_tool_result(
    entries: list[dict], tool_idx: int, tool_use_id: str
) -> tuple[int, dict] | None:
    """Find the user message holding tool_result for the given tool_use_id."""
    for j in range(tool_idx + 1, min(tool_idx + 6, len(entries))):
        e = entries[j]
        if e.get("type") != "user":
            continue
        for blk in (e.get("message") or {}).get("content") or []:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_result":
                continue
            if blk.get("tool_use_id") == tool_use_id:
                return (j, blk)
        # If we hit a user message that's NOT tool_result-only, abort —
        # the tool_use never produced a result the watchdog could see.
    return None


def _slice(entries: list[dict], last_idx: int) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for e in entries[: last_idx + 1]:
        tmp.write(json.dumps(e, ensure_ascii=False) + "\n")
    tmp.close()
    return tmp.name


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


def _tool_result_payload(tool_result_block: dict) -> dict:
    """Reconstruct a tool_response-ish object from the tool_result block.
    Original PostToolUse hooks see a richer shape (stdout/stderr split etc.);
    the corpus only needs the textual content the model would have read."""
    content = tool_result_block.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                text += (blk.get("text") or "")
    return {
        "is_error": bool(tool_result_block.get("is_error", False)),
        "content": text[:8000],
    }


def main():
    if not RAW_FILE.exists():
        print(f"ERROR: {RAW_FILE} missing — run the pushback extractor first",
              file=sys.stderr)
        sys.exit(2)

    pushbacks = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    users = json.loads(USERS_FILE.read_text(encoding="utf-8"))
    # msg_id is the index into users[]; user has uuid + file.
    msg_by_id = {u["msg_id"]: u for u in users}
    print(f"Loaded {len(pushbacks)} raw pushbacks, {len(users)} user msgs")

    # Index transcripts by filename (we'll cache parsed entries lazily).
    cache: dict[str, list[dict]] = {}
    cases = []
    skipped: dict[str, int] = {
        "transcript_missing": 0,
        "uuid_not_found": 0,
        "no_trigger_tool_use": 0,
        "no_tool_result": 0,
        "session_load_failed": 0,
    }

    for idx, pb in enumerate(pushbacks):
        u = msg_by_id.get(pb["msg_id"])
        if not u:
            skipped["uuid_not_found"] += 1
            continue
        fname = u["file"]
        path = DND_DIR / fname
        if not path.exists():
            skipped["transcript_missing"] += 1
            continue
        if fname not in cache:
            cache[fname] = _index_transcript(path)
        entries = cache[fname]

        user_idx = _find_uuid(entries, u["uuid"])
        if user_idx is None:
            skipped["uuid_not_found"] += 1
            continue

        trig = _find_trigger_tool_use(entries, user_idx)
        if trig is None:
            skipped["no_trigger_tool_use"] += 1
            continue
        tool_idx, tu_blk = trig

        tr = _find_tool_result(entries, tool_idx, tu_blk.get("id", ""))
        if tr is None:
            skipped["no_tool_result"] += 1
            continue
        result_idx, tr_blk = tr

        slice_path = _slice(entries, result_idx)
        try:
            ctx = session_mod.load(
                slice_path,
                distill=False,
                current_tool_input=tu_blk.get("input") or {},
                cwd="/home/vasis/projects_hobby/dnd-llm",
            )
        except Exception as exc:
            skipped["session_load_failed"] += 1
            print(f"  [{idx}] session.load failed: {exc!r}")
            continue
        finally:
            try:
                os.unlink(slice_path)
            except OSError:
                pass

        tool_name = tu_blk.get("name", "?")
        case = {
            "case_id": f"wl_case_{idx:03d}_{tool_name.lower()}",
            "cwd": "/home/vasis/projects_hobby/dnd-llm",
            "session_id": fname.replace(".jsonl", ""),
            "tool_name": tool_name,
            "tool_input": tu_blk.get("input") or {},
            "tool_response": _tool_result_payload(tr_blk),
            "expected": {
                "professional": False,
                "original_reason": pb["user_quote"],
                "category": pb["category"],
                "confidence": pb["confidence"],
                "why_this_action": pb["why_this_action"],
            },
            "mode": "wrong_level_positive",
            "ack_excerpt": pb["user_quote"],
            "session_context": _serialize_context(ctx),
        }
        cases.append(case)
        cwd_tail = case["cwd"].split("/")[-1]
        print(f"  [{idx:02d}] {case['case_id']:30s}  trig={tool_name:6s}  "
              f"actions={len(case['session_context']['recent_actions'])}  "
              f"{cwd_tail}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {len(cases)} cases → {OUT_FILE}")
    print(f"Skipped breakdown: {skipped}")


if __name__ == "__main__":
    main()
