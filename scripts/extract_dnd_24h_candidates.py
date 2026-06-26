"""Extract candidate user messages from last-24h dnd-llm transcripts.

For each user message whose IMMEDIATELY PRIOR assistant message carried
a code-changing tool_use (Edit/Write/MultiEdit/Bash), emit a record:

  {
    msg_id, file, uuid, ts_iso,
    user_text (cleaned, ≤2000c),
    prev_action_summary (1-line),
    prev_action_index (action_index inside the transcript),
  }

Output: tests/fixtures/wrong_level_corpus/candidates_24h.json

Dedupes against existing raw_user_messages.json so we don't re-classify
the messages already covered by the original week-long mining.

Skips:
  - subagent transcripts (`*/subagents/*.jsonl`)
  - tool_result-only user content (synthetic)
  - command artifacts / system-reminders (uses gadfly.pairs.clean_user_text)
  - user messages with no preceding code-changing action in the last
    25 records — there's nothing to attach a benchmark case to.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gadfly.pairs import clean_user_text, is_pure_tool_result  # noqa: E402

DNDDIR = Path(os.path.expanduser(
    "~/.claude/projects/-home-vasis-projects-hobby-dnd-llm"
))
OUT = ROOT / "tests" / "fixtures" / "wrong_level_corpus" / "candidates_24h.json"
EXISTING_USER_MSGS = (
    ROOT / "tests" / "fixtures" / "wrong_level_corpus" / "raw_user_messages.json"
)
TRIGGER_TOOLS = {"Edit", "Write", "MultiEdit", "Bash"}
WALK_BACK = 25
WINDOW_HOURS = 24


def _last_24h_main_transcripts() -> list[Path]:
    """All `.jsonl` files in the dnd-llm projects dir modified in the
    last WINDOW_HOURS, EXCLUDING the subagents/ subtree."""
    cutoff = time.time() - WINDOW_HOURS * 3600
    out: list[Path] = []
    for p in DNDDIR.glob("*.jsonl"):
        try:
            if p.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        # Main-agent transcripts live directly under DNDDIR; subagents
        # live under DNDDIR/<sid>/subagents/*.jsonl. The glob above
        # already restricts to depth 1, so we're filtering by location
        # for documentation purposes, but check anyway.
        if "subagents" in p.parts:
            continue
        out.append(p)
    return sorted(out, key=lambda x: x.stat().st_mtime)


def _summarize_tool_use(name: str, inp: dict) -> str:
    fp = inp.get("file_path", "?")
    if name == "Edit":
        old = (inp.get("old_string") or "")[:120]
        new = (inp.get("new_string") or "")[:120]
        return f"Edit({fp})\n  -: {old}\n  +: {new}"
    if name == "Write":
        body = (inp.get("content") or "")[:200]
        return f"Write({fp})\n  body: {body}"
    if name == "MultiEdit":
        edits = inp.get("edits") or []
        first_new = ""
        if edits and isinstance(edits[0], dict):
            first_new = (edits[0].get("new_string") or "")[:120]
        return f"MultiEdit({fp}, {len(edits)} edits)\n  first +: {first_new}"
    if name == "Bash":
        cmd = (inp.get("command") or "")[:200]
        return f"Bash: {cmd}"
    return name


def _find_prev_action(
    entries: list[dict], user_idx: int
) -> tuple[int, str] | None:
    """Walk back up to WALK_BACK records looking for the nearest assistant
    message with a code-changing tool_use. Return (assistant_idx,
    summary) or None.
    """
    lo = max(0, user_idx - WALK_BACK)
    for i in range(user_idx - 1, lo - 1, -1):
        e = entries[i]
        if e.get("type") != "assistant":
            continue
        if e.get("isSidechain"):
            continue
        msg = e.get("message") or {}
        for blk in reversed(msg.get("content") or []):
            if not isinstance(blk, dict):
                continue
            if blk.get("type") != "tool_use":
                continue
            name = blk.get("name")
            if name not in TRIGGER_TOOLS:
                continue
            inp = blk.get("input") or {}
            return i, _summarize_tool_use(name, inp)
    return None


def _action_index_at(entries: list[dict], assistant_idx: int) -> int:
    """Compute the monotonic action_index this assistant message
    represents — same counting rule session.py uses, so the trail
    rubric replay aligns with how the watchdog would have seen it."""
    n = 0
    for i in range(assistant_idx + 1):
        e = entries[i]
        if e.get("type") != "assistant" or e.get("isSidechain"):
            continue
        for blk in (e.get("message") or {}).get("content") or []:
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                n += 1
    return n


def _index_transcript(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    transcripts = _last_24h_main_transcripts()
    print(f"transcripts (last 24h, main-agent): {len(transcripts)}")
    total_size = sum(p.stat().st_size for p in transcripts) / 1024 / 1024
    print(f"total size: {total_size:.1f} MB")

    # Dedupe set — already-classified uuids from the original mining.
    seen_uuids: set[str] = set()
    if EXISTING_USER_MSGS.is_file():
        try:
            old = json.loads(EXISTING_USER_MSGS.read_text(encoding="utf-8"))
            for u in old:
                if isinstance(u, dict) and u.get("uuid"):
                    seen_uuids.add(u["uuid"])
        except Exception:
            pass
    print(f"existing uuids to skip: {len(seen_uuids)}")

    candidates: list[dict] = []
    counts: dict[str, int] = {
        "users_total": 0,
        "skip_existing": 0,
        "skip_tool_result_only": 0,
        "skip_no_text": 0,
        "skip_cleaned_empty": 0,
        "skip_no_prev_action": 0,
        "kept": 0,
    }

    for path in transcripts:
        entries = _index_transcript(path)
        for i, e in enumerate(entries):
            if e.get("type") != "user":
                continue
            if e.get("isSidechain"):
                continue
            uuid = e.get("uuid") or ""
            if not uuid:
                continue
            counts["users_total"] += 1
            if uuid in seen_uuids:
                counts["skip_existing"] += 1
                continue
            msg = e.get("message") or {}
            content = msg.get("content")
            if is_pure_tool_result(content):
                counts["skip_tool_result_only"] += 1
                continue
            # Pull text out of content (string or list of text blocks).
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        t = blk.get("text")
                        if isinstance(t, str):
                            text += (t + "\n")
            cleaned = clean_user_text(text) if text else None
            if not text:
                counts["skip_no_text"] += 1
                continue
            if not cleaned:
                counts["skip_cleaned_empty"] += 1
                continue
            prev = _find_prev_action(entries, i)
            if prev is None:
                counts["skip_no_prev_action"] += 1
                continue
            assistant_idx, action_summary = prev
            action_index = _action_index_at(entries, assistant_idx)
            ts = e.get("timestamp")
            candidates.append({
                "msg_id": len(candidates),
                "file": path.name,
                "uuid": uuid,
                "ts": ts,
                "user_text": cleaned[:2000],
                "prev_action_summary": action_summary[:600],
                "prev_action_index": action_index,
                "prev_assistant_idx": assistant_idx,
                "user_idx": i,
            })
            counts["kept"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {len(candidates)} candidates → {OUT}")
    print("Funnel:")
    for k, v in counts.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
