"""Extract candidate (agent_text, user_response) pairs from last-24h dnd-llm.

Hypothesis under test: agent's WORDS (the text blocks it emits) predict
user pushback better than the trail of its tool actions. To test, we
need the agent's SUBSTANTIVE text — the message it wrote that the user
then reacted to — paired with that user reaction.

For each non-junk user message in a main-agent transcript modified in
the last 24h, we walk back to find the immediately preceding assistant
text block (skipping tool_use blocks and sidechain messages). Both must
be non-empty after junk-stripping.

Output: tests/fixtures/wrong_level_corpus/candidates_24h_words.json
  [
    {
      msg_id, file, uuid,
      agent_text,       # cleaned, ≤4000c — what we evaluate for laziness
      agent_text_len,
      user_text,        # cleaned, ≤2000c — ground truth of user reaction
      preceding_action_summary,  # optional 1-liner of any tool_use the
                                  # agent did BEFORE the text block
      user_idx, agent_msg_idx,
    },
    ...
  ]
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
OUT = (
    ROOT / "tests" / "fixtures" / "wrong_level_corpus"
    / "candidates_24h_words.json"
)
TRIGGER_TOOLS = {"Edit", "Write", "MultiEdit", "Bash", "Read"}
WALK_BACK = 30
WINDOW_HOURS = 24
MIN_AGENT_TEXT_LEN = 80
MAX_AGENT_TEXT_LEN = 4000
MAX_USER_TEXT_LEN = 2000


def _last_24h_main_transcripts() -> list[Path]:
    cutoff = time.time() - WINDOW_HOURS * 3600
    out: list[Path] = []
    for p in DNDDIR.glob("*.jsonl"):
        try:
            if p.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        if "subagents" in p.parts:
            continue
        out.append(p)
    return sorted(out, key=lambda x: x.stat().st_mtime)


def _index_transcript(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _extract_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                t = blk.get("text")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        return "\n".join(chunks)
    return ""


def _extract_tool_use_summary(content) -> str:
    """If the assistant message contains a tool_use, return a short
    one-line summary of it. Used purely as context for the case —
    what action (if any) the agent took alongside the text block.
    """
    if not isinstance(content, list):
        return ""
    for blk in content:
        if not isinstance(blk, dict) or blk.get("type") != "tool_use":
            continue
        name = blk.get("name", "?")
        inp = blk.get("input") or {}
        if name == "Edit":
            return f"Edit({inp.get('file_path', '?')})"
        if name == "Write":
            return f"Write({inp.get('file_path', '?')})"
        if name == "MultiEdit":
            edits = inp.get("edits") or []
            return f"MultiEdit({inp.get('file_path', '?')}, {len(edits)} edits)"
        if name == "Bash":
            return f"Bash: {(inp.get('command') or '')[:120]}"
        if name == "Read":
            return f"Read({inp.get('file_path', '?')})"
        return name
    return ""


def _find_preceding_agent_text(
    entries: list[dict], user_idx: int
) -> tuple[int, str, str] | None:
    """Walk back from user_idx to find the nearest assistant message
    with a non-empty TEXT block (independent of tool_use). Returns
    (agent_idx, text, preceding_action_summary) or None.

    The 'preceding_action_summary' is taken from the SAME assistant
    message — many assistant turns combine reasoning text with a
    tool_use. We capture both so the case knows what action (if any)
    the agent's text was paired with.
    """
    lo = max(0, user_idx - WALK_BACK)
    for i in range(user_idx - 1, lo - 1, -1):
        e = entries[i]
        if e.get("type") != "assistant":
            continue
        if e.get("isSidechain"):
            continue
        msg = e.get("message") or {}
        content = msg.get("content")
        text = _extract_text(content)
        if not text or len(text) < MIN_AGENT_TEXT_LEN:
            continue
        tool_summary = _extract_tool_use_summary(content)
        return i, text, tool_summary
    return None


def main() -> None:
    transcripts = _last_24h_main_transcripts()
    print(f"transcripts: {len(transcripts)}")

    candidates: list[dict] = []
    counts: dict[str, int] = {
        "users_total": 0,
        "skip_sidechain": 0,
        "skip_tool_result_only": 0,
        "skip_cleaned_empty": 0,
        "skip_no_preceding_text": 0,
        "skip_text_too_short": 0,
        "kept": 0,
    }

    for path in transcripts:
        entries = _index_transcript(path)
        for i, e in enumerate(entries):
            if e.get("type") != "user":
                continue
            if e.get("isSidechain"):
                counts["skip_sidechain"] += 1
                continue
            uuid = e.get("uuid") or ""
            if not uuid:
                continue
            counts["users_total"] += 1

            msg = e.get("message") or {}
            content = msg.get("content")
            if is_pure_tool_result(content):
                counts["skip_tool_result_only"] += 1
                continue
            raw_user = ""
            if isinstance(content, str):
                raw_user = content
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        t = blk.get("text")
                        if isinstance(t, str):
                            raw_user += (t + "\n")
            cleaned = clean_user_text(raw_user) if raw_user else None
            if not cleaned:
                counts["skip_cleaned_empty"] += 1
                continue

            prev = _find_preceding_agent_text(entries, i)
            if prev is None:
                counts["skip_no_preceding_text"] += 1
                continue
            agent_idx, agent_text, tool_summary = prev
            if len(agent_text) < MIN_AGENT_TEXT_LEN:
                counts["skip_text_too_short"] += 1
                continue
            if len(agent_text) > MAX_AGENT_TEXT_LEN:
                agent_text = agent_text[:MAX_AGENT_TEXT_LEN] + "…[+truncated]"

            ts = e.get("timestamp")
            candidates.append({
                "msg_id": len(candidates),
                "file": path.name,
                "uuid": uuid,
                "ts": ts,
                "agent_text": agent_text,
                "agent_text_len": len(agent_text),
                "user_text": cleaned[:MAX_USER_TEXT_LEN],
                "preceding_action_summary": tool_summary[:300],
                "user_idx": i,
                "agent_msg_idx": agent_idx,
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
    if candidates:
        avg = sum(c["agent_text_len"] for c in candidates) / len(candidates)
        print(f"\nAvg agent_text length: {int(avg)} chars")
        actions = sum(1 for c in candidates if c["preceding_action_summary"])
        print(f"With paired action: {actions} / {len(candidates)}")


if __name__ == "__main__":
    main()
