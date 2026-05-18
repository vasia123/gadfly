"""Read the recent context out of a Claude Code session transcript.

Claude Code writes a JSONL transcript to a known path and hands that path to
hooks via the `transcript_path` field on the PostToolUse payload. We use it to
pull out the three things the watchdog needs:

  * the most recent user instruction — so Haiku knows what was actually asked,
  * the most recent assistant text — so Haiku sees what the agent claimed it
    was about to do,
  * the last few tool_use entries — so Haiku sees the trajectory of this turn.

The transcript schema is not formally documented and varies between Claude
Code versions, so this module parses defensively: anything malformed is
skipped, and missing context degrades to None / empty list rather than raising.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionContext:
    # Distilled current goal of the conversation, produced by goal.py
    # (Haiku one-shot, cached). When None — distillation wasn't available
    # (no API auth, error, etc.) — fall back to recent_user_requests.
    distilled_goal: str | None = None
    # Up to MAX_USER_REQUESTS most recent user messages, OLDEST first.
    # Used as fallback context when distilled_goal is unavailable, and as a
    # second signal so Haiku can spot if the distillation lost something.
    recent_user_requests: list[str] = field(default_factory=list)
    last_assistant_plan: str | None = None
    recent_actions: list[str] = field(default_factory=list)
    # Full chronological (assistant, user) pair list — consumed by the
    # journal updater. Pure transcript output; no Haiku call.
    pairs: list[Any] = field(default_factory=list)
    # Number of assistant tool_uses observed in the transcript so far.
    # Used by journal as action_index — monotonically increasing per
    # session.
    action_index: int = 0
    # The session journal as loaded from disk after the maintainer ran.
    # None in Phase 1 (journal not consumed by the verdict prompt) or
    # when journal-mode is disabled (GADFLY_JOURNAL=0). When set, the
    # watchdog prompt switches to journal-aware composition.
    journal: Any = None
    # Canonical cwd. Set by hook.main() from the PostToolUse payload.
    # Watchdog uses it to look up project_state.verdict_patterns for
    # the E2 self-calibration block.
    cwd: str = ""

    @property
    def last_user_request(self) -> str | None:
        """Back-compat / convenience: the latest user message, if any."""
        return self.recent_user_requests[-1] if self.recent_user_requests else None


MAX_USER_REQUESTS = 5


def _extract_text(content: Any) -> str | None:
    """Pull plain text out of an Anthropic-style content field.

    `content` may be a string or a list of blocks like
    [{"type": "text", "text": "..."}].
    """
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        if chunks:
            return "\n".join(chunks)
    return None


def _extract_tool_uses(content: Any) -> list[tuple[str, dict[str, Any]]]:
    """Return (tool_name, tool_input) for every tool_use block in `content`."""
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name")
                inp = block.get("input")
                if isinstance(name, str) and isinstance(inp, dict):
                    out.append((name, inp))
    return out


def _clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit}c]"


def _summarize_action(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Short, *content-aware* summary of a tool_use for the 'recent prior
    actions' list in the watchdog prompt.

    For code-changing tools we include a short preview of what was changed —
    otherwise Haiku, looking at the current edit in isolation, may flag it as
    a stub when in fact the symbol it references was introduced one edit
    earlier in the same series. Total budget is small (a few hundred chars
    per action × ~5 actions) so we keep previews tight.
    """
    fp = tool_input.get("file_path", "?")
    if tool_name == "Edit":
        old = _clip(tool_input.get("old_string", ""), 200)
        new = _clip(tool_input.get("new_string", ""), 200)
        return f"Edit({fp})\n  -: {old}\n  +: {new}"
    if tool_name == "Write":
        body = _clip(tool_input.get("content", ""), 300)
        return f"Write({fp})\n  body: {body}"
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits") or []
        first = ""
        if edits and isinstance(edits[0], dict):
            first = _clip(edits[0].get("new_string", ""), 150)
        return f"MultiEdit({fp}, {len(edits)} edits)\n  first +: {first}"
    if tool_name == "Bash":
        cmd = _clip(tool_input.get("command", ""), 200)
        return f"Bash: {cmd}"
    if tool_name == "Read":
        return f"Read({fp})"
    return tool_name


def load(
    transcript_path: str | None,
    *,
    max_actions: int = 5,
    session_id: str | None = None,
    distill: bool = True,
) -> SessionContext:
    """Read a Claude Code transcript and return distilled context.

    Always returns a SessionContext — never raises. When the file is missing,
    unreadable, or in an unexpected format, the corresponding fields stay at
    their defaults.

    `distill=True` (default) runs goal.load_or_distill on the (assistant,
    user) pairs to produce SessionContext.distilled_goal. Set distill=False
    in tests where you don't want the extra Haiku call.
    """
    ctx = SessionContext()
    if not transcript_path:
        return ctx
    path = Path(transcript_path)
    if not path.is_file():
        return ctx

    entries: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return ctx

    # Walk from the end backwards collecting up to MAX_USER_REQUESTS real
    # user messages (skipping synthetic tool_result wrappers). One message
    # rarely captures intent — the earlier ones set the goal, the later
    # ones clarify or correct it. We keep them in chronological order so
    # Haiku reads them naturally.
    collected: list[str] = []
    for entry in reversed(entries):
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        # Skip user-messages that are purely tool_result wrappers.
        if isinstance(content, list) and all(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            continue
        text = _extract_text(content)
        if not text:
            continue
        # Strip Claude-Code service tags (<system-reminder>, <command-name>,
        # <local-command-stdout>, …). Without this the watchdog's "recent
        # user messages" panel shows /compact stdout, /goal command echoes,
        # and Stop-hook system-reminders as if they were user intent — and
        # Haiku then graded the agent against THAT instead of the real goal.
        from . import goal as goal_mod
        text = goal_mod.clean_user_text(text)
        if text is None:
            continue
        collected.append(text)
        if len(collected) >= MAX_USER_REQUESTS:
            break
    ctx.recent_user_requests = list(reversed(collected))  # oldest first

    # Distil the goal (one Haiku call, cached). Only when there is at least
    # one real user message — distillation on empty conversation is silly.
    #
    # Phase 2: when the journal-aware verdict prompt is on (default), the
    # journal already owns root_goal — running the legacy distillation in
    # parallel would burn +3s per new user message for nothing. Skip it.
    # Rolling back to Phase 1 (GADFLY_JOURNAL_VERDICT=0) brings it back.
    phase2_on = os.environ.get("GADFLY_JOURNAL_VERDICT", "1") != "0"
    if distill and session_id and ctx.recent_user_requests and not phase2_on:
        try:
            from . import goal as goal_mod

            pairs = goal_mod.extract_pairs(entries)
            if pairs:
                state = goal_mod.load_or_distill(session_id=session_id, pairs=pairs)
                ctx.distilled_goal = state.goal
        except Exception:
            # Distillation must never break context loading.
            pass

    # Most recent assistant text block.
    for entry in reversed(entries):
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = _extract_text(msg.get("content"))
        if text:
            ctx.last_assistant_plan = text
            break

    # Last `max_actions` tool_use summaries, in chronological order.
    actions: list[str] = []
    total_tool_uses = 0
    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for name, inp in _extract_tool_uses(msg.get("content")):
            actions.append(_summarize_action(name, inp))
            total_tool_uses += 1
    ctx.recent_actions = actions[-max_actions:]
    ctx.action_index = total_tool_uses

    # Full pair list for the journal updater. Cheap (pure parsing,
    # already-loaded entries). Done last so a failure here can't poison
    # the rest of the context.
    try:
        from . import pairs as pairs_mod

        ctx.pairs = pairs_mod.extract_pairs(entries)
    except Exception:
        ctx.pairs = []

    return ctx
