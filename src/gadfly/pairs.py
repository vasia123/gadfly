"""(assistant_text, user_text) pair extraction + service-tag scrubbing.

Lifted out of goal.py so multiple call sites (goal distillation, journal
maintenance, session context loading) share one implementation. This is
pure plumbing — no SDK calls, no I/O.

The transcript format Claude Code writes is undocumented and varies
between releases, so all parsing is defensive: anything malformed is
skipped silently, missing fields degrade to None / [].
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Pair:
    """One (preceding assistant text, user message) tuple from the transcript."""

    assistant_text: str | None
    user_text: str

    def to_cache_hash(self) -> str:
        h = hashlib.sha256()
        h.update((self.assistant_text or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(self.user_text.encode("utf-8"))
        return h.hexdigest()[:16]


_JUNK_PATTERNS = (
    "[Request interrupted by user for tool use]",
    "[Request interrupted by user]",
)

# Tags Claude Code injects into the user-role content stream but that are
# NOT real user intent. Stripped before deciding whether the message has
# any signal left. Single source of truth for both pair extraction and
# session.recent_user_requests collection.
_STRIPPABLE_TAGS = (
    "system-reminder",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-stderr",
    "local-command-caveat",
    "task-notification",
)

_TAG_BLOCK_RE = re.compile(
    r"<(" + "|".join(_STRIPPABLE_TAGS) + r")\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


def clean_user_text(text: str) -> str | None:
    """Strip Claude-Code-injected service tags from a user message.

    Returns the cleaned text, or None when nothing meaningful is left.
    """
    if not isinstance(text, str):
        return None
    cleaned = _TAG_BLOCK_RE.sub("", text).strip()
    if not cleaned:
        return None
    if any(p in cleaned for p in _JUNK_PATTERNS):
        without_markers = cleaned
        for p in _JUNK_PATTERNS:
            without_markers = without_markers.replace(p, "")
        without_markers = without_markers.strip()
        if not without_markers:
            return None
        cleaned = without_markers
    return cleaned


def extract_text_block(content: Any) -> str | None:
    """Pull plain text out of an Anthropic-style content field."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        chunks: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
        if chunks:
            return "\n".join(chunks)
    return None


def is_pure_tool_result(content: Any) -> bool:
    """True when a user-role content is just tool_result blocks (skip)."""
    return isinstance(content, list) and all(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def extract_pairs(entries: list[dict[str, Any]]) -> list[Pair]:
    """Walk transcript entries chronologically, build (prev_assistant, user) pairs.

    - Skips synthetic tool_result-only user messages.
    - Skips `[Request interrupted by user…]` markers and command artifacts.
    - De-duplicates exact-repeat user texts in a row.
    """
    pairs: list[Pair] = []
    last_assistant_text: str | None = None
    last_user_text: str | None = None

    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            text = extract_text_block(content)
            if text:
                last_assistant_text = text
            continue

        if role == "user":
            if is_pure_tool_result(content):
                continue
            text = extract_text_block(content)
            if text is None:
                continue
            text = clean_user_text(text)
            if text is None:
                continue
            if text == last_user_text:
                continue
            pairs.append(Pair(assistant_text=last_assistant_text, user_text=text))
            last_user_text = text
            # Once consumed, assistant context belongs to this pair only.
            last_assistant_text = None

    return pairs

# ── secret redaction ───────────────────────────────────────────────────────
# Corpus fixtures are built from REAL transcripts, and a transcript contains
# whatever the agent read: config files, `env` output, curl commands. That is
# how a live OpenRouter key and a Fireworks key ended up committed to this
# repo (2026-08-21). Anything derived from a transcript must pass through
# here before it is written to disk.
#
# Two nets. Known provider prefixes catch the common formats outright. The
# shape rule catches the rest: a key/token/secret assignment whose value is a
# long literal mixing letters and digits, with no spaces, dots or slashes —
# which is what excludes `password: postgres`, `token = request.token` and
# other ordinary code.
_SECRET_PREFIX_RE = re.compile(
    r"\b(fw_|gsk_|sk-ant-|sk-or-v1-|sk-proj-|AIza|hf_|ghp_|gho_|ghs_"
    r"|github_pat_|xox[bpsa]-|AKIA|glpat-)[A-Za-z0-9_\-]{16,}"
)
_SECRET_ASSIGN_RE = re.compile(
    r"((?:api_?key|apikey|token|secret|password|credential)[A-Za-z_]{0,20}"
    r"[\\\"']*\s*[:=]\s*\\?[\"'])([^\"'\\\s]{20,})",
    re.IGNORECASE,
)


def _looks_like_secret(value: str) -> bool:
    if "REDACTED" in value.upper():
        return False
    if any(ch in value for ch in "./\\("):        # paths and code, not secrets
        return False
    return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a live credential with a placeholder."""
    if not text:
        return text
    text = _SECRET_PREFIX_RE.sub(lambda m: m.group(1) + "REDACTED", text)
    return _SECRET_ASSIGN_RE.sub(
        lambda m: m.group(1) + ("REDACTED" if _looks_like_secret(m.group(2)) else m.group(2)),
        text,
    )
