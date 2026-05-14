"""Append-only audit log of every watchdog verdict.

One file per Claude Code session under ~/.claude/gadfly/log/<session-id>.jsonl.
Each line is a self-contained JSON record carrying everything we need to
re-run a verdict and reason about it:

  ts, tool_name, tool_input_digest, payload, user_message,
  system_prompt_sha, verdict, latency_ms, error

System prompts are large and rarely change, so they are content-addressed in
~/.claude/gadfly/system_prompts/<sha>.txt and referenced from each record by
their SHA. The viewer reads them lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .verdict import Verdict


def _root() -> Path:
    """Resolve the log root *at call time* so tests that set GADFLY_LOG_DIR
    after import still get a clean directory."""
    return Path(os.environ.get("GADFLY_LOG_DIR") or Path.home() / ".claude" / "gadfly" / "log")


# Back-compat shim for callers that read this constant.
LOG_ROOT = _root()


def _system_prompts_dir() -> Path:
    # Sit next to the log dir.
    return _root().parent / "system_prompts"


def ensure_system_prompt(text: str) -> str:
    """Content-address `text`. Write it once under system_prompts/<sha>.txt.
    Return the sha. Idempotent. Never raises (returns sha even if write fails)."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    try:
        d = _system_prompts_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{sha}.txt"
        if not path.exists():
            path.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return sha


def read_system_prompt(sha: str) -> str | None:
    try:
        path = _system_prompts_dir() / f"{sha}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


def _digest_tool_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Trim the tool_input down to something log-friendly.

    We don't want the audit log to balloon to gigabytes when the agent writes
    large files, so for Write/Edit we only keep paths and short previews.
    """
    if tool_name == "Edit":
        return {
            "file_path": tool_input.get("file_path"),
            "old_string_preview": str(tool_input.get("old_string", ""))[:200],
            "new_string_preview": str(tool_input.get("new_string", ""))[:200],
        }
    if tool_name == "Write":
        return {
            "file_path": tool_input.get("file_path"),
            "content_len": len(str(tool_input.get("content", ""))),
            "content_preview": str(tool_input.get("content", ""))[:200],
        }
    if tool_name == "MultiEdit":
        return {
            "file_path": tool_input.get("file_path"),
            "edit_count": len(tool_input.get("edits", [])),
        }
    if tool_name == "Bash":
        return {
            "command": str(tool_input.get("command", ""))[:500],
            "description": tool_input.get("description"),
        }
    return {"raw": str(tool_input)[:300]}


def append(
    *,
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    verdict: Verdict,
    latency_ms: float | None = None,
    error: str | None = None,
    payload: dict[str, Any] | None = None,
    user_message: str | None = None,
    system_prompt_sha: str | None = None,
) -> None:
    """Append one record. Never raises — logging failure must not break the hook."""
    try:
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{session_id or 'unknown'}.jsonl"
        record = {
            "ts": time.time(),
            "tool_name": tool_name,
            "tool_input_digest": _digest_tool_input(tool_name, tool_input),
            "payload": payload,
            "user_message": user_message,
            "system_prompt_sha": system_prompt_sha,
            "verdict": verdict.to_log_dict(),
            "latency_ms": latency_ms,
            "error": error,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging is best-effort. Swallow.
        pass
