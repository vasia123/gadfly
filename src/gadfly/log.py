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


def _journals_dir() -> Path:
    """Content-addressed journal snapshot store. Same pattern as system_prompts."""
    return _root().parent / "journals"


def _trails_dir() -> Path:
    """Content-addressed trail snapshot store. Same pattern as journals."""
    return _root().parent / "trails"


def ensure_trail_snapshot(trail_json: str) -> str:
    """Content-address a trail JSON string. Returns sha. Idempotent.

    Audit log entries reference this sha so the trail's full breadcrumb
    list never inlines into log lines — keeps `<session>.jsonl` skimmable
    even after long sessions.
    """
    sha = hashlib.sha256(trail_json.encode("utf-8")).hexdigest()[:16]
    try:
        d = _trails_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{sha}.json"
        if not path.exists():
            path.write_text(trail_json, encoding="utf-8")
    except Exception:
        pass
    return sha


def read_trail_snapshot(sha: str) -> str | None:
    try:
        path = _trails_dir() / f"{sha}.json"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


def ensure_journal_snapshot(journal_json: str) -> str:
    """Content-address a journal JSON string. Returns sha. Idempotent.

    Audit log entries reference this sha instead of inlining the full
    journal — keeps log lines tiny even after many updates.
    """
    sha = hashlib.sha256(journal_json.encode("utf-8")).hexdigest()[:16]
    try:
        d = _journals_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{sha}.json"
        if not path.exists():
            path.write_text(journal_json, encoding="utf-8")
    except Exception:
        pass
    return sha


def read_journal_snapshot(sha: str) -> str | None:
    try:
        path = _journals_dir() / f"{sha}.json"
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return None


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
    """Append one verdict record. Never raises — logging failure must not break the hook."""
    try:
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{session_id or 'unknown'}.jsonl"
        record = {
            "type": "verdict",
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


def append_historian_event(
    *,
    session_id: str,
    cwd: str,
    chunks: int,
    findings_counts: dict[str, int],
    latency_ms: float | None,
    error: str | None,
) -> None:
    """Append one historian session-digest event. Recorded under the
    session_id (same file as the verdict log) so the viewer can show
    'this session was digested at T, producing N findings'."""
    try:
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{session_id or 'unknown'}.jsonl"
        record = {
            "type": "historian_digest",
            "ts": time.time(),
            "cwd": cwd,
            "chunks": chunks,
            "findings_counts": dict(findings_counts),
            "latency_ms": latency_ms,
            "error": error,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_journal_event(
    *,
    session_id: str,
    action_index: int,
    prior_journal_sha: str | None,
    new_journal_sha: str | None,
    diff_summary: list[str],
    latency_ms: float | None,
    error: str | None,
    skipped_reason: str | None = None,
) -> None:
    """Append one journal-update event so the viewer can show how the
    agent's worklist evolved. Same file as verdict / goal records;
    differentiated by `type`."""
    try:
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{session_id or 'unknown'}.jsonl"
        record = {
            "type": "journal_update",
            "ts": time.time(),
            "action_index": action_index,
            "prior_journal_sha": prior_journal_sha,
            "new_journal_sha": new_journal_sha,
            "diff_summary": diff_summary,
            "latency_ms": latency_ms,
            "error": error,
            "skipped_reason": skipped_reason,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_trail_event(
    *,
    session_id: str,
    action_index: int,
    prior_trail_sha: str | None,
    new_trail_sha: str | None,
    advances_trail: bool,
    drift_detected: bool,
    drift_kind: str | None,
    suppressed: bool,
    delivered_to_agent: bool,
    diff_summary: list[str],
    latency_ms: float | None,
    error: str | None,
    skipped_reason: str | None = None,
) -> None:
    """Append one trail-update event. Same file as verdict / journal / goal
    records; differentiated by `type=trail_update`. `prior_trail_sha` /
    `new_trail_sha` reference content-addressed snapshots under
    ~/.claude/gadfly/trails/<sha>.json so the line stays tiny even when
    the trail itself has 30 breadcrumbs.
    """
    try:
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{session_id or 'unknown'}.jsonl"
        record = {
            "type": "trail_update",
            "ts": time.time(),
            "action_index": action_index,
            "prior_trail_sha": prior_trail_sha,
            "new_trail_sha": new_trail_sha,
            "advances_trail": advances_trail,
            "drift_detected": drift_detected,
            "drift_kind": drift_kind,
            "suppressed": suppressed,
            "delivered_to_agent": delivered_to_agent,
            "diff_summary": diff_summary,
            "latency_ms": latency_ms,
            "error": error,
            "skipped_reason": skipped_reason,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_goal_event(
    *,
    session_id: str,
    pairs_total: int,
    pairs_new: int,
    pairs_cached: int,
    prior_goal: str | None,
    goal: str | None,
    latency_ms: float | None,
    error: str | None,
    cache_hit: bool,
) -> None:
    """Append one goal-distillation event so the viewer can show when/why
    the session goal changed. Same file as verdict records; differentiated
    by `type`."""
    try:
        root = _root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{session_id or 'unknown'}.jsonl"
        record = {
            "type": "goal_distill",
            "ts": time.time(),
            "pairs_total": pairs_total,
            "pairs_new": pairs_new,
            "pairs_cached": pairs_cached,
            "prior_goal": prior_goal,
            "goal": goal,
            "latency_ms": latency_ms,
            "error": error,
            "cache_hit": cache_hit,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
