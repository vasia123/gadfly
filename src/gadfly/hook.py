"""PostToolUse hook entrypoint.

Wiring contract (Claude Code → stdin):
  {
    "session_id": "...",
    "transcript_path": "/abs/path/to/<session-id>.jsonl",
    "cwd": "/abs/cwd",
    "hook_event_name": "PostToolUse",
    "tool_name": "Edit" | "Write" | "MultiEdit" | "Bash" | ...,
    "tool_input": {...},
    "tool_response": {...}
  }

We respond with either:
  - empty stdout + exit 0  (no comment to inject — silence is the default), or
  - JSON {"hookSpecificOutput": {"hookEventName": "PostToolUse",
          "additionalContext": "..."}} + exit 0 (warn the agent).

Hard rules:
  - We never exit non-zero. Even on internal errors. A broken watchdog must
    not break the user's Claude Code session.
  - We never raise out of main(). Every exception is swallowed (logged when
    possible) and replaced with a silent exit 0.
  - We skip cheaply when there's nothing to grade (wrong event, wrong tool,
    kill-switch set, empty stdin).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from . import log as audit_log
from . import session, watchdog
from .verdict import Verdict

WATCHED_TOOLS = {"Edit", "Write", "MultiEdit", "Bash"}


def _read_payload() -> dict[str, Any] | None:
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _emit_hook_output(output: dict[str, Any] | None) -> None:
    if output is None:
        return
    try:
        sys.stdout.write(json.dumps(output, ensure_ascii=False))
        sys.stdout.flush()
    except Exception:
        pass


def main() -> int:
    try:
        if os.environ.get("GADFLY_DISABLE") == "1":
            return 0
        # Recursion guard: when this hook is somehow triggered from inside
        # the watchdog's own inner Claude Code CLI (which sets this env), we
        # must exit immediately. The watchdog also disables setting inheritance
        # so this branch should be unreachable in practice, but treat it as
        # belt-and-braces.
        if os.environ.get("GADFLY_INTERNAL") == "1":
            return 0

        payload = _read_payload()
        if payload is None:
            return 0

        if payload.get("hook_event_name") != "PostToolUse":
            return 0

        tool_name = payload.get("tool_name")
        if tool_name not in WATCHED_TOOLS:
            return 0

        tool_input = payload.get("tool_input") or {}
        tool_response = payload.get("tool_response")
        session_id = str(payload.get("session_id") or "unknown")
        transcript_path = payload.get("transcript_path")

        if not isinstance(tool_input, dict):
            return 0

        ctx = session.load(transcript_path if isinstance(transcript_path, str) else None)

        t0 = time.perf_counter()
        result = watchdog.evaluate(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            context=ctx,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        audit_log.append(
            session_id=session_id,
            tool_name=tool_name,
            tool_input=tool_input,
            verdict=result.verdict,
            latency_ms=latency_ms,
            error=result.error,
            payload=payload,
            user_message=result.user_message,
            system_prompt_sha=result.system_prompt_sha,
        )

        _emit_hook_output(result.verdict.to_hook_output())
        return 0
    except Exception as exc:
        # Last-resort safety net. Try to log, but never propagate.
        try:
            audit_log.append(
                session_id="unknown",
                tool_name="?",
                tool_input={},
                verdict=Verdict.silent_ok(),
                error=f"hook crashed: {exc!r}",
            )
        except Exception:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
