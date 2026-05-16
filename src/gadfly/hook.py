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
from . import journal, session, watchdog
from .verdict import Verdict

WATCHED_TOOLS = {"Edit", "Write", "MultiEdit", "Bash"}


def _summarize_action_for_journal(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Compact one-liner summary for the journal maintainer.

    Larger / more structured than the verdict prompt's mini-diff — Haiku
    needs the gist (which file, which command) but the journal does not
    persist the full diff.
    """
    if tool_name == "Edit":
        return f"Edit({tool_input.get('file_path', '?')})"
    if tool_name == "Write":
        return f"Write({tool_input.get('file_path', '?')})"
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits") or []
        return f"MultiEdit({tool_input.get('file_path', '?')}, {len(edits)} edits)"
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))[:200]
        return f"Bash: {cmd}"
    return tool_name


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

        ctx = session.load(
            transcript_path if isinstance(transcript_path, str) else None,
            session_id=session_id,
        )

        # Phase-1 shadow: maintain a session journal alongside the
        # existing watchdog. The verdict prompt does NOT yet consume the
        # journal — we are gathering real-session journals first to
        # validate they look sane before switching the watchdog to read
        # from them. Opt out with GADFLY_JOURNAL=0.
        #
        # Phase-2 cutover (opt-in via GADFLY_JOURNAL_VERDICT=1): the
        # freshly-updated journal is loaded back into the session context
        # so watchdog.evaluate consumes it as primary verdict context.
        if os.environ.get("GADFLY_JOURNAL", "1") == "1":
            try:
                action_summary = _summarize_action_for_journal(tool_name, tool_input)
                result_j = journal.update_for_action(
                    session_id=session_id,
                    action_index=ctx.action_index,
                    action_summary=action_summary,
                    assistant_reasoning=ctx.last_assistant_plan,
                    pairs=ctx.pairs,
                )
                # Phase 2 default: the watchdog reads the journal as
                # primary context. Rollback to legacy by setting
                # GADFLY_JOURNAL_VERDICT=0.
                if os.environ.get("GADFLY_JOURNAL_VERDICT", "1") != "0":
                    ctx.journal = result_j.journal
            except Exception:
                # Journal must never break the hook. Swallowed silently;
                # journal.update_for_action already logs its own errors.
                pass

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
