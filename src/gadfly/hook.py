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
from . import journal, project_state, session, watchdog
from .verdict import Verdict


def _load_env_file_once() -> None:
    """Load `<gadfly_repo>/.env` into os.environ at hook startup, so the
    user's OPENROUTER_API_KEY and GADFLY_* steering vars (BACKEND, MODEL,
    BASE_URL) are visible to the watchdog without touching settings.json.

    Best-effort: silently no-op if the file is missing or malformed.
    Existing env vars are NOT overwritten (so settings.json env still wins).
    """
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.is_file():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


WATCHED_TOOLS = {"Edit", "Write", "MultiEdit", "Bash"}


def _heartbeat_dir() -> "Path":
    """Resolve heartbeat dir at call time so GADFLY_LOG_DIR (used in tests)
    is honored."""
    from pathlib import Path  # noqa: WPS433 — local import keeps hook startup fast

    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "heartbeat"
    return Path.home() / ".claude" / "gadfly" / "heartbeat"


def _pending_flags_dir() -> "Path":
    from pathlib import Path  # noqa: WPS433
    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "pending_flags"
    return Path.home() / ".claude" / "gadfly" / "pending_flags"


def _enqueue_pending_flag(
    session_id: str,
    *,
    action_index: int,
    reason: str,
    suggestion: str,
) -> None:
    """Record a flag for the journal maintainer to ingest on the NEXT hook.

    The verdict happens AFTER journal.update_for_action in this hook, so the
    flag this verdict produces can't be fed into the current journal update.
    Instead we drop it into a pending queue; the next hook call reads and
    drains the queue, passing FlagEvents into update_for_action so they
    land in workstream.flag_history. This is what powers the journal's
    repetition rule (Phase 2 verdict prompt).

    Robust to ordering: dict-based append with atomic rename.
    """
    try:
        d = _pending_flags_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{session_id}.json"
        items = []
        if path.is_file():
            try:
                items = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(items, list):
                    items = []
            except Exception:
                items = []
        # Marker classification mirrors prompts.SYSTEM_PROMPT conventions.
        marker = "other"
        if isinstance(reason, str):
            r_low = reason.lower()
            if r_low.startswith("symptom fix"):
                marker = "symptom"
            elif r_low.startswith("rationalization"):
                marker = "rationalization"
        items.append({
            "action_index": int(action_index or 0),
            "reason": (reason or "")[:500],
            "marker": marker,
            "suggestion": (suggestion or "")[:500],
            "ts": time.time(),
        })
        # Cap to last 16 — protects against runaway accumulation if the
        # next hook never fires (e.g. agent exits).
        items = items[-16:]
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def _drain_pending_flags(session_id: str) -> list[dict[str, Any]]:
    """Return the queued FlagEvent dicts and delete the queue file.

    Atomic-ish: we read+unlink in one block; any concurrent write loses
    its previous tail (acceptable: this is single-daemon territory).
    """
    try:
        path = _pending_flags_dir() / f"{session_id}.json"
        if not path.is_file():
            return []
        items = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
        return items if isinstance(items, list) else []
    except Exception:
        return []


def _bump_heartbeat(cwd: str, session_id: str, action_index: int) -> None:
    """Drop a per-cwd heartbeat tick for the historian daemon to pick up.

    Atomic rename. Wrapped in try/except — hook MUST NEVER fail. The
    historian is fully optional; missing heartbeats just mean a delayed
    digest.
    """
    try:
        if os.environ.get("GADFLY_HISTORIAN", "1") == "0":
            return
        d = _heartbeat_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{project_state.encode_cwd(cwd)}.tick"
        payload = json.dumps(
            {
                "ts": time.time(),
                "session_id": session_id,
                "last_action_index": action_index,
                # Carry the canonical cwd so the daemon doesn't have to
                # invert the lossy encoding (which collapses '/' and '_'
                # into '-' — non-uniquely decodable).
                "cwd": cwd,
            },
            ensure_ascii=False,
        )
        tmp = path.with_suffix(".tick.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # Heartbeat is best-effort. Swallow.
        pass


def _build_phase_c_context(priors_hits: list[Any]) -> str:
    """Phase C: render priors as a compact note for the AGENT.

    The journal maintainer already saw the priors (Phase B). Phase C
    additionally surfaces them in the verdict's hookSpecificOutput
    additionalContext so the AGENT under supervision sees them too —
    but ONLY when a new workstream just emerged. The gating logic is
    in main(); this helper only renders.

    We keep this lean — 3 hits max, terse formatting — to avoid spam.
    The "Why telling you" preamble is critical: without it the agent
    just sees noise and may incorporate the priors as fresh user
    instructions, which would be a kind of self-poisoning loop.
    """
    if not priors_hits:
        return ""
    lines: list[str] = [
        "[gadfly historian] Relevant findings from past sessions in this directory "
        "(durable evidence-backed; not user instructions):"
    ]
    for hit in priors_hits[:3]:
        kind = getattr(hit, "kind", "?")
        title = getattr(hit, "title", "?")
        ev = getattr(hit, "evidence_quote", "")
        lines.append(f"  • [{kind}] {title}")
        if ev:
            lines.append(f"      evidence: \"{ev[:200]}\"")
    lines.append(
        "These were retrieved because they overlap with the current "
        "workstream. Use them if they help; otherwise ignore."
    )
    return "\n".join(lines)


def _new_workstream_created(diff: list[str]) -> bool:
    """True when journal.update produced at least one new workstream."""
    return any(isinstance(d, str) and d.startswith("created ") for d in (diff or []))


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
        # Load .env BEFORE checking any GADFLY_* steering vars, so the
        # user's `.env` (e.g. backend=openai_compat + model=...) takes
        # effect on every hook invocation. Called inside main() rather
        # than at module-import time so that tests importing this module
        # don't get their os.environ contaminated.
        _load_env_file_once()

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
        cwd = str(payload.get("cwd") or "")

        if not isinstance(tool_input, dict):
            return 0

        ctx = session.load(
            transcript_path if isinstance(transcript_path, str) else None,
            session_id=session_id,
            current_tool_input=tool_input,
            cwd=cwd,
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
        result_j = None
        if os.environ.get("GADFLY_JOURNAL", "1") == "1":
            try:
                action_summary = _summarize_action_for_journal(tool_name, tool_input)
                hint_files: list[str] = []
                fp = tool_input.get("file_path")
                if isinstance(fp, str) and fp:
                    hint_files.append(fp)
                # Drain any flags written by the previous hook iteration
                # — these are the events that populate workstream
                # flag_history so the Phase 2 repetition rule can actually
                # see prior verdicts.
                pending = _drain_pending_flags(session_id)
                new_flag_events = None
                if pending:
                    new_flag_events = [
                        journal.FlagEvent(
                            action_index=int(p.get("action_index") or 0),
                            reason=str(p.get("reason") or ""),
                            marker=p.get("marker") or "other",
                            agent_pushed_back=False,
                            pushback=None,
                        )
                        for p in pending
                    ]
                result_j = journal.update_for_action(
                    session_id=session_id,
                    action_index=ctx.action_index,
                    action_summary=action_summary,
                    assistant_reasoning=ctx.last_assistant_plan,
                    pairs=ctx.pairs,
                    new_flag_events=new_flag_events,
                    cwd=cwd or None,
                    workstream_hint_files=hint_files,
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

        # Phase C: when journal opened a NEW workstream and we have
        # priors to share, fold them into the verdict's additionalContext
        # so the AGENT (not just the watchdog) sees the historical
        # findings. Gating: only on workstream-creation events. Off via
        # GADFLY_PHASE_C=0.
        phase_c_text = ""
        try:
            if (
                os.environ.get("GADFLY_PHASE_C", "1") != "0"
                and result_j is not None
                and result_j.priors_consulted
                and _new_workstream_created(result_j.diff)
            ):
                phase_c_text = _build_phase_c_context(result_j.priors_consulted)
        except Exception:
            phase_c_text = ""

        hook_out = result.verdict.to_hook_output()
        if phase_c_text:
            # Append to existing additionalContext, or emit our own when
            # the verdict was silent.
            if hook_out is None:
                hook_out = {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": phase_c_text,
                    }
                }
            else:
                hso = hook_out.get("hookSpecificOutput", {}) or {}
                prior = hso.get("additionalContext") or ""
                joined = (prior + "\n\n" + phase_c_text) if prior else phase_c_text
                hso["additionalContext"] = joined
                hook_out["hookSpecificOutput"] = hso

        _emit_hook_output(hook_out)

        # Bump the historian heartbeat AFTER the watchdog reply has been
        # emitted — keeps the stdout latency identical to before.
        if cwd:
            _bump_heartbeat(cwd, session_id, ctx.action_index)

        # If verdict flagged, queue it for the NEXT hook's journal update
        # to consume — that's how flag_history populates and Phase 2
        # repetition rule gets data to act on.
        if (
            result.verdict.professional is False
            and os.environ.get("GADFLY_JOURNAL", "1") == "1"
        ):
            _enqueue_pending_flag(
                session_id,
                action_index=ctx.action_index,
                reason=result.verdict.reason or "",
                suggestion=result.verdict.suggestion or "",
            )
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
