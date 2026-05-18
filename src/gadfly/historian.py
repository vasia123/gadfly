"""Project historian — mines all sessions in a cwd for durable findings.

This is the "cold path" of gadfly. The hook drops heartbeats; this
module reads them, identifies sessions that have gone quiet, and
distils each one into:

  - a raw episodic digest (`raw/<sid>.json`) — verbatim, source of truth,
  - new semantic findings that flow through the quarantine→promotion
    lifecycle into the project corpus.

Three design principles, drawn directly from the 2026 agent-memory
literature:

  1. **Ground-truth preservation** (MemMachine, SLEEP): the distill call
     extracts findings ONCE from the raw transcript. The semantic
     aggregator in `project_state.aggregate()` operates over those
     extractions deterministically. It NEVER re-reads its own prior
     semantic output, so it can never "drift".
  2. **ADD-only extraction** (mem0): the MCP tool returns four flat
     lists of NEW findings. No UPDATE/DELETE. Supersession is handled
     in the aggregator and via revocation.
  3. **Quarantine + provenance** (anti-MemoryGraft / MINJA): every
     finding carries the source_session + action_index + evidence_quote.
     Corrections need ≥2 sessions before going active.

The distill call uses the same SDK + recursion-guard setup as
`watchdog.py` and `journal.py`. The hook MUST NEVER fail because of
us: all entry points swallow exceptions and log via the audit log.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ThinkingConfigDisabled,
    create_sdk_mcp_server,
    query,
    tool,
)

from . import log as audit_log
from . import pairs as pairs_mod
from . import project_state as ps


# --- Tunables ----------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5"
# Distillation timeout. Set higher than the watchdog/journal default
# (60s) because per-session digest can chunk into multiple Haiku calls
# for large transcripts, and we don't want a long session to get
# orphaned on every backfill attempt. Real-world observation: 5%+ of
# >100KB sessions exceeded 60s, ~1% exceeded 120s during the
# gadfly+dnd-llm sweep. 180s covers the long tail.
DEFAULT_TIMEOUT_S = 180.0
DONE_AFTER_S = 5 * 60        # heartbeat ≥5min old → session "done"
DAEMON_POLL_S = 60           # how often --watch wakes up

# Per-chunk size for oversized transcripts. We measure by "events"
# rather than tokens for cheapness.
MAX_EVENTS_PER_CHUNK = 80

# Per-poll cap. Without this, a fresh install on a project with
# hundreds of historical sessions would burn the user's Haiku quota in
# one cycle. The daemon spreads the work across polls — at default 5
# sessions / 60-second poll, 100 backlog sessions take ~20 minutes,
# which is gentle on both quota and the user's machine. Override with
# --max-per-poll on the CLI. Use `backfill --yes-i-know-the-cost` if
# you want unrestricted throughput.
DEFAULT_MAX_PER_POLL = 5


# --- Compaction --------------------------------------------------------------


def _truncate(text: str, limit: int = 600) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text) - limit}c]"


def _read_transcript(transcript_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with transcript_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries


def _summarize_tool_use(name: str, inp: dict[str, Any]) -> str:
    """Short, content-aware one-liner for a tool_use. We do NOT keep
    full diffs in the digest — the journal does that. Here we just want
    enough to recognise the action class."""
    if not isinstance(inp, dict):
        return name
    if name == "Edit":
        return f"Edit({inp.get('file_path', '?')})"
    if name == "Write":
        body_len = len(str(inp.get("content", "")))
        return f"Write({inp.get('file_path', '?')}, {body_len}c)"
    if name == "MultiEdit":
        edits = inp.get("edits") or []
        return f"MultiEdit({inp.get('file_path', '?')}, {len(edits)} edits)"
    if name == "Bash":
        return f"Bash: {_truncate(str(inp.get('command', '')), 200)}"
    if name == "Read":
        return f"Read({inp.get('file_path', '?')})"
    if name == "Grep":
        return f"Grep({inp.get('pattern', '?')})"
    return name


@dataclass
class CompactEvent:
    """One event in a compacted transcript. Lightweight enough to
    serialise N×80 of these in a single Haiku prompt."""

    kind: str          # "user" | "assistant_text" | "assistant_tool" | "system_reminder"
    action_index: int   # monotonically increasing across the session
    text: str           # the content, already truncated

    def to_block(self) -> str:
        return f"[{self.action_index}] ({self.kind}) {self.text}"


def compact_transcript(entries: list[dict[str, Any]]) -> list[CompactEvent]:
    """Transform raw transcript entries into a compact event list.

    Drops:
      - tool_result blocks (kept on disk in the raw jsonl; not useful
        for finding extraction since they're machine output).
      - service tags (<system-reminder>, command-name, etc.) via
        pairs.clean_user_text.
    Keeps:
      - user text (cleaned),
      - assistant text,
      - assistant tool_use NAMES + summarized input.

    action_index is the running count of assistant tool_uses so a
    finding can reference "happened around action #N" consistent with
    the journal's numbering.
    """
    out: list[CompactEvent] = []
    action_index = 0

    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            # Skip pure tool_result wrappers.
            if pairs_mod.is_pure_tool_result(content):
                continue
            text = pairs_mod.extract_text_block(content)
            if text is None:
                continue
            cleaned = pairs_mod.clean_user_text(text)
            if cleaned is None:
                continue
            out.append(CompactEvent(
                kind="user",
                action_index=action_index,
                text=_truncate(cleaned),
            ))
            continue

        if role == "assistant":
            # Assistant text and tool_uses, in original order.
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        t = block.get("text")
                        if isinstance(t, str) and t.strip():
                            out.append(CompactEvent(
                                kind="assistant_text",
                                action_index=action_index,
                                text=_truncate(t.strip()),
                            ))
                    elif btype == "tool_use":
                        action_index += 1
                        out.append(CompactEvent(
                            kind="assistant_tool",
                            action_index=action_index,
                            text=_summarize_tool_use(
                                str(block.get("name") or ""),
                                block.get("input") or {},
                            ),
                        ))
            elif isinstance(content, str):
                if content.strip():
                    out.append(CompactEvent(
                        kind="assistant_text",
                        action_index=action_index,
                        text=_truncate(content.strip()),
                    ))

    return out


def render_chunk(events: list[CompactEvent]) -> str:
    """Render a chunk of compact events as a single newline-joined block."""
    return "\n".join(e.to_block() for e in events)


# --- Extraction prompt -------------------------------------------------------


EXTRACT_SYSTEM_PROMPT = """\
You are gadfly's project historian. You receive a COMPACTED transcript
of one Claude Code session and extract DURABLE FINDINGS that should
survive into the project's long-term memory. You will be called once
per finalised session.

Extract four classes of findings. For EACH item you produce, an
`evidence_quote` (≤200 chars, verbatim from the transcript) is
REQUIRED — items without evidence are rejected. Provenance discipline
is what separates a useful memory from an attack surface (see the
2026 MemoryGraft / MINJA research on persistent memory poisoning).

(1) `new_promises` — the agent explicitly committed to follow-up work
    that did NOT land in this session. Phrasings like:
      - "I'll do X later"
      - "TODO: implement Y next session"
      - "Out of scope; tracking this as a separate task"
    For each: {title (≤80c), evidence_quote, action_index}.
    DO NOT extract aspirational language the user wrote — only the
    agent's own commitments. DO NOT extract "I'll do X right now" if
    X was actually done in this session.

(2) `fulfilled_promises` — work that LANDED in this session that
    refers back to a promise from an earlier session (you don't see
    earlier sessions, but the agent often says "I finally got around
    to X" / "finished the Y task we discussed last time"). For each:
    {id_hint (the title or topic of the earlier promise as best you
    can name it), evidence_quote}.

(3) `new_corrections` — DURABLE user feedback / preferences. Patterns:
      - "don't use mocks in integration tests"
      - "always run cargo fmt before committing in this repo"
      - "in this project we prefer X over Y because Z"
    For each: {rule (≤120c imperative), why (≤200c reason), how_to_apply
    (≤200c short guide), evidence_quote}.
    BE CONSERVATIVE — single off-hand remarks are NOT corrections.
    Only extract when the user clearly states a preference that should
    apply across all future work in this project.

(4) `knowledge_updates` — KNOWLEDGE-GRAPH nodes (subsystems). A
    subsystem is a group of files with a shared purpose. Extract when
    the session reveals enough to assert "subsystem X lives in files
    A, B, C and does Y". For each: {subsystem_id (kebab-case, stable),
    title (≤80c), purpose (≤200c), files (list of repo-relative paths
    actually touched), evidence_quote}.
    DO NOT invent subsystems from a single grep or Read. A subsystem
    requires direct evidence in the session of cohesion (e.g. the
    agent edited multiple files in the group, or the user described
    them as a unit).

(5) `fulfilled_promise_ids` — IDs of currently-open project promises
    (from past sessions) that THIS session genuinely COMPLETED. You
    will receive a list of candidates in an `<open_promises>` section
    of the user message. Each entry has an `id`, a `title`, and an
    `evidence` quote from the session where the promise was made.
    Return the verbatim `id` of any promise whose work was actually
    landed in this transcript.

    Strict rules — read these carefully:
      - Include an id ONLY when the transcript shows the work was
        ACTUALLY DONE in this session. Discussion / planning /
        starting is NOT fulfillment.
      - Language-agnostic: a session can be in any language. "всё
        готово, тесты зелёные", "fait, tests passent", "終わった、
        テスト通った" all count as completion. Read the semantic
        meaning, not the words.
      - Do NOT include an id for work that was merely related but
        not closing the specific promise.
      - Be conservative. If unsure → omit. The corpus prefers a
        promise that quietly ages out over a wrong fulfillment.
      - Return ids verbatim from the candidate list. Do not invent
        new ids and do not modify the format.
      - If no candidate promise was fulfilled, return an empty list.

Output protocol — non-negotiable:
  - Your entire response is ONE and only one call to the
    `extract_findings` tool with the four lists.
  - Empty lists are FINE — most sessions produce no findings at all.
  - Do NOT write any preamble, explanation, reasoning, acknowledgement.
  - If you have nothing to add, return all four lists empty.
  - This is ADD-only extraction. Do not signal removals — that's
    handled elsewhere via revocation.
"""


EXTRACT_FINDINGS_DESCRIPTION = (
    "Submit the four lists of new findings extracted from this session. "
    "Call exactly once. Empty lists are allowed when nothing durable "
    "appeared. Every item MUST include evidence_quote."
)


EXTRACT_FINDINGS_INPUT_SCHEMA: dict[str, Any] = {
    "new_promises": list,
    "fulfilled_promises": list,
    "new_corrections": list,
    "knowledge_updates": list,
    "fulfilled_promise_ids": list,
}


def _system_prompt_sha() -> str:
    return hashlib.sha256(EXTRACT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:16]


# --- MCP tool + SDK options --------------------------------------------------


@dataclass
class _Captured:
    payload: dict[str, Any] | None = None


def _build_extract_tool(captured: _Captured):
    @tool(
        "extract_findings",
        EXTRACT_FINDINGS_DESCRIPTION,
        EXTRACT_FINDINGS_INPUT_SCHEMA,
    )
    async def extract_findings(args: dict[str, Any]) -> dict[str, Any]:
        captured.payload = args
        return {"content": [{"type": "text", "text": "findings recorded"}]}

    return extract_findings


def _build_options(captured: _Captured, model: str) -> ClaudeAgentOptions:
    server = create_sdk_mcp_server(
        "gadfly_historian",
        "1.0.0",
        [_build_extract_tool(captured)],
    )
    return ClaudeAgentOptions(
        model=model,
        system_prompt=EXTRACT_SYSTEM_PROMPT,
        mcp_servers={"gadfly_historian": server},
        allowed_tools=["mcp__gadfly_historian__extract_findings"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        # Recursion guard — empty inner-CLI settings prevent our own
        # PostToolUse hook from re-triggering. See watchdog._build_options
        # for the full rationale.
        settings="{}",
        thinking=ThinkingConfigDisabled(type="disabled"),
        # max_turns=3 (not 2 like watchdog/journal) — on very large
        # transcripts Haiku occasionally emits a text turn before
        # calling extract_findings. Giving it a third turn lets the
        # tool call land on retry within the same call rather than
        # nuking the whole digest. Observed in real corpus:
        # session 870ee492 (137MB) hit max_turns=2 once and was
        # heading toward exhaustion before this fix.
        max_turns=3,
        env={"GADFLY_INTERNAL": "1"},
    )


RunQuery = Callable[[str, ClaudeAgentOptions], Awaitable[None]]


async def _default_run_query(prompt: str, options: ClaudeAgentOptions) -> None:
    async for _ in query(prompt=prompt, options=options):
        pass


# --- User-message builder ----------------------------------------------------


MAX_OPEN_PROMISE_CANDIDATES = 20


def _candidate_open_promises(
    cwd: str,
    events: list[CompactEvent],
    *,
    max_n: int = MAX_OPEN_PROMISE_CANDIDATES,
) -> list[ps.Promise]:
    """Pick the open promises most likely to be touched by this session.

    Filtered by token overlap between session text and promise title.
    Without a filter we'd push the entire open-promise list (gadfly: 250+)
    into every prompt — too noisy and bloats prompt size. Top-N is enough
    for Haiku to detect fulfillment when relevant; if the matching open
    promise truly never overlapped the session lexically, fulfillment was
    extremely unlikely anyway.
    """
    state = ps.load_state(cwd)
    open_ps = [p for p in state.promises.values() if p.status == "open"]
    if not open_ps:
        return []
    # Sample the session text — cap to keep tokenisation cheap.
    session_text = " ".join(e.text for e in events[:400])
    sess_tokens = _tokens(session_text)
    if not sess_tokens:
        return []
    common = _common_tokens_in_corpus(state)
    scored: list[tuple[float, ps.Promise]] = []
    for p in open_ps:
        score = _kw_overlap(sess_tokens, _tokens(p.title), common_tokens=common)
        if score <= 0:
            continue
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [p for _, p in scored[:max_n]]


def render_open_promises_block(candidates: list[ps.Promise]) -> str:
    if not candidates:
        return ""
    lines: list[str] = []
    lines.append("## <open_promises>")
    lines.append(
        "Currently-open promises from PRIOR sessions in this project. If "
        "any of these were ACTUALLY COMPLETED during the work shown above "
        "(in ANY language — Russian, English, Japanese, etc.), include "
        "their verbatim `id` in the `fulfilled_promise_ids` output. "
        "Conservative bias: when uncertain, omit. Return ids verbatim, do "
        "not invent."
    )
    for p in candidates:
        ev = (p.provenance.evidence_quote or "").replace("\n", " ").strip()
        if len(ev) > 200:
            ev = ev[:200] + "…"
        lines.append(
            f"- id: {p.id}\n"
            f"  title: {p.title}\n"
            f"  evidence: \"{ev}\""
        )
    lines.append("## </open_promises>")
    return "\n".join(lines)


def build_extract_user_message(
    *,
    session_id: str,
    chunk_events: list[CompactEvent],
    chunk_index: int,
    chunk_count: int,
    previously_extracted: dict[str, Any] | None = None,
    open_promises_block: str = "",
) -> str:
    """Compose the per-chunk user message for the extractor.

    For multi-chunk sessions we pass forward the PRIOR chunks'
    extracted findings (not prior semantic state — that would break
    ground-truth preservation). This lets the extractor recognise when
    the agent fulfils in chunk 5 a promise it made in chunk 1.

    `open_promises_block` is the rendered <open_promises> section — the
    list of open project-level promises Haiku will check for
    fulfillment. Computed once for the whole session and threaded
    through every chunk so a chunk-5 completion of a chunk-1 promise
    still surfaces (same as previously_extracted).
    """
    parts: list[str] = []
    parts.append(f"## Session {session_id}  (chunk {chunk_index + 1}/{chunk_count})")
    if previously_extracted:
        parts.append(
            "## Findings extracted from earlier chunks of this same session\n"
            + json.dumps(previously_extracted, ensure_ascii=False, indent=2)
        )
    parts.append("## Compacted events")
    parts.append(render_chunk(chunk_events))
    if open_promises_block:
        parts.append(open_promises_block)
    parts.append(
        "## Task\n"
        "Extract durable findings (rules 1–5 in the system prompt). Call "
        "`extract_findings` exactly once. Empty lists are fine. Every "
        "item MUST include `evidence_quote` (≤200c verbatim from the "
        "events above). For rule (5), only return ids verbatim from the "
        "<open_promises> list, and only when the work was clearly "
        "completed in this transcript."
    )
    return "\n\n".join(parts)


# --- Findings merging --------------------------------------------------------


def _merge_findings(into: dict[str, Any], chunk: dict[str, Any]) -> None:
    for key in ("new_promises", "fulfilled_promises", "new_corrections", "knowledge_updates"):
        items = chunk.get(key) or []
        if not isinstance(items, list):
            continue
        existing = into.setdefault(key, [])
        for item in items:
            if isinstance(item, dict):
                existing.append(item)
    fp_ids = chunk.get("fulfilled_promise_ids") or []
    if isinstance(fp_ids, list):
        existing_ids = into.setdefault("fulfilled_promise_ids", [])
        seen = set(existing_ids)
        for pid in fp_ids:
            if isinstance(pid, str) and pid and pid not in seen:
                existing_ids.append(pid)
                seen.add(pid)


def _empty_findings() -> dict[str, Any]:
    return {
        "new_promises": [],
        "fulfilled_promises": [],
        "new_corrections": [],
        "knowledge_updates": [],
        "fulfilled_promise_ids": [],
    }


# --- The distill call --------------------------------------------------------


async def _extract_async(
    *,
    prompt: str,
    model: str,
    timeout_s: float,
    run_query: RunQuery,
) -> tuple[dict[str, Any] | None, str | None]:
    captured = _Captured()
    options = _build_options(captured, model)
    try:
        await asyncio.wait_for(run_query(prompt, options), timeout=timeout_s)
    except asyncio.TimeoutError:
        return None, f"timeout after {timeout_s}s"
    except FileNotFoundError as exc:
        return None, f"claude CLI not found: {exc!s}"
    except Exception as exc:
        return None, f"agent-sdk error: {exc!r}"
    if captured.payload is None:
        return None, "Haiku did not call extract_findings"
    return captured.payload, None


@dataclass
class DistillResult:
    raw_digest: dict[str, Any]    # what gets persisted to raw/<sid>.json
    error: str | None
    latency_ms: float
    chunks: int                    # number of Haiku calls made


def distill_session(
    *,
    cwd: str,
    session_id: str,
    transcript_path: Path,
    transcript_mtime: float,
    transcript_sha: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    run_query: RunQuery = _default_run_query,
) -> DistillResult:
    """Run extraction on one session's transcript. Synchronous wrapper
    around the chunked async extractor.

    Persists the raw digest to disk via `project_state.write_raw_digest`.
    Logs an audit event. Returns the digest for downstream re-aggregation.
    Never raises.
    """
    t0 = time.perf_counter()
    entries = _read_transcript(transcript_path)
    events = compact_transcript(entries)
    if not events:
        # Empty transcript or unparseable — write an empty digest so
        # we don't keep re-trying.
        empty = {
            "session_id": session_id,
            "ts": time.time(),
            "mtime": transcript_mtime,
            "sha": transcript_sha,
            "chunks": 0,
            **_empty_findings(),
        }
        ps.write_raw_digest(cwd, session_id, empty)
        return DistillResult(
            raw_digest=empty,
            error=None,
            latency_ms=(time.perf_counter() - t0) * 1000,
            chunks=0,
        )

    # Chunk by event count. For most sessions this is one chunk.
    chunks: list[list[CompactEvent]] = []
    for i in range(0, len(events), MAX_EVENTS_PER_CHUNK):
        chunks.append(events[i : i + MAX_EVENTS_PER_CHUNK])

    # Cross-session fulfillment candidates — computed once over the full
    # event stream so a chunk-5 completion of a chunk-1 setup still has
    # the candidate list available. Pure-Python, no Haiku.
    try:
        candidates = _candidate_open_promises(cwd, events)
        open_promises_block = render_open_promises_block(candidates)
    except Exception:
        open_promises_block = ""

    accumulated = _empty_findings()
    error: str | None = None
    chunks_completed = 0
    for idx, chunk in enumerate(chunks):
        prompt = build_extract_user_message(
            session_id=session_id,
            chunk_events=chunk,
            chunk_index=idx,
            chunk_count=len(chunks),
            previously_extracted=accumulated if idx > 0 else None,
            open_promises_block=open_promises_block,
        )
        try:
            payload, err = asyncio.run(
                _extract_async(
                    prompt=prompt,
                    model=model,
                    timeout_s=timeout_s,
                    run_query=run_query,
                )
            )
        except Exception as exc:
            error = f"asyncio.run failed: {exc!r}"
            break
        if err:
            error = err
            break
        if payload is None:
            error = "Haiku returned no payload"
            break
        _merge_findings(accumulated, payload)
        chunks_completed += 1

    # Attach action_index to items that didn't specify one (best-effort:
    # use the chunk's last action_index so retrieval can still order
    # findings chronologically).
    last_action_index = events[-1].action_index if events else 0
    for key in ("new_promises", "new_corrections", "knowledge_updates"):
        for item in accumulated[key]:
            if isinstance(item, dict) and "action_index" not in item:
                item["action_index"] = last_action_index

    is_partial = error is not None and chunks_completed > 0
    digest = {
        "session_id": session_id,
        "ts": time.time(),
        "mtime": transcript_mtime,
        "sha": transcript_sha,
        "chunks": len(chunks),
        "chunks_completed": chunks_completed,
        "partial": is_partial,
        "prompt_sha": _system_prompt_sha(),
        **accumulated,
    }
    latency_ms = (time.perf_counter() - t0) * 1000

    # Persistence policy:
    #   - Full success (no error): write digest, clear failure marker.
    #   - PARTIAL success (some chunks OK, later one failed): we still
    #     write what we got — losing 60 chunks because chunk 61 hit
    #     max_turns is worse than keeping the 60. Marker is also written
    #     so the daemon notes the attempt; if a later sweep with newer
    #     prompts succeeds fully, marker clears.
    #   - Total failure (no chunks completed): no digest written, just
    #     failure marker; retried until MAX_DISTILL_ATTEMPTS.
    attempts = 0
    if error is None:
        ps.write_raw_digest(cwd, session_id, digest)
        ps.clear_distill_failure(cwd, session_id)
    elif is_partial:
        ps.write_raw_digest(cwd, session_id, digest)
        attempts = ps.record_distill_failure(
            cwd, session_id, f"partial: {chunks_completed}/{len(chunks)} chunks then {error}",
        )
    else:
        attempts = ps.record_distill_failure(cwd, session_id, error)

    # Audit-log event.
    try:
        audit_log.append_historian_event(
            session_id=session_id,
            cwd=cwd,
            chunks=len(chunks),
            findings_counts={k: len(accumulated[k]) for k in accumulated},
            latency_ms=latency_ms,
            error=error if not attempts else f"{error} (attempt {attempts}/{ps.MAX_DISTILL_ATTEMPTS})",
        )
    except Exception:
        pass

    return DistillResult(
        raw_digest=digest,
        error=error,
        latency_ms=latency_ms,
        chunks=len(chunks),
    )


# --- Heartbeat reading -------------------------------------------------------


def _heartbeat_dir() -> Path:
    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "heartbeat"
    return Path.home() / ".claude" / "gadfly" / "heartbeat"


def read_heartbeat(cwd_encoded: str) -> dict[str, Any] | None:
    p = _heartbeat_dir() / f"{cwd_encoded}.tick"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_heartbeats() -> list[tuple[str, dict[str, Any], float]]:
    """Return [(cwd_encoded, payload, mtime), …] for all heartbeat files."""
    d = _heartbeat_dir()
    if not d.is_dir():
        return []
    out: list[tuple[str, dict[str, Any], float]] = []
    for p in d.glob("*.tick"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            out.append((p.stem, payload, p.stat().st_mtime))
        except Exception:
            continue
    return out


def decode_cwd(cwd_encoded: str) -> str:
    """Best-effort inverse of project_state.encode_cwd.

    We can't perfectly invert (we don't know which dashes were originally
    slashes vs underscores). For our use we just want to locate the
    transcript dir, which lives under ~/.claude/projects/<cwd_encoded>/.
    The cwd is also stored inside heartbeat payloads — prefer that when
    available.
    """
    # Most cases: leading dash → /, subsequent ones → /. The "_"
    # ambiguity is unrecoverable but Claude Code's transcript files
    # carry the cwd in user messages anyway.
    return "/" + "/".join(p for p in cwd_encoded.split("-") if p)


# --- Daemon-side helpers (used by H4) ----------------------------------------


def transcript_dir_for(cwd_encoded: str) -> Path:
    return Path.home() / ".claude" / "projects" / cwd_encoded


def find_undigested_sessions(
    cwd: str,
    *,
    cwd_encoded: str | None = None,
) -> list[tuple[str, Path, float]]:
    """Return [(session_id, transcript_path, mtime), …] for sessions in
    this cwd that have not yet been digested.

    A session is undigested when there's no raw/<sid>.json for it OR the
    transcript file's mtime is newer than what we recorded.
    """
    cwd_encoded = cwd_encoded or ps.encode_cwd(cwd)
    tdir = transcript_dir_for(cwd_encoded)
    if not tdir.is_dir():
        return []
    state = ps.load_state(cwd)
    digested = state.digested_sessions
    out: list[tuple[str, Path, float]] = []
    for tp in tdir.glob("*.jsonl"):
        sid = tp.stem
        try:
            mtime = tp.stat().st_mtime
        except OSError:
            continue
        prev = digested.get(sid)
        if prev and float(prev.get("mtime") or 0) >= mtime:
            continue
        # Skip sessions that have exhausted their retry budget — they
        # block the daemon if we keep hammering them, and the user can
        # reset by deleting <project>/failed/<sid>.json.
        if ps.session_exhausted(cwd, sid):
            continue
        out.append((sid, tp, mtime))
    # Oldest first — newer sessions reference promises from older ones.
    out.sort(key=lambda t: t[2])
    return out


def session_is_done(
    transcript_mtime: float,
    heartbeat_payload: dict[str, Any] | None,
    *,
    session_id: str | None = None,
) -> bool:
    """True when the session is past its activity window.

    The heartbeat is *per-cwd* and identifies the ONE session currently
    being worked on (Claude Code holds one active session per cwd at a
    time). Therefore:

      - if heartbeat.session_id matches this session AND heartbeat is
        fresh → this session IS the active one, NOT done.
      - if heartbeat.session_id differs (or no heartbeat) AND the
        session's own transcript hasn't been touched for DONE_AFTER_S
        → done. This lets the daemon digest historical sessions of a
        cwd while the user is actively working on a new one.
    """
    hb_sid = (heartbeat_payload or {}).get("session_id") if heartbeat_payload else None
    hb_ts = float((heartbeat_payload or {}).get("ts") or 0.0)
    if session_id and hb_sid and session_id == hb_sid:
        # This IS the current active session — done only when its own
        # heartbeat has gone stale.
        return (time.time() - hb_ts) >= DONE_AFTER_S
    # Otherwise we're looking at a historical session in the same cwd.
    # Its done-ness depends on its own transcript mtime, not the
    # heartbeat (which belongs to a different session).
    return (time.time() - transcript_mtime) >= DONE_AFTER_S


# --- Retrieval (Phase B) ----------------------------------------------------

MAX_PRIORS = 5


@dataclass
class PriorHit:
    """One retrieved finding, ready to render into a prompt block."""

    kind: str          # "promise" | "correction" | "subsystem"
    id: str
    title: str
    body: str          # human-readable summary
    evidence_quote: str
    source_session: str
    score: float


_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "by", "as", "at", "from",
    "this", "that", "these", "those", "it", "its", "into",
}


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for raw in re.split(r"[^A-Za-z0-9_]+", text.lower()):
        if len(raw) >= 3 and raw not in _STOP:
            out.add(raw)
    return out


def _kw_overlap(a: set[str], b: set[str], common_tokens: set[str] | None = None) -> float:
    """Token-overlap similarity with project-corpus-aware filtering.

    `common_tokens` is the set of tokens that appear in MANY items
    across the project corpus (e.g. 'watchdog' in a project ALL ABOUT
    the watchdog). When the only shared token is in this common set,
    the match is too generic to surface — score 0. This catches the
    real-world false-positive: stale promises about "watchdog X"
    matching every new edit in a watchdog project.

    Distinctive single-token matches still surface (e.g. 'marlin' in
    a project not centered on Marlin), so concise titles aren't lost.
    """
    if not a or not b:
        return 0.0
    shared = a & b
    if not shared:
        return 0.0
    if common_tokens and len(shared) == 1 and shared.issubset(common_tokens):
        # Only a generic project-wide token matched — not enough signal.
        return 0.0
    return len(shared) / max(1, min(len(a), len(b)))


def _common_tokens_in_corpus(state: "ps.ProjectState", threshold_frac: float = 0.15) -> set[str]:
    """Tokens appearing in ≥threshold_frac of project items (default 15%).

    Computed inline at retrieval time. Cheap — we already iterate
    these items. Result is a set of "project-wide" tokens that
    shouldn't single-handedly justify a match. E.g. for the gadfly
    project this catches 'watchdog', 'haiku', 'gadfly'.

    Threshold 15% (with 3-item floor) caught the real-world false-positive:
    'watchdog' appeared in ~12% of gadfly promise titles, enough to be
    "in the air" but not enough to be rejected by 30%. 15% is a tighter
    fit; revisit if it gates legitimate distinctive tokens.
    """
    titles: list[set[str]] = []
    for p in state.promises.values():
        if p.status == "open":
            titles.append(_tokens(p.title))
    for c in state.corrections.values():
        titles.append(_tokens(c.rule + " " + c.why))
    for s in state.subsystems.values():
        titles.append(_tokens(s.title + " " + s.purpose))
    if not titles:
        return set()
    from collections import Counter
    cnt: Counter[str] = Counter()
    for t in titles:
        for tok in t:
            cnt[tok] += 1
    threshold = max(3, int(len(titles) * threshold_frac))
    return {tok for tok, n in cnt.items() if n >= threshold}


def find_relevant_priors(
    cwd: str,
    *,
    workstream_title: str,
    file_paths: list[str] | None = None,
    max_results: int = MAX_PRIORS,
) -> list[PriorHit]:
    """Retrieve top-N priors relevant to a workstream.

    Three cheap signals fused:
      1. keyword overlap (title vs. promise.title / correction.rule /
         subsystem.title)
      2. file-path overlap (against subsystem.files)
      3. usefulness_score (outcomes feedback) — boosts repeatedly-cited
         priors

    No Haiku call. Pure-Python. The MemoryArena lesson is "make
    retrieval cheap and frequent". We add a reranker call (Phase B+1)
    only if precision proves insufficient on real corpora.
    """
    file_paths = file_paths or []
    state = ps.load_state(cwd)
    if not state.promises and not state.corrections and not state.subsystems:
        return []

    q_tokens = _tokens(workstream_title)
    file_set = {fp for fp in file_paths if fp}
    hits: list[PriorHit] = []
    _now = time.time()
    # IDF-style filter: tokens that appear in ≥30% of project items are
    # too generic to single-handedly justify a match. Computed once per
    # retrieval call.
    common = _common_tokens_in_corpus(state)

    def _recency_penalty(ts: float) -> float:
        """Penalty term for stale entries. -0.05 / week of age, capped
        at -0.5. Without this the retrieval keeps surfacing months-old
        priors that share a generic keyword with the current workstream
        (e.g. 'watchdog' in a project all about the watchdog).
        Recent evidence is much more relevant than old."""
        if ts <= 0 or _now <= ts:
            return 0.0
        age_weeks = (_now - ts) / (7 * 24 * 3600)
        return max(-0.5, -0.05 * age_weeks)

    # --- promises ---
    for pid, p in state.promises.items():
        if p.status != "open":
            continue
        score = _kw_overlap(q_tokens, _tokens(p.title), common_tokens=common)
        if score <= 0:
            continue
        score += _recency_penalty(p.last_seen_ts or p.provenance.first_seen_ts)
        if score <= 0:
            continue
        hits.append(PriorHit(
            kind="promise",
            id=pid,
            title=p.title,
            body=f"agent committed to follow up, status={p.status}",
            evidence_quote=p.provenance.evidence_quote,
            source_session=p.provenance.source_session,
            score=score,
        ))

    # --- corrections ---
    for cid, c in state.corrections.items():
        score = _kw_overlap(q_tokens, _tokens(c.rule + " " + c.why), common_tokens=common)
        # usefulness signal nudges things up/down at the margins
        score += 0.05 * c.usefulness_score
        score += _recency_penalty(c.provenance.first_seen_ts)
        if score <= 0:
            continue
        hits.append(PriorHit(
            kind="correction",
            id=cid,
            title=c.rule,
            body=(c.why or "") + ("\nhow: " + c.how_to_apply if c.how_to_apply else ""),
            evidence_quote=c.provenance.evidence_quote,
            source_session=c.provenance.source_session,
            score=score,
        ))

    # --- subsystems (use file-path overlap as primary signal) ---
    for sid_canonical, s in state.subsystems.items():
        kw_score = _kw_overlap(q_tokens, _tokens(s.title + " " + s.purpose), common_tokens=common)
        sub_files = set(s.files or [])
        file_score = 0.0
        if file_set and sub_files:
            file_score = len(file_set & sub_files) / max(1, len(file_set))
        score = max(kw_score, file_score) + 0.05 * s.usefulness_score
        # Subsystems are about CURRENT codebase shape — use last_touched
        # rather than first_seen for recency. A subsystem touched
        # yesterday is fresh even if first observed weeks ago.
        score += _recency_penalty(s.last_touched_ts or s.provenance.first_seen_ts)
        if score <= 0:
            continue
        hits.append(PriorHit(
            kind="subsystem",
            id=sid_canonical,
            title=s.title,
            body=f"{s.purpose}\nfiles: {', '.join(s.files)}",
            evidence_quote=s.provenance.evidence_quote,
            source_session=s.provenance.source_session,
            score=score,
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:max_results]


def render_priors_block(hits: list[PriorHit]) -> str:
    """Render a Phase B project_priors block for injection into the
    journal maintainer prompt."""
    if not hits:
        return ""
    lines: list[str] = []
    lines.append("## Project priors (durable findings from past sessions in this cwd)")
    lines.append(
        "These are NOT speculation. Each is grounded in a verbatim "
        "evidence_quote from a prior session. Use them when relevant; "
        "ignore otherwise. Do not flag drift just because new work doesn't "
        "match these — they're informational."
    )
    for h in hits:
        lines.append(
            f"\n### [{h.kind}] {h.title}\n"
            f"  body: {h.body[:600]}\n"
            f"  evidence: \"{h.evidence_quote}\"\n"
            f"  from: session {h.source_session[:12]}"
        )
    return "\n".join(lines)


# --- Daemon orchestration ---------------------------------------------------


def _resolve_cwd_for_encoded(cwd_encoded: str, heartbeat_payload: dict[str, Any] | None) -> str:
    """Best-effort canonical cwd. Heartbeat payload may carry it; fall
    back to decode_cwd which is lossy but workable."""
    if isinstance(heartbeat_payload, dict):
        ck = heartbeat_payload.get("cwd")
        if isinstance(ck, str) and ck.startswith("/"):
            return ck
    return decode_cwd(cwd_encoded)


def discover_work(*, force_done: bool = False) -> list[dict[str, Any]]:
    """Scan heartbeats AND any existing project dirs, return a list of
    {cwd_encoded, cwd, transcript_path, session_id, mtime, hb} dicts
    for sessions that are done and not yet digested.

    `force_done=True` ignores the heartbeat-staleness check and treats
    every undigested session as ready — used by --sweep and --backfill.
    """
    work: list[dict[str, Any]] = []

    # Build the union of (cwd) we know about: heartbeats + already-
    # digested project dirs.
    heartbeats = {cwd_encoded: payload for cwd_encoded, payload, _mtime in list_heartbeats()}

    known_encoded: set[str] = set(heartbeats.keys())
    proot = ps.project_root()
    if proot.is_dir():
        for d in proot.iterdir():
            if d.is_dir():
                known_encoded.add(d.name)

    # Order cwds by heartbeat freshness DESC — the project the user is
    # most actively working on gets digested first, so its priors are
    # ready by the time a new workstream opens. Cwds with no heartbeat
    # (purely historical, no current activity) come last. Within a
    # bucket, alphabetical for determinism.
    def _cwd_priority(enc: str) -> tuple[float, str]:
        hb = heartbeats.get(enc)
        ts = float((hb or {}).get("ts") or 0.0)
        # Negate ts so higher freshness sorts first.
        return (-ts, enc)

    for cwd_encoded in sorted(known_encoded, key=_cwd_priority):
        hb = heartbeats.get(cwd_encoded)
        cwd = _resolve_cwd_for_encoded(cwd_encoded, hb)
        for sid, tp, mtime in find_undigested_sessions(cwd, cwd_encoded=cwd_encoded):
            if not force_done and not session_is_done(mtime, hb, session_id=sid):
                # Session may still be active — skip for now, daemon will
                # come back on the next poll.
                continue
            work.append({
                "cwd_encoded": cwd_encoded,
                "cwd": cwd,
                "transcript_path": tp,
                "session_id": sid,
                "mtime": mtime,
                "hb": hb,
            })
    return work


def process_one(item: dict[str, Any], *, run_query: RunQuery = _default_run_query) -> DistillResult:
    """Distil one work item and merge it into the cwd's project state.

    Updates: raw/<sid>.json (via distill_session) + state.json with the
    digested-sessions index. The semantic aggregator runs INSIDE
    save_state-time via a rebuild_from_raw call so the result is always
    consistent with what's on disk.
    """
    cwd = item["cwd"]
    sid = item["session_id"]
    tp: Path = item["transcript_path"]
    mtime = item["mtime"]

    res = distill_session(
        cwd=cwd,
        session_id=sid,
        transcript_path=tp,
        transcript_mtime=mtime,
        run_query=run_query,
    )

    # Re-aggregate from raw/ — deterministic; preserves drift-resistance.
    state = ps.rebuild_from_raw(cwd, prompt_sha=_system_prompt_sha())
    ps.save_state(state)
    return res


def sweep_once(
    *,
    force_done: bool = False,
    run_query: RunQuery = _default_run_query,
    max_per_poll: int | None = DEFAULT_MAX_PER_POLL,
) -> dict[str, Any]:
    """Run one full pass: find work, process up to `max_per_poll` items.

    `max_per_poll=None` removes the cap (used by `backfill`). The cap
    exists so a fresh install on a project with hundreds of historical
    sessions doesn't burn Haiku quota in one cycle.

    Returns a summary dict for CLI/--status output.
    """
    work = discover_work(force_done=force_done)
    considered = len(work)
    if max_per_poll is not None and max_per_poll > 0:
        work = work[:max_per_poll]
    processed = 0
    errors = 0
    for item in work:
        try:
            res = process_one(item, run_query=run_query)
            processed += 1
            if res.error:
                errors += 1
        except Exception:
            errors += 1
    return {
        "considered": considered,
        "processed": processed,
        "errors": errors,
        "deferred": max(0, considered - processed - errors),
    }


def watch_loop(
    *,
    run_query: RunQuery = _default_run_query,
    poll_s: float = DAEMON_POLL_S,
    verbose: bool = False,
    max_per_poll: int | None = DEFAULT_MAX_PER_POLL,
) -> int:
    """Long-running daemon. Wakes every poll_s, processes done sessions.

    Returns when interrupted. Never raises out.

    `verbose=True` prints a one-line status to stderr on every poll
    cycle (whether work was found or not) so an operator tailing the
    log can confirm the daemon is alive.
    """
    import sys as _sys

    iteration = 0
    while True:
        iteration += 1
        t0 = time.time()
        try:
            summary = sweep_once(run_query=run_query, max_per_poll=max_per_poll)
            if verbose:
                hb_count = len(list_heartbeats())
                stamp = time.strftime("%H:%M:%S")
                _sys.stderr.write(
                    f"[{stamp}] poll #{iteration:>3d}  "
                    f"heartbeats={hb_count}  "
                    f"considered={summary['considered']}  "
                    f"processed={summary['processed']}  "
                    f"errors={summary['errors']}  "
                    f"deferred={summary['deferred']}  "
                    f"elapsed={time.time() - t0:.1f}s\n"
                )
                _sys.stderr.flush()
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            # Daemon failures must not crash the daemon — log and continue.
            if verbose:
                _sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] poll error: {exc!r}\n")
                _sys.stderr.flush()
        try:
            time.sleep(poll_s)
        except KeyboardInterrupt:
            return 0


# --- CLI --------------------------------------------------------------------


def _cli_status() -> int:
    proot = ps.project_root()
    print(f"gadfly historian — projects root: {proot}")
    hb_count = len(list_heartbeats())
    print(f"heartbeats:        {hb_count}")
    work = discover_work(force_done=False)
    print(f"sessions ready:    {len(work)}")
    if not proot.is_dir():
        # No corpus yet. If there's backlog AND no daemon has ever run,
        # report it as stale so first-time users see they need to start
        # the daemon.
        if len(work) > 0:
            print(
                f"\nWARNING: {len(work)} session(s) ready to digest but "
                f"no project corpus exists yet. Start the daemon: "
                f"python -m gadfly.historian watch"
            )
            return 1
        print("no project corpus on disk yet")
        return 0
    print()
    for d in sorted(proot.iterdir()):
        if not d.is_dir():
            continue
        try:
            state = ps.load_state(_resolve_cwd_for_encoded(d.name, None))
        except Exception:
            continue
        print(
            f"  {d.name}: "
            f"promises={len(state.promises)} "
            f"corrections={len(state.corrections)} "
            f"subsystems={len(state.subsystems)} "
            f"quarantine={len(state.quarantine)} "
            f"digested={len(state.digested_sessions)}"
        )
    # Non-zero exit when there's backlog AND no recent digest activity:
    # caller (systemd, cron, alerting) can treat this as "daemon stale".
    if len(work) == 0:
        return 0
    if _daemon_appears_dead():
        print(
            f"\nWARNING: {len(work)} session(s) ready to digest but the "
            f"daemon appears to have stalled (no recent state.json "
            f"updates). Start it with: python -m gadfly.historian watch"
        )
        return 1
    return 0


def _daemon_appears_dead(*, threshold_s: float = DONE_AFTER_S * 2) -> bool:
    """Heuristic: backlog exists AND no project state.json has been
    updated in `threshold_s` seconds. Defaults to 10 min — comfortably
    longer than one poll cycle even with cap=5 chewing slowly.
    """
    proot = ps.project_root()
    if not proot.is_dir():
        return False
    newest = 0.0
    for state_path in proot.glob("*/state.json"):
        try:
            mtime = state_path.stat().st_mtime
            if mtime > newest:
                newest = mtime
        except OSError:
            continue
    if newest == 0.0:
        # No state.json anywhere yet — daemon hasn't run, by definition stale.
        return True
    return (time.time() - newest) > threshold_s


def _cli_sweep() -> int:
    summary = sweep_once()
    print(f"considered={summary['considered']} processed={summary['processed']} errors={summary['errors']}")
    return 0 if summary["errors"] == 0 else 1


def _cli_watch(
    *,
    verbose: bool = False,
    poll_s: float = DAEMON_POLL_S,
    max_per_poll: int | None = DEFAULT_MAX_PER_POLL,
) -> int:
    cap = "unlimited" if not max_per_poll else str(max_per_poll)
    print(
        f"gadfly historian watching every {poll_s:.0f}s "
        f"(max {cap} sessions / poll); Ctrl-C to stop",
        flush=True,
    )
    return watch_loop(verbose=verbose, poll_s=poll_s, max_per_poll=max_per_poll)


def _cli_backfill(cwd_arg: str, confirmed: bool, limit: int | None = None) -> int:
    if not confirmed:
        print(
            "Backfill is expensive (1 Haiku call per prior session). "
            "Re-run with --yes-i-know-the-cost to proceed."
        )
        return 2
    cwd_encoded = ps.encode_cwd(cwd_arg)
    sessions = find_undigested_sessions(cwd_arg, cwd_encoded=cwd_encoded)
    total_pending = len(sessions)
    if limit is not None and limit > 0:
        sessions = sessions[:limit]
    print(
        f"backfilling {len(sessions)} session(s) for {cwd_arg}"
        + (f" (limit {limit} of {total_pending} total)" if limit else "")
    )
    processed = 0
    errors = 0
    for sid, tp, mtime in sessions:
        item = {
            "cwd_encoded": cwd_encoded,
            "cwd": cwd_arg,
            "transcript_path": tp,
            "session_id": sid,
            "mtime": mtime,
            "hb": None,
        }
        try:
            res = process_one(item)
            if res.error:
                errors += 1
                print(f"  ERR {sid}: {res.error}")
            else:
                processed += 1
                print(f"  OK  {sid}  chunks={res.chunks}  latency={res.latency_ms:.0f}ms")
        except Exception as exc:
            errors += 1
            print(f"  ERR {sid}: {exc!r}")
    print(f"done: processed={processed} errors={errors}")
    return 0 if errors == 0 else 1


def propose_claudemd(cwd: str, *, min_confidence: int = 2) -> str:
    """Render a candidate CLAUDE.md addition from the project corpus.

    Pure markdown, ready to paste. Includes:
      - active corrections with confidence >= min_confidence
      - active subsystems (knowledge graph)
    Does NOT include open promises — those are session-state, not
    durable knowledge. Does NOT include quarantine — by definition
    those haven't reached the promotion threshold.

    The user reviews and copies what they want. We never modify their
    project CLAUDE.md automatically — that's an obvious attack surface
    given the memory-poisoning literature.
    """
    state = ps.load_state(cwd)
    lines: list[str] = []

    lines.append("<!-- gadfly historian — candidate additions to CLAUDE.md -->")
    lines.append(f"<!-- generated {time.strftime('%Y-%m-%d %H:%M')}  cwd={cwd} -->")
    lines.append(f"<!-- review carefully; never auto-applied -->")
    lines.append("")

    # --- Corrections section ---
    high_conf = [
        c for c in state.corrections.values()
        if c.confidence >= min_confidence
    ]
    if high_conf:
        lines.append("## Project conventions and feedback")
        lines.append("")
        lines.append(
            "These are durable rules extracted from across multiple "
            "sessions in this project. Each is grounded in a verbatim "
            "evidence quote — verify before adopting."
        )
        lines.append("")
        for c in sorted(high_conf, key=lambda x: -x.confidence):
            lines.append(f"### {c.rule}")
            lines.append("")
            if c.why:
                lines.append(f"**Why:** {c.why}")
                lines.append("")
            if c.how_to_apply:
                lines.append(f"**How to apply:** {c.how_to_apply}")
                lines.append("")
            lines.append(
                f"_evidence: \"{c.provenance.evidence_quote}\" "
                f"(confidence {c.confidence}, seen in "
                f"{len(c.seen_in_sessions)} session(s))_"
            )
            lines.append("")

    # --- Subsystems section ---
    subs = sorted(state.subsystems.values(), key=lambda s: -s.usefulness_score)
    if subs:
        if high_conf:
            lines.append("---")
            lines.append("")
        lines.append("## Codebase map (subsystems)")
        lines.append("")
        lines.append(
            "Subsystems extracted from real session evidence — groups "
            "of files with a shared purpose. Not a substitute for grep; "
            "a stable overview of what's where."
        )
        lines.append("")
        for s in subs:
            lines.append(f"### {s.title}")
            lines.append("")
            if s.purpose:
                lines.append(s.purpose)
                lines.append("")
            if s.files:
                lines.append("**Files:**")
                lines.append("")
                for fp in s.files:
                    lines.append(f"- `{fp}`")
                lines.append("")
            lines.append(
                f"_first seen in session "
                f"{s.provenance.source_session[:12]}; last touched in "
                f"{s.last_touched_session[:12]}_"
            )
            lines.append("")

    if not high_conf and not subs:
        lines.append("(nothing high-confidence enough to propose yet — keep")
        lines.append("working in this cwd and the historian will accumulate")
        lines.append("evidence. Corrections need >=2 sessions; subsystems")
        lines.append(">=3 file mentions.)")

    return "\n".join(lines)


def _cli_propose_claudemd(cwd_arg: str, min_confidence: int) -> int:
    text = propose_claudemd(cwd_arg, min_confidence=min_confidence)
    print(text)
    return 0


def _cli_rebuild(cwd_arg: str) -> int:
    state = ps.rebuild_from_raw(cwd_arg, prompt_sha=_system_prompt_sha())
    ps.save_state(state)
    print(
        f"rebuilt: promises={len(state.promises)} "
        f"corrections={len(state.corrections)} "
        f"subsystems={len(state.subsystems)} "
        f"quarantine={len(state.quarantine)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="gadfly.historian",
                                 description="Project historian for gadfly.")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Print per-cwd state summary.")
    sub.add_parser("sweep", help="One-shot scan of ready sessions, then exit.")
    wp = sub.add_parser("watch", help="Long-running poll loop (daemon).")
    wp.add_argument("--verbose", action="store_true",
                    help="Print a one-line status to stderr on every poll cycle.")
    wp.add_argument("--poll-s", type=float, default=DAEMON_POLL_S,
                    help=f"Polling interval in seconds (default {DAEMON_POLL_S}).")
    wp.add_argument("--max-per-poll", type=int, default=DEFAULT_MAX_PER_POLL,
                    help=f"Max sessions to digest per poll (default {DEFAULT_MAX_PER_POLL}). "
                         "0 or negative removes the cap.")

    bp = sub.add_parser("backfill", help="Digest ALL prior sessions in a cwd.")
    bp.add_argument("cwd", help="Absolute path of the project (e.g. /home/u/repo).")
    bp.add_argument("--yes-i-know-the-cost", action="store_true",
                    dest="confirmed",
                    help="Required — backfill makes one Haiku call per session.")
    bp.add_argument("--limit", type=int, default=None,
                    help="Process at most N sessions, then exit. "
                         "Lets you onboard a large corpus gradually.")

    rp = sub.add_parser("rebuild", help="Re-derive semantic state from raw/ digests.")
    rp.add_argument("cwd", help="Absolute path of the project.")

    pcp = sub.add_parser(
        "propose-claudemd",
        help="Print a candidate CLAUDE.md addition from the corpus. Never auto-applies.",
    )
    pcp.add_argument("cwd", help="Absolute path of the project.")
    pcp.add_argument(
        "--min-confidence",
        type=int, default=2,
        help="Minimum correction confidence (default 2).",
    )

    pp = sub.add_parser(
        "promote",
        help="Force-promote a quarantined finding to the active store.",
    )
    pp.add_argument("cwd", help="Absolute path of the project.")
    pp.add_argument("id", help="Finding id (from the viewer or `status`).")

    rvp = sub.add_parser(
        "revoke",
        help="Move a finding to the revoked audit list. Never deletes raw evidence.",
    )
    rvp.add_argument("cwd", help="Absolute path of the project.")
    rvp.add_argument("id", help="Finding id (from the viewer or `status`).")
    rvp.add_argument("--reason", default="", help="Optional revocation reason.")

    args = ap.parse_args(argv)

    if args.cmd == "status" or args.cmd is None:
        return _cli_status()
    if args.cmd == "sweep":
        return _cli_sweep()
    if args.cmd == "watch":
        cap = args.max_per_poll if args.max_per_poll and args.max_per_poll > 0 else None
        return _cli_watch(verbose=args.verbose, poll_s=args.poll_s, max_per_poll=cap)
    if args.cmd == "backfill":
        return _cli_backfill(args.cwd, args.confirmed, limit=args.limit)
    if args.cmd == "rebuild":
        return _cli_rebuild(args.cwd)
    if args.cmd == "propose-claudemd":
        return _cli_propose_claudemd(args.cwd, args.min_confidence)
    if args.cmd == "promote":
        ok, msg = ps.promote_finding(args.cwd, args.id)
        print(msg)
        return 0 if ok else 2
    if args.cmd == "revoke":
        ok, msg = ps.revoke_finding(args.cwd, args.id, reason=args.reason)
        print(msg)
        return 0 if ok else 2
    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
