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
    # Snapshots of the CURRENT on-disk state of up to 3 files relevant
    # to the action being evaluated. Solves edit-window blindness:
    # instead of reconstructing the file's state from a series of
    # truncated diffs (which we used to do, with a language-specific
    # regex heuristic), we hand Haiku the verbatim file content. Haiku
    # already understands code semantics in every language — let it.
    per_file_snapshots: dict[str, str] = field(default_factory=dict)
    # Thin trajectory of file-touching actions: (action_index,
    # tool_name, file_path). No content — just metadata. Helps Haiku
    # see "agent has been thrashing on file X for 8 actions" without
    # any content heuristics.
    file_touch_trajectory: list[tuple[int, str, str]] = field(default_factory=list)
    # Active plan body, when the user has approved one via ExitPlanMode
    # in this session. None when no plan was approved. Gives the watchdog
    # ground truth on "is this action prescribed by the user?" instead
    # of guessing from journal workstreams.
    active_plan: str | None = None
    # Last N (assistant_text, user_text) pairs of dialogue before the
    # current action. Solves multi-turn-authorization blindness: when
    # the user's brief reply ("ставим", "go", "fais-le") refers back
    # to an EARLIER assistant proposal, last_assistant_plan alone
    # doesn't carry the context — the full back-and-forth does.
    recent_dialogue_pairs: list[tuple[str | None, str]] = field(default_factory=list)
    # Last N Bash actions with truncated command + output. Solves the
    # "diagnostic chain invisible" blind spot: when the agent ran
    # curl→grep→tail to investigate something across several turns,
    # the watchdog used to see only the latest command and lost the
    # context of what came back from earlier ones. Each entry:
    # (action_index, command, exit_code, stdout, stderr).
    recent_bash_actions: list[tuple[int, str, Any, str, str]] = field(default_factory=list)

    @property
    def last_user_request(self) -> str | None:
        """Back-compat / convenience: the latest user message, if any."""
        return self.recent_user_requests[-1] if self.recent_user_requests else None


MAX_USER_REQUESTS = 5
MAX_DIALOGUE_PAIRS = 5


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
    current_tool_input: dict[str, Any] | None = None,
    cwd: str | None = None,
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
    if cwd:
        ctx.cwd = cwd
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

    # Trajectory of file-touching actions (no content) and per-file
    # snapshots (verbatim disk content of relevant files). Together
    # they solve edit-window blindness: trajectory says "agent
    # touched X 10 times", snapshot says "this is what X looks like
    # right now". No regex / content heuristics anywhere — Haiku
    # parses the snapshot itself.
    try:
        ctx.file_touch_trajectory = extract_file_touch_trajectory(entries)
        target_file: str | None = None
        target_anchor: str | None = None
        if isinstance(current_tool_input, dict):
            fp = current_tool_input.get("file_path")
            if isinstance(fp, str) and fp:
                target_file = fp
            old = current_tool_input.get("old_string")
            if isinstance(old, str) and old.strip():
                target_anchor = old
        ctx.per_file_snapshots = extract_per_file_snapshots(
            entries,
            cwd=ctx.cwd,
            target_file=target_file,
            target_anchor=target_anchor,
        )
    except Exception:
        ctx.file_touch_trajectory = []
        ctx.per_file_snapshots = {}

    # Active plan body — when the user has approved a plan via
    # ExitPlanMode in this session, the prescription IS the
    # authoritative answer to "is this action allowed?". Without
    # this the watchdog has no signal that explicit work is
    # plan-prescribed and may flag it as scope creep / drift.
    try:
        ctx.active_plan = extract_active_plan(entries)
    except Exception:
        ctx.active_plan = None

    # Last N dialogue pairs (assistant_text, user_text). Solves
    # multi-turn-authorization blindness — the user's brief
    # "ставим" / "go" / "fais-le" may refer to a proposal that
    # lives in turn N-3 rather than N-1.
    try:
        ctx.recent_dialogue_pairs = [
            (p.assistant_text, p.user_text)
            for p in ctx.pairs[-MAX_DIALOGUE_PAIRS:]
        ]
    except Exception:
        ctx.recent_dialogue_pairs = []

    # Last N Bash actions with their output. Solves the diagnostic-
    # chain blind spot: agent runs curl@N-5 returning 400, grep@N-3
    # returning empty, then current Bash → watchdog without this
    # block sees only the latest call and decides "logs empty,
    # nothing diagnosed".
    try:
        ctx.recent_bash_actions = extract_recent_bash_actions(entries)
    except Exception:
        ctx.recent_bash_actions = []

    return ctx


# --- Per-file snapshots + trajectory ---------------------------------------


# Tools that mutate file content. Used for snapshot priority — an
# edited file's snapshot is more load-bearing than a read-only one.
_FILE_MUTATING_TOOLS = {"Edit", "Write", "MultiEdit"}
# Tools that touch a file (mutate OR read). Used for trajectory and
# as the snapshot candidate pool. Including Read here closes the
# "agent verified an upstream contract before changing downstream"
# blind spot — Haiku sees the verification step in trajectory and
# can inspect the verified file via snapshot.
_FILE_TOUCH_TOOLS = _FILE_MUTATING_TOOLS | {"Read"}
MAX_TRAJECTORY = 10
MAX_SNAPSHOT_FILES = 3
# Reference files (Read'd or recently touched but not the current target):
# 25KB head+tail. Most files fit; oversized files lose the middle section.
MAX_SNAPSHOT_BYTES = 25_000
MAX_SNAPSHOT_HEAD = 17_000
MAX_SNAPSHOT_TAIL = 6_000
# Current EDIT target gets a larger budget — the file the action mutates
# is the one Haiku must reason about most carefully. 60KB fits ~95% of
# real source files unsplit; over that we use an edit-region-centered
# window so the slice contains the area the agent actually changed.
MAX_TARGET_SNAPSHOT_BYTES = 60_000
TARGET_WINDOW_BEFORE = 25_000  # bytes before old_string anchor
TARGET_WINDOW_AFTER = 25_000   # bytes after
MAX_READABLE_FILE_BYTES = 5_000_000


def relpath_under_cwd(path: str, cwd: str) -> str:
    """Return `path` relative to `cwd` when possible, else `path` unchanged.

    Used to shorten absolute file paths in snapshot/trajectory rendering
    so Haiku reads `frontend/components/MaterialsImport.vue` instead of
    `/home/vasis/work/v-calc/frontend/components/MaterialsImport.vue`.
    The agent's verbatim `tool_input.file_path` stays absolute — only
    summary blocks shorten.
    """
    if not path or not cwd:
        return path
    try:
        rel = Path(path).resolve(strict=False).relative_to(
            Path(cwd).resolve(strict=False)
        )
        return str(rel)
    except (ValueError, OSError):
        return path


MAX_ACTIVE_PLAN_BYTES = 8000

# Deterministic strings the Claude Code plan-mode tooling emits when a
# plan is approved. We look for the most-recent occurrence in the
# transcript and harvest the plan body that follows.
_PLAN_MARKERS = (
    "## Approved Plan:",
    "Your plan has been saved to:",
)
# A close-of-block sentinel: the plan body ends when this string appears
# (Claude Code wraps each Exited-Plan-Mode notice in a system-reminder
# with this closing line).
_PLAN_END_MARKERS = (
    "<system-reminder>",
    "## Exited Plan Mode",
)


def extract_active_plan(
    entries: list[dict[str, Any]],
    *,
    max_bytes: int = MAX_ACTIVE_PLAN_BYTES,
) -> str | None:
    """Scan transcript backward for the most recent approved plan body.

    Returns the plan body text (capped at `max_bytes`) or None when no
    plan-mode approval was ever recorded. Pure transcript scrape, no
    Haiku, no markdown parsing — finds the verbatim string emitted by
    Claude Code's ExitPlanMode tooling and slices forward from there.

    Multiple plans in a session: the LATEST approval wins. Each new
    plan-approval supersedes prior ones (the user re-entered plan mode
    deliberately).
    """
    # Walk entries from newest to oldest, looking inside any text block
    # (user OR assistant role — the plan body comes back as a tool
    # result wrapped in a user-role system-reminder, but the format
    # varies between Claude Code versions, so check both).
    for entry in reversed(entries):
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        text = _extract_text(content)
        if not text:
            continue
        # Locate the latest plan marker inside this entry.
        best_marker_pos = -1
        for marker in _PLAN_MARKERS:
            pos = text.rfind(marker)
            if pos > best_marker_pos:
                best_marker_pos = pos
        if best_marker_pos < 0:
            continue
        body_start = best_marker_pos
        # Trim leading marker line so the plan body starts at its
        # first content line.
        nl = text.find("\n", body_start)
        if nl > 0:
            body_start = nl + 1
        # Find the earliest end-marker after body_start, if any.
        body_end = len(text)
        for marker in _PLAN_END_MARKERS:
            pos = text.find(marker, body_start)
            if 0 <= pos < body_end:
                body_end = pos
        body = text[body_start:body_end].strip()
        if not body:
            continue
        if len(body) > max_bytes:
            body = body[:max_bytes] + (
                f"\n\n[…plan body truncated at {max_bytes} bytes; full text in plan file…]"
            )
        return body
    return None


MAX_BASH_ACTIONS = 8
MAX_BASH_COMMAND = 300
MAX_BASH_OUTPUT_TOTAL = 800
MAX_BASH_OUTPUT_HEAD = 500
MAX_BASH_OUTPUT_TAIL = 250


def _extract_tool_uses_with_ids(content: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """Same shape as `_extract_tool_uses` but also returns the tool_use_id
    so we can match it to the tool_result block that follows in the next
    user-role entry."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                name = b.get("name")
                tid = b.get("id")
                inp = b.get("input")
                if isinstance(name, str) and isinstance(tid, str) and isinstance(inp, dict):
                    out.append((name, tid, inp))
    return out


def _extract_tool_results(content: Any) -> dict[str, Any]:
    """Return {tool_use_id: result_content} for tool_result blocks
    inside a user-role content list. `result_content` is the raw inner
    block — string or list of dicts (with `text` / `type`)."""
    out: dict[str, Any] = {}
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if isinstance(tid, str):
                    out[tid] = b.get("content")
    return out


def _bash_result_to_strings(result_content: Any) -> tuple[Any, str, str]:
    """Coerce a tool_result.content payload into (exit_code, stdout, stderr).

    Claude Code's Bash tool returns its result either as a dict
    {stdout, stderr, exit_code, ...} (newer transcripts) or as plain
    text inside a text-block list (older ones). Tolerate both shapes.
    """
    exit_code: Any = None
    stdout = ""
    stderr = ""
    if isinstance(result_content, list):
        # Inspect text blocks; sometimes a single text block contains
        # the literal JSON payload, other times plain stdout.
        for b in result_content:
            if isinstance(b, dict) and b.get("type") == "text":
                t = b.get("text")
                if isinstance(t, str):
                    # Try to parse as JSON if it looks structured.
                    stripped = t.strip()
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                exit_code = parsed.get("exit_code", exit_code)
                                if isinstance(parsed.get("stdout"), str):
                                    stdout = parsed["stdout"]
                                if isinstance(parsed.get("stderr"), str):
                                    stderr = parsed["stderr"]
                                continue
                        except json.JSONDecodeError:
                            pass
                    if not stdout:
                        stdout = t
                    else:
                        stdout += "\n" + t
    elif isinstance(result_content, dict):
        exit_code = result_content.get("exit_code", exit_code)
        if isinstance(result_content.get("stdout"), str):
            stdout = result_content["stdout"]
        if isinstance(result_content.get("stderr"), str):
            stderr = result_content["stderr"]
    elif isinstance(result_content, str):
        stdout = result_content
    return exit_code, stdout, stderr


def _head_tail_clip(text: str, *, total: int, head: int, tail: int) -> str:
    """Bytes-budget cut with explicit truncation marker. Same shape as
    `prompts._head_tail_truncate` but lives here so the extraction
    layer can prep the strings before they hit the prompt builder.
    Keeps prompt-side rendering pure formatting."""
    if not text or len(text) <= total:
        return text
    dropped = len(text) - head - tail
    return (
        text[:head]
        + f"\n[…{dropped} bytes truncated…]\n"
        + text[-tail:]
    )


def extract_recent_bash_actions(
    entries: list[dict[str, Any]],
    *,
    max_n: int = MAX_BASH_ACTIONS,
    max_cmd: int = MAX_BASH_COMMAND,
    max_output_total: int = MAX_BASH_OUTPUT_TOTAL,
    max_output_head: int = MAX_BASH_OUTPUT_HEAD,
    max_output_tail: int = MAX_BASH_OUTPUT_TAIL,
) -> list[tuple[int, str, Any, str, str]]:
    """Walk transcript chronologically, build per-action records of
    Bash tool calls + their tool_result outputs.

    Returns the last `max_n` entries as
    `[(action_index, command, exit_code, stdout, stderr), ...]`.

    Each output is head+tail truncated so a verbose log can't crowd
    out the diagnostic signal. The action_index matches the
    monotonically increasing tool_use counter used by
    `extract_file_touch_trajectory` so Haiku can correlate.
    """
    # First pass — collect all (action_index, command, tool_use_id).
    bash_calls: list[tuple[int, str, str]] = []
    action_index = 0
    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for name, tid, inp in _extract_tool_uses_with_ids(msg.get("content")):
            action_index += 1
            if name != "Bash":
                continue
            cmd = str(inp.get("command", "") or "")
            bash_calls.append((action_index, cmd, tid))

    if not bash_calls:
        return []

    # Second pass — collect tool_result blocks keyed by tool_use_id.
    results_by_id: dict[str, Any] = {}
    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for tid, payload in _extract_tool_results(msg.get("content")).items():
            results_by_id[tid] = payload

    # Build output records for the last N Bash calls.
    out: list[tuple[int, str, Any, str, str]] = []
    for idx, cmd, tid in bash_calls[-max_n:]:
        exit_code, stdout, stderr = _bash_result_to_strings(
            results_by_id.get(tid)
        )
        cmd_clipped = _clip(cmd, max_cmd)
        stdout_clipped = _head_tail_clip(
            stdout,
            total=max_output_total,
            head=max_output_head,
            tail=max_output_tail,
        )
        stderr_clipped = _head_tail_clip(
            stderr,
            total=max_output_total // 2,
            head=max_output_head // 2,
            tail=max_output_tail // 2,
        )
        out.append((idx, cmd_clipped, exit_code, stdout_clipped, stderr_clipped))
    return out


def extract_file_touch_trajectory(
    entries: list[dict[str, Any]],
) -> list[tuple[int, str, str]]:
    """Walk entries, return last MAX_TRAJECTORY (action_index, tool, path)
    tuples for Edit/Write/MultiEdit actions.

    Pure metadata. No file content, no heuristics — just "what files did
    the agent touch and when". Haiku can spot repeated-thrashing patterns
    from this without any pre-computed signal.
    """
    out: list[tuple[int, str, str]] = []
    action_index = 0
    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for name, inp in _extract_tool_uses(msg.get("content")):
            action_index += 1
            if name not in _FILE_TOUCH_TOOLS:
                continue
            fp = inp.get("file_path")
            if not isinstance(fp, str) or not fp:
                continue
            out.append((action_index, name, fp))
    return out[-MAX_TRAJECTORY:]


def _read_file_snapshot(
    file_path: str,
    *,
    cwd: str,
    max_bytes: int = MAX_SNAPSHOT_BYTES,
    max_head: int = MAX_SNAPSHOT_HEAD,
    max_tail: int = MAX_SNAPSHOT_TAIL,
    anchor: str | None = None,
    window_before: int = TARGET_WINDOW_BEFORE,
    window_after: int = TARGET_WINDOW_AFTER,
) -> str | None:
    """Read `file_path` from disk and return its content as a string.

    Returns None when:
      - path resolution fails,
      - resolved path escapes `cwd` (avoid leaking arbitrary fs),
      - file is missing / unreadable,
      - file is larger than MAX_READABLE_FILE_BYTES (5MB).

    Truncation policy when content exceeds `max_bytes`:
      - If `anchor` is provided AND found in content: slice an
        edit-region-centered window (`window_before` bytes before the
        anchor + `window_after` bytes after). This is the case for
        the CURRENT EDIT TARGET — we know where the change happened,
        we show around it. The head+tail dance loses the middle, which
        is exactly where newly-added code typically lives.
      - Otherwise: head + truncation marker + tail. Size-budget cut,
        no claim about meaning.

    Both modes emit an explicit `[…truncated…]` marker so Haiku knows
    bytes were dropped.
    """
    try:
        raw_path = Path(file_path)
        if not raw_path.is_absolute():
            if not cwd:
                return None
            raw_path = Path(cwd) / file_path
        resolved = raw_path.resolve(strict=False)
        if cwd:
            try:
                resolved.relative_to(Path(cwd).resolve(strict=False))
            except ValueError:
                # Path escapes cwd — refuse.
                return None
        if not resolved.is_file():
            return None
        try:
            size = resolved.stat().st_size
        except OSError:
            return None
        if size > MAX_READABLE_FILE_BYTES:
            return None
        with resolved.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return None
    except Exception:
        return None

    if len(content) <= max_bytes:
        return content

    # Edit-region-centered window: when the caller knows where the
    # action's edit landed (anchor = old_string for Edit/MultiEdit),
    # slice around that point. This preserves the area the agent
    # actually changed, including any nearby just-added definitions.
    if anchor:
        anchor_pos = content.find(anchor)
        if anchor_pos >= 0:
            start = max(0, anchor_pos - window_before)
            end = min(len(content), anchor_pos + len(anchor) + window_after)
            slice_text = content[start:end]
            head_dropped = start
            tail_dropped = len(content) - end
            parts: list[str] = []
            if head_dropped > 0:
                parts.append(
                    f"[…{head_dropped} bytes "
                    f"({content[:start].count(chr(10))} lines) before the edit window…]\n\n"
                )
            parts.append(slice_text)
            if tail_dropped > 0:
                parts.append(
                    f"\n\n[…{tail_dropped} bytes "
                    f"({content[end:].count(chr(10))} lines) after the edit window…]"
                )
            return "".join(parts)

    # Head/tail split with a clear marker line.
    head = content[:max_head]
    tail = content[-max_tail:]
    dropped = len(content) - max_head - max_tail
    head_lines = head.count("\n")
    tail_lines = tail.count("\n")
    dropped_lines = content.count("\n") - head_lines - tail_lines
    marker = (
        f"\n\n[…truncated middle of file: {dropped} bytes / "
        f"~{dropped_lines} lines dropped to fit prompt budget…]\n\n"
    )
    return head + marker + tail


def extract_per_file_snapshots(
    entries: list[dict[str, Any]],
    *,
    cwd: str,
    max_files: int = MAX_SNAPSHOT_FILES,
    max_bytes_per_file: int = MAX_SNAPSHOT_BYTES,
    target_file: str | None = None,
    target_anchor: str | None = None,
) -> dict[str, str]:
    """Read up to `max_files` files most relevant to the recent action
    stream, return `{file_path: snapshot_text}`.

    Files are picked from the Edit/Write/MultiEdit trajectory in the
    transcript, ordered by recency of last touch. This covers the
    vast majority of file mutations the agent makes; the rare case of
    Bash-mediated file rewrites (`> path`, `sed -i`, etc.) is
    deliberately not handled here — a filesystem-mtime sweep was
    considered and rejected as too expensive on large projects.

    When `target_file` is given, that file gets the larger
    MAX_TARGET_SNAPSHOT_BYTES budget. If `target_anchor` (typically the
    Edit's `old_string`) is also provided AND the target exceeds budget,
    the snapshot is sliced as a window centered on the anchor — the
    edit region — rather than head+tail. This prevents the common case
    where a freshly-added utility function lives in the middle of a
    growing file and gets dropped by head+tail truncation.

    Each snapshot is the file's CURRENT on-disk content (post any
    edits already applied). Reading is best-effort — failures, paths
    outside `cwd`, and oversize files are silently skipped.
    """
    last_touch_idx: dict[str, int] = {}
    action_index = 0
    for entry in entries:
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for name, inp in _extract_tool_uses(msg.get("content")):
            action_index += 1
            if name not in _FILE_TOUCH_TOOLS:
                continue
            fp = inp.get("file_path")
            if isinstance(fp, str) and fp:
                last_touch_idx[fp] = action_index

    if not last_touch_idx:
        return {}

    chosen = sorted(last_touch_idx, key=lambda f: -last_touch_idx[f])[:max_files]
    out: dict[str, str] = {}
    for fp in chosen:
        if fp == target_file:
            snap = _read_file_snapshot(
                fp,
                cwd=cwd,
                max_bytes=MAX_TARGET_SNAPSHOT_BYTES,
                anchor=target_anchor,
            )
        else:
            snap = _read_file_snapshot(fp, cwd=cwd, max_bytes=max_bytes_per_file)
        if snap is not None:
            out[fp] = snap
    return out
