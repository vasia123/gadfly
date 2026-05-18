"""Project-level semantic memory — the historian's data model.

Per-cwd corpus living under `~/.claude/gadfly/project/<cwd-encoded>/`.
Keeps three kinds of consolidated findings (semantic memory), each
backed by raw episodic digests at `raw/<sid>.json`:

  Promise      — agent said "I'll do X later"; tracks landing.
  Correction   — durable user feedback / preference. Needs ≥2 sessions
                 before graduating from quarantine.
  Subsystem    — knowledge-graph node: a group of files with a shared
                 purpose. Needs ≥3 file mentions across sessions before
                 graduating.

Design discipline (from MemoryArena / MemMachine / MemoryGraft literature):

  * Provenance on every entry — `source_session`, `source_action_index`,
    `evidence_quote` ≤200c verbatim. Anything without provenance is
    rejected at extraction time.
  * Ground-truth preservation — the consolidated state is fully
    re-derivable from `raw/<sid>.json`. The aggregator NEVER reads its
    own previous output. `rebuild()` proves this: it discards
    semantic state and reconstructs from episodic alone, byte-for-byte
    identical (modulo `ts`) on every run.
  * Quarantine → promotion — new findings land in `quarantine` and only
    graduate when their support threshold is met. Defends against
    single-shot poisoning.
  * Revocation, not deletion — revoked entries move to an audit list;
    raw evidence is never deleted.

This module is pure plumbing. The Haiku call that extracts findings
from a session transcript lives in `historian.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal


SCHEMA_VERSION = 1

# Promotion thresholds.
CORRECTION_PROMOTION_SESSIONS = 2
SUBSYSTEM_PROMOTION_FILE_MENTIONS = 3

# Age-out: how long an open promise stays "fresh" before the aggregator
# transitions it to status="aged-out". Tuned for typical week-long
# workstreams. Manual override: edit raw/<sid>.json or revoke via CLI.
PROMISE_AGE_OUT_DAYS = 7
PROMISE_AGE_OUT_S = PROMISE_AGE_OUT_DAYS * 24 * 3600

# Auto-fulfillment heuristic — completion verbs.
# Conservative list: words that strongly indicate a task was COMPLETED
# (not just discussed, investigated, planned). Past-tense / participle
# forms preferred. Tweak if false-positives prove common.
_COMPLETION_VERBS = {
    "done", "completed", "implemented", "landed", "shipped", "fixed",
    "resolved", "merged", "deployed", "added", "wired",
}
# Minimum distinct promise-nouns that must appear in a later digest's
# text for auto-fulfillment to fire. ≥2 is conservative; 1 alone is
# too generic (any session mentioning the topic would count).
_AUTOFULFILL_MIN_NOUNS = 2

# Caps to keep entries bounded.
MAX_EVIDENCE_QUOTE = 200
MAX_NOTES = 1500


# --- cwd encoding -----------------------------------------------------------


def encode_cwd(cwd: str) -> str:
    """Mirror Claude Code's encoding of cwd into the directory name it
    uses under ~/.claude/projects/.

    Observed rule from existing dirs: '/' and '_' both become '-'.
    Native dashes pass through. Case is preserved.

      /home/vasis/projects_hobby/gadfly  →  -home-vasis-projects-hobby-gadfly
      /home/vasis/projects_hobby/8mart-games
                                        →  -home-vasis-projects-hobby-8mart-games

    Defensive: also normalise any other non-[A-Za-z0-9-] runs to a
    single '-' so weird paths don't crash the layout. We've never seen
    Claude Code break this rule but it's cheap insurance.
    """
    if not isinstance(cwd, str) or not cwd:
        return "unknown"
    s = cwd.replace("/", "-").replace("_", "-")
    # Collapse any unexpected garbage to a single dash.
    s = re.sub(r"[^A-Za-z0-9-]", "-", s)
    # Collapse runs of dashes to a single dash (Claude Code doesn't, but
    # we'd hit it on edge cases like `//path` — match common sense).
    s = re.sub(r"-{2,}", "-", s)
    return s


# --- dataclasses ------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where a finding came from. Required on every entry."""

    source_session: str
    source_action_index: int
    evidence_quote: str       # ≤200c verbatim
    first_seen_ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Provenance":
        return cls(
            source_session=str(d.get("source_session") or ""),
            source_action_index=int(d.get("source_action_index") or 0),
            evidence_quote=str(d.get("evidence_quote") or "")[:MAX_EVIDENCE_QUOTE],
            first_seen_ts=float(d.get("first_seen_ts") or 0.0),
        )


PromiseStatus = Literal["open", "fulfilled", "aged-out"]


@dataclass
class Promise:
    id: str
    title: str
    provenance: Provenance
    status: PromiseStatus = "open"
    fulfilled_in_session: str | None = None
    last_seen_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "fulfilled_in_session": self.fulfilled_in_session,
            "last_seen_ts": self.last_seen_ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Promise":
        st = d.get("status")
        if st not in ("open", "fulfilled", "aged-out"):
            st = "open"
        return cls(
            id=str(d.get("id") or ""),
            title=str(d.get("title") or ""),
            provenance=Provenance.from_dict(d.get("provenance") or {}),
            status=st,  # type: ignore[arg-type]
            fulfilled_in_session=(
                str(d["fulfilled_in_session"])
                if d.get("fulfilled_in_session")
                else None
            ),
            last_seen_ts=float(d.get("last_seen_ts") or 0.0),
        )


@dataclass
class Correction:
    id: str
    rule: str
    why: str
    how_to_apply: str
    provenance: Provenance      # first occurrence
    seen_in_sessions: list[str] = field(default_factory=list)
    usefulness_score: int = 0

    @property
    def confidence(self) -> int:
        # Bounded 1..3 — semantic memory literature pattern.
        return max(1, min(3, len(self.seen_in_sessions) or 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule": self.rule,
            "why": self.why,
            "how_to_apply": self.how_to_apply,
            "provenance": self.provenance.to_dict(),
            "seen_in_sessions": list(self.seen_in_sessions),
            "usefulness_score": self.usefulness_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Correction":
        return cls(
            id=str(d.get("id") or ""),
            rule=str(d.get("rule") or ""),
            why=str(d.get("why") or ""),
            how_to_apply=str(d.get("how_to_apply") or ""),
            provenance=Provenance.from_dict(d.get("provenance") or {}),
            seen_in_sessions=[
                str(s) for s in (d.get("seen_in_sessions") or []) if isinstance(s, str)
            ],
            usefulness_score=int(d.get("usefulness_score") or 0),
        )


@dataclass
class Subsystem:
    id: str
    title: str
    purpose: str
    files: list[str]                   # repo-relative paths, sorted+deduped
    provenance: Provenance             # first occurrence
    last_touched_session: str = ""
    last_touched_ts: float = 0.0
    usefulness_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "purpose": self.purpose,
            "files": list(self.files),
            "provenance": self.provenance.to_dict(),
            "last_touched_session": self.last_touched_session,
            "last_touched_ts": self.last_touched_ts,
            "usefulness_score": self.usefulness_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Subsystem":
        return cls(
            id=str(d.get("id") or ""),
            title=str(d.get("title") or ""),
            purpose=str(d.get("purpose") or ""),
            files=sorted({str(f) for f in (d.get("files") or []) if isinstance(f, str)}),
            provenance=Provenance.from_dict(d.get("provenance") or {}),
            last_touched_session=str(d.get("last_touched_session") or ""),
            last_touched_ts=float(d.get("last_touched_ts") or 0.0),
            usefulness_score=int(d.get("usefulness_score") or 0),
        )


QuarantineKind = Literal["promise", "correction", "subsystem"]


@dataclass
class QuarantineItem:
    """A finding extracted but not yet meeting its promotion threshold."""

    kind: QuarantineKind
    payload: dict[str, Any]         # raw extracted dict (will become a Promise/etc.)
    provenance: Provenance
    seen_in_sessions: list[str] = field(default_factory=list)
    file_mentions: list[str] = field(default_factory=list)   # subsystems only
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "provenance": self.provenance.to_dict(),
            "seen_in_sessions": list(self.seen_in_sessions),
            "file_mentions": list(self.file_mentions),
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuarantineItem":
        k = d.get("kind")
        if k not in ("promise", "correction", "subsystem"):
            k = "correction"
        return cls(
            kind=k,  # type: ignore[arg-type]
            payload=dict(d.get("payload") or {}),
            provenance=Provenance.from_dict(d.get("provenance") or {}),
            seen_in_sessions=[
                str(s) for s in (d.get("seen_in_sessions") or []) if isinstance(s, str)
            ],
            file_mentions=[
                str(f) for f in (d.get("file_mentions") or []) if isinstance(f, str)
            ],
            ts=float(d.get("ts") or 0.0),
        )


@dataclass
class VerdictPattern:
    """Aggregate value-tracker for a recurring verdict pattern.

    Watchdog's analog of priors' `usefulness_score`. Without it, the
    watchdog flags the same pattern forever — even when historical
    evidence says the agent's behavior was correct each time. This is
    the MemoryArena lesson applied to the watchdog itself: storage of
    flags is cheap, USE of them for self-calibration is the hard part.

    A pattern is keyed by `fingerprint`: a deterministic kebab string
    derived from (marker, normalised noun-keywords of the reason).
    Two flags with reason "Symptom fix: composable does not exist yet"
    and "Symptom fix: composable not yet defined" should hash to the
    same fingerprint so we accumulate across phrasings.

    The signed `value_score` captures net judgment-quality across
    historical resolutions:
      +1 when a workstream that carried this flag closed `done` AND
         the agent visibly complied with the suggestion.
      -1 when a workstream closed `done` AND the agent did NOT comply
         (the flag was correctly ignored — likely a false positive).
       0 (no change) when the workstream was `abandoned` or evidence
         is ambiguous (don't penalise nor reward).
    Bounded to [-10, +10] so a long-stale pattern can still recover.
    """

    fingerprint: str          # deterministic key (see derive_fingerprint)
    marker: str               # symptom | rationalization | other
    sample_reason: str        # one canonical phrasing for human review
    total_flags: int = 0      # how many times this pattern fired
    value_score: int = 0      # bounded [-10, +10]
    last_seen_session: str = ""
    last_seen_ts: float = 0.0
    sample_workstream_ids: list[str] = field(default_factory=list)  # up to 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "marker": self.marker,
            "sample_reason": self.sample_reason,
            "total_flags": self.total_flags,
            "value_score": self.value_score,
            "last_seen_session": self.last_seen_session,
            "last_seen_ts": self.last_seen_ts,
            "sample_workstream_ids": list(self.sample_workstream_ids)[-5:],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerdictPattern":
        return cls(
            fingerprint=str(d.get("fingerprint") or ""),
            marker=str(d.get("marker") or "other"),
            sample_reason=str(d.get("sample_reason") or "")[:280],
            total_flags=int(d.get("total_flags") or 0),
            value_score=max(-10, min(10, int(d.get("value_score") or 0))),
            last_seen_session=str(d.get("last_seen_session") or ""),
            last_seen_ts=float(d.get("last_seen_ts") or 0.0),
            sample_workstream_ids=[
                str(s) for s in (d.get("sample_workstream_ids") or [])
                if isinstance(s, str)
            ],
        )


_REASON_STOP = {
    # English
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with",
    "is", "are", "was", "were", "be", "been", "by", "as", "at", "from",
    "this", "that", "these", "those", "it", "its", "without", "but", "not",
    "than", "then", "when", "where", "why", "how", "what", "which",
    "agent", "code", "fix", "edit", "file", "yet", "should", "would",
    "could", "have", "has", "had", "will", "may", "might", "can",
}


def derive_pattern_fingerprint(marker: str, reason: str) -> str:
    """Deterministic kebab-slug for grouping similar flags.

    The hard part: phrasings that share a TOPIC but use different
    verbs need to collapse:

      'Symptom fix: composable does not exist yet'
      'Symptom fix: useMail composable not yet defined'
      'Symptom fix: composable methods are not implemented'
                                  ↓
      symptom-composable  (single longest content noun)

    Strategy:
      1. Strip marker prefix ('Symptom fix:', 'Rationalization:') so the
         marker word doesn't appear in the topic token.
      2. Keep tokens ≥6 chars (content-y, not generic helpers).
      3. Filter stop words.
      4. Take the SINGLE longest token. Aggressive but lossy-in-the-
         right-direction: better to over-group similar flags into one
         pattern than to never accumulate evidence.

    Drawback: distinct topics that happen to share a long word collide
    (e.g. flags about 'authentication' and flags about 'authorization'
    both → 'auth...'). Acceptable trade in v1; revisit if false-merge
    rate matters in real corpora.
    """
    marker_norm = (marker or "other").lower().split(":", 1)[0].strip()
    if marker_norm not in ("symptom", "rationalization", "other"):
        marker_norm = "other"

    # Strip "Symptom fix:" / "Rationalization:" prefix so the marker
    # word doesn't end up as a content token.
    body = reason or ""
    m = re.match(r"\s*(symptom\s+fix|rationalization)\s*[:\.]\s*", body, re.IGNORECASE)
    if m:
        body = body[m.end():]

    candidates: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[^A-Za-z0-9_]+", body.lower()):
        if len(raw) < 6 or raw in _REASON_STOP or raw in seen:
            continue
        seen.add(raw)
        candidates.append(raw)

    if not candidates:
        return f"{marker_norm}-empty"

    # Pick the longest. Ties broken alphabetically for determinism.
    candidates.sort(key=lambda t: (-len(t), t))
    return f"{marker_norm}-{candidates[0]}"


@dataclass
class RevokedEntry:
    """Audit trail of revoked findings. Raw evidence is preserved
    elsewhere; this records why something was excluded."""

    kind: QuarantineKind
    id: str
    title: str
    reason: str
    ts: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RevokedEntry":
        return cls(
            kind=d.get("kind") or "correction",  # type: ignore[arg-type]
            id=str(d.get("id") or ""),
            title=str(d.get("title") or ""),
            reason=str(d.get("reason") or ""),
            ts=float(d.get("ts") or 0.0),
        )


@dataclass
class ProjectState:
    cwd: str
    promises: dict[str, Promise] = field(default_factory=dict)
    corrections: dict[str, Correction] = field(default_factory=dict)
    subsystems: dict[str, Subsystem] = field(default_factory=dict)
    quarantine: list[QuarantineItem] = field(default_factory=list)
    digested_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    revoked: list[RevokedEntry] = field(default_factory=list)
    # E2: watchdog self-learning. Recurring flag patterns keyed by
    # fingerprint, with a signed value_score (-10..+10). Surfaced to
    # the verdict prompt so the watchdog can recognise patterns it has
    # historically been wrong about and stay silent.
    verdict_patterns: dict[str, VerdictPattern] = field(default_factory=dict)
    prompt_sha: str = ""
    schema_version: int = SCHEMA_VERSION
    ts: float = 0.0

    # ---- I/O ---------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "promises": {k: v.to_dict() for k, v in self.promises.items()},
            "corrections": {k: v.to_dict() for k, v in self.corrections.items()},
            "subsystems": {k: v.to_dict() for k, v in self.subsystems.items()},
            "quarantine": [q.to_dict() for q in self.quarantine],
            "digested_sessions": dict(self.digested_sessions),
            "revoked": [r.to_dict() for r in self.revoked],
            "verdict_patterns": {k: v.to_dict() for k, v in self.verdict_patterns.items()},
            "prompt_sha": self.prompt_sha,
            "schema_version": self.schema_version,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectState":
        promises = {
            k: Promise.from_dict(v)
            for k, v in (d.get("promises") or {}).items()
            if isinstance(v, dict)
        }
        corrections = {
            k: Correction.from_dict(v)
            for k, v in (d.get("corrections") or {}).items()
            if isinstance(v, dict)
        }
        subsystems = {
            k: Subsystem.from_dict(v)
            for k, v in (d.get("subsystems") or {}).items()
            if isinstance(v, dict)
        }
        quarantine = [
            QuarantineItem.from_dict(q)
            for q in (d.get("quarantine") or [])
            if isinstance(q, dict)
        ]
        revoked = [
            RevokedEntry.from_dict(r)
            for r in (d.get("revoked") or [])
            if isinstance(r, dict)
        ]
        verdict_patterns = {
            k: VerdictPattern.from_dict(v)
            for k, v in (d.get("verdict_patterns") or {}).items()
            if isinstance(v, dict)
        }
        return cls(
            cwd=str(d.get("cwd") or ""),
            promises=promises,
            corrections=corrections,
            subsystems=subsystems,
            quarantine=quarantine,
            digested_sessions=dict(d.get("digested_sessions") or {}),
            revoked=revoked,
            verdict_patterns=verdict_patterns,
            prompt_sha=str(d.get("prompt_sha") or ""),
            schema_version=int(d.get("schema_version") or SCHEMA_VERSION),
            ts=float(d.get("ts") or 0.0),
        )

    def is_revoked(self, kind: QuarantineKind, finding_id: str) -> bool:
        return any(r.kind == kind and r.id == finding_id for r in self.revoked)


# --- Persistence ------------------------------------------------------------


def project_root() -> Path:
    """Project-corpus root. Honors GADFLY_LOG_DIR so tests can isolate."""
    base = os.environ.get("GADFLY_LOG_DIR")
    if base:
        return Path(base).parent / "project"
    return Path.home() / ".claude" / "gadfly" / "project"


def project_dir(cwd: str) -> Path:
    return project_root() / encode_cwd(cwd)


def state_path(cwd: str) -> Path:
    return project_dir(cwd) / "state.json"


def raw_dir(cwd: str) -> Path:
    return project_dir(cwd) / "raw"


def failed_dir(cwd: str) -> Path:
    """Per-session failure markers. distill errors land here so the
    daemon can retry a bounded number of times before giving up."""
    return project_dir(cwd) / "failed"


MAX_DISTILL_ATTEMPTS = 3


def record_distill_failure(cwd: str, session_id: str, error: str) -> int:
    """Increment the attempt counter for a failed session. Returns the
    new attempt count. Atomic enough for our single-daemon setup."""
    try:
        d = failed_dir(cwd)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{_safe_sid(session_id)}.json"
        attempts = 1
        prior: dict[str, Any] = {}
        if path.is_file():
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
                attempts = int(prior.get("attempts") or 0) + 1
            except Exception:
                prior = {}
        record = {
            "attempts": attempts,
            "last_error": error,
            "last_ts": time.time(),
            "first_ts": prior.get("first_ts") or time.time(),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8")
        return attempts
    except Exception:
        return 1


def clear_distill_failure(cwd: str, session_id: str) -> None:
    """Remove the failure marker — call on successful retry."""
    try:
        path = failed_dir(cwd) / f"{_safe_sid(session_id)}.json"
        if path.is_file():
            path.unlink()
    except Exception:
        pass


def get_distill_attempts(cwd: str, session_id: str) -> int:
    """Return the recorded attempt count for a session (0 if no marker)."""
    try:
        path = failed_dir(cwd) / f"{_safe_sid(session_id)}.json"
        if not path.is_file():
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("attempts") or 0)
    except Exception:
        return 0


def session_exhausted(cwd: str, session_id: str) -> bool:
    """True when the session has failed MAX_DISTILL_ATTEMPTS times.
    The daemon stops retrying these. Manual cleanup: delete the file
    under <project>/failed/<sid>.json to reset."""
    return get_distill_attempts(cwd, session_id) >= MAX_DISTILL_ATTEMPTS


def load_state(cwd: str, *, expected_prompt_sha: str | None = None) -> ProjectState:
    """Load state for `cwd`. Returns fresh empty state when:
    - file missing
    - schema_version mismatch
    - prompt_sha mismatch (when expected_prompt_sha is given) — drift defence.
    """
    p = state_path(cwd)
    if not p.is_file():
        return ProjectState(cwd=cwd, prompt_sha=expected_prompt_sha or "")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ProjectState(cwd=cwd, prompt_sha=expected_prompt_sha or "")
    if not isinstance(data, dict):
        return ProjectState(cwd=cwd, prompt_sha=expected_prompt_sha or "")
    state = ProjectState.from_dict(data)
    if state.schema_version != SCHEMA_VERSION:
        return ProjectState(cwd=cwd, prompt_sha=expected_prompt_sha or "")
    if expected_prompt_sha and state.prompt_sha and state.prompt_sha != expected_prompt_sha:
        return ProjectState(cwd=cwd, prompt_sha=expected_prompt_sha)
    # Make sure cwd is set even on older files that may have dropped it.
    if not state.cwd:
        state.cwd = cwd
    return state


def save_state(state: ProjectState) -> None:
    """Atomic write to state.json. Never raises."""
    try:
        d = project_dir(state.cwd)
        d.mkdir(parents=True, exist_ok=True)
        tmp = state_path(state.cwd).with_suffix(".tmp")
        # Sort keys so on-disk JSON is stable across runs — important
        # for the byte-identical rebuild assertion.
        tmp.write_text(
            json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, state_path(state.cwd))
    except Exception:
        pass


def write_raw_digest(cwd: str, session_id: str, digest: dict[str, Any]) -> None:
    """Write a per-session raw digest. Episodic memory — preserved
    verbatim; never overwritten on the same session-id once written."""
    try:
        d = raw_dir(cwd)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{_safe_sid(session_id)}.json"
        if path.exists():
            return  # never overwrite
        path.write_text(
            json.dumps(digest, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass


def read_raw_digest(cwd: str, session_id: str) -> dict[str, Any] | None:
    try:
        path = raw_dir(cwd) / f"{_safe_sid(session_id)}.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_raw_digests(cwd: str) -> list[str]:
    d = raw_dir(cwd)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def _safe_sid(sid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", sid) or "unknown"


# --- Slug / id derivation ---------------------------------------------------


def slugify(text: str, *, max_len: int = 40) -> str:
    """Lowercase kebab-case slug. Deterministic — used inside ids so the
    aggregator can be re-run idempotently."""
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").lower()).strip("-")
    if not s:
        s = "untitled"
    return s[:max_len]


def derive_id(*parts: str) -> str:
    """Combine slug parts with a short hash suffix so collisions are rare
    but the id is still human-readable."""
    base = "-".join(p for p in (slugify(x) for x in parts) if p)
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:6]
    return f"{base}-{h}" if base else h


# --- Deterministic re-aggregator -------------------------------------------


def aggregate(
    raw_digests: Iterable[dict[str, Any]],
    *,
    cwd: str,
    prompt_sha: str = "",
    revoked: list[RevokedEntry] | None = None,
    prior_usefulness: dict[tuple[QuarantineKind, str], int] | None = None,
) -> ProjectState:
    """Build a ProjectState from raw per-session digests.

    Pure function — no Haiku calls, no I/O. The whole point: rebuild can
    re-run this against existing raw/ files and reach byte-identical
    output (modulo the `ts` field which we set to the max of input
    first_seen_ts values so it's still deterministic).

    Each input digest is the JSON shape returned by historian's
    extract_findings, augmented with the session_id, action_index_max,
    and ts fields written into raw/.

    The aggregator:
      - merges promises (last-write-wins on status; ADD-only otherwise),
      - merges corrections (accumulating seen_in_sessions, promoting
        when threshold met),
      - merges subsystems (union of files, last_touched_session wins by ts,
        promoting when file-mentions threshold met),
      - applies revocations from `revoked` (excluded from output),
      - carries `usefulness_score` forward via `prior_usefulness`.
    """
    revoked = list(revoked or [])
    prior_use = dict(prior_usefulness or {})

    state = ProjectState(cwd=cwd, prompt_sha=prompt_sha, schema_version=SCHEMA_VERSION)

    # Working tally for quarantine eligibility BEFORE promotion.
    pending_corrections: dict[str, QuarantineItem] = {}
    pending_subsystems: dict[str, QuarantineItem] = {}

    digests = list(raw_digests)
    # Sort so iteration order is deterministic regardless of fs listing order.
    digests.sort(key=lambda d: (d.get("ts") or 0.0, d.get("session_id") or ""))

    max_ts = 0.0

    for digest in digests:
        sid = str(digest.get("session_id") or "")
        ts = float(digest.get("ts") or 0.0)
        max_ts = max(max_ts, ts)
        state.digested_sessions[sid] = {
            "mtime": digest.get("mtime"),
            "sha": digest.get("sha"),
            "ts": ts,
        }

        # --- promises ----------------------------------------------------
        for p in digest.get("new_promises") or []:
            if not isinstance(p, dict):
                continue
            title = str(p.get("title") or "").strip()
            quote = str(p.get("evidence_quote") or "")[:MAX_EVIDENCE_QUOTE]
            ai = int(p.get("action_index") or 0)
            if not title or not quote:
                continue
            pid = derive_id("promise", title)
            if any(r.kind == "promise" and r.id == pid for r in revoked):
                continue
            existing = state.promises.get(pid)
            if existing is None:
                prov = Provenance(
                    source_session=sid,
                    source_action_index=ai,
                    evidence_quote=quote,
                    first_seen_ts=ts,
                )
                state.promises[pid] = Promise(
                    id=pid,
                    title=title,
                    provenance=prov,
                    status="open",
                    last_seen_ts=ts,
                )
            else:
                existing.last_seen_ts = max(existing.last_seen_ts, ts)

        for fulfilled in digest.get("fulfilled_promises") or []:
            if not isinstance(fulfilled, dict):
                continue
            hint = str(fulfilled.get("id_hint") or "").strip()
            if not hint:
                continue
            pid = derive_id("promise", hint)
            existing = state.promises.get(pid)
            if existing and existing.status == "open":
                existing.status = "fulfilled"
                existing.fulfilled_in_session = sid
                existing.last_seen_ts = max(existing.last_seen_ts, ts)

        # --- corrections -------------------------------------------------
        for c in digest.get("new_corrections") or []:
            if not isinstance(c, dict):
                continue
            rule = str(c.get("rule") or "").strip()
            why = str(c.get("why") or "").strip()
            how = str(c.get("how_to_apply") or "").strip()
            quote = str(c.get("evidence_quote") or "")[:MAX_EVIDENCE_QUOTE]
            if not rule or not quote:
                continue
            cid = derive_id("correction", rule)
            if any(r.kind == "correction" and r.id == cid for r in revoked):
                continue
            # Already promoted? Just bump seen_in_sessions.
            if cid in state.corrections:
                corr = state.corrections[cid]
                if sid not in corr.seen_in_sessions:
                    corr.seen_in_sessions.append(sid)
                continue
            # Otherwise accumulate in pending.
            if cid in pending_corrections:
                pending_corrections[cid].seen_in_sessions.append(sid)
            else:
                pending_corrections[cid] = QuarantineItem(
                    kind="correction",
                    payload={
                        "id": cid,
                        "rule": rule,
                        "why": why,
                        "how_to_apply": how,
                    },
                    provenance=Provenance(
                        source_session=sid,
                        source_action_index=int(c.get("action_index") or 0),
                        evidence_quote=quote,
                        first_seen_ts=ts,
                    ),
                    seen_in_sessions=[sid],
                    ts=ts,
                )

        # --- subsystems --------------------------------------------------
        for k in digest.get("knowledge_updates") or []:
            if not isinstance(k, dict):
                continue
            sub_id_hint = str(k.get("subsystem_id") or "").strip() or slugify(
                str(k.get("title") or "")
            )
            title = str(k.get("title") or "").strip()
            purpose = str(k.get("purpose") or "").strip()
            files = sorted(
                {str(f) for f in (k.get("files") or []) if isinstance(f, str)}
            )
            quote = str(k.get("evidence_quote") or "")[:MAX_EVIDENCE_QUOTE]
            if not title or not files or not quote:
                continue
            sid_canonical = derive_id("subsystem", sub_id_hint, title)
            if any(r.kind == "subsystem" and r.id == sid_canonical for r in revoked):
                continue
            # Already promoted? Merge files / bump last_touched.
            if sid_canonical in state.subsystems:
                sub = state.subsystems[sid_canonical]
                sub.files = sorted(set(sub.files) | set(files))
                if ts >= sub.last_touched_ts:
                    sub.last_touched_session = sid
                    sub.last_touched_ts = ts
                if purpose and not sub.purpose:
                    sub.purpose = purpose
                continue
            if sid_canonical in pending_subsystems:
                pq = pending_subsystems[sid_canonical]
                pq.file_mentions = sorted(set(pq.file_mentions) | set(files))
                if sid not in pq.seen_in_sessions:
                    pq.seen_in_sessions.append(sid)
                pq.payload["files"] = pq.file_mentions
                if ts >= pq.payload.get("last_touched_ts", 0.0):
                    pq.payload["last_touched_session"] = sid
                    pq.payload["last_touched_ts"] = ts
            else:
                pending_subsystems[sid_canonical] = QuarantineItem(
                    kind="subsystem",
                    payload={
                        "id": sid_canonical,
                        "title": title,
                        "purpose": purpose,
                        "files": files,
                        "last_touched_session": sid,
                        "last_touched_ts": ts,
                    },
                    provenance=Provenance(
                        source_session=sid,
                        source_action_index=int(k.get("action_index") or 0),
                        evidence_quote=quote,
                        first_seen_ts=ts,
                    ),
                    seen_in_sessions=[sid],
                    file_mentions=files,
                    ts=ts,
                )

    # --- promotion pass ------------------------------------------------------
    for cid, q in pending_corrections.items():
        unique_sessions = sorted(set(q.seen_in_sessions))
        if len(unique_sessions) >= CORRECTION_PROMOTION_SESSIONS:
            payload = q.payload
            corr = Correction(
                id=cid,
                rule=str(payload.get("rule") or ""),
                why=str(payload.get("why") or ""),
                how_to_apply=str(payload.get("how_to_apply") or ""),
                provenance=q.provenance,
                seen_in_sessions=unique_sessions,
                usefulness_score=prior_use.get(("correction", cid), 0),
            )
            state.corrections[cid] = corr
        else:
            q.seen_in_sessions = unique_sessions
            state.quarantine.append(q)

    for sid_canonical, q in pending_subsystems.items():
        if len(q.file_mentions) >= SUBSYSTEM_PROMOTION_FILE_MENTIONS:
            payload = q.payload
            sub = Subsystem(
                id=sid_canonical,
                title=str(payload.get("title") or ""),
                purpose=str(payload.get("purpose") or ""),
                files=sorted(q.file_mentions),
                provenance=q.provenance,
                last_touched_session=str(payload.get("last_touched_session") or ""),
                last_touched_ts=float(payload.get("last_touched_ts") or 0.0),
                usefulness_score=prior_use.get(("subsystem", sid_canonical), 0),
            )
            state.subsystems[sid_canonical] = sub
        else:
            q.seen_in_sessions = sorted(set(q.seen_in_sessions))
            state.quarantine.append(q)

    # Stable order: sort quarantine list by (kind, payload-id) so output
    # is byte-identical across runs of the same inputs.
    state.quarantine.sort(key=lambda q: (q.kind, str(q.payload.get("id") or "")))
    state.revoked = list(revoked)
    state.ts = max_ts

    # Cross-session fulfillment: each digest carries
    # `fulfilled_promise_ids` — verbatim ids of open promises that
    # Haiku judged as completed during that session. Pure deterministic
    # application here keeps rebuild Haiku-free.
    _apply_cross_session_fulfillments(state, digests)

    # Age-out pass: promises whose last_seen_ts is older than the
    # latest digest by > PROMISE_AGE_OUT_S transition to aged-out.
    # Helps in mature corpora; doesn't fire on projects < 7 days old.
    # Runs AFTER fulfillment so a stale promise that just got fulfilled
    # is correctly labelled "fulfilled", not "aged-out".
    if max_ts > 0:
        cutoff = max_ts - PROMISE_AGE_OUT_S
        for p in state.promises.values():
            if p.status == "open" and p.last_seen_ts and p.last_seen_ts < cutoff:
                p.status = "aged-out"
    return state


def _apply_cross_session_fulfillments(
    state: ProjectState,
    digests: list[dict[str, Any]],
) -> None:
    """Apply per-digest `fulfilled_promise_ids` to the promise store.

    Iterates digests in chronological order (the caller already sorts).
    For each promise id listed, if the promise exists and is currently
    open, transition it to fulfilled and record the session that closed
    it. Unknown ids and already-non-open promises are silently skipped —
    Haiku may have returned a verbatim id from a candidate list whose
    promise was later revoked, or two sessions may both claim to have
    completed the same promise (first writer wins).
    """
    for digest in digests:
        sid = str(digest.get("session_id") or "")
        ts = float(digest.get("ts") or 0.0)
        ids = digest.get("fulfilled_promise_ids") or []
        if not isinstance(ids, list):
            continue
        for pid in ids:
            if not isinstance(pid, str) or not pid:
                continue
            p = state.promises.get(pid)
            if p is None or p.status != "open":
                continue
            p.status = "fulfilled"
            p.fulfilled_in_session = sid
            p.last_seen_ts = max(p.last_seen_ts, ts)


# --- Promote / revoke actions ----------------------------------------------


def find_finding(
    state: ProjectState, finding_id: str
) -> tuple[QuarantineKind, str, dict[str, Any]] | None:
    """Find a finding by id across active stores AND quarantine. Returns
    (kind, location, payload_dict) where location is 'active' or
    'quarantine'. payload is a dict view useful for diagnostics."""
    if finding_id in state.promises:
        return ("promise", "active", state.promises[finding_id].to_dict())
    if finding_id in state.corrections:
        return ("correction", "active", state.corrections[finding_id].to_dict())
    if finding_id in state.subsystems:
        return ("subsystem", "active", state.subsystems[finding_id].to_dict())
    for q in state.quarantine:
        if str(q.payload.get("id")) == finding_id:
            return (q.kind, "quarantine", q.payload)
    return None


def promote_finding(cwd: str, finding_id: str) -> tuple[bool, str]:
    """Force-promote a quarantined finding to the active store.

    Manual override for the repetition-threshold gate. Use sparingly —
    the threshold exists to filter single-shot noise. Returns
    (success, message).
    """
    state = load_state(cwd)
    for i, q in enumerate(state.quarantine):
        if str(q.payload.get("id")) != finding_id:
            continue
        if q.kind == "correction":
            corr = Correction(
                id=finding_id,
                rule=str(q.payload.get("rule") or ""),
                why=str(q.payload.get("why") or ""),
                how_to_apply=str(q.payload.get("how_to_apply") or ""),
                provenance=q.provenance,
                seen_in_sessions=list(q.seen_in_sessions),
            )
            state.corrections[finding_id] = corr
        elif q.kind == "subsystem":
            sub = Subsystem(
                id=finding_id,
                title=str(q.payload.get("title") or ""),
                purpose=str(q.payload.get("purpose") or ""),
                files=sorted(q.file_mentions),
                provenance=q.provenance,
                last_touched_session=str(q.payload.get("last_touched_session") or ""),
                last_touched_ts=float(q.payload.get("last_touched_ts") or 0.0),
            )
            state.subsystems[finding_id] = sub
        elif q.kind == "promise":
            # Promises don't live in quarantine in current design;
            # included for completeness.
            prom = Promise(
                id=finding_id,
                title=str(q.payload.get("title") or ""),
                provenance=q.provenance,
                status="open",
                last_seen_ts=q.ts,
            )
            state.promises[finding_id] = prom
        else:
            return (False, f"unknown kind {q.kind!r}")
        # Remove from quarantine.
        state.quarantine.pop(i)
        save_state(state)
        return (True, f"promoted {q.kind} {finding_id}")
    return (False, f"id {finding_id!r} not found in quarantine")


def revoke_finding(cwd: str, finding_id: str, *, reason: str = "") -> tuple[bool, str]:
    """Move a finding (active or quarantined) to the revoked audit list.

    Revocation removes it from retrieval permanently — and from any
    future aggregator re-run (rebuild honors `revoked` and excludes
    those ids). Raw evidence is untouched.
    """
    state = load_state(cwd)
    kind: QuarantineKind | None = None
    title = ""

    if finding_id in state.promises:
        kind = "promise"
        title = state.promises[finding_id].title
        state.promises.pop(finding_id)
    elif finding_id in state.corrections:
        kind = "correction"
        title = state.corrections[finding_id].rule
        state.corrections.pop(finding_id)
    elif finding_id in state.subsystems:
        kind = "subsystem"
        title = state.subsystems[finding_id].title
        state.subsystems.pop(finding_id)
    else:
        for i, q in enumerate(state.quarantine):
            if str(q.payload.get("id")) == finding_id:
                kind = q.kind
                title = str(q.payload.get("title") or q.payload.get("rule") or "")
                state.quarantine.pop(i)
                break

    if kind is None:
        return (False, f"id {finding_id!r} not found")

    state.revoked.append(RevokedEntry(
        kind=kind, id=finding_id, title=title,
        reason=reason or "manual revocation", ts=time.time(),
    ))
    save_state(state)
    return (True, f"revoked {kind} {finding_id}")


def update_verdict_pattern(
    cwd: str,
    *,
    marker: str,
    reason: str,
    delta: int,
    session_id: str,
    workstream_id: str = "",
) -> None:
    """Bump a verdict-pattern's value_score by delta.

    Called by journal._apply_outcomes_feedback when a workstream
    closes with flag_history entries. Aggregates across sessions to
    build the watchdog's self-calibration memory.
    """
    if delta == 0:
        return
    try:
        state = load_state(cwd)
        fp = derive_pattern_fingerprint(marker, reason)
        existing = state.verdict_patterns.get(fp)
        if existing is None:
            state.verdict_patterns[fp] = VerdictPattern(
                fingerprint=fp,
                marker=(marker or "other").lower().split(":", 1)[0].strip() or "other",
                sample_reason=str(reason or "")[:280],
                total_flags=1,
                value_score=max(-10, min(10, delta)),
                last_seen_session=session_id,
                last_seen_ts=time.time(),
                sample_workstream_ids=[workstream_id] if workstream_id else [],
            )
        else:
            existing.total_flags += 1
            existing.value_score = max(-10, min(10, existing.value_score + delta))
            existing.last_seen_session = session_id
            existing.last_seen_ts = time.time()
            if workstream_id and workstream_id not in existing.sample_workstream_ids:
                existing.sample_workstream_ids.append(workstream_id)
                existing.sample_workstream_ids = existing.sample_workstream_ids[-5:]
        save_state(state)
    except Exception:
        pass


def rebuild_from_raw(cwd: str, *, prompt_sha: str = "") -> ProjectState:
    """Reconstruct semantic state from existing raw/ digests on disk.

    Idempotent (modulo `ts`). Used by `gadfly historian --rebuild` for
    drift recovery and by tests to verify the aggregator is deterministic.
    """
    sids = list_raw_digests(cwd)
    digests: list[dict[str, Any]] = []
    for sid in sids:
        d = read_raw_digest(cwd, sid)
        if isinstance(d, dict):
            digests.append(d)

    # Carry forward usefulness scores from a prior state if it exists.
    prior_use: dict[tuple[QuarantineKind, str], int] = {}
    revoked: list[RevokedEntry] = []
    try:
        prior = load_state(cwd)
        for cid, c in prior.corrections.items():
            if c.usefulness_score:
                prior_use[("correction", cid)] = c.usefulness_score
        for sid_canonical, s in prior.subsystems.items():
            if s.usefulness_score:
                prior_use[("subsystem", sid_canonical)] = s.usefulness_score
        revoked = list(prior.revoked)
    except Exception:
        pass

    return aggregate(
        digests,
        cwd=cwd,
        prompt_sha=prompt_sha,
        revoked=revoked,
        prior_usefulness=prior_use,
    )
