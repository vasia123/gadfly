"""Local HTTP viewer for the gadfly audit log.

Run:  python -m gadfly.viewer  [--port 7777] [--no-browser]

Then open the URL it prints. The page lists every Claude Code session that
the watchdog has logged, and for each session shows every verdict — what
came in (the raw Claude Code payload), what we sent to Haiku (the prompt),
and what Haiku replied (the verdict).

Everything is stdlib only — no Flask / FastAPI. The audit log is the source
of truth; the viewer just renders it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import log as audit_log
from . import project_state as ps


def _list_sessions(log_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not log_dir.is_dir():
        return out
    for f in log_dir.glob("*.jsonl"):
        count = 0
        flagged = 0
        goal_events = 0
        journal_events = 0
        trail_events = 0
        trail_drifts = 0
        stop_events = 0
        stop_blocks = 0
        latest_ts: float | None = None
        last_tool: str | None = None
        # Cwd is extracted from the first record carrying it — usually
        # the first verdict's payload.cwd. Lets the sidebar display the
        # human-readable project name instead of the UUID-style session id.
        session_cwd: str | None = None
        earliest_ts: float | None = None
        try:
            with f.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Distinguish record types. Old records without "type"
                    # are treated as verdicts (backward compat).
                    rec_type = rec.get("type", "verdict")
                    if rec_type == "goal_distill":
                        goal_events += 1
                        ts = rec.get("ts")
                        if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
                            latest_ts = ts
                            last_tool = "goal"
                        continue
                    if rec_type == "journal_update":
                        journal_events += 1
                        ts = rec.get("ts")
                        if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
                            latest_ts = ts
                            last_tool = "journal"
                        continue
                    if rec_type == "trail_update":
                        trail_events += 1
                        if rec.get("drift_detected"):
                            trail_drifts += 1
                        ts = rec.get("ts")
                        if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
                            latest_ts = ts
                            last_tool = "trail"
                        continue
                    if rec_type == "stop_verdict":
                        stop_events += 1
                        if rec.get("stop_appropriate") is False:
                            stop_blocks += 1
                        ts = rec.get("ts")
                        if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
                            latest_ts = ts
                            last_tool = "stop"
                        continue
                    count += 1
                    if rec.get("verdict", {}).get("professional") is False:
                        flagged += 1
                    ts = rec.get("ts")
                    if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
                        latest_ts = ts
                        last_tool = rec.get("tool_name")
                    if isinstance(ts, (int, float)) and (earliest_ts is None or ts < earliest_ts):
                        earliest_ts = ts
                    if session_cwd is None:
                        payload = rec.get("payload")
                        if isinstance(payload, dict):
                            c = payload.get("cwd")
                            if isinstance(c, str) and c:
                                session_cwd = c
        except OSError:
            continue
        out.append(
            {
                "id": f.stem,
                "count": count,
                "flagged": flagged,
                "goal_events": goal_events,
                "journal_events": journal_events,
                "trail_events": trail_events,
                "trail_drifts": trail_drifts,
                "stop_events": stop_events,
                "stop_blocks": stop_blocks,
                "latest_ts": latest_ts,
                "earliest_ts": earliest_ts,
                "cwd": session_cwd,
                "last_tool": last_tool,
            }
        )
    out.sort(key=lambda s: s["latest_ts"] or 0, reverse=True)
    return out


def _read_session(
    log_dir: Path,
    session_id: str,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Read a session's audit log.

    Returns {"total": int, "records": [...]}. When `limit` is given, the
    returned `records` are the `limit` MOST RECENT entries (still in the
    file's natural chronological order — newest at the end). When `limit`
    is None, every record is returned.
    """
    path = log_dir / f"{session_id}.jsonl"
    if not path.is_file():
        return {"total": 0, "records": []}
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    total = len(records)
    if limit is not None and limit >= 0:
        records = records[-limit:] if limit else []
    return {"total": total, "records": records}


def _health_snapshot() -> dict[str, Any]:
    """Quick health probe: backlog + last-digest-mtime + daemon-stale guess.

    The viewer polls this to decide whether to show a 'daemon not
    running?' banner. Defensive — never raises, missing things just
    degrade to neutral values.
    """
    import time as _time

    proot = ps.project_root()
    newest_state_mtime = 0.0
    digested_total = 0
    if proot.is_dir():
        for sp in proot.glob("*/state.json"):
            try:
                mt = sp.stat().st_mtime
                if mt > newest_state_mtime:
                    newest_state_mtime = mt
                data = json.loads(sp.read_text(encoding="utf-8"))
                digested_total += len(data.get("digested_sessions") or {})
            except Exception:
                continue

    hb_dir = (
        Path(os.environ.get("GADFLY_LOG_DIR")).parent / "heartbeat"
        if os.environ.get("GADFLY_LOG_DIR")
        else Path.home() / ".claude" / "gadfly" / "heartbeat"
    )
    active_heartbeats = 0
    newest_hb_ts = 0.0
    if hb_dir.is_dir():
        for tp in hb_dir.glob("*.tick"):
            try:
                payload = json.loads(tp.read_text(encoding="utf-8"))
                ts = float(payload.get("ts") or 0.0)
                if _time.time() - ts < 600:
                    active_heartbeats += 1
                if ts > newest_hb_ts:
                    newest_hb_ts = ts
            except Exception:
                continue

    # Daemon "stale" = recent activity heard from a session AND no
    # state.json updates in 10 min. We're cautious: if there's no
    # heartbeat at all, the daemon has nothing to do, so don't warn.
    stale = (
        active_heartbeats > 0
        and (
            newest_state_mtime == 0.0
            or (_time.time() - newest_state_mtime) > 600
        )
    )
    return {
        "active_heartbeats": active_heartbeats,
        "newest_heartbeat_ts": newest_hb_ts,
        "newest_state_mtime": newest_state_mtime,
        "digested_total": digested_total,
        "daemon_stale": stale,
    }


def _list_projects() -> list[dict[str, Any]]:
    """List cwds with a historian corpus. Returns one row per cwd_encoded."""
    out: list[dict[str, Any]] = []
    proot = ps.project_root()
    if not proot.is_dir():
        return out
    for d in sorted(proot.iterdir()):
        if not d.is_dir():
            continue
        sp = d / "state.json"
        if not sp.is_file():
            continue
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "cwd_encoded": d.name,
            "cwd": data.get("cwd") or "",
            "promises": len(data.get("promises") or {}),
            "corrections": len(data.get("corrections") or {}),
            "subsystems": len(data.get("subsystems") or {}),
            "quarantine": len(data.get("quarantine") or []),
            "digested": len(data.get("digested_sessions") or {}),
            "ts": data.get("ts"),
        })
    return out


def _read_project(cwd_encoded: str) -> dict[str, Any]:
    """Return the full ProjectState for a cwd_encoded, JSON-ready."""
    proot = ps.project_root()
    sp = proot / cwd_encoded / "state.json"
    if not sp.is_file():
        return {"cwd_encoded": cwd_encoded, "missing": True}
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:
        return {"cwd_encoded": cwd_encoded, "missing": True}
    data["cwd_encoded"] = cwd_encoded
    return data


class _Handler(BaseHTTPRequestHandler):
    log_dir: Path  # set by main()

    def log_message(self, format: str, *args: Any) -> None:  # silence default access log
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body_raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_raw.decode("utf-8")) if body_raw else {}
        except Exception:
            self._send_json({"ok": False, "error": "invalid JSON"}, status=400)
            return
        if path == "/api/project/promote":
            cwd = body.get("cwd")
            fid = body.get("id")
            if not cwd or not fid:
                self._send_json({"ok": False, "error": "cwd and id required"}, status=400)
                return
            ok, msg = ps.promote_finding(str(cwd), str(fid))
            self._send_json({"ok": ok, "message": msg})
            return
        if path == "/api/project/revoke":
            cwd = body.get("cwd")
            fid = body.get("id")
            reason = body.get("reason") or ""
            if not cwd or not fid:
                self._send_json({"ok": False, "error": "cwd and id required"}, status=400)
                return
            ok, msg = ps.revoke_finding(str(cwd), str(fid), reason=str(reason))
            self._send_json({"ok": ok, "message": msg})
            return
        self._send_json({"ok": False, "error": "unknown endpoint"}, status=404)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._send_text(_INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if path == "/api/sessions":
            self._send_json(_list_sessions(self.log_dir))
            return
        if path.startswith("/api/session/"):
            sid = path[len("/api/session/"):]
            limit_raw = (query.get("limit") or [None])[0]
            limit: int | None = None
            if limit_raw is not None:
                try:
                    limit = max(0, int(limit_raw))
                except ValueError:
                    limit = None
            self._send_json(_read_session(self.log_dir, sid, limit=limit))
            return
        if path.startswith("/api/system_prompt/"):
            sha = path[len("/api/system_prompt/"):]
            # Use the same root logic as the log module so this works under
            # GADFLY_LOG_DIR.
            text = audit_log.read_system_prompt(sha)
            if text is None:
                self._send_text("not found", status=404)
                return
            self._send_text(text)
            return
        if path == "/api/projects":
            self._send_json(_list_projects())
            return
        if path == "/api/health":
            self._send_json(_health_snapshot())
            return
        if path.startswith("/api/project/"):
            cwd_encoded = path[len("/api/project/"):]
            self._send_json(_read_project(cwd_encoded))
            return
        if path.startswith("/api/journal_snapshot/"):
            sha = path[len("/api/journal_snapshot/"):]
            text = audit_log.read_journal_snapshot(sha)
            if text is None:
                self._send_text("not found", status=404)
                return
            # Try to pretty-print JSON for readability; fall back to raw.
            try:
                parsed = json.loads(text)
                pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                self._send_text(pretty, content_type="application/json; charset=utf-8")
            except Exception:
                self._send_text(text, content_type="application/json; charset=utf-8")
            return
        if path.startswith("/api/trail_snapshot/"):
            sha = path[len("/api/trail_snapshot/"):]
            text = audit_log.read_trail_snapshot(sha)
            if text is None:
                self._send_text("not found", status=404)
                return
            try:
                parsed = json.loads(text)
                pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                self._send_text(pretty, content_type="application/json; charset=utf-8")
            except Exception:
                self._send_text(text, content_type="application/json; charset=utf-8")
            return
        self._send_text("not found", status=404)


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gadfly · observation log</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {
  --bg: #0d1117;
  --bg-elev: #161b22;
  --bg-card: #1c2230;
  --rule: #272e3c;
  --rule-bright: #353d4d;
  --ink: #e8e4dc;
  --ink-dim: #9aa0aa;
  --ink-faint: #5b6270;
  --accent: #dc4a3a;
  --accent-soft: rgba(220, 74, 58, 0.12);
  --positive: #86c084;
  --warn: #d4a056;
  --lvl-instance: #d97757;
  --lvl-class: #d4a056;
  --lvl-architecture: #86c084;
  --lvl-rationalization: #6b7080;
  --lvl-unclear: #3d434f;
  --display: 'Spectral', Georgia, serif;
  --body: 'IBM Plex Sans', -apple-system, system-ui, sans-serif;
  --mono: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; }
body {
  background: var(--bg);
  color: var(--ink);
  font: 400 14px/1.55 var(--body);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}

a { color: inherit; text-decoration: none; }
button { font: inherit; background: transparent; border: 0; color: inherit; cursor: pointer; padding: 0; }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
*:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ── Layout ────────────────────────────────────────────────── */

.shell { display: grid; grid-template-columns: 296px 1fr; height: 100vh; }
aside.sidebar { background: var(--bg-elev); border-right: 1px solid var(--rule); overflow: hidden; display: flex; flex-direction: column; }
main.canvas { overflow-y: auto; background: var(--bg); }

.brand {
  padding: 26px 24px 20px;
  border-bottom: 1px solid var(--rule);
  display: flex;
  align-items: baseline;
  justify-content: space-between;
}
.brand h1 {
  font: 600 13px/1 var(--display);
  letter-spacing: 0.22em;
  text-transform: uppercase;
  margin: 0;
}
.brand .live {
  font: 400 10px/1 var(--mono);
  color: var(--positive);
  letter-spacing: 0.08em;
  display: flex; align-items: center; gap: 7px;
  text-transform: lowercase;
}
.brand .live::before {
  content: ""; width: 6px; height: 6px; border-radius: 999px; background: var(--positive);
  box-shadow: 0 0 8px rgba(134, 192, 132, 0.6);
}
.brand .live.dim { color: var(--ink-faint); }
.brand .live.dim::before { background: var(--ink-faint); box-shadow: none; }

.side-tabs { display: flex; border-bottom: 1px solid var(--rule); }
.side-tab {
  flex: 1; padding: 13px 16px;
  font: 500 10.5px/1 var(--body);
  letter-spacing: 0.10em; text-transform: uppercase;
  color: var(--ink-faint); border-bottom: 2px solid transparent;
  transition: color 120ms, border-color 120ms;
}
.side-tab:hover { color: var(--ink-dim); }
.side-tab.active { color: var(--ink); border-bottom-color: var(--accent); }

.side-filter { padding: 14px 24px; display: flex; gap: 8px; border-bottom: 1px solid var(--rule); }
.filter-pill {
  font: 500 10.5px/1 var(--body);
  letter-spacing: 0.04em; text-transform: lowercase;
  padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--rule-bright); color: var(--ink-dim);
  transition: all 120ms;
}
.filter-pill:hover { color: var(--ink); border-color: var(--ink-faint); }
.filter-pill.active { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }

.sessions, .projects-list { flex: 1; overflow-y: auto; }
.session-row {
  padding: 16px 24px 14px;
  border-bottom: 1px solid var(--rule);
  cursor: pointer; transition: background 120ms;
}
.session-row:hover { background: var(--bg-card); }
.session-row.active { background: var(--bg-card); border-left: 2px solid var(--accent); padding-left: 22px; }
.session-row .project { font: 500 13.5px/1.2 var(--body); color: var(--ink); margin-bottom: 4px; }
.session-row .when {
  font: 400 11px/1 var(--mono);
  color: var(--ink-faint);
  letter-spacing: 0.02em;
  margin-bottom: 10px;
}
.session-row .marks { display: flex; flex-wrap: wrap; gap: 10px 16px; font: 500 10px/1 var(--mono); letter-spacing: 0.04em; text-transform: lowercase; }
.session-row .marks span { color: var(--ink-faint); }
.session-row .marks .drift, .session-row .marks .block { color: var(--accent); }

.sessions-empty {
  padding: 40px 24px; color: var(--ink-faint);
  font: 400 italic 13px/1.55 var(--body); text-align: center;
}

/* ── Canvas ────────────────────────────────────────────────── */

.empty-canvas {
  display: flex; align-items: center; justify-content: center;
  height: 100%; color: var(--ink-faint);
  font: 400 italic 14px/1.55 var(--body);
}

.session-canvas { padding: 64px 72px 120px; max-width: 960px; margin: 0 auto; }

.session-header { margin-bottom: 72px; }
.session-eyebrow {
  font: 500 10px/1 var(--body); color: var(--ink-faint);
  letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 16px;
}
.session-title { font: 500 36px/1.1 var(--display); letter-spacing: -0.01em; margin: 0 0 10px; }
.session-id {
  font: 400 11.5px/1 var(--mono); color: var(--ink-faint);
  letter-spacing: 0.04em; margin-bottom: 32px;
}
.vitals {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 24px; padding-top: 24px;
  border-top: 1px solid var(--rule);
}
.vital { display: flex; flex-direction: column; gap: 8px; }
.vital-n { font: 500 32px/1 var(--display); color: var(--ink); letter-spacing: -0.01em; }
.vital-n.alert { color: var(--accent); }
.vital-n.good { color: var(--positive); }
.vital-n.dim { color: var(--ink-faint); }
.vital-l {
  font: 500 10px/1 var(--body); color: var(--ink-faint);
  letter-spacing: 0.12em; text-transform: uppercase;
}

.section { margin-bottom: 56px; }
.section-title {
  font: 600 10.5px/1 var(--body); color: var(--ink-dim);
  letter-spacing: 0.20em; text-transform: uppercase;
  margin: 0 0 24px; padding-bottom: 12px;
  border-bottom: 1px solid var(--rule);
  display: flex; align-items: baseline; gap: 14px;
}
.section-title .count {
  font: 500 10.5px/1 var(--mono); color: var(--ink-faint);
  letter-spacing: 0.04em; text-transform: none;
}

/* ── Trajectory chart (signature element) ─────────────────── */

.chart-wrap { background: var(--bg-elev); border: 1px solid var(--rule); border-radius: 2px; padding: 28px 28px 20px; }
.chart { width: 100%; display: block; }
.chart-legend {
  display: flex; gap: 24px; margin-top: 20px;
  padding-top: 18px; border-top: 1px solid var(--rule);
  flex-wrap: wrap;
}
.legend-item {
  display: flex; align-items: center; gap: 7px;
  font: 500 10px/1 var(--body);
  letter-spacing: 0.06em; text-transform: lowercase;
  color: var(--ink-dim);
}
.legend-item .n { color: var(--ink-faint); margin-left: 2px; font-family: var(--mono); }
.legend-dot { width: 8px; height: 8px; border-radius: 999px; }

.chart-empty {
  color: var(--ink-faint);
  font: 400 italic 13px/1.55 var(--body);
  text-align: center; padding: 60px 20px;
}

/* ── Moments (drift + stop) ───────────────────────────────── */

.moment {
  border-top: 1px solid var(--rule);
  padding: 22px 0;
  display: flex; flex-direction: column; gap: 14px;
}
.moment:first-child { border-top: 0; padding-top: 4px; }

.moment-head { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.moment-id { font: 500 11.5px/1 var(--mono); color: var(--ink-faint); letter-spacing: 0.04em; }
.moment-kind {
  font: 500 11px/1 var(--mono); letter-spacing: 0.04em;
  padding: 4px 10px; border-radius: 2px;
  background: var(--accent-soft); color: var(--accent);
}
.moment-kind.ok { background: rgba(134, 192, 132, 0.14); color: var(--positive); }
.moment-state {
  font: 500 10px/1 var(--body); letter-spacing: 0.10em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 2px;
}
.moment-state.delivered { background: rgba(134, 192, 132, 0.14); color: var(--positive); }
.moment-state.suppressed { background: rgba(155, 160, 170, 0.14); color: var(--ink-faint); }
.moment-state.silent { background: rgba(212, 160, 86, 0.14); color: var(--warn); }
.moment-time { font: 400 11px/1 var(--mono); color: var(--ink-faint); margin-left: auto; }

.moment-reason {
  font: 400 12.5px/1.55 var(--body); color: var(--ink-dim);
  padding-left: 16px; border-left: 2px solid var(--rule);
}
.moment-reason b { color: var(--ink); font-weight: 500; }

.moment-question {
  background: var(--bg-elev);
  border-left: 3px solid var(--accent);
  padding: 16px 20px;
  font: 400 13px/1.65 var(--body);
  color: var(--ink); white-space: pre-line;
}

.moments-empty {
  color: var(--ink-faint); font: 400 italic 13px/1.55 var(--body);
  padding: 12px 0;
}

/* ── Breadcrumbs list ──────────────────────────────────────── */

.crumbs { display: flex; flex-direction: column; }
.crumb {
  display: grid; grid-template-columns: 56px 116px 1fr;
  gap: 18px; align-items: baseline;
  padding: 12px 0;
  border-top: 1px solid var(--rule);
}
.crumb:first-child { border-top: 0; }
.crumb-id { font: 500 11px/1 var(--mono); color: var(--ink-faint); }
.crumb-level {
  font: 500 10px/1 var(--mono); letter-spacing: 0.04em;
  padding: 3px 8px; border-radius: 2px; text-align: center;
}
.crumb-level.instance       { background: rgba(217, 119, 87, 0.14);  color: var(--lvl-instance); }
.crumb-level.class          { background: rgba(212, 160, 86, 0.14);  color: var(--lvl-class); }
.crumb-level.architecture   { background: rgba(134, 192, 132, 0.14); color: var(--lvl-architecture); }
.crumb-level.rationalization{ background: rgba(155, 160, 170, 0.14); color: var(--lvl-rationalization); }
.crumb-level.unclear        { background: rgba(110, 117, 130, 0.18); color: var(--ink-faint); }
.crumb-text { font: 400 13px/1.5 var(--body); color: var(--ink); }
.crumb-action {
  font: 400 11px/1.45 var(--mono); color: var(--ink-faint);
  margin-top: 5px; white-space: pre-wrap; word-break: break-all;
}

.toggle-line {
  font: 500 10.5px/1 var(--body);
  color: var(--ink-faint);
  letter-spacing: 0.08em; text-transform: lowercase;
  padding-top: 18px; transition: color 120ms;
}
.toggle-line:hover { color: var(--ink-dim); }

/* ── Raw audit tail ────────────────────────────────────────── */

.records-section { margin-top: 24px; padding-top: 24px; border-top: 1px solid var(--rule); }
.records-list { margin-top: 16px; }
.record-row {
  padding: 11px 0; border-top: 1px solid var(--rule);
  display: grid; grid-template-columns: 110px 1fr 60px;
  gap: 18px; font: 400 11.5px/1.45 var(--mono); color: var(--ink-dim);
  align-items: baseline;
}
.record-row:first-child { border-top: 0; }
.record-row.flag { color: var(--accent); }
.record-row .type { color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.10em; font-size: 10px; }
.record-row .ts { color: var(--ink-faint); text-align: right; }

/* ── Health banner ─────────────────────────────────────────── */

.health-banner {
  margin: 0 24px 16px; padding: 12px 14px;
  border: 1px solid rgba(212, 160, 86, 0.4);
  background: rgba(212, 160, 86, 0.08);
  border-radius: 2px;
  font: 400 11.5px/1.45 var(--body); color: var(--warn);
}

/* ── Mobile ────────────────────────────────────────────────── */

@media (max-width: 760px) {
  .shell { grid-template-columns: 1fr; height: auto; }
  aside.sidebar { border-right: 0; border-bottom: 1px solid var(--rule); max-height: 50vh; }
  .session-canvas { padding: 32px 22px 80px; }
  .session-title { font-size: 26px; }
  .vitals { grid-template-columns: repeat(2, 1fr); gap: 18px; }
  .vital-n { font-size: 24px; }
  .crumb { grid-template-columns: 44px 96px 1fr; gap: 12px; }
}
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <header class="brand">
      <h1>Gadfly</h1>
      <span class="live" id="live-indicator">live</span>
    </header>
    <nav class="side-tabs" role="tablist">
      <button class="side-tab active" data-tab="sessions" role="tab" aria-selected="true">Sessions</button>
      <button class="side-tab" data-tab="projects" role="tab" aria-selected="false">Projects</button>
    </nav>
    <div class="side-filter">
      <button class="filter-pill" id="filter-drift">with drift</button>
    </div>
    <div id="health-banner-slot"></div>
    <div class="sessions" id="sessions-pane" role="tabpanel"></div>
    <div class="projects-list" id="projects-pane" role="tabpanel" style="display:none"></div>
  </aside>

  <main class="canvas" id="canvas" tabindex="0">
    <div class="empty-canvas">Select a session to read its trajectory.</div>
  </main>
</div>

<script>
/* ── Constants (mirrored from prompts.py) ───────────────────── */

const LEVEL_ORDER = ['architecture', 'class', 'instance', 'rationalization', 'unclear'];
const LEVEL_COLOR = {
  instance: '#d97757',
  class: '#d4a056',
  architecture: '#86c084',
  rationalization: '#6b7080',
  unclear: '#3d434f',
};

const TRAIL_DRIFT_QUESTIONS = {
  hardcoded_instance: "The instance you just fixed — it's a specific case of WHAT? Name the class. Where in the codebase does that class already have a slot? If the slot exists, route the next patch through it. If not, create one before adding the next patch.",
  premature_ceiling: "You climbed one abstraction level (instance → class) and stopped. What level above the class would the architectural fix live at? Does that slot already exist? If yes — why are you patching class-level instead of routing through architecture?",
  wrong_layer: "The thing you just changed — what is its responsibility (presentation / domain / persistence / scheduling / classification)? Which layer owns the bug? If they don't match — what does moving the fix to the right layer cost?",
  rule_skip: "The action bypasses a rule, convention, or contract visible in this project. Name the rule. Did you deviate for a reason, or because the bypass was more convenient?",
  incomplete_coverage: "The last action closed ONE branch of a larger equivalent set. What are the other members of its class? Will the fix cover them?",
  recon_as_work: "Several actions have been reading/grepping/diagnostics — no code committed. What concrete BUILD step have you committed in the last few moves? If none — do you have enough context to commit one now?",
  rationalization: "Your reasoning looks like a post-hoc justification of a prior wrong-level step. Which action are you defending? Would a senior reviewer accept it at face value?",
  other: "This action triggered a longitudinal three-level check. Stop and reflect: instance you fixed → class it belongs to → architecture slot."
};

const TRAIL_STOP_QUESTION =
  "Your work is being monitored, and you are about to stop. Before stopping, re-read the user's most recent ask and answer honestly:\n"
  + "1. What did the user explicitly ask for? State it in one sentence.\n"
  + "2. Walk down the asks: which parts have you ACTUALLY committed (landed in code / shipped output)? Which parts are still open?\n"
  + "3. If anything is open — is there a reason you cannot finish it now? If there is no such reason, do NOT stop: finish the open work first.\n"
  + "Ignore if the user explicitly approved partial delivery or asked for analysis-only.";

/* ── State ──────────────────────────────────────────────────── */

const state = {
  currentTab: 'sessions',
  currentSession: null,
  currentProject: null,
  sessions: [],
  projects: [],
  records: [],
  trailSnap: null,
  trailSnapSha: null,
  driftOnly: false,
  showAllCrumbs: false,
  showRawRecords: false,
  lastSessionsSig: '',
  lastCanvasSig: '',
};

/* ── Utilities ──────────────────────────────────────────────── */

function escapeHtml(s) {
  return (s == null ? '' : String(s))
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmtClock(ts) {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
}

function fmtRelDate(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const target = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diff = Math.round((today - target) / 86400000);
  if (diff === 0) return 'today';
  if (diff === 1) return 'yesterday';
  if (diff > 0 && diff < 7) return d.toLocaleDateString('en-GB', { weekday: 'short' }).toLowerCase();
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' }).toLowerCase();
}

function fmtFullDate(ts) {
  if (!ts) return '';
  return new Date(ts * 1000).toLocaleDateString('en-GB', { day: '2-digit', month: 'long', year: 'numeric' });
}

function fmtDuration(start, end) {
  if (!start || !end || end < start) return '';
  const s = end - start;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${Math.floor(s)}s`;
}

function projectName(cwd) {
  if (!cwd) return 'unknown';
  const parts = String(cwd).split('/').filter(Boolean);
  return parts.length ? parts[parts.length - 1] : 'unknown';
}

function shortSid(sid) { return sid && sid.length > 8 ? sid.slice(0, 8) : (sid || ''); }

/* ── Sidebar — sessions list ────────────────────────────────── */

function setTab(t) {
  state.currentTab = t;
  for (const el of document.querySelectorAll('.side-tab')) {
    const on = el.dataset.tab === t;
    el.classList.toggle('active', on);
    el.setAttribute('aria-selected', on ? 'true' : 'false');
  }
  document.getElementById('sessions-pane').style.display = t === 'sessions' ? '' : 'none';
  document.getElementById('projects-pane').style.display = t === 'projects' ? '' : 'none';
  if (t === 'projects') loadProjects();
}

function toggleDriftFilter() {
  state.driftOnly = !state.driftOnly;
  document.getElementById('filter-drift').classList.toggle('active', state.driftOnly);
  state.lastSessionsSig = '';
  renderSessions();
}

async function loadSessions() {
  try {
    const r = await fetch('/api/sessions');
    state.sessions = await r.json();
    renderSessions();
    if (state.currentSession) loadSession(state.currentSession, false);
  } catch (e) { console.error(e); }
}

function renderSessions() {
  const visible = state.driftOnly
    ? state.sessions.filter(s => (s.trail_drifts || 0) > 0 || (s.stop_blocks || 0) > 0 || (s.flagged || 0) > 0)
    : state.sessions;
  const sig = JSON.stringify({active: state.currentSession, drift: state.driftOnly, n: visible.length, ids: visible.slice(0, 30).map(s => s.id + ':' + s.count + ':' + (s.trail_drifts || 0))});
  if (sig === state.lastSessionsSig) return;
  state.lastSessionsSig = sig;
  const root = document.getElementById('sessions-pane');
  if (visible.length === 0) {
    root.innerHTML = `<div class="sessions-empty">${state.driftOnly ? 'No sessions with drift in the log.' : 'No sessions recorded yet.'}</div>`;
    return;
  }
  root.innerHTML = visible.map(s => {
    const proj = projectName(s.cwd);
    const when = `${fmtRelDate(s.latest_ts)} · ${fmtClock(s.latest_ts)}`;
    const marks = [];
    if (s.count > 0) marks.push(`<span>${s.count} actions</span>`);
    if (s.trail_drifts > 0) marks.push(`<span class="drift">${s.trail_drifts} drifts</span>`);
    if (s.stop_blocks > 0) marks.push(`<span class="block">${s.stop_blocks} stop blocks</span>`);
    return `<div class="session-row ${state.currentSession === s.id ? 'active' : ''}"
                 role="button" tabindex="0"
                 onclick="loadSession('${escapeHtml(s.id)}', true)"
                 onkeydown="if(event.key==='Enter'||event.key===' ')loadSession('${escapeHtml(s.id)}', true)">
              <div class="project">${escapeHtml(proj)}</div>
              <div class="when">${escapeHtml(when)} · ${escapeHtml(shortSid(s.id))}</div>
              <div class="marks">${marks.join('')}</div>
            </div>`;
  }).join('');
}

async function loadSession(id, scrollReset) {
  if (state.currentSession !== id) {
    state.currentSession = id;
    state.trailSnap = null;
    state.trailSnapSha = null;
    state.showAllCrumbs = false;
    state.showRawRecords = false;
    state.lastCanvasSig = '';
    renderSessions();
  }
  try {
    const r = await fetch('/api/session/' + encodeURIComponent(id) + '?limit=500');
    const payload = await r.json();
    state.records = Array.isArray(payload) ? payload : (payload.records || []);
  } catch (e) {
    console.error(e); state.records = [];
  }
  await ensureTrailSnap();
  renderCanvas(scrollReset);
}

async function ensureTrailSnap() {
  const trails = state.records.filter(r => r.type === 'trail_update');
  if (trails.length === 0) {
    state.trailSnap = null; state.trailSnapSha = null; return;
  }
  const latest = trails[trails.length - 1];
  const sha = latest && latest.new_trail_sha;
  if (!sha) {
    state.trailSnap = null; state.trailSnapSha = null; return;
  }
  if (sha === state.trailSnapSha && state.trailSnap) return;
  try {
    const r = await fetch('/api/trail_snapshot/' + encodeURIComponent(sha));
    if (!r.ok) { state.trailSnap = null; state.trailSnapSha = null; return; }
    const txt = await r.text();
    state.trailSnap = JSON.parse(txt);
    state.trailSnapSha = sha;
  } catch (e) {
    console.error('trail snapshot fetch failed', e);
    state.trailSnap = null; state.trailSnapSha = null;
  }
}

/* ── Main canvas ────────────────────────────────────────────── */

function sessionMeta(sid) {
  const recs = state.records;
  const tsList = recs.map(r => r.ts).filter(t => typeof t === 'number');
  const startTs = tsList.length ? Math.min(...tsList) : null;
  const endTs = tsList.length ? Math.max(...tsList) : null;
  const verdicts = recs.filter(r => (r.type || 'verdict') === 'verdict');
  const actions = verdicts.length;
  const flagged = verdicts.filter(r => r.verdict && r.verdict.professional === false).length;
  const driftEvents = recs.filter(r => r.type === 'trail_update' && r.drift_detected);
  const drifts = driftEvents.length;
  const driftsDelivered = driftEvents.filter(r => r.delivered_to_agent).length;
  const stops = recs.filter(r => r.type === 'stop_verdict');
  const stopsBlocked = stops.filter(r => r.stop_appropriate === false).length;
  let cwd = '';
  const sidebarEntry = state.sessions.find(s => s.id === sid);
  if (sidebarEntry && sidebarEntry.cwd) cwd = sidebarEntry.cwd;
  if (!cwd) {
    const carrier = recs.find(r => r.payload && r.payload.cwd);
    if (carrier) cwd = carrier.payload.cwd;
  }
  return { startTs, endTs, actions, flagged, drifts, driftsDelivered, stops, stopsBlocked, cwd };
}

function renderCanvas(scrollReset) {
  const canvas = document.getElementById('canvas');
  if (!state.currentSession) {
    canvas.innerHTML = '<div class="empty-canvas">Select a session to read its trajectory.</div>';
    return;
  }
  const meta = sessionMeta(state.currentSession);
  const sig = JSON.stringify({
    sid: state.currentSession,
    sha: state.trailSnapSha,
    nrec: state.records.length,
    crumbs: state.showAllCrumbs,
    raw: state.showRawRecords,
    meta: [meta.actions, meta.drifts, meta.driftsDelivered, meta.stopsBlocked],
  });
  if (sig === state.lastCanvasSig && !scrollReset) return;
  state.lastCanvasSig = sig;

  const project = projectName(meta.cwd);
  const dateStr = fmtFullDate(meta.startTs);
  const duration = fmtDuration(meta.startTs, meta.endTs);
  const eyebrowBits = [dateStr, duration].filter(Boolean);

  let html = '<div class="session-canvas">';
  html += `
    <header class="session-header">
      <div class="session-eyebrow">${escapeHtml(eyebrowBits.join(' · '))}</div>
      <h2 class="session-title">${escapeHtml(project)}</h2>
      <div class="session-id">${escapeHtml(state.currentSession)}</div>
      <div class="vitals">
        <div class="vital">
          <span class="vital-n">${meta.actions}</span>
          <span class="vital-l">actions</span>
        </div>
        <div class="vital">
          <span class="vital-n ${meta.drifts ? 'alert' : 'dim'}">${meta.drifts}</span>
          <span class="vital-l">drifts caught</span>
        </div>
        <div class="vital">
          <span class="vital-n ${meta.driftsDelivered ? 'good' : 'dim'}">${meta.driftsDelivered}</span>
          <span class="vital-l">delivered</span>
        </div>
        <div class="vital">
          <span class="vital-n ${meta.stopsBlocked ? 'alert' : 'dim'}">${meta.stopsBlocked}</span>
          <span class="vital-l">stop blocks</span>
        </div>
      </div>
    </header>`;

  // Trajectory
  const crumbCount = state.trailSnap?.breadcrumbs?.length || 0;
  html += `<section class="section">
    <h3 class="section-title">Trajectory <span class="count">${crumbCount} breadcrumbs</span></h3>
    <div class="chart-wrap">${renderChart()}</div>
  </section>`;

  // Drifts
  const driftRecords = state.records.filter(r => r.type === 'trail_update' && r.drift_detected);
  html += `<section class="section">
    <h3 class="section-title">Drift moments <span class="count">${driftRecords.length}</span></h3>
    ${renderDrifts(driftRecords)}
  </section>`;

  // Stops
  if (meta.stops.length > 0) {
    html += `<section class="section">
      <h3 class="section-title">Stop checks <span class="count">${meta.stops.length}</span></h3>
      ${renderStops(meta.stops)}
    </section>`;
  }

  // Breadcrumbs
  html += `<section class="section">
    <h3 class="section-title">Breadcrumbs <span class="count">full trail</span></h3>
    ${renderCrumbsList()}
  </section>`;

  // Raw audit
  html += `<section class="records-section">
    <button class="toggle-line" onclick="toggleRawRecords()">
      ${state.showRawRecords ? '− hide raw audit log' : '+ show raw audit log (' + state.records.length + ' records)'}
    </button>
    ${state.showRawRecords ? `<div class="records-list">${renderRawRecords()}</div>` : ''}
  </section>`;

  html += '</div>';
  canvas.innerHTML = html;
  if (scrollReset) canvas.scrollTo({ top: 0 });
}

function toggleAllCrumbs() { state.showAllCrumbs = !state.showAllCrumbs; state.lastCanvasSig = ''; renderCanvas(false); }
function toggleRawRecords() { state.showRawRecords = !state.showRawRecords; state.lastCanvasSig = ''; renderCanvas(false); }

/* ── Signature: the trajectory chart ──────────────────────── */

function renderChart() {
  if (!state.trailSnap || !state.trailSnap.breadcrumbs || state.trailSnap.breadcrumbs.length === 0) {
    return `<div class="chart-empty">
      No breadcrumbs recorded yet.<br>
      <span style="color: var(--ink-faint); font-size: 11px;">Either the trail module is disabled, the session predates it, or no watched tool action has fired.</span>
    </div>`;
  }
  const crumbs = state.trailSnap.breadcrumbs;
  const drifts = state.trailSnap.drift_flags || [];
  const W = 880, H = 260;
  const PAD_L = 132, PAD_R = 32, PAD_T = 28, PAD_B = 32;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const laneH = innerH / LEVEL_ORDER.length;

  const minIdx = crumbs[0].action_index;
  const maxIdx = crumbs[crumbs.length - 1].action_index;
  const xFor = (idx) => {
    if (maxIdx === minIdx) return PAD_L + innerW / 2;
    return PAD_L + ((idx - minIdx) / (maxIdx - minIdx)) * innerW;
  };
  const yFor = (level) => {
    const i = LEVEL_ORDER.indexOf(level);
    const lane = i < 0 ? LEVEL_ORDER.length - 1 : i;
    return PAD_T + lane * laneH + laneH / 2;
  };

  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Agent trajectory through abstraction levels">`;

  // Lane separator lines + labels (right-aligned in gutter)
  for (let i = 0; i < LEVEL_ORDER.length; i++) {
    const level = LEVEL_ORDER[i];
    const yTop = PAD_T + i * laneH;
    svg += `<line x1="${PAD_L}" y1="${yTop + laneH}" x2="${W - PAD_R}" y2="${yTop + laneH}" stroke="#272e3c" stroke-width="1"/>`;
    svg += `<text x="${PAD_L - 14}" y="${yTop + laneH / 2}" text-anchor="end" dominant-baseline="middle"
              fill="${LEVEL_COLOR[level]}" font-family="JetBrains Mono, monospace" font-size="10" letter-spacing="0.5">${level}</text>`;
  }
  // Top rule
  svg += `<line x1="${PAD_L}" y1="${PAD_T}" x2="${W - PAD_R}" y2="${PAD_T}" stroke="#272e3c" stroke-width="1"/>`;

  // Drift bands — vermilion blocks for delivered, slate for suppressed
  for (const d of drifts) {
    const cites = (d.cited_action_indexes || []).filter(i => i >= minIdx && i <= maxIdx);
    if (cites.length === 0) continue;
    const xMin = xFor(Math.min(...cites));
    const xMax = xFor(Math.max(...cites));
    const w = Math.max(xMax - xMin, 8);
    const delivered = d.delivered_to_agent;
    const fill = delivered ? 'rgba(220,74,58,0.10)' : 'rgba(155,160,170,0.05)';
    const stroke = delivered ? 'rgba(220,74,58,0.45)' : 'rgba(155,160,170,0.18)';
    svg += `<rect x="${xMin - 4}" y="${PAD_T + 2}" width="${w + 8}" height="${innerH - 4}"
              fill="${fill}" stroke="${stroke}" stroke-width="1" stroke-dasharray="${delivered ? '0' : '3 3'}" rx="1"/>`;
    if (delivered) {
      svg += `<text x="${xMin + w / 2 - 4}" y="${PAD_T - 8}" text-anchor="middle"
                fill="#dc4a3a" font-family="JetBrains Mono, monospace" font-size="9" letter-spacing="0.6">${escapeHtml(d.drift_kind)}</text>`;
    }
  }

  // Path
  const pts = crumbs.map(b => `${xFor(b.action_index)},${yFor(b.abstraction_level)}`).join(' ');
  svg += `<polyline points="${pts}" fill="none" stroke="rgba(232,228,220,0.18)" stroke-width="1.25"/>`;

  // Breadcrumb dots
  for (const b of crumbs) {
    const x = xFor(b.action_index);
    const y = yFor(b.abstraction_level);
    const color = LEVEL_COLOR[b.abstraction_level] || LEVEL_COLOR.unclear;
    svg += `<circle cx="${x}" cy="${y}" r="5" fill="${color}" stroke="${color}" stroke-width="1.5" opacity="0.95">`;
    svg += `<title>#${b.action_index} [${b.abstraction_level}] ${escapeHtml(b.breadcrumb_text)}</title>`;
    svg += `</circle>`;
  }

  // Axis tick labels
  svg += `<text x="${PAD_L}" y="${H - PAD_B + 20}" fill="#5b6270" font-family="JetBrains Mono, monospace" font-size="10">#${minIdx}</text>`;
  svg += `<text x="${W - PAD_R}" y="${H - PAD_B + 20}" text-anchor="end" fill="#5b6270" font-family="JetBrains Mono, monospace" font-size="10">#${maxIdx}</text>`;

  svg += '</svg>';

  // Legend
  const counts = {};
  for (const b of crumbs) counts[b.abstraction_level] = (counts[b.abstraction_level] || 0) + 1;
  const legend = LEVEL_ORDER.map(lvl => counts[lvl]
    ? `<div class="legend-item"><span class="legend-dot" style="background:${LEVEL_COLOR[lvl]}"></span>${lvl}<span class="n">${counts[lvl]}</span></div>` : ''
  ).filter(Boolean).join('');
  if (legend) svg += `<div class="chart-legend">${legend}</div>`;
  return svg;
}

/* ── Drift + stop renderers ───────────────────────────────── */

function suppressionReason(s) {
  if (!s) return '';
  const x = String(s).toLowerCase();
  if (x.includes('without citations')) return 'model claimed drift without naming specific breadcrumbs';
  if (x.includes('same kind'))         return 'this drift kind already fired in the last 3 events';
  if (x.includes('no payload'))        return 'model did not return a valid verdict';
  return s;
}

function renderDrifts(drifts) {
  if (drifts.length === 0) {
    return '<div class="moments-empty">No drift caught in this session.</div>';
  }
  return drifts.slice().reverse().map(d => {
    const kind = d.drift_kind || 'other';
    const state = d.delivered_to_agent ? 'delivered' : (d.suppressed ? 'suppressed' : 'silent');
    const stateLabel = state;
    const question = TRAIL_DRIFT_QUESTIONS[kind] || TRAIL_DRIFT_QUESTIONS.other;
    const supBlock = d.suppressed && d.skipped_reason
      ? `<div class="moment-reason"><b>suppressed:</b> ${escapeHtml(suppressionReason(d.skipped_reason))}</div>` : '';
    return `<article class="moment">
      <div class="moment-head">
        <span class="moment-id">#${d.action_index ?? '?'}</span>
        <span class="moment-kind">${escapeHtml(kind)}</span>
        <span class="moment-state ${state}">${stateLabel}</span>
        <span class="moment-time">${escapeHtml(fmtClock(d.ts))}</span>
      </div>
      ${supBlock}
      <div class="moment-question">${escapeHtml(question)}</div>
    </article>`;
  }).join('');
}

function renderStops(stops) {
  return stops.slice().reverse().map(s => {
    const ok = !!s.stop_appropriate;
    const stateClass = ok ? 'silent' : (s.delivered_to_agent ? 'delivered' : 'suppressed');
    const stateLabel = ok ? 'stop ok' : (s.delivered_to_agent ? 'forced continue' : 'logged only');
    const kindClass = ok ? 'moment-kind ok' : 'moment-kind';
    const kindLabel = ok ? 'stop_appropriate' : 'premature_stop';
    const missing = Array.isArray(s.missing_pieces) ? s.missing_pieces : [];
    const missingHtml = missing.length === 0 ? '' :
      `<div class="moment-reason"><b>missing:</b> ${missing.map(escapeHtml).join(' · ')}</div>`;
    const reasoningHtml = s.reasoning ? `<div class="moment-reason">${escapeHtml(s.reasoning)}</div>` : '';
    const qHtml = (!ok && s.delivered_to_agent)
      ? `<div class="moment-question">${escapeHtml(TRAIL_STOP_QUESTION)}</div>` : '';
    return `<article class="moment">
      <div class="moment-head">
        <span class="moment-id">stop</span>
        <span class="${kindClass}">${kindLabel}</span>
        <span class="moment-state ${stateClass}">${stateLabel}</span>
        <span class="moment-time">${escapeHtml(fmtClock(s.ts))}</span>
      </div>
      ${missingHtml}
      ${reasoningHtml}
      ${qHtml}
    </article>`;
  }).join('');
}

function renderCrumbsList() {
  if (!state.trailSnap?.breadcrumbs?.length) {
    return '<div class="moments-empty">No breadcrumbs recorded.</div>';
  }
  const crumbs = state.trailSnap.breadcrumbs;
  const showAll = state.showAllCrumbs;
  const sliced = showAll ? crumbs : crumbs.slice(-12);
  let html = '<div class="crumbs">';
  html += sliced.map(b => `
    <div class="crumb">
      <span class="crumb-id">#${b.action_index}</span>
      <span class="crumb-level ${escapeHtml(b.abstraction_level)}">${escapeHtml(b.abstraction_level)}</span>
      <div>
        <div class="crumb-text">${escapeHtml(b.breadcrumb_text || '')}</div>
        ${b.action_summary ? `<div class="crumb-action">${escapeHtml(b.action_summary)}</div>` : ''}
      </div>
    </div>`).join('');
  html += '</div>';
  if (crumbs.length > 12) {
    html += `<button class="toggle-line" onclick="toggleAllCrumbs()">${
      showAll ? '↑ show recent 12 only' : '↓ show all ' + crumbs.length
    }</button>`;
  }
  return html;
}

function renderRawRecords() {
  const recs = state.records.slice().reverse().slice(0, 300);
  return recs.map(r => {
    const t = r.type || 'verdict';
    const flagged = r.verdict?.professional === false || r.drift_detected
                  || (t === 'stop_verdict' && r.stop_appropriate === false);
    return `<div class="record-row ${flagged ? 'flag' : ''}">
      <span class="type">${escapeHtml(t)}</span>
      <span>${escapeHtml(recordSummary(r, t))}</span>
      <span class="ts">${escapeHtml(fmtClock(r.ts))}</span>
    </div>`;
  }).join('');
}

function recordSummary(r, t) {
  if (t === 'verdict') {
    const tool = r.tool_name || '?';
    const reason = r.verdict?.reason || '(silent)';
    return `${tool} — ${reason.slice(0, 100)}`;
  }
  if (t === 'journal_update') {
    return Array.isArray(r.diff_summary) && r.diff_summary.length ? r.diff_summary.join(' · ') : '(no change)';
  }
  if (t === 'trail_update') {
    if (r.drift_detected) {
      return `drift ${r.drift_kind || 'other'} (${r.delivered_to_agent ? 'delivered' : (r.suppressed ? 'suppressed' : 'silent')})`;
    }
    return r.advances_trail ? 'breadcrumb +1' : 'no change';
  }
  if (t === 'stop_verdict') {
    if (r.stop_appropriate) return 'stop appropriate';
    const missing = Array.isArray(r.missing_pieces) ? r.missing_pieces : [];
    return `premature: ${missing.length ? missing.join(', ') : '(no missing list)'}`;
  }
  if (t === 'goal_distill') {
    return r.goal ? `goal: ${String(r.goal).slice(0, 100)}` : '(goal cleared)';
  }
  if (t === 'historian_digest') {
    return `digest · ${r.chunks ?? 0} chunks`;
  }
  return '';
}

/* ── Projects view ──────────────────────────────────────── */

async function loadProjects() {
  try {
    const r = await fetch('/api/projects');
    state.projects = await r.json();
    renderProjects();
  } catch (e) { console.error(e); }
}

function renderProjects() {
  const root = document.getElementById('projects-pane');
  if (!Array.isArray(state.projects) || state.projects.length === 0) {
    root.innerHTML = '<div class="sessions-empty">No project memory yet.</div>';
    return;
  }
  root.innerHTML = state.projects.map(p => `
    <div class="session-row" role="button" tabindex="0"
         onclick="loadProject('${escapeHtml(p.cwd_encoded)}')"
         onkeydown="if(event.key==='Enter')loadProject('${escapeHtml(p.cwd_encoded)}')">
      <div class="project">${escapeHtml(projectName(p.cwd))}</div>
      <div class="when">${p.promises||0} promises · ${p.corrections||0} corrections</div>
      <div class="marks"><span>${p.subsystems||0} subsystems</span></div>
    </div>
  `).join('');
}

async function loadProject(enc) {
  state.currentProject = enc;
  try {
    const r = await fetch('/api/project/' + encodeURIComponent(enc));
    renderProject(await r.json());
  } catch (e) { console.error(e); }
}

function renderProject(data) {
  const canvas = document.getElementById('canvas');
  if (!data?.cwd) {
    canvas.innerHTML = '<div class="empty-canvas">Project not found.</div>';
    return;
  }
  let html = '<div class="session-canvas">';
  html += `<header class="session-header">
    <div class="session-eyebrow">Project memory</div>
    <h2 class="session-title">${escapeHtml(projectName(data.cwd))}</h2>
    <div class="session-id">${escapeHtml(data.cwd)}</div>
  </header>`;
  for (const kind of ['promises', 'corrections', 'subsystems']) {
    const items = Array.isArray(data[kind]) ? data[kind] : Object.values(data[kind] || {});
    if (items.length === 0) continue;
    html += `<section class="section">
      <h3 class="section-title">${kind} <span class="count">${items.length}</span></h3>`;
    for (const item of items.slice(0, 20)) {
      html += `<article class="moment">
        <div class="moment-head">
          <span class="moment-id">${escapeHtml(item.id || '')}</span>
          ${item.status ? `<span class="moment-state silent">${escapeHtml(item.status)}</span>` : ''}
        </div>
        <div class="crumb-text">${escapeHtml(item.title || '')}</div>
        ${item.evidence_quote ? `<div class="moment-reason">"${escapeHtml(item.evidence_quote)}"</div>` : ''}
      </article>`;
    }
    html += '</section>';
  }
  html += '</div>';
  canvas.innerHTML = html;
}

/* ── Health banner ───────────────────────────────────────── */

async function refreshHealth() {
  try {
    const r = await fetch('/api/health');
    const h = await r.json();
    const banner = document.getElementById('health-banner-slot');
    const live = document.getElementById('live-indicator');
    if (!h || h.daemon_stale === undefined || h.daemon_stale === null) {
      banner.innerHTML = ''; live.classList.remove('dim'); return;
    }
    if (h.daemon_stale) {
      banner.innerHTML = `<div class="health-banner">Historian daemon hasn't pulsed in a while. <span style="color: var(--ink-faint)">Run <code>python -m gadfly.historian watch</code></span></div>`;
    } else {
      banner.innerHTML = '';
    }
  } catch (e) {
    document.getElementById('live-indicator').classList.add('dim');
  }
}

/* ── Wire up ──────────────────────────────────────────── */

for (const el of document.querySelectorAll('.side-tab')) {
  el.addEventListener('click', () => setTab(el.dataset.tab));
}
document.getElementById('filter-drift').addEventListener('click', toggleDriftFilter);

refreshHealth();
loadSessions();
setInterval(() => { loadSessions(); refreshHealth(); }, 5000);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gadfly.viewer")
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GADFLY_VIEWER_PORT", "7777")),
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument(
        "--log-dir",
        default=os.environ.get("GADFLY_LOG_DIR") or str(Path.home() / ".claude" / "gadfly" / "log"),
    )
    args = ap.parse_args(argv)

    _Handler.log_dir = Path(args.log_dir)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"gadfly viewer: serving {args.log_dir}", flush=True)
    print(f"gadfly viewer: {url}", flush=True)
    if not args.no_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
