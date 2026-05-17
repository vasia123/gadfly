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
        latest_ts: float | None = None
        last_tool: str | None = None
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
                    count += 1
                    if rec.get("verdict", {}).get("professional") is False:
                        flagged += 1
                    ts = rec.get("ts")
                    if isinstance(ts, (int, float)) and (latest_ts is None or ts > latest_ts):
                        latest_ts = ts
                        last_tool = rec.get("tool_name")
        except OSError:
            continue
        out.append(
            {
                "id": f.stem,
                "count": count,
                "flagged": flagged,
                "goal_events": goal_events,
                "journal_events": journal_events,
                "latest_ts": latest_ts,
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
        self._send_text("not found", status=404)


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>gadfly — watchdog log</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --panel-2: #1d2230;
    --border: #262c3a;
    --text: #d8dde7;
    --muted: #8a93a6;
    --accent: #8ab4f8;
    --good: #6fcf97;
    --bad: #f08a8a;
    --warn: #f0c674;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  a { color: var(--accent); text-decoration: none; }
  pre, code, .mono { font: 12.5px/1.55 "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace; }
  .layout { display: grid; grid-template-columns: 340px 1fr; height: 100vh; }
  aside { background: var(--panel); border-right: 1px solid var(--border); overflow-y: auto; }
  main { overflow-y: auto; }
  .topbar { padding: 14px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }
  .topbar h1 { font-size: 14px; font-weight: 600; margin: 0; letter-spacing: 0.5px; }
  .topbar small { color: var(--muted); }
  .toggle { display: inline-flex; align-items: center; gap: 6px; background: var(--panel-2); border: 1px solid var(--border); border-radius: 999px; padding: 4px 10px; cursor: pointer; user-select: none; font-size: 12px; color: var(--muted); }
  .toggle:hover { color: var(--text); }
  .toggle.on { background: rgba(240, 138, 138, 0.18); color: var(--bad); border-color: rgba(240, 138, 138, 0.4); }
  .toggle .dot { width: 8px; height: 8px; border-radius: 999px; background: var(--muted); }
  .toggle.on .dot { background: var(--bad); }
  .session { padding: 12px 16px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .session:hover { background: var(--panel-2); }
  .session.active { background: var(--panel-2); border-left: 3px solid var(--accent); padding-left: 13px; }
  .session .id { font-family: "SF Mono", Menlo, monospace; font-size: 11.5px; color: var(--muted); word-break: break-all; }
  .session .meta { margin-top: 4px; font-size: 12px; color: var(--muted); display: flex; gap: 10px; }
  .session .meta .flagged { color: var(--bad); }
  .session .meta .count { color: var(--text); }
  .content-head { padding: 20px 28px 8px; border-bottom: 1px solid var(--border); }
  .content-head h2 { margin: 0; font-size: 15px; font-weight: 600; }
  .content-head .sid { font-family: "SF Mono", Menlo, monospace; color: var(--muted); font-size: 12px; margin-top: 4px; word-break: break-all; }
  .records { padding: 16px 28px 80px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
  .card.flag { border-color: #5a2727; }
  .card.err { border-color: #5a4a27; }
  .card.goal { border-color: #2c4a5a; background: #131a22; }
  .card.goal-err { border-color: #5a4a27; background: #131a22; }
  .card.goal .head .tool { color: #88b6d9; }
  .card.journal { border-color: #3a4a32; background: #141a14; }
  .card.journal-err { border-color: #5a4a27; background: #141a14; }
  .card.journal .head .tool { color: #a4c98a; }
  .journal-diff { padding: 10px 14px; }
  .journal-diff .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .journal-diff ul { margin: 0; padding-left: 18px; }
  .journal-diff li { font-size: 12.5px; line-height: 1.6; }
  .journal-diff li.created { color: #6fcf97; }
  .journal-diff li.dropped { color: var(--bad); }
  .journal-diff li.status { color: var(--warn); }
  .journal-skipped { padding: 8px 14px; color: var(--muted); font-size: 12px; font-style: italic; }
  .tabbar { display: flex; gap: 4px; padding: 8px 12px 0; border-bottom: 1px solid var(--border); background: var(--panel); }
  .tabbar .tab { padding: 7px 14px; cursor: pointer; color: var(--muted); border-radius: 6px 6px 0 0; user-select: none; }
  .tabbar .tab:hover { color: var(--text); }
  .tabbar .tab.active { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-bottom: 1px solid var(--bg); margin-bottom: -1px; }
  .proj-card { padding: 12px 16px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .proj-card:hover { background: var(--panel-2); }
  .proj-card.active { background: var(--panel-2); border-left: 3px solid #a4c98a; padding-left: 13px; }
  .proj-card .cwd { font-family: "SF Mono", Menlo, monospace; font-size: 12px; color: var(--text); word-break: break-all; }
  .proj-card .meta { margin-top: 4px; font-size: 12px; color: var(--muted); display: flex; gap: 10px; flex-wrap: wrap; }
  .proj-section { margin: 22px 28px; }
  .proj-section h3 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 10px; }
  .proj-item { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
  .proj-item .title { font-weight: 600; }
  .proj-item .body { color: var(--text); margin-top: 4px; font-size: 13px; line-height: 1.5; }
  .proj-item .evidence { color: var(--muted); font-style: italic; margin-top: 6px; font-size: 12px; border-left: 2px solid var(--border); padding-left: 8px; }
  .proj-item .provenance { color: var(--muted); font-size: 11px; margin-top: 4px; font-family: "SF Mono", Menlo, monospace; }
  .proj-item.quarantine { border-color: #5a4a27; }
  .proj-item.quarantine .status-pill { color: var(--warn); }
  .status-pill { display: inline-block; padding: 1px 7px; border-radius: 999px; background: rgba(138,180,248,0.12); color: var(--accent); font-size: 11px; margin-left: 6px; }
  .files-list { font-family: "SF Mono", Menlo, monospace; font-size: 11.5px; color: var(--muted); margin-top: 4px; }
  .health-banner { background: rgba(240, 198, 116, 0.12); border-bottom: 1px solid rgba(240, 198, 116, 0.4); color: var(--warn); padding: 10px 16px; font-size: 12px; line-height: 1.5; }
  .health-banner code { background: rgba(0,0,0,0.3); padding: 1px 5px; border-radius: 3px; font-family: "SF Mono", Menlo, monospace; }
  .proj-item .actions { margin-top: 8px; display: flex; gap: 6px; }
  .proj-item button { background: transparent; border: 1px solid var(--border); color: var(--muted); padding: 3px 10px; border-radius: 4px; font-size: 11px; cursor: pointer; }
  .proj-item button:hover { color: var(--text); border-color: var(--muted); }
  .proj-item button.promote { color: #a4c98a; border-color: rgba(164,201,138,0.35); }
  .proj-item button.promote:hover { background: rgba(164,201,138,0.12); }
  .proj-item button.revoke { color: var(--bad); border-color: rgba(240,138,138,0.3); }
  .proj-item button.revoke:hover { background: rgba(240,138,138,0.12); }
  .goal-block { padding: 12px 14px; }
  .goal-block .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .goal-block .body { font: 13px/1.5 -apple-system, sans-serif; white-space: pre-wrap; }
  .goal-block .body.prior { color: var(--muted); font-size: 12.5px; }
  .goal-stats { padding: 8px 14px; color: var(--muted); font-size: 12px; display: flex; gap: 14px; border-bottom: 1px solid var(--border); }
  .card .head { padding: 10px 14px; display: flex; align-items: center; gap: 12px; border-bottom: 1px solid var(--border); }
  .card .head .tool { font-weight: 600; }
  .card .head .ts { color: var(--muted); font-size: 12px; }
  .card .head .lat { color: var(--muted); font-size: 12px; margin-left: auto; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  .badge.good { background: rgba(111, 207, 151, 0.15); color: var(--good); }
  .badge.bad { background: rgba(240, 138, 138, 0.18); color: var(--bad); }
  .badge.warn { background: rgba(240, 198, 116, 0.18); color: var(--warn); }
  .verdict-msg { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  .verdict-msg.reason { background: rgba(240, 138, 138, 0.06); }
  .verdict-msg.error { background: rgba(240, 198, 116, 0.06); color: var(--warn); }
  .verdict-msg b { color: var(--muted); font-weight: 500; }
  details { border-bottom: 1px solid var(--border); }
  details:last-child { border-bottom: none; }
  details summary { padding: 9px 14px; cursor: pointer; color: var(--muted); list-style: none; user-select: none; }
  details summary::-webkit-details-marker { display: none; }
  details summary::before { content: "▸"; display: inline-block; margin-right: 8px; transition: transform 0.15s; }
  details[open] summary::before { transform: rotate(90deg); }
  details summary:hover { color: var(--text); }
  details > pre { margin: 0; padding: 12px 14px 14px; background: #0b0d12; max-height: 480px; overflow: auto; white-space: pre-wrap; word-break: break-word; }
  .empty { color: var(--muted); padding: 40px 28px; text-align: center; }
  .load-more { display: flex; justify-content: center; padding: 16px 28px 40px; }
  .load-more button { background: var(--panel); border: 1px solid var(--border); color: var(--text); padding: 8px 24px; border-radius: 6px; cursor: pointer; font: 13px/1.4 -apple-system, sans-serif; }
  .load-more button:hover { background: var(--panel-2); border-color: var(--accent); }
  .load-more .count { color: var(--muted); font-size: 12px; margin-left: 10px; }
  .sysprompt-link { font-family: "SF Mono", Menlo, monospace; font-size: 11.5px; color: var(--muted); }
  .sysprompt-link a { color: var(--accent); }
  #modal { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: none; align-items: center; justify-content: center; padding: 40px; z-index: 100; }
  #modal.open { display: flex; }
  #modal .box { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; max-width: 900px; width: 100%; max-height: 80vh; display: flex; flex-direction: column; }
  #modal .box header { padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  #modal .box header h3 { margin: 0; font-size: 14px; }
  #modal .box header button { background: transparent; border: 1px solid var(--border); color: var(--text); padding: 4px 12px; border-radius: 4px; cursor: pointer; }
  #modal .box pre { margin: 0; padding: 18px; overflow: auto; white-space: pre-wrap; }
</style>
</head>
<body>
<div class="layout">
  <aside>
    <div class="topbar">
      <h1>gadfly</h1>
      <span class="toggle" id="flagged-toggle" onclick="toggleFlaggedOnly()" title="Show only unprofessional verdicts">
        <span class="dot"></span><span>flagged only</span>
      </span>
      <small id="refresh-indicator">·</small>
    </div>
    <div class="tabbar">
      <div class="tab active" id="tab-sessions" onclick="setView('sessions')">Sessions</div>
      <div class="tab" id="tab-projects" onclick="setView('projects')">Projects</div>
    </div>
    <div id="health-banner" style="display:none"></div>
    <div id="sessions"></div>
    <div id="projects" style="display:none"></div>
  </aside>
  <main>
    <div class="content-head">
      <h2 id="session-title">select a session</h2>
      <div class="sid" id="session-sid"></div>
    </div>
    <div class="records" id="records"></div>
  </main>
</div>

<div id="modal" onclick="if(event.target===this) this.classList.remove('open')">
  <div class="box">
    <header>
      <h3 id="modal-title"></h3>
      <button onclick="document.getElementById('modal').classList.remove('open')">close</button>
    </header>
    <pre id="modal-body"></pre>
  </div>
</div>

<script>
let currentSession = null;
let sessions = [];
let currentRecords = [];
let currentTotal = 0;
let currentView = "sessions";
let projects = [];
let currentProject = null;
let lastProjectsSig = "";
let flaggedOnly = localStorage.getItem("gadfly.flaggedOnly") === "1";
// Pagination state. We always request the LAST N records from the API
// (newest live at the file's tail). `loadedCount` grows when the user
// clicks "show N more"; refresh re-fetches with the same loadedCount so
// the list size doesn't snap back.
const PAGE_SIZE = 50;
let loadedCount = PAGE_SIZE;
// Track which <details> were open keyed by data-detail-key so we can
// restore them after every re-render (otherwise the 5s auto-refresh
// would slam every panel shut).
let openDetails = new Set();
// Cached signature of last render — skip re-render entirely when nothing
// changed. Cheap and removes 99% of the "snap closed" cases.
let lastSessionsSig = "";
let lastRecordsSig = "";

function toggleFlaggedOnly() {
  flaggedOnly = !flaggedOnly;
  localStorage.setItem("gadfly.flaggedOnly", flaggedOnly ? "1" : "0");
  document.getElementById("flagged-toggle").classList.toggle("on", flaggedOnly);
  renderSessions();
  renderRecords(currentRecords);
  // If the active session is now hidden, switch to the first visible one.
  if (flaggedOnly && currentSession) {
    const visible = sessions.filter(s => s.flagged > 0);
    if (!visible.some(s => s.id === currentSession) && visible.length > 0) {
      loadSession(visible[0].id, true);
    }
  }
}

function fmtTs(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}
function fmtMs(ms) {
  if (ms == null) return "";
  if (ms < 1000) return ms.toFixed(0) + "ms";
  return (ms / 1000).toFixed(1) + "s";
}
function escapeHtml(s) {
  if (s == null) return "";
  return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}
function pretty(v) {
  if (v == null) return "(none)";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch (e) { return String(v); }
}

async function loadSessions() {
  try {
    const r = await fetch("/api/sessions");
    sessions = await r.json();
    renderSessions();
    if (currentSession) {
      // refresh active session content too
      loadSession(currentSession, false);
    } else if (sessions.length > 0) {
      loadSession(sessions[0].id, true);
    }
  } catch (e) {
    console.error(e);
  }
}

function renderSessions() {
  const root = document.getElementById("sessions");
  const visible = flaggedOnly ? sessions.filter(s => s.flagged > 0) : sessions;
  const sig = JSON.stringify({active: currentSession, flagged: flaggedOnly, list: visible});
  if (sig === lastSessionsSig) return;
  lastSessionsSig = sig;
  if (visible.length === 0) {
    root.innerHTML = '<div class="empty">' + (flaggedOnly
      ? 'no flagged verdicts — everything looks professional'
      : 'no sessions yet — run claude code with the hook wired in') + '</div>';
    return;
  }
  root.innerHTML = visible.map(s => `
    <div class="session ${currentSession===s.id?'active':''}" onclick="loadSession('${s.id}', true)">
      <div class="id">${escapeHtml(s.id)}</div>
      <div class="meta">
        <span class="count">${s.count} verdicts</span>
        ${s.flagged > 0 ? `<span class="flagged">${s.flagged} flagged</span>` : ''}
        <span>${escapeHtml(s.last_tool || '')}</span>
      </div>
      <div class="meta">
        <span>${fmtTs(s.latest_ts)}</span>
      </div>
    </div>
  `).join("");
}

async function loadSession(id, switchTo) {
  if (switchTo) {
    currentSession = id;
    loadedCount = PAGE_SIZE;  // reset pagination when switching sessions
    lastRecordsSig = "";
    renderSessions();
    document.getElementById("session-title").textContent = "session";
    document.getElementById("session-sid").textContent = id;
  }
  try {
    const r = await fetch(
      "/api/session/" + encodeURIComponent(id) + "?limit=" + loadedCount
    );
    const payload = await r.json();
    // Backward-compat: very old endpoint format was a bare list.
    if (Array.isArray(payload)) {
      currentRecords = payload;
      currentTotal = payload.length;
    } else {
      currentRecords = payload.records || [];
      currentTotal = payload.total || currentRecords.length;
    }
    renderRecords(currentRecords);
  } catch (e) {
    console.error(e);
  }
}

function loadMore() {
  loadedCount += PAGE_SIZE;
  lastRecordsSig = "";  // force re-render
  if (currentSession) loadSession(currentSession, false);
}

function detailKey(r, kind) {
  return (r.ts || 0) + "::" + kind;
}
function openAttr(r, kind) {
  return openDetails.has(detailKey(r, kind)) ? " open" : "";
}

function renderRecords(records) {
  const root = document.getElementById("records");
  if (!records) { root.innerHTML = ""; return; }
  const visible = flaggedOnly
    ? records.filter(r =>
        (r.verdict && r.verdict.professional === false)
        || (r.type === "goal_distill" && r.error)  // failed goal events stay visible
        || (r.type === "journal_update" && (r.error || r.skipped_reason))
      )
    : records;
  const sig = JSON.stringify({sid: currentSession, flagged: flaggedOnly, list: visible, total: currentTotal, loaded: loadedCount});
  if (sig === lastRecordsSig) return;
  lastRecordsSig = sig;
  if (visible.length === 0) {
    root.innerHTML = '<div class="empty">' + (flaggedOnly
      ? 'no flagged verdicts in this session'
      : 'no records') + '</div>';
    return;
  }
  // Newest first
  const newestFirst = visible.slice().reverse();
  const cardsHtml = newestFirst.map(r => {
    if (r.type === "goal_distill") return renderGoalEvent(r);
    if (r.type === "journal_update") return renderJournalEvent(r);
    const v = r.verdict || {};
    const pro = v.professional;
    const flagged = pro === false;
    const hasErr = !!r.error;
    const cls = flagged ? "card flag" : (hasErr ? "card err" : "card");
    const badge = flagged
      ? '<span class="badge bad">unprofessional</span>'
      : (hasErr ? '<span class="badge warn">error</span>' : '<span class="badge good">ok</span>');
    return `
      <div class="${cls}">
        <div class="head">
          <span class="tool">${escapeHtml(r.tool_name || '?')}</span>
          ${badge}
          <span class="ts">${fmtTs(r.ts)}</span>
          <span class="lat">${fmtMs(r.latency_ms)}</span>
        </div>
        ${flagged && v.reason ? `<div class="verdict-msg reason"><b>reason:</b> ${escapeHtml(v.reason)}</div>` : ''}
        ${flagged && v.suggestion ? `<div class="verdict-msg reason"><b>suggestion:</b> ${escapeHtml(v.suggestion)}</div>` : ''}
        ${hasErr ? `<div class="verdict-msg error"><b>error:</b> ${escapeHtml(r.error)}</div>` : ''}
        <details data-key="${detailKey(r, 'tool_input')}"${openAttr(r, 'tool_input')}>
          <summary>tool input</summary>
          <pre>${escapeHtml(pretty(r.payload?.tool_input ?? r.tool_input_digest))}</pre>
        </details>
        <details data-key="${detailKey(r, 'tool_response')}"${openAttr(r, 'tool_response')}>
          <summary>tool response</summary>
          <pre>${escapeHtml(pretty(r.payload?.tool_response))}</pre>
        </details>
        <details data-key="${detailKey(r, 'prompt')}"${openAttr(r, 'prompt')}>
          <summary>prompt sent to Haiku</summary>
          <pre>${escapeHtml(r.user_message || '(not recorded)')}</pre>
        </details>
        <details data-key="${detailKey(r, 'payload')}"${openAttr(r, 'payload')}>
          <summary>raw payload from Claude Code</summary>
          <pre>${escapeHtml(pretty(r.payload))}</pre>
        </details>
        <details data-key="${detailKey(r, 'sysprompt')}"${openAttr(r, 'sysprompt')}>
          <summary>system prompt <span class="sysprompt-link">sha=${escapeHtml(r.system_prompt_sha || '?')}</span></summary>
          <pre id="sp-${r.ts}"><button onclick="loadSysPrompt('${escapeHtml(r.system_prompt_sha || '')}', 'sp-${r.ts}')">load</button></pre>
        </details>
      </div>
    `;
  }).join("");

  // Pagination footer. Show "load more" when the audit log has older
  // records the user hasn't fetched yet. Note: `currentTotal` is the
  // *unfiltered* count from the API; when `flaggedOnly` is on, the gap
  // between `loadedCount` and `currentTotal` may still reflect entries
  // that wouldn't pass the filter, but loading them is still useful in
  // case more flagged ones live further back.
  let footerHtml = "";
  if (currentTotal > loadedCount) {
    const remaining = currentTotal - loadedCount;
    const nextStep = Math.min(PAGE_SIZE, remaining);
    footerHtml = `
      <div class="load-more">
        <button onclick="loadMore()">show ${nextStep} more</button>
        <span class="count">${loadedCount} of ${currentTotal} loaded · ${remaining} older still on disk</span>
      </div>
    `;
  } else if (currentTotal > PAGE_SIZE) {
    footerHtml = `<div class="load-more"><span class="count">all ${currentTotal} records loaded</span></div>`;
  }
  root.innerHTML = cardsHtml + footerHtml;
}

function renderGoalEvent(r) {
  const failed = !!r.error;
  const cls = failed ? "card goal-err" : "card goal";
  const badge = failed
    ? '<span class="badge warn">goal failed</span>'
    : '<span class="badge" style="background:rgba(136,182,217,0.18);color:#88b6d9;">goal updated</span>';
  const stats = [
    `pairs total: ${r.pairs_total ?? '?'}`,
    `new: ${r.pairs_new ?? 0}`,
    `cached: ${r.pairs_cached ?? 0}`,
  ].join(' · ');
  return `
    <div class="${cls}">
      <div class="head">
        <span class="tool">goal distill</span>
        ${badge}
        <span class="ts">${fmtTs(r.ts)}</span>
        <span class="lat">${fmtMs(r.latency_ms)}</span>
      </div>
      <div class="goal-stats">${escapeHtml(stats)}</div>
      ${failed ? `<div class="verdict-msg error"><b>error:</b> ${escapeHtml(r.error)}</div>` : ''}
      ${r.prior_goal ? `
        <div class="goal-block">
          <div class="label">previous goal</div>
          <div class="body prior">${escapeHtml(r.prior_goal)}</div>
        </div>` : ''}
      ${r.goal ? `
        <div class="goal-block">
          <div class="label">${r.prior_goal ? 'updated goal' : 'distilled goal'}</div>
          <div class="body">${escapeHtml(r.goal)}</div>
        </div>` : ''}
    </div>
  `;
}

function diffLineClass(line) {
  if (!line) return "";
  if (line.startsWith("created ")) return "created";
  if (line.startsWith("dropped ")) return "dropped";
  if (line.startsWith("poison:")) return "dropped";
  if (line.startsWith("refused")) return "dropped";
  if (line.includes("→") || line.includes(".flags ")) return "status";
  return "";
}

function renderJournalEvent(r) {
  const failed = !!r.error;
  const skipped = !!r.skipped_reason;
  const cls = failed ? "card journal-err" : "card journal";
  let badge;
  if (failed) {
    badge = '<span class="badge warn">journal failed</span>';
  } else if (skipped) {
    badge = '<span class="badge" style="background:rgba(138,147,166,0.18);color:var(--muted);">journal skipped</span>';
  } else {
    badge = '<span class="badge" style="background:rgba(164,201,138,0.18);color:#a4c98a;">journal updated</span>';
  }
  const diff = Array.isArray(r.diff_summary) ? r.diff_summary : [];
  const diffHtml = diff.length === 0
    ? ''
    : `
      <div class="journal-diff">
        <div class="label">changes</div>
        <ul>
          ${diff.map(d => `<li class="${diffLineClass(d)}">${escapeHtml(d)}</li>`).join('')}
        </ul>
      </div>`;
  const skippedHtml = skipped && !failed
    ? `<div class="journal-skipped">${escapeHtml(r.skipped_reason)}</div>`
    : '';
  const errHtml = failed
    ? `<div class="verdict-msg error"><b>error:</b> ${escapeHtml(r.error)}</div>`
    : '';
  const newSha = r.new_journal_sha || '';
  const priorSha = r.prior_journal_sha || '';
  return `
    <div class="${cls}">
      <div class="head">
        <span class="tool">journal #${r.action_index ?? '?'}</span>
        ${badge}
        <span class="ts">${fmtTs(r.ts)}</span>
        <span class="lat">${fmtMs(r.latency_ms)}</span>
      </div>
      ${diffHtml}
      ${skippedHtml}
      ${errHtml}
      ${newSha ? `
        <details data-key="${detailKey(r, 'journal-new')}"${openAttr(r, 'journal-new')}>
          <summary>journal after this update <span class="sysprompt-link">sha=${escapeHtml(newSha)}</span></summary>
          <pre id="jn-${r.ts}"><button onclick="loadJournalSnap('${escapeHtml(newSha)}', 'jn-${r.ts}')">load</button></pre>
        </details>` : ''}
      ${priorSha ? `
        <details data-key="${detailKey(r, 'journal-prior')}"${openAttr(r, 'journal-prior')}>
          <summary>journal before this update <span class="sysprompt-link">sha=${escapeHtml(priorSha)}</span></summary>
          <pre id="jp-${r.ts}"><button onclick="loadJournalSnap('${escapeHtml(priorSha)}', 'jp-${r.ts}')">load</button></pre>
        </details>` : ''}
    </div>
  `;
}

async function promoteFinding(fid) {
  const cwd = projects.find(p => p.cwd_encoded === currentProject)?.cwd;
  if (!cwd) { alert("project cwd unknown"); return; }
  if (!confirm("Promote finding " + fid + " out of quarantine?")) return;
  const r = await fetch("/api/project/promote", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({cwd, id: fid}),
  });
  const data = await r.json();
  if (!data.ok) { alert("promote failed: " + (data.message || data.error)); return; }
  loadProject(currentProject);
}

async function revokeFinding(fid) {
  const cwd = projects.find(p => p.cwd_encoded === currentProject)?.cwd;
  if (!cwd) { alert("project cwd unknown"); return; }
  const reason = prompt("Revoke finding " + fid + ". Reason (optional):", "") ?? "";
  const r = await fetch("/api/project/revoke", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({cwd, id: fid, reason}),
  });
  const data = await r.json();
  if (!data.ok) { alert("revoke failed: " + (data.message || data.error)); return; }
  loadProject(currentProject);
}

async function loadJournalSnap(sha, targetId) {
  if (!sha) return;
  const r = await fetch("/api/journal_snapshot/" + encodeURIComponent(sha));
  const text = await r.text();
  const el = document.getElementById(targetId);
  if (el) el.textContent = text;
}

function setView(view) {
  currentView = view;
  document.getElementById("tab-sessions").classList.toggle("active", view === "sessions");
  document.getElementById("tab-projects").classList.toggle("active", view === "projects");
  document.getElementById("sessions").style.display = view === "sessions" ? "" : "none";
  document.getElementById("projects").style.display = view === "projects" ? "" : "none";
  if (view === "projects") {
    loadProjects();
  } else {
    loadSessions();
  }
}

async function loadProjects() {
  try {
    const r = await fetch("/api/projects");
    projects = await r.json();
    renderProjects();
    if (!currentProject && projects.length > 0) {
      loadProject(projects[0].cwd_encoded);
    } else if (currentProject) {
      loadProject(currentProject);
    }
  } catch (e) {
    console.error(e);
  }
}

function renderProjects() {
  const sig = JSON.stringify({active: currentProject, list: projects});
  if (sig === lastProjectsSig) return;
  lastProjectsSig = sig;
  const root = document.getElementById("projects");
  if (projects.length === 0) {
    root.innerHTML = '<div class="empty">no project corpus yet — run the historian (`python -m gadfly.historian sweep`)</div>';
    return;
  }
  root.innerHTML = projects.map(p => `
    <div class="proj-card ${currentProject === p.cwd_encoded ? 'active' : ''}" onclick="loadProject('${escapeHtml(p.cwd_encoded)}')">
      <div class="cwd">${escapeHtml(p.cwd || p.cwd_encoded)}</div>
      <div class="meta">
        <span>${p.promises} promises</span>
        <span>${p.corrections} corrections</span>
        <span>${p.subsystems} subsystems</span>
        ${p.quarantine > 0 ? `<span style="color:var(--warn)">${p.quarantine} quarantine</span>` : ''}
        <span>${p.digested} sessions digested</span>
      </div>
    </div>
  `).join("");
}

async function loadProject(cwdEncoded) {
  currentProject = cwdEncoded;
  renderProjects();
  document.getElementById("session-title").textContent = "project memory";
  document.getElementById("session-sid").textContent = cwdEncoded;
  try {
    const r = await fetch("/api/project/" + encodeURIComponent(cwdEncoded));
    const data = await r.json();
    renderProject(data);
  } catch (e) {
    console.error(e);
  }
}

function renderProject(data) {
  const root = document.getElementById("records");
  if (data.missing) {
    root.innerHTML = '<div class="empty">project missing</div>';
    return;
  }
  const sections = [];

  // Promises
  const promises = Object.values(data.promises || {});
  if (promises.length > 0) {
    sections.push(`
      <div class="proj-section">
        <h3>Open Promises (${promises.length})</h3>
        ${promises.map(p => renderPromise(p)).join('')}
      </div>`);
  }

  // Active corrections
  const corrections = Object.values(data.corrections || {});
  if (corrections.length > 0) {
    sections.push(`
      <div class="proj-section">
        <h3>Active Corrections (${corrections.length})</h3>
        ${corrections.map(c => renderCorrection(c, false)).join('')}
      </div>`);
  }

  // Subsystems
  const subsystems = Object.values(data.subsystems || {});
  if (subsystems.length > 0) {
    sections.push(`
      <div class="proj-section">
        <h3>Subsystems (${subsystems.length})</h3>
        ${subsystems.map(s => renderSubsystem(s)).join('')}
      </div>`);
  }

  // Quarantine
  const quarantine = data.quarantine || [];
  if (quarantine.length > 0) {
    sections.push(`
      <div class="proj-section">
        <h3>Quarantine — pending promotion (${quarantine.length})</h3>
        ${quarantine.map(q => renderQuarantine(q)).join('')}
      </div>`);
  }

  if (sections.length === 0) {
    sections.push('<div class="empty">no semantic findings yet — run the historian against this cwd</div>');
  }

  root.innerHTML = sections.join('');
}

function actionButtons(fid, opts) {
  const buttons = [];
  if (opts.promote) {
    buttons.push(`<button class="promote" onclick="promoteFinding('${escapeHtml(fid)}')">promote</button>`);
  }
  if (opts.revoke) {
    buttons.push(`<button class="revoke" onclick="revokeFinding('${escapeHtml(fid)}')">revoke</button>`);
  }
  return buttons.length ? `<div class="actions">${buttons.join('')}</div>` : '';
}

function renderPromise(p) {
  const prov = p.provenance || {};
  const statusColor = p.status === "fulfilled" ? "var(--good)" : p.status === "aged-out" ? "var(--muted)" : "var(--accent)";
  return `
    <div class="proj-item">
      <div class="title">${escapeHtml(p.title || '?')} <span class="status-pill" style="color:${statusColor}">${escapeHtml(p.status || 'open')}</span></div>
      <div class="evidence">${escapeHtml(prov.evidence_quote || '')}</div>
      <div class="provenance">session ${escapeHtml(prov.source_session || '?').slice(0, 12)} · action #${prov.source_action_index || '?'}${p.fulfilled_in_session ? ' · fulfilled in ' + escapeHtml(p.fulfilled_in_session).slice(0,12) : ''}</div>
      ${actionButtons(p.id || '', {revoke: true})}
    </div>`;
}

function renderCorrection(c, isPending) {
  const prov = c.provenance || {};
  const seen = (c.seen_in_sessions || []).length;
  return `
    <div class="proj-item ${isPending ? 'quarantine' : ''}">
      <div class="title">${escapeHtml(c.rule || '?')}${isPending ? '<span class="status-pill">pending</span>' : '<span class="status-pill">active (conf ' + (c.confidence || seen || 1) + ')</span>'}</div>
      ${c.why ? `<div class="body"><b>why:</b> ${escapeHtml(c.why)}</div>` : ''}
      ${c.how_to_apply ? `<div class="body"><b>how:</b> ${escapeHtml(c.how_to_apply)}</div>` : ''}
      <div class="evidence">${escapeHtml(prov.evidence_quote || '')}</div>
      <div class="provenance">seen in ${seen || 1} session(s) · usefulness=${c.usefulness_score || 0}</div>
      ${actionButtons(c.id || '', {promote: isPending, revoke: true})}
    </div>`;
}

function renderSubsystem(s) {
  const prov = s.provenance || {};
  const files = (s.files || []).join(', ');
  return `
    <div class="proj-item">
      <div class="title">${escapeHtml(s.title || '?')} <span class="status-pill">${(s.files || []).length} files</span></div>
      ${s.purpose ? `<div class="body">${escapeHtml(s.purpose)}</div>` : ''}
      ${files ? `<div class="files-list">${escapeHtml(files)}</div>` : ''}
      <div class="evidence">${escapeHtml(prov.evidence_quote || '')}</div>
      <div class="provenance">last touched in ${escapeHtml(s.last_touched_session || '?').slice(0,12)}</div>
      ${actionButtons(s.id || '', {revoke: true})}
    </div>`;
}

function renderQuarantine(q) {
  if (q.kind === "correction") {
    return renderCorrection({...q.payload, provenance: q.provenance, seen_in_sessions: q.seen_in_sessions}, true);
  }
  if (q.kind === "subsystem") {
    const fid = q.payload?.id || '';
    return `
      <div class="proj-item quarantine">
        <div class="title">${escapeHtml(q.payload?.title || '?')} <span class="status-pill">pending subsystem</span></div>
        ${q.payload?.purpose ? `<div class="body">${escapeHtml(q.payload.purpose)}</div>` : ''}
        <div class="files-list">files mentioned: ${escapeHtml((q.file_mentions || []).join(', '))}</div>
        <div class="evidence">${escapeHtml(q.provenance?.evidence_quote || '')}</div>
        <div class="provenance">seen in ${(q.seen_in_sessions || []).length} session(s)</div>
        ${actionButtons(fid, {promote: true, revoke: true})}
      </div>`;
  }
  return `<div class="proj-item quarantine"><div class="title">${escapeHtml(q.kind)}</div></div>`;
}

async function loadSysPrompt(sha, targetId) {
  if (!sha) return;
  const r = await fetch("/api/system_prompt/" + encodeURIComponent(sha));
  const text = await r.text();
  const el = document.getElementById(targetId);
  if (el) el.textContent = text;
}

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const data = await r.json();
    const banner = document.getElementById("health-banner");
    if (data.daemon_stale) {
      banner.style.display = "";
      const ageMin = data.newest_state_mtime > 0
        ? Math.floor((Date.now() / 1000 - data.newest_state_mtime) / 60)
        : null;
      const lastTxt = ageMin === null ? "never" : ageMin + " min ago";
      banner.innerHTML =
        `⚠️ Daemon appears stale: ${data.active_heartbeats} active session(s) ` +
        `but last digest was ${escapeHtml(lastTxt)}. Start the historian: ` +
        `<code>python -m gadfly.historian watch &</code>`;
    } else {
      banner.style.display = "none";
    }
  } catch (e) { /* health is best-effort */ }
}

function tickRefresh() {
  const ind = document.getElementById("refresh-indicator");
  ind.textContent = "refreshing…";
  refreshHealth();
  loadSessions().finally(() => {
    setTimeout(() => { ind.textContent = "auto-refresh 5s"; }, 200);
  });
}

// Restore toggle state visually before first render.
if (flaggedOnly) document.getElementById("flagged-toggle").classList.add("on");

// Delegated handler — every <details> in the records pane reports its
// open/close state into the openDetails set so the next re-render keeps it.
document.getElementById("records").addEventListener("toggle", (e) => {
  const d = e.target;
  if (!(d instanceof HTMLElement) || d.tagName !== "DETAILS") return;
  const key = d.getAttribute("data-key");
  if (!key) return;
  if (d.open) openDetails.add(key); else openDetails.delete(key);
}, true);

refreshHealth();
loadSessions().then(() => {
  document.getElementById("refresh-indicator").textContent = "auto-refresh 5s";
});
setInterval(tickRefresh, 5000);
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
