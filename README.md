# gadfly

Watchdog over Claude Code sessions. After every `Edit` / `Write` / `MultiEdit` /
`Bash` action the agent takes, a `PostToolUse` hook calls Haiku (via the
Anthropic API) and asks one question:

> Is this action a *professional* approach to the task, or a corner-cutting
> shortcut?

If Haiku flags the action as unprofessional, gadfly returns a warning back to
the Claude Code session via `hookSpecificOutput.additionalContext`, and the
agent sees it inside the same turn — typically prompting it to redo the work
properly.

If Haiku says it's fine, gadfly stays silent. Silence is the default.

## Install

```bash
cd /path/to/gadfly
uv sync
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Wire into Claude Code

Add this to `~/.claude/settings.json` (global) or `.claude/settings.json`
(per project):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit|Bash",
        "hooks": [
          {
            "type": "command",
            "command": "/absolute/path/to/gadfly/.venv/bin/python -m gadfly.hook"
          }
        ]
      }
    ]
  }
}
```

See `settings.example.json` for a template. Use the direct `.venv/bin/python`
path — `uv run` adds ~300–800ms of startup overhead per invocation.

## Operate

- **Kill switch**: `export GADFLY_DISABLE=1` — hook exits 0 with no work done.
- **Audit log**: every verdict is appended to
  `~/.claude/gadfly/log/<session-id>.jsonl`, one JSON record per line.
  Override the directory with `GADFLY_LOG_DIR`.
- **Failures are silent**: missing API key, network errors, malformed
  responses — none of these break the parent Claude Code session. They are
  recorded in the audit log and the hook exits 0.

## View the log in your browser

```bash
.venv/bin/python -m gadfly.viewer       # opens http://127.0.0.1:7777 in your browser
# or
.venv/bin/gadfly-view --no-browser      # just serve, don't auto-open
```

The viewer is stdlib-only (no Flask), reads `~/.claude/gadfly/log/` directly,
and auto-refreshes every 5 seconds. For each verdict you can expand:

- the **tool input** and **tool response** the agent emitted,
- the **prompt sent to Haiku** (exactly what Haiku saw),
- the **raw payload** from Claude Code,
- the **system prompt** (content-addressed by SHA, click to load).

Sessions with at least one unprofessional verdict are highlighted, and the
record card uses a red border so flagged actions are obvious at a glance.

## Test

```bash
.venv/bin/python -m pytest -q
```

## How it works

```
Claude Code session
      │
      ▼  Edit / Write / MultiEdit / Bash completes
PostToolUse hook
      │
      ▼  python -m gadfly.hook  (reads JSON from stdin)
hook.main
  ├─ session.load(transcript_path)  →  last user request, last assistant text,
  │                                    last few tool actions
  ├─ watchdog.evaluate(...)         →  Haiku 4.5 + tool-use submit_verdict
  │                                    via direct anthropic SDK
  ├─ log.append(...)                →  one JSONL record
  └─ stdout: nothing  OR  {"hookSpecificOutput": {"hookEventName":
                            "PostToolUse", "additionalContext": "..."}}
```
