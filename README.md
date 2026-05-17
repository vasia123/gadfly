# gadfly

A watchdog that catches Claude Code agents the moment they start cutting
corners — stubbing things out, swallowing errors, fixing symptoms instead
of causes, or rationalising shortcuts ("the real X is harder than needed,
let me just use a proxy"). After every action the agent takes, gadfly
asks Haiku 4.5 one question — *is this a professional approach or a
shortcut?* — and if it's a shortcut, the critique is fed back into the
agent's session in the same turn. The agent sees the note and usually
fixes the work right away.

When the agent is being honest, gadfly stays silent. That's the default.

## What you need

- [Claude Code](https://docs.claude.com/en/docs/claude-code) installed and signed in (subscription works — no separate API key needed)
- [uv](https://github.com/astral-sh/uv) for Python deps
- Python 3.11+

## Install

```bash
git clone <this repo>
cd gadfly
uv sync
```

## Hook it into Claude Code

Add this to `~/.claude/settings.json` (global — every session) or
`<project>/.claude/settings.json` (one project only). Merge with whatever
hooks you already have — don't overwrite the whole file:

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

A ready-made template is in `settings.example.json`.

Use the direct `.venv/bin/python` path. Don't substitute `uv run` — it
adds hundreds of milliseconds per invocation and the hook runs on every
single `Edit` / `Write` / `Bash`.

That's it. Start a new Claude Code session (or run any tool in your
existing one) and gadfly is live.

## See what's happening

```bash
.venv/bin/python -m gadfly.viewer
# opens http://127.0.0.1:7777 in your browser
```

The viewer shows every session gadfly has watched, with every verdict
expanded:

- which tool call was evaluated
- the exact reasoning that came before it
- the raw payload Claude Code handed the hook
- the prompt sent to Haiku
- Haiku's verdict (professional or not, reason, suggestion)

Sessions and individual verdicts with unprofessional findings get red
highlights. There's a **"flagged only"** filter in the top bar — flip it
on to see just the cases gadfly caught.

It auto-refreshes every 5 seconds. Leave it open in a tab while you work.

## Performance

The watchdog adds ~10-20 seconds to each Edit / Write / Bash the first
time, ~3-5 seconds after Claude Code's CLI is warm. The latency is the
cost of running Haiku 4.5 against your subscription rather than
requiring a separate API key.

For long turns (the agent is editing a substantial file or running a
long bash command) this is invisible. For a flurry of small actions it's
noticeable. If it bothers you, see the "killswitch" below.

## Turn it off

Temporarily, without removing the hook:

```bash
export GADFLY_DISABLE=1
```

Any session in that shell will skip the watchdog entirely. Unset to
re-enable.

Permanently — just remove the `PostToolUse` block you added to
`settings.json`.

## Where things live

- `~/.claude/gadfly/log/<session-id>.jsonl` — one record per verdict.
  Plain JSONL, safe to inspect with `jq`.
- `~/.claude/gadfly/system_prompts/<sha>.txt` — every system prompt
  gadfly has used, content-addressed so old verdicts stay reproducible
  even after the rubric evolves.
- `~/.claude/gadfly/journal/<session-id>.json` — the per-session
  journal (root goal + active workstreams + flag history).
- `~/.claude/gadfly/project/<cwd-encoded>/` — the project corpus
  (cross-session memory), see below.

You can override the log location with `GADFLY_LOG_DIR=/some/path`.

## Cross-session memory (the historian)

Gadfly keeps two layers of state:

- **In-session journal** — what the agent is working on *right now*.
  Live by default, no setup needed.
- **Cross-session corpus** — durable findings mined from EVERY prior
  Claude Code session you've run in a given project directory.
  Discovers open promises ("I'll do X later" → next session sees it),
  user corrections that hold across sessions, and a knowledge graph
  of subsystems / files. **Opt-in**: code runs by default but the
  daemon that does the heavy lifting is not auto-started.

Set the daemon running once (in a tmux, with `&`, or via systemd-user
unit — see `docs/gadfly-historian.service.example`):

```bash
.venv/bin/python -m gadfly.historian watch &
```

It polls heartbeats every minute, digests each session after 5 min of
silence, and writes findings to
`~/.claude/gadfly/project/<cwd-encoded>/`.

Useful one-shots:

```bash
# What does each project corpus contain right now?
.venv/bin/python -m gadfly.historian status

# Single sweep, then exit. Good for cron / cron-like usage.
.venv/bin/python -m gadfly.historian sweep

# Backfill: digest ALL prior sessions in a project from scratch.
# Expensive — one Haiku call per session.
.venv/bin/python -m gadfly.historian backfill /abs/path/to/project --yes-i-know-the-cost

# Drift recovery: re-derive semantic state from existing raw digests.
# No Haiku calls. Cheap and idempotent.
.venv/bin/python -m gadfly.historian rebuild /abs/path/to/project

# What would I add to my CLAUDE.md based on findings?
.venv/bin/python -m gadfly.historian propose-claudemd /abs/path/to/project

# Force-promote a quarantined finding (lift the repetition gate
# manually — useful when you trust a single-shot user correction).
.venv/bin/python -m gadfly.historian promote /abs/path/to/project <finding-id>

# Revoke a finding (active or quarantined). Moves it to audit log;
# rebuild will continue to honor the revocation. Raw evidence is
# preserved.
.venv/bin/python -m gadfly.historian revoke /abs/path/to/project <finding-id> --reason "..."
```

Finding IDs come from the viewer's Project tab (where promote/revoke
buttons also live), from `propose-claudemd` output, or by reading
`~/.claude/gadfly/project/<cwd-encoded>/state.json` directly.

The viewer (`.venv/bin/python -m gadfly.viewer`) has a **Projects** tab
showing each cwd's promises, corrections, subsystems, and the
quarantine of pending findings waiting for repeat evidence.

### Feature flags (env vars)

The whole memory stack is controlled by these knobs. Defaults are
sensible — set anything to `0` to roll back to a more conservative mode.

| Var | Default | Effect when `0` |
|---|---|---|
| `GADFLY_DISABLE` | `0` | Disables the hook entirely (silent exit). |
| `GADFLY_JOURNAL` | `1` | No per-session journal Haiku call. |
| `GADFLY_JOURNAL_VERDICT` | `1` | Verdict prompt falls back to legacy (no journal-aware repetition rule). |
| `GADFLY_HISTORIAN` | `1` | No heartbeats written; daemon has nothing to do. |
| `GADFLY_HISTORIAN_PRIORS` | `1` | Journal-maintainer doesn't see project priors. |
| `GADFLY_PHASE_C` | `1` | Priors not surfaced to the agent on new-workstream events. |

## What it catches

Examples of what gadfly flags as unprofessional in practice:

- `git commit --no-verify` to skip a failing pre-commit hook
- Deleting a failing test file as a way of "fixing" it
- `if x is None: x = default` as a guard instead of fixing why x came None
- `try: ... except: pass` without a written-down reason
- `time.sleep()` to paper over a race condition
- Replacing a real component with a mock "because the real one is harder
  than needed", without acknowledging the trade-off

And what it stays quiet about:

- Style, formatting, naming, micro-optimisations
- One reasonable design choice over another
- Honest exploration: `Read`, `grep`, `git diff`, running tests
- TODOs tracking genuinely out-of-scope follow-up work
- Workarounds the user / agent explicitly acknowledged as workarounds

## For developers

If you want to change the rubric, calibrate against the real model, or
understand the architecture, see `CLAUDE.md`.

Quick start:

```bash
.venv/bin/python -m pytest -q       # unit tests, fully mocked
.venv/bin/python scripts/probe.py   # live calibration against real Haiku
```
