# CLAUDE.md — onboarding for agents working on gadfly

This file exists so the next agent who opens this repo can get oriented in
two minutes. Read the README for user-facing docs; read this for the
context behind why the code is shaped the way it is.

## What this is

A watchdog that supervises a Claude Code session. After each tool call the
agent makes (and via prompt context, the reasoning right before it), a
PostToolUse hook fires, asks Haiku 4.5 "is this a professional approach
or a corner-cutting shortcut?", and either stays silent or feeds the
critique back into the session via `hookSpecificOutput.additionalContext`.

The agent under supervision sees those critiques inside the same turn and
typically corrects course. The watchdog itself is non-blocking; it never
exits non-zero and never breaks the host session.

## Map of the code

```
src/gadfly/
  prompts.py    SYSTEM_PROMPT (the rubric) + build_user_message()
  watchdog.py   evaluate_async / evaluate — one-shot Haiku call via
                claude-agent-sdk + in-process MCP tool `submit_verdict`
  session.py    Reads transcript_path JSONL, returns recent_user_requests
                (up to 5, chronological), last_assistant_plan, recent_actions
                (with mini-diffs of Edit/Write so prior series-edits are
                visible)
  hook.py       PostToolUse entrypoint. python -m gadfly.hook.
  verdict.py    Verdict dataclass + to_hook_output()
  log.py        Append-only JSONL audit log + content-addressed
                system prompts under ~/.claude/gadfly/system_prompts/
  viewer.py     stdlib-only local HTTP server + single-page HTML
scripts/
  probe.py      Live regression probe against real Haiku
                (no mocks; bills the real subscription)
tests/          37+ unit tests, all mock the SDK via DI
```

## Architectural decisions — and why

These are the load-bearing decisions. Don't undo them without reading why.

### 1. Backend is claude-agent-sdk, not the direct `anthropic` package

Earlier in the project we used the direct `anthropic` SDK with an
ANTHROPIC_API_KEY. We switched to `claude-agent-sdk` because the user
prefers to bill against their existing Claude subscription rather than a
separate API key. The trade-off: cold start is ~10-20s per hook
invocation (Claude Code CLI subprocess spin-up) instead of ~100ms for a
direct HTTPS call.

If you ever consider switching back, the user's preference is explicit —
ask first. A future optimisation is a long-lived daemon that keeps one
warm `ClaudeSDKClient` per session and exposes a unix socket the hook
talks to; that gets you both subscription billing AND fast latency.

### 2. Structured output via in-process MCP tool, not text-parsing

Haiku is forced to reply by calling `submit_verdict(professional, reason,
suggestion)` exactly once. We never parse free-form text. The tool is
built inline in `watchdog._build_submit_verdict_tool` and wired into a
one-shot `create_sdk_mcp_server` per evaluation. The handler writes its
arguments into a `_Captured` container that the calling code reads after
the SDK loop finishes.

Why this is important: ~20% of unstructured Haiku replies historically
omitted the call entirely. Even with the current strict prompt, we keep
the tool-use enforcement as defense-in-depth.

### 3. Recursion guard — three independent mechanisms

The watchdog itself runs Claude Code under the hood, so it would
infinitely recurse if the inner CLI picked up the same PostToolUse hook.
We block this three ways:

- `ClaudeAgentOptions(setting_sources=[])` — inner CLI shouldn't read
  `~/.claude/settings.json` at all. **Caveat:** in practice this does
  NOT stop hooks from leaking through (observed: cc-telegram-notify
  Stop/Notification still fired from the inner CLI). So we also need…
- `ClaudeAgentOptions(hooks={})` — explicit empty hooks map handed to
  the SDK, overriding whatever it would otherwise inherit. This is the
  load-bearing one. Don't remove it.
- `ClaudeAgentOptions(env={"GADFLY_INTERNAL": "1"})` — `hook.py` checks
  this env var on startup and exits 0 immediately if set. Belt and braces
  for our OWN PostToolUse hook, even if the first two leak.

All three must stay. Removing any is a footgun.

### 4. The hook NEVER fails

`hook.main()` has a top-level `try/except Exception` that swallows
everything and exits 0. Internal errors are recorded in the audit log so
they're visible in the viewer, but they never propagate. The reason: a
broken watchdog must not break the user's actual Claude Code session.
That includes:

- missing API auth → logged "ANTHROPIC_API_KEY not set" / similar
- claude CLI not on PATH → logged "claude CLI not found"
- Haiku timeout (default 60s) → logged "timeout after Xs"
- Haiku didn't call submit_verdict → logged "Haiku did not call submit_verdict"
- malformed payload → silent exit 0

When you debug a real issue, look at the viewer or grep the JSONL for
`"error":` lines. The hook will not crash to stderr.

### 5. We send context, Haiku doesn't fetch it

Earlier sketches gave Haiku the `Read` / `Bash` tools so it could go look
up files. We dropped that: deterministic, cheap, and Haiku has only one
job — classify. We assemble everything Haiku needs (the action, the tool
response, the recent assistant text, recent prior actions with mini-diffs,
and the trail of recent user messages — not just the last one, since the
latest message is often a clarification that's meaningless without the
earlier goal-setting one) into a single user message in
`prompts.build_user_message` and call once.

If you find yourself wanting Haiku to "go look at X" — instead add X to
`session.SessionContext` and `build_user_message`. Keep the call one-shot.

### 6. The system prompt is the product

`prompts.SYSTEM_PROMPT` is the single biggest determinant of quality.
Treat it like production code:

- Every meaningful change comes with a regression test in
  `tests/test_prompts.py` asserting that the marker text is present
  (we can't unit-test Haiku's behaviour, but we can guarantee the prompt
  still mentions "root cause", "Rationalization:", etc.).
- Every meaningful change comes with a new case in `scripts/probe.py`
  exercising the change against real Haiku — both a positive case (must
  flag) and a control (must stay silent).
- The prompt is content-addressed by SHA in the audit log, so old verdicts
  remain reproducible even after the prompt changes.

Be conservative. The user's default position is "silence beats noise" —
false positives erode trust faster than false negatives. The non-rules
section of the prompt exists for that reason. When in doubt, add a
non-rule, don't add a positive rule.

### 7. The viewer is stdlib-only

`viewer.py` uses `http.server` + `BaseHTTPRequestHandler` and serves a
single inline HTML page. No Flask, no FastAPI, no JS framework, no CDN.
Auto-refresh + `JSON.stringify` signatures avoid re-rendering when
nothing changed (which would otherwise snap every `<details>` shut every
5 seconds). Open details state is tracked in a JS `Set` keyed by
`(ts, kind)` and restored on render.

If you add a new endpoint or section, keep the stdlib-only constraint.
The dependency footprint of this project is `claude-agent-sdk` and that's
it; we want to keep it that way.

## Pitfalls — real footguns we've already hit

### Don't `pkill -f "gadfly.viewer"` from inside a hook-supervised session

The pattern matches the bash command line, and the bash command line
itself contains the string "gadfly.viewer" — so pkill kills its own
parent shell. Use `fuser -k 7777/tcp` instead, or just spawn a fresh
viewer on a different port.

### Don't edit `~/.claude/settings.json` without a `.bak` first

It contains the user's hooks (cc-telegram-notify and ours), MCP servers,
plugin enables. Always copy to `settings.json.bak.gadfly` first, then
merge — never overwrite. Use `Read` + `Edit` (small patches), never
`Write` on the whole file.

### Don't make the hook do extra work — every Edit/Write/Bash pays

Cold start is already ~15s. Any extra import / disk scan / network call
on the hook hot path multiplies user-visible latency. Anything optional
or analytical (calibration scripts, batched analysis) belongs in
`scripts/`, not in `hook.py`.

### Don't trust that the transcript schema is stable

`session.py` parses Claude Code's transcript JSONL defensively because the
schema is undocumented and changes between releases. Every `dict.get`,
every `isinstance` guard there is load-bearing. If you "clean it up",
add tests for every shape you removed support for. There are existing
regression tests at `tests/test_session.py` for tool_result-only
user-messages, malformed lines, missing fields — keep them passing.

## How to develop here

```bash
# Set up
uv sync --extra dev

# Run tests (fast, all mocked)
.venv/bin/python -m pytest -q

# Live probe against real Haiku (slow, costs subscription quota)
.venv/bin/python scripts/probe.py

# Local viewer
.venv/bin/python -m gadfly.viewer

# See what the watchdog is actually flagging right now
ls -lt ~/.claude/gadfly/log/ | head
.venv/bin/python -c "
import json
for line in open('/path/to/session.jsonl'):
    r = json.loads(line)
    v = r.get('verdict', {})
    if v.get('professional') is False:
        print(r['tool_name'], '→', v.get('reason'))
"
```

When a real user-reported false positive or false negative comes up,
the workflow is:

1. Pull the offending payload out of `~/.claude/gadfly/log/<id>.jsonl`.
2. Add it as a case in `scripts/probe.py` with the *expected* verdict.
3. Run `probe.py` — confirm it's reproducible.
4. Edit `prompts.SYSTEM_PROMPT` minimally to fix the case.
5. Add a marker-assertion regression test in `tests/test_prompts.py`.
6. Re-run `probe.py` — confirm the target case AND every other case
   still behave correctly. False positives are very easy to introduce
   when you tighten the prompt; always re-run controls.

## Open / known limitations

- **Cold start ~15s per hook call.** Acceptable for long turns
  (Edit/Bash that already take seconds), painful for chains of small
  actions. Daemon-mode (one warm `ClaudeSDKClient` per session, hook is
  thin unix-socket client) is the obvious next step.
- **Haiku occasionally still replies without calling `submit_verdict`.**
  Logged as error="Haiku did not call submit_verdict". Frequency went
  down a lot after tightening the output-protocol section of
  SYSTEM_PROMPT but isn't zero. A retry-once with a stiff reminder
  would help if it gets bad.
- **Series-of-edits context.** `session.py` includes mini-diffs of the
  last 5 actions, but only of the form (Edit, file, old, new) — it does
  not group by file or accumulate. If the agent does 8 Edits to the same
  file and Haiku evaluates the 8th, it sees only the diff of edit 8 and
  a one-line summary of 3-7 earlier. For larger series, a per-file
  accumulated diff would be better. Open work.
- **No interactive blocking yet.** All verdicts go via additionalContext
  (exit 0). `decision: block` (exit 2) was scoped out for MVP because
  it's much higher-stakes if it misfires.

## What I'd want to know in your shoes

- The user prefers concise, professional responses. They notice and call
  out lazy shortcuts in *me* (the agent), so this is not theoretical
  preference — they actually verify.
- The user reads the audit log and the viewer. When you ship a change to
  the rubric, expect them to look at the next 5-10 verdicts to see if
  the calibration shifted.
- `probe.py` is the source of truth for "does the rubric work". Tests
  in `tests/` only check that the *prompt text* contains the right
  markers — they can't tell whether Haiku will actually behave.
- The plan file at `/home/vasis/.claude/plans/sprightly-crafting-puppy.md`
  was the original design doc. It's been amended in-place several times
  (we switched backends twice, scoped block-mode out, etc.). Don't trust
  it as a current spec; this CLAUDE.md is more current.
