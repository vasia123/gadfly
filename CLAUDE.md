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
  prompts.py    SYSTEM_PROMPT (legacy) + SYSTEM_PROMPT_JOURNAL (Phase 2;
                adds journal-reader, repetition rule, rationalization
                cross-check, drift guidance) + build_user_message —
                switches to journal-mode rendering when a Journal is
                passed.
  watchdog.py   evaluate_async / evaluate — one-shot Haiku call via
                claude-agent-sdk + in-process MCP tool `submit_verdict`.
                Picks SYSTEM_PROMPT_JOURNAL when
                GADFLY_JOURNAL_VERDICT=1 AND ctx.journal is set.
  journal.py    Per-session journal (the agent's worklist as Haiku
                understands it). Workstreams, statuses, flag history,
                drift. update_for_action() runs ONE Haiku call per
                hook. Persistence:
                  ~/.claude/gadfly/journal/<session_id>.json (current)
                  ~/.claude/gadfly/journals/<sha>.json (snapshots)
                See "The journal" section below.
  pairs.py      Pair / clean_user_text / extract_pairs — single source
                of truth for transcript scrubbing.
  project_state.py
                Per-cwd semantic memory: Promise / Correction / Subsystem
                dataclasses, content-addressed cwd encoding (mirrors
                Claude Code's ~/.claude/projects/<encoded>/ layout),
                deterministic re-aggregator. NO Haiku calls — pure
                aggregation. `rebuild_from_raw()` reconstructs semantic
                state from raw episodic alone, byte-identical across
                runs (drift-resistance invariant).
  historian.py  Cross-session "project memory" daemon. Mines every
                jsonl transcript in a cwd's projects/ dir, extracts
                durable findings (promises, corrections, subsystems)
                via one Haiku call per session (extract_findings MCP
                tool), persists to:
                  ~/.claude/gadfly/project/<encoded>/state.json
                  ~/.claude/gadfly/project/<encoded>/raw/<sid>.json
                CLI: python -m gadfly.historian {status,sweep,watch,
                backfill,rebuild,propose-claudemd}. Triggered by
                heartbeat files the hook writes after each PostToolUse;
                a session is "done" after 5 min of no heartbeat.
                Retrieval: find_relevant_priors() — keyword + file-path
                overlap + outcomes-feedback usefulness; top-5 priors
                injected into journal-maintainer prompt as a
                "project_priors" block when GADFLY_HISTORIAN_PRIORS=1
                (default). See "The historian" section below.
  goal.py       Legacy goal distillation. Phase 1 keeps it alongside
                the journal; Phase 3 deletes it once journal proves
                itself in production.
  session.py    Reads transcript_path JSONL, returns SessionContext:
                recent_user_requests (≤5, chronological),
                last_assistant_plan, recent_actions, pairs (full
                chronological list for journal), action_index, and
                journal (populated by hook.py in Phase 2).
  hook.py       PostToolUse entrypoint. python -m gadfly.hook.
                Runs journal.update_for_action() BEFORE
                watchdog.evaluate(), then writes a heartbeat for the
                historian. GADFLY_JOURNAL=0 disables the journal;
                GADFLY_JOURNAL_VERDICT=0 rolls back to legacy verdict
                prompt; GADFLY_HISTORIAN=0 disables heartbeats;
                GADFLY_HISTORIAN_PRIORS=0 disables priors injection.
  verdict.py    Verdict dataclass + to_hook_output()
  log.py        Append-only JSONL audit log + content-addressed
                system prompts AND journal snapshots under
                ~/.claude/gadfly/{system_prompts,journals}/<sha>.
                Record types: verdict, goal_distill, journal_update,
                historian_digest.
  viewer.py     stdlib-only local HTTP server + single-page HTML.
                Renders journal-update events with diff_summary + a
                snapshot view via /api/journal_snapshot/<sha>.
  backends/     Pluggable model transports.
    base.py        Backend Protocol + BackendResult dataclass.
    claude_sdk.py  ClaudeSDKBackend — wraps claude-agent-sdk and the
                   user's Claude Code CLI subscription. Carries the
                   recursion guards (setting_sources=[], settings="{}",
                   env={"GADFLY_INTERNAL":"1"}). Default backend.
    openai_compat.py  OpenAICompatBackend — stdlib urllib client
                   speaking the OpenAI Chat Completions wire protocol.
                   Works against OpenAI, Anthropic's OAI-compat
                   endpoint, OpenRouter, vLLM, llama.cpp, Ollama,
                   Groq, Together, and any self-hosted endpoint.
                   No new deps. Forced tool-choice guarantees
                   structured output.
    __init__.py    select_backend(name, **kwargs) factory.
                Pass `backend=` to evaluate_async/evaluate to swap
                transports; run_corpus.py exposes `--backend`,
                `--base-url`, `--api-key-env`, `--extra-headers`.
                Hook stays on claude_sdk for subscription billing.
scripts/
  probe.py      Live regression probe against real Haiku (no mocks;
                bills the real subscription). CASES A/B exercise the
                legacy prompt; CASES C/D exercise the journal-aware
                repetition rule (gated by GADFLY_JOURNAL_VERDICT=1).
  build_corpus.py
                Mines ~/.claude/gadfly/log/ for flagged verdicts the
                agent acknowledged in transcript text. For each, slices
                the source transcript at the flag point, runs current
                session.load(), and serializes the resulting
                SessionContext + tool_input/response + original verdict
                into tests/fixtures/watchdog_corpus/cases.json.
                Replayable forever, machine-portable (snapshots frozen
                inline). Re-run whenever new ack-worthy flags accumulate.
                Two modes via --mode: `positive` (default; agent acked
                the flag — recall corpus) and `negative` (agent pushed
                back; flag was a watchdog FP — precision corpus, output
                to cases_negative.json). 24 positive, 15 negative.
  run_corpus.py Replays the corpus through evaluate() with a chosen
                model (--model, default Haiku) and backend
                (--backend claude_sdk|openai_compat). For OpenAI-compat:
                --base-url, --api-key-env, --extra-headers. For negative
                corpus: --negative (auto-loads cases_negative.json,
                flips success criterion: silence = success). Prints
                per-case ✓/✗ and aggregate rate; writes JSON via --out.
                Cost ~5 min full positive corpus via subscription.
  compare_corpus.py
                Diffs two run_corpus reports. Reports regressions
                (caught→missed) and improvements (missed→caught).
                Exit 1 when regressions exceed --allow-regressions
                (default 0) — useful in CI for prompt-experiment gates.
tests/          258+ unit tests, all mock the SDK via DI. test_live.py
                is gated by GADFLY_LIVE=1 and covers Haiku integration
                for verdict, goal, and journal flows. test_watchdog_
                corpus.py replays the curated useful-flag corpus and
                asserts the aggregate catch rate stays above
                GADFLY_CORPUS_THRESHOLD (default 0.70).

                Baselines table — recall (positive 24) × precision
                (negative 15 silent-on-FP). F1 is the load-bearing
                summary; recall-only without precision is misleading:

                  Model                       recall  precision  F1
                  claude-haiku-4-5             75.0%   66.7%    0.71  ← rubric calibrated for this
                  google/gemini-3.1-flash-lite-preview
                                               62.5%   73.3%    0.68  ← 4× cheaper than Haiku
                  mistralai/mistral-small-2603 79.2%   53.3%    0.64  ← 6× cheaper; best price/F1
                  openai/gpt-5.4-nano          79.2%   26.7%    0.40  (precision-poor at this tier)
                  openai/gpt-4o-mini          100.0%   13.3%    0.24  (recall trap)
                  inclusionai/ling-2.6-1t      79.2%    6.7%    0.12  (worse recall trap)
                  claude-sonnet-4-6            12.5%   (n/a)     —    (over-cautious)

                Implication: gpt-4o-mini's 100% recall is a false
                signal — it also flags 87% of cases where the agent's
                pushback proved the watchdog wrong. The rubric is
                calibrated for Haiku's interpretation; swapping models
                requires per-model prompt re-tuning, not just config.

                40 models total tested. See docs/model_comparison.md
                for the full table + cost column. Beyond the F1-tested
                head, the long tail clusters at 0-5/24 — most models
                default to "professional=true" silently. Strong coding
                leaderboard names (Kimi K2.6, DeepSeek V4 Pro, GLM-5.1,
                gpt-5.1-codex-mini, gpt-oss-120b, Nvidia Nemotron-3
                Super 120B) all score 0/24 — they're trained to be
                helpful/agreeable, not adversarial. The bottleneck
                is *willingness to disagree*, not raw capability.
                The 6 missed cases are mostly STRUCTURAL gaps, not
                prompt bugs: rationalization detection (case_010) and
                scope-pivot detection (case_004) require journal-mode
                / workstream context which the replay deliberately
                disables; case_000 is borderline because the agent
                had already self-corrected before the second flag and
                silence is arguably the right call now. Do not chase
                these with non-rules — verify against journal-mode
                first if you ever want to lift the threshold.
                Note: DEFAULT_MAX_TURNS=2 is Haiku-specific; evaluate()
                and _build_options thread max_turns through and
                run_corpus.py exposes --max-turns (default 4) for
                cross-model runs.
```

## The journal (Phase 1 default, Phase 2 opt-in)

The watchdog used to be stateless per hook. Two failure modes drove the
journal design (see also session 7fd6dce9-dcfa-41f9-aec9-105c5b8a54b7
which is the canonical repro):

1. **Edit-window blindness.** A series of 8+ Edits pushed prior code
   changes out of the 5-action window, and a follow-up Write(MEMORY.md)
   got flagged "claimed-as-landed but no Edits" — incorrectly.
2. **Verdict echo chamber.** The watchdog has no memory of its own
   verdicts, so the same flag fires 14 times on the same topic after
   the agent already pushed back.

The journal solves both by maintaining persistent state. It tracks:

- root_goal (subsumes the goal.py distill)
- workstreams: id, title, status, origin, notes, watchdog_flags,
  flag_history (each entry carrying `agent_pushed_back`), last_touched
- drift: initial_workstream_ids (baseline) + observations

Update happens on every PostToolUse via a small Haiku call
(update_journal MCP tool). Steady-state cost is one ~3s call —
incremental: only NEW pairs + the latest action are fed.

### Two phases, two env vars

- `GADFLY_JOURNAL=1` (default): journal runs, but the verdict prompt
  is the legacy `SYSTEM_PROMPT`. Use this for shadow-mode validation:
  open the viewer, scroll the Journal cards, check workstreams match
  what the agent was actually doing across several real sessions.
- `GADFLY_JOURNAL_VERDICT=1` (opt-in, Phase 2): verdict prompt swaps
  to `SYSTEM_PROMPT_JOURNAL`, watchdog reads the journal as primary
  context. Includes the load-bearing repetition rule that solves the
  echo chamber.

### Poisoning guards

- `prompt_sha` invalidates the disk cache when `UPDATE_SYSTEM_PROMPT`
  changes — same pattern as goal cache.
- The update is rejected (carry forward + log into
  `drift.observations`) when it would drop >50% of workstreams or
  rewrite root_goal radically WITHOUT a new user message to justify
  it. A new user message lifts the guard.

### Caps

- `flag_history`: 8 most recent entries per workstream.
- `MAX_NOTES_LEN = 1500`, `MAX_WORKSTREAMS = 20`.
- If the workstream cap is hit, done/abandoned ones drop first.

## The historian (Phase A+B default, Phase C/D opt-in)

The journal lives **within** a session. The historian lives **across**
sessions. Every cwd accumulates a corpus of jsonl transcripts under
`~/.claude/projects/<cwd-encoded>/`; the historian mines that corpus
for durable findings the next session can build on.

Aligned with 2026 agent-memory literature (MemMachine, mem0,
MemoryArena, Reflexion, MemoryGraft):

- **Episodic memory** = `raw/<sid>.json` per-session digests, verbatim
  source of truth. Never overwritten. Aggregator NEVER re-reads its
  own previous output — semantic state is fully re-derivable from
  episodic alone (the drift-resistance invariant).
- **Semantic memory** = `promises.json`, `corrections.json`,
  `subsystems.json` derived from episodic. Three classes:
    - Promise — agent committed to follow-up work; tracks landing.
    - Correction — durable user feedback. Needs ≥2 sessions to leave
      quarantine.
    - Subsystem — knowledge-graph node (group of files + purpose).
      Needs ≥3 distinct file mentions.
- **Provenance + quarantine + revocation** — defends against
  MemoryGraft-style poisoning. Every finding carries
  `source_session`, `source_action_index`, `evidence_quote` ≤200c
  verbatim. Single-shot findings stay in quarantine until repetition
  threshold is met. `revoked.json` is an audit-only soft-delete.

### Trigger model

The hook (PostToolUse) writes a heartbeat at
`~/.claude/gadfly/heartbeat/<cwd-encoded>.tick` after each watched
tool call. **The historian is NEVER on the hot path** — heartbeat
write is atomic-rename, <5ms, swallowed-exceptions.

A separate daemon (`python -m gadfly.historian watch`) polls
heartbeats every 60s. A session is "done" when no heartbeat has
arrived in 5 min; then it gets digested. **Daemon is not
auto-started.** Run it manually (`watch`), or one-shot via `sweep`,
or backfill via `backfill <cwd> --yes-i-know-the-cost`.

### CLI

`python -m gadfly.historian` subcommands:
- `status` — per-cwd corpus summary
- `sweep` — one-shot: digest any sessions that are done + undigested
- `watch` — long-running poll loop
- `backfill <cwd> --yes-i-know-the-cost` — digest ALL prior sessions
  in a cwd (1789 Haiku calls for vllm-rust corpus — expensive)
- `rebuild <cwd>` — re-derive semantic state from existing raw/
  digests. **No Haiku calls.** Cheap drift recovery.
- `propose-claudemd <cwd>` — print candidate CLAUDE.md addition from
  high-confidence corrections + subsystems. Never auto-applies.

### Phase B — priors injection (default-on)

When the journal maintainer opens a new workstream, the hook calls
`historian.find_relevant_priors(workstream_title, file_paths)` and
injects the top-5 hits into the maintainer prompt as a
"Project priors" block. Retrieval signals (fused):
1. keyword overlap (title vs. promise/correction/subsystem titles)
2. file-path overlap (workstream's edited file vs. subsystem.files)
3. usefulness_score (outcomes-feedback bonus)

Default: ON. `GADFLY_HISTORIAN_PRIORS=0` rolls back.

### Phases roadmap

- **Phase A (default-on):** historian + storage + daemon + viewer.
  Validated on real corpus (gadfly project, 15 sessions, real Haiku).
- **Phase B (default-on):** priors injected into journal maintainer.
  Validated end-to-end against real corpus (top-1 retrieval
  precision = correct).
- **Phase C (NOT yet implemented):** additionalContext injection on
  workstream-creation events, so the agent (not just the watchdog)
  sees priors. Risk: false positives → noise. Wait for real-session
  evidence before enabling.
- **Phase D:** `propose-claudemd` CLI — read-only, ships.

## Architectural decisions — and why

These are the load-bearing decisions. Don't undo them without reading why.

### 1. Backend is env-selectable; production currently points at Mistral Small 4 via OpenRouter

Two production paths supported (both implemented as `Backend`s in
`src/gadfly/backends/`):

  - `claude_sdk` — wraps claude-agent-sdk + the user's Claude Code CLI
    subscription. Cold start ~10-20s; subscription billing.
  - `openai_compat` — stdlib `urllib.request` against any
    /v1/chat/completions endpoint. ~3-5s latency.

The hook reads `<gadfly_repo>/.env` at startup (see `_load_env_file_once`
in hook.py) and the watchdog then resolves the backend + model from
these env vars:

    GADFLY_BACKEND       = "openai_compat" | "claude_sdk" (default fallback)
    GADFLY_BASE_URL      = https://openrouter.ai/api/v1
    GADFLY_API_KEY_ENV   = OPENROUTER_API_KEY (name of the env var holding the key)
    GADFLY_MODEL         = mistralai/mistral-small-2603
    GADFLY_EXTRA_HEADERS = optional JSON (e.g. anthropic-version)

Current production .env values (NOT committed): Mistral Small 4 via
OpenRouter. Validation: F1 0.64 in docs/model_comparison.md, ~6× cheaper
than Haiku via OR, latency ~3s typical.

Roll back to subscription-billed Haiku by removing the GADFLY_* lines
from .env (they fall through to ClaudeSDKBackend + claude-haiku-4-5).

**Test-isolation pitfall:** `_load_env_file_once` is called inside
`main()`, not at module import. Otherwise `tests/test_hook.py`'s import
contaminates `os.environ` for every other test (we hit this; see
`test_options_block_recursion` for the `monkeypatch.delenv` guard).

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
