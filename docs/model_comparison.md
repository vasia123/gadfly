# Watchdog model comparison

Cross-model evaluation of the gadfly watchdog rubric. The current
`SYSTEM_PROMPT` is calibrated for Claude Haiku; this table answers
the question "which other models can substitute for Haiku without
degrading the watchdog?"

## Methodology

Two curated corpora replayed through the **current** prompts/stack
via `scripts/run_corpus.py`. Each case feeds the model a single
PostToolUse moment from a real Claude Code session — verbatim tool
input/response plus the SessionContext the watchdog would have seen.

- **Positive corpus** (24 cases, `cases.json`): the watchdog
  originally flagged the action and the agent ACKNOWLEDGED the
  critique in their next message ("Watchdog прав, …"). The model
  is expected to RE-FIRE — measures **recall**.
- **Negative corpus** (15 cases, `cases_negative.json`): the
  watchdog flagged but the agent successfully PUSHED BACK
  ("Watchdog ошибается / врёт / галлюцинирует"). The model is
  expected to STAY SILENT — measures **precision** (low precision
  = false-positive monster).
- **F1** = harmonic mean of recall and precision. Load-bearing
  summary; recall-only is misleading because a "flag everything"
  model trivially hits 100%.

All runs use the legacy `SYSTEM_PROMPT` (`GADFLY_JOURNAL_VERDICT=0`)
because the corpus cases predate journal-mode and don't carry
journal state. `max_turns=4` for OpenAI-compat to accommodate
slower-to-tool-call models; `max_turns=2` for Claude SDK (Haiku-tuned).

Backend: `claude_sdk` for Anthropic models (subscription billing),
`openai_compat` for everything else (OpenRouter).

## Results

Sorted by F1 (when measured), then by recall. Precision tested only
for models that crossed ~50% recall — others would dominate the
precision discussion with their silent-default behaviour.

| # | Model | Recall (24) | Precision (15) | F1 | $/M in/out |
|---|---|---|---|---|---|
| 1 | openai/gpt-4o-mini | **100.0%** | 13.3% | 0.24 | $0.75 / $3.00 |
| 2 | openai/gpt-5.4-nano | 79.2% | 26.7% | 0.40 | $0.20 / $1.25 |
| 2 | **mistralai/mistral-small-2603** (Mistral Small 4) | 79.2% | 53.3% | **0.64** | **$0.15 / $0.60** |
| 2 | inclusionai/ling-2.6-1t | 79.2% | **6.7%** | 0.12 | $0.08 / $0.63 |
| 5 | **claude-haiku-4-5** | 75.0% | **66.7%** | **0.71** | $1.00 / $5.00 |
| 6 | **google/gemini-3.1-flash-lite-preview** | 62.5% | **73.3%** | **0.68** | $0.25 / $1.50 |
| 7 | google/gemini-2.5-flash-lite | 58.3% | — | — | $0.10 / $0.40 |
| 8 | google/gemini-3.1-flash-lite (non-preview) | 54.2% | — | — | $0.25 / $1.50 |
| 9 | x-ai/grok-build-0.1 | 45.8% | — | — | $1.00 / $2.00 |
| 10 | ibm-granite/granite-4.1-8b | 33.3% | — | — | $0.05 / $0.10 |
| 11 | kwaipilot/kat-coder-pro-v2 | 29.2% | — | — | $0.30 / $1.20 |
| 11 | google/gemini-2.5-flash | 29.2% | — | — | $0.30 / $2.50 |
| 13 | inclusionai/ling-2.6-flash | 20.8% | — | — | $0.01 / $0.03 |
| 13 | arcee-ai/trinity-large-thinking | 20.8% | — | — | $0.22 / $0.85 |
| 13 | qwen/qwen3-coder-next | 20.8% | — | — | — |
| 13 | qwen/qwen3-235b-a22b-2507 | 20.8% | — | — | — |
| 17 | inclusionai/ring-2.6-1t | 16.7% | — | — | $0.08 / $0.63 |
| 18 | claude-sonnet-4-6 | 12.5% | — | — | $3.00 / $15.00 |
| 18 | qwen/qwen3.6-35b-a3b | 12.5% | — | — | $0.15 / $1.00 |
| 18 | openai/gpt-oss-120b | 12.5% | — | — | — |
| 18 | google/gemma-4-31b-it | 12.5% | — | — | $0.12 / $0.37 |
| 18 | deepseek/deepseek-v4-flash | 12.5% | — | — | $0.11 / $0.22 |
| 23 | z-ai/glm-4.7-flash | 8.3% | — | — | — |
| 23 | xiaomi/mimo-v2.5-pro | 8.3% | — | — | $1.00 / $3.00 |
| 23 | x-ai/grok-4.3 | 8.3% | — | — | $1.25 / $2.50 |
| 23 | qwen/qwen3.5-35b-a3b | 8.3% | — | — | $0.14 / $1.00 |
| 27 | qwen/qwen3.5-27b | 4.2% | — | — | $0.19 / $1.56 |
| 27 | openai/gpt-5.1-codex-mini | 4.2% | — | — | — |
| 27 | minimax/minimax-m2.7 | 4.2% | — | — | $0.28 / $1.20 |
| 27 | google/gemma-4-26b-a4b-it | 4.2% | — | — | $0.06 / $0.33 |
| 31 | qwen/qwen3.6-flash | 0.0% | — | — | $0.19 / $1.13 |
| 31 | qwen/qwen3.6-plus | 0.0% | — | — | $0.32 / $1.95 |
| 31 | qwen/qwen3.5-plus-20260420 | 0.0% | — | — | $0.30 / $1.80 |
| 31 | z-ai/glm-5.1 | 0.0% | — | — | (free promo) |
| 31 | tencent/hy3-preview | 0.0% | — | — | $0.07 / $0.26 |
| 31 | perceptron/perceptron-mk1 | 0.0% | — | — | $0.15 / $1.50 |
| 31 | nvidia/nemotron-3-super-120b-a12b | 0.0% | — | — | $0.09 / $0.45 |
| 31 | nvidia/nemotron-3-nano-30b-a3b:free | 0.0% | — | — | free |
| 31 | moonshotai/kimi-k2.6 | 0.0% | — | — | $0.73 / $3.49 |
| 31 | inception/mercury-2 | 0.0% | — | — | $0.25 / $0.75 |
| 31 | deepseek/deepseek-v4-pro | 0.0% | — | — | $0.44 / $0.87 |
| 31 | bytedance-seed/seed-2.0-mini | 0.0% | — | — | $0.10 / $0.40 |
| 31 | baidu/cobuddy:free | 0.0% | — | — | free |

## Conclusions

1. **Haiku owns the F1 podium (0.71).** Sweet spot of recall (75%)
   and precision (67%). The rubric is calibrated for Haiku's
   interpretation.

2. **Two credible non-Anthropic alternatives:**

   - **Gemini 3.1 Flash Lite Preview**: F1 0.68 — within 0.03 of
     Haiku. Higher precision than Haiku (73.3% vs 66.7%) at lower
     recall (62.5% vs 75%). $0.25/$1.50 — 4× cheaper than Haiku
     via OpenRouter.
   - **Mistral Small 4** (`mistralai/mistral-small-2603`): F1 0.64.
     Same 79.2% recall as gpt-5.4-nano but precision 53.3% (2×
     higher). $0.15/$0.60 — **6× cheaper than Haiku, 1.7× cheaper
     than Gemini 3.1 Flash Lite Preview**. The new price/quality
     champion.

   Curiosities in the Gemini line: the full `gemini-2.5-flash`
   (29.2% recall) underperforms its own `flash-lite` sibling
   (58.3%) — full models are more conservative on this rubric
   than their distilled variants. Same pattern with `gemini-3.1`:
   preview > non-preview on recall.

3. **Two recall traps to avoid.** Both flag almost everything,
   inflating recall to ~80-100% while precision collapses:
   - `openai/gpt-4o-mini`: 100% recall, 13.3% precision, F1 0.24.
     Critiques read well — specific, on-point references to
     fields/symbols — but the model finds fault in any non-trivial
     action. Motivated reasoning at scale.
   - `inclusionai/ling-2.6-1t`: 79.2% recall, **6.7% precision**
     (worse than gpt-4o-mini!), F1 0.12. Indiscriminately fires
     on actions the agent already proved were fine.

   Both unusable — would drown the supervised session in noise.

4. **gpt-5.4-nano is mediocre at its price tier.** Recall 79.2%
   ties Mistral Small 4, but precision 26.7% (half of Mistral's
   53.3%). At $0.20/$1.25 it's actually pricier than Mistral too.
   No reason to pick it over Mistral Small 4 or Gemini 3.1 Flash
   Lite Preview.

5. **Only 7 models cross 50% recall** out of 40 tested. The
   distribution is bimodal: a small head of capable models, then
   a long tail of "silent default" — 14 models score 0/24 because
   they literally never emit `professional=false`. Strong "coding
   leaderboard" models (Kimi K2.6, DeepSeek V4 Pro, GLM-5.1,
   gpt-5.1-codex-mini) cluster in this tail — they're trained to
   be helpful and agreeable, not adversarial. All from OpenAI / Anthropic /
   Google. Strong "coding leaderboard" models (Kimi K2.6, DeepSeek
   V4 Pro, GLM-5.1) score 0/24 — they default to
   `professional=true` silently because the rubric requires
   adversarial judgement they're trained against.

6. **Bigger ≠ better, and "thinking" ≠ better either.** DeepSeek
   V4 **Pro** (0/24) underperforms V4 **Flash** (3/24). Qwen3-235B
   (5/24) ties Qwen3-coder-next. GPT-oss-120B (3/24) is worse than
   gpt-5.4-nano (19/24). Nvidia Nemotron 3 Super 120B and
   inception/mercury-2 both score 0/24 despite being heavier
   models. The bottleneck is *willingness to disagree*, not raw
   capability or chain-of-thought.

7. **Switching models requires re-tuning the prompt**, not just
   the config. The SYSTEM_PROMPT explicitly tells Haiku to default
   to `professional=true` ("silence beats noise"). Other models
   take that instruction more literally than Haiku and produce
   near-uniform silence.

## How to re-run

```bash
# Positive (24 cases) — claude_sdk default
.venv/bin/python scripts/run_corpus.py --out /tmp/out.json

# Negative (15 cases) — same default backend
.venv/bin/python scripts/run_corpus.py --negative --out /tmp/out_neg.json

# OpenAI-compat against OpenRouter
set -a && . .env && set +a
.venv/bin/python scripts/run_corpus.py \
  --backend openai_compat \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY \
  --model openai/gpt-5.4-nano \
  --out /tmp/gpt5nano.json

# Compare a new run vs a saved baseline
.venv/bin/python scripts/compare_corpus.py \
  tests/fixtures/watchdog_corpus/baseline_haiku.json \
  /tmp/out.json
```

## Frozen baselines in this repo

`tests/fixtures/watchdog_corpus/`:
- `cases.json` — 24 positive cases (verbatim snapshots inline)
- `cases_negative.json` — 15 negative cases

Baselines with **both** recall and precision measured:
- `baseline_haiku.json` / `baseline_negative_haiku.json`
- `baseline_gpt-5.4-nano.json` / `baseline_negative_gpt-5.4-nano.json`
- `baseline_gpt-4o-mini.json` / `baseline_negative_gpt-4o-mini.json`
- `baseline_gemini-3.1-flash-lite-preview.json` /
  `baseline_negative_gemini-3.1-flash-lite-preview.json`
- `baseline_mistral-small-2603.json` /
  `baseline_negative_mistral-small-2603.json`
- `baseline_ling-2.6-1t.json` / `baseline_negative_ling-2.6-1t.json`

Recall only (less interesting for F1 ranking, kept for diff
tracking on future regressions):
- `baseline_sonnet.json`
- `baseline_gemini-3.1-flash-lite.json`
- `baseline_gemini-2.5-flash-lite.json` / `baseline_gemini-2.5-flash.json`
- `baseline_granite-4.1-8b.json`
- `baseline_kat-coder-pro-v2.json`
- `baseline_ling-2.6-flash.json`
- `baseline_trinity-large-thinking.json`
- `baseline_qwen3.5-35b-a3b.json` / `baseline_qwen3.6-flash.json`
- `baseline_seed-2.0-mini.json`
- `baseline_nemotron-3-super-120b.json`
- `baseline_mercury-2.json`

Use `compare_corpus.py` to diff a new run against any of these
to detect regression in either direction (caught→missed or
missed→caught) over time.
