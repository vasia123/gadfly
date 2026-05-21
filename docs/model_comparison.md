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

| # | Model | Recall (24) | Precision (15) | F1 |
|---|---|---|---|---|
| 1 | openai/gpt-4o-mini | **100.0%** | 13.3% | 0.24 |
| 2 | openai/gpt-5.4-nano | 79.2% | 26.7% | 0.40 |
| 3 | **claude-haiku-4-5** | 75.0% | **66.7%** | **0.71** |
| 4 | **google/gemini-3.1-flash-lite-preview** | 62.5% | **73.3%** | **0.68** |
| 5 | google/gemini-2.5-flash-lite | 58.3% | — | — |
| 6 | google/gemini-3.1-flash-lite (non-preview) | 54.2% | — | — |
| 7 | x-ai/grok-build-0.1 | 45.8% | — | — |
| 8 | google/gemini-2.5-flash | 29.2% | — | — |
| 9 | qwen/qwen3-coder-next | 20.8% | — | — |
| 9 | qwen/qwen3-235b-a22b-2507 | 20.8% | — | — |
| 11 | inclusionai/ring-2.6-1t | 16.7% | — | — |
| 12 | claude-sonnet-4-6 | 12.5% | — | — |
| 12 | qwen/qwen3.6-35b-a3b | 12.5% | — | — |
| 12 | openai/gpt-oss-120b | 12.5% | — | — |
| 12 | google/gemma-4-31b-it | 12.5% | — | — |
| 12 | deepseek/deepseek-v4-flash | 12.5% | — | — |
| 17 | z-ai/glm-4.7-flash | 8.3% | — | — |
| 17 | xiaomi/mimo-v2.5-pro | 8.3% | — | — |
| 17 | x-ai/grok-4.3 | 8.3% | — | — |
| 20 | qwen/qwen3.5-27b | 4.2% | — | — |
| 20 | openai/gpt-5.1-codex-mini | 4.2% | — | — |
| 20 | minimax/minimax-m2.7 | 4.2% | — | — |
| 20 | google/gemma-4-26b-a4b-it | 4.2% | — | — |
| 24 | z-ai/glm-5.1 | 0.0% | — | — |
| 24 | tencent/hy3-preview | 0.0% | — | — |
| 24 | qwen/qwen3.6-plus | 0.0% | — | — |
| 24 | qwen/qwen3.5-plus-20260420 | 0.0% | — | — |
| 24 | perceptron/perceptron-mk1 | 0.0% | — | — |
| 24 | nvidia/nemotron-3-nano-30b-a3b:free | 0.0% | — | — |
| 24 | moonshotai/kimi-k2.6 | 0.0% | — | — |
| 24 | deepseek/deepseek-v4-pro | 0.0% | — | — |
| 24 | baidu/cobuddy:free | 0.0% | — | — |

## Conclusions

1. **Haiku owns the F1 podium (0.71).** Sweet spot of recall (75%)
   and precision (67%). The rubric is calibrated for Haiku's
   interpretation.

2. **Gemini 3.1 Flash Lite Preview is the only credible
   non-Anthropic alternative.** F1 0.68 — within 0.03 of Haiku.
   Higher precision than Haiku (73.3% vs 66.7%) at lower recall
   (62.5% vs 75%). Price-wise it's $0.25/$1.50 per M tokens vs
   Haiku's $1/$5 via OpenRouter — 4× cheaper for ~95% of Haiku's
   F1. Non-preview sibling drops to 54.2% recall (untested on
   precision). Curiously the full `gemini-2.5-flash` (29.2%
   recall) underperforms its own `flash-lite` (58.3%): full
   Gemini is more conservative on the rubric than its distilled
   variant.

3. **gpt-4o-mini is a recall trap.** 100% recall is misleading;
   precision 13.3% means it also flags 87% of cases the agent
   successfully argued were watchdog false-positives. Critiques
   read well (specific, on-point references to fields/symbols)
   but the model finds fault in any non-trivial action — motivated
   reasoning at scale.

4. **gpt-5.4-nano is the cheapest OpenAI option but precision-poor.**
   Recall 79.2% beats Haiku, but precision 26.7% — three-quarters
   of its flags on previously-rebutted cases are spurious. F1 0.40,
   roughly half of Haiku's. Use Gemini 3.1 Flash Lite Preview
   instead at the same price tier.

5. **Only 6 models cross 50% recall.** All from OpenAI / Anthropic /
   Google. Strong "coding leaderboard" models (Kimi K2.6, DeepSeek
   V4 Pro, GLM-5.1) score 0/24 — they default to
   `professional=true` silently because the rubric requires
   adversarial judgement they're trained against.

6. **Bigger ≠ better.** DeepSeek V4 **Pro** (0/24) underperforms
   V4 **Flash** (3/24). Qwen3-235B (5/24) ties Qwen3-coder-next.
   GPT-oss-120B (3/24) is worse than gpt-5.4-nano (19/24).
   The bottleneck is *willingness to disagree*, not raw capability.

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
- `baseline_haiku.json` / `baseline_negative_haiku.json`
- `baseline_sonnet.json`
- `baseline_gpt-5.4-nano.json` / `baseline_negative_gpt-5.4-nano.json`
- `baseline_gpt-4o-mini.json` / `baseline_negative_gpt-4o-mini.json`
- `baseline_gemini-3.1-flash-lite-preview.json` /
  `baseline_negative_gemini-3.1-flash-lite-preview.json`
- `baseline_gemini-3.1-flash-lite.json` (recall only)
- `baseline_gemini-2.5-flash-lite.json` (recall only)
- `baseline_gemini-2.5-flash.json` (recall only)

Use `compare_corpus.py` to diff a new run against any of these
to detect regression in either direction (caught→missed or
missed→caught) over time.
