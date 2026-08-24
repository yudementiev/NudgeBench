# Setting up OpenRouter

One account gives access to every model the experiments use. Written for someone
who has not used it before.

## 1. Account and credits

1. Sign up at **https://openrouter.ai** with email, Google or GitHub. No card is
   needed to create the account.
2. In the left sidebar open **Credits → Add Credits**. The minimum purchase is
   **$5**; **$25 covers all of phase 1** ($41.50 is the planning figure including
   a 40% margin, so $50 leaves room to rerun).
   Cards, AliPay and USDC are accepted.

Credits are prepaid and shared across every model — there is no per-provider
billing and no separate signup with Anthropic, OpenAI or Google.

## 2. Create a key with a spending limit

**Dashboard → Keys → Create key.**

Name it something like `mcta-bench-phase1`, and — this is the part worth not
skipping — **set a credit limit on the key itself**, say $50. The key then cannot
spend more than that no matter what the code does. Our client enforces its own cap
too, but a limit at the provider is the one that holds if the code is wrong.

Optionally set an expiry date.

## 3. Point the code at it

Create a `.env` file in the repository root. The code reads it by itself — there
is nothing to source, and it works from any shell and any directory:

```bash
cd "/Users/yuru/Inconsistent Planning"
printf 'OPENROUTER_API_KEY=sk-or-v1-PASTE_YOURS_HERE\n' > .env
chmod 600 .env
```

`.env` is gitignored, so the key cannot be committed by accident. Verify:

```bash
git check-ignore -v .env      # should print the .gitignore rule
git status --short             # .env must not appear
```

An `export OPENROUTER_API_KEY=...` in the shell still works and takes precedence
over the file, which is handy for running one command against a different key.

Never paste the key into a tracked file, and never into a prompt to a model.

## 4. Check it works, for about a cent

```bash
.venv/bin/python -c "
from mcta.probe import Ledger, OpenRouter, Pricing, Transcript
agent = OpenRouter('google/gemini-3.7-flash', Pricing(0.375, 1.875),
                   Ledger(cap=0.05), Transcript('runs/smoke.jsonl'))
print(agent('Reply with exactly: ready'))
"
```

If this prints `ready`, the account, credits, key and client are all wired up.

Common failures:

| Symptom | Cause |
|---|---|
| `OPENROUTER_API_KEY is not set` | the export did not reach this shell |
| 401 in the transcript's `error` field | key wrong, or revoked |
| 402 | out of credits |
| 404 | model id wrong — see §6 |

## 5. Check what the models will actually honour

Two parameters decide whether experiments are meaningful, and both are worth a few
cents to verify rather than discovering mid-run:

```bash
.venv/bin/python -c "
import json
from mcta.probe import Ledger, Pricing, Transcript, probe_capabilities
print(json.dumps(probe_capabilities(
    'google/gemini-3.7-flash', Pricing(0.375, 1.875),
    Ledger(cap=0.20), Transcript('runs/caps.jsonl')), indent=2))
"
```

- **`varies_with_temperature`** must be true for any model used in **E4**, which
  measures a fraction over the sampling distribution. The GPT-5.6 family does not
  list `temperature` among its parameters at all, so `scripts/e4.py` refuses those
  models before spending anything.
- **`effort_changes_output_length`** must be true for the **reasoning-effort
  sweep**, which is two thirds of the phase-1 budget.

## 6. Model ids change

Ids are exact strings and some carry date suffixes — `deepseek/deepseek-v4-pro-20260813`,
not `deepseek/deepseek-v4-pro`. A near miss fails on the first call. Before a
campaign, check the live list against `PRICES` in `scripts/e1.py`:

```bash
curl -s https://openrouter.ai/api/v1/models | .venv/bin/python -c "
import json, sys
from scripts.e1 import PRICES
live = {m['id'] for m in json.load(sys.stdin)['data']}
for name in sorted(PRICES):
    print(('ok   ' if name in live else 'GONE ') + name)
"
```

## 7. Run the experiments

In this order. Each step is cheap and each one tells you whether the next is worth
starting.

```bash
.venv/bin/python scripts/e1.py --dry-run            # free; pipeline self-test
.venv/bin/python scripts/e1.py --pilot --model google/gemini-3.7-flash --cap 1
.venv/bin/python scripts/e1.py --model google/gemini-3.7-flash --cap 2
.venv/bin/python scripts/e1.py --model anthropic/claude-opus-5 \
    --effort low,medium,high --cap 25
.venv/bin/python scripts/e4.py --model google/gemini-3.7-flash --rollouts 20 --cap 5
```

`--cap` is a hard ceiling in dollars, checked against the worst case *before* each
call, so a run stops below it rather than just after.

Every call is written to `runs/*.jsonl` with its tokens and cost. Re-running a
command replays from that file instead of paying again, so re-analysis is free and
an interrupted run resumes where it stopped. Delete the file to force fresh calls.

## What each run answers

| Command | Question |
|---|---|
| `e1.py --dry-run` | does our own pipeline work at all |
| `e1.py --pilot` | does the model answer in a format we can parse |
| `e1.py` | is the agent present-biased, and what is its β |
| `e1.py --effort ...` | does β climb as the agent is given more room to think |
| `e4.py` | do theory-derived edits steer it, on every rollout or only most |
