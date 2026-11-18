# Running NudgeBench

Python 3.11 or newer is required. Run commands from the repository root.

## Offline checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/e1.py --dry-run
.venv/bin/python scripts/e5_deferral.py --dry-run
```

These commands do not call a model API. Corpus-dependent tests skip when the data
has not been fetched. To run those tests and the E4 reference-agent check:

```bash
./scripts/fetch_data.sh
.venv/bin/python scripts/fetch_taskbench.py
.venv/bin/python scripts/fetch_wfinstances.py
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/e4.py --dry-run
```

The fetchers download the third-party corpora; they do not purchase model calls.
`e4.py --dry-run` needs Caltech data even though it uses a reference agent.
A successful dry run validates the pipeline against that reference agent, not a
behavioural mechanism in an LLM.

## Live OpenRouter experiments

Live runs make paid requests. Use an account and API key from
[OpenRouter](https://openrouter.ai), with a provider-side spending limit appropriate
to the run. Model availability, prices and account interfaces can change; check
them before starting. The identifiers and prices in `scripts/e1.py` record the
study configuration and are not a live price feed.

Place the key in an untracked `.env` file in the repository root:

```dotenv
OPENROUTER_API_KEY=YOUR_KEY
```

The client reads this file automatically. An existing `OPENROUTER_API_KEY`
environment variable takes precedence. Protect the file and verify it is ignored:

```bash
chmod 600 .env
git check-ignore -v .env
```

Do not add the key to tracked files or transcripts.

Inspect the current provider model list before using the recorded identifiers:

```bash
curl -fsS https://openrouter.ai/api/v1/models | .venv/bin/python -c "
import json, sys
from scripts.e1 import PRICES
live = {m['id'] for m in json.load(sys.stdin)['data']}
for name in sorted(PRICES):
    print(('ok   ' if name in live else 'GONE ') + name)
"
```

Confirm the model's supported sampling and effort parameters as well as its
identifier. `probe_capabilities` in `mcta.probe.client` can perform a small paid
probe; short repeated answers can coincide by chance, so its result is diagnostic
rather than a guarantee of determinism or stochasticity.

Examples below use the study's recorded model identifiers. Use them only after
checking current availability and pricing:

```bash
.venv/bin/python scripts/e1.py --pilot --model google/gemini-3.7-flash --cap 1
.venv/bin/python scripts/e1.py --model google/gemini-3.7-flash --cap 2
.venv/bin/python scripts/e4.py --model google/gemini-3.7-flash --rollouts 20 --cap 5
.venv/bin/python scripts/e5_deferral.py --model google/gemini-3.7-flash \
  --repetitions 20 --temperature 1 --cap 3
```

`--cap` limits the client's estimated spending using the configured prices and
output-token ceiling. Its accuracy depends on those prices and provider accounting;
keep the provider-side limit as well. A low cap may stop a campaign before all
requested episodes complete.

## Archived results and replay

`runs/*.json` contains result summaries; `runs/*.jsonl` contains the recorded
requests and replies. The transcripts used for the paper are versioned. Keep them
unchanged when preparing a new campaign, and use a separate checkout for new runs.

The client checks a replay index before making a request. A replay match includes
the exact model, prompt, temperature, effort and sample identifier. An unmatched
request may become a live call. For analysis that must remain offline, use a zero
spending cap and no API key; a missing match will stop rather than buy a new call.

For example, these scripts use a zero-cap ledger to re-score the archived deferral
transcripts (no API key is needed for matched records):

```bash
.venv/bin/python scripts/robustness_e5.py google/gemini-3.7-flash 20 1
.venv/bin/python scripts/robustness_e5.py anthropic/claude-opus-5 20 1
```

A failed replay is not a fresh empirical result: investigate the missing record or
configuration mismatch. A live rerun is subject to provider sampling and model
changes, so it need not reproduce the archived outputs exactly.

## What the experiments measure

| Command | Question |
|---|---|
| `e1.py --dry-run` | Does the pipeline recover an interval containing the reference beta? |
| `e1.py --pilot` | Can responses be parsed and scored under this configuration? |
| `e1.py` | Which beta values are consistent with the observed choices under the model? |
| `e1.py --effort ...` | How do model-consistent intervals change across effort settings? |
| `e4.py` | How do observed compliance and repair costs compare across conditions? |
| `e5_deferral.py` | How often and how repeatedly does the agent defer? |

Model-consistent intervals do not identify an internal psychological parameter.
Deferral does not uniquely distinguish present bias from other decision rules.
Finite-rollout compliance is empirical evidence, not a guarantee over an LLM's
entire sampling distribution.
