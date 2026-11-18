# Archived E1 results, gemini-3.7-flash

These notes describe the initial E1 runs, not the complete study. See the
[README](../README.md) and [camera-ready paper](../publication/FAST2026_NudgeBench_camera_ready.pdf)
for the later corpus, deferral and intervention experiments.

All runs: two framings (abstract, effort), fresh randomised labels per episode,
temperature 0. Reproduce with `scripts/e1.py`; transcripts in `runs/*.jsonl`.

| band | trap depth | episodes | arithmetic | diverged | optimal | β |
|---|---|---|---|---|---|---|
| effort [0.70, 0.79] | 1 arc | 100 | 100% | 0% | 100% | **[0.789, 1]** |
| agentic [0.80, 0.99] | 1 arc | 100 | 100% | 0% | 100% | **[0.983, 1]** |
| agentic [0.80, 0.99] | 6 arcs | 60 | 100% | 0% | 100% | **[0.958, 1]** |

Recorded spend for these initial runs: $0.95 (not the total study budget).

## Interpretation

The agent chose the cost-minimal route in every reported episode, including the
strongest two-route trap. The compatible beta intervals include 1, so these
observations are consistent with patient planning under the model. They do not
identify a unique internal discount factor or exclude small present bias that
these instances cannot resolve.

The arithmetic control passes throughout. This verifies the reported route totals;
it does not rule out alternative decision rules or planning errors in other settings.
Depth-six traps also produce patient-compatible choices.

These results apply to the tested fully visible graphs and settings. Subsequent
deferral experiments found a different pattern in one model, as reported in the
camera-ready paper. The archived numbers and transcripts remain unchanged.
