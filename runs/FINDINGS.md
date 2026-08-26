# E1 results, gemini-3.7-flash

All runs: two framings (abstract, effort), fresh randomised labels per episode,
temperature 0. Reproduce with `scripts/e1.py`; transcripts in `runs/*.jsonl`.

| band | trap depth | episodes | arithmetic | diverged | optimal | β |
|---|---|---|---|---|---|---|
| effort [0.70, 0.79] | 1 arc | 100 | 100% | 0% | 100% | **[0.789, 1]** |
| agentic [0.80, 0.99] | 1 arc | 100 | 100% | 0% | 100% | **[0.983, 1]** |
| agentic [0.80, 0.99] | 6 arcs | 60 | 100% | 0% | 100% | **[0.958, 1]** |

Spend to date: $0.95.

## What this says

**No present bias, at any threshold or depth tested.** The agent resisted even the
strongest trap — one hour of work now against 118 later, versus 59 and 59 — and
took the cost-minimal route in every single episode.

**It is not a competence artifact.** The arithmetic control passes at 100%
throughout: asked to state a route and total it, the agent totals correctly. So
this is a patient planner, not a model too weak to compare the branches.

**Nor is it lack of anything to look ahead at.** The depth-6 traps make each branch
a chain of six arcs, so choosing correctly means summing six numbers per side. The
result barely moves.

## What it means for the paper

The Kleinberg–Oren model does not describe this agent's route choice on graphs of
this size. Two readings, and they are not equally comfortable:

1. Capable models plan patiently whenever the graph is fully visible, and
   present-bias models transfer to LLM agents no better than they had to. A clean
   negative result, and squarely the kind of thing a workshop about importing
   outside theory should want reported.
2. The probes have not yet reached the regime where bounded rationality appears.
   Every instance so far fits comfortably in one prompt. Real harnesses do not, and
   neither do the larger corpus instances (Johns Hopkins closures reach 55 nodes).

Distinguishing these is the next experiment, and it is cheap. Until it is run, the
honest claim is the narrow one: **on fully-visible task graphs of up to twelve
arcs, gemini-3.7-flash is not present-biased.**
