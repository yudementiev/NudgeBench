# MCTA-Bench

A benchmark for **alignment by environment design**: given an agent that plans
myopically, what is the cheapest edit to its environment that forces every
trajectory it might take to reach the goal and perform a prescribed set of
critical actions?

The formal setting is the Kleinberg–Oren model of time-inconsistent planning and
the MINIMUM-COST TASK ALIGNMENT problem of Belova, Dementiev, Fomin, Golovach,
Ignatiev and Sagunov, *Structural Alignment in Inconsistent Planning*. The theory
is developed; this repository is its first empirical evaluation.

## Status

| Component | State |
|---|---|
| Kleinberg–Oren simulator (`mcta.model`) | working, validated against the paper |
| Alignment check and witness extraction (`mcta.align`) | working |
| Corpus of real task DAGs | not started |
| Solvers (ILP, heuristics, treewidth DP) | not started |
| Agent-in-the-loop protocol | not started |

## Validation

The paper works a full example (Figure 1) and states the agent's perceived cost
at every decision point. Those numbers are the test suite:

```bash
.venv/bin/python -m pytest tests/ -q
```

Ten tests cover the three stages of the paper's story — the agent taking the
tempting shortcut, the deletion that merely diverts it, and the auxiliary arc
that finally aligns it — plus abandonment and the effect of `beta` in isolation.

## Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Layout

- `mcta/model.py` — the instance, the agent's decision rule, feasible trajectories
- `mcta/align.py` — T-avoiding arcs, the alignment verdict, violating witnesses
- `mcta/examples.py` — reference instances with behavior known in advance
