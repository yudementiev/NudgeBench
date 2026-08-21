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
| Probe harness, β recovery (`mcta.probe`) | working, validated by round trip |
| Instance generators (`mcta.generate`) | working, calibrated to attested β |
| Structural measurement (`mcta.structure`) | working, certified bounds |
| Course-prerequisite corpus (`mcta.corpus`) | working, matches published figures |
| Solvers (ILP, heuristics, treewidth DP) | not started |
| LLM client for E1 | blocked on API access |

## First result (C1)

Real curricula are structurally narrow. Across two published course-prerequisite
networks, whole networks measure treewidth 8–12 (Caltech) and at most 25 (Johns
Hopkins), and the **702 task graphs** derived from them have treewidth at most 11
with routes at most 12 courses long — against the hundreds that Maniu et al.
report for database and road networks. Caveats, including what the Johns Hopkins
figures can and cannot claim, are in [REFERENCES.md](REFERENCES.md).

```bash
./scripts/fetch_data.sh              # fetch the published data (not vendored)
.venv/bin/python scripts/census.py   # reproduce the numbers above
```

Sources for every externally grounded number are in
[REFERENCES.md](REFERENCES.md), including one citation in the paper that needs
revisiting.

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
