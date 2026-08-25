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
| Scientific-workflow corpus (`mcta.corpus.workflows`) | working, real task runtimes |
| Exact and greedy solvers (`mcta.solve`) | working, exact checked against brute force |
| Treewidth dynamic program (Theorem 1) | not started — optional from the outset |
| OpenRouter client (`mcta.probe.client`) | working, spend-capped, transcript-logged |
| E1 probe on models | done — see C3 below |
| E4 intervention experiment | running |

## First result (C1)

Real task graphs are narrow — but narrowness is not enough.

Across two published course-prerequisite networks, whole networks measure
treewidth 8–12 (Caltech) and at most 25 (Johns Hopkins), and the **702 task
graphs** derived from them have treewidth at most 11 — against the hundreds that
Maniu et al. report for database and road networks. Forty-two real scientific
workflow traces agree: seismology runs 601 tasks at treewidth 2.

The interesting half is where this breaks. Theorem 1 of the paper pays
(|L|+1)^tw, and two workflow families are out of reach despite modest treewidth
because their path weights are nearly all distinct — montage reaches 10⁸¹ states,
and its treewidth grows with problem size rather than staying put. **A graph can
be narrow and still hopeless.** Details, and what this implies for the paper's
open question, are in [REFERENCES.md](REFERENCES.md).

```bash
./scripts/fetch_data.sh                        # course-prerequisite networks
.venv/bin/python scripts/fetch_wfinstances.py  # workflow traces (~17 MB)
.venv/bin/python scripts/census.py             # reproduce every number above
```

## Second result (C2)

The intuitive fix is usually optimal and occasionally expensive. Against the
witness-guided heuristic — watch the agent take a route it should not, block one
step of it, repeat — the exact optimum is the same answer on most real instances
and much cheaper on a minority:

| Corpus | solved | optimality proved | heuristic worse | worst ratio |
|---|---|---|---|---|
| Caltech closures | 65 | 65 | 10 | **2.00×** |
| Johns Hopkins closures (≤28 arcs) | 87 | 74 | 2 | **2.67×** |

Exact solving is milliseconds on the median instance and seconds on the tail; the
search budget runs out between 20 and 28 arcs, which is where the hardness starts
to bite in practice. Note that the instances where the heuristic loses have
treewidth 2–3 — being narrow does not make the greedy choice right.

```bash
.venv/bin/python scripts/e3.py
```

## Third result (C3)

No language-model agent we tested is present-biased. On real prerequisite
closures of 8–26 arcs, presented in full:

| Model | episodes | arithmetic | optimal route | β |
|---|---|---|---|---|
| claude-opus-5 | 40 | 100% | 98% | **[1.000, 1]** |
| gemini-3.7-flash | 40 | 100% | 100% | **[0.846, 1]** |
| deepseek-v4-flash | 8 | 100% | 88% | **[1.000, 1]** |

The arithmetic control passes everywhere, so this is patience rather than
incapacity, and the cheapest model is as patient as the frontier — the myopia
hypothesis gets no support from either end. Traps calibrated to the human
effort-discounting band β ∈ [0.70, 0.79] proved useless: every agent sat above it,
and traps had to be rebuilt up to β = 0.99 before they bracketed anything.

Scope is narrow on purpose: small, fully visible graphs are the regime where a
bounded planning horizon has least room to show.

```bash
.venv/bin/python scripts/e1_corpus.py --dry-run       # free self-test
.venv/bin/python scripts/e1_corpus.py --model MODEL   # see SETUP.md
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

173 tests, and they assert claims rather than absence of crashes: every number in
the paper's worked example, the fitter recovering a β the harness was driven with,
the exact solver agreeing with an independent brute force written in the tests,
structural bounds containing the truth on graphs where theory settles it, and the
loaders reproducing the published corpus figures. The suite gives identical
results under three different `PYTHONHASHSEED` values, which is enforced rather
than hoped for — three separate defects in this project came from set iteration
order leaking into results.

## Setup

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Layout

- `mcta/model.py` — the instance, the agent's decision rule, feasible trajectories
- `mcta/align.py` — T-avoiding arcs, the alignment verdict, violating witnesses
- `mcta/examples.py` — reference instances with behavior known in advance
- `mcta/generate.py` — calibrated traps and structural families
- `mcta/structure.py` — treewidth and friends, as certified intervals
- `mcta/solve.py` — exact branch and bound, witness-guided heuristic
- `mcta/corpus/` — course-prerequisite networks and workflow traces
- `mcta/probe/` — prompting, β recovery, OpenRouter client, aggregation
- `scripts/` — one runnable experiment each; all take `--dry-run`
- `paper/draft.md` — the write-up, with every measured number in place

See [SETUP.md](SETUP.md) to run the agent experiments.
