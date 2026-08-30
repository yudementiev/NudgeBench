# MCTA-Bench

**A benchmark for alignment by environment design.** Given an agent that plans
myopically, what is the cheapest edit to its environment that forces *every*
trajectory it might take to reach the goal and perform a prescribed set of critical
actions?

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![tests](https://img.shields.io/badge/tests-179%20passing-brightgreen)
![instances](https://img.shields.io/badge/instances-18%2C045-orange)
![domains](https://img.shields.io/badge/domains-4-orange)
![transcripts](https://img.shields.io/badge/API%20calls-logged%20in%20full-lightgrey)

`agent-planning` · `present-bias` · `environment-design` · `task-graphs` ·
`time-inconsistency` · `benchmark` · `llm-agents` · `treewidth` ·
`parameterized-complexity` · `behavioral-economics`

---

## What this is

Agent harnesses are environments. Which tools are exposed, which steps are
mandatory, whether a cheap shortcut sits beside the thorough route — these choices
shape behaviour more reliably than instructions do, and practitioners make them by
intuition, with no objective being minimised and no way to check that an edit binds
every trajectory rather than the one seen in testing.

The Kleinberg–Oren model of time-inconsistent planning formalises this, and one can
then ask for the cheapest edit to the environment that aligns the agent. The theory
is developed and had never been executed. This repository is its first empirical
evaluation, and the benchmark that evaluation needed.

## The instance taxonomy

The benchmark's central artefact. **Branching** counts instances where some node
offers a choice of continuation — fan-in, several inputs to one task, is not a
decision. **Discriminating** counts those where a patient and a biased agent walk
*different* routes, which is the only kind that can measure an agent at all.

| Domain of origin | Instances | Branching | Discriminating | Arcs (median) | tw |
|---|---:|---:|---:|---|---:|
| **Agent tool decompositions** | | | | | |
| everyday APIs (TaskBench) | 4,318 | 275 | **0** | 2–9 (4) | 2 |
| multimedia (TaskBench) | 5,548 | 525 | 21 | 2–11 (4) | 2 |
| HuggingFace (TaskBench) | 7,435 | 775 | 42 | 2–11 (4) | 2 |
| **University curricula** | | | | | |
| Caltech | 169 | 169 | 28 | 7–30 (14) | 4 |
| Johns Hopkins | 533 | 495 | 119 | 6–180 (23) | 6 |
| **Scientific workflows** | | | | | |
| montage | 6 | 6 | 5 | 130–1336 (365) | 18 |
| soykb | 6 | 6 | 2 | 202–757 (479) | 12 |
| five other applications | 30 | 30 | 1 | 42–1201 | 2–3 |

**Of 17,301 agent task decompositions, 63 can pose the question at all.** The
domain agents actually operate in barely produces environments able to test
present-biased planning — which is why the benchmark also ships parametric
generators, and why domain of origin predicts structure far better than instance
count does.

```bash
.venv/bin/python scripts/taxonomy.py
```

## Results

**Environments are narrow, and that is not enough.** Real task graphs have
treewidth in single digits where prior measurements of database and road networks
report hundreds. But the theory's dynamic program pays `(|L|+1)^tw`, and two of
seven workflow families are out of reach regardless — montage reaches 10⁸¹ states,
and its treewidth *grows with problem size*.

**The intuitive repair is usually right and occasionally expensive.** Against the
witness-guided heuristic — watch the agent take a route it should not, block one
step, repeat — the exact optimum is the same answer on the median real instance and
up to **2.67× cheaper** in the tail. The instances where the heuristic loses have
treewidth 2–3: being narrow does not make the greedy choice correct.

**Present bias appears in an identifiable structure, not in general.** No agent
tested shows it on general graphs, at any capability. But in a *deferral trap* — a
large step toward the goal beside cheap ways of postponing it — one model defers at
a rate rising with the threshold, repeatedly, up to four deep, while another never
does in 91 episodes.

| big/small | Threshold | `gemini-3.7-flash` | `claude-opus-5` |
|---|---|---|---|
| 10/2 | 0.800 | 0/20 | 0/18 |
| 13/2 | 0.846 | 1/20 | 0/20 |
| 20/2 | 0.900 | 5/20 (4 repeatedly) | 0/20 |
| 20/1 | 0.950 | 5/20 | 0/20 |
| 50/1 | 0.980 | **7/20** (6 repeatedly) | 0/13 |

**Theory-derived edits steer the agent across the whole sampling distribution.**
Compliance goes from 0/240 unedited to 240/240 under either repair, on every
instance and every rollout — and the edits work despite being computed for a
discount factor the agent does not have.

## Quick start

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 179 tests, no data or API needed

./scripts/fetch_data.sh                        # course-prerequisite networks
.venv/bin/python scripts/fetch_taskbench.py    # agent task decompositions (~24 MB)
.venv/bin/python scripts/fetch_wfinstances.py  # workflow traces (~17 MB)

.venv/bin/python scripts/taxonomy.py           # the instance taxonomy
.venv/bin/python scripts/census.py             # structural parameters
.venv/bin/python scripts/e3.py                 # solvers on real instances
```

Agent experiments need an OpenRouter key — see [SETUP.md](SETUP.md). Every one of
them takes `--dry-run`, which exercises the whole pipeline against a reference
agent of known β and costs nothing.

## Layout

| Path | What |
|---|---|
| `mcta/model.py` | the instance, the agent's decision rule, feasible trajectories |
| `mcta/align.py` | T-avoiding arcs, the alignment verdict, violating witnesses |
| `mcta/generate.py` | six parametric families with analytically placed thresholds |
| `mcta/structure.py` | treewidth and friends, as certified intervals |
| `mcta/solve.py` | exact branch and bound, witness-guided heuristic |
| `mcta/corpus/` | four corpora as loaders; data is fetched, not vendored |
| `mcta/probe/` | prompting, exact β recovery, OpenRouter client, aggregation |
| `scripts/` | one runnable experiment each, all with `--dry-run` |
| `runs/` | every API call made for the paper: prompt, reply, tokens, cost |

## Reproducibility

Nothing here is meant to be taken on trust.

**Every reported number regenerates from a named script**, and the raw transcript of
every API call is versioned in `runs/`, so any figure can be re-scored without
re-spending. **NP-hard parameters are certified intervals, never point estimates** —
"treewidth at most 6" is sound whatever the truth; "is 6" would not be. **The exact
solver is validated against an independent brute force** written in the tests rather
than the library, so a shared mistake cannot hide. **The suite runs under three
different hash seeds**, because three separate defects in this project came from
Python's per-process hash randomisation leaking set-iteration order into results.

The paper's appendix documents four occasions on which a defect in our own harness
presented as a property of the agent. We think it is the part of this work most
reusable by anyone else measuring agents.

## Data

Fetched from the original authors, never redistributed here.

| Corpus | License | Weights |
|---|---|---|
| TaskBench | MIT | synthetic |
| Course-prerequisite networks | not declared — fetched, not vendored | synthetic |
| WfCommons WfInstances | LGPL-3.0 | **measured runtimes** |

Only workflow traces carry measured weights; elsewhere they are drawn from a small
integer palette, which is an assumption rather than a measurement and is reported as
one. See [REFERENCES.md](REFERENCES.md) for every externally grounded number and the
caveats against it.

## License

MIT — see [LICENSE](LICENSE).
