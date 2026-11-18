# NudgeBench

**A benchmark for alignment by environment design.** Given an agent that plans
myopically, what is the cheapest edit to its environment that forces *every*
trajectory feasible under the Kleinberg–Oren planning model to reach the goal and
perform a prescribed set of critical actions?

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![instances](https://img.shields.io/badge/instances-18%2C045-orange)
![domains](https://img.shields.io/badge/domains-3-orange)
![corpora](https://img.shields.io/badge/source%20corpora-4-orange)
![transcripts](https://img.shields.io/badge/API%20calls-logged%20in%20full-lightgrey)

`agent-planning` · `present-bias` · `environment-design` · `task-graphs` ·
`time-inconsistency` · `benchmark` · `llm-agents` · `treewidth` ·
`parameterized-complexity` · `behavioral-economics`

---

## Paper

**Nudging the Graph, Not the Agent: A Benchmark for Alignment by Environment Design**
Yuriy Dementiev · ITMO University · FAST Workshop at NeurIPS 2026

[Camera-ready PDF](publication/FAST2026_NudgeBench_camera_ready.pdf) ·
[LaTeX source and build instructions](publication/README.md) ·
[Citation metadata](CITATION.cff)

## What this is

Agent harnesses are environments. Which tools are exposed, which steps are
mandatory, whether a cheap shortcut sits beside the thorough route — these choices
shape the available actions. NudgeBench makes one environment-editing objective
precise and provides instances, solvers and behavioural probes for studying it.

The Kleinberg–Oren model of time-inconsistent planning formalises this, and one can
then ask for the cheapest edit to the environment that aligns the agent. The theory
is developed and had never been executed. This repository is its first empirical
evaluation, and the benchmark that evaluation needed.

The guarantee is conditional on the planning model. Stochastic LLM sampling can
select actions that are strictly worse under that model, so it is not equivalent
to arbitrary tie-breaking among model-optimal actions. Structural measurements,
behavioural probes and empirical intervention results answer separate questions.

## The instance taxonomy

Four source corpora (TaskBench, Caltech, Johns Hopkins and WfCommons) span three
domain groups. **Branching** counts instances where some node
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

The `tw` column is the maximum treewidth **upper bound** over the first up to 60
instances in each source/application, as implemented in `scripts/taxonomy.py`.
It is a sampled statistic, not the maximum over the complete corpus; the census
uses a different scope and course-weight scheme (see [REFERENCES.md](REFERENCES.md)).

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

**Repeated deferral is observed in a specific configuration.** On the tested
general graphs, choices are consistent with little or no present bias under the
model. In a *deferral trap* — a large step toward the goal beside cheap ways of
postponing it — one model repeatedly defers, up to four times, while the other
never does in 91 completed episodes. The trend with threshold is descriptive;
the experiments do not establish it statistically. Deferral is consistent with
present-biased planning but does not rule out local-cost heuristics or planning
errors, even when the arithmetic control passes.

| big/small | Threshold | `gemini-3.7-flash` | `claude-opus-5` |
|---|---|---|---|
| 10/2 | 0.800 | 0/20 | 0/18 |
| 13/2 | 0.846 | 1/20 | 0/20 |
| 20/2 | 0.900 | 5/20 (4 repeatedly) | 0/20 |
| 20/1 | 0.950 | 5/20 | 0/20 |
| 50/1 | 0.980 | **7/20** (6 repeatedly) | 0/13 |

**Graph edits improve compliance in the observed rollouts.** Compliance increases
from 0/240 unedited to 240/240 under either repair, with reported 95% Wilson
intervals of [0%, 2%] and [98%, 100%]. Finite rollouts do not establish success over
the entire sampling distribution. The edits were designed at β = 0.75 and tested
on an agent whose probe interval was near 1; this mismatch did not prevent success
in these trials, but does not establish general robustness to an unknown β.

**The probe computes a model-consistent interval.** It intersects linear choice
constraints analytically rather than grid-searching β (the implementation uses
floating-point arithmetic and a comparison tolerance). A nonempty interval does
not identify a unique internal parameter or behavioural mechanism. An empty
intersection rejects a shared β under the tested model constraints.

The intervention study is separate from the deferral experiment. It does not show
that modelling present bias is necessary for intervention effectiveness. Comparisons
with mandatory checkpoints and prompting remain future work.

## Quick start

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # no API calls; corpus tests skip if data is absent

./scripts/fetch_data.sh                        # course-prerequisite networks
.venv/bin/python scripts/fetch_taskbench.py    # agent task decompositions (~24 MB)
.venv/bin/python scripts/fetch_wfinstances.py  # workflow traces (~17 MB)

.venv/bin/python scripts/taxonomy.py           # the instance taxonomy
.venv/bin/python scripts/census.py             # structural parameters
.venv/bin/python scripts/e3.py                 # solvers on real instances
```

Start with the synthetic reference-agent checks; these need no fetched data or API:

```bash
.venv/bin/python scripts/e1.py --dry-run
.venv/bin/python scripts/e5_deferral.py --dry-run
```

Live agent experiments use OpenRouter — see [SETUP.md](SETUP.md). The E4 dry run
also needs the Caltech corpus. Dry runs check the implemented reference model;
they do not validate an LLM's behavioural mechanism.

## Layout

| Path | What |
|---|---|
| `mcta/model.py` | the instance, the agent's decision rule, feasible trajectories |
| `mcta/align.py` | T-avoiding arcs, the alignment verdict, violating witnesses |
| `mcta/generate.py` | nine instance generators with analytically placed thresholds |
| `mcta/structure.py` | treewidth and friends, as certified intervals |
| `mcta/solve.py` | exact branch and bound, witness-guided heuristic |
| `mcta/corpus/` | four corpora as loaders; data is fetched, not vendored |
| `mcta/probe/` | prompting, model-consistent β intervals, OpenRouter client, aggregation |
| `scripts/` | experiment runners, data fetchers and analysis utilities |
| `runs/` | archived API transcripts and result summaries |
| `publication/` | camera-ready PDF, LaTeX source and build instructions |

## Reproducibility

The experiment scripts and archived API transcripts in `runs/` support auditing
and re-scoring. Replay requires the original model, prompt, temperature, effort
and sample identifier; unmatched requests can become live calls unless the spending
cap is zero. See [SETUP.md](SETUP.md) for offline checks and replay constraints.

**NP-hard parameters are certified intervals, never point estimates** —
"treewidth at most 6" is sound whatever the truth; "is 6" would not be. **The exact
solver is validated against an independent brute force** written in the tests rather
than the library, so a shared mistake cannot hide.

Run the suite under three hash seeds to check for dependence on Python set ordering:

```bash
for seed in 0 1 42; do
  PYTHONHASHSEED=$seed .venv/bin/python -m pytest tests/ -q || exit 1
done
```

Corpus tests skip when their data is absent; fetch all three data collections above
to run the complete suite. Neither unit tests nor reference-agent dry runs call a
commercial model.

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

## Citation

```bibtex
@inproceedings{dementiev2026nudgebench,
  author = {Yuriy Dementiev},
  title = {Nudging the Graph, Not the Agent: A Benchmark for Alignment by Environment Design},
  booktitle = {Foundations of Agentic Systems Theory (FAST), NeurIPS 2026 Workshop},
  year = {2026},
  url = {https://github.com/yudementiev/NudgeBench}
}
```

## Acknowledgments

This work was supported by the Ministry of Economic Development of the Russian
Federation (IGK 000000C313925P4C0002), agreement No. 139-15-2025-010.
The authors declare no competing interests.

## License

MIT — see [LICENSE](LICENSE).
