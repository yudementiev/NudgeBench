# Nudging the Graph, Not the Agent: A Benchmark for Alignment by Environment Design

*Draft for NeurIPS 2026 FAST. Every number is reproducible today from the named
script.*

---

## Abstract

Agent harnesses are environments: the tools on offer, the ordering of steps and
the presence of shortcuts determine not only what an agent *can* do but what it
*will* do. Practitioners already align agents by editing these environments —
removing a tool, forcing a checkpoint, blocking a shortcut — without a formal
objective, a solver, or any guarantee that an edit constrains *every* trajectory
the agent might sample. A twelve-year-old model from behavioural economics, the
Kleinberg–Oren account of present-biased planning on task graphs, formalises
exactly this setting, and the accompanying Minimum-Cost Task Alignment problem
asks for the cheapest edit forcing every feasible trajectory to reach the goal and
perform a prescribed set of critical actions. The theory is well developed and has
never been executed. We introduce MCTA-Bench: 744 task graphs drawn from real
curricula and scientific workflows, annotated with the structural parameters that
govern tractability; exact and heuristic solvers; and an agent-in-the-loop probe
that estimates an agent's discount factor exactly rather than by fitting. We
report three results. Real task graphs are structurally narrow — treewidth in
single digits where prior measurements of network data report hundreds — yet
two of seven workflow families remain intractable regardless, because tractability
turns on weight diversity as much as on width. The repair an engineer would make
by hand matches the optimum on the median real instance and costs up to 2.67× more
in the tail, on instances of treewidth two. And on graphs of up to 26 arcs
presented in full, we find **no present bias in any language-model agent we
tested**, from frontier models down to the cheapest available: the discount factor
is pinned at or near 1 in every case, with arithmetic controls passing throughout.
The behavioural model that motivates this line of theory does not, at this scale,
describe the agents it is now being applied to. The intervention machinery,
however, transfers intact: theory-derived edits move compliance from 0% to 100%
across the whole sampling distribution, and do so even though they were computed
for a discount factor the agent does not have.

---

## 1. Introduction

Designing an agent's harness is already alignment work. Which tools are exposed,
which steps are mandatory, whether a cheap shortcut exists beside the thorough
route — these decisions shape behaviour more reliably than instructions do, and
practitioners make them constantly. They make them by intuition. There is no
objective being minimised, no solver, and no way to check that an edit binds every
trajectory the agent might produce rather than the one it happened to produce
during testing.

A formal account of this setting exists. Kleinberg and Oren model an agent
traversing a task DAG that evaluates plans under a present-biased discount: the
next action counts in full, everything after it is discounted by β. Such an agent
can announce a sensible plan and then abandon it on arrival, with nothing about
the environment having changed. The principal cannot alter β, but can edit the
graph — and Minimum-Cost Task Alignment asks for the cheapest edit that forces
every feasible trajectory to reach the target and traverse a prescribed set of
critical arcs.

This line has produced a detailed complexity landscape and no experiments. Nobody
has run the algorithms, measured the environments they assume, or checked the
behavioural premise against an actual agent. We do all three.

**One observation frames the whole exercise.** An agent sampling at temperature
above zero is precisely the nondeterministic tie-breaker the model's
all-feasible-paths formulation is written for. Prompting shifts the mode of an
agent's behaviour; editing its environment is supposed to constrain the entire
support. That distinction is what a theory of environment design buys over
instruction-writing, and it is testable.

## 2. Setting

An instance is `M = (G, w, s, t, β, r)` with `G` a weighted DAG. Standing at `v`,
the agent evaluates every `v`–`t` path under

    ζ(P) = w(e₁) + β · Σ_{i≥2} w(eᵢ),

abandons if `min_P ζ(P) > β·r`, and otherwise commits to `e₁` and repeats. Ties
are broken arbitrarily, so the model induces a *set* of feasible trajectories.
MCTA seeks `S ⊆ E` minimising `Σ_{e∈S} c(e)` such that in `G − S` every feasible
trajectory reaches `t` and traverses every arc of `T`. Costs are real: an
auxiliary arc given a negative deletion cost is one the solver pays to *keep*,
which lets a single objective carry both deletion and paid addition.

## 3. MCTA-Bench ✅

| Corpus | instances | source | weights |
|---|---|---|---|
| Caltech curricula | 169 | Stavrinides & Zuev, cleaned prerequisites | synthetic |
| Johns Hopkins curricula | 533 | same repository, extracted from prose | synthetic |
| Scientific workflows | 42 | WfCommons | **measured runtimes** |

Two mapping decisions are stated rather than buried. Prerequisite graphs are
conjunctive while agent paths are sequential, so an instance answers "how hard is
this shape to align", not "what should this student enrol in". And the Johns
Hopkins extraction reads catalogue disjunctions as conjunctions, which only adds
arcs — the safe direction, since treewidth never falls when arcs are added, so
those figures bound the real curriculum from above.

## 4. Environments are narrow, and that is not enough ✅

`scripts/census.py`

| Graph | treewidth | p | \|L\| |
|---|---|---|---|
| Caltech, largest component (436 nodes, 747 arcs) | 8–12 | — | — |
| Caltech, 169 instances | median 3, max 5 | max 7 | max 8 |
| Johns Hopkins, largest component (665, 1708) | ≤ 25 | — | — |
| Johns Hopkins, 533 instances | median 3, max 11 | max 12 | max 13 |
| seismology, 601 tasks | **2** | 3 | 415 |
| epigenomics, 233 tasks | **2** | 10 | 286 |
| montage, 472 tasks | **17–18** | 9 | 34 986 |
| soykb, 336 tasks | **12** | 12 | 6 610 |

The motivating claim survives. Task graphs are narrow, against the hundreds Maniu
et al. report for database and road networks — a contrast worth drawing precisely
because their corpus is network data rather than task graphs, so structural
narrowness appears to be a property of the latter specifically.

The second half is the interesting one. Theorem 1 of the underlying theory runs in
`(|L|+1)^O(tw)`, so width alone does not decide feasibility. Montage reaches 10⁸¹
states and its treewidth *grows with problem size*, 5 at 58 tasks to 18 at 472;
soykb reaches 10⁴⁶. Five of seven workflow families are trivially tractable and
two are hopeless. **A graph can be narrow and still out of reach.** This argues for
the theorem's form rather than against it, and it gives the open question of
whether an `f(tw)·m^g(p)` algorithm exists a concrete constituency: montage has
`p = 9` at every size, so such an algorithm would rescue exactly the family the
current one cannot.

One practical lever falls out. Rounding runtimes to whole seconds collapses `|L|`
from 34 986 to 385 in montage and barely moves srasearch. The principal's unit of
account is a design parameter of the intervention, not an implementation detail.

## 5. The intuitive repair is usually right and occasionally expensive ✅

`scripts/e3.py`

| Corpus | solved | optimality proved | heuristic worse | worst ratio |
|---|---|---|---|---|
| Caltech | 65 | 65 | 10 | **2.00×** |
| Johns Hopkins (≤28 arcs) | 87 | 74 | 2 | **2.67×** |

The heuristic is the engineer's move rendered as an algorithm: watch the agent
take a route it should not, block one step of it, repeat. Its median cost ratio
against the exact optimum is 1.00 — usually the local fix is simply correct. The
tail is where it is not, and the instances where it loses have **treewidth 2–3**.
Being narrow does not make the greedy choice right. Exact solving takes
milliseconds on the median instance and seconds in the tail; the search budget
runs out between 20 and 28 arcs, which is where hardness begins to bite in
practice.

## 6. Agents ✅

### 6.1 Probe

Each episode puts one agent through two conditions on one instance: state a
complete route from START, then walk it one committed action at a time. The agent
sees the whole graph in both — present bias is discounting the future, not being
blind to it, so hiding the future would measure something else. β is recovered
exactly rather than fitted: each revealed preference is a linear inequality in β,
so a trajectory pins it to an interval and the episodes of a run intersect to a
tighter one. An empty intersection is itself a finding — that no single-parameter
discount explains the agent — and is reported as such rather than averaged away.

Two controls carry the interpretation. The agent totals its own stated route, so a
model that cannot add is not mistaken for one that discounts. And replies that ran
out of output budget are counted apart from replies that ignored the format.

### 6.2 No present bias, at any capability ✅

`scripts/e1_corpus.py`, real prerequisite closures of 8–26 arcs

| Model | episodes | arithmetic | optimal route | β |
|---|---|---|---|---|
| claude-opus-5 | 40 | 100% | 98% | **[1.000, 1]** |
| gemini-3.7-flash | 40 | 100% | 100% | **[0.846, 1]** |
| deepseek-v4-flash | 8 | 100% | 88% | **[1.000, 1]** |

Synthetic traps agree. Calibrating instances to the human effort-discounting band
β ∈ [0.70, 0.79] proved useless: every agent sat above it, and traps had to be
rebuilt up to β = 0.99 before they bracketed anything. On the strongest of those —
one hour of work now against 118 later, versus 59 and 59 — gemini took the steady
route in 100 of 100 episodes.

**No agent we tested shows present bias on these instances.** The arithmetic
control passes everywhere, so this is patience rather than incapacity. The
hypothesis that myopia is a function of reasoning budget is not supported either:
the cheapest model available is as patient as the frontier.

The honest scope is narrow: on fully visible graphs of up to 26 arcs, the
Kleinberg–Oren model does not describe language-model route choice at any
capability level we could buy. What it does not establish is behaviour under
partial observability, longer horizons, or genuine execution cost — where a
bounded planning horizon has room to bite and where, we would expect, the model
has its best chance.

### 6.3 Do theory-derived edits steer an agent? ✅

Twelve instances the agent walks past a required action, three conditions
(unedited, hand-made repair, cheapest repair), 20 rollouts each at temperature 1.
On the selected cases the hand-made repair costs 29 against the optimum's 20.
Against a textbook present-biased agent both repairs align by construction and
differ only in price, so any gap in *effectiveness* measures the real agent
departing from the model.

| Condition | traversed T | rate | 95% CI |
|---|---|---|---|
| unedited | 0/240 | **0%** | [0%, 2%] |
| hand-made repair | 240/240 | **100%** | [98%, 100%] |
| cheapest repair | 240/240 | **100%** | [98%, 100%] |

**The intervention works completely** — 0% to 100% across the whole sampling
distribution, unanimous over 12 instances × 20 rollouts. That is the property
distinguishing environment design from prompting: not a shifted mode, a
constrained support.

**It is robust to getting β wrong.** The edits were computed for β = 0.75 and
applied to an agent our probe measures at β ≈ 1. They steered it perfectly anyway,
so a principal need not know the discount factor to design an effective edit —
which matters, since §6.2 suggests it may not exist.

**The optimum buys cost, not robustness.** The repairs differ on 9 of 12 instances
and the hand-made one costs 45% more (29 against 20), yet both succeed on every
rollout. The hypothesis that the cheaper repair might be more fragile finds no
support. The theory delivers the same guarantee for less, not a stronger one.

## 7. What measuring agents cost us

Three times a defect in our own harness presented as a property of the agent, each
time plausibly:

| Symptom | Read as | Actually |
|---|---|---|
| 25% unparseable replies from one model | will not follow the format | our parser could not read `**MOVE: A5**` |
| transcript replay missing every prompt | the log is useless | node order came from set iteration |
| empty replies from the weakest model | could not solve it | our output budget, exhausted mid-thought |

Each would have entered a paper as a finding. The first is the worst: those
episodes were dropped *selectively*, since whatever a model chooses to emphasise
is not a random slice, so the discount factor it produced was fitted to a biased
subsample. We therefore report the arithmetic control, the truncation column and
the full call transcript as part of the method rather than as engineering detail.

One limit of the transcript is worth recording: replay saves the cost of a rerun
only while the interpretation is unchanged. Fixing the parser changed which move
was read, which changed where the walk went, which required step prompts that were
never issued — so the stored replies no longer covered the interaction and both
runs had to be repeated.

## 8. Limitations

Weights in the curriculum corpora are synthetic; the published data has no
workload column. The conjunctive-to-sequential mapping is a modelling choice. The
Johns Hopkins figures bound its structure from above rather than measuring it. The
exact solver is branch and bound, not the treewidth dynamic program, so the
theory's main algorithm remains unexecuted. Agent results cover three models, one
protocol and two framings, with the weakest model at only 8 episodes; and the
graphs are small and fully visible, which is precisely the regime where a
bounded-horizon effect has least room to appear.

## 9. Conclusion

Environment design is an alignment primitive with a measurable objective, a
solver, and now a corpus. Two findings are ready to use: real task graphs are
narrow but not therefore tractable, and the intuitive repair is optimal on the
median instance and materially worse in a tail that structural narrowness does not
predict. The third is a negative result about the premise — the behavioural model
that motivates this theory does not describe the agents it is now aimed at, at the
scale we could test. That does not retire the theory; it locates where the
question actually lives, which is at horizons long enough or environments partial
enough that lookahead has a cost.

Everything releases: corpora, solvers, probe, and every API call made.
