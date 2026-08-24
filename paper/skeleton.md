# MCTA-Bench — draft for NeurIPS 2026 FAST

**Venue.** Foundations of Agentic Systems Theory, Paris, 12–13 Dec.
**Deadline.** 29 Aug 2026 AoE. **Format.** 7 pp full / 4 pp short, NeurIPS
template, double-blind, non-archival.

Numbers below marked ✅ are measured and reproducible today. Slots marked ⬜ wait
on the agent runs. Everything is regenerable: `scripts/census.py`, `scripts/e3.py`,
`scripts/e1.py`, `scripts/e4.py`.

---

## Title

Leading candidate: **Nudging the Graph, Not the Agent: A Benchmark for Alignment
by Environment Design**

Alternatives: *Harness Engineering, Formalized* · *Narrow but Hopeless: Measuring
the Environments We Ask Agents to Plan In* · *Can You Steer an Agent by Editing
Its Environment?*

## Abstract (≈180 words)

> Agent harnesses are environments: the available tools, the ordering of steps and
> the presence of shortcuts determine not only what an agent *can* do but what it
> *will* do. Practitioners already align agents by editing these environments —
> removing a tool, forcing a checkpoint, blocking a shortcut — but without a formal
> objective, a solver, or any guarantee that the edit constrains *every* trajectory
> the agent might sample. We connect this practice to Kleinberg–Oren present-biased
> planning and to MINIMUM-COST TASK ALIGNMENT, which asks for the cheapest set of
> edits forcing every feasible trajectory to reach the goal and perform a prescribed
> set of critical actions. The theory is developed but has never been executed. We
> introduce MCTA-Bench: 744 task graphs from real curricula and scientific
> workflows annotated with the parameters that govern tractability, reference
> solvers, and an agent-in-the-loop protocol. We find that real task graphs are
> structurally narrow yet frequently intractable anyway ⬜(and that LLM agents
> …)⬜, and we characterise when the intuitive repair provably fails.

## 1. Introduction (≈1 p)

The wedge, in order:

1. Scaffold design *is* alignment, and it is currently done by intuition.
2. A twelve-year-old formal model describes exactly this, and has never been run.
3. Three claims it makes that nobody has checked. We check them.

**The line to keep.** An agent at temperature > 0 is precisely the nondeterministic
tie-breaker the model's all-feasible-paths formulation is written for. Prompting
shifts the mode of behavior; editing the environment is supposed to constrain the
whole support.

## 2. Setting (≈0.75 p)

Kleinberg–Oren on task DAGs; ζ(P) = w(e₁) + β·Σw(eᵢ); abandonment at ζ > βr; MCTA
with real-valued costs, which subsumes deletion and paid addition in one objective.
Keep to half a column — this is background, and the contribution is empirical.

## 3. The benchmark (≈1 p)

| | instances | source |
|---|---|---|
| Caltech curricula | 169 | Stavrinides & Zuev, cleaned prerequisites |
| Johns Hopkins curricula | 533 | same repository, extracted from prose |
| Scientific workflows | 42 | WfCommons, **measured task runtimes** |

State plainly what the mapping assumes: prerequisite graphs are conjunctive, agent
paths are sequential, so instances answer "how hard is this shape to align", not
"what should this student enrol in". The Johns Hopkins extraction reads
disjunctions as conjunctions, which only ever adds arcs — the safe direction, since
treewidth never falls when arcs are added.

## 4. C1 — real environments are narrow, and that is not enough ✅

| Corpus | treewidth | p | \|L\| |
|---|---|---|---|
| Caltech, largest component | 8–12 | — | — |
| Caltech, 169 instances | median 3, max 5 | max 7 | max 8 |
| Johns Hopkins, largest component | ≤ 25 | — | — |
| Johns Hopkins, 533 instances | median 3, max 11 | max 12 | max 13 |
| seismology, 601 tasks | **2** | 3 | 415 |
| montage, 472 tasks | **17–18** | 9 | 34 986 |

Two findings, and the second is the interesting one.

**The motivation survives.** Task graphs are narrow, against the hundreds Maniu et
al. report for database and road networks. *Note for co-authors: the current draft
cites Maniu et al. in support of this claim, and their headline finding is the
opposite. Replace with our own measurement and cite them for the contrast — a
stronger sentence, and ours.*

**Narrowness is not sufficient.** Theorem 1 pays (|L|+1)^tw. Montage reaches 10⁸¹
states and its treewidth *grows with problem size*; soykb reaches 10⁴⁶. Five of
seven workflow families are trivial, two are hopeless. This argues *for* the
theorem's form — both quantities matter — and gives the Conclusion's open question
a constituency: montage has p = 9 at every size, so an f(tw)·m^g(p) algorithm would
rescue exactly the family the current one cannot.

Practical corollary: rounding runtimes to whole seconds collapses |L| from 34 986
to 385 in montage and barely moves srasearch. The principal's unit of account is a
design parameter of the intervention.

## 5. C2 — the intuitive repair is usually right and occasionally expensive ✅

| Corpus | solved | optimality proved | heuristic worse | worst ratio |
|---|---|---|---|---|
| Caltech | 65 | 65 | 10 | **2.00×** |
| Johns Hopkins (≤28 arcs) | 87 | 74 | 2 | **2.67×** |

The heuristic is the engineer's move: watch the agent take a route it should not,
block one step, repeat. Median ratio 1.00 — usually it is simply right. The tail is
where it is not, and the instances where it loses have **treewidth 2–3**: being
narrow does not make the greedy choice correct. Exact solving is milliseconds on
the median and seconds on the tail; the budget runs out between 20 and 28 arcs.

## 6. C3 — do real agents behave this way? ⬜

**E1.** Traps with whole-number weights, thresholds spread across the effort band
β ∈ [0.70, 0.79] attested by the quasi-hyperbolic meta-analysis. Two conditions:
state a route, then walk it. β is recovered by intersecting revealed preferences,
which is exact rather than fitted.

- ⬜ β per model
- ⬜ arithmetic control rate — *if this is low the cell is void, not weak*
- ⬜ plan-versus-walk divergence
- ⬜ **β against reasoning effort** — the headline if it moves

**E4.** Twelve instances the agent fails, three conditions, 20 rollouts at
temperature. On the selected cases the hand-made repair costs 29 against the
optimum's 20. Against a textbook agent both align by construction and differ only
in price, so any gap in *effectiveness* measures the real agent departing from the
model.

- ⬜ success rate per condition, with Wilson intervals

## 7. Limitations

Say these ourselves. |W| = 1 in the curriculum corpora, an assumption the data
forces. Conjunctive-to-sequential mapping. Johns Hopkins is an upper bound on its
own structure. The exact solver is branch and bound, not the treewidth DP, so
Theorem 1 remains unexecuted. ⬜ Agent results are N models, one protocol, no
replication across prompt phrasings beyond the two framings.

## 8. What this is for

Environment design as an alignment primitive, with a measurable objective, a
solver and a corpus. Everything releases.

---

## Figure plan

**Figure 1 (the one that sells it).** Two panels. Left: (|L|+1)^tw on a log axis
against treewidth, every corpus instance plotted, with a horizontal line at what is
computable — shows narrow-but-hopeless at a glance and needs no agent results.
Right: ⬜ E4 success rate by condition with Wilson bars.

**Figure 2.** The paper's own Figure 1 graph, three panels: agent takes the
shortcut; deletion merely diverts; the auxiliary arc aligns. Reuse the one-pager's
SVG.

**Table 1.** C1 corpus table. **Table 2.** C2 solver table.

## Contingency

If the agent runs do not land by 27 Aug, sections 4, 5 and 7 stand alone as a
4-page short paper: first executable benchmark, 744 real instances, the
narrow-but-hopeless finding, the heuristic's tail. Section 6 becomes "in progress"
and the abstract drops its middle clause. Weaker at an agents workshop — no LLM in
it — but complete and defensible rather than rushed.

## Division of labour

- ⬜ who writes §1–2
- ⬜ who reviews 26–27 Aug
- ⬜ author order
