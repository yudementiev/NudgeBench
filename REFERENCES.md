# External grounding

What in this benchmark rests on outside evidence, what that evidence actually
says, and where it is used. Kept separate so co-authors can audit the claims
before they reach the paper.

## Present-bias factors

Instances are calibrated so that the agent's behavior flips at a `beta` that real
agents are known to occupy. Weights in our model are effort, so the effort band is
the one that governs.

| Reward type | β | 95% CI | Used for |
|---|---|---|---|
| Monetary | 0.938 | [0.905, 0.972] | upper reference; not our setting |
| Non-monetary | 0.750 | [0.643, 0.857] | broad band for robustness sweeps |
| Real effort | 0.70–0.79 | — | **primary calibration band** |

- Cheung, Tymula, Wang, *A meta-analysis of quasi-hyperbolic discounting*
  ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5222564),
  [IZA DP 14625](https://docs.iza.org/dp14625.pdf)) — 86 studies; source of the
  monetary and non-monetary figures above.
- Cerrone, Chakraborty, Kim, Lades, *Estimating present bias and sophistication
  over effort and money*, Experimental Economics 28(6):1191–1213, 2025
  ([Cambridge Core](https://www.cambridge.org/core/journals/experimental-economics/article/estimating-present-bias-and-sophistication-over-effort-and-money/308B6EA8834BE182081267488BCFF07F))
  — effort β of 0.70–0.79, with sophistication 0.80–0.88. Present bias over real
  effort is stronger than over money. (Not Augenblick, who appears in it only as
  a cited predecessor; we had this misattributed.)

**Caveat we should state ourselves before a reviewer does.** Enke, *Complexity and
Hyperbolic Discounting* ([HBS WP 24-048](https://www.hbs.edu/ris/Publication%20Files/24-048_304d978e-f730-4523-a9e0-4370f82ebd03.pdf))
argues that structural estimates of present bias are inflated when the effect of
decision complexity is ignored, and the meta-analysis itself reports weak support
for the specifically *quasi-hyperbolic* functional form. We use these numbers to
place instances in a plausible range, not to claim the Kleinberg–Oren discount is
the true model of anyone. E1 tests that separately, and complexity is a confound
we control for through the arithmetic condition.

## Treewidth of real graphs — a citation to revisit

Maniu, Senellart, Jog, *An Experimental Study of the Treewidth of Real-World Graph
Data*, ICDT 2019 ([arXiv:1901.06862](https://arxiv.org/abs/1901.06862)).

The IEEE Access submission cites this (as [23]) in support of the claim that
structural parameters are small in practice and therefore that our parameterized
algorithms apply. **Its headline finding is the opposite**: across 25 datasets from
8 domains, treewidths of real networks are high — even road networks, among the
lowest, reach into the hundreds, which the authors say renders algorithms
exponential in treewidth unusable.

The citation is not fatal, because their corpus is database instances, social and
road networks — not task graphs. Our claim concerns layered, human-designed
workflows, which plausibly behave differently.

**Measured, on two real corpora.** The claim holds; the citation still does not
support it. Reproduce with `scripts/census.py`:

| Graph | Treewidth | p | \|L\| |
|---|---|---|---|
| Caltech, largest component (436 nodes, 747 links) | **8–12** | — | — |
| Caltech, 169 derived instances | median **3**, max **5** | median 5, max 7 | median 6, max 8 |
| Johns Hopkins, largest component (665, 1708) | **≤ 25** | — | — |
| Johns Hopkins, 533 derived instances | median **3**, max **11** | median 6, max 12 | median 7, max 13 |

Against hundreds for road networks in Maniu et al. So the right move is to cite our
own measurement for the claim and cite Maniu et al. only for the contrast — that
structural narrowness is a property of *task* graphs specifically, not of network
data generally. That is a stronger sentence than the one currently in the paper,
and it is ours.

### Scientific workflows — where the simple story breaks

Data: **WfCommons WfInstances**, https://github.com/wfcommons/WfInstances (LGPL-3.0)
— traces of real production runs in WfFormat, with **measured task runtimes**, so
these are the first instances whose weights are not an assumption. Fetch with
`scripts/fetch_wfinstances.py`; 42 traces across 7 applications.

| Application | tasks | treewidth | \|L\| | (\|L\|+1)^tw |
|---|---|---|---|---|
| seismology | 101 → 601 | **2** | 97 → 415 | 10⁴·⁰ → 10⁵·² |
| epigenomics | 41 → 233 | **2** | 50 → 286 | 10³·⁴ → 10⁴·⁹ |
| cycles | 67 → 266 | **3** | 110 → 410 | 10⁶·¹ → 10⁷·⁸ |
| 1000genome | 52 → 208 | **3** | 365 → 1442 | 10⁷·⁷ → 10⁹·⁵ |
| srasearch | 22 → 42 | **3** | 32 → 62 | 10⁴·⁶ → 10⁵·⁴ |
| soykb | 96 → 336 | 7 → **12** | 1665 → 6610 | 10²²·⁶ → 10⁴⁵·⁸ |
| montage | 58 → 472 | 5 → **18** | 1018 → 34986 | 10¹⁸·⁰ → 10⁸¹·⁸ |

**Treewidth alone is not the story, and this is the finding worth reporting.**
Five of seven families are trivially tractable — seismology runs 601 tasks at
treewidth 2. But montage's treewidth *grows with problem size*, 5 to 18, so the
family is not narrow in any size-independent sense; and both montage and soykb are
out of reach regardless, because Theorem 1 pays (|L|+1)^tw and their path weights
are nearly all distinct. A graph can be narrow and still hopeless.

This sharpens rather than damages the paper. The theorem already says the bound is
governed by treewidth *and* weight diversity together; what the corpus shows is
that real instances genuinely fall on both sides of that line, which is an argument
for the theorem's form rather than against it. It also gives the open question in
the Conclusion — whether an f(tw)·m^g(p) algorithm exists — a concrete
constituency: montage has p = 9 at every size, so an algorithm exponential in
treewidth but only polynomial in p would rescue exactly the family that the current
one cannot.

One practical lever falls out: rounding runtimes to whole seconds collapses |L|
sharply in some families (montage 34986 → 385, seismology 415 → 5) and barely at
all in others (srasearch 32 → 29). How coarsely the principal is willing to reason
is therefore a design parameter of the intervention, not an implementation detail.

### Caveats

Three caveats to carry into the write-up.

- The |L| figures assume **uniform course weights**, since neither file carries a
  workload column. |W| = 1 is an assumption, not a measurement, and realistic
  weights would widen |L|.
- The Johns Hopkins network is **extracted from catalogue prose**, and roughly two
  in five prerequisite blurbs express a disjunction that the extraction reads as a
  conjunction. That adds arcs, and since treewidth never falls when arcs are added,
  the figures above are upper bounds on the real curriculum — the safe direction
  for C1, but it means only the "≤ 25" is a claim about Johns Hopkins itself. The
  lower bound of 16 describes our reading, not their catalogue.
- Caltech closures are **small** (7–17 nodes). Johns Hopkins supplies the harder
  material, up to 55 nodes and treewidth 11, which is why both are in the corpus.

## Topological families

Structural generators imitate shapes catalogued for scientific workflows:
layered fan-out/fan-in (Montage, CyberShake), independent parallel pipelines
(Epigenomics), and series-parallel nesting.

- Bharathi et al., *Characterization of scientific workflows* — the standard
  characterisation of Montage, CyberShake, Epigenomics, LIGO and SIPHT. The
  Pegasus project publishes DAG representations of these workflows, which are a
  candidate corpus for E2.

## Candidate real-world corpora for E2

- **Course-prerequisite networks.** Stavrinides & Zuev, *Course-prerequisite
  networks for analyzing and understanding academic curricula*, Applied Network
  Science 2023 ([arXiv:2210.01269](https://arxiv.org/abs/2210.01269)) — the
  CalTech CPN is public on GitHub. Cited by our paper as [29].
- **Five Midwestern public institutions.** Yang et al., Applied Network Science
  2024 ([Springer](https://link.springer.com/article/10.1007/s41109-024-00637-z))
  — a comparative set built on the above. Cited by our paper as [31].
- **Scientific workflows.** Pegasus DAG XML for the five applications above.
- **CI/CD pipelines.** GitHub Actions workflow files parse directly into DAGs;
  large-scale mining is straightforward but the corpus would be ours to build.

The first two are the strongest starting point: they are public, they are already
cited in our paper, and they are task graphs in the sense the model intends.
