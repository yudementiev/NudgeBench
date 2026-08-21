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
- Augenblick et al., *Estimating present bias and sophistication over effort and
  money*, Experimental Economics
  ([Cambridge Core](https://www.cambridge.org/core/journals/experimental-economics/article/estimating-present-bias-and-sophistication-over-effort-and-money/308B6EA8834BE182081267488BCFF07F))
  — effort β of 0.70–0.79, with sophistication 0.80–0.88. Present bias over real
  effort is stronger than over money.

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
workflows, which plausibly behave differently. But the reference does not support
the sentence it is attached to, and E2 is what would settle the question. This is
claim **C1** in the workshop proposal, and it is now concrete rather than
rhetorical.

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
