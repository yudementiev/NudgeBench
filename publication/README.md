# Camera-ready paper

**Nudging the Graph, Not the Agent: A Benchmark for Alignment by Environment Design**
Yuriy Dementiev, ITMO University
Foundations of Agentic Systems Theory (FAST), NeurIPS 2026 Workshop

[Read the PDF](FAST2026_NudgeBench_camera_ready.pdf).

## Build

Install a TeX distribution with `latexmk` and run from the repository root:

```bash
cd publication
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build neurips_2026.tex
```

The result is `publication/build/neurips_2026.pdf`. The bibliography is embedded
in the main `.tex` file; no separate `.bib` file or BibTeX invocation is needed.
The checked-in PDF has 16 pages; the main text and acknowledgments end on page 7.

This version includes the review revisions separating model guarantees from
finite-rollout evidence, reporting model-consistent beta intervals, qualifying
the interpretation of deferral and completing the bibliography. Numerical results
and the requested funding acknowledgment are preserved; no new experiments were
run for those editorial revisions.

Source repository: <https://github.com/yudementiev/NudgeBench>.
