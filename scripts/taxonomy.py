"""The instance taxonomy: what each domain of origin actually supplies.

A benchmark's value here is less in how many instances it holds than in knowing
what kind each is and which questions it can answer. Four sources contribute
graphs with very different shapes, and the differences are the point: agent task
decompositions are small and almost never branch, curricula branch modestly,
workflows are large but shallow in choice.

Columns worth reading carefully:

  branching  instances with any node offering a choice of continuation. Fan-in
             does not count - several inputs to one task is not a decision.
  discrim.   of those, the ones where a patient and a biased agent walk different
             routes. Only these can measure an agent at all.

    .venv/bin/python scripts/taxonomy.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.corpus import cpn, taskbench, workflows  # noqa: E402
from mcta.model import Instance  # noqa: E402
from mcta.structure import treewidth_bounds, underlying  # noqa: E402
from scripts.e1_corpus import PALETTE, discriminating  # noqa: E402


def branching(instance: Instance) -> int:
    return sum(1 for v in instance.graph if instance.graph.out_degree(v) > 1)


def summarise(name: str, instances: list[Instance], note: str = "") -> dict:
    if not instances:
        print(f"  {name:<22} none")
        return {}
    branchy = [i for i in instances if branching(i) > 0]
    discrim = [i for i in branchy if discriminating(i)]
    arcs = [i.graph.number_of_edges() for i in instances]
    widths = [treewidth_bounds(underlying(i.graph)).upper for i in instances[:60]]

    print(f"  {name:<22}{len(instances):>7}{len(branchy):>11}{len(discrim):>9}"
          f"{f'{min(arcs)}-{max(arcs)}':>10}{int(statistics.median(arcs)):>7}"
          f"{max(widths):>6}  {note}")
    return {
        "instances": len(instances), "branching": len(branchy),
        "discriminating": len(discrim), "arcs_min": min(arcs), "arcs_max": max(arcs),
        "arcs_median": statistics.median(arcs), "treewidth_max_sampled": max(widths),
    }


def main() -> int:
    rng = random.Random(0)
    print(f"  {'source':<22}{'instances':>7}{'branching':>11}{'discrim.':>9}"
          f"{'arcs':>10}{'median':>7}{'tw':>6}")
    print("  " + "-" * 76)

    print("\n  AGENT TASK DECOMPOSITIONS")
    for domain in taskbench.DOMAINS:
        decompositions = taskbench.load_domain(domain)
        charges = {}
        instances = []
        for decomposition in decompositions:
            for call in decomposition.graph:
                charges.setdefault(call, rng.choice(PALETTE))
            instances.append(taskbench.to_instance(
                decomposition, weight_of=charges.__getitem__))
        summarise(domain, instances, "synthetic weights")

    print("\n  HUMAN CURRICULA")
    for loader, name in ((cpn.load_caltech, "caltech"), (cpn.load_jhu, "johns hopkins")):
        network = loader()
        charges = {course: rng.choice(PALETTE) for course in network}
        instances = [
            cpn.to_instance(network, target, weight_of=charges.__getitem__)
            for target in cpn.candidate_targets(network, min_courses=6, min_depth=3)
        ]
        summarise(name, instances, "synthetic weights")

    print("\n  SCIENTIFIC WORKFLOWS")
    traces = workflows.load_all()
    for kind in sorted({t.kind for t in traces}):
        family = [t for t in traces if t.kind == kind]
        summarise(kind, [workflows.to_instance(t) for t in family], "measured runtimes")

    print("\n  Only instances in the 'discrim.' column can measure an agent: elsewhere")
    print("  a patient and a biased planner produce the same trajectory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
