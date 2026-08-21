"""E3: what does the intuitive fix cost, against the cheapest one?

For each real instance we pick a critical action the agent currently skips, then
solve twice: once with the witness-guided heuristic, which is what an engineer
does by hand, and once exactly. The gap between them, read against the instance's
structure, is the experiment.

    .venv/bin/python scripts/e3.py
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.align import t_avoiding_arcs  # noqa: E402
from mcta.corpus.cpn import candidate_targets, load_caltech, load_jhu, to_instance  # noqa: E402
from mcta.model import Arc, Instance, analyze  # noqa: E402
from mcta.solve import (  # noqa: E402
    greedy_witness_guided,
    is_alignable,
    minimalize,
    solve_exact,
)
from mcta.structure import treewidth_bounds, underlying  # noqa: E402

NODE_BUDGET = 60_000


def pick_critical(instance: Instance) -> set[Arc] | None:
    """An action on some route to the target that the agent will not take unaided.

    Choosing an arc the agent already traverses would make the instance aligned
    from the start and measure nothing; choosing one off every route would make it
    unsatisfiable. What is wanted is the interesting middle - a genuinely reachable
    action the present-biased agent skips.
    """
    behavior = analyze(instance)
    walked = {arc for path in behavior.feasible_paths(limit=2000) for arc in path}
    for arc in sorted(instance.arcs()):
        if arc in walked:
            continue
        critical = {arc}
        # An arc is only worth requiring if some route through it exists at all.
        if arc in t_avoiding_arcs(instance, critical):
            continue
        if is_alignable(instance, critical):
            return critical
    return None


@dataclass
class Row:
    name: str
    nodes: int
    arcs: int
    treewidth: int
    greedy: float
    exact: float
    proved: bool
    greedy_ms: float
    exact_ms: float
    oracle_calls: int

    @property
    def gap(self) -> float:
        if not self.proved or self.exact <= 0:
            return float("nan")
        return (self.greedy - self.exact) / self.exact


def evaluate(name: str, instance: Instance, critical: set[Arc]) -> Row | None:
    start = time.perf_counter()
    greedy = greedy_witness_guided(instance, critical)
    greedy = minimalize(instance, critical, greedy.deleted) if greedy.aligned else greedy
    greedy_ms = (time.perf_counter() - start) * 1000
    if not greedy.aligned:
        return None

    start = time.perf_counter()
    exact = solve_exact(instance, critical, node_budget=NODE_BUDGET)
    exact_ms = (time.perf_counter() - start) * 1000
    if not exact.aligned:
        return None

    return Row(
        name=name,
        nodes=instance.graph.number_of_nodes(),
        arcs=instance.graph.number_of_edges(),
        treewidth=treewidth_bounds(underlying(instance.graph)).upper,
        greedy=greedy.cost,
        exact=exact.cost,
        proved=exact.proved_optimal,
        greedy_ms=greedy_ms,
        exact_ms=exact_ms,
        oracle_calls=exact.oracle_calls,
    )


def run(label: str, instances: list[tuple[str, Instance]]) -> list[Row]:
    rows = []
    skipped = 0
    for name, instance in instances:
        critical = pick_critical(instance)
        if critical is None:
            skipped += 1
            continue
        row = evaluate(name, instance, critical)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    print(f"\n{label}: {len(rows)} solved, {skipped} skipped (already aligned or unsatisfiable)")
    if not rows:
        return rows

    proved = [r for r in rows if r.proved]
    worse = [r for r in proved if r.greedy > r.exact + 1e-9]
    print(f"  optimality proved on {len(proved)}/{len(rows)}")
    print(f"  heuristic strictly worse on {len(worse)}/{len(proved)}")
    if proved:
        gaps = [r.gap for r in proved]
        print(
            f"  cost ratio greedy/exact: median {statistics.median(g + 1 for g in gaps):.2f}"
            f"  worst {max(gaps) + 1:.2f}"
        )
        print(
            f"  time: greedy median {statistics.median(r.greedy_ms for r in proved):.1f} ms,"
            f"  exact median {statistics.median(r.exact_ms for r in proved):.1f} ms,"
            f"  exact worst {max(r.exact_ms for r in proved):.0f} ms"
        )
    unproved = [r for r in rows if not r.proved]
    if unproved:
        print(f"  budget exhausted on {len(unproved)} (arcs {min(r.arcs for r in unproved)}"
              f"-{max(r.arcs for r in unproved)})")

    if worse:
        print("  worst cases:")
        for row in sorted(worse, key=lambda r: -r.gap)[:5]:
            print(
                f"    {row.name[:28]:28s} n={row.nodes:>3} m={row.arcs:>3} tw={row.treewidth} "
                f"greedy={row.greedy:g} exact={row.exact:g} ({row.oracle_calls} oracle calls)"
            )
    return rows


def main() -> None:
    caltech = load_caltech()
    targets = candidate_targets(caltech, min_courses=6, min_depth=3)
    run("Caltech closures", [(t, to_instance(caltech, t)) for t in targets])

    jhu = load_jhu()
    jhu_targets = candidate_targets(jhu, min_courses=6, min_depth=3)
    small = [(t, to_instance(jhu, t)) for t in jhu_targets]
    small = [(n, i) for n, i in small if i.graph.number_of_edges() <= 28]
    run("Johns Hopkins closures (<= 28 arcs)", small)


if __name__ == "__main__":
    main()
