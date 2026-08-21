"""Structural census of the real corpora - the evidence behind claim C1.

Reproduces the numbers quoted in REFERENCES.md. Run after scripts/fetch_data.sh:

    .venv/bin/python scripts/census.py
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.corpus.cpn import (  # noqa: E402
    candidate_targets,
    largest_component,
    load_caltech,
    load_jhu,
    to_instance,
)
from mcta.corpus.workflows import load_all
from mcta.corpus.workflows import to_instance as workflow_instance  # noqa: E402
from mcta.structure import (  # noqa: E402
    distinct_path_weights,
    longest_path_length,
    measure,
    treewidth_bounds,
    underlying,
)


def describe_network(name: str, graph: nx.DiGraph, published: str = "") -> None:
    active = graph.subgraph([n for n in graph if graph.degree(n) > 0]).copy()
    component = largest_component(graph)
    width = treewidth_bounds(underlying(nx.MultiDiGraph(component)))

    print(f"{name}")
    print(f"  listed              n={graph.number_of_nodes():>6} m={graph.number_of_edges():>6}")
    print(f"  non-isolated        n={active.number_of_nodes():>6} m={active.number_of_edges():>6}")
    print(
        f"  largest component   n={component.number_of_nodes():>6} "
        f"m={component.number_of_edges():>6}   treewidth {width}"
        + (f"   (published: {published})" if published else "")
    )


def describe_instances(name: str, graph: nx.DiGraph) -> None:
    targets = candidate_targets(graph, min_courses=6, min_depth=3)
    measured = [measure(to_instance(graph, target)) for target in targets]

    def spread(label: str, values: list[int]) -> str:
        ordered = sorted(values)
        p90 = ordered[min(int(0.9 * len(ordered)), len(ordered) - 1)]
        return (
            f"  {label:<10} min {min(values):>3}  median {int(statistics.median(values)):>3}"
            f"  p90 {p90:>3}  max {max(values):>3}"
        )

    print(f"\n{name}: {len(measured)} derived instances")
    print(spread("n", [p.nodes for p in measured]))
    print(spread("tw upper", [p.treewidth.upper for p in measured]))
    print(spread("p", [p.longest_path for p in measured]))
    counted = [p.distinct_path_weights for p in measured if p.distinct_path_weights is not None]
    if counted:
        print(spread("|L|", counted) + "   (uniform weights, |W|=1)")


def dp_state_exponent(treewidth: int, distinct: int | None) -> float:
    """log10 of the (|L| + 1)^tw state count Theorem 1 pays for.

    Treewidth alone does not decide whether the dynamic program runs; the theorem
    is exponential in treewidth with |L| as the base, so a narrow graph whose paths
    all weigh something different is still out of reach. This is the number that
    says which side of the line an instance falls on.
    """
    if distinct is None:
        return float("inf")
    return treewidth * math.log10(distinct + 1)


def describe_workflows() -> None:
    traces = load_all()
    print(f"\n{len(traces)} workflow traces (WfCommons, real runtimes)\n")
    header = f"  {'application':<14}{'tasks':>7}{'tw':>8}{'p':>5}{'|L|':>9}{'|L| q=1s':>10}{'log10 states':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for kind in sorted({t.kind for t in traces}):
        family = [t for t in traces if t.kind == kind]
        for trace in (min(family, key=lambda t: t.tasks), max(family, key=lambda t: t.tasks)):
            instance = workflow_instance(trace)
            width = treewidth_bounds(underlying(instance.graph))
            distinct = distinct_path_weights(instance, cap=2_000_000)
            quantized = distinct_path_weights(workflow_instance(trace, quantize=1.0), cap=2_000_000)
            exponent = dp_state_exponent(width.upper, distinct)
            verdict = "runs" if exponent <= 12 else "hopeless"
            print(
                f"  {kind:<14}{trace.tasks:>7}{str(width):>8}"
                f"{longest_path_length(instance):>5}{distinct if distinct else '>cap':>9}"
                f"{quantized if quantized else '>cap':>10}{exponent:>10.1f} {verdict}"
            )


def main() -> None:
    caltech, jhu = load_caltech(), load_jhu()

    print("=" * 72)
    print("WHOLE NETWORKS")
    print("=" * 72)
    describe_network("Caltech CPN (cleaned prerequisites)", caltech, "436 nodes, 747 links")
    print()
    describe_network("Johns Hopkins (extracted from prose; over-approximates)", jhu)

    print()
    print("=" * 72)
    print("DERIVED MCTA INSTANCES")
    print("=" * 72)
    describe_instances("Caltech", caltech)
    describe_instances("Johns Hopkins", jhu)

    print()
    print("=" * 72)
    print("SCIENTIFIC WORKFLOWS")
    print("=" * 72)
    describe_workflows()

    print()
    print("Maniu, Senellart & Jog report treewidths in the hundreds for database")
    print("and road networks. Task graphs are a different kind of object.")


if __name__ == "__main__":
    main()
