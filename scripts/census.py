"""Structural census of the real corpora - the evidence behind claim C1.

Reproduces the numbers quoted in REFERENCES.md. Run after scripts/fetch_data.sh:

    .venv/bin/python scripts/census.py
"""

from __future__ import annotations

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
from mcta.structure import measure, treewidth_bounds, underlying  # noqa: E402


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
    print("Maniu, Senellart & Jog report treewidths in the hundreds for database")
    print("and road networks. Task graphs are a different kind of object.")


if __name__ == "__main__":
    main()
