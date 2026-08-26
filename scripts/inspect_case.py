"""Look closely at one instance and what the agent did on it.

    .venv/bin/python scripts/inspect_case.py EN.510.445
"""

from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx  # noqa: E402

from mcta.model import path_weight, shortest_distances  # noqa: E402
from mcta.probe import (  # noqa: E402
    Ledger, OpenRouter, Transcript, make_presentation, run_episode,
)
from mcta.structure import measure  # noqa: E402
from scripts.e1 import PRICES  # noqa: E402
from scripts.e1_complexity import BINS, branching, collect  # noqa: E402

MODEL = "google/gemini-3.7-flash"
TRANSCRIPT = Path("runs/e1complexity-google-gemini-3.7-flash-uncertain.jsonl")


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else "EN.510.445"
    pool = collect(8)

    for name, _, _ in BINS:
        for index, (case, instance) in enumerate(pool[name]):
            if wanted not in case:
                continue

            graph = instance.graph
            distance = shortest_distances(instance)
            parameters = measure(instance)
            print(f"{case}   bin {name}")
            print(f"  {graph.number_of_nodes()} nodes, {graph.number_of_edges()} arcs, "
                  f"{branching(instance)} choice points, treewidth {parameters.treewidth}")
            print(f"  shortest route costs {distance[instance.source]:g}, "
                  f"longest path is {parameters.longest_path} arcs, reward {instance.reward:g}")
            weights = sorted({instance.weight(a) for a in instance.arcs()})
            print(f"  distinct weights {weights}")
            out = sorted(((graph.out_degree(v), v) for v in graph), reverse=True)[:4]
            print(f"  widest choice points: {[(v, d) for d, v in out]}")

            routes = list(nx.all_simple_edge_paths(graph, instance.source, instance.target))
            costs = sorted(path_weight(instance, r) for r in routes)
            print(f"  {len(routes)} distinct routes, costs from {costs[0]:g} to {costs[-1]:g}"
                  f" (median {costs[len(costs)//2]:g})")

            ledger = Ledger(cap=0.0)
            agent = OpenRouter(MODEL, PRICES[MODEL], ledger, Transcript(TRANSCRIPT),
                               max_tokens=8000)
            for repetition in range(2):
                seed = zlib.crc32(f"{name}:{index}:{repetition}".encode()) % 100_000
                agent.sample = seed
                presentation = make_presentation(
                    instance, random.Random(seed), framing="uncertain")
                episode = run_episode(instance, presentation, agent)
                walked = path_weight(instance, episode.trajectory)
                fit = episode.beta_fit()
                print(f"\n  repetition {repetition}: walked cost {walked:g} "
                      f"against optimal {distance[instance.source]:g}"
                      f"  ({walked / distance[instance.source]:.2f}x)")
                print(f"    plan kept: {episode.plan_survived_execution}, {fit.describe()}")
                for constraint in fit.constraints:
                    if constraint.upper < 0.999 and constraint.rejected:
                        print(f"    binding choice at {constraint.node}: took "
                              f"{presentation.arc_label[constraint.chosen]} over "
                              f"{presentation.arc_label[constraint.rejected]} -> {constraint.note}")
                        break
                if episode.step_replies:
                    print(f"    first move reasoning (tail):\n      "
                          f"...{episode.step_replies[0][-400:]}")
            print(f"\n  paid calls: {ledger.calls}")
            return 0

    print(f"{wanted} not in the sweep")
    return 1


if __name__ == "__main__":
    sys.exit(main())
