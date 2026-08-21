"""The Kleinberg-Oren model of time-inconsistent planning.

An instance is a tuple M = (G, w, s, t, beta, r) where G is a weighted DAG.
Standing at v, the agent evaluates every v-t path P = e_1 e_2 ... e_k under

    zeta(P) = w(e_1) + beta * sum_{i>=2} w(e_i),

abandons the task if min_P zeta(P) > beta * r, and otherwise commits to e_1
and repeats. Ties are broken arbitrarily, so the model induces a *set* of
feasible trajectories rather than a single path.

Notation follows Belova et al., "Structural Alignment in Inconsistent Planning".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import networkx as nx

# An arc is identified by (tail, head, key); the key distinguishes parallel arcs,
# which the model explicitly permits.
Arc = tuple[str, str, int]


@dataclass(frozen=True)
class Instance:
    """A time-inconsistent planning model M = (G, w, s, t, beta, r)."""

    graph: nx.MultiDiGraph
    source: str
    target: str
    beta: float
    reward: float

    def __post_init__(self) -> None:
        if not 0 < self.beta <= 1:
            raise ValueError(f"beta must lie in (0, 1], got {self.beta}")
        for node in (self.source, self.target):
            if node not in self.graph:
                raise ValueError(f"node {node!r} is not in the graph")
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("the task graph must be acyclic")
        for tail, head, key, data in self.graph.edges(keys=True, data=True):
            if "weight" not in data:
                raise ValueError(f"arc {(tail, head, key)} has no weight")
            if data["weight"] < 0:
                raise ValueError(f"arc {(tail, head, key)} has negative weight")

    @property
    def perceived_reward(self) -> float:
        """The reward as the agent perceives it at every decision point."""
        return self.beta * self.reward

    def weight(self, arc: Arc) -> float:
        tail, head, key = arc
        return self.graph.edges[tail, head, key]["weight"]

    def arcs(self) -> Iterator[Arc]:
        yield from self.graph.edges(keys=True)

    def out_arcs(self, node: str) -> Iterator[Arc]:
        yield from self.graph.out_edges(node, keys=True)

    def without(self, deleted: set[Arc]) -> "Instance":
        """This instance with `deleted` removed - the principal's intervention."""
        reduced = self.graph.copy()
        reduced.remove_edges_from(deleted)
        return Instance(reduced, self.source, self.target, self.beta, self.reward)

    def with_arcs(self, added: dict[Arc, float]) -> "Instance":
        """This instance with auxiliary arcs added, each at the given weight."""
        extended = self.graph.copy()
        for (tail, head, key), weight in added.items():
            extended.add_edge(tail, head, key=key, weight=weight)
        return Instance(extended, self.source, self.target, self.beta, self.reward)


@dataclass
class Behavior:
    """Everything about how the agent moves through one instance.

    Computed once per instance; every downstream analysis reads from here.
    """

    instance: Instance
    distance: dict[str, float] = field(default_factory=dict)
    """Plain shortest-path weight from each node to the target (inf if unreachable)."""

    zeta: dict[str, float] = field(default_factory=dict)
    """Minimum perceived cost of continuing from each node (inf if stuck)."""

    feasible: set[Arc] = field(default_factory=set)
    """Arcs the agent may commit to: minimisers at a node it does not abandon."""

    def abandons_at(self, node: str) -> bool:
        """Whether an agent standing at `node` gives up."""
        if node == self.instance.target:
            return False
        return self.zeta[node] > self.instance.perceived_reward

    def reachable_nodes(self) -> set[str]:
        """Nodes the agent can actually stand at, following feasible arcs from s."""
        seen = {self.instance.source}
        stack = [self.instance.source]
        while stack:
            node = stack.pop()
            for arc in self.instance.out_arcs(node):
                if arc in self.feasible and arc[1] not in seen:
                    seen.add(arc[1])
                    stack.append(arc[1])
        return seen

    def feasible_paths(self, limit: int = 10_000) -> list[list[Arc]]:
        """Every trajectory the agent may follow, as a list of arcs.

        A trajectory that ends anywhere other than the target is one on which the
        agent abandoned. Raises if the instance admits more than `limit` of them.
        """
        paths: list[list[Arc]] = []

        def walk(node: str, prefix: list[Arc]) -> None:
            if len(paths) > limit:
                raise RuntimeError(f"more than {limit} feasible paths")
            if node == self.instance.target or self.abandons_at(node):
                paths.append(prefix)
                return
            for arc in self.instance.out_arcs(node):
                if arc in self.feasible:
                    walk(arc[1], prefix + [arc])

        walk(self.instance.source, [])
        return paths


def shortest_distances(instance: Instance) -> dict[str, float]:
    """Plain (undiscounted) shortest-path weight from every node to the target."""
    distance = {node: math.inf for node in instance.graph}
    distance[instance.target] = 0.0
    for node in reversed(list(nx.topological_sort(instance.graph))):
        for arc in instance.out_arcs(node):
            candidate = instance.weight(arc) + distance[arc[1]]
            if candidate < distance[node]:
                distance[node] = candidate
    return distance


def perceived_cost(instance: Instance, arc: Arc, distance: dict[str, float]) -> float:
    """Cost of the cheapest continuation through `arc`, as the agent perceives it.

    The agent pays the first arc in full and discounts everything after it, so the
    best plan through `arc` costs w(arc) + beta * d(head, t).
    """
    return instance.weight(arc) + instance.beta * distance[arc[1]]


def analyze(instance: Instance) -> Behavior:
    """Compute the agent's behavior: distances, perceived costs, feasible arcs."""
    distance = shortest_distances(instance)
    zeta: dict[str, float] = {}
    feasible: set[Arc] = set()

    for node in instance.graph:
        if node == instance.target:
            zeta[node] = 0.0
            continue
        costs = {arc: perceived_cost(instance, arc, distance) for arc in instance.out_arcs(node)}
        zeta[node] = min(costs.values(), default=math.inf)

    for node in instance.graph:
        if node == instance.target or zeta[node] > instance.perceived_reward:
            # The agent either stops here or abandons; it commits to nothing.
            continue
        for arc in instance.out_arcs(node):
            if perceived_cost(instance, arc, distance) == zeta[node]:
                feasible.add(arc)

    return Behavior(instance=instance, distance=distance, zeta=zeta, feasible=feasible)


def path_perceived_cost(instance: Instance, path: list[Arc]) -> float:
    """zeta(P) - what the agent standing at the start of `path` thinks it costs."""
    if not path:
        return 0.0
    head, *rest = path
    return instance.weight(head) + instance.beta * sum(instance.weight(a) for a in rest)


def path_weight(instance: Instance, path: list[Arc]) -> float:
    """The true, undiscounted cost of walking `path`."""
    return sum(instance.weight(arc) for arc in path)
