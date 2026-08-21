"""Measuring the structural parameters that govern tractability.

This is the machinery behind claim C1. The paper's positive results are stated in
terms of treewidth, vertex cover, feedback vertex set, path length and the
diversity of path weights, and its motivation asserts that these stay small in
real task graphs. Whether they do is an empirical question, and the citation the
paper offers - Maniu, Senellart and Jog's study of real-world graph data - in fact
reports treewidths in the hundreds for the corpora it examined.

Three of these parameters are NP-hard to compute, so every one of them is reported
as a certified interval rather than a number. A claim of the form "treewidth is at
most 6" is sound whatever the true value; a claim that it "is 6" would not be.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from .model import Instance


@dataclass(frozen=True)
class Bound:
    """A certified interval for a quantity that is hard to pin down exactly."""

    lower: int
    upper: int

    @property
    def exact(self) -> bool:
        return self.lower == self.upper

    def __str__(self) -> str:
        return str(self.lower) if self.exact else f"{self.lower}-{self.upper}"


@dataclass(frozen=True)
class Parameters:
    """Everything Theorem 8 of the paper is stated in terms of, for one graph."""

    nodes: int
    arcs: int
    treewidth: Bound
    pathwidth: Bound
    vertex_cover: Bound
    feedback_vertex_set: Bound
    longest_path: int
    """p: the maximum number of arcs on any s-t path."""

    distinct_weights: int
    """|W|."""

    distinct_path_weights: int | None = None
    """|L|, the quantity the treewidth algorithm is exponential in. None if too large to count."""

    def summary(self) -> str:
        return (
            f"n={self.nodes} m={self.arcs} tw={self.treewidth} pw={self.pathwidth} "
            f"vc={self.vertex_cover} fvs={self.feedback_vertex_set} "
            f"p={self.longest_path} |W|={self.distinct_weights} "
            f"|L|={self.distinct_path_weights if self.distinct_path_weights is not None else '>cap'}"
        )


def underlying(graph: nx.MultiDiGraph) -> nx.Graph:
    """The simple undirected graph the parameters are defined on."""
    simple = nx.Graph()
    simple.add_nodes_from(graph)
    simple.add_edges_from((u, v) for u, v in graph.edges() if u != v)
    return simple


# --- treewidth --------------------------------------------------------------


def minor_min_width(graph: nx.Graph) -> int:
    """A lower bound on treewidth: the minor-min-width heuristic.

    Treewidth never increases when passing to a minor, and every graph has a vertex
    of degree at most its treewidth. So repeatedly noting the minimum degree and
    then contracting that vertex into its least-connected neighbour walks down a
    chain of minors, and the largest minimum degree seen along the way is a valid
    floor. Contracting rather than deleting is what makes this stronger than
    degeneracy, which only ever deletes.
    """
    working = nx.Graph(graph)
    working.remove_edges_from(nx.selfloop_edges(working))
    best = 0

    while working.number_of_nodes() > 1:
        victim = min(working, key=lambda v: working.degree(v))
        degree = working.degree(victim)
        best = max(best, degree)
        if degree == 0:
            working.remove_node(victim)
            continue
        # Contract into the neighbour that shares the least, which keeps the
        # resulting minor sparse and the bound as tight as possible.
        neighbours = list(working.neighbors(victim))
        into = min(neighbours, key=lambda u: working.degree(u))
        for other in neighbours:
            if other != into:
                working.add_edge(into, other)
        working.remove_node(victim)

    return best


def treewidth_bounds(graph: nx.Graph) -> Bound:
    """Bracket treewidth between a minor-based floor and a constructed decomposition.

    The upper end comes from the better of two standard elimination heuristics,
    each of which exhibits an actual tree decomposition and so certifies itself.
    """
    if graph.number_of_nodes() == 0:
        return Bound(0, 0)
    if graph.number_of_edges() == 0:
        return Bound(0, 0)

    lower = max(minor_min_width(graph), max(nx.core_number(graph).values()))
    fill_in, _ = nx.approximation.treewidth_min_fill_in(graph)
    min_degree, _ = nx.approximation.treewidth_min_degree(graph)
    upper = min(fill_in, min_degree)
    return Bound(min(lower, upper), upper)


def pathwidth_bounds(graph: nx.Graph, treewidth: Bound) -> Bound:
    """Bracket pathwidth. It is never below treewidth, and never above the bound below.

    Every graph satisfies pw <= (tw + 1) * ceil(log2 n) - 1, so a treewidth
    decomposition certifies a pathwidth ceiling without constructing one.
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return Bound(0, 0)
    ceiling = (treewidth.upper + 1) * math.ceil(math.log2(n)) - 1
    return Bound(treewidth.lower, min(ceiling, n - 1))


# --- covers and cuts --------------------------------------------------------


def vertex_cover_bounds(graph: nx.Graph) -> Bound:
    """Bracket the minimum vertex cover.

    Any matching forces one endpoint of each of its edges into the cover, which
    certifies the lower end; taking both endpoints of a maximal matching is a
    cover, which certifies the upper end at twice that.
    """
    matching = nx.maximal_matching(graph)
    isolated = {v for v in graph if graph.degree(v) == 0}
    covered = {v for edge in matching for v in edge}
    upper = len(covered)

    # The matching is maximal, so every edge touches it; a better lower bound comes
    # from a maximum matching, which is exact on bipartite graphs by Konig.
    maximum = nx.max_weight_matching(graph, maxcardinality=True) if graph.number_of_edges() else set()
    lower = len(maximum)

    remaining = graph.number_of_nodes() - len(isolated)
    return Bound(min(lower, upper), min(upper, remaining))


def feedback_vertex_set_bounds(graph: nx.Graph) -> Bound:
    """Bracket the minimum feedback vertex set of the *undirected* graph.

    The paper defines all parameters on the underlying undirected graph, where a
    DAG can still carry many cycles. A spanning forest leaves m - n + c independent
    cycles, which bounds how many vertices could possibly be needed above, and the
    cycle space dimension divided by the largest degree bounds it below.
    """
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    components = nx.number_connected_components(graph)
    cycle_space = m - n + components
    if cycle_space <= 0:
        return Bound(0, 0)

    upper = _greedy_feedback_vertex_set(graph)
    max_degree = max(dict(graph.degree()).values())
    lower = math.ceil(cycle_space / max_degree) if max_degree else 0
    return Bound(min(lower, upper), upper)


def _greedy_feedback_vertex_set(graph: nx.Graph) -> int:
    """Repeatedly remove the highest-degree vertex on a cycle until none remain."""
    working = graph.copy()
    removed = 0
    while True:
        # Strip vertices that cannot lie on a cycle.
        while True:
            leaves = [v for v in working if working.degree(v) <= 1]
            if not leaves:
                break
            working.remove_nodes_from(leaves)
        if working.number_of_nodes() == 0:
            return removed
        victim = max(working, key=lambda v: working.degree(v))
        working.remove_node(victim)
        removed += 1


# --- path structure ---------------------------------------------------------


def longest_path_length(instance: Instance) -> int:
    """p: the greatest number of arcs on any source-target path."""
    best = {node: -math.inf for node in instance.graph}
    best[instance.source] = 0
    for node in nx.topological_sort(instance.graph):
        if best[node] == -math.inf:
            continue
        for _, head, _ in instance.graph.out_edges(node, keys=True):
            best[head] = max(best[head], best[node] + 1)
    return int(best[instance.target]) if best[instance.target] > -math.inf else 0


def distinct_path_weights(instance: Instance, cap: int = 200_000) -> int | None:
    """|L|: how many distinct weights any node-to-target path can have.

    This is the quantity the treewidth dynamic program is exponential in, so it
    decides whether Theorem 1 is usable on a given graph as much as treewidth does.
    Returns None if the count exceeds `cap`, which is itself the finding.
    """
    reachable: dict[str, set[float]] = {}
    for node in reversed(list(nx.topological_sort(instance.graph))):
        if node == instance.target:
            reachable[node] = {0.0}
            continue
        weights: set[float] = set()
        for arc in instance.out_arcs(node):
            weights.update(instance.weight(arc) + w for w in reachable.get(arc[1], ()))
            if len(weights) > cap:
                return None
        reachable[node] = weights

    everything: set[float] = set()
    for weights in reachable.values():
        everything.update(weights)
        if len(everything) > cap:
            return None
    return len(everything)


def measure(instance: Instance, *, cap: int = 200_000) -> Parameters:
    """Compute every parameter Theorem 8 is stated in terms of, with honest bounds."""
    simple = underlying(instance.graph)
    treewidth = treewidth_bounds(simple)

    return Parameters(
        nodes=instance.graph.number_of_nodes(),
        arcs=instance.graph.number_of_edges(),
        treewidth=treewidth,
        pathwidth=pathwidth_bounds(simple, treewidth),
        vertex_cover=vertex_cover_bounds(simple),
        feedback_vertex_set=feedback_vertex_set_bounds(simple),
        longest_path=longest_path_length(instance),
        distinct_weights=len({instance.weight(a) for a in instance.arcs()}),
        distinct_path_weights=distinct_path_weights(instance, cap=cap),
    )
