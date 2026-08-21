"""Reference instances with known behavior, used to validate the simulator."""

from __future__ import annotations

import networkx as nx

from .model import Arc, Instance

# The auxiliary arc Alice introduces in the paper's worked example. It is not
# present in the initial graph; `figure_one` returns it separately so tests can
# add it back at the weight the paper specifies.
FIGURE_ONE_AUXILIARY: tuple[Arc, float] = (("a", "b", 0), 1.0)

FIGURE_ONE_CRITICAL: set[Arc] = {("b", "e", 0)}


def figure_one() -> Instance:
    """Figure 1 of Belova et al.: Bob's LLM deployment project.

    Bob (beta = 1/2) is offered r = 36 for reaching t. Three parallel pipelines
    lead to the target, plus a cheap prompt-only shortcut (b, f) that tempts him
    away from the critical evaluation step (b, e).

    The paper records the perceived costs at s exactly: zeta(sbet) = zeta(scft) = 20,
    zeta(sadt) = 18, zeta(sbft) = 16, so the agent follows sbft.
    """
    graph = nx.MultiDiGraph()
    arcs = [
        ("s", "a", 10.0),
        ("a", "d", 8.0),
        ("d", "t", 8.0),
        ("s", "b", 10.0),
        ("b", "e", 10.0),
        ("e", "t", 10.0),
        ("s", "c", 10.0),
        ("c", "f", 10.0),
        ("f", "t", 10.0),
        ("b", "f", 2.0),
    ]
    for tail, head, weight in arcs:
        graph.add_edge(tail, head, key=0, weight=weight)

    return Instance(graph=graph, source="s", target="t", beta=0.5, reward=36.0)


def chain(length: int, weight: float = 1.0, beta: float = 0.5, reward: float = 100.0) -> Instance:
    """A single path s -> v1 -> ... -> t, where the agent has no choice to make."""
    graph = nx.MultiDiGraph()
    nodes = ["s"] + [f"v{i}" for i in range(1, length)] + ["t"]
    for tail, head in zip(nodes, nodes[1:]):
        graph.add_edge(tail, head, key=0, weight=weight)
    return Instance(graph=graph, source="s", target="t", beta=beta, reward=reward)


def two_route_trap(
    shortcut_weight: float = 1.0,
    detour_weight: float = 6.0,
    tail_weight: float = 14.0,
    beta: float = 0.5,
    reward: float = 40.0,
) -> Instance:
    """The smallest instance where present bias alone changes the chosen route.

    Both routes reach t. The shortcut is cheap now and expensive later; the detour
    splits its cost evenly. At the defaults a patient agent compares 1 + 14 against
    6 + 6 and detours, while a biased agent compares 1 + 14/2 against 6 + 6/2 and
    takes the shortcut - so the route flips on beta alone, with the graph fixed.
    """
    graph = nx.MultiDiGraph()
    graph.add_edge("s", "shortcut", key=0, weight=shortcut_weight)
    graph.add_edge("shortcut", "t", key=0, weight=tail_weight)
    graph.add_edge("s", "detour", key=0, weight=detour_weight)
    graph.add_edge("detour", "t", key=0, weight=detour_weight)
    return Instance(graph=graph, source="s", target="t", beta=beta, reward=reward)
