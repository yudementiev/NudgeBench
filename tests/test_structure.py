"""Validation of the structural measurements behind claim C1.

Every parameter here is NP-hard to compute exactly, so the module reports
intervals. The tests check the property that matters: the interval must contain
the true value on graphs whose treewidth, vertex cover and feedback vertex set are
known from theory. A bound that excluded the truth would silently corrupt every
number in the paper.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from mcta.generate import parallel_chains, series_parallel
from mcta.model import Instance
from mcta.structure import (
    feedback_vertex_set_bounds,
    longest_path_length,
    measure,
    minor_min_width,
    treewidth_bounds,
    underlying,
    vertex_cover_bounds,
)

# Graphs whose treewidth is settled by theory.
KNOWN_TREEWIDTH = [
    (nx.path_graph(8), 1, "a path"),
    (nx.cycle_graph(9), 2, "a cycle"),
    (nx.complete_graph(6), 5, "a clique on 6"),
    (nx.balanced_tree(2, 3), 1, "a tree"),
    (nx.grid_2d_graph(4, 4), 4, "a 4x4 grid"),
    (nx.complete_bipartite_graph(3, 3), 3, "K(3,3)"),
]


@pytest.mark.parametrize("graph, truth, name", KNOWN_TREEWIDTH)
def test_treewidth_bounds_contain_the_truth(graph, truth, name):
    bound = treewidth_bounds(graph)
    assert bound.lower <= truth <= bound.upper, f"{name}: {bound} excludes {truth}"


@pytest.mark.parametrize("graph, truth, name", KNOWN_TREEWIDTH)
def test_minor_min_width_never_overshoots(graph, truth, name):
    """A lower bound that exceeded treewidth would be worse than useless."""
    assert minor_min_width(graph) <= truth, f"{name}: floor above the true treewidth"


def test_bounds_do_not_depend_on_how_the_graph_was_built():
    """Published numbers have to reproduce, so tie-breaking must not follow insertion order."""
    grid = nx.grid_2d_graph(6, 6)
    forward = nx.Graph()
    forward.add_nodes_from(sorted(grid.nodes()))
    forward.add_edges_from(grid.edges())
    backward = nx.Graph()
    backward.add_nodes_from(sorted(grid.nodes(), reverse=True))
    backward.add_edges_from(grid.edges())

    assert minor_min_width(forward) == minor_min_width(backward)
    assert treewidth_bounds(forward) == treewidth_bounds(backward)


def test_minor_min_width_beats_degeneracy_somewhere():
    """Contracting rather than deleting is the point; on a grid it pays off."""
    grid = nx.grid_2d_graph(5, 5)
    degeneracy = max(nx.core_number(grid).values())

    assert minor_min_width(grid) > degeneracy


@pytest.mark.parametrize(
    "graph, truth",
    [
        (nx.path_graph(6), 3),          # alternate vertices
        (nx.complete_graph(5), 4),      # all but one
        (nx.cycle_graph(6), 3),
        (nx.star_graph(7), 1),          # the hub alone
    ],
)
def test_vertex_cover_bounds_contain_the_truth(graph, truth):
    bound = vertex_cover_bounds(graph)
    assert bound.lower <= truth <= bound.upper, f"{bound} excludes {truth}"


@pytest.mark.parametrize(
    "graph, truth",
    [
        (nx.path_graph(6), 0),          # already a forest
        (nx.balanced_tree(2, 3), 0),
        (nx.cycle_graph(7), 1),
        (nx.complete_graph(5), 3),      # leave a tree on 2
    ],
)
def test_feedback_vertex_set_bounds_contain_the_truth(graph, truth):
    bound = feedback_vertex_set_bounds(graph)
    assert bound.lower <= truth <= bound.upper, f"{bound} excludes {truth}"


def test_longest_path_counts_arcs_not_nodes():
    graph = nx.MultiDiGraph()
    for tail, head in [("s", "a"), ("a", "b"), ("b", "t"), ("s", "t")]:
        graph.add_edge(tail, head, key=0, weight=1.0)
    instance = Instance(graph, "s", "t", 0.5, 100.0)

    assert longest_path_length(instance) == 3


def test_series_parallel_bounds_stay_at_two():
    """The regime of the paper's hardness result: the measurement must be tight here."""
    rng = random.Random(0)
    for _ in range(5):
        instance = series_parallel(3, rng).instance
        bound = treewidth_bounds(underlying(instance.graph))
        assert bound.upper <= 2, f"series-parallel graph measured at {bound}"


def test_distinct_path_weights_counts_what_the_dp_pays_for():
    """|L| is what Theorem 1 is exponential in, so it has to be counted exactly."""
    rng = random.Random(0)
    instance = parallel_chains(3, 3, rng).instance
    parameters = measure(instance)

    routes = {
        sum(instance.weight(a) for a in path)
        for path in nx.all_simple_edge_paths(instance.graph, "s", "t")
    }
    # Every distinct route weight must appear among the counted path weights.
    assert parameters.distinct_path_weights >= len(routes)


def test_measurement_reports_a_cap_rather_than_lying():
    """When |L| explodes the count is refused, not truncated."""
    rng = random.Random(1)
    instance = parallel_chains(6, 4, rng).instance
    parameters = measure(instance, cap=2)

    assert parameters.distinct_path_weights is None
    assert ">cap" in parameters.summary()
