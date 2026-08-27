"""Validation of the agent task-decomposition corpus."""

from __future__ import annotations

import networkx as nx
import pytest

from mcta.corpus.taskbench import DEFAULT_ROOT, DOMAINS, END, START, load_domain, to_instance
from mcta.model import analyze

pytestmark = pytest.mark.skipif(
    not DEFAULT_ROOT.exists(), reason="run scripts/fetch_taskbench.py to fetch the data"
)


@pytest.fixture(scope="module")
def multimedia():
    return load_domain("multimedia")


def test_every_domain_loads_as_acyclic_decompositions():
    for domain in DOMAINS:
        decompositions = load_domain(domain)
        assert len(decompositions) > 1000
        for decomposition in decompositions[:200]:
            assert nx.is_directed_acyclic_graph(decomposition.graph)
            assert decomposition.calls > 0


def test_topology_labels_match_the_graphs(multimedia):
    """TaskBench labels each record; the label must not contradict the structure."""
    for decomposition in multimedia[:400]:
        if decomposition.topology == "single":
            assert decomposition.calls == 1 or decomposition.graph.number_of_edges() == 0
        if decomposition.topology == "chain":
            assert decomposition.branching == 0, "a chain offers no choice by definition"


def test_instances_are_bracketed_and_walkable(multimedia):
    for decomposition in multimedia[:60]:
        instance = to_instance(decomposition)
        assert instance.graph.in_degree(START) == 0
        assert instance.graph.out_degree(END) == 0
        assert nx.has_path(instance.graph, START, END)
        assert not analyze(instance).abandons_at(START)


def test_node_order_is_stable(multimedia):
    """Same defect class as the course loader: prompts must regenerate identically."""
    decomposition = multimedia[0]
    first = list(to_instance(decomposition).graph.nodes())
    second = list(to_instance(decomposition).graph.nodes())

    assert first == second


def test_agent_decompositions_almost_never_offer_a_choice(multimedia):
    """The corpus's headline property, asserted so it cannot drift unnoticed.

    Dependency structure here is overwhelmingly fan-in - several inputs to one
    call - which is not a decision for an agent walking a route. Fan-out is, and it
    is rare. This is why the benchmark needs generated families at all.
    """
    branchy = [d for d in multimedia if d.branching > 0]

    assert len(branchy) / len(multimedia) < 0.15

