"""Validation of the scientific-workflow corpus.

These traces are the first instances with measured rather than assumed weights, so
the tests care about two things: that the runtimes survive the trip into an
instance intact, and that the corpus really does span both sides of the
tractability line rather than only the comfortable side.
"""

from __future__ import annotations

import networkx as nx
import pytest

from mcta.corpus.workflows import DEFAULT_ROOT, END, START, load_all, load_trace, to_instance
from mcta.model import analyze, path_weight
from mcta.structure import distinct_path_weights, treewidth_bounds, underlying

pytestmark = pytest.mark.skipif(
    not DEFAULT_ROOT.exists(), reason="run scripts/fetch_wfinstances.py to fetch the traces"
)


@pytest.fixture(scope="module")
def traces():
    return load_all()


def test_every_trace_is_an_acyclic_run_with_full_timing(traces):
    assert len(traces) >= 40
    for trace in traces:
        assert nx.is_directed_acyclic_graph(trace.graph)
        assert trace.tasks > 0
        assert all(data["runtime"] >= 0 for _, data in trace.graph.nodes(data=True))


def test_instances_are_bracketed_by_virtual_endpoints(traces):
    for trace in traces[:8]:
        instance = to_instance(trace)

        assert instance.graph.in_degree(START) == 0
        assert instance.graph.out_degree(END) == 0
        assert nx.has_path(instance.graph, START, END)
        assert nx.is_directed_acyclic_graph(instance.graph)


def test_arc_weights_are_the_measured_runtimes(traces):
    """A route must cost what performing those tasks actually cost."""
    trace = traces[0]
    instance = to_instance(trace)

    route = next(nx.all_simple_edge_paths(instance.graph, START, END))
    tasks = [head for _, head, _ in route if head != END]
    expected = sum(trace.graph.nodes[task]["runtime"] for task in tasks)

    assert path_weight(instance, route) == pytest.approx(expected)


def test_quantizing_never_increases_the_weight_diversity(traces):
    """Coarser units can only merge path weights, never split them."""
    for trace in traces[:6]:
        fine = distinct_path_weights(to_instance(trace), cap=2_000_000)
        coarse = distinct_path_weights(to_instance(trace, quantize=1.0), cap=2_000_000)

        assert fine is not None and coarse is not None
        assert coarse <= fine


def test_real_weights_are_genuinely_diverse(traces):
    """The point of this corpus: |W| is no longer 1 by assumption."""
    instance = to_instance(traces[0])
    weights = {instance.weight(arc) for arc in instance.arcs()}

    assert len(weights) > 5, "measured runtimes should not collapse to a handful of values"


def test_the_corpus_spans_both_sides_of_the_tractability_line(traces):
    """Small treewidth is necessary but not sufficient, and the corpus shows both.

    Theorem 1 pays (|L| + 1)^tw, so a narrow graph whose path weights are all
    distinct is still out of reach. Seismology sits at treewidth 2 and is trivial;
    montage at scale is hopeless despite a treewidth in the teens.
    """
    verdicts = set()
    for trace in traces:
        instance = to_instance(trace)
        width = treewidth_bounds(underlying(instance.graph)).upper
        distinct = distinct_path_weights(instance, cap=2_000_000)
        assert distinct is not None
        verdicts.add(width * (distinct + 1).bit_length() <= 60)

    assert verdicts == {True, False}, "a corpus on one side of the line tests nothing"


def test_montage_widens_as_the_problem_grows(traces):
    """Not every family stays narrow - which is the honest half of claim C1."""
    montage = sorted((t for t in traces if t.kind == "montage"), key=lambda t: t.tasks)
    widths = [treewidth_bounds(underlying(to_instance(t).graph)).upper for t in montage]

    assert widths[-1] > widths[0] + 5, f"expected growth with size, saw {widths}"


def test_parser_refuses_an_unknown_schema(tmp_path):
    """Silently misreading a future WfFormat would corrupt every number downstream."""
    rogue = tmp_path / "rogue.json"
    rogue.write_text('{"schemaVersion": "9.9", "workflow": {"specification": {"tasks": []}}}')

    with pytest.raises(ValueError, match="WfFormat"):
        load_trace(rogue)


def test_agent_walks_a_workflow_without_abandoning(traces):
    """An instance abandoned at START would measure nothing."""
    for trace in traces[:6]:
        instance = to_instance(trace)
        assert not analyze(instance).abandons_at(START)
