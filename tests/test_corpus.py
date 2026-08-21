"""Validation of the course-prerequisite corpus.

The published paper reports the network's size and that of its largest component,
so those are ground truth for the loader. Tests skip rather than fail when the
data has not been fetched, since it is deliberately not vendored.
"""

from __future__ import annotations

import networkx as nx
import pytest

from mcta.corpus import candidate_targets, largest_component, load_caltech, to_instance
from mcta.corpus.cpn import DEFAULT_PATH
from mcta.model import analyze
from mcta.structure import measure

pytestmark = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason="run scripts/fetch_data.sh to fetch the CPN data"
)


@pytest.fixture(scope="module")
def cpn():
    return load_caltech()


def test_loader_reproduces_the_published_network(cpn):
    """Stavrinides & Zuev report 771 courses and 772 prerequisite relations."""
    assert cpn.number_of_nodes() == 771
    assert cpn.number_of_edges() == 772
    assert nx.is_directed_acyclic_graph(cpn)


def test_largest_component_matches_the_published_figure(cpn):
    """The paper reports 436 nodes and 747 links in the largest component."""
    component = largest_component(cpn)

    assert component.number_of_nodes() == 436
    assert component.number_of_edges() == 747


def test_courses_carry_their_metadata(cpn):
    course = cpn.nodes["Ae 101 abc"]["course"]

    assert course.department == "Aerospace"
    assert "Fluid" in course.title


def test_derived_instances_are_well_formed(cpn):
    """Every instance must be a DAG in which the target is reachable from START."""
    for target in candidate_targets(cpn, min_courses=6, min_depth=3)[:20]:
        instance = to_instance(cpn, target)

        assert nx.is_directed_acyclic_graph(instance.graph)
        assert nx.has_path(instance.graph, "START", target)
        assert instance.graph.in_degree("START") == 0
        assert instance.graph.out_degree(target) == 0


def test_derived_instances_pose_a_real_choice(cpn):
    """An instance the agent walks with no decision to make would measure nothing."""
    targets = candidate_targets(cpn, min_courses=8, min_depth=3)
    branching = 0
    for target in targets[:20]:
        instance = to_instance(cpn, target)
        behavior = analyze(instance)
        assert not behavior.abandons_at("START")
        if any(instance.graph.out_degree(n) > 1 for n in instance.graph):
            branching += 1

    assert branching > 0, "no instance in the sample offers the agent a choice"


def test_real_curricula_are_structurally_narrow(cpn):
    """Claim C1, on real data: these task graphs sit far below the hard regime.

    The paper's motivation asserts this and cites a study of database and network
    data which in fact found treewidths in the hundreds. Course prerequisite
    closures are a different kind of object, and they are genuinely narrow - which
    is worth measuring ourselves rather than citing someone else for.
    """
    widths, depths = [], []
    for target in candidate_targets(cpn, min_courses=6, min_depth=3):
        parameters = measure(to_instance(cpn, target))
        widths.append(parameters.treewidth.upper)
        depths.append(parameters.longest_path)

    assert max(widths) <= 8, f"widest closure measured at treewidth {max(widths)}"
    assert max(depths) <= 12, f"longest curriculum route runs {max(depths)} courses"
