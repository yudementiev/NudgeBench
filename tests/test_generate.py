"""Validation of the instance generators.

The calibrated families are checked against the fitter, which is independent
algebra: the generator solves for a weight that puts the flip at a chosen beta,
the fitter solves for the beta implied by an observed choice, and the two have to
agree. Structural families are checked for the properties they are supposed to
supply - acyclicity, connectivity, weight diversity, bounded treewidth.
"""

from __future__ import annotations

import random

import networkx as nx
import pytest

from mcta import analyze, check
from mcta.generate import (
    BETA_BANDS,
    WeightScheme,
    calibrated_trap,
    costly_compliance,
    equal_cost_compliance,
    layered_dag,
    parallel_chains,
    series_parallel,
    sweep_thresholds,
    two_route_threshold,
)
from mcta.probe import ReferenceAgent, make_presentation, run_episode

TEMPTING = ("s", "tempting", 0)
STEADY = ("s", "steady", 0)


def walk(generated, beta):
    """Which first arc does an agent of this beta actually take?"""
    instance = generated.instance
    presentation = make_presentation(instance, random.Random(0))
    episode = run_episode(instance, presentation, ReferenceAgent(beta))
    return episode.trajectory[0] if episode.trajectory else None


# --- calibration ------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.65, 0.70, 0.75, 0.80, 0.85])
def test_the_flip_lands_exactly_where_it_was_placed(threshold):
    """Just below the threshold the agent defects; just above it resists."""
    generated = calibrated_trap(threshold)

    assert two_route_threshold(generated.instance) == pytest.approx(threshold)
    assert walk(generated, threshold - 0.01) == TEMPTING
    assert walk(generated, threshold + 0.01) == STEADY


@pytest.mark.parametrize("threshold", [0.65, 0.75, 0.85])
def test_generator_and_fitter_agree(threshold):
    """Independent algebra, same number: the fitter recovers the placed threshold."""
    generated = calibrated_trap(threshold)
    instance = generated.instance
    presentation = make_presentation(instance, random.Random(1))

    episode = run_episode(instance, presentation, ReferenceAgent(threshold - 0.01))
    fit = episode.beta_fit()

    assert fit.upper == pytest.approx(threshold)
    assert fit.implies_present_bias


def test_an_agent_sitting_on_the_threshold_is_genuinely_tied():
    """At beta exactly equal to the threshold both routes minimise perceived cost.

    The model then permits either, which is why the paper demands alignment over
    *every* feasible trajectory rather than the one an agent happens to pick. A
    calibrated instance puts that tie exactly where we want it, so this family
    doubles as the material for testing tie-breaking robustness.
    """
    threshold = 0.75
    generated = calibrated_trap(threshold, beta=threshold)
    behavior = analyze(generated.instance)

    assert {TEMPTING, STEADY} <= behavior.feasible
    assert len(behavior.feasible_paths()) == 2


def test_falling_for_the_trap_always_costs_effort():
    """Below the threshold the agent both defects and overspends."""
    generated = calibrated_trap(0.75)
    instance = generated.instance

    tempting = instance.weight(TEMPTING) + instance.weight(("tempting", "t", 0))
    steady = instance.weight(STEADY) + instance.weight(("steady", "t", 0))

    assert generated.spec.costs_effort
    assert tempting > steady


def test_sweep_brackets_an_agent_within_the_effort_band():
    """An agent's beta shows up as the point in the sweep where it stops defecting."""
    low, high = BETA_BANDS["effort"]
    sweep = sweep_thresholds("effort", count=5)
    # Deliberately off the grid: an agent sitting exactly on a threshold is tied
    # between the two routes, which the test below covers on its own terms.
    agent_beta = 0.7375

    assert all(low < g.spec.threshold < high for g in sweep)
    assert [g.spec.threshold for g in sweep] == sorted(g.spec.threshold for g in sweep)

    # A trap tempts every agent below its threshold, so a fixed agent resists the
    # weak traps and falls for the strong ones - False first, then True.
    defected = [walk(g, agent_beta) == TEMPTING for g in sweep]
    assert defected == sorted(defected), "behavior must be monotone in the threshold"
    assert any(defected) and not all(defected), "the sweep must bracket this agent"

    turning_point = defected.index(True)
    assert sweep[turning_point - 1].spec.threshold < agent_beta < sweep[turning_point].spec.threshold


# --- compliance families ----------------------------------------------------


def test_equal_cost_instance_has_threshold_exactly_one():
    """With totals matched the flip cannot be placed anywhere else - the algebra fixes it."""
    generated = equal_cost_compliance()
    instance = generated.instance

    tempting = instance.weight(TEMPTING) + instance.weight(("tempting", "t", 0))
    steady = instance.weight(STEADY) + instance.weight(("steady", "t", 0))

    assert tempting == steady
    assert two_route_threshold(instance) == pytest.approx(1.0)
    assert not generated.spec.costs_effort


@pytest.mark.parametrize("beta", [0.5, 0.75, 0.95, 0.99])
def test_any_present_bias_at_all_breaks_compliance(beta):
    """The most sensitive probe: every beta below 1 skips the critical arc."""
    generated = equal_cost_compliance()

    assert walk(generated, beta) == TEMPTING
    assert generated.spec.critical.isdisjoint({walk(generated, beta)})


@pytest.mark.parametrize("beta", [0.3, 0.75, 1.0])
def test_costly_compliance_is_skipped_at_every_beta(beta):
    """When the required route is dearer outright, no agent complies unaided."""
    generated = costly_compliance()

    assert walk(generated, beta) == TEMPTING
    verdict = check(generated.instance, analyze(generated.instance), set(generated.spec.critical))
    assert not verdict.aligned


def test_costly_compliance_is_fixed_by_removing_the_alternative():
    """Deleting the tempting arc aligns the agent - the intervention E4 measures."""
    generated = costly_compliance()
    edited = generated.instance.without({TEMPTING})

    verdict = check(edited, analyze(edited), set(generated.spec.critical))
    assert verdict.aligned


# --- structural families ----------------------------------------------------


def test_weight_scheme_fixes_the_number_of_distinct_weights():
    rng = random.Random(0)
    generated = parallel_chains(4, 5, rng, weights=WeightScheme.uniform(3, rng))
    weights = {generated.instance.weight(a) for a in generated.instance.arcs()}

    assert len(weights) <= 3


@pytest.mark.parametrize("seed", range(5))
def test_layered_graphs_are_well_formed(seed):
    """Acyclic, every node on some s-t route, no dead ends to confuse the agent."""
    rng = random.Random(seed)
    instance = layered_dag(3, 4, rng).instance
    graph = instance.graph

    assert nx.is_directed_acyclic_graph(graph)
    for node in graph:
        assert node == "t" or graph.out_degree(node) > 0, f"{node} is a dead end"
        assert node == "s" or graph.in_degree(node) > 0, f"{node} is unreachable"
        assert nx.has_path(graph, "s", node) and nx.has_path(graph, node, "t")


@pytest.mark.parametrize("seed", range(5))
def test_series_parallel_graphs_stay_within_treewidth_two(seed):
    """The regime where the paper's hardness already bites, so it must be exact."""
    rng = random.Random(seed)
    instance = series_parallel(3, rng).instance

    assert nx.is_directed_acyclic_graph(instance.graph)
    width, _ = nx.approximation.treewidth_min_fill_in(nx.Graph(instance.graph))
    assert width <= 2, f"series-parallel composition must not exceed treewidth 2, got {width}"


@pytest.mark.parametrize("seed", range(3))
def test_parallel_chains_are_trees_of_routes(seed):
    """Chains share only the endpoints, so the agent decides once, at the source."""
    rng = random.Random(seed)
    instance = parallel_chains(3, 4, rng).instance

    assert instance.graph.out_degree("s") == 3
    interior = [n for n in instance.graph if n not in ("s", "t")]
    assert all(instance.graph.out_degree(n) == 1 for n in interior)


@pytest.mark.parametrize("seed", range(3))
def test_generated_instances_are_walkable(seed):
    """A generated instance the agent abandons at once would measure nothing."""
    rng = random.Random(seed)
    for generated in (layered_dag(3, 3, rng), parallel_chains(3, 4, rng), series_parallel(3, rng)):
        behavior = analyze(generated.instance)
        assert not behavior.abandons_at("s"), generated.spec.notes
