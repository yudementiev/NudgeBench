"""Validation of the MCTA solvers.

The exact solver is checked against an independent brute force written here rather
than in the library, so a shared mistake cannot hide. The heuristic is checked for
what it is: always feasible when it returns aligned, and not always optimal - the
gap it leaves is the thing E3 measures.
"""

from __future__ import annotations

import itertools
import random

import pytest

from mcta.examples import FIGURE_ONE_AUXILIARY, FIGURE_ONE_CRITICAL, figure_one
from mcta.generate import costly_compliance, layered_dag
from mcta.model import Arc, Instance, analyze
from mcta.solve import (
    Intervention,
    candidate_arcs,
    greedy_witness_guided,
    is_aligned,
    is_alignable,
    minimalize,
    solve_exact,
    total_cost,
    unit_cost,
)

BF = ("b", "f", 0)
AB = FIGURE_ONE_AUXILIARY[0]


def brute_force(instance: Instance, critical: set[Arc], cost=unit_cost) -> tuple[float, int]:
    """Every subset of the candidate arcs, independently of the library's search."""
    candidates = sorted(candidate_arcs(instance, critical))
    best, count = float("inf"), 0
    for size in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            if is_aligned(instance.without(set(subset)), critical):
                count += 1
                best = min(best, total_cost(subset, cost))
    return best, count


# --- the paper's example ----------------------------------------------------


def test_exact_solver_reproduces_alices_intervention():
    """With the auxiliary arc available, the cheapest fix is to cut the shortcut.

    The paper's story in solver form: (a, b) is present but useless while (b, f)
    tempts Bob away, and removing (b, f) alone is what makes him walk sabet.
    """
    arc, weight = FIGURE_ONE_AUXILIARY
    instance = figure_one().with_arcs({arc: weight})
    critical = set(FIGURE_ONE_CRITICAL)

    result = solve_exact(instance, critical)

    assert result.aligned and result.proved_optimal
    assert result.cost == 1, "one deletion suffices once the auxiliary arc exists"

    # Two single deletions are optimal here - cutting the shortcut, or stranding the
    # node it leads to - so the solver is free to pick either. What it may not do is
    # return a graph the agent walks past the critical step.
    aligned = instance.without(set(result.deleted))
    assert is_aligned(aligned, critical)
    assert critical <= set(analyze(aligned).feasible_paths()[0])
    assert is_aligned(instance.without({BF}), critical), "the paper's own fix must also work"


def test_paid_additions_are_priced_by_keeping_them():
    """A negative deletion cost turns "keep this arc" into something to pay for.

    Charged 0.5 to retain the auxiliary arc, the solver still retains it, because
    no cheaper aligned graph exists without it - deleting both leaves Bob on sadt,
    which skips the critical step.
    """
    arc, weight = FIGURE_ONE_AUXILIARY
    instance = figure_one().with_arcs({arc: weight})

    def cost(candidate: Arc) -> float:
        return -0.5 if candidate == AB else 1.0

    result = solve_exact(instance, set(FIGURE_ONE_CRITICAL), cost)

    assert result.aligned and result.proved_optimal
    assert AB not in result.deleted, "the auxiliary arc is worth its price here"
    assert result.cost == 1, "paying to keep the arc is still cheaper than any alternative"


def test_without_the_auxiliary_arc_the_shortcut_deletion_is_not_enough():
    """The paper's point: blocking the tempting route only diverts the agent."""
    instance = figure_one().without({BF})

    assert not is_aligned(instance, set(FIGURE_ONE_CRITICAL))


# --- exact search agrees with brute force -----------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_exact_matches_brute_force(seed):
    rng = random.Random(seed)
    generated = layered_dag(2, 2, rng)
    instance = generated.instance
    behavior = analyze(instance)
    critical = {next(iter(behavior.feasible))} if behavior.feasible else set()

    expected, solutions = brute_force(instance, critical)
    result = solve_exact(instance, critical)

    if solutions == 0:
        assert not result.aligned
    else:
        assert result.aligned and result.proved_optimal
        assert result.cost == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(6))
def test_exact_matches_brute_force_with_mixed_sign_costs(seed):
    """Pruning has to stay sound when a branch could still save money later."""
    rng = random.Random(100 + seed)
    generated = layered_dag(2, 2, rng)
    instance = generated.instance
    arcs = sorted(instance.arcs())
    prices = {arc: rng.choice([-0.5, 1.0, 2.0, 3.0]) for arc in arcs}
    cost = prices.__getitem__

    behavior = analyze(instance)
    critical = {next(iter(behavior.feasible))} if behavior.feasible else set()

    expected, solutions = brute_force(instance, critical, cost)
    result = solve_exact(instance, critical, cost)

    if solutions:
        assert result.aligned and result.proved_optimal
        assert result.cost == pytest.approx(expected)


def test_solver_reports_zero_when_nothing_is_wrong():
    instance = costly_compliance().instance
    already = instance.without({("s", "tempting", 0)})

    result = solve_exact(already, {("s", "steady", 0)})

    assert result.aligned and result.cost == 0 and not result.deleted


# --- the heuristic ----------------------------------------------------------


def test_greedy_returns_a_genuinely_aligned_graph():
    """Whatever it costs, a heuristic that claims alignment must deliver it."""
    rng = random.Random(3)
    for _ in range(20):
        generated = layered_dag(3, 3, rng)
        instance = generated.instance
        behavior = analyze(instance)
        if not behavior.feasible:
            continue
        critical = {next(iter(behavior.feasible))}
        result = greedy_witness_guided(instance, critical)
        if result.aligned:
            assert is_aligned(instance.without(set(result.deleted)), critical)


def test_greedy_pays_double_on_a_known_instance():
    """The gap E3 exists to measure, pinned to one reproducible case.

    Blocking the step the agent just took diverts it onto another bad route, and
    the heuristic pays twice for what one better-chosen deletion would have fixed.
    """
    instance = layered_dag(3, 3, random.Random(0)).instance
    critical = {("L1_2", "L2_0", 0)}

    exact = solve_exact(instance, critical)
    greedy = greedy_witness_guided(instance, critical)

    assert exact.aligned and exact.proved_optimal and exact.cost == 1
    assert greedy.aligned and greedy.cost == 2


def test_greedy_never_beats_the_optimum_and_sometimes_loses():
    """Across a sweep: feasible always, optimal not always."""
    gaps = []
    for seed in range(12):
        rng = random.Random(seed)
        instance = layered_dag(3, 3, rng).instance
        behavior = analyze(instance)
        for critical_arc in list(behavior.feasible)[:3]:
            critical = {critical_arc}
            exact = solve_exact(instance, critical)
            greedy = greedy_witness_guided(instance, critical)
            if exact.aligned and greedy.aligned:
                gaps.append(greedy.cost - exact.cost)

    assert gaps, "no solvable instances were generated"
    assert all(gap >= -1e-9 for gap in gaps), "the heuristic beat the exact optimum"
    assert any(gap > 0 for gap in gaps), "a heuristic that never loses is not a baseline"


def test_minimalize_never_breaks_or_worsens_a_solution():
    rng = random.Random(7)
    for _ in range(20):
        instance = layered_dag(3, 3, rng).instance
        behavior = analyze(instance)
        if not behavior.feasible:
            continue
        critical = {next(iter(behavior.feasible))}
        greedy = greedy_witness_guided(instance, critical)
        if not greedy.aligned:
            continue

        trimmed = minimalize(instance, critical, greedy.deleted)

        assert trimmed.aligned
        assert trimmed.cost <= greedy.cost + 1e-9
        assert set(trimmed.deleted) <= set(greedy.deleted)


# --- feasibility ------------------------------------------------------------


def test_forced_route_proves_a_costly_requirement_can_be_imposed():
    generated = costly_compliance()

    assert is_alignable(generated.instance, set(generated.spec.critical))


def test_critical_arcs_are_never_offered_for_deletion():
    generated = costly_compliance()
    critical = set(generated.spec.critical)

    assert critical.isdisjoint(candidate_arcs(generated.instance, critical))


def test_unreachable_arcs_are_not_worth_paying_to_remove():
    """An arc the agent could never take should not appear as a candidate."""
    instance = figure_one()
    stranded = instance.with_arcs({("d", "c", 0): 1.0})

    candidates = candidate_arcs(stranded, set(FIGURE_ONE_CRITICAL))

    # c can still reach t, so (d, c) is usable; but nothing reaches a dead end.
    marooned = stranded.with_arcs({("e", "dead", 0): 1.0})
    assert ("e", "dead", 0) not in candidate_arcs(marooned, set(FIGURE_ONE_CRITICAL))
    assert candidates
