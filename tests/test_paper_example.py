"""Validation against the worked example of Belova et al., Figure 1.

The paper states the agent's perceived costs and chosen routes explicitly at each
stage of Alice's intervention, which makes it exact ground truth for the
simulator. Every number asserted below is quoted in the paper's text.
"""

from __future__ import annotations

import pytest

from mcta import analyze, check, path_perceived_cost, t_avoiding_arcs
from mcta.examples import (
    FIGURE_ONE_AUXILIARY,
    FIGURE_ONE_CRITICAL,
    figure_one,
    two_route_trap,
)

SA, AD, DT = ("s", "a", 0), ("a", "d", 0), ("d", "t", 0)
SB, BE, ET = ("s", "b", 0), ("b", "e", 0), ("e", "t", 0)
SC, CF, FT = ("s", "c", 0), ("c", "f", 0), ("f", "t", 0)
BF = ("b", "f", 0)
AB = FIGURE_ONE_AUXILIARY[0]


def test_perceived_costs_of_the_four_routes():
    """The paper: zeta(sbet) = zeta(scft) = 20, zeta(sadt) = 18, zeta(sbft) = 16."""
    instance = figure_one()
    assert path_perceived_cost(instance, [SB, BE, ET]) == 20
    assert path_perceived_cost(instance, [SC, CF, FT]) == 20
    assert path_perceived_cost(instance, [SA, AD, DT]) == 18
    assert path_perceived_cost(instance, [SB, BF, FT]) == 16


def test_biased_agent_takes_the_shortcut():
    """"Bob chooses the path P4 ... and will eventually follow sbft.\"""" ""
    behavior = analyze(figure_one())
    assert behavior.feasible_paths() == [[SB, BF, FT]]
    assert behavior.zeta["s"] == 16


def test_deleting_the_shortcut_diverts_rather_than_fixes():
    """Alice removes (b, f); the perceived cost at s becomes exactly beta * r = 18.

    The paper's point is that this is not enough: Bob switches to sadt, which
    still skips the critical evaluation step.
    """
    instance = figure_one().without({BF})
    behavior = analyze(instance)

    assert behavior.zeta["s"] == 18 == instance.perceived_reward
    assert not behavior.abandons_at("s")
    assert behavior.feasible_paths() == [[SA, AD, DT]]


def test_adding_the_auxiliary_arc_aligns_the_agent():
    """With (b, f) gone and (a, b) added at weight 1, Bob follows sabet.

    The paper computes the decisive perceived cost at a: 1 + (10 + 10) / 2 = 11.
    """
    arc, weight = FIGURE_ONE_AUXILIARY
    instance = figure_one().without({BF}).with_arcs({arc: weight})
    behavior = analyze(instance)

    assert behavior.zeta["a"] == 11
    assert behavior.feasible_paths() == [[SA, AB, BE, ET]]


@pytest.mark.parametrize(
    "deleted, added, expected_aligned, expected_witness",
    [
        (set(), {}, False, [SB, BF]),
        ({BF}, {}, False, [SA]),
        ({BF}, {AB: 1.0}, True, None),
    ],
    ids=["initial", "after-deletion", "after-deletion-and-addition"],
)
def test_alignment_verdict_at_each_stage(deleted, added, expected_aligned, expected_witness):
    """Only the combined intervention aligns the agent, as the paper's story goes."""
    instance = figure_one().without(deleted).with_arcs(added)
    verdict = check(instance, analyze(instance), FIGURE_ONE_CRITICAL)

    assert verdict.aligned is expected_aligned
    assert verdict.witness == expected_witness


def test_t_avoiding_arcs_identify_the_foreclosing_moves():
    """Taking (b, f) forecloses (b, e); staying on (s, b) does not."""
    instance = figure_one()
    avoiding = t_avoiding_arcs(instance, FIGURE_ONE_CRITICAL)

    assert BF in avoiding, "the shortcut past e must count as T-avoiding"
    assert SA in avoiding, "routing through a leaves b unreachable"
    assert SC in avoiding
    assert SB not in avoiding, "b can still be reached, so this arc forecloses nothing"
    assert BE not in avoiding, "a critical arc never avoids itself"


def test_present_bias_is_what_changes_the_route():
    """The same graph, two agents: only the biased one takes the tempting shortcut."""
    biased = two_route_trap(beta=0.5)
    patient = two_route_trap(beta=1.0)

    assert analyze(biased).feasible_paths() == [[("s", "shortcut", 0), ("shortcut", "t", 0)]]
    assert analyze(patient).feasible_paths() == [[("s", "detour", 0), ("detour", "t", 0)]]


def test_agent_abandons_when_the_reward_is_too_small():
    """A reward the agent cannot perceive as worth it stops it at the source."""
    instance = two_route_trap(reward=1.0)
    behavior = analyze(instance)

    assert behavior.abandons_at("s")
    assert behavior.feasible_paths() == [[]], "abandoning at s means an empty trajectory"
