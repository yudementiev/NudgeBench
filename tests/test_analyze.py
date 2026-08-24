"""Validation of the aggregation that turns episodes into reported numbers."""

from __future__ import annotations

import random

import pytest

from mcta.generate import integer_sweep
from mcta.probe import ReferenceAgent, make_presentation, run_episode
from mcta.probe.analyze import Cell, Interval, intersect, wilson
from mcta.probe.fit import BetaFit


def fit(lower: float, upper: float) -> BetaFit:
    return BetaFit(lower=lower, upper=upper)


def test_intersecting_narrows_towards_the_truth():
    result = intersect([("a", fit(0.0, 0.8)), ("b", fit(0.6, 1.0)), ("c", fit(0.0, 0.75))])

    assert result.consistent
    assert result.lower == pytest.approx(0.6)
    assert result.upper == pytest.approx(0.75)
    assert result.contributing == 3


def test_an_impossible_combination_names_the_offender():
    """An empty intersection is a finding about the agent, not a failed measurement."""
    result = intersect([("early", fit(0.0, 0.4)), ("late", fit(0.9, 1.0))])

    assert not result.consistent
    assert any("late" in conflict for conflict in result.conflicts)
    assert "does not describe this agent" in result.describe()


def test_an_episode_no_beta_explains_is_carried_as_a_conflict():
    broken = BetaFit(lower=1.0, upper=0.0)
    result = intersect([("ok", fit(0.0, 1.0)), ("broken", broken)])

    assert not result.consistent
    assert result.contributing == 1


def test_empty_input_is_reported_rather_than_guessed():
    assert intersect([]).describe() == "no usable episodes"


@pytest.mark.parametrize("beta", [0.55, 0.7375, 0.99])
def test_a_sweep_brackets_a_known_agent(beta):
    """End to end: run the real sweep, recover the beta it was driven with."""
    cell = Cell(label=f"reference {beta}")
    for index, trap in enumerate(integer_sweep("effort", count=5)):
        presentation = make_presentation(trap.instance, random.Random(index))
        cell.add(run_episode(trap.instance, presentation, ReferenceAgent(beta)))

    interval = cell.beta()

    assert interval.consistent, interval.describe()
    assert interval.lower - 1e-9 <= beta <= interval.upper + 1e-9
    assert cell.arithmetic_rate == 1.0, "the reference agent totals its own plan correctly"
    assert cell.parse_failure_rate == 0.0


def test_wilson_stays_inside_the_unit_interval_at_the_extremes():
    """E4 expects rates at 100%, where a normal interval would run past 1."""
    low, high = wilson(20, 20)

    assert 0.0 <= low <= 1.0 and high == 1.0
    assert low > 0.8, "twenty out of twenty should not admit a rate below 80%"


def test_wilson_handles_zero_successes():
    low, high = wilson(0, 20)

    assert low == 0.0 and 0.0 < high < 0.2


def test_wilson_refuses_to_invent_an_interval_from_nothing():
    low, high = wilson(0, 0)

    assert low != low and high != high  # NaN
