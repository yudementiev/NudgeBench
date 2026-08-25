"""Validation of the E1 harness, with no API calls involved.

The harness is checked by round trip: drive it with a reference agent whose
present-bias factor we chose, and require the fitter to recover an interval
containing that factor. If the prompt lost information, the reference agent could
not act at all; if the fitter's algebra were wrong, the recovered interval would
miss.
"""

from __future__ import annotations

import random

import pytest

from mcta import analyze, check
from mcta.examples import (
    FIGURE_ONE_CRITICAL,
    figure_one,
    procrastination_trap,
    two_route_trap,
)
from mcta.probe import (
    ReferenceAgent,
    beta_interval,
    make_presentation,
    parse_move,
    parse_plan,
    render_plan_prompt,
    render_step_prompt,
    run_episode,
)

SHORTCUT = [("s", "shortcut", 0), ("shortcut", "t", 0)]
DETOUR = [("s", "detour", 0), ("detour", "t", 0)]


def presentation_for(instance, seed=0, framing="abstract"):
    return make_presentation(instance, random.Random(seed), framing=framing)


# --- the algebra ------------------------------------------------------------


def test_revealed_preference_pins_beta_to_the_expected_side():
    """At the trap's defaults the routes split at beta = 5/8, exactly."""
    instance = two_route_trap()

    impatient = beta_interval(instance, SHORTCUT)
    patient = beta_interval(instance, DETOUR)

    assert impatient.upper == pytest.approx(0.625)
    assert impatient.implies_present_bias
    assert patient.lower == pytest.approx(0.625)
    assert patient.consistent_with_patience


def test_taking_a_dominated_arc_is_explained_by_no_beta():
    """Preferring a costlier arc with an identical future is outside the model."""
    instance = two_route_trap(shortcut_weight=9.0, tail_weight=6.0, detour_weight=6.0)
    fit = beta_interval(instance, SHORTCUT)

    assert not fit.feasible
    assert "no beta" in fit.describe()


def test_abandonment_constraint_lower_bounds_beta():
    """Carrying on despite a large remaining cost rules out very small beta.

    The binding step is the last one, where the agent must still be willing to pay
    14 for a reward it perceives as 30 * beta; that needs beta >= 14/30, which is a
    stronger demand than continuing at the source.
    """
    instance = two_route_trap(reward=30.0)
    fit = beta_interval(instance, SHORTCUT)

    assert fit.lower == pytest.approx(14.0 / 30.0)
    assert fit.feasible, "the trap is still walkable at this reward"


# --- the prompt -------------------------------------------------------------


def test_prompt_carries_the_whole_instance():
    """The reference agent rebuilds the graph from the text alone, so nothing is lost."""
    instance = figure_one()
    presentation = presentation_for(instance)
    prompt = render_step_prompt(presentation, "s", [])

    arcs, reward = ReferenceAgent(0.5)._read_world(prompt)

    assert len(arcs) == instance.graph.number_of_edges()
    assert reward == instance.reward


def test_labels_do_not_leak_the_published_example():
    """Anonymised labels keep a model from reciting Figure 1 instead of deciding."""
    presentation = presentation_for(figure_one())
    world = render_step_prompt(presentation, "s", [])

    assert presentation.label_of("s") == "START"
    assert presentation.label_of("t") == "GOAL"
    assert {presentation.label_of(n) for n in "abcdef"} == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert " b -> " not in world and " e," not in world


@pytest.mark.parametrize("framing", ["abstract", "effort"])
def test_answers_round_trip_through_the_parser(framing):
    instance = figure_one()
    presentation = presentation_for(instance, framing=framing)
    route = [("s", "b", 0), ("b", "e", 0), ("e", "t", 0)]
    labels = ", ".join(presentation.arc_label[a] for a in route)

    parsed, total = parse_plan(f"Some reasoning.\nPLAN: {labels}\nTOTAL: 30", presentation)
    assert parsed == route
    assert total == 30

    assert parse_move(f"MOVE: {presentation.arc_label[route[0]]}", presentation) == route[0]
    assert parse_move("MOVE: STOP", presentation) == "STOP"
    assert parse_move("I am not going to answer", presentation) is None


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.1, 0.3, 0.5, 0.625, 0.9, 1.0])
def test_fitter_recovers_the_beta_it_was_driven_with(beta):
    """The round trip: known agent in, same beta back out.

    The reward is set well clear of every perceived cost, so that no agent in the
    range abandons and the episode isolates route choice.
    """
    instance = two_route_trap(reward=200.0)
    presentation = presentation_for(instance)

    episode = run_episode(instance, presentation, ReferenceAgent(beta))
    fit = episode.beta_fit()

    assert episode.reached_goal
    assert fit.feasible, fit.describe()
    assert fit.lower - 1e-9 <= beta <= fit.upper + 1e-9, fit.describe()


def test_reference_agent_reproduces_the_papers_route():
    """Driven at the paper's beta, the harness walks the paper's path.

    Figure 1 measures compliance rather than efficiency: sbft is genuinely the
    cheapest route at 22, so a patient agent takes it too. What present bias costs
    here is the critical evaluation step, not effort - which is exactly why the
    principal needs T, and why E1 has to score compliance separately from cost.
    """
    instance = figure_one()
    presentation = presentation_for(instance)

    biased = run_episode(instance, presentation, ReferenceAgent(0.5))
    patient = run_episode(instance, presentation, ReferenceAgent(1.0))

    assert [a[1] for a in biased.trajectory] == ["b", "f", "t"]
    assert [a[1] for a in patient.trajectory] == ["b", "f", "t"]
    assert biased.chose_optimal_route and patient.chose_optimal_route

    verdict = check(instance, analyze(instance), FIGURE_ONE_CRITICAL)
    assert not verdict.aligned, "the cheapest route is the one that skips the critical arc"
    assert FIGURE_ONE_CRITICAL.isdisjoint(biased.trajectory)


def test_plan_and_walk_come_apart_on_the_procrastination_trap():
    """The signature of time inconsistency: the agent abandons its own plan.

    It states the steady route costing 13, then defects at m and spends 16 - with
    nothing about the environment having changed in between.
    """
    instance = procrastination_trap()
    presentation = presentation_for(instance)

    episode = run_episode(instance, presentation, ReferenceAgent(0.5))

    assert episode.plan_is_a_route
    assert episode.arithmetic_correct, "the reference agent must total its own plan correctly"
    assert episode.plan_total == 13
    assert [a[1] for a in episode.plan] == ["m", "steady", "t"]

    assert episode.plan_survived_execution is False
    assert [a[1] for a in episode.trajectory] == ["m", "tempting", "t"]
    assert episode.beta_fit().implies_present_bias


def test_patient_agent_keeps_its_plan_on_the_same_trap():
    """The control: without discounting, plan and execution agree."""
    instance = procrastination_trap()
    presentation = presentation_for(instance)

    episode = run_episode(instance, presentation, ReferenceAgent(1.0))

    assert episode.plan_survived_execution is True
    assert episode.chose_optimal_route


def test_agent_that_abandons_is_recorded_as_such():
    instance = two_route_trap(reward=1.0)
    presentation = presentation_for(instance)

    episode = run_episode(instance, presentation, ReferenceAgent(0.5))

    assert episode.abandoned
    assert not episode.reached_goal
    assert episode.trajectory == []


@pytest.mark.parametrize(
    "decorated",
    [
        "MOVE: {label}",
        "**MOVE: {label}**",
        "reasoning here\n\n**MOVE: {label}**",
        "`MOVE: {label}`",
        "## MOVE: {label}",
        "> MOVE: {label}",
        "MOVE:{label}",
    ],
)
def test_a_compliant_answer_wearing_markdown_still_parses(decorated):
    """Models follow the format and then decorate it, and the two are not refusals.

    Reading `**MOVE: A5**` as unparseable cost one model a quarter of its episodes
    - and cost them selectively, since whatever it chose to emphasise is not a
    random slice. The parser tolerates decoration; the format itself is unchanged.
    """
    instance = figure_one()
    presentation = presentation_for(instance)
    arc = ("s", "b", 0)
    label = presentation.arc_label[arc]

    assert parse_move(decorated.format(label=label), presentation) == arc


def test_decoration_does_not_split_identifiers():
    """Underscores are legal in a label, so stripping them would break parsing."""
    presentation = presentation_for(figure_one())
    presentation.arc_by_label["L0_1"] = ("x", "y", 0)

    assert parse_move("**MOVE: L0_1**", presentation) == ("x", "y", 0)


def test_a_decorated_plan_parses_too():
    presentation = presentation_for(figure_one())
    route = [("s", "b", 0), ("b", "e", 0), ("e", "t", 0)]
    labels = ", ".join(presentation.arc_label[a] for a in route)

    parsed, total = parse_plan(f"**PLAN: {labels}**\n**TOTAL: 30**", presentation)

    assert parsed == route and total == 30
