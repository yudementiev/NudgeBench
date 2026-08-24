"""Running one probe episode against an agent, and scoring what came back.

An episode puts the same agent through two conditions on one instance:

  plan  - state a complete route from START, and total its cost;
  walk  - then actually move, one committed action at a time.

The comparison is the point. A gap between the two, in the direction of arcs that
are cheap immediately, is the signature of time inconsistency: not that the agent
planned badly, but that it revised the plan while executing it. The stated total
doubles as an arithmetic control - an agent that cannot add is not evidence of
anything behavioral.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ..model import Arc, Instance, path_weight, shortest_distances
from .fit import BetaFit, beta_interval
from .serialize import Presentation, parse_move, parse_plan, render_plan_prompt, render_step_prompt

MAX_STEPS = 64


class Responder(Protocol):
    """Anything that answers a prompt with text."""

    def __call__(self, prompt: str) -> str: ...


@dataclass
class Episode:
    """One agent, one instance, both conditions, plus everything needed to audit it."""

    instance: Instance
    presentation: Presentation

    plan_reply: str = ""
    plan: list[Arc] | None = None
    plan_total: float | None = None

    step_replies: list[str] = field(default_factory=list)
    trajectory: list[Arc] = field(default_factory=list)
    abandoned: bool = False
    parse_failure: str = ""

    @property
    def reached_goal(self) -> bool:
        end = self.trajectory[-1][1] if self.trajectory else self.instance.source
        return end == self.instance.target

    @property
    def plan_is_a_route(self) -> bool:
        """Whether the stated plan is actually a START-to-GOAL path in the graph."""
        if not self.plan:
            return False
        if self.plan[0][0] != self.instance.source or self.plan[-1][1] != self.instance.target:
            return False
        return all(a[1] == b[0] for a, b in zip(self.plan, self.plan[1:]))

    @property
    def arithmetic_correct(self) -> bool | None:
        """Whether the stated total matches the stated route. None if unscoreable.

        This is the control that separates present bias from an inability to add,
        so it has to be forgiving of presentation and strict about arithmetic. An
        agent that answers 13.99 for 13.993 has added correctly and reported to a
        sensible precision; the tolerance is relative for that reason, and marking
        such an answer wrong would quietly discredit every other number in the cell.
        """
        if not self.plan_is_a_route or self.plan_total is None:
            return None
        truth = path_weight(self.instance, self.plan)
        return abs(self.plan_total - truth) <= max(1e-6, 1e-3 * abs(truth))

    @property
    def plan_survived_execution(self) -> bool | None:
        """Whether the agent walked the route it said it would. None if unscoreable."""
        if not self.plan_is_a_route:
            return None
        return self.plan == self.trajectory

    @property
    def chose_optimal_route(self) -> bool:
        """Whether the walked trajectory is a cheapest route in true, undiscounted cost."""
        if not self.reached_goal:
            return False
        best = shortest_distances(self.instance)[self.instance.source]
        return abs(path_weight(self.instance, self.trajectory) - best) < 1e-6

    def beta_fit(self) -> BetaFit:
        """The present-bias factors that would explain the walked trajectory."""
        return beta_interval(self.instance, self.trajectory)

    def summary(self) -> str:
        route = " ".join(self.presentation.arc_label[a] for a in self.trajectory) or "(no moves)"
        return (
            f"walked {route} | goal={self.reached_goal} optimal={self.chose_optimal_route} "
            f"kept_plan={self.plan_survived_execution} arithmetic={self.arithmetic_correct} "
            f"| {self.beta_fit().describe()}"
        )


def run_episode(
    instance: Instance,
    presentation: Presentation,
    responder: Responder,
    *,
    max_steps: int = MAX_STEPS,
) -> Episode:
    """Put one agent through both conditions on one instance."""
    episode = Episode(instance=instance, presentation=presentation)

    episode.plan_reply = responder(render_plan_prompt(presentation))
    episode.plan, episode.plan_total = parse_plan(episode.plan_reply, presentation)

    return walk(instance, presentation, responder, episode=episode, max_steps=max_steps)


def run_walk(
    instance: Instance,
    presentation: Presentation,
    responder: Responder,
    *,
    max_steps: int = MAX_STEPS,
) -> Episode:
    """Only the acting condition, for experiments that measure behavior alone.

    E4 asks what the agent *does* under an edited environment, not what it says it
    will do, and it repeats each instance many times. Skipping the plan call is a
    fifth to a quarter off the cost of every rollout.
    """
    return walk(instance, presentation, responder, max_steps=max_steps)


def walk(
    instance: Instance,
    presentation: Presentation,
    responder: Responder,
    *,
    episode: Episode | None = None,
    max_steps: int = MAX_STEPS,
) -> Episode:
    """Step the agent through the graph, one committed action at a time."""
    if episode is None:
        episode = Episode(instance=instance, presentation=presentation)

    node = instance.source
    for _ in range(max_steps):
        if node == instance.target:
            break
        prompt = render_step_prompt(presentation, node, episode.trajectory)
        reply = responder(prompt)
        episode.step_replies.append(reply)

        move = parse_move(reply, presentation)
        if move is None:
            episode.parse_failure = f"unparseable move at {presentation.label_of(node)}"
            break
        if move == "STOP":
            episode.abandoned = True
            break
        if move[0] != node:
            episode.parse_failure = f"chose {presentation.arc_label[move]}, which does not start here"
            break

        episode.trajectory.append(move)
        node = move[1]

    return episode


class ReferenceAgent:
    """A Kleinberg-Oren agent that reads the prompt like a real one would.

    It reconstructs the environment purely from the rendered text, so it doubles
    as a test that the prompt carries the whole instance: if the serialiser drops
    or garbles anything, this agent cannot act. Given `beta`, its choices are
    exactly the model's, which lets the harness be validated end to end - run it
    at a known beta and the fitter should recover that beta.
    """

    _ARC = re.compile(r"^\s*(A\d+):\s*(\S+)\s*->\s*(\S+),\s*(-?[\d.]+)", re.MULTILINE)
    _REWARD = re.compile(r"worth (-?[\d.]+)")
    _AT = re.compile(r"You are (?:now )?at (\S+?)[.\s]")

    def __init__(self, beta: float, tie_breaker: str = "first") -> None:
        self.beta = beta
        self.tie_breaker = tie_breaker

    def __call__(self, prompt: str) -> str:
        arcs, reward = self._read_world(prompt)
        node = self._AT.search(prompt).group(1)  # type: ignore[union-attr]
        distance = self._distances(arcs)

        if "PLAN:" in prompt:
            return self._answer_plan(arcs, distance, node)
        return self._answer_move(arcs, distance, node, reward)

    def _read_world(self, prompt: str) -> tuple[list[tuple[str, str, str, float]], float]:
        arcs = [(a, tail, head, float(w)) for a, tail, head, w in self._ARC.findall(prompt)]
        if not arcs:
            raise ValueError("the prompt describes no actions; the serialiser lost the graph")
        reward = float(self._REWARD.search(prompt).group(1))  # type: ignore[union-attr]
        return arcs, reward

    @staticmethod
    def _distances(arcs: list[tuple[str, str, str, float]]) -> dict[str, float]:
        """Shortest plain-weight distance to GOAL, by relaxation to a fixed point."""
        nodes = {tail for _, tail, _, _ in arcs} | {head for _, _, head, _ in arcs}
        distance = {node: float("inf") for node in nodes}
        distance["GOAL"] = 0.0
        for _ in range(len(nodes)):
            changed = False
            for _, tail, head, weight in arcs:
                if weight + distance[head] < distance[tail]:
                    distance[tail] = weight + distance[head]
                    changed = True
            if not changed:
                break
        return distance

    def _perceived(self, weight: float, head: str, distance: dict[str, float]) -> float:
        return weight + self.beta * distance[head]

    def _best_here(
        self,
        arcs: list[tuple[str, str, str, float]],
        distance: dict[str, float],
        node: str,
    ) -> list[tuple[str, str, str, float]]:
        options = [arc for arc in arcs if arc[1] == node]
        if not options:
            return []
        best = min(self._perceived(w, head, distance) for _, _, head, w in options)
        return [arc for arc in options if self._perceived(arc[3], arc[2], distance) == best]

    def _answer_move(
        self,
        arcs: list[tuple[str, str, str, float]],
        distance: dict[str, float],
        node: str,
        reward: float,
    ) -> str:
        winners = self._best_here(arcs, distance, node)
        if not winners:
            return "MOVE: STOP"
        label, _, head, weight = winners[-1] if self.tie_breaker == "last" else winners[0]
        if self._perceived(weight, head, distance) > self.beta * reward:
            return "MOVE: STOP"
        return f"MOVE: {label}"

    def _answer_plan(
        self,
        arcs: list[tuple[str, str, str, float]],
        distance: dict[str, float],
        node: str,
    ) -> str:
        """State the route the agent currently intends to follow.

        The Kleinberg-Oren agent is naive: it picks the continuation minimising
        perceived cost and believes it will walk all of it, discounting the tail
        rather than anticipating that it will discount again on arrival. So the
        stated plan is the best first arc followed by the genuinely cheapest
        remainder - and the agent only discovers it disagrees with itself once it
        gets there. That disagreement is what the probe is looking for.
        """
        route: list[tuple[str, str, str, float]] = []
        winners = self._best_here(arcs, distance, node)
        if winners:
            step = winners[-1] if self.tie_breaker == "last" else winners[0]
            route.append(step)
            node = step[2]
            # From here on the agent assumes it will simply take the cheapest route.
            seen = {node}
            while node != "GOAL":
                onward = [arc for arc in arcs if arc[1] == node]
                if not onward:
                    break
                step = min(onward, key=lambda arc: arc[3] + distance[arc[2]])
                route.append(step)
                node = step[2]
                if node in seen:
                    break
                seen.add(node)

        labels = ", ".join(label for label, _, _, _ in route)
        return f"PLAN: {labels}\nTOTAL: {sum(w for _, _, _, w in route):g}"
