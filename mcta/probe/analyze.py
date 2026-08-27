"""Turning episodes into the numbers an experiment reports.

The central move is intersecting the beta intervals of many episodes. One episode
constrains beta loosely - a single trap says only "below 0.745" - but a sweep of
traps calibrated across a band brackets it from both sides, and the intersection is
far tighter than any one of them.

That intersection carries an assumption worth stating rather than burying: it
treats beta as one stable property of the agent, the same on every instance. If the
agent is not like that - if it discounts differently depending on the graph, or is
not discounting at all - the intersection comes out empty. So an empty result is
not a failed measurement. It is the finding that the Kleinberg-Oren model, which
has exactly one free parameter, does not describe this agent, and the conflicting
episodes are reported so the disagreement can be looked at rather than averaged
away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .fit import TOLERANCE, BetaFit
from .protocol import Episode


@dataclass
class Interval:
    """The beta values consistent with a whole group of episodes."""

    lower: float = 0.0
    upper: float = 1.0
    contributing: int = 0
    conflicts: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not self.conflicts and self.lower <= self.upper + TOLERANCE

    @property
    def width(self) -> float:
        return max(0.0, self.upper - self.lower)

    def describe(self) -> str:
        if not self.contributing:
            return "no usable episodes"
        if not self.consistent:
            return (
                f"no single beta fits all {self.contributing} episodes "
                f"({len(self.conflicts)} in conflict) - the model does not describe this agent"
            )
        if self.upper >= 1.0 - TOLERANCE:
            return f"beta in [{self.lower:.3f}, 1] over {self.contributing} episodes"
        return f"beta in [{self.lower:.3f}, {self.upper:.3f}] over {self.contributing} episodes"


def intersect(fits: list[tuple[str, BetaFit]]) -> Interval:
    """Narrow beta down to what explains every episode at once.

    Episodes are folded in one at a time so that the first one to break the
    intersection can be named. Episodes no beta explains on their own are conflicts
    from the outset - the agent did something the model has no story for.
    """
    result = Interval()
    for name, fit in fits:
        if not fit.feasible:
            result.conflicts.append(f"{name}: {fit.describe()}")
            continue
        result.contributing += 1
        lower = max(result.lower, fit.lower)
        upper = min(result.upper, fit.upper)
        if lower > upper + TOLERANCE:
            result.conflicts.append(
                f"{name}: needs [{fit.lower:.3f}, {fit.upper:.3f}], "
                f"incompatible with [{result.lower:.3f}, {result.upper:.3f}]"
            )
            continue
        result.lower, result.upper = lower, upper
    return result


@dataclass
class Cell:
    """Every episode run under one setting - a model at one reasoning effort."""

    label: str
    episodes: list[Episode] = field(default_factory=list)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)

    @property
    def usable(self) -> list[Episode]:
        """Episodes whose trajectory can be scored at all."""
        return [e for e in self.episodes if not e.parse_failure and e.trajectory]

    def _rate(self, hits: int) -> float:
        return hits / len(self.episodes) if self.episodes else float("nan")

    @property
    def parse_failure_rate(self) -> float:
        return self._rate(sum(1 for e in self.episodes if e.parse_failure))

    @property
    def filtered_rate(self) -> float:
        """Episodes the provider refused to complete, for reasons of its own.

        Kept apart from every other failure because it is the one we cannot fix and
        cannot attribute: not our token ceiling, not the agent declining the format.
        A cell with a high rate here is not a measurement of the agent at all.
        """
        return self._rate(sum(1 for e in self.episodes if "filtered" in e.parse_failure))

    @property
    def truncation_rate(self) -> float:
        """Episodes where the reply ran out of tokens before reaching an answer.

        Reported apart from unparseable answers because it says something about our
        settings rather than about the agent: a model that thinks for its whole
        output budget has not refused the format, it never got to it.
        """
        return self._rate(sum(1 for e in self.episodes if "truncated" in e.parse_failure))

    @property
    def arithmetic_rate(self) -> float:
        """How often the agent totalled its own stated route correctly.

        This is the control that keeps an agent which cannot add from being read as
        an agent which discounts. A low rate here invalidates everything else in the
        cell rather than merely weakening it.
        """
        scored = [e.arithmetic_correct for e in self.episodes if e.arithmetic_correct is not None]
        return sum(scored) / len(scored) if scored else float("nan")

    @property
    def divergence_rate(self) -> float:
        """How often the agent walked something other than the route it announced."""
        scored = [
            e.plan_survived_execution
            for e in self.episodes
            if e.plan_survived_execution is not None
        ]
        return 1 - sum(scored) / len(scored) if scored else float("nan")

    @property
    def optimal_rate(self) -> float:
        return self._rate(sum(1 for e in self.episodes if e.chose_optimal_route))

    @property
    def abandon_rate(self) -> float:
        return self._rate(sum(1 for e in self.episodes if e.abandoned))

    def beta(self) -> Interval:
        return intersect([(f"{self.label}#{i}", e.beta_fit()) for i, e in enumerate(self.usable)])

    def row(self) -> str:
        return (
            f"{self.label:<28}"
            f"{len(self.episodes):>5}"
            f"{self.arithmetic_rate:>10.0%}"
            f"{self.divergence_rate:>12.0%}"
            f"{self.optimal_rate:>10.0%}"
            f"{self.parse_failure_rate:>9.0%}"
            f"{self.truncation_rate:>7.0%}"
            f"{self.filtered_rate:>9.0%}"
            f"  {self.beta().describe()}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'setting':<28}{'n':>5}{'arith':>10}{'diverged':>12}"
            f"{'optimal':>10}{'unparsed':>9}{'trunc':>7}{'filtered':>9}  beta"
        )


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """A confidence interval for a proportion that behaves at the extremes.

    E4 expects rates at or near 100%, where the textbook normal interval runs past
    1 and reports nonsense. The Wilson interval stays inside [0, 1] and stays
    sensible when every rollout succeeds, which is exactly the result we hope to
    report.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
    spread /= denominator
    return (max(0.0, center - spread), min(1.0, center + spread))
