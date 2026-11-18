"""Compute the Kleinberg-Oren discount factors consistent with observed choices.

Each chosen action contributes a linear inequality in beta:

    w(e*) + beta * d(head(e*)) <= w(e) + beta * d(head(e)).

Intersecting these constraints gives a model-consistent interval without a beta
parameter grid. Arithmetic uses floats with TOLERANCE for comparisons. An empty
interval rejects a common beta under these constraints. A nonempty interval does
not establish a unique internal discount factor or identify the behavioural
mechanism: other decision rules may generate the same observed choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..model import Arc, Instance, shortest_distances

# Slack for floating-point comparisons when intersecting half-lines.
TOLERANCE = 1e-9


@dataclass
class Constraint:
    """One revealed preference, recorded for diagnostics."""

    node: str
    chosen: Arc
    rejected: Arc | None
    lower: float
    upper: float
    note: str = ""


@dataclass
class BetaFit:
    """The set of present-bias factors consistent with an observed trajectory."""

    lower: float
    upper: float
    constraints: list[Constraint] = field(default_factory=list)
    infeasible_at: Constraint | None = None

    @property
    def feasible(self) -> bool:
        """Whether any beta explains the trajectory."""
        return self.infeasible_at is None and self.lower <= self.upper + TOLERANCE

    @property
    def consistent_with_patience(self) -> bool:
        """Whether an unbiased agent (beta = 1) could have produced this trajectory."""
        return self.feasible and self.upper >= 1.0 - TOLERANCE

    @property
    def implies_present_bias(self) -> bool:
        """Whether every model-consistent explanation requires beta < 1.

        This is conditional on the KO model, not a diagnosis excluding local-cost
        heuristics or planning errors.
        """
        return self.feasible and self.upper < 1.0 - TOLERANCE

    def describe(self) -> str:
        if not self.feasible:
            where = self.infeasible_at
            detail = f" at {where.node}: {where.note}" if where else ""
            return f"no beta explains this trajectory under the model{detail}"
        if self.consistent_with_patience:
            return f"beta in [{self.lower:.4f}, 1] - consistent with a patient agent under the model"
        return f"beta in [{self.lower:.4f}, {self.upper:.4f}] - requires beta < 1 under the model"


def beta_interval(
    instance: Instance,
    trajectory: list[Arc],
    *,
    require_completion: bool = True,
) -> BetaFit:
    """Solve for every beta under which a KO agent could follow `trajectory`.

    Each step contributes one inequality per rejected alternative. When
    `require_completion` is set, each step also contributes the constraint that
    the agent did not abandon there, which lower-bounds beta through the reward.
    """
    distance = shortest_distances(instance)
    lower, upper = 0.0, 1.0
    constraints: list[Constraint] = []

    for arc in trajectory:
        node = arc[0]
        chosen_weight = instance.weight(arc)
        chosen_distance = distance[arc[1]]

        if math.isinf(chosen_distance):
            failure = Constraint(
                node=node,
                chosen=arc,
                rejected=None,
                lower=lower,
                upper=upper,
                note="the chosen arc cannot reach the target",
            )
            return BetaFit(lower, upper, constraints, infeasible_at=failure)

        for alternative in instance.out_arcs(node):
            if alternative == arc or math.isinf(distance[alternative[1]]):
                continue
            constraint = _revealed_preference(
                node=node,
                chosen=arc,
                chosen_weight=chosen_weight,
                chosen_distance=chosen_distance,
                rejected=alternative,
                rejected_weight=instance.weight(alternative),
                rejected_distance=distance[alternative[1]],
            )
            constraints.append(constraint)
            lower, upper = max(lower, constraint.lower), min(upper, constraint.upper)
            if lower > upper + TOLERANCE:
                return BetaFit(lower, upper, constraints, infeasible_at=constraint)

        if require_completion:
            constraint = _did_not_abandon(
                node=node,
                chosen=arc,
                chosen_weight=chosen_weight,
                chosen_distance=chosen_distance,
                reward=instance.reward,
            )
            constraints.append(constraint)
            lower, upper = max(lower, constraint.lower), min(upper, constraint.upper)
            if lower > upper + TOLERANCE:
                return BetaFit(lower, upper, constraints, infeasible_at=constraint)

    return BetaFit(lower, upper, constraints)


def _revealed_preference(
    *,
    node: str,
    chosen: Arc,
    chosen_weight: float,
    chosen_distance: float,
    rejected: Arc,
    rejected_weight: float,
    rejected_distance: float,
) -> Constraint:
    """Turn "e* was preferred to e" into a bound on beta.

    Rearranging w* + beta*d* <= w + beta*d gives beta * (d* - d) <= w - w*, whose
    direction depends on the sign of the distance gap.
    """
    gap = chosen_distance - rejected_distance
    slack = rejected_weight - chosen_weight

    if abs(gap) < TOLERANCE:
        # Equal futures: the choice turns on the immediate weights alone, and beta
        # cannot explain preferring the more expensive one.
        if slack < -TOLERANCE:
            return Constraint(
                node, chosen, rejected, 1.0, 0.0,
                note=f"chose the costlier arc among equal continuations ({chosen_weight} over {rejected_weight})",
            )
        return Constraint(node, chosen, rejected, 0.0, 1.0, note="equal continuations, no constraint")

    bound = slack / gap
    if gap > 0:
        # The chosen arc leads to a worse future, so only a discounting agent takes it.
        return Constraint(node, chosen, rejected, 0.0, bound, note=f"beta <= {bound:.4f}")
    return Constraint(node, chosen, rejected, bound, 1.0, note=f"beta >= {bound:.4f}")


def _did_not_abandon(
    *,
    node: str,
    chosen: Arc,
    chosen_weight: float,
    chosen_distance: float,
    reward: float,
) -> Constraint:
    """The agent carried on, so its perceived cost stayed within its perceived reward.

    From w* + beta*d* <= beta*r we get beta * (r - d*) >= w*, which lower-bounds
    beta when the target is worth more than the remaining distance and is
    unsatisfiable when it is not.
    """
    headroom = reward - chosen_distance
    if headroom > TOLERANCE:
        return Constraint(
            node, chosen, None, chosen_weight / headroom, 1.0,
            note=f"continued rather than abandoned, so beta >= {chosen_weight / headroom:.4f}",
        )
    if chosen_weight <= TOLERANCE:
        return Constraint(node, chosen, None, 0.0, 1.0, note="free arc, abandonment says nothing")
    return Constraint(
        node, chosen, None, 1.0, 0.0,
        note=f"continued although the remaining cost {chosen_distance} exceeds the reward {reward}",
    )
