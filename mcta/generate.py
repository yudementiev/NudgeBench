"""Generating benchmark instances.

Two principles keep the corpus from being arbitrary.

*Calibrate to attested present bias.* An instance only discriminates if the beta
at which the agent's behavior flips lies in the range real agents occupy. The
quasi-hyperbolic meta-analysis of Cheung, Tymula and Wang (86 studies) puts beta
at 0.938, CI [0.905, 0.972], for monetary rewards and 0.750, CI [0.643, 0.857],
for non-monetary ones; effort-discounting experiments in the Augenblick line put
it at 0.70-0.79 for real effort. Weights here are effort, so instances are built
to switch inside the effort band rather than at some arbitrary threshold. An
instance that flips at beta = 0.2 tells us nothing about any agent we care about.

*Separate the two ways present bias hurts.* It can make an agent spend more than
it needed to, or make it skip a required action while spending no more at all.
Figure 1 of the paper is the second kind: the tempting route is also the cheapest
one, so nothing is wasted and only compliance fails. The generators below produce
each kind deliberately, because the two need different measurements.

Structural families imitate the shapes catalogued for scientific workflows -
layered fan-out/fan-in, independent parallel pipelines, series-parallel nesting -
which is where the corpus meets the treewidth question that C1 is about.

References
----------
Cheung, Tymula, Wang. A meta-analysis of quasi-hyperbolic discounting.
Augenblick et al. Estimating present bias and sophistication over effort and money.
Maniu, Senellart, Jog. An Experimental Study of the Treewidth of Real-World Graph
    Data, ICDT 2019 - which finds treewidth *high* in database and network data,
    and is the reason C1 needs testing on task graphs specifically.
Bharathi et al. Characterization of scientific workflows (Montage, CyberShake,
    Epigenomics, LIGO, SIPHT), the source of the topological families below.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import networkx as nx

from .model import Arc, Instance

# Empirical present-bias factors, for calibrating instances to agents that exist.
BETA_BANDS = {
    "money": (0.905, 0.972),
    "non_monetary": (0.643, 0.857),
    "effort": (0.70, 0.79),
}
BETA_POINT = {"money": 0.938, "non_monetary": 0.750, "effort": 0.745}

# Reward headroom over the largest perceived cost, so that instances probe route
# choice rather than abandonment unless abandonment is the point.
REWARD_SAFETY = 1.5


@dataclass(frozen=True)
class Spec:
    """What an instance was built to be, carried alongside it for analysis."""

    family: str
    threshold: float | None = None
    """The beta at which the agent's route flips, when the instance has one."""

    critical: frozenset[Arc] = frozenset()
    """Arcs the principal requires, if the instance tests compliance."""

    costs_effort: bool = True
    """Whether taking the tempting route actually wastes effort."""

    notes: str = ""


@dataclass(frozen=True)
class Generated:
    instance: Instance
    spec: Spec


# --- calibrated two-route instances -----------------------------------------


def calibrated_trap(
    threshold: float,
    *,
    immediate: float = 1.0,
    steady: float = 6.0,
    beta: float = 0.75,
    reward: float | None = None,
) -> Generated:
    """A two-route instance whose choice flips at exactly `threshold`.

    One route is cheap now and expensive later; the other splits its cost evenly.
    Preferring the tempting route requires

        immediate + beta * tail  <=  steady + beta * steady,

    so solving for the tail places the switch wherever we want it:
    tail = steady + (steady - immediate) / threshold. Below the threshold the
    agent takes the tempting route and pays immediate + tail; above it, 2 * steady.
    Since threshold < 1 always makes the tempting route the more expensive one,
    every instance of this family costs effort when the agent falls for it.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold}")
    if steady <= immediate:
        raise ValueError("the steady route must start dearer than the tempting one")

    tail = steady + (steady - immediate) / threshold
    graph = nx.MultiDiGraph()
    graph.add_edge("s", "tempting", key=0, weight=immediate)
    graph.add_edge("tempting", "t", key=0, weight=tail)
    graph.add_edge("s", "steady", key=0, weight=steady)
    graph.add_edge("steady", "t", key=0, weight=steady)

    if reward is None:
        reward = REWARD_SAFETY * tail / min(beta, threshold)

    return Generated(
        instance=Instance(graph, "s", "t", beta, reward),
        spec=Spec(
            family="calibrated_trap",
            threshold=threshold,
            costs_effort=True,
            notes=f"tempting route costs {immediate + tail:g}, steady {2 * steady:g}",
        ),
    )


def _two_route_instance(
    immediate: float,
    tempting_tail: float,
    steady_first: float,
    steady_tail: float,
    beta: float,
    reward: float | None,
) -> Instance:
    graph = nx.MultiDiGraph()
    graph.add_edge("s", "tempting", key=0, weight=immediate)
    graph.add_edge("tempting", "t", key=0, weight=tempting_tail)
    graph.add_edge("s", "steady", key=0, weight=steady_first)
    graph.add_edge("steady", "t", key=0, weight=steady_tail)
    if reward is None:
        reward = REWARD_SAFETY * max(tempting_tail, steady_tail) / 0.6
    return Instance(graph, "s", "t", beta, reward)


def two_route_threshold(instance: Instance) -> float:
    """The beta at which a two-route instance flips, recovered from the graph."""
    from .model import shortest_distances

    distance = shortest_distances(instance)
    tempting = ("s", "tempting", 0)
    steady = ("s", "steady", 0)
    gap = distance[tempting[1]] - distance[steady[1]]
    slack = instance.weight(steady) - instance.weight(tempting)
    return slack / gap if gap else float("inf")


def equal_cost_compliance(
    *,
    total: float = 14.0,
    immediate: float = 1.0,
    steady_first: float = 6.0,
    beta: float = 0.75,
    reward: float | None = None,
) -> Generated:
    """Two routes of identical total cost, one of which carries the critical arc.

    Because the totals match, the threshold is not a free parameter: it is exactly
    1. Writing the flip condition with both routes costing C gives

        (steady_first - immediate) / ((C - immediate) - (C - steady_first)) = 1,

    so *any* agent with beta < 1 defects to whichever route is cheaper right now,
    and only a perfectly patient one is indifferent. That makes this family the
    most sensitive detector of present bias we can build - it fires on any bias at
    all - at the price of not localising beta. It also puts a genuine tie at
    beta = 1, where a patient agent may go either way, which is where
    tie-breaking robustness starts to matter in practice.
    """
    if steady_first <= immediate:
        raise ValueError("the steady route must start dearer than the tempting one")
    if total <= steady_first:
        raise ValueError("the total must cover the steady route's first arc")

    instance = _two_route_instance(
        immediate, total - immediate, steady_first, total - steady_first, beta, reward
    )
    return Generated(
        instance=instance,
        spec=Spec(
            family="equal_cost_compliance",
            threshold=two_route_threshold(instance),
            critical=frozenset({("s", "steady", 0)}),
            costs_effort=False,
            notes=f"both routes cost {total:g}; only compliance separates them",
        ),
    )


def costly_compliance(
    *,
    immediate: float = 1.0,
    tempting_tail: float = 11.0,
    steady_first: float = 6.0,
    steady_tail: float = 12.0,
    beta: float = 0.75,
    reward: float | None = None,
) -> Generated:
    """The critical action sits on a route no agent would ever choose.

    Here the tempting route is cheaper both immediately and in total, so it
    dominates at every beta and even a perfectly patient agent skips the critical
    arc. This is the situation of Figure 1, isolated: what the principal wants
    genuinely costs more, which is precisely why the agent will not do it unaided
    and why the environment has to be edited at all.

    It is also the family that gives E4 its material - alignment here is not a
    matter of nudging a borderline choice but of removing the alternative.
    """
    if immediate >= steady_first or tempting_tail >= steady_tail:
        raise ValueError("the tempting route must dominate at every beta")

    instance = _two_route_instance(
        immediate, tempting_tail, steady_first, steady_tail, beta, reward
    )
    return Generated(
        instance=instance,
        spec=Spec(
            family="costly_compliance",
            threshold=None,
            critical=frozenset({("s", "steady", 0)}),
            costs_effort=False,
            notes=(
                f"tempting costs {immediate + tempting_tail:g} against "
                f"{steady_first + steady_tail:g}; no beta prefers compliance"
            ),
        ),
    )


def integer_trap(immediate: int, steady: int, tail: int, *, beta: float = 0.75,
                 reward: float | None = None) -> Generated:
    """A trap built from whole numbers, with the threshold read off afterwards.

    `calibrated_trap` fixes the threshold and solves for the tail, which lands on
    values like 13.993006993. That is fine for a reference agent and bad for a real
    one: asking a model to total awkward decimals confuses "cannot add" with
    "cannot add this", and the arithmetic control is what keeps those two apart.

    Here the weights are chosen instead and the threshold - (steady - immediate) /
    (tail - steady) - is whatever they imply. `integer_sweep` searches for triples
    whose thresholds land where we need them.
    """
    if not immediate < steady < tail:
        raise ValueError("need immediate < steady < tail for a genuine trap")

    instance = _two_route_instance(
        float(immediate), float(tail), float(steady), float(steady), beta, reward
    )
    return Generated(
        instance=instance,
        spec=Spec(
            family="integer_trap",
            threshold=two_route_threshold(instance),
            costs_effort=True,
            notes=f"weights {immediate}/{tail} against {steady}/{steady}",
        ),
    )


def integer_sweep(band: str = "effort", count: int = 5, *, limit: int = 40,
                  **kwargs) -> list[Generated]:
    """Whole-number traps whose thresholds spread across an attested band.

    Enumerates small triples, keeps those whose implied threshold falls in the band,
    and picks the `count` that are most evenly spaced - so the sweep brackets an
    agent as tightly as the calibrated version while only ever showing it integers.
    """
    low, high = BETA_BANDS[band]
    found: dict[float, tuple[int, int, int]] = {}
    for immediate in range(1, limit // 3):
        for steady in range(immediate + 1, limit):
            for tail in range(steady + 1, limit * 2):
                threshold = (steady - immediate) / (tail - steady)
                if low < threshold < high:
                    # Prefer the smallest numbers achieving each distinct threshold.
                    key = round(threshold, 6)
                    if key not in found:
                        found[key] = (immediate, steady, tail)

    if len(found) < count:
        raise ValueError(f"only {len(found)} integer thresholds in band {band}")

    ordered = sorted(found)
    picked = [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]
    return [integer_trap(*found[threshold], **kwargs) for threshold in picked]


def sweep_thresholds(
    band: str = "effort",
    count: int = 5,
    **kwargs,
) -> list[Generated]:
    """Calibrated traps spread across an empirically attested band of beta.

    The threshold is how patient an agent would have to be to resist, so a trap at
    0.85 tempts everyone below 0.85 and is the *stronger* one. An agent therefore
    falls for every trap whose threshold exceeds its own beta and resists the rest,
    and the turning point in the sweep brackets its beta from both sides.
    """
    low, high = BETA_BANDS[band]
    step = (high - low) / (count + 1)
    return [calibrated_trap(low + step * (i + 1), **kwargs) for i in range(count)]


# --- structural families ----------------------------------------------------


class WeightScheme:
    """Draws arc weights from a fixed palette, which fixes |W| directly.

    The tractable regimes of Theorem 8 are stated in terms of the number of
    distinct arc weights, so the corpus needs that as a knob rather than an
    accident of sampling.
    """

    def __init__(self, palette: list[float], rng: random.Random) -> None:
        if not palette:
            raise ValueError("the palette must offer at least one weight")
        self.palette = palette
        self.rng = rng

    def draw(self) -> float:
        return self.rng.choice(self.palette)

    @classmethod
    def uniform(cls, distinct: int, rng: random.Random, top: float = 16.0) -> "WeightScheme":
        step = top / distinct
        return cls([round(step * (i + 1), 3) for i in range(distinct)], rng)


def layered_dag(
    layers: int,
    width: int,
    rng: random.Random,
    *,
    weights: WeightScheme | None = None,
    density: float = 0.5,
    beta: float = 0.75,
    reward: float = 500.0,
) -> Generated:
    """Fan out from the source through `layers` ranks of `width`, then fan in.

    The shape of Montage and CyberShake: a broad parallel middle pinned at both
    ends. Arcs run only between consecutive ranks, which bounds pathwidth by
    roughly twice the rank width and gives the treewidth knob C1 needs. Every node
    keeps at least one arc in and one out, so no dead ends distort the agent's
    reasoning.
    """
    weights = weights or WeightScheme.uniform(4, rng)
    graph = nx.MultiDiGraph()
    ranks = [["s"]] + [[f"L{i}_{j}" for j in range(width)] for i in range(layers)] + [["t"]]

    for upper, lower in zip(ranks, ranks[1:]):
        chosen: set[tuple[str, str]] = set()
        for tail in upper:
            for head in lower:
                if rng.random() < density:
                    chosen.add((tail, head))
        for tail in upper:  # guarantee an exit
            if not any(t == tail for t, _ in chosen):
                chosen.add((tail, rng.choice(lower)))
        for head in lower:  # guarantee an entrance
            if not any(h == head for _, h in chosen):
                chosen.add((rng.choice(upper), head))
        for tail, head in sorted(chosen):
            graph.add_edge(tail, head, key=0, weight=weights.draw())

    return Generated(
        instance=Instance(graph, "s", "t", beta, reward),
        spec=Spec(
            family="layered",
            notes=f"{layers} ranks of width {width}, density {density}, |W|={len(weights.palette)}",
        ),
    )


def parallel_chains(
    chains: int,
    length: int,
    rng: random.Random,
    *,
    weights: WeightScheme | None = None,
    beta: float = 0.75,
    reward: float = 500.0,
) -> Generated:
    """Independent pipelines from source to target, in the shape of Epigenomics.

    Nothing interacts once a chain is entered, so the whole decision is made at the
    source. These are the easy cases structurally - treewidth stays at 1 - which
    makes them the control against which the layered family is read.
    """
    weights = weights or WeightScheme.uniform(4, rng)
    graph = nx.MultiDiGraph()

    for c in range(chains):
        nodes = ["s"] + [f"C{c}_{i}" for i in range(length - 1)] + ["t"]
        for tail, head in zip(nodes, nodes[1:]):
            graph.add_edge(tail, head, key=0, weight=weights.draw())

    return Generated(
        instance=Instance(graph, "s", "t", beta, reward),
        spec=Spec(
            family="parallel_chains",
            notes=f"{chains} chains of length {length}, |W|={len(weights.palette)}",
        ),
    )


def series_parallel(
    depth: int,
    rng: random.Random,
    *,
    weights: WeightScheme | None = None,
    beta: float = 0.75,
    reward: float = 500.0,
) -> Generated:
    """Recursively nested series and parallel composition, so treewidth stays <= 2.

    This is the regime the paper's hardness result lands in - MCTA is already
    NP-hard at pathwidth two - so these instances are where a small structural
    parameter provably fails to rescue the problem, and where the gap between the
    solvers should show up first.
    """
    weights = weights or WeightScheme.uniform(4, rng)
    graph = nx.MultiDiGraph()
    counter = itertools.count()

    def build(source: str, sink: str, remaining: int) -> None:
        if remaining <= 0:
            graph.add_edge(source, sink, key=len(graph.get_edge_data(source, sink) or {}),
                           weight=weights.draw())
            return
        if rng.random() < 0.5:  # series: split through a fresh middle node
            middle = f"n{next(counter)}"
            build(source, middle, remaining - 1)
            build(middle, sink, remaining - 1)
        else:  # parallel: two independent routes between the same endpoints
            build(source, sink, remaining - 1)
            build(source, sink, remaining - 1)

    build("s", "t", depth)

    return Generated(
        instance=Instance(graph, "s", "t", beta, reward),
        spec=Spec(
            family="series_parallel",
            notes=f"nesting depth {depth}, |W|={len(weights.palette)}",
        ),
    )
