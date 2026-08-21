"""Solving MINIMUM-COST TASK ALIGNMENT.

Given a model, a cost for intervening on each arc, and a set of critical arcs,
find the cheapest set of arcs to delete such that every trajectory the agent might
follow reaches the target and performs every critical action.

Costs are real, not just positive, which is what lets one formulation carry both
kinds of intervention. Deleting an arc the environment already has costs what it
costs. To model *adding* an arc for a price, put it in the graph up front and give
it a negative deletion cost: the solver then pays, relative to the baseline where
every auxiliary arc is stripped out, only for the ones it chooses to keep.

The problem is bilevel, and that is what makes it awkward. The principal moves
first, but which arcs the agent will consider depends on the shortest distances in
the graph the principal leaves behind, so deleting an arc in one corner can change
what the agent does in another. Alignment is therefore not monotone in the deleted
set - deleting more can break an instance that was already aligned, by stranding
the agent - and the usual submodular shortcuts do not apply. What we do have is a
polynomial-time oracle for "is this aligned, and if not, here is a trajectory that
breaks it", which is Lemma 2 of the paper, and the solvers below are built around
calling it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable

import networkx as nx

from .align import check
from .model import Arc, Instance, analyze

CostFn = Callable[[Arc], float]


@dataclass
class Intervention:
    """What a solver decided to do, and what it cost."""

    deleted: frozenset[Arc]
    cost: float
    aligned: bool
    solver: str = ""
    oracle_calls: int = 0
    proved_optimal: bool = False
    note: str = ""

    def summary(self) -> str:
        state = "aligned" if self.aligned else "FAILED"
        proof = " (optimal)" if self.proved_optimal else ""
        return f"{self.solver}: {state}{proof} cost={self.cost:g} |S|={len(self.deleted)}"


def unit_cost(_: Arc) -> float:
    """The paper's own reduction: minimising the number of deleted arcs."""
    return 1.0


def total_cost(deleted: Iterable[Arc], cost: CostFn) -> float:
    return sum(cost(arc) for arc in deleted)


def is_aligned(instance: Instance, critical: set[Arc]) -> bool:
    return bool(check(instance, analyze(instance), critical))


def candidate_arcs(instance: Instance, critical: set[Arc]) -> set[Arc]:
    """Arcs a solver could ever want to delete.

    Two reductions, both sound. A critical arc can never be deleted - the agent is
    required to traverse it, so removing it makes the instance unsatisfiable. And
    an arc the agent could never take is not worth paying to remove: if its head
    cannot reach the target it is never on a minimising continuation, and if its
    tail is unreachable from the source the agent never stands there. Deletion only
    ever shrinks reachability, so an arc useless at the outset stays useless.
    """
    graph = instance.graph
    reaches_target = {instance.target} | nx.ancestors(graph, instance.target)
    from_source = {instance.source} | nx.descendants(graph, instance.source)

    return {
        arc
        for arc in instance.arcs()
        if arc not in critical and arc[0] in from_source and arc[1] in reaches_target
    }


def is_alignable(instance: Instance, critical: set[Arc]) -> bool:
    """A *sufficient* test that this instance can be aligned at all.

    Deleting everything off one route that threads every critical arc is the most
    forceful move available, so finding such a route proves the instance solvable.
    The converse does not hold: the route is built greedily, taking critical arcs in
    order of their distance from the source and connecting them by shortest legs,
    and a different threading might succeed where that one fails. Read True as a
    proof and False as "no forced route was found" - `solve_exact` settles the
    remaining cases.
    """
    if is_aligned(instance, critical):
        return True
    return _forced_route(instance, critical) is not None


def _forced_route(instance: Instance, critical: set[Arc]) -> frozenset[Arc] | None:
    """Delete everything off one route that threads every critical arc.

    This is the sledgehammer: leave the agent no choice whatsoever. It is a valid
    solution whenever one exists, which makes it both a feasibility test and the
    starting incumbent for search.
    """
    graph = instance.graph
    ordered = sorted(critical, key=lambda arc: _depth(graph, instance.source, arc[0]))

    route: list[Arc] = []
    node = instance.source
    for arc in ordered:
        leg = _shortest_leg(instance, node, arc[0])
        if leg is None:
            return None
        route.extend(leg)
        route.append(arc)
        node = arc[1]

    leg = _shortest_leg(instance, node, instance.target)
    if leg is None:
        return None
    route.extend(leg)

    keep = set(route)
    deleted = frozenset(arc for arc in instance.arcs() if arc not in keep)
    reduced = instance.without(set(deleted))
    return deleted if is_aligned(reduced, critical) else None


def _depth(graph: nx.MultiDiGraph, source: str, node: str) -> int:
    try:
        return nx.shortest_path_length(graph, source, node)
    except nx.NetworkXNoPath:
        return math.inf  # type: ignore[return-value]


def _shortest_leg(instance: Instance, start: str, end: str) -> list[Arc] | None:
    """The cheapest sequence of arcs from `start` to `end`, or None if disconnected."""
    if start == end:
        return []
    try:
        nodes = nx.shortest_path(instance.graph, start, end, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    legs: list[Arc] = []
    for tail, head in zip(nodes, nodes[1:]):
        parallel = instance.graph.get_edge_data(tail, head)
        key = min(parallel, key=lambda k: parallel[k]["weight"])
        legs.append((tail, head, key))
    return legs


# --- heuristics -------------------------------------------------------------


def greedy_witness_guided(
    instance: Instance,
    critical: set[Arc],
    cost: CostFn = unit_cost,
    *,
    max_rounds: int = 200,
) -> Intervention:
    """Block the bad trajectory the oracle just showed you, and repeat.

    This is what an engineer does by hand: watch the agent take a route it should
    not, remove one step of that route, run it again. Each round deletes the
    cheapest arc on the witness, which is the locally sensible move and, as E3 is
    meant to show, not generally the right one - blocking a route can divert the
    agent onto a worse one rather than onto the intended one, exactly as deleting
    (b, f) does in the paper's own worked example.
    """
    deleted: set[Arc] = set()
    allowed = candidate_arcs(instance, critical)
    calls = 0

    for _ in range(max_rounds):
        current = instance.without(deleted)
        verdict = check(current, analyze(current), critical)
        calls += 1
        if verdict.aligned:
            return Intervention(
                frozenset(deleted), total_cost(deleted, cost), True,
                "greedy-witness", calls,
            )

        options = [arc for arc in (verdict.witness or ()) if arc in allowed and arc not in deleted]
        if not options:
            return Intervention(
                frozenset(deleted), total_cost(deleted, cost), False,
                "greedy-witness", calls, note="no deletable arc on the witness trajectory",
            )
        deleted.add(min(options, key=cost))

    return Intervention(
        frozenset(deleted), total_cost(deleted, cost), False,
        "greedy-witness", calls, note=f"gave up after {max_rounds} rounds",
    )


def minimalize(
    instance: Instance,
    critical: set[Arc],
    deleted: Iterable[Arc],
    cost: CostFn = unit_cost,
) -> Intervention:
    """Drop deletions the solution does not need, dearest first.

    Because alignment is not monotone, a deletion that looked necessary when it was
    made may have been rendered redundant - or actively harmful - by a later one.
    Restoring an arc and re-checking is cheap, and it is the one improvement that
    applies to any solution whatever produced it.
    """
    keep = set(deleted)
    calls = 0

    for arc in sorted(keep, key=cost, reverse=True):
        trial = keep - {arc}
        candidate = instance.without(trial)
        calls += 1
        if is_aligned(candidate, critical):
            keep = trial

    reduced = instance.without(keep)
    return Intervention(
        frozenset(keep), total_cost(keep, cost), is_aligned(reduced, critical),
        "minimalized", calls,
    )


# --- exact search -----------------------------------------------------------


@dataclass
class _Search:
    instance: Instance
    critical: set[Arc]
    cost: CostFn
    candidates: list[Arc]
    suffix_discount: list[float]
    """suffix_discount[i] sums the negative costs from position i on - the most a
    branch could still save, which is what makes pruning sound when costs may be
    negative."""

    best: float = math.inf
    best_set: frozenset[Arc] = frozenset()
    calls: int = 0
    exhausted: bool = True
    node_budget: int = 2_000_000


def solve_exact(
    instance: Instance,
    critical: set[Arc],
    cost: CostFn = unit_cost,
    *,
    node_budget: int = 2_000_000,
) -> Intervention:
    """Branch and bound over subsets of the candidate arcs.

    Alignment is checked by the polynomial oracle at every node, and a branch is
    abandoned once it cannot beat the incumbent. With non-negative costs that test
    is just the running total; with negative costs - the encoding for paid
    additions - a branch could still save whatever negative costs remain ahead of
    it, so the bound subtracts those. Reports `proved_optimal` only when the search
    actually finished; otherwise the answer is the best found within budget.
    """
    candidates = sorted(candidate_arcs(instance, critical), key=cost)
    already = is_aligned(instance, critical)

    # Doing nothing is optimal only when nothing pays you to act. Under the
    # encoding for paid additions some costs are negative, and then an aligned
    # instance can still be improved by stripping auxiliary arcs it does not need.
    if already and all(cost(arc) >= 0 for arc in candidates):
        return Intervention(frozenset(), 0.0, True, "exact", 1, True, "already aligned")

    suffix = [0.0] * (len(candidates) + 1)
    for i in range(len(candidates) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + min(0.0, cost(candidates[i]))

    search = _Search(
        instance=instance,
        critical=critical,
        cost=cost,
        candidates=candidates,
        suffix_discount=suffix,
        node_budget=node_budget,
    )

    # A tight incumbent is worth more than any amount of clever ordering, because
    # every branch is measured against it. Take the best cheap answer available:
    # doing nothing when that already works, otherwise the trimmed greedy solution,
    # falling back to the forced route only if the heuristic fails outright.
    if already:
        search.best = 0.0
        search.best_set = frozenset()
    else:
        seed = greedy_witness_guided(instance, critical, cost)
        if seed.aligned:
            seed = minimalize(instance, critical, seed.deleted, cost)
        if seed.aligned:
            search.best = seed.cost
            search.best_set = seed.deleted
        else:
            forced = _forced_route(instance, critical)
            if forced is not None:
                search.best = total_cost(forced, cost)
                search.best_set = forced

    _descend(search, index=0, deleted=set(), running=0.0)

    aligned = search.best < math.inf
    return Intervention(
        frozenset(search.best_set),
        search.best if aligned else math.inf,
        aligned,
        "exact",
        search.calls,
        proved_optimal=search.exhausted and aligned,
        note="" if search.exhausted else "search budget exhausted; bound not proved",
    )


def _descend(search: _Search, index: int, deleted: set[Arc], running: float) -> None:
    if search.calls >= search.node_budget:
        search.exhausted = False
        return
    if running + search.suffix_discount[index] >= search.best:
        return

    if deleted:
        reduced = search.instance.without(deleted)
        search.calls += 1
        if is_aligned(reduced, search.critical) and running < search.best:
            search.best = running
            search.best_set = frozenset(deleted)
            # Non-monotonicity means a superset could still be cheaper only if some
            # remaining cost is negative; the bound above already accounts for that.

    if index >= len(search.candidates):
        return

    arc = search.candidates[index]
    deleted.add(arc)
    _descend(search, index + 1, deleted, running + search.cost(arc))
    deleted.discard(arc)
    _descend(search, index + 1, deleted, running)


SOLVERS: dict[str, Callable[..., Intervention]] = {
    "exact": solve_exact,
    "greedy-witness": greedy_witness_guided,
}
