"""Deciding whether an instance is aligned, and exhibiting a witness when it is not.

The principal wants *every* feasible trajectory to reach the target and traverse
every arc of a prescribed critical set T. Checking this directly would mean
enumerating trajectories, of which there may be exponentially many. Lemmas 1 and
2 of Belova et al. avoid that: a path contains all of T exactly when it contains
no "T-avoiding" arc, and a single backwards pass then decides alignment and
reconstructs a violating trajectory when one exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .model import Arc, Behavior, Instance


@dataclass
class Verdict:
    """The outcome of an alignment check."""

    aligned: bool
    witness: list[Arc] | None = None
    """A feasible trajectory that abandons or skips a critical arc, if one exists."""

    reason: str = ""

    def __bool__(self) -> bool:
        return self.aligned


def descendants_with_self(graph: nx.MultiDiGraph, node: str) -> set[str]:
    return nx.descendants(graph, node) | {node}


def t_avoiding_arcs(instance: Instance, critical: set[Arc]) -> set[Arc]:
    """Arcs whose traversal makes some critical arc unreachable (Definition 1).

    An arc (u, v) is T-avoiding if some (x, y) in T is distinct from it, has x
    reachable from u, but no longer has x reachable from v. Taking such an arc
    forecloses the critical action, so by Lemma 1 a trajectory traverses all of T
    precisely when it avoids every arc of this set.
    """
    reach = {node: descendants_with_self(instance.graph, node) for node in instance.graph}
    avoiding: set[Arc] = set()

    for arc in instance.arcs():
        tail, head, _ = arc
        for critical_arc in critical:
            x = critical_arc[0]
            if arc != critical_arc and x in reach[tail] and x not in reach[head]:
                avoiding.add(arc)
                break

    return avoiding


def check(instance: Instance, behavior: Behavior, critical: set[Arc]) -> Verdict:
    """Decide whether every feasible trajectory reaches t and traverses all of T.

    Implements Lemma 2: propagate a boolean A(v) backwards, where A(v) records
    that some feasible continuation from v either abandons or gives up on a
    critical arc. The instance is aligned exactly when A(s) is false.
    """
    if not _critical_arcs_lie_on_some_path(instance, critical):
        return Verdict(
            aligned=False,
            reason="no s-t path traverses every critical arc, so no intervention can align this instance",
        )

    avoiding = t_avoiding_arcs(instance, critical)
    doomed: dict[str, bool] = {}

    for node in reversed(list(nx.topological_sort(instance.graph))):
        if node == instance.target:
            doomed[node] = False
        elif behavior.abandons_at(node):
            doomed[node] = True
        else:
            doomed[node] = any(
                doomed[arc[1]] or arc in avoiding
                for arc in instance.out_arcs(node)
                if arc in behavior.feasible
            )

    if not doomed[instance.source]:
        return Verdict(aligned=True, reason="every feasible trajectory reaches t and traverses T")

    return Verdict(
        aligned=False,
        witness=_reconstruct_witness(instance, behavior, avoiding, doomed),
        reason="a feasible trajectory abandons or skips a critical arc",
    )


def _critical_arcs_lie_on_some_path(instance: Instance, critical: set[Arc]) -> bool:
    """Whether *any* s-t path traverses every critical arc.

    Lemma 1 assumes such a path exists; without it the instance is unalignable and
    the T-avoiding characterisation does not apply. The critical arcs must form a
    chain that can be threaded in a single pass from s to t.
    """
    if not critical:
        return True

    reach = {node: descendants_with_self(instance.graph, node) for node in instance.graph}
    if any(instance.target not in reach[arc[1]] for arc in critical):
        return False
    if any(arc[0] not in reach[instance.source] for arc in critical):
        return False

    # Order the critical arcs by reachability and check they chain up.
    ordered = sorted(critical, key=lambda arc: len(reach[arc[1]]), reverse=True)
    for earlier, later in zip(ordered, ordered[1:]):
        if later[0] not in reach[earlier[1]]:
            return False
    return True


def _reconstruct_witness(
    instance: Instance,
    behavior: Behavior,
    avoiding: set[Arc],
    doomed: dict[str, bool],
) -> list[Arc]:
    """Walk from s along feasible arcs, always into trouble, to exhibit a failure."""
    path: list[Arc] = []
    node = instance.source

    while node != instance.target and not behavior.abandons_at(node):
        for arc in instance.out_arcs(node):
            if arc not in behavior.feasible:
                continue
            if arc in avoiding:
                return path + [arc]
            if doomed[arc[1]]:
                path.append(arc)
                node = arc[1]
                break
        else:  # pragma: no cover - only reachable if `doomed` is inconsistent
            raise RuntimeError(f"no failing continuation at {node!r}, but A({node!r}) is true")

    return path
