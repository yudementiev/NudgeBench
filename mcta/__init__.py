"""MCTA-Bench: alignment of present-biased agents by environment design."""

from .align import Verdict, check, t_avoiding_arcs
from .model import (
    Arc,
    Behavior,
    Instance,
    analyze,
    path_perceived_cost,
    path_weight,
    perceived_cost,
    shortest_distances,
)

__all__ = [
    "Arc",
    "Behavior",
    "Instance",
    "Verdict",
    "analyze",
    "check",
    "path_perceived_cost",
    "path_weight",
    "perceived_cost",
    "shortest_distances",
    "t_avoiding_arcs",
]
