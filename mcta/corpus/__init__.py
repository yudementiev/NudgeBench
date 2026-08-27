"""Real task graphs, loaded from published sources. See REFERENCES.md."""

from .cpn import (
    Course,
    candidate_targets,
    largest_component,
    load_caltech,
    load_jhu,
    to_instance,
)
from .taskbench import Decomposition
from .workflows import Trace, load_all, load_trace

__all__ = [
    "Course",
    "Decomposition",
    "Trace",
    "load_all",
    "load_jhu",
    "load_trace",
    "candidate_targets",
    "largest_component",
    "load_caltech",
    "to_instance",
]
