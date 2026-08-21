"""Real task graphs, loaded from published sources. See REFERENCES.md."""

from .cpn import (
    Course,
    candidate_targets,
    largest_component,
    load_caltech,
    to_instance,
)

__all__ = [
    "Course",
    "candidate_targets",
    "largest_component",
    "load_caltech",
    "to_instance",
]
