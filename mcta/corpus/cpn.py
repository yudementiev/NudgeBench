"""Course-prerequisite networks as task graphs.

Two corpora, both published by Stavrinides & Zuev alongside *Course-prerequisite
networks for analyzing and understanding academic curricula*, Applied Network
Science 2023, at https://github.com/pstavrin/Course-Prerequisite-Networks. Run
`scripts/fetch_data.sh` first; the data is not vendored here.

Caltech arrives with a hand-cleaned prerequisite column and is exact. Johns
Hopkins arrives as catalogue prose and has to be extracted, which makes it larger,
messier and - importantly - only an upper bound on its own structure; see
`load_jhu`. Together they give small, exact instances and larger, approximate
ones, which is the spread the benchmark needs.

A modelling caveat worth stating plainly, because it limits what these instances
can show. A CPN is a *dependency* graph with conjunctive semantics: an arc from
"Ma 1" to "Ma 2" means Ma 2 cannot be taken until Ma 1 has been, and a course with
three prerequisites needs all three. A path in the Kleinberg-Oren model is a
*sequence* of actions, and the agent picks one continuation at each step. The two
are not the same object, so a mapping has to be chosen rather than read off.

What we take is the reading the paper's own motivation uses when it speaks of
curriculum maps: a route through the curriculum. Fix a target course; the courses
that can reach it form its prerequisite closure; a path from an entry-level course
to the target is one order in which a student could arrive there. Weights are the
effort a course costs. The agent chooses which chain to climb, and present bias
makes it prefer whichever next course is cheapest now.

This reading keeps the graph's real topology - which is all claim C1 depends on,
since structural parameters do not look at weights at all - while being honest
that the conjunctive requirement is not being modelled. Instances derived this way
answer "how hard is this shape to align", not "what should this student enrol in".
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..model import Instance

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "cpn" / "caltech.csv"

# The published spreadsheet's own column names, typo included.
_NODE = "Node_name"
_CLEAN_PREREQS = "Prereaquisites (clean)"
_DEPARTMENT = "department_name"
_TITLE = "course_title"


@dataclass(frozen=True)
class Course:
    name: str
    title: str
    department: str


def load_caltech(path: Path | str = DEFAULT_PATH) -> nx.DiGraph:
    """Read the Caltech CPN into a DAG whose arcs run prerequisite -> dependent.

    The published file lists 771 courses and 772 prerequisite relations, every
    reference resolving to a listed course, and the result is acyclic - all of
    which this function asserts rather than assumes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/fetch_data.sh to fetch the published data"
        )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    graph = nx.DiGraph()
    for row in rows:
        name = row[_NODE].strip()
        if name:
            graph.add_node(
                name,
                course=Course(name, row[_TITLE].strip(), row[_DEPARTMENT].strip()),
            )

    unresolved: list[str] = []
    for row in rows:
        course = row[_NODE].strip()
        for prerequisite in (part.strip() for part in row[_CLEAN_PREREQS].split(",")):
            if not prerequisite:
                continue
            if prerequisite in graph:
                graph.add_edge(prerequisite, course)
            else:
                unresolved.append(prerequisite)

    if unresolved:
        raise ValueError(f"{len(unresolved)} prerequisite references do not name a listed course")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the prerequisite network contains a cycle")

    return graph


JHU_PATH = Path(__file__).resolve().parents[2] / "data" / "cpn" / "jhu.csv"

_COURSE_CODE = re.compile(r"[A-Z]{2}\.\d{3}\.\d{3}")
# Prerequisite blurbs name the dependent course after a phrase like "prior to
# enrolling in", so codes past that point are not prerequisites at all.
_DEPENDENT_CLAUSE = re.compile(
    r"\bprior to\b|\bbefore (?:taking|enrolling)\b|\bconcurrent", re.IGNORECASE
)


def load_jhu(path: Path | str = JHU_PATH, *, trim_dependent_clause: bool = True) -> nx.DiGraph:
    """Read the Johns Hopkins course listing into a prerequisite DAG.

    Unlike the Caltech file, this one carries no cleaned prerequisite column, only
    the catalogue's prose - "You must take AS.440.602 AND AS.440.606 prior to
    enrolling in AS.440.614". Course codes are extracted by pattern, the clause
    naming the dependent course is trimmed, and self-references are dropped.

    **This over-approximates the real dependency structure**, and deliberately so.
    Roughly two in five blurbs express a disjunction - "one of the following:
    X OR Y" - and reading every mentioned code as a requirement turns that choice
    into two obligations, adding arcs that the catalogue does not. The direction is
    the safe one for claim C1: treewidth never decreases when arcs are added, so a
    measurement taken here is an upper bound on the truth. If even the
    over-connected reading is narrow, the real curriculum is narrower still.

    What this graph cannot support is any claim about a *particular* course's
    requirements. It is a shape to measure, not a catalogue to advise from.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/fetch_data.sh to fetch the published data"
        )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    graph = nx.DiGraph()
    for row in rows:
        code = row["Course"].strip()
        if code:
            graph.add_node(
                code,
                course=Course(code, row["Title"].strip(), row["Department Name"].strip()),
            )

    for row in rows:
        dependent = row["Course"].strip()
        blurb = row["Pre Req"]
        if trim_dependent_clause:
            blurb = _DEPENDENT_CLAUSE.split(blurb)[0]
        for prerequisite in _COURSE_CODE.findall(blurb):
            # Codes may name courses absent from the listing, and a blurb may name
            # its own course; neither is an arc.
            if prerequisite != dependent and prerequisite in graph:
                graph.add_edge(prerequisite, dependent)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("the extracted prerequisite network contains a cycle")

    return graph


def candidate_targets(cpn: nx.DiGraph, *, min_courses: int = 6, min_depth: int = 3) -> list[str]:
    """Courses whose prerequisite closure is big and deep enough to plan through.

    A course with two prerequisites and no depth poses the agent no choice, so it
    measures nothing. `min_depth` is the longest chain of prerequisites leading to
    the course, which is the p of the resulting instance.
    """
    targets = []
    for course in cpn:
        closure = nx.ancestors(cpn, course)
        if len(closure) + 1 < min_courses:
            continue
        induced = cpn.subgraph(closure | {course})
        if nx.dag_longest_path_length(induced) >= min_depth:
            targets.append(course)
    return sorted(targets)


def to_instance(
    cpn: nx.DiGraph,
    target: str,
    *,
    weight_of=None,
    beta: float = 0.75,
    reward: float | None = None,
) -> Instance:
    """Turn the prerequisite closure of `target` into a task graph.

    A virtual START is attached to every course in the closure that has no
    prerequisite of its own, so the agent begins with a genuine choice of where to
    enter the curriculum. Arcs point in the direction of study - take the
    prerequisite, then the course that needed it.

    `weight_of` maps a course name to the effort it costs; the default charges
    every course the same, because the published data carries no unit or workload
    column. That makes |W| = 1, which is the friendliest possible regime for the
    paper's algorithms and should be reported as an assumption, not a measurement.
    """
    if target not in cpn:
        raise KeyError(f"{target!r} is not a course in this network")

    closure = nx.ancestors(cpn, target) | {target}
    induced = cpn.subgraph(closure)
    charge = weight_of or (lambda course: 1.0)

    graph = nx.MultiDiGraph()
    graph.add_node("START")
    for course in induced:
        graph.add_node(course)
    for tail, head in induced.edges():
        graph.add_edge(tail, head, key=0, weight=charge(head))

    entries = [c for c in induced if induced.in_degree(c) == 0]
    for entry in entries:
        graph.add_edge("START", entry, key=0, weight=charge(entry))

    if reward is None:
        # Generous enough that these instances probe route choice, not abandonment.
        reward = 4.0 * sum(charge(c) for c in induced)

    return Instance(graph, "START", target, beta, reward)


def largest_component(cpn: nx.DiGraph) -> nx.DiGraph:
    """The biggest weakly connected piece - the paper reports 436 courses, 747 links."""
    biggest = max(nx.weakly_connected_components(cpn), key=len)
    return cpn.subgraph(biggest).copy()
