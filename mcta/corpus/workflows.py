"""Scientific workflow executions as task graphs.

Data: WfCommons WfInstances, https://github.com/wfcommons/WfInstances - traces of
real production runs in the WfFormat schema, released under LGPL-3.0. Fetch with
`scripts/fetch_wfinstances.py`; the data is not vendored here.

These are the corpus's first instances with **real weights**. The course networks
carry no workload column, so every course had to be charged the same, which pinned
|W| to 1 and made |L| look small by construction. A workflow trace records how
long each task actually took, in seconds and to three decimals, so the weights are
measured rather than assumed.

That matters for more than realism. Theorem 1 of the paper runs in
(|L| + 1)^O(tw) time, where L collects the distinct weights of paths to the
target - so a graph can be narrow and still out of reach if its path weights are
all different. Continuous runtimes are the worst case for that quantity, and
whether real workflows fall into it is a question the course corpus could not ask.
`quantize` exists to answer the follow-up: how coarse must a schedule's units be
before the algorithm becomes usable?

The conjunctive caveat from the course networks applies here too, and more
sharply. A workflow task genuinely needs all its inputs, whereas an agent path
takes one predecessor chain. Instances derived here describe the *shape* a
principal would have to intervene on, not an executable schedule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..model import Instance

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "workflows"

SUPPORTED_SCHEMA = "1.6"
START, END = "START", "END"


@dataclass(frozen=True)
class Trace:
    """One recorded workflow execution."""

    name: str
    kind: str
    """The workflow application: montage, epigenomics, seismology, and so on."""

    path: Path
    graph: nx.DiGraph
    """Tasks and their dependencies. Node attribute `runtime` is seconds."""

    makespan: float | None = None

    @property
    def tasks(self) -> int:
        return self.graph.number_of_nodes()


def load_trace(path: Path | str) -> Trace:
    """Read one WfFormat instance into a DAG whose nodes carry measured runtimes."""
    path = Path(path)
    document = json.loads(path.read_text())

    version = document.get("schemaVersion")
    if version != SUPPORTED_SCHEMA:
        raise ValueError(
            f"{path.name} uses WfFormat {version}; this parser targets {SUPPORTED_SCHEMA}. "
            "Refusing rather than risk misreading the fields."
        )

    workflow = document["workflow"]
    specification = workflow["specification"]
    execution = workflow.get("execution", {})
    runtimes = {
        task["id"]: task.get("runtimeInSeconds")
        for task in execution.get("tasks", [])
    }

    graph = nx.DiGraph()
    for task in specification["tasks"]:
        graph.add_node(task["id"], runtime=runtimes.get(task["id"]), program=task.get("name", ""))
    for task in specification["tasks"]:
        for child in task.get("children", []):
            if child in graph:
                graph.add_edge(task["id"], child)

    missing = [n for n, data in graph.nodes(data=True) if data["runtime"] is None]
    if missing:
        raise ValueError(f"{path.name}: {len(missing)} tasks have no recorded runtime")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"{path.name}: the recorded dependencies contain a cycle")

    return Trace(
        name=path.stem,
        kind=path.parent.name,
        path=path,
        graph=graph,
        makespan=execution.get("makespanInSeconds"),
    )


def load_all(root: Path | str = DEFAULT_ROOT) -> list[Trace]:
    """Every fetched trace, sorted by application then size."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} is missing; run scripts/fetch_wfinstances.py to fetch the traces"
        )
    traces = [load_trace(p) for p in sorted(root.rglob("*.json"))]
    return sorted(traces, key=lambda t: (t.kind, t.tasks))


def to_instance(
    trace: Trace,
    *,
    quantize: float | None = None,
    beta: float = 0.75,
    reward: float | None = None,
) -> Instance:
    """Wrap a trace between a virtual START and END so it becomes an MCTA instance.

    An arc into a task is charged that task's runtime, so walking a route costs what
    performing those tasks cost. START fans out to every task with no dependency and
    every task nothing depends on runs into END, both at zero cost - they mark the
    boundary rather than adding work.

    `quantize` rounds runtimes to a multiple of the given number of seconds. Leaving
    it None keeps the measured values, which is honest but makes nearly every path
    weight distinct; setting it asks what the algorithm's |L| would be if the
    principal reasoned in coarser units.
    """
    def charge(task: str) -> float:
        runtime = trace.graph.nodes[task]["runtime"]
        if quantize:
            return round(runtime / quantize) * quantize
        return float(runtime)

    graph = nx.MultiDiGraph()
    graph.add_node(START)
    graph.add_node(END)
    for task in trace.graph:
        graph.add_node(task)

    for tail, head in trace.graph.edges():
        graph.add_edge(tail, head, key=0, weight=charge(head))

    for task in trace.graph:
        if trace.graph.in_degree(task) == 0:
            graph.add_edge(START, task, key=0, weight=charge(task))
        if trace.graph.out_degree(task) == 0:
            graph.add_edge(task, END, key=0, weight=0.0)

    if reward is None:
        reward = 4.0 * sum(charge(t) for t in trace.graph)

    return Instance(graph, START, END, beta, reward)
