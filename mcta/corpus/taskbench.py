"""Agent task-decomposition graphs from TaskBench.

Data: TaskBench~\\cite{shen2023taskbench}, MIT licensed, published by Microsoft at
https://huggingface.co/datasets/microsoft/Taskbench. Fetch with
`scripts/fetch_taskbench.py`; the data is not vendored here.

This is the only corpus in the benchmark whose graphs come from *agent* tasks
rather than from human ones. Each record is a user request decomposed into tool
calls, with the dependencies between them annotated: `task_nodes` are the calls and
`task_links` say which consumes another's output. Three tool domains ship - everyday
APIs, HuggingFace models, multimedia processing - which lets domain of origin be a
variable rather than a caveat.

Two properties shape what it can be used for.

*The records carry their own topology label.* `single` (one call), `chain` (a
sequence), `dag` (genuine branching). Only the last poses a planning choice at all,
so only those can measure an agent, and the label saves us inferring it.

*The graphs are small.* One to eight calls, three on average. That is a fact about
how agent tasks are currently decomposed, not a shortcoming of the data, and it is
worth reporting as such: the environments agents actually face are far smaller than
the curricula and workflows in the rest of the corpus.

The conjunctive caveat from the course networks applies here too. A task node needs
all its inputs, while an agent path takes one predecessor chain, so an instance
describes a shape to align rather than an executable plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..model import Instance

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "taskbench"
DOMAINS = ("dailylifeapis", "multimedia", "huggingface")

START, END = "START", "END"


@dataclass(frozen=True)
class Decomposition:
    """One user request and the tool calls it was decomposed into."""

    identifier: str
    domain: str
    topology: str
    """TaskBench's own label: single, chain, or dag."""

    request: str
    graph: nx.DiGraph

    @property
    def calls(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def branching(self) -> int:
        return sum(1 for node in self.graph if self.graph.out_degree(node) > 1)


def load_domain(domain: str, root: Path | str = DEFAULT_ROOT) -> list[Decomposition]:
    """Read one tool domain. Records are JSON lines despite the .json extension."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")
    path = Path(root) / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/fetch_taskbench.py to fetch the published data"
        )

    out: list[Decomposition] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        nodes = [n.get("task") for n in record.get("task_nodes", []) if n.get("task")]
        if not nodes:
            continue

        graph = nx.DiGraph()
        graph.add_nodes_from(sorted(set(nodes)))
        for link in record.get("task_links", []):
            source, target = link.get("source"), link.get("target")
            if source in graph and target in graph and source != target:
                graph.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(graph):
            continue

        out.append(Decomposition(
            identifier=str(record.get("id")),
            domain=domain,
            topology=record.get("type", "unknown"),
            request=record.get("user_request", ""),
            graph=graph,
        ))
    return out


def load_all(root: Path | str = DEFAULT_ROOT) -> list[Decomposition]:
    return [d for domain in DOMAINS for d in load_domain(domain, root)]


def to_instance(
    decomposition: Decomposition,
    *,
    weight_of=None,
    beta: float = 0.75,
    reward: float | None = None,
) -> Instance:
    """Bracket a decomposition between a virtual START and END.

    An arc into a call is charged that call's cost. TaskBench records no runtime or
    price per tool, so weights are supplied by the caller and are synthetic by
    necessity - the same assumption the course corpora carry, and stated for the
    same reason.
    """
    charge = weight_of or (lambda call: 1.0)
    source = decomposition.graph

    graph = nx.MultiDiGraph()
    graph.add_node(START)
    graph.add_node(END)
    for call in sorted(source):
        graph.add_node(call)
    for tail, head in sorted(source.edges()):
        graph.add_edge(tail, head, key=0, weight=float(charge(head)))
    for call in sorted(source):
        if source.in_degree(call) == 0:
            graph.add_edge(START, call, key=0, weight=float(charge(call)))
        if source.out_degree(call) == 0:
            graph.add_edge(call, END, key=0, weight=0.0)

    if reward is None:
        reward = 4.0 * sum(charge(c) for c in source)
    return Instance(graph, START, END, beta, reward)
