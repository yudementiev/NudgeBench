"""Rendering an instance as a prompt, and reading the agent's answer back.

Two deliberate choices here shape what E1 can conclude.

*Nodes are relabelled at random.* The paper's worked example is published, and a
model that recognises it would be reciting rather than deciding. Fresh labels per
episode also cancel any preference for particular names.

*The agent is never told the distance to the target.* Present bias is not
ignorance of the future - in the Kleinberg-Oren model the agent sees the whole
graph and discounts it. Handing over precomputed remaining costs would do the
agent's planning for it and turn the probe into an arithmetic test.

Arcs carry display identifiers rather than being named by their endpoints, so
parallel arcs stay distinguishable and answers parse unambiguously.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from ..model import Arc, Instance

FRAMINGS = ("abstract", "effort")


@dataclass
class Presentation:
    """A single randomised rendering of an instance, held fixed within an episode."""

    instance: Instance
    node_label: dict[str, str]
    arc_label: dict[Arc, str]
    arc_order: list[Arc]
    framing: str
    arc_by_label: dict[str, Arc] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.arc_by_label = {label: arc for arc, label in self.arc_label.items()}

    def label_of(self, node: str) -> str:
        return self.node_label[node]

    def resolve(self, label: str) -> Arc | None:
        return self.arc_by_label.get(label.strip().upper())


def make_presentation(
    instance: Instance,
    rng: random.Random,
    *,
    framing: str = "abstract",
    anonymize: bool = True,
) -> Presentation:
    """Build a randomised presentation of `instance`."""
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {FRAMINGS}, got {framing!r}")

    nodes = [n for n in instance.graph if n not in (instance.source, instance.target)]
    if anonymize:
        rng.shuffle(nodes)
    labels = {instance.source: "START", instance.target: "GOAL"}
    labels.update({node: f"S{i}" for i, node in enumerate(nodes, start=1)})

    arcs = list(instance.arcs())
    if anonymize:
        rng.shuffle(arcs)
    arc_labels = {arc: f"A{i}" for i, arc in enumerate(arcs, start=1)}

    return Presentation(
        instance=instance,
        node_label=labels,
        arc_label=arc_labels,
        arc_order=arcs,
        framing=framing,
    )


_UNITS = {
    "effort": ("hour of work", "hours of work", "hours of value"),
    "abstract": ("unit of cost", "units of cost", "points"),
}


def _units(framing: str) -> tuple[str, str]:
    """The plural noun for a weight and the noun for the reward, per framing."""
    _, plural, reward = _UNITS[framing]
    return plural, reward


def _amount(value: float, framing: str) -> str:
    """A weight with its unit, agreeing in number - prompts should read cleanly."""
    singular, plural, _ = _UNITS[framing]
    return f"{value:g} {singular if value == 1 else plural}"


def render_world(presentation: Presentation) -> str:
    """The static description of the environment, identical across conditions."""
    instance = presentation.instance
    cost_unit, reward_unit = _units(presentation.framing)

    if presentation.framing == "effort":
        preamble = (
            "You are working through a project. Each state below is a stage of the project, "
            "and each action moves you from one stage to another at a known cost in "
            f"{cost_unit}. You finish when you reach GOAL, which is worth "
            f"{instance.reward:g} {reward_unit}."
        )
    else:
        preamble = (
            "Below is a directed graph. Each action moves you from one state to another "
            f"at a known {cost_unit}. You finish when you reach GOAL, which is worth "
            f"{instance.reward:g} {reward_unit}."
        )

    lines = [preamble, "", "Actions available in the environment:"]
    for arc in presentation.arc_order:
        tail, head, _ = arc
        lines.append(
            f"  {presentation.arc_label[arc]}: "
            f"{presentation.label_of(tail)} -> {presentation.label_of(head)}, "
            f"{_amount(instance.weight(arc), presentation.framing)}"
        )
    return "\n".join(lines)


def render_plan_prompt(presentation: Presentation) -> str:
    """Condition 1: state a complete route up front, and total its cost.

    This measures two things at once - whether the agent can do the arithmetic at
    all, and what it says it intends to do before it starts moving.
    """
    cost_unit, _ = _units(presentation.framing)
    return (
        f"{render_world(presentation)}\n\n"
        "You are at START. Plan your complete route to GOAL now.\n\n"
        "Think it through, then end your reply with exactly two lines:\n"
        "PLAN: <action ids in order, comma separated>\n"
        f"TOTAL: <the total {cost_unit} of that route, as a number>"
    )


def render_step_prompt(
    presentation: Presentation,
    node: str,
    taken: list[Arc],
) -> str:
    """Condition 2: commit to one action from where the agent now stands.

    The agent still sees the entire environment; only the question is local. Any
    gap between this trajectory and the stated plan is a revision of intent, which
    is what time inconsistency means.
    """
    instance = presentation.instance
    cost_unit, _ = _units(presentation.framing)
    options = [arc for arc in presentation.arc_order if arc[0] == node]

    history = "You have not acted yet."
    if taken:
        spent = sum(instance.weight(arc) for arc in taken)
        moves = ", ".join(presentation.arc_label[arc] for arc in taken)
        history = (
            f"So far you have taken: {moves}, "
            f"spending {_amount(spent, presentation.framing)}."
        )

    lines = [
        render_world(presentation),
        "",
        f"You are now at {presentation.label_of(node)}. {history}",
        "",
        "Your options from here:",
    ]
    for arc in options:
        lines.append(
            f"  {presentation.arc_label[arc]}: to {presentation.label_of(arc[1])}, "
            f"{_amount(instance.weight(arc), presentation.framing)}"
        )
    lines += [
        "  STOP: abandon the project and take nothing.",
        "",
        "Choose exactly one action to take now.",
        "",
        "Think it through, then end your reply with exactly one line:",
        "MOVE: <a single action id, or STOP>",
    ]
    return "\n".join(lines)


# Underscore is deliberately not stripped: it is legal inside an identifier, and
# removing it would split a label like L0_0 into two tokens.
_EMPHASIS = re.compile(r"[*`#>\s]+")


def _normalise(reply: str) -> str:
    """Strip markdown decoration from the edges of every line before matching.

    Models answer the format correctly and then dress it up: `**MOVE: A5**` is a
    compliant answer wearing bold. Reading that as a refusal would drop a quarter
    of one model's episodes and, worse, drop them selectively - the surviving
    subsample is whatever the model chose not to emphasise, which is not a random
    slice of anything.
    """
    return "\n".join(_EMPHASIS.sub(" ", line).strip() for line in reply.splitlines())


_PLAN_PATTERN = re.compile(r"^PLAN\s*:\s*(.+?)$", re.MULTILINE | re.IGNORECASE)
_TOTAL_PATTERN = re.compile(r"^TOTAL\s*:\s*(-?[\d.]+)", re.MULTILINE | re.IGNORECASE)
_MOVE_PATTERN = re.compile(r"^MOVE\s*:\s*([A-Za-z0-9_]+)", re.MULTILINE | re.IGNORECASE)


def parse_plan(reply: str, presentation: Presentation) -> tuple[list[Arc] | None, float | None]:
    """Read the PLAN and TOTAL lines. The last occurrence of each wins."""
    reply = _normalise(reply)
    plan_matches = _PLAN_PATTERN.findall(reply)
    total_matches = _TOTAL_PATTERN.findall(reply)

    route: list[Arc] | None = None
    if plan_matches:
        labels = [part.strip() for part in re.split(r"[,\s>-]+", plan_matches[-1]) if part.strip()]
        resolved = [presentation.resolve(label) for label in labels]
        route = None if any(arc is None for arc in resolved) else resolved  # type: ignore[assignment]

    total = float(total_matches[-1]) if total_matches else None
    return route, total


def parse_move(reply: str, presentation: Presentation) -> Arc | str | None:
    """Read the MOVE line. Returns an arc, the string "STOP", or None if unparseable."""
    matches = _MOVE_PATTERN.findall(_normalise(reply))
    if not matches:
        return None
    token = matches[-1].strip().upper()
    if token == "STOP":
        return "STOP"
    return presentation.resolve(token)
