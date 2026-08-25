"""E4: do theory-derived edits actually steer a real agent?

Takes instances the agent walks past a required action, computes two repairs - the
one an engineer would make by hand and the cheapest one - applies each, and then
samples many rollouts at temperature above zero.

    .venv/bin/python scripts/e4.py --dry-run
    .venv/bin/python scripts/e4.py --model X --rollouts 20

The measurement is a *fraction of rollouts*, not one trajectory, and that is the
point. A present-biased agent at temperature is exactly the nondeterministic
tie-breaker the paper's formulation is written for: prompting shifts the mode of
its behavior, while editing the environment is supposed to constrain the whole
support. If the optimal edit succeeds on every rollout and the hand-made one only
on most, that difference is what the theory buys.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.corpus.cpn import candidate_targets, load_caltech, to_instance  # noqa: E402
from mcta.generate import costly_compliance  # noqa: E402
from mcta.model import Arc, Instance  # noqa: E402
from mcta.probe import (  # noqa: E402
    Ledger,
    OpenRouter,
    Pricing,
    ReferenceAgent,
    Transcript,
    make_presentation,
)
from mcta.probe.analyze import wilson  # noqa: E402
from mcta.probe.protocol import run_walk  # noqa: E402
from mcta.solve import greedy_witness_guided, minimalize, solve_exact  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e1 import PRICES, SAMPLING_MODELS  # noqa: E402  one price list
from scripts.e3 import pick_critical  # noqa: E402  reuse E3's selection exactly

RUNS = Path(__file__).resolve().parents[1] / "runs"


@dataclass
class Case:
    """One instance, the action required of it, and the two repairs."""

    name: str
    instance: Instance
    critical: set[Arc]
    greedy: frozenset[Arc]
    optimal: frozenset[Arc]
    greedy_cost: float
    optimal_cost: float

    def variants(self) -> dict[str, Instance]:
        return {
            "unedited": self.instance,
            "greedy": self.instance.without(set(self.greedy)),
            "optimal": self.instance.without(set(self.optimal)),
        }


def build_cases(limit: int) -> list[Case]:
    """Instances a present-biased agent fails, paired with both repairs.

    Synthetic ones first, because `costly_compliance` puts the required action on a
    route no agent takes unaided and so is guaranteed to need an intervention; then
    real curricula, which supply the shapes.
    """
    cases: list[Case] = []

    for index in range(2):
        generated = costly_compliance(steady_first=6 + index, steady_tail=12 + index)
        instance, critical = generated.instance, set(generated.spec.critical)
        cases.append(_make_case(f"costly_compliance_{index}", instance, critical))

    cpn = load_caltech()
    # Scan every target before selecting: stopping early would take whatever comes
    # first alphabetically rather than the cases that actually discriminate.
    for target in candidate_targets(cpn, min_courses=6, min_depth=3):
        instance = to_instance(cpn, target)
        if instance.graph.number_of_edges() > 22:
            continue
        # Same rule E3 used, so the instances where the heuristic was shown to
        # lose are the instances this experiment steers.
        critical = pick_critical(instance)
        if critical is None:
            continue
        case = _make_case(target, instance, critical)
        if case is not None:
            cases.append(case)

    # Prefer cases where the two repairs actually differ. Against a textbook
    # Kleinberg-Oren agent both are aligned by construction and differ only in
    # price, so any gap in *effectiveness* can only come from the real agent
    # departing from the model - which is what this experiment is here to find.
    cases = [c for c in cases if c is not None]
    cases.sort(key=lambda c: (c.greedy == c.optimal, -(c.greedy_cost - c.optimal_cost), c.name))
    return cases[:limit]


def _make_case(name: str, instance: Instance, critical: set[Arc]) -> Case | None:
    greedy = greedy_witness_guided(instance, critical)
    if greedy.aligned:
        greedy = minimalize(instance, critical, greedy.deleted)
    exact = solve_exact(instance, critical, node_budget=60_000)
    if not greedy.aligned or not exact.aligned:
        return None
    return Case(
        name=name,
        instance=instance,
        critical=critical,
        greedy=greedy.deleted,
        optimal=exact.deleted,
        greedy_cost=greedy.cost,
        optimal_cost=exact.cost,
    )


def rollout_rate(instance: Instance, critical: set[Arc], responder, rollouts: int,
                 seed: int, tag: str) -> tuple[int, int]:
    """How many of `rollouts` traversed every critical arc."""
    hits = 0
    for index in range(rollouts):
        presentation = make_presentation(instance, random.Random(seed * 1000 + index))
        if isinstance(responder, OpenRouter):
            responder.sample = seed * 1000 + index
            responder.tag = tag
        episode = run_walk(instance, presentation, responder)
        if episode.reached_goal and critical <= set(episode.trajectory):
            hits += 1
    return hits, rollouts


def report(results: dict) -> None:
    print(f"\n{'condition':<12}{'hits':>8}{'rate':>9}{'95% CI':>18}")
    print("-" * 47)
    for condition in ("unedited", "greedy", "optimal"):
        hits, total = results[condition]
        low, high = wilson(hits, total)
        print(f"{condition:<12}{hits:>4}/{total:<3}{hits / total:>8.0%}"
              f"{f'[{low:.0%}, {high:.0%}]':>18}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--instances", type=int, default=12)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.75, help="dry run only")
    args = parser.parse_args()

    cases = build_cases(args.instances)
    print(f"{len(cases)} cases; greedy costs "
          f"{sum(c.greedy_cost for c in cases):g} against optimal "
          f"{sum(c.optimal_cost for c in cases):g} in total")

    if args.dry_run:
        responder = ReferenceAgent(args.beta)
        ledger = None
    else:
        if args.model not in PRICES:
            print(f"No price on file for {args.model}")
            return 2
        if args.model not in SAMPLING_MODELS and args.temperature > 0:
            print(f"{args.model} does not list temperature among its parameters.")
            print("E4 measures a fraction over the sampling distribution, so its")
            print("result would be meaningless here. Pick a model from:")
            for name in sorted(SAMPLING_MODELS):
                print(f"  {name}")
            return 2
        RUNS.mkdir(exist_ok=True)
        ledger = Ledger(cap=args.cap)
        responder = OpenRouter(
            args.model, PRICES[args.model], ledger,
            Transcript(RUNS / f"e4-{args.model.replace('/', '-')}.jsonl"),
            temperature=args.temperature,
        )

    totals = {c: [0, 0] for c in ("unedited", "greedy", "optimal")}
    per_case = {}
    for index, case in enumerate(cases):
        row = {}
        for condition, variant in case.variants().items():
            hits, total = rollout_rate(
                variant, case.critical, responder, args.rollouts, index,
                tag=f"e4:{case.name}:{condition}",
            )
            totals[condition][0] += hits
            totals[condition][1] += total
            row[condition] = [hits, total]
        per_case[case.name] = row

    report({k: tuple(v) for k, v in totals.items()})

    if args.dry_run:
        unedited, _ = totals["unedited"]
        optimal, optimal_total = totals["optimal"]
        print(f"\nReference agent at beta={args.beta}: the model predicts 0% unedited "
              f"and 100% under the optimal edit.")
        ok = unedited == 0 and optimal == optimal_total
        print("Pipeline behaves as the theory says." if ok else "PIPELINE DISAGREES WITH THEORY.")
        return 0 if ok else 1

    out = RUNS / f"e4-{args.model.replace('/', '-')}.json"
    out.write_text(json.dumps({"totals": totals, "per_case": per_case}, indent=2))
    print(f"\n{ledger.summary()}")
    print(f"Results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
