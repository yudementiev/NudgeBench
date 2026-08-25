"""E1 on real task graphs rather than synthetic traps.

The traps are four to twelve arcs and fit in a glance. Real curricula branch, so
choosing well means comparing routes rather than two sums - which is the regime
where a bounded horizon would show up if it exists.

Only *discriminating* instances are used: those where a patient agent and a biased
one walk different routes. On an instance where every route costs the same, the
agent reveals nothing and the episode measures nothing.

Weights come from a small integer palette. The course data carries no workload
column, so this is synthetic and stated as such; whole numbers keep the arithmetic
control clean.

    .venv/bin/python scripts/e1_corpus.py --dry-run
    .venv/bin/python scripts/e1_corpus.py --model X --instances 20 --cap 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.corpus.cpn import candidate_targets, load_caltech, load_jhu, to_instance  # noqa: E402
from mcta.model import Instance, analyze  # noqa: E402
from mcta.probe import (  # noqa: E402
    Ledger, OpenRouter, ReferenceAgent, Transcript, make_presentation, run_episode,
)
from mcta.probe.analyze import Cell  # noqa: E402
from scripts.e1 import PRICES  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
PALETTE = [2, 3, 5, 8, 13]


def weighted(cpn, target, rng: random.Random) -> Instance:
    charges = {c: rng.choice(PALETTE) for c in cpn}
    return to_instance(cpn, target, weight_of=charges.__getitem__, beta=0.75)


def discriminating(instance: Instance) -> bool:
    """Whether patience and bias would walk different routes here."""
    behavior = analyze(instance)
    if behavior.abandons_at(instance.source):
        return False
    patient = Instance(instance.graph, instance.source, instance.target, 1.0, instance.reward)
    biased = Instance(instance.graph, instance.source, instance.target, 0.4, instance.reward)
    a, b = analyze(patient).feasible_paths(limit=200), analyze(biased).feasible_paths(limit=200)
    return bool(a and b and a[0] != b[0])


def build(count: int, seed: int = 0) -> list[tuple[str, Instance]]:
    rng = random.Random(seed)
    picked: list[tuple[str, Instance]] = []
    for loader, name in ((load_caltech, "caltech"), (load_jhu, "jhu")):
        cpn = loader()
        for target in candidate_targets(cpn, min_courses=6, min_depth=3):
            if len(picked) >= count:
                return picked
            instance = weighted(cpn, target, rng)
            if 8 <= instance.graph.number_of_edges() <= 26 and discriminating(instance):
                picked.append((f"{name}:{target}", instance))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--cap", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.75)
    args = parser.parse_args()

    cases = build(args.instances)
    arcs = [i.graph.number_of_edges() for _, i in cases]
    print(f"{len(cases)} discriminating instances, {min(arcs)}-{max(arcs)} arcs\n")

    if args.dry_run:
        responder, ledger = ReferenceAgent(args.beta), None
        label = f"reference beta={args.beta}"
    else:
        RUNS.mkdir(exist_ok=True)
        ledger = Ledger(cap=args.cap)
        tag = args.model.replace("/", "-")
        responder = OpenRouter(args.model, PRICES[args.model], ledger,
                               Transcript(RUNS / f"e1corpus-{tag}.jsonl"))
        label = f"{args.model.split('/')[-1]} on corpus"

    cell = Cell(label=label)
    for index, (name, instance) in enumerate(cases):
        for repetition in range(args.repetitions):
            seed = index * 100 + repetition
            presentation = make_presentation(instance, random.Random(seed), framing="effort")
            if isinstance(responder, OpenRouter):
                responder.sample = seed
                responder.tag = f"e1corpus:{name}:r{repetition}"
            cell.add(run_episode(instance, presentation, responder))

    print(Cell.header())
    print(cell.row())

    if args.dry_run:
        interval = cell.beta()
        ok = interval.consistent and interval.lower - 1e-6 <= args.beta <= interval.upper + 1e-6
        print("\nPipeline recovers the beta it was driven with." if ok
              else f"\nFAILED: {interval.describe()}")
        return 0 if ok else 1

    interval = cell.beta()
    out = RUNS / f"e1corpus-{args.model.replace('/', '-')}.json"
    out.write_text(json.dumps({
        "instances": len(cases), "episodes": len(cell.episodes),
        "beta_lower": interval.lower, "beta_upper": interval.upper,
        "consistent": interval.consistent, "conflicts": interval.conflicts[:10],
        "arithmetic_rate": cell.arithmetic_rate, "divergence_rate": cell.divergence_rate,
        "optimal_rate": cell.optimal_rate, "parse_failure_rate": cell.parse_failure_rate,
    }, indent=2))
    print(f"\n{ledger.summary()}\nResults in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
