"""Does the agent's discount factor depend on how hard the instance is?

The corpus probe found something the pooled number hid: episodes on larger,
busier graphs fitted a discount strictly below 1, while episodes on small ones
fitted patience, and the two were mutually incompatible. That split was noticed
only because those same episodes had earlier been lost to a token limit - which
means the effect and the truncation share a cause, namely that the agent has more
to think about.

This makes that cause the independent variable. Instances are binned by the number
of nodes offering a choice, since that is what a planning horizon actually has to
span, and beta is recovered per bin.

Two things have to be watched or the measurement repeats the earlier mistake.
Truncation is reported per bin: if the largest instances lose episodes, they lose
exactly the ones carrying the effect, and the bin's beta is then fitted to whatever
survived. And the arithmetic control is reported per bin too, because an agent that
stops adding correctly on big graphs is a different story from one that discounts.

    .venv/bin/python scripts/e1_complexity.py --dry-run
    .venv/bin/python scripts/e1_complexity.py --model X --per-bin 8 --cap 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.corpus.cpn import candidate_targets, load_caltech, load_jhu, to_instance  # noqa: E402
from mcta.model import Instance  # noqa: E402
from mcta.probe import (  # noqa: E402
    Ledger, OpenRouter, ReferenceAgent, Transcript, make_presentation, run_episode,
)
from mcta.probe.analyze import Cell  # noqa: E402
from scripts.e1 import PRICES  # noqa: E402
from scripts.e1_corpus import PALETTE, discriminating  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"

# Choice points, not arcs: an instance with many arcs but one route poses no
# planning problem at all.
BINS = [("1-3", 1, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11+", 11, 999)]


def branching(instance: Instance) -> int:
    return sum(1 for v in instance.graph if instance.graph.out_degree(v) > 1)


def collect(per_bin: int, seed: int = 0) -> dict[str, list[tuple[str, Instance]]]:
    """Discriminating instances, evenly spread across the complexity bins."""
    rng = random.Random(seed)
    pool: dict[str, list[tuple[str, Instance]]] = {name: [] for name, _, _ in BINS}

    for loader, tag in ((load_caltech, "caltech"), (load_jhu, "jhu")):
        cpn = loader()
        for target in candidate_targets(cpn, min_courses=6, min_depth=3):
            charges = {course: rng.choice(PALETTE) for course in cpn}
            instance = to_instance(cpn, target, weight_of=charges.__getitem__, beta=0.75)
            if not 8 <= instance.graph.number_of_edges() <= 90:
                continue
            if not discriminating(instance):
                continue
            width = branching(instance)
            for name, low, high in BINS:
                if low <= width <= high and len(pool[name]) < per_bin:
                    pool[name].append((f"{tag}:{target}", instance))
                    break
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--per-bin", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--framing", default="uncertain")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--cap", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=0.75)
    args = parser.parse_args()

    pool = collect(args.per_bin)
    print(f"{sum(len(v) for v in pool.values())} instances, framing={args.framing}\n")

    if args.dry_run:
        responder, ledger, tag = ReferenceAgent(args.beta), None, "dry"
    else:
        RUNS.mkdir(exist_ok=True)
        ledger = Ledger(cap=args.cap)
        tag = f"{args.model.replace('/', '-')}-{args.framing}"
        responder = OpenRouter(args.model, PRICES[args.model], ledger,
                               Transcript(RUNS / f"e1complexity-{tag}.jsonl"),
                               max_tokens=args.max_tokens)

    print(f"{'bin':<7}{'inst':>5}{'arcs':>9}{'n':>5}{'arith':>8}{'optimal':>9}"
          f"{'trunc':>7}  beta")
    print("-" * 78)

    results = {}
    for name, _, _ in BINS:
        cases = pool[name]
        if not cases:
            continue
        cell = Cell(label=name)
        for index, (case, instance) in enumerate(cases):
            for repetition in range(args.repetitions):
                # zlib.crc32, not hash(): the builtin is randomised per process,
                # so seeds - and therefore prompts - would differ between runs.
                seed = zlib.crc32(f"{name}:{index}:{repetition}".encode()) % 100_000
                if isinstance(responder, OpenRouter):
                    responder.sample = seed
                    responder.tag = f"e1cx:{name}:{case}:r{repetition}"
                cell.add(run_episode(
                    instance,
                    make_presentation(instance, random.Random(seed), framing=args.framing),
                    responder,
                ))
        arcs = [i.graph.number_of_edges() for _, i in cases]
        interval = cell.beta()
        print(f"{name:<7}{len(cases):>5}{f'{min(arcs)}-{max(arcs)}':>9}{len(cell.episodes):>5}"
              f"{cell.arithmetic_rate:>8.0%}{cell.optimal_rate:>9.0%}"
              f"{cell.truncation_rate:>7.0%}  {interval.describe()}")
        results[name] = {
            "instances": len(cases),
            "arcs_min": min(arcs), "arcs_max": max(arcs),
            "episodes": len(cell.episodes),
            "beta_lower": interval.lower, "beta_upper": interval.upper,
            "consistent": interval.consistent,
            "arithmetic_rate": cell.arithmetic_rate,
            "optimal_rate": cell.optimal_rate,
            "truncation_rate": cell.truncation_rate,
            "parse_failure_rate": cell.parse_failure_rate,
        }

    if args.dry_run:
        print("\nReference agent: beta should come back the same in every bin.")
        return 0

    out = RUNS / f"e1complexity-{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{ledger.summary()}\nResults in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
