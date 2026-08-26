"""E5: does the agent defer a large step when deferring is cheap?

The configuration comes from the agent rather than from the theory. In the one
instance of the complexity sweep where a deviation appeared, the agent stood in
front of a 13-hour step to the goal, listed three 2-hour steps that each led to a
node offering the same 13-hour step again, and took one - "minimizes immediate
expenditure". It paid 15 instead of 13 and said why.

`deferral_trap` generalises that. The threshold is fixed by the structure,
beta < 1 - small/big, so sweeping the ratio sweeps the threshold. What a two-route
trap could not ask, this can: since the choice at the next node is identical, an
agent below the threshold defers again, and the number of deferrals measures how
far the procrastination actually runs.

    .venv/bin/python scripts/e5_deferral.py --dry-run
    .venv/bin/python scripts/e5_deferral.py --model X --cap 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.generate import deferral_trap  # noqa: E402
from mcta.model import path_weight  # noqa: E402
from mcta.probe import (  # noqa: E402
    Ledger, OpenRouter, ReferenceAgent, Transcript, make_presentation, run_episode,
)
from mcta.probe.analyze import Cell, wilson  # noqa: E402
from scripts.e1 import PRICES  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"

# (big, small) pairs whose thresholds 1 - small/big spread across the band where
# agents have so far turned out to sit. The second is the configuration observed.
LADDER = [(10, 2), (13, 2), (20, 2), (20, 1), (50, 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--framing", default="uncertain")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cap", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.75)
    args = parser.parse_args()

    if args.dry_run:
        responder, ledger, tag = ReferenceAgent(args.beta), None, "dry"
    else:
        RUNS.mkdir(exist_ok=True)
        ledger = Ledger(cap=args.cap)
        tag = f"{args.model.replace('/', '-')}-{args.framing}"
        responder = OpenRouter(args.model, PRICES[args.model], ledger,
                               Transcript(RUNS / f"e5-{tag}.jsonl"),
                               temperature=args.temperature, max_tokens=args.max_tokens)

    print(f"depth {args.depth}, {args.branches} ways to defer, "
          f"{args.repetitions} repetitions, framing={args.framing}\n")
    print(f"{'big/small':>10}{'threshold':>11}{'deferred':>10}{'rate':>7}"
          f"{'95% CI':>16}{'mean cost':>11}{'direct':>8}  beta")
    print("-" * 92)

    results = {}
    for big, small in LADDER:
        generated = deferral_trap(big=big, small=small, branches=args.branches,
                                  depth=args.depth, beta=args.beta)
        instance = generated.instance
        cell = Cell(label=f"{big}/{small}")
        deferrals, costs = 0, []
        for repetition in range(args.repetitions):
            seed = zlib.crc32(f"{big}:{small}:{repetition}".encode()) % 100_000
            if isinstance(responder, OpenRouter):
                responder.sample = seed
                responder.tag = f"e5:{big}-{small}:r{repetition}"
            episode = run_episode(
                instance,
                make_presentation(instance, random.Random(seed), framing=args.framing),
                responder,
            )
            cell.add(episode)
            if episode.reached_goal:
                # Every route ends with the big step, so extra steps are deferrals.
                deferrals += len(episode.trajectory) - 1
                costs.append(path_weight(instance, episode.trajectory))

        episodes = max(1, len(costs))
        rate = sum(1 for c in costs if c > big) / episodes
        low, high = wilson(sum(1 for c in costs if c > big), episodes)
        results[f"{big}/{small}"] = {
            "threshold": generated.spec.threshold,
            "episodes": len(cell.episodes),
            "reached_goal": len(costs),
            "deferral_episodes": sum(1 for c in costs if c > big),
            "total_deferrals": deferrals,
            "mean_cost": sum(costs) / episodes if costs else None,
            "direct_cost": big,
            "beta_lower": cell.beta().lower, "beta_upper": cell.beta().upper,
            "consistent": cell.beta().consistent,
            "arithmetic_rate": cell.arithmetic_rate,
            "truncation_rate": cell.truncation_rate,
        }
        print(f"{f'{big}/{small}':>10}{generated.spec.threshold:>11.3f}"
              f"{deferrals:>10}{rate:>7.0%}{f'[{low:.0%}, {high:.0%}]':>16}"
              f"{(sum(costs) / episodes if costs else 0):>11.1f}{big:>8}  "
              f"{cell.beta().describe()}")

    if args.dry_run:
        print(f"\nReference agent at beta={args.beta}: defers wherever "
              f"beta < 1 - small/big, and not otherwise.")
        return 0

    out = RUNS / f"e5-{tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{ledger.summary()}\nResults in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
