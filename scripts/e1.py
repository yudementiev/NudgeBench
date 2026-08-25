"""E1: does an agent making local decisions plan like a present-biased one?

Sweeps traps calibrated across an empirically attested band of beta, in two
framings, and reads the agent's own beta off where it stops falling for them.

    .venv/bin/python scripts/e1.py --dry-run          # whole pipeline, no API
    .venv/bin/python scripts/e1.py --pilot --model X  # a few dollars
    .venv/bin/python scripts/e1.py --model X --effort low,medium,high

--dry-run drives the reference agent at a beta we choose, so a green dry run means
the sweep, the parsing and the intersection all work and the only untested thing
left is the model itself.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.generate import BETA_BANDS, deep_sweep, integer_sweep  # noqa: E402
from mcta.probe import (  # noqa: E402
    FRAMINGS,
    Ledger,
    OpenRouter,
    Pricing,
    ReferenceAgent,
    Transcript,
    make_presentation,
    run_episode,
)
from mcta.probe.analyze import Cell  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"

# USD per million tokens, with ids taken verbatim from the OpenRouter model list -
# a near-miss id fails the run on its first call. Verify against
# https://openrouter.ai/api/v1/models before a campaign; ids carry date suffixes
# that change.
PRICES = {
    "anthropic/claude-opus-5": Pricing(5.00, 25.00),
    "openai/gpt-5.6-sol": Pricing(2.00, 10.00),
    "openai/gpt-5.6-terra": Pricing(2.00, 12.00),
    "openai/gpt-5.6-luna": Pricing(0.20, 1.20),
    "google/gemini-3.7-flash": Pricing(0.375, 1.875),
    "google/gemini-3.6-flash": Pricing(0.75, 3.75),
    "qwen/qwen3.8-max": Pricing(2.00, 6.00),
    "qwen/qwen3.8-27b": Pricing(0.40, 3.00),
    "deepseek/deepseek-v4-pro-20260813": Pricing(1.122, 3.366),
    "deepseek/deepseek-v4-flash-0731": Pricing(0.14, 0.28),
}

# Models confirmed to give a usable sampling distribution for E4, which measures a
# fraction over one. The GPT-5.6 family does not list `temperature` among its
# supported parameters yet does vary between samples, so it is stochastic without
# the width being controllable - excluded here out of caution rather than because
# it was shown not to work. Verify with probe_capabilities before relying on any of
# these; an earlier four-sample probe called sol deterministic purely because short
# answers collided.
SAMPLING_MODELS = {
    "anthropic/claude-opus-5",
    "google/gemini-3.7-flash",
    "google/gemini-3.6-flash",
    "qwen/qwen3.8-max",
    "qwen/qwen3.8-27b",
    "deepseek/deepseek-v4-pro-20260813",
    "deepseek/deepseek-v4-flash-0731",
}


def conditions(thresholds: int, repetitions: int, framings: tuple[str, ...],
               band: str = "effort", depth: int = 1):
    """Every (trap, framing, repetition) the sweep calls for."""
    # Whole-number weights: the arithmetic control must not be confounded by
    # asking a model to total awkward decimals. Depth beyond 1 makes evaluating
    # the future cost something, which is where a bounded horizon would show.
    traps = (integer_sweep(band, count=thresholds) if depth <= 1
             else deep_sweep(band, count=thresholds, depth=depth))
    for trap in traps:
        for framing in framings:
            for repetition in range(repetitions):
                yield trap, framing, repetition


def run_cell(label: str, responder, thresholds: int, repetitions: int,
             framings: tuple[str, ...], seed: int = 0, band: str = "effort",
             depth: int = 1) -> Cell:
    cell = Cell(label=label)
    for index, (trap, framing, repetition) in enumerate(
        conditions(thresholds, repetitions, framings, band, depth)
    ):
        instance = trap.instance
        # A fresh presentation per episode: labels and option order are re-randomised
        # so neither can be what the agent is responding to.
        presentation = make_presentation(
            instance, random.Random(seed * 10_000 + index), framing=framing
        )
        if isinstance(responder, OpenRouter):
            responder.sample = index
            responder.tag = f"e1:{label}:th{trap.spec.threshold:.3f}:{framing}:r{repetition}"
        cell.add(run_episode(instance, presentation, responder))
    return cell


def dry_run(args) -> int:
    """Drive the whole experiment with an agent whose beta we already know."""
    print("Dry run: reference agent, no API calls.\n")
    print(Cell.header())

    ok = True
    for beta in (0.60, 0.7375, 0.95):
        cell = run_cell(
            f"reference beta={beta}", ReferenceAgent(beta),
            args.thresholds, args.repetitions, tuple(args.framings), band=args.band, depth=args.depth,
        )
        print(cell.row())
        interval = cell.beta()
        if not interval.consistent:
            print(f"    FAILED: {interval.describe()}")
            ok = False
        elif not interval.lower - 1e-6 <= beta <= interval.upper + 1e-6:
            print(f"    FAILED: recovered interval excludes the driving beta {beta}")
            ok = False

    print(f"\nTraps calibrated across the {args.band} band {BETA_BANDS[args.band]}.")
    print("Pipeline recovers the beta it was driven with." if ok else "\nPIPELINE BROKEN.")
    return 0 if ok else 1


def live_run(args) -> int:
    if args.model not in PRICES:
        print(f"No price on file for {args.model}. Known: {', '.join(sorted(PRICES))}")
        return 2

    RUNS.mkdir(exist_ok=True)
    tag = f"{args.model.replace('/', '-')}-{args.band}-d{args.depth}"
    transcript = Transcript(RUNS / f'e1-{tag}.jsonl')
    ledger = Ledger(cap=args.cap)

    print(f"Model {args.model}, cap ${args.cap:.2f}, transcript {transcript.path.name}\n")
    print(Cell.header())

    results = {}
    for effort in args.effort:
        responder = OpenRouter(
            args.model, PRICES[args.model], ledger, transcript,
            temperature=args.temperature, effort=None if effort == "none" else effort,
        )
        cell = run_cell(
            f"{args.model.split('/')[-1]} effort={effort}", responder,
            args.thresholds, args.repetitions, tuple(args.framings), band=args.band, depth=args.depth,
        )
        print(cell.row())
        interval = cell.beta()
        results[effort] = {
            "episodes": len(cell.episodes),
            "beta_lower": interval.lower,
            "beta_upper": interval.upper,
            "consistent": interval.consistent,
            "conflicts": interval.conflicts[:10],
            "arithmetic_rate": cell.arithmetic_rate,
            "divergence_rate": cell.divergence_rate,
            "optimal_rate": cell.optimal_rate,
            "parse_failure_rate": cell.parse_failure_rate,
        }

    out = RUNS / f'e1-{tag}.json'
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{ledger.summary()}")
    print(f"Results in {out}")

    if len(args.effort) > 1:
        print("\nEffort sweep - does the discount recede as the agent is given more room?")
        for effort in args.effort:
            row = results[effort]
            if row["consistent"]:
                print(f"  {effort:<8} beta in [{row['beta_lower']:.3f}, {row['beta_upper']:.3f}]")
            else:
                print(f"  {effort:<8} no single beta fits")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="reference agent, no API")
    parser.add_argument("--pilot", action="store_true", help="a handful of episodes")
    parser.add_argument("--model", default="google/gemini-3.7-flash")
    parser.add_argument("--effort", default="none",
                        help="comma separated: none,low,medium,high")
    parser.add_argument("--depth", type=int, default=1,
                        help="arcs per route; >1 makes lookahead expensive")
    parser.add_argument("--band", default="effort",
                        help="effort | non_monetary | money | agentic | wide")
    parser.add_argument("--thresholds", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--framings", default=",".join(FRAMINGS))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cap", type=float, default=10.0, help="hard spending ceiling in USD")
    args = parser.parse_args()

    args.effort = [e.strip() for e in args.effort.split(",") if e.strip()]
    args.framings = [f.strip() for f in args.framings.split(",") if f.strip()]
    if args.pilot:
        args.thresholds, args.repetitions = 2, 2
        args.cap = min(args.cap, 3.0)

    return dry_run(args) if args.dry_run else live_run(args)


if __name__ == "__main__":
    sys.exit(main())
