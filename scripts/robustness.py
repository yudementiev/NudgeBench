"""How many episodes hold up each bin's beta bound?

An intersected interval is only as trustworthy as the episodes forcing its edges.
One deviant episode can pull an upper bound below 1 and make a whole bin look
biased, which is the failure mode that already cost this project one wrong
conclusion. This re-scores the complexity sweep from its transcript - no new calls
- and counts, per bin, how many episodes individually rule out patience.

    .venv/bin/python scripts/robustness.py
"""

from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.probe import (  # noqa: E402
    Ledger, OpenRouter, Transcript, make_presentation, run_episode,
)
from scripts.e1 import PRICES  # noqa: E402
from scripts.e1_complexity import BINS, branching, collect  # noqa: E402

MODEL = "google/gemini-3.7-flash"
TRANSCRIPT = Path("runs/e1complexity-google-gemini-3.7-flash-uncertain.jsonl")


def main() -> int:
    # A zero cap proves this is pure replay: any real call would raise.
    ledger = Ledger(cap=0.0)
    agent = OpenRouter(MODEL, PRICES[MODEL], ledger, Transcript(TRANSCRIPT), max_tokens=8000)
    pool = collect(8)

    print(f"{'bin':<7}{'n':>4}{'rule out beta=1':>17}{'strictest':>11}"
          f"{'2nd':>8}{'non-optimal':>13}  instances forcing the bound")
    print("-" * 96)

    for name, _, _ in BINS:
        cases = pool[name]
        biased, uppers, suboptimal = [], [], 0
        for index, (case, instance) in enumerate(cases):
            for repetition in range(2):
                seed = zlib.crc32(f"{name}:{index}:{repetition}".encode()) % 100_000
                agent.sample = seed
                episode = run_episode(
                    instance,
                    make_presentation(instance, random.Random(seed), framing="uncertain"),
                    agent,
                )
                fit = episode.beta_fit()
                if not episode.chose_optimal_route:
                    suboptimal += 1
                if fit.feasible and fit.upper < 0.999:
                    biased.append((case.split(":")[-1], round(fit.upper, 3),
                                   branching(instance)))
                    uppers.append(fit.upper)

        uppers.sort()
        strictest = f"{uppers[0]:.3f}" if uppers else "-"
        second = f"{uppers[1]:.3f}" if len(uppers) > 1 else "-"
        forcing = ", ".join(f"{c}({b}br,<={u})" for c, u, b in sorted(biased, key=lambda x: x[1])[:3])
        print(f"{name:<7}{2 * len(cases):>4}{len(biased):>17}{strictest:>11}"
              f"{second:>8}{suboptimal:>13}  {forcing}")

    print(f"\npaid calls during replay: {ledger.calls}")
    print("\nA bound resting on one episode is an anecdote; on several, a measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
