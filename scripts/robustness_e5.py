"""How many episodes hold up each deferral cell's bound?

Same question as scripts/robustness.py, asked of E5. An interval pinned exactly to
an instance's threshold is what a single deferring episode produces, so the cell's
number means nothing until we know how many episodes deferred and how many merely
failed to contradict them.

    .venv/bin/python scripts/robustness_e5.py
"""

from __future__ import annotations

import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcta.generate import deferral_trap  # noqa: E402
from mcta.model import path_weight  # noqa: E402
from mcta.probe import (  # noqa: E402
    Ledger, OpenRouter, Transcript, make_presentation, run_episode,
)
from scripts.e1 import PRICES  # noqa: E402
from scripts.e5_deferral import LADDER  # noqa: E402

DEFAULT_MODEL = "google/gemini-3.7-flash"


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    transcript = Path(f"runs/e5-{model.replace('/', '-')}-uncertain.jsonl")
    repetitions = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    # Replay keys include the model, so a mismatch here silently becomes a live run.
    ledger = Ledger(cap=0.0)
    agent = OpenRouter(model, PRICES[model], ledger, Transcript(transcript),
                       temperature=temperature, max_tokens=8000)
    print(f"{model}, {transcript.name}\n")

    print(f"{'cell':>8}{'threshold':>11}{'goal':>6}{'deferred':>10}{'>1 defer':>10}"
          f"{'max':>5}{'arith':>8}{'trunc':>7}{'unpars':>8}{'stopped':>9}")
    print("-" * 84)

    for big, small in LADDER:
        generated = deferral_trap(big=big, small=small, branches=3, depth=4, beta=0.75)
        instance = generated.instance
        counts, arith, trunc, scored, unparsed, abandoned = [], 0, 0, 0, 0, 0
        for repetition in range(repetitions):
            seed = zlib.crc32(f"{big}:{small}:{repetition}".encode()) % 100_000
            agent.sample = seed
            episode = run_episode(
                instance,
                make_presentation(instance, random.Random(seed), framing="uncertain"),
                agent,
            )
            if "truncated" in episode.parse_failure:
                trunc += 1
            if "unparseable" in episode.parse_failure:
                unparsed += 1
            abandoned += episode.abandoned
            if episode.arithmetic_correct is not None:
                scored += 1
                arith += bool(episode.arithmetic_correct)
            if episode.reached_goal:
                counts.append(len(episode.trajectory) - 1)

        deferred = sum(1 for c in counts if c > 0)
        repeated = sum(1 for c in counts if c > 1)
        print(f"{f'{big}/{small}':>8}{generated.spec.threshold:>11.3f}"
              f"{f'{len(counts)}/{repetitions}':>6}{deferred:>10}{repeated:>10}"
              f"{(max(counts) if counts else 0):>5}"
              f"{(arith / scored if scored else float('nan')):>8.0%}"
              f"{trunc / repetitions:>7.0%}{unparsed / repetitions:>8.0%}"
              f"{abandoned / repetitions:>9.0%}")

    print(f"\npaid calls during replay: {ledger.calls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
