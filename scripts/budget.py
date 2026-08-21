"""What the agent experiments cost, from measured prompt sizes and live prices.

Prompt lengths are not guessed: they come from rendering the actual harness on the
actual instances. The only real unknown is how much the model writes before it
answers, so that is a parameter with two scenarios rather than a single number.

    .venv/bin/python scripts/budget.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OPENROUTER_FEE = 0.055
"""OpenRouter adds a credit fee on top of provider rates."""


@dataclass(frozen=True)
class Model:
    name: str
    input_per_m: float
    output_per_m: float
    reasoning: bool = False
    note: str = ""


# Rates in USD per million tokens, read from the OpenRouter model list.
MODELS = [
    Model("claude-opus-5", 5.00, 25.00, True, "frontier reference"),
    Model("gpt-5.6-sol", 2.50, 15.00, True, "frontier, other lab"),
    Model("gpt-5.6-terra", 2.00, 12.00, True, "mid frontier"),
    Model("qwen3.8-max", 2.00, 6.00, False, "open-weight frontier"),
    Model("deepseek-v4-pro", 1.19, 3.56, False, "open-weight, cheap output"),
    Model("gemini-3.7-flash", 0.375, 1.875, False, "cheap, high volume"),
    Model("gpt-5.6-luna", 0.20, 1.20, False, "small model, weak arithmetic expected"),
    Model("qwen3.8-27b", 0.45, 3.20, False, "small open weight"),
    Model("deepseek-v4-flash", 0.08, 0.18, False, "cheapest usable"),
]


@dataclass(frozen=True)
class Experiment:
    name: str
    episodes: int
    calls_per_episode: int
    input_tokens: int
    """Measured by rendering the real prompt."""

    what: str

    @property
    def calls(self) -> int:
        return self.episodes * self.calls_per_episode


# Prompt sizes measured with tiktoken on the rendered harness output:
# synthetic traps run about 200 tokens a prompt, Caltech closures 300-550.
E1 = Experiment(
    name="E1 probe",
    episodes=5 * 2 * 20,       # 5 calibrated thresholds x 2 framings x 20 repetitions
    calls_per_episode=3,       # one plan call, then p = 2 committed steps
    input_tokens=210,
    what="does the agent discount, and what is its beta",
)

E4_FULL = Experiment(
    name="E4 intervention (full)",
    episodes=20 * 3 * 30,      # 20 instances x {none, greedy, optimal} x 30 rollouts
    calls_per_episode=5,       # p is 4-5 on Caltech closures
    input_tokens=400,
    what="do theory-derived edits steer the agent, across the sampling distribution",
)

E4 = Experiment(
    name="E4 intervention (lean)",
    episodes=12 * 3 * 20,      # fewer instances, 20 rollouts still separates 100% from 70%
    calls_per_episode=4,       # prefer closures with p = 4
    input_tokens=380,
    what="same claim, sized to what the effect actually needs",
)

# Reasoning effort is an experimental variable here, not an overhead to minimise:
# if myopia comes from bounded lookahead, beta should climb with the budget. So E1
# is run at three levels on the models that expose the dial.
E1_EFFORT_LEVELS = 3

OUTPUT_SCENARIOS = {"brief": 150, "reasoning": 800}


def cost(model: Model, experiment: Experiment, output_tokens: int) -> float:
    calls = experiment.calls
    spend = (
        calls * experiment.input_tokens * model.input_per_m
        + calls * output_tokens * model.output_per_m
    ) / 1_000_000
    return spend * (1 + OPENROUTER_FEE)


def main() -> None:
    print("Measured prompt sizes: synthetic traps ~210 tokens, Caltech closures ~400.")
    print(f"OpenRouter credit fee of {OPENROUTER_FEE:.1%} included in every figure.\n")

    for experiment in (E1, E4, E4_FULL):
        print(f"{experiment.name}: {experiment.calls:,} calls per model - {experiment.what}")
    print()

    header = f"{'model':<20}{'E1 brief':>10}{'E1 reason':>11}{'E4 brief':>11}{'E4 reason':>11}{'both':>10}"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        e1_brief = cost(model, E1, OUTPUT_SCENARIOS["brief"])
        e1_reason = cost(model, E1, OUTPUT_SCENARIOS["reasoning"])
        e4_brief = cost(model, E4, OUTPUT_SCENARIOS["brief"])
        e4_reason = cost(model, E4, OUTPUT_SCENARIOS["reasoning"])
        worst = e1_reason + e4_reason if model.reasoning else e1_brief + e4_brief
        print(
            f"{model.name:<20}{e1_brief:>10.2f}{e1_reason:>11.2f}"
            f"{e4_brief:>11.2f}{e4_reason:>11.2f}{worst:>10.2f}"
        )

    def scenario(model: Model) -> int:
        return OUTPUT_SCENARIOS["reasoning" if model.reasoning else "brief"]

    print()
    print("Recommended plan")
    print("-" * 58)

    wide = sum(cost(m, E1, scenario(m)) for m in MODELS)
    print(f"  E1 across all {len(MODELS)} models{'':22}${wide:7.2f}")

    dial = [m for m in MODELS if m.reasoning]
    effort = sum(cost(m, E1, scenario(m)) * (E1_EFFORT_LEVELS - 1) for m in dial)
    print(f"  E1 reasoning-effort sweep on {len(dial)} models{'':10}${effort:7.2f}")

    deep_models = {"claude-opus-5", "gemini-3.7-flash"}
    deep = sum(cost(m, E4, scenario(m)) for m in MODELS if m.name in deep_models)
    print(f"  E4 lean on opus-5 + gemini-flash{'':14}${deep:7.2f}")

    subtotal = wide + effort + deep
    print(f"  {'subtotal':<44}${subtotal:7.2f}")
    print(f"  {'+40% for pilots, reruns, failed parses':<44}${subtotal * 0.4:7.2f}")
    print(f"  {'PLANNING FIGURE':<44}${subtotal * 1.4:7.2f}")


if __name__ == "__main__":
    main()
