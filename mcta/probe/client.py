"""Talking to models through OpenRouter, with the money accounted for.

Three things here matter more than the HTTP.

*A hard spending cap.* The budget for these experiments is tens of dollars and a
loop that retries forever would eat it silently. The client tracks what it has
spent from the token counts the API returns and refuses to make a call that would
cross the ceiling, rather than discovering the overrun afterwards.

*A transcript of every call.* Each request and reply is appended to a JSONL file
with its usage and cost. Analysis then runs off the transcript rather than off the
network, so re-scoring an experiment - fitting beta differently, re-parsing a
reply - costs nothing and changes no numbers.

*Replies are data.* Model output is parsed for a move, never executed and never
followed as instruction. The harness only ever reads a MOVE or PLAN line out of it.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_FEE = 0.055

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: Path | str = ENV_FILE) -> None:
    """Read KEY=value lines from `.env` into the environment, if it exists.

    Saves the commonest setup failure, which is exporting a key in one shell and
    running the experiment from another. A variable already set in the environment
    always wins, so an explicit export still overrides the file.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


class BudgetExceeded(RuntimeError):
    """Raised instead of making a call that would cross the spending ceiling."""


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens, as listed by OpenRouter."""

    input_per_m: float
    output_per_m: float

    def charge(self, prompt_tokens: int, completion_tokens: int) -> float:
        raw = (prompt_tokens * self.input_per_m + completion_tokens * self.output_per_m) / 1_000_000
        return raw * (1 + OPENROUTER_FEE)


@dataclass
class Call:
    """One request and what it cost, as written to the transcript."""

    model: str
    prompt: str
    reply: str
    temperature: float
    effort: str | None
    sample: int
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency_ms: float
    reasoning_tokens: int = 0
    """Thinking the model did before answering. Zero for models without the dial.

    This is where reasoning effort shows up: the visible reply can be identical at
    low and high effort while the thinking behind it doubles, so measuring the
    answer's length says nothing about whether the dial was honoured.
    """

    estimated: bool = False
    """True when the charge is our own arithmetic rather than the API's own figure."""

    finish_reason: str = ""
    """Why generation stopped. "length" means the reply was cut off mid-thought.

    A model that spends its whole output budget reasoning and never reaches the
    answer line looks exactly like a model that answered unparseably, and the two
    mean opposite things: one is a harness setting to raise, the other is a fact
    about the agent. Recording this keeps them apart.
    """

    tag: str = ""
    error: str = ""


@dataclass
class Ledger:
    """Running account of an experiment's spending."""

    cap: float
    spent: float = 0.0
    calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)

    def would_exceed(self, estimate: float) -> bool:
        return self.spent + estimate > self.cap

    def record(self, model: str, cost: float) -> None:
        self.spent += cost
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0.0) + cost

    def summary(self) -> str:
        parts = ", ".join(f"{m} ${c:.2f}" for m, c in sorted(self.by_model.items()))
        return f"{self.calls} calls, ${self.spent:.2f} of ${self.cap:.2f} ({parts})"


class Transcript:
    """Append-only JSONL log of every call, and a reader for re-analysis."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, call: Call) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(call), ensure_ascii=False) + "\n")

    def read(self) -> list[Call]:
        if not self.path.exists():
            return []
        return [Call(**json.loads(line)) for line in self.path.read_text().splitlines() if line]

    def replay_index(self) -> dict[tuple, str]:
        """Map a call's identity to its reply, so a rerun costs nothing.

        Keyed on the sample index as well as the prompt, because E4 deliberately
        draws repeated samples at temperature above zero and they must not collapse
        into one cached answer.

        Replies cut off at the token ceiling are deliberately excluded. They carry
        no error and plenty of text, so replaying them would hand back a half-finished
        thought as though it were an answer - and would make raising the ceiling
        useless, since the rerun would serve the same truncation from the log. Their
        absence is what lets a rerun retry exactly the calls that failed.
        """
        return {
            (call.model, call.prompt, call.temperature, call.effort, call.sample): call.reply
            for call in self.read()
            if not call.error and call.finish_reason != "length"
        }


class OpenRouter:
    """A responder backed by OpenRouter, usable anywhere the harness takes one."""

    def __init__(
        self,
        model: str,
        pricing: Pricing,
        ledger: Ledger,
        transcript: Transcript,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        effort: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        retries: int = 4,
        replay: bool = True,
        transport=None,
    ) -> None:
        self.model = model
        self.pricing = pricing
        self.ledger = ledger
        self.transcript = transcript
        if api_key is None:
            load_env()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.sample = 0
        self.tag = ""
        self.last_finish_reason = ""
        """Why the most recent call stopped. Callers read this to tell a reply
        cut off mid-thought from one that ignored the format."""
        # Record whether we will really hit the network as a flag: comparing bound
        # methods with `is` is always false, since each attribute access makes a
        # fresh object, which would silently disable the missing-key check.
        self._uses_network = transport is None
        self._transport = transport or self._post
        self._replay = transcript.replay_index() if replay else {}

    def __call__(self, prompt: str) -> str:
        key = (self.model, prompt, self.temperature, self.effort, self.sample)
        if key in self._replay:
            self.last_finish_reason = "replayed"
            return self._replay[key]

        if not self.api_key and self._uses_network:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in .env at the repository "
                f"root ({ENV_FILE}) as OPENROUTER_API_KEY=sk-or-v1-... , or export "
                "it in this shell. See SETUP.md."
            )

        # Charge the worst case against the cap before spending it, so a run stops
        # before the ceiling rather than just after.
        worst_case = self.pricing.charge(len(prompt) // 3, self.max_tokens)
        if self.ledger.would_exceed(worst_case):
            raise BudgetExceeded(
                f"next call could cost ${worst_case:.4f}; {self.ledger.summary()}"
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.effort:
            payload["reasoning"] = {"effort": self.effort}

        started = time.perf_counter()
        reply, usage, error = self._attempt(payload)
        latency = (time.perf_counter() - started) * 1000

        # OpenRouter reports what the call actually cost. Prefer that over our own
        # arithmetic, which depends on a price table that drifts; fall back to the
        # estimate only when the field is absent.
        reported = usage.get("cost")
        estimated = reported is None
        self.last_finish_reason = usage.get("_finish_reason", "")
        cost = (
            self.pricing.charge(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            if estimated
            else float(reported) * (1 + OPENROUTER_FEE)
        )
        self.ledger.record(self.model, cost)
        self.transcript.append(
            Call(
                model=self.model,
                prompt=prompt,
                reply=reply,
                temperature=self.temperature,
                effort=self.effort,
                sample=self.sample,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                cost=cost,
                latency_ms=latency,
                reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                ),
                estimated=estimated,
                finish_reason=usage.get("_finish_reason", ""),
                tag=self.tag,
                error=error,
            )
        )
        return reply

    def _attempt(self, payload: dict) -> tuple[str, dict, str]:
        delay = 1.0
        last = ""
        for attempt in range(self.retries):
            try:
                body = self._transport(payload)
            except Exception as exc:  # network, rate limit, malformed response
                last = f"{type(exc).__name__}: {exc}"
                if attempt == self.retries - 1:
                    break
                time.sleep(delay)
                delay *= 2
                continue

            choices = body.get("choices") or []
            if not choices:
                last = f"no choices in response: {str(body)[:200]}"
                time.sleep(delay)
                delay *= 2
                continue
            reason = choices[0].get("finish_reason") or ""
            usage = dict(body.get("usage", {}))
            usage["_finish_reason"] = reason
            return choices[0]["message"]["content"] or "", usage, ""

        return "", {}, last

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/mcta-bench",
                "X-Title": "MCTA-Bench",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


def probe_capabilities(model: str, pricing: Pricing, ledger: Ledger, transcript: Transcript,
                       **kwargs) -> dict:
    """Check what a model will actually honour before an experiment depends on it.

    E4 measures a fraction over the sampling distribution, so it needs a model that
    respects temperature; the effort sweep needs one that respects the reasoning
    dial. Both are worth discovering for a few cents rather than midway through a
    run whose results then mean nothing.
    """
    findings: dict[str, object] = {"model": model}
    question = "Reply with exactly one word: a random colour."

    hot = OpenRouter(model, pricing, ledger, transcript, temperature=1.0, **kwargs)
    replies = set()
    for index in range(4):
        hot.sample = index
        hot.tag = "capability:temperature"
        replies.add(hot(question).strip().lower()[:24])
    findings["varies_with_temperature"] = len(replies) > 1
    findings["temperature_samples"] = sorted(replies)

    # The dial has to be probed with something worth thinking about, and measured on
    # the thinking rather than the answer: a model asked for a single number replies
    # with a single number at every effort, while the reasoning behind it varies.
    puzzle = (
        "A train leaves A at 14:00 travelling 80 km/h. Another leaves B at 14:30 "
        "travelling 100 km/h toward A. A and B are 400 km apart. At what clock time "
        "do they meet? Give only the time."
    )
    for effort in ("low", "high"):
        agent = OpenRouter(model, pricing, ledger, transcript, temperature=0.0,
                           effort=effort, **kwargs)
        agent.tag = f"capability:effort:{effort}"
        agent(puzzle)

    spend = [c for c in transcript.read() if c.tag.startswith("capability:effort")]
    if len(spend) >= 2:
        low, high = spend[-2], spend[-1]
        findings["effort_tokens"] = {
            "low": {"reasoning": low.reasoning_tokens, "completion": low.completion_tokens},
            "high": {"reasoning": high.reasoning_tokens, "completion": high.completion_tokens},
        }
        findings["honours_effort"] = high.reasoning_tokens != low.reasoning_tokens
        findings["reports_reasoning_tokens"] = bool(low.reasoning_tokens or high.reasoning_tokens)

    return findings
