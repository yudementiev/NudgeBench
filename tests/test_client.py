"""Validation of the OpenRouter client, with no network and no spending.

Every test drives a stub transport. What is checked is the machinery that protects
the budget and the record: that a run stops before the ceiling rather than after,
that repeated samples stay independent, and that the client drops into the harness
wherever a responder is expected.
"""

from __future__ import annotations

import pytest

from mcta.examples import procrastination_trap
from mcta.probe import make_presentation, run_episode
from mcta.probe.client import (
    BudgetExceeded,
    Call,
    Ledger,
    OpenRouter,
    Pricing,
    Transcript,
)

PRICING = Pricing(input_per_m=1.0, output_per_m=4.0)


def stub(reply: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """A transport that answers with fixed text and reports fixed usage."""
    def transport(payload):
        return {
            "choices": [{"message": {"content": reply}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    return transport


def client(tmp_path, transport, **kwargs):
    return OpenRouter(
        model="stub/model",
        pricing=PRICING,
        ledger=kwargs.pop("ledger", Ledger(cap=10.0)),
        transcript=Transcript(tmp_path / "calls.jsonl"),
        api_key="test",
        transport=transport,
        **kwargs,
    )


# --- money ------------------------------------------------------------------


def test_cost_includes_the_platform_fee():
    charge = PRICING.charge(1_000_000, 1_000_000)

    assert charge == pytest.approx((1.0 + 4.0) * 1.055)


def test_spending_is_accounted_per_call(tmp_path):
    ledger = Ledger(cap=10.0)
    agent = client(tmp_path, stub("MOVE: A1"), ledger=ledger)

    agent("hello")
    agent("hello again")

    expected = 2 * PRICING.charge(100, 50)
    assert ledger.calls == 2
    assert ledger.spent == pytest.approx(expected)
    assert ledger.by_model["stub/model"] == pytest.approx(expected)


def test_the_cap_stops_a_run_before_it_overruns(tmp_path):
    """The ceiling is checked against the worst case, so it cannot be crossed."""
    ledger = Ledger(cap=0.001)
    agent = client(tmp_path, stub("MOVE: A1"), ledger=ledger, max_tokens=2048)

    with pytest.raises(BudgetExceeded):
        agent("a prompt long enough to matter " * 20)

    assert ledger.calls == 0, "a refused call must not be billed"


def test_a_run_can_finish_inside_a_generous_cap(tmp_path):
    ledger = Ledger(cap=1.0)
    agent = client(tmp_path, stub("MOVE: A1"), ledger=ledger, max_tokens=64)

    for _ in range(5):
        agent("short prompt")

    assert ledger.calls == 5 and ledger.spent < 1.0


# --- the transcript ---------------------------------------------------------


def test_every_call_is_written_down(tmp_path):
    transcript = Transcript(tmp_path / "calls.jsonl")
    agent = OpenRouter("stub/model", PRICING, Ledger(cap=10.0), transcript,
                       api_key="test", transport=stub("MOVE: A2"))
    agent.tag = "e1:trap"
    agent("what now?")

    written = transcript.read()
    assert len(written) == 1
    assert written[0].reply == "MOVE: A2"
    assert written[0].tag == "e1:trap"
    assert written[0].prompt_tokens == 100


def test_replaying_a_transcript_costs_nothing(tmp_path):
    path = tmp_path / "calls.jsonl"
    first = OpenRouter("stub/model", PRICING, Ledger(cap=10.0), Transcript(path),
                       api_key="test", transport=stub("MOVE: A3"))
    first("the question")

    ledger = Ledger(cap=10.0)
    second = OpenRouter("stub/model", PRICING, ledger, Transcript(path), api_key="test",
                        transport=stub("SHOULD NOT BE CALLED"))

    assert second("the question") == "MOVE: A3"
    assert ledger.calls == 0, "a replayed call must not be billed again"


def test_repeated_samples_stay_independent(tmp_path):
    """E4 draws many rollouts at temperature; they must not collapse into one reply."""
    path = tmp_path / "calls.jsonl"
    replies = iter(["MOVE: A1", "MOVE: A2", "MOVE: A3"])

    def varying(payload):
        return {
            "choices": [{"message": {"content": next(replies)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    agent = OpenRouter("stub/model", PRICING, Ledger(cap=10.0), Transcript(path),
                       api_key="test", temperature=1.0, transport=varying)

    seen = []
    for index in range(3):
        agent.sample = index
        seen.append(agent("same prompt"))

    assert seen == ["MOVE: A1", "MOVE: A2", "MOVE: A3"]
    assert len(Transcript(path).replay_index()) == 3


# --- failure ----------------------------------------------------------------


def test_transient_failures_are_retried(tmp_path):
    attempts = {"n": 0}

    def flaky(payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("upstream hiccup")
        return {
            "choices": [{"message": {"content": "MOVE: A1"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    agent = client(tmp_path, flaky, retries=4)
    agent.retries = 4

    assert agent("go") == "MOVE: A1"
    assert attempts["n"] == 3


def test_a_call_that_never_succeeds_is_recorded_not_raised(tmp_path):
    """A dead model should cost the run one episode, not the whole experiment."""
    def broken(payload):
        raise ConnectionError("refused")

    transcript = Transcript(tmp_path / "calls.jsonl")
    agent = OpenRouter("stub/model", PRICING, Ledger(cap=10.0), transcript,
                       api_key="test", transport=broken, retries=2)

    assert agent("go") == ""
    assert "ConnectionError" in transcript.read()[0].error


def test_missing_key_is_reported_before_any_work(tmp_path):
    agent = OpenRouter("stub/model", PRICING, Ledger(cap=10.0),
                       Transcript(tmp_path / "calls.jsonl"), api_key="")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent("go")


# --- it plugs into the harness ----------------------------------------------


def test_the_client_drives_a_real_episode(tmp_path):
    """The whole point: this is a responder, so the existing harness takes it as is."""
    instance = procrastination_trap()
    presentation = make_presentation(instance, __import__("random").Random(0))

    steady = presentation.arc_label[("m", "steady", 0)]
    first = presentation.arc_label[("s", "m", 0)]
    last = presentation.arc_label[("steady", "t", 0)]
    scripted = iter([
        f"PLAN: {first}, {steady}, {last}\nTOTAL: 13",
        f"MOVE: {first}",
        f"MOVE: {steady}",
        f"MOVE: {last}",
    ])

    def replying(payload):
        return {
            "choices": [{"message": {"content": next(scripted)}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        }

    agent = client(tmp_path, replying)
    episode = run_episode(instance, presentation, agent)

    assert episode.reached_goal
    assert episode.plan_is_a_route and episode.arithmetic_correct
    assert episode.plan_survived_execution is True


# --- picking the key up -----------------------------------------------------


def test_dotenv_is_read_when_the_shell_has_no_key(tmp_path, monkeypatch):
    """Exporting in one shell and running from another is the commonest slip."""
    from mcta.probe.client import load_env

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\nOPENROUTER_API_KEY='sk-or-v1-from-file'\n\n")

    load_env(env)

    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-from-file"


def test_an_exported_key_beats_the_file(tmp_path, monkeypatch):
    from mcta.probe.client import load_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-exported")
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=sk-or-v1-from-file\n")

    load_env(env)

    import os
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-exported"


def test_a_missing_dotenv_is_not_an_error(tmp_path):
    from mcta.probe.client import load_env

    load_env(tmp_path / "nothing-here")
