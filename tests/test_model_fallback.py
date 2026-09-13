"""Tests for the automatic model fallback chain.

An agent meant to run unattended must survive its model provider misbehaving: slugs get
retired (404), balances run dry (402), providers rate-limit (429). Without fallback, any
of those silently reduces the agent to "no edge on every market" indefinitely -- it keeps
running, keeps paying rent, and never trades, which looks like caution but is just
blindness.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import requests
from pods.polymarket.llm_estimator import LLMEstimator


def make_config(models=None, **over):
    cfg = {"polymarket": {
        "llm_models": models if models is not None else ["model-a", "model-b:free", "model-c"],
        "llm_max_spend_per_cycle_usd": 0.50,
        "llm_estimated_cost_per_call_usd": 0.01,
    }}
    cfg["polymarket"].update(over)
    return cfg


class FakeResponse:
    def __init__(self, status_code, content="ok"):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_uses_first_model_when_it_works(monkeypatch):
    est = LLMEstimator(make_config())
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, "hello"))
    assert est._chat([], "key") == "hello"
    assert est.model == "model-a"


@pytest.mark.parametrize("status", [402, 404, 429, 500, 503])
def test_falls_through_on_recoverable_statuses(monkeypatch, status):
    est = LLMEstimator(make_config())
    calls = []

    def fake_post(*a, **k):
        model = k["json"]["model"]
        calls.append(model)
        # First model always fails with the given status; second one works.
        return FakeResponse(200, "recovered") if model == "model-b:free" else FakeResponse(status)

    monkeypatch.setattr(requests, "post", fake_post)
    assert est._chat([], "key") == "recovered"
    assert calls[0] == "model-a"
    assert est.model == "model-b:free"


def test_raises_only_when_every_model_fails(monkeypatch):
    est = LLMEstimator(make_config())
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(402))
    with pytest.raises(RuntimeError, match="all 3 models failed"):
        est._chat([], "key")


def test_estimate_falls_back_to_market_price_when_chain_exhausted(monkeypatch):
    """Total LLM outage must produce 'no edge', never a made-up probability."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    est = LLMEstimator(make_config())
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(402))
    result = est.estimate("Will X?", "", market_implied_prob=0.37)
    assert result["probability"] == 0.37
    assert result["confidence"] == 0.0


def test_dead_model_not_retried_within_a_run(monkeypatch):
    """A dead model must not be re-tried on every one of 15 markets."""
    est = LLMEstimator(make_config())
    attempts = []

    def fake_post(*a, **k):
        model = k["json"]["model"]
        attempts.append(model)
        return FakeResponse(200, "ok") if model == "model-b:free" else FakeResponse(404)

    monkeypatch.setattr(requests, "post", fake_post)
    est._chat([], "key")
    est._chat([], "key")
    est._chat([], "key")
    assert attempts.count("model-a") == 1, "dead model should be tried once, then skipped"


def test_free_models_do_not_consume_spend_budget(monkeypatch):
    est = LLMEstimator(make_config(models=["free-model:free"]))
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, "ok"))
    est._chat([], "key")
    est._chat([], "key")
    assert est._spent_this_cycle == 0.0
    assert est.budget_remaining() is True


def test_paid_models_do_consume_spend_budget(monkeypatch):
    est = LLMEstimator(make_config(models=["paid-model"]))
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(200, "ok"))
    est._chat([], "key")
    assert est._spent_this_cycle == pytest.approx(0.01)


def test_legacy_single_model_config_still_works():
    cfg = {"polymarket": {"llm_model": "solo-model",
                           "llm_max_spend_per_cycle_usd": 0.5,
                           "llm_estimated_cost_per_call_usd": 0.01}}
    est = LLMEstimator(cfg)
    assert est.models == ["solo-model"]
    assert est.model == "solo-model"


def test_failed_models_get_another_chance_next_cycle(monkeypatch):
    """Free-tier 429s are temporary -- a throttled model must not stay demoted forever."""
    est = LLMEstimator(make_config())
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse(429))
    with pytest.raises(RuntimeError):
        est._chat([], "key")
    assert est._active_idx >= len(est.models)

    est.reset_cycle_spend()
    assert est._active_idx == 0
    assert est._failed_models == {}
    assert est.model == "model-a"


class ErrorBodyResponse(FakeResponse):
    """HTTP 200 with an error payload instead of a completion -- what OpenRouter returns
    for moderation blocks, capacity issues and upstream provider failures."""
    def __init__(self, payload):
        super().__init__(200)
        self._payload = payload

    def json(self):
        return self._payload


def test_falls_through_on_error_body_with_http_200(monkeypatch):
    est = LLMEstimator(make_config())

    def fake_post(*a, **k):
        if k["json"]["model"] == "model-b:free":
            return FakeResponse(200, "recovered")
        return ErrorBodyResponse({"error": {"message": "provider capacity exceeded"}})

    monkeypatch.setattr(requests, "post", fake_post)
    assert est._chat([], "key") == "recovered"


def test_falls_through_when_choices_missing(monkeypatch):
    """The exact KeyError('choices') seen in a live run."""
    est = LLMEstimator(make_config())

    def fake_post(*a, **k):
        if k["json"]["model"] == "model-b:free":
            return FakeResponse(200, "recovered")
        return ErrorBodyResponse({"id": "x", "object": "chat.completion"})

    monkeypatch.setattr(requests, "post", fake_post)
    assert est._chat([], "key") == "recovered"


def test_falls_through_on_non_json_body(monkeypatch):
    est = LLMEstimator(make_config())

    class BadJson(FakeResponse):
        def json(self):
            raise ValueError("not json")

    def fake_post(*a, **k):
        return FakeResponse(200, "recovered") if k["json"]["model"] == "model-b:free" else BadJson(200)

    monkeypatch.setattr(requests, "post", fake_post)
    assert est._chat([], "key") == "recovered"


def test_error_body_does_not_charge_spend_budget(monkeypatch):
    """A failed call must not be billed against the survival budget."""
    est = LLMEstimator(make_config(models=["paid-a", "paid-b"]))

    def fake_post(*a, **k):
        return FakeResponse(200, "ok") if k["json"]["model"] == "paid-b" else ErrorBodyResponse(
            {"error": {"message": "nope"}})

    monkeypatch.setattr(requests, "post", fake_post)
    est._chat([], "key")
    assert est._spent_this_cycle == pytest.approx(0.01), "only the successful call should be charged"
