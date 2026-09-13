"""Tests for _implied_prob's handling of Polymarket's Gamma API quirks -- specifically
the stringified-JSON outcomePrices field that silently poisoned every trade with a fake
0.5 implied price until this was caught in a real run against live market data."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pods.polymarket.research_agent import PolymarketResearchAgent


def make_config():
    return {"polymarket": {
        "llm_model": "anthropic/claude-sonnet-5",
        "llm_max_spend_per_cycle_usd": 0.5,
        "llm_estimated_cost_per_call_usd": 0.01,
    }}


def make_agent():
    return PolymarketResearchAgent(client=None, config=make_config())


def test_implied_prob_handles_real_list():
    agent = make_agent()
    market = {"outcomePrices": [0.62, 0.38]}
    assert agent._implied_prob(market) == 0.62


def test_implied_prob_handles_stringified_json_array():
    """This is the actual shape Polymarket's Gamma API returns in practice."""
    agent = make_agent()
    market = {"outcomePrices": '["0.62", "0.38"]'}
    assert agent._implied_prob(market) == 0.62


def test_implied_prob_returns_none_on_missing_field():
    agent = make_agent()
    assert agent._implied_prob({}) is None


def test_implied_prob_returns_none_on_garbage_string():
    agent = make_agent()
    market = {"outcomePrices": "not json at all"}
    assert agent._implied_prob(market) is None


def test_implied_prob_never_silently_falls_back_to_half():
    """Regression test for the bug that shipped: a bad/unparseable price must never
    become the fake value 0.5 -- it must return None so the caller skips the market."""
    agent = make_agent()
    for bad_market in [{}, {"outcomePrices": None}, {"outcomePrices": "[]"}, {"outcomePrices": "{}"}]:
        result = agent._implied_prob(bad_market)
        assert result != 0.5, f"silently produced fake 0.5 for {bad_market!r}"


def test_parse_gamma_array_handles_stringified_token_ids():
    """clobTokenIds has the same stringified-array quirk as outcomePrices -- this is what
    caused a token id of '[' to be sent as a live order identifier."""
    agent = make_agent()
    tokens = agent._parse_gamma_array('["111111", "222222"]')
    assert tokens == ["111111", "222222"]


def test_outcome_tokens_returns_yes_and_no():
    agent = make_agent()
    market = {"clobTokenIds": '["yes-token", "no-token"]'}
    assert agent._outcome_tokens(market) == ("yes-token", "no-token")


def test_outcome_tokens_returns_none_when_unusable():
    agent = make_agent()
    assert agent._outcome_tokens({}) == (None, None)
    assert agent._outcome_tokens({"clobTokenIds": '["only-one"]'}) == (None, None)


def test_outcome_prices_parses_both_sides():
    agent = make_agent()
    market = {"outcomePrices": '["0.62", "0.38"]'}
    assert agent._outcome_prices(market) == (0.62, 0.38)


def _estimate(conf=0.8):
    return {"probability": 0.0, "confidence": conf, "reasoning": ""}


def make_gate_agent():
    cfg = make_config()
    cfg["polymarket"].update({"min_edge_pct": 0.06, "min_price": 0.05,
                               "max_price": 0.95, "max_odds_ratio": 3.0})
    return PolymarketResearchAgent(client=None, config=cfg)


def test_gate_allows_normal_edge_at_mid_price():
    agent = make_gate_agent()
    est = {"probability": 0.60, "confidence": 0.8, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.50, fair_value=0.60, edge=0.10, estimate=est)
    assert ok is True and reason is None


def test_gate_rejects_extreme_longshot_prices():
    """The real failure: a $15.62 bet on a market priced at 0.45 cents."""
    agent = make_gate_agent()
    est = {"probability": 0.08, "confidence": 0.8, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.0045, fair_value=0.08, edge=0.0755, estimate=est)
    assert ok is False
    assert "outside tradeable band" in reason


def test_gate_rejects_near_certain_prices():
    agent = make_gate_agent()
    est = {"probability": 0.90, "confidence": 0.8, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.98, fair_value=0.90, edge=-0.08, estimate=est)
    assert ok is False
    assert "outside tradeable band" in reason


def test_gate_rejects_extraordinary_relative_claims():
    """In-band price, big absolute edge, but the model claims 4x the market."""
    agent = make_gate_agent()
    est = {"probability": 0.40, "confidence": 0.8, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.10, fair_value=0.40, edge=0.30, estimate=est)
    assert ok is False
    assert "max_odds_ratio" in reason


def test_gate_rejects_extraordinary_claims_in_both_directions():
    agent = make_gate_agent()
    est = {"probability": 0.10, "confidence": 0.8, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.60, fair_value=0.10, edge=-0.50, estimate=est)
    assert ok is False
    assert "max_odds_ratio" in reason


def test_gate_rejects_zero_confidence_without_noise():
    agent = make_gate_agent()
    est = {"probability": 0.60, "confidence": 0.0, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.50, fair_value=0.60, edge=0.10, estimate=est)
    assert ok is False
    assert reason is None  # fallback path shouldn't spam logs


def test_gate_rejects_insufficient_edge_quietly():
    agent = make_gate_agent()
    est = {"probability": 0.52, "confidence": 0.8, "reasoning": ""}
    ok, reason = agent._passes_trade_gates(implied=0.50, fair_value=0.52, edge=0.02, estimate=est)
    assert ok is False
    assert reason is None


def test_gate_rejects_self_declared_uninformed_estimates():
    """The dominant real-world case: model says 0.10 confidence and 'no data provided'."""
    agent = make_gate_agent()
    agent.cfg["min_llm_confidence"] = 0.40
    est = {"probability": 0.65, "confidence": 0.10, "reasoning": "no information provided"}
    ok, reason = agent._passes_trade_gates(implied=0.50, fair_value=0.65, edge=0.15, estimate=est)
    assert ok is False
    assert "no information" in reason


def test_gate_allows_confident_estimates():
    agent = make_gate_agent()
    agent.cfg["min_llm_confidence"] = 0.40
    est = {"probability": 0.65, "confidence": 0.75, "reasoning": "strong basis"}
    ok, reason = agent._passes_trade_gates(implied=0.50, fair_value=0.65, edge=0.15, estimate=est)
    assert ok is True


def test_missing_volume_field_does_not_filter_market_out():
    """Regression: failing closed on absent volume data silently discarded every market
    in a live run, and the agent reported success while scoring nothing."""
    agent = make_gate_agent()
    agent.cfg["min_volume_usd"] = 20000
    agent.cfg["exclude_keywords"] = []
    assert agent._is_worth_scoring({"question": "Will X happen?"}) is True


def test_low_volume_market_is_filtered_when_volume_known():
    agent = make_gate_agent()
    agent.cfg["min_volume_usd"] = 20000
    agent.cfg["exclude_keywords"] = []
    assert agent._is_worth_scoring({"question": "Q", "volumeNum": 500}) is False


def test_high_volume_market_passes():
    agent = make_gate_agent()
    agent.cfg["min_volume_usd"] = 20000
    agent.cfg["exclude_keywords"] = []
    assert agent._is_worth_scoring({"question": "Q", "volumeNum": 50000}) is True


def test_excluded_keywords_filter_sports_props():
    agent = make_gate_agent()
    agent.cfg["exclude_keywords"] = ["O/U", "Exact Score"]
    agent.cfg["min_volume_usd"] = 0
    assert agent._is_worth_scoring({"question": "Team A vs B: O/U 2.5 goals"}) is False
    assert agent._is_worth_scoring({"question": "Exact Score: 0-3?"}) is False
    assert agent._is_worth_scoring({"question": "Will the bill pass by June?"}) is True
