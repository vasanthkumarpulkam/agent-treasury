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
