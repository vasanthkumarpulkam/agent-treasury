"""Tests for mechanical arbitrage detection.

This is the one strategy in the repo whose edge can be verified before trading, so the
arithmetic has to be exactly right. A false positive here doesn't cost a missed
opportunity -- it buys a basket that costs MORE than it can ever pay out, which is a
guaranteed loss rather than a probabilistic one.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pods.polymarket.arbitrage_agent import ArbitrageAgent


def make_config(**over):
    cfg = {"arbitrage": {"enabled": True, "min_profit_pct": 0.02, "max_position_usd": 100.0,
                          "min_liquidity_usd": 20.0}, "polymarket": {}}
    cfg["arbitrage"].update(over)
    return cfg


class FakeClient:
    """Books keyed by token id: {token: (ask_price, ask_shares)}"""
    def __init__(self, books):
        self.books = books

    def get_orderbook(self, token_id):
        if token_id not in self.books:
            raise KeyError(token_id)
        price, shares = self.books[token_id]
        return {"asks": [{"price": str(price), "size": str(shares)}], "bids": []}


def market(yes="yes1", no="no1", question="Q?"):
    return {"question": question, "clobTokenIds": f'["{yes}", "{no}"]'}


def test_detects_binary_complement_arbitrage():
    """YES 0.46 + NO 0.50 = 0.96 to buy something that always pays 1.00."""
    client = FakeClient({"yes1": (0.46, 1000), "no1": (0.50, 1000)})
    agent = ArbitrageAgent(client, make_config())
    arbs = agent.find_binary_arbs([market()])
    assert len(arbs) == 1
    assert arbs[0]["profit_pct"] == pytest.approx(0.04, abs=1e-9)


def test_ignores_fairly_priced_binary():
    client = FakeClient({"yes1": (0.50, 1000), "no1": (0.50, 1000)})
    agent = ArbitrageAgent(client, make_config())
    assert agent.find_binary_arbs([market()]) == []


def test_ignores_overpriced_binary():
    """Sum > 1 is not an arbitrage to BUY -- buying it guarantees a loss."""
    client = FakeClient({"yes1": (0.55, 1000), "no1": (0.55, 1000)})
    agent = ArbitrageAgent(client, make_config())
    assert agent.find_binary_arbs([market()]) == []


def test_respects_min_profit_threshold():
    """0.5% edge is real but smaller than gas + slippage, so it must be skipped."""
    client = FakeClient({"yes1": (0.497, 1000), "no1": (0.498, 1000)})
    agent = ArbitrageAgent(client, make_config(min_profit_pct=0.02))
    assert agent.find_binary_arbs([market()]) == []


def test_size_limited_by_thinnest_side_of_book():
    """You can only buy what's resting. The thin leg caps the whole trade."""
    client = FakeClient({"yes1": (0.46, 1000), "no1": (0.50, 60)})  # no1 depth = 0.50*60 = $30
    agent = ArbitrageAgent(client, make_config())
    arbs = agent.find_binary_arbs([market()])
    assert arbs[0]["size_usd"] == pytest.approx(30.0, abs=0.01)


def test_skips_opportunities_too_thin_to_trade():
    client = FakeClient({"yes1": (0.46, 10), "no1": (0.50, 10)})  # ~$5 available
    agent = ArbitrageAgent(client, make_config(min_liquidity_usd=20))
    assert agent.find_binary_arbs([market()]) == []


def test_skips_markets_with_no_book():
    client = FakeClient({"yes1": (0.46, 1000)})  # no book for no1
    agent = ArbitrageAgent(client, make_config())
    assert agent.find_binary_arbs([market()]) == []


def test_detects_negrisk_basket_arbitrage():
    """Three mutually exclusive outcomes priced 0.30+0.30+0.35 = 0.95, pays 1.00."""
    client = FakeClient({"a": (0.30, 1000), "b": (0.30, 1000), "c": (0.35, 1000)})
    event = {"title": "Who wins?", "negRisk": True, "markets": [
        {"clobTokenIds": '["a", "x"]'}, {"clobTokenIds": '["b", "y"]'},
        {"clobTokenIds": '["c", "z"]'}]}
    agent = ArbitrageAgent(client, make_config())
    arbs = agent.find_basket_arbs([event])
    assert len(arbs) == 1
    assert arbs[0]["profit_pct"] == pytest.approx(0.05, abs=1e-9)
    assert len(arbs[0]["legs"]) == 3


def test_ignores_non_negrisk_events_entirely():
    """CRITICAL: without negRisk, outcomes may not be mutually exclusive. Two could both
    resolve YES, so 'buy everything cheap' is a guaranteed loss, not an arbitrage."""
    client = FakeClient({"a": (0.30, 1000), "b": (0.30, 1000)})
    event = {"title": "Unrelated markets", "negRisk": False, "markets": [
        {"clobTokenIds": '["a", "x"]'}, {"clobTokenIds": '["b", "y"]'}]}
    agent = ArbitrageAgent(client, make_config())
    assert agent.find_basket_arbs([event]) == []


def test_basket_ignored_when_any_leg_unpriceable():
    """A basket missing one leg isn't a hedge -- it's a directional bet."""
    client = FakeClient({"a": (0.30, 1000), "b": (0.30, 1000)})  # "c" has no book
    event = {"title": "Who wins?", "negRisk": True, "markets": [
        {"clobTokenIds": '["a", "x"]'}, {"clobTokenIds": '["b", "y"]'},
        {"clobTokenIds": '["c", "z"]'}]}
    agent = ArbitrageAgent(client, make_config())
    assert agent.find_basket_arbs([event]) == []


def test_proposals_share_a_bundle_id_and_buy_equal_shares():
    class ListClient(FakeClient):
        def list_active_markets(self, limit=50):
            return [market()]
        def list_active_events(self, limit=30):
            return []

    client = ListClient({"yes1": (0.40, 1000), "no1": (0.50, 1000)})
    agent = ArbitrageAgent(client, make_config())
    proposals = agent.generate_proposals(1000.0)

    assert len(proposals) == 2
    assert proposals[0].bundle_id == proposals[1].bundle_id
    assert proposals[0].bundle_id is not None
    # Equal share counts are what make the payout uniform: size/price must match.
    shares = [p.size_usd / p.limit_price for p in proposals]
    assert shares[0] == pytest.approx(shares[1], rel=0.02)
    # Arbitrage always buys the token it names.
    assert all(p.side == "buy_yes" for p in proposals)
    assert all(p.confidence == 1.0 for p in proposals)


def test_disabled_arbitrage_produces_nothing():
    client = FakeClient({})
    agent = ArbitrageAgent(client, make_config(enabled=False))
    assert agent.generate_proposals(1000.0) == []
