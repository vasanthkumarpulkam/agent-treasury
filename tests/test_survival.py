"""Tests for the survival mechanic: the agent must earn its keep or die.

These are the most important tests in the repo. If the kill-switch doesn't fire exactly
when it should, the entire premise of the system is a lie -- it's just a trading bot that
claims it will stop.
"""
import os
import sys
import time
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from storage.db import Database
from governor.governor import Governor
from governor.models import Proposal


def make_config(daily_cost=1.0, runway_days=30):
    return {
        "treasury": {"starting_capital_usd": 1000.0, "operating_reserve_pct": 0.15},
        "survival": {"daily_operating_cost_usd": daily_cost,
                     "initial_runway_days": runway_days, "charge_llm_spend": True},
        "kill_switch": {"max_drawdown_pct": 0.30, "rolling_loss_window_days": 30,
                         "rolling_loss_pct": 0.10, "min_operating_reserve_usd": 0,
                         "resurrection_mode": "manual"},
        "allocation": {"polymarket_weight": 0.5, "bitcoin_weight": 0.5,
                        "rebalance_frequency_days": 7, "per_leg_kill_drawdown_pct": 0.40},
        "position_limits": {"max_single_position_pct": 0.05, "max_daily_notional_pct": 0.20,
                             "max_total_exposure_pct": 0.80},
        "polymarket": {"market_allowlist": []},
    }


class FakeClient:
    def execute(self, proposal, approved_size_usd, mode):
        return {"fill_price": proposal.limit_price or 0.5, "fill_size_usd": approved_size_usd,
                "fees_usd": 0.0, "order_id": "fake"}


def make_governor(config=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    g = Governor(db, config or make_config(), {"polymarket": FakeClient(), "bitcoin": FakeClient()})
    return g, tmp.name


def test_starts_with_runway_proportional_to_burn_rate():
    g, path = make_governor(make_config(daily_cost=2.0, runway_days=10))
    assert g.db.get_state("operating_reserve_usd") == pytest.approx(20.0)
    os.unlink(path)


def test_metabolic_cost_drains_reserve_over_time():
    g, path = make_governor(make_config(daily_cost=10.0))
    # Pretend a full day passed since the last charge.
    g.db.set_state("last_metabolic_charge_ts", time.time() - 86400)
    before = g.db.get_state("operating_reserve_usd")
    charged = g.charge_metabolic_cost()
    after = g.db.get_state("operating_reserve_usd")
    assert charged == pytest.approx(10.0, abs=0.1)
    assert before - after == pytest.approx(10.0, abs=0.1)
    os.unlink(path)


def test_idle_agent_eventually_starves_to_death():
    """The core premise: doing nothing is NOT survival. An agent that never earns dies."""
    g, path = make_governor(make_config(daily_cost=10.0, runway_days=1))
    assert not g.is_killed()
    # Two days pass with no profit whatsoever.
    g.db.set_state("last_metabolic_charge_ts", time.time() - 2 * 86400)
    g.charge_metabolic_cost()
    g.check_kill_switch()
    assert g.is_killed(), "an agent that never earned anything must not survive indefinitely"
    assert "starvation" in g.db.get_state("killed_reason")
    os.unlink(path)


def test_profit_refills_reserve_and_extends_life():
    g, path = make_governor(make_config(daily_cost=10.0, runway_days=1))
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=40,
                 confidence=0.9, rationale="t", limit_price=50000)
    g.execute(g.review(p), mode="paper")
    position_id = g.db.open_positions(leg="bitcoin")[0]["id"]
    reserve_before = g.db.get_state("operating_reserve_usd")

    g.settle_realized_pnl(position_id, realized_pnl_usd=100.0, leg="bitcoin")
    reserve_after = g.db.get_state("operating_reserve_usd")
    assert reserve_after > reserve_before, "profit must extend the agent's life"
    os.unlink(path)


def test_unrealized_losses_count_against_survival():
    """A position deep underwater must not look identical to no position at all."""
    g, path = make_governor()
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=50,
                 confidence=0.9, rationale="t", limit_price=50000)
    g.execute(g.review(p), mode="paper")

    realized_equity = g.current_equity_usd()
    # Price halved: the open position is down 50%.
    mtm_equity = g.mark_to_market_equity({"BTC/USD": 25000})
    assert mtm_equity < realized_equity
    assert mtm_equity == pytest.approx(realized_equity - 25.0, abs=0.5)
    os.unlink(path)


def test_death_liquidates_all_open_positions():
    """A 'dead' agent still holding risk is not dead."""
    g, path = make_governor()
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=40,
                 confidence=0.9, rationale="t", limit_price=50000)
    g.execute(g.review(p), mode="paper")
    assert len(g.db.open_positions()) == 1

    g._kill("test death", price_lookup={"BTC/USD": 50000})
    assert g.is_killed()
    assert g.db.open_positions() == [], "death must flatten every position"
    os.unlink(path)


def test_death_writes_a_tombstone():
    g, path = make_governor()
    g._kill("test death", price_lookup={})
    graves = g.db.graveyard()
    assert len(graves) == 1
    assert graves[0]["generation"] == 1
    assert graves[0]["cause_of_death"] == "test death"
    os.unlink(path)


def test_dead_agent_refuses_all_proposals():
    g, path = make_governor()
    g._kill("test death", price_lookup={})
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=10,
                 confidence=0.9, rationale="t", limit_price=50000)
    assert g.review(p).approved is False
    os.unlink(path)


def test_resurrection_creates_new_generation_and_keeps_graveyard():
    g, path = make_governor()
    g.db.record_ledger_event("realized_pnl", -200.0, 800.0, leg="bitcoin")
    g._kill("first death", price_lookup={})

    g.resurrect(500.0, "second attempt")
    assert not g.is_killed()
    assert g.db.get_state("generation") == 2
    # New generation starts at its own capital, not carrying the old loss forward.
    assert g.current_equity_usd() == pytest.approx(500.0)
    # But the previous death is still on the record.
    assert len(g.db.graveyard()) == 1
    os.unlink(path)


def test_mark_to_market_drawdown_triggers_death_before_realizing_losses():
    """The kill-switch must fire on open losses, not wait for them to be realized."""
    g, path = make_governor()
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=50,
                 confidence=0.9, rationale="t", limit_price=50000)
    g.execute(g.review(p), mode="paper")
    # Nothing realized yet, so a realized-only view sees a perfectly healthy account.
    assert g.current_equity_usd() == pytest.approx(1000.0)

    # But equity is really down >30% once the open position is marked to market.
    g.db.record_ledger_event("realized_pnl", -250.0, 750.0, leg="bitcoin")
    g.check_kill_switch(price_lookup={"BTC/USD": 1000})  # position down ~98%
    assert g.is_killed()
    os.unlink(path)


def test_bundle_rejected_entirely_if_any_leg_fails():
    """Half an arbitrage is an unhedged bet, which is worse than no trade."""
    from governor.models import Proposal
    g, path = make_governor()
    g._kill("dead", price_lookup={})  # killed system rejects everything
    legs = [Proposal(leg="polymarket", market_or_symbol="a", side="buy_yes", size_usd=10,
                     confidence=1.0, rationale="arb", limit_price=0.4, bundle_id="b1"),
            Proposal(leg="polymarket", market_or_symbol="b", side="buy_yes", size_usd=10,
                     confidence=1.0, rationale="arb", limit_price=0.5, bundle_id="b1")]
    decisions = g.review_bundle(legs)
    assert all(not d.approved for d in decisions)
    os.unlink(path)


def test_bundle_rejected_if_position_limits_would_trim_a_leg():
    """A capped leg breaks the hedge, so the whole bundle must be refused."""
    from governor.models import Proposal
    g, path = make_governor()
    # max_single_position_pct 0.05 of $1000 = $50 cap; ask for more on one leg.
    legs = [Proposal(leg="polymarket", market_or_symbol="a", side="buy_yes", size_usd=20,
                     confidence=1.0, rationale="arb", limit_price=0.4, bundle_id="b1"),
            Proposal(leg="polymarket", market_or_symbol="b", side="buy_yes", size_usd=200,
                     confidence=1.0, rationale="arb", limit_price=0.5, bundle_id="b1")]
    decisions = g.review_bundle(legs)
    assert all(not d.approved for d in decisions)
    assert "unhedged" in decisions[0].reason
    os.unlink(path)


def test_bundle_approved_when_all_legs_fit():
    from governor.models import Proposal
    g, path = make_governor()
    legs = [Proposal(leg="polymarket", market_or_symbol="a", side="buy_yes", size_usd=20,
                     confidence=1.0, rationale="arb", limit_price=0.4, bundle_id="b1"),
            Proposal(leg="polymarket", market_or_symbol="b", side="buy_yes", size_usd=25,
                     confidence=1.0, rationale="arb", limit_price=0.5, bundle_id="b1")]
    decisions = g.review_bundle(legs)
    assert all(d.approved for d in decisions)
    os.unlink(path)
