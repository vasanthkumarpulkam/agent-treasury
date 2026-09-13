import time
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from storage.db import Database
from governor.governor import Governor
from governor.models import Proposal


def make_config(**overrides):
    cfg = {
        "treasury": {"starting_capital_usd": 1000.0, "operating_reserve_pct": 0.15},
        "survival": {
            "daily_operating_cost_usd": 1.0,
            "initial_runway_days": 30,
            "charge_llm_spend": True,
        },
        "kill_switch": {
            "max_drawdown_pct": 0.30,
            "rolling_loss_window_days": 30,
            "rolling_loss_pct": 0.10,
            "min_operating_reserve_usd": 0,
            "resurrection_mode": "manual",
        },
        "allocation": {
            "polymarket_weight": 0.5, "bitcoin_weight": 0.5,
            "rebalance_frequency_days": 7, "per_leg_kill_drawdown_pct": 0.40,
        },
        "position_limits": {
            "max_single_position_pct": 0.05,
            "max_daily_notional_pct": 0.20,
            "max_total_exposure_pct": 0.80,
        },
        "polymarket": {"market_allowlist": []},
    }
    cfg.update(overrides)
    return cfg


class FakeExecutionClient:
    def execute(self, proposal, approved_size_usd, mode):
        return {"fill_price": proposal.limit_price or 0.5, "fill_size_usd": approved_size_usd,
                "fees_usd": 0.0, "order_id": "fake-1"}


@pytest.fixture
def governor():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    cfg = make_config()
    clients = {"polymarket": FakeExecutionClient(), "bitcoin": FakeExecutionClient()}
    g = Governor(db, cfg, clients)
    yield g
    os.unlink(tmp.name)


def test_starting_equity_matches_config(governor):
    assert governor.current_equity_usd() == 1000.0


def test_kill_switch_fires_on_max_drawdown(governor):
    governor.db.record_ledger_event("realized_pnl", -350.0, 650.0, leg="bitcoin")
    governor.check_kill_switch()
    assert governor.is_killed() is True


def test_kill_switch_does_not_fire_on_small_loss(governor):
    governor.db.record_ledger_event("realized_pnl", -50.0, 950.0, leg="bitcoin")
    governor.check_kill_switch()
    assert governor.is_killed() is False


def test_no_proposals_accepted_once_killed(governor):
    governor.db.record_ledger_event("realized_pnl", -350.0, 650.0, leg="bitcoin")
    governor.check_kill_switch()
    assert governor.is_killed()

    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=10,
                 confidence=0.8, rationale="test", limit_price=50000)
    decision = governor.review(p)
    assert decision.approved is False
    assert "killed" in decision.reason


def test_resume_clears_kill_state(governor):
    governor.db.record_ledger_event("realized_pnl", -350.0, 650.0, leg="bitcoin")
    governor.check_kill_switch()
    assert governor.is_killed()
    governor.resume("reviewed, adjusting strategy, resuming")
    assert governor.is_killed() is False


def test_position_size_capped_by_max_single_position_pct(governor):
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=500,
                 confidence=0.9, rationale="oversized test", limit_price=50000)
    decision = governor.review(p)
    assert decision.approved is True
    assert decision.approved_size_usd == pytest.approx(50.0, abs=0.01)


def test_duplicate_proposal_rejected(governor):
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=10,
                 confidence=0.8, rationale="dup test", limit_price=50000, source_data_ts=12345.0)
    d1 = governor.review(p)
    assert d1.approved is True
    governor.execute(d1, mode="paper")

    d2 = governor.review(p)
    assert d2.approved is False
    assert "duplicate" in d2.reason


def test_expired_proposal_rejected(governor):
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=10,
                 confidence=0.8, rationale="expired test", limit_price=50000,
                 expiry_ts=time.time() - 1)
    decision = governor.review(p)
    assert decision.approved is False
    assert "expired" in decision.reason


def test_leg_zeroed_out_after_heavy_leg_drawdown(governor):
    governor.db.record_ledger_event("realized_pnl", -250.0, 750.0, leg="bitcoin")
    multiplier = governor.leg_allocation_multiplier("bitcoin")
    assert multiplier == 0.0


def test_execute_creates_open_position(governor):
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=10,
                 confidence=0.8, rationale="position test", limit_price=50000)
    decision = governor.review(p)
    governor.execute(decision, mode="paper")
    positions = governor.db.open_positions(leg="bitcoin")
    assert len(positions) == 1
    assert positions[0]["size_usd"] == pytest.approx(10.0, abs=0.01)


def test_settle_realized_pnl_sweeps_operating_reserve(governor):
    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=10,
                 confidence=0.8, rationale="settle test", limit_price=50000)
    decision = governor.review(p)
    governor.execute(decision, mode="paper")
    position_id = governor.db.open_positions(leg="bitcoin")[0]["id"]

    reserve_before = governor.db.get_state("operating_reserve_usd", 0.0)
    governor.settle_realized_pnl(position_id, realized_pnl_usd=100.0, leg="bitcoin")
    reserve_after = governor.db.get_state("operating_reserve_usd", 0.0)
    assert reserve_after - reserve_before == pytest.approx(15.0, abs=0.01)


def test_close_positions_for_leg_realizes_profit_on_long():
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp_db.name)
    cfg = make_config()
    g = Governor(db, cfg, {"polymarket": FakeExecutionClient(), "bitcoin": FakeExecutionClient()})

    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=40,
                 confidence=0.8, rationale="test", limit_price=50000)
    decision = g.review(p)
    g.execute(decision, mode="paper")

    # Price rose 10% -- a $40 long should realize +$4.
    total_pnl = g.close_positions_for_leg("bitcoin", current_price=55000)
    assert total_pnl == pytest.approx(4.0, abs=0.01)
    assert g.db.open_positions(leg="bitcoin") == []
    os.unlink(tmp_db.name)


def test_close_positions_for_leg_realizes_profit_on_short():
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp_db.name)
    cfg = make_config()
    g = Governor(db, cfg, {"polymarket": FakeExecutionClient(), "bitcoin": FakeExecutionClient()})

    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="sell", size_usd=40,
                 confidence=0.8, rationale="test", limit_price=50000)
    decision = g.review(p)
    g.execute(decision, mode="paper")

    # Price fell 10% -- a $40 short should realize +$4.
    total_pnl = g.close_positions_for_leg("bitcoin", current_price=45000)
    assert total_pnl == pytest.approx(4.0, abs=0.01)
    os.unlink(tmp_db.name)


def test_close_positions_updates_equity_via_realized_pnl():
    tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp_db.name)
    cfg = make_config()
    g = Governor(db, cfg, {"polymarket": FakeExecutionClient(), "bitcoin": FakeExecutionClient()})

    p = Proposal(leg="bitcoin", market_or_symbol="BTC/USD", side="buy", size_usd=40,
                 confidence=0.8, rationale="test", limit_price=50000)
    decision = g.review(p)
    g.execute(decision, mode="paper")

    equity_before = g.current_equity_usd()
    g.close_positions_for_leg("bitcoin", current_price=55000)
    equity_after = g.current_equity_usd()
    assert equity_after - equity_before == pytest.approx(4.0, abs=0.01)
    os.unlink(tmp_db.name)
