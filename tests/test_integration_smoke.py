import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.db import Database
from governor.governor import Governor
from governor.models import Proposal
from orchestrator.orchestrator import Orchestrator


class FakePolymarketClient:
    def list_active_markets(self, limit=50):
        return [
            {"question": "Will X happen?", "category": "politics",
             "outcomePrices": ["0.30"], "clobTokenIds": ["tok-1"]},
            {"question": "Will Y happen?", "category": "sports",
             "outcomePrices": ["0.80"], "clobTokenIds": ["tok-2"]},
        ]

    def execute(self, proposal, approved_size_usd, mode):
        return {"fill_price": proposal.limit_price, "fill_size_usd": approved_size_usd,
                "fees_usd": 0.0, "order_id": "paper-pm-1"}


class BiasedPolymarketAgent:
    def __init__(self, client, config):
        self.client = client
        self.cfg = config["polymarket"]

    def generate_proposals(self, equity_usd):
        markets = self.client.list_active_markets()
        m = markets[0]
        implied = float(m["outcomePrices"][0])
        fake_model_prob = implied + 0.15
        proposals = [Proposal(
            leg="polymarket", market_or_symbol=m["clobTokenIds"][0], side="buy_yes",
            size_usd=equity_usd * self.cfg["kelly_fraction"], confidence=0.8,
            limit_price=implied, rationale=f"smoke test forced edge, implied={implied}",
        )]
        return proposals


class FakeBitcoinClient:
    def fetch_ohlcv(self, timeframe="1h", limit=300):
        import time
        now = time.time() * 1000
        return [[now - (limit - i) * 3600_000, 100 + i, 100 + i + 1, 100 + i - 1, 100 + i, 10]
                for i in range(limit)]

    def fetch_ticker(self):
        return {"ask": 50000.0, "bid": 49990.0, "last": 49995.0}

    def execute(self, proposal, approved_size_usd, mode):
        return {"fill_price": proposal.limit_price, "fill_size_usd": approved_size_usd,
                "fees_usd": approved_size_usd * 0.006, "order_id": "paper-btc-1"}


def make_config():
    return {
        "treasury": {"starting_capital_usd": 1000.0, "operating_reserve_pct": 0.15},
        "kill_switch": {
            "max_drawdown_pct": 0.30, "rolling_loss_window_days": 30,
            "rolling_loss_pct": 0.10, "min_operating_reserve_usd": 20,
            "resurrection_mode": "manual",
        },
        "allocation": {
            "polymarket_weight": 0.5, "bitcoin_weight": 0.5,
            "rebalance_frequency_days": 7, "per_leg_kill_drawdown_pct": 0.40,
        },
        "position_limits": {
            "max_single_position_pct": 0.05, "max_daily_notional_pct": 0.20,
            "max_total_exposure_pct": 0.80,
        },
        "polymarket": {
            "min_edge_pct": 0.06, "kelly_fraction": 0.125,
            "category_allowlist": [], "market_allowlist": [], "max_markets_per_cycle": 15,
        },
        "bitcoin": {
            "symbol": "BTC/USD", "timeframe": "1h", "trend_fast_ma": 20,
            "trend_slow_ma": 100, "vol_lookback": 30, "vol_size_scaling": True,
            "max_leverage": 1.0,
        },
    }


def run_smoke_test():
    from pods.bitcoin.signal_agent import BitcoinSignalAgent

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db = Database(tmp.name)
    config = make_config()

    pm_client = FakePolymarketClient()
    btc_client = FakeBitcoinClient()
    governor = Governor(db, config, execution_clients={"polymarket": pm_client, "bitcoin": btc_client})

    pm_agent = BiasedPolymarketAgent(pm_client, config)
    btc_agent = BitcoinSignalAgent(btc_client, config)

    orch = Orchestrator(governor, pm_agent, btc_agent, mode="paper")
    result = orch.run_cycle()

    assert result["status"] == "ok", result
    assert len(result["approved"]) >= 1, "expected at least one approved trade in the smoke test"
    print(f"OK: {len(result['approved'])} approved, {len(result['rejected'])} rejected")
    print(f"Equity after cycle: ${governor.current_equity_usd():.2f}")
    for a in result["approved"]:
        p, f = a["proposal"], a["fill"]
        print(f"  filled: [{p.leg}] {p.side} {p.market_or_symbol} size=${f['fill_size_usd']:.2f} price={f['fill_price']}")

    os.unlink(tmp.name)


if __name__ == "__main__":
    run_smoke_test()
