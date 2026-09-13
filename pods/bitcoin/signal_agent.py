import logging
import numpy as np
import pandas as pd
from governor.models import Proposal

logger = logging.getLogger("bitcoin_signal_agent")


class BitcoinSignalAgent:
    def __init__(self, client, config: dict):
        self.client = client
        self.cfg = config["bitcoin"]

    def _ohlcv_df(self) -> pd.DataFrame:
        raw = self.client.fetch_ohlcv(
            timeframe=self.cfg["timeframe"],
            limit=max(self.cfg["trend_slow_ma"], self.cfg["vol_lookback"]) + 20,
        )
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        return df

    def news_dampener(self) -> float:
        return 1.0

    def compute_signal(self, df: pd.DataFrame) -> dict:
        fast = df["close"].rolling(self.cfg["trend_fast_ma"]).mean()
        slow = df["close"].rolling(self.cfg["trend_slow_ma"]).mean()
        returns = df["close"].pct_change()
        vol = returns.rolling(self.cfg["vol_lookback"]).std()

        last_fast, last_slow = fast.iloc[-1], slow.iloc[-1]
        last_vol = vol.iloc[-1]
        median_vol = vol.median()

        trend_up = last_fast > last_slow
        trend_strength = abs(last_fast - last_slow) / df["close"].iloc[-1]

        vol_scaling = 1.0
        if self.cfg.get("vol_size_scaling") and median_vol and not np.isnan(median_vol) and median_vol > 0:
            vol_scaling = float(np.clip(median_vol / max(last_vol, 1e-9), 0.25, 1.5))

        return {
            "trend_up": bool(trend_up),
            "trend_strength": float(trend_strength) if not np.isnan(trend_strength) else 0.0,
            "vol_scaling": vol_scaling,
            "last_price": float(df["close"].iloc[-1]),
        }

    def generate_proposals(self, equity_usd: float, current_side: str = None) -> list:
        df = self._ohlcv_df()
        if df["close"].isna().all() or len(df) < self.cfg["trend_slow_ma"]:
            logger.warning("Not enough BTC data yet for signal computation")
            return []

        signal = self.compute_signal(df)
        base_pct = 0.05
        dampener = self.news_dampener()
        size_pct = base_pct * signal["vol_scaling"] * dampener
        size_usd = round(equity_usd * size_pct, 2)

        desired_side = "buy" if signal["trend_up"] else "sell"
        if desired_side == current_side:
            return []

        confidence = min(signal["trend_strength"] * 10, 1.0)
        proposal = Proposal(
            leg="bitcoin",
            market_or_symbol=self.cfg["symbol"],
            side=desired_side,
            size_usd=size_usd,
            confidence=confidence,
            limit_price=signal["last_price"],
            rationale=(
                f"trend_up={signal['trend_up']} strength={signal['trend_strength']:.4f} "
                f"vol_scaling={signal['vol_scaling']:.2f} dampener={dampener:.2f}"
            ),
        )
        logger.info("Bitcoin agent proposal: %s $%.2f (confidence=%.2f)", desired_side, size_usd, confidence)
        return [proposal]
