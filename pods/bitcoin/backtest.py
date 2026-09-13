"""Walk-forward backtest for the Bitcoin mechanical signal.

Inspired by the backtest -> paper -> live discipline in
https://github.com/braedonsaunders/homerun -- you don't get to skip step one.

The benchmark that matters is BUY AND HOLD. A trend strategy that returns +20% while BTC
itself returned +60% has not made you money, it has cost you 40% plus fees for the
privilege of extra risk and effort. Most simple MA-crossover systems lose to buy-and-hold
on crypto; that is the honest null hypothesis this is here to test, not to confirm.

Run:  python3 -m pods.bitcoin.backtest
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("bitcoin_backtest")


def run_backtest(df: pd.DataFrame, cfg: dict, fee_rate: float = 0.006,
                  starting_equity: float = 1000.0) -> dict:
    """Walk forward bar by bar, flipping position when the signal flips.

    Uses only data up to bar i when deciding at bar i (no lookahead), and executes at the
    NEXT bar's open -- executing at the same close you made the decision on is the classic
    way backtests invent profits that don't exist in live trading.
    """
    fast_n = cfg["trend_fast_ma"]
    slow_n = cfg["trend_slow_ma"]
    vol_n = cfg["vol_lookback"]
    min_conf = cfg.get("min_confidence", 0.15)
    base_pct = 0.05

    close = df["close"]
    fast = close.rolling(fast_n).mean()
    slow = close.rolling(slow_n).mean()
    returns = close.pct_change()
    vol = returns.rolling(vol_n).std()
    median_vol = vol.median()

    equity = starting_equity
    position_side = None      # "buy" | "sell" | None
    position_size = 0.0
    entry_price = 0.0

    equity_curve = []
    trades = []

    start = max(fast_n, slow_n, vol_n) + 1
    for i in range(start, len(df) - 1):
        price_now = close.iloc[i]
        exec_price = df["open"].iloc[i + 1]  # execute at next bar's open

        if np.isnan(fast.iloc[i]) or np.isnan(slow.iloc[i]):
            equity_curve.append(equity)
            continue

        trend_up = fast.iloc[i] > slow.iloc[i]
        trend_strength = abs(fast.iloc[i] - slow.iloc[i]) / price_now
        confidence = min(trend_strength * 10, 1.0)
        desired = "buy" if trend_up else "sell"

        if desired != position_side and confidence >= min_conf:
            # Close existing position at exec_price.
            if position_side is not None:
                change = (exec_price / entry_price) - 1
                pnl_pct = change if position_side == "buy" else -change
                pnl = position_size * pnl_pct
                fee = position_size * fee_rate
                equity += pnl - fee
                trades.append({"side": position_side, "entry": entry_price,
                                "exit": exec_price, "pnl": pnl - fee})
                position_side = None

            # Open the new one.
            vol_scaling = 1.0
            if cfg.get("vol_size_scaling") and median_vol and median_vol > 0 and not np.isnan(vol.iloc[i]):
                vol_scaling = float(np.clip(median_vol / max(vol.iloc[i], 1e-9), 0.25, 1.5))
            position_size = equity * base_pct * vol_scaling
            equity -= position_size * fee_rate
            entry_price = exec_price
            position_side = desired

        equity_curve.append(equity)

    # Close any position still open at the end.
    if position_side is not None:
        final_price = close.iloc[-1]
        change = (final_price / entry_price) - 1
        pnl_pct = change if position_side == "buy" else -change
        equity += position_size * pnl_pct - position_size * fee_rate
        trades.append({"side": position_side, "entry": entry_price,
                        "exit": final_price, "pnl": position_size * pnl_pct})

    curve = pd.Series(equity_curve) if equity_curve else pd.Series([starting_equity])
    strat_return = (equity / starting_equity) - 1
    hold_return = (close.iloc[-1] / close.iloc[start]) - 1

    curve_returns = curve.pct_change().dropna()
    sharpe = float("nan")
    if len(curve_returns) > 1 and curve_returns.std() > 0:
        # Annualized assuming hourly bars.
        sharpe = (curve_returns.mean() / curve_returns.std()) * np.sqrt(24 * 365)

    running_max = curve.cummax()
    max_dd = float(((curve - running_max) / running_max).min()) if len(curve) else 0.0

    wins = [t for t in trades if t["pnl"] > 0]
    return {
        "bars": len(df),
        "final_equity": equity,
        "strategy_return": strat_return,
        "buy_hold_return": hold_return,
        "beats_buy_hold": strat_return > hold_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else float("nan"),
        "total_fees_paid": sum(t.get("fee", 0) for t in trades),
    }


def format_report(r: dict) -> str:
    lines = [
        "=" * 60,
        "BITCOIN SIGNAL BACKTEST",
        "=" * 60,
        f"Bars tested       : {r['bars']}",
        f"Trades            : {r['n_trades']}",
        f"Win rate          : {r['win_rate']:.1%}" if r["n_trades"] else "Win rate          : n/a",
        "",
        f"Strategy return   : {r['strategy_return']:+.2%}",
        f"Buy & hold return : {r['buy_hold_return']:+.2%}   <-- the benchmark",
        f"Sharpe (annualized): {r['sharpe']:.2f}",
        f"Max drawdown      : {r['max_drawdown']:.2%}",
        "",
    ]
    if r["beats_buy_hold"]:
        lines.append("VERDICT: beat buy-and-hold over this window.")
        lines.append("  One window is not evidence. Test other periods and regimes before")
        lines.append("  believing it -- it is easy to find a window where anything wins.")
    else:
        lines.append("VERDICT: did NOT beat buy-and-hold over this window.")
        lines.append("  On this evidence the signal destroys value versus simply holding.")
        lines.append("  Fix or replace the signal before allocating real capital to this leg.")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    import os
    import yaml
    from dotenv import load_dotenv
    from pods.bitcoin.client import BitcoinClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    load_dotenv()
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)

    client = BitcoinClient(live_mode=False, symbol=config["bitcoin"]["symbol"],
                            exchange_id=os.environ.get("BTC_EXCHANGE", "coinbase"))
    raw = client.fetch_ohlcv(timeframe=config["bitcoin"]["timeframe"], limit=1000)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    print(f"Fetched {len(df)} bars of {config['bitcoin']['symbol']} "
          f"@ {config['bitcoin']['timeframe']}")
    print(format_report(run_backtest(df, config["bitcoin"])))


if __name__ == "__main__":
    main()
