"""Survival dashboard: is it alive, and how close is it to dying?

Run:  python3 -m analysis.status
"""
import time
import yaml

from storage.db import Database


def build_status(db, config, price_lookup=None) -> dict:
    starting = db.get_state("starting_capital_usd", 0.0) or 0.0
    realized = db.sum_realized_pnl_since(0)
    equity = starting + realized

    unrealized = 0.0
    if price_lookup:
        for p in db.open_positions():
            price = price_lookup.get(p["market_or_symbol"])
            if price and p["entry_price"]:
                change = (price / p["entry_price"]) - 1
                pnl_pct = -change if p["side"] == "sell" else change
                unrealized += p["size_usd"] * pnl_pct

    mtm_equity = equity + unrealized
    hwm = db.get_state("high_water_mark_usd", starting) or starting
    reserve = db.get_state("operating_reserve_usd", 0.0)
    daily_cost = config.get("survival", {}).get("daily_operating_cost_usd", 0.0)
    ks = config["kill_switch"]

    drawdown = 1 - (mtm_equity / hwm) if hwm > 0 else 0.0
    cutoff = time.time() - ks["rolling_loss_window_days"] * 86400
    window_pnl = db.sum_realized_pnl_since(cutoff)

    birth_ts = db.get_state("birth_ts", time.time())
    return {
        "alive": not db.get_state("killed", False),
        "death_cause": db.get_state("killed_reason"),
        "generation": db.get_state("generation", 1),
        "age_days": (time.time() - birth_ts) / 86400.0,
        "starting_capital": starting,
        "equity": mtm_equity,
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_return": (mtm_equity / starting - 1) if starting else 0.0,
        "high_water_mark": hwm,
        "reserve": reserve,
        "daily_cost": daily_cost,
        "runway_days": (reserve / daily_cost) if daily_cost > 0 else float("inf"),
        "drawdown": drawdown,
        "drawdown_limit": ks["max_drawdown_pct"],
        "drawdown_headroom": ks["max_drawdown_pct"] - drawdown,
        "window_pnl": window_pnl,
        "window_loss_limit_usd": -ks["rolling_loss_pct"] * starting,
        "open_positions": len(db.open_positions()),
        "graveyard": db.graveyard(),
    }


def _bar(fraction, width=30):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(fraction * width)
    return "#" * filled + "-" * (width - filled)


def format_status(s: dict) -> str:
    lines = ["=" * 64]
    if s["alive"]:
        lines.append(f"STATUS: ALIVE   (generation {s['generation']}, age {s['age_days']:.1f} days)")
    else:
        lines.append(f"STATUS: DEAD    (generation {s['generation']})")
        lines.append(f"  cause: {s['death_cause']}")
    lines.append("=" * 64)
    lines.append(f"Capital    : ${s['starting_capital']:,.2f} -> ${s['equity']:,.2f}  ({s['total_return']:+.2%})")
    lines.append(f"  realized : ${s['realized_pnl']:+,.2f}")
    lines.append(f"  unrealized: ${s['unrealized_pnl']:+,.2f}  ({s['open_positions']} open positions)")
    lines.append("")
    lines.append("HOW CLOSE TO DEATH")
    lines.append(f"  drawdown   [{_bar(s['drawdown'] / s['drawdown_limit'] if s['drawdown_limit'] else 0)}] "
                  f"{s['drawdown']:.1%} of {s['drawdown_limit']:.0%} limit")

    runway = s["runway_days"]
    runway_str = "unlimited (no burn rate set)" if runway == float("inf") else f"{runway:.1f} days"
    runway_frac = 0.0 if runway == float("inf") else 1 - min(1.0, runway / 30)
    lines.append(f"  starvation [{_bar(runway_frac)}] reserve ${s['reserve']:.2f} "
                  f"= {runway_str} @ ${s['daily_cost']:.2f}/day")

    window_frac = 0.0
    if s["window_loss_limit_usd"] < 0:
        window_frac = max(0.0, s["window_pnl"] / s["window_loss_limit_usd"])
    lines.append(f"  loss limit [{_bar(window_frac)}] ${s['window_pnl']:+,.2f} "
                  f"of ${s['window_loss_limit_usd']:,.2f} allowed")
    lines.append("")

    if runway != float("inf") and s["alive"]:
        needed = s["daily_cost"] * 30
        lines.append(f"To survive the next 30 days it must earn ${needed:,.2f} in profit.")

    graves = s["graveyard"]
    if graves:
        lines.append("")
        lines.append(f"GRAVEYARD ({len(graves)} previous generation(s)):")
        for g in graves:
            lines.append(f"  gen {g['generation']}: lived {g['lifespan_days']:.1f}d, "
                          f"${g['starting_capital']:,.2f} -> ${g['final_equity']:,.2f} "
                          f"({g['pnl']:+,.2f}) -- {g['cause_of_death'][:50]}")
        total_burned = sum(g["pnl"] for g in graves)
        lines.append(f"  total capital destroyed across generations: ${total_burned:+,.2f}")
        if len(graves) >= 3 and total_burned < 0:
            lines.append("  NOTE: three or more generations have died losing money.")
            lines.append("  That is the experiment returning a result. Consider stopping.")
    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    with open("config/settings.yaml") as f:
        config = yaml.safe_load(f)
    db = Database(config["logging"]["db_path"])

    # Best-effort live pricing so the dashboard shows mark-to-market truth.
    price_lookup = {}
    try:
        import os
        from dotenv import load_dotenv
        from pods.bitcoin.client import BitcoinClient
        load_dotenv()
        positions = db.open_positions()
        if any(p["leg"] == "bitcoin" for p in positions):
            client = BitcoinClient(live_mode=False, symbol=config["bitcoin"]["symbol"],
                                    exchange_id=os.environ.get("BTC_EXCHANGE", "coinbase"))
            last = client.fetch_ticker()["last"]
            for p in positions:
                if p["leg"] == "bitcoin":
                    price_lookup[p["market_or_symbol"]] = last
    except Exception as e:
        print(f"(could not fetch live prices for mark-to-market: {e})\n")

    print(format_status(build_status(db, config, price_lookup)))


if __name__ == "__main__":
    main()
