"""Mechanical arbitrage on Polymarket. No forecasting required.

This is structurally different from the LLM estimator, and better: it does not try to
predict anything. It looks for prices that are internally inconsistent, where the profit
is arithmetic and can be verified BEFORE placing a trade.

Two patterns:

1. BINARY COMPLEMENT. Every binary market has a YES and a NO token, and exactly one pays
   $1 at resolution. So YES + NO must cost $1. If you can buy both for $0.97, you have
   locked $0.03 regardless of the outcome -- you don't care who wins, you own both sides.

2. NEG-RISK BASKET. Polymarket "negRisk" events are mutually exclusive and exhaustive
   (exactly one outcome resolves YES). The YES prices across all outcomes must therefore
   sum to $1. If the basket costs $0.96, buying every outcome pays $1 whoever wins.

Why this is a better foundation than prediction: the market price already aggregates
thousands of informed traders, so beating it requires information they lack. Inconsistent
prices require no information at all -- only that you check.

What can still go wrong, and is handled:
  - You pay the ASK, not the mid. Mid-price arbitrage is imaginary.
  - Depth is finite. Size is capped by what's actually resting on the book.
  - Legs must fill together (bundle_id + Governor.review_bundle), because a half-filled
    basket is an unhedged directional bet.
  - Gas/fees eat thin edges, so min_profit_pct must exceed real costs.

The remaining risk in live trading is execution: prices can move between placing legs.
That is real and this code cannot eliminate it -- it can only keep the edge wide enough
that small slippage doesn't flip the sign.
"""
import logging
import uuid

from governor.models import Proposal

logger = logging.getLogger("polymarket_arbitrage")


class ArbitrageAgent:
    def __init__(self, client, config: dict):
        self.client = client
        self.cfg = config.get("arbitrage", {})
        self.pm_cfg = config.get("polymarket", {})

    # ---------- helpers ----------

    def _book(self, token_id: str):
        """Best ask price and the size available at it. Returns (price, size_usd)."""
        try:
            book = self.client.get_orderbook(token_id)
        except Exception as e:
            logger.debug("No book for %s: %s", token_id, e)
            return None, 0.0

        asks = book.get("asks") or []
        if not asks:
            return None, 0.0
        # Polymarket returns asks ascending; best (lowest) ask is what you pay.
        try:
            best = min(asks, key=lambda a: float(a["price"]))
            price = float(best["price"])
            size_usd = float(best["size"]) * price
            return price, size_usd
        except (KeyError, ValueError, TypeError):
            return None, 0.0

    def _tokens(self, market: dict):
        raw = market.get("clobTokenIds")
        if isinstance(raw, str):
            import json
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None, None
        if not raw or len(raw) < 2:
            return None, None
        return str(raw[0]), str(raw[1])

    # ---------- pattern 1: binary complement ----------

    def find_binary_arbs(self, markets: list) -> list:
        """YES_ask + NO_ask < $1 means both sides cost less than the guaranteed payout."""
        opportunities = []
        min_profit = self.cfg.get("min_profit_pct", 0.02)
        self.best_binary = None   # closest near-miss, for reporting

        for market in markets:
            yes_token, no_token = self._tokens(market)
            if not yes_token or not no_token:
                continue

            yes_price, yes_depth = self._book(yes_token)
            no_price, no_depth = self._book(no_token)
            if yes_price is None or no_price is None:
                continue

            basket_cost = yes_price + no_price
            profit_pct = 1.0 - basket_cost
            if self.best_binary is None or profit_pct > self.best_binary[0]:
                self.best_binary = (profit_pct, market.get("question", "")[:60])
            if profit_pct < min_profit:
                continue

            # Size is limited by the thinner side of the book.
            max_size = min(yes_depth, no_depth, self.cfg.get("max_position_usd", 100.0))
            if max_size < self.cfg.get("min_liquidity_usd", 20.0):
                logger.debug("Arb found but too thin ($%.2f) on %r", max_size, market.get("question"))
                continue

            opportunities.append({
                "type": "binary_complement",
                "question": market.get("question", ""),
                "profit_pct": profit_pct,
                "size_usd": max_size,
                "legs": [(yes_token, yes_price), (no_token, no_price)],
            })
            logger.info("BINARY ARB: %r costs $%.4f, pays $1.00 -> %.2f%% locked, $%.2f available",
                        market.get("question", "")[:60], basket_cost, profit_pct * 100, max_size)

        return opportunities

    # ---------- pattern 2: neg-risk basket ----------

    def find_basket_arbs(self, events: list) -> list:
        """For mutually exclusive + exhaustive events, all YES prices must sum to $1."""
        opportunities = []
        min_profit = self.cfg.get("min_profit_pct", 0.02)
        self.best_basket = None

        for event in events:
            # Only negRisk events are guaranteed mutually exclusive AND exhaustive.
            # Applying this to any other multi-market event is simply wrong: two outcomes
            # could both resolve YES, and the "arbitrage" would be a guaranteed loss.
            if not event.get("negRisk"):
                continue

            markets = event.get("markets") or []
            if len(markets) < 2:
                continue

            legs, total_cost, min_depth = [], 0.0, float("inf")
            usable = True
            for market in markets:
                yes_token, _ = self._tokens(market)
                if not yes_token:
                    usable = False
                    break
                price, depth = self._book(yes_token)
                if price is None:
                    usable = False
                    break
                legs.append((yes_token, price))
                total_cost += price
                min_depth = min(min_depth, depth)

            if not usable or not legs:
                continue

            profit_pct = 1.0 - total_cost
            if self.best_basket is None or profit_pct > self.best_basket[0]:
                self.best_basket = (profit_pct, event.get("title", "")[:60])
            if profit_pct < min_profit:
                continue

            max_size = min(min_depth, self.cfg.get("max_position_usd", 100.0))
            if max_size < self.cfg.get("min_liquidity_usd", 20.0):
                continue

            opportunities.append({
                "type": "negrisk_basket",
                "question": event.get("title", ""),
                "profit_pct": profit_pct,
                "size_usd": max_size,
                "legs": legs,
            })
            logger.info("BASKET ARB: %r basket costs $%.4f across %d outcomes -> %.2f%% locked",
                        event.get("title", "")[:60], total_cost, len(legs), profit_pct * 100)

        return opportunities

    # ---------- proposals ----------

    def generate_proposals(self, equity_usd: float) -> list:
        if not self.cfg.get("enabled", True):
            return []

        proposals = []
        try:
            markets = self.client.list_active_markets(limit=self.cfg.get("scan_markets", 50))
        except Exception as e:
            logger.warning("Could not fetch markets for arbitrage scan: %s", e)
            markets = []

        events = []
        try:
            events = self.client.list_active_events(limit=self.cfg.get("scan_events", 30))
        except Exception as e:
            logger.debug("Could not fetch events for basket arbitrage: %s", e)

        opportunities = self.find_binary_arbs(markets) + self.find_basket_arbs(events)

        for opp in opportunities:
            bundle_id = f"arb-{uuid.uuid4().hex[:10]}"
            # Split the sized amount across legs proportional to price, so the basket is
            # bought in equal SHARE counts -- that's what makes the payout uniform.
            total_price = sum(price for _, price in opp["legs"])
            if total_price <= 0:
                continue
            shares = opp["size_usd"] / total_price

            for token_id, price in opp["legs"]:
                proposals.append(Proposal(
                    leg="polymarket",
                    market_or_symbol=token_id,
                    side="buy_yes",           # arbitrage always BUYS the token it names
                    size_usd=round(shares * price, 2),
                    confidence=1.0,           # arithmetic, not a forecast
                    limit_price=price,
                    bundle_id=bundle_id,
                    rationale=(f"[{opp['type']}] {opp['question'][:70]} -- "
                               f"locked {opp['profit_pct']*100:.2f}% (basket "
                               f"${total_price:.4f} -> $1.00), leg @ {price:.4f}"),
                ))

        # Report the closest near-miss. "Found nothing" is ambiguous -- it could mean the
        # scan is broken, the threshold is unreachable, or the market is simply efficient.
        # The best observed spread distinguishes those, and shows whether min_profit_pct
        # is set somewhere reality can actually reach.
        best_bin = getattr(self, "best_binary", None)
        best_bas = getattr(self, "best_basket", None)
        logger.info("Arbitrage scan: %d markets, %d events -> %d opportunities "
                    "(best binary %s, best basket %s; threshold %.2f%%)",
                    len(markets), len(events), len(opportunities),
                    f"{best_bin[0]*100:+.2f}% on {best_bin[1]!r}" if best_bin else "none",
                    f"{best_bas[0]*100:+.2f}% on {best_bas[1]!r}" if best_bas else "none",
                    self.cfg.get("min_profit_pct", 0.02) * 100)
        return proposals
