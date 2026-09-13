import logging
from governor.models import Proposal

logger = logging.getLogger("polymarket_research_agent")


class PolymarketResearchAgent:
    def __init__(self, client, config: dict):
        self.client = client
        self.cfg = config["polymarket"]

    def model_probability(self, market: dict) -> float:
        """PLACEHOLDER. Replace with a real estimate (LLM + research tools, or a
        statistical model) before running live."""
        return market.get("_implied_prob", 0.5)

    def _implied_prob(self, market: dict) -> float:
        try:
            return float(market["outcomePrices"][0])
        except (KeyError, IndexError, ValueError, TypeError):
            return 0.5

    def generate_proposals(self, equity_usd: float) -> list:
        proposals = []
        try:
            markets = self.client.list_active_markets(limit=self.cfg["max_markets_per_cycle"])
        except Exception as e:
            logger.warning("Failed to fetch Polymarket markets: %s", e)
            return proposals

        allowlist = set(self.cfg.get("category_allowlist") or [])

        for market in markets:
            category = market.get("category", "")
            if allowlist and category not in allowlist:
                continue

            market["_implied_prob"] = self._implied_prob(market)
            fair_value = self.model_probability(market)
            implied = market["_implied_prob"]
            edge = fair_value - implied

            if abs(edge) < self.cfg["min_edge_pct"]:
                continue

            side = "buy_yes" if edge > 0 else "buy_no"
            confidence = min(abs(edge) * 2, 1.0)
            kelly_size = equity_usd * self.cfg["kelly_fraction"] * confidence

            token_id = market.get("clobTokenIds", [None])[0]
            if token_id is None:
                continue

            proposals.append(Proposal(
                leg="polymarket",
                market_or_symbol=token_id,
                side=side,
                size_usd=round(kelly_size, 2),
                confidence=confidence,
                limit_price=implied,
                rationale=(
                    f"market='{market.get('question','?')}' implied={implied:.3f} "
                    f"model={fair_value:.3f} edge={edge:+.3f}"
                ),
            ))

        logger.info("Polymarket agent produced %d proposals from %d markets", len(proposals), len(markets))
        return proposals
