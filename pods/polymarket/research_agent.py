"""Polymarket research agent: scores active markets for edge and emits proposals.

model_probability() now delegates to an OpenRouter LLM call (see llm_estimator.py) when
OPENROUTER_API_KEY is set, and falls back to the market's own implied probability (i.e.
zero manufactured edge) otherwise. Either way, min_edge_pct / kelly_fraction downstream
are what actually protect you from a bad estimate -- they don't fix one.
"""
import logging
from governor.models import Proposal
from pods.polymarket.llm_estimator import LLMEstimator

logger = logging.getLogger("polymarket_research_agent")


class PolymarketResearchAgent:
    def __init__(self, client, config: dict):
        self.client = client
        self.cfg = config["polymarket"]
        self.llm = LLMEstimator(config)

    def _implied_prob(self, market: dict) -> float:
        try:
            return float(market["outcomePrices"][0])
        except (KeyError, IndexError, ValueError, TypeError):
            return 0.5

    def generate_proposals(self, equity_usd: float) -> list:
        proposals = []
        self.llm.reset_cycle_spend()

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

            implied = self._implied_prob(market)
            question = market.get("question", "")
            description = market.get("description", "")

            estimate = self.llm.estimate(question, description, implied)
            fair_value = estimate["probability"]
            edge = fair_value - implied

            if abs(edge) < self.cfg["min_edge_pct"]:
                continue

            # Low LLM confidence should shrink size even if the raw edge looks big --
            # confidence 0 (no API key / fallback / error) means edge is meaningless here.
            if estimate["confidence"] <= 0:
                continue

            side = "buy_yes" if edge > 0 else "buy_no"
            sizing_confidence = min(abs(edge) * 2, 1.0) * estimate["confidence"]
            kelly_size = equity_usd * self.cfg["kelly_fraction"] * sizing_confidence

            token_id = market.get("clobTokenIds", [None])[0]
            if token_id is None:
                continue

            proposals.append(Proposal(
                leg="polymarket",
                market_or_symbol=token_id,
                side=side,
                size_usd=round(kelly_size, 2),
                confidence=sizing_confidence,
                limit_price=implied,
                rationale=(
                    f"market='{question}' implied={implied:.3f} model={fair_value:.3f} "
                    f"edge={edge:+.3f} llm_confidence={estimate['confidence']:.2f} "
                    f"llm_reasoning='{estimate['reasoning']}'"
                ),
            ))

        logger.info("Polymarket agent produced %d proposals from %d markets", len(proposals), len(markets))
        return proposals
