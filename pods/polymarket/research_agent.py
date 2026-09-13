"""Polymarket research agent: scores active markets for edge and emits proposals.

model_probability() now delegates to an OpenRouter LLM call (see llm_estimator.py) when
OPENROUTER_API_KEY is set, and falls back to the market's own implied probability (i.e.
zero manufactured edge) otherwise. Either way, min_edge_pct / kelly_fraction downstream
are what actually protect you from a bad estimate -- they don't fix one.
"""
import json
import logging
from governor.models import Proposal
from pods.polymarket.llm_estimator import LLMEstimator

logger = logging.getLogger("polymarket_research_agent")


class PolymarketResearchAgent:
    def __init__(self, client, config: dict, db=None):
        self.client = client
        self.cfg = config["polymarket"]
        self.db = db  # optional: when present, every estimate is logged for calibration
        if self.cfg.get("use_debate"):
            from pods.polymarket.debate_estimator import DebateEstimator
            self.llm = DebateEstimator(config)
            logger.info("Using Bull/Bear/Judge debate estimator (3x LLM cost per market)")
        else:
            self.llm = LLMEstimator(config)

    @staticmethod
    def _parse_gamma_array(raw):
        """Several Gamma API fields (outcomePrices, clobTokenIds) come back as a
        STRINGIFIED JSON array -- the literal text '["a", "b"]' rather than a list.
        Indexing that raw string with [0] yields the character '[', which is how this
        codebase previously ended up trading a token id of '[' at a fake price of 0.5.
        Returns a real list, or None if it can't be parsed."""
        if raw is None:
            return None
        try:
            if isinstance(raw, str):
                raw = json.loads(raw)
            return list(raw)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def _outcome_tokens(self, market: dict):
        """Returns (yes_token_id, no_token_id) or (None, None)."""
        tokens = self._parse_gamma_array(market.get("clobTokenIds"))
        if not tokens or len(tokens) < 2:
            return None, None
        return str(tokens[0]), str(tokens[1])

    def _outcome_prices(self, market: dict):
        """Returns (yes_price, no_price) or (None, None)."""
        prices = self._parse_gamma_array(market.get("outcomePrices"))
        if not prices or len(prices) < 2:
            return None, None
        try:
            return float(prices[0]), float(prices[1])
        except (ValueError, TypeError):
            return None, None

    def _implied_prob(self, market: dict) -> float:
        """Polymarket's Gamma API returns outcomePrices as a STRINGIFIED JSON array
        (e.g. the literal text '["0.55", "0.45"]'), not an actual list. Indexing a raw
        string with [0] silently grabs the character '[' instead of a price and fails to
        parse as a float -- which used to fall back to 0.5 for every market, poisoning
        every edge calculation and position size with a fake 50/50 implied price. Handle
        both the (correct) list case and the (actual, observed) stringified case."""
        raw = market.get("outcomePrices")
        if raw is None:
            logger.warning("Market %r has no outcomePrices field", market.get("question", "?"))
            return None
        try:
            if isinstance(raw, str):
                raw = json.loads(raw)
            return float(raw[0])
        except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError) as e:
            logger.warning("Could not parse outcomePrices for market %r: %r (%s)",
                            market.get("question", "?"), raw, e)
            return None


    def _passes_trade_gates(self, implied, fair_value, edge, estimate):
        """Decide whether an estimate justifies a trade. Returns (will_trade, reason).

        An absolute edge threshold alone is not enough. 6 percentage points means very
        different things at different prices: at 0.50 it's a 12% relative disagreement
        with the market, but at 0.005 it's claiming the true probability is ~13x what
        thousands of traders have settled on. The second is not an edge, it's almost
        always the model being confidently wrong -- and longshot markets are exactly
        where LLMs hallucinate most and where liquidity is thinnest, so you get filled at
        a terrible price on your worst ideas. This cost us a real (paper) $15.62 bet on a
        0.45c market before the guard existed."""
        if estimate["confidence"] <= 0:
            return False, None  # normal fallback path, already logged upstream

        # The model's own confidence is a signal we should respect. In live runs it
        # reported 0.10 alongside reasoning like "no information about team strength is
        # provided, so I default to the market price" -- an explicit admission that it
        # knows nothing. Trading on that is trading on noise dressed as analysis.
        min_conf = self.cfg.get("min_llm_confidence", 0.40)
        if estimate["confidence"] < min_conf:
            return False, (f"model confidence {estimate['confidence']:.2f} below "
                            f"min_llm_confidence {min_conf} -- it told us it has no information")

        if abs(edge) < self.cfg["min_edge_pct"]:
            return False, None  # ordinary "no edge", not worth a log line

        min_price = self.cfg.get("min_price", 0.05)
        max_price = self.cfg.get("max_price", 0.95)
        if not (min_price <= implied <= max_price):
            return False, (f"market price {implied:.4f} outside tradeable band "
                            f"[{min_price}, {max_price}] -- extreme prices are illiquid and "
                            f"estimates there are unreliable")

        # Extraordinary claims: cap how far the model may disagree with the market in
        # RELATIVE terms, in both directions.
        max_ratio = self.cfg.get("max_odds_ratio", 3.0)
        if implied > 0 and fair_value > 0:
            ratio = max(fair_value / implied, implied / fair_value)
            if ratio > max_ratio:
                return False, (f"model claims {ratio:.1f}x the market's probability "
                                f"({implied:.3f} -> {fair_value:.3f}), above max_odds_ratio "
                                f"{max_ratio} -- treating as model error, not edge")

        return True, None

    def _log_estimate(self, market, question, implied, estimate, edge, traded):
        if self.db is None:
            return
        try:
            self.db.insert_estimate(
                market_id=str(market.get("id", "")),
                condition_id=market.get("conditionId", ""),
                slug=market.get("slug", ""),
                question=question,
                token_id=(market.get("clobTokenIds") or [None])[0],
                model_name=getattr(self.llm, "model", "unknown"),
                implied_prob=implied,
                model_prob=estimate["probability"],
                llm_confidence=estimate["confidence"],
                edge=edge,
                traded=traded,
                reasoning=estimate.get("reasoning", ""),
            )
        except Exception as e:
            # Never let calibration bookkeeping break the trading path.
            logger.warning("Failed to log estimate: %s", e)

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
            if implied is None:
                continue  # never trade on an unparseable/fake price

            question = market.get("question", "")
            description = market.get("description", "")

            estimate = self.llm.estimate(question, description, implied)
            fair_value = estimate["probability"]
            edge = fair_value - implied

            will_trade, reject_reason = self._passes_trade_gates(implied, fair_value, edge, estimate)
            if reject_reason:
                logger.info("No trade on %r: %s", question[:55], reject_reason)

            # Log EVERY estimate, not just the traded ones. Scoring only the trades you
            # took is survivorship bias -- it tells you nothing about whether the model
            # is actually calibrated. See analysis/calibration.py.
            self._log_estimate(market, question, implied, estimate, edge, will_trade)

            if not will_trade:
                continue

            sizing_confidence = min(abs(edge) * 2, 1.0) * estimate["confidence"]
            kelly_size = equity_usd * self.cfg["kelly_fraction"] * sizing_confidence

            yes_token, no_token = self._outcome_tokens(market)
            yes_price, no_price = self._outcome_prices(market)
            if not yes_token or not no_token or yes_price is None or no_price is None:
                logger.warning("Skipping market %r: unusable tokens/prices", question[:60])
                continue

            # On Polymarket you bet AGAINST an outcome by BUYING the NO token -- not by
            # selling YES (you can't sell a token you don't hold on the CLOB). So each
            # proposal names the specific token being bought and that token's own price.
            if edge > 0:
                side, token_id, token_price = "buy_yes", yes_token, yes_price
            else:
                side, token_id, token_price = "buy_no", no_token, no_price

            proposals.append(Proposal(
                leg="polymarket",
                market_or_symbol=token_id,
                side=side,
                size_usd=round(kelly_size, 2),
                confidence=sizing_confidence,
                limit_price=token_price,
                rationale=(
                    f"market='{question}' implied={implied:.3f} model={fair_value:.3f} "
                    f"edge={edge:+.3f} llm_confidence={estimate['confidence']:.2f} "
                    f"llm_reasoning='{estimate['reasoning']}'"
                ),
            ))

        logger.info("Polymarket agent produced %d proposals from %d markets", len(proposals), len(markets))
        return proposals
