import logging

logger = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, governor, polymarket_agent, bitcoin_agent, mode: str):
        self.governor = governor
        self.polymarket_agent = polymarket_agent
        self.bitcoin_agent = bitcoin_agent
        self.mode = mode

    def _build_price_lookup(self) -> dict:
        """Current price for every open position, so equity can be marked to market.

        Best-effort: a position we can't price is simply excluded from unrealized P&L
        rather than assumed flat, and that's logged -- silently treating an unpriceable
        position as break-even is how a dying system looks healthy."""
        lookup = {}
        positions = self.governor.db.open_positions()
        if not positions:
            return lookup

        btc_symbols = {p["market_or_symbol"] for p in positions if p["leg"] == "bitcoin"}
        if btc_symbols:
            try:
                ticker = self.bitcoin_agent.client.fetch_ticker()
                for symbol in btc_symbols:
                    lookup[symbol] = ticker["last"]
            except Exception as e:
                logger.warning("Could not price bitcoin positions: %s", e)

        for position in positions:
            if position["leg"] != "polymarket":
                continue
            token_id = position["market_or_symbol"]
            try:
                prices = self.polymarket_agent.client.get_orderbook_prices(token_id)
                if prices.get("mid") is not None:
                    lookup[token_id] = prices["mid"]
                else:
                    logger.warning("No mid price for token %s; excluded from mark-to-market", token_id)
            except Exception as e:
                logger.warning("Could not price polymarket token %s: %s", token_id, e)

        return lookup

    def run_cycle(self):
        logger.info("=== Orchestration cycle start (mode=%s) ===", self.mode)

        # 1. Pay rent for existing. This happens BEFORE anything else and regardless of
        #    whether the agent trades -- existing is not free, which is the whole point.
        self.governor.charge_metabolic_cost()

        # 2. Price every open position so the kill-switch sees mark-to-market reality,
        #    not just realized P&L.
        price_lookup = self._build_price_lookup()

        self.governor.check_kill_switch(price_lookup)
        if self.governor.is_killed():
            reason = self.governor.db.get_state("killed_reason", "unknown")
            logger.error("System is KILLED (%s). No new proposals will be evaluated. "
                         "Call governor.resume() after review to continue.", reason)
            return {"status": "killed", "reason": reason}

        equity = self.governor.mark_to_market_equity(price_lookup)
        reserve = self.governor.db.get_state("operating_reserve_usd", 0.0)
        daily_cost = self.governor.cfg.get("survival", {}).get("daily_operating_cost_usd", 0.0)
        runway = (reserve / daily_cost) if daily_cost > 0 else float("inf")
        logger.info("Equity (mark-to-market): $%.2f | reserve: $%.2f | runway: %.1f days",
                    equity, reserve, runway)

        pm_multiplier = self.governor.leg_allocation_multiplier("polymarket")
        btc_multiplier = self.governor.leg_allocation_multiplier("bitcoin")

        alloc_cfg = self.governor.cfg["allocation"]
        pm_equity = equity * alloc_cfg["polymarket_weight"] * pm_multiplier
        btc_equity = equity * alloc_cfg["bitcoin_weight"] * btc_multiplier

        proposals = []
        if pm_multiplier > 0:
            proposals += self.polymarket_agent.generate_proposals(pm_equity)
        else:
            logger.warning("Polymarket leg allocation is zeroed out; skipping proposal generation")

        if btc_multiplier > 0:
            open_btc = self.governor.db.open_positions(leg="bitcoin")
            current_side = open_btc[0]["side"] if open_btc else None
            btc_proposals = self.bitcoin_agent.generate_proposals(btc_equity, current_side=current_side)

            # If the signal is flipping sides, close the existing position (realizing its
            # actual P&L) BEFORE the new opposite-side proposal is reviewed -- otherwise
            # exposure only ever grows and realized_pnl never reflects BTC trades at all.
            if btc_proposals and open_btc:
                new_side = btc_proposals[0].side
                if new_side != current_side:
                    current_price = self.bitcoin_agent.client.fetch_ticker()["last"]
                    self.governor.close_positions_for_leg("bitcoin", current_price)

            proposals += btc_proposals
        else:
            logger.warning("Bitcoin leg allocation is zeroed out; skipping proposal generation")

        # 3. The agent pays for its own thinking.
        llm_spend = getattr(self.polymarket_agent, "llm", None)
        if llm_spend is not None and self.governor.cfg.get("survival", {}).get("charge_llm_spend", True):
            self.governor.charge_llm_spend(getattr(llm_spend, "_spent_this_cycle", 0.0))

        results = {"approved": [], "rejected": []}
        for proposal in proposals:
            decision = self.governor.review(proposal)
            if decision.approved:
                fill = self.governor.execute(decision, mode=self.mode)
                results["approved"].append({"proposal": proposal, "fill": fill})
            else:
                logger.info("REJECTED [%s] %s: %s", proposal.leg, proposal.market_or_symbol, decision.reason)
                results["rejected"].append({"proposal": proposal, "reason": decision.reason})

        logger.info("=== Cycle complete: %d approved, %d rejected ===",
                    len(results["approved"]), len(results["rejected"]))
        return {"status": "ok", **results}
