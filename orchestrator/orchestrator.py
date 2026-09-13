import logging

logger = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, governor, polymarket_agent, bitcoin_agent, mode: str):
        self.governor = governor
        self.polymarket_agent = polymarket_agent
        self.bitcoin_agent = bitcoin_agent
        self.mode = mode

    def run_cycle(self):
        logger.info("=== Orchestration cycle start (mode=%s) ===", self.mode)

        self.governor.check_kill_switch()
        if self.governor.is_killed():
            reason = self.governor.db.get_state("killed_reason", "unknown")
            logger.error("System is KILLED (%s). No new proposals will be evaluated. "
                         "Call governor.resume() after review to continue.", reason)
            return {"status": "killed", "reason": reason}

        equity = self.governor.current_equity_usd()
        logger.info("Current equity: $%.2f", equity)

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
            proposals += self.bitcoin_agent.generate_proposals(btc_equity, current_side=current_side)
        else:
            logger.warning("Bitcoin leg allocation is zeroed out; skipping proposal generation")

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
