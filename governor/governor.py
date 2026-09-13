import time
import logging

from governor.models import Proposal, GovernorDecision

logger = logging.getLogger("governor")


class KillSwitchTriggered(Exception):
    pass


class Governor:
    def __init__(self, db, config: dict, execution_clients: dict):
        self.db = db
        self.cfg = config
        self.execution_clients = execution_clients

        if self.db.get_state("starting_capital_usd") is None:
            starting = config["treasury"]["starting_capital_usd"]
            self.db.set_state("starting_capital_usd", starting)
            self.db.set_state("high_water_mark_usd", starting)
            self.db.set_state("killed", False)
            # The reserve is the agent's runway: enough rent to live for N days while it
            # tries to become profitable. After that it must have earned its keep.
            survival_cfg = config.get("survival", {})
            daily_cost = survival_cfg.get("daily_operating_cost_usd", 0.0)
            runway_days = survival_cfg.get("initial_runway_days", 30)
            seed_reserve = daily_cost * runway_days
            self.db.set_state("operating_reserve_usd", seed_reserve)
            self.db.set_state("birth_ts", time.time())
            self.db.set_state("last_metabolic_charge_ts", time.time())
            self.db.set_state("generation", self.db.get_state("generation", 1))
            self.db.record_ledger_event("deposit", starting, starting, note="initial capital")
            logger.info("Born: generation %s, $%.2f capital, $%.2f reserve (%s days runway @ $%.2f/day)",
                        self.db.get_state("generation", 1), starting, seed_reserve,
                        runway_days, daily_cost)

    def current_equity_usd(self) -> float:
        starting = self.db.get_state("starting_capital_usd")
        realized = self.db.sum_realized_pnl_since(0)
        return starting + realized

    def update_high_water_mark(self):
        equity = self.current_equity_usd()
        hwm = self.db.get_state("high_water_mark_usd", equity)
        if equity > hwm:
            self.db.set_state("high_water_mark_usd", equity)
        return max(equity, hwm)

    def is_killed(self) -> bool:
        return bool(self.db.get_state("killed", False))

    def resume(self, operator_note: str):
        equity = self.current_equity_usd()
        self.db.set_state("killed", False)
        self.db.set_state("high_water_mark_usd", equity)
        self.db.record_ledger_event("resume", 0, equity, note=operator_note)
        logger.warning("SYSTEM RESUMED by operator: %s", operator_note)

    def resurrect(self, new_capital_usd: float, operator_note: str):
        """Start a NEW generation with fresh capital. The dead generation stays in the
        graveyard -- this is a successor, not a revival. The distinction is the point: if
        you keep feeding capital to a strategy that keeps dying, the graveyard is the
        evidence telling you to stop."""
        generation = self.db.get_state("generation", 1) + 1
        self.db.set_state("generation", generation)
        self.db.set_state("starting_capital_usd", new_capital_usd)
        self.db.set_state("high_water_mark_usd", new_capital_usd)
        self.db.set_state("killed", False)
        self.db.set_state("killed_reason", None)
        self.db.set_state("birth_ts", time.time())
        self.db.set_state("last_metabolic_charge_ts", time.time())

        s = self.cfg.get("survival", {})
        self.db.set_state("operating_reserve_usd",
                          s.get("daily_operating_cost_usd", 0.0) * s.get("initial_runway_days", 30))
        self.db.reset_pnl_baseline()
        self.db.record_ledger_event("deposit", new_capital_usd, new_capital_usd,
                                     note=f"generation {generation}: {operator_note}")
        logger.warning("RESURRECTED as generation %s with $%.2f", generation, new_capital_usd)

    # ---------- mark to market ----------

    def unrealized_pnl_usd(self, price_lookup: dict) -> float:
        """Unrealized P&L across open positions, given {market_or_symbol: current_price}.

        Without this the agent is blind to its own open losses: equity based only on
        REALIZED P&L means a position 90% underwater looks identical to no position at
        all, and the kill-switch never fires so long as nothing is ever closed. That is
        the easiest way for a 'self-terminating' system to quietly fail to terminate."""
        total = 0.0
        for position in self.db.open_positions():
            price = price_lookup.get(position["market_or_symbol"])
            entry = position["entry_price"]
            if price is None or not entry or entry <= 0:
                continue
            change = (price / entry) - 1
            pnl_pct = -change if position["side"] == "sell" else change
            total += position["size_usd"] * pnl_pct
        return total

    def mark_to_market_equity(self, price_lookup: dict = None) -> float:
        """The number that actually decides whether this thing lives."""
        equity = self.current_equity_usd()
        if price_lookup:
            equity += self.unrealized_pnl_usd(price_lookup)
        return equity

    # ---------- metabolic cost: the 'must earn to live' mechanic ----------

    def charge_metabolic_cost(self) -> float:
        """Charge rent for existing, prorated since the last charge.

        This is what makes survival mean anything. With no burn rate, an agent that never
        trades survives forever -- that is not 'survives only if it makes money', it is
        just 'survives'. Rent comes out of the operating reserve, and the reserve is
        refilled ONLY by swept profits. No profit, no reserve, death by starvation."""
        s = self.cfg.get("survival", {})
        daily_cost = s.get("daily_operating_cost_usd", 0.0)
        if daily_cost <= 0:
            return 0.0

        now = time.time()
        last = self.db.get_state("last_metabolic_charge_ts", now)
        hours = min(max(0.0, (now - last) / 3600.0), 24 * 7)  # cap so a long gap can't insta-starve
        charge = daily_cost * (hours / 24.0)
        self.db.set_state("last_metabolic_charge_ts", now)
        if charge <= 0:
            return 0.0

        reserve = self.db.get_state("operating_reserve_usd", 0.0)
        self.db.set_state("operating_reserve_usd", reserve - charge)
        self.db.record_ledger_event("metabolic_cost", -charge, self.current_equity_usd(),
                                     note=f"rent for {hours:.2f}h of existence")
        logger.info("Metabolic cost: -$%.4f for %.2fh (reserve now $%.2f)",
                    charge, hours, reserve - charge)
        return charge

    def charge_llm_spend(self, amount_usd: float):
        """Debit real LLM API spend. The agent pays for its own thinking."""
        if amount_usd <= 0:
            return
        reserve = self.db.get_state("operating_reserve_usd", 0.0)
        self.db.set_state("operating_reserve_usd", reserve - amount_usd)
        self.db.record_ledger_event("llm_spend", -amount_usd, self.current_equity_usd(),
                                     note="OpenRouter API spend")

    def _liquidate_all(self, price_lookup: dict = None):
        """Flatten every open position. A 'dead' agent still holding risk is not dead."""
        price_lookup = price_lookup or {}
        for position in self.db.open_positions():
            price = price_lookup.get(position["market_or_symbol"])
            entry = position["entry_price"]
            if price is None or not entry or entry <= 0:
                logger.warning(
                    "No price for %s at liquidation; closing at entry (0 P&L booked). "
                    "The real position may still be open at the venue -- check manually.",
                    position["market_or_symbol"])
                self.settle_realized_pnl(position["id"], 0.0, leg=position["leg"])
                continue
            change = (price / entry) - 1
            pnl_pct = -change if position["side"] == "sell" else change
            pnl = position["size_usd"] * pnl_pct
            self.settle_realized_pnl(position["id"], pnl, leg=position["leg"])
            logger.info("LIQUIDATED %s: pnl=$%.2f", position["market_or_symbol"], pnl)

    def _kill(self, reason: str, price_lookup: dict = None):
        logger.error("=" * 60)
        logger.error("DEATH: %s", reason)
        logger.error("=" * 60)

        self._liquidate_all(price_lookup)

        equity = self.current_equity_usd()
        generation = self.db.get_state("generation", 1)
        birth_ts = self.db.get_state("birth_ts", time.time())
        starting = self.db.get_state("starting_capital_usd", 0.0) or 0.0
        lifespan_days = (time.time() - birth_ts) / 86400.0

        self.db.set_state("killed", True)
        self.db.set_state("killed_reason", reason)
        self.db.set_state("killed_at", time.time())
        self.db.record_ledger_event("kill", 0, equity, note=reason)
        self.db.record_death(generation=generation, birth_ts=birth_ts, death_ts=time.time(),
                              starting_capital=starting, final_equity=equity, cause=reason)

        logger.error("Generation %s lived %.2f days. $%.2f -> $%.2f (%+.1f%%). All positions flattened.",
                     generation, lifespan_days, starting, equity,
                     ((equity / starting) - 1) * 100 if starting else 0.0)
        logger.error("It will not trade again. To start a new generation with fresh capital:")
        logger.error("  python3 -m analysis.resurrect --capital <amount>")

    def check_kill_switch(self, price_lookup: dict = None):
        if self.is_killed():
            return

        # Mark to market: open losses count against survival immediately, not only once
        # somebody chooses to realize them.
        equity = self.mark_to_market_equity(price_lookup)
        hwm = self.update_high_water_mark()
        ks_cfg = self.cfg["kill_switch"]

        drawdown = 1 - (equity / hwm) if hwm > 0 else 0
        if drawdown >= ks_cfg["max_drawdown_pct"]:
            self._kill(f"max_drawdown_pct exceeded (mark-to-market): {drawdown:.1%} >= "
                        f"{ks_cfg['max_drawdown_pct']:.1%}", price_lookup)
            return

        window_days = ks_cfg["rolling_loss_window_days"]
        cutoff = time.time() - window_days * 86400
        window_pnl = self.db.sum_realized_pnl_since(cutoff)
        starting = self.db.get_state("starting_capital_usd")
        if starting > 0 and (window_pnl / starting) <= -ks_cfg["rolling_loss_pct"]:
            self._kill(
                f"rolling_loss_pct exceeded over {window_days}d: "
                f"{window_pnl/starting:.1%} <= -{ks_cfg['rolling_loss_pct']:.1%}",
                price_lookup,
            )
            return

        # Starvation: the reserve pays rent + API costs and is refilled ONLY by swept
        # profits. An agent that never earns eventually cannot pay to exist.
        reserve = self.db.get_state("operating_reserve_usd", 0.0)
        if reserve <= 0:
            self._kill(f"starvation: operating reserve exhausted (${reserve:.2f}) -- could not "
                        f"earn enough to pay for its own existence", price_lookup)
            return

    def leg_allocation_multiplier(self, leg: str) -> float:
        cfg = self.cfg["allocation"]
        cutoff = time.time() - cfg["rebalance_frequency_days"] * 86400 * 4
        leg_pnl = self.db.sum_realized_pnl_since(cutoff, leg=leg)
        starting = self.db.get_state("starting_capital_usd")
        leg_capital = starting * cfg.get(f"{leg}_weight", 0.5)
        if leg_capital <= 0:
            return 0.0
        leg_drawdown = max(0.0, -leg_pnl / leg_capital)
        if leg_drawdown >= cfg["per_leg_kill_drawdown_pct"]:
            logger.warning("Leg '%s' allocation cut to zero: drawdown %.1f%%", leg, leg_drawdown * 100)
            return 0.0
        return 1.0

    def review(self, proposal: Proposal) -> GovernorDecision:
        if self.is_killed():
            return GovernorDecision(False, proposal, reason="system is killed; awaiting manual resume()")

        if self.db.proposal_exists(proposal.idempotency_key()):
            return GovernorDecision(False, proposal, reason="duplicate proposal (idempotency key seen)")

        if time.time() > proposal.expiry_ts:
            return GovernorDecision(False, proposal, reason="proposal expired before review")

        if self.leg_allocation_multiplier(proposal.leg) <= 0:
            return GovernorDecision(False, proposal, reason=f"leg '{proposal.leg}' allocation is zeroed out")

        equity = self.current_equity_usd()
        limits = self.cfg["position_limits"]

        max_single = equity * limits["max_single_position_pct"]
        approved_size = min(proposal.size_usd, max_single)

        current_exposure = self.db.total_open_exposure_usd()
        max_total = equity * limits["max_total_exposure_pct"]
        if current_exposure + approved_size > max_total:
            approved_size = max(0.0, max_total - current_exposure)
            if approved_size <= 0:
                return GovernorDecision(False, proposal, reason="max_total_exposure_pct reached")

        today_cutoff = time.time() - 86400
        opened_today = self.db.notional_opened_since(today_cutoff)
        max_daily = equity * limits["max_daily_notional_pct"]
        if opened_today + approved_size > max_daily:
            approved_size = max(0.0, max_daily - opened_today)
            if approved_size <= 0:
                return GovernorDecision(False, proposal, reason="max_daily_notional_pct reached")

        if proposal.leg == "polymarket":
            allowlist = self.cfg["polymarket"].get("market_allowlist") or []
            if allowlist and proposal.market_or_symbol not in allowlist:
                return GovernorDecision(False, proposal, reason="market not in allowlist")

        return GovernorDecision(True, proposal, reason="approved", approved_size_usd=round(approved_size, 2))

    def execute(self, decision: GovernorDecision, mode: str) -> dict:
        if not decision.approved:
            raise ValueError("execute() called on a non-approved decision")

        proposal = decision.proposal
        proposal_id = self.db.insert_proposal(
            leg=proposal.leg,
            idempotency_key=proposal.idempotency_key(),
            market_or_symbol=proposal.market_or_symbol,
            side=proposal.side,
            size_usd=decision.approved_size_usd,
            limit_price=proposal.limit_price,
            confidence=proposal.confidence,
            rationale=proposal.rationale,
            status="approved",
            raw_json=str(proposal),
        )

        client = self.execution_clients.get(proposal.leg)
        fill = client.execute(proposal, decision.approved_size_usd, mode=mode)

        self.db.insert_fill(
            proposal_id=proposal_id,
            fill_price=fill["fill_price"],
            fill_size_usd=fill["fill_size_usd"],
            fees_usd=fill["fees_usd"],
            mode=mode,
            exchange_order_id=fill.get("order_id"),
        )
        self.db.update_proposal_status(proposal_id, "filled")
        self.db.open_position(
            leg=proposal.leg,
            market_or_symbol=proposal.market_or_symbol,
            side=proposal.side,
            size_usd=fill["fill_size_usd"],
            entry_price=fill["fill_price"],
        )

        reserve = self.db.get_state("operating_reserve_usd", 0.0)
        self.db.set_state("operating_reserve_usd", reserve - fill["fees_usd"])

        logger.info(
            "EXECUTED [%s] %s %s size=$%.2f price=%.4f fees=$%.4f mode=%s",
            proposal.leg, proposal.side, proposal.market_or_symbol,
            fill["fill_size_usd"], fill["fill_price"], fill["fees_usd"], mode,
        )
        return fill

    def close_positions_for_leg(self, leg: str, current_price: float) -> float:
        """Mark-to-market close every open position in a leg at current_price, realizing
        P&L through settle_realized_pnl for each. Returns total realized P&L.

        This exists because nothing was previously closing Bitcoin positions when the
        trend signal flipped -- new positions just kept opening on top of old ones,
        exposure grew without bound, and realized_pnl (which the entire kill-switch is
        built on) never moved from BTC trades at all. Call this before opening an
        opposite-side position, not after -- old exposure should be gone before new
        exposure is added, so max_total_exposure_pct in review() sees the real picture."""
        total_pnl = 0.0
        for position in self.db.open_positions(leg=leg):
            entry_price = position["entry_price"]
            size_usd = position["size_usd"]
            if entry_price <= 0:
                logger.warning("Skipping position %s with invalid entry_price=%s", position["id"], entry_price)
                continue
            price_change_pct = (current_price / entry_price) - 1
            # "buy" (long) profits when price rises; "sell" (short, paper-simulated only
            # -- there is no real short mechanism here) profits when price falls.
            pnl_pct = price_change_pct if position["side"] == "buy" else -price_change_pct
            pnl_usd = size_usd * pnl_pct
            self.settle_realized_pnl(position["id"], pnl_usd, leg=leg)
            total_pnl += pnl_usd
            logger.info(
                "CLOSED [%s] position %s: entry=%.4f exit=%.4f size=$%.2f pnl=$%.2f",
                leg, position["id"], entry_price, current_price, size_usd, pnl_usd,
            )
        return total_pnl

    def settle_realized_pnl(self, position_id: int, realized_pnl_usd: float, leg: str):
        self.db.close_position(position_id, realized_pnl_usd)
        equity = self.current_equity_usd() + realized_pnl_usd
        self.db.record_ledger_event("realized_pnl", realized_pnl_usd, equity, leg=leg)

        if realized_pnl_usd > 0:
            sweep_pct = self.cfg["treasury"]["operating_reserve_pct"]
            sweep = realized_pnl_usd * sweep_pct
            reserve = self.db.get_state("operating_reserve_usd", 0.0)
            self.db.set_state("operating_reserve_usd", reserve + sweep)
            self.db.record_ledger_event("reserve_sweep", sweep, equity, leg=leg,
                                         note="profit swept to operating reserve")
