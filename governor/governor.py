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
            seed_reserve = config["kill_switch"]["min_operating_reserve_usd"]
            self.db.set_state("operating_reserve_usd", seed_reserve)
            self.db.record_ledger_event("deposit", starting, starting, note="initial capital")

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

    def _kill(self, reason: str):
        equity = self.current_equity_usd()
        self.db.set_state("killed", True)
        self.db.set_state("killed_reason", reason)
        self.db.set_state("killed_at", time.time())
        self.db.record_ledger_event("kill", 0, equity, note=reason)
        logger.error("KILL SWITCH TRIGGERED: %s (equity=$%.2f)", reason, equity)

    def check_kill_switch(self):
        if self.is_killed():
            return

        equity = self.current_equity_usd()
        hwm = self.update_high_water_mark()
        ks_cfg = self.cfg["kill_switch"]

        drawdown = 1 - (equity / hwm) if hwm > 0 else 0
        if drawdown >= ks_cfg["max_drawdown_pct"]:
            self._kill(f"max_drawdown_pct exceeded: {drawdown:.1%} >= {ks_cfg['max_drawdown_pct']:.1%}")
            return

        window_days = ks_cfg["rolling_loss_window_days"]
        cutoff = time.time() - window_days * 86400
        window_pnl = self.db.sum_realized_pnl_since(cutoff)
        starting = self.db.get_state("starting_capital_usd")
        if starting > 0 and (window_pnl / starting) <= -ks_cfg["rolling_loss_pct"]:
            self._kill(
                f"rolling_loss_pct exceeded over {window_days}d: "
                f"{window_pnl/starting:.1%} <= -{ks_cfg['rolling_loss_pct']:.1%}"
            )
            return

        reserve = self.db.get_state("operating_reserve_usd", 0.0)
        if reserve < ks_cfg["min_operating_reserve_usd"]:
            profit_since_start = max(0.0, self.db.sum_realized_pnl_since(0))
            if profit_since_start <= 0:
                self._kill(
                    f"operating reserve depleted (${reserve:.2f}) with no profit to replenish it"
                )
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
