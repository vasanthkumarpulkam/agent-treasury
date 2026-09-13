"""Resolves logged estimates against real Polymarket outcomes.

Without this, the LLM makes predictions forever and nobody ever checks whether it was
right -- which is the state most "AI trading agent" projects quietly stay in. This closes
the loop so calibration.py can score the model against reality.
"""
import json
import logging
import requests

logger = logging.getLogger("resolution_checker")

GAMMA_API = "https://gamma-api.polymarket.com"

# Don't bother re-checking markets scored in the last few hours -- they can't have resolved.
DEFAULT_MIN_AGE_SECONDS = 6 * 3600


def _parse_prices(raw):
    """Gamma returns outcomePrices as a stringified JSON array (same quirk as the live
    path in research_agent.py). Returns a list of floats or None."""
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        return [float(p) for p in raw]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def fetch_market(market_id: str) -> dict:
    resp = requests.get(f"{GAMMA_API}/markets/{market_id}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def resolve_outcome(market: dict):
    """Returns 1 (YES), 0 (NO), or None if the market hasn't definitively resolved.

    A resolved Polymarket binary market settles its outcome prices to 1/0. We require a
    decisive settlement (>=0.99 or <=0.01) rather than trusting `closed` alone, since a
    market can be closed while still in dispute/UMA resolution."""
    if not market.get("closed"):
        return None
    prices = _parse_prices(market.get("outcomePrices"))
    if not prices:
        return None
    yes_price = prices[0]
    if yes_price >= 0.99:
        return 1
    if yes_price <= 0.01:
        return 0
    return None  # closed but ambiguous/disputed -- don't score it


def check_and_record(db, min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS) -> dict:
    """Check every unresolved estimate and record outcomes for those that have settled."""
    pending = db.unresolved_estimates(min_age_seconds=min_age_seconds)
    stats = {"checked": 0, "resolved": 0, "errors": 0}

    # One market can back many estimates; fetch each market only once per run.
    seen_markets = {}

    for est in pending:
        market_id = est.get("market_id")
        if not market_id:
            continue
        stats["checked"] += 1
        try:
            if market_id not in seen_markets:
                seen_markets[market_id] = fetch_market(market_id)
            outcome = resolve_outcome(seen_markets[market_id])
        except Exception as e:
            logger.warning("Failed to check market %s: %s", market_id, e)
            stats["errors"] += 1
            continue

        if outcome is not None:
            db.record_estimate_outcome(est["id"], outcome)
            stats["resolved"] += 1
            logger.info("Resolved estimate %s (%r) -> %s", est["id"], est.get("question", "")[:60], outcome)

    logger.info("Resolution check: %s", stats)
    return stats
