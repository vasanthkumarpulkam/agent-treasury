#!/usr/bin/env python3
import os
import logging
import yaml
from dotenv import load_dotenv

from storage.db import Database
from governor.governor import Governor
from pods.polymarket.client import PolymarketClient
from pods.polymarket.research_agent import PolymarketResearchAgent
from pods.polymarket.arbitrage_agent import ArbitrageAgent
from pods.bitcoin.client import BitcoinClient
from pods.bitcoin.signal_agent import BitcoinSignalAgent
from orchestrator.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    load_dotenv()
    config = load_config()

    live_mode = os.environ.get("LIVE_MODE", "false").lower() == "true"
    mode = "live" if live_mode else "paper"

    if live_mode:
        required = ["POLYMARKET_PRIVATE_KEY", "POLYMARKET_FUNDER_ADDRESS", "POLYMARKET_API_KEY",
                    "BTC_EXCHANGE_API_KEY", "BTC_EXCHANGE_API_SECRET"]
        missing = [k for k in required if not os.environ.get(k)]
        if missing:
            raise SystemExit(
                f"LIVE_MODE=true but missing required credentials: {missing}. "
                f"Set them in .env or set LIVE_MODE=false to run in paper mode."
            )
        logger.warning("Starting in LIVE mode. Real orders will be placed with real funds.")
    else:
        logger.info("Starting in PAPER mode. No real orders will be placed.")

    db = Database(config["logging"]["db_path"])

    polymarket_client = PolymarketClient(live_mode=live_mode)
    bitcoin_client = BitcoinClient(
        live_mode=live_mode,
        symbol=config["bitcoin"]["symbol"],
        exchange_id=os.environ.get("BTC_EXCHANGE", "coinbase"),
    )

    governor = Governor(db, config, execution_clients={
        "polymarket": polymarket_client,
        "bitcoin": bitcoin_client,
    })

    polymarket_agent = PolymarketResearchAgent(polymarket_client, config, db=db)
    bitcoin_agent = BitcoinSignalAgent(bitcoin_client, config)

    arbitrage_agent = ArbitrageAgent(polymarket_client, config)
    orchestrator = Orchestrator(governor, polymarket_agent, bitcoin_agent, mode=mode,
                                 arbitrage_agent=arbitrage_agent)
    result = orchestrator.run_cycle()

    logger.info("Cycle result: %s", result["status"])
    return result


if __name__ == "__main__":
    main()
