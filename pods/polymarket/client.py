import os
import logging
import requests

logger = logging.getLogger("polymarket_client")

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


class PolymarketClient:
    def __init__(self, live_mode: bool):
        self.live_mode = live_mode
        self._clob_client = None
        if live_mode:
            self._init_live_client()

    def _init_live_client(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        pk = os.environ["POLYMARKET_PRIVATE_KEY"]
        funder = os.environ["POLYMARKET_FUNDER_ADDRESS"]
        creds = ApiCreds(
            api_key=os.environ["POLYMARKET_API_KEY"],
            api_secret=os.environ["POLYMARKET_API_SECRET"],
            api_passphrase=os.environ["POLYMARKET_API_PASSPHRASE"],
        )
        self._clob_client = ClobClient(
            host=CLOB_API, key=pk, chain_id=137, creds=creds, funder=funder, signature_type=1
        )

    def list_active_markets(self, limit: int = 50) -> list:
        resp = requests.get(
            f"{GAMMA_API}/markets",
            params={"active": "true", "closed": "false", "limit": limit, "order": "volume", "ascending": "false"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def list_active_events(self, limit: int = 30) -> list:
        """Events group related markets. negRisk events are mutually exclusive and
        exhaustive, which is what makes basket arbitrage valid."""
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"active": "true", "closed": "false", "limit": limit,
                    "order": "volume", "ascending": "false"},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()

    def get_orderbook(self, token_id: str) -> dict:
        """Raw order book: {"bids": [...], "asks": [...]}. Arbitrage needs real depth at
        real ask prices, not a mid-price summary."""
        resp = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_orderbook_prices(self, token_id: str) -> dict:
        resp = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=15)
        resp.raise_for_status()
        book = resp.json()
        best_bid = float(book["bids"][0]["price"]) if book.get("bids") else None
        best_ask = float(book["asks"][0]["price"]) if book.get("asks") else None
        mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
        return {"best_bid": best_bid, "best_ask": best_ask, "mid": mid}

    def execute(self, proposal, approved_size_usd: float, mode: str) -> dict:
        if mode == "paper" or not self.live_mode:
            return self._paper_fill(proposal, approved_size_usd)
        return self._live_fill(proposal, approved_size_usd)

    def _paper_fill(self, proposal, approved_size_usd: float) -> dict:
        fill_price = proposal.limit_price if proposal.limit_price else 0.5
        fees = approved_size_usd * 0.0
        return {
            "fill_price": fill_price,
            "fill_size_usd": approved_size_usd,
            "fees_usd": fees,
            "order_id": None,
        }

    def _live_fill(self, proposal, approved_size_usd: float) -> dict:
        from py_clob_client.clob_types import OrderArgs
        from py_clob_client.order_builder.constants import BUY

        # Always BUY: the research agent already selected WHICH token (YES or NO) to buy.
        # Betting against an outcome means buying its NO token, not selling YES -- you
        # can't sell a token you don't hold on the CLOB.
        order_args = OrderArgs(
            price=proposal.limit_price,
            size=approved_size_usd / proposal.limit_price,
            side=BUY,
            token_id=proposal.market_or_symbol,
        )
        signed_order = self._clob_client.create_order(order_args)
        resp = self._clob_client.post_order(signed_order)
        logger.info("LIVE Polymarket order placed: %s", resp)
        return {
            "fill_price": proposal.limit_price,
            "fill_size_usd": approved_size_usd,
            "fees_usd": 0.0,
            "order_id": resp.get("orderID"),
        }
