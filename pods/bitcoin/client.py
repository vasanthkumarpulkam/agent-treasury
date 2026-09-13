import os
import logging

logger = logging.getLogger("bitcoin_client")


class BitcoinClient:
    def __init__(self, live_mode: bool, symbol: str = "BTC/USD", exchange_id: str = "coinbase"):
        self.live_mode = live_mode
        self.symbol = symbol
        self.exchange_id = exchange_id
        self._exchange = self._build_exchange(live_mode, exchange_id)

    def _build_exchange(self, live_mode: bool, exchange_id: str):
        import ccxt

        exchange_class = getattr(ccxt, exchange_id)
        kwargs = {"enableRateLimit": True}
        if live_mode:
            kwargs["apiKey"] = os.environ["BTC_EXCHANGE_API_KEY"]
            kwargs["secret"] = os.environ["BTC_EXCHANGE_API_SECRET"]
            passphrase = os.environ.get("BTC_EXCHANGE_API_PASSPHRASE")
            if passphrase:
                kwargs["password"] = passphrase
        return exchange_class(kwargs)

    def fetch_ohlcv(self, timeframe: str = "1h", limit: int = 300):
        return self._exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)

    def fetch_ticker(self) -> dict:
        return self._exchange.fetch_ticker(self.symbol)

    def execute(self, proposal, approved_size_usd: float, mode: str) -> dict:
        if mode == "paper" or not self.live_mode:
            return self._paper_fill(proposal, approved_size_usd)
        return self._live_fill(proposal, approved_size_usd)

    def _paper_fill(self, proposal, approved_size_usd: float) -> dict:
        ticker = self.fetch_ticker()
        fill_price = ticker["ask"] if proposal.side == "buy" else ticker["bid"]
        fees = approved_size_usd * 0.006
        return {
            "fill_price": fill_price,
            "fill_size_usd": approved_size_usd,
            "fees_usd": fees,
            "order_id": None,
        }

    def _live_fill(self, proposal, approved_size_usd: float) -> dict:
        ticker = self.fetch_ticker()
        price = ticker["ask"] if proposal.side == "buy" else ticker["bid"]
        amount_btc = approved_size_usd / price
        order = self._exchange.create_order(
            symbol=self.symbol,
            type="market",
            side=proposal.side,
            amount=amount_btc,
        )
        logger.info("LIVE BTC order placed: %s", order)
        fees_usd = sum(f.get("cost", 0) for f in order.get("fees", []) or [])
        return {
            "fill_price": order.get("average") or price,
            "fill_size_usd": approved_size_usd,
            "fees_usd": fees_usd,
            "order_id": order.get("id"),
        }
