from dataclasses import dataclass, field
import hashlib
import time


@dataclass
class Proposal:
    leg: str
    market_or_symbol: str
    side: str
    size_usd: float
    confidence: float
    rationale: str
    limit_price: float = None
    expiry_ts: float = field(default_factory=lambda: time.time() + 3600)
    source_data_ts: float = field(default_factory=time.time)

    def idempotency_key(self) -> str:
        raw = f"{self.leg}:{self.market_or_symbol}:{self.side}:{round(self.size_usd, 2)}:{int(self.source_data_ts)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class GovernorDecision:
    approved: bool
    proposal: Proposal
    reason: str = ""
    approved_size_usd: float = 0.0
