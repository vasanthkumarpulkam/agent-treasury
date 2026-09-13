import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    event_type TEXT NOT NULL,
    leg TEXT,
    amount_usd REAL NOT NULL,
    equity_after_usd REAL NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    leg TEXT NOT NULL,
    idempotency_key TEXT UNIQUE NOT NULL,
    market_or_symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size_usd REAL NOT NULL,
    limit_price REAL,
    confidence REAL,
    rationale TEXT,
    status TEXT NOT NULL,
    reject_reason TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    fill_price REAL NOT NULL,
    fill_size_usd REAL NOT NULL,
    fees_usd REAL NOT NULL,
    mode TEXT NOT NULL,
    exchange_order_id TEXT,
    FOREIGN KEY(proposal_id) REFERENCES proposals(id)
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    leg TEXT NOT NULL,
    market_or_symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    size_usd REAL NOT NULL,
    entry_price REAL NOT NULL,
    opened_ts REAL NOT NULL,
    closed_ts REAL,
    realized_pnl_usd REAL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Every probability estimate the research agent makes, whether or not it led to a trade.
-- The ones that DIDN'T clear the edge threshold matter just as much for calibration:
-- scoring only the trades you took tells you nothing about whether the model is any good.
-- The graveyard. Every generation that died, why, and what it cost.
-- If this table grows, that IS the result: it's evidence the strategy doesn't work,
-- not an inconvenience to be cleared.
CREATE TABLE IF NOT EXISTS graveyard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    birth_ts REAL NOT NULL,
    death_ts REAL NOT NULL,
    lifespan_days REAL NOT NULL,
    starting_capital REAL NOT NULL,
    final_equity REAL NOT NULL,
    pnl REAL NOT NULL,
    cause_of_death TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    market_id TEXT,
    condition_id TEXT,
    slug TEXT,
    question TEXT,
    token_id TEXT,
    model_name TEXT,
    implied_prob REAL NOT NULL,
    model_prob REAL NOT NULL,
    llm_confidence REAL,
    edge REAL,
    traded INTEGER NOT NULL DEFAULT 0,
    reasoning TEXT,
    outcome INTEGER,          -- NULL = still unresolved, 1 = resolved YES, 0 = resolved NO
    resolved_ts REAL
);
"""


class Database:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_state(self, key: str, default=None):
        with self._conn() as conn:
            row = conn.execute("SELECT value FROM system_state WHERE key=?", (key,)).fetchone()
            return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO system_state(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def record_ledger_event(self, event_type: str, amount_usd: float, equity_after_usd: float,
                             leg: str = "account", note: str = ""):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ledger_events(ts, event_type, leg, amount_usd, equity_after_usd, note) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), event_type, leg, amount_usd, equity_after_usd, note),
            )

    def sum_realized_pnl_since(self, ts_cutoff: float, leg: str = None) -> float:
        with self._conn() as conn:
            if leg:
                row = conn.execute(
                    "SELECT COALESCE(SUM(amount_usd),0) s FROM ledger_events "
                    "WHERE event_type='realized_pnl' AND ts >= ? AND leg=?",
                    (ts_cutoff, leg),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(amount_usd),0) s FROM ledger_events "
                    "WHERE event_type='realized_pnl' AND ts >= ?",
                    (ts_cutoff,),
                ).fetchone()
            return row["s"]

    def insert_proposal(self, leg, idempotency_key, market_or_symbol, side, size_usd,
                         limit_price, confidence, rationale, status, raw_json,
                         reject_reason=None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO proposals(ts, leg, idempotency_key, market_or_symbol, side, "
                "size_usd, limit_price, confidence, rationale, status, reject_reason, raw_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(idempotency_key) DO NOTHING",
                (time.time(), leg, idempotency_key, market_or_symbol, side, size_usd,
                 limit_price, confidence, rationale, status, reject_reason, raw_json),
            )
            return cur.lastrowid

    def proposal_exists(self, idempotency_key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM proposals WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            return row is not None

    def update_proposal_status(self, proposal_id: int, status: str, reject_reason: str = None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE proposals SET status=?, reject_reason=? WHERE id=?",
                (status, reject_reason, proposal_id),
            )

    def insert_fill(self, proposal_id, fill_price, fill_size_usd, fees_usd, mode,
                     exchange_order_id=None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO fills(proposal_id, ts, fill_price, fill_size_usd, fees_usd, mode, "
                "exchange_order_id) VALUES (?,?,?,?,?,?,?)",
                (proposal_id, time.time(), fill_price, fill_size_usd, fees_usd, mode,
                 exchange_order_id),
            )

    def open_position(self, leg, market_or_symbol, side, size_usd, entry_price) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO positions(leg, market_or_symbol, side, size_usd, entry_price, "
                "opened_ts, status) VALUES (?,?,?,?,?,?, 'open')",
                (leg, market_or_symbol, side, size_usd, entry_price, time.time()),
            )
            return cur.lastrowid

    def close_position(self, position_id: int, realized_pnl_usd: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE positions SET status='closed', closed_ts=?, realized_pnl_usd=? WHERE id=?",
                (time.time(), realized_pnl_usd, position_id),
            )

    def open_positions(self, leg: str = None):
        with self._conn() as conn:
            if leg:
                rows = conn.execute(
                    "SELECT * FROM positions WHERE status='open' AND leg=?", (leg,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM positions WHERE status='open'").fetchall()
            return [dict(r) for r in rows]

    def total_open_exposure_usd(self) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(size_usd),0) s FROM positions WHERE status='open'"
            ).fetchone()
            return row["s"]

    def notional_opened_since(self, ts_cutoff: float) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(size_usd),0) s FROM positions WHERE opened_ts >= ?",
                (ts_cutoff,),
            ).fetchone()
            return row["s"]

    # --- estimates / calibration ---
    def insert_estimate(self, market_id, condition_id, slug, question, token_id, model_name,
                         implied_prob, model_prob, llm_confidence, edge, traded, reasoning) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO estimates(ts, market_id, condition_id, slug, question, token_id, "
                "model_name, implied_prob, model_prob, llm_confidence, edge, traded, reasoning) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), market_id, condition_id, slug, question, token_id, model_name,
                 implied_prob, model_prob, llm_confidence, edge, 1 if traded else 0, reasoning),
            )
            return cur.lastrowid

    def unresolved_estimates(self, min_age_seconds: float = 0):
        """Estimates with no recorded outcome yet. min_age_seconds avoids re-checking
        markets that were only just scored and can't possibly have resolved."""
        cutoff = time.time() - min_age_seconds
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM estimates WHERE outcome IS NULL AND ts <= ? ORDER BY ts",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_estimate_outcome(self, estimate_id: int, outcome: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE estimates SET outcome=?, resolved_ts=? WHERE id=?",
                (outcome, time.time(), estimate_id),
            )

    def resolved_estimates(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM estimates WHERE outcome IS NOT NULL ORDER BY ts"
            ).fetchall()
            return [dict(r) for r in rows]

    def estimate_counts(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) n FROM estimates").fetchone()["n"]
            resolved = conn.execute(
                "SELECT COUNT(*) n FROM estimates WHERE outcome IS NOT NULL"
            ).fetchone()["n"]
            traded = conn.execute("SELECT COUNT(*) n FROM estimates WHERE traded=1").fetchone()["n"]
            return {"total": total, "resolved": resolved, "traded": traded}

    # --- survival / generations ---
    def record_death(self, generation, birth_ts, death_ts, starting_capital, final_equity, cause):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO graveyard(generation, birth_ts, death_ts, lifespan_days, "
                "starting_capital, final_equity, pnl, cause_of_death) VALUES (?,?,?,?,?,?,?,?)",
                (generation, birth_ts, death_ts, (death_ts - birth_ts) / 86400.0,
                 starting_capital, final_equity, final_equity - starting_capital, cause),
            )

    def graveyard(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM graveyard ORDER BY generation").fetchall()
            return [dict(r) for r in rows]

    def reset_pnl_baseline(self):
        """Mark all existing realized-P&L events as belonging to a previous generation so
        the new generation's equity starts clean at its own capital. History is kept (the
        rows stay, retagged) -- a new generation should not be able to erase the evidence
        of what the last one did."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE ledger_events SET event_type='realized_pnl_archived' "
                "WHERE event_type='realized_pnl'"
            )
