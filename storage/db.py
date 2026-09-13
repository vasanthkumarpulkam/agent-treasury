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
