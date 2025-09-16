"""Execution database schema and initialization.

Creates and manages the local SQLite database (data/portfolio_state.db)
used by the IB Execution & Trade Approval System.

Tables:
  portfolio_snapshots  — IB account state at each sync
  positions            — current and historical positions
  trade_plans          — daily trade plan containers
  planned_trades       — individual trades within plans
  order_fills          — fill data from IB
  audit_log            — every decision, approval, fill, rejection
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "portfolio_state.db"

SCHEMA_SQL = """
-- ══════════════════════════════════════════════════════
-- PORTFOLIO SNAPSHOTS
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    taken_at        TIMESTAMP NOT NULL,
    net_liquidation REAL NOT NULL,
    settled_cash    REAL NOT NULL,
    unsettled_cash  REAL NOT NULL,
    total_cash      REAL NOT NULL,
    buying_power    REAL NOT NULL,
    cushion         REAL,
    gross_position_value REAL NOT NULL,
    unrealized_pnl  REAL,
    realized_pnl_today REAL,
    is_stale        INTEGER DEFAULT 0,
    source          TEXT DEFAULT 'IB'   -- 'IB' or 'CACHED'
);

-- ══════════════════════════════════════════════════════
-- POSITIONS  (current state + historical tracking)
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    snapshot_id     TEXT REFERENCES portfolio_snapshots(snapshot_id),
    ticker          TEXT NOT NULL,
    shares          INTEGER NOT NULL,
    avg_cost        REAL NOT NULL,
    market_price    REAL,
    market_value    REAL,
    unrealized_pnl  REAL,
    unrealized_pnl_pct REAL,
    commodity_type  TEXT,
    country         TEXT,
    sector          TEXT,
    entry_date      TEXT,
    days_held       INTEGER DEFAULT 0,
    recorded_at     TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_snapshot ON positions(snapshot_id);

-- ══════════════════════════════════════════════════════
-- TRADE PLANS  (daily plan containers)
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS trade_plans (
    plan_id         TEXT PRIMARY KEY,
    created_at      TIMESTAMP NOT NULL,
    portfolio_value REAL NOT NULL,
    cash_available  REAL NOT NULL,
    cash_buffer_target REAL NOT NULL,
    cash_after_trades REAL,
    num_positions   INTEGER,
    status          TEXT NOT NULL DEFAULT 'PENDING',
        -- PENDING, PARTIALLY_APPROVED, FULLY_APPROVED, EXECUTED, EXPIRED, CANCELLED
    expires_at      TIMESTAMP NOT NULL,
    executed_at     TIMESTAMP,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_trade_plans_status ON trade_plans(status);

-- ══════════════════════════════════════════════════════
-- PLANNED TRADES  (individual trades within a plan)
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS planned_trades (
    trade_id        TEXT PRIMARY KEY,
    plan_id         TEXT NOT NULL REFERENCES trade_plans(plan_id),
    rank            INTEGER NOT NULL,
    ticker          TEXT NOT NULL,
    action          TEXT NOT NULL,    -- BUY, CLOSE, REDUCE
    shares          INTEGER NOT NULL,
    limit_price     REAL,
    estimated_cost  REAL NOT NULL,
    expected_value_pct REAL,
    conviction_score REAL,
    kelly_fraction  REAL,
    risk_flags      TEXT,            -- JSON array of strings
    constraint_headroom TEXT,        -- JSON dict
    depends_on_trade_id TEXT,        -- if this BUY depends on a CLOSE settling (T+1)
    -- Lifecycle
    status          TEXT NOT NULL DEFAULT 'PENDING',
        -- PENDING, APPROVED, REJECTED, SUBMITTED, PARTIALLY_FILLED, FILLED, CANCELLED, EXPIRED
    approved_at     TIMESTAMP,
    submitted_at    TIMESTAMP,
    filled_at       TIMESTAMP,
    fill_price      REAL,
    fill_shares     INTEGER,
    fill_commission REAL,
    rejection_reason TEXT,
    modified_shares INTEGER,
    modified_price  REAL,
    ib_order_id     INTEGER,
    ib_perm_id      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_planned_trades_plan ON planned_trades(plan_id);
CREATE INDEX IF NOT EXISTS idx_planned_trades_status ON planned_trades(status);
CREATE INDEX IF NOT EXISTS idx_planned_trades_ticker ON planned_trades(ticker);

-- ══════════════════════════════════════════════════════
-- REJECTED TRADES  (trades that didn't make the plan)
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS rejected_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id         TEXT NOT NULL REFERENCES trade_plans(plan_id),
    ticker          TEXT NOT NULL,
    action          TEXT NOT NULL,
    reason          TEXT NOT NULL,
    would_need      REAL,
    created_at      TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejected_plan ON rejected_trades(plan_id);

-- ══════════════════════════════════════════════════════
-- AUDIT LOG  (every decision, approval, fill, rejection)
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type      TEXT NOT NULL,
        -- SYNC, PLAN_CREATED, TRADE_APPROVED, TRADE_REJECTED, TRADE_MODIFIED,
        -- ORDER_SUBMITTED, ORDER_FILLED, ORDER_CANCELLED, PLAN_EXPIRED,
        -- SAFETY_HALT, CONSTRAINT_VIOLATION, ERROR
    entity_type     TEXT,             -- 'plan', 'trade', 'position', 'portfolio'
    entity_id       TEXT,
    ticker          TEXT,
    details         TEXT,             -- JSON blob with full context
    user_action     INTEGER DEFAULT 0 -- 1 if triggered by human, 0 if automated
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);

-- ══════════════════════════════════════════════════════
-- HIGH-WATER MARK & PERFORMANCE TRACKING
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS performance_tracking (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL UNIQUE,
    net_liquidation REAL NOT NULL,
    high_water_mark REAL NOT NULL,
    drawdown_pct    REAL NOT NULL,
    realized_pnl    REAL DEFAULT 0,
    num_positions   INTEGER DEFAULT 0,
    num_trades_today INTEGER DEFAULT 0,
    consecutive_losses INTEGER DEFAULT 0,
    trades_this_month INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_perf_date ON performance_tracking(date);

-- ══════════════════════════════════════════════════════
-- PENDING ORDERS  (orders still open at IB)
-- ══════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS pending_orders (
    order_id        INTEGER PRIMARY KEY,
    perm_id         INTEGER,
    ticker          TEXT NOT NULL,
    action          TEXT NOT NULL,
    total_qty       INTEGER NOT NULL,
    filled_qty      INTEGER DEFAULT 0,
    limit_price     REAL,
    order_type      TEXT,
    tif             TEXT,
    status          TEXT NOT NULL,
    submitted_at    TIMESTAMP,
    last_updated    TIMESTAMP,
    trade_id        TEXT REFERENCES planned_trades(trade_id)
);
"""


def get_connection() -> sqlite3.Connection:
    """Get a connection to the portfolio state database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_execution_db() -> None:
    """Create all execution tables if they don't exist."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Execution database initialized at %s", DB_PATH)
    finally:
        conn.close()


def audit_log(
    event_type: str,
    entity_type: str | None = None,
    entity_id: str | None = None,
    ticker: str | None = None,
    details: str | None = None,
    user_action: bool = False,
) -> None:
    """Write an audit log entry."""
    import json
    from datetime import datetime, timezone
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO audit_log (timestamp, event_type, entity_type, entity_id,
               ticker, details, user_action)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                entity_type,
                entity_id,
                ticker,
                details if isinstance(details, str) else json.dumps(details) if details else None,
                1 if user_action else 0,
            ),
        )
        conn.commit()
    finally:
        conn.close()
