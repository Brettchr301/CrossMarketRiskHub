"""IB Portfolio Sync — connects to Interactive Brokers via ib_insync.

Responsibilities:
  - Pull current cash balance (settled + unsettled)
  - Pull current positions (ticker, shares, avg cost, market value, P&L)
  - Pull net liquidation value
  - Pull pending orders
  - Store everything in local SQLite so data persists when IB is offline
  - Handle reconnection after IB Gateway daily reset (~11:45 PM ET)
  - Detect weekends/holidays
  - Graceful fallback to cached state when IB is unreachable

Scheduled to run at 4:30 PM ET daily (after market close).
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from app.execution.db import get_connection, audit_log, init_execution_db

logger = logging.getLogger(__name__)

# US Market holidays (NYSE) — no trading on these dates
# Updated annually; this covers 2025-2027
US_MARKET_HOLIDAYS_2025_2027 = {
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
}


def is_trading_day(dt: datetime | None = None) -> bool:
    """Check if a given date is a US equity trading day."""
    if dt is None:
        from zoneinfo import ZoneInfo
        dt = datetime.now(ZoneInfo("US/Eastern"))
    # Weekend
    if dt.weekday() >= 5:
        return False
    # Holiday
    date_str = dt.strftime("%Y-%m-%d")
    if date_str in US_MARKET_HOLIDAYS_2025_2027:
        return False
    return True


# ────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class IBPosition:
    ticker: str
    shares: int
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    commodity_type: str = ""
    country: str = ""
    sector: str = ""
    entry_date: str = ""
    days_held: int = 0


@dataclass(slots=True)
class IBPortfolioState:
    snapshot_id: str
    taken_at: str
    net_liquidation: float
    settled_cash: float
    unsettled_cash: float
    total_cash: float
    buying_power: float
    cushion: float
    gross_position_value: float
    unrealized_pnl: float
    realized_pnl_today: float
    positions: list[IBPosition] = field(default_factory=list)
    pending_orders: list[dict[str, Any]] = field(default_factory=list)
    is_stale: bool = False
    source: str = "IB"


@dataclass(slots=True)
class PendingOrder:
    order_id: int
    perm_id: int
    ticker: str
    action: str
    total_qty: int
    filled_qty: int
    limit_price: float
    order_type: str
    tif: str
    status: str


# ────────────────────────────────────────────────────────────────────────────
# IB SYNC SERVICE
# ────────────────────────────────────────────────────────────────────────────

class IBSyncService:
    """Connects to IB Gateway/TWS and syncs portfolio state to SQLite."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,           # 7497=TWS paper, 7496=TWS live, 4001=GW paper, 4002=GW live
        client_id: int = 10,
        timeout: int = 30,
        paper_trade: bool = True,
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.paper_trade = paper_trade
        self._ib = None

    def _connect(self) -> Any:
        """Connect to IB Gateway/TWS. Returns the ib_insync.IB instance."""
        try:
            from ib_insync import IB
        except ImportError:
            raise ImportError(
                "ib_insync is required: pip install ib_insync\n"
                "Also ensure IB Gateway or TWS is running."
            )

        ib = IB()
        ib.connect(
            host=self.host,
            port=self.port,
            clientId=self.client_id,
            timeout=self.timeout,
            readonly=True,   # read-only for sync (order_executor handles writes)
        )
        self._ib = ib
        logger.info("Connected to IB at %s:%d (clientId=%d)", self.host, self.port, self.client_id)
        return ib

    def _disconnect(self) -> None:
        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
            logger.info("Disconnected from IB")
        self._ib = None

    def sync_portfolio(self) -> IBPortfolioState:
        """Pull complete portfolio state from IB and persist to SQLite.

        Returns the portfolio state (from IB if available, cached if not).
        """
        init_execution_db()

        try:
            state = self._fetch_from_ib()
            self._persist_state(state)
            audit_log("SYNC", "portfolio", state.snapshot_id,
                      details={"source": "IB", "nlv": state.net_liquidation,
                               "positions": len(state.positions)})
            return state
        except Exception as exc:
            logger.warning("IB sync failed: %s — falling back to cached state", exc)
            cached = self._load_cached_state()
            if cached:
                audit_log("SYNC", "portfolio", cached.snapshot_id,
                          details={"source": "CACHED", "error": str(exc)})
                return cached
            # No cached state either — return empty
            logger.error("No cached portfolio state available")
            audit_log("ERROR", "portfolio", details={"error": f"IB unreachable, no cache: {exc}"})
            return self._empty_state()

    def _fetch_from_ib(self) -> IBPortfolioState:
        """Fetch live data from IB."""
        ib = self._connect()
        try:
            # Account values
            account_values = {v.tag: v.value for v in ib.accountValues()
                              if v.currency in ("USD", "")}

            nlv = float(account_values.get("NetLiquidation", 0))
            settled = float(account_values.get("SettledCash", 0))  # T+1 settled
            total_cash = float(account_values.get("TotalCashValue", 0))
            unsettled = total_cash - settled
            buying_power = float(account_values.get("BuyingPower", 0))
            cushion = float(account_values.get("Cushion", 0))
            gross_pos = float(account_values.get("GrossPositionValue", 0))
            unreal_pnl = float(account_values.get("UnrealizedPnL", 0))
            real_pnl = float(account_values.get("RealizedPnL", 0))

            # Positions
            positions: list[IBPosition] = []
            for pos in ib.positions():
                contract = pos.contract
                ticker = contract.symbol
                if contract.exchange and contract.primaryExchange:
                    # For non-US tickers, use the symbol as-is
                    pass
                shares = int(pos.position)
                if shares == 0:
                    continue
                avg_cost = float(pos.avgCost)
                # Request market data for current price
                market_price = avg_cost  # fallback
                mkt_val = shares * market_price
                unreal = mkt_val - (shares * avg_cost)
                unreal_pct = (unreal / (shares * avg_cost) * 100) if avg_cost > 0 else 0.0

                positions.append(IBPosition(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=round(avg_cost, 4),
                    market_price=round(market_price, 4),
                    market_value=round(mkt_val, 2),
                    unrealized_pnl=round(unreal, 2),
                    unrealized_pnl_pct=round(unreal_pct, 2),
                ))

            # Use portfolio items for better price data
            for item in ib.portfolio():
                for p in positions:
                    if p.ticker == item.contract.symbol:
                        p.market_price = round(item.marketPrice, 4)
                        p.market_value = round(item.marketValue, 2)
                        p.unrealized_pnl = round(item.unrealizedPNL, 2)
                        pct = (item.unrealizedPNL / (p.shares * p.avg_cost) * 100
                               if p.avg_cost > 0 and p.shares != 0 else 0.0)
                        p.unrealized_pnl_pct = round(pct, 2)
                        break

            # Pending orders
            pending: list[dict[str, Any]] = []
            for trade in ib.openTrades():
                order = trade.order
                contract = trade.contract
                pending.append({
                    "order_id": order.orderId,
                    "perm_id": order.permId,
                    "ticker": contract.symbol,
                    "action": order.action,
                    "total_qty": int(order.totalQuantity),
                    "filled_qty": int(trade.orderStatus.filled),
                    "limit_price": float(order.lmtPrice) if order.lmtPrice else None,
                    "order_type": order.orderType,
                    "tif": order.tif,
                    "status": trade.orderStatus.status,
                })

            snapshot_id = f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

            return IBPortfolioState(
                snapshot_id=snapshot_id,
                taken_at=datetime.now(timezone.utc).isoformat(),
                net_liquidation=round(nlv, 2),
                settled_cash=round(settled, 2),
                unsettled_cash=round(unsettled, 2),
                total_cash=round(total_cash, 2),
                buying_power=round(buying_power, 2),
                cushion=round(cushion, 4),
                gross_position_value=round(gross_pos, 2),
                unrealized_pnl=round(unreal_pnl, 2),
                realized_pnl_today=round(real_pnl, 2),
                positions=positions,
                pending_orders=pending,
                is_stale=False,
                source="IB",
            )
        finally:
            self._disconnect()

    def _persist_state(self, state: IBPortfolioState) -> None:
        """Save portfolio state to SQLite."""
        conn = get_connection()
        try:
            # Insert snapshot
            conn.execute(
                """INSERT OR REPLACE INTO portfolio_snapshots
                   (snapshot_id, taken_at, net_liquidation, settled_cash, unsettled_cash,
                    total_cash, buying_power, cushion, gross_position_value,
                    unrealized_pnl, realized_pnl_today, is_stale, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (state.snapshot_id, state.taken_at, state.net_liquidation,
                 state.settled_cash, state.unsettled_cash, state.total_cash,
                 state.buying_power, state.cushion, state.gross_position_value,
                 state.unrealized_pnl, state.realized_pnl_today,
                 1 if state.is_stale else 0, state.source),
            )

            # Insert positions
            now = datetime.now(timezone.utc).isoformat()
            for pos in state.positions:
                pos_id = f"pos_{state.snapshot_id}_{pos.ticker}"
                conn.execute(
                    """INSERT OR REPLACE INTO positions
                       (position_id, snapshot_id, ticker, shares, avg_cost,
                        market_price, market_value, unrealized_pnl, unrealized_pnl_pct,
                        commodity_type, country, sector, entry_date, days_held, recorded_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (pos_id, state.snapshot_id, pos.ticker, pos.shares, pos.avg_cost,
                     pos.market_price, pos.market_value, pos.unrealized_pnl,
                     pos.unrealized_pnl_pct, pos.commodity_type, pos.country,
                     pos.sector, pos.entry_date, pos.days_held, now),
                )

            # Update pending orders
            conn.execute("DELETE FROM pending_orders")  # replace all
            for po in state.pending_orders:
                conn.execute(
                    """INSERT INTO pending_orders
                       (order_id, perm_id, ticker, action, total_qty, filled_qty,
                        limit_price, order_type, tif, status, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (po["order_id"], po.get("perm_id"), po["ticker"],
                     po["action"], po["total_qty"], po.get("filled_qty", 0),
                     po.get("limit_price"), po.get("order_type"),
                     po.get("tif"), po["status"], now),
                )

            # Update performance tracking
            from zoneinfo import ZoneInfo
            today_str = datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")
            row = conn.execute(
                "SELECT high_water_mark FROM performance_tracking ORDER BY date DESC LIMIT 1"
            ).fetchone()
            prev_hwm = row["high_water_mark"] if row else state.net_liquidation
            new_hwm = max(prev_hwm, state.net_liquidation)
            dd_pct = ((new_hwm - state.net_liquidation) / new_hwm * 100) if new_hwm > 0 else 0.0

            conn.execute(
                """INSERT OR REPLACE INTO performance_tracking
                   (date, net_liquidation, high_water_mark, drawdown_pct, num_positions)
                   VALUES (?, ?, ?, ?, ?)""",
                (today_str, state.net_liquidation, new_hwm, round(dd_pct, 2),
                 len(state.positions)),
            )

            conn.commit()
            logger.info("Persisted portfolio snapshot %s (%d positions)",
                        state.snapshot_id, len(state.positions))
        finally:
            conn.close()

    def _load_cached_state(self) -> IBPortfolioState | None:
        """Load the most recent cached portfolio state from SQLite."""
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY taken_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None

            sid = row["snapshot_id"]

            # Load positions for this snapshot
            pos_rows = conn.execute(
                "SELECT * FROM positions WHERE snapshot_id = ?", (sid,)
            ).fetchall()

            positions = [
                IBPosition(
                    ticker=p["ticker"],
                    shares=p["shares"],
                    avg_cost=p["avg_cost"],
                    market_price=p["market_price"] or p["avg_cost"],
                    market_value=p["market_value"] or 0,
                    unrealized_pnl=p["unrealized_pnl"] or 0,
                    unrealized_pnl_pct=p["unrealized_pnl_pct"] or 0,
                    commodity_type=p["commodity_type"] or "",
                    country=p["country"] or "",
                    sector=p["sector"] or "",
                    entry_date=p["entry_date"] or "",
                    days_held=p["days_held"] or 0,
                ) for p in pos_rows
            ]

            # Load pending orders
            order_rows = conn.execute("SELECT * FROM pending_orders").fetchall()
            pending = [dict(r) for r in order_rows]

            return IBPortfolioState(
                snapshot_id=sid,
                taken_at=row["taken_at"],
                net_liquidation=row["net_liquidation"],
                settled_cash=row["settled_cash"],
                unsettled_cash=row["unsettled_cash"],
                total_cash=row["total_cash"],
                buying_power=row["buying_power"],
                cushion=row["cushion"],
                gross_position_value=row["gross_position_value"],
                unrealized_pnl=row["unrealized_pnl"],
                realized_pnl_today=row["realized_pnl_today"] or 0,
                positions=positions,
                pending_orders=pending,
                is_stale=True,   # always stale since it's cached
                source="CACHED",
            )
        finally:
            conn.close()

    def _empty_state(self) -> IBPortfolioState:
        """Return an empty portfolio state as last resort."""
        return IBPortfolioState(
            snapshot_id=f"empty_{uuid.uuid4().hex[:8]}",
            taken_at=datetime.now(timezone.utc).isoformat(),
            net_liquidation=0.0,
            settled_cash=0.0,
            unsettled_cash=0.0,
            total_cash=0.0,
            buying_power=0.0,
            cushion=0.0,
            gross_position_value=0.0,
            unrealized_pnl=0.0,
            realized_pnl_today=0.0,
            positions=[],
            pending_orders=[],
            is_stale=True,
            source="EMPTY",
        )


# ────────────────────────────────────────────────────────────────────────────
# CONVENIENCE — get latest state without connecting to IB
# ────────────────────────────────────────────────────────────────────────────

def get_latest_portfolio_state() -> IBPortfolioState | None:
    """Load the most recent portfolio state from local DB (no IB connection)."""
    svc = IBSyncService()
    return svc._load_cached_state()


def get_performance_history(days: int = 90) -> list[dict[str, Any]]:
    """Get performance tracking data for the last N days."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT * FROM performance_tracking
               ORDER BY date DESC LIMIT ?""", (days,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def enrich_positions_with_universe(state: IBPortfolioState) -> None:
    """Enrich position data with commodity_type/country from global universe."""
    try:
        from app.modeling.global_universe import global_universe_by_ticker
        universe = global_universe_by_ticker()
        for pos in state.positions:
            info = universe.get(pos.ticker)
            if info:
                pos.commodity_type = info.commodity_type
                pos.country = info.country
                pos.sector = info.sector
    except ImportError:
        logger.warning("Could not import global universe for position enrichment")
