"""Order Executor — submits approved trades to IB with strict safety rails.

Safety Rails (hard-coded, never overridden):
  - Max $10,000 per single order
  - Max 5 orders per day
  - Max $25,000 daily notional
  - 5% drawdown halt: no BUYs if portfolio is >5% below high-water mark
  - Emergency cash buffer: 5% of portfolio must remain in cash
  - Paper trade mode (default): logs everything, submits nothing

Order Logic:
  - Adaptive Algo for micro-caps, Limit for liquid names
  - Limit price = latest IB price ± 1.5%
  - TIF = DAY (no overnight limit orders)
  - Sells fire before buys
  - T+1 dependent trades queued for next day

Fill Handling:
  - Monitors fills in real-time via IB callbacks
  - Updates DB on partial/full fills
  - Logs commissions
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# ── Safety constants (NEVER change these without PM approval) ────
MAX_ORDER_NOTIONAL = 10_000          # $10K per order
MAX_ORDERS_PER_DAY = 5               # 5 orders per session
MAX_DAILY_NOTIONAL = 25_000          # $25K total per day
DRAWDOWN_HALT_PCT = 0.05             # 5% from HWM → no buys
CASH_BUFFER_PCT = 0.05               # 5% of NLV must remain in cash
LIMIT_OFFSET_PCT = 0.015             # ±1.5% from last price
MICRO_CAP_ADV_THRESHOLD = 500_000    # ADV < $500K → use Adaptive algo


@dataclass
class OrderResult:
    """Result from a single order submission."""
    trade_id: str
    ticker: str
    action: str
    status: str                       # SUBMITTED, REJECTED_SAFETY, PAPER_LOGGED, FAILED
    ib_order_id: int | None = None
    message: str = ""
    limit_price: float = 0.0
    shares: int = 0
    notional: float = 0.0
    algo: str = ""


@dataclass
class ExecutionSession:
    """Tracks daily execution state."""
    date: str = ""
    orders_submitted: int = 0
    notional_submitted: float = 0.0
    orders: list[OrderResult] = field(default_factory=list)
    drawdown_halt: bool = False
    error: str = ""


class OrderExecutor:
    """Submits approved trades to IB with hard safety limits.

    Parameters
    ----------
    paper_trade : bool
        If True (default), log orders but do not submit to IB.
    ib_host : str
        IB Gateway/TWS host.
    ib_port : int
        IB Gateway/TWS port (7497 = TWS paper, 4002 = Gateway paper).
    ib_client_id : int
        IB API client ID (use different IDs for sync vs execution).
    """

    def __init__(
        self,
        paper_trade: bool = True,
        ib_host: str = "127.0.0.1",
        ib_port: int = 7497,
        ib_client_id: int = 2,
    ):
        self.paper_trade = paper_trade
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_client_id = ib_client_id
        self._session = ExecutionSession()

    # ────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ────────────────────────────────────────────────────────────────

    def execute_approved_trades(
        self,
        approved_trades: list[dict[str, Any]],
        portfolio_nlv: float,
        high_water_mark: float,
        current_cash: float,
    ) -> ExecutionSession:
        """Execute list of approved trades with full safety checks.

        Parameters
        ----------
        approved_trades : list of dict
            From ApprovalQueue.get_approved_trades(). Must have:
            trade_id, ticker, action, shares, limit_price, estimated_cost,
            depends_on_trade_id, plan_id.
        portfolio_nlv : float
            Current net liquidation value.
        high_water_mark : float
            Portfolio high-water mark for drawdown check.
        current_cash : float
            Current settled cash balance.

        Returns
        -------
        ExecutionSession with all results.
        """
        from app.execution.db import audit_log, get_connection

        today = date.today().isoformat()
        self._session = ExecutionSession(date=today)

        # ── Drawdown check ───────────────────────────────────────
        if high_water_mark > 0:
            drawdown = (high_water_mark - portfolio_nlv) / high_water_mark
            if drawdown >= DRAWDOWN_HALT_PCT:
                self._session.drawdown_halt = True
                logger.warning(
                    "DRAWDOWN HALT: Portfolio %.1f%% below HWM ($%.0f → $%.0f). "
                    "No BUYs will execute.",
                    drawdown * 100, high_water_mark, portfolio_nlv,
                )
                audit_log(
                    "DRAWDOWN_HALT", "execution", today,
                    details={"drawdown_pct": round(drawdown * 100, 2),
                             "hwm": high_water_mark, "nlv": portfolio_nlv},
                )

        # ── Check daily limits from prior orders today ───────────
        conn = get_connection()
        try:
            prior = conn.execute(
                """SELECT COUNT(*) as cnt, COALESCE(SUM(fill_shares * fill_price), 0) as notional
                   FROM planned_trades
                   WHERE status IN ('SUBMITTED', 'FILLED')
                   AND submitted_at LIKE ?""",
                (f"{today}%",)
            ).fetchone()
            self._session.orders_submitted = prior["cnt"] or 0
            self._session.notional_submitted = prior["notional"] or 0.0
        finally:
            conn.close()

        # ── Separate sells and buys — sells first
        sells = [t for t in approved_trades if t["action"] in ("CLOSE", "REDUCE")]
        buys = [t for t in approved_trades if t["action"] == "BUY"]

        # ── Execute sells first ──────────────────────────────────
        for trade in sells:
            result = self._execute_single(trade, portfolio_nlv, current_cash)
            self._session.orders.append(result)
            if result.status in ("SUBMITTED", "PAPER_LOGGED"):
                current_cash += result.notional  # Cash freed (won't settle until T+1)

        # ── Execute buys ─────────────────────────────────────────
        for trade in buys:
            # Skip buys during drawdown halt
            if self._session.drawdown_halt:
                result = OrderResult(
                    trade_id=trade["trade_id"],
                    ticker=trade["ticker"],
                    action="BUY",
                    status="REJECTED_SAFETY",
                    message="Drawdown halt active — no BUYs permitted",
                )
                self._session.orders.append(result)
                self._mark_trade_status(trade["trade_id"], "CANCELLED",
                                        reason="Drawdown halt")
                continue

            # Check T+1 dependency
            dep = trade.get("depends_on_trade_id")
            if dep:
                dep_result = next(
                    (r for r in self._session.orders if r.trade_id == dep), None
                )
                if dep_result and dep_result.status in ("SUBMITTED", "PAPER_LOGGED"):
                    # Sell was just submitted today → cash not settled until T+1
                    result = OrderResult(
                        trade_id=trade["trade_id"],
                        ticker=trade["ticker"],
                        action="BUY",
                        status="REJECTED_SAFETY",
                        message=f"T+1 dependency: waiting for {dep} to settle",
                    )
                    self._session.orders.append(result)
                    self._mark_trade_status(trade["trade_id"], "PENDING",
                                            reason="Queued for T+1")
                    continue

            result = self._execute_single(trade, portfolio_nlv, current_cash)
            self._session.orders.append(result)
            if result.status in ("SUBMITTED", "PAPER_LOGGED"):
                current_cash -= result.notional

        audit_log(
            "EXECUTION_SESSION", "execution", today,
            details={
                "total_orders": len(self._session.orders),
                "submitted": sum(1 for o in self._session.orders
                                 if o.status in ("SUBMITTED", "PAPER_LOGGED")),
                "rejected": sum(1 for o in self._session.orders
                                if o.status == "REJECTED_SAFETY"),
                "paper_mode": self.paper_trade,
            },
        )

        return self._session

    # ────────────────────────────────────────────────────────────────
    # SINGLE ORDER
    # ────────────────────────────────────────────────────────────────

    def _execute_single(
        self,
        trade: dict[str, Any],
        portfolio_nlv: float,
        current_cash: float,
    ) -> OrderResult:
        """Execute one trade with all safety checks."""
        from app.execution.db import audit_log

        tid = trade["trade_id"]
        ticker = trade["ticker"]
        action = trade["action"]
        shares = trade.get("modified_shares") or trade["shares"]
        price = trade.get("modified_price") or trade["limit_price"] or 0
        notional = abs(shares * price)

        # ── Safety rail: max order size ──────────────────────────
        if notional > MAX_ORDER_NOTIONAL:
            msg = f"Notional ${notional:,.0f} exceeds ${MAX_ORDER_NOTIONAL:,} limit"
            logger.warning("SAFETY BLOCK %s: %s", ticker, msg)
            self._mark_trade_status(tid, "CANCELLED", reason=msg)
            return OrderResult(tid, ticker, action, "REJECTED_SAFETY", message=msg)

        # ── Safety rail: daily order count ───────────────────────
        if self._session.orders_submitted >= MAX_ORDERS_PER_DAY:
            msg = f"Daily order limit ({MAX_ORDERS_PER_DAY}) reached"
            logger.warning("SAFETY BLOCK %s: %s", ticker, msg)
            self._mark_trade_status(tid, "CANCELLED", reason=msg)
            return OrderResult(tid, ticker, action, "REJECTED_SAFETY", message=msg)

        # ── Safety rail: daily notional ──────────────────────────
        if self._session.notional_submitted + notional > MAX_DAILY_NOTIONAL:
            msg = (f"Would exceed daily notional limit "
                   f"(${self._session.notional_submitted + notional:,.0f} > "
                   f"${MAX_DAILY_NOTIONAL:,})")
            logger.warning("SAFETY BLOCK %s: %s", ticker, msg)
            self._mark_trade_status(tid, "CANCELLED", reason=msg)
            return OrderResult(tid, ticker, action, "REJECTED_SAFETY", message=msg)

        # ── Safety rail: emergency cash buffer (buys only) ───────
        if action == "BUY":
            min_cash = portfolio_nlv * CASH_BUFFER_PCT
            if current_cash - notional < min_cash:
                msg = (f"Cash after trade (${current_cash - notional:,.0f}) "
                       f"below emergency buffer (${min_cash:,.0f})")
                logger.warning("SAFETY BLOCK %s: %s", ticker, msg)
                self._mark_trade_status(tid, "CANCELLED", reason=msg)
                return OrderResult(tid, ticker, action, "REJECTED_SAFETY", message=msg)

        # ── Compute limit price ──────────────────────────────────
        if action in ("CLOSE", "REDUCE"):
            limit_price = round(price * (1 - LIMIT_OFFSET_PCT), 2)  # Sell slightly below
        else:
            limit_price = round(price * (1 + LIMIT_OFFSET_PCT), 2)  # Buy slightly above

        # ── Determine algo ───────────────────────────────────────
        adv = trade.get("avg_daily_volume", 1_000_000)
        algo = "ADAPTIVE" if adv < MICRO_CAP_ADV_THRESHOLD else "LIMIT"

        # ── Paper trade mode ─────────────────────────────────────
        if self.paper_trade:
            self._session.orders_submitted += 1
            self._session.notional_submitted += notional

            self._mark_trade_status(
                tid, "SUBMITTED",
                fill_price=limit_price, fill_shares=shares,
                fill_commission=0.0,
            )

            audit_log(
                "PAPER_ORDER", "trade", tid, ticker=ticker,
                details={
                    "action": action, "shares": shares,
                    "limit_price": limit_price, "algo": algo,
                    "notional": notional,
                },
            )

            logger.info(
                "PAPER ORDER: %s %d × %s @ $%.2f (%s) = $%.0f",
                action, shares, ticker, limit_price, algo, notional,
            )

            return OrderResult(
                trade_id=tid, ticker=ticker, action=action,
                status="PAPER_LOGGED", limit_price=limit_price,
                shares=shares, notional=notional, algo=algo,
                message="Paper trade logged (not submitted to IB)",
            )

        # ── Live IB order ────────────────────────────────────────
        return self._submit_ib_order(
            trade_id=tid, ticker=ticker, action=action,
            shares=shares, limit_price=limit_price,
            algo=algo, notional=notional,
        )

    def _submit_ib_order(
        self,
        trade_id: str,
        ticker: str,
        action: str,
        shares: int,
        limit_price: float,
        algo: str,
        notional: float,
    ) -> OrderResult:
        """Submit a live order to Interactive Brokers."""
        from app.execution.db import audit_log

        try:
            from ib_insync import IB, Stock, LimitOrder, Order

            ib = IB()
            ib.connect(self.ib_host, self.ib_port, clientId=self.ib_client_id,
                       timeout=15)

            contract = Stock(ticker, "SMART", "USD")
            ib.qualifyContracts(contract)

            ib_action = "SELL" if action in ("CLOSE", "REDUCE") else "BUY"

            if algo == "ADAPTIVE":
                order = Order()
                order.action = ib_action
                order.totalQuantity = shares
                order.orderType = "LMT"
                order.lmtPrice = limit_price
                order.tif = "DAY"
                order.algoStrategy = "Adaptive"
                order.algoParams = [{"tag": "adaptivePriority", "value": "Normal"}]
            else:
                order = LimitOrder(ib_action, shares, limit_price, tif="DAY")

            trade = ib.placeOrder(contract, order)

            self._session.orders_submitted += 1
            self._session.notional_submitted += notional

            # Update DB with submitted status
            self._mark_trade_status(
                trade_id, "SUBMITTED",
                ib_order_id=trade.order.orderId,
            )

            audit_log(
                "LIVE_ORDER", "trade", trade_id, ticker=ticker,
                details={
                    "action": ib_action, "shares": shares,
                    "limit_price": limit_price, "algo": algo,
                    "ib_order_id": trade.order.orderId,
                    "notional": notional,
                },
            )

            logger.info(
                "LIVE ORDER #%d: %s %d × %s @ $%.2f (%s)",
                trade.order.orderId, ib_action, shares, ticker,
                limit_price, algo,
            )

            # Wait briefly for fill status
            ib.sleep(2)
            if trade.isDone():
                fill = trade.fills[-1] if trade.fills else None
                if fill:
                    self._record_fill(
                        trade_id, ticker,
                        fill.execution.shares,
                        fill.execution.price,
                        fill.commissionReport.commission if fill.commissionReport else 0,
                    )

            ib.disconnect()

            return OrderResult(
                trade_id=trade_id, ticker=ticker, action=action,
                status="SUBMITTED", ib_order_id=trade.order.orderId,
                limit_price=limit_price, shares=shares,
                notional=notional, algo=algo,
                message="Order submitted to IB",
            )

        except Exception as e:
            logger.error("IB order failed for %s: %s", ticker, e)
            self._mark_trade_status(trade_id, "FAILED", reason=str(e))
            return OrderResult(
                trade_id=trade_id, ticker=ticker, action=action,
                status="FAILED", message=str(e),
            )

    # ────────────────────────────────────────────────────────────────
    # FILL MONITORING
    # ────────────────────────────────────────────────────────────────

    def check_pending_fills(self) -> list[dict[str, Any]]:
        """Check IB for fills on previously submitted orders.

        Call this periodically (e.g., every 60s during market hours).
        """
        from app.execution.db import get_connection

        conn = get_connection()
        try:
            pending = conn.execute(
                """SELECT trade_id, ticker, ib_order_id
                   FROM planned_trades
                   WHERE status = 'SUBMITTED' AND ib_order_id IS NOT NULL"""
            ).fetchall()
        finally:
            conn.close()

        if not pending:
            return []

        if self.paper_trade:
            # In paper mode, auto-fill submitted orders
            results = []
            for row in pending:
                self._mark_trade_status(row["trade_id"], "FILLED")
                results.append({"trade_id": row["trade_id"], "status": "FILLED",
                                "mode": "paper"})
            return results

        # Live mode: check IB
        try:
            from ib_insync import IB
            ib = IB()
            ib.connect(self.ib_host, self.ib_port, clientId=self.ib_client_id,
                       timeout=10)

            results = []
            for row in pending:
                for trade in ib.trades():
                    if trade.order.orderId == row["ib_order_id"]:
                        if trade.isDone() and trade.fills:
                            fill = trade.fills[-1]
                            self._record_fill(
                                row["trade_id"], row["ticker"],
                                fill.execution.shares,
                                fill.execution.price,
                                fill.commissionReport.commission
                                if fill.commissionReport else 0,
                            )
                            results.append({
                                "trade_id": row["trade_id"],
                                "status": "FILLED",
                                "fill_price": fill.execution.price,
                                "fill_shares": fill.execution.shares,
                            })
                        elif trade.orderStatus.status == "Cancelled":
                            self._mark_trade_status(
                                row["trade_id"], "CANCELLED",
                                reason="Cancelled by IB/exchange",
                            )
                            results.append({"trade_id": row["trade_id"],
                                            "status": "CANCELLED"})
            ib.disconnect()
            return results

        except Exception as e:
            logger.error("Fill check failed: %s", e)
            return [{"error": str(e)}]

    # ────────────────────────────────────────────────────────────────
    # DB HELPERS
    # ────────────────────────────────────────────────────────────────

    def _mark_trade_status(
        self,
        trade_id: str,
        status: str,
        reason: str = "",
        ib_order_id: int | None = None,
        fill_price: float | None = None,
        fill_shares: int | None = None,
        fill_commission: float | None = None,
    ) -> None:
        """Update a trade's status in the database."""
        from app.execution.db import get_connection
        conn = get_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            updates = {"status": status}

            if status == "SUBMITTED":
                updates["submitted_at"] = now
            elif status == "FILLED":
                updates["filled_at"] = now
            elif status in ("CANCELLED", "FAILED"):
                updates["rejection_reason"] = reason

            if ib_order_id is not None:
                updates["ib_order_id"] = ib_order_id
            if fill_price is not None:
                updates["fill_price"] = fill_price
            if fill_shares is not None:
                updates["fill_shares"] = fill_shares
            if fill_commission is not None:
                updates["fill_commission"] = fill_commission

            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [trade_id]
            conn.execute(
                f"UPDATE planned_trades SET {set_clause} WHERE trade_id = ?",
                values,
            )
            conn.commit()
        finally:
            conn.close()

    def _record_fill(
        self,
        trade_id: str,
        ticker: str,
        shares: int,
        price: float,
        commission: float,
    ) -> None:
        """Record a fill in the database and audit log."""
        from app.execution.db import audit_log

        self._mark_trade_status(
            trade_id, "FILLED",
            fill_price=price, fill_shares=shares,
            fill_commission=commission,
        )

        audit_log(
            "ORDER_FILLED", "trade", trade_id, ticker=ticker,
            details={
                "fill_shares": shares,
                "fill_price": price,
                "commission": commission,
                "net_amount": shares * price - commission,
            },
        )

        logger.info(
            "FILLED: %d × %s @ $%.2f (commission $%.2f)",
            shares, ticker, price, commission,
        )


# ── Convenience functions ────────────────────────────────────────

def run_daily_execution(paper_trade: bool = True) -> ExecutionSession:
    """One-call daily execution: load approved trades, check safety, execute.

    Designed to be called by the scheduler at 9:31 AM ET.
    """
    from app.execution.approval_queue import ApprovalQueue
    from app.execution.ib_sync import get_latest_portfolio_state

    queue = ApprovalQueue()
    approved = queue.get_approved_trades()

    if not approved:
        logger.info("No approved trades to execute")
        return ExecutionSession(date=date.today().isoformat())

    state = get_latest_portfolio_state()
    if not state:
        logger.error("Cannot execute: no portfolio state available")
        return ExecutionSession(
            date=date.today().isoformat(),
            error="No portfolio state",
        )

    executor = OrderExecutor(paper_trade=paper_trade)
    return executor.execute_approved_trades(
        approved_trades=approved,
        portfolio_nlv=state.net_liquidation,
        high_water_mark=_get_high_water_mark(),
        current_cash=state.settled_cash,
    )


def _get_high_water_mark() -> float:
    """Fetch the latest HWM from performance tracking."""
    from app.execution.db import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT high_water_mark FROM performance_tracking ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return row["high_water_mark"] if row else 0.0
    finally:
        conn.close()
