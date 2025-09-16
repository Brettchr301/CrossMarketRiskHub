"""Approval Queue — SQLite-backed trade approval workflow.

Rules:
  - Plans expire if not approved within 16 hours
  - CLOSE/REDUCE always execute before BUYs
  - BUYs dependent on CLOSE cash are queued for T+1 settlement
  - Modified shares/price must re-validate constraints
  - Full audit trail — never delete, only update status
  - Even if approved, reject silently at execution time if constraints violated
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from app.execution.db import get_connection, audit_log, init_execution_db
from app.execution.trade_prioritizer import TradePlan, PlannedTrade, RejectedTrade

logger = logging.getLogger(__name__)


class ApprovalQueue:
    """Manages the trade plan approval lifecycle in SQLite."""

    def __init__(self):
        init_execution_db()

    # ────────────────────────────────────────────────────────────────
    # SAVE PLAN
    # ────────────────────────────────────────────────────────────────

    def save_plan(self, plan: TradePlan) -> str:
        """Persist a TradePlan to the database. Returns the plan_id."""
        conn = get_connection()
        try:
            # Insert plan header
            conn.execute(
                """INSERT OR REPLACE INTO trade_plans
                   (plan_id, created_at, portfolio_value, cash_available,
                    cash_buffer_target, cash_after_trades, num_positions,
                    status, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (plan.plan_id, plan.timestamp, plan.portfolio_value,
                 plan.cash_available, plan.cash_buffer_target,
                 plan.cash_after_trades, plan.num_current_positions,
                 "PENDING", plan.expires_at),
            )

            # Insert planned trades
            for trade in plan.trades:
                conn.execute(
                    """INSERT OR REPLACE INTO planned_trades
                       (trade_id, plan_id, rank, ticker, action, shares,
                        limit_price, estimated_cost, expected_value_pct,
                        conviction_score, kelly_fraction, risk_flags,
                        constraint_headroom, depends_on_trade_id, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (trade.trade_id, plan.plan_id, trade.rank, trade.ticker,
                     trade.action, trade.shares, trade.estimated_price,
                     trade.estimated_cost, trade.expected_value_pct,
                     trade.conviction_score, trade.kelly_fraction,
                     json.dumps(trade.risk_flags),
                     json.dumps(trade.constraint_headroom),
                     trade.depends_on_trade_id, "PENDING"),
                )

            # Insert rejected trades
            now = datetime.now(timezone.utc).isoformat()
            for rej in plan.rejected_trades:
                conn.execute(
                    """INSERT INTO rejected_trades
                       (plan_id, ticker, action, reason, would_need, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (plan.plan_id, rej.ticker, rej.action, rej.reason,
                     rej.would_need, now),
                )

            conn.commit()
            logger.info("Saved trade plan %s with %d trades, %d rejected",
                        plan.plan_id, len(plan.trades), len(plan.rejected_trades))
            return plan.plan_id
        finally:
            conn.close()

    # ────────────────────────────────────────────────────────────────
    # APPROVE / REJECT / MODIFY
    # ────────────────────────────────────────────────────────────────

    def approve_trade(self, trade_id: str, user_note: str = "") -> dict[str, Any]:
        """Approve a single trade. Returns status dict."""
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM planned_trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            if not row:
                return {"ok": False, "error": f"Trade {trade_id} not found"}

            if row["status"] not in ("PENDING",):
                return {"ok": False, "error": f"Trade is {row['status']}, cannot approve"}

            # Check plan not expired
            plan = conn.execute(
                "SELECT * FROM trade_plans WHERE plan_id = ?", (row["plan_id"],)
            ).fetchone()
            if plan and plan["status"] == "EXPIRED":
                return {"ok": False, "error": "Plan has expired"}

            now = datetime.now(timezone.utc)
            if plan:
                expires = datetime.fromisoformat(plan["expires_at"].replace("Z", "+00:00"))
                if now > expires:
                    conn.execute("UPDATE trade_plans SET status = 'EXPIRED' WHERE plan_id = ?",
                                 (plan["plan_id"],))
                    conn.commit()
                    return {"ok": False, "error": "Plan has expired"}

            conn.execute(
                "UPDATE planned_trades SET status = 'APPROVED', approved_at = ? WHERE trade_id = ?",
                (now.isoformat(), trade_id),
            )

            # Update plan status
            self._update_plan_status(conn, row["plan_id"])
            conn.commit()

            audit_log("TRADE_APPROVED", "trade", trade_id, ticker=row["ticker"],
                      details={"shares": row["shares"], "action": row["action"],
                               "note": user_note},
                      user_action=True)

            return {"ok": True, "trade_id": trade_id, "status": "APPROVED",
                    "ticker": row["ticker"], "action": row["action"],
                    "shares": row["shares"]}
        finally:
            conn.close()

    def approve_multiple(self, trade_ids: list[str]) -> list[dict[str, Any]]:
        """Approve multiple trades at once."""
        return [self.approve_trade(tid) for tid in trade_ids]

    def reject_trade(self, trade_id: str, reason: str = "") -> dict[str, Any]:
        """Reject a single trade."""
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM planned_trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            if not row:
                return {"ok": False, "error": f"Trade {trade_id} not found"}

            if row["status"] not in ("PENDING", "APPROVED"):
                return {"ok": False, "error": f"Trade is {row['status']}, cannot reject"}

            conn.execute(
                """UPDATE planned_trades SET status = 'REJECTED', rejection_reason = ?
                   WHERE trade_id = ?""",
                (reason or "User rejected", trade_id),
            )
            self._update_plan_status(conn, row["plan_id"])
            conn.commit()

            audit_log("TRADE_REJECTED", "trade", trade_id, ticker=row["ticker"],
                      details={"reason": reason}, user_action=True)

            return {"ok": True, "trade_id": trade_id, "status": "REJECTED"}
        finally:
            conn.close()

    def modify_trade(
        self,
        trade_id: str,
        new_shares: int | None = None,
        new_price: float | None = None,
    ) -> dict[str, Any]:
        """Modify shares or price of a pending trade.

        Re-validates constraints after modification.
        """
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM planned_trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            if not row:
                return {"ok": False, "error": f"Trade {trade_id} not found"}
            if row["status"] not in ("PENDING", "APPROVED"):
                return {"ok": False, "error": f"Trade is {row['status']}, cannot modify"}

            updates = {}
            if new_shares is not None:
                if new_shares <= 0:
                    return {"ok": False, "error": "Shares must be positive"}
                updates["modified_shares"] = new_shares
                # Recalculate cost
                price = new_price or row["limit_price"] or 0
                new_cost = new_shares * price
                updates["estimated_cost"] = round(new_cost, 2)
                updates["shares"] = new_shares

            if new_price is not None:
                if new_price <= 0:
                    return {"ok": False, "error": "Price must be positive"}
                updates["modified_price"] = new_price
                updates["limit_price"] = new_price
                shares = new_shares or row["shares"]
                updates["estimated_cost"] = round(shares * new_price, 2)

            if not updates:
                return {"ok": False, "error": "No modifications specified"}

            # Check minimum position size
            est_cost = updates.get("estimated_cost", row["estimated_cost"])
            if row["action"] == "BUY" and est_cost < 2000:
                return {"ok": False, "error": f"Modified cost ${est_cost:,.0f} below $2,000 minimum"}

            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [trade_id]
            conn.execute(
                f"UPDATE planned_trades SET {set_clause} WHERE trade_id = ?",
                values,
            )
            conn.commit()

            audit_log("TRADE_MODIFIED", "trade", trade_id, ticker=row["ticker"],
                      details={"modifications": updates}, user_action=True)

            return {"ok": True, "trade_id": trade_id, "modifications": updates}
        finally:
            conn.close()

    # ────────────────────────────────────────────────────────────────
    # QUERY
    # ────────────────────────────────────────────────────────────────

    def get_current_plan(self) -> dict[str, Any] | None:
        """Get the most recent non-expired plan."""
        conn = get_connection()
        try:
            self._expire_old_plans(conn)

            plan_row = conn.execute(
                """SELECT * FROM trade_plans
                   WHERE status IN ('PENDING', 'PARTIALLY_APPROVED', 'FULLY_APPROVED')
                   ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
            if not plan_row:
                return None

            pid = plan_row["plan_id"]

            trades = conn.execute(
                "SELECT * FROM planned_trades WHERE plan_id = ? ORDER BY rank",
                (pid,)
            ).fetchall()

            rejected = conn.execute(
                "SELECT * FROM rejected_trades WHERE plan_id = ?",
                (pid,)
            ).fetchall()

            return {
                "plan": dict(plan_row),
                "trades": [dict(t) for t in trades],
                "rejected": [dict(r) for r in rejected],
            }
        finally:
            conn.close()

    def get_approved_trades(self, plan_id: str | None = None) -> list[dict[str, Any]]:
        """Get all approved trades ready for execution."""
        conn = get_connection()
        try:
            if plan_id:
                rows = conn.execute(
                    """SELECT * FROM planned_trades
                       WHERE plan_id = ? AND status = 'APPROVED'
                       ORDER BY rank""",
                    (plan_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT pt.* FROM planned_trades pt
                       JOIN trade_plans tp ON pt.plan_id = tp.plan_id
                       WHERE pt.status = 'APPROVED'
                       AND tp.status IN ('PARTIALLY_APPROVED', 'FULLY_APPROVED')
                       ORDER BY pt.rank"""
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_plan_history(self, limit: int = 30) -> list[dict[str, Any]]:
        """Get historical trade plans."""
        conn = get_connection()
        try:
            rows = conn.execute(
                """SELECT tp.*, 
                   (SELECT COUNT(*) FROM planned_trades WHERE plan_id = tp.plan_id) as total_trades,
                   (SELECT COUNT(*) FROM planned_trades WHERE plan_id = tp.plan_id AND status = 'FILLED') as filled_trades
                   FROM trade_plans tp
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_trade_history(self, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get historical trades, optionally filtered by ticker."""
        conn = get_connection()
        try:
            if ticker:
                rows = conn.execute(
                    """SELECT * FROM planned_trades
                       WHERE ticker = ? ORDER BY submitted_at DESC LIMIT ?""",
                    (ticker, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM planned_trades ORDER BY submitted_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_performance_stats(self) -> dict[str, Any]:
        """Compute realized P&L and performance stats from fill data."""
        conn = get_connection()
        try:
            filled = conn.execute(
                """SELECT * FROM planned_trades WHERE status = 'FILLED'
                   ORDER BY filled_at"""
            ).fetchall()

            if not filled:
                return {"total_trades": 0, "message": "No filled trades yet"}

            total = len(filled)
            buys = [r for r in filled if r["action"] == "BUY"]
            sells = [r for r in filled if r["action"] in ("CLOSE", "REDUCE")]

            total_bought = sum((r["fill_shares"] or 0) * (r["fill_price"] or 0) for r in buys)
            total_sold = sum((r["fill_shares"] or 0) * (r["fill_price"] or 0) for r in sells)
            total_commission = sum(r["fill_commission"] or 0 for r in filled)

            # Win rate from matched pairs
            return {
                "total_trades": total,
                "total_buys": len(buys),
                "total_sells": len(sells),
                "total_bought_usd": round(total_bought, 2),
                "total_sold_usd": round(total_sold, 2),
                "total_commissions": round(total_commission, 2),
                "net_realized": round(total_sold - total_bought - total_commission, 2),
            }
        finally:
            conn.close()

    # ────────────────────────────────────────────────────────────────
    # INTERNAL
    # ────────────────────────────────────────────────────────────────

    def _update_plan_status(self, conn, plan_id: str) -> None:
        """Update plan status based on constituent trade statuses."""
        trades = conn.execute(
            "SELECT status FROM planned_trades WHERE plan_id = ?", (plan_id,)
        ).fetchall()
        if not trades:
            return

        statuses = {t["status"] for t in trades}
        if statuses == {"APPROVED"}:
            new_status = "FULLY_APPROVED"
        elif "APPROVED" in statuses and ("PENDING" in statuses or "REJECTED" in statuses):
            new_status = "PARTIALLY_APPROVED"
        elif statuses <= {"FILLED", "CANCELLED"}:
            new_status = "EXECUTED"
        elif statuses <= {"REJECTED", "CANCELLED", "EXPIRED"}:
            new_status = "CANCELLED"
        else:
            new_status = "PENDING"

        conn.execute(
            "UPDATE trade_plans SET status = ? WHERE plan_id = ?",
            (new_status, plan_id),
        )

    def _expire_old_plans(self, conn) -> None:
        """Expire plans past their expiry time."""
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE trade_plans SET status = 'EXPIRED'
               WHERE status IN ('PENDING', 'PARTIALLY_APPROVED')
               AND expires_at < ?""",
            (now,),
        )
        conn.execute(
            """UPDATE planned_trades SET status = 'EXPIRED'
               WHERE status = 'PENDING'
               AND plan_id IN (SELECT plan_id FROM trade_plans WHERE status = 'EXPIRED')""",
        )
        conn.commit()

    # ────────────────────────────────────────────────────────────────
    # BULK OPERATIONS
    # ────────────────────────────────────────────────────────────────

    def approve_all_in_plan(self, plan_id: str) -> list[dict[str, Any]]:
        """Approve all pending trades in a plan."""
        conn = get_connection()
        try:
            trades = conn.execute(
                """SELECT trade_id FROM planned_trades
                   WHERE plan_id = ? AND status = 'PENDING'""",
                (plan_id,)
            ).fetchall()
            return [self.approve_trade(t["trade_id"]) for t in trades]
        finally:
            conn.close()

    def reject_all_in_plan(self, plan_id: str, reason: str = "Bulk rejection") -> list[dict[str, Any]]:
        """Reject all pending trades in a plan."""
        conn = get_connection()
        try:
            trades = conn.execute(
                """SELECT trade_id FROM planned_trades
                   WHERE plan_id = ? AND status IN ('PENDING', 'APPROVED')""",
                (plan_id,)
            ).fetchall()
            return [self.reject_trade(t["trade_id"], reason) for t in trades]
        finally:
            conn.close()
