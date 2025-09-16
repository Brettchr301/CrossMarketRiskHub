"""FastAPI Execution Routes — REST API for the execution layer.

Endpoints:
  GET  /api/execution/portfolio      — current positions, cash, NLV
  GET  /api/execution/trade-plan     — today's prioritized plan
  POST /api/execution/approve        — approve trades by ID
  POST /api/execution/reject         — reject trades by ID
  POST /api/execution/modify         — change shares or price
  POST /api/execution/approve-all    — approve all pending in a plan
  POST /api/execution/reject-all     — reject all pending in a plan
  POST /api/execution/generate-plan  — manually trigger plan generation
  POST /api/execution/execute        — manually trigger order execution
  GET  /api/execution/history        — past plans and fills
  GET  /api/execution/performance    — realized P&L, stats
  GET  /api/execution/audit-log      — audit trail
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/execution", tags=["execution"])


# ── Request / Response models ────────────────────────────────────

class ApproveRequest(BaseModel):
    trade_ids: list[str] = Field(..., description="Trade IDs to approve")
    note: str = ""

class RejectRequest(BaseModel):
    trade_ids: list[str] = Field(..., description="Trade IDs to reject")
    reason: str = ""

class ModifyRequest(BaseModel):
    trade_id: str
    new_shares: int | None = None
    new_price: float | None = None

class ExecuteRequest(BaseModel):
    paper_trade: bool = True
    plan_id: str | None = None

class GeneratePlanRequest(BaseModel):
    """Trigger manual plan generation."""
    force: bool = False


# ── PORTFOLIO ────────────────────────────────────────────────────

@router.get("/portfolio")
def get_portfolio() -> dict[str, Any]:
    """Current portfolio state from IB (or cached)."""
    from app.execution.ib_sync import get_latest_portfolio_state
    state = get_latest_portfolio_state()
    if not state:
        return {"ok": False, "error": "No portfolio state available",
                "positions": [], "nlv": 0, "cash": 0}
    return {
        "ok": True,
        "snapshot_id": state.snapshot_id,
        "nlv": state.net_liquidation,
        "settled_cash": state.settled_cash,
        "unsettled_cash": state.unsettled_cash,
        "buying_power": state.buying_power,
        "is_stale": state.is_stale,
        "source": state.source,
        "positions": [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_cost": p.avg_cost,
                "market_price": p.market_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "pnl_pct": round((p.unrealized_pnl / (p.avg_cost * p.shares)) * 100, 2)
                    if p.avg_cost and p.shares else 0,
                "commodity_type": p.commodity_type,
                "country": p.country,
                "days_held": p.days_held,
            }
            for p in state.positions
        ],
        "pending_orders": [
            {
                "ticker": o.ticker,
                "action": o.action,
                "shares": o.shares,
                "limit_price": o.limit_price,
                "status": o.status,
            }
            for o in state.pending_orders
        ],
    }


# ── TRADE PLAN ───────────────────────────────────────────────────

@router.get("/trade-plan")
def get_trade_plan() -> dict[str, Any]:
    """Get the current pending trade plan."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    plan = queue.get_current_plan()
    if not plan:
        return {"ok": True, "plan": None, "trades": [], "rejected": [],
                "message": "No active trade plan"}
    return {"ok": True, **plan}


@router.post("/generate-plan")
def generate_plan(req: GeneratePlanRequest) -> dict[str, Any]:
    """Manually trigger trade plan generation.

    Runs: IB sync → load cached decision signals → trade prioritizer → save plan.
    If req.force is True, runs a fresh backtest (expensive, takes minutes).
    """
    from app.execution.ib_sync import IBSyncService, enrich_positions_with_universe
    from app.execution.trade_prioritizer import signals_to_trade_plan
    from app.execution.approval_queue import ApprovalQueue
    from app.execution.decision_bridge import (
        load_cached_signals, load_decision_metadata, run_fresh_backtest_and_cache,
    )
    from app.portfolio.risk_manager import PortfolioConstraints

    try:
        # 1. Sync portfolio
        sync = IBSyncService()
        state = sync.sync_portfolio()
        if not state:
            raise HTTPException(500, "Failed to sync portfolio state")

        # 2. Enrich positions
        enrich_positions_with_universe(state)

        # 3. Load rebalance signals — cached or fresh
        if req.force:
            logger.info("Force flag set — running fresh backtest")
            signals = run_fresh_backtest_and_cache()
        else:
            signals = load_cached_signals()

        if not signals:
            meta = load_decision_metadata()
            msg = (f"No rebalance signals available. "
                   f"Decision recommendation: {meta.get('recommendation', 'unknown')}. "
                   f"Run backtest first or use force=true.")
            return {"ok": True, "message": msg, "plan": None}

        # 3b. Paper trading: if IB is not connected (NLV=0), use the
        #     configured portfolio capital so we can still generate plans.
        constraints = PortfolioConstraints()
        if state.net_liquidation <= 0:
            paper_capital = constraints.total_capital  # default $75K
            logger.info("No IB connection (NLV=0) — using paper capital $%,.0f", paper_capital)
            state.net_liquidation = paper_capital
            state.settled_cash = paper_capital
            state.total_cash = paper_capital
            state.buying_power = paper_capital
            state.source = "PAPER"

        # 4. Build trade plan with cash-constrained prioritization
        plan = signals_to_trade_plan(
            rebalance_signals=signals,
            portfolio_state=state,
            constraints=constraints,
        )

        # 5. Save to DB
        queue = ApprovalQueue()
        plan_id = queue.save_plan(plan)

        return {
            "ok": True,
            "plan_id": plan_id,
            "num_trades": len(plan.trades),
            "num_rejected": len(plan.rejected_trades),
            "cash_after_trades": plan.cash_after_trades,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plan generation failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Plan generation failed: {e}")


# ── APPROVE / REJECT / MODIFY ───────────────────────────────────

@router.post("/approve")
def approve_trades(req: ApproveRequest) -> dict[str, Any]:
    """Approve one or more trades."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    results = queue.approve_multiple(req.trade_ids)
    return {"ok": True, "results": results}


@router.post("/reject")
def reject_trades(req: RejectRequest) -> dict[str, Any]:
    """Reject one or more trades."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    results = [queue.reject_trade(tid, req.reason) for tid in req.trade_ids]
    return {"ok": True, "results": results}


@router.post("/modify")
def modify_trade(req: ModifyRequest) -> dict[str, Any]:
    """Modify shares or price of a pending trade."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    result = queue.modify_trade(req.trade_id, req.new_shares, req.new_price)
    return result


@router.post("/approve-all")
def approve_all(plan_id: str) -> dict[str, Any]:
    """Approve all pending trades in a plan."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    results = queue.approve_all_in_plan(plan_id)
    return {"ok": True, "results": results}


@router.post("/reject-all")
def reject_all(plan_id: str, reason: str = "Bulk rejection") -> dict[str, Any]:
    """Reject all trades in a plan."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    results = queue.reject_all_in_plan(plan_id, reason)
    return {"ok": True, "results": results}


# ── EXECUTE ──────────────────────────────────────────────────────

@router.post("/execute")
def execute_orders(req: ExecuteRequest) -> dict[str, Any]:
    """Manually trigger order execution for approved trades.

    Default: paper trade mode (logs but doesn't submit).
    """
    from app.execution.order_executor import OrderExecutor
    from app.execution.approval_queue import ApprovalQueue
    from app.execution.ib_sync import get_latest_portfolio_state

    queue = ApprovalQueue()
    approved = queue.get_approved_trades(req.plan_id)

    if not approved:
        return {"ok": True, "message": "No approved trades to execute",
                "session": None}

    state = get_latest_portfolio_state()
    if not state:
        raise HTTPException(500, "No portfolio state — run sync first")

    # Get HWM
    from app.execution.db import get_connection
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT high_water_mark FROM performance_tracking ORDER BY date DESC LIMIT 1"
        ).fetchone()
        hwm = row["high_water_mark"] if row else state.net_liquidation
    finally:
        conn.close()

    executor = OrderExecutor(paper_trade=req.paper_trade)
    session = executor.execute_approved_trades(
        approved_trades=approved,
        portfolio_nlv=state.net_liquidation,
        high_water_mark=hwm,
        current_cash=state.settled_cash,
    )

    return {
        "ok": True,
        "date": session.date,
        "orders_submitted": session.orders_submitted,
        "notional_submitted": session.notional_submitted,
        "drawdown_halt": session.drawdown_halt,
        "orders": [
            {
                "trade_id": o.trade_id,
                "ticker": o.ticker,
                "action": o.action,
                "status": o.status,
                "shares": o.shares,
                "limit_price": o.limit_price,
                "notional": o.notional,
                "algo": o.algo,
                "message": o.message,
            }
            for o in session.orders
        ],
    }


# ── HISTORY & PERFORMANCE ───────────────────────────────────────

@router.get("/history")
def get_history(
    ticker: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    """Trade history, optionally filtered by ticker."""
    from app.execution.approval_queue import ApprovalQueue
    queue = ApprovalQueue()
    trades = queue.get_trade_history(ticker=ticker, limit=limit)
    plans = queue.get_plan_history(limit=limit)
    return {"ok": True, "trades": trades, "plans": plans}


@router.get("/performance")
def get_performance() -> dict[str, Any]:
    """Realized P&L and performance stats."""
    from app.execution.approval_queue import ApprovalQueue
    from app.execution.ib_sync import get_performance_history
    queue = ApprovalQueue()
    stats = queue.get_performance_stats()
    daily = get_performance_history(days=90)
    return {"ok": True, "stats": stats, "daily_performance": daily}


@router.get("/audit-log")
def get_audit_log(
    limit: int = Query(100, ge=1, le=1000),
    entity_type: str | None = Query(None),
) -> dict[str, Any]:
    """Fetch audit log entries."""
    from app.execution.db import get_connection
    conn = get_connection()
    try:
        if entity_type:
            rows = conn.execute(
                """SELECT * FROM audit_log WHERE entity_type = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (entity_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return {"ok": True, "entries": [dict(r) for r in rows]}
    finally:
        conn.close()


# ── SYNC ─────────────────────────────────────────────────────────

@router.post("/sync")
def sync_portfolio() -> dict[str, Any]:
    """Manually trigger IB portfolio sync."""
    from app.execution.ib_sync import IBSyncService
    try:
        svc = IBSyncService()
        state = svc.sync_portfolio()
        if not state:
            return {"ok": False, "error": "Sync returned no state"}
        return {
            "ok": True,
            "nlv": state.net_liquidation,
            "settled_cash": state.settled_cash,
            "positions": len(state.positions),
            "is_stale": state.is_stale,
            "source": state.source,
        }
    except Exception as e:
        logger.error("Sync failed: %s", e)
        return {"ok": False, "error": str(e)}
