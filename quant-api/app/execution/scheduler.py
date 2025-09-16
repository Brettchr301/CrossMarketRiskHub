"""Execution Scheduler — APScheduler jobs for daily execution workflow.

Schedule (all times US/Eastern):
  4:30 PM — IB portfolio sync + trade plan generation
  4:45 PM — Discord notification with trade plan
  9:31 AM — Execute approved orders (1 min after open)
  11:00 AM — Check pending fills
  1:00 PM  — Check pending fills
  3:00 PM  — Check pending fills
"""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("US/Eastern")


def start_scheduler():
    """Start APScheduler with daily execution jobs.

    Returns the scheduler instance for shutdown.
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.warning("APScheduler not installed — scheduler disabled. "
                        "Install with: pip install APScheduler")
        return None

    scheduler = BackgroundScheduler(timezone=ET)

    # 4:30 PM ET — Sync portfolio + generate trade plan
    scheduler.add_job(
        _job_sync_and_plan,
        CronTrigger(hour=16, minute=30, timezone=ET),
        id="daily_sync_plan",
        name="Daily IB sync + plan generation",
        misfire_grace_time=600,
        replace_existing=True,
    )

    # 4:45 PM ET — Send Discord notification
    scheduler.add_job(
        _job_notify,
        CronTrigger(hour=16, minute=45, timezone=ET),
        id="daily_notify",
        name="Daily Discord notification",
        misfire_grace_time=600,
        replace_existing=True,
    )

    # 9:31 AM ET — Execute approved orders
    scheduler.add_job(
        _job_execute,
        CronTrigger(hour=9, minute=31, day_of_week="mon-fri", timezone=ET),
        id="daily_execute",
        name="Execute approved orders",
        misfire_grace_time=120,
        replace_existing=True,
    )

    # Fill checks — 11 AM, 1 PM, 3 PM ET
    for hour in (11, 13, 15):
        scheduler.add_job(
            _job_check_fills,
            CronTrigger(hour=hour, minute=0, day_of_week="mon-fri", timezone=ET),
            id=f"check_fills_{hour}",
            name=f"Check fills at {hour}:00",
            misfire_grace_time=300,
            replace_existing=True,
        )

    # Expire stale plans — midnight ET
    scheduler.add_job(
        _job_expire_plans,
        CronTrigger(hour=0, minute=5, timezone=ET),
        id="expire_plans",
        name="Expire stale trade plans",
        misfire_grace_time=3600,
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Execution scheduler started with %d jobs", len(scheduler.get_jobs()))
    return scheduler


# ── Job implementations ──────────────────────────────────────────

def _job_sync_and_plan():
    """4:30 PM: Sync IB portfolio and generate today's trade plan."""
    from app.execution.ib_sync import IBSyncService, is_trading_day, enrich_positions_with_universe
    from app.execution.trade_prioritizer import signals_to_trade_plan
    from app.execution.approval_queue import ApprovalQueue
    from app.execution.decision_bridge import load_cached_signals
    from app.portfolio.risk_manager import PortfolioConstraints

    now = datetime.now(ET)
    if not is_trading_day(now.date()):
        logger.info("Not a trading day — skipping sync/plan")
        return

    logger.info("Starting daily sync + plan generation")

    try:
        # 1. Sync portfolio
        sync = IBSyncService()
        state = sync.sync_portfolio()
        if not state:
            logger.error("Portfolio sync failed")
            return

        # 2. Enrich positions
        enrich_positions_with_universe(state)

        # 3. Load cached rebalance signals from latest backtest decision
        signals = load_cached_signals()
        if not signals:
            logger.info("No cached rebalance signals — run backtest first")
            return

        # 4. Build trade plan
        capital = state.net_liquidation if state.net_liquidation > 0 else 75_000.0
        plan = signals_to_trade_plan(
            signals=signals,
            portfolio_value=capital,
            settled_cash=state.settled_cash,
            unsettled_cash=state.unsettled_cash,
            current_positions=state.positions,
            constraints=PortfolioConstraints(),
        )

        # 5. Save plan
        queue = ApprovalQueue()
        plan_id = queue.save_plan(plan)
        logger.info("Trade plan %s saved: %d trades, %d rejected",
                     plan_id, len(plan.trades), len(plan.rejected_trades))

    except Exception as e:
        logger.error("Sync + plan job failed: %s", e, exc_info=True)


def _job_notify():
    """4:45 PM: Send Discord notification."""
    from app.execution.ib_sync import is_trading_day
    from app.execution.notifier import send_daily_notification
    from app.config import get_settings

    now = datetime.now(ET)
    if not is_trading_day(now.date()):
        return

    settings = get_settings()
    webhook_url = getattr(settings, "discord_webhook_url", "")
    try:
        send_daily_notification(webhook_url=webhook_url)
    except Exception as e:
        logger.error("Notification job failed: %s", e, exc_info=True)


def _job_execute():
    """9:31 AM: Execute approved orders."""
    from app.execution.ib_sync import is_trading_day
    from app.execution.order_executor import run_daily_execution
    from app.config import get_settings

    now = datetime.now(ET)
    if not is_trading_day(now.date()):
        logger.info("Not a trading day — skipping execution")
        return

    settings = get_settings()
    paper = not settings.live_trading_enabled
    logger.info("Starting daily execution (paper=%s)", paper)

    try:
        session = run_daily_execution(paper_trade=paper)
        logger.info("Execution complete: %d orders, $%.0f notional",
                     session.orders_submitted, session.notional_submitted)

        # Send execution report
        from app.execution.notifier import DiscordNotifier
        webhook_url = getattr(settings, "discord_webhook_url", "")
        notifier = DiscordNotifier(webhook_url=webhook_url)
        notifier.send_execution_report({
            "date": session.date,
            "orders": [
                {
                    "trade_id": o.trade_id,
                    "ticker": o.ticker,
                    "action": o.action,
                    "status": o.status,
                    "shares": o.shares,
                    "limit_price": o.limit_price,
                    "message": o.message,
                }
                for o in session.orders
            ],
            "drawdown_halt": session.drawdown_halt,
        })
    except Exception as e:
        logger.error("Execution job failed: %s", e, exc_info=True)


def _job_check_fills():
    """Periodic fill check during market hours."""
    from app.execution.order_executor import OrderExecutor
    from app.config import get_settings

    settings = get_settings()
    paper = not settings.live_trading_enabled
    executor = OrderExecutor(paper_trade=paper)
    try:
        results = executor.check_pending_fills()
        if results:
            logger.info("Fill check: %d updates", len(results))
    except Exception as e:
        logger.error("Fill check failed: %s", e)


def _job_expire_plans():
    """Midnight: expire stale plans."""
    from app.execution.approval_queue import ApprovalQueue
    from app.execution.db import get_connection

    conn = get_connection()
    try:
        queue = ApprovalQueue()
        queue._expire_old_plans(conn)
        logger.info("Expired stale plans cleanup done")
    except Exception as e:
        logger.error("Plan expiry job failed: %s", e)
    finally:
        conn.close()
