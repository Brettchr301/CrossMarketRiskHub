"""Notifier — Discord webhook + optional email for daily trade summaries.

Fires at 4:45 PM ET after portfolio sync and trade plan generation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, date
from typing import Any

logger = logging.getLogger(__name__)

# Discord embed color codes
COLOR_GREEN = 0x2ECC71     # Profit / buys
COLOR_RED = 0xE74C3C       # Loss / sells
COLOR_YELLOW = 0xF1C40F    # Warning
COLOR_BLUE = 0x3498DB      # Info


class DiscordNotifier:
    """Sends formatted execution summaries to a Discord webhook.

    Parameters
    ----------
    webhook_url : str
        Discord webhook URL (Settings → Integrations → Webhooks).
    enabled : bool
        If False, log messages only (no HTTP calls).
    """

    def __init__(self, webhook_url: str = "", enabled: bool = True):
        self.webhook_url = webhook_url
        self.enabled = enabled and bool(webhook_url)

    # ────────────────────────────────────────────────────────────────
    # MAIN: DAILY SUMMARY
    # ────────────────────────────────────────────────────────────────

    def send_daily_summary(
        self,
        portfolio_state: dict[str, Any],
        trade_plan: dict[str, Any] | None,
        performance: dict[str, Any] | None = None,
    ) -> bool:
        """Send the full daily execution summary to Discord.

        Parameters
        ----------
        portfolio_state : dict
            From ib_sync: nlv, settled_cash, unsettled_cash, positions, etc.
        trade_plan : dict or None
            From approval_queue.get_current_plan() — includes plan, trades, rejected.
        performance : dict or None
            From approval_queue.get_performance_stats().

        Returns
        -------
        bool — True if sent successfully (or logging-only mode).
        """
        embeds = []

        # ── Portfolio Status ─────────────────────────────────────
        embeds.append(self._build_portfolio_embed(portfolio_state))

        # ── Trade Plan ───────────────────────────────────────────
        if trade_plan and trade_plan.get("trades"):
            embeds.append(self._build_trade_plan_embed(trade_plan))

        # ── Rejected Trades ──────────────────────────────────────
        if trade_plan and trade_plan.get("rejected"):
            embeds.append(self._build_rejected_embed(trade_plan["rejected"]))

        # ── Performance ──────────────────────────────────────────
        if performance and performance.get("total_trades", 0) > 0:
            embeds.append(self._build_performance_embed(performance))

        return self._send(embeds)

    def send_execution_report(self, session: dict[str, Any]) -> bool:
        """Send post-execution report (after 9:31 AM orders)."""
        orders = session.get("orders", [])
        if not orders:
            return True

        submitted = [o for o in orders if o.get("status") in ("SUBMITTED", "PAPER_LOGGED")]
        rejected = [o for o in orders if o.get("status") == "REJECTED_SAFETY"]
        failed = [o for o in orders if o.get("status") == "FAILED"]

        lines = [f"**Execution Report — {session.get('date', 'today')}**\n"]

        if submitted:
            lines.append(f"✅ **Submitted: {len(submitted)}**")
            for o in submitted:
                mode = "📝 PAPER" if o.get("status") == "PAPER_LOGGED" else "🔴 LIVE"
                lines.append(
                    f"  {mode} {o['action']} {o.get('shares', '?')}× "
                    f"{o['ticker']} @ ${o.get('limit_price', 0):.2f}"
                )

        if rejected:
            lines.append(f"\n⛔ **Blocked by safety rails: {len(rejected)}**")
            for o in rejected:
                lines.append(f"  {o['ticker']}: {o.get('message', 'safety')}")

        if failed:
            lines.append(f"\n❌ **Failed: {len(failed)}**")
            for o in failed:
                lines.append(f"  {o['ticker']}: {o.get('message', 'error')}")

        if session.get("drawdown_halt"):
            lines.append("\n🚨 **DRAWDOWN HALT ACTIVE** — no BUY orders permitted")

        embed = {
            "title": "📊 Execution Report",
            "description": "\n".join(lines),
            "color": COLOR_GREEN if not rejected and not failed else COLOR_YELLOW,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self._send([embed])

    def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """Send a one-off alert (errors, drawdown events, etc.)."""
        color = {"info": COLOR_BLUE, "warning": COLOR_YELLOW,
                 "error": COLOR_RED}.get(level, COLOR_BLUE)
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return self._send([embed])

    # ────────────────────────────────────────────────────────────────
    # EMBED BUILDERS
    # ────────────────────────────────────────────────────────────────

    def _build_portfolio_embed(self, state: dict[str, Any]) -> dict:
        positions = state.get("positions", [])
        nlv = state.get("nlv", 0)
        cash = state.get("settled_cash", 0)
        unsettled = state.get("unsettled_cash", 0)

        # Position summary
        pos_lines = []
        total_unrealized = 0
        for p in sorted(positions, key=lambda x: abs(x.get("unrealized_pnl", 0)),
                        reverse=True)[:15]:
            pnl = p.get("unrealized_pnl", 0)
            total_unrealized += pnl
            arrow = "🟢" if pnl >= 0 else "🔴"
            pct = (pnl / p.get("market_value", 1)) * 100 if p.get("market_value") else 0
            pos_lines.append(
                f"{arrow} **{p.get('ticker', '?')}** "
                f"{p.get('shares', 0)} shr @ ${p.get('market_price', 0):.2f} "
                f"({pct:+.1f}%)"
            )

        if len(positions) > 15:
            pos_lines.append(f"  _...and {len(positions) - 15} more_")

        fields = [
            {"name": "💰 NLV", "value": f"${nlv:,.0f}", "inline": True},
            {"name": "💵 Settled Cash", "value": f"${cash:,.0f}", "inline": True},
            {"name": "⏳ Unsettled", "value": f"${unsettled:,.0f}", "inline": True},
            {"name": "📊 Positions", "value": f"{len(positions)}", "inline": True},
            {"name": "📈 Unrealized P&L",
             "value": f"${total_unrealized:+,.0f}", "inline": True},
        ]

        return {
            "title": f"🏦 Portfolio Status — {date.today().isoformat()}",
            "description": "\n".join(pos_lines) if pos_lines else "_No positions_",
            "color": COLOR_GREEN if total_unrealized >= 0 else COLOR_RED,
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_trade_plan_embed(self, plan_data: dict[str, Any]) -> dict:
        plan = plan_data.get("plan", {})
        trades = plan_data.get("trades", [])

        lines = [
            f"**Plan ID:** `{plan.get('plan_id', '?')}`",
            f"**Cash Available:** ${plan.get('cash_available', 0):,.0f}",
            f"**Cash After Trades:** ${plan.get('cash_after_trades', 0):,.0f}",
            "",
            "**Ranked Trades (reply number to approve):**",
        ]

        sells = [t for t in trades if t.get("action") in ("CLOSE", "REDUCE")]
        buys = [t for t in trades if t.get("action") == "BUY"]

        if sells:
            lines.append("\n📤 **SELLS:**")
            for i, t in enumerate(sells, 1):
                flags = ""
                risk_flags = t.get("risk_flags")
                if risk_flags:
                    if isinstance(risk_flags, str):
                        try:
                            risk_flags = json.loads(risk_flags)
                        except (json.JSONDecodeError, TypeError):
                            risk_flags = []
                    flags = f" ⚠️ {', '.join(risk_flags)}" if risk_flags else ""
                lines.append(
                    f"  `{i}.` CLOSE **{t.get('ticker', '?')}** "
                    f"{t.get('shares', 0)} shr @ ~${t.get('limit_price', 0):.2f} "
                    f"(${t.get('estimated_cost', 0):,.0f}){flags}"
                )

        if buys:
            lines.append("\n📥 **BUYS (by EV rank):**")
            for t in buys:
                rank = t.get("rank", "?")
                ev = t.get("expected_value_pct", 0) or 0
                conv = t.get("conviction_score", 0) or 0
                flags = ""
                risk_flags = t.get("risk_flags")
                if risk_flags:
                    if isinstance(risk_flags, str):
                        try:
                            risk_flags = json.loads(risk_flags)
                        except (json.JSONDecodeError, TypeError):
                            risk_flags = []
                    flags = f" ⚠️ {', '.join(risk_flags)}" if risk_flags else ""
                dep = ""
                if t.get("depends_on_trade_id"):
                    dep = " 🕐 T+1"
                lines.append(
                    f"  `{rank}.` BUY **{t.get('ticker', '?')}** "
                    f"{t.get('shares', 0)} shr @ ~${t.get('limit_price', 0):.2f} "
                    f"(${t.get('estimated_cost', 0):,.0f}) "
                    f"EV:{ev:+.1f}% Conv:{conv:.0f}{flags}{dep}"
                )

        return {
            "title": f"📋 Trade Plan — {len(trades)} trades",
            "description": "\n".join(lines),
            "color": COLOR_BLUE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": f"Expires: {plan.get('expires_at', 'N/A')}"},
        }

    def _build_rejected_embed(self, rejected: list[dict]) -> dict:
        lines = []
        for r in rejected:
            lines.append(
                f"❌ **{r.get('ticker', '?')}** ({r.get('action', '?')}): "
                f"{r.get('reason', 'no reason')}"
            )
            if r.get("would_need"):
                lines.append(f"   _Would need: ${r['would_need']:,.0f}_")

        return {
            "title": f"🚫 Rejected Trades ({len(rejected)})",
            "description": "\n".join(lines),
            "color": COLOR_RED,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_performance_embed(self, perf: dict[str, Any]) -> dict:
        net = perf.get("net_realized", 0)
        return {
            "title": "📈 Performance Summary",
            "fields": [
                {"name": "Total Trades", "value": str(perf.get("total_trades", 0)),
                 "inline": True},
                {"name": "Buys / Sells",
                 "value": f"{perf.get('total_buys', 0)} / {perf.get('total_sells', 0)}",
                 "inline": True},
                {"name": "Net Realized",
                 "value": f"${net:+,.2f}",
                 "inline": True},
                {"name": "Commissions",
                 "value": f"${perf.get('total_commissions', 0):,.2f}",
                 "inline": True},
            ],
            "color": COLOR_GREEN if net >= 0 else COLOR_RED,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ────────────────────────────────────────────────────────────────
    # HTTP
    # ────────────────────────────────────────────────────────────────

    def _send(self, embeds: list[dict]) -> bool:
        """Send embeds to Discord webhook."""
        if not self.enabled:
            logger.info("Discord notification (disabled/no webhook): %d embeds",
                        len(embeds))
            for e in embeds:
                logger.info("  → %s", e.get("title", "untitled"))
            return True

        import httpx

        payload = {
            "username": "CrossMarket Risk Hub",
            "embeds": embeds[:10],  # Discord max 10 embeds per message
        }

        try:
            resp = httpx.post(
                self.webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code in (200, 204):
                logger.info("Discord notification sent (%d embeds)", len(embeds))
                return True
            else:
                logger.error("Discord webhook failed: %d %s",
                             resp.status_code, resp.text)
                return False
        except Exception as e:
            logger.error("Discord webhook error: %s", e)
            return False


# ── Convenience: daily notification function ─────────────────────

def send_daily_notification(webhook_url: str = "") -> bool:
    """Full daily notification pipeline. Called by scheduler at 4:45 PM ET.

    1. Load latest portfolio state
    2. Load current trade plan
    3. Load performance stats
    4. Format and send to Discord
    """
    from app.execution.ib_sync import get_latest_portfolio_state
    from app.execution.approval_queue import ApprovalQueue

    state = get_latest_portfolio_state()
    if not state:
        logger.warning("No portfolio state available for notification")
        return False

    queue = ApprovalQueue()
    plan = queue.get_current_plan()
    perf = queue.get_performance_stats()

    # Convert IBPortfolioState to dict for notifier
    state_dict = {
        "nlv": state.net_liquidation,
        "settled_cash": state.settled_cash,
        "unsettled_cash": state.unsettled_cash,
        "buying_power": state.buying_power,
        "positions": [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_cost": p.avg_cost,
                "market_price": p.market_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
                "commodity_type": p.commodity_type,
                "country": p.country,
            }
            for p in state.positions
        ],
    }

    notifier = DiscordNotifier(webhook_url=webhook_url)
    return notifier.send_daily_summary(state_dict, plan, perf)
