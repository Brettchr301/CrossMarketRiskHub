"""Query helpers for the election database."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.election.db.models import (
    AlphaModelPrediction,
    ArbitrageOpportunity,
    BlendedProbability,
    MarketContract,
    MarketQuote,
    Race,
)


def latest_quotes_by_platform(db: Session, race_id: int) -> list[dict[str, Any]]:
    """Get the latest quote from each platform for a race."""
    contracts = db.execute(
        select(MarketContract).where(MarketContract.race_id == race_id, MarketContract.active == True)
    ).scalars().all()

    results = []
    for c in contracts:
        q = db.execute(
            select(MarketQuote)
            .where(MarketQuote.contract_id == c.id)
            .order_by(MarketQuote.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()
        if q:
            results.append({
                "platform": c.platform,
                "contract_id": c.id,
                "yes_bid": q.yes_bid,
                "yes_ask": q.yes_ask,
                "last_price": q.last_price,
                "volume_24h": q.volume_24h,
                "liquidity_score": q.liquidity_score,
                "as_of": q.as_of,
            })
    return results


def active_arbs(db: Session, min_edge: float = 0.0, limit: int = 50) -> list[ArbitrageOpportunity]:
    """Get active arbitrage opportunities above minimum edge."""
    return list(db.execute(
        select(ArbitrageOpportunity)
        .where(ArbitrageOpportunity.status == "active", ArbitrageOpportunity.net_edge_pct >= min_edge)
        .order_by(ArbitrageOpportunity.net_edge_pct.desc())
        .limit(limit)
    ).scalars().all())


def expire_old_arbs(db: Session, max_age_hours: int = 24):
    """Mark arbs older than max_age_hours as expired."""
    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    old = db.execute(
        select(ArbitrageOpportunity).where(
            ArbitrageOpportunity.status == "active",
            ArbitrageOpportunity.detected_at < cutoff,
        )
    ).scalars().all()

    for arb in old:
        arb.status = "expired"

    if old:
        db.commit()
    return len(old)


def quote_count_by_platform(db: Session) -> dict[str, int]:
    """Count total quotes by platform."""
    rows = db.execute(
        select(MarketContract.platform, func.count(MarketQuote.id))
        .join(MarketQuote, MarketQuote.contract_id == MarketContract.id)
        .group_by(MarketContract.platform)
    ).all()
    return {platform: count for platform, count in rows}
