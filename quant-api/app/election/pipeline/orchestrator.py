"""Election pipeline orchestrator.

Scheduled pipeline: ingest market data → detect arbitrage → run alpha model → alert.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

import requests

from app.election.config import get_election_settings
from app.election.db.session import get_session_factory, init_election_db

logger = logging.getLogger(__name__)

_running = False
_lock = threading.Lock()


def ingest_market_quotes(db) -> dict[str, Any]:
    """Fetch quotes from all 4 prediction market platforms."""
    from app.election.providers import (
        kalshi_election,
        metaculus,
        polymarket_election,
        predictit,
    )
    from app.election.db.models import MarketQuote, MarketContract
    from app.election.mappings.election_events import ELECTION_EVENT_MAPPINGS

    stats = {"polymarket": 0, "kalshi": 0, "predictit": 0, "metaculus": 0}
    now = datetime.now(UTC).replace(tzinfo=None)

    # Polymarket
    try:
        for event_id, mapping in ELECTION_EVENT_MAPPINGS.items():
            if mapping.poly_regex:
                quotes = polymarket_election.fetch_by_regex(mapping.poly_regex)
                for q in quotes[:3]:  # top 3 by volume
                    _upsert_quote(db, q, now)
                    stats["polymarket"] += 1
    except Exception as exc:
        logger.error("Polymarket ingestion failed: %s", exc)

    # Kalshi
    try:
        kalshi_data = kalshi_election.fetch_all_election_markets()
        for event_id, quotes in kalshi_data.items():
            for q in quotes[:3]:
                _upsert_quote(db, q, now)
                stats["kalshi"] += 1
    except Exception as exc:
        logger.error("Kalshi ingestion failed: %s", exc)

    # PredictIt
    try:
        pi_quotes = predictit.fetch_all_markets()
        for q in pi_quotes:
            _upsert_quote(db, q, now)
            stats["predictit"] += 1
    except Exception as exc:
        logger.error("PredictIt ingestion failed: %s", exc)

    # Metaculus
    try:
        mc_quotes = metaculus.fetch_election_questions()
        for q in mc_quotes:
            _upsert_quote(db, q, now)
            stats["metaculus"] += 1
    except Exception as exc:
        logger.error("Metaculus ingestion failed: %s", exc)

    db.commit()
    logger.info("Market ingestion complete: %s", stats)
    return stats


def _upsert_quote(db, quote: dict[str, Any], now: datetime):
    """Insert a market quote, auto-creating contract if needed."""
    from app.election.db.models import MarketContract, MarketQuote
    from sqlalchemy import select

    platform = quote.get("platform", "unknown")
    market_id = quote.get("platform_market_id", "")

    # Find or create contract
    contract = db.execute(
        select(MarketContract).where(
            MarketContract.platform == platform,
            MarketContract.platform_market_id == market_id,
        )
    ).scalar_one_or_none()

    if contract is None:
        contract = MarketContract(
            race_id=0,  # unlinked until matched
            platform=platform,
            platform_market_id=market_id,
            platform_question=quote.get("question", ""),
            active=True,
            discovered_at=now,
        )
        db.add(contract)
        db.flush()

    # Insert quote
    db.add(MarketQuote(
        contract_id=contract.id,
        yes_bid=quote.get("yes_bid", 0.0),
        yes_ask=quote.get("yes_ask", 0.0),
        last_price=quote.get("last_price", 0.0),
        volume_24h=quote.get("volume_24h", 0.0),
        open_interest=quote.get("open_interest", 0.0),
        liquidity_score=quote.get("liquidity_score", 0.5),
        as_of=now,
    ))


def run_arbitrage_detection(db) -> int:
    """Run all arbitrage detectors on latest quotes."""
    from app.election.arbitrage.engine import ArbitrageEngine
    from app.election.db.models import MarketContract, MarketQuote
    from sqlalchemy import select, func

    # Get latest quotes per contract
    subq = (
        select(
            MarketQuote.contract_id,
            func.max(MarketQuote.as_of).label("max_asof"),
        )
        .group_by(MarketQuote.contract_id)
        .subquery()
    )

    latest_quotes = db.execute(
        select(MarketQuote, MarketContract)
        .join(MarketContract, MarketQuote.contract_id == MarketContract.id)
        .join(subq, (MarketQuote.contract_id == subq.c.contract_id) & (MarketQuote.as_of == subq.c.max_asof))
    ).all()

    if not latest_quotes:
        return 0

    # Organize by race for cross-market detection
    quotes_by_race: dict[int, list[dict[str, Any]]] = {}
    quotes_by_race_platform: dict[tuple[int, str], list[dict[str, Any]]] = {}

    for mq, mc in latest_quotes:
        q = {
            "platform": mc.platform,
            "contract_id": mc.id,
            "race_id": mc.race_id,
            "yes_bid": mq.yes_bid,
            "yes_ask": mq.yes_ask,
            "liquidity_score": mq.liquidity_score,
        }
        quotes_by_race.setdefault(mc.race_id, []).append(q)
        quotes_by_race_platform.setdefault((mc.race_id, mc.platform), []).append(q)

    engine = ArbitrageEngine()
    signals = engine.run_all(
        db=db,
        quotes_by_race=quotes_by_race,
        quotes_by_race_platform=quotes_by_race_platform,
        aggregate_quotes={},
        component_probs={},
        market_probs={},
    )

    return len(signals)


def run_alpha_model(db) -> int:
    """Run correlation alpha model on all tracked races."""
    from app.election.correlation.alpha_model import run_alpha_model
    from app.election.db.models import Race
    from sqlalchemy import select

    race_ids = db.execute(select(Race.id)).scalars().all()
    if not race_ids:
        return 0

    signals = run_alpha_model(db, list(race_ids))
    return len(signals)


def send_discord_alert(message: str):
    """Send alert to Discord webhook if configured."""
    settings = get_election_settings()
    if not settings.discord_webhook_url:
        return
    try:
        requests.post(
            settings.discord_webhook_url,
            json={"content": message},
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Discord alert failed: %s", exc)


def run_full_pipeline():
    """Run the complete election pipeline once."""
    global _running
    with _lock:
        if _running:
            logger.info("Pipeline already running, skipping")
            return
        _running = True

    try:
        init_election_db()
        factory = get_session_factory()
        db = factory()

        try:
            # 1. Ingest market quotes
            stats = ingest_market_quotes(db)
            total_quotes = sum(stats.values())
            logger.info("Ingested %d quotes across platforms", total_quotes)

            # 2. Run arbitrage detection
            arb_count = run_arbitrage_detection(db)
            logger.info("Found %d arbitrage opportunities", arb_count)

            # 3. Run alpha model
            alpha_count = run_alpha_model(db)
            logger.info("Generated %d alpha predictions", alpha_count)

            # 4. Alert on significant arbs
            settings = get_election_settings()
            if arb_count > 0:
                from app.election.db.models import ArbitrageOpportunity
                from sqlalchemy import select

                hot_arbs = db.execute(
                    select(ArbitrageOpportunity)
                    .where(
                        ArbitrageOpportunity.status == "active",
                        ArbitrageOpportunity.net_edge_pct >= settings.arb_alert_min_edge_pct,
                    )
                    .order_by(ArbitrageOpportunity.net_edge_pct.desc())
                    .limit(5)
                ).scalars().all()

                for arb in hot_arbs:
                    send_discord_alert(
                        f"🎯 **Election Arb** [{arb.arb_type}] "
                        f"Net edge: {arb.net_edge_pct:.2f}%\n{arb.description}"
                    )
        finally:
            db.close()
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
    finally:
        with _lock:
            _running = False


def start_background_pipeline():
    """Start the election pipeline on a background thread with scheduling."""
    import time

    settings = get_election_settings()
    interval = settings.market_poll_interval_seconds

    def _loop():
        while True:
            try:
                run_full_pipeline()
            except Exception as exc:
                logger.error("Pipeline loop error: %s", exc)
            time.sleep(interval)

    thread = threading.Thread(target=_loop, daemon=True, name="election-pipeline")
    thread.start()
    logger.info("Election pipeline started (interval=%ds)", interval)
    return thread
