"""Normalize all election prediction market data in the DB.

Steps:
1. Deduplicate exact-duplicate quote rows
2. Relink all quotes against current race registry
3. Detect direction (D/R YES semantics) per question
4. Backfill MarketContract table with linked contracts + directions
5. Create/update BlendedProbability rows (P(Dem wins) across platforms)
6. Verify DB integrity
7. Emit final report
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select, func, update, distinct, delete
from sqlalchemy.orm import Session

from app.election.db.session import get_session_factory
from app.election.db.historical_models import HistoricalQuote, RaceOutcome
from app.election.db.models import MarketContract, BlendedProbability
from app.election.mappings.race_linker import link_contract_to_race
from app.election.mappings.direction_detector import detect_direction, normalize_price

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def step_1_dedupe(db: Session) -> int:
    """Remove exact-duplicate quote rows.

    Defined as: same (platform_market_id, as_of, price) tuple.
    """
    before = db.execute(select(func.count(HistoricalQuote.id))).scalar()

    # Find duplicates using a SQL window function
    sql = """
        DELETE FROM historical_quotes WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY platform_market_id, as_of, price
                    ORDER BY id
                ) AS rn
                FROM historical_quotes
            ) t WHERE rn > 1
        )
    """
    from sqlalchemy import text
    result = db.execute(text(sql))
    db.commit()

    after = db.execute(select(func.count(HistoricalQuote.id))).scalar()
    removed = before - after
    logger.info("Step 1: Dedupe — removed %d duplicates (before=%d, after=%d)", removed, before, after)
    return removed


def step_2_relink(db: Session) -> dict[str, int]:
    """Relink all quotes to race_id based on question text."""
    distinct_qs = db.execute(select(distinct(HistoricalQuote.question))).scalars().all()

    linked = 0
    unlinked = 0
    updated = 0

    for q in distinct_qs:
        if not q:
            continue
        link = link_contract_to_race(q)
        if link.race_id:
            linked += 1
        else:
            unlinked += 1
        db.execute(
            update(HistoricalQuote)
            .where(HistoricalQuote.question == q)
            .values(race_id=link.race_id)
        )
        updated += 1

    db.commit()
    logger.info("Step 2: Relink — %d questions processed, %d linked, %d unlinked",
                updated, linked, unlinked)
    return {"questions": updated, "linked": linked, "unlinked": unlinked}


def step_3_populate_contracts(db: Session) -> int:
    """Create MarketContract rows with direction flags."""
    # Clear existing contracts
    db.execute(delete(MarketContract))
    db.commit()

    # Get distinct (platform, market_id, question) combos
    rows = db.execute(
        select(
            HistoricalQuote.platform,
            HistoricalQuote.platform_market_id,
            HistoricalQuote.question,
            HistoricalQuote.race_id,
        ).distinct()
    ).all()

    n = 0
    by_direction = {"D": 0, "R": 0, "I": 0, "unknown": 0}

    for platform, mid, question, race_id in rows:
        if not mid:
            continue
        direction = detect_direction(question or "")
        is_inverted = (direction.yes_party == "R")  # normalize-to-D convention

        db.add(MarketContract(
            race_id=race_id or 0,
            candidate_id=None,
            platform=platform,
            platform_market_id=mid,
            platform_question=question,
            contract_type="binary",
            is_inverted=is_inverted,
            active=False,  # historical
        ))
        n += 1
        by_direction[direction.yes_party] += 1

        if n % 5000 == 0:
            db.commit()
            logger.info("Contracts inserted: %d", n)

    db.commit()
    logger.info("Step 3: Contracts — %d inserted. Direction breakdown: %s", n, by_direction)
    return n


def step_4_blended_probabilities(db: Session) -> int:
    """For each race, compute daily blended P(Dem wins) across platforms.

    Uses direction_detector to normalize, then averages by day per race.
    """
    # Clear existing blended
    db.execute(delete(BlendedProbability))
    db.commit()

    # Get all quotes with linked race_ids, joined to their contracts for direction
    logger.info("Step 4: Computing blended probabilities (this may take a minute)...")

    # Pull as DataFrame for efficient groupby
    rows = db.execute(
        select(
            HistoricalQuote.race_id,
            HistoricalQuote.question,
            HistoricalQuote.price,
            HistoricalQuote.as_of,
            HistoricalQuote.platform,
        ).where(HistoricalQuote.race_id.isnot(None))
    ).all()

    if not rows:
        logger.warning("No linked quotes for blended probability")
        return 0

    df = pd.DataFrame(rows, columns=["race_id", "question", "price", "as_of", "platform"])
    logger.info("Blended input: %d linked quotes", len(df))

    # Detect direction per unique question, cache
    unique_qs = df["question"].unique()
    q_direction: dict[str, str] = {}
    for q in unique_qs:
        if q:
            q_direction[q] = detect_direction(q).yes_party

    df["direction"] = df["question"].map(q_direction).fillna("unknown")

    # Filter to known directions only
    df = df[df["direction"].isin(["D", "R"])].copy()
    logger.info("After direction filter: %d quotes", len(df))

    # Normalize to P(Dem wins)
    df["p_dem"] = df.apply(
        lambda r: r["price"] if r["direction"] == "D" else 1.0 - r["price"], axis=1
    )

    # Daily average per race
    df["date"] = pd.to_datetime(df["as_of"]).dt.normalize()
    grouped = df.groupby(["race_id", "date"]).agg(
        prob=("p_dem", "mean"),
        std=("p_dem", "std"),
        n=("platform", "nunique"),
    ).reset_index()

    grouped["std"] = grouped["std"].fillna(0.05)

    n_inserted = 0
    for _, row in grouped.iterrows():
        prob = float(row["prob"])
        std = float(row["std"])
        ci_low = max(0.0, prob - 1.96 * std)
        ci_high = min(1.0, prob + 1.96 * std)
        db.add(BlendedProbability(
            race_id=int(row["race_id"]),
            candidate_id=None,
            prob=prob,
            ci_low=ci_low,
            ci_high=ci_high,
            n_platforms=int(row["n"]),
            as_of=row["date"].to_pydatetime(),
        ))
        n_inserted += 1
        if n_inserted % 10000 == 0:
            db.commit()
            logger.info("Blended inserted: %d", n_inserted)

    db.commit()
    logger.info("Step 4: Blended probabilities — %d rows inserted", n_inserted)
    return n_inserted


def step_5_integrity_check(db: Session) -> dict[str, Any]:
    """Verify DB integrity and emit a final report."""
    total_quotes = db.execute(select(func.count(HistoricalQuote.id))).scalar()
    by_platform = dict(db.execute(
        select(HistoricalQuote.platform, func.count(HistoricalQuote.id))
        .group_by(HistoricalQuote.platform)
    ).all())
    by_cycle = dict(db.execute(
        select(HistoricalQuote.cycle, func.count(HistoricalQuote.id))
        .group_by(HistoricalQuote.cycle).order_by(HistoricalQuote.cycle)
    ).all())

    linked = db.execute(
        select(func.count(HistoricalQuote.id))
        .where(HistoricalQuote.race_id.isnot(None))
    ).scalar()
    unlinked = total_quotes - linked

    contracts = db.execute(select(func.count(MarketContract.id))).scalar()
    inverted = db.execute(
        select(func.count(MarketContract.id))
        .where(MarketContract.is_inverted == True)
    ).scalar()
    blended = db.execute(select(func.count(BlendedProbability.id))).scalar()
    outcomes = db.execute(select(func.count(RaceOutcome.id))).scalar()

    return {
        "total_quotes": total_quotes,
        "linked_quotes": linked,
        "unlinked_quotes": unlinked,
        "link_pct": round(100 * linked / total_quotes, 1) if total_quotes else 0,
        "by_platform": by_platform,
        "by_cycle": by_cycle,
        "market_contracts": contracts,
        "inverted_contracts": inverted,
        "blended_probability_rows": blended,
        "race_outcomes": outcomes,
    }


def main():
    db = get_session_factory()()
    try:
        logger.info("=" * 60)
        logger.info("NORMALIZATION PIPELINE")
        logger.info("=" * 60)

        # Step 1: Dedupe
        removed = step_1_dedupe(db)

        # Step 2: Relink
        relink_stats = step_2_relink(db)

        # Step 3: Populate contracts with direction
        contracts = step_3_populate_contracts(db)

        # Step 4: Blended probabilities
        blended = step_4_blended_probabilities(db)

        # Step 5: Integrity check
        report = step_5_integrity_check(db)

        logger.info("=" * 60)
        logger.info("NORMALIZATION COMPLETE")
        logger.info("=" * 60)
        for k, v in report.items():
            logger.info("  %s: %s", k, v)

        return report
    finally:
        db.close()


if __name__ == "__main__":
    report = main()
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    import json
    print(json.dumps(report, indent=2, default=str))
