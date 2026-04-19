"""Fast normalization pipeline.

Avoids the slow per-question UPDATE pattern by:
1. Creating a temporary index on historical_quotes.question
2. Using bulk SQL patches instead of per-row ORM writes
3. Computing blended probabilities via pandas+pyarrow for speed
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select, func, distinct, delete, text
from sqlalchemy.orm import Session

from app.election.db.session import get_session_factory
from app.election.db.historical_models import HistoricalQuote, RaceOutcome
from app.election.db.models import MarketContract, BlendedProbability
from app.election.mappings.race_linker import link_contract_to_race
from app.election.mappings.direction_detector import detect_direction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def step_0_index(db: Session):
    """Create helpful index on question if missing."""
    try:
        db.execute(text("CREATE INDEX IF NOT EXISTS ix_hq_question ON historical_quotes(question)"))
        db.execute(text("CREATE INDEX IF NOT EXISTS ix_hq_platform_mid ON historical_quotes(platform, platform_market_id)"))
        db.commit()
        logger.info("Step 0: Indexes created")
    except Exception as exc:
        logger.warning("Index creation: %s", exc)


def step_1_dedupe(db: Session) -> int:
    before = db.execute(select(func.count(HistoricalQuote.id))).scalar()
    db.execute(text("""
        DELETE FROM historical_quotes WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY platform_market_id, as_of, price
                    ORDER BY id
                ) AS rn FROM historical_quotes
            ) t WHERE rn > 1
        )
    """))
    db.commit()
    after = db.execute(select(func.count(HistoricalQuote.id))).scalar()
    logger.info("Step 1: Dedupe removed %d rows (before=%d, after=%d)", before - after, before, after)
    return before - after


def step_2_relink_fast(db: Session):
    """Bulk relink using a JOIN on a temp table."""
    distinct_qs = db.execute(select(distinct(HistoricalQuote.question))).scalars().all()
    logger.info("Step 2: Processing %d distinct questions", len(distinct_qs))

    # Build mapping table in Python
    links = {}
    for q in distinct_qs:
        if not q:
            continue
        result = link_contract_to_race(q)
        if result.race_id is not None:
            links[q] = result.race_id

    logger.info("Step 2: %d questions have race_id", len(links))

    # Create temp table with mappings
    db.execute(text("DROP TABLE IF EXISTS temp_race_map"))
    db.execute(text("CREATE TEMP TABLE temp_race_map (question TEXT PRIMARY KEY, race_id INTEGER)"))

    # Bulk insert mappings
    from sqlalchemy import insert
    batch = [{"question": q, "race_id": rid} for q, rid in links.items()]
    # Use parameterized inserts
    if batch:
        db.execute(text("INSERT INTO temp_race_map (question, race_id) VALUES (:question, :race_id)"), batch)
    db.commit()
    logger.info("Step 2: Temp table populated with %d rows", len(batch))

    # Update all quotes at once via JOIN — MUCH faster than per-question UPDATEs
    # First: set race_id where mapping exists
    result = db.execute(text("""
        UPDATE historical_quotes
        SET race_id = (SELECT race_id FROM temp_race_map WHERE temp_race_map.question = historical_quotes.question)
        WHERE question IN (SELECT question FROM temp_race_map)
    """))
    logger.info("Step 2: Updated rows with race_id")
    # Set unlinked to NULL
    db.execute(text("""
        UPDATE historical_quotes SET race_id = NULL
        WHERE question NOT IN (SELECT question FROM temp_race_map)
    """))
    db.commit()
    db.execute(text("DROP TABLE temp_race_map"))
    db.commit()
    logger.info("Step 2: Relink complete")


def step_3_contracts_fast(db: Session):
    """Bulk-populate MarketContract via SQL, then patch is_inverted in chunks."""
    db.execute(text("DELETE FROM market_contracts"))
    db.commit()

    # Bulk insert distinct (platform, market_id, question, race_id) groups
    db.execute(text("""
        INSERT INTO market_contracts (race_id, candidate_id, platform, platform_market_id, platform_question, contract_type, is_inverted, active, discovered_at)
        SELECT
            COALESCE(race_id, 0),
            NULL,
            platform,
            platform_market_id,
            question,
            'binary',
            0,
            0,
            CURRENT_TIMESTAMP
        FROM (
            SELECT DISTINCT platform, platform_market_id, question, race_id
            FROM historical_quotes
            WHERE platform_market_id IS NOT NULL AND platform_market_id != ''
        ) t
    """))
    db.commit()

    n = db.execute(select(func.count(MarketContract.id))).scalar()
    logger.info("Step 3: Inserted %d contracts", n)

    # Now patch is_inverted for contracts with Republican-directional questions
    # Load all questions, detect direction, build R-list
    rows = db.execute(select(MarketContract.id, MarketContract.platform_question)).all()
    r_ids: list[int] = []
    direction_counts = {"D": 0, "R": 0, "unknown": 0, "I": 0}
    for cid, q in rows:
        if not q:
            direction_counts["unknown"] += 1
            continue
        d = detect_direction(q).yes_party
        direction_counts[d] = direction_counts.get(d, 0) + 1
        if d == "R":
            r_ids.append(cid)

    # Batch update R-directional contracts
    if r_ids:
        logger.info("Step 3: Marking %d R-directional contracts as is_inverted=true", len(r_ids))
        # Do in chunks of 1000
        for i in range(0, len(r_ids), 1000):
            chunk = r_ids[i:i + 1000]
            placeholders = ",".join(str(x) for x in chunk)
            db.execute(text(f"UPDATE market_contracts SET is_inverted = 1 WHERE id IN ({placeholders})"))
        db.commit()

    logger.info("Step 3: Direction breakdown: %s", direction_counts)


def step_4_blended_fast(db: Session) -> int:
    """Compute blended P(Dem wins) using pandas."""
    db.execute(text("DELETE FROM blended_probabilities"))
    db.commit()

    # Pull all linked quotes joined with their contract direction
    logger.info("Step 4: Loading quotes + directions...")
    sql = """
        SELECT
            hq.race_id,
            hq.price,
            hq.as_of,
            hq.platform,
            mc.is_inverted
        FROM historical_quotes hq
        LEFT JOIN market_contracts mc
          ON hq.platform = mc.platform AND hq.platform_market_id = mc.platform_market_id
        WHERE hq.race_id IS NOT NULL
    """
    df = pd.read_sql(sql, db.bind)
    logger.info("Step 4: Loaded %d linked quotes", len(df))

    if df.empty:
        return 0

    # Normalize to P(Dem wins)
    df["p_dem"] = df.apply(lambda r: (1.0 - r["price"]) if r["is_inverted"] else r["price"], axis=1)

    # Daily group
    df["date"] = pd.to_datetime(df["as_of"]).dt.normalize()
    grouped = df.groupby(["race_id", "date"]).agg(
        prob=("p_dem", "mean"),
        std=("p_dem", "std"),
        n=("platform", "nunique"),
    ).reset_index()

    grouped["std"] = grouped["std"].fillna(0.05)
    grouped["ci_low"] = (grouped["prob"] - 1.96 * grouped["std"]).clip(0, 1)
    grouped["ci_high"] = (grouped["prob"] + 1.96 * grouped["std"]).clip(0, 1)

    # Bulk insert
    records = [
        {
            "race_id": int(row["race_id"]),
            "candidate_id": None,
            "prob": float(row["prob"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "n_platforms": int(row["n"]),
            "as_of": row["date"].to_pydatetime(),
        }
        for _, row in grouped.iterrows()
    ]

    # Bulk insert in chunks
    for i in range(0, len(records), 5000):
        chunk = records[i:i + 5000]
        db.execute(text("""
            INSERT INTO blended_probabilities (race_id, candidate_id, prob, ci_low, ci_high, n_platforms, as_of)
            VALUES (:race_id, :candidate_id, :prob, :ci_low, :ci_high, :n_platforms, :as_of)
        """), chunk)
    db.commit()

    logger.info("Step 4: Inserted %d blended probability rows", len(records))
    return len(records)


def step_5_report(db: Session) -> dict[str, Any]:
    total = db.execute(select(func.count(HistoricalQuote.id))).scalar()
    linked = db.execute(select(func.count(HistoricalQuote.id)).where(HistoricalQuote.race_id.isnot(None))).scalar()
    by_platform = dict(db.execute(
        select(HistoricalQuote.platform, func.count(HistoricalQuote.id))
        .group_by(HistoricalQuote.platform)
    ).all())
    by_cycle = dict(db.execute(
        select(HistoricalQuote.cycle, func.count(HistoricalQuote.id))
        .group_by(HistoricalQuote.cycle).order_by(HistoricalQuote.cycle)
    ).all())
    contracts = db.execute(select(func.count(MarketContract.id))).scalar()
    inverted = db.execute(select(func.count(MarketContract.id)).where(MarketContract.is_inverted == True)).scalar()
    blended = db.execute(select(func.count(BlendedProbability.id))).scalar()
    outcomes = db.execute(select(func.count(RaceOutcome.id))).scalar()

    import os
    db_size_gb = os.path.getsize("C:/Users/BrettC/OneDrive/Documents/election_arb.db") / (1024**3)

    return {
        "total_quotes": total,
        "linked_quotes": linked,
        "link_pct": round(100 * linked / total, 1) if total else 0,
        "by_platform": by_platform,
        "by_cycle": by_cycle,
        "market_contracts": contracts,
        "inverted_contracts": inverted,
        "blended_probability_rows": blended,
        "race_outcomes": outcomes,
        "db_size_gb": round(db_size_gb, 3),
    }


def main():
    db = get_session_factory()()
    try:
        step_0_index(db)
        step_1_dedupe(db)
        step_2_relink_fast(db)
        step_3_contracts_fast(db)
        step_4_blended_fast(db)
        report = step_5_report(db)

        print("\n" + "=" * 60)
        print("NORMALIZATION COMPLETE")
        print("=" * 60)
        import json
        print(json.dumps(report, indent=2, default=str))
    finally:
        db.close()


if __name__ == "__main__":
    main()
