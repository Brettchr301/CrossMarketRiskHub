"""Extract tick-level prediction market data during the CLOSING HOURS of each election.

For each election cycle, pulls every trade / price point within the 48-hour window
surrounding polls closing. This is the highest-value period for:
- Vote-count response analysis
- Settlement-flow detection
- Price-discovery / efficient-market studies

Sources:
1. HuggingFace trades.parquet (27 GB, every Polymarket trade since 2020)
2. Polymarket CLOB prices-history at fidelity=1 (minute) for per-market pulls
3. Manifold bets API (tick-level, already timestamped to ms)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

HF_DIR = Path("C:/Users/BrettC/OneDrive/Documents/polymarket_huggingface_dump")
MARKETS_PARQUET = HF_DIR / "markets.parquet"
TRADES_PARQUET = HF_DIR / "trades.parquet"

# Election dates (polls close ~8pm-11pm ET on these days)
ELECTION_DATES = {
    2018: datetime(2018, 11, 6),
    2019: datetime(2019, 11, 5),
    2020: datetime(2020, 11, 3),
    2021: datetime(2021, 11, 2),
    2022: datetime(2022, 11, 8),
    2023: datetime(2023, 11, 7),
    2024: datetime(2024, 11, 5),
    2025: datetime(2025, 11, 4),
}

# Closing-hours window: 24 hours BEFORE polls close through 48 hours AFTER
# Total = 72 hours around election day
WINDOW_BEFORE_HOURS = 24
WINDOW_AFTER_HOURS = 48

ELECTION_RE = re.compile(
    r"\belection\b|senate|\bhouse\b|governor|president|midterm|"
    r"primary|caucus|nominat|congress|gubernator|mayor|recall|"
    r"ballot|referendum|legislature|democrat|republican|"
    r"\bvote|electoral|incumbent|trump|biden|harris|desantis",
    re.I,
)


def load_election_markets() -> pd.DataFrame:
    """Load the pre-filtered election markets index from HF dump."""
    logger.info("Loading markets.parquet from %s", MARKETS_PARQUET)
    df = pd.read_parquet(MARKETS_PARQUET)
    logger.info("Total Polymarket markets in HF dump: %d", len(df))

    # Find question column
    q_col = "question" if "question" in df.columns else "title"
    mask = df[q_col].fillna("").str.contains(ELECTION_RE, regex=True, na=False)
    election_df = df[mask].copy()
    logger.info("Election markets: %d", len(election_df))

    # Map each market to its closest election cycle
    # Use end_date field if present, else infer from question
    def infer_cycle(row) -> int:
        q = str(row.get(q_col, ""))
        years = re.findall(r"\b(20\d{2})\b", q)
        years = [int(y) for y in years if 2018 <= int(y) <= 2028]
        if years:
            return max(years)
        # Fallback: use end_date
        end = row.get("end_date")
        if pd.notna(end):
            try:
                ts = pd.to_datetime(end)
                return ts.year
            except Exception:
                pass
        return 0

    election_df["cycle"] = election_df.apply(infer_cycle, axis=1)
    return election_df


def extract_closing_hours_trades(markets_df: pd.DataFrame) -> pd.DataFrame:
    """Filter trades.parquet for trades in closing-hours windows across all cycles.

    Returns a DataFrame with columns: timestamp, market_id, question, price, cycle, size
    """
    logger.info("Loading trades.parquet (27+ GB, this may take a minute)...")
    # Read trades in chunks to avoid OOM
    from pyarrow import parquet as pq

    pf = pq.ParquetFile(TRADES_PARQUET)
    logger.info("Trades parquet schema: %s", pf.schema_arrow)
    logger.info("Trades row groups: %d, total rows: %d", pf.num_row_groups, pf.metadata.num_rows)

    # Build windows (UTC timestamps, approximate)
    windows: list[tuple[int, datetime, datetime]] = []
    for cycle, election_dt in ELECTION_DATES.items():
        # Polls close late ET → use UTC+5 offset
        end_poll = election_dt.replace(hour=23, minute=59)  # day end UTC
        win_start = end_poll - timedelta(hours=WINDOW_BEFORE_HOURS)
        win_end = end_poll + timedelta(hours=WINDOW_AFTER_HOURS)
        windows.append((cycle, win_start, win_end))

    # Get election market IDs by cycle
    mkt_by_cycle: dict[int, set[str]] = {}
    for cycle in ELECTION_DATES:
        subset = markets_df[markets_df["cycle"] == cycle]
        ids = set(subset["id"].astype(str))
        mkt_by_cycle[cycle] = ids
        logger.info("Cycle %d: %d election markets", cycle, len(ids))

    # Read trades in chunks, filter
    chunk_size = 500_000
    all_filtered: list[pd.DataFrame] = []

    col_names = pf.schema_arrow.names
    logger.info("Trade columns: %s", col_names)

    # Identify timestamp and market columns
    ts_col = next((c for c in ["timestamp", "created_at", "time", "block_time"] if c in col_names), None)
    mkt_col = next((c for c in ["market_id", "condition_id", "market", "conditionId"] if c in col_names), None)
    price_col = next((c for c in ["price", "price_usdc", "px"] if c in col_names), None)

    if not ts_col or not mkt_col or not price_col:
        logger.error("Missing required columns. ts=%s mkt=%s price=%s", ts_col, mkt_col, price_col)
        return pd.DataFrame()

    logger.info("Using columns: timestamp=%s market=%s price=%s", ts_col, mkt_col, price_col)

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=chunk_size, columns=[ts_col, mkt_col, price_col])):
        df = batch.to_pandas()
        if df.empty:
            continue

        # Normalize timestamp
        if df[ts_col].dtype != "datetime64[ns]":
            try:
                df[ts_col] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
            except Exception:
                df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

        df[mkt_col] = df[mkt_col].astype(str)
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

        # For each cycle, filter
        for cycle, win_start, win_end in windows:
            market_ids = mkt_by_cycle.get(cycle, set())
            if not market_ids:
                continue

            mask = (
                (df[ts_col] >= win_start)
                & (df[ts_col] <= win_end)
                & (df[mkt_col].isin(market_ids))
                & df[price_col].between(0, 1)
            )
            matched = df[mask].copy()
            if matched.empty:
                continue
            matched["cycle"] = cycle
            all_filtered.append(matched)

        if (batch_idx + 1) % 10 == 0:
            total_so_far = sum(len(d) for d in all_filtered)
            logger.info("Batch %d: %d matching trades so far", batch_idx + 1, total_so_far)

    if not all_filtered:
        return pd.DataFrame()

    result = pd.concat(all_filtered, ignore_index=True)
    result.rename(columns={ts_col: "timestamp", mkt_col: "market_id", price_col: "price"}, inplace=True)
    logger.info("Total closing-hours trades extracted: %d", len(result))
    return result


def merge_questions(trades_df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
    """Add question text to trades by joining on market_id."""
    q_col = "question" if "question" in markets_df.columns else "title"
    mkt_lookup = markets_df.set_index(markets_df["id"].astype(str))[q_col].to_dict()
    trades_df["question"] = trades_df["market_id"].map(mkt_lookup)
    return trades_df


def ingest_to_db(trades_df: pd.DataFrame) -> int:
    """Insert closing-hours trades into HistoricalQuote table."""
    from app.election.db.session import get_session_factory, init_election_db, _get_engine
    from app.election.db.models import ElectionBase
    from app.election.db.historical_models import HistoricalQuote
    from app.election.mappings.race_linker import link_contract_to_race

    init_election_db()
    ElectionBase.metadata.create_all(_get_engine())
    db = get_session_factory()()

    n = 0
    questions_linked: dict[str, int | None] = {}

    try:
        for _, row in trades_df.iterrows():
            question = str(row.get("question") or f"market_{row['market_id']}")
            if question not in questions_linked:
                questions_linked[question] = link_contract_to_race(question).race_id

            db.add(HistoricalQuote(
                race_id=questions_linked[question],
                platform="polymarket_hf",
                platform_market_id=f"hf_{row['market_id']}",
                question=question,
                cycle=int(row["cycle"]),
                price=float(row["price"]),
                as_of=row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime") else row["timestamp"],
            ))
            n += 1
            if n % 10000 == 0:
                db.commit()
                logger.info("Inserted %d rows...", n)

        db.commit()
    finally:
        db.close()

    return n


if __name__ == "__main__":
    logger.info("=== Closing Hours Tick Extraction ===")

    markets_df = load_election_markets()

    # Save filtered election markets index for reuse
    idx_path = HF_DIR / "election_markets_with_cycle.csv"
    markets_df[[c for c in markets_df.columns if c in ("id", "question", "title", "cycle", "end_date", "volume")]].to_csv(idx_path, index=False)
    logger.info("Saved election index: %s", idx_path)

    trades_df = extract_closing_hours_trades(markets_df)
    if trades_df.empty:
        logger.error("No closing-hours trades found")
        exit(1)

    trades_df = merge_questions(trades_df, markets_df)

    # Save extracted trades as CSV for inspection
    out_csv = HF_DIR / "closing_hours_trades.csv"
    trades_df.to_csv(out_csv, index=False)
    logger.info("Saved CSV: %s", out_csv)

    # Breakdown by cycle
    for cycle, count in trades_df.groupby("cycle").size().items():
        logger.info("Cycle %d: %d closing-hours trades", cycle, count)

    # Ingest to DB
    inserted = ingest_to_db(trades_df)
    logger.info("Inserted %d rows into HistoricalQuote", inserted)
    print(f"\nDone. {inserted:,} closing-hours trades ingested.")
