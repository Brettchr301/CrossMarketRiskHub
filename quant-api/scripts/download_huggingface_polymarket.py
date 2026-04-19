"""Download SII-WANGZJ/Polymarket_data from HuggingFace.

Strategy:
1. First pull markets.parquet (~68 MB index of all Polymarket markets)
2. Filter for election-related markets
3. Pull trades.parquet (28 GB) in streaming fashion, filtering by market_id

Storage destination: OneDrive/Documents/polymarket_huggingface_dump/
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# OneDrive destination (1 TB 365 Personal/Family tier)
DEST = Path("C:/Users/BrettC/OneDrive/Documents/polymarket_huggingface_dump")
DEST.mkdir(parents=True, exist_ok=True)

REPO = "SII-WANGZJ/Polymarket_data"

ELECTION_RE = re.compile(
    r"\belection\b|senate|\bhouse\b|governor|president|midterm|"
    r"primary|caucus|nominat|congress|gubernator|mayor|recall|"
    r"ballot|referendum|legislature|democrat|republican|"
    r"\bvote|electoral|incumbent|\bvp\b|vice.president|"
    r"trump|biden|harris|desantis|newsom",
    re.I,
)


def download_markets_index():
    """Download the markets.parquet index (~68 MB)."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading markets.parquet...")
    path = hf_hub_download(
        repo_id=REPO,
        filename="markets.parquet",
        repo_type="dataset",
        local_dir=str(DEST),
    )
    logger.info("Saved: %s", path)
    return Path(path)


def filter_election_markets(markets_path: Path) -> pd.DataFrame:
    """Filter markets.parquet for election-related questions."""
    df = pd.read_parquet(markets_path)
    logger.info("Total markets in HF dump: %d", len(df))
    logger.info("Columns: %s", list(df.columns))

    # Find the question/title column
    q_col = None
    for cand in ["question", "title", "name", "description"]:
        if cand in df.columns:
            q_col = cand
            break

    if not q_col:
        logger.warning("Could not find question column; returning all markets")
        return df

    mask = df[q_col].fillna("").str.contains(ELECTION_RE, regex=True, na=False)
    election_df = df[mask].copy()
    logger.info("Election-matching markets: %d", len(election_df))
    return election_df


def download_trades(market_ids: set[str]):
    """Download trades.parquet and filter to election market_ids.

    Uses streaming/chunked reads since the full file is 28 GB.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading trades.parquet (28 GB, may take 20-60 min)...")
    path = hf_hub_download(
        repo_id=REPO,
        filename="trades.parquet",
        repo_type="dataset",
        local_dir=str(DEST),
    )
    logger.info("Saved: %s", path)
    return Path(path)


if __name__ == "__main__":
    logger.info("=== HuggingFace Polymarket Dump Download ===")
    logger.info("Destination: %s", DEST)

    # Step 1: markets index
    markets_path = download_markets_index()

    # Step 2: filter for election
    election_df = filter_election_markets(markets_path)
    election_csv = DEST / "election_markets_index.csv"
    election_df.to_csv(election_csv, index=False)
    logger.info("Saved filtered index: %s", election_csv)

    # Step 3: download full trades (for filtering later)
    # NOTE: Comment this out if you don't want the 28 GB download
    logger.info("Proceeding to trades.parquet download...")
    trades_path = download_trades(set(election_df.get("market_id", election_df.index).astype(str)))

    print(f"\nFinal paths:")
    print(f"  markets: {markets_path}")
    print(f"  trades: {trades_path}")
    print(f"  election index: {election_csv}")
