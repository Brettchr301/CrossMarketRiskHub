"""Ingest Kaggle Polymarket 2024 US Election state-level data.

Source: https://www.kaggle.com/datasets/pbizil/polymarket-2024-us-election-state-data

Dataset layout after unzip under data/kaggle_polymarket_2024/:
    csv_minute/<ST>_minutely.csv   (election-day minute bars)
    csv_hour/<ST>_hourly.csv       (Aug-Nov 2024 hourly bars)
    csv_day/<ST>_daily.csv         (Mar-Nov 2024 daily bars)
    csv_week/<ST>_weekly.csv
    csv_month/<ST>_monthly.csv

Columns: Date (UTC), Timestamp (UTC), Donald Trump, Kamala Harris, Other
Each column after timestamp is a YES price (0-1) for that candidate winning the
state.  The "Other" column is the residual probability (third-party / tie).

This script emits one HistoricalQuote per (state, candidate, timestamp) at
hourly resolution, links to canonical races via link_contract_to_race, and
commits to election.historical_quotes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.election.db.historical_models import HistoricalQuote  # noqa: E402
from app.election.db.session import get_session_factory, init_election_db  # noqa: E402
from app.election.mappings.race_linker import link_contract_to_race  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kaggle-poly24")

DATA_ROOT = ROOT / "data" / "kaggle_polymarket_2024"
CANDIDATES = [
    ("Donald Trump", "R"),
    ("Kamala Harris", "D"),
]


def load_state(csv_path: Path, resolution: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["Timestamp (UTC)"], unit="s", utc=True).dt.tz_convert(None)
    return df


def ingest(resolution: str = "hour", cycle: int = 2024, limit_per_state: int | None = None) -> dict:
    folder = {
        "minute": "csv_minute",
        "hour": "csv_hour",
        "day": "csv_day",
        "week": "csv_week",
        "month": "csv_month",
    }[resolution]
    base = DATA_ROOT / folder
    if not base.exists():
        raise SystemExit(f"Missing dataset folder: {base}")

    init_election_db()
    factory = get_session_factory()

    total_rows = 0
    total_inserted = 0
    linked_races: set[int] = set()
    per_state: dict[str, int] = {}

    with factory() as s:
        for csv_file in sorted(base.glob("*.csv")):
            state = csv_file.stem.split("_")[0]
            df = load_state(csv_file, resolution)
            if limit_per_state:
                df = df.head(limit_per_state)
            total_rows += len(df)
            n_state = 0
            for cand, party in CANDIDATES:
                if cand not in df.columns:
                    continue
                question = f"Will {cand} win {state} in the 2024 Presidential Election?"
                link = link_contract_to_race(question)
                if link.race_id is not None:
                    linked_races.add(link.race_id)
                market_id = f"poly2024.state.{state}.{cand.replace(' ', '_')}"
                for _, row in df.iterrows():
                    try:
                        price = float(row[cand])
                    except (TypeError, ValueError):
                        continue
                    if not (0 <= price <= 1):
                        continue
                    ts = row["ts"].to_pydatetime().replace(tzinfo=None)
                    s.add(HistoricalQuote(
                        race_id=link.race_id,
                        platform="polymarket",
                        platform_market_id=market_id,
                        question=question,
                        cycle=cycle,
                        price=price,
                        as_of=ts,
                    ))
                    n_state += 1
                    total_inserted += 1
                    if total_inserted % 2000 == 0:
                        s.commit()
            per_state[state] = n_state
            log.info("State %s: %d quotes", state, n_state)
        s.commit()

    return {
        "states": len(per_state),
        "rows_read": total_rows,
        "quotes_inserted": total_inserted,
        "linked_races": len(linked_races),
        "per_state": per_state,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default="hour",
                    choices=["minute", "hour", "day", "week", "month"])
    ap.add_argument("--limit-per-state", type=int, default=None)
    args = ap.parse_args()

    result = ingest(args.resolution, limit_per_state=args.limit_per_state)
    print("\n==== Kaggle Polymarket 2024 ingest summary ====")
    print(f"  resolution:       {args.resolution}")
    print(f"  states:           {result['states']}")
    print(f"  rows read:        {result['rows_read']}")
    print(f"  quotes inserted:  {result['quotes_inserted']}")
    print(f"  linked races:     {result['linked_races']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
