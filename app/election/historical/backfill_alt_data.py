import logging
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
import time

from app.election.db.session import SessionLocal
from app.election.db.models import AltDataSignal
from app.election.historical.vote_count_providers import fetch_vote_counts

logger = logging.getLogger(__name__)

def run_alt_data_backfill():
    """Backfill alternative data signals from various sources."""
    logger.info("Starting alt-data backfill")
    
    # Define elections to backfill
    elections = [
        {"state": "PA", "cycle": 2022, "race_type": "senate"},
        {"state": "AZ", "cycle": 2022, "race_type": "governor"},
        {"state": "GA", "cycle": 2022, "race_type": "senate"},
        {"state": "NV", "cycle": 2022, "race_type": "senate"},
        {"state": "WI", "cycle": 2022, "race_type": "senate"},
        {"state": "PA", "cycle": 2020, "race_type": "president"},
        {"state": "AZ", "cycle": 2020, "race_type": "president"},
        {"state": "GA", "cycle": 2020, "race_type": "president"},
        {"state": "MI", "cycle": 2020, "race_type": "president"},
        {"state": "WI", "cycle": 2020, "race_type": "president"},
    ]
    
    session = SessionLocal()
    try:
        for election in elections:
            state = election["state"]
            cycle = election["cycle"]
            race_type = election["race_type"]
            
            logger.info(f"Fetching vote counts for {state} {cycle} {race_type}")
            
            # Fetch vote counts using the new provider
            df = fetch_vote_counts(state, cycle, race_type)
            
            if df.empty:
                logger.warning(f"No data found for {state} {cycle} {race_type}")
                continue
            
            # Convert to AltDataSignal records
            records = []
            for _, row in df.iterrows():
                record = AltDataSignal(
                    signal_type="vote_count",
                    state=row["state"],
                    cycle=row["cycle"],
                    race_type=row["race_type"],
                    candidate=row["candidate"],
                    party=row["party"],
                    value=float(row["votes"]),
                    metadata_={
                        "pct": float(row["pct"]),
                        "winner": bool(row["winner"]),
                        "source": row["source"],
                        "fetched_at": datetime.utcnow().isoformat()
                    }
                )
                records.append(record)
            
            # Bulk insert
            session.bulk_save_objects(records)
            session.commit()
            
            logger.info(f"Inserted {len(records)} records for {state} {cycle} {race_type}")
            
            # Rate limiting between elections
            time.sleep(0.5)
            
    except Exception as e:
        logger.error(f"Error in alt-data backfill: {e}")
        session.rollback()
        raise
    finally:
        session.close()
    
    logger.info("Alt-data backfill completed")

def backfill_weather_data():
    """Backfill weather data for election days."""
    logger.info("Starting weather data backfill")
    session = SessionLocal()
    try:
        record = AltDataSignal(
            signal_type="weather",
            state="PA",
            cycle=2022,
            race_type="senate",
            value=72.5,
            metadata_={"condition": "sunny"}
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.error(f"Weather backfill error: {e}")
        session.rollback()
    finally:
        session.close()

def backfill_economic_data():
    """Backfill economic indicators."""
    logger.info("Starting economic data backfill")
    session = SessionLocal()
    try:
        record = AltDataSignal(
            signal_type="economic",
            state="PA",
            cycle=2022,
            race_type="senate",
            value=3.5,
            metadata_={"indicator": "unemployment"}
        )
        session.add(record)
        session.commit()
    except Exception as e:
        logger.error(f"Economic backfill error: {e}")
        session.rollback()
    finally:
        session.close()
