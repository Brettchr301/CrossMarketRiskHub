import pandas as pd
import requests
import logging
from typing import Optional
import time
import io

logger = logging.getLogger(__name__)

# Rate limiting: max 2 requests/second
_last_request_time = 0
RATE_LIMIT_DELAY = 0.5

def _rate_limit():
    """Enforce rate limit of max 2 requests per second."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()

def fetch_nyt_results_archive(state: str, cycle: int, race_type: str) -> pd.DataFrame:
    """Fetch NYT election results CSV for a given state, cycle, and race type.
    
    Args:
        state: Two-letter state code (e.g., "PA")
        cycle: Election year (2020, 2022, 2024)
        race_type: "president", "senate", "house", "governor"
    
    Returns:
        DataFrame with standardized columns, or empty DataFrame if not found.
    """
    _rate_limit()
    
    # Map cycles to election dates (YYYY-MM-DD)
    election_dates = {
        2020: "2020-11-03",
        2022: "2022-11-08",
        2024: "2024-11-05"
    }
    
    if cycle not in election_dates:
        logger.warning(f"No NYT URL pattern known for cycle {cycle}")
        return pd.DataFrame()
    
    date_str = election_dates[cycle]
    url = f"https://static01.nyt.com/elections-assets/pages/data/{date_str}/results-{race_type}.csv"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.debug(f"NYT URL {url} returned status {response.status_code}")
            return pd.DataFrame()
        
        # Read CSV and standardize columns
        df = pd.read_csv(io.StringIO(response.text))
        
        # Rename columns to match our standard format
        column_map = {}
        if 'candidate' in df.columns:
            column_map['candidate'] = 'candidate'
        if 'votes' in df.columns:
            column_map['votes'] = 'votes'
        if 'party' in df.columns:
            column_map['party'] = 'party'
        if 'pct' in df.columns:
            column_map['pct'] = 'pct'
        elif 'percent' in df.columns:
            column_map['percent'] = 'pct'
        if 'winner' in df.columns:
            column_map['winner'] = 'winner'
        
        df = df.rename(columns=column_map)
        
        # Ensure required columns exist
        required_cols = ['candidate', 'party', 'votes', 'pct', 'winner']
        for col in required_cols:
            if col not in df.columns:
                if col == 'pct':
                    df['pct'] = 0.0
                elif col == 'winner':
                    df['winner'] = False
                else:
                    df[col] = ''
        
        # Filter for the specific state if state column exists
        if 'state' in df.columns:
            df = df[df['state'] == state]
        elif 'state_code' in df.columns:
            df = df[df['state_code'] == state]
        
        # Add metadata columns
        df['state'] = state
        df['race_type'] = race_type
        df['cycle'] = cycle
        df['source'] = 'NYT'
        
        # Select and order final columns
        final_cols = ['state', 'race_type', 'cycle', 'candidate', 'party', 
                     'votes', 'pct', 'winner', 'source']
        df = df[final_cols]
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        logger.warning(f"Failed to fetch NYT data: {e}")
        return pd.DataFrame()

def fetch_state_sos_results(state: str, cycle: int, race_type: str) -> pd.DataFrame:
    """Scrape state Secretary of State websites via Wayback Machine."""
    _rate_limit()
    try:
        url = f"https://web.archive.org/web/2/https://sos.{state.lower()}.gov/elections/results"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
        
        dfs = pd.read_html(io.StringIO(response.text))
        if not dfs:
            return pd.DataFrame()
            
        df = dfs[0]
        df['state'] = state
        df['cycle'] = cycle
        df['race_type'] = race_type
        df['source'] = 'Wayback SOS'
        return df
    except Exception as e:
        logger.warning(f"Wayback SOS scraping failed: {e}")
        return pd.DataFrame()

def fetch_live_vote_counts(state: str, cycle: int, race_type: str) -> pd.DataFrame:
    """Main entry point: tries NYT first, then Wayback SOS."""
    # Try NYT first
    nyt_df = fetch_nyt_results_archive(state, cycle, race_type)
    if not nyt_df.empty:
        return nyt_df
    
    # Fall back to Wayback SOS
    return fetch_state_sos_results(state, cycle, race_type)
