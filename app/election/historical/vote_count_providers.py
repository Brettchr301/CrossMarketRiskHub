import pandas as pd
import requests
import logging
import time
from typing import Optional
import re
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

def _standardize_dataframe(df: pd.DataFrame, state: str, cycle: int, 
                          race_type: str, source: str) -> pd.DataFrame:
    """Ensure DataFrame has required columns in correct order."""
    if df.empty:
        return pd.DataFrame(columns=['state', 'race_type', 'cycle', 'candidate', 'party', 'votes', 'pct', 'winner', 'source'])
    
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
    
    # Add metadata columns
    df['state'] = state
    df['race_type'] = race_type
    df['cycle'] = cycle
    df['source'] = source
    
    # Select and order final columns
    final_cols = ['state', 'race_type', 'cycle', 'candidate', 'party', 
                 'votes', 'pct', 'winner', 'source']
    
    # Keep only columns that exist
    existing_cols = [col for col in final_cols if col in df.columns]
    df = df[existing_cols]
    
    # Reorder and add any missing columns
    for col in final_cols:
        if col not in df.columns:
            df[col] = '' if col != 'pct' else 0.0
    
    return df[final_cols].reset_index(drop=True)

def fetch_openelections_results(state: str, cycle: int, race_type: str) -> pd.DataFrame:
    """Fetch certified results from OpenElections GitHub CSVs.
    
    URL pattern: https://raw.githubusercontent.com/openelections/
    openelections-data-{state_lower}/master/{cycle}/
    {cycle}{MM}{DD}__{state_lower}__general__{race_type}.csv
    
    Returns DataFrame with standardized columns or empty DataFrame on failure.
    """
    _rate_limit()
    
    # Map cycles to election dates (MMDD)
    election_dates = {
        2018: "1106",
        2020: "1103",
        2022: "1108",
        2024: "1105"
    }
    
    if cycle not in election_dates:
        logger.warning(f"No OpenElections data for cycle {cycle}")
        return pd.DataFrame()
    
    state_lower = state.lower()
    date_str = election_dates[cycle]
    
    # Map race types to OpenElections format
    race_map = {
        'president': 'president',
        'senate': 'senate',
        'house': 'house',
        'governor': 'governor'
    }
    
    if race_type not in race_map:
        logger.warning(f"Unsupported race type for OpenElections: {race_type}")
        return pd.DataFrame()
    
    url = (f"https://raw.githubusercontent.com/openelections/"
           f"openelections-data-{state_lower}/master/{cycle}/"
           f"{cycle}{date_str}__{state_lower}__general__{race_map[race_type]}.csv")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.debug(f"OpenElections URL {url} returned status {response.status_code}")
            return pd.DataFrame()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        # Standardize column names
        if 'candidate' not in df.columns and 'name' in df.columns:
            df = df.rename(columns={'name': 'candidate'})
        if 'votes' not in df.columns:
            votes_cols = [col for col in df.columns if 'votes' in col.lower()]
            if votes_cols:
                df = df.rename(columns={votes_cols[0]: 'votes'})
        if 'party' not in df.columns:
            party_cols = [col for col in df.columns if 'party' in col.lower()]
            if party_cols:
                df = df.rename(columns={party_cols[0]: 'party'})
        
        # Convert votes to numeric, handling commas
        if 'votes' in df.columns:
            df['votes'] = pd.to_numeric(df['votes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # Calculate percentage if not present
        if 'pct' not in df.columns and 'votes' in df.columns:
            total_votes = df['votes'].sum()
            if total_votes > 0:
                df['pct'] = (df['votes'] / total_votes) * 100
            else:
                df['pct'] = 0.0
        
        # Determine winner (highest votes)
        if 'winner' not in df.columns and 'votes' in df.columns and not df.empty:
            max_votes = df['votes'].max()
            df['winner'] = df['votes'] == max_votes
        
        return _standardize_dataframe(df, state, cycle, race_type, "OpenElections")
        
    except Exception as e:
        logger.warning(f"Failed to fetch OpenElections data: {e}")
        return pd.DataFrame()

def fetch_wikipedia_certified_results(state: str, cycle: int, race_type: str = "senate") -> pd.DataFrame:
    """Scrape Wikipedia infobox for certified election results.
    
    Uses MediaWiki API (api.php?action=parse) to get HTML, then parses
    the infobox table for candidate names, parties, and vote totals.
    """
    _rate_limit()
    
    # Construct page title based on election
    state_full = {
        'PA': 'Pennsylvania', 'AZ': 'Arizona', 'GA': 'Georgia',
        'NV': 'Nevada', 'WI': 'Wisconsin', 'MI': 'Michigan'
    }.get(state, state)
    
    race_titles = {
        'president': f'{cycle} United States presidential election in {state_full}',
        'senate': f'{cycle} United States Senate election in {state_full}',
        'governor': f'{cycle} {state_full} gubernatorial election',
        'house': f'{cycle} United States House of Representatives elections in {state_full}'
    }
    
    if race_type not in race_titles:
        logger.warning(f"Unsupported race type for Wikipedia: {race_type}")
        return pd.DataFrame()
    
    page_title = race_titles[race_type]
    
    try:
        # Fetch page content via MediaWiki API
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'parse',
            'page': page_title,
            'prop': 'text',
            'format': 'json',
            'section': 0  # Usually infobox is in first section
        }
        
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code != 200:
            logger.debug(f"Wikipedia API returned status {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
        if 'parse' not in data or 'text' not in data['parse']:
            return pd.DataFrame()
        
        html_content = data['parse']['text']['*']
        
        # Parse HTML for infobox data
        rows = []
        
        # Look for candidate rows in infobox
        candidate_pattern = r'<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([^<]+)</td>'
        matches = re.findall(candidate_pattern, html_content, re.IGNORECASE)
        
        for match in matches:
            candidate, party, votes = match[:3]
            # Clean up text
            candidate = re.sub(r'<[^>]+>', '', candidate).strip()
            party = re.sub(r'<[^>]+>', '', party).strip()
            votes = re.sub(r'<[^>]+>', '', votes).strip()
            
            # Remove citations like [1]
            candidate = re.sub(r'\[\d+\]', '', candidate)
            party = re.sub(r'\[\d+\]', '', party)
            votes = re.sub(r'\[\d+\]', '', votes)
            
            # Convert votes to numeric
            votes_num = 0
            try:
                votes_num = int(votes.replace(',', ''))
            except ValueError:
                continue
            
            rows.append({
                'candidate': candidate,
                'party': party,
                'votes': votes_num,
                'pct': 0.0,
                'winner': False
            })
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Calculate percentages
        total_votes = df['votes'].sum()
        if total_votes > 0:
            df['pct'] = (df['votes'] / total_votes) * 100
        
        # Determine winner (highest votes)
        if not df.empty:
            max_votes = df['votes'].max()
            df['winner'] = df['votes'] == max_votes
        
        return _standardize_dataframe(df, state, cycle, race_type, "Wikipedia")
        
    except Exception as e:
        logger.warning(f"Failed to fetch Wikipedia data: {e}")
        return pd.DataFrame()

def fetch_vote_counts(state: str, cycle: int, race_type: str) -> pd.DataFrame:
    """Orchestrator: tries sources in order until one returns data.
    
    Order: NYT CDN → OpenElections → Wikipedia → empty DataFrame
    Logs which source succeeded.
    """
    from app.election.historical.live_vote_counts import fetch_nyt_results_archive
    
    sources = [
        ("NYT", lambda: fetch_nyt_results_archive(state, cycle, race_type)),
        ("OpenElections", lambda: fetch_openelections_results(state, cycle, race_type)),
        ("Wikipedia", lambda: fetch_wikipedia_certified_results(state, cycle, race_type))
    ]
    
    for source_name, fetch_func in sources:
        df = fetch_func()
        if not df.empty:
            logger.info(f"Successfully fetched {state} {cycle} {race_type} from {source_name}")
            return df
    
    logger.warning(f"No data found for {state} {cycle} {race_type}")
    return pd.DataFrame(columns=['state', 'race_type', 'cycle', 'candidate', 'party', 
                                 'votes', 'pct', 'winner', 'source'])
