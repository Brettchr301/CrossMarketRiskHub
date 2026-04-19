# Task: Alternative Vote Count Sources (NYT/AP/Edison)

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | Public APIs, no auth needed |
| Complexity | 2 | Multiple data sources with different formats, fallback chain |
| Novelty | 1 | Follows existing provider pattern (weather.py, state_sos.py) |
| Blast Radius | 1 | Additive — new provider, doesn't change existing data |
| Existing Code | 1 | Extends live_vote_counts.py which already has NYT 2024 stub |
| **Total** | **5** | |

## Objective

The current `live_vote_counts.py` has two problems:
1. **NYT archive only covers 2024** (hardcoded URLs at lines 22-23) — no 2018/2020/2022
2. **Wayback SOS scraping times out** on all states (0 rows extracted across all cycles)

We need reliable vote count data for election-night backtesting across 2018-2024. The solution is a multi-source fallback chain:

1. **NYT Results API** (primary) — public CDN CSVs exist for 2020 and 2024. Pattern: `https://static01.nyt.com/elections-assets/pages/data/{YYYY}-{MM}-{DD}/results-{type}.csv`
2. **Associated Press (AP) via GitHub** — MIT-licensed repos with historical AP call data. Several open datasets exist on GitHub with county-level results.
3. **Wikipedia tables** — Infobox scraping for final certified results per race (not real-time, but ground-truth)
4. **OpenElections project** — `openelections.net` provides CSV downloads of certified state results for 2018-2024

## Deliverables

- [ ] `app/election/historical/live_vote_counts.py` — extend `fetch_nyt_results_archive()` to cover 2020 and 2022 cycles with correct URL patterns. Add fallback to empty DataFrame gracefully.
- [ ] `app/election/historical/vote_count_providers.py` — NEW file implementing the fallback chain:
  - `fetch_openelections_results(state: str, cycle: int, race_type: str) -> pd.DataFrame` — downloads CSV from OpenElections GitHub raw URLs
  - `fetch_wikipedia_certified_results(state: str, cycle: int) -> pd.DataFrame` — scrapes Wikipedia infobox for certified winner + vote counts
  - `fetch_vote_counts(state: str, cycle: int, race_type: str) -> pd.DataFrame` — orchestrator that tries NYT → OpenElections → Wikipedia in order, returns first non-empty result
- [ ] `app/election/db/models.py` — add `VoteCountSnapshot` model if not exists (or use existing `AltDataSignal` with signal_type="vote_count")
- [ ] `app/election/historical/backfill_alt_data.py` — integrate new vote count provider into the alt-data backfill pipeline
- [ ] `tests/test_vote_count_providers.py` — tests with mocked HTTP responses

## Constraints

- DO NOT delete or modify the Wayback SOS scraping code — keep it as last-resort fallback
- DO NOT add any API keys or paid services — all sources must be free/public
- Use `requests` for HTTP (already in requirements) — do not add new dependencies
- Rate-limit all HTTP calls: max 2 requests/second
- All DataFrames must have consistent columns: `state`, `race_type`, `cycle`, `candidate`, `party`, `votes`, `pct`, `winner`, `source`

## Exact Interface

```python
# vote_count_providers.py

def fetch_openelections_results(
    state: str,         # "PA", "AZ", etc.
    cycle: int,         # 2018, 2020, 2022, 2024
    race_type: str,     # "president", "senate", "house", "governor"
) -> pd.DataFrame:
    """Fetch certified results from OpenElections GitHub CSVs.

    URL pattern: https://raw.githubusercontent.com/openelections/
    openelections-data-{state_lower}/master/{cycle}/
    {cycle}{MM}{DD}__{state_lower}__general__{race_type}.csv

    Returns DataFrame with standardized columns or empty DataFrame on failure.
    """

def fetch_wikipedia_certified_results(
    state: str,
    cycle: int,
    race_type: str = "senate",
) -> pd.DataFrame:
    """Scrape Wikipedia infobox for certified election results.

    Uses MediaWiki API (api.php?action=parse) to get HTML, then parses
    the infobox table for candidate names, parties, and vote totals.
    """

def fetch_vote_counts(
    state: str,
    cycle: int,
    race_type: str,
) -> pd.DataFrame:
    """Orchestrator: tries sources in order until one returns data.

    Order: NYT CDN → OpenElections → Wikipedia → empty DataFrame
    Logs which source succeeded.
    """
```

## Tests to Write

1. **test_nyt_2020_url_pattern**: Mock requests.get for NYT 2020 presidential URL. Verify correct URL constructed and DataFrame parsed with expected columns.

2. **test_nyt_fallback_on_404**: Mock NYT returning 404. Verify returns empty DataFrame, no exception raised.

3. **test_openelections_pa_2022**: Mock OpenElections CSV response for PA 2022 Senate. Verify Fetterman/Oz rows parsed with correct vote counts and party labels.

4. **test_wikipedia_fallback**: Mock NYT 404, OpenElections 404, Wikipedia returning infobox HTML. Verify `fetch_vote_counts()` falls through to Wikipedia and returns parsed results.

5. **test_standardized_columns**: Verify all three providers return DataFrames with exactly the columns: `state, race_type, cycle, candidate, party, votes, pct, winner, source`.

6. **test_rate_limiting**: Verify that calling `fetch_vote_counts()` for 5 states in sequence doesn't exceed 2 req/s (mock with timing checks).

7. **test_backfill_integration**: Verify `run_alt_data_backfill()` calls the new vote count provider and persists results to DB.

## Files to Touch
- `app/election/historical/live_vote_counts.py` — modify
- `app/election/historical/vote_count_providers.py` — create
- `app/election/historical/backfill_alt_data.py` — modify
- `tests/test_vote_count_providers.py` — create

## Success Criteria
1. All 7 tests pass
2. `fetch_vote_counts("PA", 2022, "senate")` returns Fetterman as winner (via any source)
3. `fetch_vote_counts("AZ", 2022, "governor")` returns Hobbs as winner
4. At least 2 of 3 sources (NYT, OpenElections, Wikipedia) return data for 2024 cycle
5. No regressions: `python -c "from app.election.historical.live_vote_counts import fetch_nyt_results_archive"` still works



## Pre-Generated Tests (your code MUST pass these)
These tests were generated independently. Include them EXACTLY as shown.

```json
{"content": "--- FILE: tests/test_vote_count_providers.py ---\nimport pytest\nimport pandas as pd\nfrom unittest.mock import patch, MagicMock, call\n\n# Target imports based on the specification\nfrom app.election.historical.live_vote_counts import fetch_nyt_results_archive\nfrom app.election.historical.vote_count_providers import (\n    fetch_openelections_results,\n    fetch_wikipedia_certified_results,\n    fetch_vote_counts\n)\nfrom app.election.historical.backfill_alt_data import run_alt_data_backfill\n\nEXPECTED_COLUMNS = [\"state\", \"race_type\", \"cycle\", \"candidate\", \"party\", \"votes\", \"pct\", \"winner\", \"source\"]\n\n@patch(\"app.election.historical.live_vote_counts.requests.get\")\ndef test_nyt_2020_url_pattern(mock_get):\n    \"\"\"1. Verify correct NYT 2020 URL constructed and DataFrame parsed with expected columns.\"\"\"\n    mock_resp = MagicMock()\n    mock_resp.status_code = 200\n    # Provide dummy CSV content that the implementation should parse and standardize\n    mock_resp.text = \"state,race_type,cycle,candidate,party,votes,pct,winner,source\\nPA,president,2020,Biden,DEM,100,50.0,True,NYT\"\n    mock_get.return_value = mock_resp\n\n    df = fetch_nyt_results_archive(\"PA\", 2020, \"president\")\n\n    # Verify URL pattern for 2020 (Election Day: Nov 3, 2020)\n    mock_get.assert_called_once()\n    called_url = mock_get.call_args[0][0]\n    assert \"2020-11-03\" in called_url\n    assert \"results-president.csv\" in called_url\n\n    # Verify DataFrame columns\n    assert not df.empty\n    assert list(df.columns) == EXPECTED_COLUMNS\n    assert df.iloc[0][\"candidate\"] == \"Biden\"\n\n\n@patch(\"app.election.historical.live_vote_counts.requests.get\")\ndef test_nyt_fallback_on_404(mock_get):\n    \"\"\"2. Verify NYT returning 404 returns empty DataFrame, no exception raised.\"\"\"\n    mock_resp = MagicMock()\n    mock_resp.status_code = 404\n    mock_get.return_value = mock_resp\n\n    df = fetch_nyt_results_archive(\"PA\", 2020, \"president\")\n    \n    assert isinstance(df, pd.DataFrame)\n    assert df.empty\n\n\n@patch(\"app.election.historical.vote_count_providers.requests.get\")\ndef test_openelections_pa_2022(mock_get):\n    \"\"\"3. Verify OpenElections CSV response for PA 2022 Senate parses Fetterman/Oz rows.\"\"\"\n    mock_resp = MagicMock()\n    mock_resp.status_code = 200\n    # Dummy raw CSV from OpenElections\n    mock_resp.text = \"candidate,party,votes,winner\\nJohn Fetterman,DEM,2751012,True\\nMehmet Oz,REP,2487260,False\"\n    mock_get.return_value = mock_resp\n\n    df = fetch_openelections_results(\"PA\", 2022, \"senate\")\n\n    # Verify URL pattern (Election Day: Nov 8, 2022)\n    mock_get.assert_called_once()\n    called_url = mock_get.call_args[0][0]\n    assert \"openelections-data-pa\" in called_url\n    assert \"20221108__pa__general__senate.csv\" in called_url\n\n    # Verify parsed data\n    assert not df.empty\n    candidates = df[\"candidate\"].tolist()\n    assert \"John Fetterman\" in candidates\n    assert \"Mehmet Oz\" in candidates\n    \n    fetterman_row = df[df[\"candidate\"] == \"John Fetterman\"].iloc[0]\n    assert fetterman_row[\"party\"] == \"DEM\"\n    assert int(fetterman_row[\"votes\"]) == 2751012\n\n\n@patch(\"app.election.historical.vote_count_providers.fetch_wikipedia_certified_results\")\n@patch(\"app.election.historical.vote_count_providers.fetch_openelections_results\")\n@patch(\"app.election.historical.vote_count_providers.fetch_nyt_results_archive\")\ndef test_wikipedia_fallback(mock_nyt, mock_oe, mock_wiki):\n    \"\"\"4. Verify fetch_vote_counts() falls through to Wikipedia when NYT and OpenElections fail.\"\"\"\n    # Mock NYT and OpenElections returning empty DataFrames (simulating 404s)\n    mock_nyt.return_value = pd.DataFrame()\n    mock_oe.return_value = pd.DataFrame()\n    \n    # Mock Wikipedia returning valid data\n    wiki_df = pd.DataFrame([{\n        \"state\": \"PA\", \"race_type\": \"senate\", \"cycle\": 2022,\n        \"candidate\": \"John Fetterman\", \"party\": \"DEM\", \"votes\": 2751012,\n        \"pct\": 51.2, \"winner\": True, \"source\": \"Wikipedia\"\n    }])\n    mock_wiki.return_value = wiki_df\n\n    df = fetch_vote_counts(\"PA\", 2022, \"senate\")\n\n    # Verify fallback chain execution\n    mock_nyt.assert_called_once_with(\"PA\", 2022, \"senate\")\n    mock_oe.assert_called_once_with(\"PA\", 2022, \"senate\")\n    mock_wiki.assert_called_once_with(\"PA\", 2022, \"senate\")\n\n    # Verify Wikipedia data is returned\n    assert not df.empty\n    assert df.iloc[0][\"source\"] == \"Wikipedia\"\n    assert df.iloc[0][\"candidate\"] == \"John Fetterman\"\n\n\n@patch(\"app.election.historical.vote_count_providers.requests.get\")\n@patch(\"app.election.historical.live_vote_counts.requests.get\")\ndef test_standardized_columns(mock_nyt_get, mock_vp_get):\n    \"\"\"5. Verify all three providers return DataFrames with exactly the standardized columns.\"\"\"\n    # Setup mocks to return minimal valid data\n    mock_nyt_resp = MagicMock()\n    mock_nyt_resp.status_code = 200\n    mock_nyt_resp.text = \"state,race_type,cycle,candidate,party,votes,pct,winner,source\\nPA,president,2020,Biden,DEM,100,50.0,True,NYT\"\n    mock_nyt_get.return_value = mock_nyt_resp\n\n    mock_vp_resp = MagicMock()\n    mock_vp_resp.status_code = 200\n    mock_vp_resp.text = \"candidate,party,votes,winner\\nJohn Fetterman,DEM,2751012,True\"\n    mock_vp_get.return_value = mock_vp_resp\n\n    # Test NYT\n    df_nyt = fetch_nyt_results_archive(\"PA\", 2020, \"president\")\n    assert list(df_nyt.columns) == EXPECTED_COLUMNS\n\n    # Test OpenElections\n    df_oe = fetch_openelections_results(\"PA\", 2022, \"senate\")\n    assert list(df_oe.columns) == EXPECTED_COLUMNS\n\n    # Test Wikipedia (mocking the API response)\n    mock_wiki_resp = MagicMock()\n    mock_wiki_resp.status_code = 200\n    mock_wiki_resp.json.return_value = {\"parse\": {\"text\": {\"*\": \"<table><tr><td>Fetterman</td><td>DEM</td><td>2,751,012</td></tr></table>\"}}}\n    mock_vp_get.return_value = mock_wiki_resp\n    \n    df_wiki = fetch_wikipedia_certified_results(\"PA\", 2022, \"senate\")\n    assert list(df_wiki.columns) == EXPECTED_COLUMNS\n\n\n@patch(\"time.sleep\")\n@patch(\"app.election.historical.vote_count_providers.fetch_nyt_results_archive\")\n@patch(\"app.election.historical.vote_count_providers.fetch_openelections_results\")\n@patch(\"app.election.historical.vote_count_providers.fetch_wikipedia_certified_results\")\ndef test_rate_limiting(mock_wiki, mock_oe, mock_nyt, mock_sleep):\n    \"\"\"6. Verify calling fetch_vote_counts() for 5 states in sequence doesn't exceed 2 req/s.\"\"\"\n    # Make all sources fail to force maximum HTTP calls/orchestrator loops\n    mock_nyt.return_value = pd.DataFrame()\n    mock_oe.return_value = pd.DataFrame()\n    mock_wiki.return_value = pd.DataFrame()\n\n    states = [\"PA\", \"AZ\", \"GA\", \"NV\", \"WI\"]\n    for state in states:\n        fetch_vote_counts(state, 2022, \"senate\")\n\n    # To maintain max 2 requests/second, the system must sleep at least 0.5s between requests.\n    # We verify that time.sleep was called to enforce this.\n    assert mock_sleep.call_count > 0\n    \n    # Verify the sleep duration is sufficient (>= 0.5 seconds per rate limit event)\n    for sleep_call in mock_sleep.call_args_list:\n        sleep_duration = sleep_call[0][0]\n        assert sleep_duration >= 0.5\n\n\n@patch(\"app.election.historical.backfill_alt_data.fetch_vote_counts\")\ndef test_backfill_integration(mock_fetch):\n    \"\"\"7. Verify run_alt_data_backfill() calls the new provider and persists results to DB.\"\"\"\n    # Mock the orchestrator to return valid data\n    mock_fetch.return_value = pd.DataFrame([{\n        \"state\": \"PA\", \"race_type\": \"senate\", \"cycle\": 2022,\n        \"candidate\": \"Fetterman\", \"party\": \"DEM\", \"votes\": 2751012,\n        \"pct\": 51.2, \"winner\": True, \"source\": \"NYT\"\n    }])\n\n    # Mock the database session/persistence layer generically\n    with patch(\"app.election.historical.backfill_alt_data.SessionLocal\", create=True) as mock_session_local:\n        mock_session = mock_session_local.return_value.__enter__.return_value\n        \n        # Execute the backfill\n        run_alt_data_backfill()\n\n        # Verify the new provider was called\n        mock_fetch.assert_called()\n\n        # Verify database persistence was attempted (add and commit)\n        assert mock_session.add.called or mock_session.bulk_save_objects.called or mock_session.execute.called\n        assert mock_session.commit.called\n--- END FILE ---", "model": "gemini-3.1-pro-preview", "usage": {"prompt_tokens": 2002, "completion_tokens": 2657, "total_tokens": 4659}, "cost_usd": 0.035888}
```
