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
