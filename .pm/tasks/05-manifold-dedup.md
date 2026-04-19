# Task: Manifold Market Deduplication

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | Internal data pipeline, no user input |
| Complexity | 1 | Single-file change with clear dedup logic |
| Novelty | 0 | Standard dedup pattern (seen set, unique constraint) |
| Blast Radius | 1 | Only affects Manifold ingestion path |
| Existing Code | 1 | Extends manifold_history.py which has basic `seen` set |
| **Total** | **3** | |

## Objective

The Manifold scraper (`manifold_history.py`) searches 9 terms and deduplicates by market `id` within a single run via a `seen` set. However:

1. **Cross-run duplicates**: If `backfill_election_markets()` is called multiple times (which it has been — 295 markets scraped), the `seen` set resets each time. The same market gets re-fetched and potentially re-inserted into the DB as duplicate `HistoricalQuote` rows.

2. **DB-level duplicates**: There's no unique constraint on `(platform, platform_market_id, as_of)` in `HistoricalQuote`, so duplicate quotes accumulate.

3. **Keyword overlap**: Terms like "2024 presidential" and "2024 senate" can return the same market (e.g., "Will Democrats win the presidency and Senate?").

Solution: Add DB-level dedup checks before inserting, and a cleanup function for existing duplicates.

## Deliverables

- [ ] `app/election/historical/manifold_history.py` — modify `backfill_election_markets()` to:
  1. Query existing `HistoricalQuote` rows where platform="manifold" to build a set of already-ingested `platform_market_id` values
  2. Skip markets already in the DB (log "skipping {question}, already ingested")
  3. For new markets, check for existing quotes at same timestamps before inserting
- [ ] `app/election/historical/manifold_history.py` — add `deduplicate_manifold_quotes(db: Session) -> int` function:
  1. Find duplicate groups by `(platform_market_id, as_of)` where platform="manifold"
  2. Keep the row with the lowest `id` (first inserted), delete the rest
  3. Return count of deleted rows
- [ ] `tests/test_manifold_dedup.py` — tests

## Constraints

- DO NOT change the Manifold API calls or search terms
- DO NOT change the `HistoricalQuote` model or add new columns
- DO NOT delete non-duplicate data
- The dedup function must be idempotent (running it twice produces same result)
- Use SQLAlchemy ORM for all DB operations (no raw SQL)

## Exact Interface

```python
# In manifold_history.py, add to backfill_election_markets():

def backfill_election_markets(
    db: Session,  # NEW parameter — needs DB session for dedup check
    search_terms: list[str] | None = None,
    markets_per_term: int = 20,
) -> dict[str, pd.Series]:
    """Backfill Manifold election market bet histories.

    Now accepts a DB session to check for already-ingested markets.
    Skips markets whose platform_market_id already exists in historical_quotes.
    """
    # Query existing Manifold market IDs from DB
    existing_ids = set(
        db.execute(
            select(HistoricalQuote.platform_market_id)
            .where(HistoricalQuote.platform == "manifold")
            .distinct()
        ).scalars().all()
    )
    # ... rest of logic, but skip if mid in existing_ids


def deduplicate_manifold_quotes(db: Session) -> int:
    """Remove duplicate Manifold quotes. Returns count of deleted rows.

    Duplicates defined as: same platform_market_id + same as_of timestamp
    where platform = 'manifold'. Keeps lowest id.
    """
```

Note: The existing function signature doesn't take `db`. To avoid breaking callers, make `db` optional with default `None`. If `None`, skip the DB dedup check (backward compatible).

## Tests to Write

1. **test_skip_existing_market**: Insert a HistoricalQuote with platform="manifold", platform_market_id="abc123". Call `backfill_election_markets(db=session)` with a mocked API response that includes market id="abc123". Verify the market is skipped (no new quotes inserted).

2. **test_new_market_ingested**: Empty DB. Mock API returns market id="new123" with 5 bets. Verify 5 HistoricalQuote rows inserted.

3. **test_deduplicate_removes_extras**: Insert 3 HistoricalQuote rows with same (platform_market_id="x", as_of=same_timestamp). Call `deduplicate_manifold_quotes(db)`. Verify exactly 2 deleted, 1 remains (lowest id).

4. **test_deduplicate_idempotent**: Run `deduplicate_manifold_quotes()` twice. Second run returns 0 deleted.

5. **test_no_false_dedup**: Insert 3 rows with same platform_market_id but DIFFERENT as_of timestamps. Verify `deduplicate_manifold_quotes()` deletes 0.

6. **test_backward_compatible**: Call `backfill_election_markets(db=None)` — verify it still works without DB session (old behavior).

## Files to Touch
- `app/election/historical/manifold_history.py` — modify
- `tests/test_manifold_dedup.py` — create

## Success Criteria
1. All 6 tests pass
2. Running `deduplicate_manifold_quotes()` on the live DB reduces row count (if duplicates exist)
3. Re-running `backfill_election_markets()` with same search terms inserts 0 new rows (all skipped)
4. No regressions: `python -c "from app.election.historical.manifold_history import backfill_election_markets"` still works
