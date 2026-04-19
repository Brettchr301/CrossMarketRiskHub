"""Step 2: Run kalshi_multi_fidelity_backfill for all discovered 2024 events."""
import json
import sys
from datetime import date

from app.election.historical.multi_fidelity_backfill import kalshi_multi_fidelity_backfill
from app.election.historical.kalshi_history import fetch_markets_for_event
from app.election.mappings.race_linker import link_contract_to_race
from app.election.db.session import get_session_factory, init_election_db
from app.election.db.historical_models import HistoricalQuote

with open("scripts/kalshi_2024_events.json") as f:
    blob = json.load(f)
tickers = blob["events"]
print(f"Running backfill for {len(tickers)} events")

init_election_db()
db = get_session_factory()()

total_inserted = 0
success_events = []
empty_events = []
errors = {}

for idx, ticker in enumerate(tickers, 1):
    # Preflight: count markets returned by fetch_markets_for_event
    try:
        mks = fetch_markets_for_event(ticker)
    except Exception as exc:
        errors[ticker] = f"fetch_markets_for_event: {exc}"
        continue
    if not mks:
        empty_events.append(ticker)
        if idx <= 10 or idx % 20 == 0:
            print(f"  [{idx}/{len(tickers)}] {ticker}: 0 markets (preflight)")
        continue

    # Has markets — run the backfill
    try:
        results = kalshi_multi_fidelity_backfill(ticker, date(2024, 11, 5))
    except Exception as exc:
        errors[ticker] = f"backfill: {exc}"
        print(f"  [{idx}] {ticker}: backfill FAILED: {exc}")
        continue

    inserted_for_ticker = 0
    for title, series in results.items():
        try:
            link = link_contract_to_race(title)
        except Exception:
            class _L: race_id = None
            link = _L()
        for ts, price in series.items():
            db.add(HistoricalQuote(
                race_id=getattr(link, "race_id", None),
                platform="kalshi",
                platform_market_id=f"kalshi_hf_{abs(hash(title)) % 10**10}",
                question=title,
                cycle=2024,
                price=float(price),
                as_of=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            ))
            inserted_for_ticker += 1
            total_inserted += 1
    if inserted_for_ticker:
        success_events.append((ticker, len(results), inserted_for_ticker))
        print(f"  [{idx}] {ticker}: {len(results)} markets, {inserted_for_ticker} quotes")
    try:
        db.commit()
    except Exception as exc:
        print(f"    commit error: {exc}")
        db.rollback()

print("\n" + "=" * 80)
print("Backfill summary")
print("=" * 80)
print(f"Events processed: {len(tickers)}")
print(f"Events with markets (succeeded): {len(success_events)}")
print(f"Events with no markets (empty preflight): {len(empty_events)}")
print(f"Events errored: {len(errors)}")
print(f"Total historical quotes inserted: {total_inserted}")
if success_events:
    print("\nSuccessful events:")
    for t, nm, nq in success_events:
        print(f"  {t}: {nm} markets, {nq} quotes")
if errors:
    print("\nErrors:")
    for t, e in list(errors.items())[:20]:
        print(f"  {t}: {e}")

db.close()
