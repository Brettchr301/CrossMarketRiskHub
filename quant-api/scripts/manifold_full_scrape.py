"""Comprehensive Manifold Markets backfill across all election cycles.

Covers every cycle Manifold has data for (platform launched ~2021):
  - 2022 midterms
  - 2023 off-year governor races (KY, LA, MS)
  - 2024 presidential + downballot
  - 2025 off-year governor races (NJ, VA) + NYC mayor
  - 2026 midterm markets (still trading)
  - 2028 presidential primary markets (still trading)

Uses app.election.historical.manifold_history for API calls, then links each
market to a canonical race via race_linker.link_contract_to_race and inserts
tick-by-tick bet-level quotes into HistoricalQuote.
"""
from __future__ import annotations

import logging
import re
import sys
from collections import Counter, defaultdict

from app.election.historical.manifold_history import (
    bet_history_to_price_series,
    fetch_bet_history,
    search_election_markets,
)
from app.election.mappings.race_linker import link_contract_to_race
from app.election.db.session import get_session_factory, init_election_db
from app.election.db.historical_models import HistoricalQuote


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("manifold_full_scrape")


SEARCH_TERMS = [
    # Presidential
    "2020 presidential election",
    "2024 presidential election",
    "2028 presidential nomination",
    "2028 democratic primary",
    "2028 republican primary",
    # Senate
    "2022 senate",
    "2024 senate",
    "2026 senate",
    # House
    "2022 house",
    "2024 house",
    "2026 house",
    # Governor (annual)
    "2022 governor",
    "2023 kentucky governor",
    "2023 louisiana governor",
    "2023 mississippi governor",
    "2024 governor",
    "2025 virginia governor",
    "2025 new jersey governor",
    "2026 governor",
    # Off-year and specials
    "2021 virginia election",
    "2021 new jersey",
    "2021 nyc mayor",
    "2025 nyc mayor",
    "special election",
    # Ballot measures
    "ohio issue 1",
    "abortion referendum",
    # Trump-specific (huge volume)
    "trump election",
    "trump 2024",
    "trump 2028",
]

MARKETS_PER_TERM = 15
MAX_BETS_PER_MARKET = 10_000
PLATFORM = "manifold"


_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_RELEVANT_YEARS = {2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030}


def infer_cycle(question: str) -> int | None:
    """Infer election cycle from the question text.

    Prefers the most recent plausible election year mentioned. Falls back to
    None (the DB column is NOT NULL, so callers default to 0).
    """
    years = [int(y) for y in _YEAR_RE.findall(question or "")]
    candidates = [y for y in years if y in _RELEVANT_YEARS]
    if not candidates:
        return None
    # Prefer even years (federal election cycles) when available
    even = [y for y in candidates if y % 2 == 0]
    return max(even) if even else max(candidates)


def main() -> None:
    init_election_db()
    SessionFactory = get_session_factory()
    db = SessionFactory()

    total_markets = 0
    total_bets_inserted = 0
    cycle_counter: Counter[int] = Counter()
    cycle_market_counter: Counter[int] = Counter()
    platform_counter: Counter[str] = Counter()
    per_term_counter: dict[str, int] = defaultdict(int)
    linked_count = 0
    unlinked_count = 0
    empty_markets = 0
    errored_markets = 0

    seen_contract_ids: set[str] = set()

    logger.info(
        "Starting Manifold full scrape | terms=%d | markets/term=%d | max_bets/market=%d",
        len(SEARCH_TERMS), MARKETS_PER_TERM, MAX_BETS_PER_MARKET,
    )

    for term_idx, term in enumerate(SEARCH_TERMS, 1):
        logger.info("[term %d/%d] Searching: %r", term_idx, len(SEARCH_TERMS), term)
        markets = search_election_markets(term, limit=MARKETS_PER_TERM, sort="most-popular")
        if not markets:
            logger.warning("  No markets returned for term: %r", term)
            continue

        for market in markets:
            contract_id = market.get("id")
            question = (market.get("question") or "").strip()
            if not contract_id or contract_id in seen_contract_ids:
                continue
            seen_contract_ids.add(contract_id)

            total_markets += 1
            per_term_counter[term] += 1

            # Log progress every 5 markets
            if total_markets % 5 == 0:
                logger.info(
                    "  Progress: %d unique markets processed | %d bets inserted | linked=%d unlinked=%d",
                    total_markets, total_bets_inserted, linked_count, unlinked_count,
                )

            # Fetch bet-level history
            try:
                df = fetch_bet_history(contract_id, max_bets=MAX_BETS_PER_MARKET)
            except Exception as exc:
                logger.warning("  fetch_bet_history failed for %s: %s", contract_id, exc)
                errored_markets += 1
                continue

            series = bet_history_to_price_series(df)
            if series.empty:
                empty_markets += 1
                continue

            # Link to a canonical race (best-effort)
            try:
                link = link_contract_to_race(question)
                race_id = getattr(link, "race_id", None)
            except Exception:
                race_id = None

            if race_id is not None:
                linked_count += 1
            else:
                unlinked_count += 1

            cycle = infer_cycle(question)
            cycle_for_row = cycle if cycle is not None else 0

            cycle_market_counter[cycle_for_row] += 1
            platform_counter[PLATFORM] += 1

            # Insert every tick as a HistoricalQuote row
            inserted_this_market = 0
            for ts, price in series.items():
                try:
                    py_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                except Exception:
                    py_ts = ts
                db.add(HistoricalQuote(
                    race_id=race_id,
                    platform=PLATFORM,
                    platform_market_id=str(contract_id),
                    question=question[:4000] if question else "",
                    cycle=cycle_for_row,
                    price=float(price),
                    as_of=py_ts,
                ))
                inserted_this_market += 1

            total_bets_inserted += inserted_this_market
            cycle_counter[cycle_for_row] += inserted_this_market

            # Commit per market to keep transactions small and reduce lock time
            try:
                db.commit()
            except Exception as exc:
                logger.warning("  commit failed for %s: %s", contract_id, exc)
                db.rollback()
                errored_markets += 1
                continue

            logger.info(
                "  [%d] %s | cycle=%s | bets=%d | race_id=%s",
                total_markets,
                (question[:70] + "...") if len(question) > 70 else question,
                cycle_for_row if cycle is not None else "unknown",
                inserted_this_market,
                race_id,
            )

    db.close()

    # =========================================================================
    # Final report
    # =========================================================================
    bar = "=" * 80
    print("\n" + bar)
    print("Manifold comprehensive scrape summary")
    print(bar)
    print(f"Search terms executed       : {len(SEARCH_TERMS)}")
    print(f"Unique markets processed    : {total_markets}")
    print(f"Markets with bets (inserted): {total_markets - empty_markets - errored_markets}")
    print(f"Markets empty (no bets)     : {empty_markets}")
    print(f"Markets errored             : {errored_markets}")
    print(f"Linked to canonical race    : {linked_count}")
    print(f"Unlinked                    : {unlinked_count}")
    print(f"Total historical quotes     : {total_bets_inserted}")
    print()

    print("Quotes by cycle:")
    for cycle in sorted(cycle_counter.keys()):
        label = str(cycle) if cycle != 0 else "unknown"
        n_markets = cycle_market_counter.get(cycle, 0)
        print(f"  {label:>8}: {cycle_counter[cycle]:>10,} quotes  ({n_markets} markets)")
    print()

    print("Quotes by platform:")
    for plat, n in platform_counter.items():
        print(f"  {plat:>10}: {n:>10,} markets-with-quotes")
    print()

    print("Markets per search term:")
    for t in SEARCH_TERMS:
        print(f"  {per_term_counter.get(t, 0):>3}  {t}")
    print(bar)


if __name__ == "__main__":
    main()
