"""PredictIt Wayback Machine scraper.

For each election cycle (and off-years), fetch evenly distributed snapshots
of the PredictIt marketdata endpoint from the Internet Archive and insert
reconstructed contract price series into HistoricalQuote.

Task:
- Cycles: 2018, 2019, 2020, 2021, 2022, 2023, 2024
- sample_n = 50 per year
- Fallback to 20 if rate-limited
- Link via link_contract_to_race
- Infer cycle from any 4-digit year found in the contract key

Usage:  python -m scripts.predictit_wayback_scrape
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from collections import defaultdict

import requests

# Import the provider MODULE so we can patch its constants/fn in-place.
# On this host plain http://web.archive.org (port 80) is blocked; https:// works.
# Inject a browser UA and use a persistent Session with retries to survive
# the occasional transient timeout from the Wayback CDN.
from app.election.historical import predictit_history as _pih

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Force HTTPS for the Wayback CDX endpoint.
_pih.WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

# Persistent session (keeps TCP warm, reuses TLS) with a browser UA.
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": _BROWSER_UA})

# (connect_timeout, read_timeout)
_CDX_TIMEOUT = (15, 60)
_SNAP_TIMEOUT = (15, 90)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 4.0  # seconds


def _get_with_retries(url, params=None, timeout=_SNAP_TIMEOUT):
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _patched_list_wayback_snapshots(from_date, to_date, limit=500):
    try:
        params = {
            "url": _pih.PREDICTIT_URL,
            "output": "json",
            "from": from_date,
            "to": to_date,
            "limit": limit,
            "filter": "statuscode:200",
        }
        r = _get_with_retries(_pih.WAYBACK_CDX, params=params, timeout=_CDX_TIMEOUT)
        rows = r.json()
        if len(rows) < 2:
            return []
        header = rows[0]
        snapshots = []
        for row in rows[1:]:
            snap = dict(zip(header, row))
            # id_ = raw unmodified payload, HTTPS path that works on this host
            snap["archive_url"] = (
                f"https://web.archive.org/web/{snap['timestamp']}id_/"
                f"{_pih.PREDICTIT_URL}"
            )
            snapshots.append(snap)
        return snapshots
    except Exception as exc:
        print(f"    CDX fetch failed: {exc}", flush=True)
        return []


def _xml_to_markets_dict(xml_text: str) -> dict:
    """Convert PredictIt XML MarketList payload to the JSON-equivalent dict
    that backfill_from_wayback expects: {'markets': [{'name', 'contracts': [{'name','lastTradePrice'}]}]}
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {"markets": []}

    def _localname(tag: str) -> str:
        return tag.split("}", 1)[-1]

    markets_out = []
    # Walk Markets/MarketData nodes regardless of default namespace
    for md in root.iter():
        if _localname(md.tag) != "MarketData":
            continue
        m_name = None
        contracts_out = []
        for child in md:
            ln = _localname(child.tag)
            if ln == "Name" and m_name is None:
                m_name = (child.text or "").strip()
            elif ln == "Contracts":
                for mc in child:
                    if _localname(mc.tag) != "MarketContract":
                        continue
                    c_name = None
                    last = 0.0
                    for f in mc:
                        fln = _localname(f.tag)
                        if fln == "Name" and c_name is None:
                            c_name = (f.text or "").strip()
                        elif fln == "LastTradePrice":
                            try:
                                last = float((f.text or "0").strip())
                            except (ValueError, TypeError):
                                last = 0.0
                    if c_name:
                        contracts_out.append({"name": c_name, "lastTradePrice": last})
        if m_name and contracts_out:
            markets_out.append({"name": m_name, "contracts": contracts_out})
    return {"markets": markets_out}


def _patched_fetch_snapshot(archive_url):
    try:
        r = _get_with_retries(archive_url, timeout=_SNAP_TIMEOUT)
        ct = (r.headers.get("content-type") or "").lower()
        text = r.text
        # PredictIt historically served XML at this endpoint; JSON came later.
        if "xml" in ct or text.lstrip().startswith("<"):
            return _xml_to_markets_dict(text)
        # Otherwise expect JSON
        return r.json()
    except Exception as exc:
        print(f"    Snapshot fetch failed: {exc}", flush=True)
        return None


_pih.list_wayback_snapshots = _patched_list_wayback_snapshots
_pih.fetch_snapshot = _patched_fetch_snapshot

from app.election.historical.predictit_history import backfill_from_wayback  # noqa: E402
from app.election.mappings.race_linker import link_contract_to_race  # noqa: E402
from app.election.db.session import get_session_factory, init_election_db  # noqa: E402
from app.election.db.historical_models import HistoricalQuote  # noqa: E402


YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
DEFAULT_SAMPLE_N = 50
FALLBACK_SAMPLE_N = 20
INTER_YEAR_SLEEP = 5.0  # polite pause between cycles


# Match any 4-digit year 2016-2030 in the contract key text
_YEAR_RE = re.compile(r"\b(20(?:1[6-9]|2\d|30))\b")


def infer_cycle_from_key(key: str, default_year: int) -> int:
    """Find a 4-digit year (2016-2030) in the contract key; fall back to snapshot year."""
    m = _YEAR_RE.search(key)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return default_year


def run_year(year: int, sample_n: int) -> dict:
    """Run backfill for a single year. Returns stats dict."""
    from_date = f"{year}0101"
    to_date = f"{year}1231"
    print(f"  -> backfill_from_wayback(from={from_date}, to={to_date}, sample_n={sample_n})", flush=True)
    t0 = time.time()
    try:
        series_map = backfill_from_wayback(from_date, to_date, sample_n=sample_n)
    except Exception as exc:
        print(f"  -> backfill raised: {exc}", flush=True)
        traceback.print_exc()
        return {
            "year": year,
            "sample_n": sample_n,
            "contracts": 0,
            "quotes": 0,
            "elapsed_sec": time.time() - t0,
            "error": str(exc),
        }
    elapsed = time.time() - t0
    return {
        "year": year,
        "sample_n": sample_n,
        "series_map": series_map,
        "contracts": len(series_map),
        "elapsed_sec": elapsed,
        "error": None,
    }


def insert_series(db, series_map: dict, default_year: int) -> tuple[int, int]:
    """Insert reconstructed series into HistoricalQuote.

    Returns (contracts_inserted, quotes_inserted).
    """
    contracts_inserted = 0
    quotes_inserted = 0

    for key, series in series_map.items():
        if series is None or len(series) == 0:
            continue
        cycle = infer_cycle_from_key(key, default_year)

        try:
            link = link_contract_to_race(key)
        except Exception:
            class _L:
                race_id = None
            link = _L()

        market_id = f"predictit_wb_{abs(hash(key)) % 10**10}"
        row_count = 0
        for ts, price in series.items():
            try:
                as_of = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                db.add(HistoricalQuote(
                    race_id=getattr(link, "race_id", None),
                    platform="predictit",
                    platform_market_id=market_id,
                    question=key,
                    cycle=int(cycle),
                    price=float(price),
                    as_of=as_of,
                ))
                row_count += 1
            except Exception:
                continue
        if row_count:
            contracts_inserted += 1
            quotes_inserted += row_count

    try:
        db.commit()
    except Exception as exc:
        print(f"    commit error: {exc}", flush=True)
        db.rollback()
    return contracts_inserted, quotes_inserted


def main() -> int:
    print("=" * 80, flush=True)
    print("PredictIt Wayback Scraper", flush=True)
    print(f"Cycles: {YEARS}", flush=True)
    print("=" * 80, flush=True)

    init_election_db()
    Session = get_session_factory()
    db = Session()

    overall = {
        "snapshots_target": 0,
        "contracts": 0,
        "quotes": 0,
        "by_cycle": {},
    }

    try:
        for i, year in enumerate(YEARS, 1):
            print(f"\n[{i}/{len(YEARS)}] Cycle {year}", flush=True)

            # First attempt: default sample_n
            result = run_year(year, DEFAULT_SAMPLE_N)
            sample_used = DEFAULT_SAMPLE_N

            # Fallback: reduce to FALLBACK_SAMPLE_N if we got nothing
            if (result.get("error") or result.get("contracts", 0) == 0):
                print(f"  -> Empty/error with sample_n={DEFAULT_SAMPLE_N}; retrying with sample_n={FALLBACK_SAMPLE_N}", flush=True)
                time.sleep(INTER_YEAR_SLEEP)
                result = run_year(year, FALLBACK_SAMPLE_N)
                sample_used = FALLBACK_SAMPLE_N

            series_map = result.get("series_map") or {}
            if series_map:
                contracts_ins, quotes_ins = insert_series(db, series_map, year)
            else:
                contracts_ins, quotes_ins = 0, 0

            overall["snapshots_target"] += sample_used
            overall["contracts"] += contracts_ins
            overall["quotes"] += quotes_ins
            overall["by_cycle"][year] = {
                "sample_n_used": sample_used,
                "contracts_reconstructed": result.get("contracts", 0),
                "contracts_inserted": contracts_ins,
                "quotes_inserted": quotes_ins,
                "elapsed_sec": round(result.get("elapsed_sec", 0.0), 1),
                "error": result.get("error"),
            }

            print(
                f"  [{year}] reconstructed={result.get('contracts', 0)} "
                f"inserted_contracts={contracts_ins} quotes={quotes_ins} "
                f"elapsed={result.get('elapsed_sec', 0):.1f}s",
                flush=True,
            )

            # Polite pause between cycles
            if i < len(YEARS):
                time.sleep(INTER_YEAR_SLEEP)

    finally:
        db.close()

    print("\n" + "=" * 80, flush=True)
    print("PredictIt Wayback Scrape Summary", flush=True)
    print("=" * 80, flush=True)
    print(f"Snapshots targeted (sum of sample_n used): {overall['snapshots_target']}", flush=True)
    print(f"Total contracts inserted: {overall['contracts']}", flush=True)
    print(f"Total quotes inserted:    {overall['quotes']}", flush=True)
    print("\nBreakdown by cycle:", flush=True)
    print(f"  {'cycle':>6} {'sample_n':>9} {'contracts':>10} {'quotes':>8} {'elapsed(s)':>11}  error", flush=True)
    for year, stats in overall["by_cycle"].items():
        err = stats.get("error") or ""
        print(
            f"  {year:>6} {stats['sample_n_used']:>9} "
            f"{stats['contracts_inserted']:>10} {stats['quotes_inserted']:>8} "
            f"{stats['elapsed_sec']:>11}  {err}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
