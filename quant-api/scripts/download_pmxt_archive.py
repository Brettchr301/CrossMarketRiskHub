"""Download pmxt archive Polymarket Parquet snapshots.

The pmxt archive (https://archive.pmxt.dev/Polymarket) hosts hourly Parquet
snapshots of Polymarket market data. This script:

  1. Discovers the directory structure via S3-style listing or guessed paths.
  2. Downloads hourly Parquet snapshots spanning pre-election, election week,
     and post-election periods for cycles 2018 / 2020 / 2022 / 2024.
  3. Reads each Parquet with pandas, extracting timestamp/market/price/volume.
  4. Filters to election-related markets (regex on market question).
  5. Links each filtered market to a canonical race via link_contract_to_race.
  6. Inserts the rows into election.historical_quotes (HistoricalQuote).

Data is cached under data/pmxt_archive/.

NOTE on network access (2026-04-19):
    The pmxt.dev domain is currently intercepted by an ISP "Advanced Security"
    filter on this network, returning a safebrowse.io block page on HTTP and
    non-TLS bytes on port 443.  Every list/download attempt will be replaced
    with a graceful skip and a log entry.  To run successfully, either:
      - Connect via a network without the safebrowse filter, or
      - Route traffic through a VPN / alternate DNS with DoT/DoH.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import pandas as pd
import requests

# allow `python scripts/download_pmxt_archive.py` from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.election.db.historical_models import HistoricalQuote  # noqa: E402
from app.election.db.session import get_session_factory, init_election_db  # noqa: E402
from app.election.mappings.race_linker import link_contract_to_race  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pmxt")

ARCHIVE_BASES = [
    "https://archive.pmxt.dev",
    "http://archive.pmxt.dev",
]
POLYMARKET_PATH = "/Polymarket"
DATA_DIR = ROOT / "data" / "pmxt_archive"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Window spec per cycle: pre-election weeks, election week, post-election week
CYCLE_WINDOWS: dict[int, list[tuple[date, date]]] = {
    2018: [
        (date(2018, 10, 15), date(2018, 10, 21)),  # ~3wk out
        (date(2018, 11, 5), date(2018, 11, 11)),   # election week (Nov 6)
        (date(2018, 11, 12), date(2018, 11, 14)),  # post
    ],
    2020: [
        (date(2020, 10, 14), date(2020, 10, 20)),
        (date(2020, 11, 2), date(2020, 11, 8)),    # election day Nov 3
        (date(2020, 11, 9), date(2020, 11, 11)),
    ],
    2022: [
        (date(2022, 10, 17), date(2022, 10, 23)),
        (date(2022, 11, 7), date(2022, 11, 13)),   # election day Nov 8
        (date(2022, 11, 14), date(2022, 11, 16)),
    ],
    2024: [
        (date(2024, 10, 21), date(2024, 10, 27)),
        (date(2024, 11, 4), date(2024, 11, 10)),   # election day Nov 5
        (date(2024, 11, 11), date(2024, 11, 13)),
    ],
}

ELECTION_REGEX = re.compile(
    r"(president|senate|house|governor|election|midterm|congress|white\s*house|"
    r"republican|democrat|gop|dnc|trump|biden|harris|desantis|swing\s*state|"
    r"electoral\s*college|popular\s*vote)",
    re.I,
)

TIMEOUT = 60
HEADERS = {"User-Agent": "CrossMarketRiskHub/pmxt-archive (+https://example.local)"}

S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass
class PmxtObject:
    key: str
    size: int
    last_modified: str
    url: str


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _get(session: requests.Session, url: str, **kw) -> requests.Response | None:
    try:
        r = session.get(url, timeout=TIMEOUT, headers=HEADERS, **kw)
        if r.status_code >= 400:
            log.warning("GET %s -> %s", url, r.status_code)
            return None
        # Detect ISP safebrowse interception (keywords in body)
        snippet = r.text[:600].lower() if "text" in r.headers.get("content-type", "").lower() or "html" in r.headers.get("content-type", "").lower() else ""
        if ("advanced security" in snippet or "safebrowse" in snippet or
            "wifi sicuro" in snippet or "potential threat detected" in snippet or
            "theme-xdns-security" in snippet):
            log.warning("GET %s -> ISP safebrowse interception", url)
            return None
        return r
    except requests.exceptions.SSLError as e:
        log.warning("GET %s -> SSL error: %s", url, str(e)[:100])
        return None
    except Exception as e:
        log.warning("GET %s -> %s", url, str(e)[:200])
        return None


def _parse_s3_listing(xml_text: str) -> tuple[list[PmxtObject], str | None, list[str]]:
    """Parse an S3 ListBucketResult XML response."""
    objs: list[PmxtObject] = []
    common_prefixes: list[str] = []
    next_token: str | None = None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return objs, next_token, common_prefixes

    tag = root.tag.split("}")[-1]
    if tag not in ("ListBucketResult",):
        return objs, next_token, common_prefixes

    ns = {"s": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    pref = "s:" if ns else ""

    for c in root.findall(f"{pref}Contents", ns):
        key = (c.findtext(f"{pref}Key", "", ns) or "").strip()
        size = int(c.findtext(f"{pref}Size", "0", ns) or 0)
        lm = c.findtext(f"{pref}LastModified", "", ns) or ""
        if key:
            objs.append(PmxtObject(key=key, size=size, last_modified=lm, url=""))
    for p in root.findall(f"{pref}CommonPrefixes", ns):
        pk = p.findtext(f"{pref}Prefix", "", ns) or ""
        if pk:
            common_prefixes.append(pk)
    nt = root.findtext(f"{pref}NextContinuationToken", None, ns)
    if nt:
        next_token = nt
    return objs, next_token, common_prefixes


def _find_working_base(session: requests.Session) -> str | None:
    for base in ARCHIVE_BASES:
        r = _get(session, base + "/")
        if r is not None:
            log.info("Archive base reachable: %s", base)
            return base
        # Also try S3-style ?list-type=2 at bucket root
        r = _get(session, base + "/?list-type=2&prefix=Polymarket/&delimiter=/")
        if r is not None:
            log.info("Archive base reachable (S3 listing): %s", base)
            return base
    return None


def list_prefix(session: requests.Session, base: str, prefix: str) -> tuple[list[PmxtObject], list[str]]:
    """List objects under a prefix using S3 ListObjectsV2."""
    url = f"{base}/?list-type=2&prefix={prefix}&delimiter=/"
    r = _get(session, url)
    if r is None:
        return [], []
    objs, token, prefixes = _parse_s3_listing(r.text)
    while token:
        r = _get(session, url + f"&continuation-token={requests.utils.quote(token)}")
        if r is None:
            break
        more_objs, token, more_prefixes = _parse_s3_listing(r.text)
        objs.extend(more_objs)
        prefixes.extend(more_prefixes)
    for o in objs:
        o.url = f"{base}/{o.key}"
    return objs, prefixes


# ---------------------------------------------------------------------------
# Download & parse
# ---------------------------------------------------------------------------


def _in_window(ts: datetime, cycle: int) -> bool:
    for start, end in CYCLE_WINDOWS.get(cycle, []):
        if start <= ts.date() <= end:
            return True
    return False


def _pick_sample(objs: list[PmxtObject], cycle: int, max_per_window: int = 6) -> list[PmxtObject]:
    """Pick a representative subset: up to max_per_window hourly files per window."""
    by_window: dict[tuple[date, date], list[PmxtObject]] = {w: [] for w in CYCLE_WINDOWS.get(cycle, [])}
    for o in objs:
        # Extract YYYY-MM-DD or YYYYMMDD from key
        m = re.search(r"(20\d{2})[-_/]?(\d{2})[-_/]?(\d{2})", o.key)
        if not m:
            continue
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        for w in by_window:
            if w[0] <= d <= w[1]:
                by_window[w].append(o)
                break
    picked: list[PmxtObject] = []
    for w, lst in by_window.items():
        lst.sort(key=lambda x: x.key)
        step = max(1, len(lst) // max_per_window) if len(lst) > max_per_window else 1
        picked.extend(lst[::step][:max_per_window])
    return picked


def download(session: requests.Session, obj: PmxtObject) -> Path | None:
    local = DATA_DIR / obj.key.replace("/", "_")
    if local.exists() and local.stat().st_size > 0:
        return local
    r = _get(session, obj.url, stream=True)
    if r is None:
        return None
    try:
        with open(local, "wb") as fh:
            for chunk in r.iter_content(1 << 16):
                if chunk:
                    fh.write(chunk)
        return local
    except Exception as e:
        log.warning("download %s failed: %s", obj.url, e)
        if local.exists():
            local.unlink()
        return None


def parse_parquet(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        log.warning("parquet read %s failed: %s", path.name, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------


def _infer_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = {c.lower(): c for c in df.columns}
    def pick(*cands: str) -> str | None:
        for c in cands:
            if c in cols:
                return cols[c]
        return None
    return {
        "ts": pick("timestamp", "ts", "time", "as_of", "datetime"),
        "market_id": pick("market_id", "condition_id", "conditionid", "token_id", "id"),
        "question": pick("question", "market_name", "name", "title"),
        "price": pick("price", "mid", "last", "probability", "yes_price"),
        "volume": pick("volume", "volume24hr", "vol"),
    }


def insert_quotes(df: pd.DataFrame, cycle: int, source_file: str) -> int:
    if df.empty:
        return 0
    cmap = _infer_columns(df)
    if not cmap["ts"] or not cmap["price"] or not (cmap["question"] or cmap["market_id"]):
        log.warning("%s: missing required columns (%s)", source_file, cmap)
        return 0

    factory = get_session_factory()
    inserted = 0
    with factory() as s:
        for _, row in df.iterrows():
            try:
                question = str(row[cmap["question"]]) if cmap["question"] else str(row[cmap["market_id"]])
                if not ELECTION_REGEX.search(question):
                    continue
                ts_raw = row[cmap["ts"]]
                if isinstance(ts_raw, (int, float)):
                    ts = datetime.utcfromtimestamp(float(ts_raw) / (1e3 if ts_raw > 1e12 else 1))
                else:
                    ts = pd.to_datetime(ts_raw).to_pydatetime().replace(tzinfo=None)
                price = float(row[cmap["price"]])
                if not (0 <= price <= 1):
                    continue
                link = link_contract_to_race(question)
                market_id = str(row[cmap["market_id"]]) if cmap["market_id"] else question[:120]
                s.add(HistoricalQuote(
                    race_id=link.race_id,
                    platform="polymarket",
                    platform_market_id=market_id[:256],
                    question=question[:1000],
                    cycle=cycle,
                    price=price,
                    as_of=ts,
                ))
                inserted += 1
                if inserted % 500 == 0:
                    s.commit()
            except Exception as e:
                log.debug("row insert skip: %s", e)
        s.commit()
    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", nargs="+", type=int, default=[2018, 2020, 2022, 2024])
    ap.add_argument("--max-per-window", type=int, default=6)
    args = ap.parse_args()

    init_election_db()

    session = requests.Session()
    session.headers.update(HEADERS)

    base = _find_working_base(session)
    if base is None:
        log.error("pmxt archive unreachable from this network "
                  "(DNS hijack / ISP Advanced Security filter). "
                  "No Parquet files could be downloaded.  "
                  "Run this script from a network without the filter or via VPN.")
        summary = {
            "files_downloaded": 0,
            "rows_total": 0,
            "quotes_inserted": 0,
            "coverage_by_year": {},
            "status": "blocked_by_isp_safebrowse",
        }
        print("\n==== pmxt archive summary ====")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 2

    # Enumerate the Polymarket prefix
    top_objs, top_prefixes = list_prefix(session, base, "Polymarket/")
    log.info("Top-level prefixes: %d, objects: %d", len(top_prefixes), len(top_objs))

    # Gather all .parquet keys under each prefix recursively (shallow depth 2)
    candidates: list[PmxtObject] = list(top_objs)
    for p in top_prefixes:
        sub_objs, sub_prefixes = list_prefix(session, base, p)
        candidates.extend(sub_objs)
        for p2 in sub_prefixes[:50]:
            sub2_objs, _ = list_prefix(session, base, p2)
            candidates.extend(sub2_objs)

    parquet_objs = [o for o in candidates if o.key.lower().endswith(".parquet")]
    log.info("Total parquet objects discovered: %d", len(parquet_objs))

    total_files, total_rows, total_inserted = 0, 0, 0
    by_year: dict[int, dict[str, int]] = {}
    for cycle in args.cycles:
        picks = _pick_sample(parquet_objs, cycle, args.max_per_window)
        log.info("Cycle %d: %d candidate files", cycle, len(picks))
        by_year[cycle] = {"files": 0, "rows": 0, "quotes": 0}
        for obj in picks:
            local = download(session, obj)
            if local is None:
                continue
            df = parse_parquet(local)
            if df.empty:
                continue
            total_files += 1
            total_rows += len(df)
            by_year[cycle]["files"] += 1
            by_year[cycle]["rows"] += len(df)
            n = insert_quotes(df, cycle, local.name)
            total_inserted += n
            by_year[cycle]["quotes"] += n
            time.sleep(0.25)

    print("\n==== pmxt archive summary ====")
    print(f"  files_downloaded:  {total_files}")
    print(f"  rows_total:        {total_rows}")
    print(f"  quotes_inserted:   {total_inserted}")
    for y, d in by_year.items():
        print(f"  {y}: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
