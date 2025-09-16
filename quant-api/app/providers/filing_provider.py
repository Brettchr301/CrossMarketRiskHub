"""
T2 — Filing Provider
=====================
Fetches SEC EDGAR 10-K / 10-Q filings for a given ticker,
then extracts structured fundamentals (revenue, gross_margin, capex,
production_volume) using an LLM.

Cache: 90-day TTL for 10-K, 30-day TTL for 10-Q.
Wire-up: feeds into GlobalOpportunityService._fundamental_proxy().

Follows the Protocol-based provider pattern established in base.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "analysis_output" / "filing_cache"


@dataclass(slots=True)
class FilingRow:
    """One structured filing extraction."""
    ticker: str
    filing_type: str            # "10-K" or "10-Q"
    period: str                 # e.g. "2025Q4" or "2025-FY"
    revenue_mm: float           # Revenue in millions USD
    gross_margin: float         # As fraction, e.g. 0.35
    capex_mm: float             # CapEx in millions USD
    production_volume: float    # Primary production metric (bpd, mcfd, etc.)
    raw_url: str                # EDGAR filing URL
    extracted_at: datetime
    meta: dict[str, Any] = field(default_factory=dict)


class FilingProvider(Protocol):
    """Protocol for filing providers."""
    def fetch_filings(self, ticker: str, filing_types: Sequence[str] = ("10-K", "10-Q")) -> Sequence[FilingRow]:
        ...


# ---------------------------------------------------------------------------
# EDGAR helpers
# ---------------------------------------------------------------------------

_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"
_EDGAR_COMPANY = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form}&dateRange=custom&startdt={start}&enddt={end}"

# SEC requires a User-Agent identifying you
_HEADERS = {
    "User-Agent": "CrossMarketRiskHub/1.0 (contact@example.com)",
    "Accept": "application/json",
}


def _cik_for_ticker(ticker: str) -> Optional[str]:
    """Look up CIK number for a ticker via SEC EDGAR company tickers JSON."""
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
    except Exception as exc:
        logger.warning("CIK lookup failed for %s: %s", ticker, exc)
    return None


def _fetch_recent_filings(cik: str, filing_type: str = "10-K", count: int = 3) -> list[dict[str, Any]]:
    """Fetch recent filings metadata from EDGAR submissions API."""
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == filing_type and i < len(dates):
                accession_clean = accessions[i].replace("-", "")
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_docs[i]}"
                results.append({
                    "form": form,
                    "filing_date": dates[i],
                    "accession": accessions[i],
                    "doc_url": doc_url,
                })
                if len(results) >= count:
                    break
        return results

    except Exception as exc:
        logger.warning("EDGAR filings fetch failed for CIK %s: %s", cik, exc)
        return []


def _extract_filing_text_snippet(doc_url: str, max_chars: int = 15_000) -> str:
    """Download filing and extract a text snippet for LLM analysis."""
    try:
        resp = requests.get(doc_url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        text = resp.text

        # Strip HTML tags for plain text
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Try to find financial statements section
        markers = [
            "CONSOLIDATED STATEMENTS OF OPERATIONS",
            "CONSOLIDATED BALANCE SHEET",
            "SELECTED FINANCIAL DATA",
            "RESULTS OF OPERATIONS",
            "REVENUE",
            "Total revenue",
        ]
        best_start = 0
        for marker in markers:
            idx = text.upper().find(marker.upper())
            if idx >= 0:
                best_start = max(0, idx - 200)
                break

        return text[best_start : best_start + max_chars]

    except Exception as exc:
        logger.warning("Filing text extraction failed for %s: %s", doc_url, exc)
        return ""


# ---------------------------------------------------------------------------
# LLM extraction (uses DeepSeek via OpenAI-compat as configured in bridge)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """You are a financial analyst. Extract the following data points from this SEC filing excerpt.
Return ONLY a JSON object with these keys (use null if not found):
- revenue_mm: total revenue in millions USD
- gross_margin: gross margin as a decimal (e.g. 0.35 for 35%)
- capex_mm: capital expenditure in millions USD
- production_volume: primary production metric (barrels per day for oil, mcf/d for gas, TEU for shipping)
- production_unit: unit of production (bpd, mcfd, teu, etc.)
- period: reporting period (e.g. "2025Q4" or "2025-FY")

Filing excerpt:
{text}"""


def _llm_extract_fundamentals(text: str) -> dict[str, Any]:
    """Use DeepSeek API (or fallback to regex heuristics) to extract fundamentals."""
    # Try DeepSeek API first
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key and text:
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": _EXTRACTION_PROMPT.format(text=text[:12_000])},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 512,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Extract JSON from response
            json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as exc:
            logger.warning("LLM extraction failed, falling back to heuristics: %s", exc)

    # Fallback: regex heuristics
    return _regex_extract_fundamentals(text)


def _regex_extract_fundamentals(text: str) -> dict[str, Any]:
    """Fallback extraction using regex patterns for common financial data."""
    result: dict[str, Any] = {
        "revenue_mm": None,
        "gross_margin": None,
        "capex_mm": None,
        "production_volume": None,
        "production_unit": None,
        "period": None,
    }

    # Revenue patterns
    for pattern in [
        r"(?:total\s+)?revenue[s]?\s*[\$:]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|MM|M\b)",
        r"(?:total\s+)?revenue[s]?\s*[\$:]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:billion|B\b)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = float(m.group(1).replace(",", ""))
            if "billion" in pattern.lower():
                val *= 1000
            result["revenue_mm"] = val
            break

    # Gross margin
    m = re.search(r"gross\s+(?:profit\s+)?margin\s*[:\s]*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m:
        result["gross_margin"] = float(m.group(1)) / 100.0

    # CapEx
    m = re.search(r"capital\s+expenditure[s]?\s*[\$:]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|MM|M\b)", text, re.IGNORECASE)
    if m:
        result["capex_mm"] = float(m.group(1).replace(",", ""))

    # Production volume (oil/gas companies)
    m = re.search(r"(?:production|output)\s*[:\s]*([\d,]+(?:\.\d+)?)\s*(boe/d|bpd|barrels?\s*per\s*day|mcf/d|mboe/d)", text, re.IGNORECASE)
    if m:
        result["production_volume"] = float(m.group(1).replace(",", ""))
        result["production_unit"] = m.group(2).strip().lower()

    return result


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_key(ticker: str, filing_type: str) -> str:
    return hashlib.sha256(f"{ticker}:{filing_type}".encode()).hexdigest()[:16]


def _load_cached(ticker: str, filing_type: str, ttl_days: int) -> Optional[list[dict[str, Any]]]:
    """Load cached filing extraction if within TTL."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(ticker, filing_type)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_at = data.get("cached_at", 0)
            if time.time() - cached_at < ttl_days * 86400:
                logger.info("Cache hit for %s %s (age: %.1f days)", ticker, filing_type, (time.time() - cached_at) / 86400)
                return data.get("rows", [])
        except Exception:
            pass
    return None


def _save_cache(ticker: str, filing_type: str, rows: list[dict[str, Any]]) -> None:
    """Save filing extraction to disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(ticker, filing_type)
    cache_file = CACHE_DIR / f"{key}.json"
    data = {
        "ticker": ticker,
        "filing_type": filing_type,
        "cached_at": time.time(),
        "rows": rows,
    }
    cache_file.write_text(json.dumps(data, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------------

class SECFilingProvider:
    """
    Fetches SEC EDGAR filings (10-K, 10-Q) and extracts structured fundamentals.
    
    Usage:
        provider = SECFilingProvider()
        filings = provider.fetch_filings("XOM")
        for f in filings:
            print(f.revenue_mm, f.gross_margin, f.capex_mm, f.production_volume)
    """

    def __init__(self, deepseek_api_key: str | None = None):
        if deepseek_api_key:
            os.environ.setdefault("DEEPSEEK_API_KEY", deepseek_api_key)

    def fetch_filings(
        self,
        ticker: str,
        filing_types: Sequence[str] = ("10-K", "10-Q"),
    ) -> Sequence[FilingRow]:
        """Fetch and extract fundamentals from recent SEC filings."""
        all_rows: list[FilingRow] = []

        for ft in filing_types:
            ttl = 90 if ft == "10-K" else 30

            # Check cache first
            cached = _load_cached(ticker, ft, ttl_days=ttl)
            if cached:
                for row_data in cached:
                    all_rows.append(FilingRow(
                        ticker=row_data["ticker"],
                        filing_type=row_data["filing_type"],
                        period=row_data.get("period", "unknown"),
                        revenue_mm=row_data.get("revenue_mm", 0.0) or 0.0,
                        gross_margin=row_data.get("gross_margin", 0.0) or 0.0,
                        capex_mm=row_data.get("capex_mm", 0.0) or 0.0,
                        production_volume=row_data.get("production_volume", 0.0) or 0.0,
                        raw_url=row_data.get("raw_url", ""),
                        extracted_at=datetime.fromisoformat(row_data.get("extracted_at", "2026-01-01T00:00:00")),
                        meta=row_data.get("meta", {}),
                    ))
                continue

            # Fetch from EDGAR
            cik = _cik_for_ticker(ticker)
            if not cik:
                logger.warning("No CIK found for %s, skipping %s filings", ticker, ft)
                continue

            filings_meta = _fetch_recent_filings(cik, filing_type=ft, count=2)
            if not filings_meta:
                logger.warning("No %s filings found for %s (CIK %s)", ft, ticker, cik)
                continue

            rows_to_cache: list[dict[str, Any]] = []
            for fm in filings_meta:
                # Extract text snippet from filing
                text = _extract_filing_text_snippet(fm["doc_url"])
                if not text:
                    continue

                # Extract fundamentals via LLM or regex
                extracted = _llm_extract_fundamentals(text)

                row = FilingRow(
                    ticker=ticker.upper(),
                    filing_type=ft,
                    period=extracted.get("period") or fm["filing_date"][:7],
                    revenue_mm=float(extracted.get("revenue_mm") or 0.0),
                    gross_margin=float(extracted.get("gross_margin") or 0.0),
                    capex_mm=float(extracted.get("capex_mm") or 0.0),
                    production_volume=float(extracted.get("production_volume") or 0.0),
                    raw_url=fm["doc_url"],
                    extracted_at=datetime.now(UTC).replace(tzinfo=None),
                    meta={
                        "filing_date": fm["filing_date"],
                        "accession": fm["accession"],
                        "production_unit": extracted.get("production_unit"),
                    },
                )
                all_rows.append(row)
                rows_to_cache.append({
                    "ticker": row.ticker,
                    "filing_type": row.filing_type,
                    "period": row.period,
                    "revenue_mm": row.revenue_mm,
                    "gross_margin": row.gross_margin,
                    "capex_mm": row.capex_mm,
                    "production_volume": row.production_volume,
                    "raw_url": row.raw_url,
                    "extracted_at": row.extracted_at.isoformat(),
                    "meta": row.meta,
                })

            if rows_to_cache:
                _save_cache(ticker, ft, rows_to_cache)

        return all_rows

    def get_fundamentals_dict(self, ticker: str) -> dict[str, float]:
        """
        Convenience: return a flat dict compatible with _fundamental_proxy() in global_scan.py.
        Uses the most recent 10-K (preferred) or 10-Q.
        """
        filings = self.fetch_filings(ticker)
        if not filings:
            return {}

        # Prefer 10-K, fallback to most recent
        best = None
        for f in filings:
            if f.filing_type == "10-K":
                best = f
                break
        if not best:
            best = filings[0]

        return {
            "revenue_mm": best.revenue_mm,
            "gross_margin": best.gross_margin,
            "capex_mm": best.capex_mm,
            "production_volume": best.production_volume,
            "filing_type": best.filing_type,
            "period": best.period,
            "source": "sec_edgar",
        }
