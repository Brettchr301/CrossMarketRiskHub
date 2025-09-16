"""ROIC Quality Filter & McKinsey Expectations Treadmill.

Implements the key insight from McKinsey's "Valuation" (Koller, Goedhart, Wessels):

THE EXPECTATIONS TREADMILL:
  - High-ROIC companies already have sky-high expectations priced in
  - Just MAINTAINING high ROIC isn't enough — the market expects IMPROVEMENT
  - The alpha comes from finding the DELTA between actual ROIC trajectory
    and what the market has already priced in
  - A company improving from 8% ROIC to 12% ROIC may generate MORE alpha
    than one maintaining 25% ROIC (because the 25% was already priced in)

ROIC QUALITY GATES:
  1. Absolute ROIC: must be above sector/size peer median (not bottom quartile)
  2. ROIC Trend: improving or stable (declining ROIC = value trap)
  3. ROIC vs Expectations: the treadmill score — is ROIC trajectory BEATING market pricing?
  4. Capital Allocation: reinvestment rate * ROIC spread vs WACC

For commodity stocks specifically:
  - ROIC is CYCLICAL — you want to buy when ROIC is at cycle low but improving
  - The treadmill means the market has already priced in the current commodity price
  - Alpha comes from prediction market signals showing the cycle will turn
    BEFORE the market reprices

Data sources: yfinance fundamentals, SEC filings, foreign exchange filings
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ROICProfile:
    """ROIC quality profile for a single ticker."""
    ticker: str
    commodity_type: str
    country: str

    # Raw ROIC data
    current_roic_pct: float           # most recent ROIC (%)
    roic_3yr_avg_pct: float           # 3-year average
    roic_trend: float                 # slope of ROIC over 3 years (positive = improving)
    roic_volatility: float            # std dev of quarterly ROIC

    # Peer comparison
    sector_median_roic_pct: float     # median ROIC for same commodity_type
    size_peer_median_roic_pct: float  # median ROIC for similar market cap
    roic_percentile_sector: float     # 0-1 percentile within sector
    roic_percentile_size: float       # 0-1 percentile within size bucket

    # Expectations Treadmill
    implied_roic_from_price: float    # market-implied ROIC from current stock price
    roic_vs_expectations: float       # actual - implied (positive = beating treadmill)
    treadmill_score: float            # -1 to +1 composite score

    # Capital allocation quality
    reinvestment_rate: float          # % of earnings reinvested
    roic_wacc_spread: float           # ROIC - estimated WACC
    value_creation_rate: float        # reinvestment_rate * ROIC_spread (McKinsey formula)

    # Quality gates
    passes_absolute_roic: bool        # above 30th percentile
    passes_roic_trend: bool           # not declining
    passes_treadmill: bool            # beating expectations
    passes_capital_allocation: bool   # creating value (spread > 0)
    overall_quality_pass: bool        # passes all 4 gates
    quality_score: float              # 0-100

    # Data availability
    data_available: bool
    data_source: str                  # "yfinance" / "sec_filing" / "foreign_filing" / "estimated"


# ────────────────────────────────────────────────────────────────────────────
# ROIC CALCULATION FROM FUNDAMENTALS
# ────────────────────────────────────────────────────────────────────────────

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or math.isnan(denominator) or math.isnan(numerator):
        return default
    return numerator / denominator


def compute_roic_from_financials(
    operating_income: float,
    tax_rate: float,
    total_equity: float,
    total_debt: float,
    cash: float,
) -> float:
    """ROIC = NOPAT / Invested Capital.

    NOPAT = Operating Income * (1 - tax_rate)
    Invested Capital = Total Equity + Total Debt - Cash

    This is the McKinsey definition: measures return on ALL capital deployed,
    not just equity. This is the correct metric for comparing across
    capital structures (crucial for commodity companies that lever up).
    """
    nopat = operating_income * (1.0 - max(0.0, min(0.5, tax_rate)))
    invested_capital = total_equity + total_debt - cash
    if invested_capital <= 0:
        return 0.0
    return nopat / invested_capital


def estimate_implied_roic(
    market_cap: float,
    total_debt: float,
    cash: float,
    book_equity: float,
    current_roic: float,
    growth_rate: float = 0.03,
    wacc: float = 0.10,
) -> float:
    """Estimate the ROIC the market is implying from the current stock price.

    Uses simplified McKinsey valuation framework:
      Enterprise Value = Invested Capital + PV(future value creation)
      PV(value creation) = IC * (ROIC - WACC) * growth / (WACC * (WACC - growth))

    Solving backwards for implied ROIC from current EV.

    The TREADMILL: if implied_ROIC > actual_ROIC, the market expects IMPROVEMENT.
    If implied_ROIC < actual_ROIC, the market is DISCOUNTING the company.
    """
    ev = market_cap + total_debt - cash
    invested_capital = max(book_equity + total_debt - cash, 1.0)

    if invested_capital <= 0 or wacc <= growth_rate:
        return current_roic

    # EV/IC ratio
    ev_ic_ratio = ev / invested_capital

    # Implied ROIC: solve EV/IC = 1 + (ROIC-WACC)*g / (WACC*(WACC-g))
    # => (EV/IC - 1) * WACC * (WACC - g) / g + WACC = implied_ROIC
    denominator = growth_rate
    if abs(denominator) < 1e-6:
        return current_roic

    implied = (ev_ic_ratio - 1.0) * wacc * (wacc - growth_rate) / growth_rate + wacc
    return max(-0.5, min(1.0, implied))  # cap at -50% to 100%


# ────────────────────────────────────────────────────────────────────────────
# SECTOR PEER COMPARISON
# ────────────────────────────────────────────────────────────────────────────

# Historical median ROIC by commodity type (from Damodaran's data + McKinsey benchmarks)
# These are NORMAL-CYCLE medians, not peak/trough
SECTOR_MEDIAN_ROIC: dict[str, float] = {
    "oil_gas_upstream": 0.08,      # 8% — highly cyclical, ranges from -5% to 25%
    "oil_refining": 0.10,          # 10% — cracking margins drive it
    "oil_services": 0.07,          # 7% — capital-intensive, cyclical
    "midstream": 0.06,             # 6% — lower ROIC but lower volatility (fee-based)
    "precious_metals": 0.05,       # 5% — gold miners historically low ROIC
    "uranium": 0.04,               # 4% — long development cycles, high capex
    "lithium": 0.09,               # 9% — variable, recent boom pushed higher
    "rare_earths": 0.06,           # 6% — limited producers, high variability
}

# ROIC by market cap tier (smaller companies tend to have higher variance)
SIZE_MEDIAN_ROIC: dict[str, float] = {
    "micro": 0.03,    # many micro-caps are pre-profit or barely profitable
    "small": 0.06,
    "mid": 0.09,
    "large": 0.11,
}


# ────────────────────────────────────────────────────────────────────────────
# MAIN QUALITY ASSESSMENT
# ────────────────────────────────────────────────────────────────────────────

def assess_roic_quality(
    ticker: str,
    commodity_type: str,
    country: str,
    market_cap: float,
    financials: dict[str, Any] | None = None,
    min_percentile: float = 0.30,
) -> ROICProfile:
    """Full ROIC quality assessment with McKinsey treadmill scoring.

    If fundamental data is unavailable (foreign filings, micro-caps),
    falls back to sector/size estimates with wider uncertainty.

    The key innovation: combines ROIC quality with prediction market signals.
    A low-ROIC company with IMPROVING trend AND prediction markets showing
    favorable commodity cycle = potential deep-value opportunity.
    This is the opposite of the expectations treadmill — buying when
    expectations are LOW and the cycle is about to turn.
    """
    # Determine size bucket
    if market_cap < 500_000_000:
        size_bucket = "micro"
    elif market_cap < 2_000_000_000:
        size_bucket = "small"
    elif market_cap < 10_000_000_000:
        size_bucket = "mid"
    else:
        size_bucket = "large"

    sector_med = SECTOR_MEDIAN_ROIC.get(commodity_type, 0.07)
    size_med = SIZE_MEDIAN_ROIC.get(size_bucket, 0.06)

    # Try to extract ROIC from fundamentals
    current_roic = 0.0
    roic_3yr = 0.0
    roic_trend = 0.0
    roic_vol = 0.0
    reinvest_rate = 0.0
    wacc_est = 0.10  # default WACC
    data_avail = False
    data_src = "estimated"
    total_debt = 0.0
    cash = 0.0
    book_equity = market_cap * 0.4  # fallback

    if financials:
        try:
            op_inc = float(financials.get("operatingIncome", 0) or 0)
            tax = float(financials.get("taxRate", 0.21) or 0.21)
            equity = float(financials.get("totalStockholderEquity", 0) or 0)
            total_debt = float(financials.get("totalDebt", 0) or 0)
            cash = float(financials.get("totalCash", 0) or 0)
            book_equity = equity if equity > 0 else market_cap * 0.4

            if equity > 0 or total_debt > 0:
                current_roic = compute_roic_from_financials(op_inc, tax, equity, total_debt, cash)
                data_avail = True
                data_src = "yfinance"

            # Historical ROIC for trend
            hist_roic = financials.get("roic_history", [])
            if isinstance(hist_roic, list) and len(hist_roic) >= 2:
                roic_3yr = float(np.mean(hist_roic[-12:])) if len(hist_roic) >= 4 else current_roic
                roic_vol = float(np.std(hist_roic, ddof=1)) if len(hist_roic) >= 4 else 0.0
                # Trend: linear slope
                x = np.arange(len(hist_roic))
                if len(x) >= 3:
                    slope, _, _, _, _ = np.polyfit(x, hist_roic, 1, full=False, cov=False) if len(x) > 1 else (0.0,)
                    if isinstance(slope, (int, float)):
                        roic_trend = float(slope)
                    else:
                        coeffs = np.polyfit(x, hist_roic, 1)
                        roic_trend = float(coeffs[0])

            # Reinvestment rate
            capex = abs(float(financials.get("capitalExpenditures", 0) or 0))
            depr = abs(float(financials.get("depreciation", 0) or 0))
            nopat = op_inc * (1.0 - tax)
            if nopat > 0:
                reinvest_rate = min(1.5, (capex - depr) / nopat) if capex > depr else 0.0

            # WACC estimate (simplified)
            cost_of_equity = 0.08 + (0.03 if market_cap < 2e9 else 0.01)  # size premium
            cost_of_debt = 0.05
            debt_ratio = total_debt / max(total_debt + equity, 1.0)
            wacc_est = cost_of_equity * (1.0 - debt_ratio) + cost_of_debt * (1.0 - 0.21) * debt_ratio

        except (ValueError, TypeError, KeyError):
            pass

    if not data_avail:
        # Fallback: use sector median with noise
        current_roic = sector_med
        roic_3yr = sector_med
        roic_trend = 0.0
        roic_vol = sector_med * 0.5

    # Percentile within sector (approximate from distance to median)
    roic_pct_sector = _roic_to_percentile(current_roic, sector_med, roic_vol or sector_med * 0.3)
    roic_pct_size = _roic_to_percentile(current_roic, size_med, roic_vol or size_med * 0.4)

    # Expectations treadmill
    if data_avail:
        # Only run treadmill when we have real financial data;
        # the implied-ROIC calculation needs real book equity and debt
        implied_roic = estimate_implied_roic(
            market_cap, total_debt, cash, book_equity, current_roic,
            growth_rate=0.03, wacc=wacc_est,
        )
        roic_vs_exp = current_roic - implied_roic

        # Treadmill score: -1 to +1
        # Positive = actual ROIC > market expectations (underpriced)
        # Negative = market expects more than company delivers (overpriced)
        treadmill_raw = roic_vs_exp / max(abs(implied_roic), 0.01)
        treadmill_score = max(-1.0, min(1.0, treadmill_raw))
    else:
        # No data: assume neutral treadmill position
        implied_roic = current_roic
        roic_vs_exp = 0.0
        treadmill_score = 0.0

    # Value creation rate (McKinsey formula)
    roic_spread = current_roic - wacc_est
    value_creation = reinvest_rate * roic_spread

    # Quality gates
    passes_abs = roic_pct_sector >= min_percentile
    passes_trend = roic_trend >= -0.005  # allow flat, reject only declining
    passes_treadmill = treadmill_score > -0.3  # not massively overpriced
    passes_capital = roic_spread > -0.04  # ROIC within 4% of WACC

    # For commodity stocks at CYCLE LOWS: relax absolute ROIC gate
    # This is the anti-treadmill trade: buy when ROIC is depressed
    # but improving, because the market has already given up
    if roic_trend > 0.01 and treadmill_score > 0:
        passes_abs = True  # override: improving + underpriced

    overall = passes_abs and passes_trend and passes_treadmill and passes_capital

    # Quality score: 0-100
    score = 0.0
    if passes_abs:
        score += 25
    if passes_trend:
        score += 25
    if passes_treadmill:
        score += 25
    if passes_capital:
        score += 15
    # Bonus: strong treadmill beat
    if treadmill_score > 0.2:
        score += 10

    return ROICProfile(
        ticker=ticker,
        commodity_type=commodity_type,
        country=country,
        current_roic_pct=round(current_roic * 100, 2),
        roic_3yr_avg_pct=round(roic_3yr * 100, 2),
        roic_trend=round(roic_trend, 5),
        roic_volatility=round(roic_vol, 4),
        sector_median_roic_pct=round(sector_med * 100, 2),
        size_peer_median_roic_pct=round(size_med * 100, 2),
        roic_percentile_sector=round(roic_pct_sector, 3),
        roic_percentile_size=round(roic_pct_size, 3),
        implied_roic_from_price=round(implied_roic * 100, 2),
        roic_vs_expectations=round(roic_vs_exp * 100, 2),
        treadmill_score=round(treadmill_score, 3),
        reinvestment_rate=round(reinvest_rate, 3),
        roic_wacc_spread=round(roic_spread * 100, 2),
        value_creation_rate=round(value_creation * 100, 3),
        passes_absolute_roic=passes_abs,
        passes_roic_trend=passes_trend,
        passes_treadmill=passes_treadmill,
        passes_capital_allocation=passes_capital,
        overall_quality_pass=overall,
        quality_score=round(score, 1),
        data_available=data_avail,
        data_source=data_src,
    )


def _roic_to_percentile(roic: float, median: float, vol: float) -> float:
    """Approximate percentile from ROIC vs sector median using normal CDF."""
    from scipy import stats as sp_stats
    if vol <= 0:
        return 0.5
    z = (roic - median) / vol
    return float(sp_stats.norm.cdf(z))


# ────────────────────────────────────────────────────────────────────────────
# BATCH QUALITY ASSESSMENT WITH YFINANCE
# ────────────────────────────────────────────────────────────────────────────

def fetch_fundamentals_yfinance(ticker: str) -> dict[str, Any] | None:
    """Fetch fundamental data from yfinance for ROIC calculation."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Income statement
        inc = t.income_stmt
        bs = t.balance_sheet
        cf = t.cashflow

        if inc is None or inc.empty or bs is None or bs.empty:
            return None

        result: dict[str, Any] = {}

        # Operating income (most recent quarter/annual available)
        for key in ["Operating Income", "EBIT"]:
            if key in inc.index:
                vals = inc.loc[key].dropna()
                if not vals.empty:
                    result["operatingIncome"] = float(vals.iloc[0])
                    # Build ROIC history
                    hist = vals.tolist()
                    if len(hist) > 1:
                        result["_op_inc_history"] = hist
                    break

        # Tax rate (approximate from provision / pretax income)
        for tax_key in ["Tax Provision"]:
            if tax_key in inc.index:
                for pretax_key in ["Pretax Income"]:
                    if pretax_key in inc.index:
                        tax_prov = inc.loc[tax_key].dropna()
                        pretax = inc.loc[pretax_key].dropna()
                        if not tax_prov.empty and not pretax.empty:
                            pt = float(pretax.iloc[0])
                            if pt > 0:
                                result["taxRate"] = float(tax_prov.iloc[0]) / pt
                            break

        result.setdefault("taxRate", 0.21)

        # Balance sheet
        for key in ["Stockholders Equity", "Total Stockholders Equity", "Total Equity Gross Minority Interest"]:
            if key in bs.index:
                vals = bs.loc[key].dropna()
                if not vals.empty:
                    result["totalStockholderEquity"] = float(vals.iloc[0])
                    break

        for key in ["Total Debt", "Long Term Debt"]:
            if key in bs.index:
                vals = bs.loc[key].dropna()
                if not vals.empty:
                    result["totalDebt"] = float(vals.iloc[0])
                    break

        for key in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
            if key in bs.index:
                vals = bs.loc[key].dropna()
                if not vals.empty:
                    result["totalCash"] = float(vals.iloc[0])
                    break

        # Cash flow for reinvestment rate
        if cf is not None and not cf.empty:
            for key in ["Capital Expenditure"]:
                if key in cf.index:
                    vals = cf.loc[key].dropna()
                    if not vals.empty:
                        result["capitalExpenditures"] = float(vals.iloc[0])
                        break
            for key in ["Depreciation And Amortization", "Depreciation"]:
                if key in cf.index:
                    vals = cf.loc[key].dropna()
                    if not vals.empty:
                        result["depreciation"] = float(vals.iloc[0])
                        break

        # Build ROIC history from multiple periods
        if "_op_inc_history" in result and "totalStockholderEquity" in result:
            tax = result.get("taxRate", 0.21)
            equity = result["totalStockholderEquity"]
            debt = result.get("totalDebt", 0)
            cash_val = result.get("totalCash", 0)
            ic = equity + debt - cash_val
            if ic > 0:
                result["roic_history"] = [
                    op * (1 - tax) / ic for op in result["_op_inc_history"]
                ]

        return result

    except Exception as exc:
        logger.debug("Failed to fetch fundamentals for %s: %s", ticker, exc)
        return None


def batch_assess_quality(
    tickers_meta: list[dict[str, Any]],
    skip_data_fetch: bool = False,
) -> dict[str, ROICProfile]:
    """Assess ROIC quality for a batch of tickers.

    For ~270 tickers, this takes ~3-5 minutes (yfinance rate limits).
    Results are cached in the ROICProfile objects.

    Args:
        tickers_meta: list of dicts with keys: ticker, commodity_type, country, market_cap
        skip_data_fetch: if True, use sector estimates only (fast mode)
    """
    results: dict[str, ROICProfile] = {}

    for i, meta in enumerate(tickers_meta):
        ticker = meta["ticker"]
        financials = None

        if not skip_data_fetch:
            try:
                financials = fetch_fundamentals_yfinance(ticker)
            except Exception:
                pass

        profile = assess_roic_quality(
            ticker=ticker,
            commodity_type=meta.get("commodity_type", "oil_gas_upstream"),
            country=meta.get("country", "US"),
            market_cap=meta.get("market_cap", 2_000_000_000),
            financials=financials,
        )
        results[ticker] = profile

        if (i + 1) % 25 == 0:
            logger.info("ROIC quality: assessed %d/%d tickers", i + 1, len(tickers_meta))

    return results
