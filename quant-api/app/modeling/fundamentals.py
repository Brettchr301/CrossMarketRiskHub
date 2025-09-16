from __future__ import annotations

import re
from datetime import datetime, UTC
from typing import Mapping

import yfinance as yf

from app.config import get_settings
from app.modeling.types import FundamentalStatePoint


NUM_RE = r"([-+]?\d+(?:\.\d+)?)"
SHIPPING_TICKERS = {"TNK", "INSW", "STNG", "SBLK", "DHT", "FRO", "NAT", "GOGL"}


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        if hasattr(value, "iloc"):
            try:
                value = value.iloc[0]
            except Exception:
                pass
        return float(value)
    except (TypeError, ValueError):
        return default


def _pick_statement_value(frame, labels: list[str]) -> float | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    idx_map = {str(idx).strip().lower(): idx for idx in frame.index}
    for label in labels:
        target = label.lower()
        idx = idx_map.get(target)
        if idx is None:
            idx = next((v for k, v in idx_map.items() if target in k), None)
        if idx is None:
            continue
        series = frame.loc[idx]
        if hasattr(series, "dropna"):
            series = series.dropna()
        if getattr(series, "empty", True):
            continue
        return _safe_float(series.iloc[0], default=0.0)
    return None


def _latest_price(symbol: str, period: str = "20d") -> float | None:
    frame = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if frame is None or frame.empty:
        return None
    close_col = "Close" if "Close" in frame.columns else frame.columns[0]
    series = frame[close_col].dropna()
    if series.empty:
        return None
    val = _safe_float(series.iloc[-1], default=None)
    if val is None or val <= 0:
        return None
    return float(val)


def extract_guidance_from_text(text: str) -> dict[str, float]:
    lower = text.lower()
    out: dict[str, float] = {}
    patterns = {
        "production": rf"production(?:\s+growth)?\D+{NUM_RE}",
        "cost_per_unit": rf"(?:lifting|unit)\s+cost\D+{NUM_RE}",
        "capex": rf"capex\D+{NUM_RE}",
        "debt": rf"debt\D+{NUM_RE}",
        "share_count": rf"share(?:s)?\D+{NUM_RE}",
        "utilization": rf"utilization\D+{NUM_RE}",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, lower)
        if match:
            out[key] = float(match.group(1))
    return out


class FundamentalStateBuilder:
    """Quarterly base-state model (filings + transcript-friendly inputs)."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.use_real_data = self.settings.real_data_only
        if self.use_real_data:
            self._oil_spot = _latest_price("BZ=F") or _latest_price("CL=F") or 80.0
            self._shipping_proxy = _latest_price("BOAT") or _latest_price("SEA") or _latest_price("BDRY") or 20.0
        else:
            self._oil_spot = 80.0
            self._shipping_proxy = 20.0
        # sector_type in {"producer", "shipping"}
        self.templates: dict[str, dict[str, float | str]] = {
            "TNK": {
                "sector_type": "shipping",
                "production": 9000.0,  # fleet_days proxy
                "cost_per_unit": 26.0,  # opex
                "transport_cost": 8.0,  # bunkers proxy
                "sga": 52_000_000.0,
                "capex": 130_000_000.0,
                "debt": 1_050_000_000.0,
                "interest_rate": 0.064,
                "hedge_ratio": 0.15,
                "utilization": 0.94,
                "share_count": 206_000_000.0,
            },
            "INSW": {
                "sector_type": "shipping",
                "production": 7100.0,
                "cost_per_unit": 27.0,
                "transport_cost": 8.5,
                "sga": 44_000_000.0,
                "capex": 105_000_000.0,
                "debt": 730_000_000.0,
                "interest_rate": 0.063,
                "hedge_ratio": 0.12,
                "utilization": 0.93,
                "share_count": 95_000_000.0,
            },
            "STNG": {
                "sector_type": "shipping",
                "production": 12800.0,
                "cost_per_unit": 24.0,
                "transport_cost": 7.8,
                "sga": 88_000_000.0,
                "capex": 175_000_000.0,
                "debt": 1_420_000_000.0,
                "interest_rate": 0.061,
                "hedge_ratio": 0.1,
                "utilization": 0.95,
                "share_count": 170_000_000.0,
            },
            "SBLK": {
                "sector_type": "shipping",
                "production": 10400.0,
                "cost_per_unit": 18.0,
                "transport_cost": 6.5,
                "sga": 71_000_000.0,
                "capex": 145_000_000.0,
                "debt": 990_000_000.0,
                "interest_rate": 0.066,
                "hedge_ratio": 0.08,
                "utilization": 0.91,
                "share_count": 108_000_000.0,
            },
            "GOGL": {
                "sector_type": "shipping",
                "production": 11100.0,
                "cost_per_unit": 19.0,
                "transport_cost": 6.9,
                "sga": 76_000_000.0,
                "capex": 120_000_000.0,
                "debt": 1_100_000_000.0,
                "interest_rate": 0.065,
                "hedge_ratio": 0.1,
                "utilization": 0.9,
                "share_count": 204_000_000.0,
            },
            "CIVI": {
                "sector_type": "producer",
                "production": 340_000.0,  # barrels/day equivalent
                "cost_per_unit": 29.0,
                "transport_cost": 7.2,
                "sga": 320_000_000.0,
                "capex": 1_450_000_000.0,
                "debt": 4_300_000_000.0,
                "interest_rate": 0.067,
                "hedge_ratio": 0.46,
                "utilization": 0.98,
                "share_count": 98_000_000.0,
            },
            "SM": {
                "sector_type": "producer",
                "production": 220_000.0,
                "cost_per_unit": 25.0,
                "transport_cost": 6.5,
                "sga": 210_000_000.0,
                "capex": 920_000_000.0,
                "debt": 2_700_000_000.0,
                "interest_rate": 0.068,
                "hedge_ratio": 0.41,
                "utilization": 0.97,
                "share_count": 113_000_000.0,
            },
            "VTLE": {
                "sector_type": "producer",
                "production": 158_000.0,
                "cost_per_unit": 28.0,
                "transport_cost": 7.0,
                "sga": 155_000_000.0,
                "capex": 790_000_000.0,
                "debt": 2_250_000_000.0,
                "interest_rate": 0.069,
                "hedge_ratio": 0.33,
                "utilization": 0.96,
                "share_count": 49_000_000.0,
            },
        }

    def _real_base_template(self, ticker_u: str) -> tuple[dict[str, float | str], float]:
        ticker = yf.Ticker(ticker_u)
        info = {}
        try:
            info = ticker.fast_info or {}
        except Exception:
            info = {}

        fin = ticker.quarterly_financials
        bal = ticker.quarterly_balance_sheet
        cash = ticker.quarterly_cashflow

        revenue_q = _pick_statement_value(fin, ["Total Revenue", "Operating Revenue", "Revenue"])
        if not revenue_q or revenue_q <= 0:
            raise RuntimeError(f"Missing revenue for {ticker_u} from live statements.")

        cogs_q = _pick_statement_value(fin, ["Cost Of Revenue", "Cost of Revenue", "Operating Expense"])
        if cogs_q is None or cogs_q <= 0:
            cogs_q = revenue_q * 0.68
        cogs_q = abs(cogs_q)

        sga_q = _pick_statement_value(
            fin,
            [
                "Selling General And Administration",
                "Selling General Administrative",
                "Selling General Administration",
                "General And Administrative Expense",
            ],
        )
        if sga_q is None or sga_q <= 0:
            sga_q = max(revenue_q * 0.08, 10_000_000.0)

        capex_q = _pick_statement_value(cash, ["Capital Expenditure", "Capital Expenditures", "Capex"])
        if capex_q is None:
            capex_q = revenue_q * 0.12
        capex_q = abs(capex_q)

        debt = _pick_statement_value(bal, ["Total Debt", "Long Term Debt", "Current Debt"])
        if debt is None or debt <= 0:
            debt = max(revenue_q * 2.0, 250_000_000.0)

        interest_q = _pick_statement_value(
            fin, ["Interest Expense", "Net Interest Income", "Interest Expense Non Operating"]
        )
        if interest_q is None:
            interest_q = debt * 0.065 / 4.0
        interest_q = abs(interest_q)

        shares = _safe_float(
            info.get("shares")
            or info.get("sharesOutstanding")
            or _pick_statement_value(
                bal, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]
            ),
            default=0.0,
        )
        if shares <= 0:
            raise RuntimeError(f"Missing shares outstanding for {ticker_u}.")

        annual_revenue = revenue_q * 4.0
        annual_cogs = cogs_q * 4.0
        annual_sga = sga_q * 4.0
        annual_capex = capex_q * 4.0
        sector_type = "shipping" if ticker_u in SHIPPING_TICKERS else "producer"
        operating_margin = max(0.05, min(0.5, (annual_revenue - annual_cogs - annual_sga) / max(annual_revenue, 1.0)))

        if sector_type == "producer":
            realized_price = max(45.0, self._oil_spot * 0.9)
            annual_volume = annual_revenue / max(realized_price, 1.0)
            production = max(5_000.0, min(2_500_000.0, annual_volume / 365.0))
            cost_per_unit = annual_cogs / max(annual_volume, 1.0)
            transport_cost = max(1.5, min(12.0, cost_per_unit * 0.18))
            hedge_ratio = max(0.1, min(0.8, 0.45 - operating_margin * 0.2))
            utilization = max(0.9, min(0.99, 0.94 + operating_margin * 0.08))
        else:
            tce_proxy = max(12_000.0, self._shipping_proxy * 2_200.0)
            utilization = max(0.75, min(0.97, 0.82 + operating_margin * 0.35))
            fleet_days = annual_revenue / max(tce_proxy * utilization, 1.0)
            production = max(1_500.0, min(120_000.0, fleet_days))
            cost_per_unit = annual_cogs / max(production, 1.0)
            transport_cost = max(2.0, min(30.0, cost_per_unit * 0.25))
            hedge_ratio = max(0.02, min(0.3, 0.12 - operating_margin * 0.05))

        cost_per_unit = max(5.0, min(65.0, cost_per_unit))
        interest_rate = max(0.01, min(0.15, (interest_q * 4.0) / max(debt, 1.0)))

        confidence = 0.6
        for x in (revenue_q, cogs_q, sga_q, capex_q, debt, shares):
            if x and x > 0:
                confidence += 0.055
        confidence = min(confidence, 0.93)

        return (
            {
                "sector_type": sector_type,
                "production": production,
                "cost_per_unit": cost_per_unit,
                "transport_cost": transport_cost,
                "sga": annual_sga,
                "capex": annual_capex,
                "debt": debt,
                "interest_rate": interest_rate,
                "hedge_ratio": hedge_ratio,
                "utilization": utilization,
                "share_count": shares,
                "production_growth_assumption": max(-0.18, min(0.26, (operating_margin - 0.16) * 0.45)),
                "source": "yfinance_financial_statements",
            },
            confidence,
        )

    def build_state(
        self,
        ticker: str,
        guidance_period: str,
        guidance_text: str | None = None,
        overrides: Mapping[str, float] | None = None,
    ) -> FundamentalStatePoint:
        ticker_u = ticker.upper()
        source = "template_fallback"
        if self.use_real_data:
            base, confidence = self._real_base_template(ticker_u)
            source = str(base.get("source", "yfinance_financial_statements"))
        else:
            base = dict(self.templates.get(ticker_u, self._generic_template(ticker_u)))
            confidence = 0.75

        parsed = extract_guidance_from_text(guidance_text or "")
        if parsed:
            for key, value in parsed.items():
                if key in base:
                    base[key] = value
        if overrides:
            for key, value in overrides.items():
                if key in base:
                    base[key] = value

        if guidance_text:
            confidence += 0.1
            source = f"{source}+guidance_text"
        if overrides:
            confidence += 0.05
            source = f"{source}+overrides"
        confidence = min(confidence, 0.95)
        assumptions = self._dynamic_assumptions(base=base, ticker=ticker_u)
        assumptions["source"] = source
        assumptions["ticker"] = ticker_u

        return FundamentalStatePoint(
            ticker=ticker_u,
            guidance_period=guidance_period,
            sector_type=str(base["sector_type"]),
            production=float(base["production"]),
            cost_per_unit=float(base["cost_per_unit"]),
            transport_cost=float(base["transport_cost"]),
            sga=float(base["sga"]),
            capex=float(base["capex"]),
            debt=float(base["debt"]),
            interest_rate=float(base["interest_rate"]),
            hedge_ratio=float(base["hedge_ratio"]),
            utilization=float(base["utilization"]),
            share_count=float(base["share_count"]),
            confidence=confidence,
            meta_payload=assumptions,
            as_of=datetime.now(UTC).replace(tzinfo=None),
        )

    @staticmethod
    def _generic_template(ticker: str) -> dict[str, float | str]:
        if ticker.startswith(("S", "G", "T", "I")):
            sector = "shipping"
        else:
            sector = "producer"
        return {
            "sector_type": sector,
            "production": 10000.0 if sector == "shipping" else 180000.0,
            "cost_per_unit": 22.0 if sector == "shipping" else 27.0,
            "transport_cost": 7.0,
            "sga": 80_000_000.0,
            "capex": 300_000_000.0,
            "debt": 1_200_000_000.0,
            "interest_rate": 0.065,
            "hedge_ratio": 0.2,
            "utilization": 0.92,
            "share_count": 100_000_000.0,
            "production_growth_assumption": 0.03,
        }

    @staticmethod
    def _dynamic_assumptions(base: Mapping[str, float | str], ticker: str) -> dict[str, float | str]:
        sector = str(base["sector_type"])
        growth = float(base.get("production_growth_assumption", 0.03))
        leverage = max(0.2, min(2.4, float(base["debt"]) / max(float(base["sga"]), 1.0)))
        if sector == "producer":
            realized_beta = 1.0 + 0.18 * leverage
            realized_gamma = 0.35 + 0.08 * leverage
            unit_cost_beta = 0.18 + 0.03 * leverage
            transport_beta = 0.12 + 0.02 * leverage
            growth_beta = 0.2 + 0.04 * leverage
            return {
                "production_growth_assumption": max(-0.2, min(0.35, growth)),
                "growth_beta_oil": max(0.12, min(0.6, growth_beta)),
                "realized_price_beta_oil": max(0.8, min(1.7, realized_beta)),
                "realized_price_gamma_oil": max(0.15, min(1.2, realized_gamma)),
                "unit_cost_beta_oil": max(0.08, min(0.6, unit_cost_beta)),
                "transport_beta_oil": max(0.06, min(0.5, transport_beta)),
            }

        tce_beta = 1.05 + 0.16 * leverage
        tce_gamma = 0.3 + 0.1 * leverage
        util_beta = 0.08 + 0.04 * leverage
        opex_beta = 0.06 + 0.02 * leverage
        bunker_beta = 0.2 + 0.06 * leverage
        return {
            "fleet_growth_assumption": max(-0.12, min(0.2, growth)),
            "tce_beta_freight": max(0.7, min(1.9, tce_beta)),
            "tce_gamma_freight": max(0.1, min(1.3, tce_gamma)),
            "utilization_beta_freight": max(0.05, min(0.35, util_beta)),
            "opex_beta_freight": max(0.04, min(0.3, opex_beta)),
            "bunker_beta_freight": max(0.12, min(0.75, bunker_beta)),
        }
