"""
E5 — Options Vol Surface Provider
===================================
Fetches live options chains and computes alpha-grade features:
  - IV skew (25-delta put IV − ATM call IV)
  - IV term structure slope (frontmonth vs 3-month)
  - Put/call open interest ratio
  - Put/call volume ratio
  - Net gamma exposure (simplified, $MM scale)

Designed as a lightweight provider compatible with _build_feature_frame().
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt yfinance import — graceful degradation if unavailable
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


_DEFAULT_FEATURES: dict[str, float] = {
    "iv_skew": 0.0,
    "iv_term_slope": 0.0,
    "put_call_oi_ratio": 1.0,
    "put_call_vol_ratio": 1.0,
    "gamma_exposure_mm": 0.0,
}


class OptionsVolSurfaceProvider:
    """
    Compute options-implied vol surface features for equity tickers.

    Usage:
        provider = OptionsVolSurfaceProvider()
        feats = provider.get_features("XOM")
        # feats = {"iv_skew": -0.03, "iv_term_slope": 0.02, ...}
    """

    def __init__(self, cache_ttl_minutes: int = 30):
        self.cache_ttl = cache_ttl_minutes
        self._cache: dict[str, tuple[datetime, dict[str, float]]] = {}

    def get_features(self, ticker: str) -> dict[str, float]:
        """Return vol surface features for a ticker. Cache-aware."""
        now = datetime.now(UTC).replace(tzinfo=None)

        # Check cache
        cached = self._cache.get(ticker)
        if cached and (now - cached[0]).total_seconds() < self.cache_ttl * 60:
            return cached[1]

        if not _HAS_YF:
            return dict(_DEFAULT_FEATURES)

        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                return dict(_DEFAULT_FEATURES)

            # Get most liquid (nearest) chain
            nearest_chain = stock.option_chain(expirations[0])
            calls = nearest_chain.calls
            puts = nearest_chain.puts

            # Get spot price for gamma calc
            hist = stock.history(period="2d")
            spot = float(hist["Close"].iloc[-1]) if len(hist) > 0 else 0.0

            features = {
                **self._iv_skew(calls, puts, spot),
                **self._iv_term_structure(stock, expirations),
                **self._put_call_ratios(calls, puts),
                **self._gamma_exposure(calls, puts, spot),
            }

            self._cache[ticker] = (now, features)
            return features

        except Exception as exc:
            logger.debug("Options vol surface failed for %s: %s", ticker, exc)
            return dict(_DEFAULT_FEATURES)

    # -------------------------------------------------------------------
    # Feature computation
    # -------------------------------------------------------------------

    @staticmethod
    def _iv_skew(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> dict[str, float]:
        """IV skew: 25-delta put IV minus ATM call IV."""
        if calls.empty or puts.empty or spot <= 0:
            return {"iv_skew": 0.0}

        try:
            # Find ATM call IV
            call_ivs = calls.dropna(subset=["impliedVolatility"])
            if call_ivs.empty:
                return {"iv_skew": 0.0}
            atm_idx = (call_ivs["strike"] - spot).abs().idxmin()
            atm_call_iv = float(call_ivs.loc[atm_idx, "impliedVolatility"])

            # Find ~25-delta put (strike ≈ 90-95% of spot)
            target_strike = spot * 0.925
            put_ivs = puts.dropna(subset=["impliedVolatility"])
            if put_ivs.empty:
                return {"iv_skew": 0.0}
            put_idx = (put_ivs["strike"] - target_strike).abs().idxmin()
            otm_put_iv = float(put_ivs.loc[put_idx, "impliedVolatility"])

            skew = otm_put_iv - atm_call_iv
            return {"iv_skew": float(np.clip(skew, -0.5, 0.5))}
        except Exception:
            return {"iv_skew": 0.0}

    @staticmethod
    def _iv_term_structure(stock: Any, expirations: tuple[str, ...]) -> dict[str, float]:
        """IV term structure slope: 3-month ATM IV minus frontmonth ATM IV."""
        if len(expirations) < 2:
            return {"iv_term_slope": 0.0}

        try:
            today = datetime.now().date()
            exp_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in expirations]
            days_out = [(e - today).days for e in exp_dates]

            # Frontmonth
            front_idx = 0
            # Find ~60-90 day expiration
            back_idx = None
            for i, d in enumerate(days_out):
                if 50 <= d <= 120:
                    back_idx = i
                    break

            if back_idx is None:
                # Use the furthest available if no 60-90 day
                back_idx = min(len(expirations) - 1, 2)

            front_chain = stock.option_chain(expirations[front_idx])
            back_chain = stock.option_chain(expirations[back_idx])

            front_calls = front_chain.calls.dropna(subset=["impliedVolatility"])
            back_calls = back_chain.calls.dropna(subset=["impliedVolatility"])

            if front_calls.empty or back_calls.empty:
                return {"iv_term_slope": 0.0}

            # ATM IV for each expiration
            mid_front = len(front_calls) // 2
            mid_back = len(back_calls) // 2
            front_iv = float(front_calls.iloc[mid_front]["impliedVolatility"])
            back_iv = float(back_calls.iloc[mid_back]["impliedVolatility"])

            slope = back_iv - front_iv
            return {"iv_term_slope": float(np.clip(slope, -0.5, 0.5))}
        except Exception:
            return {"iv_term_slope": 0.0}

    @staticmethod
    def _put_call_ratios(calls: pd.DataFrame, puts: pd.DataFrame) -> dict[str, float]:
        """Put/call ratios by open interest and volume."""
        pc_oi = 1.0
        pc_vol = 1.0

        try:
            if not calls.empty and not puts.empty:
                call_oi = calls["openInterest"].sum()
                put_oi = puts["openInterest"].sum()
                if call_oi > 0:
                    pc_oi = float(put_oi / call_oi)

                if "volume" in calls.columns and "volume" in puts.columns:
                    call_vol = calls["volume"].sum()
                    put_vol = puts["volume"].sum()
                    if call_vol > 0:
                        pc_vol = float(put_vol / call_vol)
        except Exception:
            pass

        return {
            "put_call_oi_ratio": float(np.clip(pc_oi, 0.0, 10.0)),
            "put_call_vol_ratio": float(np.clip(pc_vol, 0.0, 10.0)),
        }

    @staticmethod
    def _gamma_exposure(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> dict[str, float]:
        """Simplified net gamma exposure in $MM scale."""
        if calls.empty or puts.empty or spot <= 0:
            return {"gamma_exposure_mm": 0.0}

        try:
            net_gamma = 0.0

            # Call gamma (positive) — weighted by proximity to ATM and OI
            for _, row in calls.iterrows():
                strike = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                if oi <= 0 or strike <= 0:
                    continue
                moneyness = abs(strike - spot) / spot
                gamma_weight = max(0.0, 1.0 - moneyness * 5.0)  # 20% OTM → 0 weight
                net_gamma += oi * gamma_weight

            # Put gamma (negative for dealers)
            for _, row in puts.iterrows():
                strike = float(row.get("strike", 0))
                oi = float(row.get("openInterest", 0) or 0)
                if oi <= 0 or strike <= 0:
                    continue
                moneyness = abs(strike - spot) / spot
                gamma_weight = max(0.0, 1.0 - moneyness * 5.0)
                net_gamma -= oi * gamma_weight

            # Scale to $MM (100 shares per contract)
            gamma_mm = net_gamma * spot * 100.0 / 1_000_000.0
            return {"gamma_exposure_mm": float(np.clip(gamma_mm, -1000.0, 1000.0))}
        except Exception:
            return {"gamma_exposure_mm": 0.0}
