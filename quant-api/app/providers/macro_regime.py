"""
E7 — Macro Regime Provider
============================
Fetches macro regime indicators (VIX, DXY, yield curve, Gold, S&P500)
and computes regime classification features.

These features capture factor rotation regimes that drive
commodity equity performance (strong dollar = bad for commodity equities,
high VIX = risk-off reduces shipping volumes, etc.)
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


# Macro tickers and their feature names
_MACRO_TICKERS = {
    "^VIX": "vix",
    "DX-Y.NYB": "dxy",
    "^TNX": "us10y",
    "GC=F": "gold",
    "^GSPC": "sp500",
    "^IRX": "us3m",  # 3-month T-bill (for yield curve spread)
}

_DEFAULT_FEATURES: dict[str, float] = {
    "macro_vix": 18.0,
    "macro_vix_ret_5d": 0.0,
    "macro_dxy": 104.0,
    "macro_dxy_ret_5d": 0.0,
    "macro_gold_ret_5d": 0.0,
    "macro_sp500_ret_5d": 0.0,
    "macro_yield_curve_spread": 0.0,
    "macro_risk_off_regime": 0.0,
}


class MacroRegimeProvider:
    """
    Compute macro regime features for integration into _build_feature_frame().

    Usage:
        provider = MacroRegimeProvider()
        feats = provider.get_features()
        # feats = {"macro_vix": 22.5, "macro_risk_off_regime": 1.0, ...}
    """

    def __init__(self, cache_ttl_minutes: int = 15):
        self.cache_ttl = cache_ttl_minutes
        self._cache: tuple[datetime, dict[str, float]] | None = None

    def get_features(self) -> dict[str, float]:
        """Return macro regime features. Cache-aware."""
        now = datetime.now(UTC).replace(tzinfo=None)

        if self._cache is not None:
            cached_at, cached_feats = self._cache
            if (now - cached_at).total_seconds() < self.cache_ttl * 60:
                return cached_feats

        if not _HAS_YF:
            return dict(_DEFAULT_FEATURES)

        features = dict(_DEFAULT_FEATURES)
        raw: dict[str, dict[str, float]] = {}

        # Fetch all macro tickers in one batch
        try:
            tickers = list(_MACRO_TICKERS.keys())
            data = yf.download(tickers, period="22d", progress=False, group_by="ticker")

            for yf_ticker, feat_name in _MACRO_TICKERS.items():
                try:
                    if len(_MACRO_TICKERS) > 1 and yf_ticker in data.columns.get_level_values(0):
                        series = data[yf_ticker]["Close"].dropna()
                    else:
                        series = data["Close"].dropna()

                    if series.empty:
                        continue

                    current = float(series.iloc[-1])
                    prev_5d = float(series.iloc[-6]) if len(series) >= 6 else current
                    prev_20d = float(series.iloc[0]) if len(series) >= 15 else current

                    raw[feat_name] = {
                        "current": current,
                        "ret_5d": (current - prev_5d) / max(prev_5d, 0.01),
                        "ret_20d": (current - prev_20d) / max(prev_20d, 0.01),
                    }

                    features[f"macro_{feat_name}"] = current
                    features[f"macro_{feat_name}_ret_5d"] = raw[feat_name]["ret_5d"]

                except Exception as exc:
                    logger.debug("Macro feature %s failed: %s", feat_name, exc)

        except Exception as exc:
            logger.warning("Macro batch download failed: %s", exc)
            return dict(_DEFAULT_FEATURES)

        # Derived features
        # Yield curve spread (10Y - 3M) — inversion signals recession risk
        us10y = raw.get("us10y", {}).get("current", 4.0)
        us3m = raw.get("us3m", {}).get("current", 4.5)
        features["macro_yield_curve_spread"] = us10y - us3m

        # Risk-off regime classifier (simple rule-based)
        vix = raw.get("vix", {}).get("current", 18.0)
        dxy_ret = raw.get("dxy", {}).get("ret_5d", 0.0)
        yc_spread = features["macro_yield_curve_spread"]

        # Risk-off when: VIX > 22, OR dollar strengthening + yield curve inverted
        risk_off = 0.0
        if vix > 25:
            risk_off = 1.0
        elif vix > 20 and dxy_ret > 0.005:
            risk_off = 0.7
        elif yc_spread < -0.5:
            risk_off = 0.5
        elif vix > 18 and dxy_ret > 0.01:
            risk_off = 0.4
        features["macro_risk_off_regime"] = risk_off

        # Risk-on boost: low VIX + contango environment + gold falling
        gold_ret = raw.get("gold", {}).get("ret_5d", 0.0)
        if vix < 15 and gold_ret < -0.01:
            features["macro_risk_on_boost"] = 1.0
        else:
            features["macro_risk_on_boost"] = 0.0

        self._cache = (now, features)
        return features

    def get_regime_label(self) -> str:
        """Return a human-readable regime label."""
        feats = self.get_features()
        risk_off = feats.get("macro_risk_off_regime", 0.0)
        if risk_off >= 0.7:
            return "risk_off"
        elif risk_off >= 0.4:
            return "cautious"
        elif feats.get("macro_risk_on_boost", 0.0) > 0:
            return "risk_on"
        else:
            return "neutral"
