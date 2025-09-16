from __future__ import annotations

from datetime import date, datetime, UTC, timedelta
from hashlib import sha256
from math import exp
from typing import Sequence

import numpy as np

from app.providers.base import EquityQuoteRow, OptionQuoteRow, PriceQuoteRow, ShippingQuoteRow


def _stable_seed(token: str) -> int:
    return int(sha256(token.encode("utf-8")).hexdigest()[:8], 16)


class FreeCommodityProvider:
    def fetch_commodity_quotes(self, symbols: Sequence[str]) -> Sequence[PriceQuoteRow]:
        now = datetime.now(UTC).replace(tzinfo=None)
        rows: list[PriceQuoteRow] = []
        for symbol in symbols:
            seed = _stable_seed(f"commodity:{symbol}:{now.date().isoformat()}")
            base = {"BRENT": 84.0, "WTI": 80.0}.get(symbol.upper(), 50.0)
            drift = ((seed % 1000) - 500) / 1000.0
            rows.append(PriceQuoteRow(symbol=symbol.upper(), price=max(1.0, base + drift), as_of=now))
        return rows


class FreeShippingProvider:
    def fetch_shipping_quotes(self, indices: Sequence[str]) -> Sequence[ShippingQuoteRow]:
        now = datetime.now(UTC).replace(tzinfo=None)
        rows: list[ShippingQuoteRow] = []
        bases = {"BDI": 1700.0, "TD3": 82.0, "BCTI": 710.0}
        for idx in indices:
            seed = _stable_seed(f"shipping:{idx}:{now.date().isoformat()}")
            base = bases.get(idx.upper(), 100.0)
            bump = ((seed % 1200) - 600) / 30.0
            rows.append(ShippingQuoteRow(index_name=idx.upper(), value=max(1.0, base + bump), as_of=now))
        return rows


class FreeEquityProvider:
    def fetch_equity_quotes(self, tickers: Sequence[str]) -> Sequence[EquityQuoteRow]:
        now = datetime.now(UTC).replace(tzinfo=None)
        rows: list[EquityQuoteRow] = []
        for ticker in tickers:
            seed = _stable_seed(f"equity:{ticker}:{now.date().isoformat()}")
            base = 8.0 + (seed % 900) / 20.0
            volume = 100_000 + (seed % 900_000)
            rows.append(
                EquityQuoteRow(
                    ticker=ticker.upper(),
                    close_price=max(1.0, round(base, 2)),
                    volume=float(volume),
                    as_of=now,
                )
            )
        return rows


class FreeOptionsProvider:
    def fetch_options_chain(self, ticker: str, spot_price: float) -> Sequence[OptionQuoteRow]:
        now = datetime.now(UTC).replace(tzinfo=None)
        expiry = date.today() + timedelta(days=60)
        rows: list[OptionQuoteRow] = []
        strikes = np.linspace(0.7 * spot_price, 1.3 * spot_price, 11)
        seed = _stable_seed(f"options:{ticker}:{date.today().isoformat()}")
        rng = np.random.default_rng(seed)
        atm_vol = 0.35 + 0.1 * rng.random()
        for strike in strikes:
            moneyness = strike / max(spot_price, 0.01)
            smile = atm_vol * (1 + 0.35 * abs(moneyness - 1.0))
            t = 60.0 / 365.0
            sigma_sqrt_t = smile * (t ** 0.5)
            mid = spot_price * 0.04 * exp(-abs(moneyness - 1.0) * 2.0) + sigma_sqrt_t * spot_price * 0.08
            spread = max(0.01, mid * 0.12)
            oi = max(20.0, 1500.0 * exp(-abs(moneyness - 1.0) * 2.5))
            rows.append(
                OptionQuoteRow(
                    ticker=ticker.upper(),
                    expiration=expiry,
                    strike=float(round(strike, 2)),
                    option_type="call",
                    bid=float(max(0.01, mid - spread / 2)),
                    ask=float(max(0.02, mid + spread / 2)),
                    implied_vol=float(smile),
                    open_interest=float(oi),
                    as_of=now,
                )
            )
            rows.append(
                OptionQuoteRow(
                    ticker=ticker.upper(),
                    expiration=expiry,
                    strike=float(round(strike, 2)),
                    option_type="put",
                    bid=float(max(0.01, mid - spread / 2)),
                    ask=float(max(0.02, mid + spread / 2)),
                    implied_vol=float(smile * (1.0 + 0.03 * (1.0 - moneyness))),
                    open_interest=float(oi * (1.0 + 0.2 * abs(1.0 - moneyness))),
                    as_of=now,
                )
            )
        return rows

