from __future__ import annotations

from datetime import date, datetime, UTC
from typing import Sequence

import pandas as pd
import yfinance as yf

from app.providers.base import EquityQuoteRow, OptionQuoteRow, PriceQuoteRow, ShippingQuoteRow


def _latest_close_map(tickers: Sequence[str], period: str = "15d") -> dict[str, tuple[float, datetime, float]]:
    if not tickers:
        return {}
    frame = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )
    out: dict[str, tuple[float, datetime, float]] = {}
    if frame.empty:
        return out

    if isinstance(frame.columns, pd.MultiIndex):
        for ticker in tickers:
            t = ticker.upper()
            if t not in frame.columns.get_level_values(0):
                continue
            sub = frame[t].dropna()
            if sub.empty:
                continue
            row = sub.iloc[-1]
            as_of = pd.Timestamp(sub.index[-1]).to_pydatetime().replace(tzinfo=None)
            close = float(row.get("Close", row.get("Adj Close", 0.0)))
            vol = float(row.get("Volume", 0.0))
            out[t] = (close, as_of, vol)
    else:
        sub = frame.dropna()
        if not sub.empty and len(tickers) == 1:
            row = sub.iloc[-1]
            as_of = pd.Timestamp(sub.index[-1]).to_pydatetime().replace(tzinfo=None)
            t = tickers[0].upper()
            close = float(row.get("Close", row.get("Adj Close", 0.0)))
            vol = float(row.get("Volume", 0.0))
            out[t] = (close, as_of, vol)
    return out


class RealCommodityProvider:
    ticker_map = {"BRENT": "BZ=F", "WTI": "CL=F"}

    def fetch_commodity_quotes(self, symbols: Sequence[str]) -> Sequence[PriceQuoteRow]:
        yf_tickers = [self.ticker_map.get(s.upper(), s) for s in symbols]
        latest = _latest_close_map(yf_tickers, period="20d")
        rows: list[PriceQuoteRow] = []
        for symbol in symbols:
            yf_symbol = self.ticker_map.get(symbol.upper(), symbol.upper())
            data = latest.get(yf_symbol.upper())
            if data is None:
                continue
            price, as_of, _ = data
            rows.append(PriceQuoteRow(symbol=symbol.upper(), price=price, as_of=as_of))
        return rows


class RealShippingProvider:
    proxy_map = {"BDI": "BDRY", "TD3": "BOAT", "BCTI": "SEA"}

    def fetch_shipping_quotes(self, indices: Sequence[str]) -> Sequence[ShippingQuoteRow]:
        yf_tickers = [self.proxy_map.get(i.upper(), i.upper()) for i in indices]
        latest = _latest_close_map(yf_tickers, period="20d")
        rows: list[ShippingQuoteRow] = []
        for idx in indices:
            yf_symbol = self.proxy_map.get(idx.upper(), idx.upper())
            data = latest.get(yf_symbol.upper())
            if data is None:
                continue
            value, as_of, _ = data
            rows.append(ShippingQuoteRow(index_name=idx.upper(), value=value, as_of=as_of))
        return rows


class RealEquityProvider:
    def fetch_equity_quotes(self, tickers: Sequence[str]) -> Sequence[EquityQuoteRow]:
        latest = _latest_close_map([t.upper() for t in tickers], period="20d")
        rows: list[EquityQuoteRow] = []
        for ticker in tickers:
            data = latest.get(ticker.upper())
            if data is None:
                continue
            close, as_of, vol = data
            rows.append(
                EquityQuoteRow(
                    ticker=ticker.upper(),
                    close_price=close,
                    volume=vol,
                    as_of=as_of,
                )
            )
        return rows


class RealOptionsProvider:
    def fetch_options_chain(self, ticker: str, spot_price: float) -> Sequence[OptionQuoteRow]:
        t = yf.Ticker(ticker.upper())
        expirations = list(t.options or [])
        if not expirations:
            return []
        expiry = expirations[0]
        chain = t.option_chain(expiry)
        now = datetime.now(UTC).replace(tzinfo=None)
        rows: list[OptionQuoteRow] = []

        def _convert(frame: pd.DataFrame, option_type: str):
            if frame is None or frame.empty:
                return
            for _, row in frame.iterrows():
                bid = float(row.get("bid", 0.0))
                ask = float(row.get("ask", 0.0))
                iv = float(row.get("impliedVolatility", 0.0))
                strike = float(row.get("strike", 0.0))
                oi = float(row.get("openInterest", 0.0))
                if strike <= 0 or (bid <= 0 and ask <= 0):
                    continue
                rows.append(
                    OptionQuoteRow(
                        ticker=ticker.upper(),
                        expiration=date.fromisoformat(expiry),
                        strike=strike,
                        option_type=option_type,
                        bid=max(0.01, bid),
                        ask=max(0.01, ask),
                        implied_vol=max(0.01, iv),
                        open_interest=max(0.0, oi),
                        as_of=now,
                    )
                )

        _convert(chain.calls, "call")
        _convert(chain.puts, "put")
        return rows

