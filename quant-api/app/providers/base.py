from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Sequence


@dataclass(slots=True)
class PredictionQuoteRow:
    provider: str
    event_id: str
    bid: float
    ask: float
    volume: float
    liquidity_score: float
    as_of: datetime

    @property
    def mid_price(self) -> float:
        return max(0.0, min(1.0, 0.5 * (self.bid + self.ask)))


@dataclass(slots=True)
class PriceQuoteRow:
    symbol: str
    price: float
    as_of: datetime


@dataclass(slots=True)
class ShippingQuoteRow:
    index_name: str
    value: float
    as_of: datetime


@dataclass(slots=True)
class EquityQuoteRow:
    ticker: str
    close_price: float
    volume: float
    as_of: datetime


@dataclass(slots=True)
class OptionQuoteRow:
    ticker: str
    expiration: date
    strike: float
    option_type: str
    bid: float
    ask: float
    implied_vol: float
    open_interest: float
    as_of: datetime


class PredictionProvider(Protocol):
    def fetch_event_quotes(self, events: Sequence[str]) -> Sequence[PredictionQuoteRow]:
        ...


class CommodityProvider(Protocol):
    def fetch_commodity_quotes(self, symbols: Sequence[str]) -> Sequence[PriceQuoteRow]:
        ...


class ShippingProvider(Protocol):
    def fetch_shipping_quotes(self, indices: Sequence[str]) -> Sequence[ShippingQuoteRow]:
        ...


class EquityProvider(Protocol):
    def fetch_equity_quotes(self, tickers: Sequence[str]) -> Sequence[EquityQuoteRow]:
        ...


class OptionsProvider(Protocol):
    def fetch_options_chain(self, ticker: str, spot_price: float) -> Sequence[OptionQuoteRow]:
        ...

