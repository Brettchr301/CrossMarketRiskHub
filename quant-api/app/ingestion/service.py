from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, UTC
from typing import Sequence

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import models
from app.ingestion.archive import write_raw_archive
from app.providers import (
    FreeCommodityProvider,
    FreeEquityProvider,
    FreeOptionsProvider,
    FreeShippingProvider,
    KalshiProvider,
    PolymarketProvider,
    RealCommodityProvider,
    RealEquityProvider,
    RealOptionsProvider,
    RealPredictionProvider,
    RealShippingProvider,
)


EVENT_SET = [
    "hormuz_closure",
    "red_sea_disruption",
    "sanctions_escalation",
    "oil_above_100",
    "opec_production_cut",
    "panama_canal_disruption",
    "china_stimulus",
    "us_spr_release",
    "us_refinery_utilization_low",
    # Commodity events
    "gold_above_3000",
    "silver_above_40",
    "copper_above_5",
    "iron_ore_above_150",
    "us_tariff_escalation",
    "china_property_crisis",
    "rare_earth_export_ban",
    "nuclear_renaissance",
    "ev_adoption_milestone",
    "lithium_oversupply",
    "potash_sanctions",
    "food_crisis",
    "carbon_price_above_100",
    "india_infrastructure_boom",
    # Tail-risk geopolitical / policy events (high alpha)
    "taiwan_strait_crisis",
    "russia_ukraine_ceasefire",
    "south_africa_grid_crisis",
    "chile_lithium_nationalization",
    "indonesia_nickel_ban",
    "us_permitting_reform",
    "eu_cbam_implementation",
    "australia_china_trade_thaw",
    "us_recession",
    "middle_east_war_escalation",
]
COMMODITIES = ["BRENT", "WTI"]
SHIPPING_INDICES = ["BDI", "TD3", "BCTI"]


class MarketIngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.real_data_only:
            self.prediction = RealPredictionProvider()
            self.commodity = RealCommodityProvider()
            self.shipping = RealShippingProvider()
            self.equity = RealEquityProvider()
            self.options = RealOptionsProvider()
        else:
            self.polymarket = PolymarketProvider()
            self.kalshi = KalshiProvider()
            self.commodity = FreeCommodityProvider()
            self.shipping = FreeShippingProvider()
            self.equity = FreeEquityProvider()
            self.options = FreeOptionsProvider()

    def ingest_all(self, db: Session, tickers: Sequence[str]) -> dict[str, int]:
        counts = {
            "prediction_quotes": self.ingest_prediction_quotes(db),
            "commodity_quotes": self.ingest_commodity_quotes(db),
            "shipping_indices": self.ingest_shipping_quotes(db),
            "equity_quotes": self.ingest_equity_quotes(db, tickers),
            "options_chain_eod": self.ingest_options_quotes(db, tickers),
        }
        db.commit()
        return counts

    def ingest_prediction_quotes(self, db: Session) -> int:
        if self.settings.real_data_only:
            rows = list(self.prediction.fetch_event_quotes(EVENT_SET)) if self.prediction else []
        else:
            rows = list(self.polymarket.fetch_event_quotes(EVENT_SET)) + list(
                self.kalshi.fetch_event_quotes(EVENT_SET)
            )
        for row in rows:
            db.add(
                models.PredictionQuote(
                    provider=row.provider,
                    event_id=row.event_id,
                    mid_price=row.mid_price,
                    bid=row.bid,
                    ask=row.ask,
                    volume=row.volume,
                    liquidity_score=row.liquidity_score,
                    as_of=row.as_of,
                    received_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )
        write_raw_archive(
            self.settings.raw_archive_path,
            "prediction_quotes",
            [asdict(x) | {"mid_price": x.mid_price} for x in rows],
        )
        if self.settings.real_data_only and not rows:
            raise RuntimeError("Real-data mode enabled but no prediction market quotes were retrieved.")
        return len(rows)

    def ingest_commodity_quotes(self, db: Session) -> int:
        rows = list(self.commodity.fetch_commodity_quotes(COMMODITIES))
        for row in rows:
            db.add(models.CommodityQuote(symbol=row.symbol, price=row.price, as_of=row.as_of))
        write_raw_archive(self.settings.raw_archive_path, "commodity_quotes", [asdict(x) for x in rows])
        if self.settings.real_data_only and not rows:
            raise RuntimeError("Real-data mode enabled but no commodity quotes were retrieved.")
        return len(rows)

    def ingest_shipping_quotes(self, db: Session) -> int:
        rows = list(self.shipping.fetch_shipping_quotes(SHIPPING_INDICES))
        for row in rows:
            db.add(models.ShippingIndexQuote(index_name=row.index_name, value=row.value, as_of=row.as_of))
        write_raw_archive(self.settings.raw_archive_path, "shipping_indices", [asdict(x) for x in rows])
        if self.settings.real_data_only and not rows:
            raise RuntimeError("Real-data mode enabled but no shipping proxy quotes were retrieved.")
        return len(rows)

    def ingest_equity_quotes(self, db: Session, tickers: Sequence[str]) -> int:
        rows = list(self.equity.fetch_equity_quotes(tickers))
        for row in rows:
            db.add(
                models.EquityQuote(
                    ticker=row.ticker,
                    close_price=row.close_price,
                    volume=row.volume,
                    as_of=row.as_of,
                )
            )
        write_raw_archive(self.settings.raw_archive_path, "equity_quotes", [asdict(x) for x in rows])
        if self.settings.real_data_only and not rows:
            raise RuntimeError("Real-data mode enabled but no equity quotes were retrieved.")
        return len(rows)

    def ingest_options_quotes(self, db: Session, tickers: Sequence[str]) -> int:
        inserted = 0
        options_records: list[dict[str, object]] = []
        for ticker in tickers:
            spot = next(
                (
                    row.close_price
                    for row in self.equity.fetch_equity_quotes([ticker])
                    if row.ticker == ticker.upper()
                ),
                15.0,
            )
            rows = self.options.fetch_options_chain(ticker, spot)
            for row in rows:
                inserted += 1
                db.add(
                    models.OptionChainQuote(
                        ticker=row.ticker,
                        expiration=row.expiration,
                        strike=row.strike,
                        option_type=row.option_type,
                        bid=row.bid,
                        ask=row.ask,
                        implied_vol=row.implied_vol,
                        open_interest=row.open_interest,
                        as_of=row.as_of,
                    )
                )
                options_records.append(asdict(row))
        write_raw_archive(self.settings.raw_archive_path, "options_chain_eod", options_records)
        if self.settings.real_data_only and inserted == 0:
            raise RuntimeError("Real-data mode enabled but no option-chain rows were retrieved.")
        return inserted
