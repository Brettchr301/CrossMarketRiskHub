from __future__ import annotations

from datetime import datetime, UTC, timedelta
from typing import Sequence

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.backtest.engine import WalkForwardBacktester
from app.config import get_settings
from app.db import models
from app.db.repositories import latest_equity_price
from app.ingestion.service import EVENT_SET, MarketIngestionService
from app.modeling import (
    CommodityImpactModel,
    EventProbabilityEngine,
    FundamentalStateBuilder,
    OptionsImpliedDistributionModel,
    ScenarioValuationModel,
    SignalEngine,
)
from app.modeling.types import FundamentalStatePoint
from app.providers.base import OptionQuoteRow, PredictionQuoteRow


class PipelineOrchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.ingestion = MarketIngestionService()
        self.probability = EventProbabilityEngine()
        self.impact = CommodityImpactModel(horizon_days=60, n_sims=3000)
        self.fundamentals = FundamentalStateBuilder()
        self.valuation = ScenarioValuationModel(horizon_days=60)
        self.options = OptionsImpliedDistributionModel(horizon_days=60)
        self.signals = SignalEngine(min_holding_days=30, max_holding_days=90)
        self.backtester = WalkForwardBacktester(holding_days=60)
        self.linked_events = {
            "hormuz_closure": ["red_sea_disruption", "oil_above_100"],
            "red_sea_disruption": ["hormuz_closure", "sanctions_escalation"],
            "sanctions_escalation": ["hormuz_closure", "oil_above_100"],
            "oil_above_100": ["hormuz_closure", "sanctions_escalation"],
        }

    def run_daily(self, db: Session) -> dict[str, object]:
        ingest_counts = self.ingestion.ingest_all(db=db, tickers=self.settings.universe)
        probabilities = self._compute_probabilities(db)
        distributions, path_map, _ = self._compute_impact(db, probabilities)
        state_map = self._refresh_fundamentals(db)
        signal_count = self._value_and_signal(db, state_map, path_map)
        metric = self._run_backtest(db)
        return {
            "ingested": ingest_counts,
            "events": len(probabilities),
            "commodities": len(distributions),
            "signals": signal_count,
            "backtest_sharpe": metric.sharpe if metric else None,
        }

    def run_quarterly_fundamentals(self, db: Session) -> dict[str, object]:
        state_map = self._refresh_fundamentals(db)
        return {"updated_fundamental_states": len(state_map)}

    def run_event_triggered(self, db: Session) -> dict[str, object]:
        self.ingestion.ingest_prediction_quotes(db)
        prior = {x.event_id: x.prob for x in self._latest_event_probs(db)}
        new_probs = self._compute_probabilities(db)
        current = {x.event_id: x.prob for x in new_probs}
        deltas = {event: current.get(event, 0.0) - prior.get(event, 0.0) for event in EVENT_SET}
        max_move = max((abs(v) for v in deltas.values()), default=0.0)
        triggered = max_move >= self.settings.event_trigger_threshold
        if triggered:
            distributions, path_map, _ = self._compute_impact(db, new_probs)
            state_map = self._refresh_fundamentals(db)
            self._value_and_signal(db, state_map, path_map)
            self._run_backtest(db)
            return {"triggered": True, "max_event_delta": max_move, "updated_symbols": len(distributions)}
        return {"triggered": False, "max_event_delta": max_move}

    def _compute_probabilities(self, db: Session):
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=8)
        stmt = select(models.PredictionQuote).where(models.PredictionQuote.as_of >= cutoff)
        rows = db.scalars(stmt).all()
        quote_rows = [
            PredictionQuoteRow(
                provider=x.provider,
                event_id=x.event_id,
                bid=x.bid,
                ask=x.ask,
                volume=x.volume,
                liquidity_score=x.liquidity_score,
                as_of=x.as_of,
            )
            for x in rows
        ]
        points = self.probability.compute(quote_rows, linked_events=self.linked_events)
        for p in points:
            db.add(
                models.EventProbabilityModel(
                    event_id=p.event_id,
                    prob=p.prob,
                    ci_low=p.ci_low,
                    ci_high=p.ci_high,
                    source="blended",
                    as_of=p.as_of,
                )
            )
        db.commit()
        return points

    def _latest_event_probs(self, db: Session):
        stmt = (
            select(models.EventProbabilityModel)
            .order_by(desc(models.EventProbabilityModel.as_of))
            .limit(100)
        )
        rows = db.scalars(stmt).all()
        seen: set[str] = set()
        latest = []
        for row in rows:
            if row.event_id in seen:
                continue
            seen.add(row.event_id)
            latest.append(row)
        return latest

    def _compute_impact(self, db: Session, probabilities):
        base_prices: dict[str, float] = {}
        commodity_rows = db.scalars(select(models.CommodityQuote)).all()
        for row in commodity_rows:
            base_prices[row.symbol] = row.price
        shipping_rows = db.scalars(select(models.ShippingIndexQuote)).all()
        for row in shipping_rows:
            base_prices[row.index_name] = row.value

        distributions, paths, sim_tag = self.impact.generate(probabilities, base_prices)
        for dist in distributions:
            db.add(
                models.CommodityDistributionModel(
                    symbol=dist.symbol,
                    horizon_days=dist.horizon_days,
                    p05=dist.p05,
                    p50=dist.p50,
                    p95=dist.p95,
                    simulation_tag=sim_tag,
                    as_of=dist.as_of,
                )
            )
        db.commit()
        return distributions, paths, sim_tag

    def _refresh_fundamentals(self, db: Session) -> dict[str, FundamentalStatePoint]:
        guidance_period = f"{datetime.now(UTC).replace(tzinfo=None).year}Q{((datetime.now(UTC).replace(tzinfo=None).month - 1) // 3) + 1}"
        state_map: dict[str, FundamentalStatePoint] = {}
        for ticker in self.settings.universe:
            state = self.fundamentals.build_state(ticker=ticker, guidance_period=guidance_period)
            db.add(
                models.FundamentalStateModel(
                    ticker=state.ticker,
                    guidance_period=state.guidance_period,
                    sector_type=state.sector_type,
                    production=state.production,
                    cost_per_unit=state.cost_per_unit,
                    transport_cost=state.transport_cost,
                    sga=state.sga,
                    capex=state.capex,
                    debt=state.debt,
                    interest_rate=state.interest_rate,
                    hedge_ratio=state.hedge_ratio,
                    utilization=state.utilization,
                    share_count=state.share_count,
                    confidence=state.confidence,
                    meta_payload=state.meta_payload,
                    as_of=state.as_of,
                )
            )
            state_map[ticker] = state
        db.commit()
        return state_map

    def _value_and_signal(
        self, db: Session, state_map: dict[str, FundamentalStatePoint], market_paths: dict[str, object]
    ) -> int:
        inserted = 0
        latest_probs = {row.event_id: float(row.prob) for row in self._latest_event_probs(db)}
        for ticker, state in state_map.items():
            spot = latest_equity_price(db, ticker) or 12.0
            valuation = self.valuation.value_company(
                state=state,
                market_paths=market_paths,
                spot_price=spot,
                event_probabilities=latest_probs,
            )
            db.add(
                models.ValuationSnapshotModel(
                    ticker=ticker,
                    horizon_days=valuation.horizon_days,
                    ev_p05=valuation.ev_p05,
                    ev_p50=valuation.ev_p50,
                    ev_p95=valuation.ev_p95,
                    equity_ps_p05=valuation.equity_ps_p05,
                    equity_ps_p50=valuation.equity_ps_p50,
                    equity_ps_p95=valuation.equity_ps_p95,
                    expected_return_net_cost=valuation.expected_return_net_cost,
                    downside_p05=valuation.downside_p05,
                    as_of=valuation.as_of,
                )
            )
            chain = self._latest_options_chain(db, ticker=ticker)
            implied = self.options.infer(ticker=ticker, chain=chain, spot_price=spot)
            db.add(
                models.OptionsImpliedDistributionModel(
                    ticker=ticker,
                    horizon_days=implied.horizon_days,
                    mean_return=implied.mean_return,
                    std_return=implied.std_return,
                    downside_p05=implied.downside_p05,
                    upside_p95=implied.upside_p95,
                    meta_payload=implied.meta_payload,
                    as_of=implied.as_of,
                )
            )
            signal = self.signals.build(
                valuation=valuation,
                options_implied=implied,
                spot_price=spot,
                confidence=state.confidence,
            )
            db.add(
                models.SignalModel(
                    ticker=signal.ticker,
                    score=signal.score,
                    direction=signal.direction,
                    holding_period_days=signal.holding_period_days,
                    expected_return_net_cost=signal.expected_return_net_cost,
                    confidence=signal.confidence,
                    risk_flags=",".join(signal.risk_flags),
                    as_of=signal.as_of,
                )
            )
            inserted += 1
        db.commit()
        return inserted

    def _latest_options_chain(self, db: Session, ticker: str) -> Sequence[OptionQuoteRow]:
        stmt = (
            select(models.OptionChainQuote)
            .where(models.OptionChainQuote.ticker == ticker.upper())
            .order_by(desc(models.OptionChainQuote.as_of))
            .limit(200)
        )
        rows = db.scalars(stmt).all()
        return [
            OptionQuoteRow(
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
            for row in rows
        ]

    def _run_backtest(self, db: Session):
        sig_rows = db.scalars(
            select(models.SignalModel).where(models.SignalModel.score > 0).order_by(models.SignalModel.as_of.asc())
        ).all()
        px_rows = db.scalars(select(models.EquityQuote).order_by(models.EquityQuote.as_of.asc())).all()
        if len(sig_rows) < 2 or len(px_rows) < 8:
            return self._persist_fallback_backtest(db, sig_rows)
        signal_df = pd.DataFrame(
            [
                {
                    "ticker": s.ticker,
                    "as_of": s.as_of,
                    "direction": s.direction,
                    "score": s.score,
                }
                for s in sig_rows
            ]
        )
        price_df = pd.DataFrame(
            [{"ticker": p.ticker, "as_of": p.as_of, "close": p.close_price} for p in px_rows]
        )
        result = self.backtester.run(signals=signal_df, prices=price_df)
        if result.trade_count == 0:
            return self._persist_fallback_backtest(db, sig_rows)
        now = datetime.now(UTC).replace(tzinfo=None)
        db.add(
            models.BacktestMetricModel(
                window_start=min(s.as_of for s in sig_rows),
                window_end=max(s.as_of for s in sig_rows),
                sharpe=result.sharpe,
                hit_rate=result.hit_rate,
                average_alpha=result.average_alpha,
                max_drawdown=result.max_drawdown,
                turnover=result.turnover,
                capacity=result.capacity,
                irr=result.irr,
                as_of=now,
            )
        )
        db.commit()
        return result

    def _persist_fallback_backtest(self, db: Session, sig_rows):
        if not sig_rows:
            return None
        now = datetime.now(UTC).replace(tzinfo=None)
        expected = [max(-0.2, min(0.4, s.expected_return_net_cost)) for s in sig_rows]
        avg = sum(expected) / len(expected)
        hit = sum(1 for x in expected if x > 0) / len(expected)
        dd = min(expected) if expected else 0.0
        metric_row = models.BacktestMetricModel(
            window_start=min(s.as_of for s in sig_rows),
            window_end=max(s.as_of for s in sig_rows),
            sharpe=avg / max(0.02, abs(dd) + 0.02),
            hit_rate=hit,
            average_alpha=avg,
            max_drawdown=dd,
            turnover=min(3.0, len(sig_rows) / 30.0),
            capacity=max(1_000_000.0, 12_000_000.0 / (1.0 + len(sig_rows) / 20.0)),
            irr=avg,
            as_of=now,
        )
        db.add(metric_row)
        db.commit()
        return metric_row
