"""Election prediction market API routes."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.election.db.models import (
    AlphaModelPrediction,
    ArbitrageOpportunity,
    BlendedProbability,
    CampaignFinance,
    Candidate,
    MarketContract,
    MarketQuote,
    PollingData,
    Race,
)
from app.election.db.session import get_election_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/election", tags=["election"])


@router.get("/races")
def list_races(db: Session = Depends(get_election_db)) -> list[dict[str, Any]]:
    """List all tracked races with latest blended probabilities."""
    races = db.execute(select(Race).order_by(Race.cycle, Race.state)).scalars().all()
    result = []
    for r in races:
        # Latest blended probability
        bp = db.execute(
            select(BlendedProbability)
            .where(BlendedProbability.race_id == r.id)
            .order_by(BlendedProbability.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()

        result.append({
            "id": r.id,
            "race_type": r.race_type,
            "state": r.state,
            "district": r.district,
            "cycle": r.cycle,
            "election_date": str(r.election_date),
            "latest_prob": bp.prob if bp else None,
            "ci_low": bp.ci_low if bp else None,
            "ci_high": bp.ci_high if bp else None,
            "n_platforms": bp.n_platforms if bp else 0,
        })
    return result


@router.get("/quotes/{race_id}")
def get_quotes(race_id: int, db: Session = Depends(get_election_db)) -> list[dict[str, Any]]:
    """Latest quotes from all platforms for a race."""
    contracts = db.execute(
        select(MarketContract).where(MarketContract.race_id == race_id)
    ).scalars().all()

    if not contracts:
        raise HTTPException(404, f"No contracts found for race {race_id}")

    result = []
    for c in contracts:
        latest = db.execute(
            select(MarketQuote)
            .where(MarketQuote.contract_id == c.id)
            .order_by(MarketQuote.as_of.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest:
            result.append({
                "contract_id": c.id,
                "platform": c.platform,
                "platform_market_id": c.platform_market_id,
                "question": c.platform_question,
                "yes_bid": latest.yes_bid,
                "yes_ask": latest.yes_ask,
                "last_price": latest.last_price,
                "volume_24h": latest.volume_24h,
                "liquidity_score": latest.liquidity_score,
                "as_of": str(latest.as_of),
            })
    return result


@router.get("/arbitrage")
def get_arbitrage(
    min_edge: float = 0.0,
    limit: int = 50,
    db: Session = Depends(get_election_db),
) -> list[dict[str, Any]]:
    """Active arbitrage opportunities sorted by net edge."""
    arbs = db.execute(
        select(ArbitrageOpportunity)
        .where(
            ArbitrageOpportunity.status == "active",
            ArbitrageOpportunity.net_edge_pct >= min_edge,
        )
        .order_by(ArbitrageOpportunity.net_edge_pct.desc())
        .limit(limit)
    ).scalars().all()

    return [
        {
            "id": a.id,
            "arb_type": a.arb_type,
            "race_id": a.race_id,
            "description": a.description,
            "gross_edge_pct": a.gross_edge_pct,
            "net_edge_pct": a.net_edge_pct,
            "buy_platform": a.buy_platform,
            "buy_price": a.buy_price,
            "sell_platform": a.sell_platform,
            "sell_price": a.sell_price,
            "confidence": a.confidence,
            "detected_at": str(a.detected_at),
            "status": a.status,
        }
        for a in arbs
    ]


@router.get("/alpha-signals")
def get_alpha_signals(
    min_confidence: float = 0.0,
    limit: int = 50,
    db: Session = Depends(get_election_db),
) -> list[dict[str, Any]]:
    """Latest alpha model predictions with confidence."""
    preds = db.execute(
        select(AlphaModelPrediction)
        .where(AlphaModelPrediction.confidence >= min_confidence)
        .order_by(AlphaModelPrediction.as_of.desc())
        .limit(limit)
    ).scalars().all()

    return [
        {
            "race_id": p.race_id,
            "predicted_prob_change": p.predicted_prob_change,
            "confidence": p.confidence,
            "top_features": p.top_features,
            "model_version": p.model_version,
            "as_of": str(p.as_of),
        }
        for p in preds
    ]


@router.get("/polling/{race_id}")
def get_polling(race_id: int, db: Session = Depends(get_election_db)) -> list[dict[str, Any]]:
    """Polling history for a race."""
    polls = db.execute(
        select(PollingData)
        .where(PollingData.race_id == race_id)
        .order_by(PollingData.poll_date.desc())
        .limit(100)
    ).scalars().all()

    return [
        {
            "pollster": p.pollster,
            "sample_size": p.sample_size,
            "candidate_id": p.candidate_id,
            "pct": p.pct,
            "poll_date": str(p.poll_date),
            "source": p.source,
        }
        for p in polls
    ]


@router.get("/finance/{candidate_id}")
def get_finance(candidate_id: int, db: Session = Depends(get_election_db)) -> list[dict[str, Any]]:
    """Campaign finance data for a candidate."""
    records = db.execute(
        select(CampaignFinance)
        .where(CampaignFinance.candidate_id == candidate_id)
        .order_by(CampaignFinance.period_end.desc())
    ).scalars().all()

    return [
        {
            "period_start": str(r.period_start),
            "period_end": str(r.period_end),
            "receipts": r.receipts,
            "disbursements": r.disbursements,
            "cash_on_hand": r.cash_on_hand,
            "individual_contributions": r.individual_contributions,
            "pac_contributions": r.pac_contributions,
        }
        for r in records
    ]


@router.get("/dashboard")
def get_dashboard(db: Session = Depends(get_election_db)) -> dict[str, Any]:
    """Aggregate dashboard payload."""
    race_count = db.execute(select(func.count(Race.id))).scalar() or 0
    contract_count = db.execute(select(func.count(MarketContract.id))).scalar() or 0
    quote_count = db.execute(select(func.count(MarketQuote.id))).scalar() or 0

    # Active arbs
    active_arbs = db.execute(
        select(ArbitrageOpportunity)
        .where(ArbitrageOpportunity.status == "active")
        .order_by(ArbitrageOpportunity.net_edge_pct.desc())
        .limit(10)
    ).scalars().all()

    # Latest alpha signals
    latest_alpha = db.execute(
        select(AlphaModelPrediction)
        .order_by(AlphaModelPrediction.as_of.desc())
        .limit(10)
    ).scalars().all()

    return {
        "summary": {
            "races_tracked": race_count,
            "contracts_tracked": contract_count,
            "total_quotes": quote_count,
            "active_arbs": len(active_arbs),
        },
        "top_arbs": [
            {
                "arb_type": a.arb_type,
                "description": a.description,
                "net_edge_pct": a.net_edge_pct,
                "confidence": a.confidence,
            }
            for a in active_arbs
        ],
        "alpha_signals": [
            {
                "race_id": p.race_id,
                "predicted_prob_change": p.predicted_prob_change,
                "confidence": p.confidence,
            }
            for p in latest_alpha
        ],
    }


@router.post("/pipeline/run")
def trigger_pipeline() -> dict[str, str]:
    """Manually trigger a pipeline run."""
    import threading
    from app.election.pipeline.orchestrator import run_full_pipeline

    threading.Thread(target=run_full_pipeline, daemon=True).start()
    return {"status": "pipeline triggered"}


@router.get("/backtest/runs")
def list_backtest_runs(db: Session = Depends(get_election_db)) -> list[dict]:
    from app.election.db.historical_models import BacktestRun
    runs = db.execute(select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(50)).scalars().all()
    return [
        {
            "id": r.id,
            "run_name": r.run_name,
            "strategy": r.strategy,
            "cycle": r.cycle,
            "n_trades": r.n_trades,
            "total_pnl": r.total_pnl,
            "win_rate": r.win_rate,
            "sharpe": r.sharpe,
            "max_drawdown": r.max_drawdown,
            "created_at": str(r.created_at),
        }
        for r in runs
    ]


@router.get("/backtest/runs/{run_id}/trades")
def get_backtest_trades(run_id: int, db: Session = Depends(get_election_db)) -> list[dict]:
    from app.election.db.historical_models import BacktestTrade
    trades = db.execute(
        select(BacktestTrade).where(BacktestTrade.run_id == run_id).order_by(BacktestTrade.entry_date)
    ).scalars().all()
    return [
        {
            "race_id": t.race_id,
            "trade_type": t.trade_type,
            "entry_date": str(t.entry_date),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "settlement_price": t.settlement_price,
            "gross_pnl": t.gross_pnl,
            "fees": t.fees,
            "net_pnl": t.net_pnl,
            "won": t.won,
            "description": t.description,
        }
        for t in trades
    ]


@router.post("/backtest/run")
def trigger_backtest(cycles: list[int] = [2022, 2024]) -> dict[str, str]:
    """Trigger backtest run in background thread."""
    import threading
    from app.election.backtest.runner import run_full_backfill_and_backtest
    threading.Thread(target=run_full_backfill_and_backtest, args=(cycles,), daemon=True).start()
    return {"status": "backtest triggered", "cycles": str(cycles)}


@router.get("/event-study/{race_id}")
def get_event_study(
    race_id: int,
    hours_before: int = 48,
    hours_after: int = 48,
    db: Session = Depends(get_election_db),
) -> dict:
    """Get aligned event study data for a race."""
    from app.election.backtest.alignment import build_event_study
    study = build_event_study(db, race_id, hours_before, hours_after)
    if study is None:
        return {"error": f"No event study data for race {race_id}"}

    return {
        "race_id": study.race_id,
        "cycle": study.cycle,
        "election_date": str(study.election_date),
        "n_market_points": len(study.market_series),
        "platforms": list(study.market_series.columns),
        "n_weather_points": len(study.weather_series) if not study.weather_series.empty else 0,
        "n_vote_count_points": len(study.vote_count_series) if study.vote_count_series is not None else 0,
        "party_registration": study.party_reg,
        "market_price_range": {
            "min": float(study.market_series.min().min()) if not study.market_series.empty else None,
            "max": float(study.market_series.max().max()) if not study.market_series.empty else None,
        },
    }


@router.get("/event-study/{race_id}/vote-response")
def vote_response(
    race_id: int,
    lag_minutes: int = 5,
    db: Session = Depends(get_election_db),
) -> list[dict]:
    """Measure market price response to each vote count update."""
    from app.election.backtest.alignment import build_event_study, price_response_to_vote_reporting
    study = build_event_study(db, race_id)
    if study is None:
        return []
    df = price_response_to_vote_reporting(study, lag_minutes)
    return df.to_dict("records") if not df.empty else []


@router.post("/backfill/alt-data")
def trigger_alt_data_backfill(cycles: list[int] = [2024]) -> dict[str, str]:
    """Trigger historical alt-data backfill in background."""
    import threading
    from app.election.historical.backfill_alt_data import run_alt_data_backfill
    threading.Thread(target=run_alt_data_backfill, args=(cycles,), daemon=True).start()
    return {"status": "alt-data backfill triggered", "cycles": str(cycles)}


@router.get("/dashboard/ui", response_class=HTMLResponse)
def get_dashboard_ui():
    """Serve the election alpha dashboard HTML."""
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


@router.get("/alpha-report")
def get_alpha_report() -> dict:
    """Latest alpha model validation report."""
    report_path = Path(__file__).parent.parent / "correlation" / "latest_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return {"error": "No validation report generated yet. POST /v1/election/alpha-train to generate."}


@router.post("/alpha-train")
def trigger_alpha_training(
    train_cycles: list[int] = [2018, 2020, 2022],
    test_cycle: int = 2024,
) -> dict[str, str]:
    """Trigger alpha model training + validation in background."""
    import threading
    from app.election.correlation.model_trainer import train_models, generate_validation_report

    def _run():
        from app.election.db.session import get_election_db
        db = next(get_election_db())
        try:
            results = train_models(db, train_cycles, test_cycle)
            report = generate_validation_report(results)
            report_path = Path(__file__).parent.parent / "correlation" / "latest_report.json"
            report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            logger.info("Alpha model training complete, report saved")
        except Exception as exc:
            logger.error("Alpha training failed: %s", exc)
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "alpha training triggered", "train_cycles": str(train_cycles), "test_cycle": str(test_cycle)}


# --- Signal monitoring endpoints ---

_signal_monitor = None

def _get_monitor():
    global _signal_monitor
    if _signal_monitor is None:
        from app.election.signals.monitor import SignalMonitor
        from app.election.config import ElectionSettings
        settings = ElectionSettings()
        _signal_monitor = SignalMonitor(
            db_path=settings.election_db_path,
            alert_webhook=settings.discord_webhook_url,
        )
    return _signal_monitor


@router.get("/signal-status")
def get_signal_status() -> dict:
    """Current narrative bias signal status."""
    return _get_monitor().get_status()


@router.get("/analog-forecast")
def get_analog_forecast() -> list[dict]:
    """Analog-based mispricing forecast for 2026 races."""
    from app.election.signals.analog_matcher import match_2026_to_analogs, get_mispricing_forecast
    matches = match_2026_to_analogs()
    return get_mispricing_forecast(matches)
