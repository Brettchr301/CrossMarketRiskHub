# Standalone Cross-Market Valuation Platform

This directory is a standalone quantitative research and signal platform focused on identifying mispriced small/mid-cap shipping and commodity equities using:

- Prediction markets (Polymarket, Kalshi)
- Commodity/freight markets
- Company fundamentals and guidance
- Options-implied distributions

The platform is intentionally isolated from the Personal Ledger Pro application.

## Structure

- `quant-api/`: FastAPI service, modeling pipeline, ingestion connectors, Celery workers, backtesting.
- `quant-dashboard/`: React dashboard for probabilities, distributions, valuations, and signals.
- `docker-compose.yml`: Local infra for Postgres + TimescaleDB and Redis.

## Core Design

- Primary signal: probability-weighted intrinsic valuation (scenario DCF + EV/EBITDA cross-check).
- Timing overlay: event-driven repricing behavior from prediction/commodity/options markets.
- Execution policy: no live broker execution by default (`LIVE_TRADING_ENABLED=false`).

## Local Run

1. Start infra:

```bash
docker compose up -d
```

2. Start API:

```bash
cd quant-api
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8100
```

3. Start workers:

```bash
cd quant-api
celery -A app.workers.celery_app.celery_app worker -B --loglevel=INFO
```

4. Start dashboard:

```bash
cd quant-dashboard
npm install
npm run dev
```

## API Endpoints

- `GET /v1/events/probabilities`
- `GET /v1/commodities/distributions`
- `GET /v1/companies/{ticker}/fundamental-state`
- `GET /v1/companies/{ticker}/valuation-distribution`
- `GET /v1/companies/{ticker}/valuation-distribution.xlsx`
- `GET /v1/options/{ticker}/implied-distribution`
- `GET /v1/signals`
- `GET /v1/backtest/metrics`

## Notes

- Free-first adapters are implemented with provider interfaces and mock-safe fallbacks.
- Real-data mode is enabled by default (`REAL_DATA_ONLY=true`).
- Raw ingestion payloads are archived to Parquet when available (CSV fallback when Parquet engine is unavailable).
- Fundamentals are intended to refresh quarterly; market overlays refresh daily and on event shocks.
