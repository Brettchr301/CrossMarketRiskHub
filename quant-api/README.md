# quant-api

Standalone API + modeling engine for cross-market EV mispricing signals.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn app.main:app --reload --port 8100
```

## Run workers

```bash
celery -A app.workers.celery_app.celery_app worker -B --loglevel=INFO
```

## Tests

```bash
python -m pytest -q
```

## Opportunity Regime Backtest (real-data mode)

```bash
python scripts/real_data_backtest.py
```

Outputs:

- `analysis_output/real_data_backtest_summary.json`
- `analysis_output/real_data_backtest_trades.csv`
