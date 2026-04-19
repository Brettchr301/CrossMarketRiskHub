"""Correlation alpha model.

Walk-forward Ridge regression predicting next-24h probability changes.
Same pattern as global_scan.py:_walkforward_fit().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from sqlalchemy.orm import Session

from app.election.correlation.feature_builder import build_feature_matrix
from app.election.db.models import AlphaModelPrediction

logger = logging.getLogger(__name__)

MODEL_VERSION = "ridge_v1"
ALPHA_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]
MIN_SAMPLES = 30


@dataclass
class AlphaSignal:
    race_id: int
    predicted_prob_change: float
    confidence: float
    top_features: dict[str, float]


def train_and_predict(
    db: Session,
    race_id: int,
    lookback_days: int = 90,
) -> AlphaSignal | None:
    """Train walk-forward Ridge and predict next-24h probability change."""
    df = build_feature_matrix(db, race_id, lookback_days)

    if df.empty or len(df) < MIN_SAMPLES:
        logger.info("Race %d: insufficient data (%d rows)", race_id, len(df))
        return None

    target_col = "target_prob_change"
    feature_cols = [c for c in df.columns if c != target_col]

    if not feature_cols:
        return None

    X = df[feature_cols].values
    y = df[target_col].values

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Walk-forward with TimeSeriesSplit
    n_splits = min(5, max(2, len(X) // 20))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    try:
        model = RidgeCV(alphas=ALPHA_GRID, cv=tscv, scoring="neg_mean_squared_error")
        model.fit(X, y)
    except Exception as exc:
        logger.error("Ridge fit failed for race %d: %s", race_id, exc)
        return None

    # Predict on latest row
    latest_x = X[-1:].copy()
    prediction = float(model.predict(latest_x)[0])

    # Feature importance
    coefs = dict(zip(feature_cols, model.coef_.tolist()))
    top = dict(sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5])

    # Confidence from R² on last fold
    last_train, last_test = list(tscv.split(X))[-1]
    test_pred = model.predict(X[last_test])
    ss_res = np.sum((y[last_test] - test_pred) ** 2)
    ss_tot = np.sum((y[last_test] - y[last_test].mean()) ** 2)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    confidence = min(1.0, r2 + 0.1)  # floor at 0.1

    return AlphaSignal(
        race_id=race_id,
        predicted_prob_change=round(prediction, 6),
        confidence=round(confidence, 4),
        top_features=top,
    )


def run_alpha_model(
    db: Session,
    race_ids: list[int],
    lookback_days: int = 90,
) -> list[AlphaSignal]:
    """Run alpha model for all races and persist predictions."""
    signals = []
    now = datetime.now(UTC).replace(tzinfo=None)

    for race_id in race_ids:
        sig = train_and_predict(db, race_id, lookback_days)
        if sig is None:
            continue
        signals.append(sig)

        row = AlphaModelPrediction(
            race_id=sig.race_id,
            predicted_prob_change=sig.predicted_prob_change,
            confidence=sig.confidence,
            top_features=sig.top_features,
            model_version=MODEL_VERSION,
            as_of=now,
        )
        db.add(row)

    if signals:
        db.commit()
        logger.info("Alpha model: %d predictions generated", len(signals))

    return signals
