"""Feature builder for correlation alpha model.

Constructs a feature matrix from polling, finance, and alt-data signals.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.election.db.models import (
    AltDataSignal,
    BlendedProbability,
    CampaignFinance,
    CorrelationFeature,
    PollingData,
)

logger = logging.getLogger(__name__)


def build_feature_matrix(
    db: Session,
    race_id: int,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Build feature matrix for a single race.

    Returns DataFrame with columns = feature names, index = as_of timestamps.
    Last column is 'target_prob_change' (next-24h blended probability change).
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    features: dict[str, pd.Series] = {}

    # 1. Blended probability history (for target variable)
    probs = db.execute(
        select(BlendedProbability)
        .where(BlendedProbability.race_id == race_id, BlendedProbability.as_of >= cutoff)
        .order_by(BlendedProbability.as_of)
    ).scalars().all()

    if len(probs) < 10:
        return pd.DataFrame()

    prob_series = pd.Series(
        [p.prob for p in probs],
        index=pd.DatetimeIndex([p.as_of for p in probs]),
    )
    prob_series = prob_series[~prob_series.index.duplicated(keep="last")]

    # Target: next-24h probability change
    target = prob_series.shift(-1) - prob_series
    features["target_prob_change"] = target

    # 2. Polling delta features (1d, 3d, 7d)
    polls = db.execute(
        select(PollingData)
        .where(PollingData.race_id == race_id, PollingData.poll_date >= cutoff.date())
        .order_by(PollingData.poll_date)
    ).scalars().all()

    if polls:
        poll_ts = pd.Series(
            [p.pct for p in polls],
            index=pd.DatetimeIndex([datetime.combine(p.poll_date, datetime.min.time()) for p in polls]),
        )
        poll_ts = poll_ts.resample("D").mean().ffill()
        for window in [1, 3, 7]:
            features[f"poll_delta_{window}d"] = poll_ts.diff(window)
        features["poll_momentum"] = poll_ts.diff(1).diff(1)  # second derivative

    # 3. Campaign finance features
    finance = db.execute(
        select(CampaignFinance).where(CampaignFinance.candidate_id.in_(
            db.execute(
                select(PollingData.candidate_id).where(PollingData.race_id == race_id)
            ).scalars().all()
        ))
    ).scalars().all()

    if finance:
        total_receipts = sum(f.receipts for f in finance)
        total_coh = sum(f.cash_on_hand for f in finance)
        if total_receipts > 0:
            features["fundraising_log"] = pd.Series(
                np.log1p(total_receipts), index=prob_series.index
            )
        if total_coh > 0:
            features["cash_on_hand_log"] = pd.Series(
                np.log1p(total_coh), index=prob_series.index
            )

    # 4. Alt-data signals
    for signal_type in ["google_trends", "wikipedia", "weather"]:
        alt_rows = db.execute(
            select(AltDataSignal).where(
                AltDataSignal.race_id == race_id,
                AltDataSignal.signal_type == signal_type,
                AltDataSignal.as_of >= cutoff,
            ).order_by(AltDataSignal.as_of)
        ).scalars().all()

        if alt_rows:
            alt_series = pd.Series(
                [a.value for a in alt_rows],
                index=pd.DatetimeIndex([a.as_of for a in alt_rows]),
            )
            alt_series = alt_series[~alt_series.index.duplicated(keep="last")]
            features[f"alt_{signal_type}"] = alt_series

    # 5. Cross-race contagion (probability changes in correlated races)
    corr_features = db.execute(
        select(CorrelationFeature).where(
            CorrelationFeature.race_id == race_id,
            CorrelationFeature.as_of >= cutoff,
        ).order_by(CorrelationFeature.as_of)
    ).scalars().all()

    if corr_features:
        for cf in corr_features:
            fname = cf.feature_name
            if fname not in features:
                features[fname] = pd.Series(dtype=float)
            features[fname].loc[cf.as_of] = cf.feature_value

    # Combine into DataFrame
    if not features:
        return pd.DataFrame()

    df = pd.DataFrame(features)
    df = df.sort_index().ffill().dropna(subset=["target_prob_change"])
    return df


def clean_features(df: pd.DataFrame, max_nan_pct: float = 0.5) -> pd.DataFrame:
    """Clean feature matrix: drop high-NaN and near-zero-variance columns.

    Args:
        df: Feature matrix with target_prob_change column.
        max_nan_pct: Maximum fraction of NaN values allowed per column.

    Returns:
        Cleaned DataFrame with only usable features.
    """
    if df.empty:
        return df

    target_col = "target_prob_change"
    feature_cols = [c for c in df.columns if c != target_col]

    keep = []
    for col in feature_cols:
        series = df[col]
        # Drop if too many NaN
        if series.isna().mean() > max_nan_pct:
            continue
        # Drop if near-zero variance
        valid = series.dropna()
        if len(valid) < 2 or valid.std() < 1e-10:
            continue
        keep.append(col)

    if target_col in df.columns:
        keep.append(target_col)

    return df[keep].dropna()
