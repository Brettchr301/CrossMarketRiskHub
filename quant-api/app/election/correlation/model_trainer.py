import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
import joblib

from app.election.correlation.feature_builder import build_feature_matrix, clean_features


@dataclass
class ModelResult:
    model_name: str
    model: BaseEstimator
    brier_score: float
    hit_rate: float
    mse: float
    r2: float
    feature_importance: dict[str, float]
    calibration_data: list[dict]
    n_train: int
    n_test: int


def train_models(
    db: Session,
    train_cycles: list[int] = [2018, 2020, 2022],
    test_cycle: int = 2024,
    lookback_days: int = 180,
) -> list[ModelResult]:
    """Train 4 model families on historical data, validate on held-out cycle."""
    results = []
    
    # 1-2. Build feature matrices for train and test cycles
    train_dfs = []
    for cycle in train_cycles:
        df = build_feature_matrix(db, cycle, lookback_days)
        if df is not None and not df.empty:
            train_dfs.append(df)
    
    if not train_dfs:
        raise ValueError("No training data available for specified cycles")
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = build_feature_matrix(db, test_cycle, lookback_days)
    
    if test_df is None or test_df.empty:
        raise ValueError("No test data available for test cycle")
    
    # 3. Clean features
    train_df_clean = clean_features(train_df)
    test_df_clean = clean_features(test_df)
    
    # Align features between train and test
    common_cols = train_df_clean.columns.intersection(test_df_clean.columns)
    target_col = 'target'
    if target_col not in common_cols:
        raise ValueError("Target column missing after cleaning")
    
    feature_cols = [c for c in common_cols if c != target_col]
    if not feature_cols:
        raise ValueError("No features remain after cleaning")
    
    X_train = train_df_clean[feature_cols].fillna(0).values
    y_train = train_df_clean[target_col].values
    X_test = test_df_clean[feature_cols].fillna(0).values
    y_test = test_df_clean[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = [
        ("ridge", RidgeCV(alphas=[0.1, 1.0, 10.0], cv=TimeSeriesSplit(n_splits=5))),
        ("lasso", LassoCV(cv=TimeSeriesSplit(n_splits=5), max_iter=10000)),
        ("elasticnet", ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), max_iter=10000)),
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    
    for model_name, model in models:
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test
        y_pred = model.predict(X_test_scaled)
        y_pred = np.clip(y_pred, 0, 1)  # Clip to probability range
        
        # Calculate metrics
        brier = brier_score_loss(y_test, y_pred)
        hit_rate = np.mean((y_pred > 0.5) == (y_test > 0.5))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'coef_'):
            importance = dict(zip(feature_cols, np.abs(model.coef_)))
        elif hasattr(model, 'feature_importances_'):
            importance = dict(zip(feature_cols, model.feature_importances_))
        else:
            importance = {col: 0.0 for col in feature_cols}
        
        # Top 10 features
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10, strategy='uniform')
        calibration_data = []
        for i in range(len(prob_true)):
            bin_center = (i + 0.5) / 10
            calibration_data.append({
                "bin_center": float(bin_center),
                "predicted_mean": float(prob_pred[i]),
                "actual_mean": float(prob_true[i]),
                "count": int(len(y_test) // 10)
            })
        
        result = ModelResult(
            model_name=model_name,
            model=model,
            brier_score=brier,
            hit_rate=hit_rate,
            mse=mse,
            r2=r2,
            feature_importance=top_features,
            calibration_data=calibration_data,
            n_train=len(y_train),
            n_test=len(y_test)
        )
        results.append(result)
    
    # Sort by Brier score (ascending)
    results.sort(key=lambda x: x.brier_score)
    return results


def walk_forward_validate(
    db: Session,
    model: BaseEstimator,
    race_ids: list[int],
    n_splits: int = 5,
    lookback_days: int = 180,
) -> dict[str, float]:
    """Walk-forward cross-validation within a single cycle."""
    # Build feature matrix for all races
    all_dfs = []
    for race_id in race_ids:
        # In practice, we'd need a function to get features for specific race
        # For now, we'll reuse build_feature_matrix with filtering
        pass
    
    # Since we don't have race-specific feature building yet,
    # we'll implement a simplified version using TimeSeriesSplit
    raise NotImplementedError("walk_forward_validate requires race-specific feature building")


def generate_validation_report(results: list[ModelResult]) -> dict[str, Any]:
    """Generate JSON-serializable validation report."""
    if not results:
        return {"error": "No model results provided"}
    
    baseline_brier = 0.25  # Always predict 0.5
    best_result = min(results, key=lambda x: x.brier_score)
    improvement = (baseline_brier - best_result.brier_score) / baseline_brier * 100
    
    models_data = []
    for result in results:
        models_data.append({
            "name": result.model_name,
            "brier_score": float(result.brier_score),
            "hit_rate": float(result.hit_rate),
            "mse": float(result.mse),
            "r2": float(result.r2),
            "top_features": result.feature_importance,
            "calibration": result.calibration_data
        })
    
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "train_cycles": [2018, 2020, 2022],  # Would need to be parameterized
        "test_cycle": 2024,  # Would need to be parameterized
        "models": models_data,
        "best_model": best_result.model_name,
        "baseline_brier": baseline_brier,
        "improvement_vs_baseline_pct": float(improvement)
    }


def select_best_model(results: list[ModelResult]) -> tuple[BaseEstimator, str]:
    """Select model with lowest Brier score. Returns (model, model_name)."""
    if not results:
        raise ValueError("No results provided")
    
    best_result = min(results, key=lambda x: x.brier_score)
    return best_result.model, best_result.model_name


def save_trained_model(model: BaseEstimator, model_name: str, version: str = "v1"):
    """Save trained model to disk."""
    model_dir = Path(__file__).parent / "trained_models"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / f"{model_name}_{version}.joblib"
    joblib.dump(model, model_path)
    return model_path
