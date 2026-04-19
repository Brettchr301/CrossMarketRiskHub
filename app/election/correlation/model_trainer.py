from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    
    # 1. Build feature matrix for all races in train_cycles
    train_dfs = []
    for cycle in train_cycles:
        df = build_feature_matrix(db, cycle, lookback_days)
        if df is not None and not df.empty:
            train_dfs.append(df)
            
    if not train_dfs:
        raise ValueError("No training data available for specified cycles")
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    if 'date' in train_df.columns:
        train_df = train_df.sort_values('date')
        
    # 2. Build feature matrix for all races in test_cycle
    test_df = build_feature_matrix(db, test_cycle, lookback_days)
    if test_df is None or test_df.empty:
        raise ValueError("No test data available for test cycle")
        
    if 'date' in test_df.columns:
        test_df = test_df.sort_values('date')
        
    # 3. Clean features (drop >50% NaN, near-zero variance)
    train_df_clean = clean_features(train_df)
    test_df_clean = clean_features(test_df)
    
    common_cols = train_df_clean.columns.intersection(test_df_clean.columns)
    target_col = 'target'
    if target_col not in common_cols:
        raise ValueError("Target column missing after cleaning")
        
    feature_cols = [c for c in common_cols if c not in (target_col, 'date', 'race_id')]
    if not feature_cols:
        raise ValueError("No features remain after cleaning")
        
    X_train = train_df_clean[feature_cols].fillna(0).values
    y_train = train_df_clean[target_col].values
    X_test = test_df_clean[feature_cols].fillna(0).values
    y_test = test_df_clean[target_col].values
    
    # 4. Train: RidgeCV, LassoCV, ElasticNetCV, RandomForestRegressor
    models = [
        ("ridge", Pipeline([('scaler', StandardScaler()), ('regressor', RidgeCV(alphas=[0.1, 1.0, 10.0], cv=TimeSeriesSplit(n_splits=5)))])),
        ("lasso", Pipeline([('scaler', StandardScaler()), ('regressor', LassoCV(cv=TimeSeriesSplit(n_splits=5), max_iter=10000))])),
        ("elasticnet", Pipeline([('scaler', StandardScaler()), ('regressor', ElasticNetCV(cv=TimeSeriesSplit(n_splits=5), max_iter=10000))])),
        ("random_forest", Pipeline([('scaler', StandardScaler()), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])),
    ]
    
    for model_name, model in models:
        model.fit(X_train, y_train)
        
        # 5. Evaluate each on test_cycle data
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)
        
        brier = float(brier_score_loss(y_test, y_pred))
        hit_rate = float(np.mean((y_pred > 0.5) == (y_test > 0.5)))
        mse = float(mean_squared_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred)) if np.var(y_test) > 0 else 0.0
        
        regressor = model.named_steps['regressor']
        if hasattr(regressor, 'coef_'):
            importance = dict(zip(feature_cols, np.abs(regressor.coef_)))
        elif hasattr(regressor, 'feature_importances_'):
            importance = dict(zip(feature_cols, regressor.feature_importances_))
        else:
            importance = {col: 0.0 for col in feature_cols}
            
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
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
        
    # 6. Return sorted by Brier score (ascending)
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
    current_year = datetime.now().year
    df = build_feature_matrix(db, current_year, lookback_days)
    if df is None or df.empty:
        return {"brier": 0.0, "hit_rate": 0.0, "mse": 0.0, "r2": 0.0, "n_predictions": 0}
        
    if 'race_id' in df.columns:
        df = df[df['race_id'].isin(race_ids)]
    elif df.index.name == 'race_id' or 'race_id' in df.index.names:
        df = df[df.index.isin(race_ids)]
        
    if df.empty:
        return {"brier": 0.0, "hit_rate": 0.0, "mse": 0.0, "r2": 0.0, "n_predictions": 0}
        
    if 'date' in df.columns:
        df = df.sort_values('date')
        
    df_clean = clean_features(df)
    target_col = 'target'
    if target_col not in df_clean.columns:
        return {"brier": 0.0, "hit_rate": 0.0, "mse": 0.0, "r2": 0.0, "n_predictions": 0}
        
    feature_cols = [c for c in df_clean.columns if c not in (target_col, 'date', 'race_id')]
    X = df_clean[feature_cols].fillna(0).values
    y = df_clean[target_col].values
    
    if len(X) < n_splits + 1:
        return {"brier": 0.0, "hit_rate": 0.0, "mse": 0.0, "r2": 0.0, "n_predictions": 0}
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    briers, hits, mses, r2s = [], [], [], []
    n_preds = 0
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        split_model = clone(model)
        split_model.fit(X_train, y_train)
        
        y_pred = split_model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)
        
        briers.append(brier_score_loss(y_test, y_pred))
        hits.append(np.mean((y_pred > 0.5) == (y_test > 0.5)))
        mses.append(mean_squared_error(y_test, y_pred))
        if np.var(y_test) > 0:
            r2s.append(r2_score(y_test, y_pred))
        else:
            r2s.append(0.0)
        n_preds += len(y_test)
        
    return {
        "brier": float(np.mean(briers)) if briers else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "mse": float(np.mean(mses)) if mses else 0.0,
        "r2": float(np.mean(r2s)) if r2s else 0.0,
        "n_predictions": n_preds
    }


def generate_validation_report(results: list[ModelResult]) -> dict[str, Any]:
    """Generate JSON-serializable validation report."""
    if not results:
        return {"error": "No model results provided"}
        
    baseline_brier = 0.25
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
        "train_cycles": [2018, 2020, 2022],
        "test_cycle": 2024,
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
