import pytest
import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.linear_model import Ridge
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.election.correlation.model_trainer import (
    ModelResult,
    train_models,
    walk_forward_validate,
    generate_validation_report,
    select_best_model
)
from app.election.correlation.feature_builder import clean_features
from app.election.api.routes import router


@patch("app.election.correlation.model_trainer.build_feature_matrix", create=True)
def test_train_models_synthetic(mock_build_features):
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_raw = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(100) * 0.1
    y = 1 / (1 + np.exp(-y_raw))

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y

    # 3 train cycles + 1 test cycle = 4 calls
    mock_build_features.side_effect = [df.iloc[:20], df.iloc[20:40], df.iloc[40:80], df.iloc[80:]]

    db_mock = MagicMock()
    results = train_models(db_mock, train_cycles=[2018, 2020, 2022], test_cycle=2024)

    assert len(results) == 4
    model_names = {r.model_name for r in results}
    assert model_names == {"ridge", "lasso", "elasticnet", "random_forest"}

    for res in results:
        assert isinstance(res, ModelResult)
        assert res.brier_score < 0.5


@patch("app.election.correlation.model_trainer.build_feature_matrix", create=True)
@patch("app.election.correlation.model_trainer.clone", side_effect=lambda x: x)
def test_walk_forward_no_leakage(mock_clone, mock_build_features):
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    time_feature = np.arange(100)
    df = pd.DataFrame({
        "date": dates,
        "race_id": [1] * 100,
        "time_feat": time_feature,
        "target": np.random.rand(100)
    })
    mock_build_features.return_value = df

    class MockEstimator(Ridge):
        def __init__(self):
            super().__init__()
            self.train_max_time = []
            self.test_min_time = []

        def fit(self, X, y):
            self.train_max_time.append(np.max(X[:, 0]))
            return self

        def predict(self, X):
            self.test_min_time.append(np.min(X[:, 0]))
            return np.zeros(len(X))

    model = MockEstimator()
    db_mock = MagicMock()

    res = walk_forward_validate(db_mock, model, race_ids=[1], n_splits=5)
    
    assert res["n_predictions"] > 0
    assert len(model.train_max_time) == 5
    for train_max, test_min in zip(model.train_max_time, model.test_min_time):
        assert train_max < test_min


def test_feature_selection():
    df = pd.DataFrame({
        "nan_1": [np.nan]*6 + [1, 2, 3, 4],
        "nan_2": [np.nan]*7 + [1, 2, 3],
        "nan_3": [np.nan]*8 + [1, 2],
        "zero_var_1": [5.0]*10,
        "zero_var_2": [1.0]*10,
        "good_1": np.random.randn(10),
        "good_2": np.random.randn(10),
        "good_3": np.random.randn(10),
        "good_4": np.random.randn(10),
        "good_5": np.random.randn(10),
        "target": np.random.rand(10)
    })

    cleaned_df = clean_features(df)

    assert cleaned_df.shape[1] == 6
    assert "nan_1" not in cleaned_df.columns
    assert "zero_var_1" not in cleaned_df.columns
    assert "good_1" in cleaned_df.columns
    assert "target" in cleaned_df.columns


def test_validation_report_json():
    results = [
        ModelResult(
            model_name="ridge",
            model=Ridge(),
            brier_score=0.159,
            hit_rate=0.67,
            mse=0.023,
            r2=0.34,
            feature_importance={"poll_delta_3d": 0.45},
            calibration_data=[{"bin_center": 0.1, "predicted_mean": 0.12, "actual_mean": 0.09, "count": 23}],
            n_train=1000,
            n_test=200
        )
    ]
    report = generate_validation_report(results)
    
    try:
        json_str = json.dumps(report)
    except TypeError:
        pytest.fail("Validation report is not JSON serializable")
        
    assert "ridge" in json_str
    assert "baseline_brier" in report


def test_best_model_selection():
    results = [
        ModelResult(model_name="m1", model=Ridge(), brier_score=0.20, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),
        ModelResult(model_name="m2", model=Ridge(), brier_score=0.15, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),
        ModelResult(model_name="m3", model=Ridge(), brier_score=0.18, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),
        ModelResult(model_name="m4", model=Ridge(), brier_score=0.22, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),
    ]
    best_model, best_name = select_best_model(results)
    
    assert best_name == "m2"
    assert best_model == results[1].model


@patch("app.election.correlation.model_trainer.build_feature_matrix", create=True)
def test_calibration_bins(mock_build_features):
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y_raw = X[:, 0] * 0.5 + np.random.randn(200) * 0.1
    y = 1 / (1 + np.exp(-y_raw))

    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y

    mock_build_features.side_effect = [df, df]

    db_mock = MagicMock()
    results = train_models(db_mock, train_cycles=[2020], test_cycle=2024)

    cal_data = results[0].calibration_data
    assert len(cal_data) > 0
    assert len(cal_data) <= 10

    for bin_data in cal_data:
        assert "bin_center" in bin_data
        assert "predicted_mean" in bin_data
        assert "actual_mean" in bin_data
        assert "count" in bin_data


def test_baseline_comparison():
    results = [
        ModelResult(
            model_name="ridge",
            model=Ridge(),
            brier_score=0.15,
            hit_rate=0.65,
            mse=0.02,
            r2=0.3,
            feature_importance={},
            calibration_data=[],
            n_train=1000,
            n_test=200
        )
    ]
    report = generate_validation_report(results)
    
    assert "baseline_brier" in report
    assert report["baseline_brier"] == 0.25
    
    assert "improvement_vs_baseline_pct" in report
    assert report["improvement_vs_baseline_pct"] == 40.0


def test_alpha_report_endpoint():
    app = FastAPI()
    app.include_router(router, prefix="/v1/election")
    client = TestClient(app)

    with patch("threading.Thread.start") as mock_thread:
        post_resp = client.post(
            "/v1/election/alpha-train", 
            json={"train_cycles": [2018, 2020, 2022], "test_cycle": 2024}
        )
        assert post_resp.status_code == 200
        assert post_resp.json() == {"status": "training triggered", "train_cycles": [2018, 2020, 2022], "test_cycle": 2024}
        mock_thread.assert_called_once()

    mock_report = {
        "generated_at": "2026-04-19T00:00:00",
        "best_model": "ridge",
        "baseline_brier": 0.25,
        "improvement_vs_baseline_pct": 36.4
    }
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value=json.dumps(mock_report)):
        get_resp = client.get("/v1/election/alpha-report")
        assert get_resp.status_code == 200
        
        data = get_resp.json()
        assert data["best_model"] == "ridge"
        assert data["baseline_brier"] == 0.25
        assert data["improvement_vs_baseline_pct"] == 36.4
