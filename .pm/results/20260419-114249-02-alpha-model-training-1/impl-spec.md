# Task: Alpha Model Training with Out-of-Sample Validation

## Classification
| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Security | 0 | No auth, internal model training |
| Complexity | 3 | Walk-forward CV, multiple feature families, Brier calibration, model comparison |
| Novelty | 2 | Extends existing Ridge stub to full training pipeline with validation report |
| Blast Radius | 2 | Changes alpha model predictions used by dashboard and alerts |
| Existing Code | 2 | Must integrate with feature_builder.py, alpha_model.py, correlation_study.py |
| **Total** | **9** | |

## Objective

The alpha model infrastructure exists but hasn't been trained or validated end-to-end:

1. **`alpha_model.py`** has `train_and_predict()` using RidgeCV with TimeSeriesSplit — but it only predicts on the latest row. No out-of-sample validation, no model comparison, no calibration metrics.

2. **`feature_builder.py`** builds features (polling deltas, campaign finance, alt-data, cross-race contagion) — but has no feature importance analysis or feature selection.

3. **No validation report** — we don't know if the model actually predicts anything useful. Need Brier scores, hit rates, and comparison vs naive baseline.

We need a full training pipeline that:
- Trains on 2018-2022 data, validates on 2024 (true out-of-sample)
- Compares Ridge vs Lasso vs ElasticNet vs Random Forest
- Produces a validation report with Brier scores, calibration curves, feature importance
- Persists the best model and its predictions

## Deliverables

- [ ] `app/election/correlation/model_trainer.py` — NEW: full training pipeline
  - `train_models(db, train_cycles, test_cycles)` — trains 4 model families
  - `walk_forward_validate(db, model, race_ids, n_splits)` — walk-forward OOS validation
  - `generate_validation_report(results) -> dict` — Brier, hit rate, calibration, feature importance
  - `select_best_model(results) -> tuple[model, str]` — picks lowest Brier model
- [ ] `app/election/correlation/alpha_model.py` — modify `run_alpha_model()` to accept a pre-trained model instead of always re-training per race. Add `model_version` tracking for the new models.
- [ ] `app/election/correlation/feature_builder.py` — add feature selection: drop features with >50% NaN, drop features with near-zero variance, add feature correlation heatmap data
- [ ] `app/election/api/routes.py` — add `GET /v1/election/alpha-report` endpoint returning the latest validation report
- [ ] `tests/test_alpha_model_training.py` — tests with synthetic data

## Constraints

- DO NOT remove existing Ridge model — keep it as one option in the comparison
- DO NOT add GPU-dependent libraries (no PyTorch, TensorFlow)
- All models must be from scikit-learn (already a dependency)
- Walk-forward validation must respect temporal ordering (no future data leakage)
- Model serialization: use joblib (already bundled with scikit-learn)
- Validation report must be JSON-serializable for the API endpoint
- DO NOT change the `AlphaModelPrediction` table schema

## Exact Interface

```python
# model_trainer.py

from dataclasses import dataclass
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator
from sqlalchemy.orm import Session


@dataclass
class ModelResult:
    model_name: str        # "ridge", "lasso", "elasticnet", "random_forest"
    model: BaseEstimator
    brier_score: float     # lower is better, 0=perfect
    hit_rate: float        # % of correct direction predictions
    mse: float             # mean squared error on OOS
    r2: float              # R² on OOS
    feature_importance: dict[str, float]  # top 10 features by |coef| or importance
    calibration_data: list[dict]  # [{bin_center, predicted_mean, actual_mean, count}]
    n_train: int
    n_test: int


def train_models(
    db: Session,
    train_cycles: list[int] = [2018, 2020, 2022],
    test_cycle: int = 2024,
    lookback_days: int = 180,
) -> list[ModelResult]:
    """Train 4 model families on historical data, validate on held-out cycle.

    Steps:
    1. Build feature matrix for all races in train_cycles
    2. Build feature matrix for all races in test_cycle
    3. Clean features (drop >50% NaN, near-zero variance)
    4. Train: RidgeCV, LassoCV, ElasticNetCV, RandomForestRegressor
    5. Evaluate each on test_cycle data
    6. Return sorted by Brier score (ascending)
    """


def walk_forward_validate(
    db: Session,
    model: BaseEstimator,
    race_ids: list[int],
    n_splits: int = 5,
    lookback_days: int = 180,
) -> dict[str, float]:
    """Walk-forward cross-validation within a single cycle.

    Returns: {brier, hit_rate, mse, r2, n_predictions}
    """


def generate_validation_report(results: list[ModelResult]) -> dict[str, Any]:
    """Generate JSON-serializable validation report.

    Returns:
    {
        "generated_at": "2026-04-19T...",
        "train_cycles": [2018, 2020, 2022],
        "test_cycle": 2024,
        "models": [
            {
                "name": "ridge",
                "brier_score": 0.159,
                "hit_rate": 0.67,
                "mse": 0.023,
                "r2": 0.34,
                "top_features": {"poll_delta_3d": 0.45, ...},
                "calibration": [{"bin": 0.1, "predicted": 0.12, "actual": 0.09, "n": 23}, ...],
            },
            ...
        ],
        "best_model": "ridge",
        "baseline_brier": 0.25,  # naive: always predict 0.5
        "improvement_vs_baseline_pct": 36.4,
    }
    """


def select_best_model(results: list[ModelResult]) -> tuple[BaseEstimator, str]:
    """Select model with lowest Brier score. Returns (model, model_name)."""
```

### API Route Addition

```python
# In routes.py, add:

@router.get("/alpha-report")
def get_alpha_report(db: Session = Depends(get_election_db)) -> dict:
    """Latest alpha model validation report."""
    # Read from a cached JSON file or recompute
    report_path = Path(__file__).parent.parent / "correlation" / "latest_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    return {"error": "No validation report generated yet. POST /v1/election/alpha-train to generate."}


@router.post("/alpha-train")
def trigger_alpha_training(
    train_cycles: list[int] = [2018, 2020, 2022],
    test_cycle: int = 2024,
) -> dict:
    """Trigger alpha model training + validation in background."""
    import threading
    from app.election.correlation.model_trainer import train_models, generate_validation_report
    def _run():
        # ... train, generate report, save to JSON
        pass
    threading.Thread(target=_run, daemon=True).start()
    return {"status": "training triggered"}
```

## Tests to Write

1. **test_train_models_synthetic**: Generate synthetic feature matrix (100 rows, 5 features, target = weighted sum + noise). Train all 4 models. Verify all return ModelResult with Brier < 0.5 (better than random).

2. **test_walk_forward_no_leakage**: Create time-indexed data. Verify that for each split, all training timestamps < all test timestamps (no future leakage).

3. **test_feature_selection**: Create matrix with 10 features, 3 of which are >50% NaN and 2 with zero variance. Verify cleaned matrix has exactly 5 features.

4. **test_validation_report_json**: Generate report from mock ModelResults. Verify it's JSON-serializable (`json.dumps(report)` doesn't raise).

5. **test_best_model_selection**: Create 4 ModelResults with Brier scores [0.20, 0.15, 0.18, 0.22]. Verify `select_best_model()` returns the model with Brier=0.15.

6. **test_calibration_bins**: Generate predictions + actuals. Verify calibration_data has 10 bins, each with bin_center, predicted_mean, actual_mean, count fields.

7. **test_baseline_comparison**: Verify report includes baseline Brier (always predict 0.5 = 0.25) and improvement percentage.

8. **test_alpha_report_endpoint**: Mock FastAPI test client. POST `/v1/election/alpha-train`, then GET `/v1/election/alpha-report`. Verify report structure.

## Files to Touch
- `app/election/correlation/model_trainer.py` — create
- `app/election/correlation/alpha_model.py` — modify
- `app/election/correlation/feature_builder.py` — modify
- `app/election/api/routes.py` — modify
- `tests/test_alpha_model_training.py` — create

## Success Criteria
1. All 8 tests pass
2. Best model achieves Brier < 0.25 (better than naive baseline) on synthetic data
3. `GET /v1/election/alpha-report` returns a well-structured JSON report
4. Feature importance reveals which features actually predict probability changes
5. No regressions: existing `run_alpha_model()` still works for live predictions
6. Report includes calibration curve data for visualization



## Pre-Generated Tests (your code MUST pass these)
These tests were generated independently. Include them EXACTLY as shown.

```json
{"content": "--- FILE: tests/test_alpha_model_training.py ---\nimport pytest\nimport json\nimport numpy as np\nimport pandas as pd\nfrom unittest.mock import MagicMock, patch\nfrom sklearn.linear_model import Ridge\nfrom fastapi.testclient import TestClient\nfrom fastapi import FastAPI\n\n# Assuming the implementation follows the spec exactly\nfrom app.election.correlation.model_trainer import (\n    ModelResult,\n    train_models,\n    walk_forward_validate,\n    generate_validation_report,\n    select_best_model\n)\n# Assuming clean_features is the function added to feature_builder.py\nfrom app.election.correlation.feature_builder import clean_features\nfrom app.election.api.routes import router\n\n\n@patch(\"app.election.correlation.model_trainer.build_feature_matrix\", create=True)\ndef test_train_models_synthetic(mock_build_features):\n    \"\"\"\n    1. Generate synthetic feature matrix (100 rows, 5 features, target = weighted sum + noise). \n    Train all 4 models. Verify all return ModelResult with Brier < 0.5 (better than random).\n    \"\"\"\n    np.random.seed(42)\n    X = np.random.randn(100, 5)\n    # Target is probability between 0 and 1\n    y_raw = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(100) * 0.1\n    y = 1 / (1 + np.exp(-y_raw))\n\n    df = pd.DataFrame(X, columns=[f\"feat_{i}\" for i in range(5)])\n    df[\"target\"] = y\n\n    # Mock the feature builder to return our synthetic data for both train and test phases\n    mock_build_features.side_effect = [df.iloc[:80], df.iloc[80:]]\n\n    db_mock = MagicMock()\n    results = train_models(db_mock, train_cycles=[2018, 2020, 2022], test_cycle=2024)\n\n    assert len(results) == 4\n    model_names = {r.model_name for r in results}\n    assert model_names == {\"ridge\", \"lasso\", \"elasticnet\", \"random_forest\"}\n\n    for res in results:\n        assert isinstance(res, ModelResult)\n        assert res.brier_score < 0.5  # Better than random guessing\n\n\n@patch(\"app.election.correlation.model_trainer.build_feature_matrix\", create=True)\ndef test_walk_forward_no_leakage(mock_build_features):\n    \"\"\"\n    2. Create time-indexed data. Verify that for each split, \n    all training timestamps < all test timestamps (no future leakage).\n    \"\"\"\n    dates = pd.date_range(start=\"2024-01-01\", periods=100, freq=\"D\")\n    df = pd.DataFrame({\n        \"date\": dates,\n        \"feat_0\": np.random.randn(100),\n        \"target\": np.random.rand(100)\n    })\n    mock_build_features.return_value = df\n\n    class MockEstimator(Ridge):\n        def __init__(self):\n            super().__init__()\n            self.splits = []\n\n        def fit(self, X, y):\n            self.train_indices = X.index\n            return self\n\n        def predict(self, X):\n            self.test_indices = X.index\n            self.splits.append((self.train_indices, self.test_indices))\n            return np.zeros(len(X))\n\n    model = MockEstimator()\n    db_mock = MagicMock()\n\n    walk_forward_validate(db_mock, model, race_ids=[1], n_splits=5)\n\n    assert len(model.splits) == 5\n    for train_idx, test_idx in model.splits:\n        train_dates = df.iloc[train_idx][\"date\"]\n        test_dates = df.iloc[test_idx][\"date\"]\n        assert train_dates.max() < test_dates.min()\n\n\ndef test_feature_selection():\n    \"\"\"\n    3. Create matrix with 10 features, 3 of which are >50% NaN and 2 with zero variance. \n    Verify cleaned matrix has exactly 5 features.\n    \"\"\"\n    df = pd.DataFrame({\n        \"nan_1\": [np.nan]*6 + [1, 2, 3, 4],      # 60% NaN\n        \"nan_2\": [np.nan]*7 + [1, 2, 3],         # 70% NaN\n        \"nan_3\": [np.nan]*8 + [1, 2],            # 80% NaN\n        \"zero_var_1\": [5.0]*10,                  # 0 variance\n        \"zero_var_2\": [1.0]*10,                  # 0 variance\n        \"good_1\": np.random.randn(10),\n        \"good_2\": np.random.randn(10),\n        \"good_3\": np.random.randn(10),\n        \"good_4\": np.random.randn(10),\n        \"good_5\": np.random.randn(10),\n        \"target\": np.random.rand(10)             # Target should be preserved\n    })\n\n    cleaned_df = clean_features(df)\n\n    assert cleaned_df.shape[1] == 6  # 5 good features + 1 target\n    assert \"nan_1\" not in cleaned_df.columns\n    assert \"zero_var_1\" not in cleaned_df.columns\n    assert \"good_1\" in cleaned_df.columns\n    assert \"target\" in cleaned_df.columns\n\n\ndef test_validation_report_json():\n    \"\"\"\n    4. Generate report from mock ModelResults. \n    Verify it's JSON-serializable (json.dumps(report) doesn't raise).\n    \"\"\"\n    results = [\n        ModelResult(\n            model_name=\"ridge\",\n            model=Ridge(),\n            brier_score=0.159,\n            hit_rate=0.67,\n            mse=0.023,\n            r2=0.34,\n            feature_importance={\"poll_delta_3d\": 0.45},\n            calibration_data=[{\"bin_center\": 0.1, \"predicted_mean\": 0.12, \"actual_mean\": 0.09, \"count\": 23}],\n            n_train=1000,\n            n_test=200\n        )\n    ]\n    report = generate_validation_report(results)\n    \n    try:\n        json_str = json.dumps(report)\n    except TypeError:\n        pytest.fail(\"Validation report is not JSON serializable\")\n        \n    assert \"ridge\" in json_str\n    assert \"baseline_brier\" in report\n\n\ndef test_best_model_selection():\n    \"\"\"\n    5. Create 4 ModelResults with Brier scores [0.20, 0.15, 0.18, 0.22]. \n    Verify select_best_model() returns the model with Brier=0.15.\n    \"\"\"\n    results = [\n        ModelResult(model_name=\"m1\", model=Ridge(), brier_score=0.20, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),\n        ModelResult(model_name=\"m2\", model=Ridge(), brier_score=0.15, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),\n        ModelResult(model_name=\"m3\", model=Ridge(), brier_score=0.18, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),\n        ModelResult(model_name=\"m4\", model=Ridge(), brier_score=0.22, hit_rate=0.5, mse=0.1, r2=0.1, feature_importance={}, calibration_data=[], n_train=10, n_test=10),\n    ]\n    best_model, best_name = select_best_model(results)\n    \n    assert best_name == \"m2\"\n    assert best_model == results[1].model\n\n\n@patch(\"app.election.correlation.model_trainer.build_feature_matrix\", create=True)\ndef test_calibration_bins(mock_build_features):\n    \"\"\"\n    6. Generate predictions + actuals. Verify calibration_data has 10 bins, \n    each with bin_center, predicted_mean, actual_mean, count fields.\n    \"\"\"\n    np.random.seed(42)\n    X = np.random.randn(200, 5)\n    y_raw = X[:, 0] * 0.5 + np.random.randn(200) * 0.1\n    y = 1 / (1 + np.exp(-y_raw))\n\n    df = pd.DataFrame(X, columns=[f\"feat_{i}\" for i in range(5)])\n    df[\"target\"] = y\n\n    mock_build_features.side_effect = [df, df]\n\n    db_mock = MagicMock()\n    results = train_models(db_mock, train_cycles=[2020], test_cycle=2024)\n\n    cal_data = results[0].calibration_data\n    assert len(cal_data) > 0\n    assert len(cal_data) <= 10  # Max 10 bins\n\n    for bin_data in cal_data:\n        assert \"bin_center\" in bin_data\n        assert \"predicted_mean\" in bin_data\n        assert \"actual_mean\" in bin_data\n        assert \"count\" in bin_data\n\n\ndef test_baseline_comparison():\n    \"\"\"\n    7. Verify report includes baseline Brier (always predict 0.5 = 0.25) \n    and improvement percentage.\n    \"\"\"\n    results = [\n        ModelResult(\n            model_name=\"ridge\",\n            model=Ridge(),\n            brier_score=0.15,\n            hit_rate=0.65,\n            mse=0.02,\n            r2=0.3,\n            feature_importance={},\n            calibration_data=[],\n            n_train=1000,\n            n_test=200\n        )\n    ]\n    report = generate_validation_report(results)\n    \n    assert \"baseline_brier\" in report\n    assert report[\"baseline_brier\"] == 0.25\n    \n    assert \"improvement_vs_baseline_pct\" in report\n    # (0.25 - 0.15) / 0.25 = 0.40 -> 40.0%\n    assert report[\"improvement_vs_baseline_pct\"] == 40.0\n\n\ndef test_alpha_report_endpoint():\n    \"\"\"\n    8. Mock FastAPI test client. POST /v1/election/alpha-train, \n    then GET /v1/election/alpha-report. Verify report structure.\n    \"\"\"\n    app = FastAPI()\n    app.include_router(router, prefix=\"/v1/election\")\n    client = TestClient(app)\n\n    # Test POST /alpha-train\n    with patch(\"threading.Thread.start\") as mock_thread:\n        post_resp = client.post(\n            \"/v1/election/alpha-train\", \n            json={\"train_cycles\": [2018, 2020, 2022], \"test_cycle\": 2024}\n        )\n        assert post_resp.status_code == 200\n        assert post_resp.json() == {\"status\": \"training triggered\"}\n        mock_thread.assert_called_once()\n\n    # Test GET /alpha-report\n    mock_report = {\n        \"generated_at\": \"2026-04-19T00:00:00\",\n        \"best_model\": \"ridge\",\n        \"baseline_brier\": 0.25,\n        \"improvement_vs_baseline_pct\": 36.4\n    }\n    \n    with patch(\"pathlib.Path.exists\", return_value=True), \\\n         patch(\"pathlib.Path.read_text\", return_value=json.dumps(mock_report)):\n        get_resp = client.get(\"/v1/election/alpha-report\")\n        assert get_resp.status_code == 200\n        \n        data = get_resp.json()\n        assert data[\"best_model\"] == \"ridge\"\n        assert data[\"baseline_brier\"] == 0.25\n        assert data[\"improvement_vs_baseline_pct\"] == 36.4\n--- END FILE ---", "model": "gemini-3.1-pro-preview", "usage": {"prompt_tokens": 2739, "completion_tokens": 3118, "total_tokens": 5857}, "cost_usd": 0.042894}
```
