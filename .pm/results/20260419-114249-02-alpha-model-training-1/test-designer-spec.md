# Test Design Request

You are a TEST DESIGNER. Generate test cases from the spec below.
You have NOT seen any implementation. Write tests that verify the SPECIFICATION.
Use the testing framework mentioned in the spec.
Output ONLY test file(s) using --- FILE: path --- / --- END FILE --- format.

## Specification
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

