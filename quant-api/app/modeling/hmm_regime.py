"""
T8 — HMM Regime-Aware Alpha Stacking
======================================
Two-state Hidden Markov Model for regime detection:
  - State 0 (risk-on):  contango, low VIX → trending/momentum regime
  - State 1 (risk-off): backwardation, high VIX → mean-reversion/defensive regime

Per-regime Ridge models produce separate predictions.
Final signal = p(risk_on) × pred_risk_on + p(risk_off) × pred_risk_off

Observables:
  - macro_risk_off_regime  (from MacroRegimeProvider)
  - brent_contango_ret     (from _build_feature_frame T4)

Uses pure numpy Baum-Welch EM by default; optional hmmlearn backend.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)

# Try hmmlearn for better numerical stability; fall back to pure numpy
try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False


# ---------------------------------------------------------------------------
# Pure-numpy 2-state Gaussian HMM (Baum-Welch EM)
# ---------------------------------------------------------------------------

class _SimpleGaussianHMM:
    """
    Minimal 2-state Gaussian HMM with diagonal covariance.
    Fits via Baum-Welch EM. No external dependencies.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 30, tol: float = 1e-4):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        # Parameters (initialized in fit)
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.covars_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_SimpleGaussianHMM":
        """Fit the HMM to observation sequences X (T × D)."""
        T, D = X.shape
        K = self.n_states

        # Initialize with kmeans-ish split
        sorted_idx = np.argsort(X[:, 0])
        split = T // K
        self.means_ = np.array([X[sorted_idx[i * split:(i + 1) * split]].mean(axis=0) for i in range(K)])
        self.covars_ = np.array([np.var(X, axis=0) + 1e-6 for _ in range(K)])
        self.startprob_ = np.ones(K) / K
        self.transmat_ = np.full((K, K), 1.0 / K)
        # Add persistence bias (regimes tend to persist)
        for i in range(K):
            self.transmat_[i, i] = 0.85
            for j in range(K):
                if j != i:
                    self.transmat_[i, j] = 0.15 / (K - 1)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step: forward-backward
            log_lik = self._log_likelihood(X)  # T × K
            alpha, scale = self._forward(log_lik)
            beta = self._backward(log_lik, scale)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = alpha[t, i] * self.transmat_[i, j] * np.exp(log_lik[t + 1, j]) * beta[t + 1, j]
                xi[t] /= xi[t].sum() + 1e-300

            # M-step
            self.startprob_ = gamma[0] / (gamma[0].sum() + 1e-300)
            for i in range(K):
                self.transmat_[i] = xi[:, i, :].sum(axis=0)
                self.transmat_[i] /= self.transmat_[i].sum() + 1e-300

                w = gamma[:, i]
                w_sum = w.sum() + 1e-300
                self.means_[i] = (w[:, None] * X).sum(axis=0) / w_sum
                diff = X - self.means_[i]
                self.covars_[i] = (w[:, None] * diff * diff).sum(axis=0) / w_sum + 1e-6

            # Check convergence
            ll = np.sum(np.log(scale + 1e-300))
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities (T × K)."""
        log_lik = self._log_likelihood(X)
        alpha, scale = self._forward(log_lik)
        beta = self._backward(log_lik, scale)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return most likely state sequence (T,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities (T × K)."""
        T, D = X.shape
        K = self.n_states
        log_lik = np.zeros((T, K))
        for k in range(K):
            diff = X - self.means_[k]
            var = self.covars_[k]
            log_lik[:, k] = -0.5 * np.sum(diff * diff / var + np.log(2 * np.pi * var), axis=1)
        return log_lik

    def _forward(self, log_lik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass with scaling."""
        T, K = log_lik.shape
        alpha = np.zeros((T, K))
        scale = np.zeros(T)
        alpha[0] = self.startprob_ * np.exp(log_lik[0])
        scale[0] = alpha[0].sum() + 1e-300
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * np.exp(log_lik[t])
            scale[t] = alpha[t].sum() + 1e-300
            alpha[t] /= scale[t]
        return alpha, scale

    def _backward(self, log_lik: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """Backward pass."""
        T, K = log_lik.shape
        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.transmat_ @ (np.exp(log_lik[t + 1]) * beta[t + 1])
            beta[t] /= scale[t + 1] + 1e-300
        return beta


# ---------------------------------------------------------------------------
# Regime-Aware Ridge Stacking
# ---------------------------------------------------------------------------

class HMMRegimeModel:
    """
    2-state HMM regime-aware alpha stacking model.

    Observables:
      - macro_risk_off_regime (0-1 from MacroRegimeProvider)
      - brent_contango_ret (contango/backwardation signal from T4)

    Two Ridge models (one per regime). Final prediction is the
    regime-probability-weighted blend:
        pred = p(risk_on) × Ridge_risk_on(X) + p(risk_off) × Ridge_risk_off(X)

    Usage:
        hmm = HMMRegimeModel()
        result = hmm.fit_predict(feature_frame, target_col="y_fwd_20d")
        # result["latest_pred"] = blended prediction
        # result["regime_probs"] = [p_risk_on, p_risk_off]
        # result["regime_label"] = "risk_on" or "risk_off"
    """

    REGIME_LABELS = {0: "risk_on", 1: "risk_off"}

    def __init__(
        self,
        observable_cols: tuple[str, ...] = ("macro_risk_off_regime", "brent_contango_ret"),
        ridge_alpha: float = 1.0,
        hmm_n_iter: int = 30,
        min_samples: int = 80,
    ):
        self.observable_cols = list(observable_cols)
        self.ridge_alpha = ridge_alpha
        self.hmm_n_iter = hmm_n_iter
        self.min_samples = min_samples
        self._hmm = None
        self._ridge_models: dict[int, Ridge] = {}

    def fit_predict(
        self,
        frame: pd.DataFrame,
        target_col: str = "y_fwd_20d",
        lookback: int = 150,
    ) -> dict[str, Any] | None:
        """
        Fit HMM on observables, train per-regime Ridge models,
        and produce regime-weighted prediction for the latest row.
        """
        feature_cols = [
            c for c in frame.columns
            if c not in {target_col, "stock_px", "stock_vol"}
            and c not in self.observable_cols
        ]
        if not feature_cols or len(frame) < self.min_samples:
            return None

        # Build observable matrix for HMM
        obs_cols_available = [c for c in self.observable_cols if c in frame.columns]
        if not obs_cols_available:
            # Fallback: use brent vol regime as sole observable
            if "high_vol_regime" in frame.columns:
                obs_cols_available = ["high_vol_regime"]
            else:
                return None

        obs_matrix = frame[obs_cols_available].fillna(0.0).values
        if obs_matrix.shape[0] < self.min_samples:
            return None

        # Fit HMM
        if _HAS_HMMLEARN:
            hmm = GaussianHMM(
                n_components=2,
                covariance_type="diag",
                n_iter=self.hmm_n_iter,
                random_state=42,
            )
            hmm.fit(obs_matrix)
            regime_probs = hmm.predict_proba(obs_matrix)  # T × 2
        else:
            hmm = _SimpleGaussianHMM(n_states=2, n_iter=self.hmm_n_iter)
            hmm.fit(obs_matrix)
            regime_probs = hmm.predict_proba(obs_matrix)

        self._hmm = hmm

        # Label regimes: the state with higher mean risk_off_regime = risk_off (state 1)
        if len(obs_cols_available) > 0:
            state_means = np.array([
                regime_probs[:, k].dot(obs_matrix[:, 0]) / (regime_probs[:, k].sum() + 1e-10)
                for k in range(2)
            ])
            # Higher observable mean = more risk-off
            if state_means[0] > state_means[1]:
                # Swap so state 0 = risk-on, state 1 = risk-off
                regime_probs = regime_probs[:, ::-1]

        # Train per-regime Ridge models (walk-forward)
        X = frame[feature_cols].values
        y = frame[target_col].values

        regime_preds = np.zeros((len(frame), 2))
        latest_models: dict[int, Ridge | None] = {0: None, 1: None}

        for regime_id in range(2):
            weights = regime_probs[:, regime_id]
            preds = []
            for i in range(lookback, len(frame)):
                train_slice = slice(i - lookback, i)
                w = weights[train_slice]
                w_sum = w.sum()
                if w_sum < 5.0:
                    # Not enough weight in this regime for this window
                    preds.append(0.0)
                    continue

                ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
                ridge.fit(X[train_slice], y[train_slice], sample_weight=w)
                pred = float(ridge.predict(X[i:i + 1])[0])
                preds.append(pred)
                latest_models[regime_id] = ridge

            # Pad early rows with 0
            full_preds = np.zeros(len(frame))
            full_preds[lookback:] = preds
            regime_preds[:, regime_id] = full_preds

        self._ridge_models = {k: v for k, v in latest_models.items() if v is not None}

        # Blended prediction: p(regime_0) × pred_0 + p(regime_1) × pred_1
        blended = (regime_probs * regime_preds).sum(axis=1)

        # Latest row results
        latest_probs = regime_probs[-1]
        latest_regime = int(np.argmax(latest_probs))
        latest_pred = float(blended[-1])

        # Walk-forward performance metrics (on blended signal)
        valid_mask = np.arange(len(frame)) >= lookback
        valid_y = y[valid_mask]
        valid_blend = blended[valid_mask]
        # Remove NaN targets (forward-looking target not available for last 20 rows)
        finite_mask = np.isfinite(valid_y) & np.isfinite(valid_blend)
        if finite_mask.sum() > 10:
            vy = valid_y[finite_mask]
            vb = valid_blend[finite_mask]
            hit_rate = float(np.mean(np.sign(vb) == np.sign(vy)))
            mae = float(np.mean(np.abs(vb - vy)))
            corr = float(np.corrcoef(vb, vy)[0, 1]) if len(vb) > 3 else 0.0
            if np.isnan(corr):
                corr = 0.0
        else:
            hit_rate = 0.5
            mae = 0.1
            corr = 0.0

        # Regime history summary
        regime_sequence = np.argmax(regime_probs, axis=1)
        risk_on_pct = float(np.mean(regime_sequence == 0))

        return {
            "latest_pred": latest_pred,
            "regime_probs": latest_probs.tolist(),  # [p_risk_on, p_risk_off]
            "regime_label": self.REGIME_LABELS.get(latest_regime, "unknown"),
            "regime_confidence": float(latest_probs[latest_regime]),
            "risk_on_pct": risk_on_pct,
            "hit_rate": hit_rate,
            "mae": mae,
            "correlation": corr,
            "n_samples": int(finite_mask.sum()) if finite_mask.sum() > 0 else 0,
            "per_regime_preds": {
                "risk_on": float(regime_preds[-1, 0]),
                "risk_off": float(regime_preds[-1, 1]),
            },
            "observable_cols": obs_cols_available,
            "backend": "hmmlearn" if _HAS_HMMLEARN else "numpy",
        }

    @property
    def current_regime(self) -> str:
        """Return the current regime label."""
        if self._hmm is None:
            return "unknown"
        return "fitted"

    def regime_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Return a summary of regime statistics for the given feature frame."""
        result = self.fit_predict(frame)
        if result is None:
            return {"error": "insufficient data"}
        return {
            "current_regime": result["regime_label"],
            "regime_confidence": result["regime_confidence"],
            "risk_on_pct_history": result["risk_on_pct"],
            "regime_hit_rate": result["hit_rate"],
            "blended_vs_regime_preds": result["per_regime_preds"],
        }
