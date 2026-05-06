"""
model.py - Ridge regression model for Kalshi lead-lag arbitrage.

Fits logit(kalshi_yes_mid) ~ f(lagged_coinbase_microprice, tau, momentum, ...).

q_settled(): substitute current microprice into all lag slots to predict
where Kalshi will quote once it finishes digesting the current spot move.
That gap between q_settled and q_ask is the raw edge.
"""

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from betbot.kalshi.features import (
    FEATURE_NAMES, N_FEATURES, FeatureVec,
    _sigmoid, _logit,
)
from betbot.kalshi.config import RIDGE_ALPHAS, HELDOUT_FRACTION


def _clip_r2(r2: float) -> float:
    """
    Clip an R^2 to a sane range. R^2 can legitimately be negative (worse than
    mean baseline), but when held-out target variance is near zero (e.g. early
    in a run when yes_mid is flat), the formula 1 - SSE/SST explodes to values
    like -1e30. Clipping to [-10, 1] preserves the "model is bad" signal
    without producing unreadable garbage in logs.
    """
    if not np.isfinite(r2):
        return -10.0
    return max(-10.0, min(1.0, float(r2)))


@dataclass
class ModelDiagnostics:
    version_id:           str
    ts_ns:                int
    n_train:              int
    ridge_alpha:          float
    r2_in_sample:         float
    r2_cv:                float
    r2_held_out:          float
    intercept:            float
    coefs:                dict   # feature_name -> float
    estimated_lag_s:      float  # beta-weighted average lag (diagnostic only)
    coef_delta_l2:        float  # L2 norm of coef change vs previous fit


class KalshiRegressionModel:
    """
    Thread-safe ridge regression.  The background refitter calls fit();
    the decision loop calls q_settled() - no locking needed in CPython
    because numpy array assignment is atomic under the GIL.
    """

    def __init__(self):
        self._lock = threading.Lock()

        self._coefs:     Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._scaler:    Optional[StandardScaler] = None
        self._version_id: str = ""
        self._prev_coefs: Optional[np.ndarray] = None

        # Publicly readable diagnostics
        self.is_fit:        bool  = False
        self.r2_cv:         float = 0.0
        self.r2_held_out:   float = 0.0
        self.last_refit_ns: int   = 0
        self.estimated_lag_s: float = 0.0
        self.last_diag: Optional[ModelDiagnostics] = None

    # ------------------------------------------------------------------
    # Fitting - called by the background refitter
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            ts_ns: np.ndarray) -> ModelDiagnostics:
        """
        Fit ridge regression on (X, y) with time-series CV.
        X shape: (n_samples, N_FEATURES)
        y shape: (n_samples,)  - logit(kalshi_yes_mid)
        ts_ns: (n_samples,) wall-clock receipt time in nanoseconds
        """
        n = len(y)
        if n < 2:
            raise ValueError(f"need >= 2 samples, got {n}")

        now_ns = time.time_ns()

        # Hold-out: most-recent HELDOUT_FRACTION of the time span
        span_ns = ts_ns.max() - ts_ns.min()
        cutoff_ns = ts_ns.max() - int(span_ns * HELDOUT_FRACTION)
        train_mask = ts_ns <= cutoff_ns
        val_mask   = ts_ns >  cutoff_ns

        n_train = int(train_mask.sum())
        if n_train < 30:
            X_train, y_train = X, y
            X_val,   y_val   = None, None
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val,   y_val   = X[val_mask],   y[val_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_va_s = scaler.transform(X_val) if X_val is not None else None

        tscv = TimeSeriesSplit(n_splits=min(5, max(2, n_train // 50)))
        mdl  = RidgeCV(alphas=RIDGE_ALPHAS, cv=tscv, scoring="r2",
                       fit_intercept=True)
        mdl.fit(X_tr_s, y_train)

        r2_in  = _clip_r2(mdl.score(X_tr_s, y_train))
        r2_cv  = _clip_r2(getattr(mdl, "best_score_", r2_in))
        r2_hld = _clip_r2(mdl.score(X_va_s, y_val)) if X_va_s is not None and len(y_val) > 0 else 0.0

        new_coefs = mdl.coef_.copy()

        # beta-weighted average lag - which historical slot explains Kalshi best
        lag_names = ["x_15", "x_30", "x_60", "x_90", "x_120"]
        lag_secs  = [15,      30,     60,     90,     120]
        lag_idx   = [FEATURE_NAMES.index(n) for n in lag_names]
        lag_betas = [abs(float(new_coefs[i])) for i in lag_idx]
        total_beta = sum(lag_betas)
        est_lag = (sum(b * s for b, s in zip(lag_betas, lag_secs)) / total_beta
                   if total_beta > 0 else 0.0)

        prev = self._prev_coefs
        coef_delta = float(np.linalg.norm(new_coefs - prev)) if (
            prev is not None and len(prev) == len(new_coefs)) else 0.0

        version_id = str(uuid.uuid4())
        diag = ModelDiagnostics(
            version_id=version_id,
            ts_ns=now_ns,
            n_train=n_train if n_train >= 30 else n,
            ridge_alpha=float(mdl.alpha_),
            r2_in_sample=r2_in,
            r2_cv=r2_cv,
            r2_held_out=r2_hld,
            intercept=float(mdl.intercept_),
            coefs={name: float(new_coefs[i]) for i, name in enumerate(FEATURE_NAMES)},
            estimated_lag_s=est_lag,
            coef_delta_l2=coef_delta,
        )

        with self._lock:
            self._prev_coefs    = self._coefs
            self._coefs         = new_coefs
            self._intercept     = float(mdl.intercept_)
            self._scaler        = scaler
            self._version_id    = version_id
            self.is_fit         = True
            self.r2_cv          = r2_cv
            self.r2_held_out    = r2_hld
            self.last_refit_ns  = now_ns
            self.estimated_lag_s = est_lag
            self.last_diag      = diag

        return diag

    # ------------------------------------------------------------------
    # Prediction - called by scheduler tick (must be fast)
    # ------------------------------------------------------------------

    def _predict_raw(self, vec: np.ndarray) -> Optional[float]:
        coefs, intercept, scaler = self._coefs, self._intercept, self._scaler
        if coefs is None or scaler is None:
            return None
        x = scaler.transform([vec])[0]
        return float(np.dot(coefs, x) + intercept)

    def q_predicted(self, fv: FeatureVec) -> Optional[float]:
        """
        What should Kalshi be quoting NOW given lagged spot history?
        Sanity check - should track yes_mid closely when model is healthy.
        """
        if not self.is_fit or not fv.complete:
            return None
        logit = self._predict_raw(fv.as_array())
        return _sigmoid(logit) if logit is not None else None

    def q_settled(self, fv: FeatureVec) -> Optional[float]:
        """
        Where will Kalshi quote once it has digested current spot?
        Uses settled_array() which substitutes x_0 into all lag slots.
        This is the primary signal for edge calculation.
        """
        if not self.is_fit or not fv.complete:
            return None
        logit = self._predict_raw(fv.settled_array())
        return _sigmoid(logit) if logit is not None else None

    @property
    def version_id(self) -> str:
        return self._version_id

    def stale_s(self) -> float:
        """Seconds since last refit."""
        if self.last_refit_ns == 0:
            return float("inf")
        return (time.time_ns() - self.last_refit_ns) / 1e9
