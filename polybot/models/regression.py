"""
regression.py — Lead-lag ridge regression: fit, predict, q_settled.

The model predicts logit(q_actual) from lagged spot features. The key
output is q_settled: substitute current microprice into every lag slot
to get where Polymarket is heading once it digests current spot.

See CLAUDE.md §3.3, §6 for the full mathematical spec.
"""

import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from polybot.models.features import FEATURE_NAMES, FeatureVec
from polybot.infra.config import RIDGE_ALPHAS, HELDOUT_WINDOW_SECONDS

# Indices of the Binance x_* features that get replaced in q_settled computation
# (x_15 .. x_120 — NOT x_now, which stays as-is)
_SETTLE_INDICES = [FEATURE_NAMES.index(n) for n in
                   ["x_15", "x_30", "x_45", "x_60", "x_90", "x_120",
                    "cb_x_15", "cb_x_30", "cb_x_60"]]
_X_NOW_IDX      = FEATURE_NAMES.index("x_now")
_CB_X_NOW_IDX   = FEATURE_NAMES.index("cb_x_now")


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class ModelDiagnostics:
    """Diagnostics computed at each refit, written to the model_versions table."""
    model_version_id:       str
    asset:                  str
    ts_ns:                  int
    n_train_samples:        int
    training_window_start_ns: int
    ridge_alpha:            float
    r2_in_sample:           float
    r2_cv_mean:             float
    r2_held_out_30min:      float
    coef_alpha:             float        # intercept
    coefs:                  dict[str, float]   # feature_name -> coef
    estimated_lag_seconds:  float
    coef_delta_l2:          float        # L2 change vs previous fit


class RegressionModel:
    """
    Thread-safe ridge regression model with atomic coefficient swap.

    The refitter background task calls fit() which internally swaps
    the live coefficients atomically. The scheduler tick reads
    predict_logit() / q_settled() at any time with no locking needed
    because numpy array assignment is atomic in CPython (GIL).
    """

    FEATURE_NAMES = FEATURE_NAMES

    def __init__(self, asset: str):
        self.asset = asset
        self._lock = threading.Lock()

        # Live model state — replaced atomically on each refit
        self._coefs: Optional[np.ndarray] = None      # shape (n_features,)
        self._intercept: float = 0.0
        self._scaler: Optional[StandardScaler] = None
        self._version_id: str = ""
        self._last_diagnostics: Optional[ModelDiagnostics] = None
        self._prev_coefs: Optional[np.ndarray] = None

        # Sanity gate state (updated after each refit)
        self.r2_cv_mean: float = 0.0
        self.r2_held_out: float = 0.0
        self.last_refit_ns: int = 0
        self.is_fit: bool = False

    # ------------------------------------------------------------------
    # Fitting (called by RegressionRefitter in background task)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,       # shape (n_samples, n_features), already in feature order
        y: np.ndarray,       # shape (n_samples,), logit(q_actual) values
        ts_ns_array: np.ndarray,  # shape (n_samples,), wall-clock ns for each row
        now_ns: int,
    ) -> ModelDiagnostics:
        """
        Fit ridge regression on training data. Atomically swaps live coefficients.
        Returns diagnostics for logging to model_versions table.
        """
        import time as _time

        n = len(y)
        if n < 2:
            raise ValueError(f"too few training samples: {n}")

        # --- Hold out most-recent HELDOUT_WINDOW_SECONDS for validation ---
        # If the configured heldout window would consume everything (early
        # warmup), shrink it so we always keep ~80% for training.
        ts_span_s = (ts_ns_array.max() - ts_ns_array.min()) / 1e9 if len(ts_ns_array) > 0 else 0
        heldout_s = min(HELDOUT_WINDOW_SECONDS, max(60, ts_span_s * 0.20))
        heldout_cutoff_ns = now_ns - int(heldout_s) * 1_000_000_000
        train_mask = ts_ns_array < heldout_cutoff_ns
        val_mask   = ts_ns_array >= heldout_cutoff_ns

        n_train = int(train_mask.sum())
        if n_train < 50:
            # Not enough held-in data — use everything for training
            X_train, y_train = X, y
            X_val,   y_val   = None, None
            n_train_used = len(y)
        else:
            X_train, y_train = X[train_mask], y[train_mask]
            X_val,   y_val   = X[val_mask],   y[val_mask]
            n_train_used = n_train

        # --- Scale features ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val) if X_val is not None else None

        # --- Time-series CV ridge ---
        tscv = TimeSeriesSplit(n_splits=5)
        model = RidgeCV(
            alphas=RIDGE_ALPHAS,
            cv=tscv,
            scoring="r2",
            fit_intercept=True,
        )
        model.fit(X_train_s, y_train)

        r2_in = float(model.score(X_train_s, y_train))
        r2_cv = float(model.best_score_) if hasattr(model, "best_score_") else r2_in

        r2_held = 0.0
        if X_val_s is not None and len(y_val) > 0:
            r2_held = float(model.score(X_val_s, y_val))

        # --- Coefficient delta vs previous fit ---
        new_coefs = model.coef_.copy()
        coef_delta = 0.0
        if self._prev_coefs is not None and len(self._prev_coefs) == len(new_coefs):
            coef_delta = float(np.linalg.norm(new_coefs - self._prev_coefs))

        # --- Estimated lag seconds (β-weighted average of lag horizons) ---
        lag_horizons = [0, 15, 30, 45, 60, 90, 120]
        lag_coef_indices = [FEATURE_NAMES.index(f"x_{l}" if l > 0 else "x_now")
                            for l in lag_horizons]
        lag_coefs = [float(new_coefs[i]) for i in lag_coef_indices]
        sum_lag_coefs = sum(abs(c) for c in lag_coefs[1:])  # exclude x_now
        if sum_lag_coefs > 0:
            estimated_lag = sum(
                abs(lag_coefs[i+1]) * lag_horizons[i+1]
                for i in range(len(lag_horizons) - 1)
            ) / sum_lag_coefs
        else:
            estimated_lag = 0.0

        version_id = str(uuid.uuid4())
        training_start_ns = int(ts_ns_array[0]) if len(ts_ns_array) > 0 else 0

        diag = ModelDiagnostics(
            model_version_id=version_id,
            asset=self.asset,
            ts_ns=now_ns,
            n_train_samples=n_train_used,
            training_window_start_ns=training_start_ns,
            ridge_alpha=float(model.alpha_),
            r2_in_sample=r2_in,
            r2_cv_mean=r2_cv,
            r2_held_out_30min=r2_held,
            coef_alpha=float(model.intercept_),
            coefs={name: float(new_coefs[i]) for i, name in enumerate(FEATURE_NAMES)},
            estimated_lag_seconds=estimated_lag,
            coef_delta_l2=coef_delta,
        )

        # --- Atomic swap ---
        with self._lock:
            self._prev_coefs = self._coefs
            self._coefs      = new_coefs
            self._intercept  = float(model.intercept_)
            self._scaler     = scaler
            self._version_id = version_id
            self._last_diagnostics = diag
            self.r2_cv_mean    = r2_cv
            self.r2_held_out   = r2_held
            self.last_refit_ns = now_ns
            self.is_fit        = True

        return diag

    # ------------------------------------------------------------------
    # Prediction (called by scheduler tick — must be fast, no allocation)
    # ------------------------------------------------------------------

    def predict_logit(self, fv: FeatureVec) -> Optional[float]:
        """
        Predict logit(q) from a FeatureVec using current feature vector.
        Returns None if model not fit or feature vector incomplete.
        """
        if not self.is_fit or not fv.complete:
            return None
        vec = fv.regression_vector()
        if vec is None:
            return None
        coefs, intercept, scaler = self._coefs, self._intercept, self._scaler
        if coefs is None or scaler is None:
            return None
        x = scaler.transform([vec])[0]
        return float(np.dot(coefs, x) + intercept)

    def q_predicted(self, fv: FeatureVec) -> Optional[float]:
        """
        What should Polymarket be quoting NOW given the lagged history?
        Sanity check: should track q_actual closely.
        """
        logit = self.predict_logit(fv)
        return _sigmoid(logit) if logit is not None else None

    def q_settled(self, fv: FeatureVec) -> Optional[float]:
        """
        Where will Polymarket quote once it digests CURRENT spot?
        Substitute x_now into every lagged slot, re-predict.
        This is the primary output used for edge calculation.
        """
        if not self.is_fit or not fv.complete:
            return None
        vec = fv.regression_vector()
        if vec is None:
            return None

        # Replace all lagged Binance + Coinbase slots with current values
        settled = list(vec)
        x_now_val  = settled[_X_NOW_IDX]
        cb_now_val = settled[_CB_X_NOW_IDX]
        for idx in _SETTLE_INDICES:
            # Binance lagged slots get x_now; Coinbase lagged slots get cb_x_now
            fname = FEATURE_NAMES[idx]
            settled[idx] = cb_now_val if fname.startswith("cb_") else x_now_val

        coefs, intercept, scaler = self._coefs, self._intercept, self._scaler
        if coefs is None or scaler is None:
            return None
        x = scaler.transform([settled])[0]
        return _sigmoid(float(np.dot(coefs, x) + intercept))

    @property
    def version_id(self) -> str:
        return self._version_id
