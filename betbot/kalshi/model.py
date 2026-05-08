"""
model.py - LightGBM multi-horizon prediction model for Kalshi lead-lag arb.

Trains one LGBMRegressor per forecast horizon (5s, 10s, 15s, 60s ahead).
Target for each model: logit(yes_bid_{t + h_seconds}) — the future bid price,
so edge = predicted_future_bid - current_ask reflects actual fill prices.

q_settled() returns the primary-horizon prediction at the current feature
vector, expressed as a probability. This is the edge signal the decision
loop uses.

Workflow:
  1. Dry run the bot to collect ticks (no live orders, no model needed)
  2. scripts/train_model.py        — fit LGBM on collected ticks, save to model_fits/
  3. scripts/tune_trading_knobs.py — grid-search Kelly tiers and exit thresholds
  4. scripts/test_all.py           — final simulation with model + tuned config
  5. scripts/run/run_kalshi_bot.py --model-file model_fits/<name>.pkl --live-orders
"""

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from betbot.kalshi.features import (
    FEATURE_NAMES, N_FEATURES, FeatureVec,
    _sigmoid, _logit,
)
from betbot.kalshi.config import (
    HELDOUT_FRACTION,
    LGBM_FORECAST_HORIZONS, LGBM_PRIMARY_HORIZON,
)


def _clip_r2(r2: float) -> float:
    if not np.isfinite(r2):
        return -10.0
    return max(-10.0, min(1.0, float(r2)))


@dataclass
class ModelDiagnostics:
    version_id:       str
    ts_ns:            int
    n_train:          int
    ridge_alpha:      float   # always 0.0 (kept for log-format compat)
    r2_in_sample:     float
    r2_cv:            float
    r2_held_out:      float
    intercept:        float   # always 0.0
    coefs:            dict    # horizon -> r2_hld, e.g. {"h5s_r2_hld": 0.32, ...}
    estimated_lag_s:  float
    coef_delta_l2:    float


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _heldout_split(X: np.ndarray, y: np.ndarray, ts_ns: np.ndarray):
    """Time-based hold-out: last HELDOUT_FRACTION of span."""
    span_ns   = ts_ns.max() - ts_ns.min()
    cutoff_ns = ts_ns.max() - int(span_ns * HELDOUT_FRACTION)
    train     = ts_ns <= cutoff_ns
    return train, ~train, int(train.sum())


# ---------------------------------------------------------------------------
# LightGBM multi-horizon model
# ---------------------------------------------------------------------------

class LGBMModel:
    """
    One LightGBM regressor per forecast horizon.

    Training:
        fit(X, y_multi, ts_ns)
          X:       (n, N_FEATURES)
          y_multi: (n, n_horizons) — logit(yes_mid_{t+h}) per column

    Inference:
        q_settled(fv)  -> probability (primary horizon prediction)
        q_at_horizon(fv, h_s) -> probability for a specific horizon
    """

    def __init__(self, horizons: list[int] = None, primary_horizon: int = None):
        import lightgbm as lgb
        self._lgb = lgb

        self._horizons    = list(horizons or LGBM_FORECAST_HORIZONS)
        self._primary_h   = primary_horizon or LGBM_PRIMARY_HORIZON
        self._primary_idx = self._horizons.index(self._primary_h)

        self._models:  list[Optional[object]] = [None] * len(self._horizons)
        self._scalers: list[Optional[StandardScaler]] = [None] * len(self._horizons)
        self._version_id: str = ""
        self._r2s_hld: list[float] = [0.0] * len(self._horizons)

        self._lock = threading.Lock()

        self.is_fit:          bool  = False
        self.r2_cv:           float = 0.0
        self.r2_held_out:     float = 0.0
        self.last_refit_ns:   int   = 0
        self.estimated_lag_s: float = float(self._primary_h)
        self.last_diag: Optional[ModelDiagnostics] = None

        self._lgb_params = {
            "objective":         "regression",
            "metric":            "rmse",
            "n_estimators":      200,
            "learning_rate":     0.05,
            "num_leaves":        31,
            "min_child_samples": 20,
            "subsample":         0.8,
            "colsample_bytree":  0.8,
            "verbose":           -1,
            "n_jobs":            1,
            "num_threads":       1,
            "verbosity":         -1,
        }

    # ---- Pickle support (strip/restore threading.Lock) ----

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state.pop("_lgb", None)
        return state

    def __setstate__(self, state):
        import lightgbm as lgb
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._lgb  = lgb
        # Ensure attributes exist for models loaded from older pkl files
        if not hasattr(self, "is_fit"):          self.is_fit          = False
        if not hasattr(self, "r2_cv"):           self.r2_cv           = 0.0
        if not hasattr(self, "r2_held_out"):     self.r2_held_out     = 0.0
        if not hasattr(self, "last_refit_ns"):   self.last_refit_ns   = 0
        if not hasattr(self, "estimated_lag_s"): self.estimated_lag_s = float(self._primary_h)
        if not hasattr(self, "last_diag"):       self.last_diag       = None

        # Strip stale `verbose` param baked into older pkl models — it conflicts
        # with `verbosity=-1` and prints "verbose=50 will be ignored" on every predict.
        for mdl in (self._models or []):
            if mdl is None:
                continue
            try:
                params = mdl.get_params()
                if params.get("verbose") not in (None, -1):
                    mdl.set_params(verbose=-1)
                if hasattr(mdl, "_other_params"):
                    mdl._other_params.pop("verbose", None)
            except Exception:
                pass

    # ---- Snapshot / restore for fit_if_better ----

    def _snapshot(self) -> dict:
        with self._lock:
            return dict(
                models   = list(self._models),
                scalers  = list(self._scalers),
                r2_hld   = self.r2_held_out,
                r2_cv    = self.r2_cv,
                r2s_hld  = list(self._r2s_hld),
                lag      = self.estimated_lag_s,
                diag     = self.last_diag,
                refit_ns = self.last_refit_ns,
            )

    def _restore(self, snap: dict) -> None:
        with self._lock:
            self._models         = snap["models"]
            self._scalers        = snap["scalers"]
            self.r2_held_out     = snap["r2_hld"]
            self.r2_cv           = snap["r2_cv"]
            self._r2s_hld        = snap["r2s_hld"]
            self.estimated_lag_s = snap["lag"]
            self.last_diag       = snap["diag"]
            self.last_refit_ns   = snap["refit_ns"]

    # ---- Training ----

    def fit(self, X: np.ndarray, y: np.ndarray, ts_ns: np.ndarray) -> ModelDiagnostics:
        """
        X:     (n, N_FEATURES)
        y:     (n, n_horizons) — each column is logit(yes_bid_{t+h}) for one horizon
        ts_ns: (n,) nanosecond timestamps of the feature rows
        """
        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_horizons = y.shape[1]
        if n_horizons != len(self._horizons):
            raise ValueError(
                f"y has {n_horizons} columns but model has {len(self._horizons)} horizons"
            )

        now_ns = time.time_ns()
        train_mask, val_mask, n_train = _heldout_split(X, y[:, 0], ts_ns)

        new_models:  list = []
        new_scalers: list = []
        r2s_in, r2s_hld  = [], []

        for col_idx in range(n_horizons):
            h_label = f"h{self._horizons[col_idx]}s"
            print(f"  Fitting horizon {col_idx+1}/{n_horizons} ({h_label})...",
                  end=" ", flush=True)

            yc    = y[:, col_idx]
            yc_tr = yc[train_mask]
            yc_va = yc[val_mask]

            scaler = StandardScaler()
            Xtr_s  = scaler.fit_transform(X[train_mask])
            Xva_s  = scaler.transform(X[val_mask]) if val_mask.sum() > 0 else None

            params = dict(self._lgb_params)
            mdl = self._lgb.LGBMRegressor(**params)

            callbacks = [self._lgb.log_evaluation(period=-1)]
            if Xva_s is not None:
                callbacks.insert(0, self._lgb.early_stopping(20, verbose=False))

            import pandas as _pd
            Xtr_df = _pd.DataFrame(Xtr_s, columns=FEATURE_NAMES)
            Xva_df = _pd.DataFrame(Xva_s, columns=FEATURE_NAMES) if Xva_s is not None else None

            mdl.fit(
                Xtr_df, yc_tr,
                eval_set=[(Xva_df, yc_va)] if Xva_df is not None else None,
                callbacks=callbacks,
            )
            print(f"  Fitted {h_label}: {mdl.n_estimators_} trees", flush=True)

            r2_in  = _clip_r2(float(1 - np.mean((yc_tr - mdl.predict(Xtr_s))**2) /
                                    max(1e-9, np.var(yc_tr))))
            if Xva_s is not None and len(yc_va) > 0:
                r2_hld = _clip_r2(float(1 - np.mean((yc_va - mdl.predict(Xva_s))**2) /
                                        max(1e-9, np.var(yc_va))))
            else:
                r2_hld = 0.0

            new_models.append(mdl)
            new_scalers.append(scaler)
            r2s_in.append(r2_in)
            r2s_hld.append(r2_hld)

        r2_hld_primary = r2s_hld[self._primary_idx]
        r2_in_primary  = r2s_in[self._primary_idx]
        version_id     = str(uuid.uuid4())

        diag = ModelDiagnostics(
            version_id=version_id, ts_ns=now_ns,
            n_train=n_train if n_train >= 30 else len(y),
            ridge_alpha=0.0, intercept=0.0,
            r2_in_sample=r2_in_primary, r2_cv=r2_in_primary, r2_held_out=r2_hld_primary,
            coefs={f"h{h}s_r2_hld": r2s_hld[i] for i, h in enumerate(self._horizons)},
            estimated_lag_s=float(self._primary_h),
            coef_delta_l2=0.0,
        )

        with self._lock:
            self._models         = new_models
            self._scalers        = new_scalers
            self._r2s_hld        = r2s_hld
            self._version_id     = version_id
            self.is_fit          = True
            self.r2_cv           = r2_in_primary
            self.r2_held_out     = r2_hld_primary
            self.last_refit_ns   = now_ns
            self.estimated_lag_s = float(self._primary_h)
            self.last_diag       = diag

        return diag

    def fit_if_better(self, X: np.ndarray, y: np.ndarray,
                      ts_ns: np.ndarray) -> tuple[ModelDiagnostics, bool]:
        """Fit a candidate; keep only if R2_hld improves (or first fit)."""
        with self._lock:
            snap_r2_hld = self.r2_held_out
            first_fit   = not self.is_fit
            snap_state  = self._snapshot()

        diag = self.fit(X, y, ts_ns)

        if first_fit or diag.r2_held_out > snap_r2_hld:
            return diag, True

        self._restore(snap_state)
        return diag, False

    # ---- Inference ----

    def _predict_horizon(self, vec: np.ndarray, horizon_idx: int) -> Optional[float]:
        import os, pandas as _pd
        mdl    = self._models[horizon_idx]
        scaler = self._scalers[horizon_idx]
        if mdl is None or scaler is None:
            return None
        x_scaled = scaler.transform([vec])[0]
        x_df = _pd.DataFrame([x_scaled], columns=FEATURE_NAMES)
        # Suppress LightGBM's C++ stderr warnings (verbosity conflict) at the fd level
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull_fd, 2)
        try:
            result = float(mdl.predict(x_df)[0])
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull_fd)
        return result

    def q_settled(self, fv: "FeatureVec") -> Optional[float]:
        """Primary edge signal: probability at primary horizon."""
        if not self.is_fit or not fv.complete:
            return None
        logit = self._predict_horizon(fv.as_array(), self._primary_idx)
        return _sigmoid(logit) if logit is not None else None

    def q_predicted(self, fv: "FeatureVec") -> Optional[float]:
        """Alias for q_settled — LGBM predicts future directly, no lag substitution."""
        return self.q_settled(fv)

    def q_settled_from_array(self, vec: np.ndarray) -> Optional[float]:
        """q_settled from a raw pre-built numpy feature array (analysis scripts)."""
        if not self.is_fit:
            return None
        logit = self._predict_horizon(vec, self._primary_idx)
        return _sigmoid(logit) if logit is not None else None

    def q_all_horizons_from_array(self, vec: np.ndarray) -> dict[int, float]:
        """Returns {horizon_s: probability} for all fitted horizons. Skips None predictions."""
        if not self.is_fit:
            return {}
        result = {}
        for idx, h in enumerate(self._horizons):
            logit = self._predict_horizon(vec, idx)
            if logit is not None:
                result[h] = _sigmoid(logit)
        return result

    def q_at_horizon(self, fv: "FeatureVec", horizon_s: int) -> Optional[float]:
        """Prediction at a specific horizon in seconds."""
        if not self.is_fit or not fv.complete:
            return None
        if horizon_s not in self._horizons:
            return None
        logit = self._predict_horizon(fv.as_array(), self._horizons.index(horizon_s))
        return _sigmoid(logit) if logit is not None else None

    def stale_s(self) -> float:
        if self.last_refit_ns == 0:
            return float("inf")
        return (time.time_ns() - self.last_refit_ns) / 1e9

    @property
    def version_id(self) -> str:
        return self._version_id

    @property
    def horizons(self) -> list[int]:
        return list(self._horizons)

    @property
    def r2s_by_horizon(self) -> dict[int, float]:
        return {h: self._r2s_hld[i] for i, h in enumerate(self._horizons)}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

import pickle as _pickle
import json as _json


def save_model(model: LGBMModel, path) -> None:
    """
    Save a trained model to disk.
      <path>.pkl  — full model object (pickle)
      <path>.json — human-readable metadata
    """
    from pathlib import Path as _Path
    p = _Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    pkl_path = str(p) + ".pkl" if not str(p).endswith(".pkl") else str(p)
    with open(pkl_path, "wb") as f:
        _pickle.dump(model, f, protocol=_pickle.HIGHEST_PROTOCOL)

    meta = {
        "model_type":      "LGBMModel",
        "is_fit":          model.is_fit,
        "r2_held_out":     model.r2_held_out,
        "estimated_lag_s": model.estimated_lag_s,
        "horizons":        model.horizons,
        "primary_horizon": model._primary_h,
        "r2s_by_horizon":  model.r2s_by_horizon,
        "n_samples":       model.last_diag.n_train if model.last_diag else None,
    }
    if model.last_diag is not None:
        d = model.last_diag
        meta["diag"] = {
            "n_train": d.n_train,
            "r2_in":   d.r2_in_sample,
            "r2_hld":  d.r2_held_out,
            "coefs":   d.coefs,
        }

    json_path = str(p.with_suffix("")) + ".json" if str(p).endswith(".pkl") else str(p) + ".json"
    with open(json_path, "w") as f:
        _json.dump(meta, f, indent=2)

    base = pkl_path[:-4] if pkl_path.endswith(".pkl") else pkl_path
    print(f"  Saved model -> {base}.pkl  (LGBMModel  R2_hld={model.r2_held_out:.3f})",
          flush=True)


def load_model(path) -> LGBMModel:
    """Load a model saved with save_model(). Returns ready-to-predict LGBMModel."""
    from pathlib import Path as _Path
    p   = _Path(str(path))
    pkl = str(p) if str(p).endswith(".pkl") else str(p) + ".pkl"
    with open(pkl, "rb") as f:
        model = _pickle.load(f)
    print(f"  Loaded model <- {pkl}  (LGBMModel  R2_hld={model.r2_held_out:.3f})",
          flush=True)
    return model


def make_model() -> LGBMModel:
    """Return a fresh, unfitted LGBMModel."""
    return LGBMModel()
