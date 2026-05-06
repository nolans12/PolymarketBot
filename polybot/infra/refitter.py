"""
refitter.py — Background task that refits the regression every 5 minutes.

Reads the last 4 hours of decision rows from the ParquetWriter's in-memory
buffer + flushed parquet files, builds (X, y) training matrices, fits the
ridge regression, and atomically swaps the coefficients into the live model.

Refit diagnostics are written to the model_versions parquet table.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from polybot.infra.config import (
    ASSETS, REFIT_INTERVAL_SECONDS, TRAINING_WINDOW_SECONDS,
    MIN_TRAIN_SIZE, PARQUET_DIR,
)
from polybot.models.features import FEATURE_NAMES
from polybot.models.regression import RegressionModel

if TYPE_CHECKING:
    from polybot.infra.parquet_writer import ParquetWriter

logger = logging.getLogger(__name__)


def _logit(p: float) -> float:
    import math
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


# Map feature names to the column names they correspond to in the decisions table
_FEAT_TO_COL = {
    "x_now":   "x_now_logKratio",
    "x_15":    "x_15_logKratio",
    "x_30":    "x_30_logKratio",
    "x_45":    "x_45_logKratio",
    "x_60":    "x_60_logKratio",
    "x_90":    "x_90_logKratio",
    "x_120":   "x_120_logKratio",
    "cb_x_now": "cb_x_now_logKratio",
    "cb_x_15":  "cb_x_15_logKratio",
    "cb_x_30":  "cb_x_30_logKratio",
    "cb_x_60":  "cb_x_60_logKratio",
    "tau":               "tau_s",
    "inv_sqrt_tau":      None,           # computed below
    "ofi_l1":            "ofi_l1",
    "ofi_l5_weighted":   "ofi_l5_weighted",
    "pm_book_imbalance": "pm_book_imbalance",
    "pm_trade_flow_30s": "pm_trade_flow_30s",
    "momentum_30s":      "momentum_30s",
    "momentum_60s":      "momentum_60s",
    "cross_asset_momentum_60s": "cross_asset_momentum_60s",
}


def _build_training_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a list of decision dicts to (X, y, ts_ns_array).
    Rows with missing q_up_ask or missing core features are dropped.
    """
    import math as _math

    X_list, y_list, ts_list = [], [], []

    for row in rows:
        # Target: logit(q_up_ask) — what Polymarket is actually quoting
        q = row.get("q_up_ask")
        if q is None or not (0.02 < q < 0.98):
            continue

        y_val = _logit(float(q))

        # Features
        feat_row = []
        skip = False
        for fname in FEATURE_NAMES:
            col = _FEAT_TO_COL.get(fname)

            if fname == "inv_sqrt_tau":
                tau = row.get("tau_s")
                if tau is None:
                    skip = True; break
                feat_row.append(1.0 / _math.sqrt(float(tau) + 1.0))
                continue

            if col is None:
                feat_row.append(0.0)
                continue

            val = row.get(col)
            if val is None:
                # Only the core Binance x_now..x_60 are required; others impute 0
                required = fname in ("x_now", "x_15", "x_30", "x_45", "x_60", "tau")
                if required:
                    skip = True; break
                feat_row.append(0.0)
            else:
                feat_row.append(float(val))

        if skip or len(feat_row) != len(FEATURE_NAMES):
            continue

        X_list.append(feat_row)
        y_list.append(y_val)
        ts_list.append(int(row.get("ts_ns", 0)))

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty(0), np.empty(0, dtype=np.int64)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(ts_list, dtype=np.int64),
    )


class RegressionRefitter:
    """
    Background asyncio task. Every REFIT_INTERVAL_SECONDS:
      1. Pull training rows from parquet + in-memory buffer
      2. Build (X, y)
      3. Fit ridge regression
      4. Atomic-swap coefficients into live model
      5. Write diagnostics to model_versions table
    """

    def __init__(
        self,
        models: dict[str, RegressionModel],
        writer: "ParquetWriter",
    ):
        self.models  = models
        self.writer  = writer
        self._running = False

    async def run(self) -> None:
        self._running = True
        logger.info("refitter started (interval=%ds, window=%ds)",
                    REFIT_INTERVAL_SECONDS, TRAINING_WINDOW_SECONDS)

        # Stagger first refit so we don't thrash at cold start
        await asyncio.sleep(REFIT_INTERVAL_SECONDS)

        while self._running:
            for asset in ASSETS:
                try:
                    await self._refit_asset(asset)
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("refitter error asset=%s", asset)

            await asyncio.sleep(REFIT_INTERVAL_SECONDS)

    def stop(self) -> None:
        self._running = False

    async def _refit_asset(self, asset: str) -> None:
        now_ns = time.time_ns()
        cutoff_ns = now_ns - TRAINING_WINDOW_SECONDS * 1_000_000_000

        # Pull rows from in-memory buffer
        rows = self.writer.get_decision_rows(asset, since_ns=cutoff_ns)

        # Also scan parquet files for older rows (first hour of each run)
        parquet_rows = await asyncio.get_event_loop().run_in_executor(
            None, _load_parquet_rows, asset, cutoff_ns, PARQUET_DIR
        )
        rows = parquet_rows + rows

        n = len(rows)
        if n < MIN_TRAIN_SIZE:
            logger.info("refitter asset=%s skip: only %d rows (need %d)",
                        asset, n, MIN_TRAIN_SIZE)
            return

        logger.info("refitter asset=%s fitting on %d rows", asset, n)

        X, y, ts_ns_arr = await asyncio.get_event_loop().run_in_executor(
            None, _build_training_matrix, rows
        )

        if len(y) < MIN_TRAIN_SIZE:
            logger.info("refitter asset=%s skip: only %d valid rows after filter",
                        asset, len(y))
            return

        model = self.models[asset]
        diag = await asyncio.get_event_loop().run_in_executor(
            None, model.fit, X, y, ts_ns_arr, now_ns
        )

        logger.info(
            "refitter asset=%s fitted version=%s n=%d alpha=%.3f "
            "r2_cv=%.3f r2_held=%.3f lag_s=%.1f delta_l2=%.4f",
            asset, diag.model_version_id[:8], diag.n_train_samples,
            diag.ridge_alpha, diag.r2_cv_mean, diag.r2_held_out_30min,
            diag.estimated_lag_seconds, diag.coef_delta_l2,
        )

        self.writer.write_model_version(asset, diag)


def _load_parquet_rows(asset: str, cutoff_ns: int, parquet_dir: Path) -> list[dict]:
    """Load decision rows from parquet files newer than cutoff_ns."""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        return []

    base = Path(parquet_dir) / "decisions"
    if not base.exists():
        return []

    rows = []
    for pfile in sorted(base.rglob(f"*asset={asset}*/*.parquet")):
        try:
            tbl = pq.read_table(pfile, columns=["ts_ns"] + _needed_parquet_cols())
            for batch in tbl.to_batches():
                d = batch.to_pydict()
                n = len(d.get("ts_ns", []))
                for i in range(n):
                    ts = d["ts_ns"][i]
                    if ts and ts >= cutoff_ns:
                        rows.append({col: d[col][i] for col in d})
        except Exception:
            pass

    return rows


def _needed_parquet_cols() -> list[str]:
    cols = set()
    for fname in FEATURE_NAMES:
        col = _FEAT_TO_COL.get(fname)
        if col:
            cols.add(col)
    cols.update(["q_up_ask", "tau_s"])
    return sorted(cols)
