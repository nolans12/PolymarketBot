"""
train_model.py — Fit a LightGBM model on historical tick data and save to model_fits/.

Loads ticks_<ASSET>.csv from a run folder (typically a dry run), fits a
LightGBM multi-horizon model, and saves it to model_fits/<name>.pkl.
Load the saved model into the live bot with --model-file.

Workflow:
  1. python scripts/run/run_kalshi_bot.py          # dry run, collect ticks
  2. python scripts/train_model.py                 # fit model on collected ticks
  3. python scripts/tune_trading_knobs.py           # optimize Kelly/threshold config
  4. python scripts/test_all.py --model-file ...    # final simulation with tuned config
  5. python scripts/run/run_kalshi_bot.py --model-file model_fits/<name>.pkl --live-orders

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --run data/2026-05-07_00-15-00_BTC
  python scripts/train_model.py --name btc_lgbm_v1
  python scripts/train_model.py --list
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

from betbot.kalshi.features import _logit, FEATURE_NAMES
from betbot.kalshi.model import LGBMModel, make_model, save_model, load_model
from betbot.kalshi.config import LGBM_FORECAST_HORIZONS, LGBM_PRIMARY_HORIZON, TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX
from pick_run import pick_run_folder

MODEL_FITS_DIR = _REPO / "model_fits"


# ---------------------------------------------------------------------------
# Ticks loading (same as replay_window)
# ---------------------------------------------------------------------------

def load_ticks(path: Path) -> dict[str, list[dict]]:
    windows: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                windows[r["window_ticker"]].append({
                    "ts_ns":     int(r["ts_ns"]),
                    "tau_s":     float(r["tau_s"]),
                    "btc_micro": float(r["btc_microprice"]),
                    "yes_bid":   float(r["yes_bid"]),
                    "yes_ask":   float(r["yes_ask"]),
                    "yes_mid":   float(r["yes_mid"]),
                    "K":         float(r["floor_strike"]),
                })
            except (KeyError, ValueError):
                pass
    for rows in windows.values():
        rows.sort(key=lambda r: r["ts_ns"])
    return windows


def build_feature_row(rows: list[dict], idx: int) -> np.ndarray | None:
    r   = rows[idx]
    K   = r["K"]
    mp  = r["btc_micro"]
    tau = max(1.0, r["tau_s"])
    if K <= 0 or mp <= 0:
        return None

    ts = r["ts_ns"]

    def lagged_mp(lag_ns: int) -> float:
        target = ts - lag_ns
        lo, hi = 0, idx - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if rows[mid]["ts_ns"] <= target:
                lo = mid
            else:
                hi = mid - 1
        return rows[lo]["btc_micro"] if rows[lo]["ts_ns"] <= target else mp

    def lagged_ym(lag_ns: int) -> float:
        target = ts - lag_ns
        lo, hi = 0, idx - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if rows[mid]["ts_ns"] <= target:
                lo = mid
            else:
                hi = mid - 1
        return rows[lo]["yes_mid"] if rows[lo]["ts_ns"] <= target else r["yes_mid"]

    try:
        x_0  = math.log(mp / K)
        x_5  = math.log(lagged_mp(5_000_000_000)  / K)
        x_10 = math.log(lagged_mp(10_000_000_000) / K)
        x_15 = math.log(lagged_mp(15_000_000_000) / K)
        x_20 = math.log(lagged_mp(20_000_000_000) / K)
        x_25 = math.log(lagged_mp(25_000_000_000) / K)
        x_30 = math.log(lagged_mp(30_000_000_000) / K)
    except (ValueError, ZeroDivisionError):
        return None

    inv_sqrt_tau = 1.0 / math.sqrt(tau + 1.0)
    spread       = max(0.0, r["yes_ask"] - r["yes_bid"])
    km5  = lagged_ym(5_000_000_000)
    km10 = lagged_ym(10_000_000_000)
    km30 = lagged_ym(30_000_000_000)

    return np.array([
        x_0, x_5, x_10, x_15, x_20, x_25, x_30,
        tau, inv_sqrt_tau, spread,
        r["yes_mid"] - km5,
        r["yes_mid"] - km10,
        r["yes_mid"] - km30,
    ], dtype=np.float64)


def build_multi_horizon_targets_for_window(
    rows: list[dict], X_list: list, ts_list: list[int], horizons_s: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build shifted targets for one window only. Never looks across window boundaries.
    rows, X_list, ts_list must all be from the same window_ticker.
    """
    if not X_list:
        return np.empty((0, len(horizons_s))), np.empty((0, len(horizons_s))), np.empty((0,))

    ts_arr = np.array(ts_list, dtype=np.int64)
    X_arr  = np.array(X_list)
    tol_ns = [2 * h * 1_000_000_000 for h in horizons_s]

    valid_idx, y_cols = [], [[] for _ in horizons_s]

    for i, ts in enumerate(ts_arr):
        ok, targets = True, []
        for h, tol in zip(horizons_s, tol_ns):
            target_ts = ts + h * 1_000_000_000
            j = int(np.searchsorted(ts_arr, target_ts))
            j = min(j, len(ts_arr) - 1)
            if j > 0 and abs(ts_arr[j-1] - target_ts) < abs(ts_arr[j] - target_ts):
                j -= 1
            if abs(ts_arr[j] - target_ts) > tol:
                ok = False
                break
            targets.append(rows[j]["yes_mid"])
        if ok and all(TRAIN_YES_MID_MIN <= t <= TRAIN_YES_MID_MAX for t in targets):
            valid_idx.append(i)
            for col, t in enumerate(targets):
                y_cols[col].append(_logit(t))

    if not valid_idx:
        return np.empty((0, len(horizons_s))), np.empty((0, len(horizons_s))), np.empty((0,))

    idx = np.array(valid_idx)
    return X_arr[idx], np.column_stack([np.array(c) for c in y_cols]), ts_arr[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_saved_models():
    MODEL_FITS_DIR.mkdir(exist_ok=True)
    jsons = sorted(MODEL_FITS_DIR.glob("**/model.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsons:
        print("No saved models in model_fits/")
        return
    print(f"{'Directory':<45} {'R2_hld':>8}  {'Horizons'}")
    print("-" * 70)
    for j in jsons:
        try:
            meta     = json.loads(j.read_text())
            name     = j.parent.name
            r2       = meta.get("r2_held_out", float("nan"))
            horizons = str(meta.get("horizons", "n/a"))
            print(f"{name:<45} {r2:>8.3f}  {horizons}")
        except Exception:
            print(f"  {j.parent}  (unreadable)")


def main():
    parser = argparse.ArgumentParser(description="Fit and save a prediction model")
    parser.add_argument("--run",   type=str, default=None,
                        help="Path to a data/<run> folder (popup if omitted)")
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--name",  type=str, default=None,
                        help="Save name under model_fits/ (auto-generated if omitted)")
    parser.add_argument("--list",  action="store_true",
                        help="List all saved models and exit")
    args = parser.parse_args()

    if args.list:
        list_saved_models()
        return

    run_dir = pick_run_folder(cli_arg=args.run, title="Select run to train on")
    asset      = args.asset.upper()
    tick_path = run_dir / f"ticks_{asset}.csv"
    if not tick_path.exists():
        sys.exit(f"ERROR: ticks file not found: {tick_path}\n"
                 f"  Expected ticks_{{ASSET}}.csv — re-run the dry bot to collect data.")

    print(f"Loading ticks from {tick_path}...", flush=True)
    windows = load_ticks(tick_path)
    if not windows:
        sys.exit("ERROR: no tick data found.")

    n_ticks = sum(len(v) for v in windows.values())
    print(f"  {len(windows)} windows  {n_ticks} ticks total", flush=True)

    # Build training samples per window — never mix rows across window boundaries.
    # Lag features (x_5..x_30) and future targets (t+h) are only valid within
    # the same 15-min market; crossing windows would corrupt both.
    all_X_parts:  list[np.ndarray] = []
    all_y_parts:  list[np.ndarray] = []
    all_ts_parts: list[np.ndarray] = []

    tickers = list(windows.items())
    n_windows = len(tickers)
    for w_idx, (ticker, rows) in enumerate(tickers):
        pct = (w_idx + 1) / n_windows * 100
        print(f"  Building features [{w_idx+1}/{n_windows}  {pct:.0f}%]  {ticker} ({len(rows)} ticks)...",
              end="\r", flush=True)

        win_rows: list[dict] = []
        win_X:    list       = []
        win_ts:   list[int]  = []

        for i, r in enumerate(rows):
            if i < 30:
                continue
            if not (TRAIN_YES_MID_MIN <= r["yes_mid"] <= TRAIN_YES_MID_MAX):
                continue
            fv = build_feature_row(rows, i)
            if fv is None:
                continue
            win_X.append(fv)
            win_ts.append(r["ts_ns"])
            win_rows.append(r)

        if not win_X:
            continue

        X_w, y_w, ts_w = build_multi_horizon_targets_for_window(
            win_rows, win_X, win_ts, LGBM_FORECAST_HORIZONS
        )
        if len(X_w) > 0:
            all_X_parts.append(X_w)
            all_y_parts.append(y_w)
            all_ts_parts.append(ts_w)

    print(flush=True)  # newline after \r progress

    if not all_X_parts:
        sys.exit("ERROR: no usable training samples.")

    X  = np.vstack(all_X_parts)
    y  = np.vstack(all_y_parts)
    ts = np.concatenate(all_ts_parts)

    # Estimate data rate from median inter-tick interval across all windows
    all_intervals = []
    for ts_w in all_ts_parts:
        if len(ts_w) > 1:
            all_intervals.extend(np.diff(ts_w).tolist())
    if all_intervals:
        median_ns = float(np.median(all_intervals))
        hz = 1e9 / median_ns if median_ns > 0 else 1.0
    else:
        hz = 1.0
    duration_min = len(X) / hz / 60
    print(f"  {len(X)} training samples across {len(all_X_parts)} windows "
          f"({duration_min:.1f} min at {hz:.1f}Hz)", flush=True)

    model = make_model()
    print(f"Fitting LightGBM model (horizons: {LGBM_FORECAST_HORIZONS}s)...", flush=True)
    if len(y) == 0:
        sys.exit("ERROR: not enough data for multi-horizon targets. "
                 "Need at least 60s of contiguous ticks — run the dry bot longer.")
    print(f"  Multi-horizon samples: {len(y)}  horizons: {LGBM_FORECAST_HORIZONS}s", flush=True)
    diag = model.fit(X, y, ts)
    print(f"  R2_hld(primary={LGBM_PRIMARY_HORIZON}s): {diag.r2_held_out:.3f}", flush=True)
    for k, v in diag.coefs.items():
        print(f"    {k}={v:.3f}", flush=True)

    import datetime
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        dir_name = args.name
    else:
        dir_name = f"{run_dir.name}_{asset}_{ts_str}"

    model_dir = MODEL_FITS_DIR / dir_name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, model_dir / "model")
    print(f"\nDone. To use this model:", flush=True)
    print(f"  python scripts/train_model.py --model-file model_fits/{dir_name}/model.pkl", flush=True)
    print(f"  python scripts/run/run_kalshi_bot.py --model-file model_fits/{dir_name}/model.pkl --live-orders", flush=True)


if __name__ == "__main__":
    main()
