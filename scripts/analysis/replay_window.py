"""
replay_window.py - Fit the regression on all available tick data, then
replay a chosen 15-minute window with a live plot showing:

  Blue  (left axis):  BTC microprice - floor_strike
  Orange (right axis, solid):  actual Kalshi YES mid
  Red   (right axis, dashed):  q_settled - model's predicted equilibrium,
                                projected as a short horizontal line
                                LOOKAHEAD_S seconds forward

Usage:
  python scripts/analysis/replay_window.py
  python scripts/analysis/replay_window.py --run data/2026-05-07_00-15-00_BTC
  python scripts/analysis/replay_window.py --asset ETH
  python scripts/analysis/replay_window.py --window KXBTC15M-26MAY070200-00
  python scripts/analysis/replay_window.py --speed 15
  python scripts/analysis/replay_window.py --lookahead 10
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from betbot.kalshi.features import FEATURE_NAMES, _logit, _sigmoid
from pick_run import pick_run_folder
from betbot.kalshi.model import LGBMModel, make_model, save_model, load_model
from betbot.kalshi.config import LGBM_FORECAST_HORIZONS, LGBM_PRIMARY_HORIZON

_MODEL_FITS_DIR = Path(__file__).resolve().parents[2] / "model_fits"

LOOKAHEAD_S_DEFAULT = 5
SPEED_DEFAULT       = 15    # ticks per second (1 tick = ~1s real data => 15x speedup)
TRAIL_S             = 120   # seconds of history to show


# ---------------------------------------------------------------------------
# Data loading
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


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

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
        tau, inv_sqrt_tau,
        spread,
        r["yes_mid"] - km5,
        r["yes_mid"] - km10,
        r["yes_mid"] - km30,
    ], dtype=np.float64)



# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _build_multi_horizon_targets(
    all_rows: list[dict], all_X: list, all_ts: list, horizons_s: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift training targets by each horizon for LGBM multi-horizon training."""
    ts_arr  = np.array(all_ts, dtype=np.int64)
    X_arr   = np.array(all_X)
    tol_ns  = [2 * h * 1_000_000_000 for h in horizons_s]

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
            targets.append(all_rows[j]["yes_mid"])
        if ok and all(0.001 <= t <= 0.999 for t in targets):
            valid_idx.append(i)
            for col, t in enumerate(targets):
                y_cols[col].append(_logit(t))

    if not valid_idx:
        return np.empty((0, len(horizons_s))), np.empty((0,)), np.empty((0,))

    idx = np.array(valid_idx)
    return X_arr[idx], np.column_stack([np.array(c) for c in y_cols]), ts_arr[idx]


def fit_model(windows: dict[str, list[dict]]) -> LGBMModel:
    """Fit a LightGBM model on all windows in the ticks file."""
    model = make_model()

    all_rows_flat = []
    all_X, all_ts = [], []

    for ticker, rows in windows.items():
        for i, r in enumerate(rows):
            if i < 30:
                continue
            if not (0.001 <= r["yes_mid"] <= 0.999):
                continue
            fv = build_feature_row(rows, i)
            if fv is None:
                continue
            all_X.append(fv)
            all_ts.append(r["ts_ns"])
            all_rows_flat.append(r)

    if not all_X:
        sys.exit("ERROR: no training data available.")

    X, y, ts = _build_multi_horizon_targets(all_rows_flat, all_X, all_ts, LGBM_FORECAST_HORIZONS)
    if len(y) == 0:
        sys.exit("ERROR: not enough data to build multi-horizon targets (need at least 60s of ticks).")
    diag, _ = model.fit_if_better(X, y, ts)
    print(f"  LGBM fit: n={diag.n_train}  "
          f"R2_hld(primary {LGBM_PRIMARY_HORIZON}s)={diag.r2_held_out:.3f}", flush=True)
    for k, v in diag.coefs.items():
        print(f"    {k}={v:.3f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       type=str,   default=None)
    parser.add_argument("--asset",     type=str,   default="BTC")
    parser.add_argument("--window",    type=str,   default=None)
    parser.add_argument("--speed",     type=float, default=SPEED_DEFAULT,
                        help="Ticks per second to advance (default 15)")
    parser.add_argument("--lookahead", type=float, default=LOOKAHEAD_S_DEFAULT)
    parser.add_argument("--trail",     type=int,   default=TRAIL_S)
    parser.add_argument("--model-file", type=str,   default=None,
                        help="Path to a saved model .pkl (skips fitting). "
                             "E.g. model_fits/my_model.pkl")
    args = parser.parse_args()

    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to replay")
    asset     = args.asset.upper()
    tick_path = run_dir / f"ticks_{asset}.csv"
    if not tick_path.exists():
        legacy = run_dir / "ticks.csv"
        if legacy.exists():
            tick_path = legacy
    if not tick_path.exists():
        sys.exit(f"ERROR: ticks file not found: {tick_path}")

    print(f"Loading ticks from {tick_path}...", flush=True)
    windows = load_ticks(tick_path)
    if not windows:
        sys.exit("ERROR: no tick data found.")

    print(f"Detected {len(windows)} 15-minute window(s) in dataset:", flush=True)
    for ticker, rows in sorted(windows.items(), key=lambda kv: kv[1][0]["ts_ns"]):
        print(f"  {ticker}  ({len(rows)} ticks)", flush=True)

    if args.window:
        replay_ticker = args.window
        if replay_ticker not in windows:
            sys.exit(f"ERROR: window {replay_ticker!r} not found.")
    else:
        replay_ticker = max(windows, key=lambda t: len(windows[t]))

    replay_rows = windows[replay_ticker]
    window_s = (replay_rows[-1]["ts_ns"] - replay_rows[0]["ts_ns"]) / 1e9
    est_s = window_s / args.speed
    print(f"\nReplaying: {replay_ticker}  ({len(replay_rows)} ticks, {window_s:.0f}s)  [1 window]",
          flush=True)
    print(f"  Speed: {args.speed}x   Est. replay time: {est_s:.0f}s", flush=True)

    if args.model_file:
        mf_path = Path(args.model_file)
        if not mf_path.is_absolute():
            # Try relative to repo root first, then model_fits/
            if (_MODEL_FITS_DIR / mf_path).exists():
                mf_path = _MODEL_FITS_DIR / mf_path
        pkl = str(mf_path) + ".pkl" if not str(mf_path).endswith(".pkl") else str(mf_path)
        print(f"Loading saved model from {pkl}...", flush=True)
        model = load_model(pkl)
    else:
        print("Fitting LightGBM model on all windows...", flush=True)
        model = fit_model(windows)
    if not model.is_fit:
        sys.exit("ERROR: model did not fit.")

    print("Pre-computing q_settled...", flush=True)
    q_settled_arr = []
    for i in range(len(replay_rows)):
        if i < 30:
            q_settled_arr.append(float("nan"))
            continue
        fv = build_feature_row(replay_rows, i)
        if fv is None:
            q_settled_arr.append(float("nan"))
            continue
        qs = model.q_settled_from_array(fv)
        q_settled_arr.append(qs if qs is not None else float("nan"))

    n_valid = sum(1 for q in q_settled_arr if not math.isnan(q))
    print(f"  Done. {n_valid} valid predictions.", flush=True)

    # Pre-compute everything as numpy arrays
    t0_ns      = replay_rows[0]["ts_ns"]
    ts_s       = np.array([(r["ts_ns"] - t0_ns) / 1e9 for r in replay_rows])
    delta_arr  = np.array([r["btc_micro"] - r["K"]     for r in replay_rows])
    yes_arr    = np.array([r["yes_mid"]                 for r in replay_rows])
    yes_ask_arr = np.array([r["yes_ask"]                for r in replay_rows])
    qs_arr     = np.array(q_settled_arr)

    # --- Figure setup ---
    fig, ax_btc = plt.subplots(figsize=(14, 5))
    ax_yes = ax_btc.twinx()
    fig.tight_layout(pad=2.5)

    # Pre-create all artists
    line_delta, = ax_btc.plot([], [], color="steelblue",  lw=1.2, label="BTC - strike")
    line_yes,   = ax_yes.plot([], [], color="darkorange", lw=1.3, label="YES mid (actual)")
    line_qs,    = ax_yes.plot([], [], color="red", lw=2.0, linestyle="--",
                               label="q_settled (model target)")
    scat_qs     = ax_yes.scatter([], [], color="red", s=40, zorder=7)
    # Vertical line segment as cheap arrow substitute
    line_arrow, = ax_yes.plot([], [], color="red", lw=1.5, alpha=0.8)

    ax_btc.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.5)
    ax_yes.axhline(0.5, color="darkorange", lw=0.6, linestyle="--", alpha=0.3)
    ax_btc.set_ylabel("BTC - strike (USD)", color="steelblue", fontsize=9)
    ax_btc.set_xlabel("seconds since window open", fontsize=9)
    ax_btc.tick_params(axis="y", labelcolor="steelblue")
    ax_yes.set_ylabel("Kalshi YES probability", color="darkorange", fontsize=9)
    ax_yes.tick_params(axis="y", labelcolor="darkorange")
    ax_yes.set_ylim(-0.05, 1.05)
    ax_btc.grid(alpha=0.2)

    h1, l1 = ax_btc.get_legend_handles_labels()
    h2, l2 = ax_yes.get_legend_handles_labels()
    ax_yes.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # Compute interval so the animation finishes in (window_duration / speed) real seconds.
    # Each frame advances 1 tick; total frames = len(replay_rows).
    # Total real time = window_duration_s / speed  =>  interval = that / n_ticks * 1000
    window_duration_s = (replay_rows[-1]["ts_ns"] - replay_rows[0]["ts_ns"]) / 1e9
    n_ticks = len(replay_rows)
    interval_ms = max(1, int(window_duration_s / args.speed / n_ticks * 1000))
    print(f"  Tick spacing: {window_duration_s/n_ticks*1000:.1f}ms  "
          f"Frame interval: {interval_ms}ms  "
          f"Replay duration: {window_duration_s/args.speed:.0f}s", flush=True)
    state = {"idx": 30}

    def _update(_):
        i = state["idx"]
        if i >= len(replay_rows):
            return line_delta, line_yes, line_qs, scat_qs, line_arrow
        state["idx"] = i + 1

        now_s = ts_s[i]
        x_min = now_s - args.trail

        # Slice visible trail
        start = max(0, np.searchsorted(ts_s, x_min))
        sl    = slice(start, i + 1)
        vis_ts    = ts_s[sl]
        vis_delta = delta_arr[sl]
        vis_yes   = yes_arr[sl]

        line_delta.set_data(vis_ts, vis_delta)
        line_yes.set_data(vis_ts, vis_yes)

        # q_settled lookahead
        qs_now = qs_arr[i]
        if not math.isnan(qs_now):
            look_s = now_s + args.lookahead
            line_qs.set_data([now_s, look_s], [qs_now, qs_now])
            scat_qs.set_offsets([[look_s, qs_now]])
            yes_now = yes_arr[i]
            line_arrow.set_data([now_s, now_s], [yes_now, qs_now])
        else:
            line_qs.set_data([], [])
            scat_qs.set_offsets(np.empty((0, 2)))
            line_arrow.set_data([], [])

        # Axis limits
        if len(vis_ts) > 0:
            ax_btc.set_xlim(x_min, now_s + args.lookahead + 5)
            d_min, d_max = float(vis_delta.min()), float(vis_delta.max())
            pad = max(50.0, (d_max - d_min) * 0.1)
            ax_btc.set_ylim(d_min - pad, d_max + pad)

        tau_now  = replay_rows[i]["tau_s"]
        edge     = (qs_now - yes_ask_arr[i]) if not math.isnan(qs_now) else float("nan")
        qs_str   = f"{qs_now:.3f}" if not math.isnan(qs_now) else "---"
        edge_str = f"{edge:+.3f}" if math.isfinite(edge) else "---"
        elapsed  = f"{int(now_s // 60):02d}:{int(now_s % 60):02d}"
        ax_btc.set_title(
            f"{replay_ticker}  |  t={elapsed}  tau={tau_now:.0f}s  "
            f"YES={yes_arr[i]:.3f}  q_settled={qs_str}  edge={edge_str}  "
            f"[{args.speed:.0f}x  {i}/{len(replay_rows)}]",
            fontsize=10,
        )

        return line_delta, line_yes, line_qs, scat_qs, line_arrow

    ani = animation.FuncAnimation(
        fig, _update,
        interval=interval_ms,
        blit=False,           # blit=True breaks twinx axis limits
        cache_frame_data=False,
    )
    plt.show()
    _ = ani


if __name__ == "__main__":
    main()
