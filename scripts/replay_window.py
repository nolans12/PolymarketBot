"""
replay_window.py — Animated replay of a 15-minute Kalshi window.

Single time-series plot with two y-axes:
  Left axis  (orange): Kalshi YES bid/ask spread + mid price
  Right axis (blue):   BTC microprice (Coinbase)

Multi-horizon model projections drawn on the LEFT axis as colored dots
at t+5s, t+10s, t+15s, t+60s ahead of the current tick.

Usage:
  python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl
  python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl --run data/first_run
  python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl --window KXBTC15M-26MAY070200-00
  python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl --speed 10
  python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl --speed 5 --trail 60
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

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

from betbot.kalshi.model import load_model
from betbot.kalshi.config import LGBM_FORECAST_HORIZONS
from pick_run import pick_run_folder

_MODEL_FITS_DIR = _REPO / "model_fits"

SPEED_DEFAULT = 10    # ticks per wall-clock second  => 10x real speed at 1Hz data
TRAIL_S       = 120   # seconds of history to show

# One colour per horizon — same order as LGBM_FORECAST_HORIZONS [5, 10, 15, 60]
_HORIZON_COLORS = ["#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]


# ---------------------------------------------------------------------------
# Data loading + feature building
# ---------------------------------------------------------------------------

def _load_ticks(path: Path) -> dict[str, list[dict]]:
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


def _build_feature_row(rows: list[dict], idx: int) -> np.ndarray | None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Animated replay of a Kalshi window")
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--run",    type=str, default=None)
    parser.add_argument("--asset",  type=str, default="BTC")
    parser.add_argument("--window", type=str, default=None,
                        help="Specific window_ticker to replay (default: longest)")
    parser.add_argument("--speed",  type=float, default=SPEED_DEFAULT,
                        help=f"Ticks per wall-clock second (default {SPEED_DEFAULT}x)")
    parser.add_argument("--trail",  type=int, default=TRAIL_S,
                        help="Seconds of history to show (default 120)")
    args = parser.parse_args()

    # --- Load model ---
    p   = Path(args.model_file)
    pkl = p if str(p).endswith(".pkl") else Path(str(p) + ".pkl")
    if not pkl.is_absolute():
        for candidate in [_REPO / pkl, _MODEL_FITS_DIR / pkl]:
            if candidate.exists():
                pkl = candidate
                break
    if not pkl.exists():
        sys.exit(f"Model not found: {pkl}")
    print(f"Loading model: {pkl}", flush=True)
    model = load_model(str(pkl))
    if not model.is_fit:
        sys.exit("Model is not fitted.")

    horizons = model.horizons   # e.g. [5, 10, 15, 60]
    colors   = (_HORIZON_COLORS + ["gray"] * len(horizons))[:len(horizons)]

    # --- Load ticks ---
    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to replay")
    asset     = args.asset.upper()
    tick_path = run_dir / f"ticks_{asset}.csv"
    if not tick_path.exists():
        sys.exit(f"ERROR: {tick_path} not found.")

    print(f"Loading ticks: {tick_path}", flush=True)
    windows = _load_ticks(tick_path)
    if not windows:
        sys.exit("ERROR: no tick data found.")

    print(f"  {len(windows)} windows:", flush=True)
    for ticker, rows in sorted(windows.items(), key=lambda kv: kv[1][0]["ts_ns"]):
        print(f"    {ticker}  ({len(rows)} ticks)", flush=True)

    replay_ticker = args.window or max(windows, key=lambda t: len(windows[t]))
    if replay_ticker not in windows:
        sys.exit(f"Window {replay_ticker!r} not found.")
    rows = windows[replay_ticker]
    print(f"\nReplaying: {replay_ticker}  ({len(rows)} ticks)", flush=True)

    # --- Pre-compute arrays ---
    t0_ns       = rows[0]["ts_ns"]
    ts_s        = np.array([(r["ts_ns"] - t0_ns) / 1e9 for r in rows])
    btc_arr     = np.array([r["btc_micro"]               for r in rows])
    k_arr       = np.array([r["K"]                       for r in rows])
    btc_vs_k    = btc_arr - k_arr
    yes_mid     = np.array([r["yes_mid"]                 for r in rows])
    yes_ask     = np.array([r["yes_ask"]                 for r in rows])
    yes_bid     = np.array([r["yes_bid"]                 for r in rows])

    # Per-tick multi-horizon predictions: shape (n_ticks, n_horizons), nan where missing
    print("Pre-computing multi-horizon predictions...", flush=True)
    qs_matrix = np.full((len(rows), len(horizons)), float("nan"))
    for i in range(len(rows)):
        if i < 30:
            continue
        fv = _build_feature_row(rows, i)
        if fv is None:
            continue
        preds = model.q_all_horizons_from_array(fv)
        for h_idx, h in enumerate(horizons):
            if h in preds:
                qs_matrix[i, h_idx] = preds[h]

    n_valid = int(np.sum(~np.isnan(qs_matrix[:, 0])))
    print(f"  {n_valid}/{len(rows)} ticks with valid predictions", flush=True)

    # --- Figure setup: single plot, two y-axes ---
    fig, ax_yes = plt.subplots(figsize=(15, 6))
    ax_btc = ax_yes.twinx()

    # YES axis artists
    line_yes_mid, = ax_yes.plot([], [], color="darkorange", lw=1.5, label="YES mid", zorder=3)
    fill_ba = [ax_yes.fill_between([], [], [], alpha=0.15, color="darkorange")]  # placeholder

    # One scatter + horizontal projection line per horizon
    proj_scats = []
    proj_lines = []
    for h, col in zip(horizons, colors):
        s = ax_yes.scatter([], [], color=col, s=60, zorder=6, label=f"q t+{h}s")
        l, = ax_yes.plot([], [], color=col, lw=1.5, linestyle="--", alpha=0.7)
        proj_scats.append(s)
        proj_lines.append(l)

    ax_yes.axhline(0.5, color="gray", lw=0.6, linestyle=":", alpha=0.35)
    ax_yes.set_ylim(-0.02, 1.02)
    ax_yes.set_ylabel("Kalshi YES probability", color="darkorange", fontsize=10)
    ax_yes.tick_params(axis="y", labelcolor="darkorange")

    # BTC axis artists
    line_btc, = ax_btc.plot([], [], color="steelblue", lw=1.0, alpha=0.7, label="BTC − K")
    ax_btc.set_ylabel("BTC microprice − strike (USD)", color="steelblue", fontsize=10)
    ax_btc.tick_params(axis="y", labelcolor="steelblue")

    ax_yes.set_xlabel("seconds since window open", fontsize=10)
    ax_yes.grid(alpha=0.2)

    # Combined legend
    handles_yes, labels_yes = ax_yes.get_legend_handles_labels()
    handles_btc, labels_btc = ax_btc.get_legend_handles_labels()
    ax_yes.legend(handles_yes + handles_btc, labels_yes + labels_btc,
                  fontsize=8, loc="upper left", ncol=3)

    # --- Animation ---
    window_dur_s = (rows[-1]["ts_ns"] - rows[0]["ts_ns"]) / 1e9
    interval_ms  = max(1, int(window_dur_s / args.speed / len(rows) * 1000))
    max_horizon  = max(horizons)
    print(f"  Speed: {args.speed}x  interval: {interval_ms}ms  "
          f"est. duration: {window_dur_s/args.speed:.0f}s", flush=True)

    state = {"idx": 30}

    def _update(_):
        i = state["idx"]
        if i >= len(rows):
            return
        state["idx"] = i + 1

        now_s  = ts_s[i]
        x_min  = now_s - args.trail
        x_max  = now_s + max_horizon + 5
        start  = max(0, int(np.searchsorted(ts_s, x_min)))
        sl     = slice(start, i + 1)

        vis_ts  = ts_s[sl]
        vis_ym  = yes_mid[sl]
        vis_ya  = yes_ask[sl]
        vis_yb  = yes_bid[sl]
        vis_btc = btc_vs_k[sl]

        # YES mid line
        line_yes_mid.set_data(vis_ts, vis_ym)

        # Bid/ask fill — remove old, redraw
        for coll in fill_ba:
            try:
                coll.remove()
            except Exception:
                pass
        fill_ba[0] = ax_yes.fill_between(
            vis_ts, vis_yb, vis_ya, alpha=0.15, color="darkorange"
        )

        # BTC − K line
        line_btc.set_data(vis_ts, vis_btc)

        # Multi-horizon projections: dot at (now_s + h, q) + dashed line from now to dot
        for h_idx, (h, s, l) in enumerate(zip(horizons, proj_scats, proj_lines)):
            q = qs_matrix[i, h_idx]
            if not math.isnan(q):
                proj_x = now_s + h
                s.set_offsets([[proj_x, q]])
                # Horizontal dashed line from current YES mid to projection dot
                l.set_data([now_s, proj_x], [yes_mid[i], q])
            else:
                s.set_offsets(np.empty((0, 2)))
                l.set_data([], [])

        # Axis limits
        ax_yes.set_xlim(x_min, x_max)
        if len(vis_btc):
            pad = max(50.0, (vis_btc.max() - vis_btc.min()) * 0.1)
            ax_btc.set_ylim(vis_btc.min() - pad, vis_btc.max() + pad)

        # Title
        tau_now = rows[i]["tau_s"]
        qs_strs = []
        for h_idx, h in enumerate(horizons):
            q = qs_matrix[i, h_idx]
            qs_strs.append(f"q+{h}s={'---' if math.isnan(q) else f'{q:.3f}'}")
        elapsed = f"{int(now_s//60):02d}:{int(now_s%60):02d}"
        fig.suptitle(
            f"{replay_ticker}  |  t={elapsed}  tau={tau_now:.0f}s  "
            f"BTC-K={btc_vs_k[i]:+,.0f}  YES={yes_mid[i]:.3f}  "
            + "  ".join(qs_strs)
            + f"  [{args.speed:.0f}x  {i}/{len(rows)}]",
            fontsize=9, fontweight="bold",
        )

    ani = animation.FuncAnimation(
        fig, _update,
        interval=interval_ms,
        blit=False,
        cache_frame_data=False,
    )
    plt.show()
    _ = ani


if __name__ == "__main__":
    main()
