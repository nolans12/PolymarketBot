"""
window_replay.py — Replay 15-min windows with model predictions + trade decisions.

Single-asset mode (original):
  python scripts/window_replay.py --model-file model_fits/<dir>/model.pkl --asset BTC

Three-asset side-by-side (3x1 subplot):
  python scripts/window_replay.py \
    --btc-model model_fits/<btc_dir>/model.pkl \
    --eth-model model_fits/<eth_dir>/model.pkl \
    --sol-model model_fits/<sol_dir>/model.pkl

Other options:
  --run       data/<run>  (popup if omitted)
  --window    specific window_ticker (single-asset mode only)
  --seed      random seed for window selection
  --all-windows  cycle through all windows (single-asset mode only)
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from betbot.kalshi.model import load_model
from betbot.kalshi.config import (
    LGBM_FORECAST_HORIZONS, LGBM_PRIMARY_HORIZON,
    THETA_FEE_TAKER, ENTRY_MODE,
    KELLY_TIERS, LAG_CLOSE_THRESHOLD, MAX_HOLD_S, FALLBACK_TAU_S,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    MIN_ENTRY_INTERVAL_S, SIZE_MIN_USD, SIZE_MAX_USD,
)
from pick_run import pick_run_folder

_MODEL_FITS_DIR = _REPO / "model_fits"

# Horizon colours — same as replay_window.py
_HORIZON_COLORS = ["#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ticks(path: Path) -> dict[str, list[dict]]:
    from analysis.tick_loader import load_ticks as _load
    return _load(path)


def build_feature_row(rows: list[dict], idx: int) -> np.ndarray | None:
    r   = rows[idx]
    K   = r["K"]
    mp  = r["btc_micro"]
    tau = max(1.0, r["tau_s"])
    if K <= 0 or mp <= 0:
        return None
    ts = r["ts_ns"]

    def lagged_mp(lag_ns):
        target = ts - lag_ns
        lo, hi = 0, idx - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if rows[mid]["ts_ns"] <= target:
                lo = mid
            else:
                hi = mid - 1
        return rows[lo]["btc_micro"] if rows[lo]["ts_ns"] <= target else mp

    def lagged_ym(lag_ns):
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

    # Depth features (zero on legacy CSV ticks that lack them)
    yes_book = r.get("yes_book") or []
    no_book  = r.get("no_book")  or []
    yes_bid_size = yes_book[0][1] if yes_book else 0.0
    yes_ask_size = no_book[0][1]  if no_book  else 0.0
    yes_best     = yes_book[0][0] if yes_book else 0.0
    no_best      = no_book[0][0]  if no_book  else 0.0
    yes_depth_5c = sum(s for p, s in yes_book if p >= yes_best - 0.05) if yes_book else 0.0
    no_depth_5c  = sum(s for p, s in no_book  if p >= no_best  - 0.05) if no_book  else 0.0

    return np.array([
        x_0, x_5, x_10, x_15, x_20, x_25, x_30,
        tau, inv_sqrt_tau, spread,
        r["yes_mid"] - km5,
        r["yes_mid"] - km10,
        r["yes_mid"] - km30,
        yes_bid_size, yes_ask_size,
        yes_depth_5c, no_depth_5c,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Trade simulator for one window
# ---------------------------------------------------------------------------

def _fee(p: float) -> float:
    return THETA_FEE_TAKER * p * (1.0 - p)


def simulate_window(rows: list[dict], preds: list[float | None],
                    kelly_tiers, exit_threshold, max_hold_s, min_tau_s,
                    wallet, entry_mode, stop_loss=None) -> list[dict]:
    """Run the decision loop over one window's ticks. Returns list of closed trades."""
    trades        = []
    pos           = None
    last_entry_ns = 0

    for r, q_set in zip(rows, preds):
        if q_set is None:
            continue

        cur_ts_ns = r["ts_ns"]
        tau       = r["tau_s"]
        yes_ask   = float(r["yes_ask"])
        yes_bid   = float(r["yes_bid"])
        yes_mid   = float(r["yes_mid"])

        entry_p   = yes_ask if entry_mode == "taker" else yes_bid
        fee_entry = _fee(yes_ask) if entry_mode == "taker" else 0.0
        edge_net  = q_set - entry_p - fee_entry

        # --- Exit ---
        if pos is not None:
            hold_s   = (cur_ts_ns - pos["entry_ts_ns"]) / 1e9
            cur_edge = q_set - yes_ask - _fee(yes_ask)
            stopped   = stop_loss is not None and cur_edge < pos["entry_edge"] - stop_loss
            lag_done  = cur_edge < exit_threshold
            timed_out = hold_s >= max_hold_s
            tau_out   = tau < min_tau_s

            if lag_done or timed_out or tau_out or stopped:
                contracts = pos["size_usd"] / max(pos["entry_p"], 1e-6)
                gross     = (yes_bid - pos["entry_p"]) * contracts
                fee_exit  = _fee(yes_bid) * pos["size_usd"]
                pnl       = gross - pos["fee_entry_usd"] - fee_exit
                reason    = ("stopped" if stopped else "lag_closed" if lag_done
                             else "max_hold" if timed_out else "min_tau")
                trades.append({
                    "entry_ts_ns": pos["entry_ts_ns"],
                    "exit_ts_ns":  cur_ts_ns,
                    "entry_tau_s": pos["entry_tau_s"],
                    "exit_tau_s":  tau,
                    "entry_p":     pos["entry_p"],
                    "exit_p":      yes_bid,
                    "q_at_entry":  pos["q_set"],
                    "size_usd":    pos["size_usd"],
                    "pnl":         pnl,
                    "gross_pnl":   gross,
                    "hold_s":      hold_s,
                    "exit_reason": reason,
                })
                pos = None

        # --- Entry ---
        secs_since = (cur_ts_ns - last_entry_ns) / 1e9
        if (pos is None
                and tau > min_tau_s
                and DECISION_YES_MID_MIN <= yes_mid <= DECISION_YES_MID_MAX
                and secs_since >= MIN_ENTRY_INTERVAL_S):
            tier_usd = 0.0
            for floor, frac in kelly_tiers:
                if edge_net >= floor:
                    tier_usd = min(wallet * frac, SIZE_MAX_USD)
                    break
            if tier_usd >= SIZE_MIN_USD:
                pos = {
                    "entry_ts_ns":  cur_ts_ns,
                    "entry_tau_s":  tau,
                    "entry_p":      entry_p,
                    "entry_edge":   edge_net,
                    "q_set":        q_set,
                    "size_usd":     tier_usd,
                    "fee_entry_usd": fee_entry * tier_usd,
                }
                last_entry_ns = cur_ts_ns

    return trades


# ---------------------------------------------------------------------------
# Plot one window
# ---------------------------------------------------------------------------

def plot_window(rows: list[dict], preds: list[float | None],
                qs_matrix: np.ndarray, horizons: list[int],
                trades: list[dict], model_name: str, ticker: str) -> None:
    t0_ns    = rows[0]["ts_ns"]
    ts_s     = np.array([(r["ts_ns"] - t0_ns) / 1e9 for r in rows])
    yes_bid  = np.array([r["yes_bid"] for r in rows])
    yes_ask  = np.array([r["yes_ask"] for r in rows])
    yes_mid  = np.array([r["yes_mid"] for r in rows])
    btc_vs_k = np.array([r["btc_micro"] - r["K"] for r in rows])
    tau_arr  = np.array([r["tau_s"] for r in rows])

    colors = (_HORIZON_COLORS + ["gray"] * len(horizons))[:len(horizons)]
    primary_idx = horizons.index(LGBM_PRIMARY_HORIZON) if LGBM_PRIMARY_HORIZON in horizons else 0

    fig, (ax_yes, ax_edge) = plt.subplots(2, 1, figsize=(16, 10),
                                           gridspec_kw={"height_ratios": [3, 1]})
    ax_btc = ax_yes.twinx()

    # --- YES bid/ask fill + mid ---
    ax_yes.fill_between(ts_s, yes_bid, yes_ask, alpha=0.15, color="darkorange", label="bid/ask spread")
    ax_yes.plot(ts_s, yes_mid, color="darkorange", lw=1.2, label="YES mid", zorder=3)

    # --- Model prediction dots (primary horizon only, for clarity) ---
    q_primary = qs_matrix[:, primary_idx]
    valid_mask = ~np.isnan(q_primary)
    if valid_mask.any():
        proj_ts = ts_s[valid_mask] + LGBM_PRIMARY_HORIZON
        ax_yes.scatter(proj_ts, q_primary[valid_mask],
                       color=colors[primary_idx], s=8, alpha=0.4, zorder=4,
                       label=f"q_pred t+{LGBM_PRIMARY_HORIZON}s (exit bid)")

    # --- Entry / exit triangles ---
    net_pnl = sum(t["pnl"] for t in trades)
    for t in trades:
        entry_s = (t["entry_ts_ns"] - t0_ns) / 1e9
        exit_s  = (t["exit_ts_ns"]  - t0_ns) / 1e9
        win     = t["pnl"] > 0
        color   = "green" if win else "red"

        ax_yes.scatter([entry_s], [t["entry_p"]], marker="^", color="green",
                       s=120, zorder=6)
        ax_yes.scatter([exit_s],  [t["exit_p"]],  marker="v", color="red",
                       s=120, zorder=6)
        # Line connecting entry to exit
        ax_yes.plot([entry_s, exit_s], [t["entry_p"], t["exit_p"]],
                    color=color, lw=1.5, alpha=0.6, zorder=5)
        # P&L annotation above exit
        ax_yes.annotate(
            f"{t['pnl']:+.3f}",
            xy=(exit_s, t["exit_p"]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=7, color=color, ha="center",
        )

    ax_yes.axhline(0.5, color="gray", lw=0.5, linestyle=":", alpha=0.4)
    ax_yes.set_ylim(-0.02, 1.02)
    ax_yes.set_ylabel("Kalshi YES probability", color="darkorange")
    ax_yes.tick_params(axis="y", labelcolor="darkorange")
    ax_yes.grid(alpha=0.2)

    # --- BTC - K on right axis ---
    ax_btc.plot(ts_s, btc_vs_k, color="steelblue", lw=0.8, alpha=0.6, label="BTC-K")
    ax_btc.set_ylabel("BTC - strike (USD)", color="steelblue")
    ax_btc.tick_params(axis="y", labelcolor="steelblue")

    # Combined legend
    h1, l1 = ax_yes.get_legend_handles_labels()
    h2, l2 = ax_btc.get_legend_handles_labels()
    entry_patch = mpatches.Patch(color="green", label="entry ^")
    exit_patch  = mpatches.Patch(color="red",   label="exit v")
    ax_yes.legend(h1 + h2 + [entry_patch, exit_patch], l1 + l2 + ["entry ^", "exit v"],
                  fontsize=8, loc="upper left", ncol=3)

    n_win    = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = n_win / len(trades) * 100 if trades else 0.0
    ax_yes.set_title(
        f"{ticker}  |  {model_name}  |  "
        f"{len(trades)} trades  win={win_rate:.0f}%  net P&L=${net_pnl:+.4f}",
        fontsize=10, fontweight="bold"
    )

    # --- Edge subplot ---
    edge_vals = []
    edge_ts   = []
    for i, (r, q) in enumerate(zip(rows, preds)):
        if q is not None:
            yes_a = float(r["yes_ask"])
            edge_vals.append(q - yes_a - _fee(yes_a))
            edge_ts.append(ts_s[i])

    if edge_ts:
        ax_edge.plot(edge_ts, edge_vals, color="darkorange", lw=0.8, alpha=0.8, label="edge")
        ax_edge.axhline(0, color="gray", lw=0.8)
        for floor, frac in KELLY_TIERS:
            ax_edge.axhline(floor, color="steelblue", lw=0.8, linestyle="--", alpha=0.6,
                            label=f"floor {floor:.3f}")
        ax_edge.axhline(LAG_CLOSE_THRESHOLD, color="red", lw=0.8, linestyle=":",
                        label=f"exit {LAG_CLOSE_THRESHOLD:.3f}")
        for t in trades:
            entry_s = (t["entry_ts_ns"] - t0_ns) / 1e9
            ax_edge.axvline(entry_s, color="green", lw=0.6, alpha=0.5)
        ax_edge.set_ylabel("Net edge")
        ax_edge.set_xlabel("seconds since window open")
        ax_edge.legend(fontsize=7, ncol=4)
        ax_edge.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay a single window with model + trade decisions")
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--run",        type=str, default=None)
    parser.add_argument("--asset",      type=str, default="BTC")
    parser.add_argument("--window",     type=str, default=None,
                        help="Specific window_ticker to replay (default: random)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Random seed for window selection (reproducible pick)")
    parser.add_argument("--all-windows", action="store_true",
                        help="Cycle through all windows one by one")
    # Trading knobs (default from config.py — override to test tuned values)
    parser.add_argument("--exit",       type=float, default=LAG_CLOSE_THRESHOLD)
    parser.add_argument("--hold-s",     type=int,   default=MAX_HOLD_S)
    parser.add_argument("--min-tau",    type=int,   default=FALLBACK_TAU_S)
    parser.add_argument("--stop-loss",  type=float, default=None)
    parser.add_argument("--wallet",     type=float, default=1000.0)
    parser.add_argument("--entry-mode", type=str,   default=ENTRY_MODE,
                        choices=["taker", "maker"])
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
    model    = load_model(str(pkl))
    horizons = model.horizons
    colors   = (_HORIZON_COLORS + ["gray"] * len(horizons))[:len(horizons)]

    # --- Load ticks (prefer parquet, fall back to csv) ---
    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to replay")
    asset     = args.asset.upper()
    from analysis.tick_loader import find_ticks_path
    tick_path = find_ticks_path(run_dir, asset)
    if tick_path is None:
        sys.exit(f"ERROR: no ticks_{asset}.parquet or .csv in {run_dir}")

    print(f"Loading ticks: {tick_path}", flush=True)
    all_windows = load_ticks(tick_path)
    if not all_windows:
        sys.exit("No tick data found.")

    print(f"  {len(all_windows)} windows available:", flush=True)
    for ticker, rows in sorted(all_windows.items(), key=lambda kv: kv[1][0]["ts_ns"]):
        entry_tau = rows[0]["tau_s"]
        print(f"    {ticker}  ({len(rows)} ticks, joined at tau={entry_tau:.0f}s)", flush=True)

    # Select which windows to show
    if args.window:
        if args.window not in all_windows:
            sys.exit(f"Window {args.window!r} not found.")
        to_show = [args.window]
    elif args.all_windows:
        to_show = sorted(all_windows.keys(), key=lambda t: all_windows[t][0]["ts_ns"])
    else:
        rng     = random.Random(args.seed)
        to_show = [rng.choice(list(all_windows.keys()))]
        print(f"\nRandom window selected: {to_show[0]}", flush=True)

    for ticker in to_show:
        rows = all_windows[ticker]
        print(f"\nReplaying: {ticker}  ({len(rows)} ticks)", flush=True)

        # Build features — respect the 30s warmup gate
        entry_tau_s = rows[0]["tau_s"]
        fvs: list[np.ndarray | None] = []
        for i, r in enumerate(rows):
            elapsed = entry_tau_s - r["tau_s"]
            fvs.append(build_feature_row(rows, i) if elapsed >= 30 else None)

        # Compute predictions
        preds: list[float | None] = []
        qs_matrix = np.full((len(rows), len(horizons)), float("nan"))
        for i, fv in enumerate(fvs):
            if fv is None:
                preds.append(None)
                continue
            q_primary = model.q_settled_from_array(fv)
            preds.append(q_primary)
            all_qs = model.q_all_horizons_from_array(fv)
            for h_idx, h in enumerate(horizons):
                if h in all_qs:
                    qs_matrix[i, h_idx] = all_qs[h]

        n_valid = sum(1 for p in preds if p is not None)
        print(f"  {n_valid}/{len(rows)} ticks with predictions", flush=True)

        # Simulate trades
        trades = simulate_window(
            rows, preds,
            kelly_tiers=KELLY_TIERS,
            exit_threshold=args.exit,
            max_hold_s=args.hold_s,
            min_tau_s=args.min_tau,
            wallet=args.wallet,
            entry_mode=args.entry_mode,
            stop_loss=args.stop_loss,
        )

        print(f"  Trades: {len(trades)}", flush=True)
        for t in trades:
            print(f"    buy@{t['entry_p']:.3f} sell@{t['exit_p']:.3f}  "
                  f"q={t['q_at_entry']:.3f}  pnl=${t['pnl']:+.4f}  "
                  f"hold={t['hold_s']:.0f}s  [{t['exit_reason']}]", flush=True)
        if trades:
            net = sum(t["pnl"] for t in trades)
            print(f"  Net P&L: ${net:+.4f}", flush=True)

        plot_window(rows, preds, qs_matrix, horizons, trades,
                    model_name=pkl.parent.name, ticker=ticker)


if __name__ == "__main__":
    main()
