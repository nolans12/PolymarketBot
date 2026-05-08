"""
test_all.py — Simulate live trading with a pre-saved model and a tuned config.

Loads a model from model_fits/ and replays ticks_<ASSET>.csv through the full
decision loop using the trading knobs from config.py (or overridden via CLI).
Reports P&L, win rate, exit breakdown, and individual trades.

Use this as the final sanity check before going live:
  - Train a model with scripts/train_model.py
  - Tune trading knobs with scripts/tune_trading_knobs.py
  - Run this to verify the full picture on held-out data

Workflow:
  1. python scripts/run/run_kalshi_bot.py          # dry run, collect ticks (24h+)
  2. python scripts/train_model.py                 # fit model, save to model_fits/
  3. python scripts/tune_trading_knobs.py          # optimize Kelly/threshold config
  4. python scripts/test_all.py --model-file ...   # final simulation with tuned config
  5. python scripts/run/run_kalshi_bot.py --model-file model_fits/<name>.pkl --live-orders

Usage:
  python scripts/test_all.py
  python scripts/test_all.py --model-file model_fits/btc_lgbm_v1.pkl
  python scripts/test_all.py --model-file model_fits/btc.pkl --train-frac 0.6
  python scripts/test_all.py --model-file model_fits/btc.pkl --no-plot
"""

import argparse
import csv
import json
import math
import sys
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

from betbot.kalshi.model import load_model, LGBMModel
from betbot.kalshi.config import (
    THETA_FEE_TAKER, THETA_FEE_MAKER,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    LAG_CLOSE_THRESHOLD, MAX_HOLD_S, FALLBACK_TAU_S,
    KELLY_TIERS, ENTRY_MODE, MIN_ENTRY_INTERVAL_S,
    LGBM_FORECAST_HORIZONS, LGBM_PRIMARY_HORIZON,
    TRAIN_PRICE_MIN, TRAIN_PRICE_MAX,
    SIZE_MIN_USD, SIZE_MAX_USD,
)
from pick_run import pick_run_folder

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

MODEL_FITS_DIR = _REPO / "model_fits"


# ---------------------------------------------------------------------------
# Tick loading
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
                    "window_ticker": r["window_ticker"],
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
            mid_i = (lo + hi + 1) // 2
            if rows[mid_i]["ts_ns"] <= target:
                lo = mid_i
            else:
                hi = mid_i - 1
        return rows[lo]["btc_micro"] if rows[lo]["ts_ns"] <= target else mp

    def lagged_ym(lag_ns: int) -> float:
        target = ts - lag_ns
        lo, hi = 0, idx - 1
        while lo < hi:
            mid_i = (lo + hi + 1) // 2
            if rows[mid_i]["ts_ns"] <= target:
                lo = mid_i
            else:
                hi = mid_i - 1
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
# Kelly sizing (mirrors scheduler.py)
# ---------------------------------------------------------------------------

def _fee(p: float) -> float:
    return THETA_FEE_TAKER * p * (1.0 - p)


def _kelly_size(edge: float, wallet: float, kelly_tiers: list) -> tuple[float, int]:
    for tier_idx, (floor, frac) in enumerate(kelly_tiers, start=1):
        if edge >= floor:
            return min(wallet * frac, SIZE_MAX_USD), tier_idx
    return 0.0, 0


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(
    flat_rows: list[dict],
    flat_X: list[np.ndarray],
    model,
    kelly_tiers: list,
    exit_threshold: float,
    max_hold_s: int,
    min_tau_s: int,
    wallet: float = 1000.0,
    entry_mode: str = "taker",
) -> tuple[list[dict], list[float | None]]:
    """
    Walk ticks in time order. Mirrors the live decision loop in scheduler.py.
    Returns (trades, edge_series) where edge_series[i] = net edge at tick i (None if no prediction).
    """
    trades         = []
    edge_series: list[float | None] = []
    pos            = None
    prev_ts_ns     = None
    last_entry_ns  = 0

    for i, (r, fv) in enumerate(zip(flat_rows, flat_X)):
        if fv is None:
            prev_ts_ns = r["ts_ns"]
            continue

        cur_ticker = r["window_ticker"]
        cur_ts_ns  = r["ts_ns"]

        # Force-close on window change or gap > 5s
        if pos is not None:
            ticker_changed = (cur_ticker != pos["window_ticker"])
            gap_s = (cur_ts_ns - (prev_ts_ns or cur_ts_ns)) / 1e9
            if ticker_changed or gap_s > 5.0:
                trades.append({
                    "entry_ts":    pos["entry_ts_ns"] / 1e9,
                    "exit_ts":     (prev_ts_ns or cur_ts_ns) / 1e9,
                    "entry_p":     pos["entry_p"],
                    "exit_p":      pos["entry_p"],
                    "size_usd":    pos["size_usd"],
                    "pnl":         -pos["fee_entry_abs"] * pos["size_usd"],
                    "gross_pnl":   0.0,
                    "hold_s":      ((prev_ts_ns or cur_ts_ns) - pos["entry_ts_ns"]) / 1e9,
                    "exit_edge":   0.0,
                    "q_settled":   pos["q_settled"],
                    "exit_reason": "window_gap",
                    "tier":        pos["tier"],
                })
                pos = None

        prev_ts_ns = cur_ts_ns
        tau     = r["tau_s"]
        yes_ask = float(r["yes_ask"])
        yes_bid = float(r["yes_bid"])
        yes_mid = float(r["yes_mid"])

        q_set = model.q_settled_from_array(fv)
        if q_set is None or not math.isfinite(q_set):
            continue

        if entry_mode == "taker":
            entry_p = yes_ask
            fee_e   = _fee(yes_ask)
        else:
            entry_p = yes_bid
            fee_e   = 0.0

        edge_net = q_set - entry_p - fee_e
        edge_series.append((cur_ts_ns, edge_net))

        # --- Exit logic ---
        if pos is not None:
            hold_s   = (cur_ts_ns - pos["entry_ts_ns"]) / 1e9
            cur_edge = q_set - yes_ask - _fee(yes_ask)
            should_exit = (
                cur_edge < exit_threshold
                or hold_s >= max_hold_s
                or tau < min_tau_s
            )
            if should_exit:
                exit_p   = yes_bid
                fee_exit = _fee(exit_p) * pos["size_usd"]
                gross    = (exit_p - pos["entry_p"]) * pos["size_usd"] / max(pos["entry_p"], 1e-6) * pos["entry_p"]
                # simpler: P&L = (exit - entry) * contracts
                contracts = pos["size_usd"] / max(pos["entry_p"], 1e-6)
                gross     = (exit_p - pos["entry_p"]) * contracts
                fee_tot   = pos["fee_entry_abs"] * pos["size_usd"] + fee_exit
                pnl       = gross - fee_tot
                trades.append({
                    "entry_ts":    pos["entry_ts_ns"] / 1e9,
                    "exit_ts":     cur_ts_ns / 1e9,
                    "entry_p":     pos["entry_p"],
                    "exit_p":      exit_p,
                    "size_usd":    pos["size_usd"],
                    "pnl":         pnl,
                    "gross_pnl":   gross,
                    "hold_s":      hold_s,
                    "exit_edge":   cur_edge,
                    "q_settled":   pos["q_settled"],
                    "exit_reason": (
                        "lag_closed" if cur_edge < exit_threshold
                        else "max_hold" if hold_s >= max_hold_s
                        else "min_tau"
                    ),
                    "tier":        pos["tier"],
                })
                pos = None

        # --- Entry logic ---
        secs_since_entry = (cur_ts_ns - last_entry_ns) / 1e9
        if (pos is None
                and tau > min_tau_s
                and DECISION_YES_MID_MIN <= yes_mid <= DECISION_YES_MID_MAX
                and secs_since_entry >= MIN_ENTRY_INTERVAL_S):
            bet_usd, tier = _kelly_size(edge_net, wallet, kelly_tiers)
            if tier > 0:
                pos = {
                    "entry_ts_ns":   cur_ts_ns,
                    "entry_p":       entry_p,
                    "entry_edge":    edge_net,
                    "size_usd":      max(bet_usd, SIZE_MIN_USD),
                    "fee_entry_abs": fee_e,
                    "q_settled":     q_set,
                    "window_ticker": cur_ticker,
                    "tier":          tier,
                }
                last_entry_ns = cur_ts_ns

    return trades, edge_series


# ---------------------------------------------------------------------------
# Single-config run
# ---------------------------------------------------------------------------

def run_simulation(
    flat_rows: list[dict],
    flat_X: list[np.ndarray],
    model,
    kelly_tiers: list,
    exit_threshold: float,
    max_hold_s: int,
    min_tau_s: int,
    wallet: float,
    entry_mode: str,
) -> dict:
    trades, edge_series = simulate(flat_rows, flat_X, model,
                                   kelly_tiers, exit_threshold,
                                   max_hold_s, min_tau_s, wallet, entry_mode)

    pnl_vals  = [t["pnl"] for t in trades]
    n_win     = sum(1 for p in pnl_vals if p > 0)
    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1

    return {
        "n_ticks":         len(flat_rows),
        "n_trades":        len(trades),
        "n_win":           n_win,
        "win_rate":        n_win / len(trades) if trades else 0.0,
        "total_pnl":       sum(pnl_vals),
        "total_gross_pnl": sum(t["gross_pnl"] for t in trades),
        "avg_pnl":         float(np.mean(pnl_vals)) if pnl_vals else 0.0,
        "pnl_std":         float(np.std(pnl_vals))  if pnl_vals else 0.0,
        "avg_hold_s":      float(np.mean([t["hold_s"] for t in trades])) if trades else 0.0,
        "exit_counts":     exit_counts,
        "trades":          trades,
        "edge_series":     edge_series,
        "kelly_tiers":     kelly_tiers,
        "exit_threshold":  exit_threshold,
        "max_hold_s":      max_hold_s,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(flat_rows: list[dict], result: dict, model_name: str,
                 train_frac: float, save_path: str | None = None) -> None:
    if not HAS_MPL:
        print("matplotlib not available — skipping plot.")
        return

    trades      = result["trades"]
    edge_series = result.get("edge_series", [])
    fig, axes   = plt.subplots(4, 1, figsize=(14, 16))

    min_entry_floor = result["kelly_tiers"][-1][0] if result["kelly_tiers"] else 0.0
    fig.suptitle(
        f"Full Simulation: {model_name}  |  "
        f"entry floor={min_entry_floor:.3f}  exit<{result['exit_threshold']:.3f}  hold<={result['max_hold_s']}s",
        fontsize=12, fontweight="bold"
    )

    ts_dt       = [datetime.datetime.fromtimestamp(r["ts_ns"] / 1e9) for r in flat_rows]
    yes_bid_arr = [r["yes_bid"] for r in flat_rows]
    yes_ask_arr = [r["yes_ask"] for r in flat_rows]
    yes_mid_arr = [r["yes_mid"] for r in flat_rows]

    # Panel 1: YES price + trades
    ax = axes[0]
    ax.fill_between(ts_dt, yes_bid_arr, yes_ask_arr, alpha=0.2, color="steelblue")
    ax.plot(ts_dt, yes_mid_arr, color="steelblue", lw=1, label="YES mid")

    if trades:
        entry_x = [datetime.datetime.fromtimestamp(t["entry_ts"]) for t in trades]
        entry_y = [t["entry_p"] for t in trades]
        exit_x  = [datetime.datetime.fromtimestamp(t["exit_ts"])  for t in trades]
        exit_y  = [t["exit_p"] for t in trades]
        ax.scatter(entry_x, entry_y, marker="^", color="green", s=60, zorder=5, label="entry")
        ax.scatter(exit_x,  exit_y,  marker="v", color="red",   s=60, zorder=5, label="exit")
        for t in trades:
            ax.plot(
                [datetime.datetime.fromtimestamp(t["entry_ts"]),
                 datetime.datetime.fromtimestamp(t["exit_ts"])],
                [t["entry_p"], t["exit_p"]],
                color="green" if t["pnl"] > 0 else "red", alpha=0.4, lw=1
            )

    if train_frac < 1.0 and train_frac > 0.0:
        split_ts = flat_rows[int(len(flat_rows) * train_frac)]["ts_ns"]
        ax.axvline(datetime.datetime.fromtimestamp(split_ts / 1e9),
                   color="gray", lw=1.5, linestyle="--",
                   label=f"train/test split ({train_frac:.0%})")

    ax.set_ylabel("YES price")
    ax.set_title("YES Market Price — Entries/Exits")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)

    # Panel 2: Cumulative P&L
    ax = axes[1]
    if trades:
        exit_times = [datetime.datetime.fromtimestamp(t["exit_ts"]) for t in trades]
        cum_pnl    = np.cumsum([t["pnl"] for t in trades])
        ax.plot(exit_times, cum_pnl, color="purple", lw=2, marker="o", ms=4)
        ax.axhline(0, color="gray", lw=0.8)
        ax.fill_between(exit_times, cum_pnl, 0,
                        where=[p >= 0 for p in cum_pnl], alpha=0.2, color="green")
        ax.fill_between(exit_times, cum_pnl, 0,
                        where=[p < 0 for p in cum_pnl], alpha=0.2, color="red")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"Cumulative P&L — {len(trades)} trades, "
                 f"win rate {result['win_rate']*100:.0f}%")
    ax.grid(alpha=0.3)

    # Panel 3: Edge over time
    ax = axes[2]
    if edge_series:
        edge_ts   = [datetime.datetime.fromtimestamp(ts / 1e9) for ts, _ in edge_series]
        edge_vals = [e for _, e in edge_series]

        if edge_ts:
            ax.plot(edge_ts, edge_vals, color="darkorange", lw=0.8, alpha=0.7, label="edge (q - ask - fee)")
            ax.axhline(0, color="gray", lw=0.8)
            # Draw Kelly tier floors as horizontal lines
            for floor, frac in result["kelly_tiers"]:
                ax.axhline(floor, color="steelblue", lw=1, linestyle="--", alpha=0.6,
                           label=f"tier floor {floor:.3f}")
            ax.axhline(result["exit_threshold"], color="red", lw=1, linestyle=":",
                       alpha=0.7, label=f"exit threshold {result['exit_threshold']:.3f}")
            # Mark entry timestamps
            if trades:
                for t in trades:
                    et = datetime.datetime.fromtimestamp(t["entry_ts"])
                    ax.axvline(et, color="green", lw=0.6, alpha=0.4)
            ax.set_ylabel("Net edge")
            ax.set_title("Edge over time (q_settled − ask − fee)  |  dashed = Kelly floors  |  green lines = entries")
            ax.legend(fontsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.grid(alpha=0.3)

    # Panel 4: q_settled vs per-trade P&L
    ax = axes[3]
    if trades:
        q_vals  = [t["q_settled"] for t in trades]
        pnl_arr = [t["pnl"] for t in trades]
        colors  = ["green" if p > 0 else "red" for p in pnl_arr]
        ax.scatter(q_vals, pnl_arr, c=colors, alpha=0.6, s=40)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("q_settled at entry")
        ax.set_ylabel("P&L per trade ($)")
        ax.set_title("q_settled vs Trade P&L")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Resolve model path
# ---------------------------------------------------------------------------

def resolve_model_path(arg: str | None) -> Path:
    if arg is None:
        jsons = sorted(MODEL_FITS_DIR.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not jsons:
            sys.exit("No saved models in model_fits/. Run scripts/train_model.py first.")
        print("Saved models:")
        for i, j in enumerate(jsons):
            try:
                meta = json.loads(j.read_text())
                r2   = meta.get("r2_held_out", float("nan"))
                print(f"  [{i}] {j.stem}  (R2_hld={r2:.3f})")
            except Exception:
                print(f"  [{i}] {j.stem}")
        choice = input("Enter number (or path): ").strip()
        try:
            return jsons[int(choice)].with_suffix(".pkl")
        except ValueError:
            return Path(choice)

    p = Path(arg)
    for base in (MODEL_FITS_DIR, _REPO, Path.cwd()):
        candidate = base / p
        pkl = Path(str(candidate) if str(candidate).endswith(".pkl") else str(candidate) + ".pkl")
        if pkl.exists():
            return pkl
    pkl = Path(str(p) if str(p).endswith(".pkl") else str(p) + ".pkl")
    if pkl.exists():
        return pkl
    sys.exit(f"Model file not found: {arg}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simulate P&L with a saved model and the current trading config")
    parser.add_argument("--model-file",  type=str, default=None)
    parser.add_argument("--run",         type=str, default=None,
                        help="Path to a data/<run> folder (popup if omitted)")
    parser.add_argument("--asset",       type=str, default="BTC")
    parser.add_argument("--train-frac",  type=float, default=0.0,
                        help="If > 0, only simulate on the last (1-frac) portion")
    parser.add_argument("--wallet",      type=float, default=1000.0,
                        help="Simulated wallet size (default $1000)")
    parser.add_argument("--exit",        type=float, default=LAG_CLOSE_THRESHOLD,
                        help=f"Exit threshold (default {LAG_CLOSE_THRESHOLD} from config)")
    parser.add_argument("--hold-s",      type=int,   default=MAX_HOLD_S,
                        help=f"Force exit after this many seconds (default {MAX_HOLD_S})")
    parser.add_argument("--min-tau",     type=int,   default=FALLBACK_TAU_S)
    parser.add_argument("--entry-mode",  type=str,   default=ENTRY_MODE,
                        choices=["taker", "maker"])
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--save",        type=str, default=None,
                        help="Save chart PNG to this path")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_file)
    print(f"Loading model: {model_path}", flush=True)
    model = load_model(str(model_path))

    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta   = json.loads(meta_path.read_text())
        r2_hld = meta.get("r2_held_out", float("nan"))
        n_samp = meta.get("n_samples", "?")
        print(f"  R2_hld={r2_hld:.3f}  n={n_samp}", flush=True)

    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to test on")
    asset     = args.asset.upper()
    tick_path = run_dir / f"ticks_{asset}.csv"
    if not tick_path.exists():
        sys.exit(f"ERROR: {tick_path} not found.")

    print(f"Loading ticks: {tick_path}", flush=True)
    windows = load_ticks(tick_path)
    n_ticks = sum(len(v) for v in windows.values())
    print(f"  {len(windows)} windows  {n_ticks} ticks", flush=True)

    flat_rows: list[dict] = []
    flat_X:    list       = []
    for rows in sorted(windows.values(), key=lambda v: v[0]["ts_ns"]):
        entry_tau_s = rows[0]["tau_s"]
        for i, r in enumerate(rows):
            elapsed = entry_tau_s - r["tau_s"]
            fv = build_feature_row(rows, i) if elapsed >= 30 else None
            flat_rows.append(r)
            flat_X.append(fv)

    valid = sum(1 for x in flat_X if x is not None)
    print(f"  {valid} ticks with complete features", flush=True)

    if args.train_frac > 0.0:
        split = int(len(flat_rows) * args.train_frac)
        flat_rows = flat_rows[split:]
        flat_X    = flat_X[split:]
        print(f"  Using last {100*(1-args.train_frac):.0f}% as test set ({len(flat_rows)} ticks)",
              flush=True)

    print(f"\nTrading config from config.py:")
    print(f"  Kelly tiers: {KELLY_TIERS}")
    print(f"  Exit threshold: {args.exit}  Max hold: {args.hold_s}s  Min tau: {args.min_tau}s")
    print(f"  Entry mode: {args.entry_mode}  Wallet: ${args.wallet:,.0f}", flush=True)

    result = run_simulation(
        flat_rows, flat_X, model,
        KELLY_TIERS, args.exit, args.hold_s, args.min_tau,
        args.wallet, args.entry_mode,
    )

    print(f"\n{'='*60}")
    print(f"  FULL SIMULATION RESULTS  ({model_path.stem})")
    print(f"{'='*60}")
    print(f"  Ticks replayed:  {result['n_ticks']}")
    print(f"  Trades:          {result['n_trades']}")
    print(f"  Win rate:        {result['win_rate']*100:.0f}%  "
          f"({result['n_win']}/{result['n_trades']})")
    print(f"  Gross P&L:       ${result['total_gross_pnl']:+.4f}  (before fees)")
    print(f"  Net P&L:         ${result['total_pnl']:+.4f}  (after fees)")
    print(f"  Avg net/trade:   ${result['avg_pnl']:+.4f}  (std ${result['pnl_std']:.4f})")
    print(f"  Avg hold time:   {result['avg_hold_s']:.0f}s")
    print(f"  Exit breakdown:  {result['exit_counts']}")
    print(f"{'='*60}")

    if result["n_trades"] > 0:
        print(f"\n  Individual trades:")
        for t in result["trades"]:
            entry_time = datetime.datetime.fromtimestamp(t["entry_ts"]).strftime("%H:%M:%S")
            print(f"    {entry_time}  buy@{t['entry_p']:.3f} -> sell@{t['exit_p']:.3f}  "
                  f"q={t['q_settled']:.3f}  pnl=${t['pnl']:+.4f}  "
                  f"hold={t['hold_s']:.0f}s  tier={t['tier']}  [{t['exit_reason']}]")

    if not args.no_plot:
        plot_results(flat_rows, result, model_path.stem,
                     args.train_frac, save_path=args.save)


if __name__ == "__main__":
    main()
