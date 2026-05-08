"""
tune_trading_knobs.py — Grid-search over Kelly tiers and exit thresholds to
maximize net P&L on historical tick data given a trained model.

This is step 3 of the offline workflow:
  1. python scripts/run/run_kalshi_bot.py          # dry run, collect ticks
  2. python scripts/train_model.py                 # fit model
  3. python scripts/tune_trading_knobs.py          # THIS — optimize Kelly/exit config
  4. python scripts/test_all.py                    # final simulation with tuned config
  5. python scripts/run/run_kalshi_bot.py --model-file ... --live-orders

The sweep covers:
  - Kelly tier edge floors (at what edge level to enter)
  - Exit threshold (when to close — edge compressed this much)
  - Max hold seconds (force-exit after this hold time)

The output tells you:
  - Best combo by net P&L
  - Best combo by Sharpe (P&L / std deviation)
  - Best combo by win rate
  - Copy-paste config snippet for config.py

Usage:
  python scripts/tune_trading_knobs.py
  python scripts/tune_trading_knobs.py --model-file model_fits/btc_lgbm_v1.pkl
  python scripts/tune_trading_knobs.py --model-file model_fits/btc.pkl --run data/2026-05-07_BTC
  python scripts/tune_trading_knobs.py --train-frac 0.6   # sweep on last 40%
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

from betbot.kalshi.model import load_model
from betbot.kalshi.config import (
    THETA_FEE_TAKER, DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    FALLBACK_TAU_S, ENTRY_MODE, SIZE_MIN_USD, SIZE_MAX_USD,
    MIN_ENTRY_INTERVAL_S,
)
from pick_run import pick_run_folder

MODEL_FITS_DIR = _REPO / "model_fits"


# ---------------------------------------------------------------------------
# Tick loading + feature building (same as test_all.py)
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
# Pre-compute model predictions for all ticks (vectorized for speed)
# ---------------------------------------------------------------------------

def precompute_predictions(flat_X: list, model) -> list[float | None]:
    """Return q_settled for every tick (None if fv is None or prediction failed)."""
    preds = []
    for fv in flat_X:
        if fv is None:
            preds.append(None)
        else:
            q = model.q_settled_from_array(fv)
            preds.append(q if (q is not None and math.isfinite(q)) else None)
    return preds


# ---------------------------------------------------------------------------
# Fast simulator (takes pre-computed predictions)
# ---------------------------------------------------------------------------

def _fee(p: float) -> float:
    return THETA_FEE_TAKER * p * (1.0 - p)


def simulate_fast(
    flat_rows: list[dict],
    preds: list[float | None],
    kelly_tiers: list[tuple[float, float]],
    exit_threshold: float,
    max_hold_s: int,
    min_tau_s: int,
    wallet: float,
    entry_mode: str,
    stop_loss: float | None = None,
) -> list[dict]:
    """
    Fast single-pass simulator. All P&L is after fees.
    Entry: buy at ask (taker) or bid (maker).
    Exit:  sell at bid (always taker on exit).
    Fees:  THETA * p * (1-p) per dollar per leg.
    Stop:  exit if cur_edge < entry_edge - stop_loss (if stop_loss is not None).
    """
    trades          = []
    pos             = None
    prev_ts_ns      = None
    last_entry_ns   = 0

    for r, q_set in zip(flat_rows, preds):
        cur_ts_ns  = r["ts_ns"]
        cur_ticker = r["window_ticker"]

        # Force-close on window boundary or large gap (> 5s)
        if pos is not None:
            gap_s = (cur_ts_ns - (prev_ts_ns or cur_ts_ns)) / 1e9
            if cur_ticker != pos["window_ticker"] or gap_s > 5.0:
                # No exit fill — just lose the entry fee
                trades.append({
                    "pnl":         -pos["fee_entry_usd"],
                    "gross_pnl":   0.0,
                    "hold_s":      ((prev_ts_ns or cur_ts_ns) - pos["entry_ts_ns"]) / 1e9,
                    "exit_reason": "window_gap",
                })
                pos = None

        prev_ts_ns = cur_ts_ns

        if q_set is None:
            continue

        tau     = r["tau_s"]
        yes_ask = float(r["yes_ask"])
        yes_bid = float(r["yes_bid"])
        yes_mid = float(r["yes_mid"])

        # Entry price and fee
        if entry_mode == "taker":
            entry_p    = yes_ask
            fee_entry  = _fee(yes_ask)   # dollars per dollar bet
        else:
            entry_p    = yes_bid
            fee_entry  = 0.0

        # Net edge = predicted_future_bid - current_ask - entry_fee
        edge_net = q_set - entry_p - fee_entry

        # --- Exit ---
        if pos is not None:
            hold_s   = (cur_ts_ns - pos["entry_ts_ns"]) / 1e9
            # Re-evaluate edge using current prices — exit when lag has closed
            cur_edge = q_set - yes_ask - _fee(yes_ask)

            stopped   = stop_loss is not None and cur_edge < pos["entry_edge"] - stop_loss
            lag_done  = cur_edge < exit_threshold
            timed_out = hold_s >= max_hold_s
            tau_out   = tau < min_tau_s

            if lag_done or timed_out or tau_out or stopped:
                exit_p    = yes_bid                             # sell into bid
                contracts = pos["size_usd"] / max(pos["entry_p"], 1e-6)
                gross     = (exit_p - pos["entry_p"]) * contracts
                fee_exit  = _fee(exit_p) * pos["size_usd"]     # exit leg fee
                pnl       = gross - pos["fee_entry_usd"] - fee_exit
                trades.append({
                    "pnl":         pnl,
                    "gross_pnl":   gross,
                    "hold_s":      hold_s,
                    "exit_reason": (
                        "stopped"    if stopped
                        else "lag_closed" if lag_done
                        else "max_hold"   if timed_out
                        else "min_tau"
                    ),
                })
                pos = None

        # --- Entry ---
        secs_since_entry = (cur_ts_ns - last_entry_ns) / 1e9
        if (pos is None
                and tau > min_tau_s
                and DECISION_YES_MID_MIN <= yes_mid <= DECISION_YES_MID_MAX
                and secs_since_entry >= MIN_ENTRY_INTERVAL_S):
            tier_usd = 0.0
            for floor, frac in kelly_tiers:
                if edge_net >= floor:
                    tier_usd = min(wallet * frac, SIZE_MAX_USD)
                    break
            if tier_usd >= SIZE_MIN_USD:
                pos = {
                    "entry_ts_ns":   cur_ts_ns,
                    "entry_p":       entry_p,
                    "entry_edge":    edge_net,
                    "size_usd":      tier_usd,
                    "fee_entry_usd": fee_entry * tier_usd,
                    "window_ticker": cur_ticker,
                }
                last_entry_ns = cur_ts_ns

    return trades


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
# Kelly tier templates: (name, [(edge_floor, wallet_fraction), ...])
# edge_floor is NET edge after fees — so already fee-adjusted at entry.
# wallet_fraction is the fraction of wallet to bet per trade.
_TIER_TEMPLATES: list[tuple[str, list[tuple[float, float]]]] = [
    # Aggressive: only enter on strong signals, size up
    ("agg_2t_hi",       [(0.12, 0.08), (0.07, 0.05)]),
    ("agg_2t_med",      [(0.10, 0.06), (0.06, 0.04)]),
    ("agg_1t_hi",       [(0.10, 0.08)]),
    ("agg_1t_med",      [(0.08, 0.06)]),
    ("agg_1t_lo",       [(0.06, 0.05)]),
    # Standard: multi-tier, enter on moderate signal too
    ("std_4t",          [(0.20, 0.08), (0.10, 0.06), (0.05, 0.04), (0.02, 0.02)]),
    ("std_3t_hi",       [(0.15, 0.08), (0.07, 0.05), (0.03, 0.025)]),
    ("std_3t_med",      [(0.10, 0.06), (0.05, 0.04), (0.02, 0.02)]),
    ("std_3t_lo",       [(0.08, 0.05), (0.04, 0.03), (0.02, 0.015)]),
    ("std_2t_hi",       [(0.10, 0.06), (0.04, 0.03)]),
    ("std_2t_med",      [(0.07, 0.05), (0.03, 0.025)]),
    ("std_2t_lo",       [(0.05, 0.04), (0.02, 0.02)]),
    # Conservative: small size, only enter on clear signal
    ("cons_1t_hi",      [(0.05, 0.03)]),
    ("cons_1t_med",     [(0.04, 0.025)]),
    ("cons_1t_lo",      [(0.03, 0.02)]),
    # Very tight entry: only trade the very best setups
    ("tight_1t",        [(0.15, 0.06)]),
    ("tight_2t",        [(0.15, 0.06), (0.08, 0.04)]),
]

# Exit threshold: close when net edge drops below this
# Lower = hold longer waiting for full lag closure; higher = take profits faster
_EXIT_THRESHOLDS = [0.000, 0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030]

# Max hold: force-exit after N seconds regardless of edge
_MAX_HOLDS = [10, 15, 20, 30, 45, 60, 90, 120]

# Stop loss: exit if edge drops this far below entry edge
# (None = disabled — rely on exit_threshold + max_hold only)
_STOP_LOSSES = [None, 0.03, 0.05, 0.08]


def run_sweep(flat_rows, preds, wallet, entry_mode, min_tau_s) -> list[dict]:
    results = []
    total = len(_TIER_TEMPLATES) * len(_EXIT_THRESHOLDS) * len(_MAX_HOLDS) * len(_STOP_LOSSES)
    done  = 0

    for tier_name, kelly_tiers in _TIER_TEMPLATES:
        for xt in _EXIT_THRESHOLDS:
            for mh in _MAX_HOLDS:
                for sl in _STOP_LOSSES:
                    trades = simulate_fast(
                        flat_rows, preds, kelly_tiers, xt, mh, min_tau_s,
                        wallet, entry_mode, stop_loss=sl,
                    )
                    pnl_vals  = [t["pnl"] for t in trades]
                    n_win     = sum(1 for p in pnl_vals if p > 0)
                    total_pnl = sum(pnl_vals)
                    std_pnl   = float(np.std(pnl_vals)) if len(pnl_vals) > 1 else 0.0
                    sharpe    = total_pnl / (std_pnl * math.sqrt(max(len(pnl_vals), 1)) + 1e-9)
                    n_lag     = sum(1 for t in trades if t["exit_reason"] == "lag_closed")
                    avg_hold  = float(np.mean([t["hold_s"] for t in trades])) if trades else 0.0

                    results.append({
                        "tier_name":    tier_name,
                        "kelly_tiers":  kelly_tiers,
                        "exit":         xt,
                        "hold_s":       mh,
                        "stop_loss":    sl,
                        "n_trades":     len(trades),
                        "win_rate":     n_win / len(trades) if trades else 0.0,
                        "pct_lag_closed": n_lag / len(trades) if trades else 0.0,
                        "avg_hold_s":   avg_hold,
                        "total_pnl":    total_pnl,
                        "avg_pnl":      float(np.mean(pnl_vals)) if pnl_vals else 0.0,
                        "std_pnl":      std_pnl,
                        "sharpe":       sharpe,
                    })
                    done += 1
                    if done % 100 == 0:
                        print(f"  {done}/{total}  "
                              f"tiers={tier_name}  exit={xt:.3f}  hold={mh}s  "
                              f"sl={'none' if sl is None else sl:.3f}  "
                              f"n={len(trades)}  pnl={total_pnl:+.4f}",
                              flush=True)

    return results


def print_top(results: list[dict], key: str, label: str, n: int = 10) -> dict:
    # Filter to results with at least 5 trades to avoid noise
    valid = [r for r in results if r["n_trades"] >= 5]
    if not valid:
        valid = results
    ranked = sorted(valid, key=lambda r: r[key], reverse=True)
    print(f"\n{'='*90}")
    print(f"  Top {n} by {label}:")
    print(f"{'='*90}")
    hdr = (f"{'Tier name':<18} {'exit':>6} {'hold':>5} {'stop':>6} "
           f"{'n':>5} {'win%':>5} {'lag%':>5} {'hold_s':>6} "
           f"{'pnl':>10} {'avg':>8} {'sharpe':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in ranked[:n]:
        sl_str = "none" if r["stop_loss"] is None else f"{r['stop_loss']:.3f}"
        print(f"{r['tier_name']:<18} {r['exit']:>6.3f} {r['hold_s']:>5} {sl_str:>6} "
              f"{r['n_trades']:>5} {r['win_rate']*100:>4.0f}% "
              f"{r['pct_lag_closed']*100:>4.0f}% {r['avg_hold_s']:>6.0f} "
              f"{r['total_pnl']:>+10.4f} {r['avg_pnl']:>+8.5f} {r['sharpe']:>+7.3f}")
    return ranked[0] if ranked else {}


def print_config_snippet(best: dict, asset: str = "") -> None:
    tiers  = best.get("kelly_tiers", [])
    exit_t = best.get("exit", 0.005)
    hold_s = best.get("hold_s", 60)
    sl     = best.get("stop_loss", None)

    label = f" ({asset})" if asset else ""
    print(f"\n{'='*70}")
    print(f"  Suggested config.py snippet{label} — copy-paste and verify:")
    print(f"{'='*70}")
    print(f"\nKELLY_TIERS = [")
    for floor, frac in tiers:
        print(f"    ({floor}, {frac}),")
    print(f"]")
    print(f"\nLAG_CLOSE_THRESHOLD = {exit_t}")
    print(f"MAX_HOLD_S          = {hold_s}")
    if sl is not None:
        print(f"STOP_THRESHOLD      = {sl}")
    else:
        print(f"# STOP_THRESHOLD disabled (best config uses max_hold only)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep Kelly tiers and exit thresholds to maximise P&L")
    parser.add_argument("--model-file", type=str, default=None)
    parser.add_argument("--run",        type=str, default=None,
                        help="Path to a data/<run> folder (popup if omitted)")
    parser.add_argument("--asset",      type=str, default="BTC")
    parser.add_argument("--train-frac", type=float, default=0.6,
                        help="Fraction used as train split; sweep runs on the remaining "
                             "held-out portion (default 0.6 = sweep on last 40%%)")
    parser.add_argument("--wallet",     type=float, default=1000.0)
    parser.add_argument("--entry-mode", type=str, default=ENTRY_MODE,
                        choices=["taker", "maker"])
    parser.add_argument("--min-tau",    type=int, default=FALLBACK_TAU_S)
    args = parser.parse_args()

    # --- Load model ---
    if args.model_file is None:
        jsons = sorted(MODEL_FITS_DIR.glob("**/*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
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
            model_path = jsons[int(choice)].with_suffix(".pkl")
        except ValueError:
            model_path = Path(choice)
    else:
        p = Path(args.model_file)
        pkl = p if str(p).endswith(".pkl") else Path(str(p) + ".pkl")
        # Try absolute, then relative to repo root, then relative to model_fits/
        if pkl.is_absolute() and pkl.exists():
            model_path = pkl
        elif (_REPO / pkl).exists():
            model_path = _REPO / pkl
        elif (MODEL_FITS_DIR / pkl).exists():
            model_path = MODEL_FITS_DIR / pkl
        else:
            model_path = _REPO / pkl

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}", flush=True)
    model = load_model(str(model_path))

    meta_path = model_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print(f"  R2_hld={meta.get('r2_held_out', float('nan')):.3f}  "
              f"n={meta.get('n_samples', '?')}", flush=True)

    # --- Load ticks ---
    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to tune on")
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

    # Restrict to held-out split
    if args.train_frac > 0.0:
        split = int(len(flat_rows) * args.train_frac)
        flat_rows = flat_rows[split:]
        flat_X    = flat_X[split:]
        print(f"  Using last {100*(1-args.train_frac):.0f}% as held-out sweep data "
              f"({len(flat_rows)} ticks)", flush=True)

    # Pre-compute q_settled predictions once for all ticks
    print("Pre-computing model predictions...", flush=True)
    preds = precompute_predictions(flat_X, model)
    n_valid = sum(1 for p in preds if p is not None)
    print(f"  {n_valid}/{len(preds)} ticks have valid predictions", flush=True)

    # Run sweep
    combos = (len(_TIER_TEMPLATES) * len(_EXIT_THRESHOLDS)
              * len(_MAX_HOLDS) * len(_STOP_LOSSES))
    print(f"\nRunning sweep ({asset}): "
          f"{len(_TIER_TEMPLATES)} tiers x {len(_EXIT_THRESHOLDS)} exits x "
          f"{len(_MAX_HOLDS)} holds x {len(_STOP_LOSSES)} stops = {combos} combos...",
          flush=True)

    results = run_sweep(flat_rows, preds, args.wallet, args.entry_mode, args.min_tau)

    best_pnl    = print_top(results, "total_pnl", "total net P&L")
    best_sharpe = print_top(results, "sharpe",    "Sharpe ratio")
    best_wr     = print_top(results, "win_rate",  "win rate (min 5 trades)")

    print(f"\n{'='*70}")
    print(f"  Recommendation ({asset}):")
    print(f"{'='*70}")
    print(f"  Best P&L:    tiers={best_pnl.get('tier_name')}  "
          f"exit={best_pnl.get('exit')}  hold={best_pnl.get('hold_s')}s  "
          f"stop={'none' if best_pnl.get('stop_loss') is None else best_pnl.get('stop_loss')}  "
          f"pnl={best_pnl.get('total_pnl', 0):+.4f}")
    print(f"  Best Sharpe: tiers={best_sharpe.get('tier_name')}  "
          f"exit={best_sharpe.get('exit')}  hold={best_sharpe.get('hold_s')}s  "
          f"stop={'none' if best_sharpe.get('stop_loss') is None else best_sharpe.get('stop_loss')}  "
          f"sharpe={best_sharpe.get('sharpe', 0):+.3f}")

    # Print config snippet for best-by-Sharpe (more stable than raw P&L)
    print_config_snippet(best_sharpe, asset=asset)

    print("Verify with:")
    print(f"  python scripts/test_all.py --model-file {model_path} --asset {asset}")


if __name__ == "__main__":
    main()
