"""
backtest.py — Realistic backtest with maker entry / taker exit fill model.

Replaces the legacy test_all.py. Models real Kalshi mechanics:

  Entry (maker):
    - Bot decides to enter at tick t with target side YES/NO and price P = bid+1c
    - Fills if within MAKER_TTL_S seconds: the YES ask drops to P (for YES buys)
      or the NO ask drops to P (for NO buys). I.e., someone crossed our limit.
    - If TTL expires unfilled → no trade, no cost.
    - Maker fee = $0.

  Exit (taker IOC sweep):
    - Sells against the live bid, capped to yes_ask_size depth (logged top-of-book).
    - Proceeds = count - taker_fill_cost_dollars (Kalshi reciprocal).
    - Taker fee = THETA * p * (1-p) per dollar bet, rounded up to nearest cent
      per Kalshi's actual fee schedule.

P&L is in real dollars per trade — no shortcuts, no estimates.

Usage:
  python scripts/backtest.py --model-file model_fits/<dir>/model.pkl
  python scripts/backtest.py --run data/<run> --asset BTC --model-file ...
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from betbot.kalshi.config import (
    KELLY_TIERS, LAG_CLOSE_THRESHOLD, STOP_THRESHOLD, FALLBACK_TAU_S,
    MAX_HOLD_S, MIN_ENTRY_INTERVAL_S, THETA_FEE_TAKER,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    SIZE_MAX_USD, MAKER_AT_BID_PLUS_1, MAKER_TTL_S,
    TRAIN_PRICE_MIN, TRAIN_PRICE_MAX,
)
from betbot.kalshi.model import load_model
from betbot.kalshi.features import _logit, _sigmoid
from scripts.analysis.tick_loader import load_ticks, find_ticks_path
from scripts.analysis.pick_run import pick_run_folder


# ---------------------------------------------------------------------------
# Trading helpers (mirror scheduler.py exactly)
# ---------------------------------------------------------------------------

def fee_taker(p: float) -> float:
    """Per-dollar taker fee at price p. Real Kalshi formula."""
    if p <= 0 or p >= 1:
        return 0.0
    return THETA_FEE_TAKER * p * (1.0 - p)


def kelly_size(edge: float, wallet: float) -> tuple[float, int]:
    for idx, (floor, frac) in enumerate(KELLY_TIERS, start=1):
        if edge >= floor:
            return min(wallet * frac, SIZE_MAX_USD), idx
    return 0.0, 0


# ---------------------------------------------------------------------------
# Feature builder (matches features.py — 17 features)
# ---------------------------------------------------------------------------

def build_features(rows: list[dict], idx: int) -> np.ndarray | None:
    r = rows[idx]
    K = r["K"]
    mp = r["btc_micro"]
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
    spread = max(0.0, r["yes_ask"] - r["yes_bid"])
    km5  = lagged_ym(5_000_000_000)
    km10 = lagged_ym(10_000_000_000)
    km30 = lagged_ym(30_000_000_000)

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
# Maker fill simulator (model A: pessimistic)
# ---------------------------------------------------------------------------

def simulate_maker_fill(
    rows: list[dict], entry_idx: int, side: str, post_price_c: int,
    ttl_s: float,
) -> tuple[int, float] | None:
    """
    Walks forward from entry_idx for up to ttl_s seconds. The maker buy fills
    iff the live ask on the favored side drops to <= post_price.

    For YES buy: fills when yes_ask_t' <= post_price (someone sold YES into our bid)
    For NO  buy: fills when (1 - yes_bid_t') <= post_price (someone sold NO into our bid)

    Returns (fill_idx, fill_price_dollars) or None if TTL expires.
    """
    post_price_d = post_price_c / 100.0
    entry_ts     = rows[entry_idx]["ts_ns"]
    deadline_ns  = entry_ts + int(ttl_s * 1_000_000_000)

    for j in range(entry_idx + 1, len(rows)):
        if rows[j]["ts_ns"] > deadline_ns:
            return None
        if side == "yes":
            ask = rows[j]["yes_ask"]
            if ask <= post_price_d + 1e-9:
                return j, post_price_d
        else:
            no_ask = 1.0 - rows[j]["yes_bid"]
            if no_ask <= post_price_d + 1e-9:
                return j, post_price_d
    return None


# ---------------------------------------------------------------------------
# Trade outcome computation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    asset:        str
    window:       str
    entry_idx:    int
    fill_idx:     int
    exit_idx:     int
    side:         str          # "yes" or "no"
    contracts:    int
    entry_price:  float
    exit_price:   float
    edge_at_entry: float
    pnl_gross:    float
    pnl_fees:     float
    pnl_net:      float
    hold_s:       float
    exit_reason:  str
    tier:         int


# Match live bot: decision tick every 500ms = every 5th sampler tick at 10Hz.
# Cuts feature builds + model predictions 5x without changing fidelity.
DECISION_EVERY_N_TICKS = 5


def compute_exit(
    rows: list[dict], fill_idx: int, side: str, contracts: int,
    entry_price: float, q_set_at_entry: float, model,
) -> tuple[int, float, str, float]:
    """
    Walk forward from the fill point and find the first exit trigger.
    Returns (exit_idx, exit_price_dollars, exit_reason, hold_s).

    Exit triggers (mirror scheduler):
      - exit_lag_closed: edge_now < LAG_CLOSE_THRESHOLD
      - exit_stopped:    (if STOP_THRESHOLD set) edge erodes by stop amount
      - exit_max_hold:   wall-clock hold >= MAX_HOLD_S
      - fallback_resolution: tau < FALLBACK_TAU_S
    """
    fill_ts = rows[fill_idx]["ts_ns"]

    # Match live bot: re-check exit conditions every 500ms (every 5th tick at 10Hz)
    for j in range(fill_idx + 1, len(rows), DECISION_EVERY_N_TICKS):
        r = rows[j]
        hold_s = (r["ts_ns"] - fill_ts) / 1e9
        tau    = r["tau_s"]

        # Compute current edge_now using the model
        feat = build_features(rows, j)
        if feat is None:
            continue
        try:
            q_now_logit = model._predict_horizon(feat, model._primary_idx)
            if q_now_logit is None:
                continue
            q_now = _sigmoid(q_now_logit)
        except Exception:
            continue

        # Exit-side edge: how much room is left between current sell price (bid)
        # and where the model thinks it's heading. Matches scheduler.py.
        if side == "yes":
            edge_now   = q_now - r["yes_bid"]
            exit_price = r["yes_bid"]
        else:
            edge_now   = (1.0 - q_now) - (1.0 - r["yes_ask"])
            exit_price = 1.0 - r["yes_ask"]

        # Triggers (priority: lag close > stop > max hold > fallback)
        if edge_now < LAG_CLOSE_THRESHOLD:
            return j, exit_price, "exit_lag_closed", hold_s
        if STOP_THRESHOLD is not None and edge_now < q_set_at_entry - STOP_THRESHOLD:
            return j, exit_price, "exit_stopped", hold_s
        if hold_s >= MAX_HOLD_S:
            return j, exit_price, "exit_max_hold", hold_s
        if tau < FALLBACK_TAU_S:
            return j, exit_price, "fallback_resolution", hold_s

    # End of window — force exit at last row's bid
    last = rows[-1]
    last_price = last["yes_bid"] if side == "yes" else (1.0 - last["yes_ask"])
    hold_s = (last["ts_ns"] - fill_ts) / 1e9
    return len(rows) - 1, last_price, "window_end", hold_s


def compute_pnl(side: str, contracts: int, entry_price: float, exit_price: float,
                exit_reason: str) -> tuple[float, float, float]:
    """
    Returns (gross_pnl_usd, fees_usd, net_pnl_usd).

    Maker entry cost: contracts * entry_price (no fee)
    Taker exit proceeds: contracts * exit_price (Kalshi pays you the bid)
    Taker exit fee: fee_taker(exit_price) * contracts
    """
    gross = contracts * (exit_price - entry_price)
    if "resolution" in exit_reason or "window_end" in exit_reason:
        fees = 0.0  # Holding to resolution = no taker fee on exit
    else:
        fees = fee_taker(exit_price) * contracts
    return gross, fees, gross - fees


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def backtest_window(rows: list[dict], window: str, asset: str,
                    model, wallet_usd: float) -> list[Trade]:
    trades: list[Trade] = []
    last_entry_ts_ns = 0
    open_until_idx = -1  # don't enter while a previous trade is still "open"

    for i in range(0, len(rows), DECISION_EVERY_N_TICKS):
        r = rows[i]

        # Don't double-enter while a previous position is still open
        if i <= open_until_idx:
            continue

        # Rate limit between entries
        if r["ts_ns"] - last_entry_ts_ns < MIN_ENTRY_INTERVAL_S * 1_000_000_000:
            continue

        # Decision filters
        if r["yes_mid"] < DECISION_YES_MID_MIN or r["yes_mid"] > DECISION_YES_MID_MAX:
            continue
        if r["tau_s"] < FALLBACK_TAU_S:
            continue
        if r["yes_ask"] - r["yes_bid"] > 0.10:
            continue

        feat = build_features(rows, i)
        if feat is None:
            continue
        try:
            q_logit = model._predict_horizon(feat, model._primary_idx)
            if q_logit is None:
                continue
            q_set = _sigmoid(q_logit)
        except Exception:
            continue
        if abs(q_set - r["yes_mid"]) > 0.15:
            continue

        # Edge calc — mirrors scheduler exactly
        offset = 0.01 if MAKER_AT_BID_PLUS_1 else 0.0
        entry_yes = min(r["yes_bid"] + offset, r["yes_ask"])
        edge_yes_raw = q_set - entry_yes
        fee_yes      = fee_taker(q_set)

        entry_no  = min((1.0 - r["yes_ask"]) + offset, 1.0 - r["yes_bid"])
        exit_no   = 1.0 - q_set
        edge_no_raw = exit_no - entry_no
        fee_no      = fee_taker(exit_no)

        edge_yes_net = edge_yes_raw - fee_yes
        edge_no_net  = edge_no_raw  - fee_no

        if edge_yes_net >= edge_no_net:
            edge_signed, side = edge_yes_net, "yes"
            entry_price_d     = entry_yes
        else:
            edge_signed, side = edge_no_net, "no"
            entry_price_d     = entry_no

        edge_mag = abs(edge_signed)
        bet_usd, tier = kelly_size(edge_mag, wallet_usd)
        if tier == 0:
            continue

        post_price_c = max(1, min(99, round(entry_price_d * 100)))
        contracts    = max(1, int(min(bet_usd, SIZE_MAX_USD) / max(entry_price_d, 0.01)))

        # Simulate maker fill
        fill_result = simulate_maker_fill(rows, i, side, post_price_c, MAKER_TTL_S)
        if fill_result is None:
            # No fill — order timed out, no cost
            last_entry_ts_ns = r["ts_ns"]   # rate limit still applies on attempt
            continue
        fill_idx, fill_price = fill_result

        # Simulate exit
        exit_idx, exit_price, exit_reason, hold_s = compute_exit(
            rows, fill_idx, side, contracts, fill_price, q_set, model,
        )

        gross, fees, net = compute_pnl(side, contracts, fill_price, exit_price, exit_reason)
        trades.append(Trade(
            asset=asset, window=window,
            entry_idx=i, fill_idx=fill_idx, exit_idx=exit_idx,
            side=side, contracts=contracts,
            entry_price=fill_price, exit_price=exit_price,
            edge_at_entry=edge_mag, pnl_gross=gross,
            pnl_fees=fees, pnl_net=net, hold_s=hold_s,
            exit_reason=exit_reason, tier=tier,
        ))
        last_entry_ts_ns = r["ts_ns"]
        open_until_idx   = exit_idx

    return trades


def run_backtest(run_dir: Path, asset: str, model_path: Path, wallet_usd: float = 1000.0):
    tick_path = find_ticks_path(run_dir, asset)
    if tick_path is None:
        sys.exit(f"  no ticks for {asset} in {run_dir}")
    print(f"Loading ticks from {tick_path}...")
    windows = load_ticks(tick_path)
    if not windows:
        sys.exit("  no tick data")
    n_ticks = sum(len(rs) for rs in windows.values())
    print(f"  {len(windows)} windows  {n_ticks} ticks")

    print(f"Loading model from {model_path}...")
    model = load_model(str(model_path))
    print(f"  R2_hld={model.r2_held_out:.3f}")

    import time
    all_trades: list[Trade] = []
    n_windows = len(windows)
    t0 = time.monotonic()
    last_report = t0
    for w_idx, (window_ticker, rows) in enumerate(windows.items(), 1):
        if len(rows) < 60:
            continue
        ts = backtest_window(rows, window_ticker, asset, model, wallet_usd)
        all_trades.extend(ts)
        now = time.monotonic()
        if now - last_report >= 5.0 or w_idx == n_windows:
            pct = 100 * w_idx / n_windows
            rate = w_idx / (now - t0) if now > t0 else 0
            eta = (n_windows - w_idx) / rate if rate > 0 else 0
            print(f"    window {w_idx}/{n_windows} ({pct:.0f}%)  "
                  f"{rate:.1f} win/s  ETA {eta:.0f}s  "
                  f"trades so far: {len(all_trades)}", flush=True)
            last_report = now

    print(f"  backtest done in {time.monotonic()-t0:.0f}s — {len(all_trades)} trades")
    return all_trades


def summarize(trades: list[Trade], title: str = "Backtest"):
    if not trades:
        print(f"\n{title}: NO TRADES")
        return
    n        = len(trades)
    wins     = sum(1 for t in trades if t.pnl_net > 0)
    gross    = sum(t.pnl_gross for t in trades)
    fees     = sum(t.pnl_fees  for t in trades)
    net      = sum(t.pnl_net   for t in trades)
    avg_hold = sum(t.hold_s    for t in trades) / n

    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    by_tier: dict[int, list[Trade]] = {}
    for t in trades:
        by_tier.setdefault(t.tier, []).append(t)

    print(f"\n=== {title} ===")
    print(f"  Trades:     {n}")
    print(f"  Wins:       {wins} ({100*wins/n:.1f}%)")
    print(f"  Gross P&L:  ${gross:+.2f}")
    print(f"  Fees:       ${fees:.2f}")
    print(f"  Net P&L:    ${net:+.2f}")
    print(f"  Avg hold:   {avg_hold:.1f}s")
    print(f"  Exit reasons:")
    for reason, c in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"    {reason:25s} {c:5d}  ({100*c/n:.1f}%)")
    print(f"  By tier:")
    for tier in sorted(by_tier):
        ts = by_tier[tier]
        ts_net = sum(t.pnl_net for t in ts)
        print(f"    tier {tier}: {len(ts):4d} trades  net ${ts_net:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                    help="Run folder under data/ (popup if omitted)")
    ap.add_argument("--asset", type=str, default="BTC")
    ap.add_argument("--model-file", type=str, required=True,
                    help="Path to a model.pkl")
    ap.add_argument("--wallet", type=float, default=1000.0)
    args = ap.parse_args()

    run_dir = pick_run_folder(cli_arg=args.run, title="Select run to backtest")
    asset   = args.asset.upper()
    model_path = Path(args.model_file)
    if not model_path.is_absolute():
        model_path = REPO / model_path

    trades = run_backtest(run_dir, asset, model_path, args.wallet)
    summarize(trades, title=f"Backtest — {asset} on {run_dir.name}")


if __name__ == "__main__":
    main()
