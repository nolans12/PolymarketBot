"""
backtest.py -- Walk-forward backtest on collected tick data.

Prompts for a data/ run folder (Tkinter popup), then loads ticks_<ASSET>.csv
from that folder. Builds lag features matching the live model schema, fits ridge
regression, simulates trading on the held-out period, and reports P&L.

Fees are included: THETA * p * (1 - p) per leg (taker on both legs by default).

Usage:
  python scripts/analysis/backtest.py
  python scripts/analysis/backtest.py --run data/2026-05-07_00-15-00_BTC --asset BTC
  python scripts/analysis/backtest.py --entry 0.025
  python scripts/analysis/backtest.py --hold-s 45
  python scripts/analysis/backtest.py --train-frac 0.65
  python scripts/analysis/backtest.py --no-plot
  python scripts/analysis/backtest.py --sweep
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_run import pick_run_folder

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from betbot.kalshi.config import (
    THETA_FEE_TAKER, TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    LAG_CLOSE_THRESHOLD, MAX_HOLD_S, FALLBACK_TAU_S,
    KELLY_TIERS,
)

# ---------------------------------------------------------------------------
# Feature engineering -- matches live model schema (short-lag set)
# ---------------------------------------------------------------------------

# Must match LOOKBACK_S in config.py (excluding 0, which is x_0)
LAGS = [5, 10, 15, 20, 25, 30]   # seconds


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features within each window (grouped by window_ticker so that
    lag shifts don't bleed across window boundaries).
    Matches the live FeatureVec schema in betbot/kalshi/features.py.
    """
    groups = []
    for ticker, g in df.groupby("window_ticker", sort=False):
        g = g.copy().sort_values("ts_ns").reset_index(drop=True)
        K = g["floor_strike"].iloc[0]
        if K <= 0:
            continue

        # Spot log-ratios vs strike
        mp = g["btc_microprice"]
        g["x_0"] = np.log(mp / K)
        for lag in LAGS:
            # shift by `lag` rows (1 row = 1 second at 1Hz)
            shifted = mp.shift(lag)
            g[f"x_{lag}"] = np.log(shifted.fillna(mp) / K)

        # Time features
        g["tau_s"]        = g["tau_s"].clip(lower=0)
        g["inv_sqrt_tau"] = 1.0 / np.sqrt(g["tau_s"] + 1.0)

        # Kalshi momentum (5s, 10s, 30s) and spread
        ym = g["yes_mid"]
        g["kalshi_momentum_5s"]  = ym - ym.shift(5).fillna(ym)
        g["kalshi_momentum_10s"] = ym - ym.shift(10).fillna(ym)
        g["kalshi_momentum_30s"] = ym - ym.shift(30).fillna(ym)
        g["kalshi_spread"]       = g["yes_ask"] - g["yes_bid"]

        # Target: logit(yes_mid) -- apply same training filter as live model
        ym_filt = ym.where(
            (ym >= TRAIN_YES_MID_MIN) & (ym <= TRAIN_YES_MID_MAX), other=np.nan
        )
        ym_c = ym_filt.clip(1e-6, 1 - 1e-6)
        g["y"] = np.log(ym_c / (1 - ym_c))

        groups.append(g)

    if not groups:
        return pd.DataFrame()
    return pd.concat(groups, ignore_index=True)


FEATURE_COLS = (
    ["x_0"] + [f"x_{l}" for l in LAGS]
    + ["tau_s", "inv_sqrt_tau",
       "kalshi_spread", "kalshi_momentum_5s", "kalshi_momentum_10s", "kalshi_momentum_30s"]
)


def settled_vec(row: pd.Series) -> np.ndarray:
    """Replace all lag x slots with x_0 to compute q_settled (matches FeatureVec.settled_array)."""
    vec = row[FEATURE_COLS].values.copy().astype(float)
    x0_idx = FEATURE_COLS.index("x_0")
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith("x_") and col != "x_0":
            vec[i] = vec[x0_idx]
    return vec


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


def simulate(df_test: pd.DataFrame, scaler: StandardScaler,
             coefs: np.ndarray, intercept: float,
             entry_threshold: float, exit_threshold: float,
             max_hold_s: int, min_tau_s: int) -> list[dict]:
    """
    Walk the test rows in time order.
    Enter when net edge (after fees) > entry_threshold AND yes_mid in decision range.
    Exit when: net edge < exit_threshold, OR held > max_hold_s, OR tau < min_tau_s.
    Fees: THETA_FEE_TAKER * p * (1 - p) per leg (taker both sides).

    Cross-window / cross-gap protection:
    - Any open position is force-closed if the window_ticker changes.
    - Any open position is force-closed if there is a timestamp gap > 5s (bot restart).
    """
    trades = []
    pos    = None   # dict with entry info when open

    rows = df_test.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
    prev_ts_ns = None
    prev_ticker = None

    for _, row in rows.iterrows():
        cur_ticker = row.get("window_ticker", "")
        cur_ts_ns  = row["ts_ns"]

        # Force-close any open position on window change or timestamp gap > 5s
        if pos is not None:
            ticker_changed = (cur_ticker != pos["window_ticker"])
            gap_s = (cur_ts_ns - (prev_ts_ns or cur_ts_ns)) / 1e9
            if ticker_changed or gap_s > 5.0:
                # Close at last known bid (use entry_ask as proxy -- worst case)
                trades.append({
                    "entry_ts":    pos["entry_ts_ns"] / 1e9,
                    "exit_ts":     (prev_ts_ns or cur_ts_ns) / 1e9,
                    "entry_p":     pos["entry_ask"],
                    "exit_p":      pos["entry_ask"],   # no movement captured = 0 gross P&L
                    "contracts":   pos["contracts"],
                    "size_usd":    pos["size_usd"],
                    "pnl":         -(pos["fee_entry"]) * pos["size_usd"],  # lose entry fee only
                    "gross_pnl":   0.0,
                    "hold_s":      (prev_ts_ns - pos["entry_ts_ns"]) / 1e9 if prev_ts_ns else 0,
                    "exit_edge":   0.0,
                    "exit_reason": "window_gap",
                })
                pos = None

        prev_ts_ns  = cur_ts_ns
        prev_ticker = cur_ticker
        tau     = row["tau_s"]
        yes_ask = float(row["yes_ask"])
        yes_bid = float(row["yes_bid"])
        yes_mid = float(row["yes_mid"])

        # Compute q_settled
        sv    = settled_vec(row).reshape(1, -1)
        sv_sc = scaler.transform(sv)
        logit = float(np.dot(coefs, sv_sc[0]) + intercept)
        q_set = _sigmoid(logit)

        # Net edge after entry fee
        fee_entry = THETA_FEE_TAKER * yes_ask * (1.0 - yes_ask)
        edge_net  = q_set - yes_ask - fee_entry

        # --- Exit logic ---
        if pos is not None:
            hold_s = (row["ts_ns"] - pos["entry_ts_ns"]) / 1e9
            # Re-evaluate edge at current price (same entry fee subtracted to stay consistent)
            current_fee = THETA_FEE_TAKER * yes_ask * (1.0 - yes_ask)
            current_edge = q_set - yes_ask - current_fee
            should_exit = (
                current_edge < exit_threshold
                or hold_s >= max_hold_s
                or tau < min_tau_s
            )
            if should_exit:
                exit_p    = yes_bid
                fee_exit  = THETA_FEE_TAKER * exit_p * (1.0 - exit_p)
                gross_pnl = (exit_p - pos["entry_ask"]) * pos["contracts"]
                fee_total = (pos["fee_entry"] + fee_exit) * pos["size_usd"]
                pnl       = gross_pnl - fee_total
                trades.append({
                    "entry_ts":    pos["entry_ts_ns"] / 1e9,
                    "exit_ts":     row["ts_ns"] / 1e9,
                    "entry_p":     pos["entry_ask"],
                    "exit_p":      exit_p,
                    "contracts":   pos["contracts"],
                    "size_usd":    pos["size_usd"],
                    "pnl":         pnl,
                    "gross_pnl":   gross_pnl,
                    "hold_s":      hold_s,
                    "exit_edge":   current_edge,
                    "exit_reason": (
                        "lag_closed" if current_edge < exit_threshold
                        else "max_hold" if hold_s >= max_hold_s
                        else "min_tau"
                    ),
                })
                pos = None

        # --- Entry logic ---
        if (pos is None
                and edge_net > entry_threshold
                and tau > min_tau_s
                and DECISION_YES_MID_MIN <= yes_mid <= DECISION_YES_MID_MAX):
            size_usd  = 1.0   # $1 per trade for clean per-dollar P&L
            contracts = size_usd / yes_ask if yes_ask > 0 else 0
            pos = {
                "entry_ts_ns":   row["ts_ns"],
                "entry_ask":     yes_ask,
                "entry_edge":    edge_net,
                "contracts":     contracts,
                "size_usd":      size_usd,
                "fee_entry":     fee_entry,
                "window_ticker": row.get("window_ticker", ""),
            }

    return trades


# ---------------------------------------------------------------------------
# Run one backtest configuration
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, train_frac: float,
                 entry_threshold: float, exit_threshold: float,
                 max_hold_s: int, min_tau_s: int,
                 ridge_alpha: float | None = None) -> dict:

    df = df.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)
    if len(df) < 200:
        return {"error": "not enough data"}

    split = int(len(df) * train_frac)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    X_train = train[FEATURE_COLS].values
    y_train = train["y"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    n_splits = min(5, max(2, len(X_train) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    if ridge_alpha is not None:
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=ridge_alpha, fit_intercept=True)
    else:
        mdl = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
                      cv=tscv, scoring="r2", fit_intercept=True)
    mdl.fit(X_sc, y_train)

    r2_train  = float(mdl.score(X_sc, y_train))
    r2_cv     = float(getattr(mdl, "best_score_", r2_train))
    X_test_sc = scaler.transform(test[FEATURE_COLS].values)
    r2_test   = float(mdl.score(X_test_sc, test["y"].values))
    alpha_used = float(getattr(mdl, "alpha_", ridge_alpha or 0))

    # Print the same [Refit] line the live bot prints
    coefs = {name: float(mdl.coef_[i]) for i, name in enumerate(FEATURE_COLS)}
    lag_names = [c for c in FEATURE_COLS if c.startswith("x_") and c != "x_0"]
    lag_secs  = [int(c.split("_")[1]) for c in lag_names]
    lag_betas = [abs(coefs[n]) for n in lag_names]
    total_b   = sum(lag_betas)
    est_lag   = (sum(b * s for b, s in zip(lag_betas, lag_secs)) / total_b
                 if total_b > 0 else 0.0)
    coef_str  = "  ".join(f"{n}={coefs[n]:+.3f}" for n in FEATURE_COLS)
    print(f"\n[Refit]  n={len(X_train)}  alpha={alpha_used:.4f}  "
          f"R2_in={r2_train:.3f}  R2_cv={r2_cv:.3f}  R2_hld={r2_test:.3f}  "
          f"lag={est_lag:.0f}s")
    print(f"         {coef_str}")

    trades = simulate(
        test, scaler, mdl.coef_, mdl.intercept_,
        entry_threshold, exit_threshold, max_hold_s, min_tau_s
    )

    pnl_vals   = [t["pnl"]       for t in trades]
    gross_vals = [t["gross_pnl"] for t in trades]
    total_pnl   = sum(pnl_vals)
    total_gross = sum(gross_vals)
    n_win       = sum(1 for p in pnl_vals if p > 0)
    exit_counts = {}
    for t in trades:
        exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1

    return {
        "n_train":         len(train),
        "n_test":          len(test),
        "r2_train":        r2_train,
        "r2_cv":           r2_cv,
        "r2_test":         r2_test,
        "alpha":           alpha_used,
        "n_trades":        len(trades),
        "n_win":           n_win,
        "win_rate":        n_win / len(trades) if trades else 0.0,
        "total_pnl":       total_pnl,
        "total_gross_pnl": total_gross,
        "avg_pnl":         np.mean(pnl_vals) if pnl_vals else 0.0,
        "pnl_std":         np.std(pnl_vals)  if pnl_vals else 0.0,
        "avg_hold_s":      np.mean([t["hold_s"] for t in trades]) if trades else 0.0,
        "exit_counts":     exit_counts,
        "trades":          trades,
        "entry_threshold": entry_threshold,
        "exit_threshold":  exit_threshold,
        "max_hold_s":      max_hold_s,
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def sweep(df: pd.DataFrame, train_frac: float) -> pd.DataFrame:
    entry_thresholds = [0.010, 0.015, 0.020, 0.030, 0.040, 0.060]
    exit_thresholds  = [0.002, 0.005, 0.010, 0.015]
    max_holds        = [20, 30, 60, 90]

    results = []
    total = len(entry_thresholds) * len(exit_thresholds) * len(max_holds)
    done  = 0

    for et in entry_thresholds:
        for xt in exit_thresholds:
            if xt >= et:
                continue
            for mh in max_holds:
                r = run_backtest(df, train_frac, et, xt, mh, min_tau_s=60)
                done += 1
                if "error" not in r:
                    results.append({
                        "entry":    et,
                        "exit":     xt,
                        "hold_s":   mh,
                        "n_trades": r["n_trades"],
                        "win_rate": r["win_rate"],
                        "total_pnl":r["total_pnl"],
                        "avg_pnl":  r["avg_pnl"],
                        "r2_test":  r["r2_test"],
                    })
                print(f"  {done}/{total} entry={et:.3f} exit={xt:.3f} hold={mh}s "
                      f"-> n={r.get('n_trades',0)} pnl={r.get('total_pnl',0):+.3f}",
                      flush=True)

    return pd.DataFrame(results).sort_values("total_pnl", ascending=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(df: pd.DataFrame, result: dict, save_path: str | None = None) -> None:
    if not HAS_MPL:
        return

    import datetime

    trades = result["trades"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"Backtest Results  |  entry>{result['entry_threshold']:.3f}  "
        f"exit<{result['exit_threshold']:.3f}  hold<={result['max_hold_s']}s  "
        f"R2_test={result['r2_test']:.3f}",
        fontsize=12, fontweight="bold"
    )

    # -- Panel 1: YES price + entries/exits --
    ax = axes[0]
    ts = [datetime.datetime.fromtimestamp(r / 1e9) for r in df["ts_ns"]]
    ax.fill_between(ts, df["yes_bid"], df["yes_ask"], alpha=0.2, color="steelblue")
    ax.plot(ts, df["yes_mid"], color="steelblue", lw=1, label="YES mid")

    if trades:
        entry_x = [datetime.datetime.fromtimestamp(t["entry_ts"]) for t in trades]
        entry_y = [t["entry_p"] for t in trades]
        exit_x  = [datetime.datetime.fromtimestamp(t["exit_ts"]) for t in trades]
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

    ax.axvline(datetime.datetime.fromtimestamp(df.iloc[int(len(df) * result.get("train_frac_used", 0.6))]["ts_ns"] / 1e9),
               color="gray", lw=1.5, linestyle="--", label="train/test split")
    ax.set_ylabel("YES price")
    ax.set_title("YES Market Price -- Entry/Exit Points")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)

    # -- Panel 2: Cumulative P&L --
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
    ax.set_title(f"Cumulative P&L -- {len(trades)} trades, "
                 f"win rate {result['win_rate']*100:.0f}%")
    ax.grid(alpha=0.3)

    # -- Panel 3: Edge at entry vs P&L scatter --
    ax = axes[2]
    if trades:
        edges = [t["exit_edge"] for t in trades]
        pnls  = [t["pnl"] for t in trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax.scatter(edges, pnls, c=colors, alpha=0.6, s=40)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Edge at exit (q_settled - yes_ask)")
        ax.set_ylabel("P&L per trade ($)")
        ax.set_title("Exit edge vs Trade P&L")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest on tick data")
    parser.add_argument("--run",         default=None,
                        help="Path to a data/<run> folder (popup if omitted)")
    parser.add_argument("--asset",       default="BTC",
                        help="Which asset's ticks file to use (default: BTC)")
    parser.add_argument("--train-frac",  type=float, default=0.6,
                        help="Fraction of ticks used for training (default 0.60)")
    min_entry = KELLY_TIERS[-1][0]   # lowest tier floor from config
    parser.add_argument("--entry",       type=float, default=min_entry,
                        help=f"Min net edge to enter (default {min_entry} from config)")
    parser.add_argument("--exit",        type=float, default=LAG_CLOSE_THRESHOLD,
                        help=f"Exit when edge drops below this (default {LAG_CLOSE_THRESHOLD} from config)")
    parser.add_argument("--hold-s",      type=int,   default=MAX_HOLD_S,
                        help=f"Force exit after this many seconds (default {MAX_HOLD_S} from config)")
    parser.add_argument("--min-tau",     type=int,   default=FALLBACK_TAU_S,
                        help=f"Don't enter if window closes in < this many s (default {FALLBACK_TAU_S} from config)")
    parser.add_argument("--sweep",       action="store_true",
                        help="Grid sweep over entry/exit/hold params")
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--save",        default=None, help="Save chart PNG to this path")
    args = parser.parse_args()

    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to backtest")
    asset     = args.asset.upper()
    tick_path = run_dir / f"ticks_{asset}.csv"
    if not tick_path.exists():
        legacy = run_dir / "ticks.csv"
        if legacy.exists():
            tick_path = legacy
    if not tick_path.exists():
        sys.exit(f"ERROR: {tick_path} not found. Run the bot first:\n"
                 "  python scripts/run/run_kalshi_bot.py")

    print(f"Loading {tick_path}...")
    raw = pd.read_csv(tick_path)
    print(f"  Raw rows:   {len(raw)}")
    print(f"  Time span:  {(raw['ts_ns'].max() - raw['ts_ns'].min()) / 1e9 / 60:.1f} minutes")
    print(f"  Windows:    {raw['window_ticker'].nunique()}")

    df = build_features(raw)
    df = df.dropna(subset=FEATURE_COLS)
    print(f"  Usable rows (features complete): {len(df)}")

    if len(df) < 200:
        sys.exit("Need at least 200 ticks with complete features. "
                 "Run the bot for at least 5 more minutes.")

    if args.sweep:
        print(f"\nParameter sweep (train_frac={args.train_frac})...")
        sweep_df = sweep(df, args.train_frac)
        print("\nTop 10 parameter combinations by total P&L:")
        print(sweep_df.head(10).to_string(index=False))
        if not args.no_plot and HAS_MPL:
            fig, ax = plt.subplots(figsize=(10, 6))
            for hold in sweep_df["hold_s"].unique():
                sub = sweep_df[sweep_df["hold_s"] == hold]
                ax.scatter(sub["entry"], sub["total_pnl"], label=f"hold={hold}s", s=60)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel("Entry threshold")
            ax.set_ylabel("Total P&L ($)")
            ax.set_title("Sweep: Entry Threshold vs P&L")
            ax.legend()
            ax.grid(alpha=0.3)
            if args.save:
                plt.savefig(args.save, dpi=150, bbox_inches="tight")
            else:
                plt.show()
        return

    # Single run
    print(f"\nRunning backtest:")
    print(f"  train_frac={args.train_frac}  entry>{args.entry}  "
          f"exit<{args.exit}  hold<={args.hold_s}s  min_tau={args.min_tau}s")

    result = run_backtest(df, args.train_frac, args.entry, args.exit,
                          args.hold_s, args.min_tau)
    result["train_frac_used"] = args.train_frac

    if "error" in result:
        sys.exit(f"Backtest error: {result['error']}")

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS  (lag schema: [5,10,15,20,25,30]s)")
    print(f"{'='*60}")
    print(f"  Train rows:      {result['n_train']}  "
          f"Test rows: {result['n_test']}")
    print(f"  Ridge alpha:     {result['alpha']:.4f}")
    print(f"  R2 train:        {result['r2_train']:.4f}  <-- in-sample")
    print(f"  R2 hld (test):   {result['r2_test']:.4f}  "
          f"<-- held-out last {100-int(args.train_frac*100)}%  (this is your live R2_hld)")
    print(f"  --- Strategy ---")
    print(f"  Trades:          {result['n_trades']}")
    print(f"  Win rate:        {result['win_rate']*100:.0f}%  "
          f"({result['n_win']}/{result['n_trades']})")
    print(f"  Gross P&L:       ${result['total_gross_pnl']:+.4f}  (before fees)")
    print(f"  Net P&L:         ${result['total_pnl']:+.4f}  (after 7% taker fees both legs)")
    print(f"  Avg net/trade:   ${result['avg_pnl']:+.4f}  (std ${result['pnl_std']:.4f})")
    print(f"  Avg hold time:   {result['avg_hold_s']:.0f}s")
    print(f"  Exit breakdown:  {result['exit_counts']}")
    print(f"{'='*60}")

    if result["n_trades"] > 0:
        print(f"\n  Individual trades:")
        for t in result["trades"]:
            import datetime
            entry_time = datetime.datetime.fromtimestamp(t["entry_ts"]).strftime("%H:%M:%S")
            print(f"    {entry_time}  buy@{t['entry_p']:.3f} -> "
                  f"sell@{t['exit_p']:.3f}  "
                  f"pnl=${t['pnl']:+.4f}  "
                  f"hold={t['hold_s']:.0f}s  [{t['exit_reason']}]")

    if not args.no_plot:
        plot_results(df, result, save_path=args.save)


if __name__ == "__main__":
    main()
