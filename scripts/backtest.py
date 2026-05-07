"""
backtest.py -- Walk-forward backtest on collected tick data.

Loads logs/ticks.csv, builds lag features, fits ridge regression,
simulates trading on the held-out period, and reports P&L.

No fee deduction (user requested simple P&L first).

Usage:
  python scripts/backtest.py                      # defaults
  python scripts/backtest.py --entry 0.025        # entry threshold
  python scripts/backtest.py --hold-s 45          # max hold seconds
  python scripts/backtest.py --train-frac 0.65    # 65% train / 35% test
  python scripts/backtest.py --no-plot            # text only
  python scripts/backtest.py --sweep              # grid sweep of params
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

LAGS = [15, 30, 60, 90, 120]   # seconds


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features within each window (grouped by window_ticker so that
    lag shifts don't bleed across window boundaries).
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
            # shift by `lag` rows (1 row = 1 second)
            shifted = mp.shift(lag)
            g[f"x_{lag}"] = np.log(shifted.fillna(mp) / K)

        # Time features
        g["tau_s"]        = g["tau_s"].clip(lower=0)
        g["inv_sqrt_tau"] = 1.0 / np.sqrt(g["tau_s"] + 1.0)

        # Spot momentum (source-agnostic — same definition for Coinbase or Binance)
        g["spot_mom_30"] = np.log(mp / mp.shift(30).fillna(mp))
        g["spot_mom_60"] = np.log(mp / mp.shift(60).fillna(mp))

        # Kalshi momentum and spread
        ym = g["yes_mid"]
        g["kalshi_mom_30"] = ym - ym.shift(30).fillna(ym)
        g["kalshi_spread"] = g["yes_ask"] - g["yes_bid"]

        # Target: logit(yes_mid)
        ym_c = ym.clip(1e-4, 1 - 1e-4)
        g["y"] = np.log(ym_c / (1 - ym_c))

        # settled_x_0: substitute current spot into all lag slots
        # (model applied to this vector → q_settled)
        g["settled_flag"] = True   # marker used below

        groups.append(g)

    if not groups:
        return pd.DataFrame()
    return pd.concat(groups, ignore_index=True)


FEATURE_COLS = (
    ["x_0"] + [f"x_{l}" for l in LAGS]
    + ["tau_s", "inv_sqrt_tau", "spot_mom_30", "spot_mom_60",
       "kalshi_spread", "kalshi_mom_30"]
)


def settled_vec(row: pd.Series) -> np.ndarray:
    """Replace all lag x slots with x_0 to compute q_settled."""
    vec = row[FEATURE_COLS].values.copy()
    x0_idx = FEATURE_COLS.index("x_0")
    for i, col in enumerate(FEATURE_COLS):
        if col.startswith("x_") and col != "x_0":
            vec[i] = vec[x0_idx]
    return vec


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(df_test: pd.DataFrame, scaler: StandardScaler,
             coefs: np.ndarray, intercept: float,
             entry_threshold: float, exit_threshold: float,
             max_hold_s: int, min_tau_s: int) -> list[dict]:
    """
    Walk the test rows in time order.
    Enter when q_settled - yes_ask > entry_threshold.
    Exit when: q_settled - yes_ask < exit_threshold, OR held > max_hold_s, OR tau < min_tau_s.
    """
    trades = []
    pos    = None   # dict with entry info when open

    rows = df_test.dropna(subset=FEATURE_COLS + ["y"]).reset_index(drop=True)

    for i, row in rows.iterrows():
        tau = row["tau_s"]

        # Compute q_settled
        sv    = settled_vec(row).reshape(1, -1)
        sv_sc = scaler.transform(sv)
        logit = float(np.dot(coefs, sv_sc[0]) + intercept)
        q_set = 1.0 / (1.0 + np.exp(-logit)) if logit >= 0 else (
                np.exp(logit) / (1.0 + np.exp(logit)))

        yes_ask = row["yes_ask"]
        yes_bid = row["yes_bid"]
        edge    = q_set - yes_ask

        # --- Exit logic ---
        if pos is not None:
            hold_s = (row["ts_ns"] - pos["entry_ts_ns"]) / 1e9
            exit_edge = q_set - pos["entry_ask"]
            should_exit = (
                exit_edge < exit_threshold
                or hold_s >= max_hold_s
                or tau < min_tau_s
            )
            if should_exit:
                exit_p = yes_bid
                pnl    = (exit_p - pos["entry_ask"]) * pos["contracts"]
                trades.append({
                    "entry_ts":  pos["entry_ts_ns"] / 1e9,
                    "exit_ts":   row["ts_ns"] / 1e9,
                    "entry_p":   pos["entry_ask"],
                    "exit_p":    exit_p,
                    "contracts": pos["contracts"],
                    "pnl":       pnl,
                    "hold_s":    hold_s,
                    "exit_edge": exit_edge,
                    "exit_reason": (
                        "lag_closed" if exit_edge < exit_threshold
                        else "max_hold"   if hold_s >= max_hold_s
                        else "min_tau"
                    ),
                })
                pos = None

        # --- Entry logic ---
        if pos is None and edge > entry_threshold and tau > min_tau_s:
            contracts = 1.0 / yes_ask if yes_ask > 0 else 0   # ~$1 per trade
            pos = {
                "entry_ts_ns": row["ts_ns"],
                "entry_ask":   yes_ask,
                "entry_edge":  edge,
                "contracts":   contracts,
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

    if ridge_alpha is not None:
        from sklearn.linear_model import Ridge
        mdl = Ridge(alpha=ridge_alpha, fit_intercept=True)
    else:
        mdl = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
                      cv=5, scoring="r2", fit_intercept=True)
    mdl.fit(X_sc, y_train)

    r2_train = float(mdl.score(X_sc, y_train))
    X_test_sc = scaler.transform(test[FEATURE_COLS].values)
    r2_test   = float(mdl.score(X_test_sc, test["y"].values))
    alpha_used = float(getattr(mdl, "alpha_", ridge_alpha or 0))

    trades = simulate(
        test, scaler, mdl.coef_, mdl.intercept_,
        entry_threshold, exit_threshold, max_hold_s, min_tau_s
    )

    pnl_vals = [t["pnl"] for t in trades]
    total_pnl   = sum(pnl_vals)
    n_win       = sum(1 for p in pnl_vals if p > 0)
    exit_counts = {}
    for t in trades:
        exit_counts[t["exit_reason"]] = exit_counts.get(t["exit_reason"], 0) + 1

    return {
        "n_train":         len(train),
        "n_test":          len(test),
        "r2_train":        r2_train,
        "r2_test":         r2_test,
        "alpha":           alpha_used,
        "n_trades":        len(trades),
        "n_win":           n_win,
        "win_rate":        n_win / len(trades) if trades else 0.0,
        "total_pnl":       total_pnl,
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

    # ── Panel 1: YES price + entries/exits ──────────────────────────────────
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
    ax.set_title("YES Market Price — Entry/Exit Points")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)

    # ── Panel 2: Cumulative P&L ─────────────────────────────────────────────
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

    # ── Panel 3: Edge at entry vs P&L scatter ───────────────────────────────
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
    parser.add_argument("--ticks",       default="logs/ticks.csv")
    parser.add_argument("--train-frac",  type=float, default=0.6,
                        help="Fraction of ticks used for training (default 0.60)")
    parser.add_argument("--entry",       type=float, default=0.020,
                        help="Min q_settled - yes_ask to enter (default 0.02)")
    parser.add_argument("--exit",        type=float, default=0.005,
                        help="Exit when q_settled - yes_ask drops below this (default 0.005)")
    parser.add_argument("--hold-s",      type=int,   default=60,
                        help="Force exit after this many seconds (default 60)")
    parser.add_argument("--min-tau",     type=int,   default=60,
                        help="Don't enter if window closes in < this many s (default 60)")
    parser.add_argument("--sweep",       action="store_true",
                        help="Grid sweep over entry/exit/hold params")
    parser.add_argument("--no-plot",     action="store_true")
    parser.add_argument("--save",        default=None, help="Save chart PNG to this path")
    args = parser.parse_args()

    tick_path = Path(args.ticks)
    if not tick_path.exists():
        sys.exit(f"ERROR: {tick_path} not found. Run the bot first:\n"
                 "  python scripts/run_kalshi_bot.py")

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

    print(f"\n{'='*55}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Train rows:    {result['n_train']}  "
          f"Test rows: {result['n_test']}")
    print(f"  Ridge alpha:   {result['alpha']:.4f}")
    print(f"  R2 train:      {result['r2_train']:.4f}")
    print(f"  R2 test:       {result['r2_test']:.4f}")
    print(f"  Trades:        {result['n_trades']}")
    print(f"  Win rate:      {result['win_rate']*100:.0f}%  "
          f"({result['n_win']}/{result['n_trades']})")
    print(f"  Total P&L:     ${result['total_pnl']:+.4f}")
    print(f"  Avg P&L/trade: ${result['avg_pnl']:+.4f}  "
          f"(std ${result['pnl_std']:.4f})")
    print(f"  Avg hold time: {result['avg_hold_s']:.0f}s")
    print(f"  Exit breakdown: {result['exit_counts']}")
    print(f"{'='*55}")

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
