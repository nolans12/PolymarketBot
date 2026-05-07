"""
validate_projection_features.py -- Head-to-head comparison: baseline features
vs. baseline + forward-projection features (CLAUDE.md section 6.6).

Loads logs/ticks.csv, builds two feature sets, fits ridge on each train slice,
simulates trading on each test slice, and prints a side-by-side comparison.

Projection features (added on top of the baseline 12):
    x_proj_30  = x_0 + spot_mom_30
    x_proj_60  = x_0 + spot_mom_60
    x_proj_120 = x_0 + 2 * spot_mom_60

CRITICAL: in settled_vec, the projection slots are NOT substituted with x_0.
Only the LAG slots (x_15..x_120) are. The projection slots keep their live
forward-drift values, which is the entire point -- q_settled then represents
"where Kalshi lands once the lag closes AND Coinbase keeps drifting at its
recent rate" rather than "where Kalshi lands assuming Coinbase stops moving."

This script mirrors the LIVE bot's gating:
  - TRAIN slice keeps only ticks with yes_mid in [TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX]
    (the band the regression is supposed to learn; outside this, logit explodes
     and pulls the fit toward boundary noise).
  - ENTRY in the simulator only fires when yes_mid is in
    [DECISION_YES_MID_MIN, DECISION_YES_MID_MAX] (matches the in-flight
     entry-gate the live bot uses). Open positions still run exit logic
     regardless of yes_mid -- same as live.

Output is ASCII-only so it survives the Windows console (cp1252).

Usage:
    python scripts/validate_projection_features.py
    python scripts/validate_projection_features.py --train-frac 0.7
    python scripts/validate_projection_features.py --no-filter   # off, for reference
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_run import pick_run_folder

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# Pull the live-bot gating constants from the shared config so they can't
# drift between live and backtest behaviour.
from betbot.kalshi.config import (
    TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    THETA_FEE_TAKER as CONFIG_THETA_FEE,
)

# Validator-local fee theta. Defaults to 0.07 (= CLAUDE.md section 5.4 spec
# AND the realistic Kalshi curve in the user's chart: peak 1.75% per leg at
# p=0.5). The live config currently has THETA_FEE=0.03 as a working calibration;
# CLI --theta-fee can override.
DEFAULT_THETA_FEE = 0.07


def _taker_fee_per_dollar(p: float, theta: float) -> float:
    """Kalshi taker fee per $1 notional, per leg.

    Matches the live model in betbot/kalshi/scheduler.py:
        fee_per_dollar(p) = theta * p * (1 - p)

    Curve peaks at theta/4 at p=0.5; goes to 0 at the extremes. Fully
    consistent with the Kalshi fee distribution chart (theta=0.07 there)."""
    if p <= 0 or p >= 1:
        return 0.0
    return theta * p * (1.0 - p)

# Match the feature-engineering windows used in scripts/backtest.py.
LAGS = [15, 30, 60, 90, 120]

LAG_COL_SET = {f"x_{l}" for l in LAGS}   # slots replaced by x_0 in settled_vec

BASELINE_COLS = (
    ["x_0"] + [f"x_{l}" for l in LAGS]
    + ["tau_s", "inv_sqrt_tau", "spot_mom_30", "spot_mom_60",
       "kalshi_spread", "kalshi_mom_30"]
)

PROJ_COLS = BASELINE_COLS + ["x_proj_30", "x_proj_60", "x_proj_120"]


# ---------------------------------------------------------------------------
# Feature engineering (mirrors scripts/backtest.py + the projection extras)
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """
    Compute features within each window so lag shifts don't bleed across
    boundaries. Returns ALL columns (baseline + projection); the projection
    columns are then optionally dropped at fit time depending on the variant
    being tested.

    `lags`: which lookback horizons to materialise as x_<lag> columns.
            Defaults to the module-global LAGS = [15, 30, 60, 90, 120].
    """
    if lags is None:
        lags = LAGS
    groups = []
    for _, g in df.groupby("window_ticker", sort=False):
        g = g.copy().sort_values("ts_ns").reset_index(drop=True)
        K = g["floor_strike"].iloc[0]
        if K <= 0:
            continue

        mp = g["btc_microprice"]
        g["x_0"] = np.log(mp / K)
        for lag in lags:
            shifted = mp.shift(lag)
            g[f"x_{lag}"] = np.log(shifted.fillna(mp) / K)

        g["tau_s"]        = g["tau_s"].clip(lower=0)
        g["inv_sqrt_tau"] = 1.0 / np.sqrt(g["tau_s"] + 1.0)

        g["spot_mom_30"] = np.log(mp / mp.shift(30).fillna(mp))
        g["spot_mom_60"] = np.log(mp / mp.shift(60).fillna(mp))

        # Forward-projected log-K ratios -- assume recent drift continues.
        g["x_proj_30"]  = g["x_0"] + g["spot_mom_30"]
        g["x_proj_60"]  = g["x_0"] + g["spot_mom_60"]
        g["x_proj_120"] = g["x_0"] + 2.0 * g["spot_mom_60"]

        ym = g["yes_mid"]
        g["kalshi_mom_30"] = ym - ym.shift(30).fillna(ym)
        g["kalshi_spread"] = g["yes_ask"] - g["yes_bid"]

        ym_c = ym.clip(1e-4, 1 - 1e-4)
        g["y"] = np.log(ym_c / (1 - ym_c))

        groups.append(g)

    if not groups:
        return pd.DataFrame()
    return pd.concat(groups, ignore_index=True)


def settled_vec(row: pd.Series, feature_cols: list[str],
                lag_set: set[str] = None) -> np.ndarray:
    """
    Build the q_settled feature vector: replace ONLY the lag slots with x_0.
    Projection slots (x_proj_*) keep their live values.

    `lag_set`: which feature names count as "lag slots". Defaults to the
               module-global LAG_COL_SET (= the 5 baseline lags).
    """
    if lag_set is None:
        lag_set = LAG_COL_SET
    vec = row[feature_cols].values.copy()
    x0_idx = feature_cols.index("x_0")
    for i, col in enumerate(feature_cols):
        if col in lag_set:
            vec[i] = vec[x0_idx]
    return vec


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

def simulate(df_test: pd.DataFrame, scaler: StandardScaler, coefs: np.ndarray,
             intercept: float, feature_cols: list[str],
             entry_threshold: float, exit_threshold: float,
             max_hold_s: int, min_tau_s: int,
             use_entry_gate: bool, theta_fee: float,
             entry_mode: str = "taker",
             lag_set: set[str] = None) -> list[dict]:
    """
    entry_mode:
      "taker" -- enter at yes_ask, pay theta_fee*p*(1-p) entry fee
      "maker" -- enter at yes_bid, pay 0 entry fee (UPPER BOUND: assumes
                 every limit fills instantly at the bid; real fill rate
                 is < 100% and adverse selection is unmodelled)
    Exit is always taker (sell at yes_bid, pay exit fee).
    """
    if entry_mode not in ("taker", "maker"):
        raise ValueError(f"entry_mode must be 'taker' or 'maker', got {entry_mode!r}")
    """
    Walk the test rows in time order. Mirrors the live bot's ordering:
      - Always run exit logic first (even outside DECISION_YES_MID range).
      - Only allow entry when yes_mid is inside [DECISION_YES_MID_MIN,
        DECISION_YES_MID_MAX] AND tau > min_tau_s AND edge > entry_threshold.
    """
    trades = []
    pos    = None

    rows = df_test.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)

    for _, row in rows.iterrows():
        tau = row["tau_s"]

        sv    = settled_vec(row, feature_cols, lag_set).reshape(1, -1)
        sv_sc = scaler.transform(sv)
        logit = float(np.dot(coefs, sv_sc[0]) + intercept)
        q_set = (1.0 / (1.0 + np.exp(-logit))) if logit >= 0 else (
                np.exp(logit) / (1.0 + np.exp(logit)))

        yes_ask = row["yes_ask"]
        yes_bid = row["yes_bid"]
        yes_mid = row["yes_mid"]
        edge    = q_set - yes_ask

        # ---- Exit (runs regardless of entry gate) -------------------------
        if pos is not None:
            hold_s    = (row["ts_ns"] - pos["entry_ts_ns"]) / 1e9
            exit_edge = q_set - pos["entry_p"]
            should_exit = (
                exit_edge < exit_threshold
                or hold_s >= max_hold_s
                or tau < min_tau_s
            )
            if should_exit:
                exit_p    = yes_bid
                contracts = pos["contracts"]
                # $1 notional per trade -> bet_usd = 1.0
                # Entry fee depends on entry mode (0 if maker).
                # Exit is always taker.
                entry_fee = pos["entry_fee"]
                exit_fee  = _taker_fee_per_dollar(exit_p, theta_fee) * 1.0
                gross_pnl = (exit_p - pos["entry_p"]) * contracts
                net_pnl   = gross_pnl - entry_fee - exit_fee
                trades.append({
                    "entry_p":   pos["entry_p"],
                    "exit_p":    exit_p,
                    "pnl":       net_pnl,
                    "gross_pnl": gross_pnl,
                    "fees":      entry_fee + exit_fee,
                    "hold_s":    hold_s,
                    "exit_edge": exit_edge,
                })
                pos = None

        # ---- Entry (gated by yes_mid range when use_entry_gate=True) -----
        if pos is None and edge > entry_threshold and tau > min_tau_s:
            if use_entry_gate and (yes_mid < DECISION_YES_MID_MIN
                                    or yes_mid > DECISION_YES_MID_MAX):
                continue
            if entry_mode == "maker":
                entry_p   = yes_bid           # our limit posted at bid, fills at bid
                entry_fee = 0.0               # maker fee = 0 on Kalshi
            else:  # taker
                entry_p   = yes_ask
                entry_fee = _taker_fee_per_dollar(yes_ask, theta_fee) * 1.0
            if entry_p <= 0:
                continue
            contracts = 1.0 / entry_p
            pos = {
                "entry_ts_ns": row["ts_ns"],
                "entry_p":     entry_p,
                "entry_fee":   entry_fee,
                "entry_edge":  edge,
                "contracts":   contracts,
            }

    return trades


# ---------------------------------------------------------------------------
# Run one backtest configuration
# ---------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, feature_cols: list[str], train_frac: float,
                 entry_threshold: float, exit_threshold: float,
                 max_hold_s: int, min_tau_s: int,
                 use_train_filter: bool, use_entry_gate: bool,
                 theta_fee: float, entry_mode: str = "taker",
                 lag_set: set[str] = None) -> dict:

    df_full = df.dropna(subset=feature_cols + ["y"]).reset_index(drop=True)
    if len(df_full) < 200:
        return {"error": "not enough data"}

    split = int(len(df_full) * train_frac)
    train_raw = df_full.iloc[:split]
    test      = df_full.iloc[split:]

    if use_train_filter:
        ym = train_raw["yes_mid"]
        train = train_raw[(ym >= TRAIN_YES_MID_MIN) & (ym <= TRAIN_YES_MID_MAX)]
    else:
        train = train_raw

    if len(train) < 100:
        return {"error": f"train slice too small after filter (n={len(train)})"}

    X_train = train[feature_cols].values
    y_train = train["y"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    mdl = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
                  cv=5, scoring="r2", fit_intercept=True)
    mdl.fit(X_sc, y_train)

    r2_train  = float(mdl.score(X_sc, y_train))
    X_test_sc = scaler.transform(test[feature_cols].values)
    r2_test   = float(mdl.score(X_test_sc, test["y"].values))

    trades = simulate(test, scaler, mdl.coef_, mdl.intercept_, feature_cols,
                      entry_threshold, exit_threshold, max_hold_s, min_tau_s,
                      use_entry_gate=use_entry_gate, theta_fee=theta_fee,
                      entry_mode=entry_mode, lag_set=lag_set)

    pnl_vals = [t["pnl"] for t in trades]
    n_win    = sum(1 for p in pnl_vals if p > 0)

    return {
        "n_train":     len(train),
        "n_train_raw": len(train_raw),
        "n_test":      len(test),
        "r2_train":    r2_train,
        "r2_test":     r2_test,
        "alpha":       float(mdl.alpha_),
        "n_trades":    len(trades),
        "n_win":       n_win,
        "win_rate":    n_win / len(trades) if trades else 0.0,
        "total_pnl":   sum(pnl_vals),
        "avg_pnl":     np.mean(pnl_vals) if pnl_vals else 0.0,
        "feature_cols": feature_cols,
        "coefs":       mdl.coef_,
        "intercept":   float(mdl.intercept_),
    }


def sweep(df: pd.DataFrame, feature_cols: list[str], train_frac: float,
          entry_thresholds: list[float], exit_thresholds: list[float],
          max_holds: list[int],
          use_train_filter: bool, use_entry_gate: bool,
          theta_fee: float, entry_mode: str = "taker",
          lag_set: set[str] = None) -> pd.DataFrame:
    rows = []
    for et in entry_thresholds:
        for xt in exit_thresholds:
            if xt >= et:
                continue
            for mh in max_holds:
                r = run_backtest(df, feature_cols, train_frac, et, xt, mh, 60,
                                 use_train_filter, use_entry_gate, theta_fee,
                                 entry_mode, lag_set)
                if "error" not in r:
                    rows.append({
                        "entry":     et,
                        "exit":      xt,
                        "hold_s":    mh,
                        "n_trades":  r["n_trades"],
                        "win_rate":  r["win_rate"],
                        "total_pnl": r["total_pnl"],
                        "avg_pnl":   r["avg_pnl"],
                        "r2_test":   r["r2_test"],
                    })
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filter comparison: filter-ON (live-bot) vs filter-OFF on baseline features
# ---------------------------------------------------------------------------

def run_filter_comparison(df: pd.DataFrame, train_frac: float, theta_fee: float):
    """Sweep BASELINE features twice: filter-ON (mirrors live bot) and
    filter-OFF (no train filter, no entry gate). Both with realistic fees."""
    entry_thresholds = [0.010, 0.015, 0.020, 0.030, 0.040, 0.060]
    exit_thresholds  = [0.002, 0.005, 0.010, 0.015]
    max_holds        = [20, 30, 60, 90]

    print("Running sweep with FILTER ON (live-bot behaviour)...")
    sweep_on = sweep(df, BASELINE_COLS, train_frac,
                     entry_thresholds, exit_thresholds, max_holds,
                     use_train_filter=True, use_entry_gate=True,
                     theta_fee=theta_fee)

    print("Running sweep with FILTER OFF (train all rows, enter at any yes_mid)...")
    sweep_off = sweep(df, BASELINE_COLS, train_frac,
                      entry_thresholds, exit_thresholds, max_holds,
                      use_train_filter=False, use_entry_gate=False,
                      theta_fee=theta_fee)

    if sweep_on.empty or sweep_off.empty:
        sys.exit("ERROR: at least one sweep produced zero trades.")

    width = 78
    bar   = "=" * width

    print("\n" + bar)
    print("  TOP 10 -- FILTER ON  (train yes_mid in [0.05,0.95]; "
          "enter only in [0.10,0.90])")
    print(bar)
    print(sweep_on.head(10).to_string(index=False))

    print("\n" + bar)
    print("  TOP 10 -- FILTER OFF (train all rows; enter at any yes_mid)")
    print(bar)
    print(sweep_off.head(10).to_string(index=False))

    # Head-to-head at filter-OFF's best params (the 'naive' winner) -------
    best_off = sweep_off.iloc[0]
    fit_on   = run_backtest(df, BASELINE_COLS, train_frac,
                            float(best_off["entry"]), float(best_off["exit"]),
                            int(best_off["hold_s"]), 60,
                            use_train_filter=True, use_entry_gate=True,
                            theta_fee=theta_fee)
    fit_off  = run_backtest(df, BASELINE_COLS, train_frac,
                            float(best_off["entry"]), float(best_off["exit"]),
                            int(best_off["hold_s"]), 60,
                            use_train_filter=False, use_entry_gate=False,
                            theta_fee=theta_fee)

    fmt = "  {:<22} {:>14}  {:>14}  {:>10}"
    print("\n" + bar)
    print("  HEAD-TO-HEAD AT FILTER-OFF'S BEST PARAMS"
          f"  (entry={best_off['entry']}  exit={best_off['exit']}  "
          f"hold={int(best_off['hold_s'])}s)")
    print(bar)
    print(fmt.format("Metric", "Filter ON", "Filter OFF", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("R2 test",
                     f"{fit_on['r2_test']:.4f}",
                     f"{fit_off['r2_test']:.4f}",
                     f"{fit_off['r2_test'] - fit_on['r2_test']:+.4f}"))
    print(fmt.format("R2 train",
                     f"{fit_on['r2_train']:.4f}",
                     f"{fit_off['r2_train']:.4f}",
                     f"{fit_off['r2_train'] - fit_on['r2_train']:+.4f}"))
    print(fmt.format("Train rows used",
                     f"{fit_on['n_train']}",
                     f"{fit_off['n_train']}",
                     ""))
    print(fmt.format("# trades",
                     f"{fit_on['n_trades']}",
                     f"{fit_off['n_trades']}",
                     f"{fit_off['n_trades'] - fit_on['n_trades']:+d}"))
    print(fmt.format("Win rate",
                     f"{fit_on['win_rate']*100:.1f}%",
                     f"{fit_off['win_rate']*100:.1f}%",
                     f"{(fit_off['win_rate'] - fit_on['win_rate'])*100:+.1f}pp"))
    print(fmt.format("Total P&L (net)",
                     f"${fit_on['total_pnl']:+.4f}",
                     f"${fit_off['total_pnl']:+.4f}",
                     f"${fit_off['total_pnl'] - fit_on['total_pnl']:+.4f}"))
    print(fmt.format("Avg P&L/trade",
                     f"${fit_on['avg_pnl']:+.4f}",
                     f"${fit_off['avg_pnl']:+.4f}",
                     ""))

    # Head-to-head at each set's own best params --------------------------
    best_on = sweep_on.iloc[0]
    print("\n" + bar)
    print("  BEST CONFIG IN EACH MODE")
    print(bar)
    print(fmt.format("Metric", "Filter ON best", "Filter OFF best", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("Best params",
                     f"e={best_on['entry']} x={best_on['exit']} h={int(best_on['hold_s'])}",
                     f"e={best_off['entry']} x={best_off['exit']} h={int(best_off['hold_s'])}",
                     ""))
    print(fmt.format("Total P&L (net)",
                     f"${best_on['total_pnl']:+.4f}",
                     f"${best_off['total_pnl']:+.4f}",
                     f"${best_off['total_pnl'] - best_on['total_pnl']:+.4f}"))
    print(fmt.format("# trades",
                     f"{int(best_on['n_trades'])}",
                     f"{int(best_off['n_trades'])}",
                     f"{int(best_off['n_trades']) - int(best_on['n_trades']):+d}"))
    print(fmt.format("Win rate",
                     f"{best_on['win_rate']*100:.1f}%",
                     f"{best_off['win_rate']*100:.1f}%",
                     f"{(best_off['win_rate'] - best_on['win_rate'])*100:+.1f}pp"))
    print(fmt.format("R2 test",
                     f"{best_on['r2_test']:.4f}",
                     f"{best_off['r2_test']:.4f}",
                     f"{best_off['r2_test'] - best_on['r2_test']:+.4f}"))

    # Verdict --------------------------------------------------------------
    print("\n" + bar)
    print("  VERDICT")
    print(bar)
    common_lift = fit_off["total_pnl"] - fit_on["total_pnl"]
    best_lift   = float(best_off["total_pnl"]) - float(best_on["total_pnl"])
    print(f"  P&L (filter OFF - filter ON) at OFF's best params:  ${common_lift:+.4f}")
    print(f"  P&L (filter OFF - filter ON) at each side's best:   ${best_lift:+.4f}")

    if best_lift > 0.5:
        print("\n  -> FILTER HURTS. Trading the boundary regions adds enough P&L "
              "to offset their noise.")
    elif best_lift < -0.5:
        print("\n  -> FILTER HELPS. Excluding boundary trades produces better "
              "net-of-fee P&L.")
    else:
        print("\n  -> WASH. Filter choice is within noise on this dataset.")


# ---------------------------------------------------------------------------
# Lookback comparison: baseline lags vs baseline + x_5/x_10
# ---------------------------------------------------------------------------

def run_lookback_comparison(raw: pd.DataFrame, train_frac: float, theta_fee: float):
    """Sweep two lookback grids on baseline (non-projection) features, NO
    FILTERS, taker-both fees:

      - LAGS_BASELINE = [15, 30, 60, 90, 120]   (current spec)
      - LAGS_EXTENDED = [5, 10, 15, 30, 60, 90, 120]   (+ 5s and 10s)

    Question: does adding short lags (5s, 10s) catch sub-15s lag arbitrage
    the baseline misses?

    Both variants train on the full data (no yes_mid filter) and enter at
    any yes_mid (no entry gate) -- this matches the 'filter OFF' winner
    from earlier comparisons.
    """
    LAGS_BASELINE = [15, 30, 60, 90, 120]
    LAGS_EXTENDED = [5, 10, 15, 30, 60, 90, 120]

    cols_baseline = (
        ["x_0"] + [f"x_{l}" for l in LAGS_BASELINE]
        + ["tau_s", "inv_sqrt_tau", "spot_mom_30", "spot_mom_60",
           "kalshi_spread", "kalshi_mom_30"]
    )
    cols_extended = (
        ["x_0"] + [f"x_{l}" for l in LAGS_EXTENDED]
        + ["tau_s", "inv_sqrt_tau", "spot_mom_30", "spot_mom_60",
           "kalshi_spread", "kalshi_mom_30"]
    )
    lag_set_baseline = {f"x_{l}" for l in LAGS_BASELINE}
    lag_set_extended = {f"x_{l}" for l in LAGS_EXTENDED}

    print("Building features with BASELINE lags [15,30,60,90,120]...")
    df_base = build_features(raw, lags=LAGS_BASELINE).dropna(subset=cols_baseline)
    print(f"  Usable rows: {len(df_base):,}")

    print("Building features with EXTENDED lags [5,10,15,30,60,90,120]...")
    df_ext  = build_features(raw, lags=LAGS_EXTENDED).dropna(subset=cols_extended)
    print(f"  Usable rows: {len(df_ext):,}")

    entry_thresholds = [0.010, 0.015, 0.020, 0.030, 0.040, 0.060]
    exit_thresholds  = [0.002, 0.005, 0.010, 0.015]
    max_holds        = [20, 30, 60, 90]

    print("\nRunning sweep on BASELINE lags (no filters, taker fees)...")
    sweep_base = sweep(df_base, cols_baseline, train_frac,
                       entry_thresholds, exit_thresholds, max_holds,
                       use_train_filter=False, use_entry_gate=False,
                       theta_fee=theta_fee, entry_mode="taker",
                       lag_set=lag_set_baseline)

    print("Running sweep on EXTENDED lags (no filters, taker fees)...")
    sweep_ext  = sweep(df_ext, cols_extended, train_frac,
                       entry_thresholds, exit_thresholds, max_holds,
                       use_train_filter=False, use_entry_gate=False,
                       theta_fee=theta_fee, entry_mode="taker",
                       lag_set=lag_set_extended)

    if sweep_base.empty or sweep_ext.empty:
        sys.exit("ERROR: at least one sweep produced zero trades.")

    width = 78
    bar   = "=" * width

    print("\n" + bar)
    print("  TOP 10 -- BASELINE lags [15,30,60,90,120]")
    print(bar)
    print(sweep_base.head(10).to_string(index=False))

    print("\n" + bar)
    print("  TOP 10 -- EXTENDED lags [5,10,15,30,60,90,120]")
    print(bar)
    print(sweep_ext.head(10).to_string(index=False))

    # Head-to-head at baseline's best params ----------------------------
    best_b = sweep_base.iloc[0]
    fit_b  = run_backtest(df_base, cols_baseline, train_frac,
                          float(best_b["entry"]), float(best_b["exit"]),
                          int(best_b["hold_s"]), 60,
                          use_train_filter=False, use_entry_gate=False,
                          theta_fee=theta_fee, entry_mode="taker",
                          lag_set=lag_set_baseline)
    fit_e  = run_backtest(df_ext, cols_extended, train_frac,
                          float(best_b["entry"]), float(best_b["exit"]),
                          int(best_b["hold_s"]), 60,
                          use_train_filter=False, use_entry_gate=False,
                          theta_fee=theta_fee, entry_mode="taker",
                          lag_set=lag_set_extended)

    fmt = "  {:<22} {:>14}  {:>14}  {:>10}"
    print("\n" + bar)
    print("  HEAD-TO-HEAD AT BASELINE'S BEST PARAMS"
          f"  (entry={best_b['entry']}  exit={best_b['exit']}  "
          f"hold={int(best_b['hold_s'])}s)")
    print(bar)
    print(fmt.format("Metric", "Baseline lags", "Extended lags", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("R2 test",
                     f"{fit_b['r2_test']:.4f}",
                     f"{fit_e['r2_test']:.4f}",
                     f"{fit_e['r2_test'] - fit_b['r2_test']:+.4f}"))
    print(fmt.format("R2 train",
                     f"{fit_b['r2_train']:.4f}",
                     f"{fit_e['r2_train']:.4f}",
                     f"{fit_e['r2_train'] - fit_b['r2_train']:+.4f}"))
    print(fmt.format("Ridge alpha",
                     f"{fit_b['alpha']:.4f}",
                     f"{fit_e['alpha']:.4f}",
                     ""))
    print(fmt.format("# features",
                     f"{len(cols_baseline)}",
                     f"{len(cols_extended)}",
                     f"+{len(cols_extended) - len(cols_baseline)}"))
    print(fmt.format("# trades",
                     f"{fit_b['n_trades']}",
                     f"{fit_e['n_trades']}",
                     f"{fit_e['n_trades'] - fit_b['n_trades']:+d}"))
    print(fmt.format("Win rate",
                     f"{fit_b['win_rate']*100:.1f}%",
                     f"{fit_e['win_rate']*100:.1f}%",
                     f"{(fit_e['win_rate'] - fit_b['win_rate'])*100:+.1f}pp"))
    print(fmt.format("Total P&L (net)",
                     f"${fit_b['total_pnl']:+.4f}",
                     f"${fit_e['total_pnl']:+.4f}",
                     f"${fit_e['total_pnl'] - fit_b['total_pnl']:+.4f}"))
    print(fmt.format("Avg P&L/trade (net)",
                     f"${fit_b['avg_pnl']:+.4f}",
                     f"${fit_e['avg_pnl']:+.4f}",
                     ""))

    # Each side's own best ---------------------------------------------
    best_e = sweep_ext.iloc[0]
    print("\n" + bar)
    print("  BEST CONFIG IN EACH MODE")
    print(bar)
    print(fmt.format("Metric", "Baseline best", "Extended best", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("Best params",
                     f"e={best_b['entry']} x={best_b['exit']} h={int(best_b['hold_s'])}",
                     f"e={best_e['entry']} x={best_e['exit']} h={int(best_e['hold_s'])}",
                     ""))
    print(fmt.format("Total P&L (net)",
                     f"${best_b['total_pnl']:+.4f}",
                     f"${best_e['total_pnl']:+.4f}",
                     f"${best_e['total_pnl'] - best_b['total_pnl']:+.4f}"))
    print(fmt.format("# trades",
                     f"{int(best_b['n_trades'])}",
                     f"{int(best_e['n_trades'])}",
                     f"{int(best_e['n_trades']) - int(best_b['n_trades']):+d}"))
    print(fmt.format("Win rate",
                     f"{best_b['win_rate']*100:.1f}%",
                     f"{best_e['win_rate']*100:.1f}%",
                     f"{(best_e['win_rate'] - best_b['win_rate'])*100:+.1f}pp"))
    print(fmt.format("R2 test",
                     f"{best_b['r2_test']:.4f}",
                     f"{best_e['r2_test']:.4f}",
                     f"{best_e['r2_test'] - best_b['r2_test']:+.4f}"))

    # Coefficient table for extended ------------------------------------
    print("\n" + bar)
    print("  EXTENDED-VARIANT COEFFICIENTS (sorted by |beta|)")
    print(bar)
    coef_pairs = sorted(zip(cols_extended, fit_e["coefs"]), key=lambda kv: -abs(kv[1]))
    print(f"  {'Feature':<22} {'beta (standardised)':>20}")
    print(f"  {'-'*22} {'-'*20}")
    for name, c in coef_pairs:
        flag = "  <- new short lag" if name in ("x_5", "x_10") else ""
        print(f"  {name:<22} {c:>20.4f}{flag}")
    print(f"  intercept: {fit_e['intercept']:.4f}")

    # Verdict ----------------------------------------------------------
    print("\n" + bar)
    print("  VERDICT")
    print(bar)
    common_lift = fit_e["total_pnl"] - fit_b["total_pnl"]
    best_lift   = float(best_e["total_pnl"]) - float(best_b["total_pnl"])
    short_betas = [c for n, c in zip(cols_extended, fit_e["coefs"]) if n in ("x_5", "x_10")]
    short_max_abs = max(abs(c) for c in short_betas) if short_betas else 0.0

    print(f"  P&L lift at baseline's best params:  ${common_lift:+.4f}")
    print(f"  P&L lift at each side's own best:    ${best_lift:+.4f}")
    print(f"  Max |beta| on short lags (x_5,x_10): {short_max_abs:.4f}")

    if best_lift > 0.5:
        print("\n  -> KEEP. Short lags improve out-of-sample P&L on this dataset.")
    elif best_lift < -0.5:
        print("\n  -> REJECT. Adding short lags hurts out-of-sample P&L.")
    else:
        print("\n  -> MARGINAL. Effect within noise; revisit with more data.")


# ---------------------------------------------------------------------------
# Maker comparison: taker-both vs maker-entry / taker-exit
# ---------------------------------------------------------------------------

def run_maker_comparison(df: pd.DataFrame, train_frac: float, theta_fee: float):
    """Sweep BASELINE features twice with FILTER ON (live-bot mode):
       - taker entry / taker exit  (current execution mode)
       - maker entry / taker exit  (proposed; UPPER BOUND assumes 100% fill)

    The entry SIGNAL is identical (q_settled - yes_ask > threshold) for both
    modes. What differs is the execution price (yes_bid for maker, yes_ask
    for taker) and the entry fee (0 for maker)."""

    entry_thresholds = [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060]
    exit_thresholds  = [0.002, 0.005, 0.010, 0.015]
    max_holds        = [20, 30, 60, 90]

    print("Running sweep: TAKER entry + TAKER exit (current behaviour)...")
    sweep_t = sweep(df, BASELINE_COLS, train_frac,
                    entry_thresholds, exit_thresholds, max_holds,
                    use_train_filter=True, use_entry_gate=True,
                    theta_fee=theta_fee, entry_mode="taker")

    print("Running sweep: MAKER entry + TAKER exit (upper bound, 100% fill)...")
    sweep_m = sweep(df, BASELINE_COLS, train_frac,
                    entry_thresholds, exit_thresholds, max_holds,
                    use_train_filter=True, use_entry_gate=True,
                    theta_fee=theta_fee, entry_mode="maker")

    if sweep_t.empty or sweep_m.empty:
        sys.exit("ERROR: at least one sweep produced zero trades.")

    width = 78
    bar   = "=" * width

    print("\n" + bar)
    print("  TOP 10 -- TAKER both legs  (current execution mode)")
    print(bar)
    print(sweep_t.head(10).to_string(index=False))

    print("\n" + bar)
    print("  TOP 10 -- MAKER entry / TAKER exit  (UPPER BOUND, 100% fill rate)")
    print(bar)
    print(sweep_m.head(10).to_string(index=False))

    # Head-to-head at TAKER's best params --------------------------------
    best_t = sweep_t.iloc[0]
    fit_t  = run_backtest(df, BASELINE_COLS, train_frac,
                          float(best_t["entry"]), float(best_t["exit"]),
                          int(best_t["hold_s"]), 60,
                          use_train_filter=True, use_entry_gate=True,
                          theta_fee=theta_fee, entry_mode="taker")
    fit_m  = run_backtest(df, BASELINE_COLS, train_frac,
                          float(best_t["entry"]), float(best_t["exit"]),
                          int(best_t["hold_s"]), 60,
                          use_train_filter=True, use_entry_gate=True,
                          theta_fee=theta_fee, entry_mode="maker")

    fmt = "  {:<22} {:>14}  {:>14}  {:>10}"
    print("\n" + bar)
    print("  HEAD-TO-HEAD AT TAKER'S BEST PARAMS"
          f"  (entry={best_t['entry']}  exit={best_t['exit']}  "
          f"hold={int(best_t['hold_s'])}s)")
    print(bar)
    print(fmt.format("Metric", "Taker/Taker", "Maker/Taker", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("# trades",
                     f"{fit_t['n_trades']}",
                     f"{fit_m['n_trades']}",
                     f"{fit_m['n_trades'] - fit_t['n_trades']:+d}"))
    print(fmt.format("Win rate",
                     f"{fit_t['win_rate']*100:.1f}%",
                     f"{fit_m['win_rate']*100:.1f}%",
                     f"{(fit_m['win_rate'] - fit_t['win_rate'])*100:+.1f}pp"))
    print(fmt.format("Total P&L (net)",
                     f"${fit_t['total_pnl']:+.4f}",
                     f"${fit_m['total_pnl']:+.4f}",
                     f"${fit_m['total_pnl'] - fit_t['total_pnl']:+.4f}"))
    print(fmt.format("Avg P&L/trade (net)",
                     f"${fit_t['avg_pnl']:+.4f}",
                     f"${fit_m['avg_pnl']:+.4f}",
                     ""))

    # Each side's own best ------------------------------------------------
    best_m = sweep_m.iloc[0]
    print("\n" + bar)
    print("  BEST CONFIG IN EACH MODE")
    print(bar)
    print(fmt.format("Metric", "Taker/Taker best", "Maker/Taker best", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("Best params",
                     f"e={best_t['entry']} x={best_t['exit']} h={int(best_t['hold_s'])}",
                     f"e={best_m['entry']} x={best_m['exit']} h={int(best_m['hold_s'])}",
                     ""))
    print(fmt.format("Total P&L (net)",
                     f"${best_t['total_pnl']:+.4f}",
                     f"${best_m['total_pnl']:+.4f}",
                     f"${best_m['total_pnl'] - best_t['total_pnl']:+.4f}"))
    print(fmt.format("# trades",
                     f"{int(best_t['n_trades'])}",
                     f"{int(best_m['n_trades'])}",
                     f"{int(best_m['n_trades']) - int(best_t['n_trades']):+d}"))
    print(fmt.format("Win rate",
                     f"{best_t['win_rate']*100:.1f}%",
                     f"{best_m['win_rate']*100:.1f}%",
                     f"{(best_m['win_rate'] - best_t['win_rate'])*100:+.1f}pp"))

    # Verdict + caveats ---------------------------------------------------
    print("\n" + bar)
    print("  VERDICT  (UPPER BOUND -- assumes 100% fill on every limit)")
    print(bar)
    common_lift = fit_m["total_pnl"] - fit_t["total_pnl"]
    best_lift   = float(best_m["total_pnl"]) - float(best_t["total_pnl"])
    pct_lift    = (best_lift / float(best_t["total_pnl"]) * 100.0
                   if best_t["total_pnl"] > 0 else float("inf"))
    print(f"  P&L lift at taker's best params:    ${common_lift:+.4f}")
    print(f"  P&L lift at each side's own best:   ${best_lift:+.4f}  ({pct_lift:+.0f}%)")
    print()
    print("  Caveats (this is an UPPER BOUND, not a realised forecast):")
    print("  - Real maker fill rate is < 100%. Per CLAUDE.md section 11.1, fill")
    print("    rate must be measured via Stage A pilot before trusting this.")
    print("  - Adverse selection unmodelled. When our limit fills, the")
    print("    counterparty may be a faster lag-arb bot dumping stale inventory.")
    print("  - The entry signal is unchanged (still q_settled - yes_ask >")
    print("    threshold). A maker-aware signal could enter on a different")
    print("    rule and produce different P&L.")
    print()
    if best_lift > 1.0:
        print("  -> WORTH PURSUING. Maker entry materially improves the upper")
        print("     bound. Next step: Stage A fill-rate pilot per section 11.1.")
    elif best_lift > 0.25:
        print("  -> MARGINAL UPSIDE. Even at 100% fill the lift is small;")
        print("     real fills <100% may erase it. Probably not worth")
        print("     building Stage B around.")
    else:
        print("  -> NOT WORTH IT. The maker entry doesn't even help in the")
        print("     optimistic case. Strategy is fee/edge-bound, not")
        print("     execution-bound.")


# ---------------------------------------------------------------------------
# Main: side-by-side comparison
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Side-by-side model comparisons")
    p.add_argument("--run",        default=None,
                   help="Path to a data/<run> folder (popup if omitted)")
    p.add_argument("--asset",      default="BTC",
                   help="Which asset's ticks file to use (default: BTC)")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--no-filter",  action="store_true",
                   help="(projection mode) Disable training+entry filters")
    p.add_argument("--mode",       choices=["projection", "filters", "maker", "lookback"],
                   default="projection",
                   help="projection: baseline vs +x_proj features. "
                        "filters: filter-ON vs filter-OFF on baseline features. "
                        "maker: taker-both vs maker-entry/taker-exit (upper bound, "
                        "assumes 100%% maker fill rate). "
                        "lookback: baseline lags vs +x_5/+x_10 (no filters, taker fees).")
    p.add_argument("--theta-fee",  type=float, default=DEFAULT_THETA_FEE,
                   help=f"Taker fee theta for fee_per_dollar = theta*p*(1-p). "
                        f"Default {DEFAULT_THETA_FEE} (Kalshi curve in user's chart). "
                        f"Live config currently has {CONFIG_THETA_FEE}.")
    args = p.parse_args()

    use_train_filter = not args.no_filter
    use_entry_gate   = not args.no_filter
    theta_fee        = args.theta_fee

    run_dir   = pick_run_folder(cli_arg=args.run, title="Select run to compare")
    asset     = args.asset.upper()
    tick_path = run_dir / f"ticks_{asset}.csv"
    if not tick_path.exists():
        legacy = run_dir / "ticks.csv"
        if legacy.exists():
            tick_path = legacy
    if not tick_path.exists():
        sys.exit(f"ERROR: {tick_path} not found.")

    print(f"Loading {tick_path}...")
    raw = pd.read_csv(tick_path)
    print(f"  Raw rows:  {len(raw):,}")
    print(f"  Time span: {(raw['ts_ns'].max() - raw['ts_ns'].min()) / 1e9 / 60:.1f} min")
    print(f"  Windows:   {raw['window_ticker'].nunique()}")
    print(f"  Filters:   train yes_mid in [{TRAIN_YES_MID_MIN}, {TRAIN_YES_MID_MAX}]"
          f"  entry yes_mid in [{DECISION_YES_MID_MIN}, {DECISION_YES_MID_MAX}]"
          f"  ({'ON' if use_train_filter else 'OFF'})")

    df = build_features(raw)
    df = df.dropna(subset=PROJ_COLS)

    # Show how the row count breaks down once the live filters are applied.
    n_total      = len(df)
    train_split  = int(n_total * args.train_frac)
    train_slice  = df.iloc[:train_split]
    test_slice   = df.iloc[train_split:]
    n_train_kept = ((train_slice["yes_mid"] >= TRAIN_YES_MID_MIN)
                    & (train_slice["yes_mid"] <= TRAIN_YES_MID_MAX)).sum()
    n_test_entry = ((test_slice["yes_mid"] >= DECISION_YES_MID_MIN)
                    & (test_slice["yes_mid"] <= DECISION_YES_MID_MAX)).sum()
    print(f"  Total rows (features complete): {n_total:,}")
    print(f"  Train slice: {len(train_slice):,} rows  "
          f"(after train-filter: {int(n_train_kept):,} kept)")
    print(f"  Test slice:  {len(test_slice):,} rows  "
          f"(entry-eligible: {int(n_test_entry):,} -- "
          f"others can still close existing positions)")
    print(f"  Fee model:   theta_fee={theta_fee}  (peak {theta_fee*0.25*100:.2f}% per leg "
          f"at p=0.5); taker-on-entry + taker-on-exit applied to every simulated trade")
    print(f"  Mode:        {args.mode}\n")

    if args.mode == "filters":
        return run_filter_comparison(df, args.train_frac, theta_fee)
    if args.mode == "maker":
        return run_maker_comparison(df, args.train_frac, theta_fee)
    if args.mode == "lookback":
        return run_lookback_comparison(raw, args.train_frac, theta_fee)

    if n_total < 500:
        sys.exit("Need at least 500 ticks with complete features.")

    entry_thresholds = [0.010, 0.015, 0.020, 0.030, 0.040, 0.060]
    exit_thresholds  = [0.002, 0.005, 0.010, 0.015]
    max_holds        = [20, 30, 60, 90]

    print(f"Running sweep on BASELINE ({len(BASELINE_COLS)} features)...")
    sweep_baseline = sweep(df, BASELINE_COLS, args.train_frac,
                           entry_thresholds, exit_thresholds, max_holds,
                           use_train_filter, use_entry_gate, theta_fee)

    print(f"Running sweep on PROJECTION ({len(PROJ_COLS)} features = baseline + 3 x_proj)...")
    sweep_proj = sweep(df, PROJ_COLS, args.train_frac,
                       entry_thresholds, exit_thresholds, max_holds,
                       use_train_filter, use_entry_gate, theta_fee)

    if sweep_baseline.empty or sweep_proj.empty:
        sys.exit("ERROR: at least one sweep produced zero trades. "
                 "Try --no-filter to compare without gating, or collect more data.")

    best_b = sweep_baseline.iloc[0]
    fit_b  = run_backtest(df, BASELINE_COLS, args.train_frac,
                          float(best_b["entry"]), float(best_b["exit"]),
                          int(best_b["hold_s"]), 60,
                          use_train_filter, use_entry_gate, theta_fee)
    fit_p  = run_backtest(df, PROJ_COLS, args.train_frac,
                          float(best_b["entry"]), float(best_b["exit"]),
                          int(best_b["hold_s"]), 60,
                          use_train_filter, use_entry_gate, theta_fee)

    width = 78
    bar   = "=" * width

    print("\n" + bar)
    print("  TOP 10 -- BASELINE")
    print(bar)
    print(sweep_baseline.head(10).to_string(index=False))

    print("\n" + bar)
    print("  TOP 10 -- WITH PROJECTION FEATURES")
    print(bar)
    print(sweep_proj.head(10).to_string(index=False))

    print("\n" + bar)
    print("  HEAD-TO-HEAD AT BASELINE'S BEST PARAMS"
          f"  (entry={best_b['entry']}  exit={best_b['exit']}  hold={int(best_b['hold_s'])}s)")
    print(bar)
    fmt = "  {:<22} {:>14}  {:>14}  {:>10}"
    print(fmt.format("Metric", "Baseline", "+ Projection", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("R2 test",
                     f"{fit_b['r2_test']:.4f}",
                     f"{fit_p['r2_test']:.4f}",
                     f"{fit_p['r2_test'] - fit_b['r2_test']:+.4f}"))
    print(fmt.format("R2 train",
                     f"{fit_b['r2_train']:.4f}",
                     f"{fit_p['r2_train']:.4f}",
                     f"{fit_p['r2_train'] - fit_b['r2_train']:+.4f}"))
    print(fmt.format("Ridge alpha",
                     f"{fit_b['alpha']:.4f}",
                     f"{fit_p['alpha']:.4f}",
                     ""))
    print(fmt.format("Train rows kept",
                     f"{fit_b['n_train']}/{fit_b['n_train_raw']}",
                     f"{fit_p['n_train']}/{fit_p['n_train_raw']}",
                     ""))
    print(fmt.format("# trades",
                     f"{fit_b['n_trades']}",
                     f"{fit_p['n_trades']}",
                     f"{fit_p['n_trades'] - fit_b['n_trades']:+d}"))
    print(fmt.format("Win rate",
                     f"{fit_b['win_rate']*100:.1f}%",
                     f"{fit_p['win_rate']*100:.1f}%",
                     f"{(fit_p['win_rate'] - fit_b['win_rate'])*100:+.1f}pp"))
    print(fmt.format("Total P&L (net)",
                     f"${fit_b['total_pnl']:+.4f}",
                     f"${fit_p['total_pnl']:+.4f}",
                     f"${fit_p['total_pnl'] - fit_b['total_pnl']:+.4f}"))
    print(fmt.format("Avg P&L/trade (net)",
                     f"${fit_b['avg_pnl']:+.4f}",
                     f"${fit_p['avg_pnl']:+.4f}",
                     ""))

    print("\n" + bar)
    print("  BEST CONFIG IN EACH FEATURE SET")
    print(bar)
    best_p = sweep_proj.iloc[0]
    print(fmt.format("Metric", "Baseline best", "+ Proj best", "delta"))
    print(fmt.format("-" * 22, "-" * 14, "-" * 14, "-" * 10))
    print(fmt.format("Best params",
                     f"e={best_b['entry']} x={best_b['exit']} h={int(best_b['hold_s'])}",
                     f"e={best_p['entry']} x={best_p['exit']} h={int(best_p['hold_s'])}",
                     ""))
    print(fmt.format("Total P&L",
                     f"${best_b['total_pnl']:+.4f}",
                     f"${best_p['total_pnl']:+.4f}",
                     f"${best_p['total_pnl'] - best_b['total_pnl']:+.4f}"))
    print(fmt.format("Win rate",
                     f"{best_b['win_rate']*100:.1f}%",
                     f"{best_p['win_rate']*100:.1f}%",
                     f"{(best_p['win_rate'] - best_b['win_rate'])*100:+.1f}pp"))
    print(fmt.format("R2 test",
                     f"{best_b['r2_test']:.4f}",
                     f"{best_p['r2_test']:.4f}",
                     f"{best_p['r2_test'] - best_b['r2_test']:+.4f}"))

    print("\n" + bar)
    print("  PROJECTION-VARIANT COEFFICIENTS")
    print("  (beta values on standardised features; magnitude reflects predictive weight)")
    print(bar)
    coef_pairs = sorted(zip(PROJ_COLS, fit_p["coefs"]), key=lambda kv: -abs(kv[1]))
    print(f"  {'Feature':<22} {'beta (standardised)':>20}")
    print(f"  {'-'*22} {'-'*20}")
    for name, c in coef_pairs:
        flag = "  <- projection" if name.startswith("x_proj_") else ""
        print(f"  {name:<22} {c:>20.4f}{flag}")
    print(f"  intercept: {fit_p['intercept']:.4f}")

    print("\n" + bar)
    print("  VERDICT")
    print(bar)
    pnl_lift_at_baseline = fit_p["total_pnl"] - fit_b["total_pnl"]
    pnl_lift_best        = float(best_p["total_pnl"]) - float(best_b["total_pnl"])
    r2_lift              = fit_p["r2_test"] - fit_b["r2_test"]
    proj_coefs           = [c for n, c in zip(PROJ_COLS, fit_p["coefs"]) if n.startswith("x_proj_")]
    proj_max_abs         = max(abs(c) for c in proj_coefs)

    print(f"  P&L lift at baseline's best params:  ${pnl_lift_at_baseline:+.4f}")
    print(f"  P&L lift at each set's own best:     ${pnl_lift_best:+.4f}")
    print(f"  R2_test lift:                         {r2_lift:+.4f}")
    print(f"  Max |beta| on projection features:    {proj_max_abs:.4f}")

    if pnl_lift_best > 0.5 and pnl_lift_at_baseline >= -0.25:
        print("\n  -> KEEP. Projection features improve out-of-sample P&L.")
    elif pnl_lift_best < -0.5:
        print("\n  -> REJECT. Projection features hurt out-of-sample P&L.")
    else:
        print("\n  -> MARGINAL. Effect within noise; revisit with more data.")


if __name__ == "__main__":
    main()
