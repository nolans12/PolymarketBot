"""
tune.py — Sweep Kelly tier configurations through the realistic backtest.

Replaces tune_trading_knobs.py. Uses the maker-entry / taker-exit fill model
in scripts/backtest.py so P&L reflects true Kalshi mechanics.

Output: prints the top-N tier configurations by net P&L, plus a config.py
snippet you can paste in.

Usage:
  python scripts/tune.py --model-file model_fits/<dir>/model.pkl
  python scripts/tune.py --run data/<run> --asset BTC --model-file ...
"""

import argparse
import sys
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import betbot.kalshi.config as cfg
from betbot.kalshi.model import load_model
from scripts.analysis.tick_loader import load_ticks, find_ticks_path
from scripts.analysis.pick_run import pick_run_folder
from scripts.backtest import backtest_window, summarize, Trade


# Sweep grid — edge floors and wallet fractions to try per tier.
# Tier 1 is the highest-confidence (highest edge floor) bet.
EDGE_FLOORS_T1 = [0.10, 0.12, 0.15, 0.20]
EDGE_FLOORS_T2 = [0.05, 0.06, 0.08, 0.10]
EDGE_FLOORS_T3 = [0.02, 0.03, 0.04]
WALLET_FRACS_T1 = [0.04, 0.06, 0.08]
WALLET_FRACS_T2 = [0.02, 0.04]
WALLET_FRACS_T3 = [0.01, 0.02]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--asset", type=str, default="BTC")
    ap.add_argument("--model-file", type=str, required=True)
    ap.add_argument("--wallet", type=float, default=1000.0)
    ap.add_argument("--top", type=int, default=10,
                    help="show top-N configurations by net P&L")
    ap.add_argument("--max-windows", type=int, default=80,
                    help="cap on number of windows to evaluate per config "
                         "(default 80; pass 0 for all). With 100+ configs in "
                         "the sweep this matters a lot — 80 windows is plenty "
                         "to pick good tiers.")
    args = ap.parse_args()

    run_dir    = pick_run_folder(cli_arg=args.run, title="Select run to tune on")
    asset      = args.asset.upper()
    model_path = Path(args.model_file)
    if not model_path.is_absolute():
        model_path = REPO / model_path

    tick_path = find_ticks_path(run_dir, asset)
    if tick_path is None:
        sys.exit(f"  no ticks for {asset} in {run_dir}")
    print(f"Loading ticks from {tick_path}...")
    windows = load_ticks(tick_path)
    if not windows:
        sys.exit("  no tick data")
    print(f"  {len(windows)} windows  {sum(len(rs) for rs in windows.values())} ticks")

    # Subsample windows so the sweep finishes in reasonable time
    window_items = [(k, v) for k, v in windows.items() if len(v) >= 60]
    if args.max_windows and args.max_windows > 0 and len(window_items) > args.max_windows:
        # Take a stratified sample (evenly spaced) for variety
        step = len(window_items) // args.max_windows
        window_items = window_items[::step][:args.max_windows]
        print(f"  Sampled {len(window_items)} windows for the sweep "
              f"(use --max-windows 0 to use all)")

    print(f"Loading model from {model_path}...")
    model = load_model(str(model_path))
    print(f"  R2_hld={model.r2_held_out:.3f}")

    # Build the parameter grid
    configs = []
    for ef1, ef2, ef3 in product(EDGE_FLOORS_T1, EDGE_FLOORS_T2, EDGE_FLOORS_T3):
        if not (ef1 > ef2 > ef3):
            continue
        for wf1, wf2, wf3 in product(WALLET_FRACS_T1, WALLET_FRACS_T2, WALLET_FRACS_T3):
            if not (wf1 >= wf2 >= wf3):
                continue
            configs.append([
                (ef1, wf1), (ef2, wf2), (ef3, wf3),
            ])
    print(f"  Sweeping {len(configs)} tier configurations...")

    import time
    results = []
    t0 = time.monotonic()
    last_report = t0
    for i, tiers in enumerate(configs, 1):
        # Patch config.KELLY_TIERS in place — backtest reads it dynamically
        cfg.KELLY_TIERS[:] = tiers
        all_trades: list[Trade] = []
        for window_ticker, rows in window_items:
            all_trades.extend(backtest_window(rows, window_ticker, asset, model, args.wallet))

        n = len(all_trades)
        if n == 0:
            continue
        wins = sum(1 for t in all_trades if t.pnl_net > 0)
        net  = sum(t.pnl_net for t in all_trades)
        results.append({
            "tiers": tiers, "trades": n, "wins": wins, "net": net,
            "win_rate": wins / n,
        })
        now = time.monotonic()
        if now - last_report >= 10.0:
            pct = 100 * i / len(configs)
            rate = i / (now - t0) if now > t0 else 0
            eta = (len(configs) - i) / rate if rate > 0 else 0
            print(f"  ... {i}/{len(configs)} configs ({pct:.0f}%)  "
                  f"ETA {eta:.0f}s", flush=True)
            last_report = now

    if not results:
        sys.exit("  no configurations produced any trades")

    results.sort(key=lambda r: -r["net"])

    print(f"\n=== Top {args.top} configurations by net P&L ===\n")
    print(f"{'rank':>4}  {'trades':>6}  {'wins':>5}  {'wr':>6}  {'net':>9}  tiers")
    for rank, r in enumerate(results[:args.top], 1):
        tiers_str = "  ".join(f"({ef:.2f},{wf:.2f})" for ef, wf in r["tiers"])
        print(f"{rank:>4}  {r['trades']:>6}  {r['wins']:>5}  "
              f"{100*r['win_rate']:>5.1f}%  ${r['net']:>+8.2f}  {tiers_str}")

    best = results[0]
    print(f"\n=== Best configuration — paste into config.py ===\n")
    print("KELLY_TIERS = [")
    for ef, wf in best["tiers"]:
        print(f"    ({ef:.2f}, {wf:.2f}),")
    print("]")
    print(f"\n  Net P&L: ${best['net']:+.2f}  Trades: {best['trades']}  Wins: {best['wins']}")


if __name__ == "__main__":
    main()
