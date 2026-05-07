"""
live_plot.py — Live multi-asset rolling chart for an active bot run.

One subplot per asset (BTC, ETH, SOL, XRP).
Each subplot: left axis = spot microprice - floor_strike, right axis = Kalshi YES mid.
Markers: green ^ on entry, red v on exit.

Reads ticks_<ASSET>.csv and decisions_<ASSET>.jsonl from the selected run folder.
Refreshes every 2 seconds so you can watch a live run in real time.

Usage:
  python scripts/analysis/live_plot.py                    # popup to pick run folder
  python scripts/analysis/live_plot.py --run data/<run>   # skip popup
  python scripts/analysis/live_plot.py --assets BTC ETH   # subset of assets
  python scripts/analysis/live_plot.py --window 60        # 60-minute rolling window
  python scripts/analysis/live_plot.py --interval 2       # refresh every 2s
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_run import pick_run_folder

ASSET_COLORS = {
    "BTC": "steelblue",
    "ETH": "mediumseagreen",
    "SOL": "darkorchid",
    "XRP": "darkorange",
}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_ticks_tail(path: Path, window_s: float) -> list[dict]:
    if not path.exists():
        return []
    cutoff = time.time() - window_s
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    ts = int(r["ts_ns"]) / 1e9
                    if ts >= cutoff:
                        K = float(r["floor_strike"])
                        rows.append({
                            "ts":      datetime.datetime.fromtimestamp(ts),
                            "delta":   float(r["btc_microprice"]) - K,
                            "yes_mid": (float(r["yes_bid"]) + float(r["yes_ask"])) / 2.0,
                        })
                except (KeyError, ValueError):
                    pass
    except Exception:
        pass
    return rows


def read_decisions_tail(path: Path, window_s: float) -> list[dict]:
    if not path.exists():
        return []
    cutoff = time.time() - window_s
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r["ts_ns"] / 1e9 >= cutoff and r.get("yes_mid") not in (None, 0):
                        rows.append(r)
                except Exception:
                    pass
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Per-asset subplot draw
# ---------------------------------------------------------------------------

def redraw_asset(ax_spot, ax_yes, run_dir: Path, asset: str,
                 window_s: float, color: str) -> None:
    ax_spot.cla()
    ax_yes.cla()

    ticks_file     = run_dir / f"ticks_{asset}.csv"
    decisions_file = run_dir / f"decisions_{asset}.jsonl"

    ticks     = read_ticks_tail(ticks_file, window_s)
    decisions = read_decisions_tail(decisions_file, window_s)

    if not ticks:
        ax_spot.set_title(f"{asset} — waiting for data...", fontsize=10)
        ax_spot.set_ylabel("spot - strike (USD)", color="gray", fontsize=8)
        return

    ts      = [r["ts"]      for r in ticks]
    delta   = [r["delta"]   for r in ticks]
    yes_mid = [r["yes_mid"] for r in ticks]

    ax_spot.plot(ts, delta, color=color, lw=1.1, label=f"{asset} - strike")
    ax_spot.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.5)
    ax_spot.set_ylabel("spot - strike (USD)", color=color, fontsize=8)
    ax_spot.tick_params(axis="y", labelcolor=color, labelsize=7)

    ax_yes.plot(ts, yes_mid, color="darkorange", lw=1.1, label="YES mid")
    ax_yes.axhline(0.5, color="darkorange", lw=0.6, linestyle="--", alpha=0.4)
    ax_yes.set_ylabel("YES prob", color="darkorange", fontsize=8)
    ax_yes.tick_params(axis="y", labelcolor="darkorange", labelsize=7)
    ax_yes.set_ylim(0, 1)

    # q_settled from the decisions log — model's predicted price
    pred_rows = [(r["ts_ns"], r["q_settled"]) for r in decisions
                 if r.get("q_settled") not in (None, "") and r.get("yes_mid", 0) > 0]
    if pred_rows:
        pred_ts  = [datetime.datetime.fromtimestamp(t / 1e9) for t, _ in pred_rows]
        pred_qs  = [float(q) for _, q in pred_rows]
        ax_yes.plot(pred_ts, pred_qs, color="mediumpurple", lw=1.2,
                    linestyle="--", alpha=0.85, label="q_settled (model)")

    entry_rows = [r for r in decisions if r.get("event") == "entry"]
    exit_rows  = [r for r in decisions if r.get("event") in
                  ("exit_lag_closed", "exit_stopped", "exit_max_hold", "fallback_resolution")]

    if entry_rows:
        ex = [datetime.datetime.fromtimestamp(r["ts_ns"] / 1e9) for r in entry_rows]
        ey = [r["yes_ask"] for r in entry_rows]
        ax_yes.scatter(ex, ey, marker="^", color="green", s=80, zorder=6, label="entry")

    if exit_rows:
        xx = [datetime.datetime.fromtimestamp(r["ts_ns"] / 1e9) for r in exit_rows]
        xy = [r["yes_bid"] for r in exit_rows]
        ax_yes.scatter(xx, xy, marker="v", color="red", s=80, zorder=6, label="exit")

    ax_spot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_spot.grid(alpha=0.2)
    ax_spot.tick_params(axis="x", labelsize=7)

    n_e     = len(entry_rows)
    n_x     = len(exit_rows)
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    ax_spot.set_title(
        f"{asset}  —  {n_e} entries  {n_x} exits  [{now_str}]",
        fontsize=10,
    )

    h1, l1 = ax_spot.get_legend_handles_labels()
    h2, l2 = ax_yes.get_legend_handles_labels()
    ax_yes.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Live rolling chart for an active bot run")
    parser.add_argument("--run",      type=str, default=None,
                        help="Path to a data/<run> folder (popup if omitted)")
    parser.add_argument("--assets",   nargs="+", default=["BTC", "ETH", "SOL", "XRP"],
                        help="Assets to show (default: BTC ETH SOL XRP)")
    parser.add_argument("--window",   type=int, default=60,
                        help="Rolling window in minutes (default 60)")
    parser.add_argument("--interval", type=int, default=2,
                        help="Refresh interval in seconds (default 2)")
    args = parser.parse_args()

    run_dir  = pick_run_folder(cli_arg=args.run, title="Select run to monitor")
    assets   = [a.upper() for a in args.assets]
    # Only show assets that have a ticks file in this run
    assets   = [a for a in assets if (run_dir / f"ticks_{a}.csv").exists()]
    if not assets:
        sys.exit(f"No ticks files found in {run_dir}. Is the bot running?")

    window_s = args.window * 60
    n        = len(assets)

    print(f"Monitoring: {run_dir}", flush=True)
    print(f"Assets: {', '.join(assets)}  Window: {args.window}m  Refresh: {args.interval}s",
          flush=True)

    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    ax_pairs = []
    for ax in axes:
        ax_yes = ax.twinx()
        ax_pairs.append((ax, ax_yes))

    fig.tight_layout(pad=2.5, h_pad=3.0)

    def _update(_frame):
        for (ax_spot, ax_yes), asset in zip(ax_pairs, assets):
            color = ASSET_COLORS.get(asset, "steelblue")
            redraw_asset(ax_spot, ax_yes, run_dir, asset, window_s, color)
        fig.tight_layout(pad=2.5, h_pad=3.0)

    _update(0)
    ani = animation.FuncAnimation(
        fig, _update,
        interval=args.interval * 1000,
        cache_frame_data=False,
    )
    plt.show()
    _ = ani


if __name__ == "__main__":
    main()
