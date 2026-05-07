"""
live_plot.py — Live dual-axis rolling chart.

Top axis : Coinbase BTC microprice (left y) + Kalshi YES mid (right y)
Markers  : green ^ on entry, red v on exit

Reads logs/ticks.csv (1Hz, written by the bot) for prices and
logs/decisions.jsonl for trade events. Keeps a 15-minute rolling window.

Usage:
  python scripts/live_plot.py
  python scripts/live_plot.py --window 10   # 10-minute window
  python scripts/live_plot.py --interval 2  # refresh every 2s
"""

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import numpy as np
import datetime


TICKS_PATH     = Path("logs/ticks.csv")
DECISIONS_PATH = Path("logs/decisions.jsonl")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_ticks_tail(path: Path, window_s: float) -> list[dict]:
    """Read ticks.csv rows from the last window_s seconds."""
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
                            "ts":       datetime.datetime.fromtimestamp(ts),
                            "delta":    float(r["btc_microprice"]) - K,
                            "cb_delta": float(r.get("cb_microprice") or r["btc_microprice"]) - K,
                            "bn_delta": float(r.get("bn_microprice") or r["btc_microprice"]) - K,
                            "yes_mid":  (float(r["yes_bid"]) + float(r["yes_ask"])) / 2.0,
                        })
                except (KeyError, ValueError):
                    pass
    except Exception:
        pass
    return rows


def read_decisions_tail(path: Path, window_s: float) -> list[dict]:
    """Read decision rows from the last window_s seconds."""
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
                    ts = r["ts_ns"] / 1e9
                    if ts >= cutoff:
                        rows.append(r)
                except Exception:
                    pass
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def redraw(ax_btc, ax_yes, window_s: float) -> None:
    ax_btc.cla()
    ax_yes.cla()

    ticks     = read_ticks_tail(TICKS_PATH, window_s)
    decisions = read_decisions_tail(DECISIONS_PATH, window_s)

    if not ticks:
        ax_btc.set_title("Waiting for data from bot...")
        return

    ts       = [r["ts"]       for r in ticks]
    cb_delta = [r["cb_delta"] for r in ticks]
    bn_delta = [r["bn_delta"] for r in ticks]
    yes_mid  = [r["yes_mid"]  for r in ticks]

    # Both feeds delta-from-strike on left axis
    ax_btc.plot(ts, cb_delta, color="steelblue",  lw=1.2, label="Coinbase − strike")
    ax_btc.plot(ts, bn_delta, color="darkorchid", lw=1.2, label="Binance − strike", alpha=0.8)
    ax_btc.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.5)
    ax_btc.set_ylabel("BTC − strike (USD)", color="gray")
    ax_btc.tick_params(axis="y", labelcolor="steelblue")

    # Kalshi YES mid on right axis
    ax_yes.plot(ts, yes_mid, color="darkorange", lw=1.3, label="Kalshi YES mid")
    ax_yes.axhline(0.5, color="darkorange", lw=0.7, linestyle="--", alpha=0.4)
    ax_yes.set_ylabel("Kalshi YES probability", color="darkorange")
    ax_yes.tick_params(axis="y", labelcolor="darkorange")
    ax_yes.set_ylim(0, 1)

    # Entry / exit markers — plotted on the YES axis
    entry_rows = [r for r in decisions if r["event"] == "entry"]
    exit_rows  = [r for r in decisions if r["event"] in
                  ("exit_lag_closed", "exit_stopped", "exit_max_hold", "fallback_resolution")]

    if entry_rows:
        ex = [datetime.datetime.fromtimestamp(r["ts_ns"] / 1e9) for r in entry_rows]
        ey = [r["yes_ask"] for r in entry_rows]
        ax_yes.scatter(ex, ey, marker="^", color="green", s=120, zorder=6, label="entry")

    if exit_rows:
        xx = [datetime.datetime.fromtimestamp(r["ts_ns"] / 1e9) for r in exit_rows]
        xy = [r["yes_bid"] for r in exit_rows]
        ax_yes.scatter(xx, xy, marker="v", color="red", s=120, zorder=6, label="exit")

    # X axis formatting
    ax_btc.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_btc.grid(alpha=0.25)

    # Combined legend
    h1, l1 = ax_btc.get_legend_handles_labels()
    h2, l2 = ax_yes.get_legend_handles_labels()
    ax_yes.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    n_entries = len(entry_rows)
    n_exits   = len(exit_rows)
    ax_btc.set_title(
        f"Kalshi BTC  —  {n_entries} entries  {n_exits} exits  "
        f"(last {window_s//60:.0f} min)  [{now_str}]",
        fontsize=11,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",   type=int, default=15,
                        help="Rolling window in minutes (default 15)")
    parser.add_argument("--interval", type=int, default=2,
                        help="Refresh interval in seconds (default 2)")
    args = parser.parse_args()

    window_s = args.window * 60

    fig, ax_btc = plt.subplots(figsize=(14, 5))
    ax_yes = ax_btc.twinx()
    fig.tight_layout(pad=2.5)

    def _update(_frame):
        redraw(ax_btc, ax_yes, window_s)
        fig.tight_layout(pad=2.5)

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
