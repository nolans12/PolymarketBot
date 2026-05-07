"""
analyze_run.py -- Visualize a completed (or in-progress) dry-run decision log.

Prompts for a data/ run folder (Tkinter popup), then reads decisions_<ASSET>.jsonl
from that folder and produces:
  1. Kalshi YES price over time (bid / ask / model q_settled / q_predicted)
  2. Edge magnitude over time with Kelly tier floor lines
  3. Model R2 and estimated lag over time
  4. Hypothetical cumulative P&L (simulated from logged entry/exit events)
  5. Abstention reason breakdown

Usage:
  python scripts/analysis/analyze_run.py
  python scripts/analysis/analyze_run.py --run data/2026-05-07_00-15-00_BTC
  python scripts/analysis/analyze_run.py --asset ETH
  python scripts/analysis/analyze_run.py --save results.png
  python scripts/analysis/analyze_run.py --live --interval 5
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from betbot.kalshi.config import KELLY_TIERS, THETA_FEE_TAKER as THETA_FEE
from pick_run import pick_run_folder


# ---------------------------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------------------------

def load_log(path: Path) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Summary stats (printed to terminal)
# ---------------------------------------------------------------------------

def print_summary(rows: list) -> None:
    if not rows:
        print("No rows found.")
        return

    import datetime
    t0 = rows[0]["ts_ns"] / 1e9
    t1 = rows[-1]["ts_ns"] / 1e9
    span_min = (t1 - t0) / 60

    events = defaultdict(int)
    abstentions = defaultdict(int)
    for r in rows:
        events[r["event"]] += 1
        if r.get("abstention_reason"):
            abstentions[r["abstention_reason"]] += 1

    r2s  = [r["model_r2_hld"] for r in rows if r.get("model_r2_hld", 0) > 0]
    lags = [r["model_lag_s"]  for r in rows if r.get("model_lag_s",  0) > 0]
    edges = [r["edge_magnitude"] for r in rows
             if r.get("edge_magnitude") and r["edge_magnitude"] > -50]

    entries = [r for r in rows if r["event"] == "entry"]
    exits   = [r for r in rows if r["event"] in
               ("exit_lag_closed", "exit_stopped", "fallback_resolution")]

    print("=" * 60)
    print("  DRY-RUN SUMMARY")
    print("=" * 60)
    print(f"  Span:          {span_min:.1f} minutes  ({len(rows)} ticks)")
    print(f"  Windows:       {len(set(r['window_ticker'] for r in rows))}")
    print()
    print("  Events:")
    for ev, n in sorted(events.items(), key=lambda x: -x[1]):
        print(f"    {ev:30s} {n:5d}")
    print()
    if abstentions:
        print("  Abstention reasons:")
        for reason, n in sorted(abstentions.items(), key=lambda x: -x[1]):
            print(f"    {reason:30s} {n:5d}")
        print()
    if r2s:
        print(f"  Model R2 (cv):  median={np.median(r2s):.3f}  "
              f"p10={np.percentile(r2s,10):.3f}  p90={np.percentile(r2s,90):.3f}")
    if lags:
        print(f"  Est. lag (s):   median={np.median(lags):.0f}  "
              f"min={min(lags):.0f}  max={max(lags):.0f}")
    if edges:
        print(f"  Edge magnitude: median={np.median(edges):.4f}  "
              f"p90={np.percentile(edges,90):.4f}  p99={np.percentile(edges,99):.4f}")
    print(f"  Entries:       {len(entries)}")
    print(f"  Exits:         {len(exits)}")
    print()

    pnl_rows = compute_pnl(rows)
    if pnl_rows:
        total_pnl = sum(p["pnl"] for p in pnl_rows)
        total_bet = sum(p["size_usd"] for p in pnl_rows)
        n_win = sum(1 for p in pnl_rows if p["pnl"] > 0)
        print(f"  Simulated P&L: ${total_pnl:+.2f}  "
              f"(ROI {total_pnl/total_bet*100:.1f}%  "
              f"win-rate {n_win}/{len(pnl_rows)})")
        print()
        for ev_type, label in [("exit_lag_closed", "lag_closed"),
                                ("exit_stopped",    "stopped"),
                                ("fallback_resolution", "resolution")]:
            sub = [p for p in pnl_rows if p["exit_event"] == ev_type]
            if sub:
                avg_pnl = np.mean([p["pnl"] for p in sub])
                print(f"    {label:15s}: {len(sub):3d} trades  avg_pnl=${avg_pnl:+.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# P&L simulation from entry/exit pairs
# ---------------------------------------------------------------------------

def compute_pnl(rows: list) -> list:
    results = []
    open_entry = None

    for r in rows:
        if r["event"] == "entry":
            open_entry = r
        elif r["event"] in ("exit_lag_closed", "exit_stopped", "fallback_resolution"):
            if open_entry is None:
                continue
            side      = open_entry.get("favored_side", "yes")
            entry_p   = open_entry.get("yes_ask", 0)
            size_usd  = open_entry.get("would_bet_usd", 0)
            contracts = size_usd / max(entry_p, 1e-6)

            if side == "yes":
                exit_p = r.get("yes_bid", entry_p)
            else:
                exit_p = 1.0 - r.get("yes_ask", 1.0 - entry_p)

            if r["event"] == "fallback_resolution":
                exit_p = r.get("q_settled") or exit_p

            gross_in  = contracts * entry_p
            gross_out = contracts * exit_p
            fee_in    = THETA_FEE * entry_p * (1 - entry_p) * gross_in
            fee_out   = THETA_FEE * exit_p  * (1 - exit_p)  * gross_out
            pnl       = gross_out - gross_in - fee_in - fee_out

            results.append({
                "entry_ts":   open_entry["ts_ns"] / 1e9,
                "exit_ts":    r["ts_ns"] / 1e9,
                "exit_event": r["event"],
                "side":       side,
                "entry_p":    entry_p,
                "exit_p":     exit_p,
                "size_usd":   size_usd,
                "pnl":        pnl,
            })
            open_entry = None

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw(axes, ax3b, rows: list) -> None:
    import datetime

    for ax in axes:
        ax.cla()
    ax3b.cla()

    if not rows:
        return

    ts = [datetime.datetime.fromtimestamp(r["ts_ns"] / 1e9) for r in rows]

    yes_bid  = np.array([r["yes_bid"]  for r in rows])
    yes_ask  = np.array([r["yes_ask"]  for r in rows])
    yes_mid  = (yes_bid + yes_ask) / 2.0
    q_set    = np.array([r["q_settled"]   if r.get("q_settled")   is not None else np.nan for r in rows])
    q_pred   = np.array([r["q_predicted"] if r.get("q_predicted") is not None else np.nan for r in rows])
    edge_mag = np.array([r["edge_magnitude"] if r.get("edge_magnitude", -99) > -50 else np.nan for r in rows])
    r2       = np.array([r.get("model_r2_hld", 0) for r in rows])
    lag_s    = np.array([r.get("model_lag_s", 0)  for r in rows])

    entry_rows = [r for r in rows if r["event"] == "entry"]
    entry_ts   = [datetime.datetime.fromtimestamp(r["ts_ns"]/1e9) for r in entry_rows]
    entry_p    = [r["yes_ask"] for r in entry_rows]

    exit_lag_rows  = [r for r in rows if r["event"] == "exit_lag_closed"]
    exit_stop_rows = [r for r in rows if r["event"] == "exit_stopped"]

    def _mid_at(t):
        try:
            return float(yes_mid[ts.index(t)])
        except (ValueError, IndexError):
            return 0.5

    exit_lag_ts  = [datetime.datetime.fromtimestamp(r["ts_ns"]/1e9) for r in exit_lag_rows]
    exit_lag_p   = [_mid_at(datetime.datetime.fromtimestamp(r["ts_ns"]/1e9)) for r in exit_lag_rows]
    exit_stop_ts = [datetime.datetime.fromtimestamp(r["ts_ns"]/1e9) for r in exit_stop_rows]
    exit_stop_p  = [_mid_at(datetime.datetime.fromtimestamp(r["ts_ns"]/1e9)) for r in exit_stop_rows]

    pnl_rows  = compute_pnl(rows)
    cum_pnl_x = [datetime.datetime.fromtimestamp(p["exit_ts"]) for p in pnl_rows]
    cum_pnl_y = np.cumsum([p["pnl"] for p in pnl_rows]) if pnl_rows else []

    abstentions = defaultdict(int)
    for r in rows:
        abstentions[r.get("abstention_reason") or r["event"]] += 1

    # -- Panel 1: YES price + model --
    ax = axes[0]
    ax.fill_between(ts, yes_bid, yes_ask, alpha=0.15, color="steelblue", label="bid/ask spread")
    ax.plot(ts, yes_mid, color="steelblue", lw=1.2, label="YES mid")
    ax.plot(ts, q_set,   color="tomato",    lw=1.5, linestyle="--", label="q_settled (model)")
    ax.plot(ts, q_pred,  color="orange",    lw=1.0, linestyle=":",  label="q_predicted (sanity)")
    if entry_ts:
        ax.scatter(entry_ts, entry_p, marker="^", color="green", s=80, zorder=5, label="entry")
    if exit_lag_ts:
        ax.scatter(exit_lag_ts, exit_lag_p, marker="v", color="lime", s=80, zorder=5, label="exit_lag_closed")
    if exit_stop_ts:
        ax.scatter(exit_stop_ts, exit_stop_p, marker="x", color="red", s=80, zorder=5, label="exit_stopped")
    ax.set_ylabel("Probability")
    ax.set_title("Kalshi YES Price vs Model Predictions")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)

    # -- Panel 2: Edge magnitude --
    ax = axes[1]
    ax.plot(ts, edge_mag, color="purple", lw=1.0, label="edge_magnitude")
    colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db"]
    for (floor, frac), col in zip(KELLY_TIERS, colors):
        ax.axhline(floor, color=col, lw=0.8, linestyle="--",
                   label=f"tier floor {floor:.2f} ({frac*100:.0f}% wallet)")
    ax.set_ylabel("Edge (probability pts)")
    ax.set_title("Net Edge Magnitude and Kelly Tier Thresholds")
    ax.legend(fontsize=7, loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # -- Panel 3: Model quality --
    ax = axes[2]
    ax.plot(ts, r2,    color="dodgerblue", lw=1.5, label="R2 hld")
    ax.axhline(0.20, color="red",   lw=0.8, linestyle="--", label="R2=0.20 abstain floor")
    ax.axhline(0.50, color="green", lw=0.8, linestyle="--", label="R2=0.50 healthy")
    ax3b.plot(ts, lag_s, color="darkorange", lw=1.0, linestyle=":", label="est. lag (s)")
    ax.set_ylabel("R2 (hld)")
    ax3b.set_ylabel("Estimated lag (s)", color="darkorange")
    ax.set_title("Model Quality: R2_hld and Estimated Kalshi Lag")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(alpha=0.3)

    # -- Panel 4: Cumulative P&L + abstention pie --
    ax = axes[3]
    if len(cum_pnl_y):
        ax.plot(cum_pnl_x, cum_pnl_y, color="green", lw=2, label="cumulative P&L")
        ax.axhline(0, color="gray", lw=0.8)
        ax.fill_between(cum_pnl_x, cum_pnl_y, 0,
                        where=[y >= 0 for y in cum_pnl_y], alpha=0.2, color="green")
        ax.fill_between(cum_pnl_x, cum_pnl_y, 0,
                        where=[y < 0 for y in cum_pnl_y], alpha=0.2, color="red")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.set_title("Simulated Cumulative P&L (after fees)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.grid(alpha=0.3)
    else:
        if abstentions:
            labels_pie = list(abstentions.keys())
            sizes_pie  = [abstentions[k] for k in labels_pie]
            ax.pie(sizes_pie, labels=labels_pie, autopct="%1.0f%%", startangle=90)
            ax.set_title("Tick Breakdown (no trades yet - model still warming up)")


def plot(rows: list, save_path=None) -> None:
    if not HAS_MPL:
        print("matplotlib not installed -- skipping charts. pip install matplotlib")
        return
    if not rows:
        return

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.suptitle("Kalshi BTC Lag-Arb Dry Run", fontsize=14, fontweight="bold")
    ax3b = axes[2].twinx()

    _draw(axes, ax3b, rows)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


def live_plot(log_path: Path, interval_s: int = 10) -> None:
    if not HAS_MPL:
        print("matplotlib not installed -- pip install matplotlib")
        return

    import matplotlib.animation as animation

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.suptitle("Kalshi BTC Lag-Arb  [LIVE]", fontsize=14, fontweight="bold")
    ax3b = axes[2].twinx()

    def _update(_frame):
        if not log_path.exists():
            return
        rows = load_log(log_path)
        if not rows:
            return
        _draw(axes, ax3b, rows)
        import datetime
        fig.suptitle(
            f"Kalshi BTC Lag-Arb  [LIVE]  --  {len(rows)} ticks  "
            f"last {datetime.datetime.now().strftime('%H:%M:%S')}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

    _update(0)
    ani = animation.FuncAnimation(
        fig, _update,
        interval=interval_s * 1000,
        cache_frame_data=False,
    )
    plt.show()
    _ = ani


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze a Kalshi dry-run JSONL log")
    parser.add_argument("--run",      default=None,
                        help="Path to a data/<run> folder (popup if omitted)")
    parser.add_argument("--asset",    default="BTC",
                        help="Which asset's decisions file to load (default: BTC)")
    parser.add_argument("--save",     default=None, help="Save chart to this PNG path")
    parser.add_argument("--no-plot",  action="store_true")
    parser.add_argument("--live",     action="store_true",
                        help="Live-updating plot - refreshes while the bot runs")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds for --live (default 10)")
    args = parser.parse_args()

    run_dir  = pick_run_folder(cli_arg=args.run, title="Select run to analyze")
    log_path = run_dir / f"decisions_{args.asset.upper()}.jsonl"

    # Fallback: legacy single-asset files without asset suffix
    if not log_path.exists():
        legacy = run_dir / "decisions.jsonl"
        if legacy.exists():
            log_path = legacy

    if args.live:
        if not log_path.exists():
            print(f"Waiting for {log_path} to appear...", flush=True)
        live_plot(log_path, interval_s=args.interval)
        return

    if not log_path.exists():
        sys.exit(f"ERROR: log file not found: {log_path}\n"
                 "Start the bot first: python scripts/run/run_kalshi_bot.py")

    rows = load_log(log_path)
    if not rows:
        sys.exit("Log file is empty - bot may not have started yet.")

    print_summary(rows)
    if not args.no_plot:
        plot(rows, save_path=args.save)


if __name__ == "__main__":
    main()
