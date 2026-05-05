"""
stats.py — Post-run analysis of bot performance from JSONL logs.

Usage:
    python3 stats.py                        # analyse logs/scans.jsonl + trades.jsonl
    python3 stats.py --log-dir /path/to/logs

Outputs:
    - Signal frequency by asset and mode
    - Entry condition hit-rate (p_jj >= threshold, delta >= epsilon)
    - Trade summary: count, avg size, avg expected return
    - Wallet trajectory over time
    - Window entry timing (how many seconds left when signal fired)
    - p_jj distribution across all scans
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def fmt(n, decimals=2):
    return f"{n:.{decimals}f}"


def pct(n):
    return f"{n*100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="Bot performance stats")
    parser.add_argument("--log-dir", default="logs", help="Directory containing JSONL logs")
    args = parser.parse_args()

    scans  = load_jsonl(os.path.join(args.log_dir, "scans.jsonl"))
    trades = load_jsonl(os.path.join(args.log_dir, "trades.jsonl"))
    wallet = load_jsonl(os.path.join(args.log_dir, "wallet.jsonl"))

    if not scans:
        print(f"No scan data found in {args.log_dir}/scans.jsonl")
        sys.exit(1)

    print("=" * 65)
    print("POLYMARKET BOT — SESSION STATS")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------
    total_scans  = len(scans)
    total_trades = len([t for t in trades if t.get("event") == "signal"])
    dry_run      = any(t.get("dry_run") for t in trades)
    assets       = sorted({s["asset"] for s in scans})
    ts_start     = min(s["ts"] for s in scans)
    ts_end       = max(s["ts"] for s in scans)
    duration_h   = (ts_end - ts_start) / 3600

    print(f"\nMode:         {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Duration:     {fmt(duration_h, 1)}h")
    print(f"Total scans:  {total_scans:,}  ({total_scans // max(1, len(assets))} loops)")
    print(f"Assets:       {', '.join(assets)}")
    print(f"Signals:      {total_trades}")
    print(f"Signal rate:  {pct(total_trades / max(1, total_scans // max(1, len(assets))))}")

    # ------------------------------------------------------------------
    # Scan health: p_jj distribution
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("p_jj DISTRIBUTION (all scans with obs >= 10)")
    warmup_scans = [s for s in scans if s.get("obs", 0) >= 10]
    if warmup_scans:
        pjj_vals = [s["p_jj"] for s in warmup_scans]
        buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.87), (0.87, 0.95), (0.95, 1.01)]
        for lo, hi in buckets:
            count = sum(1 for v in pjj_vals if lo <= v < hi)
            bar = "#" * (count * 40 // max(1, len(pjj_vals)))
            print(f"  [{lo:.2f},{hi:.2f})  {bar:<40} {count:>6}  ({pct(count/max(1,len(pjj_vals)))})")
    else:
        print("  No post-warmup scans yet.")

    # ------------------------------------------------------------------
    # Entry condition breakdown
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("ENTRY CONDITION BREAKDOWN (post-warmup scans)")
    if warmup_scans:
        p_jj_min = warmup_scans[0].get("p_jj_min", 0.87)
        epsilon  = warmup_scans[0].get("epsilon", 0.04)
        in_mode  = [s for s in warmup_scans if s.get("mode") is not None]
        pjj_pass = [s for s in in_mode if s["p_jj"] >= p_jj_min]
        both     = [s for s in pjj_pass if s["delta"] >= epsilon]
        print(f"  In valid mode:        {len(in_mode):>7,}  ({pct(len(in_mode)/max(1,len(warmup_scans)))})")
        print(f"  p_jj >= {p_jj_min}:      {len(pjj_pass):>7,}  ({pct(len(pjj_pass)/max(1,len(in_mode)))} of in-mode)")
        print(f"  + delta >= {epsilon}:    {len(both):>7,}  ({pct(len(both)/max(1,len(pjj_pass)))} of p_jj-pass)")
        print(f"  Signals fired:        {total_trades:>7,}")

    # ------------------------------------------------------------------
    # Per-asset signal breakdown
    # ------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("SIGNALS BY ASSET")
    print(f"  {'Asset':<12} {'Scans':>7} {'Signals':>8} {'Hit%':>7} {'AvgDelta':>10} {'AvgPjj':>8}")
    by_asset = defaultdict(list)
    for s in warmup_scans:
        by_asset[s["asset"]].append(s)
    for asset in sorted(by_asset):
        rows = by_asset[asset]
        sigs = [r for r in rows if r.get("signal")]
        avg_delta = sum(r["delta"] for r in rows) / max(1, len(rows))
        avg_pjj   = sum(r["p_jj"] for r in rows) / max(1, len(rows))
        print(f"  {asset:<12} {len(rows):>7,} {len(sigs):>8,} {pct(len(sigs)/max(1,len(rows))):>7} "
              f"{avg_delta:>+10.4f} {avg_pjj:>8.4f}")

    # ------------------------------------------------------------------
    # Trade details
    # ------------------------------------------------------------------
    signal_trades = [t for t in trades if t.get("event") == "signal"]
    if signal_trades:
        print("\n" + "-" * 65)
        print("TRADE DETAILS")
        sizes    = [t["size_usd"] for t in signal_trades]
        returns  = [t["expected_return"] for t in signal_trades]
        deltas   = [t["delta"] for t in signal_trades]
        secs_rem = [t["secs_left"] for t in signal_trades]
        modes    = [t.get("mode") for t in signal_trades]

        print(f"  Total trades:          {len(signal_trades)}")
        print(f"  Mode 1 (hi-conf):      {modes.count(1)}")
        print(f"  Mode 2 (discount):     {modes.count(2)}")
        print(f"  Avg size:              ${sum(sizes)/len(sizes):.2f}")
        print(f"  Avg expected return:   {pct(sum(returns)/len(returns))}")
        print(f"  Avg delta:             {sum(deltas)/len(deltas):+.4f}")
        print(f"  Avg secs left in win:  {sum(secs_rem)/len(secs_rem):.0f}s")
        print(f"  Total capital at risk: ${sum(sizes):.2f}")

        print(f"\n  {'Asset':<12} {'#':>4} {'AvgQ':>7} {'AvgDelta':>10} {'AvgExpR':>9} {'AvgSecs':>9}")
        by_asset_t = defaultdict(list)
        for t in signal_trades:
            by_asset_t[t["asset"]].append(t)
        for asset in sorted(by_asset_t):
            ts = by_asset_t[asset]
            print(f"  {asset:<12} {len(ts):>4} "
                  f"{sum(t['q'] for t in ts)/len(ts):>7.3f} "
                  f"{sum(t['delta'] for t in ts)/len(ts):>+10.4f} "
                  f"{pct(sum(t['expected_return'] for t in ts)/len(ts)):>9} "
                  f"{sum(t['secs_left'] for t in ts)/len(ts):>9.0f}s")

    # ------------------------------------------------------------------
    # Wallet trajectory
    # ------------------------------------------------------------------
    if wallet:
        print("\n" + "-" * 65)
        print("WALLET TRAJECTORY")
        first_bal = wallet[0]["wallet_usd"]
        last_bal  = wallet[-1]["wallet_usd"]
        min_bal   = min(w["wallet_usd"] for w in wallet)
        max_bal   = max(w["wallet_usd"] for w in wallet)
        print(f"  Start:  ${first_bal:.2f}")
        print(f"  End:    ${last_bal:.2f}")
        print(f"  Min:    ${min_bal:.2f}")
        print(f"  Max:    ${max_bal:.2f}")
        pnl = last_bal - first_bal
        print(f"  PnL:    ${pnl:+.2f}  ({pct(pnl/max(1,first_bal))})")

    # ------------------------------------------------------------------
    # Order outcomes (closed orders)
    # ------------------------------------------------------------------
    closed = [t for t in trades if t.get("event") == "order_closed"]
    if closed:
        print("\n" + "-" * 65)
        print("ORDER OUTCOMES")
        status_counts = defaultdict(int)
        for t in closed:
            status_counts[t["status"]] += 1
        for status, count in sorted(status_counts.items()):
            print(f"  {status:<15} {count}")

    print("\n" + "=" * 65)
    print("Run 'python3 stats.py' after each session to see performance.")
    print()


if __name__ == "__main__":
    main()
