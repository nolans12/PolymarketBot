"""
watch_decisions.py — Live colored decision feed with per-asset edge status line.

Tails decisions_*.jsonl from the latest run folder. Prints meaningful events
(entries, exits, rejections) as they happen, and keeps a live status bar at the
bottom showing the current edge, position, and tau for each asset.

Usage:
  python scripts/watch_decisions.py            # latest run folder
  python scripts/watch_decisions.py --run data/2026-05-07_16-30-00_BTC_ETH
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ANSI
G    = "\033[92m"
R    = "\033[91m"
Y    = "\033[93m"
M    = "\033[95m"
C    = "\033[96m"
DIM  = "\033[2m"
RST  = "\033[0m"
BOLD = "\033[1m"
ERASE_LINE  = "\033[2K"
CURSOR_UP   = "\033[1A"
CURSOR_SAVE    = "\033[s"
CURSOR_RESTORE = "\033[u"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def latest_run() -> Path:
    runs = sorted(DATA_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        if r.is_dir():
            return r
    sys.exit("No run folders found in data/")


def edge_color(edge: float) -> str:
    if edge <= 0:
        return DIM
    if edge >= 0.06:
        return G
    if edge >= 0.03:
        return Y
    return C


def fmt_event(asset: str, row: dict) -> str | None:
    event  = row.get("event", "")
    reason = row.get("abstention_reason", "")
    side   = (row.get("favored_side") or "").upper()
    tau    = row.get("tau_s", 0)
    bid    = row.get("yes_bid", 0)
    ask    = row.get("yes_ask", 0)
    edge   = row.get("edge_magnitude", 0)
    tier   = row.get("tier", 0)
    ts     = time.strftime("%H:%M:%S")

    if event == "entry":
        color = G if side == "YES" else M
        size  = row.get("would_bet_usd", 0)
        return (
            f"{BOLD}{color}  ▲ BUY  {asset} {side:3s}{RST}"
            f"  bid={bid:.3f} ask={ask:.3f}"
            f"  edge={edge:.4f} tier={tier} ${size:.2f}"
            f"  tau={tau:.0f}s  {DIM}{ts}{RST}"
        )

    if event and event.startswith("exit"):
        exit_type = event.replace("exit_", "").upper().replace("_", " ")
        color = G if "LAG_CLOSED" in exit_type else (R if "STOPPED" in exit_type else Y)
        return (
            f"{BOLD}{color}  ▼ EXIT {asset} [{exit_type}]{RST}"
            f"  bid={bid:.3f} ask={ask:.3f}"
            f"  edge={edge:.4f}  tau={tau:.0f}s  {DIM}{ts}{RST}"
        )

    if event == "fallback_resolution":
        return (
            f"{Y}  ⏎ RESOLVE {asset}{RST}"
            f"  tau={tau:.0f}s  {DIM}{ts}{RST}"
        )

    if event == "abstain" and reason == "entry_order_rejected":
        attempted = row.get("would_bet_usd") or 0.0
        q_set     = row.get("q_settled")
        q_str     = f" q={q_set:.3f}" if q_set is not None else ""
        return (
            f"{R}  ✗ ORDER REJECTED {asset} {side}{RST}"
            f"  edge={edge:.4f}{q_str}  tried@{attempted:.2f}"
            f"  book={bid:.3f}/{ask:.3f}  {DIM}{ts}{RST}"
        )

    if event == "abstain" and reason == "entry_unfilled":
        attempted = row.get("would_bet_usd") or 0.0
        q_set     = row.get("q_settled")
        q_str     = f" q={q_set:.3f}" if q_set is not None else ""
        return (
            f"{Y}  ~ UNFILLED {asset} {side}{RST}"
            f"  edge={edge:.4f}{q_str}  tried@{attempted:.2f}"
            f"  book={bid:.3f}/{ask:.3f} — no liquidity, backing off 10s"
            f"  {DIM}{ts}{RST}"
        )

    return None


def fmt_status(asset_states: dict) -> str:
    parts = []
    for asset in sorted(asset_states):
        s = asset_states[asset]
        edge    = s.get("edge", 0.0)
        tau     = s.get("tau", 0.0)
        bid     = s.get("bid", 0.0)
        ask     = s.get("ask", 0.0)
        spot    = s.get("spot", 0.0)
        strike  = s.get("strike", 0.0)
        pos     = s.get("pos", "")
        ec      = edge_color(edge)
        pos_tag = f" {G}[{pos}]{RST}" if pos else f" {DIM}[flat]{RST}"
        if spot > 0 and strike > 0:
            delta = spot - strike
            dc    = G if delta >= 0 else R
            spot_part = f" {dc}spot=${spot:,.2f} (Δ{delta:+.2f}){RST}"
        else:
            spot_part = ""
        parts.append(
            f"{BOLD}{asset}{RST}{pos_tag}"
            f"{spot_part}"
            f" {ec}edge={edge:.4f}{RST}"
            f" {DIM}bid={bid:.2f} ask={ask:.2f} tau={tau:.0f}s{RST}"
        )
    ts = time.strftime("%H:%M:%S")
    return f"{DIM}─── {ts} ──{RST}  " + "   ".join(parts)


def tail_file(path: Path, pos: int) -> tuple[list[dict], int]:
    rows = []
    try:
        with open(path, "r") as f:
            f.seek(pos)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            pos = f.tell()
    except FileNotFoundError:
        pass
    return rows, pos


def read_last_tick(run_dir: Path, asset: str) -> dict | None:
    """Read the latest tick from parquet (preferred) or CSV (legacy).
    Returns {'spot': ..., 'strike': ...} or None."""
    parquet = run_dir / f"ticks_{asset}.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(parquet, columns=["btc_microprice", "floor_strike"])
            if t.num_rows == 0:
                return None
            return {
                "spot":   float(t.column("btc_microprice")[-1].as_py() or 0.0),
                "strike": float(t.column("floor_strike")[-1].as_py() or 0.0),
            }
        except Exception:
            return None

    csv = run_dir / f"ticks_{asset}.csv"
    if not csv.exists():
        return None
    try:
        with open(csv, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size < 2:
                return None
            chunk = min(4096, size)
            f.seek(size - chunk)
            tail = f.read().splitlines()
            if len(tail) < 2:
                return None
            last = tail[-1].decode("utf-8", errors="ignore")
            parts = last.split(",")
            if len(parts) < 12:
                return None
            return {
                "spot":   float(parts[2])  if parts[2]  else 0.0,
                "strike": float(parts[10]) if parts[10] else 0.0,
            }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run()
    print(f"{C}  Watching: {run_dir}{RST}")
    print(f"{DIM}  ▲ BUY (green)  ▼ EXIT (green=lag closed, red=stopped, yellow=other)  ✗ REJECTED (red){RST}")
    print()

    positions:    dict[Path, int]  = {}
    asset_states: dict[str, dict]  = {}
    status_lines  = 0   # how many lines the status bar currently occupies

    def clear_status():
        nonlocal status_lines
        for _ in range(status_lines):
            sys.stdout.write(f"{CURSOR_UP}{ERASE_LINE}")
        status_lines = 0

    def print_status():
        nonlocal status_lines
        if not asset_states:
            return
        line = fmt_status(asset_states)
        print(line, flush=True)
        status_lines = 1

    while True:
        # Discover new jsonl files
        for jf in sorted(run_dir.glob("decisions_*.jsonl")):
            if jf not in positions:
                positions[jf] = 0
                asset = jf.stem.replace("decisions_", "")
                asset_states[asset] = {}
                clear_status()
                print(f"{DIM}  + tracking {asset}{RST}", flush=True)
                print_status()

        # Refresh spot prices from ticks file (parquet preferred, csv fallback)
        for asset in list(asset_states):
            tick = read_last_tick(run_dir, asset)
            if tick:
                asset_states[asset]["spot"]   = tick["spot"]
                asset_states[asset]["strike"] = tick["strike"]

        # Process new rows
        events_this_tick = []
        for jf in list(positions):
            asset = jf.stem.replace("decisions_", "")
            rows, new_pos = tail_file(jf, positions[jf])
            positions[jf] = new_pos
            for row in rows:
                # Always update live state from every row
                edge = row.get("edge_magnitude", 0.0)
                if edge < -50:
                    edge = 0.0
                state = asset_states.setdefault(asset, {})
                state["edge"] = edge
                state["tau"]  = row.get("tau_s", 0.0)
                state["bid"]  = row.get("yes_bid", 0.0)
                state["ask"]  = row.get("yes_ask", 0.0)
                # Track open position from events
                ev = row.get("event", "")
                if ev == "entry":
                    side = row.get("favored_side") or ""
                    state["pos"] = f"{side.upper()}@{row.get('yes_ask', 0):.2f}"
                elif ev and (ev.startswith("exit") or ev == "fallback_resolution"):
                    state["pos"] = ""

                line = fmt_event(asset, row)
                if line:
                    events_this_tick.append(line)

        # Print events (clear status first, print event, redraw status)
        if events_this_tick:
            clear_status()
            for line in events_this_tick:
                print(line, flush=True)
            print_status()
        else:
            # Just refresh the status bar in place
            clear_status()
            print_status()

        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{DIM}  stopped{RST}")
