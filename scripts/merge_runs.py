"""
merge_runs.py — Merge tick CSVs from multiple dry runs into a single data folder.

Pools ticks from two or more run folders, groups by window_ticker, sorts by ts_ns,
and deduplicates. The output is a new data/merged_<name>/ folder with one
ticks_<ASSET>.csv per asset, ready for train_model.py.

Key correctness guarantee:
  - window_ticker is the ground truth for 15-min window identity. Ticks from the
    same window across different runs (even with a mid-window gap) are merged into
    one window group, sorted by ts_ns.
  - tau_s is preserved as-recorded (it was the real seconds-to-close at collection
    time). The training code uses entry_tau_s - r["tau_s"] >= 30 to skip the cold
    lag period at the START of each run's contribution to a window — this still
    works correctly on merged data because the warmup is measured from the first
    tick in the merged window group.
  - Ticks within the same window from different runs that have a time gap will
    produce a gap in the ts_ns sequence. The target builder's timestamp-tolerance
    check will naturally reject training samples whose future target falls across
    the gap — no special handling needed.

Usage:
  # Interactive (select 2+ run folders via popup)
  python scripts/merge_runs.py

  # Specify runs directly
  python scripts/merge_runs.py --runs data/run1 data/run2 data/run3

  # Custom output name
  python scripts/merge_runs.py --runs data/run1 data/run2 --name my_merged

  # Specific assets only
  python scripts/merge_runs.py --assets BTC ETH
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
import datetime

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

DATA_DIR = _REPO / "data"

TICK_COLUMNS = [
    "ts_ns", "tau_s", "btc_microprice", "btc_bid", "btc_ask",
    "cb_microprice", "bn_microprice", "yes_bid", "yes_ask", "yes_mid",
    "floor_strike", "window_ticker",
]


def pick_runs_interactive() -> list[Path]:
    """List available run folders and let user select multiple by number."""
    runs = sorted(
        [d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith("merged_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        sys.exit(f"No run folders found in {DATA_DIR}")

    print("Available runs:")
    for i, r in enumerate(runs):
        assets = sorted(set(
            p.stem.replace("ticks_", "")
            for p in r.glob("ticks_*.csv")
        ))
        print(f"  [{i}] {r.name}  ({', '.join(assets)})")

    raw = input("Enter numbers to merge (space-separated, e.g. 0 1 2): ").strip()
    try:
        idxs = [int(x) for x in raw.split()]
    except ValueError:
        sys.exit("Invalid input.")
    if len(idxs) < 2:
        sys.exit("Need at least 2 run folders to merge.")
    return [runs[i] for i in idxs]


def load_ticks_from_run(run_dir: Path, asset: str) -> list[dict]:
    """Load all tick rows from a single run for one asset."""
    path = run_dir / f"ticks_{asset}.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "ts_ns":          int(r["ts_ns"]),
                    "tau_s":          r["tau_s"],
                    "btc_microprice": r["btc_microprice"],
                    "btc_bid":        r["btc_bid"],
                    "btc_ask":        r["btc_ask"],
                    "cb_microprice":  r["cb_microprice"],
                    "bn_microprice":  r["bn_microprice"],
                    "yes_bid":        r["yes_bid"],
                    "yes_ask":        r["yes_ask"],
                    "yes_mid":        r["yes_mid"],
                    "floor_strike":   r["floor_strike"],
                    "window_ticker":  r["window_ticker"],
                })
            except (KeyError, ValueError):
                pass
    return rows


def merge_asset(run_dirs: list[Path], asset: str) -> tuple[list[dict], dict]:
    """
    Merge tick rows for one asset across all run_dirs.
    Returns (merged_rows_sorted, stats).
    """
    all_rows: list[dict] = []
    rows_per_run = {}
    for run_dir in run_dirs:
        loaded = load_ticks_from_run(run_dir, asset)
        rows_per_run[run_dir.name] = len(loaded)
        all_rows.extend(loaded)

    if not all_rows:
        return [], {"rows_per_run": rows_per_run, "total_raw": 0,
                    "duplicates_removed": 0, "windows": 0, "final": 0}

    total_raw = len(all_rows)

    # Sort by ts_ns globally
    all_rows.sort(key=lambda r: r["ts_ns"])

    # Deduplicate on ts_ns (keep first occurrence)
    seen_ts: set[int] = set()
    deduped: list[dict] = []
    for r in all_rows:
        if r["ts_ns"] not in seen_ts:
            seen_ts.add(r["ts_ns"])
            deduped.append(r)
    duplicates_removed = total_raw - len(deduped)

    # Group by window_ticker to report stats
    windows: dict[str, list[dict]] = defaultdict(list)
    for r in deduped:
        windows[r["window_ticker"]].append(r)

    # Validate: for each window, check for gaps and partial coverage
    gap_warnings = []
    for ticker, w_rows in sorted(windows.items()):
        w_rows.sort(key=lambda r: r["ts_ns"])
        entry_tau  = float(w_rows[0]["tau_s"])
        exit_tau   = float(w_rows[-1]["tau_s"])
        n_ticks    = len(w_rows)

        # Detect gaps: any inter-tick interval > 5s is a gap (data was offline)
        gaps = []
        for i in range(1, len(w_rows)):
            gap_s = (w_rows[i]["ts_ns"] - w_rows[i-1]["ts_ns"]) / 1e9
            if gap_s > 5.0:
                gaps.append((float(w_rows[i-1]["tau_s"]), float(w_rows[i]["tau_s"]), gap_s))

        if gaps:
            gap_strs = [f"  {g[0]:.0f}s->{g[1]:.0f}s (gap={g[2]:.0f}s)" for g in gaps]
            gap_warnings.append(
                f"  {ticker}: {n_ticks} ticks, tau {entry_tau:.0f}->{exit_tau:.0f}s, "
                f"{len(gaps)} gap(s):\n" + "\n".join(gap_strs)
            )

    stats = {
        "rows_per_run":       rows_per_run,
        "total_raw":          total_raw,
        "duplicates_removed": duplicates_removed,
        "windows":            len(windows),
        "final":              len(deduped),
        "gap_warnings":       gap_warnings,
    }
    return deduped, stats


def write_ticks(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TICK_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in TICK_COLUMNS})


def detect_assets(run_dirs: list[Path]) -> list[str]:
    """Return sorted list of assets present in any of the run folders."""
    assets: set[str] = set()
    for d in run_dirs:
        for p in d.glob("ticks_*.csv"):
            assets.add(p.stem.replace("ticks_", ""))
    return sorted(assets)


def main():
    parser = argparse.ArgumentParser(
        description="Merge tick CSVs from multiple dry runs into one data folder")
    parser.add_argument("--runs",   nargs="+", type=str, default=None,
                        help="Paths to run folders to merge (interactive if omitted)")
    parser.add_argument("--name",   type=str, default=None,
                        help="Output folder name under data/ (auto-generated if omitted)")
    parser.add_argument("--assets", nargs="+", type=str, default=None,
                        help="Assets to merge (default: all assets found in any run)")
    args = parser.parse_args()

    # --- Resolve run folders ---
    if args.runs:
        run_dirs = []
        for r in args.runs:
            p = Path(r)
            if not p.is_absolute():
                p = _REPO / p
            if not p.exists():
                sys.exit(f"Run folder not found: {p}")
            run_dirs.append(p)
        if len(run_dirs) < 2:
            sys.exit("Need at least 2 run folders to merge.")
    else:
        run_dirs = pick_runs_interactive()

    print(f"\nMerging {len(run_dirs)} run(s):")
    for d in run_dirs:
        print(f"  {d.name}")

    # --- Resolve assets ---
    assets = [a.upper() for a in args.assets] if args.assets else detect_assets(run_dirs)
    if not assets:
        sys.exit("No ticks_<ASSET>.csv files found in the specified runs.")
    print(f"Assets: {', '.join(assets)}")

    # --- Output folder ---
    if args.name:
        out_name = args.name if args.name.startswith("merged_") else f"merged_{args.name}"
    else:
        ts_str   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_tags = "_".join(d.name[:10] for d in run_dirs)  # first 10 chars of each
        out_name = f"merged_{ts_str}_{run_tags}"
    out_dir = DATA_DIR / out_name
    if out_dir.exists():
        sys.exit(f"Output folder already exists: {out_dir}\nChoose a different --name.")

    print(f"\nOutput: {out_dir}")

    # --- Merge each asset ---
    any_data = False
    for asset in assets:
        rows, stats = merge_asset(run_dirs, asset)

        if not rows:
            print(f"\n  [{asset}] no data found in any run — skipping")
            continue

        any_data = True
        out_path = out_dir / f"ticks_{asset}.csv"
        write_ticks(rows, out_path)

        print(f"\n  [{asset}]")
        for run_name, n in stats["rows_per_run"].items():
            print(f"    {run_name}: {n:,} ticks")
        print(f"    Duplicates removed: {stats['duplicates_removed']}")
        print(f"    Windows (unique):   {stats['windows']}")
        print(f"    Final ticks:        {stats['final']:,}")
        print(f"    Saved -> {out_path}")

        if stats.get("gap_warnings"):
            print(f"\n    WARNING: {len(stats['gap_warnings'])} window(s) have data gaps.")
            print("    These gaps are handled correctly by train_model.py (targets across")
            print("    a gap are rejected by the timestamp-tolerance check). Details:")
            for w in stats["gap_warnings"]:
                print(w)

    if not any_data:
        out_dir.rmdir() if out_dir.exists() else None
        sys.exit("No data found for any asset in the specified runs.")

    print(f"\nDone. To train on merged data:")
    print(f"  python scripts/train_model.py --run {out_dir}")


if __name__ == "__main__":
    main()
