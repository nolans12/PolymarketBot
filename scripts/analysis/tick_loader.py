"""
tick_loader.py — Unified tick loader for parquet (new) and csv (legacy).

Returns dict[window_ticker -> list[dict]] in the canonical schema used by
all training/analysis scripts. The new parquet format includes top-10 book
depth on each side; the legacy CSV format does not (depth fields are 0).
"""

from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _to_canonical(r: dict) -> dict:
    """Coerce a raw row (csv str or parquet typed) to the canonical schema
    used by feature builders. Always includes depth fields (zero if absent)."""
    yes_book = r.get("yes_book_top10") or []
    no_book  = r.get("no_book_top10")  or []
    # Parquet returns list-of-dict; CSV (legacy) won't have these columns.
    return {
        "ts_ns":          int(r["ts_ns"]),
        "tau_s":          float(r["tau_s"]),
        "btc_micro":      float(r["btc_microprice"]),
        "btc_bid":        float(r.get("btc_bid", 0) or 0),
        "btc_ask":        float(r.get("btc_ask", 0) or 0),
        "yes_bid":        float(r["yes_bid"]),
        "yes_ask":        float(r["yes_ask"]),
        "yes_mid":        float(r["yes_mid"]),
        "yes_bid_size":   float(r.get("yes_bid_size", 0) or 0),
        "yes_ask_size":   float(r.get("yes_ask_size", 0) or 0),
        "K":              float(r["floor_strike"]),
        "window_ticker":  str(r["window_ticker"]),
        "yes_book":       [(float(d["price"]), float(d["size"])) for d in yes_book],
        "no_book":        [(float(d["price"]), float(d["size"])) for d in no_book],
    }


def load_ticks(path: Path) -> dict[str, list[dict]]:
    """Load ticks from a single file. Auto-detects parquet vs csv."""
    path = Path(path)
    if not path.exists():
        # Try the other extension as a fallback
        alt = path.with_suffix(".parquet" if path.suffix == ".csv" else ".csv")
        if alt.exists():
            path = alt
        else:
            return {}

    if path.suffix == ".parquet":
        return _load_parquet(path)
    return _load_csv(path)


def _load_parquet(path: Path) -> dict[str, list[dict]]:
    import time
    import pyarrow.parquet as pq

    t0 = time.monotonic()
    print(f"  opening parquet...", flush=True)
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    print(f"  {total_rows:,} rows across {pf.num_row_groups} row groups; "
          f"streaming in batches...", flush=True)

    windows: dict[str, list[dict]] = defaultdict(list)
    seen = 0
    batch_size = 50_000
    last_report = t0

    # Stream by batches so we never hold the whole pylist in memory at once
    for batch in pf.iter_batches(batch_size=batch_size):
        # Pull columns once — vastly faster than per-row dict lookups
        ts_ns          = batch.column("ts_ns").to_pylist()
        tau_s          = batch.column("tau_s").to_pylist()
        btc_micro      = batch.column("btc_microprice").to_pylist()
        btc_bid        = batch.column("btc_bid").to_pylist()
        btc_ask        = batch.column("btc_ask").to_pylist()
        yes_bid        = batch.column("yes_bid").to_pylist()
        yes_ask        = batch.column("yes_ask").to_pylist()
        yes_mid        = batch.column("yes_mid").to_pylist()
        yes_bid_size   = batch.column("yes_bid_size").to_pylist()
        yes_ask_size   = batch.column("yes_ask_size").to_pylist()
        floor_strike   = batch.column("floor_strike").to_pylist()
        window_ticker  = batch.column("window_ticker").to_pylist()
        yes_book_top10 = batch.column("yes_book_top10").to_pylist()
        no_book_top10  = batch.column("no_book_top10").to_pylist()

        n = len(ts_ns)
        for i in range(n):
            try:
                yes_book = yes_book_top10[i] or []
                no_book  = no_book_top10[i]  or []
                row = {
                    "ts_ns":        int(ts_ns[i]),
                    "tau_s":        float(tau_s[i] or 0),
                    "btc_micro":    float(btc_micro[i] or 0),
                    "btc_bid":      float(btc_bid[i] or 0),
                    "btc_ask":      float(btc_ask[i] or 0),
                    "yes_bid":      float(yes_bid[i] or 0),
                    "yes_ask":      float(yes_ask[i] or 0),
                    "yes_mid":      float(yes_mid[i] or 0),
                    "yes_bid_size": float(yes_bid_size[i] or 0),
                    "yes_ask_size": float(yes_ask_size[i] or 0),
                    "K":            float(floor_strike[i] or 0),
                    "window_ticker": str(window_ticker[i]),
                    "yes_book":     [(float(d["price"]), float(d["size"])) for d in yes_book],
                    "no_book":      [(float(d["price"]), float(d["size"])) for d in no_book],
                }
            except (TypeError, ValueError, KeyError):
                continue
            windows[row["window_ticker"]].append(row)

        seen += n
        now = time.monotonic()
        if now - last_report >= 2.0:
            pct = 100 * seen / total_rows if total_rows else 0
            rate = seen / (now - t0) if now > t0 else 0
            print(f"    {seen:,}/{total_rows:,} ({pct:.0f}%)  "
                  f"{rate/1000:.0f}k rows/s", flush=True)
            last_report = now

    print(f"  loaded {seen:,} rows into {len(windows)} windows in "
          f"{time.monotonic()-t0:.1f}s; sorting...", flush=True)
    t_sort = time.monotonic()
    for rows in windows.values():
        rows.sort(key=lambda r: r["ts_ns"])
    print(f"  sorted in {time.monotonic()-t_sort:.1f}s", flush=True)
    return windows


def _load_csv(path: Path) -> dict[str, list[dict]]:
    import csv as _csv
    windows: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        for r in reader:
            try:
                row = _to_canonical(r)
            except (KeyError, ValueError):
                continue
            windows[row["window_ticker"]].append(row)
    for rows in windows.values():
        rows.sort(key=lambda r: r["ts_ns"])
    return windows


def find_ticks_path(run_dir: Path, asset: str) -> Path | None:
    """Return the ticks file path for an asset, preferring parquet over csv."""
    parquet = run_dir / f"ticks_{asset}.parquet"
    if parquet.exists():
        return parquet
    csv = run_dir / f"ticks_{asset}.csv"
    if csv.exists():
        return csv
    return None
