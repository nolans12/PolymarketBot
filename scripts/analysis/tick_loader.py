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
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    pylist = table.to_pylist()
    windows: dict[str, list[dict]] = defaultdict(list)
    for r in pylist:
        try:
            row = _to_canonical(r)
        except (KeyError, TypeError, ValueError):
            continue
        windows[row["window_ticker"]].append(row)
    for rows in windows.values():
        rows.sort(key=lambda r: r["ts_ns"])
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
