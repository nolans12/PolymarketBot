"""
tick_logger.py — Writes raw ticks to parquet for backtesting & training.

One row per sampler tick. Buffers rows in memory and flushes to parquet in
chunks (default 5000 rows ~= 8 min at 10Hz). On close, flushes remainder.

The parquet file is written as multiple row groups via append-by-rewrite
(pyarrow doesn't support true append mode without ParquetWriter). For
simplicity we use ParquetWriter with snappy compression — single file,
multiple row groups, streaming.

Schema includes top-10 book depth on each side as native list-of-struct
columns (no JSON parsing needed at read time).
"""

import threading
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

FLUSH_EVERY_ROWS = 5000

# Schema for parquet output
_BOOK_LEVEL_TYPE = pa.list_(pa.struct([
    pa.field("price", pa.float64()),
    pa.field("size",  pa.float64()),
]))

SCHEMA = pa.schema([
    pa.field("ts_ns",          pa.int64()),
    pa.field("tau_s",          pa.float32()),
    pa.field("btc_microprice", pa.float64()),
    pa.field("btc_bid",        pa.float64()),
    pa.field("btc_ask",        pa.float64()),
    pa.field("cb_microprice",  pa.float64()),
    pa.field("bn_microprice",  pa.float64()),
    pa.field("yes_bid",        pa.float32()),
    pa.field("yes_ask",        pa.float32()),
    pa.field("yes_mid",        pa.float32()),
    pa.field("yes_bid_size",   pa.float32()),     # size at top YES bid
    pa.field("yes_ask_size",   pa.float32()),     # size at top NO bid (= YES ask)
    pa.field("yes_book_top10", _BOOK_LEVEL_TYPE), # top 10 YES bid levels
    pa.field("no_book_top10",  _BOOK_LEVEL_TYPE), # top 10 NO bid levels
    pa.field("floor_strike",   pa.float64()),
    pa.field("window_ticker",  pa.string()),
])


class TickLogger:
    def __init__(self, path: Path):
        # If caller passed a .csv path (legacy), redirect to .parquet
        if str(path).endswith(".csv"):
            path = path.with_suffix(".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path  = path
        self._lock  = threading.Lock()
        self._rows: list[dict] = []
        self._writer: Optional[pq.ParquetWriter] = None

    def _ensure_writer(self) -> pq.ParquetWriter:
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._path, SCHEMA,
                compression="snappy",
                use_dictionary=True,
            )
        return self._writer

    def log(self, ts_ns: int, tau_s: float,
            btc_microprice: float, btc_bid: float, btc_ask: float,
            cb_microprice: float, bn_microprice: float,
            yes_bid: float, yes_ask: float, yes_mid: float,
            floor_strike: float, window_ticker: str,
            yes_book_top10: list[tuple[float, float]] | None = None,
            no_book_top10:  list[tuple[float, float]] | None = None) -> None:
        yes_levels = yes_book_top10 or []
        no_levels  = no_book_top10  or []
        yes_bid_sz = yes_levels[0][1] if yes_levels else 0.0
        # YES ask = 1 - top NO bid. ask size = size resting at that NO bid level.
        yes_ask_sz = no_levels[0][1] if no_levels else 0.0

        row = {
            "ts_ns":          int(ts_ns),
            "tau_s":          float(tau_s),
            "btc_microprice": float(btc_microprice),
            "btc_bid":        float(btc_bid),
            "btc_ask":        float(btc_ask),
            "cb_microprice":  float(cb_microprice),
            "bn_microprice":  float(bn_microprice),
            "yes_bid":        float(yes_bid),
            "yes_ask":        float(yes_ask),
            "yes_mid":        float(yes_mid),
            "yes_bid_size":   float(yes_bid_sz),
            "yes_ask_size":   float(yes_ask_sz),
            "yes_book_top10": [{"price": float(p), "size": float(s)} for p, s in yes_levels],
            "no_book_top10":  [{"price": float(p), "size": float(s)} for p, s in no_levels],
            "floor_strike":   float(floor_strike),
            "window_ticker":  str(window_ticker),
        }
        with self._lock:
            self._rows.append(row)
            if len(self._rows) >= FLUSH_EVERY_ROWS:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._rows:
            return
        table  = pa.Table.from_pylist(self._rows, schema=SCHEMA)
        writer = self._ensure_writer()
        writer.write_table(table)
        self._rows.clear()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
