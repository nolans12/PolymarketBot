"""
tick_logger.py — Writes raw ticks to parquet for backtesting & training.

Rolls over to a new file every ROLL_EVERY_ROWS so each output file is
self-contained (has a valid footer). If the process is killed, only the
current in-flight chunk is lost — everything previously rolled over is
intact and readable.

Output layout for path = data/<run>/ticks_BTC.parquet:
    data/<run>/ticks_BTC.parquet              <- finalized first chunk (or single chunk)
    data/<run>/ticks_BTC.0001.parquet         <- finalized chunk 1
    data/<run>/ticks_BTC.0002.parquet         <- finalized chunk 2
    ...

The tick loader (scripts/analysis/tick_loader.py) reads all chunks matching
the pattern and concatenates them.
"""

import threading
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

ROLL_EVERY_ROWS  = 60_000   # ~100 min at 10Hz — each finalized chunk ~10 MB
FLUSH_EVERY_ROWS = 5_000    # write to disk (but keep writer open) at this cadence

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
    pa.field("yes_bid_size",   pa.float32()),
    pa.field("yes_ask_size",   pa.float32()),
    pa.field("yes_book_top10", _BOOK_LEVEL_TYPE),
    pa.field("no_book_top10",  _BOOK_LEVEL_TYPE),
    pa.field("floor_strike",   pa.float64()),
    pa.field("window_ticker",  pa.string()),
])


class TickLogger:
    def __init__(self, path: Path):
        if str(path).endswith(".csv"):
            path = path.with_suffix(".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._base_path = path                  # e.g. data/<run>/ticks_BTC.parquet
        self._lock      = threading.Lock()
        self._rows: list[dict] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._cur_path: Optional[Path] = None
        self._cur_rows_written: int = 0
        self._chunk_idx: int = 0

    def _next_path(self) -> Path:
        """Pick the next chunk path that doesn't already exist on disk."""
        # If no chunks yet, use the base path. After that, .0001, .0002, etc.
        while True:
            if self._chunk_idx == 0:
                candidate = self._base_path
            else:
                stem = self._base_path.with_suffix("")  # strip .parquet
                candidate = stem.with_name(f"{stem.name}.{self._chunk_idx:04d}.parquet")
            if not candidate.exists():
                return candidate
            self._chunk_idx += 1

    def _open_writer(self) -> pq.ParquetWriter:
        self._cur_path = self._next_path()
        self._writer = pq.ParquetWriter(
            self._cur_path, SCHEMA,
            compression="snappy",
            use_dictionary=True,
        )
        self._cur_rows_written = 0
        self._chunk_idx += 1
        return self._writer

    def _ensure_writer(self) -> pq.ParquetWriter:
        if self._writer is None:
            return self._open_writer()
        return self._writer

    def _maybe_roll(self) -> None:
        """If the current writer has accumulated enough rows, close it and start a new file."""
        if self._writer is not None and self._cur_rows_written >= ROLL_EVERY_ROWS:
            self._writer.close()
            self._writer = None

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
        yes_ask_sz = no_levels[0][1]  if no_levels  else 0.0

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
        self._cur_rows_written += len(self._rows)
        self._rows.clear()
        self._maybe_roll()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
