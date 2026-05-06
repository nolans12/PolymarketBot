"""
parquet_writer.py — Hive-partitioned Parquet writer for all logged tables.

Tables and partition schemes (CLAUDE.md §9.1):
  decisions/              dt=YYYY-MM-DD/asset=btc/h=HH.parquet
  binance_ticks/          dt=YYYY-MM-DD/asset=btc/h=HH.parquet
  polymarket_snapshots/   dt=YYYY-MM-DD/asset=btc/h=HH.parquet
  polymarket_trades/      dt=YYYY-MM-DD/asset=btc/h=HH.parquet
  chainlink_ticks/        dt=YYYY-MM-DD/asset=btc/h=HH.parquet
  window_outcomes/        dt=YYYY-MM-DD/asset=btc.parquet   (no h= partition)
  model_versions/         dt=YYYY-MM-DD/asset=btc.parquet   (no h= partition)

One ParquetWriter per (table, asset, hour). Writers are opened lazily on first
row and closed + rotated when the hour changes. All in-memory rows are flushed
to disk every PARQUET_FLUSH_S seconds by the background flush task.

Usage:
    writer = ParquetWriter(parquet_dir=Path("logs"))
    asyncio.create_task(writer.run())          # background flush loop
    writer.write_binance_tick(asset, row)
    writer.write_decision(asset, row)
    await writer.close()                       # flush + close all writers on shutdown
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from polybot.infra.config import PARQUET_DIR, PARQUET_FLUSH_S, ASSETS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyArrow schemas — one per table (CLAUDE.md §9.2 – §9.5)
# ---------------------------------------------------------------------------

DECISIONS_SCHEMA = pa.schema([
    ("ts_ns",                    pa.int64()),
    ("asset",                    pa.string()),
    ("window_ts",                pa.int32()),
    ("tau_s",                    pa.float32()),
    # Binance spot inputs
    ("S_mid",                    pa.float64()),
    ("microprice",               pa.float64()),
    ("spot_spread",              pa.float32()),
    ("spot_bid_size_l1",         pa.float32()),
    ("spot_ask_size_l1",         pa.float32()),
    # Coinbase spot inputs (L1 from ticker)
    ("cb_mid",                   pa.float64()),
    ("cb_microprice",            pa.float64()),
    # Strike
    ("K",                        pa.float64()),
    ("K_uncertain",              pa.bool_()),
    # Lagged Binance microprice features
    ("x_now_logKratio",          pa.float32()),
    ("x_15_logKratio",           pa.float32()),
    ("x_30_logKratio",           pa.float32()),
    ("x_45_logKratio",           pa.float32()),
    ("x_60_logKratio",           pa.float32()),
    ("x_90_logKratio",           pa.float32()),
    ("x_120_logKratio",          pa.float32()),
    # Lagged Coinbase microprice features
    ("cb_x_now_logKratio",       pa.float32()),
    ("cb_x_15_logKratio",        pa.float32()),
    ("cb_x_30_logKratio",        pa.float32()),
    ("cb_x_60_logKratio",        pa.float32()),
    # Microstructure
    ("ofi_l1",                   pa.float32()),
    ("ofi_l5_weighted",          pa.float32()),
    ("pm_book_imbalance",        pa.float32()),
    ("pm_trade_flow_30s",        pa.float32()),
    ("momentum_30s",             pa.float32()),
    ("momentum_60s",             pa.float32()),
    ("cross_asset_momentum_60s", pa.float32()),
    # Diagnostic vol
    ("sigma_per_sec_realized",   pa.float32()),
    # Polymarket book
    ("q_up_bid",                 pa.float32()),
    ("q_up_ask",                 pa.float32()),
    ("q_up_mid",                 pa.float32()),
    ("q_down_bid",               pa.float32()),
    ("q_down_ask",               pa.float32()),
    ("q_down_mid",               pa.float32()),
    ("poly_spread_up",           pa.float32()),
    ("poly_spread_down",         pa.float32()),
    ("poly_depth_up_l1",         pa.float32()),
    ("poly_depth_down_l1",       pa.float32()),
    # Model output
    ("model_version_id",         pa.string()),
    ("q_predicted",              pa.float32()),
    ("q_settled",                pa.float32()),
    ("q_predicted_minus_q_actual", pa.float32()),
    # Edge
    ("edge_up_raw",              pa.float32()),
    ("edge_down_raw",            pa.float32()),
    ("fee_up_per_dollar",        pa.float32()),
    ("fee_down_per_dollar",      pa.float32()),
    ("slippage_up_per_dollar",   pa.float32()),
    ("slippage_down_per_dollar", pa.float32()),
    ("edge_up_net",              pa.float32()),
    ("edge_down_net",            pa.float32()),
    ("edge_signed",              pa.float32()),
    ("edge_magnitude",           pa.float32()),
    ("favored_side",             pa.string()),
    # Action
    ("event",                    pa.string()),
    ("chosen_side",              pa.string()),
    ("tier",                     pa.int8()),
    ("would_bet_usd",            pa.float32()),
    ("bet_price",                pa.float32()),
    ("bet_payout_contracts",     pa.float32()),
    ("abstention_reason",        pa.string()),
    # Position state
    ("has_open_position",        pa.bool_()),
    ("position_side",            pa.string()),
    ("position_entry_tau",       pa.float32()),
    ("position_entry_price",     pa.float32()),
    ("position_edge_at_entry",   pa.float32()),
    # Feed staleness
    ("binance_stale_ms",         pa.int32()),
    ("coinbase_stale_ms",        pa.int32()),
    ("polymarket_stale_ms",      pa.int32()),
    ("chainlink_stale_ms",       pa.int32()),
    ("circuit_active",           pa.string()),
])

BINANCE_TICKS_SCHEMA = pa.schema([
    ("ts_ns",          pa.int64()),
    ("asset",          pa.string()),
    ("best_bid",       pa.float64()),
    ("best_ask",       pa.float64()),
    ("best_bid_size",  pa.float64()),
    ("best_ask_size",  pa.float64()),
    ("mid",            pa.float64()),
    ("microprice",     pa.float64()),
    ("spread",         pa.float32()),
    ("ofi_l1",         pa.float32()),
    ("ofi_l5",         pa.float32()),
    ("last_update_id", pa.int64()),
])

POLYMARKET_SNAPSHOTS_SCHEMA = pa.schema([
    ("ts_ns",          pa.int64()),
    ("asset",          pa.string()),
    ("token_id",       pa.string()),
    ("side",           pa.string()),   # 'up' or 'down'
    ("best_bid",       pa.float32()),
    ("best_ask",       pa.float32()),
    ("best_bid_size",  pa.float32()),
    ("best_ask_size",  pa.float32()),
    ("mid",            pa.float32()),
    ("spread",         pa.float32()),
    ("depth_l1",       pa.float32()),
    # Top-3 levels (price, size) for each side — enough for VWAP fill model
    ("bid_p1",         pa.float32()),  ("bid_s1", pa.float32()),
    ("bid_p2",         pa.float32()),  ("bid_s2", pa.float32()),
    ("bid_p3",         pa.float32()),  ("bid_s3", pa.float32()),
    ("ask_p1",         pa.float32()),  ("ask_s1", pa.float32()),
    ("ask_p2",         pa.float32()),  ("ask_s2", pa.float32()),
    ("ask_p3",         pa.float32()),  ("ask_s3", pa.float32()),
])

POLYMARKET_TRADES_SCHEMA = pa.schema([
    ("ts_ns",    pa.int64()),
    ("asset",    pa.string()),
    ("token_id", pa.string()),
    ("side",     pa.string()),
    ("price",    pa.float32()),
    ("size",     pa.float32()),
])

CHAINLINK_TICKS_SCHEMA = pa.schema([
    ("ts_ns",          pa.int64()),    # wall clock receipt
    ("asset",          pa.string()),
    ("oracle_ts_ms",   pa.int64()),    # Chainlink observation time
    ("price",          pa.float64()),
    ("window_ts",      pa.int32()),    # which window this tick belongs to
    ("is_K",           pa.bool_()),    # True if this tick was accepted as K
])

WINDOW_OUTCOMES_SCHEMA = pa.schema([
    ("asset",                  pa.string()),
    ("window_ts",              pa.int32()),
    ("close_ts",               pa.int32()),
    ("K",                      pa.float64()),
    ("close_price",            pa.float64()),
    ("outcome",                pa.int8()),      # 1=Up, 0=Down, -1=unknown
    ("n_decisions",            pa.int16()),
    ("n_entries_attempted",    pa.int16()),
    # Entry info (null if no entry)
    ("entry_tier",             pa.int8()),
    ("entry_tau_s",            pa.float32()),
    ("entry_side",             pa.string()),
    ("entry_size_usd",         pa.float32()),
    ("entry_price",            pa.float32()),
    ("entry_edge_signed",      pa.float32()),
    ("entry_payout_contracts", pa.float32()),
    ("entry_q_settled",        pa.float32()),
    ("entry_model_version_id", pa.string()),
    # Exit info — populated at replay time
    ("exit_reason",            pa.string()),
    ("exit_tau_s",             pa.float32()),
    ("exit_price",             pa.float32()),
    ("exit_holding_seconds",   pa.float32()),
    ("realized_pnl_usd",       pa.float32()),
    ("realized_pnl_per_dollar", pa.float32()),
])

MODEL_VERSIONS_SCHEMA = pa.schema([
    ("ts_ns",                       pa.int64()),
    ("asset",                       pa.string()),
    ("model_version_id",            pa.string()),
    ("n_train_samples",             pa.int32()),
    ("training_window_start_ns",    pa.int64()),
    ("ridge_alpha",                 pa.float32()),
    ("r2_in_sample",                pa.float32()),
    ("r2_cv_mean",                  pa.float32()),
    ("r2_held_out_30min",           pa.float32()),
    ("coef_alpha",                  pa.float32()),
    ("coef_x_now",                  pa.float32()),
    ("coef_x_15",                   pa.float32()),
    ("coef_x_30",                   pa.float32()),
    ("coef_x_45",                   pa.float32()),
    ("coef_x_60",                   pa.float32()),
    ("coef_x_90",                   pa.float32()),
    ("coef_x_120",                  pa.float32()),
    ("coef_tau",                    pa.float32()),
    ("coef_inv_sqrt_tau",           pa.float32()),
    ("coef_ofi_l1",                 pa.float32()),
    ("coef_ofi_l5_weighted",        pa.float32()),
    ("coef_pm_book_imbalance",      pa.float32()),
    ("coef_pm_trade_flow_30s",      pa.float32()),
    ("coef_momentum_30s",           pa.float32()),
    ("coef_momentum_60s",           pa.float32()),
    ("coef_cross_asset_momentum_60s", pa.float32()),
    ("estimated_lag_seconds",       pa.float32()),
    ("coef_delta_l2",               pa.float32()),
])

TABLE_SCHEMAS = {
    "decisions":             DECISIONS_SCHEMA,
    "binance_ticks":         BINANCE_TICKS_SCHEMA,
    "polymarket_snapshots":  POLYMARKET_SNAPSHOTS_SCHEMA,
    "polymarket_trades":     POLYMARKET_TRADES_SCHEMA,
    "chainlink_ticks":       CHAINLINK_TICKS_SCHEMA,
    "window_outcomes":       WINDOW_OUTCOMES_SCHEMA,
    "model_versions":        MODEL_VERSIONS_SCHEMA,
}

# Tables that use hour partitioning (others just dt= + asset=)
HOURLY_TABLES = {"decisions", "binance_ticks", "polymarket_snapshots",
                 "polymarket_trades", "chainlink_ticks"}


# ---------------------------------------------------------------------------
# Writer internals
# ---------------------------------------------------------------------------

class _PartitionWriter:
    """
    Row buffer for a single (table, asset, dt, h) partition directory.

    Each flush writes a timestamped .parquet file inside the partition dir.
    Multiple files per partition is standard hive format — DuckDB reads the
    whole directory with a glob. No persistent file handle is held between
    flushes, so files are always immediately readable.

    Path layout:
      hourly:  <base>/<table>/dt=X/asset=Y/h=HH/<flush_ts_ns>.parquet
      daily:   <base>/<table>/dt=X/asset=Y/<flush_ts_ns>.parquet
    """

    def __init__(self, dir_path: Path, schema: pa.Schema):
        dir_path.mkdir(parents=True, exist_ok=True)
        self.dir_path = dir_path
        self.schema = schema
        self._rows: list[dict] = []
        self._row_count = 0
        self._flush_seq = 0

    def append(self, row: dict) -> None:
        self._rows.append(row)

    def flush(self) -> int:
        if not self._rows:
            return 0
        table = _rows_to_table(self._rows, self.schema)
        n = len(self._rows)
        self._rows = []

        fname = f"{time.time_ns()}_{self._flush_seq:04d}.parquet"
        self._flush_seq += 1
        out_path = self.dir_path / fname

        writer = pq.ParquetWriter(str(out_path), self.schema,
                                  compression="zstd", compression_level=3)
        writer.write_table(table)
        writer.close()

        self._row_count += n
        return n

    def close(self) -> None:
        self.flush()


def _rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    """Convert a list of row dicts to a PyArrow Table, coercing types."""
    cols: dict[str, list] = {field.name: [] for field in schema}
    for row in rows:
        for field in schema:
            val = row.get(field.name)
            cols[field.name].append(val)
    arrays = []
    for field in schema:
        try:
            arr = pa.array(cols[field.name], type=field.type)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Fallback: cast nulls/wrong types to null
            arr = pa.array([None] * len(rows), type=field.type)
        arrays.append(arr)
    return pa.table(arrays, schema=schema)


def _partition_dir(base: Path, table: str, asset: str, dt: str, h: int) -> Path:
    """Return the directory for this partition (multiple files may live here)."""
    if table in HOURLY_TABLES:
        return base / table / f"dt={dt}" / f"asset={asset}" / f"h={h:02d}"
    else:
        return base / table / f"dt={dt}" / f"asset={asset}"


# ---------------------------------------------------------------------------
# Public ParquetWriter class
# ---------------------------------------------------------------------------

class ParquetWriter:
    """
    Async-safe row buffer + background flush loop.

    All write_* methods are synchronous and non-blocking — they append to an
    in-memory buffer. The background flush task calls flush() every
    PARQUET_FLUSH_S seconds and rotates files on hour boundaries.
    """

    def __init__(self, parquet_dir: Path = PARQUET_DIR):
        self._base = parquet_dir
        self._writers: dict[tuple, _PartitionWriter] = {}
        self._running = False

    # ------------------------------------------------------------------
    # Background flush task
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(PARQUET_FLUSH_S)
            self._flush_all()

    def stop(self) -> None:
        self._running = False

    async def close(self) -> None:
        """Flush remaining rows and close all open writers."""
        self.stop()
        self._flush_all()
        for pw in self._writers.values():
            try:
                pw.close()
            except Exception as exc:
                logger.error("parquet close error: %s", exc)
        self._writers.clear()
        logger.info("parquet_writer closed all writers")

    def _flush_all(self) -> None:
        now_dt, now_h = _dt_and_hour()
        stale_keys = []
        total = 0
        for key, pw in self._writers.items():
            _, _, dt, h = key
            n = pw.flush()
            total += n
            # Rotate if hour or date changed
            if dt != now_dt or h != now_h:
                pw.close()
                stale_keys.append(key)
        for k in stale_keys:
            del self._writers[k]
        if total:
            logger.debug("parquet flush wrote %d rows across %d writers", total, len(self._writers))

    def _get_writer(self, table: str, asset: str) -> _PartitionWriter:
        dt, h = _dt_and_hour()
        key = (table, asset, dt, h)
        if key not in self._writers:
            dir_path = _partition_dir(self._base, table, asset, dt, h)
            schema = TABLE_SCHEMAS[table]
            self._writers[key] = _PartitionWriter(dir_path, schema)
            logger.info("parquet partition opened %s", dir_path)
        return self._writers[key]

    # ------------------------------------------------------------------
    # Write methods — one per table
    # ------------------------------------------------------------------

    def write_decision(self, asset: str, row: dict) -> None:
        self._get_writer("decisions", asset).append(row)

    def write_binance_tick(self, asset: str, row: dict) -> None:
        self._get_writer("binance_ticks", asset).append(row)

    def write_polymarket_snapshot(self, asset: str, row: dict) -> None:
        self._get_writer("polymarket_snapshots", asset).append(row)

    def write_polymarket_trade(self, asset: str, row: dict) -> None:
        self._get_writer("polymarket_trades", asset).append(row)

    def write_chainlink_tick(self, asset: str, row: dict) -> None:
        self._get_writer("chainlink_ticks", asset).append(row)

    def write_window_outcome(self, asset: str, row: dict) -> None:
        self._get_writer("window_outcomes", asset).append(row)

    def write_model_version(self, asset: str, row: dict) -> None:
        self._get_writer("model_versions", asset).append(row)

    def force_flush(self) -> None:
        """Flush immediately — called by scheduler on graceful shutdown."""
        self._flush_all()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_and_hour() -> tuple[str, int]:
    now = datetime.now(tz=timezone.utc)
    return now.strftime("%Y-%m-%d"), now.hour
