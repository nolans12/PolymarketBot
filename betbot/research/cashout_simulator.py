"""
cashout_simulator.py — Replay exit decisions against logged parquet data.

Reads decisions/ and polymarket_book_snapshots/ from the parquet store and
walks forward from each entry event, applying the cash-out exit rule to
determine realized P&L.

Usage:
    python -m polybot.research.cashout_simulator \
        --parquet-dir logs \
        --lag-close 0.005 \
        --stop 0.030 \
        --fallback-tau 10 \
        --out window_outcomes_sim.parquet

See CLAUDE.md §10.2 for the canonical exit logic.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

THETA = 0.05   # taker fee coefficient


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntryRow:
    window_ts:            int
    asset:                str
    ts_ns:                int
    entry_side:           str
    entry_price:          float   # ask price paid
    entry_size_usd:       float
    entry_payout_contracts: float
    entry_edge_signed:    float
    entry_tau_s:          float
    entry_tier:           int
    entry_q_settled:      Optional[float]
    entry_model_version_id: Optional[str]
    outcome:              Optional[int]   # 1=Up, 0=Down; filled in from window_outcomes


@dataclass
class ExitResult:
    exit_reason:        str    # "lag_closed" | "stopped_out" | "resolution"
    exit_tau_s:         float
    exit_price:         float  # bid we sold into, or 1.0/0.0 at resolution
    holding_seconds:    float
    realized_pnl_usd:   float
    realized_pnl_per_dollar: float


# ---------------------------------------------------------------------------
# P&L formula (CLAUDE.md §10.2)
# ---------------------------------------------------------------------------

def _realized_pnl(
    entry: EntryRow,
    exit_reason: str,
    exit_price: float,
) -> tuple[float, float]:
    contracts = entry.entry_payout_contracts
    cost      = entry.entry_size_usd
    entry_fee = THETA * entry.entry_price * (1.0 - entry.entry_price) * cost

    if exit_reason in ("lag_closed", "stopped_out"):
        gross     = contracts * exit_price
        exit_fee  = THETA * exit_price * (1.0 - exit_price) * gross
        pnl       = gross - cost - entry_fee - exit_fee
    else:
        # resolution: exit_price is 1.0 (win) or 0.0 (loss)
        if exit_price >= 1.0:
            pnl = contracts - cost - entry_fee
        else:
            pnl = -cost - entry_fee

    per_dollar = pnl / cost if cost > 0 else 0.0
    return pnl, per_dollar


# ---------------------------------------------------------------------------
# VWAP fill model (simplified — uses best bid from logged decision row)
# ---------------------------------------------------------------------------

def _vwap_exit_price(
    entry_side: str,
    decision_row: dict,
) -> float:
    """
    Returns the exit price for a given decision row.

    In Phase 1 we only log L1 bids/asks, not full order-book depth snapshots,
    so we use the logged best_bid as a conservative proxy for the exit fill.
    Phase 2 can refine this with VWAP against polymarket_book_snapshots.
    """
    if entry_side == "up":
        bid = decision_row.get("q_up_bid")
    else:
        bid = decision_row.get("q_down_bid")
    return float(bid) if bid is not None else 0.5


# ---------------------------------------------------------------------------
# Single-entry exit simulation
# ---------------------------------------------------------------------------

def simulate_exit(
    entry: EntryRow,
    decisions_after: list[dict],
    lag_close: float,
    stop: float,
    fallback_tau: float,
) -> ExitResult:
    """
    Walk forward from entry, apply exit rules, return ExitResult.

    decisions_after: list of decision row dicts ordered by ts_ns, each must
    have edge_up_net, edge_down_net, q_up_bid, q_down_bid, tau_s.
    """
    edge_at_entry = entry.entry_edge_signed

    for row in decisions_after:
        tau_now = float(row.get("tau_s", 0.0))

        # Current net edge on the open side
        if entry.entry_side == "up":
            edge_now = row.get("edge_up_net")
        else:
            edge_now = row.get("edge_down_net")

        exit_price = _vwap_exit_price(entry.entry_side, row)
        hold_s     = entry.entry_tau_s - tau_now

        # 1. Resolution fallback at small τ
        if tau_now < fallback_tau:
            terminal  = 1.0 if (entry.outcome == 1) == (entry.entry_side == "up") else 0.0
            pnl, ppd  = _realized_pnl(entry, "resolution", terminal)
            return ExitResult("resolution", tau_now, terminal, hold_s, pnl, ppd)

        if edge_now is None:
            continue

        # 2. Lag closed → profit-take
        if edge_now < lag_close:
            pnl, ppd = _realized_pnl(entry, "lag_closed", exit_price)
            return ExitResult("lag_closed", tau_now, exit_price, hold_s, pnl, ppd)

        # 3. Stop-loss
        if edge_now < edge_at_entry - stop:
            pnl, ppd = _realized_pnl(entry, "stopped_out", exit_price)
            return ExitResult("stopped_out", tau_now, exit_price, hold_s, pnl, ppd)

    # Window expired without an exit trigger → resolution
    terminal = 1.0 if (entry.outcome == 1) == (entry.entry_side == "up") else 0.0
    pnl, ppd = _realized_pnl(entry, "resolution", terminal)
    return ExitResult("resolution", 0.0, terminal, entry.entry_tau_s, pnl, ppd)


# ---------------------------------------------------------------------------
# DuckDB data loading
# ---------------------------------------------------------------------------

def _load_decisions(parquet_dir: Path) -> list[dict]:
    decisions_glob = str(parquet_dir / "decisions" / "**" / "*.parquet")
    con = duckdb.connect()
    # Deduplicate on ts_ns in case the bot was restarted mid-run and wrote
    # overlapping rows into the same partition directory.
    rows = con.execute(f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY asset, ts_ns ORDER BY ts_ns) AS _rn
            FROM read_parquet('{decisions_glob}', hive_partitioning=true)
        ) WHERE _rn = 1
        ORDER BY ts_ns ASC
    """).fetchall()
    cols = [d[0] for d in con.description if d[0] != "_rn"]
    # strip the _rn column from results
    rn_idx = next(i for i, d in enumerate(con.description) if d[0] == "_rn")
    con.close()
    return [dict(zip(cols, [v for i, v in enumerate(r) if i != rn_idx])) for r in rows]


def _load_window_outcomes(parquet_dir: Path) -> dict[tuple, int]:
    """Returns {(asset, window_ts): outcome} mapping."""
    outcomes_glob = str(parquet_dir / "window_outcomes" / "**" / "*.parquet")
    con = duckdb.connect()
    try:
        rows = con.execute(f"""
            SELECT asset, window_ts, outcome
            FROM read_parquet('{outcomes_glob}', hive_partitioning=true)
            WHERE outcome IS NOT NULL
        """).fetchall()
        con.close()
        return {(r[0], r[1]): r[2] for r in rows}
    except Exception:
        con.close()
        return {}


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    parquet_dir: Path,
    lag_close: float,
    stop: float,
    fallback_tau: float,
    since: str | None = None,     # ISO date string e.g. "2026-05-05T14:00:00"
) -> list[dict]:
    """
    Run the cash-out simulator over all logged decisions.
    Returns a list of output rows (one per entry event).

    `since` lets you ignore data before a restart or bad-data period.
    Example: since="2026-05-05T18:00:00" skips everything before 6pm.
    """
    logger.info("Loading decisions from %s", parquet_dir)
    all_rows = _load_decisions(parquet_dir)

    if since:
        import datetime
        cutoff_ns = int(datetime.datetime.fromisoformat(since).timestamp() * 1e9)
        before = len(all_rows)
        all_rows = [r for r in all_rows if r.get("ts_ns", 0) >= cutoff_ns]
        logger.info("--since filter: dropped %d rows before %s (%d remaining)",
                    before - len(all_rows), since, len(all_rows))
    outcomes = _load_window_outcomes(parquet_dir)
    logger.info("Loaded %d decision rows, %d window outcomes", len(all_rows), len(outcomes))

    # Group rows by (asset, window_ts) for fast forward-walk lookups
    from collections import defaultdict
    by_window: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row.get("asset"), row.get("window_ts"))
        by_window[key].append(row)

    results = []

    for key, rows in by_window.items():
        asset, window_ts = key
        window_outcome = outcomes.get((asset, window_ts))

        # Find entry rows (there should be at most one per window per policy)
        entry_rows_raw = [r for r in rows if r.get("event") == "entry"]
        if not entry_rows_raw:
            continue

        for entry_raw in entry_rows_raw:
            entry = EntryRow(
                window_ts=window_ts,
                asset=asset,
                ts_ns=entry_raw["ts_ns"],
                entry_side=entry_raw.get("chosen_side", "up"),
                entry_price=float(entry_raw.get("bet_price") or 0.5),
                entry_size_usd=float(entry_raw.get("would_bet_usd") or 0.0),
                entry_payout_contracts=float(entry_raw.get("bet_payout_contracts") or 0.0),
                entry_edge_signed=float(entry_raw.get("edge_signed") or 0.0),
                entry_tau_s=float(entry_raw.get("tau_s") or 0.0),
                entry_tier=int(entry_raw.get("tier") or 0),
                entry_q_settled=entry_raw.get("q_settled"),
                entry_model_version_id=entry_raw.get("model_version_id"),
                outcome=window_outcome,
            )

            # Decisions after this entry, within the same window
            after = [r for r in rows if r["ts_ns"] > entry.ts_ns]
            result = simulate_exit(entry, after, lag_close, stop, fallback_tau)

            results.append({
                "asset":                  asset,
                "window_ts":              window_ts,
                "entry_ts_ns":            entry.ts_ns,
                "entry_side":             entry.entry_side,
                "entry_tier":             entry.entry_tier,
                "entry_tau_s":            entry.entry_tau_s,
                "entry_price":            entry.entry_price,
                "entry_size_usd":         entry.entry_size_usd,
                "entry_payout_contracts": entry.entry_payout_contracts,
                "entry_edge_signed":      entry.entry_edge_signed,
                "entry_q_settled":        entry.entry_q_settled,
                "entry_model_version_id": entry.entry_model_version_id,
                "outcome":                window_outcome,
                "exit_reason":            result.exit_reason,
                "exit_tau_s":             result.exit_tau_s,
                "exit_price":             result.exit_price,
                "exit_holding_seconds":   result.holding_seconds,
                "realized_pnl_usd":       result.realized_pnl_usd,
                "realized_pnl_per_dollar": result.realized_pnl_per_dollar,
                # Sweep params (for joining in parameter_sweep)
                "lag_close_threshold":    lag_close,
                "stop_threshold":         stop,
                "fallback_tau_s":         fallback_tau,
            })

    logger.info("Simulation complete: %d entries processed", len(results))
    return results


def _to_parquet(rows: list[dict], out_path: Path) -> None:
    if not rows:
        logger.warning("No entries to write")
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd")
    logger.info("Wrote %d rows to %s", len(rows), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cash-out exit simulator")
    p.add_argument("--parquet-dir", default="logs", help="Root of Parquet store")
    p.add_argument("--lag-close",   type=float, default=0.005)
    p.add_argument("--stop",        type=float, default=0.030)
    p.add_argument("--fallback-tau", type=float, default=10.0)
    p.add_argument("--since", default=None,
                   help="Ignore data before this ISO timestamp, e.g. 2026-05-05T18:00:00")
    p.add_argument("--out", default="window_outcomes_sim.parquet",
                   help="Output Parquet file path")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    results = run_simulation(
        parquet_dir=Path(args.parquet_dir),
        lag_close=args.lag_close,
        stop=args.stop,
        fallback_tau=args.fallback_tau,
        since=args.since,
    )
    _to_parquet(results, Path(args.out))
