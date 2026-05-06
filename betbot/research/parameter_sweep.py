"""
parameter_sweep.py — Sweep exit-rule parameters over the dry-run parquet data.

Runs the cash-out simulator across all combinations of:
    LAG_CLOSE_THRESHOLD ∈ {0.002, 0.003, 0.005, 0.008, 0.012, 0.020}
    STOP_THRESHOLD      ∈ {0.020, 0.030, 0.050, 0.080, 0.120}
    FALLBACK_TAU        ∈ {5, 10, 20, 30}

For each combination, computes total P&L, ROI, win rate, exit mix.
Writes a summary Parquet and prints the top-N combinations by Sharpe.

Usage:
    python -m polybot.research.parameter_sweep \
        --parquet-dir logs \
        --out sweep_results.parquet \
        --top 20

See CLAUDE.md §10.3 for the sweep specification.
"""

import argparse
import logging
import math
import statistics
from itertools import product
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polybot.research.cashout_simulator import run_simulation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep grid (CLAUDE.md §10.3)
# ---------------------------------------------------------------------------

LAG_CLOSE_GRID = [0.002, 0.003, 0.005, 0.008, 0.012, 0.020]
STOP_GRID      = [0.020, 0.030, 0.050, 0.080, 0.120]
FALLBACK_GRID  = [5.0, 10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# Summary stats for one parameter combination
# ---------------------------------------------------------------------------

def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_entries": 0, "total_pnl_usd": 0.0, "total_wagered_usd": 0.0,
            "roi": 0.0, "win_rate": 0.0, "sharpe": 0.0,
            "n_lag_closed": 0, "n_stopped": 0, "n_resolution": 0,
            "lag_closed_pct": 0.0, "mean_hold_s": 0.0,
            "mean_pnl_per_dollar": 0.0, "std_pnl_per_dollar": 0.0,
        }

    pnls          = [r["realized_pnl_usd"] for r in rows]
    pnl_per_dollar = [r["realized_pnl_per_dollar"] for r in rows]
    wagered       = [r["entry_size_usd"] for r in rows]
    hold_s        = [r["exit_holding_seconds"] for r in rows]

    n             = len(rows)
    total_pnl     = sum(pnls)
    total_wagered = sum(wagered)
    roi           = total_pnl / total_wagered if total_wagered > 0 else 0.0
    win_rate      = sum(1 for p in pnls if p > 0) / n
    mean_ppd      = statistics.mean(pnl_per_dollar)
    std_ppd       = statistics.stdev(pnl_per_dollar) if n > 1 else 0.0
    sharpe        = mean_ppd / std_ppd if std_ppd > 0 else 0.0

    n_lag   = sum(1 for r in rows if r["exit_reason"] == "lag_closed")
    n_stop  = sum(1 for r in rows if r["exit_reason"] == "stopped_out")
    n_res   = sum(1 for r in rows if r["exit_reason"] == "resolution")

    return {
        "n_entries":              n,
        "total_pnl_usd":         round(total_pnl, 4),
        "total_wagered_usd":     round(total_wagered, 4),
        "roi":                   round(roi, 6),
        "win_rate":              round(win_rate, 4),
        "sharpe":                round(sharpe, 4),
        "n_lag_closed":          n_lag,
        "n_stopped":             n_stop,
        "n_resolution":          n_res,
        "lag_closed_pct":        round(n_lag / n, 4),
        "mean_hold_s":           round(statistics.mean(hold_s), 1),
        "mean_pnl_per_dollar":   round(mean_ppd, 6),
        "std_pnl_per_dollar":    round(std_ppd, 6),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    parquet_dir: Path,
    lag_close_grid: list[float] = LAG_CLOSE_GRID,
    stop_grid:      list[float] = STOP_GRID,
    fallback_grid:  list[float] = FALLBACK_GRID,
    since: str | None = None,
) -> list[dict]:
    """
    Run all parameter combinations. Returns list of summary dicts,
    one per (lag_close, stop, fallback_tau) combination.
    """
    combos = list(product(lag_close_grid, stop_grid, fallback_grid))
    total  = len(combos)
    logger.info("Sweeping %d parameter combinations", total)

    all_results = []

    for i, (lag_close, stop, fallback) in enumerate(combos, 1):
        logger.info("[%d/%d] lag_close=%.3f stop=%.3f fallback_tau=%.0f",
                    i, total, lag_close, stop, fallback)

        sim_rows = run_simulation(
            parquet_dir=parquet_dir,
            lag_close=lag_close,
            stop=stop,
            fallback_tau=fallback,
            since=since,
        )

        summary = _summarize(sim_rows)
        summary.update({
            "lag_close_threshold": lag_close,
            "stop_threshold":      stop,
            "fallback_tau_s":      fallback,
        })
        all_results.append(summary)

    # Sort by Sharpe descending
    all_results.sort(key=lambda r: r["sharpe"], reverse=True)
    return all_results


def _print_top(results: list[dict], top: int = 20) -> None:
    print(f"\n{'='*90}")
    print(f"  TOP {min(top, len(results))} PARAMETER COMBINATIONS BY SHARPE RATIO")
    print(f"{'='*90}")
    hdr = (
        f"{'Rank':>4}  {'LC':>6}  {'Stop':>6}  {'FTau':>4}  "
        f"{'N':>5}  {'P&L':>8}  {'ROI':>7}  {'Sharpe':>7}  "
        f"{'WinRate':>7}  {'LagCls%':>7}  {'HoldS':>6}"
    )
    print(hdr)
    print("-" * 90)
    for rank, r in enumerate(results[:top], 1):
        print(
            f"{rank:>4}  {r['lag_close_threshold']:>6.3f}  {r['stop_threshold']:>6.3f}  "
            f"{r['fallback_tau_s']:>4.0f}  {r['n_entries']:>5}  "
            f"{r['total_pnl_usd']:>8.2f}  {r['roi']:>7.4f}  {r['sharpe']:>7.4f}  "
            f"{r['win_rate']:>7.3f}  {r['lag_closed_pct']:>7.3f}  {r['mean_hold_s']:>6.1f}"
        )
    print(f"{'='*90}\n")

    if results:
        best = results[0]
        print("BEST COMBINATION:")
        print(f"  lag_close_threshold = {best['lag_close_threshold']}")
        print(f"  stop_threshold      = {best['stop_threshold']}")
        print(f"  fallback_tau_s      = {best['fallback_tau_s']}")
        print(f"  Total P&L:  ${best['total_pnl_usd']:.2f}")
        print(f"  ROI:        {best['roi']*100:.3f}%")
        print(f"  Sharpe:     {best['sharpe']:.4f}")
        print(f"  Win rate:   {best['win_rate']*100:.1f}%")
        print(f"  Lag-close%: {best['lag_closed_pct']*100:.1f}%")
        print()


def _to_parquet(results: list[dict], out_path: Path) -> None:
    if not results:
        logger.warning("No sweep results to write")
        return
    table = pa.Table.from_pylist(results)
    pq.write_table(table, out_path, compression="zstd")
    logger.info("Wrote sweep results to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter sweep over exit-rule thresholds")
    p.add_argument("--parquet-dir", default="logs")
    p.add_argument("--out", default="sweep_results.parquet")
    p.add_argument("--top", type=int, default=20, help="Print top-N by Sharpe")
    p.add_argument("--lag-close", nargs="*", type=float, default=None,
                   help="Override lag_close grid (space-separated floats)")
    p.add_argument("--stop", nargs="*", type=float, default=None,
                   help="Override stop grid")
    p.add_argument("--fallback-tau", nargs="*", type=float, default=None,
                   help="Override fallback_tau grid")
    p.add_argument("--since", default=None,
                   help="Ignore data before this ISO timestamp, e.g. 2026-05-05T18:00:00")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    results = run_sweep(
        parquet_dir=Path(args.parquet_dir),
        lag_close_grid=args.lag_close or LAG_CLOSE_GRID,
        stop_grid=args.stop or STOP_GRID,
        fallback_grid=args.fallback_tau or FALLBACK_GRID,
        since=args.since,
    )

    _print_top(results, top=args.top)
    _to_parquet(results, Path(args.out))
