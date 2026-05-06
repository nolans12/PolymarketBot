"""
polybot_metrics.py — SSH-friendly DuckDB diagnostic queries over Parquet logs.

Commands:
  summary   — P&L summary by asset/day (CLAUDE.md §10.5 Metric 1)
  model     — Regression model health (R², lag stability, coefficients)
  edge      — Edge distribution stats (Metric 3)
  engage    -- Engagement rate by tau bucket (Metric 2)
  edge-slope — Edge realization slope (CLAUDE.md §10.6)
  sweep     — Top parameter combinations from sweep_results.parquet

Usage examples:
    python -m polybot.cli.polybot_metrics summary
    python -m polybot.cli.polybot_metrics summary --asset btc --since 2026-05-04
    python -m polybot.cli.polybot_metrics model --latest
    python -m polybot.cli.polybot_metrics model --since 2026-05-04
    python -m polybot.cli.polybot_metrics edge --asset btc
    python -m polybot.cli.polybot_metrics engage --asset btc
    python -m polybot.cli.polybot_metrics edge-slope --sweep sweep_results.parquet
    python -m polybot.cli.polybot_metrics sweep --top 10 --file sweep_results.parquet
"""

import argparse
import sys
from pathlib import Path

import duckdb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _con(parquet_dir: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def _decisions_glob(parquet_dir: Path) -> str:
    return str(parquet_dir / "decisions" / "**" / "*.parquet")


def _models_glob(parquet_dir: Path) -> str:
    return str(parquet_dir / "model_versions" / "**" / "*.parquet")


def _outcomes_glob(parquet_dir: Path) -> str:
    return str(parquet_dir / "window_outcomes" / "**" / "*.parquet")


def _since_filter(since: str | None, col: str = "ts_ns") -> str:
    if not since:
        return ""
    return f"AND to_timestamp({col}/1e9) >= '{since}'"


def _asset_filter(asset: str | None) -> str:
    if not asset:
        return ""
    return f"AND asset = '{asset.lower()}'"


def _print_table(con: duckdb.DuckDBPyConnection, sql: str, title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    try:
        result = con.execute(sql)
        cols   = [d[0] for d in con.description]
        rows   = result.fetchall()
        if not rows:
            print("  (no data)")
        else:
            # Column widths
            widths = [max(len(str(c)), max((len(str(r[i])) for r in rows), default=0))
                      for i, c in enumerate(cols)]
            fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
            print(fmt.format(*cols))
            print("  " + "  ".join("-" * w for w in widths))
            for row in rows:
                print(fmt.format(*[str(v) if v is not None else "NULL" for v in row]))
    except Exception as exc:
        print(f"  ERROR: {exc}")
    print()


# ---------------------------------------------------------------------------
# summary — P&L by asset / day  (CLAUDE.md §10.5 Metric 1)
# ---------------------------------------------------------------------------

def cmd_summary(args: argparse.Namespace) -> None:
    parquet_dir = Path(args.parquet_dir)
    dg          = _decisions_glob(parquet_dir)
    asset_f     = _asset_filter(args.asset)
    since_f     = _since_filter(args.since)

    con = _con(parquet_dir)

    # --- Decision-level summary ---
    _print_table(con, f"""
        SELECT
            asset,
            CAST(to_timestamp(ts_ns/1e9) AS DATE) AS day,
            COUNT(*) FILTER (WHERE event != 'sample') AS n_decisions,
            COUNT(*) FILTER (WHERE event = 'sample')  AS n_samples,
            COUNT(*) FILTER (WHERE event = 'entry')   AS n_entries,
            COUNT(*) FILTER (WHERE event = 'abstain') AS n_abstains,
            ROUND(COUNT(*) FILTER (WHERE event='entry') * 100.0
                / NULLIF(COUNT(*) FILTER (WHERE event != 'sample'), 0), 2) AS entry_rate_pct,
            ROUND(AVG(CASE WHEN edge_magnitude > -50 AND event != 'sample'
                          THEN edge_magnitude END), 4) AS mean_edge_mag,
            COUNT(*) FILTER (WHERE abstention_reason = 'model_warmup') AS warmup_abstains,
            COUNT(*) FILTER (WHERE abstention_reason = 'edge_below_floor') AS below_floor,
            COUNT(*) FILTER (WHERE circuit_active IS NOT NULL AND event != 'sample') AS circuit_trips
        FROM read_parquet('{dg}', hive_partitioning=true)
        WHERE 1=1 {asset_f} {since_f}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """, "DECISION SUMMARY BY ASSET / DAY")

    # --- Abstention breakdown ---
    _print_table(con, f"""
        SELECT
            asset,
            abstention_reason,
            COUNT(*) AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY asset), 2) AS pct
        FROM read_parquet('{dg}', hive_partitioning=true)
        WHERE event = 'abstain' {asset_f} {since_f}
        GROUP BY 1, 2
        ORDER BY 1, COUNT(*) DESC
    """, "ABSTENTION BREAKDOWN")

    con.close()


# ---------------------------------------------------------------------------
# model — Regression health  (CLAUDE.md §10.4)
# ---------------------------------------------------------------------------

def cmd_model(args: argparse.Namespace) -> None:
    parquet_dir = Path(args.parquet_dir)
    mg          = _models_glob(parquet_dir)
    asset_f     = _asset_filter(args.asset)
    since_f     = _since_filter(args.since)

    con = _con(parquet_dir)

    if args.latest:
        _print_table(con, f"""
            SELECT
                asset,
                to_timestamp(ts_ns/1e9) AS refit_at,
                model_version_id,
                n_train_samples,
                ROUND(r2_cv_mean, 4) AS r2_cv,
                ROUND(r2_in_sample, 4) AS r2_train,
                ROUND(r2_held_out_30min, 4) AS r2_held_out,
                ROUND(estimated_lag_seconds, 1) AS lag_s,
                ROUND(ridge_alpha, 4) AS ridge_alpha,
                ROUND(coef_delta_l2, 4) AS coef_delta_l2
            FROM read_parquet('{mg}', hive_partitioning=true)
            WHERE 1=1 {asset_f}
            ORDER BY ts_ns DESC
            LIMIT 2
        """, "LATEST MODEL VERSION(S)")

        _print_table(con, f"""
            SELECT
                asset,
                to_timestamp(ts_ns/1e9) AS refit_at,
                ROUND(coef_alpha, 4)           AS intercept,
                ROUND(coef_x_now, 4)           AS x_now,
                ROUND(coef_x_15, 4)            AS x_15,
                ROUND(coef_x_30, 4)            AS x_30,
                ROUND(coef_x_45, 4)            AS x_45,
                ROUND(coef_x_60, 4)            AS x_60,
                ROUND(coef_x_90, 4)            AS x_90,
                ROUND(coef_x_120, 4)           AS x_120,
                ROUND(coef_tau, 4)             AS tau,
                ROUND(coef_ofi_l1, 4)          AS ofi_l1,
                ROUND(coef_pm_book_imbalance, 4) AS pm_imbalance
            FROM read_parquet('{mg}', hive_partitioning=true)
            WHERE 1=1 {asset_f}
            ORDER BY ts_ns DESC
            LIMIT 2
        """, "LATEST COEFFICIENTS")
    else:
        _print_table(con, f"""
            SELECT
                asset,
                ROUND(APPROX_QUANTILE(r2_cv_mean, 0.10), 4) AS p10_r2_cv,
                ROUND(APPROX_QUANTILE(r2_cv_mean, 0.50), 4) AS median_r2_cv,
                ROUND(APPROX_QUANTILE(r2_cv_mean, 0.90), 4) AS p90_r2_cv,
                ROUND(AVG(r2_cv_mean), 4)                   AS mean_r2_cv,
                COUNT(*) AS n_refits,
                ROUND(AVG(estimated_lag_seconds), 1)         AS mean_lag_s,
                ROUND(STDDEV(estimated_lag_seconds), 1)      AS sd_lag_s,
                ROUND(MIN(estimated_lag_seconds), 1)         AS min_lag_s,
                ROUND(MAX(estimated_lag_seconds), 1)         AS max_lag_s
            FROM read_parquet('{mg}', hive_partitioning=true)
            WHERE 1=1 {asset_f} {since_f}
            GROUP BY asset
        """, "MODEL R² DISTRIBUTION + LAG STABILITY")

        _print_table(con, f"""
            SELECT
                asset,
                date_trunc('hour', to_timestamp(ts_ns/1e9)) AS hr,
                COUNT(*)                               AS n_refits,
                ROUND(AVG(r2_cv_mean), 4)              AS mean_r2_cv,
                ROUND(AVG(estimated_lag_seconds), 1)   AS mean_lag_s,
                ROUND(STDDEV(estimated_lag_seconds), 1) AS sd_lag_s
            FROM read_parquet('{mg}', hive_partitioning=true)
            WHERE 1=1 {asset_f} {since_f}
            GROUP BY 1, 2
            ORDER BY 1, 2
        """, "MODEL STABILITY OVER TIME (HOURLY)")

    con.close()


# ---------------------------------------------------------------------------
# edge — Edge distribution  (CLAUDE.md §10.5 Metric 3)
# ---------------------------------------------------------------------------

def cmd_edge(args: argparse.Namespace) -> None:
    parquet_dir = Path(args.parquet_dir)
    dg          = _decisions_glob(parquet_dir)
    asset_f     = _asset_filter(args.asset)
    since_f     = _since_filter(args.since)

    con = _con(parquet_dir)

    _print_table(con, f"""
        SELECT
            asset,
            COUNT(*) AS n_decisions,
            ROUND(AVG(edge_magnitude), 5)                     AS mean_abs_edge,
            ROUND(AVG(edge_signed), 5)                        AS mean_signed_edge,
            ROUND(STDDEV(edge_signed), 5)                     AS sd_signed_edge,
            ROUND(APPROX_QUANTILE(edge_magnitude, 0.50), 5)   AS p50_edge,
            ROUND(APPROX_QUANTILE(edge_magnitude, 0.90), 5)   AS p90_edge,
            ROUND(APPROX_QUANTILE(edge_magnitude, 0.99), 5)   AS p99_edge,
            ROUND(AVG(edge_up_net), 5)                        AS mean_edge_up,
            ROUND(AVG(edge_down_net), 5)                      AS mean_edge_down
        FROM read_parquet('{dg}', hive_partitioning=true)
        WHERE edge_magnitude > -50
          AND coinbase_stale_ms < 10000
          AND polymarket_stale_ms < 60000
          {asset_f} {since_f}
        GROUP BY asset
    """, "EDGE DISTRIBUTION (CLAUDE.md §10.5 Metric 3)")

    _print_table(con, f"""
        SELECT
            asset,
            CASE
                WHEN tau_s > 240 THEN '1_T300_T241'
                WHEN tau_s > 60  THEN '2_T240_T61'
                WHEN tau_s > 10  THEN '3_T60_T11'
                ELSE                  '4_T10_T0'
            END AS tau_bucket,
            COUNT(*) AS n,
            ROUND(AVG(edge_magnitude) FILTER (WHERE edge_magnitude > -50), 5) AS mean_edge,
            ROUND(COUNT(*) FILTER (WHERE event='entry') * 100.0 / COUNT(*), 2) AS entry_rate_pct
        FROM read_parquet('{dg}', hive_partitioning=true)
        WHERE 1=1 {asset_f} {since_f}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """, "EDGE BY TAU BUCKET")

    con.close()


# ---------------------------------------------------------------------------
# engage — Engagement rate by tau bucket  (CLAUDE.md §10.5 Metric 2)
# ---------------------------------------------------------------------------

def cmd_engage(args: argparse.Namespace) -> None:
    parquet_dir = Path(args.parquet_dir)
    dg          = _decisions_glob(parquet_dir)
    asset_f     = _asset_filter(args.asset)
    since_f     = _since_filter(args.since)

    con = _con(parquet_dir)

    _print_table(con, f"""
        SELECT
            asset,
            CASE
                WHEN tau_s > 240 THEN '1_T300_T241'
                WHEN tau_s > 60  THEN '2_T240_T61'
                WHEN tau_s > 10  THEN '3_T60_T11'
                ELSE                  '4_T10_T0'
            END AS tau_bucket,
            COUNT(*) FILTER (WHERE event != 'sample') AS n_decisions,
            COUNT(*) FILTER (WHERE event = 'entry')   AS n_entries,
            COUNT(*) FILTER (WHERE event = 'abstain') AS n_abstains,
            ROUND(COUNT(*) FILTER (WHERE event='entry') * 100.0
                / NULLIF(COUNT(*) FILTER (WHERE event != 'sample'), 0), 2) AS entry_rate_pct,
            ROUND(COUNT(*) FILTER (WHERE abstention_reason='model_warmup') * 100.0
                / NULLIF(COUNT(*) FILTER (WHERE event != 'sample'), 0), 2) AS warmup_pct,
            ROUND(COUNT(*) FILTER (WHERE abstention_reason='model_low_r2') * 100.0
                / NULLIF(COUNT(*) FILTER (WHERE event != 'sample'), 0), 2) AS low_r2_pct,
            ROUND(COUNT(*) FILTER (WHERE abstention_reason='edge_below_floor') * 100.0
                / NULLIF(COUNT(*) FILTER (WHERE event='abstain'), 0), 2) AS below_floor_of_abstains_pct
        FROM read_parquet('{dg}', hive_partitioning=true)
        WHERE event != 'sample' {asset_f} {since_f}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """, "ENGAGEMENT RATE BY TAU BUCKET (CLAUDE.md §10.5 Metric 2)")

    con.close()


# ---------------------------------------------------------------------------
# edge-slope — Edge realization slope  (CLAUDE.md §10.6)
# ---------------------------------------------------------------------------

def cmd_edge_slope(args: argparse.Namespace) -> None:
    sweep_file = Path(args.sweep)
    if not sweep_file.exists():
        print(f"ERROR: sweep file not found: {sweep_file}")
        print("Run parameter_sweep.py first.")
        sys.exit(1)

    con = duckdb.connect()

    _print_table(con, f"""
        WITH bets AS (
            SELECT
                entry_edge_signed AS edge_at_entry,
                realized_pnl_per_dollar AS realized
            FROM read_parquet('{sweep_file}')
            WHERE entry_edge_signed IS NOT NULL
        ),
        buckets AS (
            SELECT
                WIDTH_BUCKET(edge_at_entry, 0.0, 0.30, 12) AS bucket,
                COUNT(*)                   AS n,
                ROUND(AVG(edge_at_entry), 5)  AS mean_edge,
                ROUND(AVG(realized), 5)       AS mean_realized,
                ROUND(STDDEV(realized), 5)    AS sd_realized,
                ROUND(MIN(realized), 5)       AS min_realized,
                ROUND(MAX(realized), 5)       AS max_realized
            FROM bets
            GROUP BY 1
        )
        SELECT * FROM buckets
        ORDER BY bucket
    """, f"EDGE REALIZATION SLOPE — {sweep_file.name}  (CLAUDE.md §10.6)")

    # Compute OLS slope manually for the headline number
    rows = con.execute(f"""
        SELECT
            entry_edge_signed AS x,
            realized_pnl_per_dollar AS y
        FROM read_parquet('{sweep_file}')
        WHERE entry_edge_signed IS NOT NULL
          AND realized_pnl_per_dollar IS NOT NULL
    """).fetchall()

    if len(rows) >= 2:
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        n  = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den > 0 else float("nan")
        print(f"  OLS slope (edge → realized P&L per dollar): {slope:.4f}")
        print(f"  Interpretation: slope > 0 means edge predicts realized P&L.")
        print(f"  slope ≈ 0.5 means costs eat ~half the raw signal (still profitable).")
        print(f"  slope ≤ 0 means the model has no predictive value — do not trade.")
        print()

    con.close()


# ---------------------------------------------------------------------------
# sweep — Print top combinations from sweep_results.parquet
# ---------------------------------------------------------------------------

def cmd_sweep(args: argparse.Namespace) -> None:
    sweep_file = Path(args.file)
    if not sweep_file.exists():
        print(f"ERROR: {sweep_file} not found. Run parameter_sweep.py first.")
        sys.exit(1)

    con = duckdb.connect()

    _print_table(con, f"""
        SELECT
            lag_close_threshold,
            stop_threshold,
            fallback_tau_s,
            n_entries,
            ROUND(total_pnl_usd, 2)      AS total_pnl,
            ROUND(roi * 100, 3)           AS roi_pct,
            ROUND(sharpe, 4)              AS sharpe,
            ROUND(win_rate * 100, 1)      AS win_rate_pct,
            ROUND(lag_closed_pct * 100, 1) AS lag_closed_pct,
            ROUND(mean_hold_s, 1)         AS mean_hold_s
        FROM read_parquet('{sweep_file}')
        ORDER BY sharpe DESC
        LIMIT {args.top}
    """, f"TOP {args.top} PARAMETER COMBINATIONS (by Sharpe)")

    # Phase 1 decision criteria (CLAUDE.md §10.7)
    print("PHASE 1 → PHASE 2 DECISION CRITERIA CHECK (best combo):")
    result = con.execute(f"""
        SELECT
            total_pnl_usd > 0                AS criterion_positive_pnl,
            lag_closed_pct >= 0.50           AS criterion_lag_closed_50pct,
            win_rate >= 0.50                 AS criterion_win_rate_50pct,
            n_entries >= 5                   AS criterion_min_entries
        FROM read_parquet('{sweep_file}')
        ORDER BY sharpe DESC
        LIMIT 1
    """).fetchone()
    if result:
        labels = ["Positive P&L", "≥50% lag-close exits", "≥50% win rate", "≥5 entries"]
        for label, passed in zip(labels, result):
            mark = "✓" if passed else "✗"
            print(f"  [{mark}] {label}")
    print()
    con.close()


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polybot-metrics",
        description="DuckDB diagnostic queries over PolymarketBot Parquet logs",
    )
    p.add_argument("--parquet-dir", default="logs", help="Root of Parquet store")
    sub = p.add_subparsers(dest="command", required=True)

    # summary
    s = sub.add_parser("summary", help="P&L summary by asset/day")
    s.add_argument("--asset", default=None)
    s.add_argument("--since", default=None, help="ISO date, e.g. 2026-05-04")

    # model
    m = sub.add_parser("model", help="Regression health stats")
    m.add_argument("--asset", default=None)
    m.add_argument("--since", default=None)
    m.add_argument("--latest", action="store_true",
                   help="Show only the most recent model version")

    # edge
    e = sub.add_parser("edge", help="Edge distribution stats")
    e.add_argument("--asset", default=None)
    e.add_argument("--since", default=None)

    # engage
    g = sub.add_parser("engage", help="Engagement rate by tau bucket")
    g.add_argument("--asset", default=None)
    g.add_argument("--since", default=None)

    # edge-slope
    sl = sub.add_parser("edge-slope", help="Edge realization slope diagnostic")
    sl.add_argument("--sweep", default="sweep_results.parquet",
                    help="sweep_results.parquet output from parameter_sweep.py")

    # sweep
    sw = sub.add_parser("sweep", help="Top parameter combinations from sweep")
    sw.add_argument("--top", type=int, default=10)
    sw.add_argument("--file", default="sweep_results.parquet")

    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    dispatch = {
        "summary":    cmd_summary,
        "model":      cmd_model,
        "edge":       cmd_edge,
        "engage":     cmd_engage,
        "edge-slope": cmd_edge_slope,
        "sweep":      cmd_sweep,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
