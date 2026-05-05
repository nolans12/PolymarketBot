"""
bot.py — Main loop for the Dual-Mode HFT Polymarket Bot.

Startup:
    python3 bot.py                    # live trading at EPSILON from .env
    python3 bot.py --dry-run          # simulate trades, no real orders
    python3 bot.py --epsilon 0.04     # override epsilon for this session

Monitor:
    tail -f logs/bot.log
    tail -f logs/bot_scans.jsonl
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time

import config
from market_data import MarketDataClient
from markov import MarkovEstimator
from model import evaluate, classify_mode
from executor import Executor


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    # Rotate at 20 MB, keep 7 backups (~140 MB max for bot.log)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=20 * 1024 * 1024, backupCount=7, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])


def open_jsonl(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "a", buffering=1)  # line-buffered: each write flushes


def rotate_jsonl(path: str, max_bytes: int = 50 * 1024 * 1024, backups: int = 3) -> None:
    """Rotate a JSONL file if it exceeds max_bytes. Called at bot startup."""
    if not os.path.exists(path) or os.path.getsize(path) < max_bytes:
        return
    for i in range(backups - 1, 0, -1):
        src = f"{path}.{i}"
        dst = f"{path}.{i+1}"
        if os.path.exists(src):
            os.replace(src, dst)
    os.replace(path, f"{path}.1")
    logger_pre = logging.getLogger(__name__)
    logger_pre.info(f"Rotated {path} (exceeded {max_bytes // 1024 // 1024} MB)")


def write_jsonl(fh, row: dict) -> None:
    fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Polymarket Dual-Mode HFT Bot")
    parser.add_argument("--dry-run", action="store_true", help="Log trades without placing real orders")
    parser.add_argument("--epsilon", type=float, default=None, help="Edge threshold override")
    return parser.parse_args()


def validate_config() -> None:
    missing = [v for v in ["PRIVATE_KEY", "API_KEY", "API_SECRET", "API_PASSPHRASE"]
               if not getattr(config, v, "")]
    if missing:
        print(f"ERROR: Missing .env values: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your credentials.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    dry_run = args.dry_run or config.DRY_RUN
    epsilon = args.epsilon if args.epsilon is not None else config.EPSILON

    setup_logging(config.LOG_FILE)
    logger = logging.getLogger(__name__)

    log_dir = os.path.dirname(config.LOG_FILE)
    scan_log_path   = os.path.join(log_dir, "scans.jsonl")
    trade_log_path  = os.path.join(log_dir, "trades.jsonl")
    wallet_log_path = os.path.join(log_dir, "wallet.jsonl")

    logger.info("=" * 60)
    logger.info("Polymarket Dual-Mode HFT Bot starting")
    logger.info(f"  dry_run={dry_run} | epsilon={epsilon} | p_jj_min={config.P_JJ_MIN}")
    logger.info(f"  poll_interval={config.POLL_INTERVAL}s | assets={len(config.ASSETS)}")
    logger.info(f"  scan log:   {scan_log_path}")
    logger.info(f"  trade log:  {trade_log_path}")
    logger.info(f"  wallet log: {wallet_log_path}")
    logger.info("=" * 60)

    if not dry_run:
        validate_config()

    market_client = MarketDataClient(clob_host=config.CLOB_HOST)

    executor = Executor(
        private_key=config.PRIVATE_KEY,
        api_key=config.API_KEY,
        api_secret=config.API_SECRET,
        api_passphrase=config.API_PASSPHRASE,
        clob_host=config.CLOB_HOST,
        chain_id=config.CHAIN_ID,
        dry_run=dry_run,
    )

    estimators: dict[str, MarkovEstimator] = {
        asset: MarkovEstimator(
            window_seconds=config.MARKOV_WINDOW_SECONDS,
            poll_interval=config.POLL_INTERVAL,
        )
        for asset in config.ASSETS
    }

    logger.info("Verifying 5-minute markets are available...")
    startup_prices = market_client.get_all_prices()
    if not startup_prices:
        logger.error("No 5-minute markets found at startup. Check Polymarket API.")
        sys.exit(1)
    logger.info(f"Markets ready: {list(startup_prices.keys())}")

    # open_orders: asset -> order_id (live mode only)
    open_orders: dict[str, str] = {}
    scan_count  = 0
    trade_count = 0

    # Rotate large JSONL files before opening (50 MB cap, 3 backups each)
    rotate_jsonl(scan_log_path)
    rotate_jsonl(wallet_log_path)

    scan_fh   = open_jsonl(scan_log_path)
    trade_fh  = open_jsonl(trade_log_path)
    wallet_fh = open_jsonl(wallet_log_path)

    logger.info("Entering main loop. Ctrl+C to stop.")

    try:
        while True:
            try:
                loop_start = time.time()
                scan_count += 1

                # ----------------------------------------------------------
                # Prune filled/cancelled open orders so we can re-enter
                # ----------------------------------------------------------
                for asset in list(open_orders):
                    status_data = executor.get_order_status(open_orders[asset])
                    if status_data is None:
                        continue
                    status = status_data.get("status", "")
                    if status in ("MATCHED", "FILLED", "CANCELLED", "EXPIRED", "CANCELED"):
                        logger.info(f"ORDER CLOSED | {asset} | {open_orders[asset]} | status={status}")
                        write_jsonl(trade_fh, {
                            "ts": round(time.time(), 3),
                            "event": "order_closed",
                            "asset": asset,
                            "order_id": open_orders[asset],
                            "status": status,
                        })
                        del open_orders[asset]

                # ----------------------------------------------------------
                # Wallet balance
                # ----------------------------------------------------------
                wallet_balance = executor.get_wallet_balance()
                if wallet_balance <= 0 and config.FALLBACK_BALANCE > 0:
                    logger.warning(
                        f"Wallet balance API returned 0 — using fallback ${config.FALLBACK_BALANCE:.2f}. "
                        f"Deposit USDC to Polymarket to fix this."
                    )
                    wallet_balance = config.FALLBACK_BALANCE
                write_jsonl(wallet_fh, {
                    "ts": round(loop_start, 3),
                    "wallet_usd": round(wallet_balance, 4),
                    "open_orders": len(open_orders),
                    "scan": scan_count,
                })

                if wallet_balance <= 0 and not dry_run:
                    logger.warning("Wallet balance is 0 and no fallback set — skipping this loop")
                    time.sleep(config.POLL_INTERVAL)
                    continue

                # ----------------------------------------------------------
                # Market prices (CLOB midpoints, auto window-rollover)
                # ----------------------------------------------------------
                t_fetch_start = time.time()
                all_prices = market_client.get_all_prices()
                fetch_ms = round((time.time() - t_fetch_start) * 1000)
                secs_left = market_client.seconds_until_next_window()

                if not all_prices:
                    logger.warning("No market data returned — skipping loop")
                    time.sleep(config.POLL_INTERVAL)
                    continue

                # ----------------------------------------------------------
                # Per-asset scan
                # ----------------------------------------------------------
                for asset in config.ASSETS:
                    info = all_prices.get(asset)
                    if info is None:
                        logger.debug(f"SCAN | {asset} | no market data — skip")
                        continue

                    q        = info["q"]
                    token_id = info["token_id"]

                    estimators[asset].update(q)
                    p_jj, p_mine = estimators[asset].estimate(q)
                    delta = p_mine - q
                    mode  = classify_mode(q)
                    obs   = estimators[asset].n_observations

                    signal = evaluate(
                        asset=asset,
                        q=q,
                        p_jj=p_jj,
                        p_mine=p_mine,
                        epsilon=epsilon,
                        p_jj_min=config.P_JJ_MIN,
                        wallet_balance_usd=wallet_balance,
                    )

                    # Always write every scan for post-processing / dry replay
                    write_jsonl(scan_fh, {
                        "ts":        round(time.time(), 3),
                        "asset":     asset,
                        "q":         round(q, 4),
                        "p_mine":    round(p_mine, 4),
                        "p_jj":      round(p_jj, 4),
                        "delta":     round(delta, 4),
                        "obs":       obs,
                        "secs_left": secs_left,
                        "mode":      mode,
                        "signal":    signal is not None,
                        "wallet":    round(wallet_balance, 2),
                        "fetch_ms":  fetch_ms,
                        "epsilon":   epsilon,
                        "p_jj_min":  config.P_JJ_MIN,
                        "token_id":  token_id,
                    })

                    logger.info(
                        f"SCAN | {asset:<10} | q={q:.3f} | p_mine={p_mine:.3f} | "
                        f"Δ={delta:+.3f} | p_jj={p_jj:.3f} | "
                        f"wallet=${wallet_balance:.2f} | obs={obs} | "
                        f"win={secs_left}s | {'SIGNAL' if signal else 'no-entry'}"
                    )

                    if signal is None:
                        continue

                    if asset in open_orders:
                        logger.info(f"SKIP | {asset} | open order already exists: {open_orders[asset]}")
                        continue

                    # ------------------------------------------------------
                    # Execute (or simulate) trade
                    # ------------------------------------------------------
                    trade_row = {
                        "ts":              round(time.time(), 3),
                        "event":           "signal",
                        "asset":           asset,
                        "mode":            signal.mode,
                        "q":               round(signal.q, 4),
                        "p_mine":          round(signal.p_mine, 4),
                        "p_jj":            round(signal.p_jj, 4),
                        "delta":           round(signal.delta, 4),
                        "expected_return": round(signal.expected_return, 4),
                        "size_usd":        round(signal.size_usd, 2),
                        "wallet_pct":      round(signal.wallet_pct, 4),
                        "wallet_usd":      round(wallet_balance, 2),
                        "secs_left":       secs_left,
                        "dry_run":         dry_run,
                        "token_id":        token_id,
                        "order_id":        None,
                    }

                    if dry_run:
                        logger.info(
                            f"DRY RUN TRADE | {asset} | mode={signal.mode} | "
                            f"q={signal.q:.3f} | r={signal.expected_return:.1%} | "
                            f"Δ={signal.delta:+.3f} | size=${signal.size_usd:.2f} "
                            f"({signal.wallet_pct:.1%} of ${wallet_balance:.2f})"
                        )
                        trade_count += 1
                        write_jsonl(trade_fh, trade_row)
                    else:
                        order_id = executor.place_order(
                            token_id=token_id,
                            side="BUY",
                            size_usd=signal.size_usd,
                            price=signal.q,
                        )
                        if order_id:
                            open_orders[asset] = order_id
                            trade_count += 1
                            trade_row["order_id"] = order_id
                            write_jsonl(trade_fh, trade_row)
                            logger.info(
                                f"TRADE | {asset} | mode={signal.mode} | "
                                f"q={signal.q:.3f} | r={signal.expected_return:.1%} | "
                                f"Δ={signal.delta:+.3f} | size=${signal.size_usd:.2f} "
                                f"({signal.wallet_pct:.1%} of wallet) | order_id={order_id}"
                            )

                # ----------------------------------------------------------
                # Loop summary
                # ----------------------------------------------------------
                elapsed = time.time() - loop_start
                logger.info(
                    f"LOOP {scan_count} | {elapsed:.2f}s (fetch={fetch_ms}ms) | "
                    f"trades={trade_count} | open_orders={len(open_orders)} | "
                    f"wallet=${wallet_balance:.2f}"
                )

                sleep_secs = max(0.0, config.POLL_INTERVAL - elapsed)
                time.sleep(sleep_secs)

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Unhandled error in loop: {e}", exc_info=True)
                logger.info("Sleeping 60s before retry...")
                time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Shutdown requested. Exiting.")

    finally:
        scan_fh.close()
        trade_fh.close()
        wallet_fh.close()
        logger.info(f"Bot stopped. scans={scan_count} | trades={trade_count}")


if __name__ == "__main__":
    main()
