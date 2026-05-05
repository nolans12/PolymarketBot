"""
bot.py — Main loop for the Dual-Mode HFT Polymarket Bot.

Startup:
    python3 bot.py                    # live trading at EPSILON from .env
    python3 bot.py --dry-run          # simulate trades, no real orders
    python3 bot.py --epsilon 0.04     # override epsilon for this session

Monitor:
    tail -f logs/bot.log
"""

import argparse
import logging
import os
import sys
import time

import config
from market_data import MarketDataClient
from markov import MarkovEstimator
from model import evaluate
from executor import Executor


def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )


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


def main():
    args = parse_args()
    dry_run = args.dry_run or config.DRY_RUN
    epsilon = args.epsilon if args.epsilon is not None else config.EPSILON

    setup_logging(config.LOG_FILE)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Polymarket Dual-Mode HFT Bot starting")
    logger.info(f"  dry_run={dry_run} | epsilon={epsilon} | p_jj_min={config.P_JJ_MIN}")
    logger.info(f"  poll_interval={config.POLL_INTERVAL}s | assets={len(config.ASSETS)}")
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

    # Verify markets are discoverable at startup
    logger.info("Verifying 5-minute markets are available...")
    startup_prices = market_client.get_all_prices()
    if not startup_prices:
        logger.error("No 5-minute markets found at startup. Check Polymarket API.")
        sys.exit(1)
    logger.info(f"Markets ready: {list(startup_prices.keys())}")

    open_orders: dict[str, str] = {}
    scan_count = 0
    trade_count = 0

    logger.info("Entering main loop. Ctrl+C to stop.")

    while True:
        try:
            loop_start = time.time()
            scan_count += 1

            # Fetch live wallet balance — sizing is a % of this
            wallet_balance = executor.get_wallet_balance()
            if wallet_balance <= 0 and not dry_run:
                logger.warning("Wallet balance is 0 or unavailable — skipping this loop")
                time.sleep(config.POLL_INTERVAL)
                continue

            # Prices and token IDs come together — window rollover handled inside
            all_prices = market_client.get_all_prices()
            secs_left = market_client.seconds_until_next_window()

            for asset in config.ASSETS:
                info = all_prices.get(asset)
                if info is None:
                    logger.debug(f"SCAN | {asset} | no market data — skip")
                    continue

                q = info["q"]
                token_id = info["token_id"]

                estimators[asset].update(q)
                p_jj, p_mine = estimators[asset].estimate(q)

                signal = evaluate(
                    asset=asset,
                    q=q,
                    p_jj=p_jj,
                    p_mine=p_mine,
                    epsilon=epsilon,
                    p_jj_min=config.P_JJ_MIN,
                    wallet_balance_usd=wallet_balance,
                )

                logger.info(
                    f"SCAN | {asset:<10} | q={q:.3f} | p_mine={p_mine:.3f} | "
                    f"Δ={p_mine - q:+.3f} | p_jj={p_jj:.3f} | "
                    f"wallet=${wallet_balance:.2f} | obs={estimators[asset].n_observations} | "
                    f"win={secs_left}s | {'SIGNAL' if signal else 'no-entry'}"
                )

                if signal is None:
                    continue

                if asset in open_orders:
                    logger.info(f"SKIP | {asset} | open order already exists: {open_orders[asset]}")
                    continue

                if dry_run:
                    logger.info(
                        f"DRY RUN TRADE | {asset} | mode={signal.mode} | "
                        f"q={signal.q:.3f} | r={signal.expected_return:.1%} | "
                        f"Δ={signal.delta:+.3f} | size=${signal.size_usd:.2f} "
                        f"({signal.wallet_pct:.1%} of ${wallet_balance:.2f})"
                    )
                    trade_count += 1
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
                        logger.info(
                            f"TRADE | {asset} | mode={signal.mode} | "
                            f"q={signal.q:.3f} | r={signal.expected_return:.1%} | "
                            f"Δ={signal.delta:+.3f} | size=${signal.size_usd:.2f} "
                            f"({signal.wallet_pct:.1%} of wallet) | order_id={order_id}"
                        )

            elapsed = time.time() - loop_start
            logger.info(
                f"LOOP {scan_count} | {elapsed:.1f}s | "
                f"trades={trade_count} | open_orders={len(open_orders)} | "
                f"wallet=${wallet_balance:.2f}"
            )

            time.sleep(max(0, config.POLL_INTERVAL - elapsed))

        except KeyboardInterrupt:
            logger.info("Shutdown requested. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            logger.info("Sleeping 60s before retry...")
            time.sleep(60)

    logger.info(f"Bot stopped. scans={scan_count} | trades={trade_count}")


if __name__ == "__main__":
    main()
