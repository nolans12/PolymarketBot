"""
test_trade.py — Phase 0 connectivity proof.

Run this on the Hetzner server BEFORE running the full bot.
It verifies the entire chain:
  1. .env credentials load correctly
  2. Polymarket CLOB API is reachable from this IP
  3. API key derivation works
  4. A real market exists and has a price
  5. A minimal test order can be placed and cancelled

Usage:
    python3 test_trade.py                  # full connectivity test
    python3 test_trade.py --derive-creds   # just print API key/secret/passphrase
    python3 test_trade.py --list-markets   # just list 5-min markets and exit
"""

import argparse
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Load .env before importing config
from dotenv import load_dotenv
load_dotenv()

import config
from market_data import MarketDataClient
from executor import Executor


def check_env() -> bool:
    logger.info("--- Step 1: Checking .env credentials ---")
    ok = True
    for var in ["PRIVATE_KEY", "API_KEY", "API_SECRET", "API_PASSPHRASE"]:
        val = getattr(config, var, "")
        if val and val not in ("0xYOUR_PRIVATE_KEY_HERE", "YOUR_API_KEY"):
            logger.info(f"  {var}: OK ({val[:8]}...)")
        else:
            logger.error(f"  {var}: MISSING or placeholder — fill in .env")
            ok = False
    return ok


def check_api_reachable() -> bool:
    logger.info("--- Step 2: Checking Polymarket CLOB API reachability ---")
    import requests
    try:
        resp = requests.get(f"{config.CLOB_HOST}/markets", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        n = len(data.get("data", []))
        logger.info(f"  API reachable. Returned {n} markets in first page.")
        return True
    except Exception as e:
        logger.error(f"  API unreachable: {e}")
        return False


def find_one_market() -> tuple[str, str]:
    """Return (asset_name, token_id) for the first 5-min market found."""
    logger.info("--- Step 3: Finding a 5-minute market ---")
    client = MarketDataClient(clob_host=config.CLOB_HOST)
    markets = client.find_5min_markets()
    if not markets:
        logger.error("  No 5-minute markets found. Check market_data.py keyword matching.")
        return "", ""
    asset, token_id = next(iter(markets.items()))
    logger.info(f"  Found: {asset} → token_id={token_id}")
    return asset, token_id


def get_midpoint(token_id: str) -> float:
    logger.info("--- Step 4: Fetching current market price ---")
    client = MarketDataClient(clob_host=config.CLOB_HOST)
    q = client.get_midpoint(token_id)
    if q is None:
        logger.error("  Could not fetch midpoint — market may be inactive")
        return 0.0
    logger.info(f"  Current q^(w) = {q:.4f}")
    return q


def place_and_cancel_test_order(token_id: str, q: float) -> bool:
    logger.info("--- Step 5: Placing minimal test order ---")
    executor = Executor(
        private_key=config.PRIVATE_KEY,
        api_key=config.API_KEY,
        api_secret=config.API_SECRET,
        api_passphrase=config.API_PASSPHRASE,
        clob_host=config.CLOB_HOST,
        chain_id=config.CHAIN_ID,
        dry_run=False,
    )

    # Place the smallest possible order: $9 at current price
    order_id = executor.place_order(
        token_id=token_id,
        side="BUY",
        size_usd=9.0,
        price=round(q, 2),
    )

    if not order_id:
        logger.error("  Order placement FAILED")
        return False

    logger.info(f"  Order placed: {order_id}")
    logger.info("  Cancelling test order...")

    cancelled = executor.cancel_order(order_id)
    if cancelled:
        logger.info("  Order cancelled successfully.")
    else:
        logger.warning(f"  Cancel may have failed — check Polymarket dashboard for order {order_id}")

    return True


def derive_and_print_creds() -> None:
    logger.info("Deriving API credentials from private key...")
    executor = Executor(
        private_key=config.PRIVATE_KEY,
        api_key="",
        api_secret="",
        api_passphrase="",
        clob_host=config.CLOB_HOST,
        chain_id=config.CHAIN_ID,
        dry_run=True,
    )
    creds = executor.derive_api_credentials()
    print("\n--- Copy these into your .env file ---")
    print(f"API_KEY={creds['api_key']}")
    print(f"API_SECRET={creds['api_secret']}")
    print(f"API_PASSPHRASE={creds['api_passphrase']}")
    print("--------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 connectivity test")
    parser.add_argument("--derive-creds", action="store_true", help="Derive and print API credentials from private key")
    parser.add_argument("--list-markets", action="store_true", help="List all 5-min markets and exit")
    args = parser.parse_args()

    if args.derive_creds:
        derive_and_print_creds()
        return

    if args.list_markets:
        client = MarketDataClient(clob_host=config.CLOB_HOST)
        markets = client.find_5min_markets()
        print(f"\nFound {len(markets)} 5-minute markets:")
        for name, token_id in markets.items():
            print(f"  {name}: {token_id}")
        return

    print("\n=== Polymarket Bot — Phase 0 Connectivity Test ===\n")

    if not check_env():
        print("\nFIX your .env file and re-run.")
        sys.exit(1)

    if not check_api_reachable():
        print("\nAPI not reachable — check network/firewall on this server.")
        sys.exit(1)

    asset, token_id = find_one_market()
    if not token_id:
        sys.exit(1)

    q = get_midpoint(token_id)
    if q == 0.0:
        sys.exit(1)

    success = place_and_cancel_test_order(token_id, q)

    print()
    if success:
        print("=== ALL STEPS PASSED ===")
        print("Infrastructure confirmed. You can now run the full bot:")
        print("  python3 bot.py --dry-run       # dry run first")
        print("  python3 bot.py --epsilon 0.04  # live Phase 1")
    else:
        print("=== TEST FAILED — check errors above ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
