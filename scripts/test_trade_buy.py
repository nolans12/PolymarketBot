"""
test_trade_buy.py — Place ONE small BUY order on Polymarket as a live-API smoke test.

Picks the current 5-min BTC market, reads the live Up-token best ask, and posts
a $5 BUY at the ask. Default is "place a real order" — pass --dry-run to print
the intent without sending.

Usage:
    python scripts/test_trade_buy.py                    # $5 buy on Up side, real order
    python scripts/test_trade_buy.py --asset eth        # ETH instead of BTC
    python scripts/test_trade_buy.py --side down        # buy Down shares
    python scripts/test_trade_buy.py --usd 10           # $10 instead of $5
    python scripts/test_trade_buy.py --dry-run          # print intent, do not place

The script:
  1. Resolves the current window's market via Gamma REST
  2. Connects to the Polymarket CLOB WS, reads one book snapshot to find ask
  3. Places a single GTC limit order at the observed ask (taker-style)
  4. Prints the order_id and saves it to .last_test_order so the SELL script
     can find the position
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from polybot.clients.polymarket_rest import current_window_ts, fetch_market
from polybot.clients.polymarket_ws import PolymarketWS
from polybot.execution.orders import OrderClient
from polybot.state.poly_book import PolyBook


PRIVATE_KEY    = os.getenv("PRIVATE_KEY", "")
API_KEY        = os.getenv("API_KEY", "")
API_SECRET     = os.getenv("API_SECRET", "")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "")
CLOB_HOST      = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID       = int(os.getenv("CHAIN_ID", "137"))
FUNDER         = os.getenv("FUNDER", "")

LAST_ORDER_FILE = Path(".last_test_order.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Place ONE small BUY on Polymarket")
    p.add_argument("--asset", default="btc", choices=["btc", "eth"])
    p.add_argument("--side",  default="up",  choices=["up", "down"])
    p.add_argument("--usd",   type=float, default=5.0,
                   help="Order size in USD (default 5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print intent without placing the order")
    p.add_argument("--book-wait", type=float, default=5.0,
                   help="Seconds to wait for first WS book snapshot (default 5)")
    return p.parse_args()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


async def get_best_ask(asset: str, up_token: str, down_token: str,
                       wait_s: float) -> tuple[float, float]:
    """Subscribe to the CLOB WS, wait for a book snapshot, return (up_ask, down_ask)."""
    books = {asset: PolyBook(asset)}
    books[asset].set_tokens(up_token, down_token)

    got_book = asyncio.Event()

    def on_update(_asset: str, book: PolyBook) -> None:
        if book.up and book.up.best_ask < float("inf"):
            got_book.set()

    ws = PolymarketWS(books=books, on_update=on_update)
    ws.subscribe(asset, up_token, down_token)

    run_task = asyncio.create_task(ws.run())
    try:
        await asyncio.wait_for(got_book.wait(), timeout=wait_s)
    except asyncio.TimeoutError:
        ws.stop()
        run_task.cancel()
        try:
            await run_task
        except (asyncio.CancelledError, Exception):
            pass
        fail(f"No book snapshot from Polymarket WS within {wait_s}s")

    up_ask   = books[asset].up.best_ask   if books[asset].up   else 0.0
    down_ask = books[asset].down.best_ask if books[asset].down else 0.0

    ws.stop()
    run_task.cancel()
    try:
        await run_task
    except (asyncio.CancelledError, Exception):
        pass

    return up_ask, down_ask


async def main() -> None:
    args = parse_args()

    print("=" * 70)
    print(f"  Polymarket smoke test — BUY ${args.usd:.2f} on {args.asset.upper()} {args.side.upper()}")
    print(f"  Mode: {'DRY RUN (no order placed)' if args.dry_run else 'LIVE (real order)'}")
    print("=" * 70)

    # --- Sanity check credentials ---
    if not args.dry_run:
        for name, val in [("PRIVATE_KEY", PRIVATE_KEY), ("API_KEY", API_KEY),
                          ("API_SECRET", API_SECRET), ("API_PASSPHRASE", API_PASSPHRASE)]:
            if not val:
                fail(f"{name} is not set in .env")
        if not FUNDER:
            print("WARNING: FUNDER not set — orders will sign with PRIVATE_KEY's own address")

    # --- Resolve market ---
    win_ts = current_window_ts()
    print(f"\n[1/4] Resolving market for window_ts={win_ts}…")
    market = fetch_market(args.asset, win_ts)
    if not market:
        fail(f"Could not resolve {args.asset} market for window_ts={win_ts}")
    print(f"      slug={market['slug']}")
    print(f"      up_token={market['up_token_id'][:16]}…")
    print(f"      down_token={market['down_token_id'][:16]}…")
    print(f"      tick_size={market['tick_size']}")

    # --- Read best ask via WS ---
    print(f"\n[2/4] Connecting to CLOB WS for live book…")
    up_ask, down_ask = await get_best_ask(
        args.asset, market["up_token_id"], market["down_token_id"], args.book_wait,
    )
    print(f"      Up ask:   {up_ask:.4f}")
    print(f"      Down ask: {down_ask:.4f}")

    target_token = market["up_token_id"] if args.side == "up" else market["down_token_id"]
    target_price = up_ask if args.side == "up" else down_ask
    if target_price <= 0 or target_price >= 1:
        fail(f"Bad ask price: {target_price}")

    expected_shares = round(args.usd / target_price, 2)
    print(f"\n[3/4] Order plan:")
    print(f"      side={args.side}  size_usd=${args.usd:.2f}  price={target_price:.4f}")
    print(f"      → {expected_shares} shares of {args.side.upper()} at {target_price:.4f}")

    # --- Place order ---
    print(f"\n[4/4] {'(DRY RUN — skipping)' if args.dry_run else 'Placing order…'}")
    client = OrderClient(
        private_key=PRIVATE_KEY,
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
        clob_host=CLOB_HOST,
        chain_id=CHAIN_ID,
        dry_run=args.dry_run,
        funder=FUNDER or None,
    )

    order_id = client.place_order(
        token_id=target_token,
        side="BUY",
        size_usd=args.usd,
        price=target_price,
    )

    if not order_id:
        fail("place_order returned None — see logs above for the API error")

    # --- Save details so test_trade_sell.py can find the position ---
    details = {
        "ts":          int(time.time()),
        "asset":       args.asset,
        "side":        args.side,
        "token_id":    target_token,
        "size_usd":    args.usd,
        "size_shares": expected_shares,
        "entry_price": target_price,
        "order_id":    order_id,
        "window_ts":   win_ts,
        "slug":        market["slug"],
        "dry_run":     args.dry_run,
    }
    LAST_ORDER_FILE.write_text(json.dumps(details, indent=2))

    print()
    print("=" * 70)
    print(f"  ORDER ID: {order_id}")
    print(f"  Saved details to {LAST_ORDER_FILE}")
    print(f"  Run: python scripts/test_trade_sell.py    # to close the position")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
