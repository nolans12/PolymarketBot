"""
test_trade_sell.py — Close the position from test_trade_buy.py at the live bid.

Reads .last_test_order.json to find the position. Connects to the CLOB WS,
reads the current best bid for that token, and places a SELL order at that
price for the same number of shares. This is the "cash-out exit" the bot
would do in production when the lag-arb edge has compressed.

Usage:
    python scripts/test_trade_sell.py            # close the most recent test position
    python scripts/test_trade_sell.py --dry-run  # print intent only
    python scripts/test_trade_sell.py --shares 10.5  # override share count
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
    p = argparse.ArgumentParser(description="Close a position opened by test_trade_buy.py")
    p.add_argument("--dry-run", action="store_true",
                   help="Print intent without placing the order")
    p.add_argument("--shares", type=float, default=None,
                   help="Override share count (default: read from .last_test_order.json)")
    p.add_argument("--book-wait", type=float, default=5.0,
                   help="Seconds to wait for first WS book snapshot (default 5)")
    p.add_argument("--order-file", default=str(LAST_ORDER_FILE),
                   help="Path to the saved buy details JSON")
    return p.parse_args()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


async def get_best_bid(token_id: str, other_token: str,
                       wait_s: float) -> float:
    """Subscribe to the CLOB WS and return the best bid for token_id."""
    asset = "btc"  # tag only — books dict is keyed by this
    books = {asset: PolyBook(asset)}
    # We subscribe to both tokens because the CLOB WS expects both sides.
    # The token we care about may be either Up or Down; PolyBook stores them
    # in the same object regardless.
    books[asset].set_tokens(token_id, other_token)

    got_book = asyncio.Event()

    def on_update(_asset: str, book: PolyBook) -> None:
        if book.up and book.up.token_id == token_id and book.up.best_bid > 0:
            got_book.set()
        elif book.down and book.down.token_id == token_id and book.down.best_bid > 0:
            got_book.set()

    ws = PolymarketWS(books=books, on_update=on_update)
    ws.subscribe(asset, token_id, other_token)

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
        fail(f"No book snapshot for token {token_id[:16]}… within {wait_s}s")

    book = books[asset]
    if book.up and book.up.token_id == token_id:
        bid = book.up.best_bid
    elif book.down and book.down.token_id == token_id:
        bid = book.down.best_bid
    else:
        bid = 0.0

    ws.stop()
    run_task.cancel()
    try:
        await run_task
    except (asyncio.CancelledError, Exception):
        pass

    return bid


async def main() -> None:
    args = parse_args()
    order_file = Path(args.order_file)
    if not order_file.exists():
        fail(f"{order_file} not found — run test_trade_buy.py first")

    details = json.loads(order_file.read_text())
    asset       = details["asset"]
    token_id    = details["token_id"]
    side        = details["side"]
    size_shares = args.shares if args.shares is not None else details["size_shares"]
    entry_price = details["entry_price"]
    buy_order   = details.get("order_id", "?")
    was_dry_run = details.get("dry_run", False)

    # Find the OTHER token (the opposite side of the same window)
    # We need it to subscribe to the WS even though we're only selling one side.
    if side == "up":
        # The buy script saved up's token; we need the down token. Reload from market.
        from polybot.clients.polymarket_rest import fetch_market
        market = fetch_market(asset, details["window_ts"])
        if not market:
            fail("Could not re-resolve market — has the window already closed?")
        other_token = market["down_token_id"]
    else:
        from polybot.clients.polymarket_rest import fetch_market
        market = fetch_market(asset, details["window_ts"])
        if not market:
            fail("Could not re-resolve market — has the window already closed?")
        other_token = market["up_token_id"]

    print("=" * 70)
    print(f"  Polymarket smoke test — SELL (cash-out)")
    print(f"  Mode: {'DRY RUN (no order placed)' if args.dry_run else 'LIVE (real order)'}")
    print("=" * 70)
    print(f"\nLoaded position from {order_file}:")
    print(f"  asset={asset}  side={side}  shares={size_shares}  entry={entry_price:.4f}")
    print(f"  buy_order_id={buy_order}{'  (was dry-run)' if was_dry_run else ''}")

    if was_dry_run and not args.dry_run:
        print("\nWARNING: the BUY was a dry run, so there's no real position to close.")
        print("         Re-run with --dry-run, or run a real BUY first.")
        return

    print(f"\n[1/3] Connecting to CLOB WS for live bid…")
    bid = await get_best_bid(token_id, other_token, args.book_wait)
    if bid <= 0 or bid >= 1:
        fail(f"Bad bid price: {bid}")
    print(f"      Best bid: {bid:.4f}")

    proceeds = size_shares * bid
    pnl_gross = proceeds - details["size_usd"]
    print(f"\n[2/3] Cash-out plan:")
    print(f"      sell {size_shares} shares at {bid:.4f}")
    print(f"      gross proceeds: ${proceeds:.2f}  (entry was ${details['size_usd']:.2f})")
    print(f"      gross P&L (pre-fee): ${pnl_gross:+.2f}")

    print(f"\n[3/3] {'(DRY RUN — skipping)' if args.dry_run else 'Placing SELL order…'}")
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

    sell_order_id = client.place_order(
        token_id=token_id,
        side="SELL",
        size_usd=proceeds,
        price=bid,
    )

    if not sell_order_id:
        fail("place_order returned None — see logs above for the API error")

    # Append to the order file so we have full round-trip history
    details["sell"] = {
        "ts":            int(time.time()),
        "exit_price":    bid,
        "exit_shares":   size_shares,
        "exit_proceeds": proceeds,
        "pnl_gross":     pnl_gross,
        "order_id":      sell_order_id,
        "dry_run":       args.dry_run,
    }
    order_file.write_text(json.dumps(details, indent=2))

    print()
    print("=" * 70)
    print(f"  SELL ORDER ID: {sell_order_id}")
    print(f"  Round-trip details written back to {order_file}")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
