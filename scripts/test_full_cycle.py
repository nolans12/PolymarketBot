"""
test_full_cycle.py — End-to-end smoke test: buy, wait, sell.

Places a small BUY, waits a few seconds, then closes the position with a
SELL at the live bid. This is the closest single-script approximation of
what the bot does in production: enter on edge, exit on lag-close.

Usage:
    python scripts/test_full_cycle.py                    # $5 BTC Up, hold 30s, sell
    python scripts/test_full_cycle.py --hold 10          # hold 10s instead of 30s
    python scripts/test_full_cycle.py --asset eth --usd 5
    python scripts/test_full_cycle.py --dry-run          # plan only, no orders

Safety:
  - Default size is $5 (the smallest sensible Polymarket size)
  - Default --hold is 30s; with both BUY + SELL fees the round-trip costs
    ~5-7% of size, so the cash-out price has to move very little before
    P&L flips negative. Don't be alarmed by a small loss.
  - --dry-run does the full plan without placing any order.
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

CYCLE_FILE = Path(".last_full_cycle.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Buy → wait → sell smoke test")
    p.add_argument("--asset", default="btc", choices=["btc", "eth"])
    p.add_argument("--side",  default="up",  choices=["up", "down"])
    p.add_argument("--usd",   type=float, default=5.0)
    p.add_argument("--hold",  type=float, default=30.0,
                   help="Seconds to hold the position before selling (default 30)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--book-wait", type=float, default=5.0)
    return p.parse_args()


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


class LiveBook:
    """Keeps a CLOB WS connection alive so we can read live bid/ask repeatedly."""

    def __init__(self, asset: str, up_token: str, down_token: str):
        self.asset = asset
        self.books = {asset: PolyBook(asset)}
        self.books[asset].set_tokens(up_token, down_token)
        self._ready = asyncio.Event()
        self._ws = PolymarketWS(books=self.books, on_update=self._on_update)
        self._ws.subscribe(asset, up_token, down_token)
        self._task: asyncio.Task | None = None

    def _on_update(self, _asset: str, book: PolyBook) -> None:
        if (book.up and book.up.best_ask < float("inf") and
            book.down and book.down.best_ask < float("inf")):
            self._ready.set()

    async def start(self, wait_s: float) -> None:
        self._task = asyncio.create_task(self._ws.run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=wait_s)
        except asyncio.TimeoutError:
            await self.stop()
            fail(f"No book snapshot within {wait_s}s")

    async def stop(self) -> None:
        self._ws.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    @property
    def book(self) -> PolyBook:
        return self.books[self.asset]


async def main() -> None:
    args = parse_args()

    print("=" * 70)
    print(f"  Polymarket smoke test — FULL CYCLE")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE (real money)'}")
    print(f"  Plan: BUY ${args.usd:.2f} {args.asset.upper()} {args.side.upper()} → "
          f"hold {args.hold:.0f}s → SELL")
    print("=" * 70)

    if not args.dry_run:
        for name, val in [("PRIVATE_KEY", PRIVATE_KEY), ("API_KEY", API_KEY),
                          ("API_SECRET", API_SECRET), ("API_PASSPHRASE", API_PASSPHRASE)]:
            if not val:
                fail(f"{name} is not set in .env")

    # --- Resolve market ---
    win_ts = current_window_ts()
    print(f"\n[1/6] Resolving market window_ts={win_ts}…")
    market = fetch_market(args.asset, win_ts)
    if not market:
        fail(f"Could not resolve {args.asset} market for window_ts={win_ts}")
    print(f"      slug={market['slug']}")

    target_token = market["up_token_id"] if args.side == "up" else market["down_token_id"]
    other_token  = market["down_token_id"] if args.side == "up" else market["up_token_id"]

    # --- Open WS feed ---
    print(f"\n[2/6] Connecting to CLOB WS…")
    feed = LiveBook(args.asset, market["up_token_id"], market["down_token_id"])
    await feed.start(args.book_wait)

    def best_ask_for_target() -> float:
        b = feed.book
        if args.side == "up":
            return b.up.best_ask if b.up else 0.0
        return b.down.best_ask if b.down else 0.0

    def best_bid_for_target() -> float:
        b = feed.book
        if args.side == "up":
            return b.up.best_bid if b.up else 0.0
        return b.down.best_bid if b.down else 0.0

    entry_price = best_ask_for_target()
    if entry_price <= 0 or entry_price >= 1:
        await feed.stop()
        fail(f"Bad ask price: {entry_price}")

    expected_shares = round(args.usd / entry_price, 2)
    print(f"      entry ask = {entry_price:.4f}  → {expected_shares} shares")

    client = OrderClient(
        private_key=PRIVATE_KEY,
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
        clob_host=CLOB_HOST,
        chain_id=CHAIN_ID,
        dry_run=args.dry_run,
    )

    # --- BUY ---
    print(f"\n[3/6] Placing BUY…")
    buy_id = client.place_order(
        token_id=target_token,
        side="BUY",
        size_usd=args.usd,
        price=entry_price,
    )
    if not buy_id:
        await feed.stop()
        fail("BUY failed — check logs above")
    print(f"      buy_order_id={buy_id}")

    # --- Hold ---
    print(f"\n[4/6] Holding {args.hold:.0f}s — watching the book…")
    t_end = time.time() + args.hold
    while time.time() < t_end:
        bid = best_bid_for_target()
        ask = best_ask_for_target()
        remaining = int(t_end - time.time())
        print(f"      [{remaining:>3}s left] bid={bid:.4f} ask={ask:.4f}",
              flush=True)
        await asyncio.sleep(min(5.0, max(1.0, remaining)))

    # --- SELL ---
    exit_bid = best_bid_for_target()
    if exit_bid <= 0 or exit_bid >= 1:
        await feed.stop()
        fail(f"Bad bid price at exit: {exit_bid}")

    proceeds = expected_shares * exit_bid
    pnl_gross = proceeds - args.usd

    print(f"\n[5/6] Placing SELL at bid={exit_bid:.4f}…")
    print(f"      gross P&L (pre-fee): ${pnl_gross:+.2f} on ${args.usd:.2f}")
    sell_id = client.place_order(
        token_id=target_token,
        side="SELL",
        size_usd=proceeds,
        price=exit_bid,
    )

    await feed.stop()

    if not sell_id:
        fail("SELL failed — check logs above. Position may still be open!")

    # --- Save round-trip ---
    details = {
        "ts":           int(time.time()),
        "asset":        args.asset,
        "side":         args.side,
        "token_id":     target_token,
        "size_usd":     args.usd,
        "size_shares":  expected_shares,
        "entry_price":  entry_price,
        "exit_price":   exit_bid,
        "hold_seconds": args.hold,
        "buy_order_id":  buy_id,
        "sell_order_id": sell_id,
        "pnl_gross":    pnl_gross,
        "dry_run":      args.dry_run,
    }
    CYCLE_FILE.write_text(json.dumps(details, indent=2))

    print()
    print("=" * 70)
    print(f"  buy_order_id  = {buy_id}")
    print(f"  sell_order_id = {sell_id}")
    print(f"  entry={entry_price:.4f}  exit={exit_bid:.4f}  "
          f"gross P&L=${pnl_gross:+.2f}")
    print(f"  details written to {CYCLE_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted — if a BUY went through, run test_trade_sell.py to close it.")
