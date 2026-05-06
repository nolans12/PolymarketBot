"""
demo_coinbase.py — Live Coinbase BTC/ETH price feed demo.

Prints microprice, spread, bid/ask with sizes as fast as ticks arrive.

Usage:
    python scripts/demo_coinbase.py
    python scripts/demo_coinbase.py --hz 10    # cap print rate per asset
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.clients.coinbase_ws import CoinbaseWS
from polybot.infra.config import ASSETS
from polybot.state.coinbase_book import CoinbaseBook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coinbase live feed demo")
    p.add_argument("--hz", type=float, default=5.0,
                   help="Max print rate per asset in Hz (default 5)")
    p.add_argument("--verbose", action="store_true",
                   help="Show INFO-level websocket logs")
    return p.parse_args()


async def main(max_hz: float) -> None:
    books: dict[str, CoinbaseBook] = {asset: CoinbaseBook(asset) for asset in ASSETS}
    last_print: dict[str, float] = {asset: 0.0 for asset in ASSETS}
    min_interval = 1.0 / max_hz

    header_printed = False

    def on_update(asset: str, book: CoinbaseBook) -> None:
        nonlocal header_printed
        now = time.time()
        if now - last_print.get(asset, 0.0) < min_interval:
            return
        last_print[asset] = now

        if not book.ready:
            return

        if not header_printed:
            print(
                f"{'asset':>4} | {'bid':>12} {'bsz':>9} | "
                f"{'ask':>12} {'asz':>9} | "
                f"{'microprice':>12} | {'spread':>8} | "
                f"{'last_trade':>12} | {'stale_ms':>8}"
            )
            print("-" * 110)
            header_printed = True

        spread = book.best_ask - book.best_bid
        stale  = book.stale_ms()

        print(
            f"{asset.upper():>4} | "
            f"{book.best_bid:>12.2f} {book.best_bid_size:>9.4f} | "
            f"{book.best_ask:>12.2f} {book.best_ask_size:>9.4f} | "
            f"{book.microprice:>12.4f} | "
            f"{spread:>8.4f} | "
            f"{book.last_trade_price:>12.2f} | "
            f"{stale:>8d}",
            flush=True,
        )

    client = CoinbaseWS(books=books, on_update=on_update)
    print(f"Connecting to Coinbase (assets: {', '.join(ASSETS)}, max {max_hz:.0f} Hz per asset)…",
          file=sys.stderr)

    try:
        await client.run()
    except KeyboardInterrupt:
        client.stop()
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    asyncio.run(main(args.hz))
