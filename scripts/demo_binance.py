"""
demo_binance.py — Live Binance feed demo.

Prints microprice, spread, and OFI for BTC and ETH as fast as ticks arrive.

Usage:
    python scripts/demo_binance.py
    python scripts/demo_binance.py --hz 10    # cap print rate per asset
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Make sure the repo root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.clients.binance_ws import BinanceWS
from polybot.infra.config import ASSETS
from polybot.state.spot_book import SpotBook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binance live feed demo")
    p.add_argument("--hz", type=float, default=5.0,
                   help="Max print rate per asset in Hz (default 5)")
    p.add_argument("--verbose", action="store_true",
                   help="Show INFO-level websocket logs")
    return p.parse_args()


async def main(max_hz: float) -> None:
    books: dict[str, SpotBook] = {asset: SpotBook(asset) for asset in ASSETS}
    last_print: dict[str, float] = {asset: 0.0 for asset in ASSETS}
    min_interval = 1.0 / max_hz

    header_printed = False

    def on_update(asset: str, book: SpotBook) -> None:
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
                f"{'ofi_l1':>10} | {'ofi_l5':>10} | {'stale_ms':>8}"
            )
            print("-" * 120)
            header_printed = True

        ofi_l1 = book._ofi_l1_acc   # peek — don't drain in demo
        ofi_l5 = book._ofi_l5_acc
        stale = book.stale_ms()

        print(
            f"{asset.upper():>4} | "
            f"{book.best_bid:>12.2f} {book.best_bid_size:>9.4f} | "
            f"{book.best_ask:>12.2f} {book.best_ask_size:>9.4f} | "
            f"{book.microprice:>12.4f} | "
            f"{book.spread:>8.4f} | "
            f"{ofi_l1:>+10.1f} | "
            f"{ofi_l5:>+10.1f} | "
            f"{stale:>8d}",
            flush=True,
        )

    client = BinanceWS(books=books, on_update=on_update)
    print(f"Connecting to Binance (assets: {', '.join(ASSETS)}, max {max_hz:.0f} Hz per asset)…",
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
