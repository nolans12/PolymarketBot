"""
spot_feed.py — Protocol contract for any spot-price feed.

A SpotFeed is anything that runs in the asyncio loop, ingests upstream
ticker data, and pushes (price, bid, ask, bid_size, ask_size) tuples
into a shared SpotBook via book.apply_ticker(). Concrete implementations
live in coinbase_feed.py and binance_feed.py.
"""

from typing import Protocol


class SpotFeed(Protocol):
    """Minimal contract every spot feed must satisfy."""

    async def run(self) -> None:
        """Run the feed loop until stop() is called. Reconnects on disconnect."""
        ...

    def stop(self) -> None:
        """Signal the run() loop to exit on its next iteration."""
        ...
