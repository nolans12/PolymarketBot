"""
binance_feed.py — Binance bookTicker stream for BTC spot.

The bookTicker stream emits one event on every top-of-book change with
the fields needed for microprice: b (bid price), a (ask price), B (bid
qty), A (ask qty). No auth required.

NOTE: Binance is geo-blocked from US IPs. Set SPOT_SOURCE=binance only
if running from outside the US or behind a non-US VPN.
"""

import asyncio
import json
import time

import websockets

from betbot.kalshi.book import SpotBook
from betbot.kalshi.config import BINANCE_WS


class BinanceFeed:
    """
    Subscribes to Binance's bookTicker stream for BTCUSDT.
    Applies every top-of-book change to the shared SpotBook.
    Reconnects automatically on disconnect.
    """

    MAX_HZ = 20.0   # cap ingest rate; bookTicker fires very fast on BTCUSDT

    def __init__(self, book: SpotBook):
        self._book    = book
        self._running = False

    async def run(self) -> None:
        self._running = True
        min_interval = 1.0 / self.MAX_HZ
        last_t = 0.0

        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15,
                ) as ws:
                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        bid_s = msg.get("b")
                        ask_s = msg.get("a")
                        bsz_s = msg.get("B")
                        asz_s = msg.get("A")
                        if not (bid_s and ask_s and bsz_s and asz_s):
                            continue

                        # Rate-limit
                        now = time.time()
                        if now - last_t < min_interval:
                            continue
                        last_t = now

                        try:
                            bid = float(bid_s)
                            ask = float(ask_s)
                            bsz = float(bsz_s)
                            asz = float(asz_s)
                            if bid <= 0 or ask <= 0 or bid >= ask:
                                continue
                        except (ValueError, TypeError):
                            continue

                        # Use mid as the trade-price proxy — bookTicker
                        # doesn't carry last_trade.
                        price = (bid + ask) / 2.0
                        self._book.apply_ticker(price, bid, ask, bsz, asz)

            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                await asyncio.sleep(2)

    def stop(self) -> None:
        self._running = False
