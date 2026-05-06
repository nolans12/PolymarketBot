"""
coinbase_feed.py — Coinbase Advanced Trade WebSocket ticker feed.

Self-contained: no dependency on betbot.clients.coinbase_ws (which has
broken polybot.* imports). Feeds a CoinbaseBook at up to 20 Hz.
"""

import asyncio
import json
import time

import websockets

from betbot.kalshi.book import CoinbaseBook
from betbot.kalshi.config import COINBASE_WS, COINBASE_PRODUCT


class CoinbaseFeed:
    """
    Subscribes to Coinbase Advanced Trade ticker channel for BTC-USD.
    Applies every tick to the shared CoinbaseBook.
    Reconnects automatically on disconnect.
    """

    MAX_HZ = 20.0   # cap ingest rate; Coinbase fires on every trade

    def __init__(self, book: CoinbaseBook,
                 product: str = COINBASE_PRODUCT):
        self._book    = book
        self._product = product
        self._running = False

    async def run(self) -> None:
        self._running = True
        sub = {
            "type":        "subscribe",
            "product_ids": [self._product],
            "channel":     "ticker",
        }
        min_interval = 1.0 / self.MAX_HZ
        last_t = 0.0

        while self._running:
            try:
                async with websockets.connect(
                    COINBASE_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15,
                ) as ws:
                    await ws.send(json.dumps(sub))

                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        if msg.get("channel") != "ticker":
                            continue

                        for ev in msg.get("events", []):
                            for tick in ev.get("tickers", []):
                                if tick.get("product_id") != self._product:
                                    continue

                                # Rate-limit
                                now = time.time()
                                if now - last_t < min_interval:
                                    continue
                                last_t = now

                                price_s = tick.get("price")
                                bid_s   = tick.get("best_bid")
                                ask_s   = tick.get("best_ask")
                                bsz_s   = tick.get("best_bid_quantity")
                                asz_s   = tick.get("best_ask_quantity")
                                if not price_s:
                                    continue
                                try:
                                    price = float(price_s)
                                    bid   = float(bid_s)   if bid_s  else price
                                    ask   = float(ask_s)   if ask_s  else price
                                    bsz   = float(bsz_s)   if bsz_s  else 1.0
                                    asz   = float(asz_s)   if asz_s  else 1.0
                                    if bid <= 0 or ask <= 0 or bid >= ask:
                                        continue
                                except (ValueError, TypeError):
                                    continue

                                self._book.apply_ticker(price, bid, ask, bsz, asz)

            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                await asyncio.sleep(2)

    def stop(self) -> None:
        self._running = False
