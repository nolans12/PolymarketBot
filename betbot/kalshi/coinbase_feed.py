"""
coinbase_feed.py — Coinbase Advanced Trade WebSocket ticker feed.

Single-product mode: CoinbaseFeed(book, product="BTC-USD")
Multi-product mode:  CoinbaseFeed(books={"BTC-USD": btc_book, "ETH-USD": eth_book, ...})

One WebSocket connection handles all subscribed products; each tick is
routed to the matching SpotBook. Reconnects automatically on disconnect.
"""

import asyncio
import json
import time
from typing import Optional

import websockets

from betbot.kalshi.book import SpotBook
from betbot.kalshi.config import COINBASE_WS, COINBASE_PRODUCT


class CoinbaseFeed:
    """
    Subscribes to Coinbase Advanced Trade ticker channel.
    Supports one or multiple products over a single WebSocket connection.

    Single-asset:  CoinbaseFeed(book, product="BTC-USD")
    Multi-asset:   CoinbaseFeed(books={"BTC-USD": btc_book, "ETH-USD": eth_book})
    """

    MAX_HZ = 20.0   # per-product rate cap

    def __init__(self, book: Optional[SpotBook] = None,
                 product: str = COINBASE_PRODUCT,
                 books: Optional[dict[str, SpotBook]] = None):
        if books is not None:
            self._books = books
        else:
            self._books = {product: book}
        self._running = False
        # Per-product last-ingest timestamp for rate-limiting
        self._last_t: dict[str, float] = {p: 0.0 for p in self._books}

    async def run(self) -> None:
        self._running = True
        products = list(self._books.keys())
        sub = {
            "type":        "subscribe",
            "product_ids": products,
            "channel":     "ticker",
        }
        min_interval = 1.0 / self.MAX_HZ

        while self._running:
            try:
                async with websockets.connect(
                    COINBASE_WS,
                    ping_interval=10,
                    ping_timeout=5,
                    open_timeout=10,
                    close_timeout=5,
                ) as ws:
                    await ws.send(json.dumps(sub))

                    while self._running:
                        # Watchdog: if no message in 15s, force a reconnect
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        except asyncio.TimeoutError:
                            print("  *** Coinbase WS silent >15s, reconnecting ***", flush=True)
                            break
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        if msg.get("channel") != "ticker":
                            continue

                        for ev in msg.get("events", []):
                            for tick in ev.get("tickers", []):
                                pid = tick.get("product_id", "")
                                book = self._books.get(pid)
                                if book is None:
                                    continue

                                # Per-product rate-limit
                                now = time.time()
                                if now - self._last_t.get(pid, 0.0) < min_interval:
                                    continue
                                self._last_t[pid] = now

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

                                book.apply_ticker(price, bid, ask, bsz, asz)

            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                await asyncio.sleep(2)

    def stop(self) -> None:
        self._running = False
