"""
kalshi_rest_feed.py - Polls Kalshi REST API for live market quotes.

Drop-in replacement for the WebSocket feed. Uses the SAME endpoint and auth
that test_trade.py uses (and which is proven to work). Polls every second.

Endpoint:  GET /trade-api/v2/markets/{ticker}
Response:  {"market": {"yes_bid_dollars": "0.84", "yes_ask_dollars": "0.86", ...}}

This eliminates all WebSocket heartbeat / stale-connection issues.
Kalshi's rate limit is 20 reads/sec on basic tier; 1 Hz polling is safe.
"""

import asyncio
import time
from typing import Optional

import aiohttp

from betbot.kalshi.auth import auth_headers, load_private_key
from betbot.kalshi.book import KalshiBook
from betbot.kalshi.config import KALSHI_REST, KALSHI_KEY_ID

POLL_INTERVAL_S = 1.0    # poll every 1s — Kalshi limit is 20 reads/sec
MAX_FAILURES    = 5      # consecutive failures before forcing a session reset


class KalshiRestFeed:
    """
    Live Kalshi market quote feed via REST polling.

    Polls GET /trade-api/v2/markets/{ticker} every second and feeds yes_bid /
    yes_ask into KalshiBook via apply_ticker_update().

    Usage:
        feed = KalshiRestFeed(book, key_id=..., pk=...)
        await feed.run()                 # runs until stop()
        feed.update_ticker("KX...")      # switch markets on window rollover
        feed.stop()
    """

    def __init__(self, book: KalshiBook, key_id: str = KALSHI_KEY_ID, pk=None):
        if pk is None:
            pk = load_private_key()

        self._book    = book
        self._key_id  = key_id
        self._pk      = pk
        self._ticker  = book.ticker
        self._running = False
        self._fails   = 0

    def update_ticker(self, new_ticker: str) -> None:
        """Switch which market we're polling (called on window rollover)."""
        self._ticker = new_ticker
        self._fails  = 0

    def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        self._running = True

        while self._running:
            session = aiohttp.ClientSession()
            try:
                while self._running:
                    t0 = time.monotonic()
                    ticker = self._ticker
                    if not ticker:
                        await asyncio.sleep(1.0)
                        continue

                    ok = await self._poll_once(session, ticker)
                    if ok:
                        self._fails = 0
                    else:
                        self._fails += 1
                        if self._fails >= MAX_FAILURES:
                            print(f"  [Kalshi REST] {self._fails} consecutive failures - "
                                  f"resetting HTTP session", flush=True)
                            self._fails = 0
                            break  # break inner -> close+recreate session

                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0.0, POLL_INTERVAL_S - elapsed))
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"  [Kalshi REST] outer error: {type(e).__name__}: {e}",
                      flush=True)
            finally:
                try:
                    await session.close()
                except Exception:
                    pass
            await asyncio.sleep(0.5)

    async def _poll_once(self, session: aiohttp.ClientSession, ticker: str) -> bool:
        path    = f"/trade-api/v2/markets/{ticker}"
        url     = KALSHI_REST + path
        headers = auth_headers(self._pk, self._key_id, "GET", path)

        try:
            async with session.get(url, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 429:
                    # Rate-limited: back off a bit
                    await asyncio.sleep(2.0)
                    return False
                if r.status != 200:
                    text = await r.text()
                    print(f"  [Kalshi REST] HTTP {r.status} on {ticker}: {text[:120]}",
                          flush=True)
                    return False
                data = await r.json()
        except asyncio.TimeoutError:
            print(f"  [Kalshi REST] timeout polling {ticker}", flush=True)
            return False
        except Exception as e:
            print(f"  [Kalshi REST] {type(e).__name__}: {e}", flush=True)
            return False

        mkt = data.get("market") or data
        bid = mkt.get("yes_bid_dollars")
        ask = mkt.get("yes_ask_dollars")
        if bid is None or ask is None:
            return False

        try:
            yes_bid = float(bid)
            yes_ask = float(ask)
        except (TypeError, ValueError):
            return False

        # Use ticker_update path: it bypasses the orderbook reconstruction
        # entirely and stamps last_update_ns. This is what we want.
        if yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid:
            # Allow zero-spread (yes_bid == yes_ask) by hand
            self._book.yes_bid = yes_bid
            self._book.yes_ask = yes_ask
            self._book.yes_mid = (yes_bid + yes_ask) / 2.0
            self._book.last_update_ns = time.time_ns()
            self._book.ready = True
            now_s = int(self._book.last_update_ns // 1_000_000_000)
            if now_s > self._book._last_ring_s:
                self._book._ring.append((now_s, self._book.yes_mid))
                self._book._last_ring_s = now_s
        return True
