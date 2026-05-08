"""
kalshi_ws_feed.py — Live Kalshi orderbook over WebSocket.

Subscribes to the orderbook_delta channel and feeds yes_bid/yes_ask into
KalshiBook in real time (sub-100ms latency vs the 100ms REST poll).

Drop-in replacement for KalshiRestFeed. Same public surface:
    feed = KalshiWsFeed(book, key_id=..., pk=...)
    await feed.run()
    feed.update_ticker("KX...")
    feed.stop()

Robustness:
- 10s message watchdog (auto-reconnect on silence)
- Verifies first snapshot arrives within 10s of subscribe (else reconnect)
- Tracks seq numbers to detect gaps; on gap → reconnect for fresh snapshot
- Falls back to a clean book reset on every reconnect
"""

import asyncio
import json
import time
from typing import Optional

import websockets

from betbot.kalshi.auth import auth_headers, load_private_key
from betbot.kalshi.book import KalshiBook
from betbot.kalshi.config import KALSHI_KEY_ID

KALSHI_WS_URL  = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH        = "/trade-api/ws/v2"
SILENCE_LIMIT  = 10.0   # seconds of silence before reconnecting
SNAPSHOT_LIMIT = 10.0   # max wait for first snapshot after subscribe


class KalshiWsFeed:
    def __init__(self, book: KalshiBook, key_id: str = KALSHI_KEY_ID, pk=None):
        if pk is None:
            pk = load_private_key()
        self._book    = book
        self._key_id  = key_id
        self._pk      = pk
        self._ticker  = book.ticker
        self._running = False
        self._cmd_id  = 0
        self._last_seq: Optional[int] = None
        self._reconnect_event = asyncio.Event()

    def update_ticker(self, new_ticker: str) -> None:
        self._ticker = new_ticker
        self._last_seq = None
        # Force the current connection to reconnect with new subscription
        self._reconnect_event.set()

    def stop(self) -> None:
        self._running = False
        self._reconnect_event.set()

    async def run(self) -> None:
        self._running = True
        backoff = 1.0
        while self._running:
            try:
                await self._connect_and_stream()
                backoff = 1.0  # reset on successful loop
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"  [Kalshi WS] {type(e).__name__}: {e} — reconnecting in {backoff:.0f}s",
                      flush=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 15.0)

    async def _connect_and_stream(self) -> None:
        ticker = self._ticker
        if not ticker:
            await asyncio.sleep(1.0)
            return

        hdrs = auth_headers(self._pk, self._key_id, "GET", WS_PATH)

        async with websockets.connect(
            KALSHI_WS_URL,
            additional_headers=hdrs,
            ping_interval=10,
            ping_timeout=5,
            open_timeout=10,
            close_timeout=5,
        ) as ws:
            # Reset book state on every fresh connect — we'll get a snapshot
            self._book._yes_book.clear()
            self._book._no_book.clear()
            self._last_seq = None

            self._cmd_id += 1
            sub = {
                "id":  self._cmd_id,
                "cmd": "subscribe",
                "params": {
                    "channels":       ["orderbook_delta"],
                    "market_tickers": [ticker],
                },
            }
            await ws.send(json.dumps(sub))

            self._reconnect_event.clear()
            got_snapshot = False
            sub_t0 = time.monotonic()

            while self._running and not self._reconnect_event.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=SILENCE_LIMIT)
                except asyncio.TimeoutError:
                    print(f"  [Kalshi WS] silent >{SILENCE_LIMIT:.0f}s on {ticker} — reconnecting",
                          flush=True)
                    return

                # Check we got a snapshot in time
                if not got_snapshot and time.monotonic() - sub_t0 > SNAPSHOT_LIMIT:
                    print(f"  [Kalshi WS] no snapshot for {ticker} after {SNAPSHOT_LIMIT:.0f}s — reconnecting",
                          flush=True)
                    return

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                mtype = msg.get("type", "")
                seq   = msg.get("seq")

                if mtype == "subscribed":
                    continue

                if mtype == "error":
                    print(f"  [Kalshi WS] error: {msg}", flush=True)
                    return

                # Sequence-gap detection
                if seq is not None:
                    if self._last_seq is not None and seq != self._last_seq + 1:
                        print(f"  [Kalshi WS] seq gap on {ticker} ({self._last_seq} -> {seq}) — reconnecting",
                              flush=True)
                        return
                    self._last_seq = seq

                payload = msg.get("msg") or {}

                if mtype == "orderbook_snapshot":
                    yes_fp = payload.get("yes_dollars_fp") or []
                    no_fp  = payload.get("no_dollars_fp")  or []
                    self._book.apply_snapshot(yes_fp, no_fp)
                    got_snapshot = True
                    continue

                if mtype == "orderbook_delta":
                    side       = payload.get("side")          # "yes" or "no"
                    price_str  = payload.get("price_dollars")
                    delta_str  = payload.get("delta_fp")
                    if side and price_str is not None and delta_str is not None:
                        try:
                            price = float(price_str)
                            delta = float(delta_str)
                            self._book.apply_delta(side, price, delta)
                        except (TypeError, ValueError):
                            pass

            # Reconnect-event triggered (new ticker on rollover) — exit cleanly
            self._reconnect_event.clear()
