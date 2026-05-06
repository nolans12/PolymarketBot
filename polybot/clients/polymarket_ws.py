"""
polymarket_ws.py — Polymarket CLOB market WebSocket client.

Subscribes to Up + Down tokens for all configured assets.
Implements the mandatory 60s silent-freeze watchdog (py-clob-client #292).

Protocol (Polymarket WS docs):
  - Connect to wss://ws-subscriptions-clob.polymarket.com/ws/market
  - Send subscribe message for each asset's token pair
  - Keep-alive: send literal "PING" every 10s; server sends "PING" every 5s,
    reply with "PONG" within 10s
  - Events: book (full snapshot), price_change (incremental), last_trade_price,
    tick_size_change
  - Silent-freeze bug: if no book/price_change in 60s, force reconnect.
    If reconnect fails twice consecutively, fall back to REST polling.

Token IDs are set by window.py at each window rollover via set_tokens().
The WS client then resubscribes for the new token pair.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets

from polybot.infra.config import POLY_CLOB_WS, ASSETS
from polybot.state.poly_book import PolyBook

logger = logging.getLogger(__name__)

PING_INTERVAL_S      = 10.0   # we send PING this often
WATCHDOG_TIMEOUT_S   = 60.0   # reconnect if no book/price_change for this long
RECONNECT_DELAY_S    = 2.0
MAX_RECONNECT_BEFORE_REST = 2  # after this many consecutive fails, log degraded mode


class PolymarketWS:
    """
    Manages the Polymarket CLOB WebSocket for all active asset token pairs.

    Usage:
        ws = PolymarketWS(books)
        asyncio.create_task(ws.run())
        # later, when a new window opens:
        ws.subscribe(asset='btc', up_token='...', down_token='...')
    """

    def __init__(
        self,
        books: dict[str, PolyBook],
        on_update: Callable[[str, PolyBook], None] | None = None,
    ):
        self.books = books
        self.on_update = on_update
        self._running = False

        # token_id -> asset; rebuilt on each subscribe call
        self._token_to_asset: dict[str, str] = {}

        # Currently subscribed token pairs per asset: asset -> [up_id, down_id]
        self._subscriptions: dict[str, list[str]] = {}

        # Signalled when subscriptions change so the WS loop resubscribes
        self._resub_event = asyncio.Event()

        # Last time we received a meaningful market event (book or price_change)
        self._last_market_event_s: float = time.time()

        # Consecutive reconnect failures (for REST-fallback logic)
        self._consec_failures: int = 0

        # Reference to the live WS connection (for sending PONG)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — call as an asyncio task."""
        self._running = True
        while self._running:
            try:
                await self._connect_and_stream()
                self._consec_failures = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._consec_failures += 1
                delay = min(RECONNECT_DELAY_S * self._consec_failures, 30.0)
                logger.warning(
                    "polymarket_ws reconnect consec=%d delay=%.1fs err=%s",
                    self._consec_failures, delay, exc,
                )
                if self._consec_failures >= MAX_RECONNECT_BEFORE_REST:
                    logger.error(
                        "polymarket_ws %d consecutive failures — running in degraded mode "
                        "(REST-only fallback; live book updates paused)",
                        self._consec_failures,
                    )
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._running = False

    def subscribe(self, asset: str, up_token: str, down_token: str) -> None:
        """
        Register a new token pair for an asset. Triggers resubscription on
        the live connection at the next opportunity.
        """
        self._subscriptions[asset] = [up_token, down_token]
        self._token_to_asset[up_token] = asset
        self._token_to_asset[down_token] = asset
        self._resub_event.set()
        logger.info(
            "polymarket_ws subscribe queued asset=%s up=%s down=%s",
            asset, up_token[:8] + "…", down_token[:8] + "…",
        )

    def all_token_ids(self) -> list[str]:
        ids = []
        for pair in self._subscriptions.values():
            ids.extend(pair)
        return ids

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        logger.info("polymarket_ws connecting url=%s", POLY_CLOB_WS)

        async with websockets.connect(
            POLY_CLOB_WS,
            ping_interval=None,   # we handle keep-alive manually
            ping_timeout=None,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            logger.info("polymarket_ws connected")
            self._last_market_event_s = time.time()

            # Subscribe to any already-known token pairs
            await self._send_subscriptions(ws)

            # Run ping, watchdog, and message loop concurrently
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._ping_loop(ws))
                tg.create_task(self._watchdog_loop(ws))
                tg.create_task(self._resub_loop(ws))
                tg.create_task(self._message_loop(ws))

    async def _message_loop(self, ws) -> None:
        async for raw in ws:
            if not self._running:
                break

            # Server keep-alive: literal "PING" string (not JSON)
            if isinstance(raw, str) and raw.strip() == "PING":
                await ws.send("PONG")
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            await self._dispatch(msg)

    async def _ping_loop(self, ws) -> None:
        """Send literal PING every 10s to keep the connection alive."""
        while self._running:
            await asyncio.sleep(PING_INTERVAL_S)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _watchdog_loop(self, ws) -> None:
        """
        Reconnect if no book/price_change event in WATCHDOG_TIMEOUT_S seconds.
        This is the mandatory fix for the silent-freeze bug (py-clob-client #292).
        """
        while self._running:
            await asyncio.sleep(5.0)
            age = time.time() - self._last_market_event_s
            if age > WATCHDOG_TIMEOUT_S:
                logger.warning(
                    "polymarket_ws silent-freeze detected (no event for %.0fs) — reconnecting",
                    age,
                )
                await ws.close()
                return  # exits TaskGroup -> outer loop reconnects

    async def _resub_loop(self, ws) -> None:
        """Watch for new subscription requests and send them to the live WS."""
        while self._running:
            await self._resub_event.wait()
            self._resub_event.clear()
            await self._send_subscriptions(ws)

    async def _send_subscriptions(self, ws) -> None:
        """Send one subscribe message per asset (all tokens in one message per asset)."""
        for asset, token_ids in self._subscriptions.items():
            msg = {
                "type": "market",
                "assets_ids": token_ids,
                "custom_feature_enabled": True,
            }
            try:
                await ws.send(json.dumps(msg))
                logger.info(
                    "polymarket_ws subscribed asset=%s tokens=%s",
                    asset, [t[:8] + "…" for t in token_ids],
                )
            except Exception as exc:
                logger.warning("polymarket_ws subscribe send failed: %s", exc)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, msg: dict | list) -> None:
        # The WS can send a list of events in one frame
        if isinstance(msg, list):
            for item in msg:
                await self._dispatch(item)
            return

        event_type = msg.get("event_type") or msg.get("type", "")

        if event_type == "book":
            self._handle_book(msg)
        elif event_type == "price_change":
            self._handle_price_change(msg)
        elif event_type == "last_trade_price":
            self._handle_last_trade(msg)
        elif event_type == "tick_size_change":
            logger.info("polymarket_ws tick_size_change: %s", msg)
        # else: ignore (e.g. confirmation messages)

    def _handle_book(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", "")
        asset = self._token_to_asset.get(asset_id)
        if not asset:
            return

        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        book = self.books[asset]
        book.handle_book(asset_id, bids, asks)
        self._last_market_event_s = time.time()

        if self.on_update:
            self.on_update(asset, book)

    def _handle_price_change(self, msg: dict) -> None:
        # price_change can be keyed by asset_id directly, or wrap a changes list
        asset_id = msg.get("asset_id", "")
        asset = self._token_to_asset.get(asset_id)
        if not asset:
            return

        # Changes may be in msg["changes"] or the event itself is one change
        changes = msg.get("changes", [msg])
        book = self.books[asset]
        book.handle_price_change(asset_id, changes)
        self._last_market_event_s = time.time()

        if self.on_update:
            self.on_update(asset, book)

    def _handle_last_trade(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", "")
        asset = self._token_to_asset.get(asset_id)
        if not asset:
            return

        try:
            price = float(msg.get("price", 0))
            size = float(msg.get("size", 0))
        except (TypeError, ValueError):
            return

        self.books[asset].handle_last_trade_price(asset_id, price, size)
