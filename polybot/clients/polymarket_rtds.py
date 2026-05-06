"""
polymarket_rtds.py — Polymarket RTDS WebSocket client (Chainlink oracle prices).

Subscribes to crypto_prices_chainlink for BTC/USD and ETH/USD.
Used exclusively for K-capture at window boundaries (CLAUDE.md §5.4).

Protocol:
  Endpoint: wss://ws-live-data.polymarket.com
  Subscribe:
    {"action": "subscribe", "subscriptions": [{
        "topic": "crypto_prices_chainlink",
        "type": "*",
        "filters": "{\"symbol\": \"btc/usd\"}"
    }]}
  Heartbeat: send literal "PING" every 5s
  Message:
    {
      "topic": "crypto_prices_chainlink",
      "type": "...",
      "timestamp": <server_send_ms>,
      "payload": {"symbol": "btc/usd", "timestamp": <oracle_obs_ms>, "value": "..."}
    }

K-capture: first tick where payload.timestamp >= window_open_ms.
Deadline: if no qualifying tick within 5000ms of window open, K_uncertain=True.

Also detects window rollovers: when wall clock crosses a 5-min boundary, fires
WindowState.rollover() and triggers gamma REST resolution for the new token IDs.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import requests
import websockets

from polybot.infra.config import (
    POLY_RTDS_WS, ASSETS, CHAINLINK_SYMBOLS, GAMMA_API, WINDOW_SECONDS,
)
from polybot.state.window import WindowState
from polybot.clients.polymarket_rest import fetch_market, current_window_ts

logger = logging.getLogger(__name__)

PING_INTERVAL_S      = 5.0
RECONNECT_DELAY_S    = 2.0
MAX_RECONNECT        = 10
STALE_TIMEOUT_S      = 90.0   # circuit breaker: abstain if no tick for 90s
ROLLOVER_LOOKAHEAD_S = 2.0    # start polling for new market this many seconds before close


def _subscribe_msg(symbol: str) -> str:
    return json.dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": json.dumps({"symbol": symbol}),
        }]
    })


class PolymarketRTDS:
    """
    Manages the RTDS WebSocket connection.

    Drives K-capture and window-rollover logic for all configured assets.
    At each rollover, fetches the new market's token IDs from gamma REST
    and notifies registered WindowState objects.
    """

    def __init__(
        self,
        windows: dict[str, WindowState],
        on_k_captured: Callable[[str, WindowState], None] | None = None,
    ):
        self.windows = windows          # asset -> WindowState
        self.on_k_captured = on_k_captured

        self._running = False
        self._last_tick_s: dict[str, float] = {a: 0.0 for a in ASSETS}

        # Track which window_ts we've already fetched market data for
        self._fetched_window_ts: dict[str, int] = {a: 0 for a in ASSETS}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True

        # Bootstrap: resolve the current window for all assets before connecting
        await self._resolve_current_windows()

        attempt = 0
        while self._running:
            try:
                await self._connect_and_stream()
                attempt = 0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                attempt += 1
                delay = min(RECONNECT_DELAY_S * attempt, 30.0)
                logger.warning(
                    "rtds reconnect attempt=%d delay=%.1fs err=%s", attempt, delay, exc
                )
                if attempt >= MAX_RECONNECT:
                    logger.error("rtds max reconnects reached — giving up")
                    break
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._running = False

    def stale_ms(self, asset: str) -> int:
        last = self._last_tick_s.get(asset, 0.0)
        if last == 0.0:
            return 999_999
        return int((time.time() - last) * 1000)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        logger.info("rtds connecting url=%s", POLY_RTDS_WS)

        async with websockets.connect(
            POLY_RTDS_WS,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5,
        ) as ws:
            logger.info("rtds connected")

            # Subscribe for each asset's Chainlink symbol
            for asset in ASSETS:
                symbol = CHAINLINK_SYMBOLS[asset]
                await ws.send(_subscribe_msg(symbol))
                logger.info("rtds subscribed asset=%s symbol=%s", asset, symbol)

            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._ping_loop(ws))
                tg.create_task(self._rollover_monitor())
                tg.create_task(self._message_loop(ws))

    async def _message_loop(self, ws) -> None:
        async for raw in ws:
            if not self._running:
                break

            if isinstance(raw, str) and raw.strip() == "PING":
                await ws.send("PONG")
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            topic = msg.get("topic", "")
            # Actual topic observed: "crypto_prices" (docs say "crypto_prices_chainlink"
            # but the server sends "crypto_prices" for the Chainlink stream)
            if topic in ("crypto_prices_chainlink", "crypto_prices"):
                self._handle_price_tick(msg)

    async def _ping_loop(self, ws) -> None:
        while self._running:
            await asyncio.sleep(PING_INTERVAL_S)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _rollover_monitor(self) -> None:
        """
        Poll wall clock every second. When we cross a 5-min boundary,
        trigger rollover for all assets and fetch new market data.
        """
        last_window_ts = current_window_ts()

        while self._running:
            await asyncio.sleep(1.0)

            now_window_ts = current_window_ts()
            if now_window_ts != last_window_ts:
                last_window_ts = now_window_ts
                logger.info("rtds window boundary crossed new_ts=%d", now_window_ts)

                for asset in ASSETS:
                    self.windows[asset].rollover(now_window_ts)

                # Fetch new market data in background (non-blocking)
                asyncio.create_task(self._fetch_markets(now_window_ts))

            # Also check K deadlines every second
            for asset in ASSETS:
                self.windows[asset].check_k_deadline()

    # ------------------------------------------------------------------
    # Tick handling
    # ------------------------------------------------------------------

    def _handle_price_tick(self, msg: dict) -> None:
        """
        Handle both message formats observed from the RTDS API:

        Batch (on subscribe): payload.data = [{timestamp, value}, ...]
        Live tick:            payload.data = [{timestamp, value}]  (single entry)
                              OR payload.timestamp + payload.value (legacy format)

        The batch contains recent history; we process every tick so that
        K-capture works even when we connect mid-window.
        """
        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            return

        symbol = payload.get("symbol", "").lower()

        # Map symbol to asset
        asset = None
        for a, sym in CHAINLINK_SYMBOLS.items():
            if sym == symbol:
                asset = a
                break
        if asset is None:
            return

        # Collect all (oracle_ts_ms, price) pairs from this message
        ticks: list[tuple[int, float]] = []

        data_arr = payload.get("data")
        if data_arr and isinstance(data_arr, list):
            # Batch or single-entry live tick
            for entry in data_arr:
                try:
                    ticks.append((int(entry["timestamp"]), float(entry["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            # Legacy single-tick format: payload.timestamp + payload.value
            oracle_ts_ms = payload.get("timestamp")
            value_raw = payload.get("value")
            if oracle_ts_ms is not None and value_raw is not None:
                try:
                    ticks.append((int(oracle_ts_ms), float(value_raw)))
                except (TypeError, ValueError):
                    pass

        if not ticks:
            return

        self._last_tick_s[asset] = time.time()
        window = self.windows[asset]

        for oracle_ts_ms, price in ticks:
            captured = window.apply_chainlink_tick(oracle_ts_ms, price)
            if captured and self.on_k_captured:
                self.on_k_captured(asset, window)

        # First tick per asset: log so we know RTDS is alive
        if not getattr(self, "_logged_first_tick", {}).get(asset):
            if not hasattr(self, "_logged_first_tick"):
                self._logged_first_tick = {}
            self._logged_first_tick[asset] = True
            last_oracle_ms, last_price = ticks[-1]
            logger.info(
                "rtds first tick asset=%s price=%.2f oracle_ts_ms=%d "
                "window_open_ts=%d window_open_ms=%d K_set=%s",
                asset, last_price, last_oracle_ms,
                window.open_ts, window.open_ts * 1000,
                window.K is not None,
            )

    # ------------------------------------------------------------------
    # Market resolution
    # ------------------------------------------------------------------

    async def _resolve_current_windows(self) -> None:
        """Bootstrap: resolve the current window for all assets on startup."""
        now_ts = current_window_ts()
        for asset in ASSETS:
            self.windows[asset].rollover(now_ts)
        await self._fetch_markets(now_ts)

    async def _fetch_markets(self, window_ts: int) -> None:
        """Fetch gamma REST for all assets and populate WindowState token IDs."""
        loop = asyncio.get_event_loop()
        for asset in ASSETS:
            if self._fetched_window_ts[asset] == window_ts:
                continue  # already fetched
            try:
                market = await loop.run_in_executor(
                    None, fetch_market, asset, window_ts
                )
                if market:
                    self.windows[asset].apply_market(market)
                    self._fetched_window_ts[asset] = window_ts
                    logger.info(
                        "rtds market resolved asset=%s slug=%s up=%s… down=%s…",
                        asset, market["slug"],
                        market["up_token_id"][:12],
                        market["down_token_id"][:12],
                    )
                    # Fire rollover callbacks now that token IDs are populated
                    for cb in self.windows[asset]._on_rollover:
                        try:
                            cb(self.windows[asset])
                        except Exception as exc:
                            logger.error("rollover callback error: %s", exc)
                else:
                    logger.warning(
                        "rtds market not found asset=%s window_ts=%d", asset, window_ts
                    )
            except Exception as exc:
                logger.error(
                    "rtds market fetch failed asset=%s window_ts=%d err=%s",
                    asset, window_ts, exc,
                )
