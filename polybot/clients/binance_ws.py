"""
binance_ws.py — Binance L2 diff stream + aggTrade feed for BTC and ETH.

Subscribes to:
  btcusdt@depth@100ms, ethusdt@depth@100ms  — L2 diffs (100ms cadence)
  btcusdt@aggTrade,    ethusdt@aggTrade      — trade prints

L2 sequence protocol (Binance docs):
  1. Open WS, buffer incoming diff events.
  2. GET /api/v3/depth?symbol=BTCUSDT&limit=1000 for REST snapshot.
  3. Drop buffered events where finalUpdateId <= lastUpdateId from snapshot.
  4. Apply remaining buffered events in order; then stream live diffs.
  5. On gap (firstUpdateId > local lastUpdateId+1): reconnect + re-snapshot.

Combined stream URL format:
  wss://stream.binance.com:9443/stream?streams=<s1>/<s2>/...

Message envelope for combined stream:
  {"stream": "btcusdt@depth@100ms", "data": { ...diff payload... }}
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Callable

import requests
import websockets

from polybot.infra.config import BINANCE_WS_HOST, BINANCE_SYMBOLS, ASSETS
from polybot.state.spot_book import SpotBook

logger = logging.getLogger(__name__)

BINANCE_REST = "https://api.binance.com"
DEPTH_LIMIT = 1000
RECONNECT_DELAY_S = 2.0
MAX_RECONNECT_ATTEMPTS = 10
SNAPSHOT_TIMEOUT_S = 10.0


def _depth_snapshot(symbol: str) -> dict:
    """Fetch REST depth snapshot. Raises on failure."""
    url = f"{BINANCE_REST}/api/v3/depth"
    resp = requests.get(url, params={"symbol": symbol.upper(), "limit": DEPTH_LIMIT},
                        timeout=SNAPSHOT_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def _build_stream_url(symbols: list[str]) -> str:
    streams = []
    for sym in symbols:
        streams.append(f"{sym}@depth@100ms")
        streams.append(f"{sym}@aggTrade")
    path = "/".join(streams)
    return f"{BINANCE_WS_HOST}/stream?streams={path}"


class BinanceWS:
    """
    Manages one combined WebSocket connection for all configured assets.
    Calls `on_update(asset, book)` after each L2 book update so consumers
    (demo script, scheduler) can react to state changes.
    """

    def __init__(
        self,
        books: dict[str, SpotBook],
        on_update: Callable[[str, SpotBook], None] | None = None,
    ):
        self.books = books          # asset -> SpotBook
        self.on_update = on_update  # optional callback after each update
        self._symbols = [BINANCE_SYMBOLS[a] for a in ASSETS]
        self._running = False

        # Per-symbol pre-snapshot diff buffer
        self._diff_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=2000))
        # Per-symbol snapshot state
        self._snap_applied: dict[str, bool] = {s: False for s in self._symbols}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop: connect, snapshot, stream. Reconnects on failure."""
        self._running = True
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
                logger.warning("binance_ws reconnect attempt=%d delay=%.1fs err=%s",
                               attempt, delay, exc)
                if attempt >= MAX_RECONNECT_ATTEMPTS:
                    logger.warning("binance_ws %d reconnect failures — continuing with Coinbase fallback",
                                   attempt)
                    attempt = MAX_RECONNECT_ATTEMPTS  # cap backoff, don't give up
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _connect_and_stream(self) -> None:
        url = _build_stream_url(self._symbols)
        logger.info("binance_ws connecting url=%s", url)

        # Reset state for clean reconnect
        for sym in self._symbols:
            self._diff_buf[sym].clear()
            self._snap_applied[sym] = False

        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            logger.info("binance_ws connected")

            # Kick off background snapshot fetches (one per symbol)
            snap_tasks = {
                sym: asyncio.create_task(self._fetch_and_apply_snapshot(sym))
                for sym in self._symbols
            }

            try:
                async for raw in ws:
                    if not self._running:
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    stream = msg.get("stream", "")
                    data = msg.get("data", {})
                    event_type = data.get("e", "")

                    # Parse symbol from stream name: "btcusdt@depth@100ms" -> "btcusdt"
                    sym = stream.split("@")[0] if stream else ""

                    if event_type == "depthUpdate":
                        await self._handle_depth(sym, data)
                    elif event_type == "aggTrade":
                        self._handle_agg_trade(sym, data)

            finally:
                for task in snap_tasks.values():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

    async def _fetch_and_apply_snapshot(self, sym: str) -> None:
        """Fetch REST snapshot and apply it, then flush buffered diffs."""
        await asyncio.sleep(0.5)  # let a few diffs buffer first
        loop = asyncio.get_event_loop()
        try:
            snap = await loop.run_in_executor(None, _depth_snapshot, sym)
        except Exception as exc:
            logger.error("binance snapshot fetch failed sym=%s err=%s", sym, exc)
            raise

        asset = self._sym_to_asset(sym)
        book = self.books[asset]
        book.apply_snapshot(snap["bids"], snap["asks"], snap["lastUpdateId"])
        self._snap_applied[sym] = True

        logger.info("binance snapshot applied sym=%s lastUpdateId=%d bids=%d asks=%d",
                    sym, snap["lastUpdateId"],
                    len(snap["bids"]), len(snap["asks"]))

        # Flush pre-snapshot buffer
        buf = self._diff_buf[sym]
        while buf:
            ev = buf.popleft()
            ok = book.apply_diff(
                ev["b"], ev["a"], ev["U"], ev["u"]
            )
            if not ok:
                logger.warning("binance gap during flush sym=%s — reconnect needed", sym)
                raise RuntimeError(f"sequence gap on {sym} after snapshot flush")

    async def _handle_depth(self, sym: str, data: dict) -> None:
        asset = self._sym_to_asset(sym)
        if asset not in self.books:
            return

        book = self.books[asset]

        if not self._snap_applied[sym]:
            # Buffer diffs until snapshot is ready
            self._diff_buf[sym].append(data)
            return

        ok = book.apply_diff(data["b"], data["a"], data["U"], data["u"])
        if not ok:
            logger.warning("binance sequence gap sym=%s — reconnecting", sym)
            raise RuntimeError(f"sequence gap on {sym}")

        if self.on_update:
            self.on_update(asset, book)

    def _handle_agg_trade(self, sym: str, data: dict) -> None:
        # Currently just updates last trade price on the book's microprice stamp.
        # Trade flow features are accumulated in spot_book OFI; this is informational.
        asset = self._sym_to_asset(sym)
        if asset in self.books:
            book = self.books[asset]
            # Tag last trade for staleness monitoring
            book.last_update_ns = time.time_ns()

    @staticmethod
    def _sym_to_asset(sym: str) -> str:
        return sym.replace("usdt", "")


# ------------------------------------------------------------------
# Demo / standalone entry point
# ------------------------------------------------------------------

async def _demo() -> None:
    """Print live microprice + OFI every depth update."""
    import sys

    books: dict[str, SpotBook] = {asset: SpotBook(asset) for asset in ASSETS}
    last_print: dict[str, float] = {asset: 0.0 for asset in ASSETS}

    def on_update(asset: str, book: SpotBook) -> None:
        now = time.time()
        # Print at most ~10 Hz per asset to keep terminal readable
        if now - last_print.get(asset, 0) < 0.1:
            return
        last_print[asset] = now

        ofi_l1, ofi_l5 = book._ofi_l1_acc, book._ofi_l5_acc  # peek without draining
        print(
            f"{asset.upper():3s} | "
            f"bid={book.best_bid:>12.2f} ({book.best_bid_size:>8.4f}) | "
            f"ask={book.best_ask:>12.2f} ({book.best_ask_size:>8.4f}) | "
            f"mid={book.mid:>12.4f} | "
            f"μp={book.microprice:>12.4f} | "
            f"spread={book.spread:>8.4f} | "
            f"ofi_l1={ofi_l1:>+10.2f} | "
            f"ofi_l5={ofi_l5:>+10.2f}",
            flush=True,
        )

    client = BinanceWS(books=books, on_update=on_update)
    print("Connecting to Binance… (Ctrl+C to stop)", file=sys.stderr)

    try:
        await client.run()
    except KeyboardInterrupt:
        client.stop()
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_demo())
