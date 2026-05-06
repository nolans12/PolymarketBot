"""
coinbase_ws.py — Coinbase Advanced Trade WebSocket ticker feed for BTC and ETH.

Subscribes to the public ticker channel (no auth required). The ticker provides:
  - last trade price
  - best_bid / best_ask with sizes (enough to compute microprice)

These feed CoinbaseBook's 1-Hz ring buffer, giving the scheduler the same
microprice_at(lag_s) interface as SpotBook but from the Coinbase venue.

Why both Binance and Coinbase?
  Binance and Coinbase market makers reprice at slightly different speeds.
  If Coinbase leads Binance (or vice versa), the regression will learn the
  cross-venue lag structure and encode it in its coefficient vector, giving
  a richer q_settled estimate than either venue alone.

Wire format (Coinbase Advanced Trade WS):
  Subscribe: {"type":"subscribe","product_ids":["BTC-USD","ETH-USD"],"channel":"ticker"}
  Response:  {"channel":"ticker","events":[{"type":"snapshot"|"update",
              "tickers":[{"product_id":"BTC-USD","price":"...","best_bid":"...",
                          "best_ask":"...","best_bid_quantity":"...","best_ask_quantity":"..."}]}]}
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets

from polybot.infra.config import COINBASE_WS_HOST, COINBASE_SYMBOLS, ASSETS
from polybot.state.coinbase_book import CoinbaseBook

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S    = 2.0
MAX_RECONNECT_DELAY  = 30.0
MAX_RECONNECT_TRIES  = 10


class CoinbaseWS:
    """
    Manages one WebSocket connection to Coinbase Advanced Trade for all
    configured assets. Calls on_update(asset, book) after each ticker update.
    """

    def __init__(
        self,
        books: dict[str, CoinbaseBook],
        on_update: Optional[Callable[[str, CoinbaseBook], None]] = None,
    ):
        self.books = books
        self.on_update = on_update
        self._product_ids = [COINBASE_SYMBOLS[a] for a in ASSETS]
        self._running = False

    async def run(self) -> None:
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
                delay = min(RECONNECT_DELAY_S * attempt, MAX_RECONNECT_DELAY)
                logger.warning("coinbase_ws reconnect attempt=%d delay=%.1fs err=%s",
                               attempt, delay, exc)
                if attempt >= MAX_RECONNECT_TRIES:
                    logger.error("coinbase_ws max reconnects reached — giving up")
                    break
                await asyncio.sleep(delay)

    def stop(self) -> None:
        self._running = False

    async def _connect_and_stream(self) -> None:
        url = COINBASE_WS_HOST
        logger.info("coinbase_ws connecting url=%s products=%s", url, self._product_ids)

        subscribe_msg = {
            "type": "subscribe",
            "product_ids": self._product_ids,
            "channel": "ticker",
        }

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=15,
        ) as ws:
            logger.info("coinbase_ws connected")
            await ws.send(json.dumps(subscribe_msg))

            async for raw in ws:
                if not self._running:
                    break
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                channel = msg.get("channel", "")
                if channel != "ticker":
                    continue

                for ev in msg.get("events", []):
                    for tick in ev.get("tickers", []):
                        self._handle_ticker(tick)

    def _handle_ticker(self, tick: dict) -> None:
        product_id = tick.get("product_id", "")
        asset = self._product_to_asset(product_id)
        if asset not in self.books:
            return

        price_str   = tick.get("price")
        bid_str     = tick.get("best_bid")
        ask_str     = tick.get("best_ask")
        bid_qty_str = tick.get("best_bid_quantity")
        ask_qty_str = tick.get("best_ask_quantity")

        if not price_str:
            return

        try:
            price = float(price_str)
            bid   = float(bid_str)   if bid_str     else price
            ask   = float(ask_str)   if ask_str     else price
            bsz   = float(bid_qty_str) if bid_qty_str else 1.0
            asz   = float(ask_qty_str) if ask_qty_str else 1.0
        except (ValueError, TypeError):
            return

        if bid <= 0 or ask <= 0 or bid >= ask:
            return

        book = self.books[asset]
        book.apply_ticker(price, bid, ask, bsz, asz)

        if self.on_update:
            self.on_update(asset, book)

    @staticmethod
    def _product_to_asset(product_id: str) -> str:
        # "BTC-USD" -> "btc", "ETH-USD" -> "eth"
        return product_id.split("-")[0].lower()


# ------------------------------------------------------------------
# Demo / standalone entry point
# ------------------------------------------------------------------

async def _demo() -> None:
    import sys

    books: dict[str, CoinbaseBook] = {asset: CoinbaseBook(asset) for asset in ASSETS}
    last_print: dict[str, float] = {}

    def on_update(asset: str, book: CoinbaseBook) -> None:
        now = time.time()
        if now - last_print.get(asset, 0) < 0.2:
            return
        last_print[asset] = now
        print(
            f"{asset.upper():3s} | "
            f"bid={book.best_bid:>12.2f} ({book.best_bid_size:>10.6f}) | "
            f"ask={book.best_ask:>12.2f} ({book.best_ask_size:>10.6f}) | "
            f"mid={book.mid:>12.4f} | "
            f"μp={book.microprice:>12.4f} | "
            f"last={book.last_trade_price:>12.2f}",
            flush=True,
        )

    client = CoinbaseWS(books=books, on_update=on_update)
    print("Connecting to Coinbase… (Ctrl+C to stop)", file=sys.stderr)
    try:
        await client.run()
    except KeyboardInterrupt:
        client.stop()
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_demo())
