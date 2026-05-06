"""
main.py — Entrypoint for the polybot service.

Owns the asyncio.TaskGroup that coordinates all six long-running tasks:
  1. Binance WS        — live L2 spot book for BTC + ETH
  2. Polymarket CLOB WS — live Up/Down order book for active windows
  3. Polymarket RTDS WS — Chainlink oracle prices for K-capture
  4. Scheduler          — 10-second decision tick loop
  5. Parquet writer     — background 60-second flush loop
  6. (Stage 2B) Regression refitter — 5-minute refit loop

Window rollover wiring:
  RTDS detects a new 5-min boundary → WindowState.rollover() →
  registered callbacks update PolyBook token IDs and resubscribe
  the CLOB WS to the new token pair.
"""

import asyncio
import logging
import signal
import sys

import structlog

from polybot.infra.config import ASSETS, DRY_RUN, PARQUET_DIR
from polybot.infra.parquet_writer import ParquetWriter
from polybot.infra.scheduler import Scheduler
from polybot.infra.refitter import RegressionRefitter
from polybot.clients.binance_ws import BinanceWS
from polybot.clients.coinbase_ws import CoinbaseWS
from polybot.clients.polymarket_ws import PolymarketWS
from polybot.clients.polymarket_rtds import PolymarketRTDS
from polybot.models.regression import RegressionModel
from polybot.state.spot_book import SpotBook
from polybot.state.coinbase_book import CoinbaseBook
from polybot.state.poly_book import PolyBook
from polybot.state.window import WindowState


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(message)s",
    )


async def _run() -> None:
    log = logging.getLogger("polybot.main")
    log.info("polybot starting DRY_RUN=%s parquet_dir=%s", DRY_RUN, PARQUET_DIR)

    # --- Shared state objects ---
    spot_books:     dict[str, SpotBook]        = {a: SpotBook(a)        for a in ASSETS}
    coinbase_books: dict[str, CoinbaseBook]    = {a: CoinbaseBook(a)    for a in ASSETS}
    poly_books:     dict[str, PolyBook]        = {a: PolyBook(a)        for a in ASSETS}
    windows:        dict[str, WindowState]     = {a: WindowState(a)     for a in ASSETS}
    models:         dict[str, RegressionModel] = {a: RegressionModel(a) for a in ASSETS}

    # --- Infrastructure ---
    writer    = ParquetWriter(parquet_dir=PARQUET_DIR)
    refitter  = RegressionRefitter(models=models, writer=writer)
    scheduler = Scheduler(
        spot_books=spot_books,
        coinbase_books=coinbase_books,
        poly_books=poly_books,
        windows=windows,
        writer=writer,
        model=models,   # dict; scheduler picks per-asset model
    )

    # --- Feed clients ---
    poly_ws     = PolymarketWS(books=poly_books)
    coinbase_ws = CoinbaseWS(books=coinbase_books)

    def _on_rollover(win: WindowState) -> None:
        """
        Called by WindowState.rollover() when a new 5-min window opens
        AND the gamma REST fetch has completed (token IDs are populated).
        Updates PolyBook and resubscribes the CLOB WS.
        """
        asset = win.asset
        if win.up_token_id and win.down_token_id:
            poly_books[asset].set_tokens(win.up_token_id, win.down_token_id)
            poly_ws.subscribe(asset, win.up_token_id, win.down_token_id)
            log.info(
                "rollover wired asset=%s slug=%s up=%s… down=%s…",
                asset, win.slug,
                win.up_token_id[:12], win.down_token_id[:12],
            )

    for asset in ASSETS:
        windows[asset].register_rollover_callback(_on_rollover)

    binance_ws = BinanceWS(books=spot_books)
    rtds       = PolymarketRTDS(windows=windows)

    # --- Graceful shutdown ---
    stop_event = asyncio.Event()

    def _handle_signal(*_):
        log.warning("shutdown signal received")
        stop_event.set()
        for client in (binance_ws, coinbase_ws, poly_ws, rtds, scheduler, refitter):
            client.stop()
        writer.stop()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # Windows: signal handlers not supported in asyncio loop

    # --- TaskGroup: run all six tasks concurrently ---
    log.info("starting all tasks")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(binance_ws.run(),   name="binance_ws")
            tg.create_task(coinbase_ws.run(),  name="coinbase_ws")
            tg.create_task(poly_ws.run(),      name="poly_ws")
            tg.create_task(rtds.run(),         name="rtds")
            tg.create_task(scheduler.run(),    name="scheduler")
            tg.create_task(refitter.run(),     name="refitter")
            tg.create_task(writer.run(),       name="parquet_writer")
            tg.create_task(_shutdown_watcher(stop_event, tg), name="shutdown")
    except* KeyboardInterrupt:
        pass
    except* Exception as eg:
        log.error("task group error: %s", eg.exceptions)
    finally:
        log.info("flushing parquet writer…")
        await writer.close()
        log.info("polybot stopped cleanly")


async def _shutdown_watcher(stop_event: asyncio.Event, tg) -> None:
    """Cancel all tasks in the group when stop_event is set."""
    await stop_event.wait()
    # Cancelling ourselves cancels the whole TaskGroup
    raise asyncio.CancelledError("shutdown requested")


def cli_entry() -> None:
    _configure_logging()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli_entry()
