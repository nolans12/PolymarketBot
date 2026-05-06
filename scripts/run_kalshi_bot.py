"""
run_kalshi_bot.py — Entry point for the Kalshi lead-lag arbitrage bot.

Wires together:
  CoinbaseFeed  -> CoinbaseBook
  KalshiRestFeed -> KalshiBook   (REST polling at 1Hz; no WebSocket)
  Scheduler     (sampler + refitter + decision + window manager)

Usage:
  python scripts/run_kalshi_bot.py [--wallet 1000] [--log logs/decisions.jsonl]
"""

import argparse
import asyncio
import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betbot.kalshi.auth import load_private_key
from betbot.kalshi.book import CoinbaseBook, KalshiBook
from betbot.kalshi.coinbase_feed import CoinbaseFeed
from betbot.kalshi.kalshi_rest_feed import KalshiRestFeed
from betbot.kalshi.scheduler import Scheduler, _discover_market, _list_active_markets
from betbot.kalshi.config import KALSHI_KEY_ID


async def _fetch_btc_spot() -> float:
    """Quick one-shot Coinbase REST fetch for current BTC price (USD)."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("https://api.coinbase.com/v2/prices/BTC-USD/spot",
                             timeout=aiohttp.ClientTimeout(total=5)) as r:
                data = await r.json()
                return float(data["data"]["amount"])
    except Exception:
        return 0.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kalshi_bot")


def parse_args():
    p = argparse.ArgumentParser(description="Kalshi BTC lag-arbitrage bot (dry run)")
    p.add_argument("--wallet", type=float, default=1000.0, help="Simulated wallet USD")
    p.add_argument("--log",    type=str,   default="logs/decisions.jsonl",
                   help="JSONL file to append decision rows")
    p.add_argument("--ticks",  type=str,   default="logs/ticks.csv",
                   help="CSV file to append 1s raw ticks for backtesting")
    p.add_argument("--ticker", type=str,   default=None,
                   help="Override market ticker (else auto-discover)")
    p.add_argument("--fresh",  action="store_true",
                   help="Delete existing log/tick files before starting (clean run)")
    return p.parse_args()


async def main():
    args = parse_args()

    print("=== Kalshi BTC Lead-Lag Arbitrage Bot ===", flush=True)
    print(f"  Wallet:     ${args.wallet:,.0f} (simulated)", flush=True)
    print(f"  Log:        {args.log}", flush=True)
    print(f"  Ticks:      {args.ticks}", flush=True)

    if args.fresh:
        for p_str in [args.log, args.ticks]:
            p = Path(p_str)
            if p.exists():
                p.unlink()
                print(f"  Deleted old {p}", flush=True)

    # ---- Discover active market ----
    if args.ticker:
        # Minimal stub when ticker is overridden — no floor_strike lookup
        ticker       = args.ticker
        floor_strike = 0.0
        close_time   = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)
        print(f"  Ticker:     {ticker} (manual override)", flush=True)
    else:
        print("  Discovering active Kalshi BTC 15m market...", flush=True)
        mkt = None
        for attempt in range(1, 13):   # retry for up to 60s (window boundary gap)
            mkt = await _discover_market()
            if mkt:
                break
            print(f"  No active market yet (attempt {attempt}/12) — "
                  f"waiting 5s for window to open...", flush=True)
            await asyncio.sleep(5)
        if not mkt:
            print("ERROR: no active KXBTC15M market after 60s. "
                  "Pass --ticker to override.", flush=True)
            sys.exit(1)
        ticker       = mkt["ticker"]
        floor_strike = float(mkt.get("floor_strike") or 0.0)
        ct_str       = mkt.get("close_time", "")
        try:
            close_time = datetime.datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        except Exception:
            close_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)
        print(f"  Ticker:     {ticker}", flush=True)
        print(f"  Strike:     ${floor_strike:,.2f}", flush=True)
        print(f"  Closes at:  {close_time.strftime('%H:%M:%S UTC')}", flush=True)

    # ---- Book state ----
    cb_book = CoinbaseBook()
    kb_book = KalshiBook()
    kb_book.set_window(ticker, floor_strike, close_time)

    # ---- Feed clients ----
    pk           = load_private_key()
    cb_feed      = CoinbaseFeed(cb_book)
    kalshi_feed  = KalshiRestFeed(kb_book, key_id=KALSHI_KEY_ID, pk=pk)

    # ---- Scheduler ----
    log_path  = Path(args.log)
    tick_path = Path(args.ticks)
    scheduler = Scheduler(cb_book, kb_book, kalshi_feed,
                          wallet_usd=args.wallet,
                          log_path=log_path,
                          tick_path=tick_path)

    print("  Starting feeds (Coinbase WS + Kalshi REST 1Hz polling)...\n", flush=True)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(cb_feed.run(),     name="coinbase_feed")
            tg.create_task(kalshi_feed.run(), name="kalshi_rest")
            tg.create_task(scheduler.run(),   name="scheduler")
    except* KeyboardInterrupt:
        pass
    except* asyncio.CancelledError:
        pass
    finally:
        cb_feed.stop()
        kalshi_feed.stop()
        scheduler.stop()
        print("\nBot stopped.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
