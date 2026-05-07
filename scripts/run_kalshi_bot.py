"""
run_kalshi_bot.py — Entry point for the Kalshi lead-lag arbitrage bot.

Wires together:
  CoinbaseFeed | BinanceFeed -> SpotBook   (selected by SPOT_SOURCE env var)
  KalshiRestFeed             -> KalshiBook (REST polling at 1Hz)
  Scheduler                  (sampler + refitter + decision + window manager)

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
from betbot.kalshi.book import SpotBook, KalshiBook
from betbot.kalshi.coinbase_feed import CoinbaseFeed
from betbot.kalshi.binance_feed import BinanceFeed
from betbot.kalshi.kalshi_rest_feed import KalshiRestFeed
from betbot.kalshi.scheduler import Scheduler, _discover_market, _list_active_markets
from betbot.kalshi.config import KALSHI_KEY_ID, SPOT_SOURCE
from betbot.kalshi.orders import get_balance_usd


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
                   help="Delete existing decisions log before starting. "
                        "Ticks file is KEPT so the model can bootstrap from it.")
    p.add_argument("--seed-ticks", type=str, default=None,
                   help="Bootstrap the model from this ticks CSV instead of --ticks. "
                        "Use when you want to cold-start from a specific dataset.")
    p.add_argument("--live-orders", action="store_true",
                   help="REAL MONEY: place actual Kalshi orders on entry/exit. "
                        "Reads your real balance from Kalshi as the wallet. "
                        "Without this flag, the bot is decision-logger only.")
    p.add_argument("--max-bet-pct", type=float, default=0.10,
                   help="Hard cap on per-trade bet, fraction of starting wallet "
                        "(default 0.10; only applies with --live-orders)")
    p.add_argument("--daily-loss-pct", type=float, default=0.05,
                   help="Halt when realized loss exceeds this fraction of starting "
                        "wallet (default 0.05; only applies with --live-orders)")
    return p.parse_args()


async def main():
    args = parse_args()

    print("=== Kalshi BTC Lead-Lag Arbitrage Bot ===", flush=True)
    if args.live_orders:
        print("  *** !!! LIVE ORDERS MODE -- REAL MONEY AT RISK !!! ***", flush=True)
        print(f"  Wallet:     (will query real Kalshi balance at startup)", flush=True)
    else:
        print(f"  Wallet:     ${args.wallet:,.0f} (simulated; bot logs decisions only)", flush=True)
    print(f"  Log:        {args.log}", flush=True)
    print(f"  Ticks:      {args.ticks}", flush=True)
    print(f"  Spot feed:  {SPOT_SOURCE}", flush=True)

    if args.fresh:
        # Only wipe the decisions log — keep ticks so bootstrap has data to train on
        p = Path(args.log)
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
    spot_book = SpotBook()
    kb_book   = KalshiBook()
    kb_book.set_window(ticker, floor_strike, close_time)

    # ---- Feed clients ----
    pk          = load_private_key()
    if SPOT_SOURCE == "coinbase":
        spot_feed = CoinbaseFeed(spot_book)
    elif SPOT_SOURCE == "binance":
        spot_feed = BinanceFeed(spot_book)
    else:
        # config.py validates this on import; this branch is unreachable.
        raise RuntimeError(f"Unknown SPOT_SOURCE: {SPOT_SOURCE!r}")
    kalshi_feed = KalshiRestFeed(kb_book, key_id=KALSHI_KEY_ID, pk=pk)

    # ---- Live orders: query real Kalshi wallet balance ----
    wallet_usd = float(args.wallet)
    if args.live_orders:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            real_balance = await get_balance_usd(s, pk)
        if real_balance <= 0:
            sys.exit("ERROR: --live-orders requires a positive Kalshi balance. "
                     "Got $0.00 from /portfolio/balance -- check API auth and account funding.")
        wallet_usd = real_balance
        print(f"  Live wallet (Kalshi balance): ${wallet_usd:,.2f}", flush=True)

    # ---- Scheduler ----
    log_path   = Path(args.log)
    tick_path  = Path(args.ticks)
    # --seed-ticks lets you bootstrap from a specific file (e.g. a known-good
    # backtest dataset) while writing new live ticks to a separate file.
    seed_path  = Path(args.seed_ticks) if args.seed_ticks else tick_path
    scheduler = Scheduler(spot_book, kb_book, kalshi_feed,
                          wallet_usd=wallet_usd,
                          log_path=log_path,
                          tick_path=tick_path,
                          seed_tick_path=seed_path,
                          pk=pk,
                          live_orders=args.live_orders,
                          max_bet_pct=args.max_bet_pct,
                          daily_loss_limit_pct=args.daily_loss_pct)

    print(f"  Starting feeds ({SPOT_SOURCE} WS + Kalshi REST 1Hz polling)...\n", flush=True)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(spot_feed.run(),   name=f"{SPOT_SOURCE}_feed")
            tg.create_task(kalshi_feed.run(), name="kalshi_rest")
            tg.create_task(scheduler.run(),   name="scheduler")
    except* KeyboardInterrupt:
        pass
    except* asyncio.CancelledError:
        pass
    finally:
        spot_feed.stop()
        kalshi_feed.stop()
        scheduler.stop()
        print("\nBot stopped.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
