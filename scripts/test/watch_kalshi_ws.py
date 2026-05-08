"""
watch_kalshi_ws.py — Standalone Kalshi WebSocket sanity check.

Connects to wss://api.elections.kalshi.com/trade-api/ws/v2, discovers the
soonest-closing market in the requested series, subscribes to orderbook_delta,
and prints live yes_bid/yes_ask + total contract depth on each book change.

Use this to verify:
  - Auth works (no 401 / unknown command errors)
  - Snapshot arrives within seconds
  - Deltas stream continuously when the book moves
  - The book actually has size at the quoted prices

Usage:
  python scripts/test/watch_kalshi_ws.py                    # BTC default
  python scripts/test/watch_kalshi_ws.py --series KXETH15M
  python scripts/test/watch_kalshi_ws.py --ticker KXBTC15M-26MAY080230-30
"""

import argparse
import asyncio
import datetime as _dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from betbot.kalshi.auth import auth_headers, load_private_key
from betbot.kalshi.config import KALSHI_REST, KALSHI_KEY_ID

WS_URL  = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"

# ANSI
G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
C   = "\033[96m"
DIM = "\033[2m"
RST = "\033[0m"


async def discover_ticker(series: str) -> str | None:
    """Find the soonest-closing open market in the series."""
    url    = f"{KALSHI_REST}/trade-api/v2/markets"
    params = {"series_ticker": series, "status": "open", "limit": 50}
    async with aiohttp.ClientSession() as s:
        async with s.get(url, params=params,
                         timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            data = await r.json()

    best = None
    best_close = float("inf")
    for mkt in data.get("markets", []):
        ct = mkt.get("close_time", "")
        if ct and mkt.get("status") in ("open", "active"):
            try:
                ep = _dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp()
                if ep < best_close:
                    best_close = ep
                    best       = mkt
            except Exception:
                pass
    return best["ticker"] if best else None


def book_top(book: dict[float, float]) -> tuple[float, float]:
    """Return (best_price, size_at_that_price) for a side. (0, 0) if empty.
    Uses a >=1 contract threshold to avoid phantom levels with tiny residue."""
    live = {p: s for p, s in book.items() if s >= 1.0}
    if not live:
        return 0.0, 0.0
    best_p = max(live)
    return best_p, live[best_p]


async def run(ticker: str, dump_raw: bool = False):
    pk = load_private_key()
    hdrs = auth_headers(pk, KALSHI_KEY_ID, "GET", WS_PATH)

    print(f"{C}  Connecting to Kalshi WS ({ticker})...{RST}")

    yes_book: dict[float, float] = defaultdict(float)
    no_book:  dict[float, float] = defaultdict(float)

    snapshot_t = None
    n_msgs     = 0
    n_deltas   = 0
    last_print = 0.0

    async with websockets.connect(
        WS_URL, additional_headers=hdrs,
        ping_interval=10, ping_timeout=5, open_timeout=10,
    ) as ws:
        sub = {
            "id":  1,
            "cmd": "subscribe",
            "params": {
                "channels":       ["orderbook_delta"],
                "market_tickers": [ticker],
            },
        }
        await ws.send(json.dumps(sub))
        t0 = time.monotonic()
        print(f"{DIM}  subscribed, waiting for snapshot...{RST}")

        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
            except asyncio.TimeoutError:
                print(f"{R}  silent >30s — connection stalled{RST}")
                return

            n_msgs += 1
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            mtype = msg.get("type", "")

            if mtype == "subscribed":
                print(f"{G}  ✓ subscribed{RST}  {DIM}sid={msg.get('sid')}{RST}")
                continue
            if mtype == "error":
                print(f"{R}  error: {msg}{RST}")
                return

            payload = msg.get("msg") or {}

            if mtype == "orderbook_snapshot":
                snapshot_t = time.monotonic() - t0
                yes_book.clear()
                no_book.clear()
                for p, sz in payload.get("yes_dollars_fp") or []:
                    yes_book[float(p)] = float(sz)
                for p, sz in payload.get("no_dollars_fp") or []:
                    no_book[float(p)] = float(sz)
                print(f"{G}  ✓ snapshot received in {snapshot_t*1000:.0f}ms"
                      f"  yes_levels={len(yes_book)}  no_levels={len(no_book)}{RST}")
                if dump_raw:
                    print(f"{DIM}  RAW SNAPSHOT: {json.dumps(payload, indent=2)[:1500]}{RST}")

            elif mtype == "orderbook_delta":
                if dump_raw and n_deltas < 30:
                    print(f"{DIM}  RAW DELTA #{n_deltas}: {json.dumps(payload)}{RST}")
                n_deltas += 1
                side  = payload.get("side")
                p_str = payload.get("price_dollars")
                d_str = payload.get("delta_fp")
                if side and p_str is not None and d_str is not None:
                    try:
                        price = float(p_str)
                        delta = float(d_str)
                    except (TypeError, ValueError):
                        continue
                    book = yes_book if side == "yes" else no_book
                    book[price] = book.get(price, 0.0) + delta
                    if book[price] <= 0:
                        book.pop(price, None)

            # Throttle prints to once per 250ms
            now = time.monotonic()
            if now - last_print < 0.25:
                continue
            last_print = now

            yes_top_p, yes_top_sz = book_top(yes_book)
            no_top_p,  no_top_sz  = book_top(no_book)
            if yes_top_p == 0 or no_top_p == 0:
                continue

            yes_ask     = 1.0 - no_top_p
            yes_ask_sz  = no_top_sz
            yes_bid     = yes_top_p
            yes_bid_sz  = yes_top_sz
            spread_c    = (yes_ask - yes_bid) * 100

            ts = time.strftime("%H:%M:%S")
            print(
                f"{DIM}{ts}{RST}  "
                f"YES bid={G}{yes_bid:.2f}{RST} ({yes_bid_sz:.0f}c)"
                f"  ask={Y}{yes_ask:.2f}{RST} ({yes_ask_sz:.0f}c)"
                f"  spread={spread_c:.1f}c"
                f"  {DIM}msgs={n_msgs}{RST}",
                flush=True,
            )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXBTC15M",
                    help="series ticker (KXBTC15M, KXETH15M, KXSOL15M, KXXRP15M)")
    ap.add_argument("--ticker", default=None,
                    help="explicit market ticker (skips discovery)")
    ap.add_argument("--raw", action="store_true",
                    help="dump raw WS messages for first 30 deltas")
    args = ap.parse_args()

    ticker = args.ticker
    if not ticker:
        print(f"{C}  Discovering soonest-closing {args.series} market...{RST}")
        ticker = await discover_ticker(args.series)
        if not ticker:
            sys.exit(f"  no open markets found in series {args.series}")

    try:
        await run(ticker, dump_raw=args.raw)
    except KeyboardInterrupt:
        print(f"\n{DIM}  stopped{RST}")


if __name__ == "__main__":
    asyncio.run(main())
