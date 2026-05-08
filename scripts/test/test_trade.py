"""
test_trade.py — Live round-trip sanity check using the WS feed.

Connects to Kalshi WS, waits for the book to populate (so we know the live
top-of-book size), then executes a market BUY → 5s hold → market SELL with
no user confirmation. Reports actual fills vs intent so you can see exactly
what the live book gave you.

Usage:
  python scripts/test/test_trade.py
  python scripts/test/test_trade.py --target-usd 1.0 --hold-s 5
"""

import argparse
import asyncio
import datetime as _dt
import json
import sys
import time
import uuid
from pathlib import Path

import aiohttp
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from betbot.kalshi.auth import auth_headers, load_private_key
from betbot.kalshi.config import KALSHI_REST, KALSHI_KEY_ID, KALSHI_SERIES

WS_URL  = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"

# ANSI
G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
C   = "\033[96m"
DIM = "\033[2m"
RST = "\033[0m"
BOLD = "\033[1m"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

async def discover_market(session: aiohttp.ClientSession) -> dict | None:
    path   = "/trade-api/v2/markets"
    params = {"series_ticker": KALSHI_SERIES, "status": "open", "limit": 20}
    async with session.get(KALSHI_REST + path, params=params,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()
    best, best_close = None, float("inf")
    for mkt in data.get("markets", []):
        ct = mkt.get("close_time", "")
        if ct and mkt.get("status") in ("open", "active"):
            try:
                ep = _dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp()
                if ep < best_close:
                    best_close, best = ep, mkt
            except Exception:
                pass
    return best


# ---------------------------------------------------------------------------
# WS book state (lives only as long as this script runs)
# ---------------------------------------------------------------------------

class LiveBook:
    def __init__(self):
        self.yes_book: dict[float, float] = {}   # YES bids
        self.no_book:  dict[float, float] = {}   # NO bids
        self.ready  = False

    def apply_snapshot(self, yes_fp, no_fp):
        self.yes_book = {float(p): float(s) for p, s in yes_fp}
        self.no_book  = {float(p): float(s) for p, s in no_fp}
        self.ready    = True

    def apply_delta(self, side, price, delta):
        book = self.yes_book if side == "yes" else self.no_book
        book[price] = book.get(price, 0.0) + delta
        if book[price] <= 0:
            book.pop(price, None)

    def top(self, side: str) -> tuple[float, float]:
        book = self.yes_book if side == "yes" else self.no_book
        live = {p: s for p, s in book.items() if s >= 1.0}
        if not live:
            return 0.0, 0.0
        p = max(live)
        return p, live[p]

    def yes_bid_ask(self) -> tuple[float, float, float, float]:
        """Returns (yes_bid, yes_bid_size, yes_ask, yes_ask_size)."""
        yb_p, yb_s = self.top("yes")
        nb_p, nb_s = self.top("no")
        if yb_p == 0 or nb_p == 0:
            return 0.0, 0.0, 0.0, 0.0
        yes_ask = 1.0 - nb_p
        return yb_p, yb_s, yes_ask, nb_s


async def ws_loop(book: LiveBook, ticker: str, ready_event: asyncio.Event):
    pk   = load_private_key()
    hdrs = auth_headers(pk, KALSHI_KEY_ID, "GET", WS_PATH)
    async with websockets.connect(
        WS_URL, additional_headers=hdrs,
        ping_interval=10, ping_timeout=5, open_timeout=10,
    ) as ws:
        await ws.send(json.dumps({
            "id":     1,
            "cmd":    "subscribe",
            "params": {"channels": ["orderbook_delta"], "market_tickers": [ticker]},
        }))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type", "")
            if mtype == "error":
                print(f"{R}  WS error: {msg}{RST}")
                return
            payload = msg.get("msg") or {}
            if mtype == "orderbook_snapshot":
                book.apply_snapshot(payload.get("yes_dollars_fp") or [],
                                    payload.get("no_dollars_fp")  or [])
                ready_event.set()
            elif mtype == "orderbook_delta":
                side  = payload.get("side")
                p_str = payload.get("price_dollars")
                d_str = payload.get("delta_fp")
                if side and p_str is not None and d_str is not None:
                    try:
                        book.apply_delta(side, float(p_str), float(d_str))
                    except (TypeError, ValueError):
                        pass


# ---------------------------------------------------------------------------
# Order placement (market FoK)
# ---------------------------------------------------------------------------

async def place_market_order(session: aiohttp.ClientSession, pk,
                             ticker: str, action: str, side: str,
                             count: int) -> dict | None:
    """Limit + IOC at extreme price — sweeps all available depth at any price."""
    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "time_in_force":   "immediate_or_cancel",
        "action":          action,
        "side":            side,
        "count":           count,
    }
    if action == "buy":
        body["yes_price" if side == "yes" else "no_price"] = 99
    else:
        body["yes_price" if side == "yes" else "no_price"] = 1

    hdrs = auth_headers(pk, KALSHI_KEY_ID, "POST", path)
    hdrs["Content-Type"] = "application/json"
    print(f"{DIM}  -> body: {json.dumps(body)}{RST}")
    try:
        async with session.post(KALSHI_REST + path, headers=hdrs,
                                json=body,
                                timeout=aiohttp.ClientTimeout(total=15)) as r:
            text = await r.text()
            print(f"{DIM}  <- HTTP {r.status}: {text}{RST}")
            if r.status not in (200, 201):
                print(f"{R}  ORDER REJECTED {r.status}: {text}{RST}")
                return None
            data = json.loads(text)
            return data.get("order", data)
    except Exception as e:
        print(f"{R}  Order exception: {e}{RST}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-usd", type=float, default=1.0,
                    help="approx dollars to spend on the test buy")
    ap.add_argument("--hold-s", type=float, default=5.0,
                    help="seconds to hold before selling")
    args = ap.parse_args()

    pk = load_private_key()

    async with aiohttp.ClientSession() as session:
        # ── Discover ticker ─────────────────────────────────────────────────
        print(f"{C}  Discovering active {KALSHI_SERIES} market...{RST}")
        mkt = await discover_market(session)
        if not mkt:
            sys.exit(f"  no open markets in {KALSHI_SERIES}")
        ticker = mkt["ticker"]
        floor  = float(mkt.get("floor_strike") or 0)
        print(f"{C}  Ticker: {ticker}  Strike: ${floor:,.2f}{RST}")

        # ── Subscribe via WS ────────────────────────────────────────────────
        book = LiveBook()
        ready = asyncio.Event()
        ws_task = asyncio.create_task(ws_loop(book, ticker, ready))

        try:
            print(f"{DIM}  waiting for WS snapshot...{RST}")
            try:
                await asyncio.wait_for(ready.wait(), timeout=15)
            except asyncio.TimeoutError:
                sys.exit("  WS snapshot timeout — connection issue")

            # Give book a sec to settle from initial deltas
            await asyncio.sleep(1.0)

            yes_bid, yes_bid_sz, yes_ask, yes_ask_sz = book.yes_bid_ask()
            if yes_ask == 0:
                sys.exit("  book empty — nothing to trade against")

            print(f"{G}  Live book: YES bid={yes_bid:.2f} ({yes_bid_sz:.0f}c)  "
                  f"ask={yes_ask:.2f} ({yes_ask_sz:.0f}c){RST}")

            # ── Size the order to target_usd, capped to ask depth ──────────
            count_target  = max(1, int(args.target_usd / max(yes_ask, 0.01)))
            count_capped  = max(1, min(count_target, int(yes_ask_sz)))
            est_cost      = count_capped * yes_ask
            print(f"{C}  Plan: market BUY {count_capped} contracts (target ${args.target_usd:.2f}, "
                  f"limited to {int(yes_ask_sz)} available at ask)  est_cost=${est_cost:.2f}{RST}")

            # ── BUY ────────────────────────────────────────────────────────
            print(f"\n{BOLD}[1/2] Sending market BUY {count_capped}x YES...{RST}")
            t0 = time.monotonic()
            buy = await place_market_order(session, pk, ticker, "buy", "yes", count_capped)
            buy_ms = (time.monotonic() - t0) * 1000
            if not buy:
                sys.exit("  BUY rejected")
            buy_filled    = int(float(buy.get("fill_count_fp") or 0))
            buy_cost_usd  = float(buy.get("taker_fill_cost_dollars") or 0)
            buy_avg_price = (buy_cost_usd / buy_filled) if buy_filled > 0 else 0.0
            buy_status    = buy.get("status", "?")
            print(f"  Status: {buy_status}  Filled: {buy_filled}/{count_capped}  "
                  f"avg=${buy_avg_price:.3f}  cost=${buy_cost_usd:.2f}  ({buy_ms:.0f}ms)")
            if buy_filled <= 0:
                sys.exit(f"{R}  BUY filled 0 contracts — no position to sell{RST}")

            # ── HOLD ───────────────────────────────────────────────────────
            print(f"\n{DIM}[hold] {args.hold_s:.0f}s...{RST}")
            await asyncio.sleep(args.hold_s)

            # Show book at sell time
            yes_bid_now, yes_bid_sz_now, yes_ask_now, yes_ask_sz_now = book.yes_bid_ask()
            print(f"{G}  Live book now: YES bid={yes_bid_now:.2f} ({yes_bid_sz_now:.0f}c)  "
                  f"ask={yes_ask_now:.2f} ({yes_ask_sz_now:.0f}c){RST}")

            # ── SELL ───────────────────────────────────────────────────────
            print(f"\n{BOLD}[2/2] Sending market SELL {buy_filled}x YES...{RST}")
            t0 = time.monotonic()
            sell = await place_market_order(session, pk, ticker, "sell", "yes", buy_filled)
            sell_ms = (time.monotonic() - t0) * 1000
            if not sell:
                print(f"{R}  SELL rejected — POSITION STILL OPEN, check Kalshi UI{RST}")
                sys.exit(1)
            sell_filled       = int(float(sell.get("fill_count_fp") or 0))
            # For SELL orders, taker_fill_cost_dollars is what the BUYER paid for the
            # complementary side. Our actual proceeds = count - taker_fill_cost_dollars.
            sell_buyer_cost   = float(sell.get("taker_fill_cost_dollars") or 0)
            sell_proceeds_usd = max(0.0, sell_filled - sell_buyer_cost)
            sell_avg_price    = (sell_proceeds_usd / sell_filled) if sell_filled > 0 else 0.0
            sell_status       = sell.get("status", "?")
            print(f"  Status: {sell_status}  Filled: {sell_filled}/{buy_filled}  "
                  f"avg=${sell_avg_price:.3f}  proceeds=${sell_proceeds_usd:.2f}  ({sell_ms:.0f}ms)")

            # ── P&L (real, from actual fill costs) ─────────────────────────
            buy_fees  = float(buy.get("taker_fees_dollars")  or 0)
            sell_fees = float(sell.get("taker_fees_dollars") or 0)
            pnl       = sell_proceeds_usd - buy_cost_usd - buy_fees - sell_fees
            print(f"\n{BOLD}  Round-trip summary:{RST}")
            print(f"    Bought {buy_filled}x @ ${buy_avg_price:.3f}  cost=${buy_cost_usd:.2f}  fees=${buy_fees:.3f}")
            print(f"    Sold   {sell_filled}x @ ${sell_avg_price:.3f}  proceeds=${sell_proceeds_usd:.2f}  fees=${sell_fees:.3f}")
            color = G if pnl >= 0 else R
            print(f"    {color}Net P&L: ${pnl:+.2f}{RST}")
            if buy_filled != sell_filled:
                print(f"{R}  WARNING: {buy_filled - sell_filled} contracts unsold — CHECK KALSHI UI{RST}")

        finally:
            ws_task.cancel()
            try:
                await ws_task
            except (asyncio.CancelledError, Exception):
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{DIM}  stopped{RST}")
