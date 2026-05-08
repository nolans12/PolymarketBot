"""
test_trade_maker.py — Maker entry / taker exit round-trip via WS.

Verifies the maker workflow on Kalshi:
  1. Connect to WS, wait for snapshot
  2. POST a resting BUY YES limit at the bid (or bid+1c with --aggressive)
  3. Poll for fill up to --ttl seconds
  4. If filled: hold --hold-s, then SELL via taker IOC at current bid
  5. If not filled: cancel the order

P&L uses correct Kalshi field semantics:
  Buy cost      = taker_fill_cost_dollars  (or maker_fill_cost_dollars if maker)
  Sell proceeds = fill_count - taker_fill_cost_dollars  (Kalshi reciprocal)
  Fees          = maker_fees_dollars + taker_fees_dollars  (maker is 0)

No confirmation prompt — fires immediately. Use small --target-usd.

Usage:
  python scripts/test/test_trade_maker.py
  python scripts/test/test_trade_maker.py --aggressive --target-usd 1.0 --ttl 30 --hold-s 5
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

G    = "\033[92m"
R    = "\033[91m"
Y    = "\033[93m"
C    = "\033[96m"
DIM  = "\033[2m"
RST  = "\033[0m"
BOLD = "\033[1m"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

async def discover_market(session: aiohttp.ClientSession) -> dict | None:
    path   = "/trade-api/v2/markets"
    params = {"series_ticker": KALSHI_SERIES, "status": "open", "limit": 50}
    async with session.get(KALSHI_REST + path, params=params,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()
    now = time.time()
    candidates = []
    for mkt in data.get("markets", []):
        ct = mkt.get("close_time", "")
        if not ct or mkt.get("status") not in ("open", "active"):
            continue
        try:
            ep = _dt.datetime.fromisoformat(ct.replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        ttl = ep - now
        if ttl < 60:
            continue
        try:
            yb = float(mkt.get("yes_bid_dollars") or 0)
            ya = float(mkt.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            continue
        if yb <= 0 or ya <= 0 or ya <= yb:
            continue
        ym = (yb + ya) / 2.0
        candidates.append({"ttl": ttl, "yb": yb, "ya": ya, "centrality": abs(ym - 0.5), "mkt": mkt})
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c["centrality"], -c["ttl"]))
    return candidates[0]["mkt"]


# ---------------------------------------------------------------------------
# Live WS book
# ---------------------------------------------------------------------------

class LiveBook:
    def __init__(self):
        self.yes_book: dict[float, float] = {}
        self.no_book:  dict[float, float] = {}
        self.ready = False

    def apply_snapshot(self, yes_fp, no_fp):
        self.yes_book = {float(p): float(s) for p, s in yes_fp}
        self.no_book  = {float(p): float(s) for p, s in no_fp}
        self.ready    = True

    def apply_delta(self, side, price, delta):
        book = self.yes_book if side == "yes" else self.no_book
        book[price] = book.get(price, 0.0) + delta
        if book[price] <= 0:
            book.pop(price, None)

    def yes_bid_ask(self) -> tuple[float, float, float, float]:
        yes_live = {p: s for p, s in self.yes_book.items() if s >= 1.0}
        no_live  = {p: s for p, s in self.no_book.items()  if s >= 1.0}
        if not yes_live or not no_live:
            return 0.0, 0.0, 0.0, 0.0
        yb_p = max(yes_live)
        nb_p = max(no_live)
        return yb_p, yes_live[yb_p], 1.0 - nb_p, no_live[nb_p]


async def ws_loop(book: LiveBook, ticker: str, ready: asyncio.Event):
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
                ready.set()
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
# REST helpers
# ---------------------------------------------------------------------------

async def place_limit_resting(session, pk, ticker, action, side, price_cents, count) -> dict | None:
    """Resting limit order — no time_in_force, sits on the book until filled or cancelled."""
    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          action,
        "side":            side,
        "count":           count,
    }
    body["yes_price" if side == "yes" else "no_price"] = price_cents

    hdrs = auth_headers(pk, KALSHI_KEY_ID, "POST", path)
    hdrs["Content-Type"] = "application/json"
    print(f"{DIM}  -> body: {json.dumps(body)}{RST}")
    async with session.post(KALSHI_REST + path, headers=hdrs, json=body,
                            timeout=aiohttp.ClientTimeout(total=15)) as r:
        text = await r.text()
        print(f"{DIM}  <- HTTP {r.status}: {text[:400]}{RST}")
        if r.status not in (200, 201):
            return None
        return json.loads(text).get("order")


async def place_taker_ioc(session, pk, ticker, action, side, count) -> dict | None:
    """Limit + IOC at sweep price — fills whatever is on the book right now."""
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
    async with session.post(KALSHI_REST + path, headers=hdrs, json=body,
                            timeout=aiohttp.ClientTimeout(total=15)) as r:
        text = await r.text()
        print(f"{DIM}  <- HTTP {r.status}: {text[:400]}{RST}")
        if r.status not in (200, 201):
            return None
        return json.loads(text).get("order")


async def get_order(session, pk, order_id) -> dict | None:
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    hdrs = auth_headers(pk, KALSHI_KEY_ID, "GET", path)
    async with session.get(KALSHI_REST + path, headers=hdrs,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200:
            return None
        return (await r.json()).get("order")


async def cancel_order(session, pk, order_id) -> bool:
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    hdrs = auth_headers(pk, KALSHI_KEY_ID, "DELETE", path)
    async with session.delete(KALSHI_REST + path, headers=hdrs,
                              timeout=aiohttp.ClientTimeout(total=10)) as r:
        return r.status in (200, 204)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-usd", type=float, default=1.0,
                    help="approximate dollars to spend on the buy")
    ap.add_argument("--ttl",        type=int,   default=30,
                    help="seconds to wait for the maker buy to fill before cancelling")
    ap.add_argument("--hold-s",     type=int,   default=5,
                    help="seconds to hold after fill before exiting")
    ap.add_argument("--aggressive", action="store_true",
                    help="post at bid+1c (one tick inside spread) instead of at bid")
    args = ap.parse_args()

    pk = load_private_key()

    async with aiohttp.ClientSession() as session:
        # ── Discover ───────────────────────────────────────────────────────
        print(f"{C}  Discovering active {KALSHI_SERIES} market...{RST}")
        mkt = await discover_market(session)
        if not mkt:
            sys.exit(f"  no tradeable market in {KALSHI_SERIES}")
        ticker = mkt["ticker"]
        floor  = float(mkt.get("floor_strike") or 0)
        print(f"{C}  Ticker: {ticker}  Strike: ${floor:,.2f}{RST}")

        # ── Subscribe via WS ───────────────────────────────────────────────
        book   = LiveBook()
        ready  = asyncio.Event()
        ws_task = asyncio.create_task(ws_loop(book, ticker, ready))

        try:
            print(f"{DIM}  waiting for WS snapshot...{RST}")
            try:
                await asyncio.wait_for(ready.wait(), timeout=15)
            except asyncio.TimeoutError:
                sys.exit("  WS snapshot timeout")
            await asyncio.sleep(1.0)  # let initial deltas settle

            yes_bid, yes_bid_sz, yes_ask, yes_ask_sz = book.yes_bid_ask()
            if yes_ask == 0 or yes_bid == 0:
                sys.exit("  book empty")
            spread_c = round((yes_ask - yes_bid) * 100)
            print(f"{G}  Live book: YES bid={yes_bid:.2f} ({yes_bid_sz:.0f}c)  "
                  f"ask={yes_ask:.2f} ({yes_ask_sz:.0f}c)  spread={spread_c}c{RST}")

            # ── Decide post price ──────────────────────────────────────────
            if args.aggressive and spread_c < 2:
                print(f"{Y}  spread < 2c — falling back to passive (at bid){RST}")
                args.aggressive = False
            if args.aggressive:
                post_cents = round(yes_bid * 100) + 1
                mode_label = "AGGRESSIVE (bid+1c)"
            else:
                post_cents = round(yes_bid * 100)
                mode_label = "PASSIVE (at bid)"
            post_cents   = max(1, min(99, post_cents))
            post_dollars = post_cents / 100.0
            count        = max(1, int(round(args.target_usd / post_dollars)))

            print(f"{C}  Plan ({mode_label}): post BUY {count}x YES @ {post_cents}c "
                  f"≈ ${count * post_dollars:.2f}, wait up to {args.ttl}s for fill{RST}")

            # ── POST resting maker buy ─────────────────────────────────────
            print(f"\n{BOLD}[1/4] Posting maker BUY {count}x YES @ {post_cents}c...{RST}")
            t_post = time.monotonic()
            buy = await place_limit_resting(session, pk, ticker, "buy", "yes", post_cents, count)
            if not buy:
                sys.exit(f"{R}  POST rejected{RST}")
            order_id    = buy.get("order_id")
            initial_fill = int(float(buy.get("fill_count_fp") or 0))
            initial_status = buy.get("status", "?")
            print(f"  Order ID: {order_id}  Status: {initial_status}  "
                  f"Initial fill: {initial_fill}/{count}")

            # If it crossed and filled immediately, treat as taker
            if initial_fill >= count:
                print(f"{Y}  Note: filled immediately — order crossed the spread (taker, not maker){RST}")
                final_buy = buy
            else:
                # ── Poll for fill ──────────────────────────────────────────
                print(f"\n{BOLD}[2/4] Polling for fill (TTL {args.ttl}s)...{RST}")
                final_buy = buy
                last_fill = initial_fill
                for elapsed in range(1, args.ttl + 1):
                    await asyncio.sleep(1.0)
                    state = await get_order(session, pk, order_id)
                    if not state:
                        continue
                    cur_fill = int(float(state.get("fill_count_fp") or 0))
                    cur_st   = state.get("status", "?")
                    if cur_fill != last_fill:
                        print(f"  {elapsed}s: status={cur_st}  filled={cur_fill}/{count}")
                        last_fill = cur_fill
                    if cur_fill >= count:
                        final_buy = state
                        break
                else:
                    # ── No fill — cancel ───────────────────────────────────
                    print(f"\n{BOLD}[3/4] No fill within TTL. Cancelling {order_id}...{RST}")
                    cancelled = await cancel_order(session, pk, order_id)
                    state     = await get_order(session, pk, order_id)
                    final_fill = int(float(state.get("fill_count_fp") or 0)) if state else last_fill
                    print(f"  Cancelled: {cancelled}  Final fill: {final_fill}/{count}")
                    if final_fill <= 0:
                        print(f"\n{Y}  RESULT: NO FILL — maker post sat at {post_cents}c for {args.ttl}s{RST}")
                        return
                    print(f"{Y}  Partial maker fill: {final_fill} contracts. Continuing to sell.{RST}")
                    final_buy = state

            # ── Pull buy details ───────────────────────────────────────────
            buy_filled    = int(float(final_buy.get("fill_count_fp") or 0))
            buy_taker_cost = float(final_buy.get("taker_fill_cost_dollars") or 0)
            buy_maker_cost = float(final_buy.get("maker_fill_cost_dollars") or 0)
            buy_cost_usd  = buy_taker_cost + buy_maker_cost
            buy_taker_fee = float(final_buy.get("taker_fees_dollars") or 0)
            buy_maker_fee = float(final_buy.get("maker_fees_dollars") or 0)
            buy_fees      = buy_taker_fee + buy_maker_fee
            buy_avg_price = (buy_cost_usd / buy_filled) if buy_filled > 0 else 0.0
            t_fill        = time.monotonic()
            print(f"\n{G}  ✓ FILLED {buy_filled}x  avg=${buy_avg_price:.3f}  cost=${buy_cost_usd:.2f}  "
                  f"fees=${buy_fees:.3f} (maker=${buy_maker_fee:.3f}, taker=${buy_taker_fee:.3f})  "
                  f"fill_time={t_fill - t_post:.1f}s{RST}")

            # ── Hold ───────────────────────────────────────────────────────
            print(f"\n{DIM}[hold] {args.hold_s}s...{RST}")
            await asyncio.sleep(args.hold_s)
            yes_bid_now, _, yes_ask_now, _ = book.yes_bid_ask()
            print(f"{G}  Live book now: bid={yes_bid_now:.2f}  ask={yes_ask_now:.2f}{RST}")

            # ── Taker exit ─────────────────────────────────────────────────
            print(f"\n{BOLD}[4/4] Taker SELL {buy_filled}x YES (IOC sweep)...{RST}")
            sell = await place_taker_ioc(session, pk, ticker, "sell", "yes", buy_filled)
            if not sell:
                print(f"{R}  SELL rejected — POSITION STILL OPEN, check Kalshi UI{RST}")
                sys.exit(1)
            sell_filled    = int(float(sell.get("fill_count_fp") or 0))
            # Reciprocal: proceeds = count - taker_fill_cost_dollars
            sell_buyer_cost = float(sell.get("taker_fill_cost_dollars") or 0)
            sell_proceeds  = max(0.0, sell_filled - sell_buyer_cost)
            sell_fees      = float(sell.get("taker_fees_dollars") or 0)
            sell_avg_price = (sell_proceeds / sell_filled) if sell_filled > 0 else 0.0
            print(f"  Filled: {sell_filled}/{buy_filled}  avg=${sell_avg_price:.3f}  "
                  f"proceeds=${sell_proceeds:.2f}  fees=${sell_fees:.3f}")

            # ── P&L ────────────────────────────────────────────────────────
            pnl = sell_proceeds - buy_cost_usd - buy_fees - sell_fees
            print(f"\n{BOLD}  Round-trip summary (maker entry → taker exit):{RST}")
            print(f"    Bought {buy_filled}x @ ${buy_avg_price:.3f}  cost=${buy_cost_usd:.2f}  fees=${buy_fees:.3f}")
            print(f"    Sold   {sell_filled}x @ ${sell_avg_price:.3f}  proceeds=${sell_proceeds:.2f}  fees=${sell_fees:.3f}")
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
