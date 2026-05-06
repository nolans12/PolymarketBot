"""
test_trade.py — Buy ~$1 of YES on BTC 15-min, wait 10 seconds, then sell it back.

Verifies the full round-trip: buy fill -> hold -> sell fill.

Usage:
  python scripts/test_trade.py           # prompts before executing
  python scripts/test_trade.py --dry-run # prints what it would do, no orders placed

Auth: KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_FILE (or KALSHI_PRIVATE_KEY_PEM) from .env
"""

import argparse
import asyncio
import base64
import json
import sys
import time
import uuid
import datetime
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betbot.kalshi.auth import load_private_key
from betbot.kalshi.config import KALSHI_REST, KALSHI_KEY_ID, KALSHI_SERIES

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

HOLD_SECONDS = 10


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _sign(pk, ts: str, method: str, path: str) -> str:
    msg = (ts + method + path).encode()
    sig = pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def _auth_headers(pk, method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": _sign(pk, ts, method, path),
        "Content-Type":            "application/json",
    }


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

async def discover_market(session: aiohttp.ClientSession) -> dict | None:
    path   = "/trade-api/v2/markets"
    params = {"series_ticker": KALSHI_SERIES, "status": "open", "limit": 20}
    async with session.get(KALSHI_REST + path, params=params,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()

    best_mkt   = None
    best_close = float("inf")
    for mkt in data.get("markets", []):
        ct = mkt.get("close_time", "")
        if ct and mkt.get("status") in ("open", "active"):
            try:
                epoch = datetime.datetime.fromisoformat(
                    ct.replace("Z", "+00:00")).timestamp()
                if epoch < best_close:
                    best_close = epoch
                    best_mkt   = mkt
            except Exception:
                pass
    return best_mkt


async def refresh_market(session: aiohttp.ClientSession, pk,
                         ticker: str) -> dict | None:
    """Re-fetch a single market to get the latest bid/ask."""
    path = f"/trade-api/v2/markets/{ticker}"
    hdrs = _auth_headers(pk, "GET", path)
    async with session.get(KALSHI_REST + path, headers=hdrs,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200:
            return None
        data = await r.json()
    return data.get("market", data)


async def place_order(session: aiohttp.ClientSession, pk,
                      ticker: str, action: str, yes_price_cents: int,
                      count: int, dry_run: bool) -> dict | None:
    """
    action: "buy" or "sell"
    yes_price_cents: limit price in cents (1-99)
    count: integer number of contracts
    """
    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          action,
        "side":            "yes",
        "count":           count,
        "yes_price":       yes_price_cents,
    }

    label = "BUY " if action == "buy" else "SELL"
    if dry_run:
        print(f"  [DRY RUN] {label} {count}x YES @ {yes_price_cents}c")
        print(f"  Body: {json.dumps(body)}")
        return {"order": {"order_id": "DRY-RUN", "status": "dry_run", "filled_count": count}}

    hdrs = _auth_headers(pk, "POST", path)
    async with session.post(KALSHI_REST + path, headers=hdrs,
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=15)) as r:
        text = await r.text()
        if r.status not in (200, 201):
            print(f"  ERROR {r.status}: {text}")
            print(f"  Sent: {json.dumps(body)}")
            return None
        return json.loads(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Buy YES then sell 10s later to verify round-trip on Kalshi")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-usd", type=float, default=1.0)
    args = parser.parse_args()

    if not KALSHI_KEY_ID:
        sys.exit("ERROR: KALSHI_API_KEY_ID not set in .env")

    pk = load_private_key()

    async with aiohttp.ClientSession() as session:

        # ── 1. Discover market ──────────────────────────────────────────────
        print("Discovering active BTC 15-min market...")
        mkt = await discover_market(session)
        if not mkt:
            sys.exit("ERROR: no open KXBTC15M market found")

        ticker      = mkt["ticker"]
        floor_str   = mkt.get("floor_strike", "?")
        close_time  = mkt.get("close_time", "")
        yes_ask_str = mkt.get("yes_ask_dollars", "")
        yes_bid_str = mkt.get("yes_bid_dollars", "")

        try:
            close_dt = datetime.datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            tau_s    = max(0, close_dt.timestamp() - time.time())
        except Exception:
            tau_s = 0

        print(f"  Ticker:      {ticker}")
        print(f"  Strike:      ${floor_str:,.2f}")
        print(f"  Closes in:   {tau_s:.0f}s")
        print(f"  YES bid/ask: ${yes_bid_str} / ${yes_ask_str}")

        if not yes_ask_str:
            sys.exit("ERROR: market has no YES ask price")

        yes_ask_usd   = float(yes_ask_str)
        yes_ask_cents = round(yes_ask_usd * 100)
        count         = max(1, int(round(args.target_usd / yes_ask_usd)))
        actual_cost   = count * yes_ask_usd

        print(f"\n  Plan:")
        print(f"    1. BUY  {count}x YES @ {yes_ask_cents}c  (~${actual_cost:.2f})")
        print(f"    2. Wait {HOLD_SECONDS}s")
        print(f"    3. SELL {count}x YES @ current bid")

        if not args.dry_run:
            confirm = input("\nProceed? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Aborted.")
                return

        # ── 2. BUY ─────────────────────────────────────────────────────────
        print(f"\n[1/3] Placing BUY {count}x YES @ {yes_ask_cents}c...")
        buy_resp = await place_order(session, pk, ticker,
                                     "buy", yes_ask_cents, count, args.dry_run)
        if not buy_resp:
            sys.exit("Buy failed — aborting before any sell.")

        buy_order = buy_resp.get("order", buy_resp)
        buy_id     = buy_order.get("order_id", buy_order.get("id", "?"))
        buy_filled = buy_order.get("filled_count", 0)
        print(f"    Order ID: {buy_id}")
        print(f"    Status:   {buy_order.get('status', '?')}")
        print(f"    Filled:   {buy_filled} / {count}")

        # ── 3. WAIT ─────────────────────────────────────────────────────────
        print(f"\n[2/3] Holding for {HOLD_SECONDS}s...")
        for remaining in range(HOLD_SECONDS, 0, -1):
            print(f"    {remaining}s...", end="\r", flush=True)
            await asyncio.sleep(1)
        print()

        # ── 4. Get current bid for sell price ───────────────────────────────
        print("[3/3] Fetching current market price for sell...")
        fresh = await refresh_market(session, pk, ticker)
        if fresh and not args.dry_run:
            yes_bid_now   = fresh.get("yes_bid_dollars", yes_bid_str)
            yes_ask_now   = fresh.get("yes_ask_dollars", yes_ask_str)
            yes_bid_cents = round(float(yes_bid_now) * 100)
            print(f"    YES bid/ask now: ${yes_bid_now} / ${yes_ask_now}")
        else:
            # dry-run: invent a bid 1c below ask
            yes_bid_cents = max(1, yes_ask_cents - 1)
            print(f"    [DRY RUN] Using simulated bid: {yes_bid_cents}c")

        # ── 5. SELL ─────────────────────────────────────────────────────────
        print(f"    Placing SELL {count}x YES @ {yes_bid_cents}c...")
        sell_resp = await place_order(session, pk, ticker,
                                      "sell", yes_bid_cents, count, args.dry_run)
        if not sell_resp:
            print("  SELL FAILED — position may still be open, check Kalshi UI")
            sys.exit(1)

        sell_order  = sell_resp.get("order", sell_resp)
        sell_id     = sell_order.get("order_id", sell_order.get("id", "?"))
        sell_filled = sell_order.get("filled_count", 0)
        print(f"    Order ID: {sell_id}")
        print(f"    Status:   {sell_order.get('status', '?')}")
        print(f"    Filled:   {sell_filled} / {count}")

        # ── 6. P&L summary ──────────────────────────────────────────────────
        gross_in  = count * yes_ask_usd
        gross_out = count * (yes_bid_cents / 100.0)
        pnl       = gross_out - gross_in
        print(f"\n  Round-trip summary:")
        print(f"    Bought {count}x @ {yes_ask_cents}c  = ${gross_in:.2f}")
        print(f"    Sold   {count}x @ {yes_bid_cents}c  = ${gross_out:.2f}")
        print(f"    P&L:   ${pnl:+.2f}  (before fees)")
        if args.dry_run:
            print("  [DRY RUN — no real orders placed]")


if __name__ == "__main__":
    asyncio.run(main())
