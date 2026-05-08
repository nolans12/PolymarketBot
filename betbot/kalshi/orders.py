"""
orders.py -- Kalshi REST order placement and portfolio balance.

All orders use type="market" + time_in_force="immediate_or_cancel".
This means: fill as much as the book offers right now, cancel the rest —
never rests, never blocks. Partial fills are reported via filled_count.
"""

import json
import logging
import uuid
from typing import Optional

import aiohttp

from betbot.kalshi.auth import auth_headers
from betbot.kalshi.config import KALSHI_REST, KALSHI_KEY_ID, SIZE_MAX_USD

log = logging.getLogger(__name__)

ORDER_PATH   = "/trade-api/v2/portfolio/orders"
BALANCE_PATH = "/trade-api/v2/portfolio/balance"


async def place_order(session: aiohttp.ClientSession, pk,
                      ticker: str, action: str, side: str,
                      price_cents: int, count: int) -> Optional[dict]:
    """
    Place a market FoK order. Fills immediately and completely or rejects — never rests.

    action:      "buy" or "sell"
    side:        "yes" or "no"
    price_cents: kept for P&L accounting only — market order ignores it for matching
    count:       integer contract count (>=1), hard-capped to SIZE_MAX_USD worth

    Returns the order dict on success (filled_count > 0), None on rejection.
    """
    if count < 1:
        log.debug("place_order: count<1, skipping")
        return None

    # Hard cap: never send more contracts than SIZE_MAX_USD allows at this price
    price_usd  = max(0.01, min(0.99, price_cents / 100.0))
    max_count  = max(1, int(SIZE_MAX_USD / price_usd))
    count      = min(int(count), max_count)

    # Kalshi requires a price field even on market orders — acts as a
    # max-price ceiling for buys and min-price floor for sells.
    # For market FoK we want "any price", so 99/1 are the extremes.
    body: dict = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "market",
        "time_in_force":   "immediate_or_cancel",
        "action":          action,
        "side":            side,
        "count":           count,
    }
    if action == "buy":
        body["yes_price" if side == "yes" else "no_price"] = 99
    else:
        body["yes_price" if side == "yes" else "no_price"] = 1

    hdrs = auth_headers(pk, KALSHI_KEY_ID, "POST", ORDER_PATH)
    hdrs["Content-Type"] = "application/json"

    try:
        async with session.post(KALSHI_REST + ORDER_PATH, headers=hdrs,
                                json=body,
                                timeout=aiohttp.ClientTimeout(total=15)) as r:
            text = await r.text()
            if r.status not in (200, 201):
                log.debug("Order REJECTED %s: %s", r.status, text)
                return None
            data = json.loads(text)
            return data.get("order", data)
    except Exception as e:
        log.debug("Order exception: %s", e)
        return None


async def get_balance_usd(session: aiohttp.ClientSession, pk) -> float:
    """Returns current cash balance in USD. Returns 0.0 on failure."""
    hdrs = auth_headers(pk, KALSHI_KEY_ID, "GET", BALANCE_PATH)
    try:
        async with session.get(KALSHI_REST + BALANCE_PATH, headers=hdrs,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                log.debug("Balance query failed %s", r.status)
                return 0.0
            data = await r.json()
            balance_cents = data.get("balance")
            if balance_cents is None:
                log.debug("Balance response missing 'balance': %s", data)
                return 0.0
            return float(balance_cents) / 100.0
    except Exception as e:
        log.debug("Balance query exception: %s", e)
        return 0.0
