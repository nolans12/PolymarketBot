"""
orders.py -- Kalshi REST order placement and portfolio balance.

Body shape on POST /portfolio/orders matches scripts/test_trade.py
(known-good): no time_in_force, no expiration_ts. Default Kalshi
behaviour is rest-until-filled-or-canceled. We always cross the
spread for entry-as-taker / exit-as-taker, so taker fills happen on
placement and we don't need to poll.
"""

import json
import logging
import uuid
from typing import Optional

import aiohttp

from betbot.kalshi.auth import auth_headers
from betbot.kalshi.config import KALSHI_REST, KALSHI_KEY_ID

log = logging.getLogger(__name__)

ORDER_PATH   = "/trade-api/v2/portfolio/orders"
BALANCE_PATH = "/trade-api/v2/portfolio/balance"


async def place_order(session: aiohttp.ClientSession, pk,
                      ticker: str, action: str, side: str,
                      price_cents: int, count: int) -> Optional[dict]:
    """
    Place a limit order. Returns the inner `order` dict on success, None on failure.

    action: "buy" or "sell"
    side:   "yes" or "no"
    price_cents: 1..99 (clamped here defensively)
    count: integer contract count (>=1)
    """
    if count < 1:
        log.warning("place_order: count<1, refusing to send")
        return None
    price_cents = max(1, min(99, int(price_cents)))

    body: dict = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          action,
        "side":            side,
        "count":           int(count),
    }
    if side == "yes":
        body["yes_price"] = price_cents
    elif side == "no":
        body["no_price"] = price_cents
    else:
        log.error(f"place_order: invalid side {side!r}")
        return None

    hdrs = auth_headers(pk, KALSHI_KEY_ID, "POST", ORDER_PATH)
    hdrs["Content-Type"] = "application/json"

    try:
        async with session.post(KALSHI_REST + ORDER_PATH, headers=hdrs,
                                json=body,
                                timeout=aiohttp.ClientTimeout(total=15)) as r:
            text = await r.text()
            if r.status not in (200, 201):
                log.error(f"Order REJECTED {r.status}: {text}  body={json.dumps(body)}")
                return None
            data = json.loads(text)
            return data.get("order", data)
    except Exception as e:
        log.error(f"Order placement exception: {e}  body={json.dumps(body)}")
        return None


async def get_balance_usd(session: aiohttp.ClientSession, pk) -> float:
    """
    Query the current cash balance in USD. Returns 0.0 on failure (caller
    should treat that as fatal -- can't size trades against zero).

    Kalshi returns balance in cents.
    """
    hdrs = auth_headers(pk, KALSHI_KEY_ID, "GET", BALANCE_PATH)
    try:
        async with session.get(KALSHI_REST + BALANCE_PATH, headers=hdrs,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                log.error(f"Balance query failed {r.status}: {await r.text()}")
                return 0.0
            data = await r.json()
            balance_cents = data.get("balance")
            if balance_cents is None:
                log.error(f"Balance response missing 'balance' field: {data}")
                return 0.0
            return float(balance_cents) / 100.0
    except Exception as e:
        log.error(f"Balance query exception: {e}")
        return 0.0
