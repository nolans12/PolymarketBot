"""
market_data.py — Polymarket 5-minute crypto market discovery and price polling.

Markets follow the slug pattern: {coin}-updown-5m-{unix_timestamp}
where unix_timestamp is the window start time (UTC, rounded to 5-min boundaries).

Each event has one market with two tokens:
  clobTokenIds[0] = UP token
  clobTokenIds[1] = DOWN token
  outcomePrices[0] = current UP price = q^(w)
  outcomePrices[1] = current DOWN price = 1 - q^(w)
"""

import time
import logging
import json
import requests
from typing import Optional

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"

COIN_SLUGS = {
    "BTC": "btc",
    "ETH": "eth",
    "SOL": "sol",
    "XRP": "xrp",
}

WINDOW_SECONDS = 300  # 5 minutes


def current_window_ts() -> int:
    """Return the Unix timestamp of the current 5-min window start."""
    now = int(time.time())
    return now - (now % WINDOW_SECONDS)


def next_window_ts() -> int:
    return current_window_ts() + WINDOW_SECONDS


def fetch_market(coin: str, window_ts: int) -> Optional[dict]:
    """
    Fetch a single 5-min market event from the gamma API.
    Returns the parsed market dict with token IDs and prices, or None.
    """
    slug = f"{COIN_SLUGS[coin]}-updown-5m-{window_ts}"
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        events = data if isinstance(data, list) else data.get("events", data.get("data", []))
        if not events:
            return None

        event = events[0]
        markets = event.get("markets", [])
        if not markets:
            return None

        m = markets[0]
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        outcome_prices = json.loads(m.get("outcomePrices", "[]"))
        outcomes = json.loads(m.get("outcomes", "[]"))

        if len(token_ids) < 2 or len(outcome_prices) < 2:
            return None

        # outcomes[0] = "Up", outcomes[1] = "Down"
        up_idx = 0 if outcomes[0].lower() == "up" else 1
        down_idx = 1 - up_idx

        return {
            "coin": coin,
            "window_ts": window_ts,
            "slug": slug,
            "up_token_id": token_ids[up_idx],
            "down_token_id": token_ids[down_idx],
            "up_price": float(outcome_prices[up_idx]),    # q^(w) for UP
            "down_price": float(outcome_prices[down_idx]), # q^(w) for DOWN
            "end_date": m.get("endDate"),
            "active": m.get("active", False),
            "closed": m.get("closed", False),
            "condition_id": m.get("conditionId"),
        }
    except Exception as e:
        logger.warning(f"Failed to fetch {coin} window {window_ts}: {e}")
        return None


class MarketDataClient:
    """
    Fetches and caches the current 5-minute markets for all coins.
    Automatically detects window rollovers and re-fetches token IDs.
    """

    def __init__(self, clob_host: str = CLOB_HOST):
        self.clob_host = clob_host
        self._cache: dict[str, dict] = {}   # coin -> market dict
        self._cache_ts: int = 0             # window_ts these were fetched for

    def _refresh_if_needed(self) -> None:
        """Re-fetch markets if the window has rolled over."""
        ts = current_window_ts()
        if ts != self._cache_ts:
            logger.info(f"New 5-min window: {ts}. Fetching market token IDs...")
            new_cache = {}
            for coin in COIN_SLUGS:
                market = fetch_market(coin, ts)
                if market:
                    new_cache[coin] = market
                    logger.info(
                        f"  {coin}: up_token={market['up_token_id'][:12]}... "
                        f"up_price={market['up_price']:.3f}"
                    )
                else:
                    logger.warning(f"  {coin}: market not found for window {ts}")
                time.sleep(0.1)
            self._cache = new_cache
            self._cache_ts = ts

    def get_all_prices(self) -> dict[str, dict]:
        """
        Return current prices for all assets.

        Returns dict keyed by asset name e.g. "BTC-UP", "BTC-DOWN":
          {
            "BTC-UP":   {"q": 0.595, "token_id": "...", "coin": "BTC", "side": "UP"},
            "BTC-DOWN": {"q": 0.405, "token_id": "...", "coin": "BTC", "side": "DOWN"},
            ...
          }

        Prices come directly from outcomePrices in the gamma API —
        no separate CLOB midpoint call needed.
        """
        self._refresh_if_needed()
        result = {}
        for coin, market in self._cache.items():
            if market.get("closed") or not market.get("active"):
                continue
            result[f"{coin}-UP"] = {
                "q": market["up_price"],
                "token_id": market["up_token_id"],
                "coin": coin,
                "side": "UP",
            }
            result[f"{coin}-DOWN"] = {
                "q": market["down_price"],
                "token_id": market["down_token_id"],
                "coin": coin,
                "side": "DOWN",
            }
        return result

    def seconds_until_next_window(self) -> int:
        """How many seconds until the current window expires."""
        now = int(time.time())
        window_end = current_window_ts() + WINDOW_SECONDS
        return max(0, window_end - now)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    client = MarketDataClient()
    prices = client.get_all_prices()
    secs = client.seconds_until_next_window()

    print(f"\nCurrent 5-minute window prices ({secs}s remaining):\n")
    print(f"{'Asset':<12} {'q^(w)':>8}  Token ID (first 20 chars)")
    print("-" * 55)
    for asset, info in sorted(prices.items()):
        print(f"{asset:<12} {info['q']:>8.4f}  {info['token_id'][:20]}...")
