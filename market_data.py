"""
market_data.py — Polymarket CLOB market discovery and price polling.

Responsibilities:
- Find the correct token IDs for BTC/ETH/SOL/XRP 5-minute up/down markets
- Poll current mid-market price (q^w) for each token
- Cache token IDs so we don't re-fetch the market list every poll cycle
"""

import requests
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"

# Known 5-minute crypto market keywords to match against market descriptions.
# Polymarket market slugs/questions change over time — we search by keyword.
ASSET_KEYWORDS = {
    "BTC-UP":   ["btc", "bitcoin", "up", "higher", "above"],
    "BTC-DOWN": ["btc", "bitcoin", "down", "lower", "below"],
    "ETH-UP":   ["eth", "ethereum", "up", "higher", "above"],
    "ETH-DOWN": ["eth", "ethereum", "down", "lower", "below"],
    "SOL-UP":   ["sol", "solana", "up", "higher", "above"],
    "SOL-DOWN": ["sol", "solana", "down", "lower", "below"],
    "XRP-UP":   ["xrp", "ripple", "up", "higher", "above"],
    "XRP-DOWN": ["xrp", "ripple", "down", "lower", "below"],
}


class MarketDataClient:
    def __init__(self, clob_host: str = CLOB_HOST):
        self.clob_host = clob_host
        self._token_cache: dict[str, str] = {}  # asset_name -> token_id

    def list_markets(self, next_cursor: str = "") -> dict:
        url = f"{self.clob_host}/markets"
        params = {"next_cursor": next_cursor} if next_cursor else {}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def find_5min_markets(self) -> dict[str, str]:
        """
        Search all active Polymarket markets for 5-minute crypto up/down markets.
        Returns a dict of asset_name -> token_id for YES outcome tokens.
        Run this once at startup (or with --list flag) to populate the token cache.
        """
        found: dict[str, str] = {}
        next_cursor = ""

        while True:
            data = self.list_markets(next_cursor)
            markets = data.get("data", [])

            for market in markets:
                question = (market.get("question") or "").lower()
                description = (market.get("description") or "").lower()
                text = question + " " + description

                # Only look at 5-minute markets
                if "5" not in text and "five" not in text and "5-min" not in text:
                    continue
                if "minute" not in text and "min" not in text:
                    continue

                tokens = market.get("tokens", [])
                for asset_name, keywords in ASSET_KEYWORDS.items():
                    if asset_name in found:
                        continue
                    coin_kw = keywords[:2]   # e.g. ["btc", "bitcoin"]
                    direction_kw = keywords[2:]  # e.g. ["up", "higher", "above"]
                    if (any(k in text for k in coin_kw) and
                            any(k in text for k in direction_kw)):
                        # Take the YES token
                        for token in tokens:
                            outcome = (token.get("outcome") or "").lower()
                            if outcome in ("yes", "up", "higher", "above"):
                                found[asset_name] = token["token_id"]
                                logger.info(f"Found {asset_name}: token_id={token['token_id']} | {question[:80]}")
                                break

            next_cursor = data.get("next_cursor", "")
            if not next_cursor or not markets:
                break

        return found

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """
        Fetch the current best-bid / best-ask midpoint for a token.
        This is q^(w) — the market-implied probability.
        Returns None if the orderbook is empty or unreachable.
        """
        url = f"{self.clob_host}/midpoint"
        try:
            resp = requests.get(url, params={"token_id": token_id}, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            mid = data.get("mid")
            if mid is None:
                return None
            return float(mid)
        except Exception as e:
            logger.warning(f"Failed to fetch midpoint for {token_id}: {e}")
            return None

    def get_all_midpoints(self, token_ids: dict[str, str]) -> dict[str, Optional[float]]:
        """
        Fetch midpoints for all tracked assets.
        token_ids: dict of asset_name -> token_id
        Returns: dict of asset_name -> q^(w) (or None if unavailable)
        """
        result = {}
        for asset_name, token_id in token_ids.items():
            result[asset_name] = self.get_midpoint(token_id)
            time.sleep(0.1)  # small delay to avoid hammering the API
        return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    client = MarketDataClient()

    if "--list" in sys.argv:
        print("Searching for 5-minute crypto markets on Polymarket...")
        markets = client.find_5min_markets()
        print(f"\nFound {len(markets)} markets:")
        for name, token_id in markets.items():
            print(f"  {name}: {token_id}")
        if not markets:
            print("No markets found. The market slugs may have changed.")
            print("Try: GET https://clob.polymarket.com/markets and search manually.")
    else:
        print("Usage: python3 market_data.py --list")
        print("Lists all discovered 5-minute crypto markets and their token IDs.")
