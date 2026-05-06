"""
polymarket_rest.py — Slug → token ID resolution for 5-minute crypto markets.

The CLOB book WS subscribes by token IDs (not slug), so each window boundary
the bot resolves the new market via the gamma REST API. This module is
deliberately minimal — the WS clients carry the live data; REST is only
used at window rollovers and as a sanity fallback.

Slug pattern (CLAUDE.md §2):
    {asset}-updown-5m-{window_open_unix_ts}    where ts % 300 == 0
"""

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 300


def current_window_ts(now_s: Optional[int] = None) -> int:
    """Return the unix timestamp of the current 5-min window's open boundary."""
    t = int(now_s if now_s is not None else time.time())
    return t - (t % WINDOW_SECONDS)


def next_window_ts(now_s: Optional[int] = None) -> int:
    return current_window_ts(now_s) + WINDOW_SECONDS


def seconds_until_next_window(now_s: Optional[int] = None) -> int:
    t = int(now_s if now_s is not None else time.time())
    return max(0, current_window_ts(t) + WINDOW_SECONDS - t)


def slug_for(asset: str, window_ts: int) -> str:
    """e.g. ('btc', 1777968000) -> 'btc-updown-5m-1777968000'."""
    return f"{asset.lower()}-updown-5m-{window_ts}"


def fetch_market(
    asset: str,
    window_ts: int,
    gamma_api: str = "https://gamma-api.polymarket.com",
    timeout: float = 10.0,
) -> Optional[dict]:
    """
    Fetch a single 5-min market event from gamma. Returns:
        {
          "asset":         "btc",
          "window_ts":     1777968000,
          "slug":          "btc-updown-5m-1777968000",
          "up_token_id":   "...",
          "down_token_id": "...",
          "condition_id":  "...",
          "tick_size":     0.01,
          "active":        True,
          "closed":        False,
          "end_date":      "...",
        }
    or None if the event is missing/malformed.
    """
    slug = slug_for(asset, window_ts)
    try:
        resp = requests.get(
            f"{gamma_api}/events",
            params={"slug": slug},
            timeout=timeout,
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
        token_ids      = json.loads(m.get("clobTokenIds", "[]"))
        outcomes       = json.loads(m.get("outcomes", "[]"))
        if len(token_ids) < 2 or len(outcomes) < 2:
            return None

        up_idx   = 0 if outcomes[0].lower() == "up" else 1
        down_idx = 1 - up_idx

        return {
            "asset":         asset.lower(),
            "window_ts":     window_ts,
            "slug":          slug,
            "up_token_id":   token_ids[up_idx],
            "down_token_id": token_ids[down_idx],
            "condition_id":  m.get("conditionId"),
            "tick_size":     float(m.get("orderPriceMinTickSize", "0.01") or 0.01),
            "active":        bool(m.get("active", False)),
            "closed":        bool(m.get("closed", False)),
            "end_date":      m.get("endDate"),
        }
    except Exception as e:
        logger.warning(f"gamma fetch failed for {slug}: {e}")
        return None
