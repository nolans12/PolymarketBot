"""
coinbase_book.py — Live Coinbase L1 state per asset, fed from the ticker channel.

Unlike SpotBook (which maintains a full L2 order book from diffs), CoinbaseBook
is fed directly by the ticker's best_bid/best_ask/quantity fields. No full book
reconstruction needed — the ticker gives us exactly what we need for microprice.

Computed fields:
  - microprice = (best_bid × ask_size + best_ask × bid_size) / (bid_size + ask_size)
  - mid        = (best_bid + best_ask) / 2
  - 1-Hz microprice ring buffer (300 slots = 5 minutes)
"""

import time
from collections import deque
from typing import Optional

RING_BUFFER_S = 300


class CoinbaseBook:
    """
    Lightweight L1 state updated from Coinbase ticker WS events.
    Same ring-buffer interface as SpotBook so the scheduler can use
    microprice_at(lag_s) identically for both feeds.
    """

    def __init__(self, asset: str):
        self.asset = asset

        self.best_bid: float = 0.0
        self.best_ask: float = float("inf")
        self.best_bid_size: float = 0.0
        self.best_ask_size: float = 0.0

        self.mid: float = 0.0
        self.microprice: float = 0.0
        self.last_trade_price: float = 0.0

        self._ring: deque[tuple[int, float]] = deque(maxlen=RING_BUFFER_S)
        self._last_ring_s: int = 0

        self.last_update_ns: int = 0
        self.ready: bool = False

    def apply_ticker(
        self,
        price: float,
        best_bid: float,
        best_ask: float,
        bid_size: float,
        ask_size: float,
    ) -> None:
        """Update from a Coinbase ticker event."""
        self.last_trade_price = price
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.best_bid_size = bid_size
        self.best_ask_size = ask_size

        self.mid = (best_bid + best_ask) / 2.0

        total = bid_size + ask_size
        if total > 0:
            self.microprice = (best_bid * ask_size + best_ask * bid_size) / total
        else:
            self.microprice = self.mid

        now_ns = time.time_ns()
        self.last_update_ns = now_ns
        self.ready = True

        now_s = int(now_ns // 1_000_000_000)
        if now_s > self._last_ring_s:
            self._ring.append((now_s, self.microprice))
            self._last_ring_s = now_s

    def stale_ms(self) -> int:
        if self.last_update_ns == 0:
            return 999_999
        return (time.time_ns() - self.last_update_ns) // 1_000_000

    def microprice_at(self, lag_s: int) -> Optional[float]:
        """Return microprice from `lag_s` seconds ago via the ring buffer."""
        if not self._ring:
            return None
        target_s = int(time.time()) - lag_s
        for ts, mp in reversed(self._ring):
            if ts <= target_s:
                return mp
        return None
