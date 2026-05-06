"""
book.py — Live market state for Coinbase (spot) and Kalshi (betting market).

CoinbaseBook: L1 microprice from ticker channel + 5-min ring buffer of 1Hz samples.
KalshiBook:   yes/no order book reconstructed from WS snapshots/deltas + ring buffer.
"""

import math
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional


_RING = 300   # 5-minute ring buffer at 1 sample/s


# ---------------------------------------------------------------------------
# Coinbase
# ---------------------------------------------------------------------------

class CoinbaseBook:
    """Coinbase L1 state from the ticker channel. Computes microprice."""

    def __init__(self):
        self.best_bid:      float = 0.0
        self.best_ask:      float = float("inf")
        self.best_bid_size: float = 0.0
        self.best_ask_size: float = 0.0
        self.mid:           float = 0.0
        self.microprice:    float = 0.0
        self.last_trade:    float = 0.0

        self._ring: deque[tuple[int, float]] = deque(maxlen=_RING)
        self._last_ring_s: int = 0
        self.last_update_ns: int = 0
        self.ready: bool = False

    def apply_ticker(self, price: float, bid: float, ask: float,
                     bid_size: float, ask_size: float) -> None:
        self.last_trade    = price
        self.best_bid      = bid
        self.best_ask      = ask
        self.best_bid_size = bid_size
        self.best_ask_size = ask_size

        self.mid = (bid + ask) / 2.0
        total = bid_size + ask_size
        self.microprice = (bid * ask_size + ask * bid_size) / total if total > 0 else self.mid

        now_ns = time.time_ns()
        self.last_update_ns = now_ns
        self.ready = True

        now_s = int(now_ns // 1_000_000_000)
        if now_s > self._last_ring_s:
            self._ring.append((now_s, self.microprice))
            self._last_ring_s = now_s

    def microprice_at(self, lag_s: int) -> Optional[float]:
        """Microprice from lag_s seconds ago. Falls back to oldest sample during warmup."""
        if not self._ring:
            return None
        target_s = int(time.time()) - lag_s
        for ts, mp in reversed(self._ring):
            if ts <= target_s:
                return mp
        return self._ring[0][1]

    def stale_ms(self) -> int:
        if self.last_update_ns == 0:
            return 999_999
        return int((time.time_ns() - self.last_update_ns) // 1_000_000)


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

class KalshiBook:
    """
    Kalshi 15-min BTC yes/no order book.

    The yes_book maps dollar prices (0.0–1.0) to resting dollar size.
    The no_book is the same for NO side.
    Top-of-book:
        yes_bid = max(yes_book)                  # best YES buyer
        yes_ask = 1.0 - max(no_book)             # best YES seller = 1 - best NO buyer
    """

    def __init__(self):
        self.ticker:       str = ""
        self.floor_strike: float = 0.0           # K — BTC price at window open
        self.close_time:   Optional[datetime] = None

        self.yes_bid: float = 0.0
        self.yes_ask: float = 1.0
        self.yes_mid: float = 0.5

        self._yes_book: dict[float, float] = {}
        self._no_book:  dict[float, float] = {}

        # 5-min ring buffer of yes_mid at 1 Hz (for momentum features)
        self._ring: deque[tuple[int, float]] = deque(maxlen=_RING)
        self._last_ring_s: int = 0

        self.last_update_ns: int = 0
        self.ready: bool = False

    # -- Window management --------------------------------------------------

    def set_window(self, ticker: str, floor_strike: float,
                   close_time: datetime) -> None:
        """Called when a new 15-min window opens."""
        self.ticker       = ticker
        self.floor_strike = floor_strike
        self.close_time   = close_time
        self._yes_book.clear()
        self._no_book.clear()
        self.ready = False
        self.last_update_ns = 0

    # -- Book updates -------------------------------------------------------

    def apply_snapshot(self, yes_fp: list, no_fp: list) -> None:
        self._yes_book = {float(p): float(s) for p, s in yes_fp}
        self._no_book  = {float(p): float(s) for p, s in no_fp}
        self._recompute()

    def apply_delta(self, side: str, price: float, delta: float) -> None:
        book = self._yes_book if side == "yes" else self._no_book
        book[price] = book.get(price, 0.0) + delta
        if book[price] <= 0:
            book.pop(price, None)
        self._recompute()

    def _recompute(self) -> None:
        # Always stamp the update time — even if book is empty, we received data
        now_ns = time.time_ns()
        self.last_update_ns = now_ns

        if not self._yes_book or not self._no_book:
            return
        yb = max(self._yes_book)
        ya = 1.0 - max(self._no_book)
        if ya < yb:
            return   # genuinely crossed book — skip (equal spread is valid)
        self.yes_bid = yb
        self.yes_ask = max(ya, yb)  # never let ask go below bid
        self.yes_mid = (self.yes_bid + self.yes_ask) / 2.0
        self.ready = True

        now_s = int(now_ns // 1_000_000_000)
        if now_s > self._last_ring_s:
            self._ring.append((now_s, self.yes_mid))
            self._last_ring_s = now_s

    def apply_last_price(self, last_price: float) -> None:
        """Update mid from a last_price tick when bid/ask aren't in the message."""
        if last_price <= 0 or last_price >= 1:
            return
        self.yes_mid = last_price
        now_ns = time.time_ns()
        self.last_update_ns = now_ns
        self.ready = True
        now_s = int(now_ns // 1_000_000_000)
        if now_s > self._last_ring_s:
            self._ring.append((now_s, self.yes_mid))
            self._last_ring_s = now_s

    def apply_ticker_update(self, yes_bid: float, yes_ask: float) -> None:
        """
        Fast path: update from the ticker channel when the orderbook is quiet.
        Ticker fires on every trade and quote change — more frequent than deltas.
        """
        if yes_bid <= 0 or yes_ask <= 0 or yes_ask <= yes_bid:
            return
        self.yes_bid = yes_bid
        self.yes_ask = yes_ask
        self.yes_mid = (yes_bid + yes_ask) / 2.0
        now_ns = time.time_ns()
        self.last_update_ns = now_ns
        self.ready = True
        now_s = int(now_ns // 1_000_000_000)
        if now_s > self._last_ring_s:
            self._ring.append((now_s, self.yes_mid))
            self._last_ring_s = now_s

    # -- Queries ------------------------------------------------------------

    def tau_s(self) -> float:
        """Seconds until window close."""
        if self.close_time is None:
            return float(900)
        return max(0.0, (self.close_time - datetime.now(timezone.utc)).total_seconds())

    def yes_mid_at(self, lag_s: int) -> Optional[float]:
        """yes_mid from lag_s seconds ago (for Kalshi momentum features)."""
        if not self._ring:
            return None
        target_s = int(time.time()) - lag_s
        for ts, mid in reversed(self._ring):
            if ts <= target_s:
                return mid
        return self._ring[0][1]

    @property
    def yes_depth(self) -> float:
        """Total resting dollar size on the YES bid side."""
        return sum(self._yes_book.values())

    @property
    def no_depth(self) -> float:
        """Total resting dollar size on the NO bid side."""
        return sum(self._no_book.values())

    def stale_ms(self) -> int:
        if self.last_update_ns == 0:
            return 999_999
        return int((time.time_ns() - self.last_update_ns) // 1_000_000)
