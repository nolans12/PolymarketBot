"""
spot_book.py — Live Binance L2 order book state per asset.

Maintained by binance_ws.py. Computed fields:
  - mid         = (best_bid + best_ask) / 2
  - microprice  = (best_bid × ask_size + best_ask × bid_size) / (bid_size + ask_size)
  - top 5 bid/ask levels
  - 1-Hz microprice ring buffer (300 slots = 5 minutes)
  - OFI accumulator (Cont-Kukanov-Stoikov, levels 1-5)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


TOP_N = 5
RING_BUFFER_S = 300  # 5 minutes at 1 sample/s


@dataclass
class Level:
    price: float
    size: float


@dataclass
class OFISnapshot:
    ts_ns: int
    ofi_l1: float       # signed OFI at level 1
    ofi_l5: float       # weighted OFI levels 1-5


class SpotBook:
    """
    Thread-safe-by-convention: only the binance_ws coroutine writes;
    the scheduler tick reads. asyncio single-threaded guarantees no races.
    """

    def __init__(self, asset: str):
        self.asset = asset

        # Raw book: sorted dicts price -> size
        # bids descending, asks ascending maintained manually
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}

        # Computed L1
        self.best_bid: float = 0.0
        self.best_ask: float = float("inf")
        self.best_bid_size: float = 0.0
        self.best_ask_size: float = 0.0

        # Derived
        self.mid: float = 0.0
        self.microprice: float = 0.0
        self.spread: float = 0.0

        # Top-N snapshots
        self.top_bids: list[Level] = []
        self.top_asks: list[Level] = []

        # 1-Hz microprice ring buffer (indexed by unix second)
        self._ring: deque[tuple[int, float]] = deque(maxlen=RING_BUFFER_S)
        self._last_ring_s: int = 0

        # OFI accumulator — resets after each read
        self._ofi_l1_acc: float = 0.0
        self._ofi_l5_acc: float = 0.0

        # Previous top-5 sizes for OFI delta
        self._prev_bid_sizes: list[float] = [0.0] * TOP_N
        self._prev_ask_sizes: list[float] = [0.0] * TOP_N

        # Staleness
        self.last_update_ns: int = 0
        self.last_update_id: int = 0

        # Snapshot ready flag (False until first full snapshot applied)
        self.ready: bool = False

    # ------------------------------------------------------------------
    # Book management
    # ------------------------------------------------------------------

    def apply_snapshot(self, bids: list, asks: list, last_update_id: int) -> None:
        """Replace entire local book from REST snapshot."""
        self._bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self._asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = last_update_id
        self._recompute()
        self.ready = True

    def apply_diff(self, bids: list, asks: list, first_update_id: int, final_update_id: int) -> bool:
        """
        Apply a WS diff event. Returns False if a sequence gap was detected
        (caller should reconnect and re-snapshot).
        """
        if not self.ready:
            return True  # still buffering pre-snapshot diffs; caller handles

        # Binance sequence rule: first valid event satisfies
        #   first_update_id <= last_update_id + 1 <= final_update_id
        if first_update_id > self.last_update_id + 1:
            return False  # gap

        if final_update_id <= self.last_update_id:
            return True  # stale, already applied — skip silently

        for p, q in bids:
            price, qty = float(p), float(q)
            if qty == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = qty

        for p, q in asks:
            price, qty = float(p), float(q)
            if qty == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

        self.last_update_id = final_update_id
        self._recompute()
        return True

    def _recompute(self) -> None:
        """Recompute all derived fields after a book change."""
        now_ns = time.time_ns()
        self.last_update_ns = now_ns

        # Top-N levels
        sorted_bids = sorted(self._bids.items(), reverse=True)
        sorted_asks = sorted(self._asks.items())

        self.top_bids = [Level(p, q) for p, q in sorted_bids[:TOP_N]]
        self.top_asks = [Level(p, q) for p, q in sorted_asks[:TOP_N]]

        if not self.top_bids or not self.top_asks:
            return

        self.best_bid = self.top_bids[0].price
        self.best_ask = self.top_asks[0].price
        self.best_bid_size = self.top_bids[0].size
        self.best_ask_size = self.top_asks[0].size

        self.spread = self.best_ask - self.best_bid
        self.mid = (self.best_bid + self.best_ask) / 2.0

        # Microprice: imbalance-weighted fair value
        total = self.best_bid_size + self.best_ask_size
        if total > 0:
            self.microprice = (
                self.best_bid * self.best_ask_size
                + self.best_ask * self.best_bid_size
            ) / total
        else:
            self.microprice = self.mid

        # OFI accumulation (Cont-Kukanov-Stoikov)
        cur_bid_sizes = [lvl.size for lvl in self.top_bids] + [0.0] * (TOP_N - len(self.top_bids))
        cur_ask_sizes = [lvl.size for lvl in self.top_asks] + [0.0] * (TOP_N - len(self.top_asks))

        weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # linear decay for levels 1-5

        ofi_l1 = cur_bid_sizes[0] - self._prev_bid_sizes[0] - (cur_ask_sizes[0] - self._prev_ask_sizes[0])
        ofi_l5 = sum(
            w * ((cur_bid_sizes[i] - self._prev_bid_sizes[i]) - (cur_ask_sizes[i] - self._prev_ask_sizes[i]))
            for i, w in enumerate(weights)
        )

        self._ofi_l1_acc += ofi_l1
        self._ofi_l5_acc += ofi_l5

        self._prev_bid_sizes = cur_bid_sizes
        self._prev_ask_sizes = cur_ask_sizes

        # 1-Hz ring buffer update
        now_s = int(now_ns // 1_000_000_000)
        if now_s > self._last_ring_s:
            self._ring.append((now_s, self.microprice))
            self._last_ring_s = now_s

    # ------------------------------------------------------------------
    # Read interface for scheduler / features
    # ------------------------------------------------------------------

    def stale_ms(self) -> int:
        """Milliseconds since last book update."""
        if self.last_update_ns == 0:
            return 999_999
        return (time.time_ns() - self.last_update_ns) // 1_000_000

    def microprice_at(self, lag_s: int) -> Optional[float]:
        """
        Return microprice from `lag_s` seconds ago using the ring buffer.

        If the ring doesn't yet have a sample that old (warmup period), falls
        back to the oldest available sample so the regression can still warm
        up. Once the buffer fills, this returns the correct lagged value.
        """
        if not self._ring:
            return None
        target_s = int(time.time()) - lag_s
        # Linear scan from newest; ring is small (300 entries max)
        for ts, mp in reversed(self._ring):
            if ts <= target_s:
                return mp
        # Warmup fallback: return oldest available sample
        return self._ring[0][1]

    def drain_ofi(self) -> tuple[float, float]:
        """Return accumulated OFI since last drain and reset accumulators."""
        l1 = self._ofi_l1_acc
        l5 = self._ofi_l5_acc
        self._ofi_l1_acc = 0.0
        self._ofi_l5_acc = 0.0
        return l1, l5
