"""
poly_book.py — Live Polymarket CLOB order book state per asset.

Maintained by polymarket_ws.py. Tracks:
  - Best bid/ask for Up and Down tokens
  - Top-5 depth per side
  - Book imbalance (depth_up - depth_down) / total
  - Trade flow accumulator (net Yes-buying volume over 30s)
  - Staleness timestamp for circuit-breaker checks
"""

import time
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


TOP_N = 5
TRADE_FLOW_WINDOW_S = 30


@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class TokenBook:
    """One side (Up or Down) of a Polymarket binary market."""
    token_id: str
    side: str              # 'up' or 'down'

    best_bid: float = 0.0
    best_ask: float = 1.0
    best_bid_size: float = 0.0
    best_ask_size: float = 0.0
    mid: float = 0.5
    spread: float = 1.0

    top_bids: list[BookLevel] = field(default_factory=list)
    top_asks: list[BookLevel] = field(default_factory=list)

    # depth_l1 = bid_size_l1 + ask_size_l1 (proxy for liquidity at touch)
    depth_l1: float = 0.0

    # Raw bids/asks dicts: price -> size (WS price_change gives absolute sizes)
    _bids: dict[float, float] = field(default_factory=dict, repr=False)
    _asks: dict[float, float] = field(default_factory=dict, repr=False)

    def apply_snapshot(self, bids: list[dict], asks: list[dict]) -> None:
        """Apply a full `book` WS event. bids/asks are [{price, size}] dicts."""
        self._bids = {}
        self._asks = {}
        for lvl in bids:
            p, q = float(lvl["price"]), float(lvl["size"])
            if q > 0:
                self._bids[p] = q
        for lvl in asks:
            p, q = float(lvl["price"]), float(lvl["size"])
            if q > 0:
                self._asks[p] = q
        self._recompute()

    def apply_price_change(self, side: str, price: float, size: float) -> None:
        """
        Apply a `price_change` WS event.
        side = 'BUY' (bid) or 'SELL' (ask); size is the new absolute resting qty.
        """
        if side == "BUY":
            if size == 0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = size
        else:  # SELL
            if size == 0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = size
        self._recompute()

    def _recompute(self) -> None:
        sorted_bids = sorted(self._bids.items(), reverse=True)
        sorted_asks = sorted(self._asks.items())

        self.top_bids = [BookLevel(p, q) for p, q in sorted_bids[:TOP_N]]
        self.top_asks = [BookLevel(p, q) for p, q in sorted_asks[:TOP_N]]

        if self.top_bids:
            self.best_bid = self.top_bids[0].price
            self.best_bid_size = self.top_bids[0].size
        if self.top_asks:
            self.best_ask = self.top_asks[0].price
            self.best_ask_size = self.top_asks[0].size

        self.mid = (self.best_bid + self.best_ask) / 2.0
        self.spread = self.best_ask - self.best_bid
        self.depth_l1 = self.best_bid_size + self.best_ask_size


class PolyBook:
    """
    Per-asset Polymarket book containing Up and Down token books.

    Token IDs are set at window rollover via window.py; until set the
    book is in an uninitialized state (ready=False).
    """

    def __init__(self, asset: str):
        self.asset = asset
        self.ready: bool = False

        self.up: Optional[TokenBook] = None
        self.down: Optional[TokenBook] = None

        # token_id -> which side
        self._token_map: dict[str, str] = {}

        # Trade flow accumulator: deque of (ts_s, signed_volume)
        # positive = net Up-buying, negative = net Down-buying
        self._trade_flow: deque[tuple[float, float]] = deque()

        self.last_update_ns: int = 0

    # ------------------------------------------------------------------
    # Window rollover — called by window.py when a new market opens
    # ------------------------------------------------------------------

    def set_tokens(self, up_token_id: str, down_token_id: str) -> None:
        """Reset book for a new 5-minute window."""
        self.up = TokenBook(token_id=up_token_id, side="up")
        self.down = TokenBook(token_id=down_token_id, side="down")
        self._token_map = {
            up_token_id: "up",
            down_token_id: "down",
        }
        self._trade_flow.clear()
        self.ready = False  # awaiting first WS snapshot

    def token_ids(self) -> list[str]:
        if self.up and self.down:
            return [self.up.token_id, self.down.token_id]
        return []

    # ------------------------------------------------------------------
    # WS event handlers
    # ------------------------------------------------------------------

    def handle_book(self, asset_id: str, bids: list[dict], asks: list[dict]) -> bool:
        """Handle a full `book` snapshot event. Returns True if handled."""
        side = self._token_map.get(asset_id)
        if not side:
            return False
        token_book = self.up if side == "up" else self.down
        token_book.apply_snapshot(bids, asks)
        self.last_update_ns = time.time_ns()
        # Ready once the Up side has data (training needs q_up_ask).
        # Down side can lag in low-liquidity windows; don't block on it.
        if self.up:
            up_has_data = bool(self.up._bids or self.up._asks)
            if up_has_data:
                self.ready = True
        return True

    def handle_price_change(self, asset_id: str, changes: list[dict]) -> bool:
        """
        Handle a `price_change` event.
        Each change: {"asset_id": ..., "side": "BUY"|"SELL", "price": ..., "size": ...}
        """
        side = self._token_map.get(asset_id)
        if not side:
            return False
        token_book = self.up if side == "up" else self.down
        for ch in changes:
            token_book.apply_price_change(
                ch["side"], float(ch["price"]), float(ch["size"])
            )
        self.last_update_ns = time.time_ns()
        # Flip ready once Up side accumulates data via incremental updates
        if self.up and (self.up._bids or self.up._asks):
            self.ready = True
        return True

    def handle_last_trade_price(self, asset_id: str, price: float, size: float) -> None:
        """Record trade print for trade-flow feature."""
        side = self._token_map.get(asset_id)
        if not side:
            return
        # positive = Up-buying, negative = Down-buying
        signed_vol = size if side == "up" else -size
        self._trade_flow.append((time.time(), signed_vol))
        self.last_update_ns = time.time_ns()

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def book_imbalance(self) -> float:
        """
        (depth_up - depth_down) / (depth_up + depth_down).
        Range [-1, 1]. Positive = more Up liquidity at touch.
        """
        if not (self.up and self.down):
            return 0.0
        d_up = self.up.depth_l1
        d_dn = self.down.depth_l1
        total = d_up + d_dn
        return (d_up - d_dn) / total if total > 0 else 0.0

    def trade_flow_30s(self) -> float:
        """Net signed volume over the last 30 seconds."""
        cutoff = time.time() - TRADE_FLOW_WINDOW_S
        # Drop expired entries
        while self._trade_flow and self._trade_flow[0][0] < cutoff:
            self._trade_flow.popleft()
        return sum(v for _, v in self._trade_flow)

    def stale_ms(self) -> int:
        if self.last_update_ns == 0:
            return 999_999
        return (time.time_ns() - self.last_update_ns) // 1_000_000
