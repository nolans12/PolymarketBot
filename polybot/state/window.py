"""
window.py — Current 5-minute window state per asset.

Tracks:
  - open_ts / close_ts (unix seconds)
  - K: Chainlink oracle price at window open (first tick at or after boundary)
  - K_uncertain: True if K was not observed within 5s of boundary
  - slug, up_token_id, down_token_id, condition_id
  - tau_s(): seconds remaining in current window

K-capture algorithm (CLAUDE.md §5.4):
  K = first Chainlink tick where payload.timestamp >= window_open_ms
  If no tick within window_open_ms + 5000ms: mark K_uncertain=True, abstain entire window.

Window rollover:
  Called by the RTDS client when it detects a new window boundary.
  Notifies PolyBook and PolymarketWS to resubscribe for the new token pair.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 300
K_CAPTURE_DEADLINE_MS = 5000  # ms after window open to capture K before marking uncertain


@dataclass
class WindowState:
    asset: str

    open_ts: int = 0           # unix seconds
    close_ts: int = 0          # unix seconds

    K: Optional[float] = None  # Chainlink strike price
    K_uncertain: bool = True   # True until K captured or deadline passed
    K_capture_ts_ns: int = 0   # when K was captured (wall clock)

    slug: str = ""
    up_token_id: str = ""
    down_token_id: str = ""
    condition_id: str = ""
    tick_size: float = 0.01
    active: bool = False
    closed: bool = False

    # Callbacks registered by orchestrator
    _on_rollover: list[Callable] = field(default_factory=list, repr=False)

    def tau_s(self, now_s: Optional[float] = None) -> float:
        """Seconds remaining in the current window."""
        t = now_s if now_s is not None else time.time()
        return max(0.0, self.close_ts - t)

    def is_active(self, now_s: Optional[float] = None) -> bool:
        """True if a window is open and not yet closed."""
        t = now_s if now_s is not None else time.time()
        return self.open_ts > 0 and t < self.close_ts and not self.closed

    def register_rollover_callback(self, cb: Callable) -> None:
        """Register a callback invoked on each window rollover with this WindowState."""
        self._on_rollover.append(cb)

    def apply_market(self, market: dict) -> None:
        """
        Populate token IDs and metadata from a gamma REST fetch_market() result.
        Called at window boundary after REST resolution.
        """
        self.slug = market["slug"]
        self.up_token_id = market["up_token_id"]
        self.down_token_id = market["down_token_id"]
        self.condition_id = market.get("condition_id", "")
        self.tick_size = market.get("tick_size", 0.01)
        self.active = market.get("active", False)
        self.closed = market.get("closed", False)

    def apply_chainlink_tick(self, payload_ts_ms: int, price: float) -> bool:
        """
        Attempt K-capture from a Chainlink tick.

        Returns True if this tick was accepted as K (first tick at or after
        window open boundary), False if it arrived too late or K already set.

        Called by polymarket_rtds.py for every incoming Chainlink tick.
        """
        if self.open_ts == 0:
            return False  # no active window yet

        window_open_ms = self.open_ts * 1000

        # K already captured for this window
        if self.K is not None:
            return False

        # Too early — tick predates the window boundary
        if payload_ts_ms < window_open_ms:
            return False

        # Within window — accept as K
        self.K = price
        self.K_uncertain = False
        self.K_capture_ts_ns = time.time_ns()
        lag_ms = (time.time_ns() // 1_000_000) - payload_ts_ms
        logger.info(
            "window K captured asset=%s K=%.2f window_ts=%d lag_from_oracle_ms=%d",
            self.asset, price, self.open_ts, lag_ms,
        )
        return True

    def check_k_deadline(self) -> None:
        """
        Call periodically after window open. If K hasn't been captured within
        K_CAPTURE_DEADLINE_MS of the boundary, mark K_uncertain=True.
        """
        if self.K is not None or self.open_ts == 0:
            return
        elapsed_ms = (time.time_ns() // 1_000_000) - (self.open_ts * 1000)
        if elapsed_ms > K_CAPTURE_DEADLINE_MS:
            if not self.K_uncertain:
                logger.warning(
                    "window K_uncertain asset=%s window_ts=%d elapsed_ms=%d — abstaining this window",
                    self.asset, self.open_ts, elapsed_ms,
                )
            self.K_uncertain = True

    def rollover(self, new_open_ts: int) -> None:
        """
        Transition to a new 5-minute window. Resets K and token IDs.
        Fires registered rollover callbacks so poly_book and WS client resubscribe.
        """
        prev_ts = self.open_ts
        self.open_ts = new_open_ts
        self.close_ts = new_open_ts + WINDOW_SECONDS
        self.K = None
        self.K_uncertain = True
        self.K_capture_ts_ns = 0
        self.slug = ""
        self.up_token_id = ""
        self.down_token_id = ""
        self.condition_id = ""
        self.active = False
        self.closed = False

        logger.info(
            "window rollover asset=%s prev_ts=%d new_ts=%d",
            self.asset, prev_ts, new_open_ts,
        )

        for cb in self._on_rollover:
            try:
                cb(self)
            except Exception as exc:
                logger.error("window rollover callback failed: %s", exc)
