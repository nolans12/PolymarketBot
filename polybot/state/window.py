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
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Read directly here to avoid a circular import on infra.config.
_K_SOURCE = os.getenv("K_SOURCE", "spot").lower()

WINDOW_SECONDS = 300
K_CAPTURE_DEADLINE_MS = 5000  # ms after window open to capture K before marking uncertain
# Cold-start fallback: if we connect mid-window and the most recent Chainlink
# tick predates window open (Chainlink updates every ~30s on price-deviation
# triggers, so this is common), accept it as a best-effort proxy K once we're
# this many ms into the window. K_uncertain is set so live trading abstains;
# training samples can still use the K for log(microprice/K) features.
COLD_START_FALLBACK_MS = 30_000


@dataclass
class WindowState:
    asset: str

    open_ts: int = 0           # unix seconds
    close_ts: int = 0          # unix seconds

    K: Optional[float] = None  # Chainlink strike price
    K_uncertain: bool = True   # True until K captured or deadline passed
    K_capture_ts_ns: int = 0   # when K was captured (wall clock)

    # Most-recent Chainlink price observed (survives rollovers).
    # Used by check_k_deadline() to apply a fallback K when no fresh tick
    # has arrived within COLD_START_FALLBACK_MS of window open.
    latest_chainlink_price: Optional[float] = None
    latest_chainlink_ts_ms: int = 0

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

    def set_k_from_spot(self, price: float) -> bool:
        """
        Capture K directly from a spot feed (Binance microprice) at the window
        boundary. Used in lieu of Chainlink RTDS, which has high latency and
        coverage issues for cold starts.

        K is treated as canonical (K_uncertain=False) since the spot feed is
        live and continuous. The trade-off: K is no longer the exact value
        Polymarket will resolve against, but for our regression's K-relative
        features that's fine — features are log(microprice/K), and a small
        constant offset in K shifts every row equally.
        """
        if self.open_ts == 0 or price is None or price <= 0:
            return False
        if self.K is not None:
            return False
        self.K = float(price)
        self.K_uncertain = False
        self.K_capture_ts_ns = time.time_ns()
        logger.info(
            "window K from spot asset=%s K=%.2f window_ts=%d",
            self.asset, self.K, self.open_ts,
        )
        self._log_active_window_banner(uncertain=False)
        return True

    def apply_chainlink_tick(self, payload_ts_ms: int, price: float) -> bool:
        """
        Attempt K-capture from a Chainlink tick. Disabled when K_SOURCE != 'chainlink'.

        Returns True if this tick was accepted as K (first tick at or after
        window open boundary), False if it arrived too late or K already set.
        """
        # When K is sourced from spot (default), Chainlink ticks don't drive K.
        if _K_SOURCE != "chainlink":
            return False

        if self.open_ts == 0:
            return False  # no active window yet

        window_open_ms = self.open_ts * 1000
        now_ms = time.time_ns() // 1_000_000

        # Cache every tick we see — survives rollovers, used by deadline fallback.
        if payload_ts_ms >= self.latest_chainlink_ts_ms:
            self.latest_chainlink_price = price
            self.latest_chainlink_ts_ms = payload_ts_ms

        # K already captured for this window
        if self.K is not None:
            return False

        # Tick is at or after window open — strict (canonical) capture
        if payload_ts_ms >= window_open_ms:
            self.K = price
            self.K_uncertain = False
            self.K_capture_ts_ns = time.time_ns()
            lag_ms = now_ms - payload_ts_ms
            logger.info(
                "window K captured asset=%s K=%.2f window_ts=%d lag_from_oracle_ms=%d",
                self.asset, price, self.open_ts, lag_ms,
            )
            self._log_active_window_banner(uncertain=False)
            return True

        # Tick predates window open. Cold-start fallback: if we're already
        # more than COLD_START_FALLBACK_MS into the window with no K, accept
        # this tick as a best-effort proxy and flag K_uncertain.
        elapsed_in_window_ms = now_ms - window_open_ms
        if elapsed_in_window_ms > COLD_START_FALLBACK_MS:
            self.K = price
            self.K_uncertain = True
            self.K_capture_ts_ns = time.time_ns()
            logger.warning(
                "window K cold-start fallback asset=%s K=%.2f window_ts=%d "
                "elapsed_in_window_ms=%d oracle_age_ms=%d (K_uncertain=True)",
                self.asset, price, self.open_ts, elapsed_in_window_ms,
                window_open_ms - payload_ts_ms,
            )
            self._log_active_window_banner(uncertain=True)
            return True

        return False

    def _log_active_window_banner(self, uncertain: bool) -> None:
        """One-line banner so the operator can confirm the live K + market."""
        flag = " [K_UNCERTAIN]" if uncertain else ""
        logger.info(
            "RUNNING %s UP OR DOWN 5 MINUTE WINDOW WITH PRICE TO BEAT: %.2f"
            " (window_ts=%d slug=%s)%s",
            self.asset.upper(), self.K, self.open_ts,
            self.slug or "?", flag,
        )

    def check_k_deadline(self) -> None:
        """
        Call periodically after window open. Two checks:
          1. K_CAPTURE_DEADLINE_MS — flag K_uncertain if no fresh tick yet.
          2. COLD_START_FALLBACK_MS — apply cached pre-window Chainlink price
             as fallback K so feature engineering can run during the rest of
             this window.

        Disabled when K_SOURCE != 'chainlink' (spot path owns K capture).
        """
        if _K_SOURCE != "chainlink":
            return
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

        # Fallback: apply most recent cached Chainlink price as proxy K.
        if (elapsed_ms > COLD_START_FALLBACK_MS
                and self.latest_chainlink_price is not None):
            self.K = self.latest_chainlink_price
            self.K_uncertain = True   # decisions still abstain; samples can run
            self.K_capture_ts_ns = time.time_ns()
            oracle_age_ms = (self.open_ts * 1000) - self.latest_chainlink_ts_ms
            logger.warning(
                "window K deadline fallback asset=%s K=%.2f window_ts=%d "
                "elapsed_in_window_ms=%d oracle_age_ms=%d (K_uncertain=True)",
                self.asset, self.K, self.open_ts, elapsed_ms, oracle_age_ms,
            )
            self._log_active_window_banner(uncertain=True)

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
