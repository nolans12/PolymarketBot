"""
fills.py — Open order tracker and daily P&L cap enforcement.

Tracks the lifecycle of each real order placed:
  placed → confirmed/failed → exited → settled

Also enforces the daily hard-stop loss limit (CLAUDE.md §12).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from polybot.infra.config import DAILY_LOSS_PCT_HARD

logger = logging.getLogger(__name__)


@dataclass
class OpenOrder:
    order_id:      str
    asset:         str
    side:          str       # "up" or "down"
    token_id:      str
    size_usd:      float
    size_shares:   float
    entry_price:   float
    placed_ns:     int
    window_ts:     int
    confirmed:     bool = False
    filled_shares: float = 0.0
    fill_price:    float = 0.0


class FillTracker:
    """
    Tracks open orders and realized P&L for the session.
    One instance shared across all assets.
    """

    def __init__(self, starting_wallet_usd: float):
        self.starting_wallet  = starting_wallet_usd
        self._open: dict[str, OpenOrder] = {}   # order_id -> OpenOrder
        self._realized_pnl    = 0.0
        self._session_start_s = time.time()

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def record_entry(self, order: OpenOrder) -> None:
        self._open[order.order_id] = order
        logger.info(
            "fill_tracker entry recorded order=%s asset=%s side=%s "
            "shares=%.4f price=%.4f usd=%.2f",
            order.order_id[:12], order.asset, order.side,
            order.size_shares, order.entry_price, order.size_usd,
        )

    def record_exit(
        self,
        order_id: str,
        exit_price: float,
        exit_shares: float,
        exit_reason: str,
    ) -> Optional[float]:
        """
        Record an exit fill. Returns realized P&L in USD, or None if
        the order_id wasn't tracked.
        """
        entry = self._open.pop(order_id, None)
        if entry is None:
            logger.warning("fill_tracker: unknown order_id %s on exit", order_id)
            return None

        theta   = 0.05
        cost    = entry.size_usd
        entry_fee = theta * entry.entry_price * (1 - entry.entry_price) * cost

        gross     = exit_shares * exit_price
        exit_fee  = theta * exit_price * (1 - exit_price) * gross
        pnl       = gross - cost - entry_fee - exit_fee

        self._realized_pnl += pnl

        logger.info(
            "fill_tracker exit reason=%s asset=%s side=%s "
            "exit_price=%.4f pnl=%.4f session_pnl=%.4f",
            exit_reason, entry.asset, entry.side,
            exit_price, pnl, self._realized_pnl,
        )
        return pnl

    def confirm_order(self, order_id: str, filled_shares: float, fill_price: float) -> None:
        if order_id in self._open:
            self._open[order_id].confirmed    = True
            self._open[order_id].filled_shares = filled_shares
            self._open[order_id].fill_price    = fill_price

    def cancel_unconfirmed(self, order_id: str) -> None:
        removed = self._open.pop(order_id, None)
        if removed:
            logger.warning("fill_tracker: order %s cancelled/unconfirmed", order_id[:12])

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def has_open_order(self, asset: str, window_ts: int) -> bool:
        return any(
            o.asset == asset and o.window_ts == window_ts
            for o in self._open.values()
        )

    def open_order_for(self, asset: str, window_ts: int) -> Optional[OpenOrder]:
        for o in self._open.values():
            if o.asset == asset and o.window_ts == window_ts:
                return o
        return None

    def get_token_id(self, asset: str, window_ts: int) -> Optional[str]:
        o = self.open_order_for(asset, window_ts)
        return o.token_id if o else None

    # ------------------------------------------------------------------
    # Daily loss cap  (CLAUDE.md §12)
    # ------------------------------------------------------------------

    def hard_stop_triggered(self) -> bool:
        """
        Returns True if realized session loss exceeds DAILY_LOSS_PCT_HARD
        of the starting wallet. When True, the scheduler must refuse all
        new entries and the operator should be paged.
        """
        if self._realized_pnl >= 0:
            return False
        loss_pct = abs(self._realized_pnl) / self.starting_wallet
        triggered = loss_pct >= DAILY_LOSS_PCT_HARD
        if triggered:
            logger.error(
                "HARD STOP: session P&L=%.2f (%.1f%% of wallet=%.2f) exceeds limit %.1f%%",
                self._realized_pnl, loss_pct * 100,
                self.starting_wallet, DAILY_LOSS_PCT_HARD * 100,
            )
        return triggered

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def n_open(self) -> int:
        return len(self._open)
