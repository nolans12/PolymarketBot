"""
position.py — In-memory position tracker per asset per window.

Tracks one open position at a time (one position per window per asset,
as per CLAUDE.md §3.7). Resets at window rollover.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    side:           str     # "up" or "down"
    entry_tau:      float   # tau_s at entry
    entry_price:    float   # q_ask at entry
    entry_edge:     float   # signed edge at entry (for stop-loss reference)
    size_usd:       float   # USD wagered
    contracts:      float   # size_usd / entry_price
    window_ts:      int     # which window this belongs to


class PositionTracker:
    """
    One per asset. Holds at most one open position per window.
    Thread-safe-by-convention (asyncio single thread).
    """

    def __init__(self, asset: str):
        self.asset = asset
        self._position: Optional[Position] = None

    @property
    def has_open(self) -> bool:
        return self._position is not None

    @property
    def position(self) -> Optional[Position]:
        return self._position

    def open(self, pos: Position) -> None:
        self._position = pos

    def close(self) -> None:
        self._position = None

    def reset_for_window(self, window_ts: int) -> None:
        """Called at window rollover. Clears any position from the old window."""
        if self._position and self._position.window_ts != window_ts:
            self._position = None

    def position_row_fields(self) -> dict:
        """Fields to merge into the decision row for position state columns."""
        if self._position is None:
            return {
                "has_open_position":     False,
                "position_side":         None,
                "position_entry_tau":    None,
                "position_entry_price":  None,
                "position_edge_at_entry": None,
            }
        p = self._position
        return {
            "has_open_position":     True,
            "position_side":         p.side,
            "position_entry_tau":    float(p.entry_tau),
            "position_entry_price":  float(p.entry_price),
            "position_edge_at_entry": float(p.entry_edge),
        }
