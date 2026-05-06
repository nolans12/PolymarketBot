"""
fees.py — Polymarket taker fee model.

Fee curve: fee_per_dollar(p) = THETA * p * (1 - p)
  THETA ≈ 0.05 (borrowed from 15-min docs; confirmed empirically in Phase 2)
  Peaks at ~3.15% when p = 0.5, drops toward 0 near p = 0 or p = 1.

See CLAUDE.md §5.5.
"""

from polybot.infra.config import THETA_FEE


def fee_per_dollar(p: float) -> float:
    """Taker fee per dollar wagered at probability p."""
    p = max(0.0, min(1.0, p))
    return THETA_FEE * p * (1.0 - p)


def round_trip_fee(entry_p: float, exit_p: float) -> float:
    """Combined entry + exit taker fee per dollar of the entry stake."""
    return fee_per_dollar(entry_p) + fee_per_dollar(exit_p)
