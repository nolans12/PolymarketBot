"""
edge.py — Edge calculation and Kelly tier sizing.

The single number that drives all entry decisions:
  edge_magnitude = |q_settled - q_actual| - fee - slippage

See CLAUDE.md §3.5 (edge formula) and §3.7 (Kelly tier table).
"""

from dataclasses import dataclass
from typing import Optional

from polybot.infra.config import KELLY_TIERS, SIZE_MIN_USD, SIZE_MAX_USD
from polybot.models.fees import fee_per_dollar
from polybot.models.slippage import slippage_per_dollar
from polybot.state.poly_book import PolyBook


@dataclass
class EdgeResult:
    edge_up_raw:              float
    edge_down_raw:            float
    fee_up_per_dollar:        float
    fee_down_per_dollar:      float
    slippage_up_per_dollar:   float
    slippage_down_per_dollar: float
    edge_up_net:              float
    edge_down_net:            float
    edge_signed:              float   # positive = Up favored, negative = Down
    edge_magnitude:           float   # abs(edge_signed)
    favored_side:             str     # "up" or "down"
    tier:                     int     # 0 = no trade, 1-5 = Kelly tier
    kelly_fraction:           float   # 0.0 if no trade
    bet_size_usd:             float   # 0.0 if no trade; already clamped

    def as_dict(self) -> dict:
        return {
            "edge_up_raw":              self.edge_up_raw,
            "edge_down_raw":            self.edge_down_raw,
            "fee_up_per_dollar":        self.fee_up_per_dollar,
            "fee_down_per_dollar":      self.fee_down_per_dollar,
            "slippage_up_per_dollar":   self.slippage_up_per_dollar,
            "slippage_down_per_dollar": self.slippage_down_per_dollar,
            "edge_up_net":              self.edge_up_net,
            "edge_down_net":            self.edge_down_net,
            "edge_signed":              self.edge_signed,
            "edge_magnitude":           self.edge_magnitude,
            "favored_side":             self.favored_side,
        }


def compute_edge(
    q_settled: float,
    poly: PolyBook,
    wallet_usd: float,
    intended_size_usd: Optional[float] = None,
) -> Optional[EdgeResult]:
    """
    Compute net edge on both sides and determine the favored side + Kelly tier.

    q_settled: regression's estimate of where Polymarket will quote once
               it has digested current spot.
    poly:      current Polymarket book (Up + Down TokenBook).
    wallet_usd: total wallet size (for Kelly sizing).

    Returns None if the book isn't ready.
    """
    if not poly.ready or not poly.up or not poly.down:
        return None

    up = poly.up
    dn = poly.down

    if up.best_ask <= 0 or dn.best_ask <= 0:
        return None

    # --- Raw edge ---
    edge_up_raw   = q_settled - up.best_ask           # positive = Up underpriced
    edge_down_raw = (1.0 - q_settled) - dn.best_ask   # positive = Down underpriced

    # --- Fees ---
    fee_up   = fee_per_dollar(up.best_ask)
    fee_down = fee_per_dollar(dn.best_ask)

    # --- Slippage (size-dependent; use a probe size if not specified) ---
    probe = intended_size_usd if intended_size_usd else min(50.0, wallet_usd * 0.03)
    slip_up   = slippage_per_dollar(up, probe, "buy")
    slip_down = slippage_per_dollar(dn, probe, "buy")

    # --- Net edge ---
    edge_up_net   = edge_up_raw   - fee_up   - slip_up
    edge_down_net = edge_down_raw - fee_down - slip_down

    # --- Favored side and signed/magnitude edge ---
    if edge_up_net >= edge_down_net:
        edge_signed  =  edge_up_net
        favored_side = "up"
    else:
        edge_signed  = -edge_down_net   # negative = Down is favored
        favored_side = "down"

    edge_magnitude = abs(edge_signed)

    # --- Kelly tier lookup ---
    tier, fraction = _kelly_tier(edge_magnitude)

    # --- Bet sizing ---
    if tier > 0 and wallet_usd > 0:
        raw_size = wallet_usd * fraction
        bet_size = max(SIZE_MIN_USD, min(SIZE_MAX_USD, raw_size))
    else:
        bet_size = 0.0

    return EdgeResult(
        edge_up_raw=edge_up_raw,
        edge_down_raw=edge_down_raw,
        fee_up_per_dollar=fee_up,
        fee_down_per_dollar=fee_down,
        slippage_up_per_dollar=slip_up,
        slippage_down_per_dollar=slip_down,
        edge_up_net=edge_up_net,
        edge_down_net=edge_down_net,
        edge_signed=edge_signed,
        edge_magnitude=edge_magnitude,
        favored_side=favored_side,
        tier=tier,
        kelly_fraction=fraction,
        bet_size_usd=bet_size,
    )


def _kelly_tier(magnitude: float) -> tuple[int, float]:
    """
    Look up the Kelly tier from KELLY_TIERS.
    Returns (tier_number, wallet_fraction). tier_number=0 means abstain.
    """
    for i, (threshold, fraction) in enumerate(KELLY_TIERS, start=1):
        if magnitude >= threshold:
            return i, fraction
    return 0, 0.0
