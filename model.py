"""
model.py — Dual-Mode HFT entry filter and dynamic Kelly position sizing.

Implements the GABIGOL dual-mode model from the paper:

  Mode 1 (High-Confidence Scalping): q^(w) ∈ [0.60, 0.97]
  Mode 2 (Extreme Discount):         q^(w) ∈ [0.087, 0.60)

Entry condition (both modes):
  p_jj >= P_JJ_MIN  AND  Δ^(w) = p_mine - q >= EPSILON  →  ENTER

Expected return formula (Table 8.1.2):
  r = (1 - q^(w)) / q^(w)

Dynamic Kelly sizing:
  Position size = kelly_fraction(delta) * wallet_balance_usd
  kelly_fraction scales with edge magnitude (see config.KELLY_TIERS)
  Clamped to [SIZE_MIN, SIZE_MAX]
"""

import logging
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger(__name__)

MODE1_LOW  = config.MODE1_LOW
MODE1_HIGH = config.MODE1_HIGH
MODE2_LOW  = config.MODE2_LOW
MODE2_HIGH = config.MODE2_HIGH


@dataclass
class TradeSignal:
    asset: str
    mode: int
    q: float           # market price (entry price)
    p_mine: float      # Markov-estimated true probability
    p_jj: float        # Markov self-transition probability
    delta: float       # p_mine - q  (the edge in probability space)
    expected_return: float  # r = (1 - q) / q
    size_usd: float    # position size in USD
    wallet_pct: float  # fraction of wallet being risked
    side: str          # always "BUY" (we buy YES tokens)


def classify_mode(q: float) -> Optional[int]:
    """Return mode 1 or 2 for a given market price, or None if out of range."""
    if MODE1_LOW <= q <= MODE1_HIGH:
        return 1
    if MODE2_LOW <= q < MODE2_HIGH:
        return 2
    return None


def expected_return(q: float) -> float:
    """r = (1 - q) / q — paper's Table 8.1.2 return formula."""
    if q <= 0:
        return 0.0
    return (1.0 - q) / q


def kelly_fraction(delta: float) -> float:
    """
    Return the wallet fraction to risk for a given edge (delta).
    Uses tiered Kelly from config.KELLY_TIERS.
    Returns 0.0 if delta is below all tiers (should not trade).
    """
    for min_delta, fraction in config.KELLY_TIERS:
        if delta >= min_delta:
            return fraction
    return 0.0


def kelly_size(delta: float, wallet_balance_usd: float) -> float:
    """
    Compute position size in USD using dynamic Kelly fraction.
    Clamped to [SIZE_MIN, SIZE_MAX].
    """
    fraction = kelly_fraction(delta)
    if fraction == 0.0:
        return 0.0
    raw = fraction * wallet_balance_usd
    return max(config.SIZE_MIN, min(config.SIZE_MAX, raw))


def evaluate(
    asset: str,
    q: float,
    p_jj: float,
    p_mine: float,
    epsilon: float,
    p_jj_min: float,
    wallet_balance_usd: float,
) -> Optional[TradeSignal]:
    """
    Run the dual-mode entry filter for one asset at one point in time.

    Returns a TradeSignal if entry conditions are met, None otherwise.
    wallet_balance_usd: current USDC balance in the Polymarket account.
    """
    mode = classify_mode(q)
    if mode is None:
        logger.debug(f"{asset} | q={q:.3f} | out of range — skip")
        return None

    delta = p_mine - q

    if p_jj < p_jj_min:
        logger.debug(f"{asset} | mode={mode} | q={q:.3f} | p_jj={p_jj:.3f} < {p_jj_min} — skip")
        return None

    if delta < epsilon:
        logger.debug(f"{asset} | mode={mode} | q={q:.3f} | Δ={delta:.3f} < {epsilon} — skip")
        return None

    size = kelly_size(delta, wallet_balance_usd)
    if size <= 0:
        return None

    fraction = kelly_fraction(delta)
    r = expected_return(q)

    signal = TradeSignal(
        asset=asset,
        mode=mode,
        q=q,
        p_mine=p_mine,
        p_jj=p_jj,
        delta=delta,
        expected_return=r,
        size_usd=size,
        wallet_pct=fraction,
        side="BUY",
    )

    logger.info(
        f"SIGNAL | {asset} | mode={mode} | q={q:.3f} | p_mine={p_mine:.3f} | "
        f"Δ={delta:.3f} | r={r:.1%} | p_jj={p_jj:.3f} | "
        f"size=${size:.2f} ({fraction:.1%} of wallet)"
    )

    return signal
