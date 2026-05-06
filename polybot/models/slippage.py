"""
slippage.py — Slippage estimator from Polymarket book depth.

In Phase 1 we use a simple linear depth model: estimate the VWAP of filling
`size_usd` worth of contracts against the current top-of-book, and return
the difference vs the best ask/bid as slippage per dollar.

Phase 2 can replace this with a more sophisticated multi-level VWAP model
once we have real fill data to calibrate against.

See CLAUDE.md §3.5 (edge calculation).
"""

from typing import Optional
from polybot.state.poly_book import TokenBook


def slippage_per_dollar(
    book: TokenBook,
    size_usd: float,
    side: str,          # "buy" or "sell"
) -> float:
    """
    Estimate slippage per dollar for a market order of `size_usd` USD
    against the given TokenBook.

    For a buy (taking asks): slippage = VWAP(fill) - best_ask
    For a sell (taking bids): slippage = best_bid - VWAP(fill)

    Returns a non-negative slippage value in probability units.
    """
    if size_usd <= 0:
        return 0.0

    if side == "buy":
        levels = book.top_asks   # ascending price
        best   = book.best_ask
    else:
        levels = book.top_bids   # descending price
        best   = book.best_bid

    if not levels or best <= 0:
        return 0.005  # fallback: 0.5% when depth unknown

    remaining_usd = size_usd
    total_cost    = 0.0
    total_filled  = 0.0

    for lvl in levels:
        price, size_contracts = lvl.price, lvl.size
        if size_contracts <= 0:
            continue
        lvl_usd = price * size_contracts
        take    = min(remaining_usd, lvl_usd)
        filled  = take / price
        total_cost   += take
        total_filled += filled
        remaining_usd -= take
        if remaining_usd <= 0:
            break

    if total_filled <= 0:
        return 0.005

    vwap = total_cost / total_filled  # average price per contract

    slip = abs(vwap - best)
    # Cap at 2% to avoid blowing up on thin books
    return min(slip, 0.02)
