"""
decision.py — Entry and exit logic for one decision tick.

apply_decision() is called by the scheduler after edge has been computed.
It returns a dict of fields to merge into the decision row, and mutates
the PositionTracker in-place.

Entry rule: CLAUDE.md §3.7
Exit rule:  CLAUDE.md §3.8
"""

from typing import Optional

from polybot.infra.config import (
    LAG_CLOSE_THRESHOLD, STOP_THRESHOLD, FALLBACK_TAU_S, KELLY_TIERS,
)
from polybot.models.edge import EdgeResult
from polybot.strategy.position import Position, PositionTracker


def apply_decision(
    asset: str,
    tau_s: float,
    window_ts: int,
    edge: Optional[EdgeResult],
    tracker: PositionTracker,
    q_up_bid: Optional[float],
    q_down_bid: Optional[float],
    q_settled: Optional[float],
) -> dict:
    """
    Core decision logic. Returns a dict with event/action fields to merge
    into the decision row. Mutates tracker (opens/closes position).

    Caller must ensure edge is not None before calling with model output.
    Returns abstain if edge is None or below floor.
    """
    # Sync position tracker to current window
    tracker.reset_for_window(window_ts)

    pos_fields = tracker.position_row_fields()

    # --- No edge computed (model not ready, book not ready, etc.) ---
    if edge is None:
        return {**pos_fields, "event": "abstain",
                "abstention_reason": "edge_below_floor",
                "tier": 0, "would_bet_usd": 0.0}

    # --- Exit logic (runs before entry — one position max per window) ---
    if tracker.has_open:
        pos = tracker.position
        return _evaluate_exit(pos, tau_s, edge, tracker,
                              q_up_bid, q_down_bid, q_settled, pos_fields)

    # --- Entry logic ---
    if edge.tier == 0:
        return {**pos_fields, "event": "abstain",
                "abstention_reason": "edge_below_floor",
                "tier": 0, "would_bet_usd": 0.0}

    # Open position
    side = edge.favored_side
    entry_price = (q_up_bid if side == "up" else q_down_bid) or (
        1 - edge.edge_down_raw if side == "down" else edge.edge_up_raw
    )
    # Use ask price for entry (we're taker-buying)
    entry_price_ask = (
        (1 - edge.edge_down_raw + edge.slippage_down_per_dollar) if side == "down"
        else (q_settled - edge.edge_up_raw + edge.slippage_up_per_dollar)
    )
    # Simpler: reconstruct from edge_*_raw
    # entry_price = q_settled - edge_up_raw + edge_up_raw = q_settled ... wrong
    # Just use the ask implied by edge: ask = q_settled - edge_up_raw (for up side)
    # or ask = (1-q_settled) - edge_down_raw for down
    # Actually we stored raw in EdgeResult: ask = q_settled - edge_up_raw
    if side == "up":
        ask_price = q_settled - edge.edge_up_raw if q_settled else 0.5
    else:
        ask_price = (1.0 - q_settled) - edge.edge_down_raw if q_settled else 0.5

    ask_price = max(0.01, min(0.99, ask_price))
    contracts = edge.bet_size_usd / ask_price if ask_price > 0 else 0.0

    pos = Position(
        side=side,
        entry_tau=tau_s,
        entry_price=ask_price,
        entry_edge=edge.edge_signed,
        size_usd=edge.bet_size_usd,
        contracts=contracts,
        window_ts=window_ts,
    )
    tracker.open(pos)

    return {
        **tracker.position_row_fields(),
        "event":              "entry",
        "abstention_reason":  None,
        "chosen_side":        side,
        "tier":               edge.tier,
        "would_bet_usd":      float(edge.bet_size_usd),
        "bet_price":          float(ask_price),
        "bet_payout_contracts": float(contracts),
    }


def _evaluate_exit(
    pos: Position,
    tau_s: float,
    edge: EdgeResult,
    tracker: PositionTracker,
    q_up_bid: Optional[float],
    q_down_bid: Optional[float],
    q_settled: Optional[float],
    pos_fields: dict,
) -> dict:
    """Evaluate whether to exit the open position this tick."""

    # Current edge on the open side
    if pos.side == "up":
        edge_now = edge.edge_up_net
        exit_bid = q_up_bid
    else:
        edge_now = edge.edge_down_net
        exit_bid = q_down_bid

    # 1. Resolution fallback
    if tau_s < FALLBACK_TAU_S:
        tracker.close()
        return {
            **tracker.position_row_fields(),
            "event": "fallback_resolution",
            "abstention_reason": None,
            "chosen_side": pos.side,
            "tier": 0, "would_bet_usd": 0.0,
            "bet_price": exit_bid,
        }

    # 2. Lag closed → profit-take
    if edge_now is not None and edge_now < LAG_CLOSE_THRESHOLD:
        tracker.close()
        return {
            **tracker.position_row_fields(),
            "event": "exit_lag_closed",
            "abstention_reason": None,
            "chosen_side": pos.side,
            "tier": 0, "would_bet_usd": 0.0,
            "bet_price": exit_bid,
        }

    # 3. Stop-loss
    if edge_now is not None and edge_now < pos.entry_edge - STOP_THRESHOLD:
        tracker.close()
        return {
            **tracker.position_row_fields(),
            "event": "exit_stopped",
            "abstention_reason": None,
            "chosen_side": pos.side,
            "tier": 0, "would_bet_usd": 0.0,
            "bet_price": exit_bid,
        }

    # 4. Hold
    return {
        **pos_fields,
        "event": "hold",
        "abstention_reason": None,
        "chosen_side": pos.side,
        "tier": 0, "would_bet_usd": 0.0,
    }
