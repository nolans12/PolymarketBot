"""
markov.py — 30-minute rolling Markov state estimator.

Given a stream of q^(w) observations (market-implied probabilities),
builds a transition matrix over discretized states and returns:
  - p_jj:  self-transition probability of the current state (persistence signal)
  - p_mine: Markov-estimated true probability for the current state

State discretization (from the paper):
  States are defined by q^(w) ranges. We use fine-grained bins so the
  matrix captures the full [0.087, 0.97] operating range.
"""

import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

# State bin edges — covers the full operating range of the model.
# States outside [0.05, 0.98] are considered inactive (no trade).
BIN_EDGES = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 1.00
])
N_STATES = len(BIN_EDGES) - 1  # 20 states


def digitize(q: float) -> int:
    """Map a probability q to a state index [0, N_STATES-1]."""
    idx = int(np.searchsorted(BIN_EDGES, q, side="right")) - 1
    return max(0, min(idx, N_STATES - 1))


def state_midpoint(state_idx: int) -> float:
    """Return the midpoint probability of a state bin."""
    return float((BIN_EDGES[state_idx] + BIN_EDGES[state_idx + 1]) / 2)


class MarkovEstimator:
    """
    Maintains a rolling 30-minute window of q^(w) observations and
    computes the Markov transition matrix on demand.

    Usage:
        estimator = MarkovEstimator(window_seconds=1800, poll_interval=30)
        estimator.update(q)          # call each poll cycle
        p_jj, p_mine = estimator.estimate(q)
    """

    def __init__(self, window_seconds: int = 1800, poll_interval: int = 30):
        # Maximum observations in the rolling window
        max_obs = window_seconds // poll_interval + 1
        self._history: deque[float] = deque(maxlen=max_obs)
        self._min_obs = 10  # need at least this many observations to estimate

    def update(self, q: float) -> None:
        """Add a new market price observation."""
        self._history.append(q)

    def estimate(self, q_current: float) -> tuple[float, float]:
        """
        Compute p_jj and p_mine for the current market price q_current.

        Returns:
            (p_jj, p_mine)
            p_jj:   probability of staying in the current state (persistence)
            p_mine: Markov-estimated true probability for this state

        Returns (0.0, q_current) if insufficient history.
        """
        if len(self._history) < self._min_obs:
            logger.debug(f"Insufficient history ({len(self._history)} obs), returning no-edge estimate")
            return 0.0, q_current

        observations = list(self._history)
        states = [digitize(q) for q in observations]
        current_state = digitize(q_current)

        # Build transition count matrix
        counts = np.zeros((N_STATES, N_STATES), dtype=float)
        for i in range(len(states) - 1):
            counts[states[i], states[i + 1]] += 1

        # Normalize rows to get transition probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero for unvisited states
        row_sums = np.where(row_sums == 0, 1, row_sums)
        transition_matrix = counts / row_sums

        # p_jj: self-transition probability of current state
        p_jj = float(transition_matrix[current_state, current_state])

        # p_mine: expected next-state probability, computed as the
        # probability-weighted midpoint of the distribution over next states
        # given we are in current_state now.
        next_state_dist = transition_matrix[current_state]  # shape (N_STATES,)
        midpoints = np.array([state_midpoint(s) for s in range(N_STATES)])
        p_mine = float(np.dot(next_state_dist, midpoints))

        logger.debug(
            f"Markov | state={current_state} | q={q_current:.3f} | "
            f"p_jj={p_jj:.3f} | p_mine={p_mine:.3f} | obs={len(self._history)}"
        )

        return p_jj, p_mine

    @property
    def n_observations(self) -> int:
        return len(self._history)
