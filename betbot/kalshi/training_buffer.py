"""
training_buffer.py — Rolling window of (features, target) training samples.

Thread-safe deque capped to TRAINING_WINDOW_S seconds of history.
The refitter reads it every 5 minutes; the sampler writes every second.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from betbot.kalshi.config import TRAINING_WINDOW_S


@dataclass(slots=True)
class _Sample:
    ts_ns: int
    X:     np.ndarray   # feature vector, shape (N_FEATURES,)
    y:     float        # logit(kalshi_yes_mid) — training target


class TrainingBuffer:
    """
    Append training samples as they arrive; trim old ones automatically.
    All methods are thread-safe (called from both the asyncio event loop
    via run_in_executor and the main loop).
    """

    def __init__(self, window_s: int = TRAINING_WINDOW_S):
        self._window_ns: int = window_s * 1_000_000_000
        self._data: deque[_Sample] = deque()
        self._lock = threading.Lock()

    def append(self, X: np.ndarray, y: float) -> None:
        """Add a sample, dropping anything older than the rolling window."""
        now_ns  = time.time_ns()
        cutoff  = now_ns - self._window_ns
        sample  = _Sample(ts_ns=now_ns, X=X.copy(), y=y)
        with self._lock:
            self._data.append(sample)
            while self._data and self._data[0].ts_ns < cutoff:
                self._data.popleft()

    def append_with_ts(self, X: np.ndarray, y: float, ts_ns: int) -> None:
        """
        Append a sample using an explicit timestamp instead of now().
        Used by Scheduler._bootstrap_from_history() to replay ticks.csv on
        startup so the model is fit before the live decision loop runs.
        """
        cutoff = time.time_ns() - self._window_ns
        if ts_ns < cutoff:
            return
        sample = _Sample(ts_ns=ts_ns, X=X.copy(), y=y)
        with self._lock:
            self._data.append(sample)

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y, ts_ns) arrays for all samples in the rolling window.
        Returns empty arrays if the buffer is empty.
        """
        with self._lock:
            if not self._data:
                empty = np.empty((0,), dtype=np.float64)
                return empty.reshape(0, 0), empty, empty
            rows = list(self._data)

        X    = np.vstack([s.X    for s in rows])
        y    = np.array([s.y    for s in rows], dtype=np.float64)
        ts   = np.array([s.ts_ns for s in rows], dtype=np.int64)
        return X, y, ts

    def __len__(self) -> int:
        return len(self._data)

    def span_minutes(self) -> float:
        """Wall-clock span of buffered data in minutes."""
        with self._lock:
            if len(self._data) < 2:
                return 0.0
            span_ns = self._data[-1].ts_ns - self._data[0].ts_ns
        return span_ns / 1e9 / 60.0
