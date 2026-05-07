"""
tick_logger.py — Writes 1-second raw ticks to CSV for backtesting.

One row per sampler tick. Appends to existing file so a run can be resumed.
Schema is intentionally minimal — backtest.py derives all features from these columns.
"""

import csv
import threading
from pathlib import Path

COLUMNS = [
    "ts_ns",              # wall-clock nanoseconds
    "tau_s",              # seconds until window close
    "btc_microprice",     # primary spot feed (SPOT_SOURCE) microprice
    "btc_bid",            # primary spot feed best bid
    "btc_ask",            # primary spot feed best ask
    "cb_microprice",      # Coinbase microprice (always logged, even when SPOT_SOURCE=binance)
    "bn_microprice",      # Binance microprice (always logged, even when SPOT_SOURCE=coinbase)
    "yes_bid",            # Kalshi YES best bid (dollars)
    "yes_ask",            # Kalshi YES best ask (dollars)
    "yes_mid",            # (yes_bid + yes_ask) / 2
    "floor_strike",       # K — BTC strike for this window
    "window_ticker",      # e.g. KXBTC15M-26MAY061515-15
]


class TickLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        write_header = not path.exists() or path.stat().st_size == 0
        self._fh  = open(path, "a", newline="", buffering=1)
        self._csv = csv.writer(self._fh)
        if write_header:
            self._csv.writerow(COLUMNS)
            self._fh.flush()

    def log(self, ts_ns: int, tau_s: float,
            btc_microprice: float, btc_bid: float, btc_ask: float,
            cb_microprice: float, bn_microprice: float,
            yes_bid: float, yes_ask: float, yes_mid: float,
            floor_strike: float, window_ticker: str) -> None:
        with self._lock:
            self._csv.writerow([
                ts_ns,
                f"{tau_s:.1f}",
                f"{btc_microprice:.2f}",
                f"{btc_bid:.2f}",
                f"{btc_ask:.2f}",
                f"{cb_microprice:.2f}",
                f"{bn_microprice:.2f}",
                f"{yes_bid:.4f}",
                f"{yes_ask:.4f}",
                f"{yes_mid:.4f}",
                f"{floor_strike:.2f}",
                window_ticker,
            ])

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()
