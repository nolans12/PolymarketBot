"""
features.py — Feature vector builder for the Kalshi lead-lag regression.

Target:  y = logit(kalshi_yes_mid_t)
Features: lagged log(coinbase_microprice / K) at [0,5,10,15,20,25,30]s,
          tau, 1/sqrt(tau), spot momentum, Kalshi spread + momentum.

q_settled: substitute x_0 (current spot) into all lagged slots to predict
where Kalshi will quote once it finishes digesting the current spot move.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from betbot.kalshi.book import SpotBook, KalshiBook

# Ordered feature names — this order matches as_array() and settled_array()
FEATURE_NAMES = [
    "x_0",               # log(microprice_now / K)
    "x_5",               # log(microprice_{t-5s} / K)
    "x_10",
    "x_15",
    "x_20",
    "x_25",
    "x_30",
    "tau_s",             # seconds until window close
    "inv_sqrt_tau",      # 1 / sqrt(tau + 1) — emphasises near-close moves
    "kalshi_spread",       # yes_ask - yes_bid (proxy for book liquidity)
    "kalshi_momentum_5s",  # yes_mid_now - yes_mid_{t-5s}  — fast Kalshi trend
    "kalshi_momentum_10s", # yes_mid_now - yes_mid_{t-10s} — medium Kalshi trend
    "kalshi_momentum_30s", # yes_mid_now - yes_mid_{t-30s} — slow Kalshi trend
]
N_FEATURES = len(FEATURE_NAMES)

# Indices of lag slots that get replaced by x_0 when computing q_settled
_LAG_INDICES = [FEATURE_NAMES.index(n) for n in ["x_5", "x_10", "x_15", "x_20", "x_25", "x_30"]]
_X0_IDX      = FEATURE_NAMES.index("x_0")


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _log_ratio(num: float, den: float) -> float:
    if den <= 0 or num <= 0:
        return 0.0
    return math.log(num / den)


@dataclass
class FeatureVec:
    x_0:   float
    x_5:   float
    x_10:  float
    x_15:  float
    x_20:  float
    x_25:  float
    x_30:  float
    tau_s: float
    inv_sqrt_tau: float
    kalshi_spread: float
    kalshi_momentum_5s:  float
    kalshi_momentum_10s: float
    kalshi_momentum_30s: float
    complete: bool    # False during cold-start (ring buffer not warm yet)

    def as_array(self) -> np.ndarray:
        return np.array([
            self.x_0, self.x_5, self.x_10, self.x_15, self.x_20, self.x_25, self.x_30,
            self.tau_s, self.inv_sqrt_tau,
            self.kalshi_spread,
            self.kalshi_momentum_5s, self.kalshi_momentum_10s, self.kalshi_momentum_30s,
        ], dtype=np.float64)

    def settled_array(self) -> np.ndarray:
        """
        Replace all lagged spot slots with x_0 (current microprice/K).
        Model applied to this vector predicts q_settled — where Kalshi will
        quote once it has fully priced in the current spot move.
        """
        a = self.as_array()
        for i in _LAG_INDICES:
            a[i] = self.x_0
        return a


def build_features(spot: SpotBook, kb: KalshiBook) -> Optional[FeatureVec]:
    """
    Construct a FeatureVec from current live state.
    Returns None if either book isn't ready or K is unknown.
    """
    if not spot.ready or not kb.ready or kb.floor_strike <= 0:
        return None

    K      = kb.floor_strike
    mp_now = spot.microprice
    tau    = kb.tau_s()

    if mp_now <= 0 or K <= 0:
        return None

    # Lagged microprices — fall back to mp_now during ring warmup so model
    # gets a feature vector (complete=False flags it as unreliable for training)
    mp5   = spot.microprice_at(5)
    mp10  = spot.microprice_at(10)
    mp15  = spot.microprice_at(15)
    mp20  = spot.microprice_at(20)
    mp25  = spot.microprice_at(25)
    mp30  = spot.microprice_at(30)

    x_0   = _log_ratio(mp_now, K)
    x_5   = _log_ratio(mp5   or mp_now, K)
    x_10  = _log_ratio(mp10  or mp_now, K)
    x_15  = _log_ratio(mp15  or mp_now, K)
    x_20  = _log_ratio(mp20  or mp_now, K)
    x_25  = _log_ratio(mp25  or mp_now, K)
    x_30  = _log_ratio(mp30  or mp_now, K)

    inv_sqrt_tau = 1.0 / math.sqrt(tau + 1.0)

    kalshi_spread = kb.yes_ask - kb.yes_bid
    km5  = kb.yes_mid_at(5)
    km10 = kb.yes_mid_at(10)
    km30 = kb.yes_mid_at(30)
    kalshi_momentum_5s  = (kb.yes_mid - km5)  if km5  is not None else 0.0
    kalshi_momentum_10s = (kb.yes_mid - km10) if km10 is not None else 0.0
    kalshi_momentum_30s = (kb.yes_mid - km30) if km30 is not None else 0.0

    # complete = ring buffer has at least 30s of real history
    complete = (mp30 is not None) and spot.ready and kb.ready

    return FeatureVec(
        x_0=x_0, x_5=x_5, x_10=x_10, x_15=x_15, x_20=x_20, x_25=x_25, x_30=x_30,
        tau_s=tau, inv_sqrt_tau=inv_sqrt_tau,
        kalshi_spread=kalshi_spread,
        kalshi_momentum_5s=kalshi_momentum_5s, kalshi_momentum_10s=kalshi_momentum_10s,
        kalshi_momentum_30s=kalshi_momentum_30s,
        complete=complete,
    )
