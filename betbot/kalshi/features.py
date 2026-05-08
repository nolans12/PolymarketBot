"""
features.py — Feature vector builder for the Kalshi lead-lag bot.

Features:
  x_0 .. x_30      log(btc_microprice_lag / K) at 0,5,10,15,20,25,30s lags
  tau_s             seconds until window close
  inv_sqrt_tau      1 / sqrt(tau + 1)
  kalshi_spread     yes_ask - yes_bid
  kalshi_lag_5s     yes_mid_now - yes_mid_{t-5s}
  kalshi_lag_10s    yes_mid_now - yes_mid_{t-10s}
  kalshi_lag_30s    yes_mid_now - yes_mid_{t-30s}
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from betbot.kalshi.book import SpotBook, KalshiBook

# Ordered feature names — must match as_array() order exactly.
# NOTE: The 4 depth features at the end were added when WS feed launched.
# Models trained before then have 13 features; models trained after have 17.
# `model.json["feature_names"]` records what each saved model expects.
FEATURE_NAMES = [
    "x_0",            # log(microprice_now / K)
    "x_5",            # log(microprice_{t-5s} / K)
    "x_10",
    "x_15",
    "x_20",
    "x_25",
    "x_30",
    "tau_s",          # seconds until window close
    "inv_sqrt_tau",   # 1 / sqrt(tau + 1)
    "kalshi_spread",  # yes_ask - yes_bid
    "kalshi_lag_5s",  # yes_mid_now - yes_mid_{t-5s}
    "kalshi_lag_10s", # yes_mid_now - yes_mid_{t-10s}
    "kalshi_lag_30s", # yes_mid_now - yes_mid_{t-30s}
    "yes_bid_size",   # contracts at best YES bid
    "yes_ask_size",   # contracts at best YES ask (= best NO bid size)
    "yes_depth_5c",   # total YES bid contracts within 5c of best
    "no_depth_5c",    # total NO  bid contracts within 5c of best
]
N_FEATURES = len(FEATURE_NAMES)


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
    inv_sqrt_tau:  float
    kalshi_spread: float
    kalshi_lag_5s:  float
    kalshi_lag_10s: float
    kalshi_lag_30s: float
    yes_bid_size:   float
    yes_ask_size:   float
    yes_depth_5c:   float
    no_depth_5c:    float
    complete: bool    # False during cold-start (ring buffer not warm yet)

    def as_array(self) -> np.ndarray:
        return np.array([
            self.x_0, self.x_5, self.x_10, self.x_15, self.x_20, self.x_25, self.x_30,
            self.tau_s, self.inv_sqrt_tau,
            self.kalshi_spread,
            self.kalshi_lag_5s, self.kalshi_lag_10s, self.kalshi_lag_30s,
            self.yes_bid_size, self.yes_ask_size,
            self.yes_depth_5c, self.no_depth_5c,
        ], dtype=np.float64)


def build_features(spot: SpotBook, kb: KalshiBook) -> Optional[FeatureVec]:
    """Construct a FeatureVec from current live state."""
    if not spot.ready or not kb.ready or kb.floor_strike <= 0:
        return None

    K      = kb.floor_strike
    mp_now = spot.microprice
    tau    = kb.tau_s()

    if mp_now <= 0 or K <= 0:
        return None

    mp5  = spot.microprice_at(5)
    mp10 = spot.microprice_at(10)
    mp15 = spot.microprice_at(15)
    mp20 = spot.microprice_at(20)
    mp25 = spot.microprice_at(25)
    mp30 = spot.microprice_at(30)

    x_0  = _log_ratio(mp_now,         K)
    x_5  = _log_ratio(mp5  or mp_now, K)
    x_10 = _log_ratio(mp10 or mp_now, K)
    x_15 = _log_ratio(mp15 or mp_now, K)
    x_20 = _log_ratio(mp20 or mp_now, K)
    x_25 = _log_ratio(mp25 or mp_now, K)
    x_30 = _log_ratio(mp30 or mp_now, K)

    inv_sqrt_tau = 1.0 / math.sqrt(tau + 1.0)

    kalshi_spread  = kb.yes_ask - kb.yes_bid
    km5  = kb.yes_mid_at(5)
    km10 = kb.yes_mid_at(10)
    km30 = kb.yes_mid_at(30)
    kalshi_lag_5s  = (kb.yes_mid - km5)  if km5  is not None else 0.0
    kalshi_lag_10s = (kb.yes_mid - km10) if km10 is not None else 0.0
    kalshi_lag_30s = (kb.yes_mid - km30) if km30 is not None else 0.0

    # ---- Depth features (zero if WS feed not in use) ----
    yes_top, no_top = kb.top_n_levels(10)
    yes_bid_size = yes_top[0][1] if yes_top else 0.0
    yes_ask_size = no_top[0][1]  if no_top  else 0.0
    yes_best     = yes_top[0][0] if yes_top else 0.0
    no_best      = no_top[0][0]  if no_top  else 0.0
    yes_depth_5c = sum(s for p, s in yes_top if p >= yes_best - 0.05) if yes_top else 0.0
    no_depth_5c  = sum(s for p, s in no_top  if p >= no_best  - 0.05) if no_top  else 0.0

    # complete = ring buffer has at least 30s of real history
    complete = (mp30 is not None) and spot.ready and kb.ready

    return FeatureVec(
        x_0=x_0, x_5=x_5, x_10=x_10, x_15=x_15, x_20=x_20, x_25=x_25, x_30=x_30,
        tau_s=tau, inv_sqrt_tau=inv_sqrt_tau,
        kalshi_spread=kalshi_spread,
        kalshi_lag_5s=kalshi_lag_5s, kalshi_lag_10s=kalshi_lag_10s, kalshi_lag_30s=kalshi_lag_30s,
        yes_bid_size=yes_bid_size, yes_ask_size=yes_ask_size,
        yes_depth_5c=yes_depth_5c, no_depth_5c=no_depth_5c,
        complete=complete,
    )
