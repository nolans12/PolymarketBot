"""
features.py — Feature engineering for the lead-lag regression.

Single entry point: build_features(spot, cb_book, poly, win, now_s)
Returns a FeatureVec dataclass with every input the regression needs,
plus a dict of raw column values ready to merge into a decision row.

Feature groups (see CLAUDE.md §6.2):
  Binance lagged log-K ratios   x_now .. x_120
  Coinbase lagged log-K ratios  cb_x_now .. cb_x_60
  Time features                 tau, inv_sqrt_tau
  Binance microstructure        ofi_l1, ofi_l5, momentum_30s, momentum_60s
  Polymarket microstructure     pm_book_imbalance, pm_trade_flow_30s
  Cross-asset                   cross_asset_momentum_60s
  Volatility (diagnostic only)  sigma_per_sec_realized
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from polybot.infra.config import LOOKBACK_HORIZONS_S
from polybot.state.spot_book import SpotBook
from polybot.state.coinbase_book import CoinbaseBook
from polybot.state.poly_book import PolyBook
from polybot.state.window import WindowState

# Coinbase lookback horizons — sparser than Binance because ticker fires less often
CB_LOOKBACK_HORIZONS_S = [0, 15, 30, 60]

# Realised-vol EWMA half-life in seconds
_VOL_HALFLIFE_S  = 30.0
_VOL_ALPHA       = 1.0 - math.exp(-1.0 / _VOL_HALFLIFE_S)

# Module-level EWMA state per asset (reset on import — fine for a single process)
_ewma_var: dict[str, float] = {}


@dataclass
class FeatureVec:
    """
    All regression inputs for one decision tick on one asset.
    None means the feature is unavailable (feed not ready, buffer cold, etc.).
    The regression must handle None by imputing or skipping those rows.
    """
    asset: str

    # ---- Binance lagged log(microprice/K) ----
    x_now:  Optional[float] = None   # log(microprice_t / K)
    x_15:   Optional[float] = None
    x_30:   Optional[float] = None
    x_45:   Optional[float] = None
    x_60:   Optional[float] = None
    x_90:   Optional[float] = None
    x_120:  Optional[float] = None

    # ---- Coinbase lagged log(microprice/K) ----
    cb_x_now: Optional[float] = None
    cb_x_15:  Optional[float] = None
    cb_x_30:  Optional[float] = None
    cb_x_60:  Optional[float] = None

    # ---- Time ----
    tau:          Optional[float] = None   # seconds remaining in window
    inv_sqrt_tau: Optional[float] = None   # 1 / sqrt(tau + 1)

    # ---- Binance microstructure ----
    ofi_l1:          Optional[float] = None
    ofi_l5_weighted: Optional[float] = None
    momentum_30s:    Optional[float] = None   # log(mp_now / mp_30s_ago)
    momentum_60s:    Optional[float] = None

    # ---- Polymarket microstructure ----
    pm_book_imbalance: Optional[float] = None
    pm_trade_flow_30s: Optional[float] = None

    # ---- Cross-asset ----
    cross_asset_momentum_60s: Optional[float] = None

    # ---- Volatility (diagnostic, not used in regression) ----
    sigma_per_sec_realized: Optional[float] = None

    # ---- Completeness flag ----
    # True when the minimum set needed to run the regression is present.
    # The regression checks this before predicting.
    complete: bool = False

    def as_dict(self) -> dict:
        """
        Return a flat dict matching the column names in the decisions table.
        Used by the scheduler to merge into the decision row.
        """
        return {
            "x_now_logKratio":        self.x_now,
            "x_15_logKratio":         self.x_15,
            "x_30_logKratio":         self.x_30,
            "x_45_logKratio":         self.x_45,
            "x_60_logKratio":         self.x_60,
            "x_90_logKratio":         self.x_90,
            "x_120_logKratio":        self.x_120,
            "cb_x_now_logKratio":     self.cb_x_now,
            "cb_x_15_logKratio":      self.cb_x_15,
            "cb_x_30_logKratio":      self.cb_x_30,
            "cb_x_60_logKratio":      self.cb_x_60,
            "ofi_l1":                 self.ofi_l1,
            "ofi_l5_weighted":        self.ofi_l5_weighted,
            "pm_book_imbalance":      self.pm_book_imbalance,
            "pm_trade_flow_30s":      self.pm_trade_flow_30s,
            "momentum_30s":           self.momentum_30s,
            "momentum_60s":           self.momentum_60s,
            "cross_asset_momentum_60s": self.cross_asset_momentum_60s,
            "sigma_per_sec_realized": self.sigma_per_sec_realized,
        }

    def regression_vector(self) -> Optional[list[float]]:
        """
        Return the ordered feature list consumed by the regression model.
        Returns None if the feature vector is incomplete.

        Order must match RegressionModel.FEATURE_NAMES exactly.
        """
        if not self.complete:
            return None
        return [
            self.x_now,
            self.x_15,
            self.x_30,
            self.x_45,
            self.x_60,
            self.x_90,
            self.x_120,
            self.cb_x_now  if self.cb_x_now  is not None else 0.0,
            self.cb_x_15   if self.cb_x_15   is not None else 0.0,
            self.cb_x_30   if self.cb_x_30   is not None else 0.0,
            self.cb_x_60   if self.cb_x_60   is not None else 0.0,
            self.tau,
            self.inv_sqrt_tau,
            self.ofi_l1          if self.ofi_l1          is not None else 0.0,
            self.ofi_l5_weighted if self.ofi_l5_weighted is not None else 0.0,
            self.pm_book_imbalance if self.pm_book_imbalance is not None else 0.0,
            self.pm_trade_flow_30s if self.pm_trade_flow_30s is not None else 0.0,
            self.momentum_30s     if self.momentum_30s      is not None else 0.0,
            self.momentum_60s     if self.momentum_60s      is not None else 0.0,
            self.cross_asset_momentum_60s if self.cross_asset_momentum_60s is not None else 0.0,
        ]


# Canonical feature name list — must stay in sync with regression_vector() above.
FEATURE_NAMES = [
    "x_now", "x_15", "x_30", "x_45", "x_60", "x_90", "x_120",
    "cb_x_now", "cb_x_15", "cb_x_30", "cb_x_60",
    "tau", "inv_sqrt_tau",
    "ofi_l1", "ofi_l5_weighted",
    "pm_book_imbalance", "pm_trade_flow_30s",
    "momentum_30s", "momentum_60s",
    "cross_asset_momentum_60s",
]


def build_features(
    spot: SpotBook,
    cb_book: Optional[CoinbaseBook],
    poly: PolyBook,
    win: WindowState,
    now_s: float,
    ofi_l1: float,
    ofi_l5: float,
    cross_spot: Optional[SpotBook] = None,
    cross_win: Optional[WindowState] = None,
) -> FeatureVec:
    """
    Compute all features for one asset at one tick.

    ofi_l1 / ofi_l5 are passed in pre-drained from SpotBook.drain_ofi()
    because draining is destructive and must happen exactly once per tick.

    Returns a FeatureVec. Check .complete before using .regression_vector().
    """
    fv = FeatureVec(asset=spot.asset)

    K = win.K
    if not K or K <= 0:
        return fv  # no strike yet — nothing to compute

    # ---- tau ----
    tau = win.tau_s(now_s)
    fv.tau = float(tau)
    fv.inv_sqrt_tau = 1.0 / math.sqrt(tau + 1.0)

    # ---- Binance lagged log-K features ----
    if spot.ready:
        lags   = [0, 15, 30, 45, 60, 90, 120]
        fields = ["x_now", "x_15", "x_30", "x_45", "x_60", "x_90", "x_120"]
        for lag_s, fname in zip(lags, fields):
            mp = spot.microprice_at(lag_s)
            if mp and mp > 0:
                setattr(fv, fname, float(math.log(mp / K)))

        # Momentum (Binance)
        mp_now = spot.microprice_at(0)
        mp_30  = spot.microprice_at(30)
        mp_60  = spot.microprice_at(60)
        if mp_now and mp_30 and mp_30 > 0:
            fv.momentum_30s = float(math.log(mp_now / mp_30))
        if mp_now and mp_60 and mp_60 > 0:
            fv.momentum_60s = float(math.log(mp_now / mp_60))

        # Realised vol EWMA — 1-tick EWMA of (log return)^2
        # Uses mp_now vs mp_30 as a proxy for the 30s return
        if mp_now and mp_30 and mp_30 > 0:
            r = math.log(mp_now / mp_30)
            key = spot.asset
            prev_var = _ewma_var.get(key, r * r)
            new_var = _VOL_ALPHA * r * r + (1.0 - _VOL_ALPHA) * prev_var
            _ewma_var[key] = new_var
            # Convert variance-per-30s to per-second
            fv.sigma_per_sec_realized = float(math.sqrt(new_var / 30.0))

    # ---- OFI ----
    fv.ofi_l1          = float(ofi_l1)
    fv.ofi_l5_weighted = float(ofi_l5)

    # ---- Coinbase lagged log-K features ----
    if cb_book and cb_book.ready:
        for lag_s, fname in [(0, "cb_x_now"), (15, "cb_x_15"),
                             (30, "cb_x_30"), (60, "cb_x_60")]:
            mp = cb_book.microprice_at(lag_s)
            if mp and mp > 0:
                setattr(fv, fname, float(math.log(mp / K)))

    # ---- Polymarket microstructure ----
    if poly.ready and poly.up and poly.down:
        fv.pm_book_imbalance = float(poly.book_imbalance())
        fv.pm_trade_flow_30s = float(poly.trade_flow_30s())

    # ---- Cross-asset momentum ----
    if cross_spot and cross_spot.ready and cross_win and cross_win.K and cross_win.K > 0:
        c_now = cross_spot.microprice_at(0)
        c_60  = cross_spot.microprice_at(60)
        if c_now and c_60 and c_60 > 0:
            fv.cross_asset_momentum_60s = float(math.log(c_now / c_60))

    # ---- Completeness check ----
    # Minimum required: x_now through x_60 (5 Binance lags) + tau
    binance_core = (fv.x_now is not None and fv.x_15 is not None and
                    fv.x_30  is not None and fv.x_45 is not None and
                    fv.x_60  is not None)
    fv.complete = binance_core and fv.tau is not None

    return fv
