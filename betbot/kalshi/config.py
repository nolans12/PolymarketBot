"""
config.py — All tunables for the Kalshi 15-min BTC lag-arb bot.

Credentials are loaded from .env. Strategy parameters are version-controlled here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

KALSHI_REST     = "https://api.elections.kalshi.com"
KALSHI_API_PATH = "/trade-api/v2"
KALSHI_KEY_ID   = os.getenv("KALSHI_API_KEY_ID", "").strip()
KALSHI_PEM_FILE = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
KALSHI_PEM_INLINE = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")

# 15-min BTC Up/Down series
KALSHI_SERIES   = "KXBTC15M"

# ---------------------------------------------------------------------------
# Coinbase (primary spot feed — no VPN needed)
# ---------------------------------------------------------------------------

COINBASE_WS      = "wss://advanced-trade-ws.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

WINDOW_SECONDS = 900   # 15-min Kalshi market

# ---------------------------------------------------------------------------
# Microprice ring buffer (feeds lagged features)
# ---------------------------------------------------------------------------

RING_BUFFER_S = 300    # 5 minutes at 1 sample/s

# Lag horizons in seconds — must match FEATURE_NAMES in features.py
LOOKBACK_S = [0, 15, 30, 60, 90, 120]

# ---------------------------------------------------------------------------
# Training / model
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_S      = 1.0          # add training sample every N seconds
REFIT_INTERVAL_S       = 300          # refit model every 5 minutes
TRAINING_WINDOW_S      = 4 * 3600     # rolling 4-hour training window
MIN_TRAIN_SAMPLES      = 360          # first fit after ~6 min at 1 sample/s
RIDGE_ALPHAS           = [0.001, 0.01, 0.1, 1.0, 10.0]
HELDOUT_FRACTION       = 0.20         # hold out last 20% for out-of-sample R²

# Training-sample filter: only train on yes_mid in [MIN, MAX].
# Extreme tails (p>0.95 or p<0.05) have logit values that disproportionately
# pull the regression. Filtering them keeps the model focused on the
# uncertain-middle regime where the lag-arb edge actually lives.
TRAIN_YES_MID_MIN      = 0.05
TRAIN_YES_MID_MAX      = 0.95

# Decision filter: only ENTER new positions when yes_mid is well inside the
# trained regime. Slightly tighter than the training range so the model is
# extrapolating less at the boundaries when we trade. Open positions still
# run exit logic regardless of this filter.
DECISION_YES_MID_MIN   = 0.10
DECISION_YES_MID_MAX   = 0.90

# Model sanity gates — abstain when these are violated
MODEL_MIN_CV_R2        = 0.05         # permissive during early data collection
MODEL_MAX_DISAGREEMENT = 0.20         # |q_predicted - q_actual| before we distrust model

# ---------------------------------------------------------------------------
# Decision loop
# ---------------------------------------------------------------------------

DECISION_INTERVAL_S = 10              # scheduler tick every 10s

# ---------------------------------------------------------------------------
# Fees  (Kalshi 15-min taker estimate; calibrate from live fills)
# taker_fee(p) = THETA * p * (1 - p)
# ---------------------------------------------------------------------------

THETA_FEE = 0.03

# ---------------------------------------------------------------------------
# Kelly tier table — evaluated top-to-bottom, first match wins
# (delta_min, wallet_fraction)
# ---------------------------------------------------------------------------

KELLY_TIERS = [
    (0.30, 0.10),
    (0.15, 0.08),
    (0.08, 0.05),
    (0.04, 0.03),
    (0.02, 0.015),
]

SIZE_MIN_USD = 5.0
SIZE_MAX_USD = 500.0

# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------

LAG_CLOSE_THRESHOLD = 0.005   # exit when edge compresses to ½ cent
STOP_THRESHOLD      = 0.03    # cut loss if edge erodes 3 cents below entry
FALLBACK_TAU_S      = 60      # hold-to-resolution at this τ (1 min before close)

# Circuit breakers
COINBASE_STALE_MS_MAX = 10_000   # 10s
KALSHI_STALE_MS_MAX   = 15_000   # 15s
WIDE_SPREAD_THRESHOLD = 0.12     # abstain if Kalshi spread > 12 cents

# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------

DRY_RUN    = os.getenv("DRY_RUN", "true").lower() == "true"
LOG_DIR    = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
