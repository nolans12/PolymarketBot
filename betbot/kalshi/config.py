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
# Spot feed (Coinbase or Binance — picks via SPOT_SOURCE env var)
# ---------------------------------------------------------------------------
# Coinbase: US-legal, no VPN required, default.
# Binance:  geo-blocked from US IPs, requires VPN if running from US.
# Both feed the same SpotBook and produce identical features — swapping
# is a config flag with no model retrain required.

SPOT_SOURCE = os.getenv("SPOT_SOURCE", "coinbase").strip().lower()
if SPOT_SOURCE not in ("coinbase", "binance"):
    raise ValueError(
        f"SPOT_SOURCE must be 'coinbase' or 'binance', got {SPOT_SOURCE!r}"
    )

COINBASE_WS      = "wss://advanced-trade-ws.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"

BINANCE_WS       = "wss://stream.binance.com:443/ws/btcusdt@bookTicker"
BINANCE_PRODUCT  = "BTCUSDT"

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

WINDOW_SECONDS = 900   # 15-min Kalshi market

# ---------------------------------------------------------------------------
# Microprice ring buffer (feeds lagged features)
# ---------------------------------------------------------------------------

RING_BUFFER_S = 300    # 5 minutes at 1 sample/s

# Lag horizons in seconds — must match FEATURE_NAMES in features.py.
# Short-lag set for testing fast-lag arb hypothesis (~10s expected lag).
LOOKBACK_S = [0, 5, 10, 15, 20, 25, 30]

# ---------------------------------------------------------------------------
# Training / model
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_S      = 0.1          # add training sample every 100ms
REFIT_INTERVAL_S       = 300          # refit model every 5 minutes
TRAINING_WINDOW_S      = 4 * 3600     # rolling 4-hour training window
MIN_TRAIN_SAMPLES      = 3600        # first fit after ~6 min at 10 samples/s
RIDGE_ALPHAS           = [0.001, 0.01, 0.1, 1.0, 10.0]
HELDOUT_FRACTION       = 0.20         # hold out last 20% for out-of-sample R²

# Training-sample filter: train on all yes_mid values.
# Clamp to (0.001, 0.999) only to keep logit finite — no regime exclusion.
TRAIN_YES_MID_MIN      = 0.001
TRAIN_YES_MID_MAX      = 0.999

# Decision filter: only ENTER new positions when yes_mid is in [MIN, MAX].
# Matches the training filter -- we don't trade in regimes the model
# wasn't trained on. Open positions still run exit logic regardless.
DECISION_YES_MID_MIN   = 0.01
DECISION_YES_MID_MAX   = 0.99

# Model sanity gates — abstain when these are violated
MODEL_MIN_CV_R2        = 0.05         # permissive during early data collection
MODEL_MAX_DISAGREEMENT = 0.20         # |q_predicted - q_actual| before we distrust model

# ---------------------------------------------------------------------------
# Decision loop
# ---------------------------------------------------------------------------

DECISION_INTERVAL_S    = 0.5         # decision tick every 500ms
MIN_ENTRY_INTERVAL_S   = 10.0        # minimum seconds between consecutive entries

# ---------------------------------------------------------------------------
# Fees  (Kalshi taker formula = THETA * p * (1 - p) per dollar bet, per leg)
# Maker orders on Kalshi pay 0. Strategy is maker-entry / taker-exit.
# ---------------------------------------------------------------------------
THETA_FEE_TAKER = 0.07
THETA_FEE_MAKER = 0.0

# Execution mode for the entry leg.
#   "maker" -- post limit at yes_bid (or 1-yes_ask for NO), pay 0 entry fee.
#              Realises spread + fee savings (~+43% upper bound on this dataset
#              per scripts/compare_models.py --mode maker), conditional on the
#              limit actually filling.
#   "taker" -- cross to yes_ask (or 1-yes_bid for NO), pay taker fee on entry.
# Exit is ALWAYS taker (sell into the bid for fast unwind on lag-close).
ENTRY_MODE = os.getenv("ENTRY_MODE", "taker").strip().lower()
if ENTRY_MODE not in ("taker", "maker"):
    raise ValueError(
        f"ENTRY_MODE must be 'taker' or 'maker', got {ENTRY_MODE!r}"
    )

# ---------------------------------------------------------------------------
# Kelly tier table — evaluated top-to-bottom, first match wins
# (delta_min, wallet_fraction)
# ---------------------------------------------------------------------------

# Kelly tiers: original 5-tier floor structure (0.02 .. 0.30) with wallet
# fractions bumped ~1.5x for more aggressive sizing per user preference.
# Top tier puts 15% of wallet on highest-confidence trades; lowest tier
# fires at 0.02 net edge with 2.5% of wallet.
KELLY_TIERS = [
    (0.30, 0.10),
    (0.15, 0.08),
    (0.08, 0.05),
    (0.04, 0.03),
    (0.02, 0.015),
]

SIZE_MIN_USD = 1.0
SIZE_MAX_USD = 25.0

# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------

LAG_CLOSE_THRESHOLD = 0.005      # exit when edge compresses below this (backtest optimum)
STOP_THRESHOLD      = 0.03    # cut loss if edge erodes 3 cents below entry
FALLBACK_TAU_S      = 60      # hold-to-resolution at this τ (1 min before close)
MAX_HOLD_S          = 90      # force-exit at the bid if the trade hasn't lag-closed
                              # or stopped within this many seconds (backtest optimum).

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
