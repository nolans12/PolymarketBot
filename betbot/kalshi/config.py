"""
config.py — Tunables for the Kalshi 15-min lag-arb bot.

The model is trained offline (scripts/train_model.py) and loaded at startup
via --model-file. Nothing here controls training — only trading behaviour.

Credentials come from .env. Everything else is version-controlled here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Kalshi API
# ---------------------------------------------------------------------------

KALSHI_REST       = "https://api.elections.kalshi.com"
KALSHI_API_PATH   = "/trade-api/v2"
KALSHI_KEY_ID     = os.getenv("KALSHI_API_KEY_ID", "").strip()
KALSHI_PEM_FILE   = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
KALSHI_PEM_INLINE = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")

# 15-min Up/Down series per asset
KALSHI_SERIES_BTC = "KXBTC15M"
KALSHI_SERIES_ETH = "KXETH15M"
KALSHI_SERIES_SOL = "KXSOL15M"
KALSHI_SERIES_XRP = "KXXRP15M"

KALSHI_SERIES = KALSHI_SERIES_BTC   # default (single-asset mode)

KALSHI_ASSETS = {
    "BTC": KALSHI_SERIES_BTC,
    "ETH": KALSHI_SERIES_ETH,
    "SOL": KALSHI_SERIES_SOL,
    "XRP": KALSHI_SERIES_XRP,
}

# ---------------------------------------------------------------------------
# Spot feed
# ---------------------------------------------------------------------------
# SPOT_SOURCE = "coinbase"  — US-legal, no VPN required (default)
# SPOT_SOURCE = "binance"   — geo-blocked from US IPs, requires VPN

SPOT_SOURCE = os.getenv("SPOT_SOURCE", "binance").strip().lower()
if SPOT_SOURCE not in ("coinbase", "binance"):
    raise ValueError(f"SPOT_SOURCE must be 'coinbase' or 'binance', got {SPOT_SOURCE!r}")

COINBASE_WS      = "wss://advanced-trade-ws.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"

COINBASE_PRODUCTS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
}

BINANCE_WS      = "wss://stream.binance.com:443/ws/btcusdt@bookTicker"
BINANCE_PRODUCT = "BTCUSDT"

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

WINDOW_SECONDS = 900   # 15-min Kalshi market

# ---------------------------------------------------------------------------
# Ring buffer (feeds lagged features to the live model)
# ---------------------------------------------------------------------------

RING_BUFFER_S = 300    # 5 minutes — covers the longest lag feature (x_30)

# ---------------------------------------------------------------------------
# LightGBM model horizons (must match the saved model)
# ---------------------------------------------------------------------------

LGBM_FORECAST_HORIZONS = [5, 10, 15, 60]  # seconds ahead
LGBM_PRIMARY_HORIZON   = 10               # which horizon feeds q_settled / edge

# ---------------------------------------------------------------------------
# Tick sampling (for data collection during dry runs)
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_S = 0.1   # write a training sample every 100ms during dry run

# ---------------------------------------------------------------------------
# Training-sample filter (applied in train_model.py / tune_trading_knobs.py / test_all.py)
# Filters both the feature rows (yes_mid) and target rows (yes_bid) to exclude
# extreme near-0 / near-1 prices where the market is essentially resolved already.
# ---------------------------------------------------------------------------

TRAIN_PRICE_MIN = 0.001
TRAIN_PRICE_MAX = 0.999

# Legacy aliases — kept so old code doesn't break during transition
TRAIN_YES_MID_MIN = TRAIN_PRICE_MIN
TRAIN_YES_MID_MAX = TRAIN_PRICE_MAX

HELDOUT_FRACTION = 0.2   # last 20% of time-span held out for model validation

# ---------------------------------------------------------------------------
# Decision loop
# ---------------------------------------------------------------------------

DECISION_INTERVAL_S  = 0.5    # decision tick every 500ms
MIN_ENTRY_INTERVAL_S = 10.0   # minimum seconds between consecutive entries

# Decision filter: only enter when yes_mid is in this range
DECISION_YES_MID_MIN = 0.01
DECISION_YES_MID_MAX = 0.99

# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------
# Kalshi taker fee = THETA * p * (1 - p) per dollar bet, per leg.
# Maker orders pay 0. Entry is maker (post resting); exit is taker (sweep IOC).

THETA_FEE_TAKER = 0.07
THETA_FEE_MAKER = 0.0
EXIT_SLIP_CENTS = 0.0    # taker IOC sweeps at the bid — no extra slippage modeled here

ENTRY_MODE = os.getenv("ENTRY_MODE", "maker").strip().lower()
if ENTRY_MODE not in ("taker", "maker"):
    raise ValueError(f"ENTRY_MODE must be 'taker' or 'maker', got {ENTRY_MODE!r}")

# ---------------------------------------------------------------------------
# Maker entry parameters (used only when ENTRY_MODE=maker)
# ---------------------------------------------------------------------------
# Maker post price: at bid (cheapest) or bid+1c (faster fills, 1c more expensive).
# Default is bid+1 — joins inside the spread for higher fill probability.
MAKER_AT_BID_PLUS_1 = True

# How many seconds to wait for the resting limit to fill before cancelling.
# Short TTL = miss more entries but stay fresh on edge calc.
MAKER_TTL_S = 5.0

# Polling cadence for maker fill check. Kalshi limit is 20 reads/sec.
MAKER_POLL_S = 0.5


# Total wallet the bot is allowed to use.
# With --live-orders the real Kalshi balance is fetched, then CAPPED to this value.
# Set this to however much you want at risk (real balance can be higher, bot won't touch the rest).
WALLET_BALANCE = 100.0

# Hard cap: bot will NEVER place a single order worth more than SIZE_MAX_USD.
# Enforced twice: in dollar sizing AND in contract count — two independent checks.
SIZE_MIN_USD = 0.25
SIZE_MAX_USD = 2.0

# ---------------------------------------------------------------------------
# Kelly tier table  (edge_floor, wallet_fraction)
# Evaluated top-to-bottom; first match wins.
# ---------------------------------------------------------------------------

KELLY_TIERS = [
    (0.10, 0.06),
    (0.06, 0.04),
    (0.02, 0.02),   # add a small-size tier for ETH-level edges
]

# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------

LAG_CLOSE_THRESHOLD = 0.005   # exit when edge compresses below this
STOP_THRESHOLD      = None    # set to a float (e.g. 0.03) to enable stop-loss; None = disabled
FALLBACK_TAU_S      = 60      # hold to resolution when < this many seconds remain
MAX_HOLD_S          = 15      # force-exit if still open after this many seconds

# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------

COINBASE_STALE_MS_MAX = 10_000   # abstain if Coinbase feed silent > 10s
KALSHI_STALE_MS_MAX   = 15_000   # abstain if Kalshi feed silent > 15s
WIDE_SPREAD_THRESHOLD = 0.12     # abstain if Kalshi bid/ask spread > 12 cents

# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
