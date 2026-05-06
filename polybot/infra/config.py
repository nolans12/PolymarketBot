"""
config.py — All tunables for the Polymarket lag-arb bot.

API credentials are loaded from .env (never hardcoded here).
Strategy parameters live here and are version-controlled.

See CLAUDE.md for the complete design specification.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials (from .env — never hardcode these)
# ---------------------------------------------------------------------------
PRIVATE_KEY    = os.getenv("PRIVATE_KEY", "")
API_KEY        = os.getenv("API_KEY", "")
API_SECRET     = os.getenv("API_SECRET", "")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "")

# Polymarket CLOB host (REST)
CLOB_HOST      = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID       = int(os.getenv("CHAIN_ID", "137"))

# Funder address: the wallet that actually holds USDC. Required when using a
# relayer signing key (PRIVATE_KEY signs orders; FUNDER holds the collateral).
FUNDER         = os.getenv("FUNDER", "")

# ---------------------------------------------------------------------------
# Venue endpoints
# ---------------------------------------------------------------------------

# Binance global L2 stream. Geo-blocked from US/UK/Ontario; if unreachable,
# Stage 1A pivots to Coinbase Advanced Trade WS per CLAUDE.md §5.1.
BINANCE_WS_HOST = os.getenv("BINANCE_WS_HOST", "wss://stream.binance.com:9443")

# Polymarket CLOB market WS — no auth required for market channel.
POLY_CLOB_WS    = os.getenv("POLY_CLOB_WS", "wss://ws-subscriptions-clob.polymarket.com/ws/market")

# Polymarket RTDS WS — Chainlink price stream for K capture at window boundaries.
POLY_RTDS_WS    = os.getenv("POLY_RTDS_WS", "wss://ws-live-data.polymarket.com")

# Public Polygon RPC — used by polybot.state.wallet for on-chain USDC balance.
POLYGON_RPC     = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

# Gamma REST — slug → token IDs.
GAMMA_API       = os.getenv("GAMMA_API", "https://gamma-api.polymarket.com")

# Coinbase Advanced Trade WS — public ticker channel, no auth required.
COINBASE_WS_HOST = os.getenv("COINBASE_WS_HOST", "wss://advanced-trade-ws.coinbase.com")

# ---------------------------------------------------------------------------
# Spot feed selection
#
# When ENABLE_BINANCE=true the bot subscribes to Binance L2 and uses it as
# the primary spot feed. When ENABLE_COINBASE=true the bot subscribes to
# Coinbase Advanced Trade ticker as a fallback / additional feature source.
# At least one must be true.
# ---------------------------------------------------------------------------

ENABLE_BINANCE  = os.getenv("ENABLE_BINANCE",  "true").lower() == "true"
ENABLE_COINBASE = os.getenv("ENABLE_COINBASE", "false").lower() == "true"

# K-capture source. "spot" uses the live Binance microprice at window open
# (more reliable, slightly different from Polymarket's resolution oracle).
# "chainlink" uses Polymarket RTDS - aka the "price to beat" (canonical but high-latency / gap-prone).
K_SOURCE = os.getenv("K_SOURCE", "spot").lower()

# ---------------------------------------------------------------------------
# Assets
#
# Phase 1 covers BTC and ETH only (CLAUDE.md §1). Alts (SOL, XRP) revisited
# in Phase 2 if BTC/ETH proves profitable.
# ---------------------------------------------------------------------------

ASSETS = ["btc", "eth"]

BINANCE_SYMBOLS = {
    "btc": "btcusdt",
    "eth": "ethusdt",
}

COINBASE_SYMBOLS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
}

CHAINLINK_SYMBOLS = {
    "btc": "btc/usd",
    "eth": "eth/usd",
}

# ---------------------------------------------------------------------------
# Window timing
# ---------------------------------------------------------------------------

WINDOW_SECONDS    = 300                    # Polymarket 5-min market window
DECISION_INTERVAL = int(os.getenv("POLL_INTERVAL", "10"))   # scheduler tick cadence
SAMPLE_INTERVAL   = float(os.getenv("SAMPLE_INTERVAL", "1.0"))  # training sampler cadence

# ---------------------------------------------------------------------------
# Microprice / OFI / history
# ---------------------------------------------------------------------------

MICROPRICE_RING_SECONDS = 300              # rolling 5-min ring buffer of 1s samples
OFI_WINDOW_SECONDS      = 30
OFI_BOOK_LEVELS         = 5

# Lookback grid for the regression's lagged-microprice features (CLAUDE.md §6.2)
LOOKBACK_HORIZONS_S = [0, 15, 30, 45, 60, 90, 120]

# ---------------------------------------------------------------------------
# Regression refitter
# ---------------------------------------------------------------------------

REFIT_INTERVAL_SECONDS    = 300            # 5 min
TRAINING_WINDOW_SECONDS   = 4 * 3600       # 4 h rolling window
# At 1s sampler + 10s decision ticks combined, expect ~66 rows/min.
# 30 min warmup = ~2,000 rows minimum before first fit.
# 4h window = ~15,840 rows at steady state.
MIN_TRAIN_SIZE            = 1800           # ~30 min of combined 1s+5s rows for first fit
HELDOUT_WINDOW_SECONDS    = 1800           # most-recent 30 min held out for validation
RIDGE_ALPHAS              = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

# ---------------------------------------------------------------------------
# Sanity gates around the model (CLAUDE.md §7)
# ---------------------------------------------------------------------------

MODEL_MIN_CV_R2           = 0.10
MODEL_MAX_DISAGREEMENT    = 0.15           # |q_predicted - q_actual|
MODEL_MAX_STALE_SECONDS   = 900            # 15 min

# ---------------------------------------------------------------------------
# Kelly tier table (CLAUDE.md §3.7)
# Evaluated top-to-bottom — first match wins.
# ---------------------------------------------------------------------------

KELLY_TIERS = [
    (0.30, 0.10),     # delta >= 0.30 -> 10% of wallet
    (0.15, 0.08),
    (0.08, 0.05),
    (0.04, 0.03),
    (0.02, 0.015),    # lowest tier; below this we abstain
]

# Hard position-size bounds in USD
SIZE_MIN_USD = 9.0
SIZE_MAX_USD = 800.0

# ---------------------------------------------------------------------------
# Fee model (CLAUDE.md §5.5) — taker fee f(p) = THETA * p * (1-p)
# THETA borrowed from 15-min market docs; confirmed empirically in Phase 2.
# ---------------------------------------------------------------------------

THETA_FEE = 0.05

# ---------------------------------------------------------------------------
# Exit-rule defaults (sweep parameters in research/parameter_sweep.py)
# ---------------------------------------------------------------------------

LAG_CLOSE_THRESHOLD = 0.005                # exit when edge has compressed to ~half a cent
STOP_THRESHOLD      = 0.03                 # exit if edge erodes 3 cents below entry
FALLBACK_TAU_S      = 10                   # default to resolution at this τ

# ---------------------------------------------------------------------------
# Circuit breakers (CLAUDE.md §12)
# ---------------------------------------------------------------------------

BINANCE_STALE_MS_MAX  = 5000
COINBASE_STALE_MS_MAX = 10000   # Coinbase ticker fires less frequently than Binance L2
POLY_STALE_MS_MAX     = 60000
RTDS_STALE_MS_MAX     = 90000
WIDE_SPREAD_THRESHOLD = 0.10
WALL_CLOCK_SKEW_MS    = 2000
DAILY_LOSS_PCT_HARD   = 0.05               # hard stop (Phase 2)
KILL_SWITCH_PATH      = "/run/polybot/STOP"

# ---------------------------------------------------------------------------
# Operational
# ---------------------------------------------------------------------------

DRY_RUN          = os.getenv("DRY_RUN", "true").lower() == "true"
LOG_DIR          = Path(os.getenv("LOG_DIR", "logs"))
PARQUET_DIR      = Path(os.getenv("PARQUET_DIR", "logs"))
PARQUET_FLUSH_S  = 10                      # flush in-memory rows to disk every 10s


def validate() -> list[str]:
    """Return a list of missing required values (empty list = OK)."""
    missing = []
    for key in ("PRIVATE_KEY", "API_KEY", "API_SECRET", "API_PASSPHRASE"):
        if not globals().get(key):
            missing.append(key)
    return missing
