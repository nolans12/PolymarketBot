"""
config.py — All model parameters for the Dual-Mode HFT Bot (GABIGOL).

API credentials are loaded from .env (never hardcoded here).
Model parameters live here and are version-controlled.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials (from .env — never hardcode these)
# ---------------------------------------------------------------------------
PRIVATE_KEY    = os.getenv("PRIVATE_KEY", "")
API_KEY        = os.getenv("API_KEY", "")
API_SECRET     = os.getenv("API_SECRET", "")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "")
CLOB_HOST      = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID       = int(os.getenv("CHAIN_ID", "137"))

# ---------------------------------------------------------------------------
# Dual-Mode model parameters (Section 8.1 of the paper)
# ---------------------------------------------------------------------------

# Entry filter — both modes must satisfy:
#   p_jj >= P_JJ_MIN   (Markov self-transition persistence threshold)
#   Δ^(w) >= EPSILON   (minimum edge = p_mine - q_market)
P_JJ_MIN = 0.87

# Edge threshold — phased rollout:
#   Phase 1 (conservative): EPSILON = 0.04
#   Phase 2 (full model):   EPSILON = 0.02
EPSILON = float(os.getenv("EPSILON", "0.04"))

# Mode 1 — High-Confidence Scalping: q^(w) ∈ [MODE1_LOW, MODE1_HIGH]
MODE1_LOW  = 0.60
MODE1_HIGH = 0.97

# Mode 2 — Extreme Discount Entry: q^(w) ∈ [MODE2_LOW, MODE2_HIGH)
MODE2_LOW  = 0.087
MODE2_HIGH = 0.60

# ---------------------------------------------------------------------------
# Markov estimator parameters (Section 8.1.1)
# ---------------------------------------------------------------------------

MARKOV_WINDOW_SECONDS = 1800   # 30-minute rolling lookback

MARKOV_BIN_EDGES = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.90, 0.95, 1.00
]

MARKOV_MIN_OBSERVATIONS = 10

# ---------------------------------------------------------------------------
# Dynamic Kelly position sizing
#
# Position size = Kelly_fraction(delta) * wallet_balance
# Kelly fraction scales with edge (delta = p_mine - q):
#
#   delta >= 0.30  →  MAX_BET_PCT  (10% of wallet) — very strong edge
#   delta >= 0.15  →  8% of wallet
#   delta >= 0.08  →  5% of wallet
#   delta >= 0.04  →  3% of wallet  (Phase 1 minimum entry)
#   delta >= 0.02  →  1.5% of wallet (Phase 2 minimum entry)
#
# Hard floor and ceiling in USD regardless of wallet size.
# ---------------------------------------------------------------------------

# Tiered Kelly fractions: list of (min_delta, wallet_fraction)
# Evaluated top-to-bottom — first match wins.
KELLY_TIERS = [
    (0.30, 0.10),   # delta >= 0.30 → 10% of wallet
    (0.15, 0.08),   # delta >= 0.15 → 8%
    (0.08, 0.05),   # delta >= 0.08 → 5%
    (0.04, 0.03),   # delta >= 0.04 → 3%
    (0.02, 0.015),  # delta >= 0.02 → 1.5%
]

# Hard position size bounds in USD (overrides Kelly if outside range)
SIZE_MIN = 9.0
SIZE_MAX = 800.0

# ---------------------------------------------------------------------------
# Bot operational parameters
# ---------------------------------------------------------------------------

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
DRY_RUN       = os.getenv("DRY_RUN", "false").lower() == "true"

ASSETS = [
    "BTC-UP",
    "BTC-DOWN",
    "ETH-UP",
    "ETH-DOWN",
    "SOL-UP",
    "SOL-DOWN",
    "XRP-UP",
    "XRP-DOWN",
]

LOG_FILE = os.getenv("LOG_FILE", "logs/bot.log")
