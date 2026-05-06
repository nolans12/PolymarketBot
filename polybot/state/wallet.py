"""
wallet.py — On-chain USDC balance reader for the funder address.

The bot's PRIVATE_KEY is a Polymarket relayer signing key, which holds zero
USDC. The actual collateral lives at the FUNDER address (the user's
Polymarket account, e.g. their Google/Magic Link wallet). Reading the
funder's USDC balance therefore requires a direct on-chain call rather than
the CLOB's get_balance_allowance — that endpoint only sees the signer.

USDC on Polygon is bridged USDC (USDC.e), 6 decimals.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

USDC_CONTRACT  = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
BALANCE_OF_SIG = "0x70a08231"  # ERC-20 balanceOf(address) selector


def usdc_balance_onchain(address: str, polygon_rpc: str, timeout: float = 10.0) -> Optional[float]:
    """
    Read USDC balance for an address directly from Polygon via public RPC.

    Returns balance in USDC (decimals already applied). Returns None on
    network/parse error so the caller can distinguish "0 balance" from
    "couldn't read balance."
    """
    if not address:
        return None
    try:
        padded = address.lower().replace("0x", "").zfill(64)
        payload = {
            "jsonrpc": "2.0",
            "method":  "eth_call",
            "params":  [
                {"to": USDC_CONTRACT, "data": BALANCE_OF_SIG + padded},
                "latest",
            ],
            "id": 1,
        }
        resp = requests.post(polygon_rpc, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("result", "0x0")
        return int(raw, 16) / 1e6
    except Exception as e:
        logger.warning(f"On-chain USDC balance fetch failed for {address}: {e}")
        return None
