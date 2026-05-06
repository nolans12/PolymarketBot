"""
wallet.py — On-chain USDC balance reader and signer-address helper.

The bot uses an EOA-mode Polymarket account: the address derived from
PRIVATE_KEY both signs orders AND holds the USDC collateral. There is no
separate "funder" / proxy wallet.

Two USDC contracts on Polygon:
  - USDC.e (bridged, legacy):  0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
  - USDC   (native, current):  0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359
We sum across both because deposits from different sources can land in
either.
"""

import logging
from typing import Optional

import requests
from eth_account import Account

logger = logging.getLogger(__name__)

USDC_E_CONTRACT  = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # bridged (legacy)
USDC_N_CONTRACT  = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # native (current)
BALANCE_OF_SIG   = "0x70a08231"  # ERC-20 balanceOf(address) selector


def _balance_one(address: str, contract: str, polygon_rpc: str,
                 timeout: float) -> Optional[float]:
    padded = address.lower().replace("0x", "").zfill(64)
    payload = {
        "jsonrpc": "2.0",
        "method":  "eth_call",
        "params":  [
            {"to": contract, "data": BALANCE_OF_SIG + padded},
            "latest",
        ],
        "id": 1,
    }
    try:
        resp = requests.post(polygon_rpc, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json().get("result", "0x0")
        return int(raw, 16) / 1e6
    except Exception as e:
        logger.warning(f"USDC balance fetch failed contract={contract[:10]}… err={e}")
        return None


def signer_address(private_key: str) -> str:
    """Return the EOA address derived from a private key (the signer/funder)."""
    if not private_key:
        return ""
    pk = private_key if private_key.startswith("0x") else "0x" + private_key
    return Account.from_key(pk).address


def usdc_balance_onchain(address: str, polygon_rpc: str,
                         timeout: float = 10.0) -> Optional[float]:
    """
    Read USDC balance for an address directly from Polygon via public RPC.

    Sums balances across both the legacy bridged USDC.e and the new native
    USDC contracts. Returns balance in USDC (decimals already applied).
    Returns None if BOTH RPC calls failed (so the caller can distinguish
    "0 balance" from "couldn't read balance").
    """
    if not address:
        return None
    bal_e = _balance_one(address, USDC_E_CONTRACT, polygon_rpc, timeout)
    bal_n = _balance_one(address, USDC_N_CONTRACT, polygon_rpc, timeout)
    if bal_e is None and bal_n is None:
        return None
    total = (bal_e or 0.0) + (bal_n or 0.0)
    if bal_e and bal_n:
        logger.info(f"USDC balance for {address[:10]}…: bridged=${bal_e:.2f} native=${bal_n:.2f} total=${total:.2f}")
    return total
