"""
check_balance.py — Verify wallet balance before going live.

Checks:
  1. On-chain USDC balance at your FUNDER address (Polygon RPC)
  2. CLOB layer balance (what the exchange sees)
  3. API credential validity

Usage:
    python scripts/check_balance.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

PRIVATE_KEY    = os.getenv("PRIVATE_KEY", "")
API_KEY        = os.getenv("API_KEY", "")
API_SECRET     = os.getenv("API_SECRET", "")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "")
CLOB_HOST      = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID       = int(os.getenv("CHAIN_ID", "137"))
FUNDER         = os.getenv("FUNDER", "")
POLYGON_RPC    = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

print("=" * 60)
print("PolymarketBot — Pre-flight balance check")
print("=" * 60)

# 1. On-chain USDC
print(f"\n1. On-chain USDC at funder {FUNDER or '(not set)'}:")
if FUNDER:
    from polybot.state.wallet import usdc_balance_onchain
    bal = usdc_balance_onchain(FUNDER, POLYGON_RPC)
    if bal is not None:
        print(f"   ${bal:.2f} USDC  {'✓' if bal > 0 else '⚠ ZERO — fund your wallet first'}")
    else:
        print("   ERROR: could not read on-chain balance (check POLYGON_RPC)")
else:
    print("   SKIPPED — FUNDER not set in .env")

# 2. CLOB balance + API credentials
print(f"\n2. CLOB API credentials + balance:")
if not API_KEY or API_KEY == "YOUR_API_KEY":
    print("   SKIPPED — API_KEY not set. Run: python scripts/derive_creds.py")
else:
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        client = ClobClient(
            host=CLOB_HOST,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=2 if FUNDER else 0,
            funder=FUNDER or None,
        )
        client.set_api_creds(
            type("C", (), {
                "api_key": API_KEY,
                "api_secret": API_SECRET,
                "api_passphrase": API_PASSPHRASE,
            })()
        )
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        data   = client.get_balance_allowance(params)
        clob_bal = float(data.get("balance", 0.0))
        print(f"   API credentials: ✓ valid")
        print(f"   CLOB balance: ${clob_bal:.2f} USDC")
        if clob_bal == 0:
            print("   ⚠ CLOB balance is 0. You may need to deposit USDC to the CLOB.")
            print("   See: https://docs.polymarket.com/")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Check that API_KEY/SECRET/PASSPHRASE are correct and try re-running")
        print("   derive_creds.py if needed.")

print("\n" + "=" * 60)
print("If both balances look good, set DRY_RUN=false in .env and start the bot.")
print("=" * 60)
