"""
check_balance.py — Verify wallet balance before going live.

EOA-mode only: the address derived from PRIVATE_KEY both signs orders and
holds USDC. Checks:
  1. On-chain USDC at the signer EOA (Polygon RPC)
  2. CLOB-layer balance the exchange sees
  3. API credential validity

Usage:
    python scripts/check_balance.py
"""

import os
import sys
from pathlib import Path

# Allow running this script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

PRIVATE_KEY    = os.getenv("PRIVATE_KEY", "")
API_KEY        = os.getenv("API_KEY", "")
API_SECRET     = os.getenv("API_SECRET", "")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "")
CLOB_HOST      = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID       = int(os.getenv("CHAIN_ID", "137"))
POLYGON_RPC    = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

from polybot.state.wallet import signer_address, usdc_balance_onchain

signer = signer_address(PRIVATE_KEY)

print("=" * 60)
print("PolymarketBot — Pre-flight balance check")
print("=" * 60)

if not signer:
    print("\nERROR: PRIVATE_KEY not set in .env")
    sys.exit(1)

print(f"\nSigner / funder EOA: {signer}")

# 1. On-chain USDC
print(f"\n1. On-chain USDC at {signer}:")
bal_onchain = usdc_balance_onchain(signer, POLYGON_RPC)
if bal_onchain is not None:
    flag = "✓" if bal_onchain > 0 else "⚠ ZERO — deposit USDC to this address first"
    print(f"   ${bal_onchain:.2f} USDC  {flag}")
else:
    print("   ERROR: could not read on-chain balance (check POLYGON_RPC)")

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
            signature_type=0,
            funder=signer,
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
        if clob_bal == 0 and bal_onchain and bal_onchain > 0:
            print("   ⚠ On-chain shows USDC but CLOB sees $0 — your account may need")
            print("     to approve USDC spending. Place ONE small manual trade on")
            print("     polymarket.com to trigger the approval, then re-run this check.")
    except Exception as e:
        print(f"   ERROR: {e}")
        print("   Check API_KEY/SECRET/PASSPHRASE; re-run derive_creds.py if needed.")

print("\n" + "=" * 60)
print("If both balances look good, set DRY_RUN=false in .env and start the bot.")
print("=" * 60)
