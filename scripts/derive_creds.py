"""
derive_creds.py — One-time script to derive Polymarket CLOB API credentials
from your wallet private key (EOA-mode only).

Run this ONCE before starting the live bot. Prints API_KEY/SECRET/PASSPHRASE
for you to paste into .env.

Usage:
    python scripts/derive_creds.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
CLOB_HOST   = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID    = int(os.getenv("CHAIN_ID", "137"))

if not PRIVATE_KEY or PRIVATE_KEY == "0xYOUR_PRIVATE_KEY_HERE":
    print("ERROR: PRIVATE_KEY not set in .env")
    sys.exit(1)

try:
    from py_clob_client.client import ClobClient
except ImportError:
    print("ERROR: py-clob-client not installed. Run: pip install -e '.[dev]'")
    sys.exit(1)

from polybot.state.wallet import signer_address

signer = signer_address(PRIVATE_KEY)
print(f"Connecting to {CLOB_HOST} (chain={CHAIN_ID})")
print(f"Signer EOA: {signer}")
print()

try:
    client = ClobClient(
        host=CLOB_HOST,
        key=PRIVATE_KEY,
        chain_id=CHAIN_ID,
        signature_type=0,
        funder=signer,
    )
    creds = client.create_or_derive_api_creds()

    print("=" * 60)
    print("Add these to your .env file on the VM:")
    print("=" * 60)
    print(f"API_KEY={creds.api_key}")
    print(f"API_SECRET={creds.api_secret}")
    print(f"API_PASSPHRASE={creds.api_passphrase}")
    print("=" * 60)
    print()
    print("Then verify with:")
    print("  python scripts/check_balance.py")

except Exception as e:
    print(f"ERROR deriving credentials: {e}")
    print()
    print("Common causes:")
    print("  - PRIVATE_KEY is wrong or not a valid Polygon wallet key")
    print("  - The signer EOA isn't registered with Polymarket. Visit polymarket.com,")
    print("    log in with this MetaMask account at least once, then re-run.")
    print("  - Network issue reaching the CLOB host")
    sys.exit(1)
