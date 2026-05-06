"""
derive_creds.py — One-time script to derive Polymarket CLOB API credentials
from your wallet private key.

Run this ONCE on your VM before starting the live bot. It prints the three
values (API_KEY, API_SECRET, API_PASSPHRASE) to add to your .env file.

Usage:
    python scripts/derive_creds.py

Requires PRIVATE_KEY and FUNDER in your .env (or as env vars).
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
FUNDER      = os.getenv("FUNDER", "")
CLOB_HOST   = os.getenv("CLOB_HOST", "https://clob.polymarket.com")
CHAIN_ID    = int(os.getenv("CHAIN_ID", "137"))

if not PRIVATE_KEY or PRIVATE_KEY == "0xYOUR_PRIVATE_KEY_HERE":
    print("ERROR: PRIVATE_KEY not set in .env")
    sys.exit(1)

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.constants import POLYGON
except ImportError:
    print("ERROR: py-clob-client not installed. Run: pip install -e '.[dev]'")
    sys.exit(1)

print(f"Connecting to {CLOB_HOST} (chain={CHAIN_ID})...")
print(f"Funder: {FUNDER or '(using signer as funder)'}")
print()

try:
    client = ClobClient(
        host=CLOB_HOST,
        key=PRIVATE_KEY,
        chain_id=CHAIN_ID,
        signature_type=2 if FUNDER else 0,
        funder=FUNDER or None,
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
    print("  - FUNDER address doesn't match the private key's associated account")
    print("  - Network issue reaching the CLOB host")
    sys.exit(1)
