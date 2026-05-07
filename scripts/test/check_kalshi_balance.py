"""
check_kalshi_balance.py — Verify Kalshi account access and read balance.

Kalshi v2 API uses RSA-PSS HTTP signatures:
  - X-Access-Key:        the API key ID (UUID-like string)
  - X-Timestamp:         current epoch in milliseconds
  - X-Signature:         RSA-PSS(SHA-256, MGF1) over `timestamp + method + path`,
                         base64-encoded

Required .env entries:
    KALSHI_API_KEY_ID=...
    KALSHI_PRIVATE_KEY_FILE=path/to/kalshi_rsa.pem    (RSA PEM, no passphrase)
  - or -
    KALSHI_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----\n..."

Usage:
    python scripts/check_kalshi_balance.py
    python scripts/check_kalshi_balance.py --demo                # demo env
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


PROD_HOST = "https://api.elections.kalshi.com"
DEMO_HOST = "https://demo-api.kalshi.co"

API_PREFIX = "/trade-api/v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true",
                   help="Use Kalshi demo environment instead of production.")
    return p.parse_args()


def load_private_key():
    """Load the RSA private key from KALSHI_PRIVATE_KEY_FILE or _PEM."""
    pem_inline = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    pem_file   = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")

    if pem_inline:
        pem_bytes = pem_inline.encode("utf-8").replace(b"\\n", b"\n")
    elif pem_file:
        path = Path(pem_file).expanduser()
        if not path.exists():
            sys.exit(f"ERROR: KALSHI_PRIVATE_KEY_FILE points to missing file: {path}")
        pem_bytes = path.read_bytes()
    else:
        sys.exit(
            "ERROR: set KALSHI_PRIVATE_KEY_FILE (path to PEM) or "
            "KALSHI_PRIVATE_KEY_PEM (inline) in .env"
        )

    return serialization.load_pem_private_key(pem_bytes, password=None)


def sign_request(private_key, method: str, path: str) -> tuple[str, str]:
    """Returns (timestamp_str, base64_signature). Path includes /trade-api/v2/..."""
    ts_ms = str(int(time.time() * 1000))
    msg   = (ts_ms + method.upper() + path).encode("utf-8")
    sig   = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return ts_ms, base64.b64encode(sig).decode("ascii")


def kalshi_get(host: str, path: str, key_id: str, private_key,
               params: dict | None = None) -> dict:
    """
    Kalshi signs the PATH ONLY, never the query string. Pass query args
    via `params` so they end up on the URL but stay out of the signature.
    """
    full_path = API_PREFIX + path
    ts, sig = sign_request(private_key, "GET", full_path)
    headers = {
        "Accept":        "application/json",
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }
    r = requests.get(host + full_path, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def main() -> None:
    args = parse_args()

    key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    if not key_id:
        sys.exit("ERROR: KALSHI_API_KEY_ID not set in .env")

    host = DEMO_HOST if args.demo else PROD_HOST
    private_key = load_private_key()

    print("=" * 60)
    print(f"Kalshi pre-flight balance check  [{'DEMO' if args.demo else 'PROD'}]")
    print(f"Host:    {host}")
    print(f"Key ID:  {key_id[:8]}…{key_id[-4:]}")
    print("=" * 60)

    # 1. Balance endpoint — also doubles as auth check
    print("\n1. Account balance:")
    try:
        data = kalshi_get(host, "/portfolio/balance", key_id, private_key)
    except Exception as e:
        sys.exit(f"   FAILED: {e}\n   Check key id, PEM file, and clock skew.")

    # Response shape: {"balance": <int cents>}
    bal_cents = int(data.get("balance", 0))
    bal_usd   = bal_cents / 100.0
    flag = "✓" if bal_usd > 0 else "⚠ ZERO — deposit USD via Kalshi UI first"
    print(f"   ${bal_usd:,.2f}   {flag}")

    # 2. Open positions (sanity — should be 0 on a fresh account)
    print("\n2. Open positions:")
    try:
        pos = kalshi_get(host, "/portfolio/positions", key_id, private_key)
        market_pos = pos.get("market_positions", []) or []
        event_pos  = pos.get("event_positions", []) or []
        if not market_pos and not event_pos:
            print("   (none)")
        else:
            for p in market_pos:
                print(f"   - {p.get('ticker'):<35}  qty={p.get('position'):>5}  "
                      f"avg=${p.get('market_exposure', 0)/100:.2f}")
    except Exception as e:
        print(f"   WARN: {e}")

    # 3. Resting orders
    print("\n3. Resting orders:")
    try:
        orders = kalshi_get(host, "/portfolio/orders", key_id, private_key,
                            params={"status": "resting"})
        rows = orders.get("orders", []) or []
        if not rows:
            print("   (none)")
        else:
            for o in rows:
                print(f"   - {o.get('ticker'):<35}  side={o.get('side')}  "
                      f"qty={o.get('remaining_count')}  px={o.get('yes_price')}")
    except Exception as e:
        print(f"   WARN: {e}")

    print("\n" + "=" * 60)
    print("If balance > 0 you're ready to scan markets and stream odds.")
    print("Next: python scripts/run_kalshi_bot.py --fresh")
    print("=" * 60)


if __name__ == "__main__":
    main()
