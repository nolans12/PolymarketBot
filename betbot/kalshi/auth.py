"""
auth.py - Kalshi API authentication (RSA-PSS request signing).

Shared by the REST feed, test_trade, and any other Kalshi REST caller.
There is no WebSocket auth in this codebase; the Kalshi WS path is no longer
used (REST polling proved more reliable than the WS heartbeat protocol).
"""

import base64
import os
import sys
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def load_private_key():
    """Load the Kalshi RSA private key from .env (file or inline PEM)."""
    pem_inline = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    pem_file   = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
    if pem_inline:
        pem = pem_inline.encode().replace(b"\\n", b"\n")
    elif pem_file:
        p = Path(pem_file).expanduser()
        if not p.exists():
            sys.exit(f"ERROR: KALSHI_PRIVATE_KEY_FILE missing: {p}")
        pem = p.read_bytes()
    else:
        sys.exit("ERROR: set KALSHI_PRIVATE_KEY_FILE or KALSHI_PRIVATE_KEY_PEM in .env")
    return serialization.load_pem_private_key(pem, password=None)


def sign(pk, ts: str, method: str, path: str) -> str:
    msg = (ts + method.upper() + path).encode()
    sig = pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def auth_headers(pk, key_id: str, method: str, path: str) -> dict:
    """Build the three KALSHI-ACCESS-* headers for a REST request."""
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sign(pk, ts, method, path),
    }
