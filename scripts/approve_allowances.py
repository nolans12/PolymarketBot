"""
approve_allowances.py — One-time on-chain approvals so the Polymarket CLOB
can move USDC and conditional tokens on your behalf.

Polymarket's UI does this automatically on your first trade, but the UI is
geoblocked in many regions. This script performs the same approvals directly
via signed transactions through your POLYGON_RPC.

Approvals granted (all from your signer EOA):
  1. USDC → Exchange (CTF Exchange contract)
  2. USDC → Neg-Risk Adapter (used by 5-min crypto markets)
  3. USDC → Neg-Risk CTF Exchange
  4. Conditional Tokens → Exchange (setApprovalForAll)
  5. Conditional Tokens → Neg-Risk Adapter
  6. Conditional Tokens → Neg-Risk CTF Exchange

Cost: ~$0.05 in MATIC gas total (Polygon is cheap).

Usage:
    python scripts/approve_allowances.py             # check current allowances
    python scripts/approve_allowances.py --execute   # send the approval txs
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from web3 import Web3
from eth_account import Account


PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
POLYGON_RPC = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

# Polymarket contract addresses on Polygon mainnet
USDC_E             = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_NATIVE        = Web3.to_checksum_address("0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359")
CONDITIONAL_TOKENS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
EXCHANGE           = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
NEG_RISK_ADAPTER   = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")
NEG_RISK_EXCHANGE  = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")

# ERC-20 ABI (just approve + allowance)
ERC20_ABI = [
    {"name": "approve", "type": "function", "inputs": [
        {"name": "spender", "type": "address"},
        {"name": "amount",  "type": "uint256"}],
     "outputs": [{"type": "bool"}], "stateMutability": "nonpayable"},
    {"name": "allowance", "type": "function", "inputs": [
        {"name": "owner",   "type": "address"},
        {"name": "spender", "type": "address"}],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
]

# ERC-1155 ABI (setApprovalForAll + isApprovedForAll)
ERC1155_ABI = [
    {"name": "setApprovalForAll", "type": "function", "inputs": [
        {"name": "operator", "type": "address"},
        {"name": "approved", "type": "bool"}],
     "outputs": [], "stateMutability": "nonpayable"},
    {"name": "isApprovedForAll", "type": "function", "inputs": [
        {"name": "owner",    "type": "address"},
        {"name": "operator", "type": "address"}],
     "outputs": [{"type": "bool"}], "stateMutability": "view"},
]

MAX_UINT256 = 2**256 - 1


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true",
                   help="Send the approval transactions (default: check only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not PRIVATE_KEY:
        fail("PRIVATE_KEY not set in .env")

    pk = PRIVATE_KEY if PRIVATE_KEY.startswith("0x") else "0x" + PRIVATE_KEY
    acct = Account.from_key(pk)
    signer = acct.address
    print(f"Signer: {signer}")

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC, request_kwargs={"timeout": 15}))
    if not w3.is_connected():
        fail(f"Could not connect to {POLYGON_RPC}")

    chain_id = w3.eth.chain_id
    print(f"Connected to chain_id={chain_id}")
    if chain_id != 137:
        fail(f"Expected Polygon mainnet (137), got chain_id={chain_id}")

    matic = w3.eth.get_balance(signer) / 1e18
    print(f"MATIC balance: {matic:.4f}  {'✓' if matic >= 0.1 else '⚠ low — need ~0.1 MATIC for gas'}")

    print()
    print("=" * 70)
    print("Current allowances:")
    print("=" * 70)

    # ERC-20 USDC allowances
    erc20_targets = [
        ("USDC.e  → Exchange",          USDC_E,      EXCHANGE),
        ("USDC.e  → NegRiskAdapter",    USDC_E,      NEG_RISK_ADAPTER),
        ("USDC.e  → NegRiskExchange",   USDC_E,      NEG_RISK_EXCHANGE),
        ("USDC    → Exchange",          USDC_NATIVE, EXCHANGE),
        ("USDC    → NegRiskAdapter",    USDC_NATIVE, NEG_RISK_ADAPTER),
        ("USDC    → NegRiskExchange",   USDC_NATIVE, NEG_RISK_EXCHANGE),
    ]
    erc1155_targets = [
        ("CTF     → Exchange",          CONDITIONAL_TOKENS, EXCHANGE),
        ("CTF     → NegRiskAdapter",    CONDITIONAL_TOKENS, NEG_RISK_ADAPTER),
        ("CTF     → NegRiskExchange",   CONDITIONAL_TOKENS, NEG_RISK_EXCHANGE),
    ]

    needs_approval = []  # list of (label, contract_addr, abi, fn_name, args)

    for label, token, spender in erc20_targets:
        c = w3.eth.contract(address=token, abi=ERC20_ABI)
        cur = c.functions.allowance(signer, spender).call()
        ok = cur >= 10**24  # any "huge" value is functionally unlimited
        print(f"  {label:30}  {'✓' if ok else '✗'}  cur={cur}")
        if not ok:
            needs_approval.append((label, token, ERC20_ABI, "approve", [spender, MAX_UINT256]))

    for label, token, spender in erc1155_targets:
        c = w3.eth.contract(address=token, abi=ERC1155_ABI)
        ok = bool(c.functions.isApprovedForAll(signer, spender).call())
        print(f"  {label:30}  {'✓' if ok else '✗'}")
        if not ok:
            needs_approval.append((label, token, ERC1155_ABI, "setApprovalForAll", [spender, True]))

    print()
    if not needs_approval:
        print("All approvals already granted. Re-run check_balance.py — your CLOB")
        print("balance should now reflect your deposit.")
        return

    print(f"{len(needs_approval)} approvals missing.")
    if not args.execute:
        print()
        print("To send the approvals, re-run with --execute:")
        print("  python scripts/approve_allowances.py --execute")
        print()
        print(f"Estimated gas: ~{0.005 * len(needs_approval):.3f} MATIC total (~$0.01)")
        return

    if matic < 0.05:
        fail(f"Not enough MATIC for gas: have {matic:.4f}, need ~0.05. Send some MATIC to {signer}.")

    print()
    print("=" * 70)
    print(f"Sending {len(needs_approval)} approval transactions…")
    print("=" * 70)

    nonce = w3.eth.get_transaction_count(signer)
    gas_price = w3.eth.gas_price

    for label, addr, abi, fn_name, fn_args in needs_approval:
        c = w3.eth.contract(address=addr, abi=abi)
        fn = getattr(c.functions, fn_name)(*fn_args)
        try:
            estimated_gas = fn.estimate_gas({"from": signer})
        except Exception:
            estimated_gas = 100_000
        tx = fn.build_transaction({
            "from":     signer,
            "nonce":    nonce,
            "gas":      int(estimated_gas * 1.2),
            "gasPrice": gas_price,
            "chainId":  137,
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print(f"  {label}: tx={tx_hash.hex()}")
        nonce += 1

    print()
    print("All transactions submitted. Waiting for confirmations…")
    # We sent them sequentially with incrementing nonces; the last one's receipt
    # implies all earlier ones confirmed too.
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status != 1:
        fail(f"Last tx failed: {tx_hash.hex()}")

    print(f"Last tx confirmed in block {receipt.blockNumber}")
    print()
    print("Re-run python scripts/check_balance.py — CLOB balance should now show")
    print("your deposit (may take 30-60s for Polymarket to index the on-chain state).")


if __name__ == "__main__":
    main()
