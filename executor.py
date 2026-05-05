"""
executor.py — Polymarket CLOB order placement and account management.

Assumes all USDC is already in the Polymarket account.
No on-chain transfers needed — POLYGON_RPC not required.
"""

import logging
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, BalanceAllowanceParams, AssetType
from py_clob_client.constants import POLYGON

logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self,
        private_key: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        clob_host: str,
        chain_id: int = POLYGON,
        dry_run: bool = False,
        funder: str = None,
    ):
        self.dry_run = dry_run
        self.client = ClobClient(
            host=clob_host,
            key=private_key,
            chain_id=chain_id,
            signature_type=2,
            funder=funder or None,
        )

        if api_key:
            self.client.set_api_creds(
                type("Creds", (), {
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "api_passphrase": api_passphrase,
                })()
            )

        logger.info(f"Executor initialised | dry_run={dry_run}")

    def get_wallet_balance(self) -> float:
        """
        Return the current USDC balance available in the Polymarket account.
        Used by the bot to size positions dynamically as a % of wallet.
        Returns 0.0 on failure (bot will use SIZE_MIN as fallback).
        """
        if self.dry_run:
            return 1000.0  # dummy balance for dry runs

        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            data = self.client.get_balance_allowance(params)
            return float(data.get("balance", 0.0))
        except Exception as e:
            logger.warning(f"Could not fetch wallet balance: {e} — using 0")
            return 0.0

    def place_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
        price: float,
    ) -> Optional[str]:
        """
        Place a GTC limit order on the CLOB.

        token_id:  Polymarket YES token ID
        side:      "BUY"
        size_usd:  position size in USD
        price:     limit price (= q^w, current market mid)

        Returns order_id string, or None on failure.
        """
        size_shares = round(size_usd / price, 2)

        if self.dry_run:
            logger.info(
                f"DRY RUN | {side} {size_shares} shares @ {price:.4f} "
                f"(${size_usd:.2f}) token={token_id[:12]}..."
            )
            return "DRY_RUN_ORDER_ID"

        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size_shares,
                side=side,
            )
            resp = self.client.create_and_post_order(order_args)
            order_id = resp.get("orderID") or resp.get("id")
            logger.info(
                f"ORDER PLACED | token={token_id[:12]}... | side={side} | "
                f"shares={size_shares} | price={price:.4f} | order_id={order_id}"
            )
            return order_id
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        if self.dry_run:
            logger.info(f"DRY RUN | would cancel order {order_id}")
            return True
        try:
            self.client.cancel(order_id)
            logger.info(f"ORDER CANCELLED | order_id={order_id}")
            return True
        except Exception as e:
            logger.error(f"Cancel failed for {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[dict]:
        if self.dry_run:
            return {"status": "DRY_RUN"}
        try:
            return self.client.get_order(order_id)
        except Exception as e:
            logger.error(f"get_order failed for {order_id}: {e}")
            return None

    def derive_api_credentials(self) -> dict:
        """
        Derive API key/secret/passphrase from private key.
        Run once on first setup via: python3 test_trade.py --derive-creds
        """
        creds = self.client.create_or_derive_api_creds()
        return {
            "api_key": creds.api_key,
            "api_secret": creds.api_secret,
            "api_passphrase": creds.api_passphrase,
        }
