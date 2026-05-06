"""
orders.py — Polymarket CLOB order placement (EOA-mode only).

The bot uses an EOA-mode Polymarket account: the EOA derived from PRIVATE_KEY
both signs orders AND holds the USDC collateral. No proxy wallet is involved
(`signature_type=0`).
"""

import logging
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    AssetType,
    BalanceAllowanceParams,
    OrderArgs,
    OrderType,
)
from py_clob_client.constants import POLYGON

# Some SDK versions expose PartialCreateOrderOptions; fall back gracefully.
try:
    from py_clob_client.clob_types import PartialCreateOrderOptions
except ImportError:
    PartialCreateOrderOptions = None  # type: ignore

from polybot.state.wallet import signer_address

logger = logging.getLogger(__name__)


class OrderClient:
    """Thin wrapper around py-clob-client for order lifecycle. Phase 2 only."""

    def __init__(
        self,
        private_key: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        clob_host: str,
        chain_id: int = POLYGON,
        dry_run: bool = False,
    ):
        self.dry_run = dry_run
        self._signer = signer_address(private_key)

        # EOA mode: signature_type=0, funder is the signer's own address.
        self.client = ClobClient(
            host=clob_host,
            key=private_key,
            chain_id=chain_id,
            signature_type=0,
            funder=self._signer,
        )

        if api_key:
            self.client.set_api_creds(
                type("Creds", (), {
                    "api_key":        api_key,
                    "api_secret":     api_secret,
                    "api_passphrase": api_passphrase,
                })()
            )

        logger.info(
            f"OrderClient initialised | dry_run={dry_run} | signer={self._signer}"
        )

    def get_clob_balance(self) -> float:
        """
        CLOB-layer USDC balance the exchange sees for the signer EOA.
        Compare against on-chain USDC via polybot.state.wallet.usdc_balance_onchain.
        """
        if self.dry_run:
            return 1000.0
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            data = self.client.get_balance_allowance(params)
            return float(data.get("balance", 0.0))
        except Exception as e:
            logger.warning(f"Could not fetch CLOB balance: {e} — using 0")
            return 0.0

    def _is_neg_risk(self, token_id: str) -> bool:
        """Query the CLOB to check if this token is part of a neg-risk market."""
        try:
            result = self.client.get_neg_risk(token_id)
            # SDK returns a bool or {"neg_risk": bool} depending on version
            if isinstance(result, dict):
                return bool(result.get("neg_risk", False))
            return bool(result)
        except Exception as exc:
            logger.warning(f"get_neg_risk failed for {token_id[:12]}…: {exc} — assuming False")
            return False

    def place_order(
        self,
        token_id: str,
        side: str,
        size_usd: float,
        price: float,
    ) -> Optional[str]:
        """Place a GTC limit order. Returns order_id or None.

        Auto-detects neg-risk markets (the 5-min crypto Up/Down markets are
        all neg-risk) and signs accordingly. Without neg_risk=True the CLOB
        rejects with 'order_version_mismatch'.
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

            neg_risk = self._is_neg_risk(token_id)
            logger.info(f"placing order neg_risk={neg_risk} token={token_id[:12]}…")

            if PartialCreateOrderOptions is not None:
                options = PartialCreateOrderOptions(neg_risk=neg_risk)
                signed_order = self.client.create_order(order_args, options)
            else:
                signed_order = self.client.create_order(order_args)

            # Diagnostic dump of the order body the CLOB will receive
            try:
                body = signed_order.dict() if hasattr(signed_order, "dict") else dict(signed_order.__dict__)
            except Exception:
                body = repr(signed_order)
            logger.info(f"SIGNED ORDER BODY: {body}")

            resp = self.client.post_order(signed_order, OrderType.GTC)
            order_id = resp.get("orderID") or resp.get("id")
            if not order_id:
                logger.error(f"post_order returned no id; full response: {resp}")
                return None
            logger.info(
                f"ORDER PLACED | token={token_id[:12]}... | side={side} | "
                f"shares={size_shares} | price={price:.4f} | "
                f"neg_risk={neg_risk} | order_id={order_id}"
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
        Derive API key/secret/passphrase from the private key.
        Run once on first setup.
        """
        creds = self.client.create_or_derive_api_creds()
        return {
            "api_key":        creds.api_key,
            "api_secret":     creds.api_secret,
            "api_passphrase": creds.api_passphrase,
        }
