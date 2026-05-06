"""
demo_polymarket_ws.py — Live Polymarket CLOB book demo.

Resolves the current BTC and ETH 5-minute markets via gamma REST,
then streams the Up/Down order book and prints best bid/ask + imbalance.

Usage:
    python scripts/demo_polymarket_ws.py
    python scripts/demo_polymarket_ws.py --hz 2
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.clients.polymarket_rest import fetch_market, current_window_ts
from polybot.clients.polymarket_ws import PolymarketWS
from polybot.infra.config import ASSETS
from polybot.state.poly_book import PolyBook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket CLOB live feed demo")
    p.add_argument("--hz", type=float, default=2.0,
                   help="Max print rate per asset in Hz (default 2)")
    p.add_argument("--verbose", action="store_true",
                   help="Show INFO-level websocket logs")
    return p.parse_args()


async def main(max_hz: float) -> None:
    books: dict[str, PolyBook] = {asset: PolyBook(asset) for asset in ASSETS}
    last_print: dict[str, float] = {asset: 0.0 for asset in ASSETS}
    min_interval = 1.0 / max_hz

    header_printed = False

    def on_update(asset: str, book: PolyBook) -> None:
        nonlocal header_printed
        now = time.time()
        if now - last_print.get(asset, 0.0) < min_interval:
            return
        if not book.ready:
            return
        last_print[asset] = now

        if not header_printed:
            print(
                f"{'asset':>4} | {'up_bid':>6} {'up_ask':>6} {'up_mid':>6} | "
                f"{'dn_bid':>6} {'dn_ask':>6} {'dn_mid':>6} | "
                f"{'imbal':>7} | {'tf30s':>8} | {'stale_ms':>8}"
            )
            print("-" * 100)
            header_printed = True

        up, dn = book.up, book.down
        imbal = book.book_imbalance()
        tf = book.trade_flow_30s()
        stale = book.stale_ms()

        print(
            f"{asset.upper():>4} | "
            f"{up.best_bid:>6.3f} {up.best_ask:>6.3f} {up.mid:>6.3f} | "
            f"{dn.best_bid:>6.3f} {dn.best_ask:>6.3f} {dn.mid:>6.3f} | "
            f"{imbal:>+7.3f} | "
            f"{tf:>+8.2f} | "
            f"{stale:>8d}",
            flush=True,
        )

    # Resolve current window token IDs via gamma REST
    ws_client = PolymarketWS(books=books, on_update=on_update)

    window_ts = current_window_ts()
    print(f"Resolving markets for window_ts={window_ts}…", file=sys.stderr)

    resolved_any = False
    for asset in ASSETS:
        market = fetch_market(asset, window_ts)
        if market:
            up_id = market["up_token_id"]
            dn_id = market["down_token_id"]
            books[asset].set_tokens(up_id, dn_id)
            ws_client.subscribe(asset, up_id, dn_id)
            print(
                f"  {asset.upper()}: slug={market['slug']}  "
                f"up={up_id[:12]}…  down={dn_id[:12]}…",
                file=sys.stderr,
            )
            resolved_any = True
        else:
            print(f"  {asset.upper()}: market not found for window_ts={window_ts}", file=sys.stderr)

    if not resolved_any:
        print("No markets resolved — check gamma API or try the next window.", file=sys.stderr)
        sys.exit(1)

    print("Connecting to Polymarket CLOB WS… (Ctrl+C to stop)", file=sys.stderr)

    try:
        await ws_client.run()
    except KeyboardInterrupt:
        ws_client.stop()
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(main(args.hz))
