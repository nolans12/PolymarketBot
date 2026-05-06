"""
demo_rtds.py — Live Polymarket RTDS Chainlink price stream demo.

Shows incoming oracle ticks, K-capture events, and window rollover timing.

Usage:
    python scripts/demo_rtds.py
    python scripts/demo_rtds.py --verbose
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.clients.polymarket_rtds import PolymarketRTDS
from polybot.infra.config import ASSETS
from polybot.state.window import WindowState


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polymarket RTDS Chainlink demo")
    p.add_argument("--verbose", action="store_true", help="Show INFO-level logs")
    return p.parse_args()


async def main() -> None:
    windows: dict[str, WindowState] = {asset: WindowState(asset) for asset in ASSETS}

    def on_k_captured(asset: str, window: WindowState) -> None:
        print(
            f"[K-CAPTURE] {asset.upper():3s}  K={window.K:>12.4f}  "
            f"window_ts={window.open_ts}  tau={window.tau_s():.0f}s remaining  "
            f"uncertain={window.K_uncertain}",
            flush=True,
        )

    # Print every Chainlink tick by monkey-patching _handle_price_tick
    client = PolymarketRTDS(windows=windows, on_k_captured=on_k_captured)

    _orig_handle = client._handle_price_tick

    def _patched_handle(msg: dict) -> None:
        payload = msg.get("payload", {})
        if not isinstance(payload, dict):
            _orig_handle(msg)
            return
        symbol = payload.get("symbol", "?")
        asset_label = symbol.split("/")[0].upper() if "/" in symbol else symbol.upper()
        asset_key = asset_label.lower()

        # Print each tick in the batch (or single live tick)
        ticks = []
        data_arr = payload.get("data")
        if data_arr and isinstance(data_arr, list):
            ticks = [(int(e["timestamp"]), float(e["value"])) for e in data_arr if "timestamp" in e]
        elif payload.get("timestamp") is not None:
            ticks = [(int(payload["timestamp"]), float(payload["value"]))]

        now_ms = int(time.time() * 1000)
        if len(ticks) > 1:
            # Batch on subscribe — just print summary
            first_ts, first_val = ticks[0]
            last_ts, last_val = ticks[-1]
            print(
                f"[BATCH] {asset_label:3s}  {len(ticks)} ticks  "
                f"span={(last_ts-first_ts)/1000:.0f}s  "
                f"latest_price={last_val:>12.4f}  age_ms={now_ms-last_ts:>5d}",
                flush=True,
            )
        else:
            for oracle_ts_ms, price in ticks:
                age_ms = now_ms - oracle_ts_ms
                k_set = windows.get(asset_key, WindowState("")).K is not None
                print(
                    f"[TICK]  {asset_label:3s}  price={price:>12.4f}  "
                    f"oracle_ts={oracle_ts_ms}  age_ms={age_ms:>5d}  K_set={k_set}",
                    flush=True,
                )
        _orig_handle(msg)

    client._handle_price_tick = _patched_handle

    # Print window state on startup
    print(f"{'asset':>5} | {'open_ts':>12} | {'close_ts':>12} | {'tau_s':>7} | K")
    print("-" * 70)

    print("Connecting to RTDS… (Ctrl+C to stop)", file=sys.stderr)
    try:
        await client.run()
    except KeyboardInterrupt:
        client.stop()
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(main())
