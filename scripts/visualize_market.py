"""
visualize_market.py — Collect 3 minutes of live data, then plot statically.

Runs both feeds simultaneously for COLLECT_S seconds, then renders a
two-axis static chart showing Binance BTC microprice vs Polymarket Up
implied probability over time. Any visible lag between the lines is the
arbitrage edge the regression exploits.

Usage:
    python scripts/visualize_market.py
    python scripts/visualize_market.py --collect 180
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.clients.binance_ws import BinanceWS
from polybot.clients.polymarket_ws import PolymarketWS
from polybot.clients.polymarket_rest import fetch_market, current_window_ts
from polybot.state.spot_book import SpotBook
from polybot.state.poly_book import PolyBook
from polybot.infra.config import ASSETS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", type=int, default=180,
                   help="Seconds to collect data before plotting (default 180)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

async def collect(collect_s: int) -> tuple[list, list, list, list, str]:
    """
    Returns (b_times, b_prices, p_times, p_prices, slug).
    Uses BinanceWS and PolymarketWS exactly as the demo scripts do.
    """
    b_times:  list[float] = []
    b_prices: list[float] = []
    p_times:  list[float] = []
    p_prices: list[float] = []
    slug = "unknown"

    spot_books = {a: SpotBook(a) for a in ASSETS}
    poly_books = {a: PolyBook(a) for a in ASSETS}

    last_b_t = 0.0

    def on_binance(asset: str, book: SpotBook) -> None:
        nonlocal last_b_t
        if asset != "btc" or not book.ready:
            return
        t = time.time()
        if t - last_b_t < 0.2:   # cap at ~5 Hz
            return
        last_b_t = t
        b_times.append(t)
        b_prices.append(book.microprice)

    def on_poly(asset: str, book: PolyBook) -> None:
        if asset != "btc" or not book.ready:
            return
        up = book.up
        if up and 0 < up.best_bid and up.best_ask < 1:
            p_times.append(time.time())
            p_prices.append(up.mid)

    binance_ws = BinanceWS(books=spot_books, on_update=on_binance)
    poly_ws    = PolymarketWS(books=poly_books, on_update=on_poly)

    # Resolve current BTC market for Polymarket
    window_ts = current_window_ts()
    market = fetch_market("btc", window_ts)
    if market:
        slug = market["slug"]
        poly_books["btc"].set_tokens(market["up_token_id"], market["down_token_id"])
        poly_ws.subscribe("btc", market["up_token_id"], market["down_token_id"])
        print(f"  Market: {slug}", flush=True)
    else:
        print("  WARNING: BTC market not found — Polymarket feed will be empty", flush=True)

    print(f"  Collecting for {collect_s}s…", flush=True)

    start = time.time()
    stop_flag = asyncio.Event()

    async def _timer():
        await asyncio.sleep(collect_s)
        stop_flag.set()
        binance_ws.stop()
        poly_ws.stop()

    async def _progress():
        while not stop_flag.is_set():
            elapsed = time.time() - start
            b_last = f"{b_prices[-1]:,.2f}" if b_prices else "—"
            p_last = f"{p_prices[-1]:.4f}" if p_prices else "—"
            print(
                f"\r  t={elapsed:4.0f}s  "
                f"Binance: {b_last} ({len(b_prices)} pts)  "
                f"Polymarket Up: {p_last} ({len(p_prices)} pts)  "
                f"[{collect_s - elapsed:.0f}s left]   ",
                end="", flush=True,
            )
            await asyncio.sleep(1.0)

    # Run feeds; cancel all tasks once the timer fires.
    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(binance_ws.run()),
        loop.create_task(poly_ws.run()),
        loop.create_task(_progress()),
    ]

    await _timer()   # blocks for collect_s seconds, then sets stop_flag + calls .stop()

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print()
    return b_times, b_prices, p_times, p_prices, slug


# ---------------------------------------------------------------------------
# Static plot
# ---------------------------------------------------------------------------

def plot(b_times, b_prices, p_times, p_prices, slug, collect_s):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    if not b_times:
        print("No Binance data collected — cannot plot.", file=sys.stderr)
        return
    if not p_times:
        print("No Polymarket data collected — plotting Binance only.", file=sys.stderr)

    # Convert unix timestamps to datetime
    b_dt = [datetime.fromtimestamp(t) for t in b_times]
    p_dt = [datetime.fromtimestamp(t) for t in p_times]

    fig, ax_b = plt.subplots(figsize=(15, 6))
    ax_p = ax_b.twinx()

    fig.patch.set_facecolor("#111111")
    ax_b.set_facecolor("#111111")
    for ax in (ax_b, ax_p):
        ax.tick_params(colors="#bbbbbb", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # Plot Binance microprice
    ax_b.plot(b_dt, b_prices, color="#4da6ff", linewidth=1.4,
              label=f"Binance microprice  ({len(b_prices)} pts)", zorder=3)

    # Plot Polymarket Up mid as a step-style line (prices only change on events)
    if p_times:
        ax_p.step(p_dt, p_prices, color="#ff8c42", linewidth=1.6,
                  where="post", label=f"Polymarket Up mid  ({len(p_prices)} pts)",
                  zorder=4)
        ax_p.set_ylim(
            max(0.0, min(p_prices) - 0.05),
            min(1.0, max(p_prices) + 0.05),
        )

    # Labels
    ax_b.set_ylabel("Binance BTC microprice  (USD)", color="#4da6ff", fontsize=10)
    ax_p.set_ylabel("Polymarket Up implied prob  (0–1)", color="#ff8c42", fontsize=10)
    ax_b.set_xlabel("Time (UTC)", color="#aaaaaa", fontsize=9)

    ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax_b.xaxis.get_majorticklabels(), rotation=25, ha="right")
    ax_b.grid(axis="y", color="#222222", linewidth=0.5, linestyle="--")

    # Legend combining both axes
    lines_b, labels_b = ax_b.get_legend_handles_labels()
    lines_p, labels_p = ax_p.get_legend_handles_labels()
    ax_b.legend(lines_b + lines_p, labels_b + labels_p,
                loc="upper left", facecolor="#1c1c1c",
                edgecolor="#444444", labelcolor="#cccccc", fontsize=9)

    duration = b_times[-1] - b_times[0] if len(b_times) >= 2 else 0
    ax_b.set_title(
        f"{slug}  —  {collect_s}s collection  "
        f"({duration:.0f}s span)\n"
        "Look for orange line trailing blue at inflection points → that delay is the lag",
        color="#dddddd", fontsize=10, pad=10,
    )

    fig.tight_layout(pad=1.5)

    # Annotation: mark the first inflection where lag is visible
    # (just a text hint, not auto-detected)
    fig.text(0.5, 0.002,
             "Tip: zoom in on a sharp Binance move and see how many seconds before "
             "Polymarket (orange) catches up",
             ha="center", color="#666666", fontsize=8)

    outfile = Path("lag_plot.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"\nSaved to: {outfile.resolve()}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"=== BTC Lag Visualizer ===")
    print(f"Collecting {args.collect}s of live data from Binance + Polymarket…")

    try:
        b_times, b_prices, p_times, p_prices, slug = asyncio.run(
            collect(args.collect)
        )
    except KeyboardInterrupt:
        print("\nStopped early — plotting what we have…")
        # asyncio.run won't return partial data on KeyboardInterrupt
        # so we can't recover; just exit
        sys.exit(0)

    print(f"\nCollected: Binance {len(b_prices)} pts, Polymarket {len(p_prices)} pts")
    print("Rendering plot…")

    plot(b_times, b_prices, p_times, p_prices, slug, args.collect)


if __name__ == "__main__":
    main()
