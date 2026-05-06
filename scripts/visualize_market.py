"""
visualize_market.py — Live side-by-side plot of Binance BTC microprice vs
Polymarket BTC Up implied probability.

Two y-axes on one chart:
  Left axis  (blue)  — Binance BTC microprice in USD
  Right axis (orange) — Polymarket Up-share mid price (0-1 = implied prob)

The lag we're looking for: Binance moves first, Polymarket follows
seconds later. You should see the orange line trailing the blue line
at price inflection points.

Usage:
    python scripts/visualize_market.py
    python scripts/visualize_market.py --window 120   # show last 120s (default 180)
    python scripts/visualize_market.py --hz 4          # plot refresh rate (default 4)
"""

import argparse
import asyncio
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # works headless-ish on Windows; fallback below
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as mticker
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from polybot.clients.binance_ws import BinanceWS
from polybot.clients.polymarket_ws import PolymarketWS
from polybot.clients.polymarket_rest import fetch_market, current_window_ts
from polybot.infra.config import ASSETS
from polybot.state.spot_book import SpotBook
from polybot.state.poly_book import PolyBook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live BTC lag visualizer")
    p.add_argument("--window", type=int, default=180,
                   help="Seconds of history to show (default 180)")
    p.add_argument("--hz", type=float, default=4.0,
                   help="Plot refresh rate in Hz (default 4)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared data buffers (written by asyncio tasks, read by matplotlib animation)
# ---------------------------------------------------------------------------

class DataBuffers:
    def __init__(self, maxlen: int):
        self.times_binance: deque[float] = deque(maxlen=maxlen)
        self.microprice: deque[float] = deque(maxlen=maxlen)

        self.times_poly: deque[float] = deque(maxlen=maxlen)
        self.poly_up_mid: deque[float] = deque(maxlen=maxlen)

        # For the right axis scale: track current K so we can overlay it
        self.current_K: float = 0.0
        self.window_ts: int = 0
        self.slug: str = ""

        # Status strings for title
        self.binance_stale_ms: int = 0
        self.poly_stale_ms: int = 0


# ---------------------------------------------------------------------------
# Asyncio feed tasks
# ---------------------------------------------------------------------------

async def run_feeds(bufs: DataBuffers, stop_event: asyncio.Event) -> None:
    spot_books = {asset: SpotBook(asset) for asset in ASSETS}
    poly_books = {asset: PolyBook(asset) for asset in ASSETS}

    def on_binance(asset: str, book: SpotBook) -> None:
        if asset != "btc" or not book.ready:
            return
        bufs.times_binance.append(time.time())
        bufs.microprice.append(book.microprice)
        bufs.binance_stale_ms = book.stale_ms()

    def on_poly(asset: str, book: PolyBook) -> None:
        if asset != "btc" or not book.ready:
            return
        up = book.up
        if up and up.best_bid > 0 and up.best_ask < 1:
            bufs.times_poly.append(time.time())
            bufs.poly_up_mid.append(up.mid)
            bufs.poly_stale_ms = book.stale_ms()

    binance_ws = BinanceWS(books=spot_books, on_update=on_binance)
    poly_ws = PolymarketWS(books=poly_books, on_update=on_poly)

    # Resolve current BTC market
    window_ts = current_window_ts()
    market = fetch_market("btc", window_ts)
    if market:
        bufs.current_K = 0.0  # will be set once RTDS connects; REST doesn't have it live
        bufs.window_ts = window_ts
        bufs.slug = market["slug"]
        poly_books["btc"].set_tokens(market["up_token_id"], market["down_token_id"])
        poly_ws.subscribe("btc", market["up_token_id"], market["down_token_id"])
        print(f"Resolved: {market['slug']}", file=sys.stderr)
        print(f"  Up token:   {market['up_token_id'][:16]}…", file=sys.stderr)
        print(f"  Down token: {market['down_token_id'][:16]}…", file=sys.stderr)
    else:
        print("WARNING: Could not resolve BTC market from gamma. Polymarket feed will be empty.",
              file=sys.stderr)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(binance_ws.run())
        tg.create_task(poly_ws.run())
        tg.create_task(_stop_watcher(stop_event, binance_ws, poly_ws))


async def _stop_watcher(stop_event: asyncio.Event, *clients) -> None:
    await stop_event.wait()
    for c in clients:
        c.stop()


# ---------------------------------------------------------------------------
# Matplotlib animation
# ---------------------------------------------------------------------------

def build_plot(bufs: DataBuffers, window_s: int, refresh_ms: int):
    fig, ax_left = plt.subplots(figsize=(14, 6))
    ax_right = ax_left.twinx()

    fig.patch.set_facecolor("#0f0f0f")
    ax_left.set_facecolor("#0f0f0f")

    for ax in (ax_left, ax_right):
        ax.tick_params(colors="#cccccc", labelsize=9)
        ax.spines["bottom"].set_color("#444444")
        ax.spines["top"].set_color("#444444")
        ax.spines["left"].set_color("#4477cc")
        ax.spines["right"].set_color("#cc7733")

    line_binance, = ax_left.plot([], [], color="#4da6ff", linewidth=1.5,
                                  label="Binance microprice (USD)")
    line_poly, = ax_right.plot([], [], color="#ff8c42", linewidth=1.5,
                                label="Polymarket Up mid (prob)")

    # Strike price horizontal line on left axis
    line_K = ax_left.axhline(y=0, color="#888888", linewidth=0.8,
                              linestyle="--", alpha=0.5, label="K (strike)")

    ax_left.set_ylabel("Binance BTC microprice (USD)", color="#4da6ff", fontsize=10)
    ax_right.set_ylabel("Polymarket Up implied prob", color="#ff8c42", fontsize=10)
    ax_left.yaxis.label.set_color("#4da6ff")
    ax_right.yaxis.label.set_color("#ff8c42")
    ax_right.set_ylim(0, 1)

    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_left.tick_params(axis="x", rotation=30)

    # Legend
    lines = [line_binance, line_poly, line_K]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc="upper left",
                   facecolor="#1a1a1a", edgecolor="#444444",
                   labelcolor="#cccccc", fontsize=9)

    title = ax_left.set_title("", color="#eeeeee", fontsize=11, pad=10)
    fig.tight_layout()

    def update(_frame):
        now = time.time()
        cutoff = now - window_s

        # Binance data
        b_times = list(bufs.times_binance)
        b_mp = list(bufs.microprice)
        p_times = list(bufs.times_poly)
        p_mid = list(bufs.poly_up_mid)

        # Filter to window
        b_pairs = [(t, v) for t, v in zip(b_times, b_mp) if t >= cutoff]
        p_pairs = [(t, v) for t, v in zip(p_times, p_mid) if t >= cutoff]

        if b_pairs:
            bt, bv = zip(*b_pairs)
            bt_dt = [datetime.fromtimestamp(t) for t in bt]
            line_binance.set_data(bt_dt, bv)
            ax_left.set_xlim(datetime.fromtimestamp(cutoff),
                             datetime.fromtimestamp(now + 2))
            lo, hi = min(bv), max(bv)
            pad = max((hi - lo) * 0.1, 5.0)
            ax_left.set_ylim(lo - pad, hi + pad)

            # Update K line if we have it
            if bufs.current_K > 0:
                line_K.set_ydata([bufs.current_K, bufs.current_K])

        if p_pairs:
            pt, pv = zip(*p_pairs)
            pt_dt = [datetime.fromtimestamp(t) for t in pt]
            line_poly.set_data(pt_dt, pv)

        # Title with live stats
        b_n = len(b_pairs)
        p_n = len(p_pairs)
        slug = bufs.slug or "resolving…"
        b_last = f"{bv[-1]:,.2f}" if b_pairs else "—"
        p_last = f"{pv[-1]:.3f}" if p_pairs else "—"
        b_stale = bufs.binance_stale_ms
        p_stale = bufs.poly_stale_ms

        title.set_text(
            f"BTC Lag Visualizer  |  {slug}\n"
            f"Binance μp: {b_last} ({b_n} pts, stale {b_stale}ms)   "
            f"Poly Up mid: {p_last} ({p_n} pts, stale {p_stale}ms)   "
            f"[last {window_s}s]"
        )

        return line_binance, line_poly, line_K, title

    ani = FuncAnimation(fig, update, interval=refresh_ms,
                        blit=False, cache_frame_data=False)
    return fig, ani


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    window_s = args.window
    refresh_ms = int(1000 / args.hz)

    # Buffer enough for the display window plus some headroom
    maxlen = max(window_s * 20, 2000)  # Binance fires ~10Hz, poly ~2Hz
    bufs = DataBuffers(maxlen=maxlen)

    stop_event = asyncio.Event()

    # Run asyncio feeds in a background thread
    import threading

    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_feeds(bufs, stop_event))

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    print(f"Connecting feeds… (window={window_s}s, refresh={args.hz:.0f}Hz)", file=sys.stderr)
    print("Close the plot window to exit.", file=sys.stderr)

    fig, ani = build_plot(bufs, window_s, refresh_ms)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        loop.call_soon_threadsafe(stop_event.set)
        thread.join(timeout=3)
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
