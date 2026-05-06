"""
visualize_market.py — Collect live data then plot all lag-indicator combos.

Collects COLLECT_S seconds from Binance + Coinbase + Polymarket, then
produces a multi-panel static chart with every combination useful for
seeing the lag:

  Panel 1 (main): Binance microprice + mid + Coinbase last-trade
                  vs Polymarket Up bid/ask/mid
  Panel 2:        Spread between Polymarket Up ask and Down ask (should sum ~1)
  Panel 3:        % change from start — Binance microprice + Coinbase
                  vs Poly Up mid (shift Binance right to align with Poly)

Usage:
    python scripts/visualize_market.py
    python scripts/visualize_market.py --collect 180
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets

from polybot.clients.binance_ws import BinanceWS
from polybot.clients.polymarket_ws import PolymarketWS
from polybot.clients.polymarket_rest import fetch_market, current_window_ts
from polybot.state.spot_book import SpotBook
from polybot.state.poly_book import PolyBook
from polybot.infra.config import ASSETS

COINBASE_WS = "wss://advanced-trade-ws.coinbase.com"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", type=int, default=180,
                   help="Seconds to collect (default 180)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Coinbase ticker feed (public, no auth)
# ---------------------------------------------------------------------------

async def _coinbase_ticker(data: dict, stop_flag: asyncio.Event) -> None:
    """
    Subscribe to Coinbase Advanced Trade WS ticker channel for BTC-USD.
    Captures last-trade price and computes microprice from best bid/ask when available.
    Appends to data['cb_times'] / data['cb_price'] / data['cb_micro'].
    """
    subscribe_msg = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "ticker",
    }
    last_cb_t = 0.0
    reconnect_delay = 2.0
    attempt = 0

    while not stop_flag.is_set():
        try:
            attempt += 1
            async with websockets.connect(
                COINBASE_WS,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=10,
            ) as ws:
                attempt = 0
                await ws.send(json.dumps(subscribe_msg))
                async for raw in ws:
                    if stop_flag.is_set():
                        return
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    channel = msg.get("channel", "")
                    # subscriptions ack and heartbeats — skip
                    if channel not in ("ticker", ""):
                        continue

                    for ev in msg.get("events", []):
                        for tick in ev.get("tickers", []):
                            if tick.get("product_id") != "BTC-USD":
                                continue

                            price_str = tick.get("price")
                            bid_str   = tick.get("best_bid")
                            ask_str   = tick.get("best_ask")
                            bid_qty   = tick.get("best_bid_quantity")
                            ask_qty   = tick.get("best_ask_quantity")

                            if not price_str:
                                continue
                            try:
                                price = float(price_str)
                            except ValueError:
                                continue

                            t = time.time()
                            if t - last_cb_t < 0.1:
                                continue
                            last_cb_t = t
                            data["cb_times"].append(t)
                            data["cb_price"].append(price)

                            # Compute microprice if bid/ask/sizes are present
                            try:
                                if bid_str and ask_str and bid_qty and ask_qty:
                                    bid  = float(bid_str)
                                    ask  = float(ask_str)
                                    bsz  = float(bid_qty)
                                    asz  = float(ask_qty)
                                    if bsz + asz > 0:
                                        mp = (bid * asz + ask * bsz) / (bsz + asz)
                                        data["cb_micro"].append(mp)
                                    else:
                                        data["cb_micro"].append(price)
                                else:
                                    data["cb_micro"].append(price)
                            except (ValueError, TypeError):
                                data["cb_micro"].append(price)

        except asyncio.CancelledError:
            return
        except Exception as exc:
            if stop_flag.is_set():
                return
            delay = min(reconnect_delay * attempt, 15.0)
            print(f"\r  [Coinbase WS err={exc!r} retry in {delay:.0f}s]   ",
                  end="", flush=True)
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

async def collect(collect_s: int):
    """Collect all series; returns a dict of named lists."""
    data = {
        "b_times":  [],   # Binance timestamps
        "b_micro":  [],   # Binance microprice
        "b_mid":    [],   # Binance plain mid
        "cb_times": [],   # Coinbase timestamps
        "cb_price": [],   # Coinbase last-trade price
        "cb_micro": [],   # Coinbase microprice (bid*ask_sz + ask*bid_sz / total)
        "p_times":  [],   # Polymarket timestamps
        "p_up_bid": [],
        "p_up_ask": [],
        "p_up_mid": [],
        "p_dn_bid": [],
        "p_dn_ask": [],
        "p_dn_mid": [],
        "slug":     "unknown",
    }

    spot_books = {a: SpotBook(a) for a in ASSETS}
    poly_books = {a: PolyBook(a) for a in ASSETS}

    last_b_t = 0.0

    def on_binance(asset: str, book: SpotBook) -> None:
        nonlocal last_b_t
        if asset != "btc" or not book.ready:
            return
        t = time.time()
        if t - last_b_t < 0.2:
            return
        last_b_t = t
        data["b_times"].append(t)
        data["b_micro"].append(book.microprice)
        data["b_mid"].append(book.mid)

    def on_poly(asset: str, book: PolyBook) -> None:
        if asset != "btc" or not book.ready:
            return
        up = book.up
        dn = book.down
        if not (up and dn):
            return
        if not (0 < up.best_bid and up.best_ask < 1):
            return
        data["p_times"].append(time.time())
        data["p_up_bid"].append(up.best_bid)
        data["p_up_ask"].append(up.best_ask)
        data["p_up_mid"].append(up.mid)
        data["p_dn_bid"].append(dn.best_bid)
        data["p_dn_ask"].append(dn.best_ask)
        data["p_dn_mid"].append(dn.mid)

    binance_ws = BinanceWS(books=spot_books, on_update=on_binance)
    poly_ws    = PolymarketWS(books=poly_books, on_update=on_poly)

    window_ts = current_window_ts()
    market = fetch_market("btc", window_ts)
    if market:
        data["slug"] = market["slug"]
        poly_books["btc"].set_tokens(market["up_token_id"], market["down_token_id"])
        poly_ws.subscribe("btc", market["up_token_id"], market["down_token_id"])
        print(f"  Market: {market['slug']}", flush=True)
    else:
        print("  WARNING: BTC market not found", flush=True)

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
            el = time.time() - start
            b_last  = f"{data['b_micro'][-1]:,.2f}"  if data["b_micro"]  else "—"
            cb_last = f"{data['cb_price'][-1]:,.2f}" if data["cb_price"] else "no CB"
            p_last  = f"{data['p_up_mid'][-1]:.4f}"  if data["p_up_mid"] else "—"
            print(
                f"\r  t={el:4.0f}s  Binance: {b_last} ({len(data['b_micro'])}pts)  "
                f"Coinbase: {cb_last} ({len(data['cb_price'])}pts)  "
                f"Poly: {p_last} ({len(data['p_up_mid'])}pts)  "
                f"[{collect_s - el:.0f}s left]   ",
                end="", flush=True,
            )
            await asyncio.sleep(1.0)

    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(binance_ws.run()),
        loop.create_task(poly_ws.run()),
        loop.create_task(_coinbase_ticker(data, stop_flag)),
        loop.create_task(_progress()),
    ]
    await _timer()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    print()
    return data


# ---------------------------------------------------------------------------
# Multi-panel static plot
# ---------------------------------------------------------------------------

def plot(data: dict, collect_s: int) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    from datetime import datetime

    slug    = data["slug"]
    b_times = data["b_times"]
    p_times = data["p_times"]
    cb_times = data["cb_times"]

    if not b_times:
        print("No Binance data — cannot plot.", file=sys.stderr)
        return

    b_dt  = [datetime.fromtimestamp(t) for t in b_times]
    p_dt  = [datetime.fromtimestamp(t) for t in p_times]
    cb_dt = [datetime.fromtimestamp(t) for t in cb_times]

    has_poly     = bool(p_times)
    has_coinbase = bool(cb_times)
    duration = b_times[-1] - b_times[0] if len(b_times) >= 2 else 0

    # ---- Figure layout: 3 rows ----
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#111111")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                           height_ratios=[3, 1.2, 1.2])

    def _style(ax):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.grid(axis="y", color="#222222", linewidth=0.4, linestyle="--")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7)

    # ================================================================
    # Panel 1: Spot prices vs Polymarket Up probability
    # ================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1r = ax1.twinx()
    _style(ax1)
    ax1r.set_facecolor("#111111")
    ax1r.tick_params(colors="#ff8c42", labelsize=8)

    # Binance mid (dashed, light blue)
    ax1.plot(b_dt, data["b_mid"], color="#6699cc", linewidth=0.9,
             alpha=0.5, linestyle="--", label="Binance mid")
    # Binance microprice (solid, bright blue)
    ax1.plot(b_dt, data["b_micro"], color="#4da6ff", linewidth=1.6,
             label="Binance microprice")
    # Coinbase last-trade (green)
    if has_coinbase:
        ax1.plot(cb_dt, data["cb_price"], color="#44cc88", linewidth=1.0,
                 alpha=0.55, linestyle="--", label="Coinbase last trade")
        if data["cb_micro"]:
            ax1.plot(cb_dt[:len(data["cb_micro"])], data["cb_micro"],
                     color="#00ff99", linewidth=1.5, label="Coinbase microprice")

    if has_poly:
        # Up bid (lower bound)
        ax1r.step(p_dt, data["p_up_bid"], color="#cc5500", linewidth=0.9,
                  alpha=0.6, where="post", linestyle=":", label="Poly Up bid")
        # Up ask
        ax1r.step(p_dt, data["p_up_ask"], color="#ff6600", linewidth=0.9,
                  alpha=0.6, where="post", linestyle="-.", label="Poly Up ask")
        # Up mid (clearest signal)
        ax1r.step(p_dt, data["p_up_mid"], color="#ff8c42", linewidth=2.0,
                  where="post", label="Poly Up mid")
        # Down mid (moves inversely)
        ax1r.step(p_dt, data["p_dn_mid"], color="#cc44ff", linewidth=1.2,
                  alpha=0.7, where="post", linestyle="--", label="Poly Down mid")

        lo = max(0.0, min(min(data["p_up_bid"]), min(data["p_dn_mid"])) - 0.04)
        hi = min(1.0, max(max(data["p_up_ask"]), max(data["p_dn_mid"])) + 0.04)
        ax1r.set_ylim(lo, hi)

    ax1.set_ylabel("BTC price (USD)", color="#4da6ff", fontsize=9)
    ax1r.set_ylabel("Polymarket implied prob", color="#ff8c42", fontsize=9)

    lines1,  labels1  = ax1.get_legend_handles_labels()
    lines1r, labels1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labels1 + labels1r,
               loc="upper left", facecolor="#1c1c1c", edgecolor="#444",
               labelcolor="#ccc", fontsize=8, ncol=2)
    ax1.set_title(
        f"{slug}  —  {collect_s}s collection  ({duration:.0f}s actual)\n"
        "Blue = Binance (microprice solid, mid dashed).  "
        "Green = Coinbase last trade.  Orange = Poly Up.  Purple = Poly Down.",
        color="#dddddd", fontsize=9, pad=6,
    )

    # ================================================================
    # Panel 2: Poly Up ask + Down ask (should sum to ~1.0 + 2×fee)
    # ================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    _style(ax2)

    if has_poly:
        ax2.step(p_dt, data["p_up_ask"], color="#ff8c42", linewidth=1.4,
                 where="post", label="Up ask")
        ax2.step(p_dt, data["p_dn_ask"], color="#cc44ff", linewidth=1.4,
                 where="post", label="Down ask")
        p_sum = [u + d for u, d in zip(data["p_up_ask"], data["p_dn_ask"])]
        ax2.step(p_dt, p_sum, color="#888888", linewidth=0.9, alpha=0.6,
                 where="post", linestyle="--", label="Up ask + Down ask")
        ax2.axhline(1.0, color="#555555", linewidth=0.7, linestyle=":")
        ax2.set_ylim(min(min(data["p_up_ask"]), min(data["p_dn_ask"])) - 0.02,
                     max(p_sum) + 0.02)
    ax2.set_ylabel("Ask prices", color="#bbbbbb", fontsize=9)
    ax2.legend(loc="upper left", facecolor="#1c1c1c", edgecolor="#444",
               labelcolor="#ccc", fontsize=8)
    ax2.set_title("Polymarket Up ask vs Down ask  (sum ≈ 1 + fees)",
                  color="#aaaaaa", fontsize=8, pad=4)

    # ================================================================
    # Panel 3: % change from start — all spot feeds vs Poly Up mid
    #          Shift any spot curve RIGHT by estimated lag to align with Poly
    # ================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3r = ax3.twinx()
    _style(ax3)
    ax3r.set_facecolor("#111111")
    ax3r.tick_params(colors="#ff8c42", labelsize=8)

    if len(data["b_micro"]) >= 2:
        base_mp = data["b_micro"][0]
        b_pct = [(v / base_mp - 1) * 100 for v in data["b_micro"]]
        ax3.plot(b_dt, b_pct, color="#4da6ff", linewidth=1.4,
                 label="Binance microprice Δ%")
        ax3.axhline(0, color="#444444", linewidth=0.5)

    if has_coinbase and len(data["cb_price"]) >= 2:
        base_cb = data["cb_price"][0]
        cb_pct = [(v / base_cb - 1) * 100 for v in data["cb_price"]]
        ax3.plot(cb_dt, cb_pct, color="#44cc88", linewidth=0.8,
                 alpha=0.5, linestyle="--", label="Coinbase last trade Δ%")
    if has_coinbase and len(data["cb_micro"]) >= 2:
        base_cm = data["cb_micro"][0]
        cm_pct = [(v / base_cm - 1) * 100 for v in data["cb_micro"]]
        cb_micro_dt = cb_dt[:len(data["cb_micro"])]
        ax3.plot(cb_micro_dt, cm_pct, color="#00ff99", linewidth=1.2,
                 label="Coinbase microprice Δ%")

    ax3.set_ylabel("Spot Δ% from start", color="#4da6ff", fontsize=8)

    if has_poly and len(data["p_up_mid"]) >= 2:
        base_pm = data["p_up_mid"][0]
        p_pct = [(v - base_pm) * 100 for v in data["p_up_mid"]]
        ax3r.step(p_dt, p_pct, color="#ff8c42", linewidth=1.6,
                  where="post", label="Poly Up mid Δ pp")
        ax3r.axhline(0, color="#664400", linewidth=0.5)
        ax3r.set_ylabel("Poly Up Δ prob-points", color="#ff8c42", fontsize=8)

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines3r, labels3r = ax3r.get_legend_handles_labels()
    ax3.legend(lines3 + lines3r, labels3 + labels3r,
               loc="upper left", facecolor="#1c1c1c", edgecolor="#444",
               labelcolor="#ccc", fontsize=8)
    ax3.set_title(
        "% change from start — shift a spot curve RIGHT by estimated lag to align with Poly",
        color="#aaaaaa", fontsize=8, pad=4,
    )
    ax3.set_xlabel("Time (local)", color="#aaaaaa", fontsize=8)

    # ================================================================
    # Save + show
    # ================================================================
    outfile = Path("lag_plot.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"Saved: {outfile.resolve()}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    print("=== BTC Lag Visualizer ===")
    print(f"Collecting {args.collect}s of live data…")

    try:
        data = asyncio.run(collect(args.collect))
    except KeyboardInterrupt:
        print("\nStopped early.")
        sys.exit(0)

    print(f"Collected: Binance {len(data['b_micro'])} pts, "
          f"Coinbase last-trade {len(data['cb_price'])} pts "
          f"(microprice {len(data['cb_micro'])} pts), "
          f"Polymarket {len(data['p_up_mid'])} pts")
    print("Rendering plot…")
    plot(data, args.collect)


if __name__ == "__main__":
    main()
