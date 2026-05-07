"""
visualize_market.py — Collect 60s of Binance/Coinbase BTC + Kalshi 15-min odds, then plot.

Goal: visually confirm whether Kalshi's yes probability LAGS spot.

  Panel 1 (main): Spot microprice (left axis) vs Kalshi YES bid/ask (right axis)
  Panel 2:        Absolute Δ from start on dual axes — lag is visible here.

Usage:
    python scripts/visualize_market.py
    python scripts/visualize_market.py --collect 120
    python scripts/visualize_market.py --spot binance      # use Binance (VPN required)
    python scripts/visualize_market.py --spot coinbase     # use Coinbase (US, no VPN)
    python scripts/visualize_market.py --ticker KXBTC15M-26MAY061415-00
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from betbot.kalshi.config import SPOT_SOURCE as _DEFAULT_SPOT_SOURCE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COINBASE_WS  = "wss://advanced-trade-ws.coinbase.com"
BINANCE_WS   = "wss://stream.binance.com:9443"
KALSHI_REST  = "https://api.elections.kalshi.com"
API_PREFIX   = "/trade-api/v2"
SPRINT_SERIES = {"btc": "KXBTC15M", "eth": "KXETH15M"}

# ---------------------------------------------------------------------------
# Kalshi auth
# ---------------------------------------------------------------------------

def load_private_key():
    pem_inline = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    pem_file   = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
    if pem_inline:
        pem_bytes = pem_inline.encode().replace(b"\\n", b"\n")
    elif pem_file:
        p = Path(pem_file).expanduser()
        if not p.exists():
            sys.exit(f"ERROR: KALSHI_PRIVATE_KEY_FILE missing: {p}")
        pem_bytes = p.read_bytes()
    else:
        sys.exit("ERROR: set KALSHI_PRIVATE_KEY_FILE or KALSHI_PRIVATE_KEY_PEM in .env")
    return serialization.load_pem_private_key(pem_bytes, password=None)


def _sign(pk, method: str, path: str) -> tuple[str, str]:
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    sig = pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return ts, base64.b64encode(sig).decode()


def _auth_headers(pk, key_id: str, method: str, path: str) -> dict:
    ts, sig = _sign(pk, method, path)
    return {"KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig}


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

def _close_within_minutes(m: dict, n: int) -> bool:
    from datetime import datetime, timezone
    ct = m.get("close_time", "")
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False
    return 0 < (dt - datetime.now(timezone.utc)).total_seconds() <= n * 60


def discover_kalshi_market(key_id: str, pk, asset: str) -> dict:
    series    = SPRINT_SERIES[asset]
    sign_path = f"{API_PREFIX}/markets"
    headers   = {"Accept": "application/json",
                 **_auth_headers(pk, key_id, "GET", sign_path)}
    r = requests.get(KALSHI_REST + sign_path, headers=headers,
                     params={"status": "open", "series_ticker": series, "limit": 200},
                     timeout=10)
    r.raise_for_status()
    markets = r.json().get("markets", []) or []
    sprints = [m for m in markets if _close_within_minutes(m, 16)]
    if not sprints:
        markets.sort(key=lambda m: m.get("close_time", "9999"))
        if not markets:
            sys.exit(f"No open {asset.upper()} markets found in {series}.")
        print(f"  WARN: no sprint closing within 16 min — using nearest market")
        sprints = [markets[0]]
    return sprints[0]


# ---------------------------------------------------------------------------
# Coinbase collector
# ---------------------------------------------------------------------------

async def collect_coinbase(product: str, data: dict, stop: asyncio.Event):
    sub = {"type": "subscribe", "product_ids": [product], "channel": "ticker"}
    last_t = 0.0
    while not stop.is_set():
        try:
            async with websockets.connect(
                COINBASE_WS, ping_interval=20, ping_timeout=10, open_timeout=10,
            ) as ws:
                await ws.send(json.dumps(sub))
                async for raw in ws:
                    if stop.is_set():
                        return
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("channel") not in ("ticker", ""):
                        continue
                    for ev in msg.get("events", []):
                        for tick in ev.get("tickers", []):
                            if tick.get("product_id") != product:
                                continue
                            price_s = tick.get("price")
                            bid_s   = tick.get("best_bid")
                            ask_s   = tick.get("best_ask")
                            bq_s    = tick.get("best_bid_quantity")
                            aq_s    = tick.get("best_ask_quantity")
                            if not price_s:
                                continue
                            t = time.time()
                            if t - last_t < 0.05:
                                continue
                            last_t = t
                            try:
                                price = float(price_s)
                                if bid_s and ask_s and bq_s and aq_s:
                                    bid = float(bid_s); ask = float(ask_s)
                                    bsz = float(bq_s);  asz = float(aq_s)
                                    mp = (bid * asz + ask * bsz) / (bsz + asz) if (bsz + asz) > 0 else price
                                else:
                                    mp = price
                            except (ValueError, TypeError):
                                continue
                            data["cb_t"].append(t)
                            data["cb_price"].append(price)
                            data["cb_micro"].append(mp)
        except asyncio.CancelledError:
            return
        except Exception:
            if stop.is_set():
                return
            await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Binance collector (L2 book — microprice from best bid/ask + sizes)
# ---------------------------------------------------------------------------

async def collect_binance(symbol: str, data: dict, stop: asyncio.Event):
    """Binance combined stream: bookTicker gives best bid/ask/sizes in real time."""
    url  = f"{BINANCE_WS}/ws/{symbol.lower()}@bookTicker"
    last_t = 0.0
    while not stop.is_set():
        try:
            async with websockets.connect(
                url, ping_interval=20, ping_timeout=10, open_timeout=10,
            ) as ws:
                async for raw in ws:
                    if stop.is_set():
                        return
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    # bookTicker: {b: best_bid, B: bid_qty, a: best_ask, A: ask_qty}
                    bid_s = msg.get("b"); ask_s = msg.get("a")
                    bsz_s = msg.get("B"); asz_s = msg.get("A")
                    if not (bid_s and ask_s and bsz_s and asz_s):
                        continue
                    t = time.time()
                    if t - last_t < 0.05:
                        continue
                    last_t = t
                    try:
                        bid = float(bid_s); ask = float(ask_s)
                        bsz = float(bsz_s); asz = float(asz_s)
                        mid = (bid + ask) / 2.0
                        mp  = (bid * asz + ask * bsz) / (bsz + asz) if (bsz + asz) > 0 else mid
                    except (ValueError, TypeError):
                        continue
                    data["cb_t"].append(t)
                    data["cb_price"].append(mid)
                    data["cb_micro"].append(mp)
        except asyncio.CancelledError:
            return
        except Exception:
            if stop.is_set():
                return
            await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Kalshi collector — REST polling at 1 Hz (no WebSocket).
# Same endpoint and auth as test_trade.py, which is the proven-working path.
# ---------------------------------------------------------------------------

async def collect_kalshi(ticker: str, data: dict, stop: asyncio.Event,
                         key_id: str, pk):
    """
    Poll GET /trade-api/v2/markets/{ticker} every second and append the
    yes_bid_dollars / yes_ask_dollars to the data dict.

    No WebSocket, no orderbook reconstruction, no heartbeat. Each poll is
    an independent authenticated REST request — if one fails, the next one
    starts fresh.
    """
    import aiohttp
    POLL_S = 1.0
    path   = f"{API_PREFIX}/markets/{ticker}"
    url    = KALSHI_REST + path

    fails = 0
    while not stop.is_set():
        async with aiohttp.ClientSession() as session:
            try:
                while not stop.is_set():
                    t0      = time.monotonic()
                    headers = _auth_headers(pk, key_id, "GET", path)
                    try:
                        async with session.get(url, headers=headers,
                                               timeout=aiohttp.ClientTimeout(total=5)) as r:
                            if r.status == 429:
                                await asyncio.sleep(2.0)
                                continue
                            if r.status != 200:
                                fails += 1
                                if fails >= 5:
                                    print(f"  [Kalshi REST] {fails} failures, resetting session",
                                          flush=True)
                                    fails = 0
                                    break
                                await asyncio.sleep(1.0)
                                continue
                            payload = await r.json()
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        fails += 1
                        print(f"  [Kalshi REST] {type(e).__name__}: {e}", flush=True)
                        await asyncio.sleep(1.0)
                        continue

                    fails = 0
                    mkt   = payload.get("market") or payload
                    bid_s = mkt.get("yes_bid_dollars")
                    ask_s = mkt.get("yes_ask_dollars")
                    if bid_s is not None and ask_s is not None:
                        try:
                            yb = float(bid_s)
                            ya = float(ask_s)
                            if yb > 0 and ya > 0 and ya >= yb:
                                now = time.time()
                                data["k_t"].append(now)
                                data["k_bid"].append(yb)
                                data["k_ask"].append(ya)
                                data["k_mid"].append((yb + ya) / 2.0)
                        except (TypeError, ValueError):
                            pass

                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0.0, POLL_S - elapsed))
            except asyncio.CancelledError:
                return
        await asyncio.sleep(0.5)


# ---------------------------------------------------------------------------
# Collection runner
# ---------------------------------------------------------------------------

async def collect(collect_s: int, asset: str, ticker: str,
                  key_id: str, pk, spot: str = "binance") -> dict:
    data = {
        "cb_t": [], "cb_price": [], "cb_micro": [],
        "k_t":  [], "k_bid": [],   "k_ask": [],   "k_mid": [],
    }
    cb_product = "BTC-USD" if asset == "btc" else "ETH-USD"
    bn_symbol  = "BTCUSDT" if asset == "btc" else "ETHUSDT"
    stop       = asyncio.Event()
    start      = time.time()

    async def _timer():
        while not stop.is_set():
            elapsed   = time.time() - start
            remaining = collect_s - elapsed
            cb_pts    = len(data["cb_micro"])
            k_pts     = len(data["k_mid"])
            src       = spot.capitalize()
            print(f"\r  [{elapsed:5.1f}s / {collect_s}s]  "
                  f"{src}: {cb_pts} pts  Kalshi: {k_pts} pts  "
                  f"[{remaining:.0f}s left]   ",
                  end="", flush=True)
            if elapsed >= collect_s:
                stop.set()
                return
            await asyncio.sleep(1.0)

    spot_task = (
        asyncio.create_task(collect_binance(bn_symbol, data, stop))
        if spot == "binance"
        else asyncio.create_task(collect_coinbase(cb_product, data, stop))
    )
    tasks = [
        spot_task,
        asyncio.create_task(collect_kalshi(ticker, data, stop, key_id, pk)),
        asyncio.create_task(_timer()),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    print()
    return data


# ---------------------------------------------------------------------------
# Static lag-detection plot
# ---------------------------------------------------------------------------

def plot(data: dict, asset: str, ticker: str, strike: float, collect_s: int):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    from datetime import datetime

    cb_t     = data["cb_t"]
    k_t      = data["k_t"]
    cb_micro = data["cb_micro"]
    cb_price = data["cb_price"]
    k_bid    = data["k_bid"]
    k_ask    = data["k_ask"]
    k_mid    = data["k_mid"]

    if not cb_t:
        print("No Coinbase data — cannot plot.", file=sys.stderr)
        return
    if not k_t:
        print("No Kalshi data — cannot plot.", file=sys.stderr)
        return

    cb_dt = [datetime.fromtimestamp(t) for t in cb_t]
    k_dt  = [datetime.fromtimestamp(t) for t in k_t]

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#111111")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.5, height_ratios=[1.2, 1])

    def _style(ax, ylabel="", ycolor="#cccccc"):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.grid(color="#1e1e1e", linewidth=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, color=ycolor, fontsize=9)

    asset_upper = asset.upper()
    strike_str  = f"${strike:,.2f}"

    # ================================================================
    # Panel 1: raw prices on dual axes
    # ================================================================
    ax1  = fig.add_subplot(gs[0])
    ax1r = ax1.twinx()
    _style(ax1, f"{asset_upper} price (USD)", "#00e676")
    ax1r.set_facecolor("#111111")
    ax1r.tick_params(colors="#ff8c42", labelsize=8)

    ax1.plot(cb_dt, cb_price, color="#44cc88", linewidth=0.8,
             alpha=0.5, linestyle="--", label="Coinbase last trade")
    ax1.plot(cb_dt, cb_micro, color="#00e676", linewidth=1.6,
             label="Coinbase microprice")

    ax1r.step(k_dt, [b * 100 for b in k_bid], color="#cc5500", linewidth=0.8,
              where="post", alpha=0.6, linestyle=":", label="Kalshi YES bid (¢)")
    ax1r.step(k_dt, [a * 100 for a in k_ask], color="#ff6600", linewidth=0.8,
              where="post", alpha=0.6, linestyle="-.", label="Kalshi YES ask (¢)")
    ax1r.step(k_dt, [m * 100 for m in k_mid], color="#ff8c42", linewidth=2.0,
              where="post", label="Kalshi YES mid (¢)")

    ax1r.set_ylabel("Kalshi YES probability (¢)", color="#ff8c42", fontsize=9)
    lo = max(0, min(k_bid) * 100 - 5) if k_bid else 0
    hi = min(100, max(k_ask) * 100 + 5) if k_ask else 100
    ax1r.set_ylim(lo, hi)

    h1, l1   = ax1.get_legend_handles_labels()
    h1r, l1r = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h1r, l1 + l1r, loc="upper left",
               facecolor="#1c1c1c", edgecolor="#444", labelcolor="#ccc",
               fontsize=8, ncol=2)
    ax1.set_title(
        f"{ticker}  |  Strike: {strike_str}  |  {collect_s}s collection\n"
        f"Green = Coinbase microprice (USD).  Orange = Kalshi YES probability.",
        color="#dddddd", fontsize=9, pad=6,
    )

    # ================================================================
    # Panel 2: absolute change from start — dual axes so scales match
    #   Left  (green): Coinbase microprice Δ USD
    #   Right (orange): Kalshi YES mid Δ probability-points (×100 = cents)
    #
    # If Kalshi lags Coinbase, the orange line will mirror the green
    # line but shifted right by N seconds.
    # ================================================================
    ax2  = fig.add_subplot(gs[1], sharex=ax1)
    ax2r = ax2.twinx()
    _style(ax2, f"Coinbase Δ USD from start", "#00e676")
    ax2r.set_facecolor("#111111")
    ax2r.tick_params(colors="#ff8c42", labelsize=8)
    ax2.axhline(0, color="#444444", linewidth=0.6, linestyle="--")
    ax2r.axhline(0, color="#442200", linewidth=0.6, linestyle="--")

    if len(cb_micro) >= 2:
        base_cb  = cb_micro[0]
        cb_delta = [v - base_cb for v in cb_micro]
        ax2.plot(cb_dt, cb_delta, color="#00e676", linewidth=1.8,
                 label=f"Coinbase microprice  Δ USD  (base={base_cb:,.2f})")

    if len(k_mid) >= 2:
        base_km  = k_mid[0]
        k_delta  = [(v - base_km) * 100 for v in k_mid]   # in cents (prob-points)
        ax2r.step(k_dt, k_delta, color="#ff8c42", linewidth=1.8,
                  where="post",
                  label=f"Kalshi YES mid  Δ cents  (base={base_km*100:.1f}¢)")

    ax2r.set_ylabel("Kalshi YES mid Δ cents", color="#ff8c42", fontsize=9)
    ax2.set_xlabel("Time (local)", color="#aaaaaa", fontsize=8)

    h2,  l2  = ax2.get_legend_handles_labels()
    h2r, l2r = ax2r.get_legend_handles_labels()
    ax2.legend(h2 + h2r, l2 + l2r, loc="upper left",
               facecolor="#1c1c1c", edgecolor="#444", labelcolor="#ccc", fontsize=8)
    ax2.set_title(
        "LAG DETECTION — if Kalshi lags Coinbase, orange trails green by N seconds.\n"
        "Left axis = Coinbase Δ USD.  Right axis = Kalshi YES mid Δ cents.",
        color="#aaaaaa", fontsize=8, pad=4,
    )

    outfile = Path(f"lag_plot_{asset}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"Saved: {outfile.resolve()}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--collect", type=int, default=60,
                   help="Seconds to collect (default 60).")
    p.add_argument("--asset", choices=["btc", "eth"], default="btc")
    p.add_argument("--ticker", default=None,
                   help="Kalshi ticker; skip auto-discovery.")
    p.add_argument("--spot", choices=["binance", "coinbase"], default=_DEFAULT_SPOT_SOURCE,
                   help="Spot feed: binance (VPN, higher freq) or coinbase (US, no VPN). "
                        "Default: SPOT_SOURCE env var (currently %(default)s).")
    return p.parse_args()


def main():
    args   = parse_args()
    key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    if not key_id:
        sys.exit("ERROR: KALSHI_API_KEY_ID not set in .env")
    pk = load_private_key()

    print(f"=== {args.asset.upper()} lag visualizer — {args.collect}s collect then plot  [spot={args.spot}] ===")

    if args.ticker:
        ticker = args.ticker
        strike = 0.0
    else:
        print("  Discovering Kalshi 15-min market…")
        m      = discover_kalshi_market(key_id, pk, args.asset)
        ticker = m["ticker"]
        strike = float(m.get("floor_strike") or m.get("strike_value") or 0)
        print(f"  Ticker: {ticker}  Strike: ${strike:,.2f}  Closes: {m.get('close_time')}")

    print(f"  Collecting {args.collect}s from Coinbase + Kalshi…")
    try:
        data = asyncio.run(collect(args.collect, args.asset, ticker, key_id, pk, spot=args.spot))
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)

    print(f"  Coinbase: {len(data['cb_micro'])} pts   Kalshi: {len(data['k_mid'])} pts")
    if not data["cb_micro"] or not data["k_mid"]:
        sys.exit("Not enough data collected — check feeds and try again.")

    print("  Rendering plot…")
    plot(data, args.asset, ticker, strike, args.collect)


if __name__ == "__main__":
    main()
