"""
demo_kalshi_ws.py — Stream live BTC / ETH Kalshi market odds.

Discovers the next-to-close BTC and ETH crypto markets via REST, then
subscribes to the Kalshi v2 trade WebSocket and prints a live ticker:

    [12:30:01] KXBTCD-25MAY07-T108250  yes_bid=58 yes_ask=61  last=59  vol=412
    [12:30:01] KXETHD-25MAY07-T2750    yes_bid=44 yes_ask=47  last=45  vol=88

Auth: same RSA-PSS scheme as REST. Signature path on the upgrade is
"/trade-api/ws/v2".

Usage:
    python scripts/demo_kalshi_ws.py
    python scripts/demo_kalshi_ws.py --demo
    python scripts/demo_kalshi_ws.py --series KXBTCD KXETHD
    python scripts/demo_kalshi_ws.py --tickers KXBTCD-25MAY07-T108250 KXETHD-25MAY07-T2750
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


PROD_REST = "https://api.elections.kalshi.com"
DEMO_REST = "https://demo-api.kalshi.co"
PROD_WS   = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEMO_WS   = "wss://demo-api.kalshi.co/trade-api/ws/v2"

API_PREFIX = "/trade-api/v2"
WS_PATH    = "/trade-api/ws/v2"

# Confirmed 15-minute over/under crypto series on Kalshi.
# Example URL: https://kalshi.com/markets/kxbtc15m/bitcoin-price-up-down/kxbtc15m-26may061300
SPRINT_SERIES = ["KXBTC15M", "KXETH15M"]

# Backup list (broader categories) used only by --list mode to help the user
# discover other crypto series.
DEFAULT_SERIES = [
    "KXBTC15M", "KXETH15M",
    "KXBTC", "KXETH",
    "KXBTCD", "KXETHD",
    "KXBTCRESH", "KXETHRESH",
]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def load_private_key():
    pem_inline = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    pem_file   = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")
    if pem_inline:
        pem_bytes = pem_inline.encode("utf-8").replace(b"\\n", b"\n")
    elif pem_file:
        path = Path(pem_file).expanduser()
        if not path.exists():
            sys.exit(f"ERROR: KALSHI_PRIVATE_KEY_FILE missing: {path}")
        pem_bytes = path.read_bytes()
    else:
        sys.exit("ERROR: set KALSHI_PRIVATE_KEY_FILE or KALSHI_PRIVATE_KEY_PEM in .env")
    return serialization.load_pem_private_key(pem_bytes, password=None)


def sign(private_key, method: str, path: str) -> tuple[str, str]:
    ts_ms = str(int(time.time() * 1000))
    msg   = (ts_ms + method.upper() + path).encode("utf-8")
    sig   = private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return ts_ms, base64.b64encode(sig).decode("ascii")


def auth_headers(private_key, key_id: str, method: str, path: str) -> dict:
    ts, signature = sign(private_key, method, path)
    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

def list_open_markets(host: str, key_id: str, pk, series_tickers: list[str]) -> list[dict]:
    """
    Return a flat list of open markets across the given series, sorted by
    close_time ascending (earliest expiry first).
    """
    out = []
    sign_path = f"{API_PREFIX}/markets"  # signature covers PATH ONLY, no query
    for series in series_tickers:
        headers = {"Accept": "application/json",
                   **auth_headers(pk, key_id, "GET", sign_path)}
        params  = {"status": "open", "series_ticker": series, "limit": 200}
        try:
            r = requests.get(host + sign_path, headers=headers,
                             params=params, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"  WARN: list {series}: {e}", file=sys.stderr)
            continue
        markets = r.json().get("markets", []) or []
        out.extend(markets)
    out.sort(key=lambda m: m.get("close_time", "9999"))
    return out


def list_all_crypto_series(host: str, key_id: str, pk) -> list[dict]:
    """Browse every Kalshi event series tagged Crypto so we can find
    short-horizon markets without guessing series tickers."""
    sign_path = f"{API_PREFIX}/series"
    headers = {"Accept": "application/json",
               **auth_headers(pk, key_id, "GET", sign_path)}
    out: list[dict] = []
    cursor = ""
    while True:
        params = {"limit": 200, "category": "Crypto"}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(host + sign_path, headers=headers,
                             params=params, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"  WARN: list series: {e}", file=sys.stderr)
            return out
        body = r.json() or {}
        out.extend(body.get("series", []) or [])
        cursor = body.get("cursor") or ""
        if not cursor:
            break
    return out


def pick_next_market(markets: list[dict], asset_hint: str) -> dict | None:
    """First open market whose ticker contains the asset hint (BTC/ETH).
    Markets are pre-sorted by close_time ascending, so this returns the
    soonest-closing one — i.e. the 15-min sprint if one exists."""
    hint = asset_hint.upper()
    for m in markets:
        if hint in m.get("ticker", "").upper():
            return m
    return None


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

async def stream(ws_url: str, key_id: str, pk, tickers: list[str],
                 strike_by_ticker: dict[str, str] | None = None) -> None:
    headers = auth_headers(pk, key_id, "GET", WS_PATH)
    print(f"  Connecting: {ws_url}")
    async with websockets.connect(
        ws_url,
        additional_headers=list(headers.items()),
        ping_interval=None,   # disable WS-level pings; Kalshi needs app-level pings
        open_timeout=10,
    ) as ws:
        print(f"  Connected. Subscribing to {len(tickers)} tickers…")

        async def _keepalive():
            pid = 100
            while True:
                await asyncio.sleep(10)
                try:
                    await ws.send(json.dumps({"id": pid, "cmd": "ping"}))
                    pid += 1
                except Exception:
                    return
        asyncio.create_task(_keepalive())

        # ticker = last trade prints; orderbook_delta = book mutations we
        # reconstruct top-of-book from. Kalshi's `ticker` payload doesn't
        # include best bid/ask, so the book channel is mandatory.
        sub_msgs = [
            {"id": 1, "cmd": "subscribe",
             "params": {"channels": ["ticker"], "market_tickers": tickers}},
            {"id": 2, "cmd": "subscribe",
             "params": {"channels": ["orderbook_delta"], "market_tickers": tickers}},
            {"id": 3, "cmd": "subscribe",
             "params": {"channels": ["trade"], "market_tickers": tickers}},
        ]
        for m in sub_msgs:
            await ws.send(json.dumps(m))

        # Per-ticker state. yes_book / no_book are dicts of {price_cents: size}
        # from which we derive top-of-book bid/ask.
        state = {
            t: {
                "yes_book": {},   # buyer-side resting orders for YES
                "no_book":  {},   # buyer-side resting orders for NO
                "last":     None,
                "vol":      0,
            } for t in tickers
        }

        def best_yes_bid(s) -> float | None:
            # Highest dollar price a YES buyer will pay (0.0–1.0)
            return max(s["yes_book"]) if s["yes_book"] else None

        def best_yes_ask(s) -> float | None:
            # YES ask = 1.0 - best NO bid (highest price a NO buyer will pay)
            return (1.0 - max(s["no_book"])) if s["no_book"] else None

        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue

            t = msg.get("type")
            payload = msg.get("msg") or {}

            if t in ("subscribed", "pong"):
                if t == "subscribed":
                    print(f"  ack id={msg.get('id')} channel={payload.get('channel')}",
                          flush=True)
                continue
            if t == "error":
                print(f"  ERROR: {msg}", file=sys.stderr, flush=True)
                continue

            ticker = payload.get("market_ticker") or payload.get("ticker")
            if ticker not in state:
                continue
            s = state[ticker]
            now = time.strftime("%H:%M:%S")
            changed = False

            if t == "orderbook_snapshot":
                # Format: yes_dollars_fp / no_dollars_fp = [["0.7100", "129.00"], ...]
                # Prices are dollar floats (0.0 – 1.0).
                s["yes_book"] = {
                    float(p): float(sz)
                    for p, sz in payload.get("yes_dollars_fp", [])
                }
                s["no_book"] = {
                    float(p): float(sz)
                    for p, sz in payload.get("no_dollars_fp", [])
                }
                changed = True

            elif t == "orderbook_delta":
                # Format: price_dollars (str float), delta_fp (str float), side
                side      = payload.get("side")
                price_raw = payload.get("price_dollars")
                delta_raw = payload.get("delta_fp")
                if price_raw is None or delta_raw is None or side not in ("yes", "no"):
                    continue
                price = float(price_raw)
                delta = float(delta_raw)
                book  = s["yes_book"] if side == "yes" else s["no_book"]
                book[price] = book.get(price, 0.0) + delta
                if book[price] <= 0:
                    book.pop(price, None)
                changed = True

            elif t == "ticker":
                # last_price in dollars
                px = payload.get("price") or payload.get("last_price")
                if px is not None:
                    s["last"] = float(px)
                    changed = True

            elif t == "trade":
                px = payload.get("yes_price") or payload.get("price")
                if px is not None:
                    s["last"] = float(px)
                    s["vol"] = (s["vol"] or 0) + float(payload.get("count", 0))
                    changed = True

            if changed:
                yb = best_yes_bid(s)   # 0.0–1.0
                ya = best_yes_ask(s)   # 0.0–1.0
                lp = s["last"]
                # Display as cents (×100) so 0.71 → 71¢ = "71% UP"
                yb_s    = f"{yb*100:5.1f}¢" if yb is not None else "    -"
                ya_s    = f"{ya*100:5.1f}¢" if ya is not None else "    -"
                no_s    = f"{(1-yb)*100:5.1f}¢" if yb is not None else "    -"
                lp_s    = f"{lp*100:5.1f}¢" if lp is not None else "    -"
                strike  = (strike_by_ticker or {}).get(ticker, "?")
                print(f"[{now}] {ticker:<32}  K={strike:<12}  "
                      f"UP bid={yb_s} ask={ya_s}  DOWN ask={no_s}  "
                      f"last={lp_s}  vol={s['vol']:.0f}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true",
                   help="Use Kalshi demo environment.")
    p.add_argument("--series", nargs="*", default=None,
                   help="Series tickers to scan (e.g. KXBTC KXETH). "
                        "Default: KXBTC KXETH KXBTCD KXETHD.")
    p.add_argument("--tickers", nargs="*", default=None,
                   help="Subscribe to these market tickers directly, skipping discovery.")
    p.add_argument("--list", action="store_true",
                   help="List all open crypto series + nearest-close markets and exit.")
    return p.parse_args()


def _close_within_minutes(m: dict, max_minutes: int) -> bool:
    """True if the market closes within `max_minutes` from now."""
    from datetime import datetime, timezone
    ct = m.get("close_time", "")
    if not ct:
        return False
    try:
        # Kalshi uses ISO 8601 with Z; normalise to +00:00.
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except ValueError:
        return False
    secs_to_close = (dt - datetime.now(timezone.utc)).total_seconds()
    return 0 < secs_to_close <= max_minutes * 60


def main() -> None:
    args = parse_args()

    key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    if not key_id:
        sys.exit("ERROR: KALSHI_API_KEY_ID not set in .env")
    pk = load_private_key()

    rest_host = DEMO_REST if args.demo else PROD_REST
    ws_url    = DEMO_WS   if args.demo else PROD_WS

    print("=" * 60)
    print(f"Kalshi WS demo  [{'DEMO' if args.demo else 'PROD'}]  — 15-min crypto over/under")
    print("=" * 60)

    # ----- --list mode: dump every crypto series + nearest market and exit
    if args.list:
        series_rows = list_all_crypto_series(rest_host, key_id, pk)
        print(f"\n  Found {len(series_rows)} crypto series:")
        for s in series_rows:
            print(f"    {s.get('ticker'):<20}  {s.get('title')}")
        # Also list every open crypto market across those series, sorted by
        # close_time, so the user can see the 15-min sprints if they exist.
        all_series_tickers = [s["ticker"] for s in series_rows] or DEFAULT_SERIES
        markets = list_open_markets(rest_host, key_id, pk, all_series_tickers)
        print(f"\n  Open markets ({len(markets)} total, soonest first):")
        for m in markets[:50]:
            print(f"    {m.get('ticker'):<35}  closes={m.get('close_time')}  "
                  f"strike={m.get('strike_value') or m.get('subtitle')}")
        return

    strike_by_ticker: dict[str, str] = {}

    if args.tickers:
        tickers = args.tickers
        print(f"  Using user-supplied tickers: {tickers}")
    else:
        series_list = args.series or SPRINT_SERIES
        print(f"  Scanning 15-min sprint series: {series_list}")
        markets = list_open_markets(rest_host, key_id, pk, series_list)
        if not markets:
            sys.exit("  No open 15-min sprint markets found. "
                     "Pass --tickers if you have one, or --list to browse.")

        # Each event holds many strike rows. Group by event_ticker, pick the
        # event closing soonest, then within it pick the at-the-money strike.
        from collections import defaultdict
        by_event = defaultdict(list)
        for m in markets:
            by_event[m.get("event_ticker", "")].append(m)

        # Sort events by close_time of their first market (markets share
        # close_time within an event).
        events_sorted = sorted(
            by_event.items(),
            key=lambda kv: kv[1][0].get("close_time", "9999"),
        )

        tickers = []
        for label, hint in [("BTC", "BTC"), ("ETH", "ETH")]:
            chosen_event = None
            for ev_ticker, ev_markets in events_sorted:
                if hint in ev_ticker.upper() and _close_within_minutes(
                    ev_markets[0], 16
                ):
                    chosen_event = (ev_ticker, ev_markets)
                    break
            if not chosen_event:
                print(f"  WARN: no 15-min {label} sprint event closing within 16 min",
                      file=sys.stderr)
                continue
            ev_ticker, ev_markets = chosen_event
            # Pick the at-the-money market — the one whose yes_price is
            # nearest to 50 (which is what "Up or Down" implies). Falls back
            # to the median strike if prices aren't populated.
            def atm_score(m):
                yp = m.get("yes_bid")
                if yp is None:
                    return 9999
                return abs(int(yp) - 50)
            atm = min(ev_markets, key=atm_score)
            tickers.append(atm["ticker"])
            # Surface whichever strike-ish field Kalshi populated. The 15-min
            # binary "Up or Down" markets typically use floor_strike (the
            # open price at slot start) and/or rules_primary text.
            strike_disp = (
                atm.get("strike_value")
                or atm.get("floor_strike")
                or atm.get("cap_strike")
                or atm.get("subtitle")
                or atm.get("yes_sub_title")
                or atm.get("title")
                or "?"
            )
            strike_by_ticker[atm["ticker"]] = str(strike_disp)
            print(f"  Picked {label}: {atm['ticker']}  closes={atm.get('close_time')}  "
                  f"strike={strike_disp}  (event={ev_ticker}, {len(ev_markets)} rows)")
            # Dump every populated field once so we can see what the API gives us.
            non_null = {k: v for k, v in atm.items()
                        if v not in (None, "", [], {}) and k != "rules_primary"}
            print(f"    fields: {non_null}")

        if not tickers:
            sys.exit("  Could not pick any 15-min tickers.")

    try:
        asyncio.run(stream(ws_url, key_id, pk, tickers, strike_by_ticker))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
