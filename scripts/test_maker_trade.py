"""
test_maker_trade.py -- Maker-entry / taker-exit round-trip on Kalshi.

Sister to scripts/test_trade.py (which goes taker-on-both). This one:

  1. Discovers an active KXBTC15M market.
  2. Posts a BUY YES limit @ yes_bid (passive) or yes_bid+1c (aggressive).
     This is a MAKER order -- it does NOT cross the spread and does NOT
     fill immediately.
  3. Polls the order's filled_count every second up to --ttl seconds.
  4. If the limit FILLS within TTL:
       a. Hold for --hold-s seconds.
       b. Place a SELL YES limit @ current yes_bid (this DOES cross --
          we're hitting the bid as a taker for fast exit).
       c. Report round-trip P&L net of taker exit fee (maker entry is free).
  5. If the limit DOES NOT fill within TTL:
       a. DELETE the resting order (cancel it).
       b. Report 'no fill' result -- this is the data point we need from
          a Stage-A pilot per CLAUDE.md section 11.1.

The point of this script is to verify two things:
  A. Kalshi's API actually accepts and tracks a resting limit at the bid.
  B. We can detect fills + cancel orders programmatically (the building
     blocks for the eventual OrderManager in WI-5 Stage B).

Auth: KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_FILE (or _PEM) from .env.

Usage:
    python scripts/test_maker_trade.py --dry-run
    python scripts/test_maker_trade.py --target-usd 1.0 --ttl 30 --hold-s 10
    python scripts/test_maker_trade.py --aggressive   # post at bid+1c
"""

import argparse
import asyncio
import base64
import json
import sys
import time
import uuid
import datetime
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from betbot.kalshi.auth import load_private_key
from betbot.kalshi.config import (
    KALSHI_REST, KALSHI_KEY_ID, KALSHI_SERIES,
    THETA_FEE_TAKER,
)

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


# ---------------------------------------------------------------------------
# Auth (same shape as scripts/test_trade.py)
# ---------------------------------------------------------------------------

def _sign(pk, ts: str, method: str, path: str) -> str:
    msg = (ts + method + path).encode()
    sig = pk.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def _auth_headers(pk, method: str, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": _sign(pk, ts, method, path),
        "Content-Type":            "application/json",
    }


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

async def discover_market(session: aiohttp.ClientSession,
                          verbose: bool = False) -> dict | None:
    """
    Find a tradeable open KXBTC15M market. Prefers markets with central
    prices (yes_mid near 0.5) and at least 60s to close so the test has
    time to fill + hold + exit. Falls back progressively if nothing
    central is open.
    """
    path   = "/trade-api/v2/markets"
    params = {"series_ticker": KALSHI_SERIES, "status": "open", "limit": 200}
    async with session.get(KALSHI_REST + path, params=params,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        data = await r.json()

    now = time.time()
    candidates = []
    rejected   = []  # (ticker, bid, ask, ttl, reason) -- for debug if 0 candidates
    for mkt in data.get("markets", []):
        ct = mkt.get("close_time", "")
        ticker = mkt.get("ticker", "?")
        if not ct or mkt.get("status") not in ("open", "active"):
            rejected.append((ticker, None, None, None, "status/close_time"))
            continue
        try:
            epoch = datetime.datetime.fromisoformat(
                ct.replace("Z", "+00:00")).timestamp()
        except Exception:
            rejected.append((ticker, None, None, None, "bad close_time"))
            continue
        try:
            yb = float(mkt.get("yes_bid_dollars") or 0)
            ya = float(mkt.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            rejected.append((ticker, None, None, None, "bad bid/ask"))
            continue
        ttl = epoch - now
        if yb <= 0 or ya <= 0 or ya <= yb:
            rejected.append((ticker, yb, ya, ttl, "degenerate book"))
            continue
        if round(yb * 100) < 1:
            rejected.append((ticker, yb, ya, ttl, "bid rounds <1c"))
            continue
        if round(ya * 100) > 99:
            rejected.append((ticker, yb, ya, ttl, "ask rounds >99c"))
            continue
        if ttl < 60:
            rejected.append((ticker, yb, ya, ttl, f"ttl={ttl:.0f}s <60"))
            continue
        ym = (yb + ya) / 2.0
        candidates.append({
            "ttl_s":   ttl,
            "yes_bid": yb,
            "yes_ask": ya,
            "yes_mid": ym,
            "centrality": abs(ym - 0.5),
            "mkt":     mkt,
        })

    if verbose:
        print(f"  Found {len(candidates)} tradeable candidate market(s) "
              f"out of {len(data.get('markets', []))} returned:")
        for c in sorted(candidates, key=lambda c: c["centrality"])[:10]:
            print(f"    OK   {c['mkt']['ticker']}  bid={c['yes_bid']:.3f} "
                  f"ask={c['yes_ask']:.3f}  mid={c['yes_mid']:.3f}  "
                  f"ttl={c['ttl_s']:.0f}s")
        if not candidates and rejected:
            print(f"  All rejected. First 15 reasons:")
            for t, b, a, ttl, why in rejected[:15]:
                bs = f"{b:.3f}" if b is not None else "?"
                as_ = f"{a:.3f}" if a is not None else "?"
                ts = f"{ttl:.0f}s" if ttl is not None else "?"
                print(f"    SKIP {t}  bid={bs} ask={as_} ttl={ts}  ({why})")

    if not candidates:
        return None
    # Prefer central prices. Among similarly central, prefer more time left
    # (so we don't race the close).
    candidates.sort(key=lambda c: (c["centrality"], -c["ttl_s"]))
    return candidates[0]["mkt"]


async def refresh_market(session: aiohttp.ClientSession, pk,
                         ticker: str) -> dict | None:
    path = f"/trade-api/v2/markets/{ticker}"
    hdrs = _auth_headers(pk, "GET", path)
    async with session.get(KALSHI_REST + path, headers=hdrs,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200:
            return None
        data = await r.json()
    return data.get("market", data)


async def place_order(session: aiohttp.ClientSession, pk,
                      ticker: str, action: str, yes_price_cents: int,
                      count: int, dry_run: bool) -> dict | None:
    """
    action: 'buy' or 'sell'
    yes_price_cents: limit price in cents (1-99)
    count: integer number of contracts

    Body shape matches scripts/test_trade.py exactly (which is known to
    work). Kalshi's default order behaviour (no time_in_force /
    expiration_ts) is 'rest until filled or canceled' -- exactly what we
    want for the maker entry leg. The taker exit leg crosses immediately
    because we set yes_price to the current best bid.

    NOTE on race condition for maker entries: between read and post, the
    ask can drop to/below our limit and the order would cross (= taker
    fill). We detect this after placement: if the response shows
    filled_count > 0 immediately, we report it as a TAKER fill rather
    than silently mis-attributing the trade.
    """
    path = "/trade-api/v2/portfolio/orders"
    body = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "type":            "limit",
        "action":          action,
        "side":            "yes",
        "count":           count,
        "yes_price":       yes_price_cents,
    }

    label = "BUY " if action == "buy" else "SELL"
    if dry_run:
        print(f"  [DRY RUN] {label} {count}x YES @ {yes_price_cents}c")
        print(f"  Body: {json.dumps(body)}")
        return {"order": {"order_id": f"DRY-RUN-{uuid.uuid4().hex[:8]}",
                          "status": "dry_run", "filled_count": 0}}

    hdrs = _auth_headers(pk, "POST", path)
    async with session.post(KALSHI_REST + path, headers=hdrs,
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=15)) as r:
        text = await r.text()
        if r.status not in (200, 201):
            print(f"  ERROR {r.status}: {text}")
            print(f"  Sent: {json.dumps(body)}")
            return None
        return json.loads(text)


async def get_order(session: aiohttp.ClientSession, pk,
                    order_id: str) -> dict | None:
    """Fetch the live state of a previously-placed order."""
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    hdrs = _auth_headers(pk, "GET", path)
    async with session.get(KALSHI_REST + path, headers=hdrs,
                           timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200:
            return None
        data = await r.json()
    return data.get("order", data)


async def cancel_order(session: aiohttp.ClientSession, pk,
                       order_id: str, dry_run: bool) -> bool:
    """DELETE the order. Returns True if cancel call succeeded."""
    if dry_run:
        print(f"  [DRY RUN] CANCEL {order_id}")
        return True
    path = f"/trade-api/v2/portfolio/orders/{order_id}"
    hdrs = _auth_headers(pk, "DELETE", path)
    async with session.delete(KALSHI_REST + path, headers=hdrs,
                              timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status not in (200, 204):
            text = await r.text()
            print(f"  CANCEL ERROR {r.status}: {text}")
            return False
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Maker-entry / taker-exit round-trip on Kalshi")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what we would do, don't place real orders")
    parser.add_argument("--target-usd", type=float, default=1.0,
                        help="Approximate notional for the buy (default $1)")
    parser.add_argument("--ttl",        type=int, default=30,
                        help="Seconds to wait for the maker buy to fill (default 30)")
    parser.add_argument("--hold-s",     type=int, default=10,
                        help="Seconds to hold after fill before selling (default 10)")
    parser.add_argument("--aggressive", action="store_true",
                        help="Post at yes_bid+1 cent (1 tick inside the spread). "
                             "Default posts at yes_bid (joins existing queue).")
    args = parser.parse_args()

    if not KALSHI_KEY_ID:
        sys.exit("ERROR: KALSHI_API_KEY_ID not set in .env")

    pk = load_private_key()

    async with aiohttp.ClientSession() as session:

        # ---- 1. Discover market ----------------------------------------
        print("Discovering active BTC 15-min market with central prices + time to spare...")
        mkt = await discover_market(session, verbose=True)
        if not mkt:
            sys.exit("ERROR: no tradeable KXBTC15M market right now. "
                     "Need: bid >= 1c, ask <= 99c, ask > bid, >= 60s to close. "
                     "Try again after the next 15-minute boundary.")

        ticker     = mkt["ticker"]
        yes_bid    = float(mkt.get("yes_bid_dollars") or 0)
        yes_ask    = float(mkt.get("yes_ask_dollars") or 0)
        floor_str  = mkt.get("floor_strike", "?")
        close_time = mkt.get("close_time", "")
        try:
            close_dt = datetime.datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            tau_s    = max(0, close_dt.timestamp() - time.time())
        except Exception:
            tau_s = 0

        print(f"  Ticker:      {ticker}")
        print(f"  Strike:      ${float(floor_str):,.2f}")
        print(f"  Closes in:   {tau_s:.0f}s")
        print(f"  YES bid/ask: ${yes_bid:.3f} / ${yes_ask:.3f}  (spread {yes_ask - yes_bid:.3f})")

        # Need at least a 2-cent spread for an aggressive post to make sense.
        spread_c = round((yes_ask - yes_bid) * 100)
        if args.aggressive and spread_c < 2:
            print("  WARNING: spread is < 2c -- posting at bid+1 would equal ask "
                  "and effectively be a taker. Falling back to passive.")
            args.aggressive = False

        # ---- 2. Decide post price + size --------------------------------
        if args.aggressive:
            post_cents = round(yes_bid * 100) + 1
            mode_label = "AGGRESSIVE (bid+1c)"
        else:
            post_cents = round(yes_bid * 100)
            mode_label = "PASSIVE (at bid)"

        # Kalshi limits are 1..99c. Clamp defensively even though discovery
        # already rejects markets outside that range.
        post_cents = max(1, min(99, post_cents))

        post_dollars = post_cents / 100.0
        if post_dollars <= 0:
            sys.exit(f"ERROR: post_dollars resolved to {post_dollars} (post_cents={post_cents}). "
                     "This shouldn't happen after discovery filtering -- inspect the market response.")
        count     = max(1, int(round(args.target_usd / post_dollars)))
        gross_buy = count * post_dollars

        print(f"\n  Plan ({mode_label}):")
        print(f"    1. POST BUY  {count}x YES @ {post_cents}c  (~${gross_buy:.2f}) -- maker, GTC")
        print(f"    2. Wait up to {args.ttl}s for fill")
        print(f"    3. If filled: hold {args.hold_s}s, then SELL @ current bid (taker, IOC)")
        print(f"    4. If not filled: cancel the order")
        print(f"  Fees: maker entry = $0; taker exit = THETA_FEE_TAKER * p_exit * (1-p_exit) * notional")
        print(f"        (THETA_FEE_TAKER from config = {THETA_FEE_TAKER})")

        if not args.dry_run:
            confirm = input("\nProceed? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Aborted.")
                return

        # ---- 3. POST the maker buy --------------------------------------
        print(f"\n[1/4] Posting BUY {count}x YES @ {post_cents}c (resting limit)...")
        post_t0  = time.time()
        buy_resp = await place_order(session, pk, ticker, "buy",
                                     post_cents, count, args.dry_run)
        if not buy_resp:
            sys.exit("Place-order failed.")
        buy_order = buy_resp.get("order", buy_resp)
        order_id  = buy_order.get("order_id", buy_order.get("id"))
        if not order_id:
            sys.exit(f"No order_id in response: {buy_resp}")
        initial_filled = int(buy_order.get("filled_count", 0) or 0)
        print(f"    Order ID:       {order_id}")
        print(f"    Initial status: {buy_order.get('status', '?')}")
        print(f"    Filled at post: {initial_filled} / {count}")

        # If the limit somehow crossed and filled fully on placement, we're
        # actually a taker. Skip polling and go straight to hold + sell.
        if initial_filled >= count:
            print(f"    NOTE: order filled fully at placement -- this was a TAKER fill.")
            fill_t        = post_t0
            filled_count  = count
            filled_status = "executed"
        else:
            # ---- 4. POLL for fill --------------------------------------
            print(f"\n[2/4] Polling order status every 1s for up to {args.ttl}s...")
            filled_count  = initial_filled
            filled_status = buy_order.get("status", "resting")
            fill_t        = None

            for elapsed in range(1, args.ttl + 1):
                await asyncio.sleep(1.0)
                if args.dry_run:
                    # Pretend the fill happens at half the TTL for demo flow.
                    if elapsed >= max(1, args.ttl // 2):
                        filled_count = count
                        filled_status = "executed"
                        fill_t = time.time()
                        print(f"    [DRY RUN] Simulated fill at t={elapsed}s "
                              f"(filled {filled_count}/{count})")
                        break
                else:
                    state = await get_order(session, pk, order_id)
                    if not state:
                        print(f"    {elapsed}s: poll failed (no response)")
                        continue
                    new_filled = int(state.get("filled_count", 0) or 0)
                    new_status = state.get("status", "?")
                    if new_filled != filled_count or new_status != filled_status:
                        print(f"    {elapsed}s: status={new_status}  filled={new_filled}/{count}")
                        filled_count  = new_filled
                        filled_status = new_status
                    if filled_count >= count:
                        fill_t = time.time()
                        break

            if filled_count < count:
                # ---- 5a. NO FILL: cancel the order ---------------------
                print(f"\n[3/4] Order did not fully fill within {args.ttl}s "
                      f"(filled {filled_count}/{count}). Canceling...")
                ok = await cancel_order(session, pk, order_id, args.dry_run)
                if ok:
                    print(f"    Order canceled successfully.")
                else:
                    print(f"    !! Cancel call failed -- check Kalshi UI manually.")
                print(f"\n  RESULT: NO_FILL")
                print(f"    Would-have-been-buy: {count}x @ {post_cents}c  (no execution)")
                if filled_count > 0:
                    print(f"    Partial fill: {filled_count} contracts at maker price -- ")
                    print(f"    you now hold {filled_count} YES at ${post_dollars:.3f} entry. "
                          f"Sell manually if desired.")
                if args.dry_run:
                    print("  [DRY RUN -- no real orders placed]")
                return

        # ---- 6. HOLD ----------------------------------------------------
        time_to_fill = fill_t - post_t0 if fill_t else 0.0
        print(f"\n[3/4] FILLED in {time_to_fill:.1f}s. Holding for {args.hold_s}s...")
        for remaining in range(args.hold_s, 0, -1):
            print(f"    {remaining}s...", end="\r", flush=True)
            await asyncio.sleep(1)
        print()

        # ---- 7. SELL (taker) -------------------------------------------
        print("[4/4] Fetching current market for taker exit...")
        fresh = await refresh_market(session, pk, ticker)
        if fresh and not args.dry_run:
            try:
                yes_bid_now = float(fresh.get("yes_bid_dollars") or 0)
                yes_ask_now = float(fresh.get("yes_ask_dollars") or 0)
            except (TypeError, ValueError):
                yes_bid_now = yes_bid
                yes_ask_now = yes_ask
            print(f"    YES bid/ask now: ${yes_bid_now:.3f} / ${yes_ask_now:.3f}")
            sell_cents = round(yes_bid_now * 100)
        else:
            sell_cents = max(1, post_cents)   # dry-run flat
            yes_bid_now = sell_cents / 100.0
            print(f"    [DRY RUN] Using simulated bid: {sell_cents}c")

        print(f"    Placing SELL {count}x YES @ {sell_cents}c (taker)...")
        sell_resp = await place_order(session, pk, ticker, "sell",
                                      sell_cents, count, args.dry_run)
        if not sell_resp:
            print("  SELL FAILED -- position may still be open, check Kalshi UI")
            sys.exit(1)
        sell_order  = sell_resp.get("order", sell_resp)
        sell_id     = sell_order.get("order_id", sell_order.get("id", "?"))
        sell_filled = int(sell_order.get("filled_count", 0) or 0)
        print(f"    Order ID: {sell_id}")
        print(f"    Status:   {sell_order.get('status', '?')}")
        print(f"    Filled:   {sell_filled} / {count}")

        # ---- 8. P&L summary --------------------------------------------
        gross_in  = count * post_dollars
        gross_out = count * (sell_cents / 100.0)
        # maker entry: 0 fee; taker exit: THETA * p_exit * (1 - p_exit) * notional_per_dollar
        sell_p    = sell_cents / 100.0
        exit_fee_per_dollar = THETA_FEE_TAKER * sell_p * (1 - sell_p) if 0 < sell_p < 1 else 0
        exit_fee  = exit_fee_per_dollar * gross_out
        gross_pnl = gross_out - gross_in
        net_pnl   = gross_pnl - exit_fee

        print(f"\n  RESULT: FILLED + EXITED")
        print(f"    Bought {count}x @ {post_cents}c (maker) = ${gross_in:.4f}")
        print(f"    Sold   {count}x @ {sell_cents}c (taker) = ${gross_out:.4f}")
        print(f"    Gross P&L:                              ${gross_pnl:+.4f}")
        print(f"    Taker exit fee (THETA*p*(1-p)*notional): -${exit_fee:.4f}")
        print(f"    Net P&L (maker entry, taker exit):      ${net_pnl:+.4f}")
        print(f"    Time-to-fill: {time_to_fill:.1f}s")

        if args.dry_run:
            print("  [DRY RUN -- no real orders placed]")


if __name__ == "__main__":
    asyncio.run(main())
