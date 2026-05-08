"""
run_kalshi_bot.py - Entry point for the Kalshi lead-lag arbitrage bot.

Single-asset mode (default, BTC only):
  python scripts/run/run_kalshi_bot.py

Multi-asset mode (BTC + ETH + SOL + XRP, one independent model each):
  python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP

Each run creates a timestamped folder under data/, e.g.:
  data/2026-05-07_00-15-00_BTC_ETH/
    ticks_BTC.csv
    decisions_BTC.jsonl
    ticks_ETH.csv
    decisions_ETH.jsonl
"""

import argparse
import asyncio
import datetime
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from betbot.kalshi.auth import load_private_key
from betbot.kalshi.book import SpotBook, KalshiBook
from betbot.kalshi.coinbase_feed import CoinbaseFeed
from betbot.kalshi.binance_feed import BinanceFeed
from betbot.kalshi.kalshi_rest_feed import KalshiRestFeed
from betbot.kalshi.kalshi_ws_feed   import KalshiWsFeed
from betbot.kalshi.scheduler import Scheduler, _discover_market
from betbot.kalshi.config import (
    KALSHI_KEY_ID, SPOT_SOURCE,
    KALSHI_ASSETS, COINBASE_PRODUCTS,
    WALLET_BALANCE,
)
from betbot.kalshi.orders import get_balance_usd
from betbot.kalshi.model import load_model

MODEL_FITS_DIR = Path(__file__).resolve().parents[2] / "model_fits"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress LightGBM's own logger (it prints noisy warnings during inference)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
log = logging.getLogger("kalshi_bot")

ALL_ASSETS = list(KALSHI_ASSETS.keys())   # ["BTC", "ETH", "SOL", "XRP"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def parse_args():
    p = argparse.ArgumentParser(description="Kalshi lag-arbitrage bot")
    p.add_argument("--assets", nargs="+", default=["BTC"],
                   choices=ALL_ASSETS,
                   help="Assets to trade (default: BTC). "
                        "Example: --assets BTC ETH SOL XRP")
    p.add_argument("--wallet", type=float, default=1000.0,
                   help="Simulated wallet USD (split equally across assets)")
    p.add_argument("--fresh", action="store_true",
                   help="Delete existing decisions logs before starting. "
                        "Ticks files are KEPT so models can bootstrap.")
    p.add_argument("--live-orders", action="store_true",
                   help="REAL MONEY: place actual Kalshi orders. "
                        "Reads real Kalshi balance as wallet.")
    p.add_argument("--max-bet-pct", type=float, default=0.10,
                   help="Hard cap on per-trade bet, fraction of per-asset wallet")
    p.add_argument("--daily-loss-pct", type=float, default=0.05,
                   help="Halt when realized loss exceeds this fraction of per-asset wallet")
    p.add_argument("--rest", action="store_true",
                   help="Use Kalshi REST 100ms polling instead of WebSocket. "
                        "Default: WebSocket (lower latency).")
    p.add_argument("--model-file", type=str, default=None,
                   help="Path to a pre-trained model .pkl (from model_fits/). "
                        "Skips live warmup — bot trades from tick 1. "
                        "Example: --model-file model_fits/btc_lgbm_v1.pkl  "
                        "For per-asset: --model-file BTC=model_fits/btc.pkl,ETH=model_fits/eth.pkl")
    return p.parse_args()


def _parse_model_files(model_file_arg: str | None, assets: list[str]) -> dict[str, str | None]:
    """
    Parse --model-file into a per-asset dict.
    Formats:
      "model_fits/foo.pkl"            -> all assets use this model
      "BTC=model_fits/b.pkl,ETH=..."  -> per-asset
    Returns {asset: path_or_none}.
    """
    result = {a: None for a in assets}
    if not model_file_arg:
        return result

    if "=" in model_file_arg:
        for part in model_file_arg.split(","):
            part = part.strip()
            if "=" in part:
                a, p = part.split("=", 1)
                result[a.upper()] = p.strip()
    else:
        for a in assets:
            result[a] = model_file_arg

    return result


def _load_model_for_asset(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        # Try relative to MODEL_FITS_DIR, then repo root, then cwd
        for base in (MODEL_FITS_DIR, Path(__file__).resolve().parents[2], Path.cwd()):
            candidate = base / p
            if candidate.exists() or Path(str(candidate) + ".pkl").exists():
                p = candidate
                break
    pkl = str(p) if str(p).endswith(".pkl") else str(p) + ".pkl"
    if not Path(pkl).exists():
        sys.exit(f"ERROR: model file not found: {pkl}")
    return load_model(pkl)


async def setup_asset(asset: str, pk, wallet_per_asset: float,
                      run_dir: Path, fresh: bool,
                      live_orders: bool, max_bet_pct: float,
                      daily_loss_pct: float,
                      spot_book: SpotBook,
                      preloaded_model=None,
                      use_ws: bool = False):
    """
    Discover the active Kalshi market for one asset, wire up its
    KalshiBook + KalshiRestFeed + Scheduler, and return them.
    """
    series = KALSHI_ASSETS[asset]

    print(f"  [{asset}] Discovering active Kalshi 15m market ({series})...", flush=True)
    mkt = None
    attempt = 0
    while mkt is None:
        mkt = await _discover_market(series, prefer_strike=None)
        if mkt:
            break
        attempt += 1
        print(f"  [{asset}] No active market (attempt {attempt}, {attempt*5}s elapsed) - waiting 5s...",
              flush=True)
        await asyncio.sleep(5)

    ticker       = mkt["ticker"]
    floor_strike = float(mkt.get("floor_strike") or 0.0)
    ct_str       = mkt.get("close_time", "")
    try:
        close_time = datetime.datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
    except Exception:
        close_time = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15)

    print(f"  [{asset}] Ticker: {ticker}  Strike: {floor_strike:,.4f}  "
          f"Closes: {close_time.strftime('%H:%M:%S UTC')}", flush=True)

    kb_book = KalshiBook()
    kb_book.set_window(ticker, floor_strike, close_time)
    if use_ws:
        kalshi_feed = KalshiWsFeed(kb_book, key_id=KALSHI_KEY_ID, pk=pk)
    else:
        kalshi_feed = KalshiRestFeed(kb_book, key_id=KALSHI_KEY_ID, pk=pk)

    log_path  = run_dir / f"decisions_{asset}.jsonl"
    tick_path = run_dir / f"ticks_{asset}.parquet"

    if fresh and log_path.exists():
        log_path.unlink()
        print(f"  [{asset}] Deleted old {log_path}", flush=True)

    scheduler = Scheduler(
        spot_book, kb_book, kalshi_feed,
        wallet_usd=wallet_per_asset,
        log_path=log_path,
        tick_path=tick_path,
        pk=pk,
        live_orders=live_orders,
        max_bet_pct=max_bet_pct,
        daily_loss_limit_pct=daily_loss_pct,
        series=series,
        preloaded_model=preloaded_model,
    )

    return kalshi_feed, scheduler, asset



async def main():
    args = parse_args()
    assets = [a.upper() for a in args.assets]

    # Create timestamped run folder: data/YYYY-MM-DD_HH-MM-SS_BTC_ETH/
    ts_str  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts_str}_{'_'.join(assets)}"
    run_dir  = DATA_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== Kalshi Lead-Lag Arbitrage Bot ===", flush=True)
    print(f"  Assets:     {', '.join(assets)}", flush=True)
    print(f"  Spot feed:  {SPOT_SOURCE}", flush=True)
    print(f"  Run folder: {run_dir}", flush=True)

    # ---- Pre-load models if --model-file was specified ----
    model_file_map = _parse_model_files(args.model_file, assets)
    preloaded_models: dict[str, object] = {}
    for asset in assets:
        mf = model_file_map.get(asset)
        if mf:
            print(f"  [{asset}] Loading pre-trained model from {mf}...", flush=True)
            preloaded_models[asset] = _load_model_for_asset(mf)
        else:
            preloaded_models[asset] = None
    if any(preloaded_models.values()):
        print(f"  Mode: PRELOADED MODEL -- bot trades from tick 1, no warmup needed",
              flush=True)
    else:
        print(f"  Mode: DRY RUN (data collection) -- no model loaded, bot will abstain. "
              f"Pass --model-file to enable trading.", flush=True)

    pk = load_private_key()

    # ---- Wallet split ----
    wallet_usd = float(args.wallet)
    if args.live_orders:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            real_balance = await get_balance_usd(s, pk)
        if real_balance <= 0:
            sys.exit("ERROR: --live-orders requires positive Kalshi balance.")
        wallet_usd = min(real_balance, WALLET_BALANCE)
        print(f"  Live wallet (Kalshi balance): ${real_balance:,.2f}  capped to: ${wallet_usd:,.2f}  (WALLET_BALANCE in config.py)", flush=True)
    else:
        print(f"  Wallet:     ${wallet_usd:,.0f} (simulated, split across {len(assets)} assets)",
              flush=True)

    wallet_per_asset = wallet_usd / len(assets)
    print(f"  Per-asset:  ${wallet_per_asset:,.2f}", flush=True)

    # ---- One SpotBook per asset (all fed by one CoinbaseFeed WS) ----
    spot_books: dict[str, SpotBook] = {asset: SpotBook() for asset in assets}
    product_to_book = {COINBASE_PRODUCTS[asset]: spot_books[asset] for asset in assets}

    if SPOT_SOURCE == "coinbase":
        spot_feed = CoinbaseFeed(books=product_to_book)
    elif SPOT_SOURCE == "binance":
        if len(assets) == 1 and assets[0] == "BTC":
            spot_feed = BinanceFeed(spot_books["BTC"])
        else:
            print("  WARNING: Binance multi-asset not yet supported; falling back to Coinbase.",
                  flush=True)
            spot_feed = CoinbaseFeed(books=product_to_book)
    else:
        raise RuntimeError(f"Unknown SPOT_SOURCE: {SPOT_SOURCE!r}")

    # ---- Per-asset setup - all assets discover markets in parallel ----
    use_ws = not args.rest
    results = await asyncio.gather(*[
        setup_asset(
            asset, pk, wallet_per_asset,
            run_dir, args.fresh,
            args.live_orders, args.max_bet_pct, args.daily_loss_pct,
            spot_books[asset],
            preloaded_model=preloaded_models[asset],
            use_ws=use_ws,
        )
        for asset in assets
    ])

    asset_configs = [
        (name, kalshi_feed, scheduler)
        for kalshi_feed, scheduler, name in results
        if kalshi_feed is not None
    ]

    if not asset_configs:
        sys.exit("ERROR: no assets could be initialized.")

    kalshi_kind = "REST" if args.rest else "WS"
    print(f"\n  Starting feeds ({SPOT_SOURCE} WS + {len(asset_configs)}x Kalshi {kalshi_kind})...\n",
          flush=True)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(spot_feed.run(), name=f"{SPOT_SOURCE}_feed")
            for asset, kalshi_feed, scheduler in asset_configs:
                tg.create_task(kalshi_feed.run(), name=f"kalshi_rest_{asset}")
                tg.create_task(scheduler.run(),   name=f"scheduler_{asset}")
    except* KeyboardInterrupt:
        pass
    except* asyncio.CancelledError:
        pass
    finally:
        spot_feed.stop()
        for _, kalshi_feed, scheduler in asset_configs:
            kalshi_feed.stop()
            scheduler.stop()
        print("\nBot stopped.", flush=True)
        print(f"Data saved to: {run_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
