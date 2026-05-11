# PolymarketBot (Kalshi branch)

Lag-arbitrage bot for Kalshi's 15-minute BTC/ETH/SOL/XRP binary markets.

**Edge:** Coinbase / Binance spot price leads Kalshi's order book by ~15–90s.
The bot buys when Kalshi is cheap relative to where spot says it should be,
exits when the gap closes. Strategy is market-neutral — direction of BTC
doesn't matter, only whether Kalshi catches up before the window closes.

**How it actually trades:**
- **Maker entry** — posts a resting limit at `bid + 1c` and waits up to 5s for someone to hit it. Pays $0 fee on entry.
- **Taker exit** — sweeps the bid IOC when edge closes (or after 15s max). Pays Kalshi taker fee `0.07 × p × (1−p)` on exit only.
- **WebSocket-driven** — both Kalshi book and Binance/Coinbase spot are streamed in real time.

---

## Workflow

```
1. COLLECT  — 24h+ dry run via WebSocket. Writes parquet ticks with top-10 depth.
              python scripts/run/run_kalshi_bot.py --assets BTC

2. TRAIN    — Fit a 17-feature LightGBM model (13 lag/momentum + 4 depth).
              python scripts/train_model.py --run data/<run> --asset BTC

3. TUNE     — Sweep Kelly tier (edge_floor, wallet_fraction) configs and
              paste the winning KELLY_TIERS into config.py.
              python scripts/tune.py --run data/<run> --asset BTC \
                --model-file model_fits/<dir>/model.pkl

4. BACKTEST — Confirm the tuned config: full per-trade P&L, exit reasons.
              python scripts/backtest.py --run data/<run> --asset BTC \
                --model-file model_fits/<dir>/model.pkl

5. GO LIVE  — Static model, trades from tick 1 with the tuned tiers.
              python scripts/run/run_kalshi_bot.py --assets BTC --live-orders \
                --model-file model_fits/<dir>/model.pkl
```

See **RUN.md** for exact commands and expected output at each step.

---

## How it works

1. **Spot feed** (Binance default, Coinbase fallback) streams microprice via WebSocket.
2. **Kalshi feed** subscribes to `orderbook_delta` over WebSocket, reconstructs full L2 book in `KalshiBook`.
3. The **sampler loop** writes one parquet row every 100ms with: spot prices, Kalshi top-of-book, top-10 depth on each side, tau, and the active window ticker.
4. The **decision loop** runs every 500ms. It builds a 17-feature vector, predicts `q_settled` (10s ahead probability), computes edge for both YES and NO sides assuming maker entry / taker exit, picks the favored side, and fires a maker post if edge clears a Kelly tier.
5. The **maker workflow** posts at `bid+1c`, polls `get_order` every 0.5s up to 5s, and cancels on TTL. Real fills set the `Position` state with the actual fill price/cost.
6. The **exit logic** monitors edge_now every tick. When edge compresses below the threshold (or 15s elapses), it sends a taker IOC sweep at the bid. Real proceeds = `count − taker_fill_cost_dollars` (Kalshi reciprocal pricing).
7. Models are **static** — never retrained live. Collect more dry-run data and retrain offline if needed.

---

## Quick start

```bash
# 1. Set up credentials
cp .env.example .env
# edit .env: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_FILE, SPOT_SOURCE=binance

# 2. Install deps
pip install -r requirements.txt

# 3. Smoke tests
python scripts/test/check_kalshi_balance.py     # auth works
python scripts/test/watch_kalshi_ws.py          # live book streaming
python scripts/test/test_trade_maker.py         # $1 maker round-trip

# 4. Start a 24h+ dry run for fresh training data
python scripts/run/run_kalshi_bot.py --assets BTC
```

---

## Multi-asset

```bash
# Dry run across multiple assets (one model trained per asset)
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP

# Live with per-asset models
python scripts/run/run_kalshi_bot.py \
  --assets BTC ETH SOL \
  --live-orders \
  --model-file BTC=model_fits/btc/model.pkl,ETH=model_fits/eth/model.pkl,SOL=model_fits/sol/model.pkl
```

Each asset gets its own `KalshiBook`, WebSocket subscription, and `Scheduler`.
One Binance/Coinbase WebSocket connection feeds all `SpotBook`s.

---

## Files

| Path | Purpose |
|---|---|
| `betbot/kalshi/scheduler.py` | Decision/sampler/window loops, maker entry workflow |
| `betbot/kalshi/orders.py` | `place_order` (IOC sweep), `place_resting_limit`, `get_order`, `cancel_order` |
| `betbot/kalshi/kalshi_ws_feed.py` | WebSocket orderbook subscription with snapshot + delta replay |
| `betbot/kalshi/kalshi_rest_feed.py` | REST 100ms polling fallback (--rest flag) |
| `betbot/kalshi/coinbase_feed.py` / `binance_feed.py` | spot WS feeds |
| `betbot/kalshi/book.py` | `KalshiBook` (L2 reconstruction), `SpotBook` |
| `betbot/kalshi/features.py` | 17-feature vector builder |
| `betbot/kalshi/model.py` | LightGBM multi-horizon model with back-compat for 13-feature models |
| `betbot/kalshi/tick_logger.py` | Parquet writer with snappy compression + top-10 depth columns |
| `betbot/kalshi/config.py` | All trading knobs and defaults |
| `scripts/run/run_kalshi_bot.py` | Main entrypoint |
| `scripts/train_model.py` | Train a 17-feature LightGBM on parquet ticks |
| `scripts/backtest.py` | **NEW** realistic maker-entry / taker-exit backtest |
| `scripts/tune.py` | **NEW** Kelly tier sweeper using the new backtest |
| `scripts/watch_decisions.py` | Live colored event feed + status bar |
| `scripts/test/test_trade.py` | $1 taker round-trip via WS |
| `scripts/test/test_trade_maker.py` | $1 maker entry / taker exit via WS |
| `scripts/test/watch_kalshi_ws.py` | Standalone WS book viewer |
| `scripts/analysis/tick_loader.py` | Unified parquet/csv tick loader |
| `scripts/analysis/analyze_run.py` | Post-run multi-panel charts |
| `scripts/analysis/live_plot.py` | Rolling chart during a live run |
| `data/<run>/` | Per-run parquet ticks + decision logs (gitignored) |
| `model_fits/<run>/` | Saved model.pkl + model.json (gitignored) |
| `CLAUDE.md` | Architecture, model design, full config reference |
| `RUN.md` | Step-by-step operational instructions |
