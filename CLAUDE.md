# CLAUDE.md - Kalshi Lag-Arb Bot

## What this is

A bot that trades **Kalshi 15-minute BTC/ETH/SOL/XRP binary markets** by exploiting the lag between Coinbase/Binance spot price moves and Kalshi's order-book repricing. Spot leads; Kalshi follows with a ~15-90s delay. We post a maker bid at `bid+1c` when the model says edge exists, exit as taker IOC when the gap closes.

**No forecasting.** We predict where Kalshi's *quote* is heading, not BTC's price.

---

## Repo layout

```
betbot/kalshi/          <- the entire live bot
  config.py             trading knobs + env vars (Kelly tiers, maker TTL, spot source)
  auth.py               Kalshi RSA-PSS signing
  book.py               SpotBook + KalshiBook (state)
  coinbase_feed.py      Coinbase WS -> SpotBook (multi-asset, one connection)
  binance_feed.py       Binance WS -> SpotBook (single-asset BTC for now)
  kalshi_ws_feed.py     Kalshi WS orderbook_delta -> KalshiBook (default, ms-latency)
  kalshi_rest_feed.py   Kalshi REST 100ms polling -> KalshiBook (--rest fallback)
  features.py           FeatureVec (17 features) + build_features()
  model.py              LGBMModel (LightGBM multi-horizon, static — never retrained live)
  tick_logger.py        Parquet writer with snappy compression + top-10 depth columns
  orders.py             place_order (IOC sweep), place_resting_limit, get_order, cancel_order
  scheduler.py          sampler / decision / window manager + maker entry workflow

scripts/run/
  run_kalshi_bot.py     main entrypoint (single or multi-asset)

scripts/
  train_model.py        fit LightGBM on parquet ticks, save to model_fits/
  backtest.py           NEW: realistic maker-entry / taker-exit backtest
  tune.py               NEW: sweep Kelly tier configs via the new backtest
  watch_decisions.py    live colored event feed + status bar

scripts/test/
  test_trade.py             real $1 taker round-trip via WS
  test_trade_maker.py       real $1 maker entry / taker exit via WS
  watch_kalshi_ws.py        standalone WS book viewer with depth
  check_kalshi_balance.py   auth + balance smoke check

scripts/analysis/
  tick_loader.py        unified parquet (preferred) / csv (legacy) loader
  pick_run.py           shared run-folder picker (Tkinter popup)
  analyze_run.py        post-run multi-panel charts
  live_plot.py          rolling chart while bot runs
  replay_window.py      animated single-window replay with model projections

data/<YYYY-MM-DD_HH-MM-SS>_<ASSETS>/   output from each bot run (gitignored)
  ticks_BTC.parquet     10Hz raw ticks with top-10 depth on each side (snappy)
  decisions_BTC.jsonl   decision log

model_fits/<run>_<ASSET>_<date>/        saved trained models (gitignored)
  model.pkl             LightGBM model weights (17 features now; legacy 13-feature also loadable)
  model.json            metadata: R2 scores, feature names, horizons
```

---

## Offline workflow

```
1. COLLECT — 24h+ of fresh WS ticks (parquet + top-10 depth):
     python scripts/run/run_kalshi_bot.py --assets BTC

2. TRAIN — fit a 17-feature LightGBM model:
     python scripts/train_model.py --run data/<run> --asset BTC

3. TUNE — sweep Kelly tier (edge_floor, wallet_fraction) configurations:
     python scripts/tune.py --run data/<run> --asset BTC \
       --model-file model_fits/<dir>/model.pkl
   Paste the suggested KELLY_TIERS into config.py.

4. BACKTEST — confirm the tuned config (realistic maker/taker fill model):
     python scripts/backtest.py --run data/<run> --asset BTC \
       --model-file model_fits/<dir>/model.pkl

5. GO LIVE:
     python scripts/run/run_kalshi_bot.py --assets BTC --live-orders \
       --model-file model_fits/<dir>/model.pkl
```

---

## Strategy

Every 500ms the decision loop builds a 17-feature vector, runs it through a
**LightGBM multi-horizon model** (trained offline) that predicts
`logit(yes_mid_{t+h})` for h in [5, 10, 15, 60]s. The 10s prediction
(`q_settled`) feeds edge calc.

**Edge math (maker entry, taker exit):**
- YES side: `edge_yes = q_set − (yes_bid + 1c) − taker_fee(q_set)`
- NO side: `edge_no = (1 − q_set) − ((1 − yes_ask) + 1c) − taker_fee(1 − q_set)`
- Maker entry fee = $0; only the taker exit pays the Kalshi fee
- Taker fee per dollar = `0.07 × p × (1 − p)`

If `max(edge_yes, edge_no)` clears a Kelly tier floor, the bot:
1. Posts a resting limit at `bid + 1c` on the favored side
2. Polls `get_order` every 0.5s for up to `MAKER_TTL_S` (5s)
3. If filled (even partially) → set `Position` with the actual fill price/cost
4. If TTL expires → cancel + back off

The model is **static**: trained offline on parquet ticks, never retrained
during a live run. The sampler still writes ticks during a live run so you
can collect more training data if needed.

---

## Features (17 total, in `features.py`)

| Feature | Description |
|---------|-------------|
| `x_0` … `x_30` | `log(microprice_lag / K)` at 0, 5, 10, 15, 20, 25, 30s lags (7 cols) |
| `tau_s` | seconds until window close |
| `inv_sqrt_tau` | `1 / sqrt(tau + 1)` |
| `kalshi_spread` | `yes_ask − yes_bid` |
| `kalshi_lag_5s/10s/30s` | `yes_mid_now − yes_mid_{t-Ns}` |
| `yes_bid_size` | size at top YES bid (contracts) |
| `yes_ask_size` | size at top NO bid (= YES ask size) |
| `yes_depth_5c` | total YES bid contracts within 5c of best |
| `no_depth_5c` | total NO bid contracts within 5c of best |

The 4 depth features are populated only when the WebSocket feed is in use
(default). Models trained on legacy CSV runs without depth still load — the
inference path truncates the feature vector to match what the saved scaler
expects.

---

## Config knobs (`config.py` + `.env`)

| Key | Default | What it does |
|-----|---------|------|
| `SPOT_SOURCE` | `binance` | `coinbase` or `binance` (binance needs non-US IP) |
| `KALSHI_API_KEY_ID` | required | Kalshi key |
| `KALSHI_PRIVATE_KEY_FILE` | required | path to .pem |
| `ENTRY_MODE` | `maker` | `maker` (post resting) or `taker` (legacy IOC sweep) |
| `MAKER_AT_BID_PLUS_1` | `True` | post at bid+1c (faster fills) vs at bid |
| `MAKER_TTL_S` | `5.0` | cancel maker post if not filled within this many seconds |
| `MAKER_POLL_S` | `0.5` | how often to poll order state during the wait |
| `KELLY_TIERS` | see config | `[(edge_floor, wallet_fraction), ...]` — tune with `tune.py` |
| `LAG_CLOSE_THRESHOLD` | 0.005 | exit when edge falls below this |
| `STOP_THRESHOLD` | None | optional stop-loss (edge erosion); None = disabled |
| `FALLBACK_TAU_S` | 60 | hold to resolution at this tau |
| `MAX_HOLD_S` | 15 | force-exit after this many wall-clock seconds |
| `THETA_FEE_TAKER` | 0.07 | fee model: `theta * p * (1-p)` per dollar bet, taker only |
| `WALLET_BALANCE` | 100.0 | wallet cap (real Kalshi balance can be higher) |
| `SIZE_MAX_USD` | 2.0 | per-order ceiling |

---

## Decision to advance to live trading

Advance when all of these hold from `backtest.py`:

- `R2_hld(10s) > 0.25`
- Net P&L positive after fees
- Win rate > 50%
- > 50% of exits are `exit_lag_closed`
- Avg hold time 5–15s
- Maker fill rate > 30%

---

## Critical Kalshi quirks (learned the hard way)

1. **Order response uses `fill_count_fp` not `filled_count`.** This was a months-long bug where the bot thought every entry failed but real money was filling. Always read `fill_count_fp` (a string like `"3.00"`).

2. **`taker_fill_cost_dollars` on a SELL is what the BUYER paid, not your proceeds.** Kalshi reciprocal pricing: real proceeds = `fill_count_fp − taker_fill_cost_dollars`.

3. **`type: "market"` was removed in Feb 2026.** Use limit + IOC at extreme prices (99 for buys, 1 for sells) to express market-like sweep behavior.

4. **WS subscribe message format**: `{"id": N, "cmd": "subscribe", "params": {"channels": ["orderbook_delta"], "market_tickers": ["..."]}}`. Anything else returns "Unknown command".

5. **The orderbook only contains bids on each side.** YES "ask" is synthetic: `1 − max(no_book)`. To buy YES at price P, you can post a YES bid at P or a NO ask (= 1−P) — they match against opposite-side bids.

6. **Maker post at the bid is unlikely to fill quickly** — there's a queue. Posting at `bid+1c` (one tick inside the spread) is usually the right tradeoff between cost and fill probability.

---

## Known issues / caveats

- `THETA_FEE_TAKER = 0.07` is the published Kalshi schedule; live trades should match within rounding. The actual fee is in `taker_fees_dollars` in every order response — `compute_pnl` in the bot uses real values.
- Slippage model is conservative (`min(0.02, size/depth × 0.01)`). Real fills often beat this on fat books.
- Maker fill simulator in `backtest.py` is **pessimistic (Model A)** — fills only when the live ask drops to your post price. Real fills are higher because makers also catch order-flow that walks into them at-bid. Backtest is therefore a lower bound on fill rate.
- Feed staleness: stale checks fire only after a feed has been ready and gone silent — won't false-alarm during startup. WS auto-reconnects on >10s silence and on sequence gaps.
- LGBM needs 60s+ of future data per sample for multi-horizon targets. Very short dry runs (< 5 min) won't produce enough targets to fit.
