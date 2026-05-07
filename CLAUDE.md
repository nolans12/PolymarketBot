# CLAUDE.md - Kalshi Lag-Arb Bot

## What this is

A bot that trades **Kalshi 15-minute BTC/ETH/SOL/XRP binary markets** by exploiting the lag between Coinbase spot price moves and Kalshi's order-book repricing. Coinbase leads; Kalshi follows with a ~15-90s delay. We buy when Kalshi hasn't caught up yet, exit when it does.

**No forecasting.** We predict where Kalshi's *quote* is heading, not BTC's price.

---

## Repo layout

```
betbot/kalshi/          <- the entire live bot
  config.py             trading knobs + env vars (Kelly tiers, thresholds, spot source)
  auth.py               Kalshi RSA-PSS signing
  book.py               SpotBook + KalshiBook (state)
  coinbase_feed.py      Coinbase WS -> SpotBook (multi-asset, one connection)
  binance_feed.py       Binance WS -> SpotBook (alternative spot feed)
  kalshi_rest_feed.py   Kalshi REST 1Hz polling -> KalshiBook
  features.py           FeatureVec + build_features()
  model.py              LGBMModel (LightGBM multi-horizon, static — never retrained live)
  tick_logger.py        CSV tick writer (dry-run data collection)
  scheduler.py          sampler (CSV only) / decision / window manager

scripts/run/
  run_kalshi_bot.py     main entrypoint (single or multi-asset)

scripts/
  train_model.py        fit LightGBM on dry-run ticks, save to model_fits/
  tune_trading_knobs.py grid-sweep Kelly tiers + exit thresholds to maximise P&L
  test_all.py           full simulation with loaded model + tuned config
  replay_window.py      animated replay of one 15-min window with model projections

scripts/analysis/
  analyze_run.py        visualize decisions.jsonl
  live_plot.py          live rolling spot/Kalshi/model-prediction chart
  pick_run.py           shared run-folder picker (Tkinter popup)

scripts/test/
  test_trade.py         real $1 round-trip sanity check
  test_maker_trade.py   maker-entry round-trip test
  check_kalshi_balance.py   auth + balance smoke check

data/<YYYY-MM-DD_HH-MM-SS>_<ASSETS>/   output from each bot run (gitignored)
  ticks_BTC.csv         1Hz raw ticks (spot + Kalshi book)
  decisions_BTC.jsonl   decision log

model_fits/<run>_<ASSET>_<date>/        saved trained models (gitignored)
  model.pkl             LightGBM model weights
  model.json            metadata: R2 scores, feature names, horizons
```

---

## Offline workflow (train → tune → verify → deploy)

```
Step 1 — Dry run (collect training data):
  python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP
  No model needed. Collects ticks to data/<run>/ticks_{ASSET}.csv.
  Let it run for 24+ hours per asset.

Step 2 — Train model (one per asset):
  python scripts/train_model.py --run data/<run>
  Fits LightGBM on the collected ticks.
  Saves to model_fits/<run>_<ASSET>_<date>/model.pkl

Step 3 — Tune trading knobs:
  python scripts/tune_trading_knobs.py --model-file model_fits/<dir>/model.pkl
  Sweeps Kelly tier configs + exit thresholds on the held-out data.
  Prints a config.py snippet. Apply to betbot/kalshi/config.py manually.

Step 4 — Full simulation:
  python scripts/test_all.py --model-file model_fits/<dir>/model.pkl
  Replays ticks with the model + tuned config. Verify P&L and exit breakdown.

Step 5 — Go live:
  python scripts/run/run_kalshi_bot.py --model-file model_fits/<dir>/model.pkl --live-orders
  Static model, no warmup, trades from tick 1.
```

---

## Strategy

Every 0.5 seconds the decision loop builds a feature vector of lagged Coinbase microprice log-ratios vs strike, Kalshi microstructure, and time-to-close. A **LightGBM multi-horizon model** (trained offline) predicts `logit(yes_mid_{t+h})` for h in [5, 10, 15, 60]s. The primary horizon (10s) feeds `q_settled`. Edge = `q_settled - yes_ask - fees`. If edge clears a Kelly tier floor, enter; exit when edge compresses back to zero, spot reverses past the stop threshold, or the window is about to close.

The model is **static**: trained offline on dry-run data, never retrained during a live run. The sampler still writes ticks to CSV during a live run so you can collect more training data if needed.

---

## Features (13 total, in `features.py`)

| Feature | Description |
|---------|-------------|
| `x_0` ... `x_30` | `log(microprice_lag / K)` at 0, 5, 10, 15, 20, 25, 30s lags |
| `tau_s` | seconds until window close |
| `inv_sqrt_tau` | `1 / sqrt(tau + 1)` |
| `kalshi_spread` | `yes_ask - yes_bid` |
| `kalshi_lag_5s/10s/30s` | `yes_mid_now - yes_mid_{t-Ns}` |

---

## Config knobs (`config.py` + `.env`)

Only operational tunables live here — no training/model parameters.

| Key | Default | What it does |
|-----|---------|------|
| `SPOT_SOURCE` | `coinbase` | `coinbase` or `binance` |
| `KALSHI_API_KEY_ID` | required | Kalshi key |
| `KALSHI_PRIVATE_KEY_FILE` | required | path to .pem |
| `KELLY_TIERS` | see config | `[(edge_floor, wallet_fraction), ...]` — tune with `tune_trading_knobs.py` |
| `LAG_CLOSE_THRESHOLD` | 0.005 | exit when edge falls below this |
| `STOP_THRESHOLD` | 0.03 | stop-loss: exit if edge erodes by this |
| `FALLBACK_TAU_S` | 60 | hold to resolution at this tau |
| `MAX_HOLD_S` | 90 | force-exit after this many seconds |
| `THETA_FEE_TAKER` | 0.07 | fee model: `theta * p * (1-p)` per leg |
| `ENTRY_MODE` | `taker` | `taker` or `maker` |
| `DRY_RUN` | `true` | set `false` for real orders |

---

## Decision to advance to live trading

Advance when all of these hold on `test_all.py` simulation:

- R2_hld (10s horizon) > 0.25
- Net P&L positive after fees
- Win rate > 50%
- >50% of exits are `lag_closed`
- Avg hold time 15-90s

---

## Known issues / caveats

- `THETA_FEE_TAKER = 0.07` is from Kalshi's published schedule, not calibrated from real fills. Run `scripts/test/test_trade.py` before trusting simulation P&L.
- Slippage model is crude (`min(0.02, size/depth * 0.01)`). Will understate at tier-1 sizing on thin books.
- Feed staleness: `cb.ready` and `kb.ready` don't check recency within a tick. If a feed goes silent without disconnecting, the bot won't notice until the WS drops.
- LGBM needs 60s+ of future data per sample for multi-horizon targets. Very short dry runs (<5 min) won't produce enough targets to fit.
