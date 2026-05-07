# CLAUDE.md - Kalshi Lag-Arb Bot

## What this is

A bot that trades **Kalshi 15-minute BTC/ETH/SOL/XRP binary markets** by exploiting the lag between Coinbase spot price moves and Kalshi's order-book repricing. Coinbase leads; Kalshi follows with a ~15-90s delay. We buy when Kalshi hasn't caught up yet, exit when it does.

**No forecasting.** We predict where Kalshi's *quote* is heading, not BTC's price.

---

## Repo layout

```
betbot/kalshi/          <- the entire live bot
  config.py             config + env vars
  auth.py               Kalshi RSA-PSS signing
  book.py               SpotBook + KalshiBook (state)
  coinbase_feed.py      Coinbase WS -> SpotBook (multi-asset, one connection)
  binance_feed.py       Binance WS -> SpotBook (alternative spot feed)
  kalshi_rest_feed.py   Kalshi REST 1Hz polling -> KalshiBook
  features.py           FeatureVec + build_features()
  model.py              KalshiRegressionModel (RidgeCV)
  training_buffer.py    rolling (X, y, ts) buffer
  tick_logger.py        1Hz CSV writer
  scheduler.py          sampler / refitter / decision / window manager
  orders.py             Kalshi order placement (Phase 2)

scripts/run/
  run_kalshi_bot.py     main entrypoint

scripts/analysis/
  analyze_run.py        visualize decisions.jsonl
  backtest.py           walk-forward replay on ticks.csv
  compare_models.py     baseline vs projection features head-to-head
  replay_window.py      animated replay of one 15-min window
  live_plot.py          live rolling spot/Kalshi plot
  pick_run.py           shared run-folder picker (Tkinter popup)

scripts/test/
  test_trade.py         real $1 round-trip sanity check
  test_maker_trade.py   maker-entry round-trip test
  check_kalshi_balance.py   auth + balance smoke test

data/<timestamp>_<ASSETS>/   output from each bot run
  ticks_BTC.csv         1Hz raw ticks (spot + Kalshi book)
  decisions_BTC.jsonl   10s decision log
```

---

## Strategy in one paragraph

Every 10 seconds the scheduler builds a feature vector of lagged Coinbase microprice log-ratios vs strike, Kalshi microstructure, and time-to-close. A `RidgeCV` regression predicts `logit(yes_mid)` from these features. `q_settled` is computed by substituting the current spot value into all lag slots — this is the model's estimate of where Kalshi will quote once it finishes catching up. Edge = `q_settled - yes_ask - fees`. If edge clears a Kelly tier floor, enter; exit when edge compresses back to zero (lag closed), spot reverses past the stop threshold, or the window is about to close.

---

## Features (13 total, in `features.py`)

| Feature | Description |
|---------|-------------|
| `x_0` ... `x_30` | `log(microprice_lag / K)` at 0, 5, 10, 15, 20, 25, 30s lags |
| `tau_s` | seconds until window close |
| `inv_sqrt_tau` | `1 / sqrt(tau + 1)` |
| `kalshi_spread` | `yes_ask - yes_bid` |
| `kalshi_momentum_5s/10s/30s` | `yes_mid - yes_mid_lag` |

`q_settled` replaces all lag slots with `x_0` before predicting — answers "where does Kalshi land once it has seen the current spot in every lookback slot?"

---

## Config knobs (`config.py` + `.env`)

| Key | Default | What it does |
|-----|---------|------|
| `SPOT_SOURCE` | `coinbase` | `coinbase` or `binance` |
| `KALSHI_API_KEY_ID` | required | Kalshi key |
| `KALSHI_PRIVATE_KEY_FILE` | required | path to .pem |
| `REFIT_INTERVAL_S` | 300 | seconds between model refits |
| `TRAINING_WINDOW_S` | 14400 | rolling training window (4h) |
| `KELLY_TIERS` | see config | `[(edge_floor, wallet_fraction), ...]` |
| `LAG_CLOSE_THRESHOLD` | 0.005 | exit when edge falls below this |
| `STOP_THRESHOLD` | 0.03 | stop-loss: exit if edge erodes by this |
| `FALLBACK_TAU_S` | 60 | hold to resolution at this tau |
| `THETA_FEE_TAKER` | 0.07 | fee model: `theta * p * (1-p)` per leg |
| `DRY_RUN` | `true` | set `false` for real orders |

---

## Running

See `RUN.md` for step-by-step instructions.

```bash
# Single asset (BTC)
python scripts/run/run_kalshi_bot.py

# Multi-asset
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP

# Live orders (real money)
python scripts/run/run_kalshi_bot.py --live-orders
```

Each run creates `data/<timestamp>_<assets>/` with all output files.

---

## Analysis scripts

All scripts show a Tkinter popup to pick the run folder. Pass `--run data/<folder>` to skip the popup.

```bash
python scripts/analysis/analyze_run.py            # visualize decisions
python scripts/analysis/backtest.py               # walk-forward P&L
python scripts/analysis/replay_window.py          # animated window replay
python scripts/analysis/compare_models.py --mode filters   # filter comparison
python scripts/analysis/compare_models.py --mode maker     # maker vs taker
```

---

## Decision to advance to live trading

Advance when all of these hold on out-of-sample backtest data:

- Median R2_cv > 0.25
- Estimated lag stable in 15-120s range
- >50% of exits are `lag_closed`
- Simulated P&L positive on best sweep params
- `|q_predicted - yes_mid|` mean < 0.03

---

## Known issues / caveats

- `THETA_FEE_TAKER = 0.07` is from Kalshi's published schedule, not calibrated from real fills. Run `scripts/test/test_trade.py` before trusting backtest P&L numbers.
- Slippage model is crude (`min(0.02, size/depth * 0.01)`). Will understate at tier-1 sizing on thin books.
- Config thresholds (`MODEL_MIN_CV_R2`, `WIDE_SPREAD_THRESHOLD`) exist in `config.py` but scheduler hardcodes slightly stricter values - two places to tune.
- Feed staleness: `cb.ready` and `kb.ready` don't check recency within a tick. If a feed goes silent without disconnecting, the bot won't notice until the WS drops.
