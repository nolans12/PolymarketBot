# RUN.md — Running the Kalshi Lag-Arb Bot

## Prerequisites

```
.env file with:
  KALSHI_API_KEY_ID=<your key>
  KALSHI_PRIVATE_KEY_FILE=~/.kalshi/kalshi_rsa.pem
  SPOT_SOURCE=binance        # or coinbase (US-legal, no VPN required)
```

```bash
pip install -r requirements.txt
```

Verify auth + WS feed both work:
```bash
python scripts/test/check_kalshi_balance.py
python scripts/test/watch_kalshi_ws.py             # live book, top-of-book sizes
python scripts/test/test_trade.py                  # ~$1 round-trip taker test
python scripts/test/test_trade_maker.py            # ~$1 round-trip maker entry test
```

---

## Pipeline overview

```
1. COLLECT  — 24h+ of fresh ticks via WebSocket (parquet + top-10 depth per side)
2. TRAIN    — fit a LightGBM 17-feature model on those ticks
3. TUNE     — sweep Kelly tier configs through the realistic backtest;
              paste the winning KELLY_TIERS into config.py
4. BACKTEST — re-run with the tuned config to confirm net P&L / win rate /
              exit-reason distribution before risking real money
5. GO LIVE  — bot uses maker entries (bid+1c, 5s TTL) and taker IOC exits
```

**Tune before backtest:** `backtest.py` reads `KELLY_TIERS` from `config.py` at
runtime, so running it before tuning gives you a meaningless number tied to
whatever the current tier config happens to be. `tune.py` already runs the
backtest internally for every candidate config — it just sweeps and reports
the best one.

The bot writes parquet (`ticks_BTC.parquet`) with top-10 book depth on both
sides — that depth feeds 4 new model features (`yes_bid_size`, `yes_ask_size`,
`yes_depth_5c`, `no_depth_5c`) that the bot has access to live.

---

## Step 1 — Collect fresh data (dry run)

```bash
# Default: Binance WS spot + Kalshi WS book + parquet ticks with depth
python scripts/run/run_kalshi_bot.py --assets BTC

# Multi-asset
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP
```

Output is saved to `data/<YYYY-MM-DD_HH-MM-SS>_<ASSETS>/`:
- `ticks_BTC.parquet` — 10Hz raw ticks with full top-10 book depth
- `decisions_BTC.jsonl` — decision log (all abstain — no model loaded)

**Storage**: ~80 MB per asset per 24h with snappy compression.

Watch the bot in another terminal:
```bash
python scripts/watch_decisions.py
```

Live book sanity check:
```bash
python scripts/test/watch_kalshi_ws.py
```

Stop after 24+ hours with `Ctrl+C`.

---

## Step 2 — Train model

```bash
# Interactive: prompts for run folder + asset
python scripts/train_model.py

# Direct
python scripts/train_model.py --run data/2026-05-08_<run>_BTC --asset BTC
```

Model is saved to `model_fits/<run>_<ASSET>_<date>/model.pkl` with metadata in
`model.json`. Targets `R2_hld(10s) > 0.25` to be viable.

The new feature set is 17 columns (13 lag/momentum + 4 depth):

| Feature | What it captures |
|---|---|
| `x_0` … `x_30` | log(microprice_lag / strike) at 0,5,10,15,20,25,30s lags |
| `tau_s`, `inv_sqrt_tau` | seconds until window close, plus its inverse-sqrt |
| `kalshi_spread` | yes_ask − yes_bid |
| `kalshi_lag_5s/10s/30s` | yes_mid drift over the last 5s/10s/30s |
| `yes_bid_size`, `yes_ask_size` | size at top YES bid and at top NO bid (= YES ask) |
| `yes_depth_5c`, `no_depth_5c` | total contracts within 5c of best bid on each side |

---

## Step 3 — Tune Kelly tiers

Run this **before** the standalone backtest. The tuner sweeps every
`(edge_floor, wallet_fraction)` configuration through the exact same backtest
engine that `backtest.py` uses, so it's the right tool for *finding* the
config. `backtest.py` is the right tool for *confirming* the chosen config.

```bash
python scripts/tune.py \
  --run data/<run> \
  --asset BTC \
  --model-file model_fits/<dir>/model.pkl
```

Prints the top-N configs by net P&L plus a copy-pasteable snippet for
`betbot/kalshi/config.py`:

```
=== Best configuration — paste into config.py ===

KELLY_TIERS = [
    (0.10, 0.06),
    (0.06, 0.04),
    (0.03, 0.02),
]

  Net P&L: $+12.45  Trades: 87  Wins: 53
```

Paste the snippet into `betbot/kalshi/config.py` (replace the existing
`KELLY_TIERS`), then move to Step 4.

---

## Step 4 — Backtest the tuned config

Now that `KELLY_TIERS` reflects the tuner's choice, this gives you the full
per-trade picture:

```bash
python scripts/backtest.py \
  --run data/<run> \
  --asset BTC \
  --model-file model_fits/<dir>/model.pkl
```

The backtest models the actual Kalshi maker workflow:

| Stage | What it does |
|---|---|
| **Entry decision** | Same as live bot — model edge clears a Kelly floor |
| **Maker post** | Posts a resting limit at `bid + 1c` |
| **Maker fill** | Conservative: fills only if the live ask drops to ≤ post price within `MAKER_TTL_S` (5s). If TTL expires unfilled, no trade is taken. |
| **Exit triggers** | `exit_lag_closed` (edge ≤ 0.005) > `exit_stopped` > `exit_max_hold` (15s) > `fallback_resolution` (tau < 60s) |
| **Taker exit** | Sweeps current bid as IOC; proceeds = `count − taker_fill_cost` per Kalshi reciprocal pricing. Real taker fee = `0.07 × p × (1 − p) × count` |
| **P&L** | Per-trade: `gross = contracts × (exit − entry)`, `fees = taker_fee(exit) × contracts`, `net = gross − fees` |

Output: total trades, win rate, gross/fees/net P&L, breakdown by exit reason and Kelly tier.

**Advance to live trading when all hold:**
- `R2_hld(10s) > 0.25`
- Net P&L positive after fees
- Win rate > 50%
- > 50% of exits are `exit_lag_closed`
- Avg hold time 5–15s

---

## Step 5 — Go live

```bash
# Single asset
python scripts/run/run_kalshi_bot.py \
  --assets BTC \
  --live-orders \
  --model-file model_fits/<dir>/model.pkl

# Multi-asset with per-asset models
python scripts/run/run_kalshi_bot.py \
  --assets BTC ETH SOL \
  --live-orders \
  --model-file BTC=model_fits/btc/model.pkl,ETH=model_fits/eth/model.pkl,SOL=model_fits/sol/model.pkl

# Cap bet size and daily loss
python scripts/run/run_kalshi_bot.py \
  --assets BTC \
  --live-orders \
  --model-file model_fits/<dir>/model.pkl \
  --max-bet-pct 0.05 \
  --daily-loss-pct 0.03
```

**Bot defaults (set in `betbot/kalshi/config.py`):**

| Knob | Default | Meaning |
|---|---|---|
| `ENTRY_MODE` | `maker` | post resting limit, wait for fill |
| `MAKER_AT_BID_PLUS_1` | `True` | post at bid+1c (faster fills than at bid) |
| `MAKER_TTL_S` | `5.0` | cancel if not filled in 5s |
| `MAX_HOLD_S` | `15` | force-exit after 15s |
| `LAG_CLOSE_THRESHOLD` | `0.005` | exit when edge compresses below 0.5c |
| `WALLET_BALANCE` | `100.0` | wallet cap (real balance can be higher) |
| `SIZE_MAX_USD` | `2.0` | per-order ceiling |

**What the live bot does on every entry:**

```
*** MAKER ENTRY [BTC] YES  3/3 @ $0.485  cost=$1.46  edge=0.072  tier=2  tau=620s ***
*** LIVE EXIT [BTC] exit_lag_closed  YES  entry=0.485 → exit=0.518  pnl=+$0.08  hold=8s ***
```

If the maker post doesn't fill within 5s, you'll see the bot back off and the
watcher logs an `entry_unfilled` abstain. No money was committed.

**Watch the bot live:**
```bash
python scripts/watch_decisions.py
```

Status bar: live spot price, Δ-from-strike, edge, position, bid/ask, tau.
Events: BUY/EXIT lines color-coded by outcome.

---

## Diagnostics

| Script | Purpose |
|---|---|
| `scripts/test/check_kalshi_balance.py` | auth + balance smoke check |
| `scripts/test/watch_kalshi_ws.py` | standalone WS book viewer with depth |
| `scripts/test/test_trade.py` | $1 taker round-trip via WS |
| `scripts/test/test_trade_maker.py` | $1 maker entry / taker exit round-trip via WS |
| `scripts/watch_decisions.py` | tail decisions log + live status bar |
| `scripts/analysis/analyze_run.py` | post-run charts (price, edge, P&L, abstains) |
| `scripts/analysis/live_plot.py` | rolling chart while bot is running |

---

## Health check

| Metric | Healthy | Bad |
|---|---|---|
| `R2_hld(10s)` | > 0.25 | < 0.10 |
| Net P&L (backtest) | positive | negative |
| Win rate | > 50% | < 40% |
| % exits `lag_closed` | > 50% | < 30% |
| Avg hold | 5–15s | > 30s or < 2s |
| Maker fill rate | > 30% | < 10% |

If maker fill rate is too low, raise `MAKER_AT_BID_PLUS_1` confidence (already
on by default) or extend `MAKER_TTL_S`. If it's too high, the entry edge
threshold is too generous — re-tune.

---

## Stop

`Ctrl+C` — drains gracefully, prints `Bot stopped.` and the run folder path.
Any open positions on Kalshi will NOT be force-closed automatically — check
the Kalshi UI manually if a window is mid-trade when you stop.
