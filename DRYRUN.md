# Kalshi BTC Lag-Arb — Dry Run Instructions

> **Goal:** Run the bot for 4+ hours collecting live data, let it train a regression model automatically, then visualize what trades it would have taken and whether the strategy has edge.

---

## How It Works (Quick Mental Model)

| Loop | Rate | What it does |
|------|------|------|
| **Sampler** | every 1s | Reads Coinbase microprice + Kalshi book → builds feature vector → appends to training buffer |
| **Refitter** | every 5 min | Pulls last 4h of samples → fits ridge regression → swaps new model in live |
| **Decision** | every 10s | Computes q_settled (where Kalshi will go), edge = q_settled - yes_ask - fees, applies Kelly sizing. Logs entry/exit/abstain to JSONL. No real orders. |
| **Window mgr** | every 10s | Detects 15-min window rollover, discovers next ticker, reconnects WS automatically |

**Timeline from cold start:**
- `0:00` — Feeds connect. Sampler starts collecting at 1 sample/s.
- `~6 min` — 360 samples accumulated. First model fit attempted.
- `~6-30 min` — Model fit but with little data. R2 will be low; most ticks abstain with `model_low_r2`.
- `~1 hr+` — Model has enough history to be meaningful. R2 should climb above 0.10.
- `~4 hr` — Full 4-hour rolling window. Model is as good as it gets on this data.

**The 10s decision rate vs 1s sample rate:** These are independent. The sampler feeds the training buffer at 1 Hz (3,600 samples/hour). The decision loop only runs every 10s. More training data = better model, even though decisions are infrequent.

---

## Prerequisites

```powershell
# .env must have these set:
KALSHI_API_KEY_ID=<your key>
KALSHI_PRIVATE_KEY_FILE=<path to your .pem>   # OR
KALSHI_PRIVATE_KEY_PEM=<inline pem with \n>
```

Install dependencies if not already done:
```powershell
pip install websockets aiohttp cryptography scikit-learn numpy python-dotenv matplotlib
```

---

## Step 0 — Start a Clean Run

If you have old data from a previous session, use `--fresh` to delete it before starting:

```powershell
python scripts/run_kalshi_bot.py --fresh
```

This deletes `logs/decisions.jsonl` and `logs/ticks.csv` before connecting, so `analyze_run.py` and `backtest.py` will only see data from this run. Without `--fresh`, the bot appends to existing files.

---

## Step 1 — Start the Bot

```powershell
cd C:\Users\steve\Repos\PolymarketBot
python scripts/run_kalshi_bot.py
```

Expected startup output:
```
=== Kalshi BTC Lead-Lag Arbitrage Bot ===
  Wallet:     $1,000 (simulated)
  Log:        logs/decisions.jsonl
  Discovering active Kalshi BTC 15m market...
  Ticker:     KXBTC15M-26MAY061515-15
  Strike:     $81,360.36
  Closes at:  19:15:00 UTC
  Starting feeds...

[abstain               ] tau=  847s  bid=0.520 ask=0.530  q_set=--- q_pred=---  edge=---  buf=9(0.1m)  R2=0.00  model_warmup
```

**If the market is mid-reset** (window just closed, next not open yet):
```
  No active market yet (attempt 1/12) — waiting 5s for window to open...
```
This is normal. It retries every 5s for up to 60s until the new window opens.

---

## Step 2 — Let It Run (4+ Hours)

Leave the terminal open and go to work. Every 10 seconds it prints one line:

```
[abstain               ] tau=  720s  bid=0.510 ask=0.520  q_set=--- q_pred=---  edge=---  buf=360(6.0m)  R2=0.00  model_warmup
[abstain               ] tau=  710s  bid=0.510 ask=0.520  q_set=0.561 q_pred=0.508  edge=0.0312  buf=370(6.2m)  R2=0.18  edge_below_floor
[entry                 ] tau=  700s  bid=0.510 ask=0.520  q_set=0.591 q_pred=0.512  edge=0.0521  buf=380(6.3m)  R2=0.21
[hold                  ] tau=  690s  bid=0.520 ask=0.530  q_set=0.589 q_pred=0.521  edge=0.0412  buf=390(6.5m)  R2=0.21
[exit_lag_closed       ] tau=  680s  bid=0.570 ask=0.580  q_set=0.575 q_pred=0.568  edge=0.0021  buf=400(6.7m)  R2=0.22
```

Columns:
- `tau` — seconds until this window closes
- `bid/ask` — Kalshi YES bid/ask
- `q_set` — where model predicts Kalshi is heading (the trade signal)
- `q_pred` — where model thinks Kalshi should be now (sanity check, should track bid/ask)
- `edge` — net edge after fees (blank = abstaining)
- `buf` — training buffer size and time span
- `R2` — model cross-validation R2 (quality gauge; 0.10+ = useful, 0.25+ = healthy)

Every 5 minutes a refit line prints:
```
[Refit] n=1800 alpha=0.100 R2_cv=0.312 R2_hld=0.287 lag=47s dCoef=0.041
```
- `lag=47s` means Kalshi is currently ~47 seconds behind Coinbase.
- `dCoef` — how much the model changed vs last fit (small = stable).

**Window rollovers every 15 min** are automatic:
```
INFO  window rollover -> KXBTC15M-26MAY061600-15  K=81412.55
```

---

## Step 2b — Backtest on Collected Ticks (after 1+ hour)

```powershell
# Default single run
python scripts/backtest.py

# Tune the parameters
python scripts/backtest.py --entry 0.030 --exit 0.008 --hold-s 45

# Sweep all parameter combinations (takes ~30s)
python scripts/backtest.py --sweep

# Save charts to PNG
python scripts/backtest.py --save backtest_results.png
```

Output shows:
- R2 on train vs test (model quality; > 0.10 means signal exists)
- Every individual trade: entry time, buy price, sell price, P&L, hold time, why it exited
- Cumulative P&L curve
- Sweep table sorted by total P&L (when using `--sweep`)

---

## Step 3 — Visualize the Results

Run this any time (even while the bot is still running):

```powershell
python scripts/analyze_run.py
```

This opens 4 charts:

1. **Kalshi YES price** — bid/ask band, YES mid (blue), q_settled (red dashed), q_predicted (orange dotted). Entry markers (green triangles), exit markers. If q_settled consistently leads the mid price, the signal is real.

2. **Edge magnitude** — purple line vs Kelly tier thresholds (dashed). Ticks above the lowest threshold (0.02) are eligible entry candidates.

3. **Model quality** — R2_cv over time (blue) and estimated Kalshi lag in seconds (orange). You want R2 above 0.10 and lag stable in the 15-90s range.

4. **Simulated P&L** — cumulative profit from all logged entry/exit pairs (after fee model). If no trades yet, shows the abstention breakdown pie chart instead.

To save the chart as a PNG instead of opening it:
```powershell
python scripts/analyze_run.py --save results.png
```

---

## Step 4 — Read the Terminal Summary

`analyze_run.py` also prints a text summary:

```
============================================================
  DRY-RUN SUMMARY
============================================================
  Span:          247.3 minutes  (1484 ticks)
  Windows:       17

  Events:
    abstain                          1380
    entry                              34
    hold                               58
    exit_lag_closed                    28
    exit_stopped                        4
    fallback_resolution                 2

  Abstention reasons:
    model_warmup                      360
    edge_below_floor                  890
    model_low_r2                       80
    already_engaged                    50

  Model R2 (cv):   median=0.312  p10=0.148  p90=0.441
  Est. lag (s):    median=47  min=18  max=89
  Edge magnitude:  median=0.0142  p90=0.0381  p99=0.0821
  Entries:        34
  Exits:          34

  Simulated P&L: $+12.43  (ROI 36.6%  win-rate 26/34)

    lag_closed      :  28 trades  avg_pnl=+$0.52
    stopped         :   4 trades  avg_pnl=-$0.31
    resolution      :   2 trades  avg_pnl=-$0.18
============================================================
```

### What to look for

| Metric | Healthy | Bad |
|--------|---------|-----|
| Median R2 cv | > 0.25 | < 0.10 |
| Estimated lag | 15–90s | < 5s or > 150s |
| % exits as lag_closed | > 50% | < 30% |
| Edge p90 | > 0.03 | < 0.01 |
| P&L | positive | negative |
| q_pred tracks yes_mid | yes_mid residual < 0.03 | > 0.10 |

If **median R2 < 0.10**: Kalshi is not predictable from Coinbase — the strategy thesis doesn't hold on today's data.

If **estimated lag < 5s**: Kalshi's market makers are already fast enough to eliminate the edge. Check back during high-volatility periods.

If **P&L positive and lag_closed > 50%**: the edge is real. The bot is correctly identifying and timing the lag.

---

## Useful Commands While Running

```powershell
# Check how many ticks have been logged
(Get-Content logs\decisions.jsonl | Measure-Object -Line).Lines

# Watch the last few decisions live
Get-Content logs\decisions.jsonl -Wait -Tail 5

# See how many entries/exits happened so far
Select-String '"event":"entry"' logs\decisions.jsonl | Measure-Object
Select-String '"event":"exit_lag_closed"' logs\decisions.jsonl | Measure-Object

# Analyze mid-run (while bot is still running)
python scripts/analyze_run.py
```

---

## To Stop the Bot

Press `Ctrl+C` in the terminal. The bot will drain gracefully and print `Bot stopped.`

The `logs/decisions.jsonl` file is safe to read while the bot runs — each line is flushed immediately.

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/run_kalshi_bot.py` | Start the bot |
| `scripts/analyze_run.py` | Visualize the results |
| `scripts/test_trade.py` | Place a real $1 round-trip trade (buy + sell) |
| `betbot/kalshi/scheduler.py` | Core: sampler + refitter + decision + window manager |
| `betbot/kalshi/model.py` | Ridge regression, q_settled, q_predicted |
| `betbot/kalshi/features.py` | Feature engineering (12 features) |
| `betbot/kalshi/book.py` | CoinbaseBook + KalshiBook state |
| `betbot/kalshi/coinbase_feed.py` | Coinbase WS feed (20 Hz) |
| `betbot/kalshi/kalshi_rest_feed.py` | Kalshi REST polling feed (1Hz, same path as test_trade.py) |
| `betbot/kalshi/auth.py` | Shared Kalshi RSA-PSS request signing |
| `logs/decisions.jsonl` | All tick data (grows during run) |
