# RUN.md - Running the Kalshi Lag-Arb Bot

## Prerequisites

```
.env file with:
  KALSHI_API_KEY_ID=<your key>
  KALSHI_PRIVATE_KEY_FILE=~/.kalshi/kalshi_rsa.pem
  DRY_RUN=true
```

```bash
pip install websockets aiohttp cryptography scikit-learn numpy pandas python-dotenv matplotlib
```

Verify auth works:
```bash
python scripts/test/check_kalshi_balance.py
```

---

## Start the bot

```bash
# BTC only (default)
python scripts/run/run_kalshi_bot.py

# Multiple assets - one independent model each
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP

# Simulated wallet (default $1000, split equally across assets)
python scripts/run/run_kalshi_bot.py --assets BTC ETH --wallet 500

# Real orders (requires positive Kalshi balance, reads live balance)
python scripts/run/run_kalshi_bot.py --live-orders

# Delete old decisions logs before starting (keeps ticks for model bootstrap)
python scripts/run/run_kalshi_bot.py --fresh
```

Output is saved to `data/<YYYY-MM-DD_HH-MM-SS>_<ASSETS>/` per run.

---

## What you'll see

**Startup:**
```
=== Kalshi Lead-Lag Arbitrage Bot ===
  Assets:     BTC
  Spot feed:  coinbase
  Run folder: data/2026-05-07_14-00-00_BTC
  [BTC] Discovering active Kalshi 15m market (KXBTC15M)...
  [BTC] Ticker: KXBTC15M-26MAY071400-15  Strike: 97,500.0000  Closes: 14:15:00 UTC
```

**Every 10 seconds:**
```
[abstain] tau=720s bid=0.510 ask=0.520 q_set=--- edge=--- buf=9(0.1m) R2=0.00 model_warmup
[abstain] tau=590s bid=0.510 ask=0.520 q_set=0.541 edge=0.013 buf=370(6.2m) R2=0.18 edge_below_floor
[entry  ] tau=580s bid=0.510 ask=0.520 q_set=0.591 edge=0.052 buf=380(6.3m) R2=0.22
[hold   ] tau=570s bid=0.525 ask=0.535 q_set=0.588 edge=0.041 buf=390(6.5m) R2=0.22
[exit_lag_closed] tau=560s bid=0.572 ask=0.582 q_set=0.576 edge=0.002 buf=400(6.7m) R2=0.23
```

**Every 5 minutes:**
```
[Refit] n=1800 alpha=0.100 R2_cv=0.31 R2_hld=0.28 lag=47s dCoef=0.04
```
`lag=47s` = Kalshi is currently ~47 seconds behind Coinbase. Target: 15-90s, stable.

**Window rollovers** happen automatically every 15 minutes.

---

## Warmup timeline

| Time | State |
|------|-------|
| 0:00 | Feeds connect, sampler starts |
| ~6 min | 360 samples collected, first model fit attempted |
| ~30 min | Model stabilizing; R2 climbing |
| ~4 hr | Full 4-hour rolling window, model at full quality |

---

## Analyze results

All analysis scripts show a Tkinter popup to select the run folder. Pass `--run data/<folder>` to skip.

```bash
# Charts: YES price vs model, edge over time, R2/lag, simulated P&L
python scripts/analysis/analyze_run.py

# Pick an asset if running multi-asset
python scripts/analysis/analyze_run.py --asset ETH

# Walk-forward backtest on collected ticks
python scripts/analysis/backtest.py

# Parameter sweep (takes ~30s)
python scripts/analysis/backtest.py --sweep

# Animated replay of one 15-min window
python scripts/analysis/replay_window.py

# Live scrolling plot while bot runs
python scripts/analysis/live_plot.py
```

---

## Health check metrics

| Metric | Healthy | Bad |
|--------|---------|-----|
| Median R2_cv | > 0.25 | < 0.10 |
| Estimated lag | 15-90s stable | < 5s or > 150s |
| % exits lag_closed | > 50% | < 30% |
| Edge p90 | > 0.03 | < 0.01 |
| q_pred vs yes_mid | mean diff < 0.03 | > 0.10 |
| Simulated P&L | positive | negative |

If R2 < 0.10: Kalshi isn't predictable from Coinbase on today's data. The edge isn't there right now.

If lag < 5s: Kalshi's market makers are faster than the strategy requires. Try during higher-volatility periods.

---

## Useful PowerShell one-liners

```powershell
# Count decisions logged
(Get-Content data\<run>\decisions_BTC.jsonl | Measure-Object -Line).Lines

# Watch live
Get-Content data\<run>\decisions_BTC.jsonl -Wait -Tail 5

# Count entries
Select-String '"event":"entry"' data\<run>\decisions_BTC.jsonl | Measure-Object
```

---

## Stop

`Ctrl+C` — drains gracefully, prints `Bot stopped.` and the run folder path.
