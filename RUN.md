# RUN.md - Running the Kalshi Lag-Arb Bot

## Prerequisites

```
.env file with:
  KALSHI_API_KEY_ID=<your key>
  KALSHI_PRIVATE_KEY_FILE=~/.kalshi/kalshi_rsa.pem
  DRY_RUN=true
```

```bash
pip install -r requirements.txt
```

Verify auth works:
```bash
python scripts/test/check_kalshi_balance.py
```

---

## Step 1 — Dry run (collect training data)

Run the bot in dry mode for at least 24 hours. No real orders are placed. All ticks are saved to CSV.

```bash
# BTC only
python scripts/run/run_kalshi_bot.py

# Multiple assets
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP
```

Output is saved to `data/<YYYY-MM-DD_HH-MM-SS>_<ASSETS>/`:
- `ticks_BTC.csv` — raw ticks (spot + Kalshi book)
- `decisions_BTC.jsonl` — decision log (all abstain, no model loaded)

**What you'll see:**
```
=== Kalshi Lead-Lag Arbitrage Bot ===
  Assets:     BTC
  Mode: DRY RUN (data collection) -- no model loaded, bot will abstain.
  [BTC] Ticker: KXBTC15M-26MAY071400-15  Strike: 97,500  Closes: 14:15:00 UTC

[abstain              ] tau=720s  BTC=$97,432  bid=0.510 ask=0.520  q_set=--- edge=---  model_not_loaded
```

Stop after 24+ hours with `Ctrl+C`.

---

## Step 2 — Train model

```bash
# Interactive: picks run folder with popup
python scripts/train_model.py

# Specify run folder directly
python scripts/train_model.py --run data/2026-05-07_00-00-00_BTC

# Custom save name
python scripts/train_model.py --name btc_lgbm_v1

# List saved models
python scripts/train_model.py --list
```

Output:
```
Loading ticks from data/2026-05-07_00-00-00_BTC/ticks_BTC.csv...
  12 windows  43200 ticks total
  42850 training samples  (71.4 min at 10Hz)
Fitting LightGBM model (horizons: [5, 10, 15, 60]s)...
  Multi-horizon samples: 41920  horizons: [5, 10, 15, 60]s
  R2_hld(primary=10s): 0.347
    h5s_r2_hld=0.412
    h10s_r2_hld=0.347
    h15s_r2_hld=0.289
    h60s_r2_hld=0.198
  Saved model -> model_fits/btc_lgbm_v1.pkl  (LGBMModel  R2_hld=0.347)
```

A model with `R2_hld > 0.25` on the 10s horizon is viable.

---

## Step 3 — Tune trading knobs

Grid-sweeps Kelly tier configurations and exit thresholds on the held-out portion of your data to find the optimal config.

```bash
# Interactive: picks model and run folder
python scripts/tune_trading_knobs.py

# Specify both directly
python scripts/tune_trading_knobs.py --model-file model_fits/btc_lgbm_v1.pkl --run data/2026-05-07_00-00-00_BTC

# Tune on last 40% of data (default: --train-frac 0.6)
python scripts/tune_trading_knobs.py --model-file model_fits/btc_lgbm_v1.pkl
```

Output:
```
Running sweep: 10 tier templates × 5 exit thresholds × 5 hold times = 250 combinations...

  Top 10 by Sharpe ratio:
  Tier name            exit  hold_s     n  win%        pnl      avg  sharpe
  conservative_lo     0.005      60   120   62%    +0.8400  +0.0070  +2.341
  standard_3t         0.005      45   98    59%    +0.6800  +0.0069  +2.105
  ...

  Suggested config.py snippet (copy-paste and verify):

KELLY_TIERS = [
    (0.08, 0.05),
    (0.04, 0.03),
    (0.02, 0.015),
]

LAG_CLOSE_THRESHOLD = 0.005
MAX_HOLD_S          = 60
```

Apply the recommended snippet to `config.py`, then verify with Step 4.

---

## Step 4 — Full simulation

Replays the data with the trained model and the tuned config to verify the full picture.

```bash
# Interactive: picks model and run folder
python scripts/test_all.py

# Specify directly
python scripts/test_all.py --model-file model_fits/btc_lgbm_v1.pkl --run data/2026-05-07_BTC

# Only simulate on last 40% of data (test set)
python scripts/test_all.py --model-file model_fits/btc_lgbm_v1.pkl --train-frac 0.6
```

Output shows:
- Entry/exit markers on YES price chart
- Cumulative P&L curve
- Per-trade breakdown with q_settled, hold time, exit reason
- Total P&L after 7% taker fees

**Advance to live trading when:** net P&L is positive on the test set, win rate > 50%, most exits are `lag_closed`.

---

## Step 5 — Live trading

```bash
# Load pre-trained model — trades immediately, no warmup
python scripts/run/run_kalshi_bot.py --model-file model_fits/btc_lgbm_v1.pkl --live-orders

# Multi-asset with per-asset models
python scripts/run/run_kalshi_bot.py \
  --assets BTC ETH \
  --model-file BTC=model_fits/btc.pkl,ETH=model_fits/eth.pkl \
  --live-orders

# Cap bet size and daily loss
python scripts/run/run_kalshi_bot.py \
  --model-file model_fits/btc_lgbm_v1.pkl \
  --live-orders \
  --max-bet-pct 0.05 \
  --daily-loss-pct 0.03
```

**What you'll see:**
```
  Mode: PRELOADED MODEL -- bot trades from tick 1, no warmup needed
  [Preloaded] Using pre-trained LGBMModel  R2_hld=0.347  -- ready to trade immediately

[entry                ] tau=580s  BTC=$97,500  bid=0.510 ask=0.520  q_set=0.591 edge=0.052  ...
  [LIVE ENTRY] side=yes filled=5@52c =$2.60 edge=0.052 tier=2

[exit_lag_closed      ] tau=560s  BTC=$97,498  bid=0.572 ask=0.582  q_set=0.576 edge=0.002  ...
  [LIVE EXIT:exit_lag_closed] side=yes entry=0.520 exit=0.572 pnl=+0.24 USD hold=20s
```

---

## Health check

| Metric | Healthy | Bad |
|--------|---------|-----|
| R2_hld (10s horizon) | > 0.25 | < 0.10 |
| Net P&L (test simulation) | positive | negative |
| Win rate | > 50% | < 40% |
| % exits lag_closed | > 50% | < 30% |
| Avg hold time | 15-90s | > 120s or < 5s |

---

## Analysis

```bash
# Visualize decisions log
python scripts/analysis/analyze_run.py

# Animated replay of one 15-min window
python scripts/analysis/replay_window.py
python scripts/analysis/replay_window.py --model-file model_fits/btc_lgbm_v1.pkl

# Live scrolling plot while bot runs
python scripts/analysis/live_plot.py
```

---

## Useful PowerShell one-liners

```powershell
# Count decisions logged
(Get-Content data\<run>\decisions_BTC.jsonl | Measure-Object -Line).Lines

# Watch live
Get-Content data\<run>\decisions_BTC.jsonl -Wait -Tail 5

# Count entries
Select-String '"event":"entry"' data\<run>\decisions_BTC.jsonl | Measure-Object

# Count ticks collected
(Get-Content data\<run>\ticks_BTC.csv | Measure-Object -Line).Lines
```

---

## Stop

`Ctrl+C` — drains gracefully, prints `Bot stopped.` and the run folder path.
