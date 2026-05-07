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
# All 4 assets (recommended — more data, trains better models)
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP

# BTC only
python scripts/run/run_kalshi_bot.py
```

Output is saved to `data/<YYYY-MM-DD_HH-MM-SS>_<ASSETS>/`:
- `ticks_BTC.csv` — raw 1Hz ticks (spot + Kalshi book)
- `decisions_BTC.jsonl` — decision log (all abstain, no model loaded)

**What you'll see:**
```
=== Kalshi Lead-Lag Arbitrage Bot ===
  Assets:     BTC, ETH, SOL, XRP
  Mode: DRY RUN (data collection) -- no model loaded, bot will abstain.
  [BTC] Ticker: KXBTC15M-26MAY071400-15  Strike: 97,500  Closes: 14:15:00 UTC

[abstain] tau=720s  BTC-K=+$32  bid=0.510 ask=0.520  q_set=--- edge=---  model_not_loaded
```

Stop after 24+ hours with `Ctrl+C`.

---

## Step 2 — Train model

One model per asset. Run separately for each.

```bash
# Interactive: prompts for run folder and asset
python scripts/train_model.py

# Specify run folder directly
python scripts/train_model.py --run data/2026-05-07_00-00-00_BTC_ETH_SOL_XRP

# List saved models
python scripts/train_model.py --list
```

Model is saved to `model_fits/<run>_<ASSET>_<date>/`:
- `model.pkl` — the trained LightGBM model
- `model.json` — metadata (R2 scores, feature names, horizons)

**Expected output:**
```
Loading ticks from data/2026-05-07_BTC_ETH_SOL_XRP/ticks_BTC.csv...
  12 windows  43200 ticks  (detected ~1.0 Hz)
  Filtering: 42850 rows in [0.001, 0.999] yes_mid range
  Building features [12/12  100%]  KXBTC15M-26MAY150000-00...
  Multi-horizon samples: 41920  horizons: [5, 10, 15, 60]s
  Fitting horizon 1/4 (h5s)...
  Fitting horizon 2/4 (h10s)...
  Fitting horizon 3/4 (h15s)...
  Fitting horizon 4/4 (h60s)...
  R2_hld(primary=10s): 0.347
  Saved -> model_fits/2026-05-07_BTC_ETH_SOL_XRP_BTC_20260507_020000/model.pkl
```

A model with `R2_hld > 0.25` on the 10s horizon is viable.

---

## Step 3 — Tune trading knobs

Grid-sweeps Kelly tier configurations and exit thresholds on the held-out data to find the optimal config.

```bash
# Interactive: prompts for model dir and run folder
python scripts/tune_trading_knobs.py

# Specify both directly
python scripts/tune_trading_knobs.py \
  --model-file model_fits/2026-05-07_BTC_ETH_SOL_XRP_BTC_20260507_020000/model.pkl \
  --run data/2026-05-07_00-00-00_BTC_ETH_SOL_XRP
```

**Expected output:**
```
Running sweep...
  Top results by Sharpe:
  Tier config         exit    hold_s    n   win%     pnl    sharpe
  conservative_lo    0.005      60    120   62%   +0.840    +2.341

  Suggested config.py snippet:
  KELLY_TIERS = [(0.10, 0.03), (0.06, 0.02)]
  LAG_CLOSE_THRESHOLD = 0.005
  MAX_HOLD_S = 60
```

Apply the suggested snippet to `betbot/kalshi/config.py`, then verify with Step 4.

---

## Step 4 — Full simulation

Replays all data with the trained model and tuned config to verify end-to-end performance.

```bash
# Interactive: prompts for model and run folder
python scripts/test_all.py

# Specify directly
python scripts/test_all.py \
  --model-file model_fits/2026-05-07_BTC_ETH_SOL_XRP_BTC_20260507_020000/model.pkl \
  --run data/2026-05-07_00-00-00_BTC_ETH_SOL_XRP
```

Output: 4-panel chart showing:
1. YES price + entry/exit markers
2. Cumulative P&L
3. Edge over time with Kelly tier floors
4. q_settled vs P&L scatter

**Advance to live trading when all hold:**
- R2_hld (10s horizon) > 0.25
- Net P&L positive after fees
- Win rate > 50%
- > 50% of exits are `lag_closed`
- Avg hold time 15–90s

---

## Step 5 — Live trading

```bash
# Single asset
python scripts/run/run_kalshi_bot.py \
  --model-file model_fits/<dir>/model.pkl \
  --live-orders

# Multi-asset with per-asset models
python scripts/run/run_kalshi_bot.py \
  --assets BTC ETH SOL XRP \
  --model-file BTC=model_fits/btc_dir/model.pkl,ETH=model_fits/eth_dir/model.pkl \
  --live-orders

# Cap bet size and daily loss
python scripts/run/run_kalshi_bot.py \
  --model-file model_fits/<dir>/model.pkl \
  --live-orders \
  --max-bet-pct 0.05 \
  --daily-loss-pct 0.03
```

**What you'll see:**
```
  Mode: PRELOADED MODEL -- bot trades from tick 1, no warmup needed

[entry      ] tau=580s  BTC-K=+$32  bid=0.510 ask=0.520  q_set=0.591 edge=0.052
  [LIVE ENTRY] side=yes filled=5@52c =$2.60 edge=0.052 tier=2

[exit_lag_closed] tau=560s  BTC-K=+$28  bid=0.572 ask=0.582  q_set=0.576 edge=0.002
  [LIVE EXIT] entry=0.520 exit=0.572 pnl=+0.24 USD hold=20s
```

---

## Analysis tools

```bash
# Animated replay of a 15-min window with multi-horizon model projections
# LEFT axis: Kalshi YES bid/ask + model projections at t+5s/10s/15s/60s
# RIGHT axis: BTC microprice − strike (centered near zero)
python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl
python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl --speed 3 --trail 60
python scripts/replay_window.py --model-file model_fits/<dir>/model.pkl --window KXBTC15M-26MAY070200-00

# Visualize decisions log from a completed run
python scripts/analysis/analyze_run.py

# Live scrolling chart while bot is running
python scripts/analysis/live_plot.py
```

---

## Health check

| Metric | Healthy | Bad |
|--------|---------|-----|
| R2_hld (10s horizon) | > 0.25 | < 0.10 |
| Net P&L (simulation) | positive | negative |
| Win rate | > 50% | < 40% |
| % exits lag_closed | > 50% | < 30% |
| Avg hold time | 15–90s | > 120s or < 5s |

---

## Useful PowerShell one-liners

```powershell
# Count ticks collected
(Get-Content data\<run>\ticks_BTC.csv | Measure-Object -Line).Lines

# Watch decisions live
Get-Content data\<run>\decisions_BTC.jsonl -Wait -Tail 5

# Count entries only
Select-String '"event":"entry"' data\<run>\decisions_BTC.jsonl | Measure-Object

# Count abstains
Select-String '"event":"abstain"' data\<run>\decisions_BTC.jsonl | Measure-Object
```

---

## Stop

`Ctrl+C` — drains gracefully, prints `Bot stopped.` and the run folder path.
