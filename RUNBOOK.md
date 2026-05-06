# PolymarketBot — Runbook

Everything you need to set up, run, monitor, and analyze the Phase 1 dry run.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setup](#2-setup)
3. [Environment Configuration](#3-environment-configuration)
4. [Running the Bot](#4-running-the-bot)
5. [Monitoring the Dry Run](#5-monitoring-the-dry-run)
6. [Visualizing Live Market Data](#6-visualizing-live-market-data)
7. [Post-Run Analysis](#7-post-run-analysis)
8. [Understanding the Results](#8-understanding-the-results)
9. [Deciding Whether to Go Live](#9-deciding-whether-to-go-live)
10. [Running Dry Run vs Live on Different Machines](#10-running-dry-run-vs-live-on-different-machines)
11. [Troubleshooting](#11-troubleshooting)
12. [File Reference](#12-file-reference)

---

## 1. Prerequisites

- Python 3.11+
- Windows 10/11, macOS, or Linux (Ubuntu recommended for headless server)
- No Polymarket account required for Phase 1 (read-only)
- **VPN recommended** — Binance L2 is the primary spot feed and is geo-blocked
  from US IPs. If you can't run a VPN, set `ENABLE_BINANCE=false` and
  `ENABLE_COINBASE=true` in `.env` to fall back to Coinbase ticker.

---

## 2. Setup

```powershell
# Clone and enter the repo
cd C:\Users\steve\Repos\PolymarketBot

# Create virtualenv and install
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate       # Mac/Linux

pip install -e ".[dev]"
```

---

## 3. Environment Configuration

### 3.1 Create your `.env`

```powershell
copy .env.example .env
```

### 3.2 Required settings for Phase 1 dry run

Open `.env` and confirm these are set:

```env
DRY_RUN=true          # MUST be true — prevents any real order placement
CLOB_HOST=https://clob.polymarket.com
CHAIN_ID=137
FALLBACK_BALANCE=0

# Decision tick cadence in seconds (5–10 typical)
POLL_INTERVAL=5

# Training sampler cadence in seconds. Lower = more training rows.
# 1.0 gives ~60 rows/min/asset; 0.5 doubles that.
SAMPLE_INTERVAL=1.0

# Spot feed selection (at least one must be true)
ENABLE_BINANCE=true     # geo-blocked from US — needs VPN
ENABLE_COINBASE=false   # works from US natively; set true if Binance unreliable

# K-capture source.
#   spot      = Binance microprice at window open (default; reliable, slightly
#               different from Polymarket's true Chainlink resolution oracle).
#   chainlink = Polymarket RTDS Chainlink ticks (canonical but high-latency,
#               often misses cold starts and rollovers).
K_SOURCE=spot
```

**Everything else (PRIVATE_KEY, API_KEY, etc.) can be left as placeholders
for the dry run.** Those are only needed for Phase 2 live trading.

### 3.3 Optional overrides

```env
# Where Parquet logs are written (default: ./logs/)
LOG_DIR=logs
PARQUET_DIR=logs
```

### 3.4 All strategy parameters

These are in `polybot/infra/config.py` with hardcoded defaults.
You do NOT need to change them for the dry run — the bot sweeps them
offline after the run.

| Parameter | Default | Description |
|---|---|---|
| `KELLY_TIERS` | 5 tiers (0.02–0.30) | Bet sizing thresholds |
| `LAG_CLOSE_THRESHOLD` | `0.005` | Exit when edge compresses to this |
| `STOP_THRESHOLD` | `0.03` | Stop-loss: exit if edge drops 3¢ below entry |
| `FALLBACK_TAU_S` | `10` | Default to resolution at this τ remaining |
| `MIN_TRAIN_SIZE` | `1800` | Rows needed before first refit attempt (~30 min) |
| `REFIT_INTERVAL_SECONDS` | `300` | Refit cadence after a successful fit (5 min) |
| `TRAINING_WINDOW_SECONDS` | `14400` | Rolling window the regression trains on (4 h) |
| `MODEL_MIN_CV_R2` | `0.10` | Minimum model R² to allow trading |
| `MODEL_MAX_DISAGREEMENT` | `0.15` | Max |q_predicted − q_actual| before abstaining |
| `MODEL_MAX_STALE_SECONDS` | `900` | Abstain if model not refit in 15 min |
| `WIDE_SPREAD_THRESHOLD` | `0.10` | Abstain if Polymarket spread > 10¢ |
| `BINANCE_STALE_MS_MAX` | `5000` | Circuit breaker: Binance feed age |
| `COINBASE_STALE_MS_MAX` | `10000` | Circuit breaker: Coinbase feed age |
| `POLY_STALE_MS_MAX` | `60000` | Circuit breaker: Polymarket feed age |

After a refit attempt is **skipped** (not enough rows), the refitter retries
in 60s instead of waiting the full 5 minutes — so cold starts converge fast.

---

## 4. Running the Bot

### 4.1 Start the dry run (keep terminal open)

```powershell
cd C:\Users\steve\Repos\PolymarketBot
.venv\Scripts\Activate.ps1
python -m polybot.main
```

The bot starts these concurrent tasks:
- `binance_ws` — Binance L2 spot stream (skipped if `ENABLE_BINANCE=false`)
- `coinbase_ws` — Coinbase ticker stream (skipped if `ENABLE_COINBASE=false`)
- `poly_ws` — Polymarket CLOB order book stream
- `rtds` — Polymarket Chainlink oracle stream + window-rollover monitor
- `scheduler` — Decision tick (cadence = `POLL_INTERVAL`)
- `k_capturer` — 1 Hz spot-driven K capture (used when `K_SOURCE=spot`)
- `sampler` — Training-row writer (cadence = `SAMPLE_INTERVAL`)
- `refitter` — Ridge regression refit (every `REFIT_INTERVAL_SECONDS` after a fit)
- `parquet_writer` — Flushes logs to disk every 10 seconds
- `shutdown_watcher` — Watches for kill switch file

### 4.2 Expected startup sequence

```
INFO  polybot starting DRY_RUN=True parquet_dir=logs
INFO  spot feeds: binance=True coinbase=False
INFO  scheduler started (DRY_RUN=True, interval=5s)
INFO  sampler started interval=1.00s assets=['btc', 'eth']
INFO  refitter started (interval=300s, window=14400s)
INFO  binance_ws connected
INFO  polymarket_ws connected
INFO  polymarket_ws subscribed asset=btc tokens=['…']
INFO  rtds connected
INFO  window K from spot asset=btc K=81239.38 window_ts=1778046900
INFO  RUNNING BTC UP OR DOWN 5 MINUTE WINDOW WITH PRICE TO BEAT: 81239.38 (window_ts=1778046900 slug=btc-updown-5m-1778046900)
INFO  polymarket_ws first book asset=btc side=up   bids=65 asks=34
INFO  polymarket_ws first book asset=btc side=down bids=34 asks=65
```

You'll then see the per-minute heartbeat lines:

```
INFO  sampler BTC: wrote=58 no_K=1 poly_not_ready=0 no_spot=0 bad_q=0 incomplete=0 errors=0
INFO  training warmup — BTC: 124/1800 rows (6%) | ETH: 12/1800 rows (0%)
```

**Sampler stat decoding:**
- `wrote=` — training rows written this minute (target ≈ 60 with `SAMPLE_INTERVAL=1.0`)
- `no_K=` — K not yet captured for this window (only at startup or rollover)
- `poly_not_ready=` — Polymarket book empty/not received yet
- `no_spot=` — neither Binance nor Coinbase has fresh data
- `bad_q=` — `q_up_ask` outside `(0.02, 0.98)` — markets resolved to extremes
- `incomplete=` — features missing (Binance ring buffer not warm yet)
- `errors=` — uncaught exceptions in the sampler

**First ~30 minutes:** All decisions show `abstain=model_warmup` — the
regression needs ≥ `MIN_TRAIN_SIZE` rows (default 1800, ~30 min combined
sample + decision rows) before the first fit attempt.

**After ~30 minutes:** First fit attempt. Even when it fits, you'll likely
see `r2_cv` very low or negative for the first hour — there isn't enough
price-relative variance yet. Decisions still abstain with `model_low_r2`.

**After 2–4 hours:** Once BTC has had real K-relative movement
(≥ 0.5 % in either direction across the training window), `r2_cv` should
climb above 0.10 and the bot starts evaluating real edges.

### 4.3 Stopping the bot

**Graceful stop (preferred):**
```powershell
# In a second terminal
python -m polybot.cli.polybot_ctl kill-switch
```

**Or just Ctrl+C** in the bot's terminal — it handles SIGINT cleanly.

**Do NOT close the terminal by force** during a Parquet flush — use Ctrl+C
or the kill switch to avoid corrupting in-flight writes.

### 4.4 Running headless on a server (Linux)

```bash
# Install as a systemd service (see CLAUDE.md §11.1 for the full unit file)
sudo systemctl start polybot
sudo systemctl status polybot
journalctl -u polybot -f --output=cat    # live log tail
```

---

## 5. Monitoring the Dry Run

All commands below query the live Parquet logs with DuckDB — safe to run
while the bot is writing.

### 5.1 Check bot health (quick status)

```powershell
python -m polybot.cli.polybot_ctl status
```

### 5.2 Decision summary — what the bot is doing

```powershell
# All assets, all time
python -m polybot.cli.polybot_metrics summary

# BTC only, today
python -m polybot.cli.polybot_metrics summary --asset btc --since 2026-05-05
```

Shows: decisions per day, entry rate, mean edge, abstention breakdown (warmup
vs below_floor vs circuit breakers).

### 5.3 Model health — is the regression working?

```powershell
# Current model: R², lag estimate, coefficients
python -m polybot.cli.polybot_metrics model --latest

# Historical trend: R² and lag stability over the run
python -m polybot.cli.polybot_metrics model --since 2026-05-05
```

**Healthy signs:**
- `r2_cv` (cross-validation R²) > 0.25
- `lag_s` (estimated lag in seconds) between 15 and 90
- `sd_lag_s` (hourly standard deviation) < 25

**Warning signs:**
- `r2_cv` < 0.10 consistently → Polymarket may not actually lag Coinbase in the current regime
- `lag_s` < 5 → Market makers are too fast; edge is too small
- `lag_s` > 200 or wild swings → Model is fitting noise

### 5.4 Edge distribution — is there signal?

```powershell
python -m polybot.cli.polybot_metrics edge --asset btc
python -m polybot.cli.polybot_metrics edge --asset eth
```

**Healthy:** `mean_abs_edge` between 0.005 and 0.025, `p99/mean` ratio of 5–15.

### 5.5 Engagement rate by window position

```powershell
python -m polybot.cli.polybot_metrics engage --asset btc
```

Shows entry rate broken into τ buckets (early window vs late window).
Expect 1–10% entry rate excluding warmup ticks.

### 5.6 Direct DuckDB queries

```powershell
# Install duckdb CLI or use Python
python -c "
import duckdb
con = duckdb.connect()
print(con.execute(\"SELECT asset, event, COUNT(*) FROM read_parquet('logs/decisions/**/*.parquet', hive_partitioning=true) GROUP BY 1,2 ORDER BY 1,3 DESC\").df())
"
```

Or with the DuckDB CLI:
```bash
duckdb -c "SELECT * FROM read_parquet('logs/decisions/**/*.parquet', hive_partitioning=true) LIMIT 5"
```

### 5.7 Overnight / multi-hour run workflow

The regression genuinely needs hours of varied data before R² climbs above
the trade gate. Recommended sequence:

```powershell
# 1. (Optional) wipe stale logs from prior failed runs
Remove-Item -Recurse logs

# 2. Confirm .env: DRY_RUN=true, ENABLE_BINANCE=true, K_SOURCE=spot
notepad .env

# 3. Start the bot and leave the terminal open overnight
python -m polybot.main
```

**Expected progression:**

| Elapsed | What you should see |
|---|---|
| 0–60 s | All feeds connect, K captured from spot, first window banner logged |
| 1 min | First `sampler BTC: wrote=58…` line; warmup at ~6 % |
| ~30 min | Warmup hits 100 %, first refit attempt fires |
| 1 h | R² typically still negative or near 0 — too little spot variance |
| 2–4 h | If BTC has moved ≥ 0.5 %, R² should climb toward / above 0.10 |
| 4 h+ | Bot may start evaluating real edges; decisions other than abstain |

In the morning:

```powershell
# Was the model fitting? Did R² progress?
python -m polybot.cli.polybot_metrics model --asset btc

# Did the model ever produce non-NULL edges?
python -m polybot.cli.polybot_metrics edge --asset btc

# Did decisions transition out of model_warmup?
python -m polybot.cli.polybot_metrics summary --asset btc

# How much K-relative spot movement did we actually have?
python -c "
import pyarrow.parquet as pq
from pathlib import Path
files = sorted([f for f in Path('logs/decisions').rglob('*.parquet') if 'asset=btc' in str(f)])
xs, qs = [], []
for f in files:
    pf = pq.ParquetFile(str(f))
    d = pf.read(columns=['x_now_logKratio', 'q_up_ask']).to_pydict()
    xs.extend([v for v in d['x_now_logKratio'] if v is not None])
    qs.extend([v for v in d['q_up_ask']         if v is not None])
import statistics
print(f'x_now stdev = {statistics.stdev(xs):.6f}  range = {max(xs)-min(xs):.6f}')
print(f'q_up_ask stdev = {statistics.stdev(qs):.4f}  range = {max(qs)-min(qs):.4f}')
"
```

**Interpreting `x_now` stdev:**

- `< 0.0005` — BTC was essentially flat in K-relative space; the regression
  has nothing to learn. Re-run during a more volatile period.
- `0.001 – 0.005` — moderate movement; the model should produce non-trivial R².
- `> 0.005` — strong signal; if R² is still bad with this much variance,
  the lag-arb thesis genuinely doesn't hold for this regime.

---

## 6. Visualizing Live Market Data

The visualizer shows live BTC/ETH prices from Binance, Coinbase, and
Polymarket in a single terminal dashboard — useful for sanity-checking
that all feeds are working before starting the full dry run.

```powershell
# Run visualizer (separate from the main bot — don't run both simultaneously)
python scripts/visualize_market.py
```

**What you see:**
- Panel 1: Polymarket Yes probability (Up/Down for BTC and ETH)
- Panel 2: Polymarket bid/ask spread
- Panel 3: Binance microprice + Coinbase microprice (both BTC and ETH)
- Panel 4: Feed staleness (ms since last update)

**Use this to verify:**
1. All four feeds are connecting (no `N/A` values after 30 seconds)
2. Binance and Coinbase prices are close (within ~$10)
3. Polymarket probability moves when spot moves sharply

Press `Ctrl+C` to exit.

---

## 7. Post-Run Analysis

Run these after the 24-hour dry run is complete.

### 7.1 Simulate cash-out exits (default thresholds)

```powershell
python -m polybot.research.cashout_simulator `
    --parquet-dir logs `
    --lag-close 0.005 `
    --stop 0.030 `
    --fallback-tau 10 `
    --out sim_default.parquet
```

**If you restarted the bot mid-run** and want to skip the bad early data,
use `--since` with the ISO timestamp of when the clean run began:

```powershell
python -m polybot.research.cashout_simulator `
    --parquet-dir logs `
    --since 2026-05-05T18:00:00 `
    --out sim_clean.parquet
```

The simulator also automatically deduplicates rows by `(asset, ts_ns)` so
overlapping data from a restart within the same hour doesn't double-count.

### 7.2 Sweep all threshold combinations (~120 combinations)

This takes a few minutes — it re-runs the exit simulator for every parameter
combination.

```powershell
python -m polybot.research.parameter_sweep `
    --parquet-dir logs `
    --out sweep_results.parquet `
    --top 20

# With --since to exclude bad early data:
python -m polybot.research.parameter_sweep `
    --parquet-dir logs `
    --since 2026-05-05T18:00:00 `
    --out sweep_results.parquet
```

Output: table of top-20 combinations by Sharpe ratio, plus the best combo
highlighted.

### 7.3 View sweep results

```powershell
# Top combinations by Sharpe
python -m polybot.cli.polybot_metrics sweep --file sweep_results.parquet --top 10

# Edge realization slope (CLAUDE.md §10.6 — the decisive diagnostic)
python -m polybot.cli.polybot_metrics edge-slope --sweep sweep_results.parquet
```

### 7.4 Full diagnostic sequence

```powershell
# 1. Model quality over the run
python -m polybot.cli.polybot_metrics model

# 2. Edge distribution
python -m polybot.cli.polybot_metrics edge

# 3. Sweep
python -m polybot.research.parameter_sweep --parquet-dir logs --out sweep_results.parquet

# 4. Best parameters
python -m polybot.cli.polybot_metrics sweep --file sweep_results.parquet

# 5. Edge realization slope — the decisive test
python -m polybot.cli.polybot_metrics edge-slope --sweep sweep_results.parquet
```

---

## 8. Understanding the Results

### 8.1 The decisive test — edge realization slope

```
polybot-metrics edge-slope --sweep sweep_results.parquet
```

Prints a table bucketed by `entry_edge_signed` (x-axis) vs
`mean_realized_pnl_per_dollar` (y-axis), plus an OLS slope.

| Slope | Meaning |
|---|---|
| ≤ 0 | Model has no predictive value. Do not trade. |
| 0 < slope < 0.3 | Edge is real but costs eat most of it. Consider whether it's worth it. |
| 0.3 – 0.7 | Strong signal. Costs eat 30–70% of raw edge. Profitable strategy. |
| > 0.7 | Excellent. Most of raw edge realizes as P&L. |

### 8.2 Model quality thresholds

| Metric | Target | Meaning if missed |
|---|---|---|
| Median CV-R² | > 0.25 | < 0.10 = Polymarket doesn't actually lag spot |
| Lag estimate `lag_s` | 15 – 90s | < 5s = market makers too fast; > 120s = noisy |
| `sd_lag_s` hourly | < 25s | Wild swings = model fitting noise |
| `q_predicted` residual | < 0.03 | Model isn't tracking Polymarket's current level |

### 8.3 Strategy health thresholds

| Metric | Target | Where to check |
|---|---|---|
| Total P&L | Positive | `polybot_metrics sweep` |
| Lag-close exit rate | ≥ 50% | `polybot_metrics sweep` |
| Win rate | ≥ 50% | `polybot_metrics sweep` |
| Mean edge | 0.005 – 0.025 | `polybot_metrics edge` |
| Entry rate (excl. warmup) | 1 – 10% | `polybot_metrics engage` |
| Model-disagree abstain rate | < 5% | `polybot_metrics summary` |

---

## 9. Deciding Whether to Go Live

Advance to Phase 2 (real money) **only if all of the following are true**
after the 24-hour dry run:

- [ ] **Total P&L is positive** with the best parameter combination
- [ ] **Median CV-R² > 0.25** across all regression refits
- [ ] **Estimated lag is stable** (sd < 25s) and in the 15–120s range
- [ ] **`q_predicted` mean residual < 0.03** — model tracks market level
- [ ] **Edge realization slope > 0** and statistically significant
- [ ] **No systematic directional bias** (`|mean_signed_edge| < 0.008`)
- [ ] **≥ 50% of exits are lag-close** (thesis is actually working)
- [ ] **Win rate > 50%** in the top 3 tiers (edge ≥ 0.08)
- [ ] **< 5% of windows had `K_uncertain=true`**

If results are mixed (slope is positive but small, P&L near zero), do NOT
go live. Instead, extend the dry run to 7 days and re-evaluate. The 24h
sample is small; a near-zero result can flip either direction with more data.

---

## 10. Running Dry Run vs Live on Different Machines

**Yes — you can run the dry run locally and the live bot on your VM.** Here's exactly how:

### Dry run on your US machine (Windows)

```env
DRY_RUN=true
PARQUET_DIR=logs          # stays local
```

- Coinbase + Polymarket feeds work fine from any US IP.
- Binance may or may not connect depending on your VPN. Either way, the
  Coinbase fallback (added in the circuit breaker fix) keeps decisions flowing.
- All parquet logs accumulate in `./logs/` on this machine.

### Live trading on your VM (non-US IP)

```env
DRY_RUN=false
PARQUET_DIR=/opt/polybot/logs
PRIVATE_KEY=0x...         # required
API_KEY=...
API_SECRET=...
API_PASSPHRASE=...
FUNDER=0x...
```

- Binance connects cleanly (non-US IP, no geo-block).
- Both Binance + Coinbase feed the regression → better signal quality.
- Real orders placed on Polymarket.

### Step-by-step: going live on the VM

**Step 1 — Fund your Polymarket wallet**

Deposit USDC to your Polymarket account (the address shown on your profile page).
Start small: $50–$100 for the first week.

**Step 2 — Derive API credentials (run once)**

```bash
# On the VM, with PRIVATE_KEY set in .env:
python scripts/derive_creds.py
```

Copy the three printed values into `.env`:
```env
API_KEY=...
API_SECRET=...
API_PASSPHRASE=...
```

**Step 3 — Verify balance and credentials**

```bash
python scripts/check_balance.py
```

Both on-chain USDC and CLOB balance should show your deposited amount.
If CLOB balance is 0, you may need to approve USDC allowance via the
Polymarket UI first.

**Step 4 — Apply best parameters from dry run sweep**

Edit `polybot/infra/config.py` on the VM with the best values from
`polybot-metrics sweep`:

```python
LAG_CLOSE_THRESHOLD = 0.005   # replace with your sweep winner
STOP_THRESHOLD      = 0.030
FALLBACK_TAU_S      = 10
```

**Step 5 — Set DRY_RUN=false**

```env
DRY_RUN=false
FUNDER=0xYOUR_POLYMARKET_ADDRESS
```

**Step 6 — Start the bot**

```bash
python -m polybot.main
```

The bot cold-starts the regression (model_warmup for ~30–60 minutes),
then begins placing real orders when edge is detected.

**Watch for these log lines:**
```
OrderClient initialised | dry_run=False | funder=0x...
wallet balance: $XX.XX USDC
refitter asset=btc fitted version=... r2_cv=0.3XX lag_s=47.3
ORDER PLACED | token=... | side=BUY | shares=XX.XX | price=0.72 | order_id=...
fill_tracker entry recorded order=... asset=btc side=up shares=41.67 price=0.7200 usd=30.00
```

**Hard stop:** If session losses exceed 5% of starting wallet, the bot logs
`HARD STOP` at ERROR level and refuses all new entries. Investigate before
restarting.

### What to transfer from dry run to live

The regression **does not transfer** — it refits from scratch on live data
and is trading within ~1 hour.

What does transfer: the three threshold values (`LAG_CLOSE_THRESHOLD`,
`STOP_THRESHOLD`, `FALLBACK_TAU_S`) from your sweep. Edit those three
lines in `polybot/infra/config.py` on the VM before starting.

### Live-API smoke tests (run BEFORE starting the bot)

Before flipping `DRY_RUN=false` and running the full bot on real money, prove
that order placement actually works with three small standalone scripts.

**Step A — Dry-run check (no money moves)**

```bash
# Plans the trade, prints intent, doesn't place an order. Validates that
# market resolution + WS book reading + order-args construction all work.
python scripts/test_trade_buy.py --dry-run
```

You should see the resolved slug, the live Up/Down asks, and the order plan.
No order_id is created on the exchange.

**Step B — Single $5 BUY (real money)**

```bash
python scripts/test_trade_buy.py
```

Verify:
- `ORDER PLACED | side=BUY | shares=… | order_id=…` appears in the log
- An entry shows up in your Polymarket activity page
- `.last_test_order.json` is written to disk

**Step C — Cash out the position**

```bash
python scripts/test_trade_sell.py
```

Verify:
- `ORDER PLACED | side=SELL | …` appears
- The position closes on Polymarket activity page
- `.last_test_order.json` now has a `sell` block with the exit price + gross P&L

**Step D — Full cycle in one shot**

```bash
# Default: $5 BUY → hold 30s → SELL. Tweak --usd / --hold / --side as needed.
python scripts/test_full_cycle.py

# Or fully simulate without spending:
python scripts/test_full_cycle.py --dry-run
```

This is the closest single-script approximation of one bot trade. Expect a
small loss (~5–7% of size) on a 30s round-trip due to the entry+exit fees;
the bot's lag-arb edge is what makes the math work in production.

**If any test fails**, capture the full error output and the contents of
`.last_test_order.json`. The most common failure modes:

| Error | Cause | Fix |
|---|---|---|
| `Could not resolve … market` | Window just closed/rolled; gamma not yet caught up | Wait 5–10s and retry |
| `403 Forbidden` on order POST | API credentials invalid or signed for wrong wallet | Re-run `derive_creds.py` and re-paste into `.env` |
| `Insufficient balance` | CLOB collateral is 0 even though on-chain USDC is funded | Approve USDC allowance via the Polymarket UI |
| `ERROR: bad ask price: 0.0` | WS connected but book event was empty (one-sided market) | Retry — usually self-corrects within a window |
| `No book snapshot within 5s` | Polymarket WS slow on first connect | Pass `--book-wait 15` to give it more time |

Only after **all four steps succeed** should you flip `DRY_RUN=false` and
launch the full bot.

---

## 11. Troubleshooting

Patterns that came up during dry-run development. Match the symptom and
apply the listed fix.

### `sampler BTC: wrote=0 no_K=60` — K never gets captured

The window opened but no Chainlink or spot price was applied as K.

- `K_SOURCE=spot` (default) requires `ENABLE_BINANCE=true` (or
  `ENABLE_COINBASE=true`) and a connected, fresh feed. Confirm with
  `binance_ws connected` / `coinbase_ws connected` lines at startup.
- `K_SOURCE=chainlink` requires Polymarket RTDS to deliver a fresh tick
  within 30 s of window open. RTDS often misses cold starts; switch to
  `K_SOURCE=spot` unless you specifically need oracle K.

### `sampler ETH: wrote=0 poly_not_ready=60`

The Polymarket WS is connected but the Up book has zero levels. Common
late in a window when one side has fully resolved. Self-corrects at the
next rollover. If it persists across multiple windows, check your token
IDs in the rollover line vs the Polymarket UI.

### `sampler …: wrote=0 incomplete=60`

Features failed the completeness check. Check the one-shot
`sampler X incomplete (first occurrence): …` line — the False fields
tell you which lag slot is missing. If `bn_ready=False`, Binance isn't
connected. If `cb_ready=False` and Binance is also down, you have no
spot feed.

### Refitter logs `n=0` despite `fitting on 475 rows`

Bookkeeping bug — the model still fits on all rows, the count is just
mis-reported. Restart to pick up the fix from this session.

### `r2_cv` stays negative for hours

Not necessarily a bug. Run the morning `x_now stdev` check (§5.7). If
stdev is very low (< 0.0005), BTC simply didn't move enough K-relatively
for the regression to learn. Wait for a more volatile period or extend
the run.

### `polymarket_ws reconnect consec=N err=unhandled errors in a TaskGroup`

The previous error message hid the real exception. The current code
unwraps `BaseExceptionGroup` and logs `err_type=…`. The `demo_polymarket_ws.py`
script works fine because it doesn't share the event loop with the
sampler / parquet writer / refitter — under bot load, a slow PONG reply
(>10 s) gets the connection dropped server-side. Reconnects are normal;
they only become a problem if `consec` keeps climbing past ~5.

### Heartbeat keeps logging "training warmup" after a successful fit

Cosmetic. After the first successful fit the line switches to
`training rolling-window status — BTC: NNN rows, r2_cv=0.XXX`. If you
still see `training warmup` after a `fitted version=…` line, restart to
pick up the fix.

### Logs missing the "RUNNING BTC UP OR DOWN…" banner

The banner fires when K transitions from `None` → set, via any of three
paths (spot, strict Chainlink, deadline fallback). If you never see it,
no K was ever captured — see the first troubleshooting entry above.

---

## 12. File Reference

### Bot source

| File | Purpose |
|---|---|
| `polybot/main.py` | Entry point — 8-task asyncio TaskGroup |
| `polybot/infra/config.py` | **All tunables** — strategy params, feed URLs, circuit breakers |
| `polybot/infra/scheduler.py` | 10-second decision tick |
| `polybot/infra/refitter.py` | Background regression refit every 5 min |
| `polybot/infra/parquet_writer.py` | Parquet log writer |
| `polybot/models/features.py` | Feature engineering (20 features) |
| `polybot/models/regression.py` | Ridge regression — fit, q_predicted, q_settled |
| `polybot/models/edge.py` | Edge calculation + Kelly tier sizing |
| `polybot/models/fees.py` | Taker fee curve: `Θ * p * (1-p)` |
| `polybot/models/slippage.py` | VWAP slippage estimator from book depth |
| `polybot/strategy/decision.py` | Entry / hold / exit state machine |
| `polybot/strategy/position.py` | In-memory position tracker |
| `polybot/clients/binance_ws.py` | Binance L2 WebSocket client |
| `polybot/clients/coinbase_ws.py` | Coinbase ticker WebSocket client |
| `polybot/clients/polymarket_ws.py` | Polymarket CLOB WebSocket client |
| `polybot/clients/polymarket_rtds.py` | Polymarket Chainlink oracle stream |
| `polybot/state/spot_book.py` | Binance L2 state (microprice, OFI, ring buffer) |
| `polybot/state/coinbase_book.py` | Coinbase L1 state (microprice, ring buffer) |
| `polybot/state/poly_book.py` | Polymarket order book state |
| `polybot/state/window.py` | 5-min window state (K, open_ts, slug) |

### Analysis tools

| File | Purpose |
|---|---|
| `polybot/research/cashout_simulator.py` | Replay exit decisions against logged data |
| `polybot/research/parameter_sweep.py` | Sweep LAG_CLOSE × STOP × FALLBACK_TAU |
| `polybot/cli/polybot_metrics.py` | DuckDB diagnostic CLI |
| `polybot/cli/polybot_ctl.py` | Bot control (status, kill switch) |
| `scripts/visualize_market.py` | Live terminal dashboard |

### Logs written during dry run

```
logs/
  decisions/dt=YYYY-MM-DD/asset=btc/h=HH.parquet   ← primary — one row per 10s tick
  decisions/dt=YYYY-MM-DD/asset=eth/h=HH.parquet
  model_versions/dt=YYYY-MM-DD/asset=btc.parquet    ← one row per 5-min refit
  model_versions/dt=YYYY-MM-DD/asset=eth.parquet
  window_outcomes/dt=YYYY-MM-DD/asset=btc.parquet   ← one row per resolved window
  window_outcomes/dt=YYYY-MM-DD/asset=eth.parquet
```

Generated by post-run analysis:
```
sweep_results.parquet          ← all 120 threshold combinations, ranked by Sharpe
sim_default.parquet            ← cashout_simulator output for default thresholds
```
