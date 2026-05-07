# PolymarketBot (Kalshi branch)

Lag-arbitrage bot for Kalshi's 15-minute BTC/ETH/SOL/XRP binary markets.

**Edge:** Coinbase spot price leads Kalshi's order book by ~15-90 seconds. The bot buys when Kalshi is cheap relative to where spot says it should be, exits when the gap closes. Strategy is market-neutral — whether BTC goes up or down doesn't matter, only whether Kalshi catches up before the window closes.

---

## Workflow

```
Step 1 — Dry run (collect data, no real orders):
  python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP
  Let it run for 24+ hours to build a solid training dataset.

Step 2 — Train model:
  python scripts/train_model.py
  Fits a LightGBM multi-horizon model on your collected ticks.
  Saves to model_fits/<run>_<ASSET>_<date>/model.pkl

Step 3 — Tune trading knobs:
  python scripts/tune_trading_knobs.py --model-file model_fits/<dir>/model.pkl
  Grid-sweeps Kelly tiers + exit thresholds on the held-out data.
  Apply the suggested config snippet to config.py.

Step 4 — Full simulation:
  python scripts/test_all.py --model-file model_fits/<dir>/model.pkl
  Replays ticks with model + tuned config. Verify P&L before going live.

Step 5 — Go live:
  python scripts/run/run_kalshi_bot.py --model-file model_fits/<dir>/model.pkl --live-orders
  Static model, trades from tick 1, no warmup.
```

See **RUN.md** for the full workflow with exact commands and expected output.

---

## How it works

1. **Coinbase WS** streams BTC/ETH/SOL/XRP microprice in real time (one connection, all assets)
2. **Kalshi REST** polls each asset's order book at 1 Hz
3. A **LightGBM model** (trained offline) predicts `yes_mid` at 4 horizons: 5s, 10s, 15s, 60s ahead. The 10s prediction is the primary edge signal.
4. `q_settled` = model's 10s-ahead probability forecast
5. Edge = `q_settled - yes_ask - fees`. If edge clears a Kelly tier, enter. Exit when edge compresses to zero.
6. Model is **static** — never retrained live. Collect more dry-run data and retrain offline if needed.

---

## Quick start

```bash
# 1. Set up credentials
cp .env.example .env
# edit .env: KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_FILE

# 2. Install deps
pip install -r requirements.txt

# 3. Check auth
python scripts/test/check_kalshi_balance.py

# 4. Dry run all 4 assets (collect training data)
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP
```

---

## Multi-asset

```bash
# Dry run all assets
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP

# Live with per-asset models
python scripts/run/run_kalshi_bot.py \
  --assets BTC ETH SOL XRP \
  --model-file BTC=model_fits/btc_run/model.pkl,ETH=model_fits/eth_run/model.pkl \
  --live-orders
```

Each asset gets its own independent KalshiBook and Scheduler. One Coinbase WebSocket connection feeds all four SpotBooks.

---

## Files

| Path | Purpose |
|------|---------|
| `betbot/kalshi/` | All bot code |
| `scripts/run/run_kalshi_bot.py` | Main entrypoint |
| `scripts/train_model.py` | Fit LightGBM on dry-run ticks, save to model_fits/ |
| `scripts/tune_trading_knobs.py` | Grid-sweep Kelly tiers and exit thresholds |
| `scripts/test_all.py` | Full simulation with loaded model + tuned config |
| `scripts/replay_window.py` | Animated replay of one 15-min window with model projections |
| `scripts/analysis/analyze_run.py` | Visualize decisions log |
| `scripts/analysis/live_plot.py` | Live rolling chart while bot runs |
| `scripts/test/` | Auth checks and round-trip trade tests |
| `data/<run>/` | Per-run tick CSVs and decision logs (gitignored) |
| `model_fits/<run>/` | Saved trained models: model.pkl + model.json (gitignored) |
| `CLAUDE.md` | Architecture, model design, config reference |
| `RUN.md` | Step-by-step operating instructions |
