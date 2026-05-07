# PolymarketBot (Kalshi branch)

Lag-arbitrage bot for Kalshi's 15-minute BTC/ETH/SOL/XRP binary markets.

**Edge:** Coinbase spot price leads Kalshi's order book by ~15-90 seconds. The bot buys when Kalshi is cheap relative to where spot says it should be, exits when the gap closes. Strategy is market-neutral — whether BTC goes up or down doesn't matter, only whether Kalshi catches up before the window closes.

---

## Workflow

```
Step 1 — Dry run (collect data, no real orders):
  python scripts/run/run_kalshi_bot.py
  Let it run for 24+ hours to build a solid training dataset.

Step 2 — Train model:
  python scripts/train_model.py
  Fits a LightGBM multi-horizon model on your collected ticks.
  Saves to model_fits/<name>.pkl

Step 3 — Test model:
  python scripts/test_model.py --model-file model_fits/<name>.pkl
  Simulates P&L on the same data. Use --sweep to tune entry/exit thresholds.

Step 4 — Go live:
  python scripts/run/run_kalshi_bot.py --model-file model_fits/<name>.pkl --live-orders
  Boots instantly (model pre-trained, no warmup wait).
  Continues to refit every 5 min as live data arrives.
```

---

## How it works

1. **Coinbase WS** streams BTC microprice in real time
2. **Kalshi REST** polls the order book at 1 Hz
3. A **LightGBM model** (trained offline on dry-run data) predicts `yes_mid` at multiple horizons: 5s, 10s, 15s, 60s ahead. The 10s prediction is the primary edge signal.
4. `q_settled` = model's 10s-ahead forecast at the current feature vector
5. Edge = `q_settled - yes_ask - fees`. If edge clears a Kelly tier, enter. Exit when edge compresses to zero.
6. In the background the bot keeps refitting every 5 min — if the new fit beats R2_hld of the loaded model, it adopts it.

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

# 4. Dry run (collect data)
python scripts/run/run_kalshi_bot.py
```

See **RUN.md** for the full workflow including training and live deployment.

---

## Multi-asset

```bash
python scripts/run/run_kalshi_bot.py --assets BTC ETH SOL XRP
```

Each asset gets its own independent model, KalshiBook, and Scheduler. One Coinbase WebSocket connection feeds all four SpotBooks.

---

## Files

| Path | Purpose |
|------|---------|
| `betbot/kalshi/` | All bot code |
| `scripts/run/run_kalshi_bot.py` | Main entrypoint |
| `scripts/train_model.py` | Fit LightGBM on dry-run ticks, save to model_fits/ |
| `scripts/test_model.py` | Simulate P&L with a saved model, tune thresholds |
| `scripts/analysis/` | Visualization (analyze_run, replay_window, live_plot) |
| `scripts/test/` | Auth checks and round-trip trade tests |
| `data/<run>/` | Per-run tick CSVs and decision logs |
| `model_fits/` | Saved trained models (.pkl + .json metadata) |
| `CLAUDE.md` | Architecture, model design, config reference |
| `RUN.md` | Step-by-step operating instructions |
