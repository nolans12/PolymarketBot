# PolymarketBot (Kalshi branch)

Lag-arbitrage bot for Kalshi's 15-minute BTC/ETH/SOL/XRP binary markets.

**Edge:** Coinbase spot price leads Kalshi's order book by ~15-90 seconds. The bot buys when Kalshi is cheap relative to where spot says it should be, exits when the gap closes. Strategy is market-neutral — whether BTC goes up or down doesn't matter, only whether Kalshi catches up before the window closes.

---

## Quick start

```bash
# 1. Set up credentials
cp .env.example .env
# edit .env with your KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_FILE

# 2. Install deps
pip install -r requirements.txt

# 3. Check auth
python scripts/test/check_kalshi_balance.py

# 4. Run (dry run, no real orders)
python scripts/run/run_kalshi_bot.py
```

Output goes to `data/<timestamp>_BTC/`. Analyze with:
```bash
python scripts/analysis/analyze_run.py
```

See **RUN.md** for full instructions.

---

## How it works

1. **Coinbase WS** streams BTC microprice in real time
2. **Kalshi REST** polls the order book at 1 Hz
3. A **ridge regression** learns the lag relationship from recent history (refits every 5 min on a rolling 4-hour window)
4. `q_settled` = model prediction with current spot substituted into all lag slots = "where Kalshi will quote once it catches up"
5. Edge = `q_settled - yes_ask - fees`. If edge clears a Kelly tier, enter. Exit when edge compresses to zero.

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
| `scripts/analysis/` | Post-processing and visualization |
| `scripts/test/` | Auth checks and round-trip trade tests |
| `data/<run>/` | Per-run tick CSVs and decision logs |
| `CLAUDE.md` | Architecture, model design, config reference |
| `RUN.md` | Step-by-step operating instructions |

---

## Status

Phase 1 — dry run / data collection. Real orders are off by default (`DRY_RUN=true`).
