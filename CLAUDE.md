# CLAUDE.md — Kalshi 15-Minute BTC Lag-Arbitrage Bot

> **Status:** Phase 1 dry run — data collection + offline backtest. Real money is **not** placed by the live decision loop. The only path that touches money is `scripts/test_trade.py` (a one-off round-trip sanity check). This document is the source of truth for the project's design, architecture, and reasoning. Read it end-to-end before generating any code.

---

## 1. Project goal

Build an automated trading bot that takes positions in **Kalshi's 15-minute BTC Up/Down binary markets** (`KXBTC15M` series) by exploiting the lag between **Coinbase BTC spot price movements** and Kalshi's order-book repricing. The bot enters when Kalshi has not yet caught up to where Coinbase says fair value is, and exits when Kalshi has caught up — collecting the lag-close as profit, regardless of how the underlying window resolves.

**The strategy is cash-out lag arbitrage. There is one model, one edge calculation, one entry rule, one exit rule.**

### 1.1 Edge thesis

Coinbase's order book is the leading indicator. Kalshi's REST-quoted YES/NO prices take some number of seconds to reprice after Coinbase moves. During that interval, Kalshi's quoted probability differs from the spot-implied probability by an amount that — when it exceeds round-trip costs (entry fee + entry slippage + exit fee + exit slippage + spread) — is exploitable.

The user has empirically observed this lag in the `lag_plot_btc.png` plot saved at the repo root. The dry run quantifies it via the regression's fitted coefficients (see §6.4) and confirms it via realized P&L in backtest replay.

**The bot's job, distilled to one sentence:** every 10 seconds, compute a single number — the net edge `delta` in probability points after fees and slippage — and feed that number into the tiered Kelly table to produce a bet size. If `delta` is below the lowest tier threshold, abstain. Otherwise enter at the prescribed size and exit when `delta` collapses (lag closed) or inverts (thesis broken).

### 1.2 Why Kalshi + Coinbase (the new stack)

The previous Polymarket + Binance stack didn't work because both venues are **geo-restricted from US IPs**. Polymarket's CLOB blocks US persons; Binance's L2 spot feed blocks US connections. Running the bot from a US machine without a VPN was effectively impossible.

The Kalshi + Coinbase stack is **fully US-legal**:

- **Kalshi** is the CFTC-regulated US prediction market exchange. It runs `KXBTC15M`, a series of 15-minute "BTC ≥ strike at close" binary markets with multiple strike bins per window. No VPN required.
- **Coinbase Advanced Trade** is the spot feed. The WebSocket ticker channel exposes top-of-book bid/ask plus quantities, which is enough to compute microprice (the regression's primary spot input). No VPN required.

This branch (`kalshi`) is a fresh implementation built from scratch in `betbot/kalshi/`. The legacy `betbot/clients/`, `betbot/models/`, `betbot/state/`, `betbot/strategy/`, `betbot/infra/`, `betbot/research/`, and `betbot/main.py` directories are **dead code** — they import from a non-existent `polybot.*` package and date back to the Polymarket era. Do not edit or run any of them; the live bot lives entirely under `betbot/kalshi/`.

### 1.3 Why not Black-Scholes binary?

Black-Scholes prices an option by assuming GBM with volatility σ; it answers "what's the no-arbitrage probability the underlying ends above K at horizon τ?" That's a forecasting question. We're not forecasting — we're predicting where Kalshi's *quote* will be in N seconds based on where Coinbase's *spot* is right now. That's a regression problem on (Kalshi price) vs (lagged spot features), and modeling it as such is more honest, more accurate, and removes an entire layer of estimation noise (volatility) from the critical path.

### 1.4 Hard constraint

Every entry decision is reducible to: "regression-predicted settled probability `q_settled`, market price `q_actual` at Kalshi YES ask, edge `delta = |q_settled − q_actual| − fee − slippage` clears the tier threshold, bet wallet fraction prescribed by Kelly tier T." Every exit decision is reducible to: "Kalshi has caught up (`delta` has compressed below the lag-close threshold → take profit), or spot reversed (`delta` has inverted by more than the stop threshold → cut loss), or τ is small enough to default to resolution."

### 1.5 Non-goals

- We are **not** building a market-making bot. No quoting both sides.
- We are **not** chasing sub-second latency arbitrage. Our edge window is 15-90+ seconds.
- We are **not** trading anything other than `KXBTC15M`. ETH support is removed; the previous `KXETH15M` plumbing has been deleted.
- We are **not** predicting BTC's close-time price. We are predicting Kalshi's *next several seconds* of quote movement.

---

## 2. The market mechanic

Every 15 minutes, Kalshi opens a fresh batch of `KXBTC15M-{date}{time}` markets — one per strike bin (e.g. one for "BTC ≥ $108,000", one for "BTC ≥ $108,500", etc.). Each market resolves YES = $1 if BTC's settlement price at the close boundary is at-or-above its `floor_strike`, otherwise NO = $1.

- **Strike `K` (floor_strike):** a fixed dollar threshold set by Kalshi when the market opens. **It is NOT the spot price at window-open.** Each window has multiple strike bins; the bot picks the at-the-money bin (the one whose strike is closest to current BTC spot) for maximum lag-arb potential.
- **Resolution price:** Kalshi's BTC settlement price at the close boundary (Kalshi publishes its own settlement methodology; the bot doesn't depend on it because the strategy exits before close).
- **YES contract** pays $1 if close ≥ floor_strike, else $0. NO is the complement. Prices live in `[0.00, 1.00]`.
- `yes_ask` is the cheapest dollar price to buy a YES share; `yes_bid` is the highest dollar price someone will pay for a YES share. `no_ask = 1.00 − yes_bid` and `no_bid = 1.00 − yes_ask`.
- Window ticker is deterministic in the form `KXBTC15M-{YYMMMDDHHMM}-{strike_index}`, e.g. `KXBTC15M-26MAY061515-15`.

**Cash-out is supported.** A YES position can be exited at any time before close by selling into the order book at the prevailing YES bid. This is what makes the strategy viable. The exit price is set by the bid, not by resolution — we are exiting *into the market*, not *waiting for settlement*.

---

## 3. The strategy

### 3.1 Conceptual model

Treat Coinbase as ground truth and Kalshi as a delayed function of Coinbase. At any moment `t`:

- `q_actual_t` = what Kalshi *is* quoting (use `yes_mid = (yes_bid + yes_ask) / 2` as the reference).
- `q_settled_t` = what Kalshi *would* quote if it had finished digesting all spot moves up to time `t`.
- `delta_t = q_settled_t − q_actual_t` (signed, on the side we're considering).

When `delta` is large and positive on the YES side, Kalshi is offering YES shares cheaper than fair value. We buy. Some seconds later, Kalshi's book updates to reflect the spot move, `q_actual` rises toward `q_settled`, the lag closes, and we sell back at the new (higher) market bid.

The model's job is to compute `q_settled_t` from observable spot data. That's what the lead-lag regression does.

### 3.2 Why microprice, not midpoint

The spot-side input we use is **microprice**, the imbalance-weighted fair value:

```
microprice = (best_bid × ask_size + best_ask × bid_size) / (bid_size + ask_size)
```

When the bid is heavy, microprice is closer to the ask (next trade likely lifts the ask). When the ask is heavy, microprice is closer to the bid. This is the cleanest available proxy for "where is spot heading in the next few seconds" using only L1 book data.

Coinbase's `ticker` channel exposes `best_bid`, `best_ask`, `best_bid_quantity`, `best_ask_quantity` on every trade or quote update — exactly what microprice needs. See `betbot/kalshi/coinbase_feed.py` and `CoinbaseBook.apply_ticker()` in `book.py`.

### 3.3 The lead-lag regression — the only model

We fit a ridge regression that predicts the logit of Kalshi's current YES mid-price from Coinbase spot history and a few microstructure features. The model's parameters are refit every 5 minutes on a rolling 4-hour window of recent (X, y) pairs.

**Target:**

```
y_t = logit(yes_mid_t)        where yes_mid_t = (yes_bid_t + yes_ask_t) / 2
```

**Features** (12 total, defined in `betbot/kalshi/features.py:FEATURE_NAMES`):

```
Spot history (microprice / K, in log space) — the lead-lag core:
  x_0    = log(microprice_now      / K)
  x_15   = log(microprice_{t-15s}  / K)
  x_30   = log(microprice_{t-30s}  / K)
  x_60   = log(microprice_{t-60s}  / K)
  x_90   = log(microprice_{t-90s}  / K)
  x_120  = log(microprice_{t-120s} / K)

Time:
  tau_s         = seconds until window close
  inv_sqrt_tau  = 1 / √(tau_s + 1)         # near-close moves matter more

Spot momentum:
  cb_momentum_30s = log(microprice_now / microprice_{t-30s})
  cb_momentum_60s = log(microprice_now / microprice_{t-60s})

Kalshi microstructure:
  kalshi_spread       = yes_ask − yes_bid           # liquidity proxy
  kalshi_momentum_30s = yes_mid_now − yes_mid_{t-30s}
```

**Model:** ridge regression with cross-validated regularization. `RidgeCV` over alphas `[0.001, 0.01, 0.1, 1.0, 10.0]`, `TimeSeriesSplit` CV (5 folds when data permits), `StandardScaler` normalization. Implemented in `betbot/kalshi/model.py:KalshiRegressionModel`.

**Two derived predictions are computed at every decision tick:**

```
# 1. q_predicted: what the model thinks Kalshi SHOULD be quoting given current data.
#    Sanity check: if this is far from yes_mid, the model is broken or the regime
#    has shifted (regime change, stale fit, feed problem).
q_predicted = sigmoid(model.predict(current_features_as_array))

# 2. q_settled: what Kalshi WILL quote once it has digested current spot.
#    Substitute x_0 (current spot/K) into every lookback slot, then predict.
q_settled = sigmoid(model.predict(features.settled_array()))
```

`features.settled_array()` clones the live feature vector and replaces every `x_15..x_120` slot with `x_0`. This forces the regression to evaluate "what would the model quote if Kalshi had already seen the current spot in every historical slot?" That output is `q_settled` — our forecast of where Kalshi will arrive once it finishes processing current spot. **This is the single most important output of the model.**

### 3.4 The lag is learned, not hard-coded

The fitted β coefficients on `x_0..x_120` *are* the lag distribution. They tell us how much of Kalshi's current quote is explained by spot at each lookback horizon. Three patterns the data could show:

- **Kalshi is fast (no lag).** `β_0` is dominant; `β_15..β_120` are near zero. Kalshi's quote is best explained by spot right now — no lag to arbitrage. **Strategy thesis is dead.**
- **Kalshi lags by ~60 seconds.** `β_0` is small, `β_60` is largest, neighboring β's decay smoothly. This is the regime our strategy needs.
- **Mixed / no clear lag.** All β's moderate, no clear peak. Strategy can still work but edge is smaller and noisier.

We don't have to pick which regime is true — the regression tells us. For a single human-readable summary number, the model logs the β-weighted average lag at each refit (`KalshiRegressionModel.fit()` → `ModelDiagnostics.estimated_lag_s`):

```
estimated_lag_s = (15·|β_15| + 30·|β_30| + 60·|β_60| + 90·|β_90| + 120·|β_120|)
                / (|β_15| + |β_30| + |β_60| + |β_90| + |β_120|)
```

This is logged every 5 minutes for human interpretability. The bot doesn't *use* it (it uses the full coefficient vector); the bot logs it. If `estimated_lag_s` drifts from 45s down to 12s over a few days, that's quantitative evidence that Kalshi's market makers are getting faster and our edge is shrinking.

### 3.5 Computing the edge

Every 10 seconds, after computing `q_settled`, compute the edge on each side. Implemented in `Scheduler._tick()` in `scheduler.py`.

```python
edge_up_raw = q_settled - yes_ask                       # buy YES at ask
edge_no_raw = (1 - q_settled) - (1 - yes_bid)           # buy NO at (1 - yes_bid)
                                                        # equals  yes_bid - q_settled

# Cost adjustments (per dollar bet):
fee_up = THETA_FEE * yes_ask * (1 - yes_ask)            # THETA_FEE = 0.07
fee_no = THETA_FEE * (1 - yes_bid) * yes_bid

# Slippage from current Kalshi book depth — see _slippage() helper
slip_up = min(0.02, intended_size_usd / yes_depth * 0.01)
slip_no = min(0.02, intended_size_usd / no_depth  * 0.01)

edge_up_net = edge_up_raw - fee_up - slip_up
edge_no_net = edge_no_raw - fee_no - slip_no

# Pick best side, report magnitude for tier lookup
if edge_up_net >= edge_no_net:
    edge_signed, favored_side = edge_up_net, "yes"
else:
    edge_signed, favored_side = edge_no_net, "no"

edge_magnitude = abs(edge_signed)
```

`edge_magnitude` is the number that drives every entry decision. If it's below the lowest tier floor (0.02), abstain. Otherwise look up the tier and bet the corresponding wallet fraction on `favored_side` at the corresponding ask.

### 3.6 Worked example

Consider a single 10-second decision tick on an at-the-money KXBTC15M market:

```
Window opened ~5 minutes ago, 10 minutes (600s) remaining.
floor_strike (K) = 108,000
tau_s            = 600

Coinbase right now:
  microprice_t           = 108,300        ($300 above strike)
  microprice 60s ago     = 108,150
  microprice 120s ago    = 108,050

Kalshi right now:
  yes_bid = 0.62, yes_ask = 0.64, yes_mid = 0.63
  yes_depth = $400 on the YES bid side

Fitted regression (rolling 4h, cross-validated):
  β_0   = 0.4          (small — spot-now barely explains Kalshi-now)
  β_60  = 4.1          (DOMINANT — lag is ~60s)
  β_120 = 1.5
  others moderate

Step 1 — q_predicted (sanity)
  Apply model to live feature vector → ≈ logit(0.61). Close to yes_mid 0.63;
  the model is healthy.

Step 2 — q_settled
  Substitute x_0 = log(108300/108000) into x_15..x_120 slots and predict.
  → q_settled ≈ 0.71

Step 3 — edge
  edge_up_raw = 0.71 - 0.64 = 0.07
  fee_up      = 0.07 * 0.64 * 0.36 = 0.0161
  slip_up     = min(0.02, 30 / 400 * 0.01) = 0.00075
  edge_up_net = 0.07 - 0.0161 - 0.00075 = 0.0532

  edge_magnitude = 0.0532, favored_side = "yes"

Step 4 — Kelly tier lookup
  KELLY_TIERS = [(0.30, 0.10), (0.15, 0.08), (0.08, 0.05),
                 (0.04, 0.03), (0.02, 0.015)]
  0.0532 falls in the (0.04, 0.03) tier → bet 3% of wallet.

Step 5 — entry (Phase 2 only; Phase 1 just logs)
  wallet = $1,000  ⇒  bet_usd = $30
  contracts = $30 / $0.64 = ~46.9 YES shares

Step 6 — wait. Over the next ~60s, if the lag closes:
  yes_ask rises 0.64 → ~0.70, yes_bid follows to ~0.69.
  Sell 46.9 contracts at yes_bid = 0.69:
    gross         = 46.9 * 0.69 = $32.36
    entry_fee     = 0.07 * 0.64 * 0.36 * 30 = $0.484
    exit_fee      = 0.07 * 0.69 * 0.31 * 32.36 = $0.485
    realized_pnl  = 32.36 - 30 - 0.484 - 0.485 = $1.39
    ROI on trade  = 4.6%
```

The trade made money because Kalshi caught up to spot. **The over/under outcome at resolution doesn't enter into the P&L on this trade — we already exited.** That's the entire point.

### 3.7 Entry rule

Every 10 seconds (in `_decision_loop()` → `_tick()`):

1. Active 15-min market loaded? If `not kb.ready or not cb.ready`, abstain (`data_not_ready`).
2. Read `tau_s = kb.tau_s()`, `yes_bid`, `yes_ask`, `yes_mid`.
3. Run sanity gates (see §7) — abstain on any failure.
4. Build features, compute `q_predicted` and `q_settled`.
5. If `|q_predicted − yes_mid| > 0.15` → abstain (`model_disagrees_market`).
6. Compute `edge_up_net`, `edge_no_net`, pick favored side, `edge_magnitude`.
7. If we already have an open position in this window → skip entry, run exit logic.
8. If `tau_s < FALLBACK_TAU_S` (60s) → abstain (`tau_too_small`).
9. If `yes_ask − yes_bid > 0.10` → abstain (`wide_spread`). *(Note: config defines `WIDE_SPREAD_THRESHOLD = 0.12` but the scheduler hardcodes 0.10 — see §12.)*
10. If `edge_magnitude > tier_floor` (lowest tier = 0.02) → enter on `favored_side` at the corresponding ask. Size = `wallet × kelly_fraction(edge_magnitude)`. Otherwise abstain (`edge_below_floor`).
11. Persist the `DecisionRow` (JSONL).

**One position per window.** Never flip sides mid-window. Never stack on the same side. Positions auto-clear on window rollover (`Scheduler._do_rollover()`).

**Tier table** (`KELLY_TIERS` in `config.py`, evaluated top-to-bottom, first match wins):

```python
KELLY_TIERS = [
    (0.30, 0.10),    # delta >= 0.30 → 10% wallet
    (0.15, 0.08),    # delta >= 0.15 →  8%
    (0.08, 0.05),    # delta >= 0.08 →  5%
    (0.04, 0.03),    # delta >= 0.04 →  3%
    (0.02, 0.015),   # delta >= 0.02 →  1.5%
]
```

### 3.8 Exit rule

```python
LAG_CLOSE_THRESHOLD = 0.005   # exit when edge has compressed to ½ cent
STOP_THRESHOLD      = 0.03    # exit if edge erodes 3 cents below entry edge
FALLBACK_TAU_S      = 60      # default to resolution at this τ (1 min before close)
```

Implemented in `Scheduler._evaluate_exit()`:

- For YES position: `edge_now = q_settled − yes_ask`; exit_price = current `yes_bid`.
- For NO position: `edge_now = (1 − q_settled) − (1 − yes_bid)`; exit_price = `1 − yes_ask`.

Decision priority:
1. If `edge_now < LAG_CLOSE_THRESHOLD` → **`exit_lag_closed`** (profit-taking).
2. Else if `edge_now < entry_edge − STOP_THRESHOLD` → **`exit_stopped`** (stop-loss).
3. Else if `tau_s < FALLBACK_TAU_S` → **`fallback_resolution`** (let the contract settle).
4. Else hold.

The thresholds are sweep parameters in the dry-run replay (`scripts/backtest.py --sweep`).

### 3.9 Process model

Single Python 3.11+ process, `asyncio.TaskGroup`. Four concurrent tasks (`scripts/run_kalshi_bot.py`):

1. **`CoinbaseFeed.run()`** — Coinbase Advanced Trade WS ticker for BTC-USD. Pushes ticks into `CoinbaseBook` at up to 20 Hz.
2. **`KalshiRestFeed.run()`** — REST poll `GET /trade-api/v2/markets/{ticker}` every 1 second. Pushes (yes_bid, yes_ask) into `KalshiBook`.
3. **`Scheduler.run()`** — itself a TaskGroup spawning four sub-loops:
   - `_sampler_loop()` — 1 Hz: build feature vector, append to `TrainingBuffer`, write raw tick to CSV.
   - `_refitter_loop()` — every 5 min: pull buffer arrays, fit `KalshiRegressionModel` in a thread executor, atomic-swap coefficients.
   - `_decision_loop()` — every 10 s: sanity gates, `q_settled`, edge, Kelly tier, entry/exit logic, log JSONL row.
   - `_window_manager_loop()` — every 10 s: pre-discover next ticker when `tau_s < 120`, switch on rollover when `tau_s ≤ 0`.

---

## 4. Architecture

### 4.1 Layout (only `betbot/kalshi/` is live)

```
/repo
  /betbot
    /kalshi                         # ← THE ENTIRE LIVE BOT IS HERE
      __init__.py
      config.py                     # All tunables; loads .env
      auth.py                       # Kalshi RSA-PSS request signing
      book.py                       # CoinbaseBook + KalshiBook (state)
      coinbase_feed.py              # Coinbase WS ticker → CoinbaseBook
      kalshi_rest_feed.py           # Kalshi REST polling → KalshiBook
      features.py                   # FeatureVec + build_features()
      model.py                      # KalshiRegressionModel (RidgeCV)
      training_buffer.py            # Rolling (X, y, ts) buffer
      tick_logger.py                # 1Hz CSV writer for backtests
      scheduler.py                  # Sampler / refitter / decision / window mgr
    /clients, /models, /state,      # ← DEAD legacy code (Polymarket era).
    /strategy, /infra, /research,   #   Imports from non-existent polybot.*
    /execution, /cli, main.py       #   Do not touch.
  /scripts
    run_kalshi_bot.py               # Live entrypoint
    backtest.py                     # Walk-forward replay on logs/ticks.csv
    analyze_run.py                  # Visualize logs/decisions.jsonl
    test_trade.py                   # One-off real $1 round-trip
    check_kalshi_balance.py         # Kalshi auth smoke test
    visualize_market.py             # Live 60s lag-visualization tool
    demo_binance.py, demo_coinbase.py  # Legacy demo scripts (dead)
  CLAUDE.md                         # This file
  DRYRUN.md                         # Step-by-step Phase 1 ops guide
  RUNBOOK.md                        # Older operations doc (partly stale)
  lag_plot_btc.png                  # Empirical lag observation
  pyproject.toml                    # Project metadata (still says "polybot" — vestigial)
  requirements.txt
  .env.example                      # Template for KALSHI_* + DRY_RUN
```

### 4.2 Data flow

```
Coinbase WS ──ticker tick──► CoinbaseBook (microprice + 5min ring buffer)
                                                │
Kalshi REST 1Hz poll ──(yes_bid, yes_ask)──► KalshiBook (yes_mid + ring buffer)
                                                │
                                                ▼
              build_features() ──FeatureVec──┐
                                             │
                                             ▼
   ┌──────────────► Sampler 1Hz ───► TrainingBuffer ───► Refitter 5min ───┐
   │                                                                       │
   └─── Decision 10s ────► q_settled, edge, Kelly tier ──► entry / exit ◄──┘
                                                              │
                                                              ▼
                                                logs/decisions.jsonl
                                                logs/ticks.csv
```

### 4.3 Process model

Single `asyncio` event loop. The CPU-heavy regression refit runs via `loop.run_in_executor(None, model.fit, ...)` so it doesn't block the decision tick. Coefficient swap inside `KalshiRegressionModel` is guarded by a `threading.Lock`.

---

## 5. Data sources and clients

### 5.1 Coinbase Advanced Trade WebSocket

Implemented in `betbot/kalshi/coinbase_feed.py`.

- **Endpoint:** `wss://advanced-trade-ws.coinbase.com`
- **Auth:** none required for the public `ticker` channel. (The legacy plan to use JWT-authed `level2` for OFI features is **not** implemented in the kalshi branch — we use only the unauthenticated ticker channel.)
- **Subscription:** `{"type":"subscribe","product_ids":["BTC-USD"],"channel":"ticker"}`.
- **Rate-limit:** ingest is capped client-side at 20 Hz (`MAX_HZ`).
- **Reconnect:** automatic on disconnect, 2-second backoff.
- **State derived in `CoinbaseBook`:**
  - `mid` = (best_bid + best_ask) / 2
  - **`microprice`** — the regression's primary input
  - 5-minute ring buffer of microprice samples at 1 Hz (powers `microprice_at(lag_s)` for lagged features)
  - `last_update_ns` — staleness watchdog input

### 5.2 Kalshi REST polling

Implemented in `betbot/kalshi/kalshi_rest_feed.py`. **The Kalshi branch deliberately does not use Kalshi's WebSocket** — early experiments hit silent-freeze and heartbeat issues, and REST polling at 1 Hz is more reliable and well within Kalshi's basic-tier 20 reads/sec limit.

- **Endpoint:** `GET /trade-api/v2/markets/{ticker}` against `https://api.elections.kalshi.com`.
- **Auth:** RSA-PSS over `timestamp + method + path`, headers `KALSHI-ACCESS-{KEY,TIMESTAMP,SIGNATURE}`. See `betbot/kalshi/auth.py`.
- **Cadence:** 1 Hz (`POLL_INTERVAL_S = 1.0`).
- **Response fields used:** `market.yes_bid_dollars`, `market.yes_ask_dollars` → fed into `KalshiBook.apply_ticker_update()`.
- **Failure handling:** 5 consecutive failures → close session and reopen; on HTTP 429 → 2 s backoff.
- **Ticker switching:** `feed.update_ticker(new_ticker)` on window rollover; the next poll uses the new ticker. No reconnect required.

### 5.3 Kalshi market discovery

Implemented in `Scheduler._list_active_markets()` and `_discover_market()`.

- Lists `KXBTC15M` markets with `status ∈ {open, active}` via `GET /trade-api/v2/markets?series_ticker=KXBTC15M&status=open`.
- Sorts by `close_time` (ascending), then by `floor_strike`.
- Picks the **at-the-money** market in the soonest-closing batch — i.e. the strike whose `floor_strike` is closest to current Coinbase BTC spot. This is the bin with the most uncertainty, and therefore the most lag-arb potential. (Deep ITM/OTM bins sit at 0.99/0.01 and offer no edge.)
- On rollover (`_window_manager_loop`): pre-discovers the next market when `tau_s < 120s`, switches when `tau_s ≤ 0`.

### 5.4 Kalshi fees (current model)

Kalshi charges a taker fee on each filled contract. The bot models it as:

```
fee_per_dollar(p) = THETA_FEE * p * (1 - p)        # THETA_FEE = 0.07
```

Peak ~1.75% per leg at p = 0.5. Round-trip cost (entry as taker + exit as taker) at p = 0.5 is roughly **3.5% of position notional** before slippage. The Kelly tier floor of 2% net edge accounts for this implicitly; the raw edge `q_settled − yes_ask` must exceed ~5% before the trade clears the lowest tier.

`THETA_FEE = 0.07` is a working estimate borrowed from Kalshi's published 2025 schedule and **must be calibrated against actual fills** before any conclusions about live profitability. See `scripts/test_trade.py` for the round-trip sanity check that produces a real fee number.

### 5.5 Slippage

Modeled crudely in `_slippage()`:

```python
def _slippage(book_depth: float, size_usd: float) -> float:
    if book_depth <= 0:
        return 0.02
    return min(0.02, size_usd / book_depth * 0.01)   # capped at 2 cents
```

This is **not** a VWAP-against-the-ladder model. It assumes a smooth penalty proportional to `size / depth`, capped at 2 cents per share. Phase 1 doesn't trade so this only affects the simulated edge calculation; in Phase 2 it should be replaced with a true depth-walking estimator.

---

## 6. The lead-lag regression in detail

### 6.1 Why this is the right model

The strategy thesis is "Kalshi lags Coinbase." The most direct way to express that hypothesis as a model is *literally* a regression of Kalshi's quote on lagged Coinbase prices. We don't model an option, infer a volatility, or assume a probability distribution. We just need to know how Kalshi's quote depends on recent spot history, and the regression learns that from data.

This formulation has properties no Black-Scholes formulation has:

- **Self-falsifying.** If R² is poor, Kalshi isn't actually predictable from spot the way we hypothesized. The dry run will tell us this directly.
- **Self-adapting.** As Kalshi's market makers get faster, the lag profile (β coefficients) shifts; the model picks this up at every refit. No human re-tuning needed.
- **No σ to be wrong about.** Volatility estimation is no longer a critical-path input.
- **The edge is a pure number, with units of probability.** `q_settled − yes_ask` is already in probability points, the same units as the Kelly tier thresholds. No translation needed.

### 6.2 Feature design

See §3.3 for the full list. A few notes on choices:

- The lookback grid `[0, 15, 30, 60, 90, 120]` seconds is intentionally dense in the 0-90s range and sparse beyond, since the user's empirical observation suggests Kalshi's lag falls in that range.
- `kalshi_spread` and `kalshi_momentum_30s` give the regression a way to learn that wide-spread or fast-moving Kalshi books need different treatment.
- All spot history is in `log(microprice / K)` so windows are comparable across the training data regardless of strike level.

### 6.3 Refitting

Implemented in `Scheduler._refitter_loop()` and `KalshiRegressionModel.fit()`.

- **Cadence:** every `REFIT_INTERVAL_S = 300` seconds (5 min).
- **Training data:** rolling `TRAINING_WINDOW_S = 4 * 3600` seconds (4 h).
- **Filter:** only train on samples where `yes_mid ∈ [0.05, 0.95]`. Extreme tails have huge logit values that disproportionately pull the regression and don't carry lag-arb signal anyway (the bot abstains in those regimes regardless).
- **Hold-out:** the most recent 20% of the time span is used for `r2_held_out` (out-of-sample sanity).
- **Cross-validation:** `TimeSeriesSplit(n_splits=min(5, n_train // 50))` over `RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]`.
- **Diagnostics logged every refit** (`ModelDiagnostics`): `n_train`, `ridge_alpha`, `r2_in_sample`, `r2_cv`, `r2_held_out`, all 12 coefficients, `estimated_lag_s`, `coef_delta_l2`.
- **Atomic swap:** new coefficients replace old under a lock; the decision loop never sees a half-updated model.

### 6.4 Cold start vs bootstrap

The regression needs about `MIN_TRAIN_SAMPLES = 360` samples (≈ 6 minutes at 1 Hz) before the first fit, and ideally several hours before coefficients are stable.

**Cold start:** if `logs/ticks.csv` doesn't exist, the bot enters a `model_warmup` abstention loop until enough live samples accumulate.

**Bootstrap from history:** if `logs/ticks.csv` exists from a prior run, `Scheduler._bootstrap_from_history()` replays the last 4 h of ticks at startup, rebuilds feature vectors using historical lagged microprices, and seeds the `TrainingBuffer`. If ≥ 360 samples are loaded, the model fits **before the first decision tick** so we don't waste an entire warmup phase. This is one of the most useful features for iterative development — kill the bot, change a feature, restart, and you're trading-ready immediately.

### 6.5 Computing `q_settled`

The core computation (`KalshiRegressionModel.q_settled()`):

```python
def q_settled(self, fv: FeatureVec) -> Optional[float]:
    if not self.is_fit or not fv.complete:
        return None
    logit = self._predict_raw(fv.settled_array())   # x_15..x_120 := x_0
    return _sigmoid(logit) if logit is not None else None

def q_predicted(self, fv: FeatureVec) -> Optional[float]:
    if not self.is_fit or not fv.complete:
        return None
    logit = self._predict_raw(fv.as_array())        # genuine lagged history
    return _sigmoid(logit) if logit is not None else None
```

`q_settled` is the headline output. `q_predicted` is a diagnostic — if it diverges substantially from `yes_mid`, the model is broken or the regime has shifted, and the bot abstains via `model_disagrees_market`.

---

## 7. Sanity gates and circuit breakers

The regression is the entire model, so we're careful about when to trust it. From `Scheduler._tick()`:

| Condition                                          | Action                                            |
|----------------------------------------------------|---------------------------------------------------|
| `not kb.ready or not cb.ready`                     | Abstain. `data_not_ready`                         |
| Model not yet fit                                  | Abstain. `model_warmup`                           |
| `model.stale_s() > 900` (last refit > 15 min ago)  | Abstain. `model_stale`                            |
| `model.r2_cv < 0.10`                               | Abstain. `model_low_r2` *(config says 0.05 — see §12)* |
| `fv is None` or `not fv.complete`                  | Abstain. `features_incomplete`                    |
| `q_pred is None or q_set is None`                  | Abstain. `model_predict_failed`                   |
| `\|q_predicted − yes_mid\| > 0.15`                 | Abstain. `model_disagrees_market`                 |
| `tau_s < FALLBACK_TAU_S` (60s)                     | Abstain (entry). `tau_too_small`                  |
| `yes_ask − yes_bid > 0.10`                         | Abstain. `wide_spread` *(config says 0.12 — see §12)* |
| `edge_magnitude < 0.02` (lowest tier floor)        | Abstain. `edge_below_floor`                       |
| Open position exists in current window             | Skip entry, run exit logic                        |
| Window rollover                                    | Drop position state (positions don't carry across windows) |

**Note:** `_tick()` checks `cb.ready` and `kb.ready` but does not gate on staleness within the tick. A book that received one update minutes ago and went silent will still report `ready=True`. The constants `COINBASE_STALE_MS_MAX` and `KALSHI_STALE_MS_MAX` exist in `config.py` but are unused in the decision path. See §12 for the full caveat.

These collectively express: *we only trade when the model has been validated on recent data, currently agrees on Kalshi's level, and the market itself looks tradeable.*

---

## 8. Phase 1 — data collection and dry-run analysis

### 8.1 Goals

1. Verify both feeds (Coinbase WS, Kalshi REST) run cleanly for many hours.
2. Log every input the regression and the exit simulator could possibly need: spot, microprice, lagged microprices, K, yes_bid/ask/mid, computed features, model predictions, fitted coefficients, hypothetical entry decisions, hypothetical exit triggers.
3. Validate the strategy thesis quantitatively via the regression's fit quality (`r2_cv`, `estimated_lag_s` stability) and the realized P&L from `scripts/backtest.py`.
4. Sweep entry/exit thresholds to find the best operating point.

### 8.2 What is NOT done in Phase 1

- No order placement. No authenticated Kalshi POST. No real money at risk via the bot.
- The single exception is `scripts/test_trade.py`, which places one $1 buy + one $1 sell to verify auth and fee accounting — run manually, never from the bot loop.
- No live tuning of the regression's structural parameters. The regression refits its coefficients automatically, but we don't change feature engineering during a run.
- No live exit decisions in the JSONL — exits are simulated at backtest time against the logged book.

### 8.3 Pre-flight checks

- Both feeds streaming for 5+ minutes:
  - `cb_book.last_update_ns` advancing on every Coinbase tick
  - `kb_book.last_update_ns` advancing on every Kalshi poll
- `floor_strike` populated (non-zero) on the discovered ticker.
- At least one full 15-min window has rolled over cleanly (look for the `INFO  window rollover -> ...` line).
- `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_FILE` (or `_PEM`) loaded from `.env` (verify with `python scripts/check_kalshi_balance.py`).
- `logs/` directory writable, ≥ 1 GB free on disk (CSV + JSONL grow ~50 MB/day).

### 8.4 During the run

Leave `python scripts/run_kalshi_bot.py` running in a terminal (or under `nohup` / `tmux` / a headless VM). One line per 10s scheduler tick prints to stdout, plus one `[Refit]` line every 5 minutes. Tail the JSONL with:

```bash
tail -f logs/decisions.jsonl | jq 'select(.event != "abstain")'
```

Resist tuning anything live. Fixed inputs over many hours is the entire point.

### 8.5 Post-run validation

Run `python scripts/analyze_run.py` after at least 1-4 hours of data. The script produces (a) charts of yes_mid vs `q_settled` / `q_predicted`, (b) edge magnitude over time, (c) `r2_cv` and `estimated_lag_s` over time, (d) hypothetical cumulative P&L from logged entry/exit pairs, and (e) a terminal summary. See `DRYRUN.md` for a full walkthrough.

Healthy thresholds:

| Metric                 | Healthy        | Bad           |
|------------------------|----------------|---------------|
| Median R²_cv           | > 0.25         | < 0.10        |
| Estimated lag          | 15-90s, stable | <5s or >150s  |
| % exits as `lag_closed`| > 50%          | < 30%         |
| Edge p90 magnitude     | > 0.03         | < 0.01        |
| `\|q_pred − yes_mid\|` | mean < 0.03    | mean > 0.10   |
| Simulated P&L          | positive       | negative      |

If **median R² < 0.10**: Kalshi is not predictable from Coinbase on today's data — the strategy thesis doesn't hold. Don't advance.

If **estimated lag < 5s**: Kalshi's market makers are already fast enough to eliminate the edge. Try again during higher-volatility periods.

---

## 9. Logging and storage

The kalshi branch uses **plain text formats** (CSV + JSONL), not Parquet. This is a deliberate simplification — at our data volume, Parquet's compression and columnar advantages don't matter, and CSV/JSONL is trivially inspectable from any shell.

### 9.1 `logs/decisions.jsonl`

One line per 10s scheduler tick, written by `Scheduler._log_decision()`. Schema (`DecisionRow`):

```
ts_ns               int       wall-clock receipt nanoseconds
tau_s               float     seconds until window close
yes_bid             float     Kalshi YES bid
yes_ask             float     Kalshi YES ask
yes_mid             float     (yes_bid + yes_ask) / 2
q_predicted         float?    sigmoid(model.predict(features))
q_settled           float?    sigmoid(model.predict(features.settled_array()))
edge_up_raw         float?    q_settled − yes_ask
edge_up_net         float?    edge_up_raw − fee − slippage
edge_magnitude      float     abs(best signed edge); -99 sentinel when not computed
favored_side        str?      "yes" / "no"
event               str       abstain / entry / hold / exit_lag_closed / exit_stopped / fallback_resolution
abstention_reason   str?      e.g. "model_warmup", "edge_below_floor"
tier                int       0 = abstain; 1-5 = Kelly tier index
would_bet_usd       float
has_open            bool
model_r2_cv         float
model_lag_s         float     β-weighted average lag at the active fit
window_ticker       str       e.g. "KXBTC15M-26MAY061515-15"
```

### 9.2 `logs/ticks.csv`

One row per 1 Hz sampler tick, written by `TickLogger.log()`. Schema:

```
ts_ns           int
tau_s           float
btc_microprice  float
btc_bid         float
btc_ask         float
yes_bid         float
yes_ask         float
yes_mid         float
floor_strike    float
window_ticker   str
```

This is the canonical input for `scripts/backtest.py` (which rebuilds features from these raw columns and replays a strategy variant) and for `Scheduler._bootstrap_from_history()` (which seeds the live training buffer from prior runs).

---

## 10. Phase 1 backtest — tuning the strategy

### 10.1 `scripts/backtest.py`

Walk-forward replay on `logs/ticks.csv`:

1. Load all ticks; group by `window_ticker` so lagged features don't bleed across boundaries.
2. Build the same 12 features used live (plus `kalshi_mom_30`, etc.).
3. Fit `RidgeCV` on the train fraction (default 65%); evaluate on the test fraction.
4. Replay the test fraction tick-by-tick with parameterized entry/exit thresholds.
5. Report per-trade P&L, cumulative P&L curve, R² on train/test.

```bash
python scripts/backtest.py                     # defaults
python scripts/backtest.py --entry 0.030 --exit 0.008 --hold-s 45
python scripts/backtest.py --sweep             # grid over (entry, exit, hold)
python scripts/backtest.py --save backtest.png
```

### 10.2 Parameter sweep

The sweep grid is defined in `backtest.py`. It iterates over `entry_threshold ∈ {0.020, 0.025, 0.030, ...}`, `exit_threshold` (the equivalent of `LAG_CLOSE_THRESHOLD`), and `hold_seconds` (effective `FALLBACK_TAU` for the simulated trade).

For each combination, replay the test fraction and report total P&L, ROI, win rate, mean hold time. Sort by total P&L; the top of that table is your candidate live configuration.

### 10.3 Decision criteria for advancing to Phase 2

Advance to Phase 2 (small live-money trial) if **all** of:

1. **Total P&L positive** in the best parameter combination from `--sweep`.
2. **Median CV-R² > 0.25** across all refits.
3. **`estimated_lag_s` stable** (sd across hourly means < 25 s) and within 15-120 s.
4. **`|q_predicted − yes_mid|` mean < 0.03** — the model tracks Kalshi's level reliably.
5. **At least 50% of exits are `lag_closed`** — the thesis actually works.
6. **Hit rate > 50% in the upper Kelly tiers** (tiers 1-3, edge ≥ 0.08).
7. **No more than 5% of windows had no `floor_strike`** (rollover gap issues).
8. **No persistent feed staleness warnings** in the log.

Do not advance if median R² < 0.10, lag-close exit rate < 30%, or P&L is negative on the best parameter combination.

---

## 11. Phase 2 — live trading on Kalshi (not yet active)

When the gates above pass, the live-trading path replaces the dry-run gating in `Scheduler._tick()` with real REST orders against Kalshi.

- **Order endpoint:** `POST /trade-api/v2/portfolio/orders` with the same RSA-PSS auth used by the REST feed.
- **Order shape:** `{ticker, action: "buy"|"sell", side: "yes"|"no", count, type: "limit", yes_price|no_price, time_in_force: "IOC"|"GTC", client_order_id}`.
- **Round-trip verification:** `scripts/test_trade.py` already does buy ($1 worth of YES IOC at top of book) → wait 10 s → sell. Use it to confirm the auth path, fee math, and fill semantics on day one of live trading.
- **Risk limits before going live:**
  - Wallet starts at ≤ $50 USDT.
  - Top Kelly tier is capped at 5% of wallet (tier 1's 10% is too aggressive for first contact).
  - Daily realized loss > 5% of starting wallet → hard stop.
  - One open position max across the bot.
  - All circuit breakers from §7 are still active.

Phase 2 is its own pull request. Do not edit `_tick()` to actually call the orders endpoint until `Phase 1 → Phase 2` decision criteria are met.

---

## 12. Algorithm soundness review

The user requested an honest assessment of whether the algorithm in this CLAUDE.md is sound. Here it is.

### 12.1 What is conceptually sound

- **Lag arbitrage as the edge.** The thesis is concrete and empirically grounded — the user's `lag_plot_btc.png` and the dry-run regression's `estimated_lag_s` both quantify it. If lag exists at 15-90 s and exceeds round-trip fees (~3.5%), the strategy makes money. If it doesn't, the bot abstains via `model_low_r2` and we don't lose money.
- **Regression-based forecasting is the right framing.** Predicting where Kalshi's *quote* will be in N seconds is a regression problem on lagged spot features. We don't need Black-Scholes; we don't need a volatility estimate; we don't need any forecasting of BTC itself. The model has one job: learn the lag profile from data.
- **Cash-out exit decouples P&L from resolution outcome.** We exit before the window closes, so whether BTC ends above or below `floor_strike` doesn't matter — we already collected the lag-close as profit.
- **`q_settled` via slot-substitution is a clean construction.** `q_predicted` (full lagged history) is the sanity check; `q_settled` (current spot in every slot) is the forecast. Both come from the same fitted coefficients, so there's no separate model to keep in sync.
- **One position per window** prevents stacking and racing yourself.
- **Self-falsifying via `r2_cv` and `model_disagrees_market` gates.** If the model is bad, the bot abstains. The strategy can't "fail silently" by losing money on bogus signals.
- **Adaptive refitting.** As Kalshi's market makers get faster, the lag distribution shifts; the 5-minute refit captures it.
- **Bootstrap from `logs/ticks.csv`** removes the warmup penalty for iterative development.
- **Training filter `yes_mid ∈ [0.05, 0.95]`** is a smart restriction — extreme tails carry no lag-arb signal and would dominate the loss function via huge logit values.

### 12.2 Concerns and things to watch

These don't break the algorithm but are worth knowing about.

#### Configuration / code drift

- **Threshold inconsistencies.** `config.py` defines `MODEL_MIN_CV_R2 = 0.05`, `MODEL_MAX_DISAGREEMENT = 0.20`, and `WIDE_SPREAD_THRESHOLD = 0.12`, but `scheduler.py` hardcodes `0.10`, `0.15`, and `0.10` respectively (lines 478, 492, 529). The hardcoded values win at runtime; the config values are dead. Fix is straightforward — wire the config constants in. Until then, the runtime behavior is stricter than the config implies, and there are two places to tune the same thresholds.

#### Cost model

- **`THETA_FEE = 0.07` is a guess.** Kalshi's actual taker fee schedule has rounding rules (`ceil(0.07 × C × P × (1−P))`) that the smooth `THETA × p × (1−p)` approximation papers over. At small bet sizes (a few contracts) the rounding can dominate the modeled fee. Calibrate against real fills from `scripts/test_trade.py` before trusting any backtest P&L number to within 50%.
- **Slippage model is crude** (capped at 2 cents, linear in size/depth). On thin Kalshi books it will *understate* real slippage when tier 1 (10% of wallet) tries to lift more depth than exists at the inside price. Phase 1 doesn't trade so this only contaminates the simulated edge; Phase 2 needs a proper VWAP-against-the-ladder estimator.

#### Sample rate

- **1 Hz Kalshi polling is plenty for 15-90 s lags but tight for shorter ones.** If Kalshi reprices in ≤ 1 s the polling will alias the signal. Acceptable today; flag if `estimated_lag_s` ever drops below ~5 s.
- **Coinbase ticker fires on every trade**, so the 1 Hz sampler effectively downsamples it. Microprice is averaged at 1 Hz into the ring buffer — fine for lag features at 15+ second horizons.

#### Strike interpretation

- **`floor_strike` (Kalshi) and the strike `K` (Polymarket) are different concepts.** On Polymarket's now-deprecated 5-min markets, `K` was the spot price at window-open (Chainlink oracle snapshot). On Kalshi `floor_strike` is a fixed dollar threshold per strike bin. The regression's `log(microprice / K)` feature on Kalshi is therefore a **moneyness** measure (how far above/below the strike are we, in log dollars), not a "spot has moved this much from open" measure. This works but means the feature has a different interpretation. Make sure this is reflected in any reasoning about why the regression's coefficients look the way they look.

#### Stop-loss math at small entry edges

- **The stop trigger is `edge_now < entry_edge − STOP_THRESHOLD`.** With `STOP_THRESHOLD = 0.03`, a tier-5 entry (entry_edge = 0.02) wants to stop when edge < -0.01. If `edge_signed` can't go that negative in the time before `FALLBACK_TAU_S`, the stop never fires and the position drifts to resolution. Not catastrophic, but tier-5 positions carry more "go to resolution" risk than the tier table implies. Consider a tier-aware stop (e.g. `STOP = max(0.02, entry_edge × 1.5)`).

#### Feed staleness gating

- **`_tick()` checks `cb.ready` and `kb.ready` but not staleness within the tick.** A book that received one update ten minutes ago and then went silent will still report `ready=True`. The display side (`_print_tick`) has staleness counters, but the decision logic doesn't gate on them. Add explicit `cb.stale_ms() > COINBASE_STALE_MS_MAX` and `kb.stale_ms() > KALSHI_STALE_MS_MAX` gates (the constants are already in `config.py`, just unused in the decision path).

#### Adverse selection at the lag close

- **When the lag closes, every other lag-arb bot is also exiting.** Real Phase 2 fills will be worse than the bid we model — the simulator assumes we hit the bid we see, but in practice the bid moves before our IOC arrives. The dry-run simulator can't capture this; only Phase 2 small-money trials can. Plan for actual P&L to come in 25-50% below backtest expectations.

#### "Lag exists" empirically vs persistently

- The user's `lag_plot_btc.png` shows lag exists *now*. The strategy assumes it persists at 15-90 s for many windows. If it doesn't (e.g. during a quiet weekend), the bot abstains rather than loses money — but it also doesn't make money. Plan for variable engagement rates.

### 12.3 Verdict

The algorithm is **conceptually sound and implementation-ready for Phase 1**. The main risks are *empirical* (does the lag persist? is it big enough net of real fees?) rather than *structural* (is the math right?). The dry run plus `scripts/backtest.py` is the correct vehicle for answering those empirical questions before any real money is committed.

Pre-Phase-2 fixes (in priority order):

1. Calibrate `THETA_FEE` from a real round-trip (`scripts/test_trade.py`). Currently 0.07 is borrowed from Kalshi's published 2025 schedule and **not verified against actual fills**. Could be off by enough to flip backtest P&L sign.
2. Wire `MODEL_MIN_CV_R2`, `MODEL_MAX_DISAGREEMENT`, and `WIDE_SPREAD_THRESHOLD` from config into `scheduler.py`. Currently hardcoded at stricter values; one place to tune is better than two.
3. Add explicit `stale_ms()` gates in `_tick()`. Constants already exist in `config.py`; just need to be referenced.
4. Replace the slippage model with a true VWAP-against-the-ladder estimator using Kalshi's full order-book endpoint. The current `min(0.02, size/depth × 0.01)` model will understate slippage on thin books at tier-1 sizing.
5. Make the stop threshold tier-aware so tier-5 entries actually have a meaningful stop. Currently with `STOP = 0.03` and tier-5 `entry_edge = 0.02`, the stop fires only at `edge < −0.01`, which may never be reached before fallback resolution.

---

## 13. Caveats and open questions

- **Kalshi WebSocket is intentionally unused.** Early experiments showed silent freezes on the Kalshi WS feed, and the REST polling at 1 Hz proved more reliable. If Kalshi improves their WS path, switching could give us sub-second Kalshi observation, which would tighten the model — but there's no urgency.
- **`pyproject.toml` still says `name = "polybot"` and lists CLI scripts that no longer exist.** Vestigial; will be cleaned up. `requirements.txt` similarly carries `py-clob-client` which is unused. Don't trust either as a source of truth — `betbot/kalshi/` is.
- **`betbot/main.py` is broken** (imports from a non-existent `polybot.*` package). Do not try to run it. The actual entrypoint is `scripts/run_kalshi_bot.py`.
- **No `cli/polybot_metrics` or `polybot-ctl` equivalents on the kalshi branch.** Operations is just stdout + JSONL inspection. SystemD / journald wiring described in older docs is from the Polymarket era; if needed, the dry run is small enough to run under `tmux` or `nohup` and tail the JSONL.
- **One-asset only.** ETH support is removed; the original CLAUDE.md's `assets: [btc, eth]` plumbing is gone. Restoring it requires duplicating `KALSHI_SERIES = "KXBTC15M"` to `KXETH15M`, running two `KalshiBook` instances, and adding `cross_asset_momentum_60s` features that look at the other asset. Out of scope for this branch.
- **The `RUNBOOK.md` and `DRYRUN.md` documents partially predate the Kalshi rewrite.** `DRYRUN.md` is reasonably current. `RUNBOOK.md` still references Binance and Polymarket env vars; treat its first sections as stale and the file-reference at the end as broadly correct.
- **24h sample is small.** ~96 windows × N strikes × ~150 ticks/window is overfitting territory for parameter selection. Mitigations: prefer parameters in stable plateaus rather than sharp peaks of the P&L surface; treat any Phase 2 first week as the real out-of-sample test; extend the dry run to 7 days if 24h results are ambiguous.

---

## 14. Prior art and references

- **Stoikov 2018 "The Micro-Price"** — foundational paper on microprice as a forward predictor of midpoint. Required reading for understanding why microprice is the right input to the regression.
- **Cont-Kukanov-Stoikov 2014** — Order Flow Imbalance. Currently unused but the natural next feature if `r2_cv` proves underwhelming.
- **Hayashi-Yoshida 2005** — non-synchronous covariance estimator for tick data from two venues. Useful for offline lag analysis as a sanity check on what the regression's β coefficients are saying.
- **Kalshi API docs** — `https://trading-api.readme.io/reference/getmarket` etc. The auth scheme (RSA-PSS over `timestamp + method + path`) is implemented in `betbot/kalshi/auth.py`.
- **Coinbase Advanced Trade WebSocket docs** — the `ticker` channel is unauthenticated and exposes `best_bid`, `best_ask`, `best_bid_quantity`, `best_ask_quantity`, which is everything we need for microprice.

The lag-arbitrage strategy itself isn't novel (every prediction-market shop has tried it), but the specific implementation — ridge regression on lagged log-microprice features with adaptive refit and cash-out exit — is uncommon in the published Kalshi/Polymarket bots and is the central contribution of this codebase.
