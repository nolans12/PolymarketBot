# CLAUDE.md — Kalshi 15-Minute BTC Lag-Arbitrage Bot

> ## 🛑 MANDATORY WORKFLOW — read before doing ANYTHING in this repo
>
> This document specifies five pending work items (see **§15**) that change the structure of the codebase. They are non-trivial, touch many files, and will produce a broken intermediate state if executed without discipline.
>
> **Before starting any work described in this document, you MUST:**
>
> 1. **Invoke `/superpowers`.** Every refactor, deletion, and feature add in §14, §5.6, §6.6, §11.1, and §15 is to be executed under the `/superpowers` skill. The skill enforces a plan-first / verify-after workflow that is required for changes of this scope.
> 2. **Invoke `/gsd`.** This document is long (~1400+ lines) and references many files across `betbot/kalshi/`, `scripts/`, `pyproject.toml`, `.env.example`, and the dead legacy directories. `/gsd` keeps the relevant context loaded efficiently and prevents you from re-fetching the same files.
>
> **After all five work items in §15 are complete and merged, you MUST also:**
>
> 3. **Invoke `/ultra-review`.** This is the closing-step verification pass. It is non-optional. The work is not "done" — and §5.6 / §6.6 / §11.1 / §15 must NOT have their `PENDING` markers removed — until `/ultra-review` has run and passed. See **§15.9** for what the review covers and what counts as passing.
>
> **Do not skip any of these three invocations.** If you (the AI assistant or human maintainer) start editing files without first invoking `/superpowers` and `/gsd`, stop and restart the session under the correct workflow. If you mark any work item complete without running `/ultra-review`, undo that and run the review first. There is no acceptable reason to bypass any of this — the work items are exactly the kind of multi-file, multi-step changes these skills exist for.
>
> **First lines of your first response in the working session should be:**
>
> ```
> Invoking /superpowers and /gsd before starting work on CLAUDE.md §15.
> Will invoke /ultra-review at the end before marking any work item done.
> ```
>
> If you are reading this and have not invoked `/superpowers` and `/gsd`: stop here. Go invoke them. Come back. If you have completed the work but have not yet run `/ultra-review`: stop. Run it. Do not declare the work done until it has passed.

---

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
- **Coinbase Advanced Trade** is the default spot feed. The WebSocket ticker channel exposes top-of-book bid/ask plus quantities, which is enough to compute microprice (the regression's primary spot input). No VPN required.

**Spot feed is pluggable.** The bot is being refactored (see §5.6 and §15 WI-3) so that **Coinbase or Binance can serve as the spot feed interchangeably**, exposing the exact same features to the model. Pick whichever has lower latency in your region. Coinbase is the default and works from US IPs; Binance is faster in many other regions but is geo-blocked from the US (or requires a VPN). Both feed the same `SpotBook` and produce identical feature vectors, so the regression doesn't know or care which one is upstream — swapping is a config-flag flip with no model retraining required.

This branch (`kalshi`) is a fresh implementation built from scratch in `betbot/kalshi/`. The legacy `betbot/clients/`, `betbot/models/`, `betbot/state/`, `betbot/strategy/`, `betbot/infra/`, `betbot/research/`, `betbot/execution/`, `betbot/cli/`, and `betbot/main.py` are **dead code** — they import from a non-existent `polybot.*` package and date back to the Polymarket era. They are scheduled for deletion (see §14 / §15 WI-1).

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

Spot momentum (source-agnostic — same definitions whether the spot feed is Coinbase or Binance):
  spot_momentum_30s = log(microprice_now / microprice_{t-30s})
  spot_momentum_60s = log(microprice_now / microprice_{t-60s})

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

**Maker fees are zero on Kalshi.** The above formula applies to taker fills only. Switching the entry leg from taker to maker (limit order at the existing best bid, wait for someone to take it) eliminates the entry fee and captures rather than pays the bid-ask spread — roughly halving total round-trip cost. This is non-trivial relative to the strategy's net edge and is specified as a separate work item: see **§11.1** for the full design and **§15 WI-5** for the execution checklist.

### 5.5 Slippage

Modeled crudely in `_slippage()`:

```python
def _slippage(book_depth: float, size_usd: float) -> float:
    if book_depth <= 0:
        return 0.02
    return min(0.02, size_usd / book_depth * 0.01)   # capped at 2 cents
```

This is **not** a VWAP-against-the-ladder model. It assumes a smooth penalty proportional to `size / depth`, capped at 2 cents per share. Phase 1 doesn't trade so this only affects the simulated edge calculation; in Phase 2 it should be replaced with a true depth-walking estimator.

### 5.6 Pluggable spot feed — Coinbase or Binance, same features (PENDING — see §15 WI-3)

**Status:** specified, not yet implemented. Tracked as Work Item 3 in §15.

**Goal.** Make the spot-feed source a configuration choice, not a code rewrite. Today, `betbot/kalshi/coinbase_feed.py` writes into `betbot/kalshi/book.py:CoinbaseBook`, which is named after Coinbase but is otherwise generic (microprice, ring buffer, staleness). After this refactor, the same infrastructure will accept either Coinbase or Binance as the upstream, with the regression seeing identical features either way. The user picks whichever has lower latency in their region; the model doesn't know or care which.

**The principle:** features are spot-source-agnostic. There is one `SpotBook`, one feature schema, one regression. The only thing that changes between Coinbase mode and Binance mode is which WebSocket is feeding the book.

**Concrete refactor:**

1. **Rename `CoinbaseBook` → `SpotBook`** in `betbot/kalshi/book.py`. The class is already generic (it consumes top-of-book ticks and produces microprice + ring buffer); the name is the only Coinbase-specific thing about it. Update the only call site (`scripts/run_kalshi_bot.py`) and the variable name `cb_book` → `spot_book` everywhere it appears.

2. **Define a `SpotFeed` abstract interface** in a new file `betbot/kalshi/spot_feed.py`:
   ```python
   class SpotFeed(Protocol):
       async def run(self) -> None: ...
       def stop(self) -> None: ...
   ```
   Both feeds expose this minimal contract; both push into a shared `SpotBook` via `SpotBook.apply_ticker(price, bid, ask, bid_size, ask_size)`.

3. **Keep `CoinbaseFeed`** in `betbot/kalshi/coinbase_feed.py` as the Coinbase implementation. Update its constructor signature to take a `SpotBook` (renamed) instead of `CoinbaseBook`.

4. **Add `BinanceFeed`** in a new file `betbot/kalshi/binance_feed.py`. Subscribes to Binance's `bookTicker` stream (`wss://stream.binance.com:9443/ws/btcusdt@bookTicker`), which exposes `b` (best bid), `a` (best ask), `B` (bid qty), `A` (ask qty) on every top-of-book change. Maps these into `SpotBook.apply_ticker()` with the same semantics as the Coinbase feed. Document that Binance is geo-blocked from US IPs and requires a VPN if running from the US.

5. **Rename feature names in `features.py`** to drop the Coinbase prefix:
   - `cb_momentum_30s` → `spot_momentum_30s`
   - `cb_momentum_60s` → `spot_momentum_60s`
   - The `x_0`..`x_120` log-ratios are already source-agnostic (no rename needed).
   - `kalshi_spread` and `kalshi_momentum_30s` stay as-is (they're Kalshi-specific, not spot-specific).
6. **Add `SPOT_SOURCE` to `betbot/kalshi/config.py`:**
   ```python
   SPOT_SOURCE = os.getenv("SPOT_SOURCE", "coinbase").lower()
   assert SPOT_SOURCE in ("coinbase", "binance"), \
       f"SPOT_SOURCE must be 'coinbase' or 'binance', got {SPOT_SOURCE}"
   ```
   Also add the Binance WebSocket URL constant alongside the existing Coinbase one.

7. **Wire the feed selection in `scripts/run_kalshi_bot.py`:**
   ```python
   from betbot.kalshi.config import SPOT_SOURCE
   from betbot.kalshi.coinbase_feed import CoinbaseFeed
   from betbot.kalshi.binance_feed import BinanceFeed
   from betbot.kalshi.book import SpotBook

   spot_book = SpotBook()
   if SPOT_SOURCE == "coinbase":
       spot_feed = CoinbaseFeed(spot_book)
   elif SPOT_SOURCE == "binance":
       spot_feed = BinanceFeed(spot_book)
   ```
   Print which source is in use at startup so the operator sees it in the log header.

8. **Update `.env.example`** to document the new flag with both options. Default to `coinbase` (US-friendly).

9. **Update `scripts/backtest.py`** to use the renamed feature columns (`cb_mom_30` → `spot_mom_30` etc.) and rebuild features the same way regardless of which feed produced the historical ticks.

10. **Tick CSV stays compatible.** `logs/ticks.csv` doesn't carry feature names, only raw `btc_microprice`, `btc_bid`, `btc_ask` etc. — the same column names work for both feeds. **No data migration needed.**

**Acceptance criteria for WI-3:**

- `python scripts/run_kalshi_bot.py --fresh` runs cleanly with `SPOT_SOURCE=coinbase` (default).
- `SPOT_SOURCE=binance python scripts/run_kalshi_bot.py --fresh` runs cleanly when the host has a route to Binance (or fails fast with a clear error if geo-blocked).
- `grep -r "CoinbaseBook\|cb_book\|cb_momentum" --include="*.py" betbot/` returns zero matches.
- Feature schema in a fresh `[Refit]` log line shows `spot_momentum_30s`, `spot_momentum_60s` (no `cb_` prefix).
- Regression refits identically on Coinbase-sourced and Binance-sourced data when run on similar time windows.

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

### 6.6 Forward-projection features — incorporating Coinbase's expected drift (PENDING — see §15 WI-4)

**Status:** specified, not yet implemented. This section is the design spec for the next planned feature-engineering change.

**Motivation.** The current `q_settled` answers: *"where will Kalshi be once it has fully caught up to current spot?"* It assumes spot stops moving the moment we make the prediction. In reality, when spot has been trending up over the last 30-60 seconds, market makers on Kalshi typically price in some of the expected continuation — even if there were zero lag, Kalshi would already be quoted higher than the spot-implied probability. The current model partially captures this through the `spot_momentum_30s` and `spot_momentum_60s` coefficients, but only as an additive momentum term; it does not project spot forward and feed that projection through the full lag-coefficient stack.

The intent of this addition: make `q_settled` answer the richer question *"where will Kalshi be once it has caught up to spot, AND spot has continued to drift at its current pace for ~60 seconds"*.

**The new features.** Add three forward-projected log-K ratios to `FEATURE_NAMES` in `betbot/kalshi/features.py`, mirroring the existing backward-looking lag features:

```
x_proj_30  = log(projected_microprice_at_t+30s  / K)  ≈  x_0 + spot_momentum_30s
x_proj_60  = log(projected_microprice_at_t+60s  / K)  ≈  x_0 + spot_momentum_60s
x_proj_120 = log(projected_microprice_at_t+120s / K)  ≈  x_0 + 2 × spot_momentum_60s
```

The projection is a simple log-linear extrapolation: assume the recent 30s/60s drift continues at the same rate. This is the same shape used by Polymarket-era `betbot/models/features.py` for its multi-asset projection (the dead reference file the user pointed at). Because the projection uses `spot_momentum_*` rather than venue-specific names, it works identically whether the spot feed is Coinbase or Binance — see §5.6 for the spot-feed abstraction.

**Updates required in `features.py`:**

1. Add `x_proj_30`, `x_proj_60`, `x_proj_120` to `FEATURE_NAMES` (at the end, so existing positional indices don't shift).
2. Add the three matching fields to the `FeatureVec` dataclass.
3. In `build_features()`, compute them after `spot_momentum_30s` and `spot_momentum_60s` are already available:
   ```python
   x_proj_30  = x_0 + spot_momentum_30s
   x_proj_60  = x_0 + spot_momentum_60s
   x_proj_120 = x_0 + 2.0 * spot_momentum_60s
   ```
4. Add them to `as_array()` in the same order as `FEATURE_NAMES`.
5. **Critical:** in `settled_array()`, **do NOT replace the projection slots with `x_0`.** Only the lag slots (`x_15`..`x_120`) get overwritten. The projection slots stay at their live values, because their whole purpose is to encode the forward-drift contribution that `q_settled` should preserve.

**Updates required elsewhere:**

- `betbot/kalshi/model.py`: no functional changes needed — `RidgeCV` adapts to the new feature dimensionality automatically. The β-weighted `estimated_lag_seconds` calculation uses `["x_15", "x_30", "x_60", "x_90", "x_120"]` and should be left alone (projection features are forward-looking, not lags).
- `betbot/kalshi/scheduler.py`: in `_bootstrap_from_history()`, append the three new projection values after the existing momentum computation, in the same order as the live `FeatureVec.as_array()`.
- `scripts/backtest.py`: in `build_features()`, after computing `cb_mom_30` and `cb_mom_60`, add `g["x_proj_30"] = g["x_0"] + g["cb_mom_30"]`, etc. Add the new column names wherever the model is fit.
- Old `logs/ticks.csv` files remain valid (the raw schema doesn't change), but a fresh fit is needed because the regression has new features.

**What the regression learns.** At training time, the projection features are linear combinations of features the model already has, so in pure fitting terms they're redundant — the regression could already express the same combination via separate β coefficients on `x_0` and `spot_momentum_30s`. The added value is **at extrapolation time** (`q_settled`), where the substitution rule treats lag slots and projection slots differently:

- Lag slots: substitute `x_0` (assumes Kalshi has caught up to current Coinbase).
- Projection slots: keep live value (assumes Coinbase keeps drifting at its recent rate).

The combined `q_settled` then represents *"where Kalshi lands once both effects play out — lag closes AND Coinbase continues drifting."* If Kalshi's market makers price in forward expectation at all, the projection coefficients will be non-zero and `q_settled` will shift further in the trend direction than the lag-only version would. If they don't, the coefficients stay near zero and the model degenerates back to the pure lag-arb formulation — no harm done.

**Caveat — small forecasting assumption.** This addition introduces a modest forecasting element (linear extrapolation of recent drift) that the §1.3 stance against forecasting deliberately avoided. The justification is that the horizon is short (≤120 s), the extrapolation is linear and parameter-free, and the regression is free to learn that the projection features carry zero weight if the assumption doesn't hold. It's still a step away from "pure lag arbitrage with no forecasting" — flag it in any post-mortem if a Phase 2 loss looks attributable to a momentum reversal that the projection over-extrapolated.

**Validation plan.** After implementation:

1. Refit on a recent 4-h slice and inspect the projection-feature β coefficients in `[Refit]` log lines. Non-zero coefficients = market makers do price in some forward drift.
2. Compare `q_settled` distribution before/after on a held-out window. The new `q_settled` should show modestly more variance and a slight shift in the direction of recent momentum.
3. Run `scripts/backtest.py --sweep` before and after to compare realized P&L. If P&L improves on out-of-sample data, keep. If it degrades or is flat, the forward-drift hypothesis isn't supported and the features can be removed.

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

### 11.1 Maker-entry execution model (PENDING — see §15 WI-5)

**Status:** specified, not yet implemented. Tracked as Work Item 5 in §15.

**Motivation.** The default execution model — taker on both legs — gives away a lot of edge. Kalshi's taker fee is `ceil(0.07 × C × P × (1−P))` per fill; makers pay zero. Combined with the bid-ask spread on each leg, round-trip costs are approximately **3.5% of position notional** before slippage. On a strategy whose net edge barely clears the 2% Kelly tier floor, that's most of the available P&L going to fees and spread. Switching the entry leg from taker to maker recovers the entry fee (≈1.75% at p=0.5) and the half-spread (typically 1-1.5¢ on Kalshi), reducing round-trip cost to roughly 1.75%. If entry fill rate is high enough, this roughly **doubles** realized P&L per trade.

**Worked cost example** (same numbers as the §6.6 example, both modes side-by-side):

```
yes_ask = 0.64, yes_bid = 0.62, q_settled = 0.79
Lag closes; market moves to yes_ask 0.71, yes_bid 0.69.

TAKER both sides (current behavior):
  entry: cross to 0.64, fee = 0.07 × 0.64 × 0.36 ≈ 0.016
  exit:  hit bid 0.69,  fee ≈ 0.015
  net P&L per $1 = (0.69 − 0.64) − 0.031 = 0.019    (1.9%)

MAKER entry, TAKER exit (this work item):
  entry: post 0.62, fill 0.62, fee = 0
  exit:  hit 0.69,             fee ≈ 0.015
  net P&L per $1 = 0.07 − 0.015 = 0.055              (5.5%)
```

**The catch — fill probability.** Limit orders only fill when someone takes them. The lag-arb thesis is "edge exists for 15–90 seconds" — if our limit doesn't fill in that window, we miss the trade entirely. Worse, when our limit DOES fill, the counterparty is sometimes the *informed* side: a faster lag-arb bot dumping its now-stale inventory at our limit price (adverse selection). The cost case above is the upper bound; real P&L = (cost savings on fills) − (opportunity cost of misses) − (adverse selection on filled trades). The unknown is the fill rate, which can't be answered from current `logs/decisions.jsonl` data alone.

**The hybrid: maker entry, taker exit.** This is the recommended starting point. Rationale:

- **Entry tolerates fill uncertainty.** The signal window is 15-90 s; if a 30 s limit doesn't fill, we cancel and re-evaluate at the next decision tick. Missing an entry is OK — we just abstain.
- **Exit does NOT tolerate fill uncertainty.** When `edge_now < LAG_CLOSE_THRESHOLD`, every second we don't exit is risk we carry for free. A limit-exit that fails to fill could turn a profitable trade into a loss when the next adverse move arrives. The 1.75% taker fee on exit is acceptable insurance.
- **Stop-loss as taker, always.** Same reasoning, more so. When we want out, get out.

This captures most of the maker savings without the tail risk of a stuck position.

**Two-stage implementation plan.** Don't write the full execution machinery before validating the assumption. Do it in two stages:

**Stage A — Phase-1 simulated pilot** (cheap; pure dry-run instrumentation):

1. In `Scheduler._tick()`, when an `entry` event would fire, also log a "would-have-posted-limit-at" record: `{ts, side, post_price, intended_size, signal_at_post}`.
2. In a follow-up walk over the next N seconds of `decisions.jsonl` (or `ticks.csv`), check whether the post side ever traded at-or-better than `post_price`. Mark the simulated order as `filled` or `unfilled`.
3. After 1-7 days of data, compute fill rate by post-price strategy:
   - Passive (post at existing best bid): `P(fill | post = yes_bid_t, within 30s)`
   - Aggressive (one tick inside the spread): `P(fill | post = yes_bid_t + tick, within 30s)`
4. Compute simulated P&L using `(filled_trades × maker_savings) − (missed_trades × taker_baseline_pnl)`. Compare to current taker-only baseline.

**Stage A is gating.** Proceed to Stage B only if simulated maker P&L beats taker P&L by ≥ 25% on out-of-sample data. If fill rates are <40% and simulated maker P&L is worse than taker, the strategy is at its execution ceiling — don't build the machinery.

**Stage B — full execution machinery** (real orders; Phase 2):

1. **`OrderManager` class** in `betbot/kalshi/orders.py`:
   - `place_limit(side, price, size, ttl_s) -> client_order_id`
   - `cancel(client_order_id)` and `cancel_all()`
   - Tracks open-orders state; reconciles against `GET /trade-api/v2/portfolio/orders` every decision tick.
2. **Fill detection.** Poll order status (or subscribe to fill events if Kalshi exposes them) and call back into `Scheduler` when our entry limit fills, transitioning state from "limit-posted" to "open-position".
3. **Cancel-on-stale logic.** If 30 s elapse without a fill, cancel and re-evaluate the entry signal. If the signal still says go, re-post (possibly at an updated price). If not, abstain.
4. **Edge-calc update in `_tick()`.** When maker entry is enabled, `fee_up = 0.0` (not `THETA × p × (1-p)`); `fee_no = 0.0`. Exit fee unchanged. The Kelly tier floor effectively rises (more edge captured per dollar) but the floor itself stays at 0.02.
5. **State machine.** `_pos` becomes `Optional[Position | PendingOrder]`. New states: `pending_entry` (limit posted, waiting for fill), `open_position` (filled, watching for exit), `pending_exit` (only used for stop-loss order in flight). Window rollover cancels any pending orders before clearing state.
6. **Backtest infrastructure.** Add a `--maker-entry` flag to `scripts/backtest.py` that simulates limit fills against logged top-of-book + trade-flow data. Same fill model as Stage A's pilot, applied to historical replays.

**What does NOT change:**

- Feature engineering (§3.3, §6.6).
- The regression model itself.
- `q_settled` computation.
- Sizing / Kelly tier table.
- Exit logic (still taker on lag-closed, taker on stopped, hold-to-resolution at small τ).

**Configuration.** Add to `betbot/kalshi/config.py`:

```python
ENTRY_MODE     = os.getenv("ENTRY_MODE", "taker")    # "taker" | "maker"
MAKER_POST_TTL_S        = 30        # cancel and re-evaluate after this
MAKER_POST_OFFSET_TICKS = 0         # 0 = at best bid (passive), 1 = one tick inside
```

Default to `taker` so existing behavior is preserved until Stage B is validated. Flip to `maker` only after Stage A passes the gate.

**Caveat — Phase 2 only.** Stage B writes real orders to Kalshi. It must not be merged or default-enabled until the live `_tick()` order-placement path is otherwise approved (the Phase 2 decision criteria in §10.3 must pass first). Stage A is safe to merge in Phase 1 because it only logs.

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
- **Polymarket-era dead code lingers in the repo.** `pyproject.toml` says `name = "polybot"` and lists CLI scripts that don't exist. `requirements.txt` carries `py-clob-client`. `betbot/main.py` is broken. Several `betbot/` subdirectories (`clients/`, `models/`, `state/`, etc.) import from a non-existent `polybot.*` package. `RUNBOOK.md` first half references Binance and Polymarket env vars. `scripts/visualize_market.py` carries a Binance code path. **See §14 for step-by-step removal instructions.** None of this affects the live bot but it confuses anyone reading the repo.
- **One-asset only.** ETH support is removed; the original CLAUDE.md's `assets: [btc, eth]` plumbing is gone. Restoring it requires duplicating `KALSHI_SERIES = "KXBTC15M"` to `KXETH15M`, running two `KalshiBook` instances, and adding `cross_asset_momentum_60s` features that look at the other asset. Out of scope for this branch.
- **24h sample is small.** ~96 windows × N strikes × ~150 ticks/window is overfitting territory for parameter selection. Mitigations: prefer parameters in stable plateaus rather than sharp peaks of the P&L surface; treat any Phase 2 first week as the real out-of-sample test; extend the dry run to 7 days if 24h results are ambiguous.
- **Forward-projection feature is specified but not implemented.** See §6.6 for the design and the validation plan. Treat any backtest results that pre-date the implementation as a baseline; re-run the sweep after the projection features land to see whether they help.

---

## 14. Codebase cleanup — removing Polymarket-era dead code

The repository carries a substantial amount of dead code from the Polymarket + Binance era that pre-dates the Kalshi rewrite. None of it is imported by the live bot (`scripts/run_kalshi_bot.py` → `betbot/kalshi/*`), but it confuses anyone reading the repo, breaks `grep`-based code search, and prevents `python -m betbot` from running cleanly. This section is the step-by-step plan for removing it. Do not skip steps — work top to bottom.

### 14.1 Pre-flight

Before deleting anything, confirm the live bot still runs end-to-end:

```bash
python scripts/check_kalshi_balance.py     # auth works
python scripts/run_kalshi_bot.py --fresh   # bot starts, prints abstain lines, refits after ~6 min
```

Stop the bot with Ctrl-C. If both commands succeed, the live path doesn't depend on anything in the dead code, and deletion is safe.

### 14.2 Delete dead directories under `betbot/`

These directories all contain Polymarket-era modules whose imports reference a non-existent `polybot.*` package. None of them are imported by `betbot/kalshi/*` or by any script under `scripts/`. Delete the whole directory in each case.

```
betbot/clients/        # binance_ws (legacy), polymarket_*, old coinbase_ws
betbot/models/         # old features.py (worth keeping a copy outside the repo as
                       #   a reference for the projection-feature pattern in §6.6,
                       #   then delete)
betbot/state/          # spot_book, poly_book, coinbase_book, window, wallet
betbot/strategy/       # decision, position
betbot/execution/      # orders, fills
betbot/infra/          # config, parquet_writer, refitter, scheduler
betbot/research/       # cashout_simulator, parameter_sweep
betbot/cli/            # polybot_ctl, polybot_metrics
```

**Important:** `betbot/clients/binance_ws.py` is the **legacy** Polymarket-era Binance L2 client — broken imports, dead. Delete it. The new `BinanceFeed` introduced in §5.6 / §15 WI-3 is a separate, clean implementation that lives at `betbot/kalshi/binance_feed.py` and uses the simpler `bookTicker` stream rather than full L2. Don't try to salvage `clients/binance_ws.py` — write the new one fresh against the `SpotFeed` interface.

`betbot/tests/` is currently empty; leave it (or delete it — either is fine).

### 14.3 Delete dead files

```
betbot/main.py                 # polybot.* imports, would crash on import
scripts/demo_binance.py        # polybot.* imports
scripts/demo_coinbase.py       # polybot.* imports — note: this is NOT betbot/kalshi/coinbase_feed.py, which IS used and stays
```

Sanity check before each delete: `grep -r "from <module>" --include="*.py"` should return zero matches outside the file being deleted.

### 14.4 Edit `betbot/__init__.py`

Replace the existing two-line docstring:

```python
"""betbot — Kalshi 15-minute BTC lag-arbitrage bot.

See CLAUDE.md for the full design specification.
"""

__version__ = "0.3.0"
```

(Bump the version to mark the cleanup boundary.)

### 14.5 Rewrite `pyproject.toml`

The current `pyproject.toml` lists name `polybot`, a Polymarket-era description, three CLI entry points that don't exist (`polybot`, `polybot-metrics`, `polybot-ctl`), and Polymarket-only deps (`py-clob-client`, `web3`). Replace with:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "betbot"
version = "0.3.0"
description = "Kalshi 15-minute BTC lag-arbitrage bot"
requires-python = ">=3.11"
readme = "README.md"

dependencies = [
  "aiohttp>=3.9.0",
  "websockets>=12.0",
  "cryptography>=42.0.0",
  "python-dotenv>=1.0.0",
  "numpy>=1.24.0",
  "pandas>=2.1.0",
  "scikit-learn>=1.3.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "matplotlib>=3.8.0",
]

[tool.setuptools.packages.find]
include = ["betbot*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["betbot/tests"]
```

Notes: dropped `py-clob-client`, `web3`, `pyarrow`, `duckdb`, `requests`, `structlog`. Added `aiohttp` (used by Kalshi REST feed). Moved `matplotlib` to optional dev deps since only the analysis scripts need it. `requests` is still imported by `scripts/check_kalshi_balance.py` — either replace those imports with `aiohttp`/`urllib` calls in the same edit, or keep `requests` in the main deps list (a single legacy import is acceptable).

### 14.6 Trim `requirements.txt`

Match `pyproject.toml`. Drop `py-clob-client`, `pyarrow`, `duckdb`, `structlog`. Add `aiohttp`. Final file:

```
aiohttp>=3.9.0
websockets>=12.0
cryptography>=42.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
pandas>=2.1.0
scikit-learn>=1.3.0
```

Optional: regenerate via `pip-compile pyproject.toml -o requirements.txt` if pip-tools is available.

### 14.7 Strip Polymarket from `.env.example`

The current `.env.example` carries commented-out `PRIVATE_KEY`, `API_KEY`, `API_SECRET`, `API_PASSPHRASE`, plus required `CLOB_HOST` and `CHAIN_ID` lines (none of which are used by the live Kalshi bot). Trim to just:

```
# Kalshi API credentials
KALSHI_API_KEY_ID=YOUR_KALSHI_KEY_ID
KALSHI_PRIVATE_KEY_FILE=~/.kalshi/kalshi_rsa.pem
# Or inline (use \n escapes for newlines):
# KALSHI_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----\n..."

# Operational
DRY_RUN=true
LOG_DIR=logs
```

Delete `CLOB_HOST`, `CHAIN_ID`, `FALLBACK_BALANCE`, the entire `POLYMARKET — DEPRECATED` block, and the optional Binance/Polymarket endpoint overrides.

### 14.8 Replace or delete `RUNBOOK.md`

The first half of `RUNBOOK.md` references Binance, Polymarket, the `polybot` CLI, and the `K_SOURCE` setting — all dead. The simplest fix is to delete the file entirely and let `DRYRUN.md` be the operations doc. If the file-reference table at the end is worth keeping, lift it into `DRYRUN.md` as a new section and then delete `RUNBOOK.md`.

### 14.9 Adapt `scripts/visualize_market.py` to the new spot-feed abstraction

This script already supports both Coinbase and Binance via a `--spot {binance,coinbase}` flag. **Keep both code paths** — they align with the spot-feed abstraction in §5.6 / §15 WI-3. Update only:

- Remove any inline copy of Binance auth, JWT generation, or other complexity that doesn't belong in a visualization script.
- Replace any direct WebSocket subscription code with imports from the new `betbot/kalshi/coinbase_feed.py` and `betbot/kalshi/binance_feed.py` modules (after WI-3 lands), so the script and the live bot share one implementation per feed.
- Have `--spot` default to whatever `SPOT_SOURCE` is set to in `.env`, with the CLI flag as override.

Do **not** delete the Binance code path. Binance is a first-class supported spot feed under the new abstraction.

### 14.10 Remove residual Kalshi WebSocket naming and references

The Kalshi WebSocket feed was already removed in favor of REST polling (see `betbot/kalshi/kalshi_rest_feed.py`'s docstring), but a few naming and documentation artifacts still reference WS. Clean them up:

1. **`betbot/kalshi/scheduler.py`** — the `Scheduler` class has `self._ws = kalshi_feed` (line ~196) where `kalshi_feed` is actually a `KalshiRestFeed` instance. The `_ws` name is misleading. Rename throughout the class:
   ```
   self._ws  →  self._kalshi_feed
   ```
   Also update the constructor parameter name from `kalshi_feed` to be consistent if needed, and update every usage site (`self._ws.update_ticker(...)` etc.).

2. **`betbot/kalshi/auth.py`** — the docstring says *"There is no WebSocket auth in this codebase; the Kalshi WS path is no longer used"*. Trim this to just describe what the file does (REST request signing) without the historical note. Future maintainers don't need to know about a path that's already gone.

3. **`betbot/kalshi/kalshi_rest_feed.py`** — the docstring opens with *"Drop-in replacement for the WebSocket feed."* Reframe: REST polling is the canonical feed, not a "replacement." Remove the comparison to the (now deleted) WS path.

4. **`DRYRUN.md`** — search for "WebSocket" or "WS" and remove any references that imply the Kalshi side uses one. The Coinbase WS reference is correct and stays.

5. **`scripts/visualize_market.py`** — the comment *"Kalshi collector — REST polling at 1 Hz (no WebSocket)"* is fine as a positive statement of what the script does. Leave it.

6. **Verification:**
   ```bash
   grep -ri "kalshi.*websocket\|kalshi.*ws_\|kalshi_ws\|self\._ws" --include="*.py" --include="*.md"
   ```
   Should return zero matches outside CLAUDE.md (which references the historical decision in §5.2 and §13).

7. **Hard rule going forward:** the Kalshi side is REST-polling-only. Do not reintroduce a Kalshi WebSocket implementation without an explicit RFC. The REST path proved more reliable and is well within Kalshi's rate limits at 1 Hz.

### 14.11 Verification — should be a clean grep

After the edits, the following greps should return zero matches (or only matches inside CLAUDE.md, .env.example comments, and explicit historical notes):

```bash
grep -r "polybot"        --include="*.py"  --include="*.toml"
grep -r "polymarket"     --include="*.py"  --include="*.toml" --include="*.md"
grep -r "py-clob-client" --include="*.toml" --include="*.txt"
grep -r "CLOB_HOST"      --include="*.py"  --include="*.md"
grep -r "CHAIN_ID"       --include="*.py"  --include="*.md"
grep -r "PRIVATE_KEY"    --include="*.py"  --include="*.md"   # except KALSHI_PRIVATE_KEY_*
grep -r "chainlink"      --include="*.py"
grep -r "RTDS"           --include="*.py"
grep -ri "kalshi.*ws\b\|kalshi_ws\|self\._ws" --include="*.py"   # Kalshi WS residue
grep -r "CoinbaseBook\|cb_momentum\|cb_book"  --include="*.py"   # post-WI-3 generic naming
```

Note: `binance` is NOT on this grep list — Binance is a supported spot feed under the new abstraction, so references in `betbot/kalshi/binance_feed.py`, `config.py`, `.env.example`, and `scripts/visualize_market.py` are expected and correct.

Allowed remaining matches:

- `CLAUDE.md` — historical context (e.g. §1.2 explaining why Kalshi+Coinbase replaced the old stack, §13 listing what was removed).
- `lag_plot_btc.png` — file name only, no code reference.

If any other file shows a match, the cleanup isn't complete. Iterate until the greps are clean.

### 14.12 Final smoke test

```bash
python -c "import betbot; import betbot.kalshi.scheduler"   # imports cleanly, no polybot.* errors
python scripts/run_kalshi_bot.py --fresh                    # bot starts with default SPOT_SOURCE=coinbase
SPOT_SOURCE=binance python scripts/run_kalshi_bot.py --fresh   # works if region permits, fails fast otherwise
pytest betbot/tests                                          # passes (or "no tests collected" — both fine)
```

Commit as a single PR titled "Remove Polymarket dead code; rename Kalshi WS residue; introduce SpotFeed abstraction" with `git rm -r` listed in the description for auditability.

### 14.13 What stays

For clarity, the following are **not** dead code and must be preserved:

- Everything under `betbot/kalshi/` (after the WI-3 renames): `book.py` (with `SpotBook` class), `coinbase_feed.py`, `binance_feed.py` (new), `kalshi_rest_feed.py`, `auth.py`, `config.py`, `features.py`, `model.py`, `scheduler.py`, `tick_logger.py`, `training_buffer.py`.
- `scripts/run_kalshi_bot.py`, `scripts/test_trade.py`, `scripts/check_kalshi_balance.py`, `scripts/backtest.py`, `scripts/analyze_run.py` — all use the Kalshi stack.
- `scripts/visualize_market.py` after the §14.9 adaptation.
- `CLAUDE.md`, `DRYRUN.md`, `.gitignore`, `lag_plot_btc.png`.

---

## 15. Pending work items — what Claude Code needs to execute

> ### 🛑 STOP — workflow check
>
> **At the start of the session:** if you have not invoked **`/superpowers`** and **`/gsd`**, do not start working through this list. Go back to the top of this file and read the mandatory-workflow block. The work below WILL break things if attempted ad-hoc.
>
> **At the end of the session:** before declaring any work item done, removing any PENDING marker, or collapsing this section, you MUST invoke **`/ultra-review`** as the closing verification gate. See §15.9 for what the review covers. The work is not done until ultra-review has passed.
>
> **First action of the working session:** post the lines:
>
> ```
> Invoking /superpowers and /gsd before starting work on CLAUDE.md §15.
> Will invoke /ultra-review at the end before marking any work item done.
> ```
>
> and actually invoke `/superpowers` and `/gsd` immediately. Save `/ultra-review` for after WI-1 through WI-5 are merged.

This section is the master checklist. Each work item points at the detailed spec elsewhere in this document. Do them in the suggested order; later items assume earlier ones are landed.

### 15.1 Work-item summary

| WI  | Title                                              | Spec ref | Complexity | Status |
|-----|----------------------------------------------------|----------|------------|--------|
| WI-1 | Remove Polymarket-era dead code                   | §14.1 – §14.8 | Low–Medium | ☐ Not started |
| WI-2 | Remove residual Kalshi-WS naming and references   | §14.10   | Low        | ☐ Not started |
| WI-3 | Pluggable spot feed (Coinbase ↔ Binance)          | §5.6     | Medium     | ☐ Not started |
| WI-4 | Forward-projection features (`x_proj_*`)          | §6.6     | Medium     | ☐ Not started |
| WI-5 | Maker-entry execution (Stage A pilot + Stage B)   | §11.1    | Medium–High | ☐ Not started |
| **GATE** | **Run `/ultra-review`** — closing verification | §15.9 | Required | ☐ Not run |

The `/ultra-review` row is **not** an additional WI; it is a mandatory gate that runs after WI-1 through WI-5 are merged. Until it has run and passed, the work is not done — see §15.8 / §15.9.

### 15.2 Suggested execution order and rationale

1. **WI-1 first.** Removing the dead Polymarket directories and config cruft strips ~250 KB of code that breaks `grep`-based search and confuses any reader (including future Claude Code sessions). Doing this first means every subsequent step works against a cleaner mental model. Commit and push after WI-1 lands; do not bundle it with later items.
2. **WI-2 second.** Tiny renaming/docstring pass. Catches `self._ws` and a few historical comments. Commit separately.
3. **WI-3 third.** This is the biggest refactor: rename `CoinbaseBook` → `SpotBook`, introduce `BinanceFeed`, rename feature columns, add `SPOT_SOURCE` config. Doing this AFTER WI-1 means there's no risk of accidentally salvaging dead code; doing it BEFORE WI-4 means the projection features in WI-4 can use the final feature names (`spot_momentum_*`) directly.
4. **WI-4 fourth.** Adds three new features and validates them against backtest. Builds on the cleaned, abstracted base from WI-1/2/3.
5. **WI-5 Stage A fifth.** Once features are stable, instrument the dry-run loop to log "would-have-posted limit" records and start collecting fill-rate data. Completely safe — no real orders. Run for at least a week before deciding on Stage B.
6. **WI-5 Stage B last (and only conditionally).** Build the full execution machinery only if Stage A's pilot data shows maker entry is worth it (≥25% P&L lift, ≥50% fill rate). If pilot data says no, skip Stage B and document the decision in CAVEATS.md.
7. **`/ultra-review` after all merge.** Closing verification gate per §15.9. Only after this passes do you remove PENDING markers and collapse §15.

Each WI is its own pull request (Stage A and Stage B of WI-5 are separate PRs). Do not stack them. `/ultra-review` runs against the post-merge `kalshi` branch HEAD, not against any individual PR.

### 15.3 WI-1 — Remove Polymarket-era dead code

**Spec:** §14.1 through §14.8.

**Files touched:** `betbot/clients/`, `betbot/models/`, `betbot/state/`, `betbot/strategy/`, `betbot/execution/`, `betbot/infra/`, `betbot/research/`, `betbot/cli/` (entire directories deleted); `betbot/main.py`, `scripts/demo_binance.py`, `scripts/demo_coinbase.py` (deleted); `betbot/__init__.py`, `pyproject.toml`, `requirements.txt`, `.env.example`, `RUNBOOK.md` (edited).

**Acceptance criteria:**
- All greps in §14.11 return zero matches in `*.py` / `*.toml` / `*.md` / `*.txt` (allowed exceptions in §14.11 only).
- `python -c "import betbot; import betbot.kalshi.scheduler"` succeeds with no errors.
- `python scripts/run_kalshi_bot.py --fresh` starts cleanly and runs for 5+ minutes.

**PR title:** `Remove Polymarket / Binance-legacy dead code`

### 15.4 WI-2 — Remove residual Kalshi-WS naming and references

**Spec:** §14.10.

**Files touched:** `betbot/kalshi/scheduler.py` (rename `self._ws` → `self._kalshi_feed` throughout); `betbot/kalshi/auth.py` (trim docstring); `betbot/kalshi/kalshi_rest_feed.py` (reframe docstring); `DRYRUN.md` (drop any Kalshi-WS implications).

**Acceptance criteria:**
- `grep -ri "kalshi.*ws\b\|kalshi_ws\|self\._ws" --include="*.py" --include="*.md"` returns matches only in CLAUDE.md (historical references).
- Bot still runs.

**PR title:** `Clean up residual Kalshi-WebSocket naming`

### 15.5 WI-3 — Pluggable spot feed (Coinbase ↔ Binance, same features)

**Spec:** §5.6.

**Files touched:**
- `betbot/kalshi/book.py` — rename `CoinbaseBook` → `SpotBook`.
- `betbot/kalshi/coinbase_feed.py` — update constructor to take `SpotBook`.
- `betbot/kalshi/binance_feed.py` — **new file**, `BinanceFeed` class against the `SpotFeed` Protocol.
- `betbot/kalshi/spot_feed.py` — **new file**, the `SpotFeed` Protocol definition.
- `betbot/kalshi/features.py` — rename `cb_momentum_30s` → `spot_momentum_30s`, etc.
- `betbot/kalshi/scheduler.py` — update variable names (`cb_book` → `spot_book`); `_bootstrap_from_history()` uses new column names.
- `betbot/kalshi/config.py` — add `SPOT_SOURCE`, add Binance WS URL constant.
- `scripts/run_kalshi_bot.py` — wire feed selection from `SPOT_SOURCE`.
- `scripts/backtest.py` — use renamed feature columns.
- `scripts/visualize_market.py` — adapt to use the new feed classes (per §14.9).
- `.env.example` — document `SPOT_SOURCE` flag.

**Acceptance criteria** (from §5.6):
- Bot runs with `SPOT_SOURCE=coinbase` (default).
- Bot runs with `SPOT_SOURCE=binance` (or fails fast with a clear error if geo-blocked).
- `grep -r "CoinbaseBook\|cb_book\|cb_momentum" --include="*.py" betbot/` returns zero matches.
- `[Refit]` log line shows `spot_momentum_30s`, `spot_momentum_60s` (no `cb_` prefix).
- A backtest run on `logs/ticks.csv` collected in Coinbase mode and another collected in Binance mode produce comparable feature distributions and refit coefficients on similar time windows.

**PR title:** `Introduce pluggable SpotFeed abstraction (Coinbase ↔ Binance)`

### 15.6 WI-4 — Forward-projection features

**Spec:** §6.6.

**Files touched:**
- `betbot/kalshi/features.py` — add `x_proj_30`, `x_proj_60`, `x_proj_120` to `FEATURE_NAMES`, `FeatureVec`, `as_array()`, `build_features()`. **Critical:** do NOT add these indexes to `_LAG_INDICES` — they must NOT be substituted in `settled_array()`.
- `betbot/kalshi/scheduler.py` — `_bootstrap_from_history()` appends the three new projection values in the right order.
- `scripts/backtest.py` — compute `x_proj_*` columns, include in feature matrix.
- (Optional) Old `logs/ticks.csv` files remain valid; existing models need a refit to incorporate the new features. A `--fresh` run is the cleanest way to validate.

**Acceptance criteria** (from §6.6):
- `[Refit]` log lines show non-zero β coefficients on at least one of `x_proj_30`, `x_proj_60`, `x_proj_120` after a few hours of live data.
- `q_settled` distribution shifts modestly in the direction of recent momentum compared to the pre-WI-4 baseline (sanity check).
- `python scripts/backtest.py --sweep` produces a P&L number that is **at least as good** as the pre-WI-4 baseline on out-of-sample data. If it's worse, the projection hypothesis isn't supported and the features should be removed (revert WI-4).

**PR title:** `Add forward-projection features to capture Coinbase drift expectation`

### 15.7 WI-5 — Maker-entry execution (limit orders on entry, taker on exit)

**Spec:** §11.1.

**Two-stage. Do NOT skip Stage A.**

#### Stage A — Phase-1 simulated pilot (cheap; safe to run during dry-run)

**Files touched:**
- `betbot/kalshi/scheduler.py` — when an `entry` event would fire, append a `would_have_posted` record to `decisions.jsonl` (or a parallel `maker_pilot.jsonl`) with `{ts, side, post_price, intended_size}`. No real orders.
- `scripts/analyze_run.py` (or a new `scripts/analyze_maker_pilot.py`) — walk forward from each `would_have_posted` record and check whether the post side ever traded at-or-better than `post_price` within `MAKER_POST_TTL_S` (default 30 s). Emit fill-rate and simulated maker-vs-taker P&L tables.

**Acceptance criteria for Stage A:**
- 1-7 days of pilot data collected during the standard dry run.
- Fill-rate report broken down by post strategy: passive (post at `yes_bid`) and aggressive (`yes_bid + 1 tick`).
- Simulated maker P&L vs current taker baseline computed on the same trade signals.
- **Gate to Stage B:** simulated maker P&L beats taker baseline by ≥ 25% on out-of-sample data, AND fill rate ≥ 50% on at least one post strategy. If either fails, **stop**: the strategy is at its execution ceiling on takers; do not build the order machinery.

**PR title (Stage A):** `Add maker-entry simulation pilot to dry-run logging`

#### Stage B — Full execution machinery (Phase 2 only)

**Do not start Stage B until:** Stage A passes the gate above, AND all four Phase-2 decision criteria in §10.3 are met (model is healthy, edge is real, etc.).

**Files touched:**
- `betbot/kalshi/orders.py` — **new file**, `OrderManager` class with `place_limit()`, `cancel()`, `cancel_all()`, plus state reconciliation against `GET /trade-api/v2/portfolio/orders`.
- `betbot/kalshi/scheduler.py` — extend the position state machine to include `pending_entry` and `pending_exit`. Window rollover cancels pending orders before clearing state. Update `_tick()` so that when `ENTRY_MODE == "maker"`, entry posts a limit instead of crossing; cancel-on-stale after `MAKER_POST_TTL_S`. Edge calc updated: `fee_up = 0.0` when maker entry is in use.
- `betbot/kalshi/config.py` — add `ENTRY_MODE`, `MAKER_POST_TTL_S`, `MAKER_POST_OFFSET_TICKS` (defaults: `taker`, 30, 0).
- `scripts/backtest.py` — add `--maker-entry` flag that simulates limit fills against logged top-of-book + trade-flow data using the same model as Stage A.
- `.env.example` — document `ENTRY_MODE` flag.

**Acceptance criteria for Stage B:**
- `ENTRY_MODE=taker python scripts/run_kalshi_bot.py` runs with identical behavior to pre-WI-5 (no regression).
- `ENTRY_MODE=maker` mode can be exercised end-to-end on Kalshi demo / paper environment.
- `OrderManager` correctly cancels stale orders within `MAKER_POST_TTL_S` of posting.
- Realized fill rate on real Kalshi within 20% of Stage-A simulated fill rate. If real fills are dramatically lower, the simulation model needs work — kick back to Stage A before scaling.
- One small live trial ($25-50 wallet) shows realized P&L per trade higher than the matched taker-mode baseline, with no orders left orphaned (no leaked limits sitting open after a window rollover).

**PR title (Stage B):** `Add OrderManager and maker-entry execution mode (off by default)`

**What does NOT change:**

- Feature engineering, the regression model, `q_settled` computation, sizing, exit logic. WI-5 is purely an execution change. The strategy decision "should we be long YES" is identical; only "how do we get long" changes.

**Risk note.** The exit leg stays as a taker (cross to bid on `lag_closed`, cross to bid on `stopped_out`). This is non-negotiable for now. Maker exits are tempting (zero fee on both legs!) but introduce fill-uncertainty on the time-critical leg, which can turn winners into losers when the next adverse Coinbase move arrives before the exit limit fills. Maker-exits would be a separate WI-6 if Stage B succeeds and we want to push further.

### 15.8 Definition of done — across all five work items

The whole sequence is "done" when, **in this exact order**:

1. All five PRs are merged (WI-1 through WI-4 plus WI-5 Stage A; WI-5 Stage B may be deferred — see below).
2. The bot runs for 24+ hours under `SPOT_SOURCE=coinbase` plus a short startup verification under `SPOT_SOURCE=binance`.
3. A backtest sweep on the new feature set shows positive P&L on out-of-sample data.
4. Stage A pilot data has been collected and analyzed; the maker-vs-taker P&L comparison and fill-rate report are committed to the repo (e.g. `docs/maker_pilot_2026-MM-DD.md`).
5. `grep` smoke tests in §14.11 are clean.
6. **`/ultra-review` is invoked and passes.** See §15.9 for what the review covers and what counts as passing. Do not proceed to step 7 until this gate is green.
7. CLAUDE.md is updated: remove "PENDING" markers from §5.6, §6.6, and §11.1 headers; collapse §15 to a brief "all WIs complete as of `<date>`" note (or delete it entirely). If WI-5 Stage B is deferred, leave §11.1 marked PENDING and §15.7-Stage-B in §15 with a note explaining the defer.
8. `CAVEATS.md` (if it exists) has the corresponding caveats moved to a "resolved" section.

**On deferring WI-5 Stage B.** Stage B is the only WI that touches real money. It's reasonable to merge Stages 1-4 plus WI-5 Stage A, run the dry-run pilot for a week or two, then make the Stage B decision based on the pilot data. If pilot data says maker entry isn't worth it, drop Stage B entirely and document the decision in CAVEATS.md.

**The order matters.** Steps 1-5 are work and data collection; step 6 is review; steps 7-8 are documentation that records the work as done. Updating CLAUDE.md (step 7) before `/ultra-review` (step 6) has run is forbidden — the doc is the source of truth and must not claim "done" before the review confirms it.

### 15.9 `/ultra-review` — what it covers and what passing means

`/ultra-review` is the closing-step verification skill. It is **mandatory** before any work item is marked done in CLAUDE.md or any PENDING marker is removed.

**What `/ultra-review` is expected to check** (this is the spec for the assistant invoking it; the skill itself defines the exact procedure):

1. **Spec ↔ implementation alignment.** For each work item (WI-1 through WI-5), every concrete instruction in its spec section (§14, §5.6, §6.6, §11.1) maps to a real change in the merged code. No spec bullet is silently skipped. No code change exists that isn't traceable back to a spec bullet.
2. **Acceptance criteria are met.** Re-run every acceptance check listed under §15.3 – §15.7. Every grep returns the expected zero/non-zero result. Every smoke test passes. No "passes locally" hand-waving. For WI-5, this includes verifying the Stage A pilot actually ran for the required duration and the fill-rate analysis was committed.
3. **Cross-references in CLAUDE.md still resolve.** Every `§N.M` and `§N WI-X` pointer in the document points at a real, correctly-numbered section. After step 7 collapses §15, audit the doc again for broken pointers.
4. **No half-finished refactors.** Search for stray references to deleted symbols (`CoinbaseBook`, `cb_momentum_*`, `polybot.*`, `self._ws`, etc.) anywhere in the repo, not just in `betbot/kalshi/`. A name that survived in a comment or a docstring counts as a finding.
5. **Behavior preservation where claimed.** §6.6 and §5.6 both claim "no behavior change in the regression's pure fitting path." Verify this empirically: refit on the same training data before and after WI-3/WI-4 and confirm the lag-feature β coefficients are unchanged within numerical noise. §11.1 claims "WI-5 Stage A is logging-only and does not change live behavior" — verify by diffing decision-loop output before and after WI-5 Stage A on the same input.
6. **Documentation honesty.** Every "FIXED" or "complete" claim in CLAUDE.md, CAVEATS.md, and PR descriptions corresponds to actual code state. No aspirational claims.
7. **Operational readiness.** Bot starts cleanly, runs for at least 30 minutes without crashing, produces sensible `[Refit]` log lines, and abstains for the right reasons during the warmup period. If WI-5 Stage B was implemented, also verify no orphaned limit orders remain after a forced shutdown + restart.

**What "passing" means:**

- Every check above returns green, or
- Any failures are addressed by additional commits before re-running the review, or
- Any failures the maintainer chooses to accept are explicitly logged in CAVEATS.md with rationale before the work is marked done.

**What "not passing" means:**

- Do not remove PENDING markers from §5.6 / §6.6 / §11.1.
- Do not collapse §15.
- Do not declare the work item complete in any PR description, commit message, or external communication.
- File the failures as new caveats in CAVEATS.md or as new work items in §15, then either fix them or explicitly accept them with reason.

**The review is one shot per "complete" claim.** If the review finds issues, fix them, then re-run the full review. Do not selectively re-run only the parts that failed — the work as a whole gets re-reviewed. This prevents drift between fixes and ensures any side-effect from the fix itself gets caught.

---

## 16. Prior art and references

- **Stoikov 2018 "The Micro-Price"** — foundational paper on microprice as a forward predictor of midpoint. Required reading for understanding why microprice is the right input to the regression.
- **Cont-Kukanov-Stoikov 2014** — Order Flow Imbalance. Currently unused but the natural next feature if `r2_cv` proves underwhelming.
- **Hayashi-Yoshida 2005** — non-synchronous covariance estimator for tick data from two venues. Useful for offline lag analysis as a sanity check on what the regression's β coefficients are saying.
- **Kalshi API docs** — `https://trading-api.readme.io/reference/getmarket` etc. The auth scheme (RSA-PSS over `timestamp + method + path`) is implemented in `betbot/kalshi/auth.py`.
- **Coinbase Advanced Trade WebSocket docs** — the `ticker` channel is unauthenticated and exposes `best_bid`, `best_ask`, `best_bid_quantity`, `best_ask_quantity`, which is everything we need for microprice.

The lag-arbitrage strategy itself isn't novel (every prediction-market shop has tried it), but the specific implementation — ridge regression on lagged log-microprice features with adaptive refit and cash-out exit — is uncommon in the published Kalshi/Polymarket bots and is the central contribution of this codebase.
