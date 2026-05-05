# CLAUDE.md — Polymarket 5-Minute Crypto Lag-Arbitrage Bot

> **Status:** Planning / pre-implementation. This document is the source of truth for the project's design, architecture, and reasoning. Read it end-to-end before generating any code.

---

## 1. Project goal

Build an automated trading bot that takes positions in Polymarket's 5-minute "Bitcoin Up or Down" and "Ethereum Up or Down" binary prediction markets by exploiting the lag between Coinbase spot price movements and Polymarket order-book repricing. The bot enters when Polymarket has not yet caught up to where Coinbase says fair value is, and exits when Polymarket has caught up — collecting the lag-close as profit, regardless of how the underlying window resolves.

**The strategy is cash-out lag arbitrage. There is one model, one edge calculation, one entry rule, one exit rule.**

**Edge thesis.** Coinbase's order book is the leading indicator. Polymarket's CLOB takes some number of seconds to reprice after Coinbase moves. During that interval, Polymarket's quoted probability differs from the spot-implied probability by an amount that exceeds round-trip costs (entry fee + entry slippage + exit fee + exit slippage + spread). When that condition holds, we enter. When the lag closes, we exit. This is the entire bot.

**The bot's job, distilled to one sentence:** *every 10 seconds, compute a single number — the net edge `delta` in probability points after fees and slippage — and feed that number into the tiered Kelly table to produce a bet size.* If `delta` is below the lowest tier threshold, abstain. Otherwise enter at the prescribed size and exit when `delta` collapses (lag closed) or inverts (thesis broken).

**Why not Black-Scholes binary?** Black-Scholes prices an option by assuming GBM with volatility σ; it answers "what's the no-arbitrage probability the underlying ends above K at horizon τ?" That's a forecasting question. We're not forecasting — we're predicting where Polymarket's *quote* will be in N seconds based on where Coinbase's *spot* is right now. That's a regression problem on (Polymarket price) vs (lagged spot features), and modeling it as such is more honest, more accurate, and removes an entire layer of estimation noise (volatility) from the critical path.

**Why not pure latency arbitrage at sub-second timescales?** Polygon-validator-adjacent bots already own that niche. Our edge window is 10-90 seconds, which is achievable with normal infrastructure but invisible to traders who poll Polymarket manually.

Phase 1 is read-only. We collect 24 hours of tick-level data, fit and validate the regression, simulate the strategy across a sweep of exit-rule parameters, and only commit real capital after verifying that the edge actually realizes as P&L.

**Hard constraint.** Every entry decision is reducible to: "regression-predicted settled probability `q_settled`, market price `q_actual` at Polymarket ask, edge `delta = |q_settled − q_actual| − fee − slippage` clears the tier threshold, bet wallet fraction prescribed by Kelly tier T." Every exit decision is reducible to: "Polymarket has caught up (`delta` has compressed below the lag-close threshold → take profit), or spot reversed (`delta` has inverted by more than the stop threshold → cut loss), or τ is small enough to default to resolution."

**Non-goals:**

- We are not building a market-making bot. No quoting both sides.
- We are not chasing sub-second latency arbitrage.
- We are not building anything that requires placing or cancelling orders faster than ~1 second end-to-end.
- We are not predicting BTC's close-time price. We are predicting Polymarket's *next several seconds* of quote movement.

---

## 2. The market mechanic

Every 5 minutes, Polymarket opens a new market on each of `BTC`, `ETH`. The market asks: "Will the asset's price at the close of this 5-minute window be ≥ the price at the open?"

- **Strike `K`** = Chainlink BTC/USD oracle price snapshot at the window-open boundary (`t mod 300 == 0`).
- **Resolution price** = Chainlink BTC/USD oracle price snapshot at the window-close boundary.
- Yes shares pay $1 if close ≥ open, else $0. No shares are the complement.
- Yes-share price `q ∈ [0, 1]` *is* the implied probability of Up.
- Window slug is deterministic: `{asset}-updown-5m-{window_open_unix_ts}`.

**Critical:** The strike `K` is the **Chainlink oracle's first observation at or after the window boundary**, not Coinbase or Binance spot at that instant. Subscribe to Polymarket's RTDS WebSocket `crypto_prices_chainlink` channel filtered to `btc/usd` (or `eth/usd`) and capture the first tick at or after each window-open timestamp. This is non-negotiable — wrong K destroys the regression's K-relative features.

**Cash-out is supported.** A Yes-share position can be exited at any time before window close by selling into the order book at the prevailing bid. This is what makes the strategy viable. The exit price is set by the bid, not by resolution — we are exiting *into the market*, not *waiting for the oracle*.

---

## 3. The strategy

### 3.1 Conceptual model

Treat Coinbase as ground truth and Polymarket as a delayed function of Coinbase. At any moment `t`:

- `q_actual_t` = what Polymarket *is* quoting (the Yes-share ask).
- `q_settled_t` = what Polymarket *would* quote if it had finished digesting all spot moves up to time `t`.
- `delta_t = q_settled_t − q_actual_t` (signed, on the side we're considering).

When `delta` is large and positive on the Up side, Polymarket is offering Up shares cheaper than fair value. We buy. Some seconds later, Polymarket's book updates to reflect the spot move, `q_actual` rises toward `q_settled`, the lag closes, and we sell back at the new (higher) market price.

The model's job is to compute `q_settled_t` from observable spot data. That's what the lead-lag regression does.

### 3.2 Why microprice, not midpoint

The spot-side input we use is **microprice**, the imbalance-weighted fair value:

```
microprice = (best_bid × ask_size + best_ask × bid_size) / (bid_size + ask_size)
```

When the bid is heavy, microprice is closer to the ask (next trade likely lifts the ask). When the ask is heavy, microprice is closer to the bid. This is the cleanest available proxy for "where is spot heading in the next few seconds" using only L1 book data.

Using microprice rather than midpoint moves our spot signal forward in time by 1-10 seconds. That compounds with Polymarket's 30-90 second lag to produce a larger and earlier edge signal. Midpoint trails microprice; if we used midpoint we'd be racing other lag-arb bots that already use microprice, and we'd lose.

### 3.3 The lead-lag regression — the only model

We fit a logistic regression that predicts Polymarket's current Yes-share price from Coinbase spot history. The model's parameters are refit every 5 minutes on a rolling 4-hour window of recent (q, features) pairs.

**Target:**

```
y_t = logit(q_actual_t)        where q_actual_t is the Polymarket Yes-ask at time t
```

**Features (window-aware — all referenced to the current window's strike K):**

```
x_now    = log(microprice_t      / K)
x_15     = log(microprice_{t-15s} / K)
x_30     = log(microprice_{t-30s} / K)
x_45     = log(microprice_{t-45s} / K)
x_60     = log(microprice_{t-60s} / K)
x_90     = log(microprice_{t-90s} / K)
x_120    = log(microprice_{t-120s}/ K)

tau      = τ (seconds remaining in window)
inv_sqrt_tau = 1/√(τ + 1)

# Microstructure features
ofi_30s          # spot OFI over last 30s
pm_book_imbalance_t
momentum_30s
momentum_60s
cross_asset_momentum_60s
```

**Model:**

```
logit(q_t) = α + Σ βₖ · x_k + γ · tau + δ · inv_sqrt_tau
                + θ_OFI · ofi_30s + θ_PM · pm_book_imbalance + ...
```

Ridge regression, regularization chosen by cross-validation. ~12-15 features total, fit on ~14,400 samples per asset (4 hours × 3,600 seconds, downsampled to 1-second observations).

**Two derived predictions are computed at every decision tick:**

```
# 1. What the model thinks Polymarket SHOULD be quoting given current data.
#    Useful as a sanity check: if this is far from q_actual, something is wrong
#    (regime shift, model is stale, feed problem, etc.)
q_predicted = sigmoid(logit_q  given current-and-lagged spot)

# 2. What Polymarket WILL quote once it has digested current spot.
#    Substitute current microprice into every lookback slot:
logit_q_settled = α + (Σ βₖ) · x_now
                    + γ · tau + δ · inv_sqrt_tau
                    + θ_OFI · ofi_30s + θ_PM · pm_book_imbalance + ...
q_settled = sigmoid(logit_q_settled)
```

`q_settled` is our forecast of where Polymarket will arrive once it finishes processing current spot. This is the single most important output of the model.

### 3.4 The lag is learned, not hard-coded

The fitted β coefficients *are* the lag distribution. They tell us how much of Polymarket's current quote is explained by spot at each lookback horizon. Three patterns the data could show:

**Polymarket is fast (no lag).** β₀ (the `x_now` coefficient) is dominant; β₁..β₆ are near zero. Polymarket's quote is best explained by spot right now — no lag to arbitrage. **Strategy thesis is dead.**

**Polymarket lags by ~60 seconds.** β₀ is small, β₄ (the 60s-ago coefficient) is largest, neighboring β's decay smoothly on either side. This is the regime our strategy needs.

**Mixed / no clear lag.** All β's moderate, no clear peak. Polymarket responds to a weighted average of recent spot. Strategy can still work but edge is smaller and noisier.

We don't have to pick which regime is true — the regression tells us. For a single human-readable summary number, compute the β-weighted average lag at each refit:

```
estimated_lag_seconds = (15·β₁ + 30·β₂ + 45·β₃ + 60·β₄ + 90·β₅ + 120·β₆)
                       / (β₁ + β₂ + β₃ + β₄ + β₅ + β₆)
```

This is logged every 5 minutes for human interpretability and dashboard display. The bot doesn't *use* it (it uses the full coefficient vector); the bot logs it. If `estimated_lag_seconds` drifts from 45s down to 12s over a few days, that's quantitative evidence that Polymarket's market makers are getting faster and our edge is shrinking.

### 3.5 Computing the edge — the single number that drives all decisions

Every 10 seconds, after computing `q_settled`, compute the edge on each side:

```python
edge_up_raw   = q_settled - q_up_ask                 # signed; positive = buy Up
edge_down_raw = (1 - q_settled) - q_down_ask         # signed; positive = buy Down

# Cost adjustments (per dollar bet):
#   - Polymarket taker fee: Theta * p * (1-p) where Theta ≈ 0.05
#   - Slippage: estimated from depth at our intended bet size
fee_up   = THETA * q_up_ask   * (1 - q_up_ask)
fee_down = THETA * q_down_ask * (1 - q_down_ask)
slip_up   = estimate_slippage(asset, side='up',   size=intended_size)
slip_down = estimate_slippage(asset, side='down', size=intended_size)

edge_up_net   = edge_up_raw   - fee_up   - slip_up
edge_down_net = edge_down_raw - fee_down - slip_down

# The "edge" — a single signed number representing best opportunity this tick
if edge_up_net > edge_down_net:
    edge_signed = edge_up_net    # positive = buy Up
    favored_side = 'up'
else:
    edge_signed = -edge_down_net # negative = buy Down (sign flipped for consistency)
    favored_side = 'down'

# The magnitude is what the Kelly tier table consumes
edge_magnitude = abs(edge_signed)
```

`edge_magnitude` is the number that drives every entry decision. If `edge_magnitude < 0.02` (lowest tier floor), abstain. Otherwise look up the tier and bet the corresponding wallet fraction on `favored_side` at the corresponding ask price.

### 3.6 Worked example

To make the above concrete, consider a single 10-second decision tick on the BTC market:

```
Window opened 2 minutes ago.
K = 100,000             (Chainlink at window open)
τ = 180 seconds remaining

Coinbase right now:
  microprice_t = 100,500    (BTC is $500 above strike)
  60 seconds ago, microprice was 100,200    ($200 above)
  120 seconds ago, microprice was 100,050   ($50 above)

Polymarket right now:
  q_up_ask = 0.72           (market says 72% chance Up)

The fitted regression has learned (over the last 4 hours of data):
  - β₀ (current spot)   = 0.5
  - β₄ (60s-ago spot)   = 4.2     <-- dominant coefficient, lag is ~60s
  - β₆ (120s-ago spot)  = 1.8
  - other β's smaller

Step 1: Confirm Polymarket's current quote is consistent with the lagged history
        (sanity check; q_predicted should be close to q_actual).

  logit(q_predicted) = α + β₀·log(100500/100000) + β₄·log(100200/100000)
                       + β₆·log(100050/100000) + ... + tau/microstructure terms
                     ≈ logit(0.71)            (close to observed 0.72; model is healthy)

Step 2: Compute q_settled by substituting current microprice into all lag slots.

  logit(q_settled) = α + (β₀ + β₄ + β₆ + ...) · log(100500/100000)
                     + tau/microstructure terms
                   ≈ logit(0.79)            (this is where Polymarket is heading)

Step 3: Compute edge.

  edge_up_raw = 0.79 - 0.72 = 0.07           (7 cents per share)
  fee_up      = 0.05 * 0.72 * 0.28 = 0.0101  (~1 cent)
  slip_up     = ~0.005                       (depth-dependent estimate)

  edge_up_net = 0.07 - 0.0101 - 0.005 = 0.0549

  edge_magnitude = 0.0549
  favored_side   = 'up'

Step 4: Look up Kelly tier.

  edge_magnitude = 0.0549 falls in the (0.04, 0.03) tier
    -> bet 3% of wallet on Up at q_up_ask = 0.72

Step 5: Place bet (Phase 2 only; in Phase 1 we just log the would-be entry).

  If wallet = $1,000:
    bet_size = $30
    contracts = $30 / $0.72 = ~41.7 Up shares

Step 6: Wait. Over the next ~60 seconds, if the lag closes as predicted:

  q_up_ask rises from 0.72 → ~0.78 (catching up to where spot says it should be)
  q_up_bid rises correspondingly to ~0.77

  We sell our 41.7 shares at q_up_bid = 0.77:
    gross proceeds = 41.7 * 0.77 = $32.11
    exit fee = 0.05 * 0.77 * 0.23 * 32.11 = $0.28
    entry fee already paid = 0.05 * 0.72 * 0.28 * 30 = $0.30
    net P&L = 32.11 - 30 - 0.28 - 0.30 = $1.53

  Realized return: $1.53 / $30 = 5.1% on the trade.
```

The trade made money because Polymarket caught up to spot, regardless of whether BTC ends up above 100,000 at T+0. **That's the entire point of the strategy.** The over/under outcome at resolution doesn't enter into our P&L on this trade — we already exited.

### 3.7 Entry rule

Every 10 seconds:

```
1. Active 5m market? If no, idle.
2. Compute τ.
3. Read inputs: microprice, lagged microprices, K, τ, OFI, PM book, etc.
4. Compute q_settled and q_predicted from the latest fitted regression.
5. Compute edge_up_net, edge_down_net, edge_magnitude, favored_side.
6. If we already have an open position in this window: skip entry, run exit logic.
7. If edge_magnitude > tier_floor:
     Enter on favored_side, size = wallet * kelly_fraction(edge_magnitude).
   Else:
     Abstain. Reason = 'edge_below_floor'.
8. Persist decision row.
```

**One position per window per asset.** Never flip sides mid-window. Never stack positions on the same side in the same window.

**Tier table (user-specified):**

```python
KELLY_TIERS = [
    (0.30, 0.10),    # delta >= 0.30 -> 10% wallet
    (0.15, 0.08),    # delta >= 0.15 -> 8%
    (0.08, 0.05),    # delta >= 0.08 -> 5%
    (0.04, 0.03),    # delta >= 0.04 -> 3%
    (0.02, 0.015),   # delta >= 0.02 -> 1.5%
]
```

Note that the regression handles the τ-effect implicitly through its coefficients — we don't need a separate τ-conditional tier floor table. At small τ the model's q_settled changes rapidly with spot, so edges naturally appear bigger; at large τ they appear smaller. The tier floor is a flat 0.02 across all τ.

### 3.8 Exit rule

```python
LAG_CLOSE_THRESHOLD = 0.005   # exit when edge has compressed to half a cent
STOP_THRESHOLD      = 0.03    # exit if edge erodes by 3 cents below entry
FALLBACK_TAU        = 10      # default to resolution at this τ

def evaluate_exit(position, current_state):
    """
    position:
        side ('up'|'down'), entry_price, entry_tau, size_usd, edge_at_entry
    current_state:
        q_settled_now, q_up_ask_now, q_down_ask_now, q_up_bid_now, q_down_bid_now,
        tau_now
    """
    if position.side == 'up':
        edge_now = current_state.q_settled_now - current_state.q_up_ask_now
        exit_bid = current_state.q_up_bid_now
    else:
        edge_now = (1 - current_state.q_settled_now) - current_state.q_down_ask_now
        exit_bid = current_state.q_down_bid_now

    # 1. Lag closed -> profit-taking exit
    if edge_now < LAG_CLOSE_THRESHOLD:
        return ('exit', 'lag_closed', exit_bid)

    # 2. Thesis broken -> stop-loss exit
    if edge_now < position.edge_at_entry - STOP_THRESHOLD:
        return ('exit', 'stopped_out', exit_bid)

    # 3. Resolution fallback at small τ
    if current_state.tau_now < FALLBACK_TAU:
        return ('hold_to_resolution', None, None)

    # 4. Otherwise hold
    return ('hold', None, None)
```

The thresholds are sweep parameters in the dry-run replay (§10.3).

### 3.9 Process model

WebSocket handlers update state (spot book, Polymarket book, Chainlink window). The scheduler tick reads state, calls the regression, applies entry/exit logic. The regression refit runs as a separate background task every 5 minutes on the rolling history table.

---

## 4. Architecture

### 4.1 Layout

```
/repo
  /clients
    coinbase_ws.py           # WebSocket client for live BTC/ETH spot
    polymarket_ws.py         # WebSocket client for Polymarket CLOB book
    polymarket_rest.py       # REST client (orders, market metadata, fills)
    polymarket_rtds.py       # WebSocket client for RTDS Chainlink stream
  /state
    spot_book.py             # Live mid, top-N bid/ask, microprice, OFI accumulator
    poly_book.py             # Live Yes/No best bid/ask, depth, our open orders
    window.py                # Current window: open_ts, close_ts, K, slug, tokens
    history.py               # Rolling buffers for returns, OFI, features, q's
  /models
    regression.py            # Lead-lag ridge regression: fit, predict, q_settled
    features.py              # Feature engineering from raw spot/PM state
    edge.py                  # Computes edge_up_net, edge_down_net, edge_magnitude
    fees.py                  # Fee curve fee(price) for net-edge calculation
    slippage.py              # Slippage estimator from current PM book depth
  /strategy
    decision.py              # The 8-step decision loop
    kelly.py                 # Tier table, sizing logic
    entry.py                 # Entry rule (§3.7)
    exit.py                  # Cash-out exit rule (§3.8)
    risk.py                  # Daily loss cap, max open positions, circuit breakers
  /execution
    orders.py                # Order placement, retries (Phase 2)
    fills.py                 # Position tracking, P&L attribution
  /infra
    scheduler.py             # Main 10s tick loop
    refitter.py              # Background task: refit regression every 5 min
    config.py                # All tunables in one place
    logger.py                # Structured tick log -> Parquet for backtest replay
    secrets.py               # API keys, wallet config (env-loaded)
  /research
    backtest.py              # Replay logged ticks
    cashout_simulator.py     # Computes P&L from logged future books
    refit_offline.py         # Offline regression fitting + cross-validation
    parameter_sweep.py       # Sweep LAG_CLOSE x STOP x FALLBACK_TAU + ridge_alpha
    diagnostics.py           # R^2, lag-stability, edge-realization slope
  /cli
    polybot_metrics.py       # SSH-friendly metric queries via DuckDB
    polybot_ctl.py           # Pause/resume/status over Unix domain socket
  /tests
    ...
  CLAUDE.md
  README.md
  pyproject.toml
```

### 4.2 Data flow

```
Coinbase WS  ──tick──►  spot_book ──┐
                                    │
Polymarket WS ──update──► poly_book ┼──► scheduler (10s) ──► features ──► regression ──► edge ──► kelly ──► (Phase 2) orders
                                    │              │                            │
RTDS Chainlink ──tick──► window ────┘              ▼                            ▼
                                              parquet writer                refitter (every 5 min)
                                                                                │
                                                                                └──► models/regression.py
```

### 4.3 Process model

Single Python 3.11+ process, asyncio, `asyncio.TaskGroup`. Long-running tasks: Coinbase WS, Polymarket CLOB WS, Polymarket RTDS WS, scheduler, parquet writer, regression refitter. Six tasks total, all under one event loop.

---

## 5. Data sources and clients

### 5.1 Coinbase Advanced Trade WebSocket

- **Endpoint:** `wss://advanced-trade-ws.coinbase.com`
- **Auth:** JWT (ES256), regenerated per subscribe message; tokens expire after 120s. Use `coinbase.jwt_generator.build_ws_jwt(api_key, api_secret)`.
- **Channels:** `ticker`, `level2` (envelope name `l2_data`), `market_trades`, `heartbeats`. **One subscribe message per channel.**
- **`level2` semantics:** `new_quantity` is **absolute size at price level**, not a delta. `new_quantity == "0"` means level removed. Initial `type: "snapshot"` event replaces local book; subsequent `type: "update"` events apply diffs. We must compute OFI ourselves from observed size changes.
- **Sequence:** `sequence_num` strictly monotonic per connection. Gap → drop local book, reconnect, get fresh snapshot.
- **Library:** `coinbase-advanced-py` for SDK convenience, or raw `websockets` + `coinbase.jwt_generator` for tightest control. Use SDK for Phase 1.
- **State derived in `spot_book.py`:**
  - `mid` = (best_bid + best_ask) / 2
  - **`microprice`** — the regression's primary input, updated on every L2 event
  - `top_5_bid_levels[]`, `top_5_ask_levels[]` (price + size)
  - `last_trade_price`, `last_trade_size`
  - 1-second sampled microprice ring buffer over last 300 seconds (this is what the regression's lagged features read from)
  - per-event OFI accumulator (Cont-Kukanov-Stoikov, levels 1-5)

### 5.2 Polymarket CLOB WebSocket

- **Endpoint:** `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- **No auth required** for market channel. Subscribe by `assets_ids` (Yes token + No token), not by slug:
  ```json
  {"type":"market","assets_ids":["<UP_TOKEN>","<DOWN_TOKEN>"],"custom_feature_enabled":true}
  ```
- Send `PING` (literal text, not JSON) every 10s. Server sends `PING` every 5s; reply `PONG` within 10s.
- **Events:** `book` (full snapshot — bids/asks arrays), `price_change` (incremental — size is new resting size, not delta), `last_trade_price`, `tick_size_change`.
- **Known issue:** silent freeze ([py-clob-client #292](https://github.com/Polymarket/py-clob-client/issues/292)). Watchdog: if no `book` or `price_change` in 60s, force reconnect; if reconnect fails twice, fall back to REST polling.
- **REST `/book` is unreliable** ([#180](https://github.com/Polymarket/py-clob-client/issues/180)) — returns stale 0.99/0.01 ghost books. Always prefer WS book; use `get_price()` for sanity checks only.

### 5.3 Polymarket REST

- **Library:** `py-clob-client-v2` (current, post CLOB v2 / pUSD migration). Use the older `py-clob-client` only as fallback.
- **REST host:** `https://clob.polymarket.com`. **Gamma:** `https://gamma-api.polymarket.com` (slug → token resolution).
- **Slug format:** `{btc|eth}-updown-5m-{window_open_unix_ts}` where `ts % 300 == 0`. Resolve at each window boundary:
  ```python
  now = int(time.time())
  window_ts = now - (now % 300)
  slug = f"btc-updown-5m-{window_ts}"
  # GET https://gamma-api.polymarket.com/markets?slug={slug}
  # -> conditionId, clobTokenIds [up_token, down_token], tickSize, feesEnabled
  ```
- **Phase 1 calls used:** `get_market`, `get_price` (sanity), `get_midpoint`, `get_spread`, `get_tick_size`, `get_server_time`. No order placement.
- **Auth tiers** (Phase 2 only): L0 read, L1 EOA private key, L2 HMAC API key. For automated trading, L2 with proxy wallet (sig type 1) avoids per-trade gas.

### 5.4 Polymarket RTDS (Chainlink prices)

- **Endpoint:** `wss://ws-live-data.polymarket.com` — separate from the CLOB WS.
- **Subscribe BTC:**
  ```json
  {"action":"subscribe","subscriptions":[{
    "topic":"crypto_prices_chainlink","type":"*",
    "filters":"{\"symbol\":\"btc/usd\"}"}]}
  ```
- **Heartbeat:** `PING` every 5s.
- **Message:** `{topic, type, timestamp, payload:{symbol, timestamp, value}}`. Outer `timestamp` is server send (ms); inner `payload.timestamp` is Chainlink oracle observation time (ms).
- **K capture algorithm:**
  ```
  window_ts_ms = window_open_unix_ts * 1000
  K = first tick where payload.timestamp >= window_ts_ms
  ```
  If no such tick within window_ts_ms + 5000ms, mark window `K_uncertain=true` and abstain from all decisions in this window.

### 5.5 Polymarket fees

5-minute crypto markets charge dynamic taker fees, makers free.

```
fee_per_dollar(p) = Theta * p * (1 - p)   # Theta ≈ 0.05, peak ~3.15% at p=0.5
```

For this strategy, fees are paid on **both legs** (entry as taker; exit as taker if lag-closed or stopped out, fee-free if held to resolution). Total round-trip cost when entering at price `a` and exiting at price `e`:

```
total_cost = fee(a) + fee(e) + spread_at_exit + slippage_in + slippage_out
```

The lag-driven edge (the regression's `q_settled − q_actual`) must exceed this total cost for the trade to be net positive. The actual Θ for 5-min markets is borrowed from 15-min docs and confirmed empirically only after first live trades.

**Maker-rebate optimization (deferred to Phase 2):** Polymarket pays makers a rebate of 25-50% of taker fees from a daily pool. Posting the *exit* leg as a maker order (limit at our target exit price) captures the rebate and dodges the exit fee. In Phase 1 we model exits as takers (worst case); in Phase 2 we add a maker-exit mode and compare.

---

## 6. The lead-lag regression in detail

### 6.1 Why this is the right model for this strategy

The strategy thesis is "Polymarket lags Coinbase." The most direct way to express that hypothesis as a model is *literally* a regression of Polymarket's quote on lagged Coinbase prices. There's no need to model an option, infer a volatility, or assume a probability distribution. We just need to know how Polymarket's quote depends on recent spot history, and the regression learns that from data.

This formulation has properties no Black-Scholes formulation has:

- **Self-falsifying.** If the regression's R² is poor, Polymarket isn't actually predictable from spot the way we hypothesized. The dry run will tell us this directly.
- **Self-calibrating to costs.** We can include `q_actual_t-1` as a feature if we want and the model will learn whatever momentum is already priced in; we don't have to worry about double-counting signals.
- **Self-adapting.** As Polymarket's market makers get faster, the lag profile (β coefficients) shifts; the model picks this up at every refit. No human re-tuning needed.
- **No σ to be wrong about.** Volatility estimation is no longer a critical-path input. We log realized vol as a diagnostic, but a σ bug can't fake an edge signal anymore.
- **The edge is a pure number, with units of probability.** `q_settled - q_actual` is already in probability points, the same units as the Kelly tier thresholds. No translation needed.

### 6.2 Feature design

Every feature is referenced to the current window's strike K so that windows are comparable across the training data. The lookback grid is intentionally dense in the 0-90s range and sparse beyond, since most plausible Polymarket lags fall in that range.

```
Spot history (microprice / K, in log space):
  x_now    = log(microprice_t      / K)
  x_15     = log(microprice_{t-15s} / K)
  x_30     = log(microprice_{t-30s} / K)
  x_45     = log(microprice_{t-45s} / K)
  x_60     = log(microprice_{t-60s} / K)
  x_90     = log(microprice_{t-90s} / K)
  x_120    = log(microprice_{t-120s}/ K)

Time:
  tau           = τ in seconds
  inv_sqrt_tau  = 1 / √(τ + 1)         # makes near-close moves matter more

Spot microstructure:
  ofi_30s             # signed Order Flow Imbalance over last 30s, z-scored
  ofi_l5_weighted     # multi-level OFI
  momentum_30s        # log(microprice_t / microprice_{t-30s})
  momentum_60s

Polymarket microstructure:
  pm_book_imbalance   # (depth_up - depth_down) / total
  pm_trade_flow_30s   # net Yes-buying minus No-buying volume

Cross-asset (when modeling BTC, this is ETH; vice versa):
  cross_momentum_60s
```

About 14-16 features total. Ridge regression with regularization `α` chosen by 5-fold time-series cross-validation each refit. (`α` is a Phase 1 sweep parameter alongside the exit thresholds.)

The features are computed at every 10-second decision tick AND at every 1-second sample for training. The training samples vastly outnumber decision ticks, which is fine — we want a richly-trained model.

### 6.3 The training loop

```python
class RegressionRefitter:
    """
    Background task: every 5 minutes, refit the regression from the last 4 hours
    of (q_actual, features) pairs. New coefficients are atomically swapped into
    the live model used by the scheduler tick.
    """
    REFIT_INTERVAL_SECS = 300
    TRAINING_WINDOW_SECS = 4 * 3600

    async def run(self, history: HistoryStore, model: RegressionModel):
        while True:
            await asyncio.sleep(self.REFIT_INTERVAL_SECS)
            try:
                training_data = history.fetch_window(self.TRAINING_WINDOW_SECS)
                if len(training_data) < MIN_TRAIN_SIZE:
                    continue                                # cold-start, skip
                X, y = build_design_matrix(training_data)
                new_coefs, diagnostics = fit_ridge_cv(X, y)
                model.atomic_swap(new_coefs)
                log_model_version(new_coefs, diagnostics)
            except Exception as e:
                logger.exception("refit_failed", error=str(e))
                # Keep using current coefficients; do not crash
```

Refit diagnostics logged at every refit: `R²` (in-sample and CV), `n_train_samples`, fitted coefficients, the derived `estimated_lag_seconds`, prediction MSE on the most recent 30 minutes (which is held out from training), and L2 norm of coefficient delta vs previous fit (stability metric — large deltas suggest regime shift).

### 6.4 Cold start

The regression needs ~1 hour of data minimum to fit anything meaningful, ~4 hours for stable coefficients. During this period, the bot abstains entirely. Phase 1 dry run divides into:

```
Hour 0-1:    All WS feeds live, history accumulating, σ warming up.
             Decisions table is being populated, but every row has
             event = 'abstain' with abstention_reason = 'model_warmup'.

Hour 1:      First regression fit attempted. If R² > 0.1, model goes live.
             Otherwise wait another hour and retry.

Hour 1-4:    Model is live but coefficients still settling. Bot operates
             normally. Trades are simulated/logged, no real money even in
             Phase 2 trial mode.

Hour 4-24:   Model has 4 hours of training data. Considered fully warmed up.
             Refits every 5 min on rolling 4-hour window.
```

For future runs (after we've collected one good 24-hour history), we can bootstrap from a pickled prior model. Phase 1 starts cold.

### 6.5 Computing q_settled

This is the core computation each decision tick uses.

```python
def compute_q_settled(model, current_features):
    """
    What WILL Polymarket be quoting once it has digested current spot?
    Substitute current microprice into every lag slot.
    """
    # Build a "settled" feature vector: every spot lookback uses x_now
    settled_features = current_features.copy()
    for lag_key in ['x_15', 'x_30', 'x_45', 'x_60', 'x_90', 'x_120']:
        settled_features[lag_key] = current_features['x_now']
    # Microstructure features stay as-is (they describe current conditions)

    logit_q_settled = model.predict_logit(settled_features)
    return sigmoid(logit_q_settled)


def compute_q_predicted(model, current_features):
    """
    What SHOULD Polymarket be quoting right now given lagged spot history?
    Sanity check: should be close to q_actual under healthy conditions.
    """
    return sigmoid(model.predict_logit(current_features))
```

`q_settled` is the headline output. `q_predicted` is a diagnostic — if it diverges substantially from `q_actual`, something is wrong (regime shift, stale model, feed problem) and we should be more cautious.

---

## 7. Sanity gates and circuit breakers around the regression

The regression is the entire model, so we need to be careful when to trust it.

| Condition                                          | Action                                  |
|----------------------------------------------------|-----------------------------------------|
| Model not yet fit                                  | Abstain. `abstention_reason='model_warmup'` |
| Model R² (rolling 30-min held-out) < 0.10          | Abstain. `abstention_reason='model_low_r2'` |
| `\|q_predicted − q_actual\| > 0.15` (model very wrong) | Abstain. `abstention_reason='model_disagrees_market'` |
| Coefficient delta vs previous fit > some threshold | Log + run normally (regime shift is real); refit window may need to shrink |
| Last successful refit > 15 minutes ago             | Abstain. `abstention_reason='model_stale'`  |

These gates collectively express: *we only trade when the model has been validated on recent data and currently agrees on Polymarket's level (even if we believe the level will move)*. A model that thinks Polymarket should be at 0.40 when it's actually at 0.72 isn't telling us about lag — it's telling us the model is broken or the regime has shifted.

---

## 8. Phase 1 — 24-hour dry run

### 8.1 Goals

1. Verify all four data feeds run cleanly for 24 hours under systemd on the Ubuntu VM.
2. Log every input the regression and the cash-out simulator could possibly need: spot, microprice, lagged microprices, K, q's, microstructure features, model predictions, fitted coefficients, hypothetical entry decisions, hypothetical exit triggers.
3. Resolve every window's outcome and write a clean `window_outcomes` table.
4. Validate the strategy thesis quantitatively via the regression's fit quality and the edge-realization slope.
5. Sweep exit-rule parameters and the ridge regularization to find the best operating point.

### 8.2 What is NOT done in Phase 1

- No order placement. No authenticated Polymarket calls. No real money at risk.
- No live tuning of the regression's structural parameters (lookback grid, feature set). The regression refits its coefficients automatically, but we don't change feature engineering during the run.
- No live exit decisions — exits are simulated at replay time against the logged future order book.

### 8.3 Pre-flight checks

- All four WS connections established and streaming for 30+ minutes.
- Microprice computation produces values that track midpoint in calm regimes (typical |microprice − mid| < 0.5 × spread) and diverge during bursts.
- At least one full 5-minute window has resolved cleanly with `K` captured at boundary and `close_price` captured at close.
- DuckDB query against the parquet directory returns expected row counts.
- systemd unit auto-restarts on simulated crash.
- Disk has ≥ 5GB free.

### 8.4 During the run

Manual SSH check-ins via `polybot-metrics summary --since=runstart` are safe — read-only DuckDB queries against parquet. After hour 4, also check `polybot-metrics model --latest` to inspect the live regression (coefficients, R², estimated_lag_seconds).

Resist tuning anything live. Fixed inputs over 24 hours is the entire point.

### 8.5 Post-run validation

Sanity checks before declaring the run successful:

1. **No data gaps.** Coinbase `sequence_num` strictly monotonic; heartbeats ≥ 1/s; Polymarket WS produced events per active window; Chainlink RTDS produced ≥ 1 tick per 60s.
2. **All windows resolved.** Expected ≈ 288 windows × 2 assets = 576. Allow 1-2% loss to startup/shutdown edges.
3. **Microprice rarely extreme** in calm regimes; excursions correlate with subsequent midpoint moves (validates microprice as forward predictor).
4. **Each window has ≈ 30 decision rows** (5 min / 10 s) with 1 K-capture event each.
5. **Model R² distribution.** Histogram CV-R² across all refits; median should be > 0.3 for healthy operation. If median < 0.1, Polymarket isn't actually predictable from spot the way the strategy assumes.
6. **Lag-stability.** `estimated_lag_seconds` should be relatively stable hour-to-hour, ideally in the 15-90s range. Wild swings (5s one hour, 200s the next) suggest the model is fitting noise, not signal.
7. **`q_predicted` tracks `q_actual` closely.** Plot q_predicted vs q_actual over the run; should hug the diagonal with low residual variance. This is the cleanest sanity check on the model.
8. **Model-disagreement abstention rate < 5%.** If the model frequently disagrees with the market (`|q_predicted - q_actual| > 0.15`), the model is mis-specified.

---

## 9. Logging and storage

### 9.1 Format and partitioning

Apache Parquet, zstd compression level 3, hive-partitioned:

```
logs/
  decisions/dt=2026-05-04/asset=btc/h=14.parquet
  decisions/dt=2026-05-04/asset=eth/h=14.parquet
  coinbase_ticks/dt=2026-05-04/asset=btc/h=14.parquet
  polymarket_book_snapshots/dt=2026-05-04/asset=btc/h=14.parquet
  polymarket_trades/dt=2026-05-04/asset=btc/h=14.parquet
  chainlink_ticks/dt=2026-05-04/asset=btc/h=14.parquet
  window_outcomes/dt=2026-05-04/asset=btc.parquet      (no h= partition)
  model_versions/dt=2026-05-04/asset=btc.parquet       (no h= partition)
```

One file per hour per asset. Use `pyarrow.parquet.ParquetWriter` opened once per hour, batch-flushed every 60 seconds.

Estimated total disk: ~50 MB/day compressed, both assets combined.

### 9.2 The `decisions` table — primary tuning source

One row per 10s scheduler tick.

| Column                      | Type    | Notes |
|-----------------------------|---------|-------|
| ts_ns                       | int64   | wall-clock receipt timestamp |
| asset                       | dict    | "btc"/"eth" |
| window_ts                   | int32   | window open unix s |
| tau_s                       | float32 | seconds until close |
| **Spot inputs**             |         |       |
| S_mid                       | float64 | Coinbase midpoint |
| microprice                  | float64 | imbalance-weighted fair value |
| spot_spread                 | float32 | Coinbase ask − bid |
| spot_bid_size_l1            | float32 | |
| spot_ask_size_l1            | float32 | |
| **Strike**                  |         |       |
| K                           | float64 | Chainlink window-open snapshot |
| K_uncertain                 | bool    | true if K was estimated, not observed |
| **Lagged spot features**    |         |       |
| x_now_logKratio             | float32 | log(microprice_t / K) |
| x_15_logKratio              | float32 | log(microprice_{t-15} / K) |
| x_30_logKratio              | float32 | |
| x_45_logKratio              | float32 | |
| x_60_logKratio              | float32 | |
| x_90_logKratio              | float32 | |
| x_120_logKratio             | float32 | |
| **Microstructure**          |         |       |
| ofi_l1                      | float32 | |
| ofi_l5_weighted             | float32 | |
| pm_book_imbalance           | float32 | |
| pm_trade_flow_30s           | float32 | |
| momentum_30s                | float32 | |
| momentum_60s                | float32 | |
| cross_asset_momentum_60s    | float32 | |
| **Diagnostic vol** (logged but not used) |    |       |
| sigma_per_sec_realized      | float32 | EWMA realized vol; logged for diagnostics only |
| **Polymarket book**         |         |       |
| q_up_bid                    | float32 | |
| q_up_ask                    | float32 | |
| q_up_mid                    | float32 | |
| q_down_bid                  | float32 | |
| q_down_ask                  | float32 | |
| q_down_mid                  | float32 | |
| poly_spread_up              | float32 | |
| poly_spread_down            | float32 | |
| poly_depth_up_l1            | float32 | top-of-book size for Up bid+ask |
| poly_depth_down_l1          | float32 | |
| **Model output**            |         |       |
| model_version_id            | string  | links to model_versions table |
| q_predicted                 | float32 | model's prediction of current q_actual (sanity) |
| q_settled                   | float32 | model's prediction of where q is heading |
| q_predicted_minus_q_actual  | float32 | sanity-check residual; should be small |
| **Edge (the headline number)** |       |       |
| edge_up_raw                 | float32 | q_settled − q_up_ask |
| edge_down_raw               | float32 | (1 − q_settled) − q_down_ask |
| fee_up_per_dollar           | float32 | THETA * q_up_ask * (1 - q_up_ask) |
| fee_down_per_dollar         | float32 | |
| slippage_up_per_dollar      | float32 | from depth model |
| slippage_down_per_dollar    | float32 | |
| edge_up_net                 | float32 | edge_up_raw − fee_up − slippage_up |
| edge_down_net               | float32 | edge_down_raw − fee_down − slippage_down |
| edge_signed                 | float32 | best signed edge (positive=Up favored, negative=Down) |
| edge_magnitude              | float32 | abs(edge_signed) — Kelly tier input |
| favored_side                | dict    | "up"/"down" |
| **Action this tick**        |         |       |
| event                       | dict    | "abstain" / "entry" / "hold" / "exit_lag_closed" / "exit_stopped" / "fallback_resolution" |
| chosen_side                 | dict    | "up"/"down"/null |
| tier                        | int8    | 0=abstain, 1..5 (only at entry) |
| would_bet_usd               | float32 | Kelly tier × wallet, 0 if not entering |
| bet_price                   | float32 | q_ask of chosen side at entry, null otherwise |
| bet_payout_contracts        | float32 | would_bet_usd / bet_price at entry |
| abstention_reason           | dict    | nullable; one of: edge_below_floor, model_warmup, model_low_r2, model_disagrees_market, model_stale, circuit_breaker_*, data_stale, sigma_uninitialized, wide_spread, already_engaged, k_uncertain |
| **Position state**          |         |       |
| has_open_position           | bool    | |
| position_side               | dict    | nullable |
| position_entry_tau          | float32 | nullable |
| position_entry_price        | float32 | nullable |
| position_edge_at_entry      | float32 | signed entry edge for stop-loss reference |
| **Feed staleness**          |         |       |
| coinbase_stale_ms           | int32   | |
| polymarket_stale_ms         | int32   | |
| chainlink_stale_ms          | int32   | |
| circuit_active              | dict    | nullable; circuit breaker name if any |

**Invariants:**

1. **One row per 10s tick, always.** Even when abstaining.
2. **`edge_magnitude` is always populated when the model is fit.** Sentinel `-99.0` if model is warming up or unfit.
3. **The `event` column is the canonical record.** Downstream queries filter on it.
4. **Position state is updated immediately after an entry/exit event.**

### 9.3 The `model_versions` table

One row per regression refit (every 5 minutes). Keyed by `model_version_id` for joins from `decisions`.

| Column                      | Type    | Notes |
|-----------------------------|---------|-------|
| ts_ns                       | int64   | refit completion time |
| asset                       | dict    | |
| model_version_id            | string  | UUID |
| n_train_samples             | int32   | rows used for training |
| training_window_start_ns    | int64   | |
| ridge_alpha                 | float32 | regularization param |
| r2_in_sample                | float32 | |
| r2_cv_mean                  | float32 | 5-fold time-series CV |
| r2_held_out_30min           | float32 | held-out validation slice |
| coef_alpha                  | float32 | intercept |
| coef_x_now                  | float32 | β₀ |
| coef_x_15                   | float32 | |
| coef_x_30                   | float32 | |
| coef_x_45                   | float32 | |
| coef_x_60                   | float32 | |
| coef_x_90                   | float32 | |
| coef_x_120                  | float32 | |
| coef_tau                    | float32 | |
| coef_inv_sqrt_tau           | float32 | |
| coef_ofi_l1                 | float32 | |
| coef_ofi_l5_weighted        | float32 | |
| coef_pm_book_imbalance      | float32 | |
| coef_pm_trade_flow_30s      | float32 | |
| coef_momentum_30s           | float32 | |
| coef_momentum_60s           | float32 | |
| coef_cross_asset_momentum_60s | float32 | |
| estimated_lag_seconds       | float32 | β-weighted average lag, see §3.4 |
| coef_delta_l2               | float32 | L2 norm of coef change vs previous fit |

The `model_versions` table is essential for backtest replay: the cash-out simulator and the parameter sweep both need to know which coefficients were live at any given decision tick.

### 9.4 The `window_outcomes` table

| Column                          | Type    | Notes |
|---------------------------------|---------|-------|
| asset                           | dict    | |
| window_ts                       | int32   | |
| close_ts                        | int32   | |
| K                               | float64 | |
| close_price                     | float64 | Chainlink at close |
| outcome                         | int8    | 1 if close ≥ K else 0 |
| n_decisions                     | int16   | |
| n_entries_attempted             | int16   | always 0 or 1 by policy |
| **Entry info (if entered)**     |         |       |
| entry_tier                      | int8    | |
| entry_tau_s                     | float32 | |
| entry_side                      | dict    | |
| entry_size_usd                  | float32 | |
| entry_price                     | float32 | |
| entry_edge_signed               | float32 | |
| entry_payout_contracts          | float32 | |
| entry_q_settled                 | float32 | model output at entry, for diagnostic |
| entry_model_version_id          | string  | |
| **Exit info (computed at replay)** | |  |
| exit_reason                     | dict    | "lag_closed"/"stopped_out"/"resolution"/null |
| exit_tau_s                      | float32 | |
| exit_price                      | float32 | bid we sold into (or 0/1 if held to resolution) |
| exit_holding_seconds            | float32 | |
| **P&L**                         |         |       |
| realized_pnl_usd                | float32 | per the cash-out P&L formula in §10 |
| realized_pnl_per_dollar         | float32 | |

In Phase 1, exit and P&L fields are populated **only at backtest/replay time**.

### 9.5 Other tables

- **`coinbase_ticks`**: raw L2 events and trades. Used to reconstruct microprice and lagged features offline at finer granularity than the live aggregation.
- **`polymarket_book_snapshots`**: every WS book/price_change event with full top-5 levels per side. **The cash-out simulator uses this for VWAP exit-fill modeling.**
- **`polymarket_trades`**: trade prints from PM, for trade flow features.
- **`chainlink_ticks`**: every Chainlink observation. Validates K capture and resolution prices.

---

## 10. Phase 1 backtest — tuning the strategy

The central analytic question of the dry run: **given the logged data, what choice of regression hyperparameters and exit thresholds produces the best P&L per unit risk, and is that P&L positive after all costs?**

### 10.1 Replay invariants

- Read decision rows in `ts_ns` order. Never join on a future row except via the explicit cash-out simulator.
- When evaluating a hypothetical strategy variant, use the model coefficients that were **live at that decision tick** (via `model_version_id`). Do not retroactively apply a better-trained model to old decisions; that's look-ahead bias.
- Use the `ts_ns` you assigned on receipt as the canonical clock. Do NOT use venue-supplied event times for ordering.

### 10.2 Cash-out simulator (`research/cashout_simulator.py`)

Same structure as before. For each entry, walk forward through `decisions` and `polymarket_book_snapshots` applying the exit rules.

```python
def simulate_exit(entry_row, decisions_after_entry, book_snapshots, outcome,
                  LAG_CLOSE=0.005, STOP=0.03, FALLBACK_TAU=10):
    edge_at_entry = entry_row.entry_edge_signed

    for next_dec in decisions_after_entry:
        # Strategy uses live edge_*_net columns from the logged decision —
        # those columns already incorporate the live model's q_settled.
        if entry_row.entry_side == 'up':
            edge_now = next_dec.edge_up_net
            exit_target_bid = next_dec.q_up_bid
        else:
            edge_now = next_dec.edge_down_net
            exit_target_bid = next_dec.q_down_bid

        # VWAP fill against logged book depth
        exit_price = vwap_fill_against_book(
            book_snapshots, next_dec.ts_ns,
            side=entry_row.entry_side,
            size_contracts=entry_row.entry_payout_contracts,
            target_bid=exit_target_bid,
        )

        if edge_now < LAG_CLOSE:
            return ('lag_closed', next_dec.tau_s, exit_price,
                    entry_row.entry_tau_s - next_dec.tau_s)

        if edge_now < edge_at_entry - STOP:
            return ('stopped_out', next_dec.tau_s, exit_price,
                    entry_row.entry_tau_s - next_dec.tau_s)

        if next_dec.tau_s < FALLBACK_TAU:
            terminal = 1.0 if (entry_row.entry_side == 'up') == (outcome == 1) else 0.0
            return ('resolution', 0, terminal, entry_row.entry_tau_s)

    terminal = 1.0 if (entry_row.entry_side == 'up') == (outcome == 1) else 0.0
    return ('resolution', 0, terminal, entry_row.entry_tau_s)


def realized_pnl(entry_row, exit_reason, exit_price, THETA=0.05):
    contracts = entry_row.entry_payout_contracts
    cost = entry_row.entry_size_usd
    entry_fee = THETA * entry_row.entry_price * (1 - entry_row.entry_price) * cost

    if exit_reason in ('lag_closed', 'stopped_out'):
        gross_proceeds = contracts * exit_price
        exit_fee = THETA * exit_price * (1 - exit_price) * gross_proceeds
        return gross_proceeds - cost - entry_fee - exit_fee
    else:  # 'resolution'
        if exit_price >= 1.0:
            return contracts - cost - entry_fee
        else:
            return -cost - entry_fee
```

### 10.3 Parameter sweep

Sweep three exit thresholds plus the regression's ridge regularization:

```
LAG_CLOSE_THRESHOLD ∈ {0.002, 0.003, 0.005, 0.008, 0.012, 0.020}
STOP_THRESHOLD      ∈ {0.020, 0.030, 0.050, 0.080, 0.120}
FALLBACK_TAU        ∈ {5, 10, 20, 30}
RIDGE_ALPHA         ∈ {0.01, 0.1, 1.0, 10.0}     # determines via offline-refit replay
```

The first three are exit-rule sweeps and use the live-fitted models in the logged data. The fourth requires re-fitting the regression offline at each ridge value and re-running the cashout simulator with the alternative coefficients. That's expensive but worth doing once.

For each combination, replay the entire 24h dataset and report total P&L, ROI, win rate, and Sharpe.

### 10.4 Model diagnostics

Before any strategy P&L is meaningful, validate the model itself:

```sql
-- Distribution of CV-R^2 across refits
SELECT
  asset,
  approx_quantile(r2_cv_mean, 0.1) AS p10_r2,
  approx_quantile(r2_cv_mean, 0.5) AS median_r2,
  approx_quantile(r2_cv_mean, 0.9) AS p90_r2
FROM model_versions
GROUP BY asset;
```

Healthy: median R² > 0.3, p10 R² > 0.15. If R² is consistently below 0.1, the strategy thesis isn't supported and no sweep will rescue it.

```sql
-- Stability of estimated_lag_seconds over the run
SELECT
  asset,
  date_trunc('hour', to_timestamp(ts_ns/1e9)) AS hr,
  AVG(estimated_lag_seconds) AS mean_lag,
  STDDEV(estimated_lag_seconds) AS sd_lag
FROM model_versions
GROUP BY asset, hr
ORDER BY asset, hr;
```

Healthy: mean lag in 15-90s range, sd_lag < 20s. Wild swings = the model is fitting noise.

```sql
-- q_predicted vs q_actual sanity
SELECT
  asset,
  AVG(ABS(q_predicted - q_up_ask)) AS mean_abs_residual,
  AVG((q_predicted - q_up_ask) * (q_predicted - q_up_ask)) AS mse
FROM decisions
WHERE event != 'abstain' OR abstention_reason IS NULL
GROUP BY asset;
```

Healthy: mean_abs_residual < 0.03. The model should be tracking Polymarket's level closely; the edge signal comes from `q_settled` (the forecast), not from `q_predicted` (the fit).

### 10.5 The three headline metrics

#### Metric 1 — Realized P&L

```sql
SELECT
  asset,
  date_trunc('day', to_timestamp(window_ts)) AS day,
  COUNT(*) FILTER (WHERE entry_tier > 0) AS n_entries,
  SUM(realized_pnl_usd) AS pnl_usd,
  SUM(entry_size_usd) AS gross_wagered,
  SUM(realized_pnl_usd) / NULLIF(SUM(entry_size_usd), 0) AS roi,
  SUM(realized_pnl_usd) / 1000.0 AS bankroll_return_pct,  -- $1000 starting wallet
  COUNT(*) FILTER (WHERE exit_reason = 'lag_closed') AS n_lag_closed,
  COUNT(*) FILTER (WHERE exit_reason = 'stopped_out') AS n_stopped,
  COUNT(*) FILTER (WHERE exit_reason = 'resolution') AS n_resolved,
  AVG(exit_holding_seconds) FILTER (WHERE entry_tier > 0) AS mean_hold_s,
  AVG(realized_pnl_per_dollar) FILTER (WHERE exit_reason = 'lag_closed') AS roi_lag_closed,
  AVG(realized_pnl_per_dollar) FILTER (WHERE exit_reason = 'stopped_out') AS roi_stopped,
  AVG(realized_pnl_per_dollar) FILTER (WHERE exit_reason = 'resolution') AS roi_resolved
FROM window_outcomes_with_pnl
WHERE params_id = :chosen_params
GROUP BY 1, 2;
```

A healthy strategy has 50%+ of exits as `lag_closed` (the thesis is actually working). Lots of `stopped_out` or `resolution` exits suggest the lag isn't real or our exit thresholds are mis-tuned.

#### Metric 2 — Engagement rate

```sql
SELECT
  asset,
  CASE
    WHEN tau_s > 240 THEN 'T-300..T-241'
    WHEN tau_s > 60  THEN 'T-240..T-61'
    WHEN tau_s > 10  THEN 'T-60..T-11'
    ELSE 'T-10..T-0'
  END AS tau_bucket,
  COUNT(*) AS n_decisions,
  COUNT(*) FILTER (WHERE event = 'entry') AS n_entries,
  COUNT(*) FILTER (WHERE event = 'abstain') AS n_abstains,
  AVG(CASE WHEN event = 'entry' THEN 1.0 ELSE 0.0 END) AS entry_rate,
  AVG(edge_magnitude) FILTER (WHERE edge_magnitude > -50) AS mean_edge_observed,
  COUNT(*) FILTER (WHERE abstention_reason = 'edge_below_floor') * 1.0
    / NULLIF(COUNT(*) FILTER (WHERE event='abstain'), 0) AS pct_below_floor,
  COUNT(*) FILTER (WHERE abstention_reason = 'model_warmup') * 1.0 / COUNT(*) AS pct_warmup,
  COUNT(*) FILTER (WHERE abstention_reason = 'model_low_r2') * 1.0 / COUNT(*) AS pct_low_r2
FROM decisions
WHERE asset = 'btc'
GROUP BY 1, 2
ORDER BY 1, 2;
```

Healthy: 1-10% per-cycle entry rate (excluding warmup). The warmup hours should show 100% abstentions with reason `model_warmup`.

#### Metric 3 — Average observed edge across all polls

```sql
SELECT
  asset,
  COUNT(*) AS n_decisions,
  AVG(edge_magnitude) AS mean_abs_edge,
  AVG(edge_signed) AS mean_signed_edge,                  -- ≈0 if unbiased
  STDDEV(edge_signed) AS sd_signed_edge,
  approx_quantile(edge_magnitude, 0.5) AS p50,
  approx_quantile(edge_magnitude, 0.9) AS p90,
  approx_quantile(edge_magnitude, 0.99) AS p99,
  AVG(edge_up_net) AS mean_edge_up,
  AVG(edge_down_net) AS mean_edge_down
FROM decisions
WHERE asset = 'btc'
  AND coinbase_stale_ms < 5000
  AND polymarket_stale_ms < 60000
  AND edge_magnitude > -50
GROUP BY asset;
```

Healthy values:
- `mean_abs_edge` between 0.005 and 0.025.
- `|mean_signed_edge|` < 0.005 (no systematic directional bias).
- `p99 / mean_abs_edge` ratio of 5-15 (heavy right tail is the strategy's natural shape).

### 10.6 The decisive diagnostic — edge realization slope

```sql
WITH bets AS (
  SELECT
    entry_edge_signed AS edge_at_entry,
    realized_pnl_per_dollar AS realized
  FROM window_outcomes_with_pnl
  WHERE entry_tier > 0
    AND params_id = :chosen_params
)
SELECT
  width_bucket(edge_at_entry, 0.0, 0.30, 12) AS bucket,
  COUNT(*) AS n,
  AVG(edge_at_entry) AS mean_edge,
  AVG(realized) AS mean_realized,
  STDDEV(realized) AS sd_realized
FROM bets
GROUP BY 1
ORDER BY 1;
```

Plot `mean_edge` (x) vs `mean_realized` (y). Linear positive slope is the strategy's signature. Slope of 1.0 means we capture all reported edge as profit (impossible after costs); slope of 0.5 means costs eat half of the raw signal but the strategy is profitable. Slope ≤ 0 means the model has zero predictive value — do not trade.

### 10.7 Decision criteria for advancing to Phase 2

Advance to Phase 2 (small live-money trial) if and only if all of the following hold:

1. **Total P&L positive** with the best parameter combination from §10.3.
2. **Median CV-R²** for the regression > 0.25 across all refits.
3. **`estimated_lag_seconds`** is stable (sd across hourly means < 25s) and sits in the 15-120s range.
4. **`q_predicted` mean absolute residual** < 0.03 — the model tracks Polymarket's level reliably.
5. **Edge realization slope** statistically distinguishable from zero with p < 0.10.
6. **`|mean_signed_edge| < 0.008`** — no systematic directional bias.
7. **At least 50% of exits are `lag_closed`** — the thesis actually works.
8. **Hit rate > 50% in tiers 3-5**.
9. **No more than 5% of windows had `K_uncertain=true`**.
10. **WS reconnect events < 10** over the run.

Do not advance if:
- Best parameters lose money on 24h.
- Median R² < 0.10 (the strategy thesis is wrong; no parameter sweep can rescue it).
- Edge realization slope ≤ 0.
- Lag-close exit rate < 30%.
- σ-realized estimates clustered at unreasonable values (median annualized > 200%) — feed quality problem.
- Polymarket WS silent-freeze recurred more than 3× despite watchdog.

If results are mixed — slope is positive but small, P&L is near zero after costs — that's diagnostic information. Iterate on the same logged data: try richer features, different lookback grids, smarter slippage estimation. Do not advance to live trading until criteria 1-10 are satisfied.

---

## 11. Operations on a headless Ubuntu VM

### 11.1 systemd unit

`/etc/systemd/system/polybot.service`:

```ini
[Unit]
Description=Polymarket Crypto Lag-Arb Bot (dry-run)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=polybot
WorkingDirectory=/opt/polybot
EnvironmentFile=/etc/polybot/secrets.env
ExecStart=/opt/polybot/.venv/bin/python -m polybot.main
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=polybot
MemoryMax=2G
TasksMax=256
KillSignal=SIGTERM
TimeoutStopSec=30
Environment=POLYBOT_KILL_SWITCH=/run/polybot/STOP

[Install]
WantedBy=multi-user.target
```

### 11.2 Logging

structlog → stdout → journald. JSON renderer when not a tty. journald rotation:

```
SystemMaxUse=2G
SystemMaxFileSize=200M
MaxRetentionSec=14day
```

Parquet rotation by filename (one file per hour per asset).

### 11.3 SSH-friendly tooling

- **`journalctl -u polybot -f --output=cat`** — live tail
- **`journalctl -u polybot -f --output=json | jq 'select(.level=="error")'`** — errors only
- **`duckdb -c "SELECT * FROM 'logs/decisions/dt=*/asset=btc/h=*.parquet' LIMIT 10"`** — quick inspect
- **`pq head /path/to/file.parquet`** — Rust pqrs CLI
- **`polybot-metrics summary --asset btc --since=2026-05-04`** — wraps §10.5 queries
- **`polybot-metrics model --latest`** — current model coefficients, R², estimated_lag_seconds
- **`polybot-metrics model --since=runstart`** — model stability over time
- **`polybot-metrics sweep --top=10`** — best parameter combinations from `parameter_sweep.py`
- **`polybot-metrics edge-slope --params=:chosen`** — §10.6 diagnostic
- **`polybot-ctl status | pause | resume`** — Unix domain socket control

### 11.4 Kill switches

- `touch /run/polybot/STOP` — graceful exit on next decision cycle
- `systemctl stop polybot` — SIGTERM, 30s grace
- Never `kill -9` — corrupts in-flight Parquet writes

### 11.5 Secrets

`/etc/polybot/secrets.env`, mode 0600, owned by polybot user. structlog processors must redact any key matching `KEY|SECRET|PASSPHRASE|PRIVATE`.

---

## 12. Risk and circuit breakers

| Trigger                                       | Action                                  |
|-----------------------------------------------|-----------------------------------------|
| Coinbase WS no message in 5s                  | abstain new entries; reconnect; (Phase 2) force-exit any open position |
| Polymarket WS no event in 60s                 | abstain; force reconnect; if reconnect fails twice, force-exit via REST |
| Chainlink RTDS no tick in 90s                 | abstain                                 |
| Coinbase sequence_num gap                     | drop book, fresh snapshot, abstain 30s  |
| Polymarket spread > 0.10                      | abstain (round-trip too costly)         |
| Tick-size change                              | log + revalidate active orders          |
| **Model warmup not complete**                 | abstain                                 |
| **Model CV-R² < 0.10 on most recent fit**     | abstain                                 |
| **\|q_predicted − q_actual\| > 0.15**         | abstain (model disagrees with market)   |
| **Last refit > 15 min ago**                   | abstain (model stale)                   |
| Daily realized loss > 5% of starting wallet   | HARD STOP, page operator (Phase 2)      |
| ≥ 1 open position in this window already      | refuse new entry                        |
| ≥ 2 open positions across assets              | refuse new entry                        |
| Wall clock vs Coinbase server time > 2s       | abstain                                 |
| `/run/polybot/STOP` exists                    | graceful shutdown                       |
| K_uncertain on current window                 | abstain entire window                   |

A lag-arb position whose data feed has gone stale is the worst case — we don't know if the lag has closed or widened. In Phase 2, always force-exit on persistent feed staleness rather than wait.

---

## 13. Phase 2 preview (not in scope yet)

After dry-run analysis identifies best parameters:

1. Authenticate Polymarket (L2 HMAC + proxy wallet sig type 1).
2. Fund wallet with $50-100 USDC; one-time approvals via `approvals.py`.
3. Run with the best parameter combination, *only* at the most conservative tier (delta ≥ 0.30, 10% wallet) for 1 week.
4. Bootstrap the regression from the dry-run final coefficients rather than cold-starting.
5. Compare realized fills against backtest expectations.
6. Implement maker-exit mode (post limit order at target exit price for fee rebate) and A/B test against taker-exit.
7. Scale bankroll only if realized P&L is within ±25% of backtest expectations.

Phase 2 is its own document and its own pull request.

---

## 14. Caveats and open questions

- **CLOB v2 / pUSD migration** is recent (early 2026). py-clob-client-v2 has the new EIP-712 v2 signature scheme. Phase 1 read-only is unaffected; Phase 2 must use v2.
- **Geographic restriction:** Polymarket's global CLOB restricts US persons. Phase 1 read-only is fine; Phase 2 from US must use Polymarket US (separate API at `api.polymarket.us`). Confirm jurisdictional compliance before live trading.
- **5-minute markets are newer than 15-minute markets.** Fee coefficient Θ=0.05 is borrowed by analogy from 15m docs and confirmed only after first live trades.
- **Polymarket WS silent-freeze (#292)** is real and unresolved; watchdog is mandatory.
- **The "Polymarket lags Coinbase by 30-90s" assumption** is anecdotal. The dry-run regression validates or invalidates this directly. If the fitted lag is consistently small (< 5s), the strategy is unprofitable and we don't advance.
- **Microprice as a forward predictor** has well-known half-life of ~1-10 seconds in equity markets. Crypto half-life on Coinbase is unconfirmed in literature for our purposes; the fitted regression's coefficient on `x_now` vs lagged x's effectively measures it.
- **Adverse selection at lag-close** — when the lag closes, every other lag-arb bot is also trying to exit. Exit fills can be materially worse than mid-bid. The dry run's VWAP-against-logged-book simulation captures part of this but not the *competitive* dynamic; real Phase 2 fills may be worse.
- **The 24h sample is small for parameter selection.** ~290 entries × 120+ parameter combinations is overfitting territory. Mitigations: (a) prefer parameters in stable plateaus rather than sharp peaks of the P&L surface, (b) treat Phase 2's first week as the real out-of-sample test, (c) extend the dry run to 7 days if 24h results are ambiguous.
- **Regression refit cost.** Ridge regression on ~14k samples × ~15 features fits in well under a second; refit every 5 minutes is trivially cheap. If the feature set grows substantially (e.g., adding tree-based models or many polynomial features), revisit the refit cadence.
- **Look-ahead bias in the offline ridge_alpha sweep.** Re-fitting the regression with different α at replay time and applying it retroactively is technically anachronistic — it uses information unavailable at the live decision tick. Mitigation: when sweeping α, use only data strictly prior to each replayed tick (rolling-origin cross-validation), not the full-run dataset. This is more expensive but honest.

---

## 15. Prior art to learn from

- **Archetapp gist `7680adabc48f812a561ca79d73cbac69`** — confirms slug format, RTDS Chainlink as strike source. Their bot is hold-to-resolution late-window entry, not lag arbitrage; we're doing something different.
- **KaustubhPatange / `polymarket-trade-engine`** — TS reference for lifecycle state machine, market discovery, fill tracking. Architecturally useful even though strategy differs.
- **NautilusTrader Polymarket integration** — production-grade Rust+Python framework with clean instrument loading and signature handling.
- **Polymarket/agent-skills (`websocket.md`)** — authoritative WS protocol reference.
- **Polymarket/real-time-data-client** — TS reference SDK for RTDS.
- **Cont-Kukanov-Stoikov 2014** — foundational paper on Order Flow Imbalance. Required reading for the OFI feature in §6.2.
- **Stoikov 2018 "The Micro-Price"** — foundational paper on microprice as a forward predictor of midpoint. Required reading for understanding why microprice is the right input to the regression.
- **Hayashi-Yoshida 2005** — non-synchronous covariance estimator for tick data from two venues. Useful for offline lag analysis as a sanity check on what the regression's β coefficients are telling us.

Read these before reinventing anything. The OFI math, microprice theory, and WS protocols are all solved problems; what we're building on top is a *lead-lag regression with adaptive refitting and cash-out exit logic* that none of the published Polymarket bots implement.
