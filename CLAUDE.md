# CLAUDE.md — Polymarket 5-Minute Crypto Up/Down Trading Bot

> **Status:** Planning / pre-implementation. This document is the source of truth for the project's design, architecture, and reasoning. Read it end-to-end before generating any code.

---

## 1. Project goal

Build an automated trading bot that takes positions in Polymarket's 5-minute "Bitcoin Up or Down" and "Ethereum Up or Down" binary prediction markets. The bot generates an independent probability estimate `p` of the Yes (Up) outcome and bets whenever the spread between `p` and the Polymarket implied probability `q` exceeds a tier threshold, sized by a tiered Kelly fraction of bankroll.

**Two strategies are first-class concerns** and must both be supported by the live data collection and evaluated head-to-head in the 24-hour dry-run backtest:

- **Strategy A — Hold to resolution.** Bet when edge appears, hold until window resolution at T+0. Edge thesis: model probability is more accurate than market probability for the close-time outcome.
- **Strategy B — Cash-out scalping.** Bet when Polymarket lags spot, sell back into the market when Polymarket catches up. Edge thesis: Polymarket order book takes seconds-to-tens-of-seconds to reprice after Coinbase moves; that lag is exploitable round-trip.

Phase 1 is read-only. We collect 24 hours of tick-level data, run both strategies as hypotheticals against the logged data, and compare which performed better before committing real capital.

**Hard constraint:** Every trade decision is reducible to "model probability `p`, market price `q`, edge `delta = p − q` net of fees and slippage clears tier T's threshold, bet wallet fraction prescribed by tier T." This is true under both strategies; the strategies differ only in *exit* logic and in *which `p` is computed* (resolution probability for A, near-term lag-arb fair value for B).

**Non-goals:**

- We are not building a market-making bot. No quoting both sides.
- We are not chasing sub-second latency arbitrage. Polygon-validator-adjacent bots already own that niche.
- We are not building anything that requires placing or cancelling orders faster than ~1 second end-to-end.

---

## 2. The market mechanic

Every 5 minutes, Polymarket opens a new market on each of `BTC`, `ETH`. The market asks: "Will the asset's price at the close of this 5-minute window be ≥ the price at the open?"

- **Strike `K`** = Chainlink BTC/USD oracle price snapshot at the window-open boundary (`t mod 300 == 0`).
- **Resolution price** = Chainlink BTC/USD oracle price snapshot at the window-close boundary.
- Yes shares pay $1 if close ≥ open, else $0. No shares are the complement.
- Yes-share price `q ∈ [0, 1]` *is* the implied probability of Up.
- Window slug is deterministic: `{asset}-updown-5m-{window_open_unix_ts}`.

**Critical:** The strike `K` is the **Chainlink oracle's first observation at or after the window boundary**, not Coinbase or Binance spot at that instant. Subscribe to Polymarket's RTDS WebSocket `crypto_prices_chainlink` channel filtered to `btc/usd` (or `eth/usd`) and capture the first tick at or after each window-open timestamp. This is non-negotiable — wrong K destroys probability calculations near window close.

**Cash-out is supported.** A Yes-share position can be exited at any time before window close by selling into the order book at the prevailing bid. This is what makes Strategy B viable.

---

## 3. Two strategies, one decision loop

The decision loop runs every 10 seconds. Both strategies share the same data inputs and the same edge-detection scaffolding. They diverge only in (a) which probability is being computed and (b) what the exit policy is.

### 3.1 Strategy A — Hold to resolution

- **Probability `p_A`:** the probability that the asset closes above K at window resolution, given current spot, time remaining, and volatility. Computed via Black-Scholes binary on midpoint spot.
- **Entry condition:** edge `|p_A − q| > tier_floor(τ)` net of fees and slippage. One bet per window — once we have a position in this window, we don't add to it.
- **Exit:** hold to resolution. P&L = (1 − bet_price) on win, −bet_price on loss, per dollar wagered.
- **Best at:** small τ (high signal-to-noise on the close-time outcome); markets where mispricings persist to resolution.
- **Failure mode:** at large τ, the model probability is fuzzy (GBM is wrong about crypto tails) and the bet is held through 4 minutes of unmodeled tail risk.

### 3.2 Strategy B — Cash-out scalping (lag arbitrage)

- **Probability `p_B`:** the probability that an idealized arbitrageur would assign right now if they had instant access to Coinbase's full microstructure. Computed via Black-Scholes binary on **microprice** (imbalance-weighted fair value) instead of midpoint, which captures impending midpoint movement that hasn't reached the trade tape yet. At small τ, microprice → spot price → near-deterministic outcome.
- **Entry condition:** edge `|p_B − q| > tier_floor(τ)` net of fees and slippage. The thesis is that Polymarket will catch up to where Coinbase already is. One position per window per asset; do not flip sides.
- **Exit:** **dynamic, not held to resolution.** Three exit triggers, whichever fires first:
  1. **Lag closed:** Polymarket has caught up. Specifically, `|p_B − q| < lag_close_threshold` (e.g., 0.005). Sell back into the bid for realized round-trip profit.
  2. **Lag widened against us / thesis broken:** spot reversed and `p_B` has moved decisively *against* our position, by more than `stop_threshold` (e.g., 0.03). Cut the loss.
  3. **Resolution fallback:** if neither (1) nor (2) fires before τ < 10s, default to hold-to-resolution (the Strategy A exit). At small τ the lag-arb interpretation merges with the resolution interpretation anyway.
- **Best at:** any τ where Polymarket lag is observable and round-trip costs (2× spread + 2× fee + 2× slippage) are smaller than raw edge.
- **Failure mode:** every leg pays spread and fees, so the per-trip cost floor is ~2-6% of bet size; lag must exceed this to be net positive. Adverse selection at the moment of "lag closed" — many bots try to exit simultaneously, exit fills can be worse than entry.

### 3.3 Decision loop (per 10s tick, both strategies in parallel)

```
1. Is there an active 5m market? If no, idle.
2. Compute time_remaining τ = window_close - now.
3. Read inputs (always):
     - S_mid (Coinbase midpoint)
     - microprice (Coinbase imbalance-weighted price)
     - K (captured at window open from Chainlink RTDS)
     - σ (per-second EWMA on midpoint log returns)
     - Polymarket book: q_up_bid, q_up_ask, q_down_bid, q_down_ask
4. Compute probabilities:
     - p_A = black_scholes_binary(S_mid, K, σ, τ)
     - p_B = black_scholes_binary(microprice, K, σ, τ)
5. Compute edges (per side, per strategy):
     - edge_A_up = p_A - q_up_ask - fee(q_up_ask) - slip
     - edge_A_down = (1 - p_A) - q_down_ask - fee(q_down_ask) - slip
     - edge_B_up = p_B - q_up_ask - fee(q_up_ask) - slip
     - edge_B_down = (1 - p_B) - q_down_ask - fee(q_down_ask) - slip
6. Apply microstructure adjustments (as logit-space additions to p_A and p_B):
     OFI on Coinbase, Polymarket book imbalance, spot-PM lead-lag residual,
     cross-asset BTC↔ETH momentum, multi-horizon momentum (5s/15s/30s).
7. Resolve action per strategy independently:
     For each strategy S in {A, B}:
       If we already have an open position from S in this window:
         (Strategy A) abstain on entry side; do nothing.
         (Strategy B) re-evaluate exit conditions (3.2), close if triggered.
       Else if max(edge_S_up, edge_S_down) > tier_floor(τ):
         Mark as would-bet, compute size from Kelly tier table.
       Else:
         Abstain with reason.
8. Persist decision row (one per 10s tick, with both strategies' would-be actions).
9. (Phase 2 only) submit live orders for the strategy chosen as "live" in config.
```

In Phase 1 we never submit orders. Both strategies are simulated continuously and logged side-by-side in the same row, so backtest replay can compare them on identical data.

### 3.4 Tier floor by τ

Edge thresholds vary by time-to-close. Full Kelly tier table from the user's spec:

```python
KELLY_TIERS = [
    (0.30, 0.10),    # delta >= 0.30 -> 10% wallet
    (0.15, 0.08),    # delta >= 0.15 -> 8%
    (0.08, 0.05),    # delta >= 0.08 -> 5%
    (0.04, 0.03),    # delta >= 0.04 -> 3%
    (0.02, 0.015),   # delta >= 0.02 -> 1.5%
]
```

τ-conditional floor on which tiers are eligible:

| τ window         | Min eligible delta | Rationale                                        |
|------------------|--------------------|--------------------------------------------------|
| `τ > 240s`       | 0.15 (tier 4-5)    | Information sparse; only take large, clear edges |
| `60s < τ ≤ 240s` | 0.04 (tier 2-5)    | Sweet spot for both strategies                   |
| `10s ≤ τ ≤ 60s`  | 0.02 (tier 1-5)    | Last entry window; take what's there             |
| `τ < 10s`        | 0.04 (tier 2-5)    | Latency-arb only; no time for thesis to play out |

This is a starting heuristic, tuned in the dry-run replay phase.

### 3.5 The processes own state, not decisions

WebSocket handlers update state (spot book, Polymarket book, Chainlink window). A separate scheduler tick reads state and runs the 9-step decision loop. This separation is important because it makes the system trivially backtestable — the scheduler can be driven by replayed timestamps rather than wall-clock time during replay.

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
    history.py               # Rolling buffers for returns, OFI, features
  /models
    volatility.py            # EWMA + realized vol estimators
    base_estimator.py        # Black-Scholes binary p_A and p_B
    edge.py                  # Microstructure signal computation
    fusion.py                # Logit-space combination -> final p
    fees.py                  # Fee curve fee(price) for net-edge calculation
  /strategy
    decision.py              # The 9-step loop
    kelly.py                 # Tier table, sizing logic
    exit.py                  # Cash-out exit logic for Strategy B
    risk.py                  # Daily loss cap, max open positions, circuit breakers
  /execution
    orders.py                # Order placement, retries, slippage estimation (Phase 2)
    fills.py                 # Position tracking, P&L attribution
  /infra
    scheduler.py             # Main 10s tick loop
    config.py                # All tunables in one place
    logger.py                # Structured tick log -> Parquet for backtest replay
    secrets.py               # API keys, wallet config (env-loaded)
  /research
    backtest.py              # Replay logged ticks, score model variants
    cashout_simulator.py     # Computes Strategy B P&L from logged future books
    fit_fusion.py            # Logistic regression for fusion weights
    calibration.py           # Reliability diagrams, Platt/isotonic recalibration
    policy_compare.py        # Side-by-side metric report for A vs B
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
Polymarket WS ──update──► poly_book ┼──► scheduler (10s tick) ──► decision.py ──► (Phase 2) orders.py
                                    │              │
RTDS Chainlink ──tick──► window ────┘              ▼
                                              parquet writer
```

### 4.3 Process model

Single Python 3.11+ process, asyncio, `asyncio.TaskGroup`. Long-running tasks: Coinbase WS, Polymarket CLOB WS, Polymarket RTDS WS, scheduler, parquet writer. Five tasks total, all under one event loop.

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
  - `microprice` = (best_bid × ask_size + best_ask × bid_size) / (bid_size + ask_size)
  - `top_5_bid_levels[]`, `top_5_ask_levels[]` (price + size)
  - `last_trade_price`, `last_trade_size`
  - 1-second OHLC ring buffer over last 300 seconds
  - 1-second log return ring buffer (from midpoint), feeds EWMA σ
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

For Strategy B, fees are paid on **both legs**. Total round-trip cost when entering at price `a` and exiting at price `e`:

```
total_cost = fee(a) + fee(e) + spread_at_exit + slippage_in + slippage_out
```

Net edge required to break even on Strategy B is therefore much higher than for Strategy A. The actual Θ for 5-min markets is borrowed from 15-min docs and must be confirmed in Phase 2 from real fills.

---

## 6. Probability models

### 6.1 Black-Scholes binary (shared base)

For the probability that GBM with per-second volatility σ ends above K at horizon τ given current price S:

```
d2 = (ln(S/K) - 0.5 * σ² * τ) / (σ * √τ)
p = Φ(d2)                          # standard normal CDF
```

We use **r = 0** (drift over 5 minutes is dominated by noise).

**Edge cases:**
- `τ ≤ 1.0`: clamp τ to 1.0 in the formula. If `S > K + ε`, return 1.0; if `S < K − ε`, return 0.0.
- `|d2| > 6`: saturate to 0.0 / 1.0.
- σ uninitialized (warmup): mark abstain with `sigma_uninitialized`.

**Implementation:** `scipy.stats.norm.cdf` directly; do not roll your own erf.

**Volatility:** EWMA on **midpoint** 1-second log returns (not last-trade — bid-ask bounce inflates raw σ by 5-20%). λ = 0.94 baseline (RiskMetrics; ~11s half-life). Range λ ∈ {0.90, 0.94, 0.97} swept in dry-run analysis.

**Warmup:** abstain for first 60-120s after startup until ≥ 60 observations of returns. Optionally seed from REST candle history.

### 6.2 Two p's, one formula

The same formula computes both strategy probabilities; only the input price differs:

```python
p_A = bs_binary(S=spot_book.mid,        K=window.K, sigma=σ, tau=τ)
p_B = bs_binary(S=spot_book.microprice, K=window.K, sigma=σ, tau=τ)
```

Microprice for Strategy B because it's the imbalance-weighted forward fair value — captures impending movement that hasn't reached the trade tape. At small τ, microprice and midpoint converge; at large τ, microprice is a stronger predictor of the next several seconds of midpoint movement.

For Phase 1 keep them separate so we can attribute edge to either source.

### 6.3 Microstructure adjustments (logit-space)

Each signal is z-scored against its own rolling 1-hour distribution, then summed in logit space:

```
logit(p_adjusted) = logit(p_strategy) + Σᵢ wᵢ * z_score(signalᵢ)
```

Signals:

| Signal                        | Source                    | Initial weight |
|-------------------------------|---------------------------|----------------|
| Spot OFI (levels 1-5 weighted)| Coinbase L2 events        | 0.40           |
| Spot momentum 5s              | Coinbase mid log return   | 0.10           |
| Spot momentum 30s             | Coinbase mid log return   | 0.20           |
| Polymarket book imbalance     | Polymarket WS book        | 0.15           |
| Polymarket trade flow 30s     | Polymarket WS trades      | 0.20           |
| Lead-lag residual             | Rolling regression        | 0.30           |
| Cross-asset OFI (other coin)  | Coinbase                  | 0.10           |

Initial weights are placeholders. Fit by `research/fit_fusion.py` once dry-run data is collected.

### 6.4 Order Flow Imbalance (Cont-Kukanov-Stoikov)

Per book event, signed flow:

```
e_n = I(P_b_n ≥ P_b_{n-1}) * q_b_n
    - I(P_b_n ≤ P_b_{n-1}) * q_b_{n-1}
    - I(P_a_n ≤ P_a_{n-1}) * q_a_n
    + I(P_a_n ≥ P_a_{n-1}) * q_a_{n-1}
```

Aggregate over 1-second bins. Multi-level extension: compute per level m=1..5, sum with weights `[1, 0.5, 0.25, 0.125, 0.0625]`. Z-score against rolling 60-120s window. Decay forward at half-life ~5-15s.

### 6.5 Lead-lag residual

Rolling regression of `Δlogit(q_polymarket_t)` on `Δlog(spot_t-k)` for k ∈ {0, 5, 10, 15, 30, 60}; residual at k* = argmax correlation is the signal. For Phase 1, log raw streams; compute the residual offline first to find typical k*, then move the computation online.

---

## 7. Exit logic for Strategy B

Exit logic only matters for Strategy B (A is hold-to-resolution, trivial).

```python
# Called every 10s for each open Strategy B position
def evaluate_exit_B(position, current_state):
    # position: side ('up'|'down'), entry_price, entry_tau, size, p_B_at_entry
    # current_state: p_B_now, q_bid_now (matching side), τ_now

    # 1. Lag closed?
    if position.side == 'up':
        edge_now = current_state.p_B_now - current_state.q_up_ask_now
        edge_at_entry = position.p_B_at_entry - position.entry_price
    else:  # 'down'
        edge_now = (1 - current_state.p_B_now) - current_state.q_down_ask_now
        edge_at_entry = (1 - position.p_B_at_entry) - position.entry_price

    if edge_now < LAG_CLOSE_THRESHOLD:        # e.g. 0.005
        return ('exit', 'lag_closed', current_state.q_bid_now)

    # 2. Thesis broken?
    if edge_now < edge_at_entry - STOP_THRESHOLD:  # e.g. 0.03 erosion
        return ('exit', 'stopped_out', current_state.q_bid_now)

    # 3. Resolution fallback
    if current_state.tau_now < 10:
        return ('hold_to_resolution', None, None)

    return ('hold', None, None)
```

The thresholds `LAG_CLOSE_THRESHOLD` and `STOP_THRESHOLD` are sweep parameters in the dry-run replay. Reasonable starting values: 0.005 and 0.03 respectively.

**Important:** Strategy B never holds two positions on opposite sides of the same window. If we're long Up and the model flips to favor Down, we exit Up; we do *not* simultaneously buy Down (that would be paying spread + fees to lock in a loss).

---

## 8. Phase 1 — 24-hour dry run

### 8.1 Goals

1. Verify all four data feeds run cleanly for 24 hours under systemd on the Ubuntu VM.
2. Log every input that any tuning sweep would need: spot, microprice, K, σ, p_A, p_B, q's, microstructure features, hypothetical decisions for both strategies.
3. Resolve every window's outcome and write a clean `window_outcomes` table.
4. Have enough information in the logs that any of the policy variants in §10 can be evaluated *offline* without re-running the bot live.

### 8.2 What is NOT done in Phase 1

- No order placement. No authenticated Polymarket calls. No real money at risk.
- No live tuning. Weights, thresholds, tier floors are fixed at startup from `config.py`.
- No live policy comparison — that happens at replay time on the logged data.

### 8.3 Pre-flight checks before starting the 24h run

- All four WS connections established and streaming for 30+ minutes.
- σ has finished warmup and produces values in [30%, 120%] annualized for BTC.
- At least one full 5-minute window has resolved cleanly with `K` captured at boundary and `close_price` captured at close.
- DuckDB query against the parquet directory returns expected row counts.
- systemd unit auto-restarts on simulated crash.
- Disk has ≥ 5GB free (24h run produces ~50MB; the buffer is for log overflows and journald).

### 8.4 During the run

Manual SSH check-ins via `polybot-metrics summary --asset btc --since=runstart` should be safe — they're read-only DuckDB queries against the parquet directory and don't touch the bot. Resist the urge to tune anything live; fixed inputs over 24 hours is the entire point.

### 8.5 Post-run validation

Sanity checks before declaring the run successful and proceeding to backtest analysis:

1. **No data gaps.** Coinbase `sequence_num` strictly monotonic; heartbeats ≥ 1/s; Polymarket WS produced events per active window; Chainlink RTDS produced ≥ 1 tick per 60s.
2. **All windows resolved.** Expected ≈ 288 windows × 2 assets = 576. Allow 1-2% loss to startup/shutdown edges.
3. **σ in plausible range.** Histogram of σ_per_sec → annualized; tail values > 200% indicate microstructure-noise blow-ups.
4. **p_A, p_B distributions reasonable.** Both should span [0.05, 0.95] over the run; if collapsed near 0.5, σ is too high; if bimodal at 0/1, τ=0 saturation is happening too aggressively.
5. **q's correlate with p_A.** Bin decisions by p_A in 10 deciles; mean q in each decile should track diagonally (efficient market sanity).
6. **Microprice rarely far from midpoint** in calm regimes (typical |microprice − mid| < 0.5 × spread); excursions are real signal.
7. **Each window has ≈ 30 decision rows** (5 min / 10 s) with 1 K-capture event each.

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
  window_outcomes/dt=2026-05-04/asset=btc.parquet  (no h= partition; small)
```

One file per hour per asset. Keeps individual files at 5-50 MB, easily DuckDB-queryable, atomic to rewrite. Use `pyarrow.parquet.ParquetWriter` opened once per hour, batch-flushed every 60 seconds (do NOT write per-row).

Estimated total disk: ~50 MB/day compressed (~250 MB uncompressed), both assets combined.

### 9.2 The `decisions` table — primary tuning source

One row per 10s scheduler tick, written even when abstaining. Both strategies' would-be decisions in the same row.

| Column                      | Type    | Notes |
|-----------------------------|---------|-------|
| ts_ns                       | int64   | wall-clock receipt timestamp |
| asset                       | dict    | "btc"/"eth" |
| window_ts                   | int32   | window open unix s |
| tau_s                       | float32 | seconds until close |
| **Spot inputs**             |         |       |
| S_mid                       | float64 | Coinbase midpoint |
| S_last                      | float64 | last trade price |
| microprice                  | float64 | imbalance-weighted fair value |
| spot_spread                 | float32 | Coinbase ask − bid |
| spot_bid_size_l1            | float32 | for OFI / microprice diagnostics |
| spot_ask_size_l1            | float32 | |
| **Strike**                  |         |       |
| K                           | float64 | Chainlink window-open snapshot |
| K_chainlink_ts_ms           | int64   | oracle observation time |
| K_uncertain                 | bool    | true if K was estimated, not observed |
| **Volatility**              |         |       |
| sigma_per_sec               | float32 | EWMA σ |
| sigma_initialized           | bool    | false during warmup |
| **Probability — Strategy A**|         |       |
| p_A                         | float32 | bs_binary on midpoint |
| p_A_adjusted                | float32 | post-fusion, post-microstructure |
| **Probability — Strategy B**|         |       |
| p_B                         | float32 | bs_binary on microprice |
| p_B_adjusted                | float32 | post-fusion, post-microstructure |
| **Polymarket book**         |         |       |
| q_up_bid                    | float32 | best bid for Up |
| q_up_ask                    | float32 | best ask for Up |
| q_up_mid                    | float32 | mid of Up |
| q_down_bid                  | float32 | best bid for Down |
| q_down_ask                  | float32 | best ask for Down |
| q_down_mid                  | float32 | |
| poly_spread_up              | float32 | q_up_ask − q_up_bid |
| poly_spread_down            | float32 | |
| poly_depth_up_l1            | float32 | top-of-book size for Up bid+ask |
| poly_depth_down_l1          | float32 | |
| **Edges (per strategy, per side, net of fees+slippage)** | | |
| edge_A_up_net               | float32 | p_A_adjusted − q_up_ask − fee − slip |
| edge_A_down_net             | float32 | (1−p_A_adjusted) − q_down_ask − fee − slip |
| edge_B_up_net               | float32 | p_B_adjusted − q_up_ask − fee − slip |
| edge_B_down_net             | float32 | (1−p_B_adjusted) − q_down_ask − fee − slip |
| edge_A_best_signed          | float32 | signed: edge_A_up_net if Up favored, −edge_A_down_net otherwise |
| edge_A_best_abs             | float32 | max(edge_A_up_net, edge_A_down_net) |
| edge_B_best_signed          | float32 | signed for Strategy B |
| edge_B_best_abs             | float32 | max for Strategy B |
| **Microstructure features** |         |       |
| ofi_l1                      | float32 | level-1 OFI 1s window |
| ofi_l5_weighted             | float32 | levels 1-5 with decay |
| pm_book_imbalance           | float32 | (depth_up − depth_down)/total |
| leadlag_resid               | float32 | spot-PM lag residual z-scored |
| momentum_5s                 | float32 | log(S_mid_t / S_mid_{t-5}) |
| momentum_15s                | float32 | |
| momentum_30s                | float32 | |
| cross_asset_momentum_30s    | float32 | other asset's momentum_30s |
| **Strategy A action**       |         |       |
| A_chosen_side               | dict    | "up"/"down"/"abstain" |
| A_tier                      | int8    | 0=abstain, 1..5 |
| A_would_bet_usd             | float32 | Kelly tier × wallet, 0 if abstain |
| A_bet_price                 | float32 | q_ask of chosen side, null if abstain |
| A_bet_payout_contracts      | float32 | A_would_bet_usd / A_bet_price |
| A_abstention_reason         | dict    | nullable; see §3.3 |
| **Strategy B action**       |         |       |
| B_chosen_side               | dict    | "up"/"down"/"abstain" |
| B_tier                      | int8    | |
| B_would_bet_usd             | float32 | |
| B_bet_price                 | float32 | |
| B_bet_payout_contracts      | float32 | |
| B_abstention_reason         | dict    | |
| **State flags**             |         |       |
| coinbase_stale_ms           | int32   | ms since last Coinbase event |
| polymarket_stale_ms         | int32   | ms since last Polymarket event |
| chainlink_stale_ms          | int32   | ms since last Chainlink tick |
| circuit_active              | dict    | nullable; circuit breaker name if any |

**Invariants the writer must enforce:**

1. **One row per 10s tick, even when abstaining.** If we don't write, engagement-rate metrics are uncomputable.
2. **Edge columns are populated even when abstaining.** When inputs are unhealthy (data stale, σ uninitialized), write sentinel `-99.0` so it's distinguishable from a real near-zero edge. The 0.0 / null distinction matters for filtering at metric time.
3. **Both strategies' actions are independently logged.** A row may have `A_chosen_side='up'` and `B_chosen_side='abstain'` simultaneously — that's expected and informative.

### 9.3 The `window_outcomes` table

Written when a window resolves (close-time Chainlink tick observed). One row per window per asset.

| Column                          | Type    | Notes |
|---------------------------------|---------|-------|
| asset                           | dict    | |
| window_ts                       | int32   | open |
| close_ts                        | int32   | open + 300 |
| K                               | float64 | strike |
| close_price                     | float64 | Chainlink at close |
| outcome                         | int8    | 1 if close ≥ K else 0 |
| n_decisions                     | int16   | rows in `decisions` for this window |
| **Strategy A first-bet info**   |         |       |
| A_n_would_bet                   | int16   | non-abstain count |
| A_first_tier                    | int8    | tier of first non-abstain (0 if none) |
| A_first_tau_s                   | float32 | τ at first bet |
| A_first_side                    | dict    | "up"/"down"/null |
| A_first_size_usd                | float32 | |
| A_first_price                   | float32 | bet_price at entry |
| A_first_payout_contracts        | float32 | |
| **Strategy B first-bet info**   |         |       |
| B_n_would_bet                   | int16   | |
| B_first_tier                    | int8    | |
| B_first_tau_s                   | float32 | |
| B_first_side                    | dict    | |
| B_first_size_usd                | float32 | |
| B_first_price                   | float32 | |
| B_first_payout_contracts        | float32 | |
| **Strategy B exit info (computed at replay)** |  |  |
| B_exit_reason                   | dict    | "lag_closed"/"stopped_out"/"resolution"/null |
| B_exit_tau_s                    | float32 | τ at exit |
| B_exit_price                    | float32 | bid we sold into (or 0/1 if held to resolution) |
| B_exit_holding_seconds          | float32 | exit_tau − entry_tau (negative since τ counts down) |

The exit fields are computed at backtest time (§10.2), not live.

### 9.4 Other tables

- **`coinbase_ticks`**: raw L2 events and trades. Used to reconstruct OFI offline at finer granularity than the live aggregation, and to validate microprice computation.
- **`polymarket_book_snapshots`**: every WS book/price_change event with full top-5 levels per side. **This is the table the cash-out simulator uses to look up future exit prices.**
- **`polymarket_trades`**: trade prints from PM, for trade flow signals.
- **`chainlink_ticks`**: every Chainlink observation. Validates K capture and resolution prices.

---

## 10. Phase 1 backtest — comparing the two strategies

This is the central analytic question of the dry run: **which strategy made more money over the 24 hours, and with what risk profile?**

### 10.1 Replay invariants

- Read decision rows in `ts_ns` order. Never join on a future row except via the explicit cash-out simulator (§10.2).
- Use the `ts_ns` you assigned on receipt as the canonical clock; do NOT use venue-supplied event times for ordering (they're trade-engine times, not arrival times).
- For Strategy A P&L, only `window_outcomes.outcome` and the `A_first_*` columns are needed.
- For Strategy B P&L, you need `B_first_*` PLUS look-ahead access to `polymarket_book_snapshots` for exit price simulation.

### 10.2 Strategy B cash-out simulator (`research/cashout_simulator.py`)

This is the single hardest piece of replay logic. For each B entry, simulate exit by stepping forward through the order book log and applying the exit rules from §7.

```python
def simulate_B_exit(entry_row, decisions_after_entry, book_snapshots,
                    LAG_CLOSE=0.005, STOP=0.03):
    """
    entry_row: a `decisions` row where B_chosen_side != 'abstain'
    decisions_after_entry: subsequent decisions in the same window for this asset
    book_snapshots: polymarket book ticks within the window, after entry
    Returns (exit_reason, exit_tau, exit_price, holding_seconds)
    """
    edge_at_entry = entry_row.edge_B_best_signed  # signed by chosen side

    for next_dec in decisions_after_entry:
        # Recompute current edge on the position's side
        if entry_row.B_chosen_side == 'up':
            edge_now = next_dec.edge_B_up_net
            exit_bid = next_dec.q_up_bid
        else:
            edge_now = next_dec.edge_B_down_net
            exit_bid = next_dec.q_down_bid

        # Lag closed
        if edge_now < LAG_CLOSE:
            return ('lag_closed', next_dec.tau_s, exit_bid,
                    entry_row.tau_s - next_dec.tau_s)

        # Stopped out
        if edge_now < edge_at_entry - STOP:
            return ('stopped_out', next_dec.tau_s, exit_bid,
                    entry_row.tau_s - next_dec.tau_s)

        # Time-based fallback at small τ
        if next_dec.tau_s < 10:
            # Hold to resolution; exit price = outcome
            outcome = window_outcomes.outcome
            terminal = 1.0 if (entry_row.B_chosen_side == 'up') == (outcome == 1) else 0.0
            return ('resolution', 0, terminal, entry_row.tau_s)

    # Window ended without explicit exit — treat as resolution
    outcome = window_outcomes.outcome
    terminal = 1.0 if (entry_row.B_chosen_side == 'up') == (outcome == 1) else 0.0
    return ('resolution', 0, terminal, entry_row.tau_s)


def realized_pnl_B(entry_row, exit_reason, exit_price):
    """
    P&L per dollar wagered, accounting for fees on both legs.
    Entry: paid `entry_row.B_bet_price` per contract, `B_would_bet_usd` total
    Exit: received `exit_price` per contract
    Fees: pay taker fee on entry; if exit is a taker (lag_closed/stopped_out) pay again.
          If exit is resolution, no exit fee (Polymarket auto-redeems).
    """
    contracts = entry_row.B_bet_payout_contracts
    gross_proceeds = contracts * exit_price
    cost = entry_row.B_would_bet_usd
    entry_fee = THETA * entry_row.B_bet_price * (1 - entry_row.B_bet_price) * cost

    if exit_reason in ('lag_closed', 'stopped_out'):
        exit_fee = THETA * exit_price * (1 - exit_price) * gross_proceeds
    else:  # resolution
        exit_fee = 0.0

    return gross_proceeds - cost - entry_fee - exit_fee
```

The simulator is deterministic given the logged book history and the threshold parameters. Sweep `LAG_CLOSE ∈ {0.003, 0.005, 0.01, 0.02}` and `STOP ∈ {0.02, 0.03, 0.05, 0.10}` and plot the P&L surface — somewhere on that surface is Strategy B's best-case operating point.

**Slippage realism:** at exit, the bid you can hit is constrained by depth. If your position is larger than the top-of-book bid size, you walk down: average exit price = VWAP across the next levels until size is satisfied. Apply this in the simulator using the depth columns logged in `polymarket_book_snapshots`.

### 10.3 Strategy A P&L (trivial)

```python
def realized_pnl_A(window_row):
    if window_row.A_first_tier == 0:
        return 0.0
    won = (window_row.A_first_side == 'up' and window_row.outcome == 1) or \
          (window_row.A_first_side == 'down' and window_row.outcome == 0)
    contracts = window_row.A_first_payout_contracts
    cost = window_row.A_first_size_usd
    entry_fee = THETA * window_row.A_first_price * (1 - window_row.A_first_price) * cost
    if won:
        return contracts - cost - entry_fee   # contracts redeem at $1
    else:
        return -cost - entry_fee
```

### 10.4 Policy comparison report (`research/policy_compare.py`)

Run all three of your headline metrics for each strategy variant, side-by-side. Output is a Rich table on the terminal and a parquet table for further analysis.

Variants to compare:

| Variant ID                | Strategy | Exit policy                                  |
|---------------------------|----------|----------------------------------------------|
| `A_hold`                  | A        | hold to resolution                           |
| `B_dynamic_005_03`        | B        | LAG_CLOSE=0.005, STOP=0.03                   |
| `B_dynamic_010_05`        | B        | LAG_CLOSE=0.010, STOP=0.05                   |
| `B_fixed_30s`             | B        | exit at exactly entry_tau − 30 (force taker) |
| `B_fixed_60s`             | B        | exit at exactly entry_tau − 60               |
| `B_fixed_120s`            | B        | exit at entry_tau − 120                      |
| `A_or_B_first_to_fire`    | mixed    | take whichever strategy fires first per window |
| `A_and_B_independent`     | mixed    | both run, separate bankrolls (40%/60% split) |

For each variant, compute and report:

- **Realized P&L** (USD, total over 24h)
- **ROI** (P&L / total wagered)
- **Bankroll return** (P&L / starting bankroll)
- **Win rate**
- **Average win, average loss, ratio**
- **Sharpe-like per-bet ratio** = mean / sd × √n
- **Max drawdown** of cumulative P&L
- **Engagement rate per cycle** (% of decision rows that fired)
- **Engagement rate per window** (% of windows with ≥ 1 fired bet)
- **Average observed edge per cycle** (mean of `edge_X_best_abs` for that strategy)
- **Average observed edge at entry** (mean of `edge_X_best_signed` on rows where action was taken)
- **Edge realization slope:** for fired bets, regress realized return against `edge_at_entry`. Slope ≈ 1.0 = honest model; slope < 0.3 = edge is mostly noise; slope ≤ 0 = the model has zero predictive value.
- **Per-tier breakdown:** all of the above bucketed by tier 1..5.
- **Per-τ-bucket breakdown:** by entry-time bucket (T-300..T-241, T-240..T-61, T-60..T-11, T-10..T-0).
- **Strategy B specific:** distribution of `B_exit_reason`, average holding seconds, lag-close hit rate (% of B trades that exited via "lag_closed" vs. stopped out vs. fell through to resolution).

### 10.5 The three headline metrics, formal SQL

These are computed once per variant in `policy_compare.py` and printed to terminal. All queries run against the Parquet directory via DuckDB. Examples below are for Strategy A; substitute `A_*` → `B_*` and join the cash-out simulator output for Strategy B.

#### Metric 1 — Realized P&L

```sql
-- Strategy A daily P&L
SELECT
  asset,
  date_trunc('day', to_timestamp(window_ts)) AS day,
  COUNT(*) FILTER (WHERE A_first_tier > 0) AS n_bets,
  SUM(realized_pnl_A) AS pnl_usd,
  SUM(A_first_size_usd) AS gross_wagered,
  SUM(realized_pnl_A) / NULLIF(SUM(A_first_size_usd), 0) AS roi,
  SUM(realized_pnl_A) / 1000.0 AS bankroll_return_pct  -- starting bankroll = $1000
FROM window_outcomes_with_pnl
GROUP BY 1, 2
ORDER BY 1, 2;
```

For Strategy B, the equivalent query joins on the cash-out simulator's output table (one row per B entry with `B_realized_pnl`, `B_exit_reason`, `B_exit_tau_s`).

```sql
-- Strategy B P&L (cash-out simulator output already joined)
SELECT
  asset,
  date_trunc('day', to_timestamp(window_ts)) AS day,
  COUNT(*) FILTER (WHERE B_first_tier > 0) AS n_bets,
  SUM(realized_pnl_B) AS pnl_usd,
  SUM(B_first_size_usd) AS gross_wagered,
  -- breakdown by exit reason
  COUNT(*) FILTER (WHERE B_exit_reason = 'lag_closed') AS n_lag_closed,
  COUNT(*) FILTER (WHERE B_exit_reason = 'stopped_out') AS n_stopped,
  COUNT(*) FILTER (WHERE B_exit_reason = 'resolution') AS n_resolved,
  AVG(B_exit_holding_seconds) FILTER (WHERE B_first_tier > 0) AS mean_hold_s
FROM window_outcomes_with_pnl
GROUP BY 1, 2;
```

#### Metric 2 — Engagement rate

The *per-cycle* rate is the headline. Compute both per-cycle and per-window so we can separate "model fires often" from "policy fires often" (one bet per window collapses many fires into one).

```sql
-- Strategy A per-cycle engagement, by τ bucket
SELECT
  asset,
  CASE
    WHEN tau_s > 240 THEN 'T-300..T-241'
    WHEN tau_s > 60  THEN 'T-240..T-61'
    WHEN tau_s > 10  THEN 'T-60..T-11'
    ELSE 'T-10..T-0'
  END AS tau_bucket,
  COUNT(*) AS n_decisions,
  COUNT(*) FILTER (WHERE A_chosen_side != 'abstain') AS n_engaged_A,
  COUNT(*) FILTER (WHERE B_chosen_side != 'abstain') AS n_engaged_B,
  AVG(CASE WHEN A_chosen_side != 'abstain' THEN 1.0 ELSE 0.0 END) AS engagement_rate_A,
  AVG(CASE WHEN B_chosen_side != 'abstain' THEN 1.0 ELSE 0.0 END) AS engagement_rate_B,
  AVG(edge_A_best_abs) FILTER (WHERE edge_A_best_abs > -50) AS mean_edge_A,
  AVG(edge_B_best_abs) FILTER (WHERE edge_B_best_abs > -50) AS mean_edge_B
FROM decisions
WHERE asset = 'btc'
GROUP BY 1, 2
ORDER BY 1, 2;
```

Healthy ranges: 0.5%–8% per-cycle engagement, depending on tier floors. Below 0.5% means circuit breakers or warmup are dominating; above 10% means the floor is too low or edges are systematically overstated. Strategy B's per-cycle engagement is expected to be 2-5× higher than Strategy A's because microprice signals at large τ that midpoint doesn't.

#### Metric 3 — Average observed edge across all polls

The unconditional distribution of model-vs-market disagreement, regardless of whether we acted.

```sql
SELECT
  asset,
  COUNT(*) AS n_decisions,
  -- Strategy A
  AVG(edge_A_best_abs) AS mean_abs_edge_A,
  AVG(edge_A_best_signed) AS mean_signed_edge_A,    -- ≈0 if unbiased
  STDDEV(edge_A_best_signed) AS sd_signed_edge_A,
  approx_quantile(edge_A_best_abs, 0.5) AS p50_abs_edge_A,
  approx_quantile(edge_A_best_abs, 0.9) AS p90_abs_edge_A,
  approx_quantile(edge_A_best_abs, 0.99) AS p99_abs_edge_A,
  -- Strategy B
  AVG(edge_B_best_abs) AS mean_abs_edge_B,
  AVG(edge_B_best_signed) AS mean_signed_edge_B,
  STDDEV(edge_B_best_signed) AS sd_signed_edge_B,
  approx_quantile(edge_B_best_abs, 0.5) AS p50_abs_edge_B,
  approx_quantile(edge_B_best_abs, 0.99) AS p99_abs_edge_B
FROM decisions
WHERE asset = 'btc'
  AND coinbase_stale_ms < 5000
  AND polymarket_stale_ms < 60000
  AND sigma_initialized
GROUP BY asset;
```

Healthy values:
- `mean_abs_edge` between 0.005 and 0.025 (50bp to 2.5%); larger means likely miscalibration
- `|mean_signed_edge|` < 0.005 (no systematic directional bias)
- `mean_abs_edge_B ≥ mean_abs_edge_A` is expected — microprice is a more aggressive predictor

### 10.6 The decisive diagnostic — edge realization slope

This is the single most important plot. For each variant, regress realized P&L per dollar wagered against the edge observed at entry.

```sql
-- Strategy A
WITH bets AS (
  SELECT
    edge_A_best_signed AS edge_at_entry,
    realized_pnl_A / A_first_size_usd AS realized_per_dollar
  FROM window_outcomes_with_pnl
  WHERE A_first_tier > 0
)
SELECT
  width_bucket(edge_at_entry, -0.10, 0.30, 16) AS bucket,
  COUNT(*) AS n,
  AVG(edge_at_entry) AS mean_edge,
  AVG(realized_per_dollar) AS mean_realized,
  STDDEV(realized_per_dollar) AS sd_realized
FROM bets
GROUP BY 1
ORDER BY 1;
```

Plot mean_edge (x) vs mean_realized (y). The relationship should be **linear with slope ≈ 1**: a 5% reported edge should translate to a 5% realized return per dollar wagered. Slope < 0.3: most reported edge is illusory. Slope ≤ 0: model is anti-predictive — do not trade live under any circumstances.

The same diagnostic for Strategy B uses `edge_B_best_signed` and the cash-out P&L. Comparing slopes across variants is the cleanest signal of which strategy is actually picking up real signal vs. noise.

### 10.7 Decision criteria for advancing to Phase 2

Advance to Phase 2 (small live-money trial) if and only if:

1. **At least one variant** has positive total P&L over 24h with hit rate > 50% in tiers ≥ 3.
2. The winning variant's edge realization slope (§10.6) is statistically distinguishable from zero with p < 0.10 (note: 24h sample is small; this is suggestive, not conclusive).
3. `|mean_signed_edge|` < 0.008 for the winning strategy (no systematic directional bias).
4. No more than 5% of windows had `K_uncertain=true` (Chainlink capture is reliable).
5. WS reconnect events < 10 over the run.

Do not advance if:
- Best variant lost money.
- Edge realization slope ≤ 0 for all variants.
- σ estimates clustered at unreasonable values (median annualized > 200%) — units bug.
- Polymarket WS silent-freeze recurred more than 3× despite watchdog.

If none of the variants are clearly profitable but Strategy B's slope is distinguishable from zero and Strategy A's isn't, that is itself useful information: it means the lag-arbitrage thesis is real but not yet large enough to clear costs. The next iteration should focus on reducing per-trip cost (smaller bets, only-tier-5 entries, maker orders for exit).

---

## 11. Operations on a headless Ubuntu VM

### 11.1 systemd unit

`/etc/systemd/system/polybot.service`:

```ini
[Unit]
Description=Polymarket Crypto Bot (dry-run)
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

structlog → stdout → journald. JSON renderer when not a tty. Rotate journald via `/etc/systemd/journald.conf`:

```
SystemMaxUse=2G
SystemMaxFileSize=200M
MaxRetentionSec=14day
```

Parquet rotation via filename (one file per hour per asset).

### 11.3 SSH-friendly tooling

- **`journalctl -u polybot -f --output=cat`** — live tail
- **`journalctl -u polybot -f --output=json | jq 'select(.level=="error")'`** — errors only
- **`duckdb -c "SELECT * FROM 'logs/decisions/dt=2026-05-04/asset=btc/h=*.parquet' LIMIT 10"`** — quick inspect
- **`pq head /path/to/file.parquet`** — Rust pqrs CLI for quick file inspection
- **`polybot-metrics summary --asset btc --since=2026-05-04`** — wraps DuckDB queries from §10
- **`polybot-metrics policy-compare`** — runs the variant table from §10.4
- **`polybot-ctl status | pause | resume`** — Unix domain socket control

### 11.4 Kill switches

- `touch /run/polybot/STOP` — bot exits gracefully on next decision cycle
- `systemctl stop polybot` — SIGTERM, 30s grace
- Never `kill -9` — corrupts in-flight Parquet writes

### 11.5 Secrets

`/etc/polybot/secrets.env`, mode 0600, owned by polybot user. `KEY=VALUE` per line, no quoting. structlog processors must redact any key containing `KEY|SECRET|PASSPHRASE|PRIVATE` on output. Never commit, never log.

---

## 12. Risk and circuit breakers

In Phase 1 these are abstention triggers; in Phase 2 they additionally prevent order placement.

| Trigger                                    | Action                                  |
|--------------------------------------------|-----------------------------------------|
| Coinbase WS no message in 5s               | abstain; reconnect after 10s            |
| Polymarket WS no event in 60s              | abstain; force reconnect                |
| Chainlink RTDS no tick in 90s              | abstain; reconnect                      |
| Coinbase sequence_num gap                  | drop book, fresh snapshot, abstain 30s  |
| σ > 5× rolling median (regime shift)       | abstain 60s                             |
| Polymarket spread > 0.10                   | abstain                                 |
| Tick-size change                           | log + revalidate active orders          |
| Daily realized loss > 5% of starting wallet| HARD STOP, page operator (Phase 2)      |
| ≥ 1 open position in window already        | refuse new (per strategy)               |
| ≥ 2 open positions across assets           | refuse new                              |
| Wall clock vs Coinbase server time > 2s    | abstain (boundary trust broken)         |
| `/run/polybot/STOP` exists                 | graceful shutdown                       |
| K_uncertain on current window              | abstain entire window                   |

---

## 13. Phase 2 preview (not in scope yet)

After dry-run analysis selects a winning variant:

1. Authenticate Polymarket (L2 HMAC + proxy wallet sig type 1).
2. Fund wallet with $50-100 USDC; one-time approvals via `approvals.py`.
3. Run *only* the winning variant, *only* at the most conservative tier (delta ≥ 0.30, 10% wallet) for 1 week.
4. Compare realized fills against backtest expectations (slippage, fee model accuracy, exit fill realism for Strategy B).
5. Scale bankroll only if realized P&L is within ±25% of backtest expectations.

Phase 2 is its own document and its own pull request.

---

## 14. Caveats and open questions

- **CLOB v2 / pUSD migration** is recent (early 2026). py-clob-client-v2 has the new EIP-712 v2 signature scheme. Phase 1 read-only is unaffected; Phase 2 must use v2.
- **Geographic restriction:** Polymarket's global CLOB restricts US persons. Phase 1 read-only is fine; Phase 2 from US must use Polymarket US (separate API at `api.polymarket.us`). Confirm jurisdictional compliance before live trading.
- **5-minute markets are newer than 15-minute markets.** Fee coefficient Θ=0.05 is borrowed by analogy from 15m docs and confirmed empirically only after first live trades.
- **Polymarket WS silent-freeze (#292)** is real and unresolved; watchdog is mandatory.
- **The "Polymarket lags Coinbase by 30-90s" figure** is anecdotal. Your dry-run cross-correlation analysis is the dispositive measurement — and Strategy B's profitability depends on it being real.
- **EWMA λ=0.94 baseline** is from RiskMetrics (1994). Crypto-1-second-optimal λ may differ; sweep in dry run.
- **Strategy B's edge depends on round-trip costs being smaller than raw lag.** Fee curve, spread distribution, and exit-side liquidity are all measurable in Phase 1 — if the math doesn't work, the dry run will show it without losing any real money.
- **Microprice as a forward predictor** has well-known half-life of ~1-10 seconds in equity markets. Crypto half-life on Coinbase is unconfirmed in the literature for our purposes; the dry run measures it directly via `cross_corr(microprice_t, mid_{t+k})`.

---

## 15. Prior art to learn from

- **Archetapp gist `7680adabc48f812a561ca79d73cbac69`** — confirms slug format, RTDS Chainlink as strike source.
- **KaustubhPatange / `polymarket-trade-engine`** — TS reference for lifecycle state machine, market discovery, fill tracking.
- **joicodev / `polymarket-bot`** — JS implementation of exactly our Black-Scholes + EWMA + logit-fusion stack; port the math.
- **NautilusTrader Polymarket integration** — production-grade Rust+Python framework with clean instrument loading and signature handling.
- **Polymarket/agent-skills (`websocket.md`)** — authoritative WS protocol reference.
- **Polymarket/real-time-data-client** — TS reference SDK for RTDS.

Read these before reinventing anything. The math, the lifecycle, and the WS protocols are all solved problems; what we're adding is the *side-by-side evaluation of two distinct exit policies on the same logged data*, which none of them do.
