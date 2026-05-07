"""
scheduler.py — Core orchestrator for the Kalshi lead-lag arbitrage bot.

Three concurrent loops:
  1. Sampler (1s): build FeatureVec → append to TrainingBuffer
  2. Refitter (5 min): pull TrainingBuffer → refit KalshiRegressionModel
  3. Decision (10s): compute q_settled/edge, apply Kelly tiers, log decisions

Also manages 15-min window transitions: pre-discovers the next ticker when
tau < 2 min, switches the Kalshi feed to the new ticker on rollover.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp

from betbot.kalshi.book import SpotBook, KalshiBook
from betbot.kalshi.config import (
    DECISION_INTERVAL_S, FALLBACK_TAU_S,
    KALSHI_REST, KALSHI_SERIES, KELLY_TIERS,
    LAG_CLOSE_THRESHOLD, MIN_TRAIN_SAMPLES,
    REFIT_INTERVAL_S, SAMPLE_INTERVAL_S,
    STOP_THRESHOLD, THETA_FEE_TAKER, TRAINING_WINDOW_S,
    TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    ENTRY_MODE, MAX_HOLD_S, MIN_ENTRY_INTERVAL_S,
)
from betbot.kalshi.features import FeatureVec, _logit, _sigmoid, build_features
from betbot.kalshi.orders import place_order as kalshi_place_order
from betbot.kalshi.model import KalshiRegressionModel, ModelDiagnostics
from betbot.kalshi.tick_logger import TickLogger
from betbot.kalshi.training_buffer import TrainingBuffer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fee / slippage helpers
# ---------------------------------------------------------------------------

def _fee(p: float) -> float:
    """Taker fee per dollar bet at price p. Always applied to the exit leg."""
    return THETA_FEE_TAKER * p * (1.0 - p)


def _entry_fee(p: float) -> float:
    """Entry-leg fee per dollar bet. Zero in maker mode (we post passively
    at the existing best bid and pay no fee when filled). When ENTRY_MODE
    is taker, the entry leg pays the same taker fee as the exit."""
    if ENTRY_MODE == "maker":
        return 0.0
    return _fee(p)


def _slippage(book_depth: float, size_usd: float) -> float:
    if book_depth <= 0:
        return 0.02
    return min(0.02, size_usd / (book_depth + 1e-9) * 0.01)


def _kelly_size(edge: float, wallet: float) -> tuple[float, int]:
    """Return (bet_usd, tier_index 1-5).  Returns (0, 0) if below floor."""
    for tier_idx, (floor, frac) in enumerate(KELLY_TIERS, start=1):
        if edge >= floor:
            return wallet * frac, tier_idx
    return 0.0, 0


# ---------------------------------------------------------------------------
# Decision log row (printed + optionally written to JSONL)
# ---------------------------------------------------------------------------

@dataclass
class DecisionRow:
    ts_ns:          int
    tau_s:          float
    yes_bid:        float
    yes_ask:        float
    yes_mid:        float
    q_predicted:    Optional[float]
    q_settled:      Optional[float]
    edge_up_raw:    Optional[float]
    edge_up_net:    Optional[float]
    edge_magnitude: float
    favored_side:   Optional[str]
    event:          str          # abstain / entry / hold / exit_lag_closed / exit_stopped / fallback_resolution
    abstention_reason: Optional[str]
    tier:           int
    would_bet_usd:  float
    has_open:       bool
    model_r2_hld:   float
    model_lag_s:    float
    window_ticker:  str


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------

@dataclass
class Position:
    side:            str    # "yes" (buy yes) or "no" (buy no)
    entry_price:     float
    entry_tau_s:     float
    entry_edge:      float  # signed edge at entry
    size_usd:        float
    contracts:       float


# ---------------------------------------------------------------------------
# Window discovery (REST)
# ---------------------------------------------------------------------------

async def _list_active_markets(series: str = KALSHI_SERIES) -> list[dict]:
    """Return all active markets in the series, sorted by close_time then floor_strike."""
    import datetime as _dt
    url    = f"{KALSHI_REST}/trade-api/v2/markets"
    params = {"series_ticker": series, "status": "open", "limit": 50}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params,
                                   timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    log.warning("list_markets HTTP %s", r.status)
                    return []
                data = await r.json()

        active = []
        for mkt in data.get("markets", []):
            ct     = mkt.get("close_time", "")
            status = mkt.get("status", "")
            if ct and status in ("open", "active"):
                try:
                    epoch = _dt.datetime.fromisoformat(
                        ct.replace("Z", "+00:00")).timestamp()
                    mkt["_close_epoch"] = epoch
                    active.append(mkt)
                except Exception:
                    pass
        active.sort(key=lambda m: (m["_close_epoch"], float(m.get("floor_strike") or 0.0)))
        return active
    except Exception as e:
        log.warning("list_markets error: %s", e)
        return []


async def _discover_market(series: str = KALSHI_SERIES,
                           prefer_strike: Optional[float] = None) -> Optional[dict]:
    """
    Return the full market dict for the active market to track.

    Among all soonest-closing markets, pick the one whose floor_strike is closest
    to prefer_strike (typically current BTC spot) — this is the at-the-money
    market with the most lag potential. If prefer_strike is None, picks the
    median strike of the soonest-closing batch.
    """
    active = await _list_active_markets(series)
    if not active:
        log.warning("discover_market: no active markets in series %s", series)
        return None

    # Group by close time — there may be multiple strike levels closing together
    earliest = active[0]["_close_epoch"]
    same_window = [m for m in active if abs(m["_close_epoch"] - earliest) < 1.0]

    if prefer_strike is not None and prefer_strike > 0:
        same_window.sort(key=lambda m: abs(float(m.get("floor_strike") or 0.0) - prefer_strike))
        return same_window[0]

    # No spot reference — pick the median strike (likely closest to ATM)
    same_window.sort(key=lambda m: float(m.get("floor_strike") or 0.0))
    return same_window[len(same_window) // 2]


async def _discover_ticker(series: str = KALSHI_SERIES) -> Optional[str]:
    """Return only the ticker string of the soonest-closing active market."""
    mkt = await _discover_market(series)
    return mkt["ticker"] if mkt else None


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """
    Wires SpotBook + KalshiBook + KalshiRegressionModel + TrainingBuffer
    into three concurrent async loops.

    Usage:
        sched = Scheduler(spot_book, kb_book, kalshi_feed)
        await sched.run()   # runs until cancelled
    """

    def __init__(self, cb: SpotBook, kb: KalshiBook,
                 kalshi_feed,          # KalshiRestFeed — must expose update_ticker()
                 wallet_usd: float = 1000.0,
                 log_path: Optional[Path] = None,
                 tick_path: Optional[Path] = None,
                 seed_tick_path: Optional[Path] = None,  # bootstrap from this file (defaults to tick_path)
                 pk=None,                       # RSA private key for order signing
                 live_orders: bool = False,     # if True: place real Kalshi orders
                 max_bet_pct: float = 0.10,     # hard cap: never bet > this % of starting wallet
                 daily_loss_limit_pct: float = 0.05):  # halt if realized loss > this % of starting wallet
        self._cb        = cb
        self._kb        = kb
        self._ws        = kalshi_feed
        self._wallet    = wallet_usd
        self._log_path  = log_path

        self._pk             = pk
        self._live_orders    = bool(live_orders) and pk is not None
        self._session: Optional["aiohttp.ClientSession"] = None
        self._starting_wallet = float(wallet_usd)
        self._max_bet_usd     = float(wallet_usd) * max_bet_pct
        self._daily_loss_cap  = -float(wallet_usd) * daily_loss_limit_pct
        self._realized_pnl    = 0.0
        self._halted          = False

        self._model   = KalshiRegressionModel()
        self._buf     = TrainingBuffer(window_s=TRAINING_WINDOW_S)
        self._pos:    Optional[Position] = None
        self._running = False
        self._last_entry_t: float = 0.0  # monotonic time of last entry

        # Track window transitions in sampler — skip first 30s of samples after
        # a window change so lagged spot features don't bleed from prior window.
        self._sampler_ticker: str   = ""
        self._sampler_window_start: float = 0.0  # monotonic time of last window switch

        self._log_fh = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_path, "a")

        self._tick_path: Optional[Path] = tick_path
        # Bootstrap reads from seed_tick_path (may differ from tick_path when --seed-ticks is used)
        self._seed_tick_path: Optional[Path] = seed_tick_path if seed_tick_path else tick_path
        self._tick_logger: Optional[TickLogger] = None
        if tick_path:
            self._tick_logger = TickLogger(tick_path)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True

        # Live-orders mode needs an aiohttp session for order placement.
        if self._live_orders:
            self._session = aiohttp.ClientSession()
            print(f"  *** LIVE ORDERS ENABLED ***", flush=True)
            print(f"  Wallet (real Kalshi balance): ${self._starting_wallet:,.2f}", flush=True)
            print(f"  Max bet per trade:            ${self._max_bet_usd:,.2f}", flush=True)
            print(f"  Daily loss circuit breaker:   ${-self._daily_loss_cap:,.2f}", flush=True)

        # Warm-start: replay logs/ticks.csv so the model is fit immediately
        # instead of waiting 6+ minutes of live data accumulation.
        n_loaded = self._bootstrap_from_history()
        if n_loaded >= MIN_TRAIN_SAMPLES:
            try:
                X, y, ts_ns = self._buf.get_arrays()
                diag = self._model.fit(X, y, ts_ns)
                self._on_refit(diag)
                print(f"  [Bootstrap] {n_loaded} samples loaded (see [Refit] above)",
                      flush=True)
            except Exception as e:
                print(f"  [Bootstrap] initial fit failed: {e}", flush=True)
        elif n_loaded > 0:
            print(f"  [Bootstrap] {n_loaded} samples loaded "
                  f"(need {MIN_TRAIN_SAMPLES} to fit; will warm up further "
                  f"from live data)", flush=True)
        else:
            print(f"  [Bootstrap] no historical ticks found; cold-starting "
                  f"(first fit in ~{MIN_TRAIN_SAMPLES} seconds)", flush=True)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._sampler_loop())
                tg.create_task(self._refitter_loop())
                tg.create_task(self._decision_loop())
                tg.create_task(self._window_manager_loop())
        finally:
            if self._session is not None:
                await self._session.close()
                self._session = None

    def stop(self) -> None:
        self._running = False
        if self._tick_logger:
            self._tick_logger.close()

    # ------------------------------------------------------------------
    # Bootstrap: replay logs/ticks.csv into the training buffer so the
    # model is already fit when the decision loop starts.
    # ------------------------------------------------------------------

    def _bootstrap_from_history(self) -> int:
        """
        Read self._tick_path and rebuild feature vectors using the EXACT same
        logic as scripts/backtest.py: group rows by window_ticker, compute lags
        only within each window (no cross-window bleed), skip the first 30 rows
        of each window.  This guarantees the initial live R² matches the backtest.
        """
        if not self._seed_tick_path or not self._seed_tick_path.exists():
            return 0

        import csv as _csv
        import numpy as _np
        from collections import defaultdict

        cutoff_ns = time.time_ns() - TRAINING_WINDOW_S * 1_000_000_000

        # --- 1. Load all rows from the last TRAINING_WINDOW_S, keyed by window ---
        windows: dict[str, list[dict]] = defaultdict(list)
        try:
            with open(self._seed_tick_path, newline="") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    try:
                        ts = int(r["ts_ns"])
                        if ts < cutoff_ns:
                            continue
                        ticker = r.get("window_ticker", "")
                        windows[ticker].append({
                            "ts_ns":     ts,
                            "btc_micro": float(r["btc_microprice"]),
                            "yes_mid":   float(r["yes_mid"]),
                            "yes_bid":   float(r["yes_bid"]),
                            "yes_ask":   float(r["yes_ask"]),
                            "K":         float(r["floor_strike"]),
                            "tau_s":     float(r["tau_s"]),
                        })
                    except (KeyError, ValueError, TypeError):
                        continue
        except Exception as e:
            print(f"  [Bootstrap] error reading {self._seed_tick_path}: {e}", flush=True)
            return 0

        if not windows:
            return 0

        # --- 2. For each window, sort by ts and compute features within-window only ---
        n_loaded = 0
        for ticker, rows in windows.items():
            rows.sort(key=lambda r: r["ts_ns"])

            def lagged_in_window(field: str, target_ts: int, idx: int) -> float:
                """Walk backwards within this window only."""
                for j in range(idx - 1, -1, -1):
                    if rows[j]["ts_ns"] <= target_ts:
                        return rows[j][field]
                return rows[0][field]

            for i, r in enumerate(rows):
                # Skip first 30 rows of each window — lag features aren't valid yet
                if i < 30:
                    continue
                K       = r["K"]
                mp_now  = r["btc_micro"]
                yes_mid = r["yes_mid"]
                if K <= 0 or mp_now <= 0:
                    continue
                if not (TRAIN_YES_MID_MIN <= yes_mid <= TRAIN_YES_MID_MAX):
                    continue

                ts    = r["ts_ns"]
                mp5   = lagged_in_window("btc_micro", ts -  5 * 1_000_000_000, i)
                mp10  = lagged_in_window("btc_micro", ts - 10 * 1_000_000_000, i)
                mp15  = lagged_in_window("btc_micro", ts - 15 * 1_000_000_000, i)
                mp20  = lagged_in_window("btc_micro", ts - 20 * 1_000_000_000, i)
                mp25  = lagged_in_window("btc_micro", ts - 25 * 1_000_000_000, i)
                mp30  = lagged_in_window("btc_micro", ts - 30 * 1_000_000_000, i)
                ym5   = lagged_in_window("yes_mid",   ts -  5 * 1_000_000_000, i)
                ym10  = lagged_in_window("yes_mid",   ts - 10 * 1_000_000_000, i)
                ym30  = lagged_in_window("yes_mid",   ts - 30 * 1_000_000_000, i)

                try:
                    x_0  = math.log(mp_now / K)
                    x_5  = math.log(mp5  / K) if mp5  > 0 else x_0
                    x_10 = math.log(mp10 / K) if mp10 > 0 else x_0
                    x_15 = math.log(mp15 / K) if mp15 > 0 else x_0
                    x_20 = math.log(mp20 / K) if mp20 > 0 else x_0
                    x_25 = math.log(mp25 / K) if mp25 > 0 else x_0
                    x_30 = math.log(mp30 / K) if mp30 > 0 else x_0
                except (ValueError, ZeroDivisionError):
                    continue

                tau          = max(1.0, r["tau_s"])
                inv_sqrt_tau = 1.0 / math.sqrt(tau + 1.0)
                kalshi_spr   = max(0.0, r["yes_ask"] - r["yes_bid"])
                kalshi_mom_5  = yes_mid - ym5
                kalshi_mom_10 = yes_mid - ym10
                kalshi_mom_30 = yes_mid - ym30

                X = _np.array([
                    x_0, x_5, x_10, x_15, x_20, x_25, x_30,
                    tau, inv_sqrt_tau,
                    kalshi_spr, kalshi_mom_5, kalshi_mom_10, kalshi_mom_30,
                ], dtype=_np.float64)
                y = _logit(yes_mid)

                self._buf.append_with_ts(X, y, ts)
                n_loaded += 1

        return n_loaded

    # ------------------------------------------------------------------
    # Loop 1: 1-second sampler
    # ------------------------------------------------------------------

    async def _sampler_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                cb, kb = self._cb, self._kb
                fv = build_features(cb, kb)

                # Detect window change and reset warmup timer
                cur_ticker = kb.ticker
                if cur_ticker != self._sampler_ticker:
                    self._sampler_ticker = cur_ticker
                    self._sampler_window_start = t0

                # Skip first 30s of samples after window switch — lag features
                # (x_5..x_30) still contain microprice from the prior window,
                # which poisons the regression the same way it does in the backtest.
                secs_since_window_start = t0 - self._sampler_window_start
                window_warmed = secs_since_window_start >= 30.0

                if (fv is not None and fv.complete and window_warmed
                        and TRAIN_YES_MID_MIN <= kb.yes_mid <= TRAIN_YES_MID_MAX):
                    y = _logit(kb.yes_mid)
                    self._buf.append(fv.as_array(), y)

                # Write raw tick to CSV regardless of feature completeness
                if self._tick_logger and cb.ready and kb.ready:
                    self._tick_logger.log(
                        ts_ns=time.time_ns(),
                        tau_s=kb.tau_s(),
                        btc_microprice=cb.microprice,
                        btc_bid=cb.best_bid,
                        btc_ask=cb.best_ask,
                        yes_bid=kb.yes_bid,
                        yes_ask=kb.yes_ask,
                        yes_mid=kb.yes_mid,
                        floor_strike=kb.floor_strike,
                        window_ticker=kb.ticker,
                    )
            except Exception as e:
                log.debug("sampler error: %s", e)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, SAMPLE_INTERVAL_S - elapsed))

    # ------------------------------------------------------------------
    # Loop 2: 5-minute refitter
    # ------------------------------------------------------------------

    async def _refitter_loop(self) -> None:
        # Wait until we have enough data before first fit
        while self._running and len(self._buf) < MIN_TRAIN_SAMPLES:
            await asyncio.sleep(10)

        while self._running:
            t0 = time.monotonic()
            try:
                X, y, ts_ns = self._buf.get_arrays()
                if len(y) >= MIN_TRAIN_SAMPLES:
                    diag, accepted = await asyncio.get_event_loop().run_in_executor(
                        None, self._model.fit_if_better, X, y, ts_ns
                    )
                    self._on_refit(diag, accepted)
            except Exception as e:
                log.warning("refit error: %s", e)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, REFIT_INTERVAL_S - elapsed))

    def _on_refit(self, diag: ModelDiagnostics, accepted: bool = True) -> None:
        coef_str = "  ".join(
            f"{name}={diag.coefs[name]:+.3f}" for name in diag.coefs
        )
        status = "ACCEPTED" if accepted else "REJECTED (kept prior model)"
        print(
            f"\n[Refit]  n={diag.n_train}  alpha={diag.ridge_alpha:.4f}  "
            f"R2_in={diag.r2_in_sample:.3f}  R2_cv={diag.r2_cv:.3f}  "
            f"R2_hld={diag.r2_held_out:.3f}  lag={diag.estimated_lag_s:.0f}s  "
            f"[{status}]"
            f"\n         {coef_str}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Loop 3: 10-second decision
    # ------------------------------------------------------------------

    async def _decision_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                await self._tick()
            except Exception as e:
                log.warning("decision tick error: %s", e)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, DECISION_INTERVAL_S - elapsed))

    async def _tick(self) -> None:
        now_ns   = time.time_ns()
        kb       = self._kb
        cb       = self._cb
        model    = self._model

        if not kb.ready or not cb.ready:
            self._log_decision(DecisionRow(
                ts_ns=now_ns, tau_s=kb.tau_s(), yes_bid=0, yes_ask=0, yes_mid=0,
                q_predicted=None, q_settled=None,
                edge_up_raw=None, edge_up_net=None,
                edge_magnitude=-99.0, favored_side=None,
                event="abstain", abstention_reason="data_not_ready",
                tier=0, would_bet_usd=0.0, has_open=self._pos is not None,
                model_r2_hld=model.r2_held_out, model_lag_s=model.estimated_lag_s,
                window_ticker=kb.ticker,
            ))
            return

        tau      = kb.tau_s()
        yes_bid  = kb.yes_bid
        yes_ask  = kb.yes_ask
        yes_mid  = kb.yes_mid

        fv = build_features(cb, kb)

        # ---- Sanity gates ----
        if not model.is_fit:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_warmup")
            return
        if model.stale_s() > 900:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_stale")
            return
        # Gate on held-out R² only. R2_cv from TimeSeriesSplit is unreliable
        # when training data spans multiple windows — CV fold boundaries land
        # on window transitions where lag features are stale, pushing CV score
        # to -0.5 even when the model is excellent (R2_hld > 0.85).
        # R2_hld is the clean signal: last 20% of training data, no bleed.
        if model.r2_held_out < 0.20:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_low_r2")
            return
        if fv is None or not fv.complete:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "features_incomplete")
            return

        q_pred    = model.q_predicted(fv)
        q_set     = model.q_settled(fv)

        if q_pred is None or q_set is None:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_predict_failed")
            return

        if abs(q_pred - yes_mid) > 0.15:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_disagrees_market")
            return

        # ---- Edge calculation ----
        edge_up_raw  = q_set - yes_ask
        edge_no_raw  = (1.0 - q_set) - (1.0 - yes_bid)  # buying No = selling Yes at bid

        # Entry-leg fee depends on ENTRY_MODE (0 for maker, taker fee for taker).
        # Exit leg is always taker -- not in this calc, applied at exit time.
        fee_up   = _entry_fee(yes_ask)
        fee_no   = _entry_fee(1.0 - yes_bid)

        # Use a rough intended size for slippage estimate
        rough_size = self._wallet * KELLY_TIERS[-1][1]
        slip_up  = _slippage(kb.yes_depth, rough_size)
        slip_no  = _slippage(kb.no_depth, rough_size)

        edge_up_net = edge_up_raw - fee_up - slip_up
        edge_no_net = edge_no_raw - fee_no - slip_no

        if edge_up_net >= edge_no_net:
            edge_signed = edge_up_net
            favored     = "yes"
        else:
            edge_signed = edge_no_net
            favored     = "no"

        edge_mag = abs(edge_signed)

        # ---- Daily loss circuit breaker ----
        if self._halted:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "halted_daily_loss")
            return

        # ---- Open position: run exit logic ----
        if self._pos is not None:
            await self._evaluate_exit(now_ns, tau, yes_bid, yes_ask, yes_mid,
                                       q_pred, q_set, edge_up_raw, edge_up_net,
                                       edge_mag, favored)
            return

        # ---- No position: run entry logic ----
        # Only enter new positions inside the trained regime. The model is
        # fit on yes_mid in [TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX]; we keep
        # entries strictly inside that range so the model is interpolating,
        # not extrapolating, when it produces q_settled.
        if yes_mid < DECISION_YES_MID_MIN or yes_mid > DECISION_YES_MID_MAX:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "extreme_probability")
            return

        if tau < FALLBACK_TAU_S:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "tau_too_small")
            return

        if yes_ask - yes_bid > 0.10:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "wide_spread")
            return

        # Rate-limit entries: never place more than one entry per MIN_ENTRY_INTERVAL_S
        secs_since_entry = time.monotonic() - self._last_entry_t
        if secs_since_entry < MIN_ENTRY_INTERVAL_S:
            self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "entry_rate_limited")
            return

        bet_usd, tier = _kelly_size(edge_mag, self._wallet)
        if tier == 0:
            row = DecisionRow(
                ts_ns=now_ns, tau_s=tau, yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
                q_predicted=q_pred, q_settled=q_set,
                edge_up_raw=edge_up_raw, edge_up_net=edge_up_net,
                edge_magnitude=edge_mag, favored_side=favored,
                event="abstain", abstention_reason="edge_below_floor",
                tier=0, would_bet_usd=0.0, has_open=False,
                model_r2_hld=model.r2_held_out, model_lag_s=model.estimated_lag_s,
                window_ticker=kb.ticker,
            )
            self._log_decision(row)
            self._print_tick(row)
            return

        # Entry. Maker fills at the bid (we post a resting limit there);
        # taker crosses to the ask. For NO positions, the equivalent flip:
        # maker NO posts at (1 - yes_ask), taker NO crosses at (1 - yes_bid).
        if ENTRY_MODE == "maker":
            entry_price = yes_bid if favored == "yes" else (1.0 - yes_ask)
        else:  # taker
            entry_price = yes_ask if favored == "yes" else (1.0 - yes_bid)

        # Hard cap: never deploy more than max_bet_usd on a single trade.
        bet_usd = min(bet_usd, self._max_bet_usd)
        contracts = bet_usd / max(entry_price, 1e-6)

        # ---- Live order placement (if enabled) ----
        if self._live_orders:
            # On Kalshi, "buy YES" crosses to yes_ask; "buy NO" crosses to no_ask=1-yes_bid.
            if favored == "yes":
                action, side, price_cents = "buy", "yes", round(yes_ask * 100)
            else:
                action, side, price_cents = "buy", "no",  round((1.0 - yes_bid) * 100)
            count_int = max(1, int(contracts))
            if count_int < 1:
                self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "size_too_small")
                return
            order = await kalshi_place_order(
                self._session, self._pk, kb.ticker, action, side, price_cents, count_int,
            )
            if not order:
                log.error(f"ENTRY ORDER REJECTED side={favored} -- staying flat")
                self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "entry_order_rejected")
                return
            filled = int(order.get("filled_count") or 0)
            if filled <= 0:
                log.warning(f"Entry order accepted but unfilled (resting): {order}")
                self._abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "entry_unfilled")
                return
            actual_entry_price = price_cents / 100.0
            actual_size_usd    = filled * actual_entry_price
            self._pos = Position(
                side=favored, entry_price=actual_entry_price, entry_tau_s=tau,
                entry_edge=edge_signed, size_usd=actual_size_usd, contracts=float(filled),
            )
            self._last_entry_t = time.monotonic()
            print(f"\n  [LIVE ENTRY] side={favored} filled={filled}@{price_cents}c "
                  f"=${actual_size_usd:.2f} edge={edge_mag:.3f} tier={tier}", flush=True)
        else:
            # Dry-run: track an idealised position for P&L bookkeeping.
            self._pos = Position(
                side=favored, entry_price=entry_price, entry_tau_s=tau,
                entry_edge=edge_signed, size_usd=bet_usd, contracts=contracts,
            )
            self._last_entry_t = time.monotonic()

        row = DecisionRow(
            ts_ns=now_ns, tau_s=tau, yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
            q_predicted=q_pred, q_settled=q_set,
            edge_up_raw=edge_up_raw, edge_up_net=edge_up_net,
            edge_magnitude=edge_mag, favored_side=favored,
            event="entry", abstention_reason=None,
            tier=tier, would_bet_usd=bet_usd, has_open=True,
            model_r2_hld=model.r2_held_out, model_lag_s=model.estimated_lag_s,
            window_ticker=kb.ticker,
        )
        self._log_decision(row)
        self._print_tick(row)

    async def _evaluate_exit(self, now_ns, tau, yes_bid, yes_ask, yes_mid,
                              q_pred, q_set, edge_up_raw, edge_up_net,
                              edge_mag, favored) -> None:
        pos = self._pos
        if pos is None:
            return

        if pos.side == "yes":
            edge_now = q_set - yes_ask
            exit_price = yes_bid
        else:
            edge_now = (1.0 - q_set) - (1.0 - yes_bid)
            exit_price = 1.0 - yes_ask

        reason = None
        hold_s = pos.entry_tau_s - tau
        if edge_now < LAG_CLOSE_THRESHOLD:
            reason = "exit_lag_closed"
        elif edge_now < pos.entry_edge - STOP_THRESHOLD:
            reason = "exit_stopped"
        elif hold_s >= MAX_HOLD_S:
            reason = "exit_max_hold"
        elif tau < FALLBACK_TAU_S:
            reason = "fallback_resolution"

        if reason:
            # ---- Live exit order placement ----
            # On "fallback_resolution" we let the contract settle (no taker
            # exit, no fees). For lag_closed / stopped / max_hold we hit
            # the bid as a taker.
            actual_exit_price = exit_price
            actual_gross      = pos.contracts * exit_price
            if self._live_orders and "resolution" not in reason:
                if pos.side == "yes":
                    action, side, price_cents = "sell", "yes", round(yes_bid * 100)
                else:
                    action, side, price_cents = "sell", "no",  round((1.0 - yes_ask) * 100)
                count_int = max(1, int(pos.contracts))
                order = await kalshi_place_order(
                    self._session, self._pk, self._kb.ticker,
                    action, side, price_cents, count_int,
                )
                if not order:
                    log.error(f"EXIT ORDER REJECTED reason={reason} -- KEEPING position open, "
                              f"will retry next tick")
                    return
                filled = int(order.get("filled_count") or 0)
                if filled <= 0:
                    log.warning(f"Exit order accepted but unfilled (resting); will retry: {order}")
                    return
                actual_exit_price = price_cents / 100.0
                actual_gross      = filled * actual_exit_price
                if filled < count_int:
                    log.warning(f"Partial exit fill: {filled}/{count_int} contracts")

            entry_fee = _entry_fee(pos.entry_price) * pos.size_usd
            exit_fee  = _fee(actual_exit_price) * actual_gross if "resolution" not in reason else 0.0
            pnl = actual_gross - pos.size_usd - entry_fee - exit_fee

            tag = "LIVE EXIT" if self._live_orders and "resolution" not in reason else "EXIT"
            print(
                f"\n  [{tag}:{reason}] side={pos.side} entry={pos.entry_price:.3f} "
                f"exit={actual_exit_price:.3f} pnl={pnl:+.2f} USD hold={hold_s:.0f}s",
                flush=True,
            )
            self._pos = None

            # Track realized P&L; halt on daily-loss breach.
            if self._live_orders:
                self._realized_pnl += pnl
                if self._realized_pnl < self._daily_loss_cap and not self._halted:
                    self._halted = True
                    log.error(
                        f"DAILY LOSS LIMIT TRIPPED: realized=${self._realized_pnl:.2f} "
                        f"<= cap ${self._daily_loss_cap:.2f}. Bot halted."
                    )
                    print(
                        f"\n  *** DAILY LOSS LIMIT TRIPPED: realized=${self._realized_pnl:.2f} "
                        f"-- HALTING (no new entries) ***", flush=True,
                    )

        row = DecisionRow(
            ts_ns=now_ns, tau_s=tau, yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
            q_predicted=q_pred, q_settled=q_set,
            edge_up_raw=edge_up_raw, edge_up_net=edge_up_net,
            edge_magnitude=edge_mag, favored_side=favored,
            event=reason or "hold",
            abstention_reason=None,
            tier=0, would_bet_usd=0.0,
            has_open=self._pos is not None,
            model_r2_hld=self._model.r2_held_out,
            model_lag_s=self._model.estimated_lag_s,
            window_ticker=self._kb.ticker,
        )
        self._log_decision(row)
        self._print_tick(row)

    def _abstain(self, now_ns, tau, yes_bid, yes_ask, yes_mid, reason) -> None:
        row = DecisionRow(
            ts_ns=now_ns, tau_s=tau, yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
            q_predicted=None, q_settled=None,
            edge_up_raw=None, edge_up_net=None,
            edge_magnitude=-99.0, favored_side=None,
            event="abstain", abstention_reason=reason,
            tier=0, would_bet_usd=0.0,
            has_open=self._pos is not None,
            model_r2_hld=self._model.r2_held_out,
            model_lag_s=self._model.estimated_lag_s,
            window_ticker=self._kb.ticker,
        )
        self._log_decision(row)
        self._print_tick(row)

    # ------------------------------------------------------------------
    # Loop 4: window transition manager
    # ------------------------------------------------------------------

    async def _window_manager_loop(self) -> None:
        import datetime as _dt
        next_mkt: Optional[dict] = None   # full market dict for the next window

        while self._running:
            tau = self._kb.tau_s()

            # Pre-discover next window when < 2 min remaining
            if tau < 120 and next_mkt is None:
                mkt = await _discover_market()
                if mkt and mkt["ticker"] != self._kb.ticker:
                    next_mkt = mkt
                    log.info("pre-discovered next window: %s", mkt["ticker"])

            # Switch when current window has closed
            if tau <= 0:
                if next_mkt:
                    self._do_rollover(next_mkt)
                    next_mkt = None
                else:
                    # Window boundary gap — the next market isn't open yet.
                    # Retry every 5s until it appears (Kalshi takes a few seconds
                    # between the old market closing and the new one going active).
                    for attempt in range(1, 25):
                        await asyncio.sleep(5)
                        mkt = await _discover_market()
                        if mkt and mkt["ticker"] != self._kb.ticker:
                            self._do_rollover(mkt)
                            next_mkt = None
                            break
                        log.info("window gap: waiting for next market (attempt %d/24)",
                                 attempt)
                    else:
                        log.warning("window gap: no new market found after 2 min, "
                                    "continuing with current ticker")

            await asyncio.sleep(10)

    def _do_rollover(self, mkt: dict) -> None:
        import datetime as _dt
        ticker       = mkt["ticker"]
        floor_strike = float(mkt.get("floor_strike") or 0.0)
        ct           = mkt.get("close_time", "")
        try:
            close_time = _dt.datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except Exception:
            import datetime as _dt2
            close_time = _dt2.datetime.now(_dt2.timezone.utc) + _dt2.timedelta(minutes=15)

        log.info("window rollover -> %s  K=%.2f", ticker, floor_strike)
        self._kb.set_window(ticker, floor_strike, close_time)
        self._ws.update_ticker(ticker)
        self._pos = None    # positions don't carry across windows

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_decision(self, row: DecisionRow) -> None:
        if self._log_fh is None:
            return
        import json as _json
        self._log_fh.write(_json.dumps({
            "ts_ns":           row.ts_ns,
            "tau_s":           row.tau_s,
            "yes_bid":         row.yes_bid,
            "yes_ask":         row.yes_ask,
            "yes_mid":         row.yes_mid,
            "q_predicted":     row.q_predicted,
            "q_settled":       row.q_settled,
            "edge_up_raw":     row.edge_up_raw,
            "edge_up_net":     row.edge_up_net,
            "edge_magnitude":  row.edge_magnitude,
            "favored_side":    row.favored_side,
            "event":           row.event,
            "abstention_reason": row.abstention_reason,
            "tier":            row.tier,
            "would_bet_usd":   row.would_bet_usd,
            "has_open":        row.has_open,
            "model_r2_hld":    row.model_r2_hld,
            "model_lag_s":     row.model_lag_s,
            "window_ticker":   row.window_ticker,
        }) + "\n")
        self._log_fh.flush()

    def _print_tick(self, row: DecisionRow) -> None:
        buf_n   = len(self._buf)
        buf_min = self._buf.span_minutes()
        q_s     = f"{row.q_settled:.3f}" if row.q_settled is not None else "---"
        q_p     = f"{row.q_predicted:.3f}" if row.q_predicted is not None else "---"
        edge_s  = f"{row.edge_magnitude:.4f}" if row.edge_magnitude > -50 else "---"

        # Feed staleness — how many ms since last update from each feed
        cb_stale_ms = int((time.time_ns() - self._cb.last_update_ns) / 1e6)
        kb_stale_ms = int((time.time_ns() - self._kb.last_update_ns) / 1e6)
        cb_ok = "CB:OK " if cb_stale_ms < 5000 else f"CB:STALE({cb_stale_ms}ms)"
        kb_ok = "KB:OK" if kb_stale_ms < 15000 else f"KB:STALE({kb_stale_ms}ms)"

        print(
            f"[{row.event:22s}] "
            f"tau={row.tau_s:5.0f}s  "
            f"BTC=${self._cb.microprice:,.2f}  "
            f"bid={row.yes_bid:.3f} ask={row.yes_ask:.3f}  "
            f"q_set={q_s} q_pred={q_p}  "
            f"edge={edge_s}  "
            f"buf={buf_n}({buf_min:.1f}m)  "
            f"R2hld={self._model.r2_held_out:.2f}  "
            f"{cb_ok} {kb_ok}  "
            f"{row.abstention_reason or ''}",
            flush=True,
        )
