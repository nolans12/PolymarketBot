"""
scheduler.py — Core orchestrator for the Kalshi lead-lag arbitrage bot.

Two concurrent loops:
  1. Sampler (0.1s): build FeatureVec → write raw tick to CSV
  2. Decision (0.5s): compute q_settled/edge, apply Kelly tiers, log decisions

Also manages 15-min window transitions: pre-discovers the next ticker when
tau < 2 min, switches the Kalshi feed to the new ticker on rollover.

The model is static: loaded at startup via --model-file, never retrained.
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
    LAG_CLOSE_THRESHOLD, SAMPLE_INTERVAL_S,
    STOP_THRESHOLD, THETA_FEE_TAKER,
    TRAIN_YES_MID_MIN, TRAIN_YES_MID_MAX,
    DECISION_YES_MID_MIN, DECISION_YES_MID_MAX,
    ENTRY_MODE, MAX_HOLD_S, MIN_ENTRY_INTERVAL_S,
    COINBASE_STALE_MS_MAX, KALSHI_STALE_MS_MAX,
    SIZE_MAX_USD, SIZE_MIN_USD, EXIT_SLIP_CENTS,
)
from betbot.kalshi.features import FeatureVec, _logit, _sigmoid, build_features
from betbot.kalshi.orders import place_order as kalshi_place_order
from betbot.kalshi.model import LGBMModel, make_model, load_model
from betbot.kalshi.tick_logger import TickLogger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fee / slippage helpers
# ---------------------------------------------------------------------------

def _fee(p: float) -> float:
    """Taker fee per dollar bet at price p. Always applied to the exit leg."""
    return THETA_FEE_TAKER * p * (1.0 - p)


def _entry_fee(p: float) -> float:
    """Entry-leg fee per dollar bet. Zero in maker mode."""
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
            return min(wallet * frac, SIZE_MAX_USD), tier_idx
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
    entry_mono_s:    float = 0.0  # wall-clock monotonic time at entry


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
                    log.debug("list_markets HTTP %s", r.status)
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
        log.debug("list_markets error: %s", e)
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
        log.debug("discover_market: no active markets in series %s", series)
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
    Wires SpotBook + KalshiBook + static LGBMModel into two concurrent async loops.

    Usage:
        sched = Scheduler(spot_book, kb_book, kalshi_feed, preloaded_model=model)
        await sched.run()   # runs until cancelled
    """

    def __init__(self, cb: SpotBook, kb: KalshiBook,
                 kalshi_feed,          # KalshiRestFeed — must expose update_ticker()
                 wallet_usd: float = 1000.0,
                 log_path: Optional[Path] = None,
                 tick_path: Optional[Path] = None,
                 pk=None,
                 live_orders: bool = False,
                 max_bet_pct: float = 0.10,
                 daily_loss_limit_pct: float = 0.05,
                 series: str = KALSHI_SERIES,
                 preloaded_model: Optional[LGBMModel] = None):
        self._cb        = cb
        self._kb        = kb
        self._ws        = kalshi_feed
        self._wallet    = wallet_usd
        self._log_path  = log_path

        self._pk             = pk
        self._live_orders    = bool(live_orders) and pk is not None
        self._session: Optional[aiohttp.ClientSession] = None
        self._starting_wallet = float(wallet_usd)
        self._max_bet_usd     = float(wallet_usd) * max_bet_pct
        self._daily_loss_cap  = -float(wallet_usd) * daily_loss_limit_pct
        self._realized_pnl    = 0.0
        self._halted          = False

        self._series  = series
        self._model: Optional[LGBMModel] = preloaded_model
        self._pos:    Optional[Position] = None
        self._running = False
        self._last_entry_t: float = 0.0
        self._last_heartbeat_s: float = 0.0
        self._last_stale_warn_s: float = 0.0
        self._asset: str = series.replace("KXBTC15M", "BTC").replace("KXETH15M", "ETH").replace("KXSOL15M", "SOL").replace("KXXRP15M", "XRP")

        # Track window transitions in sampler — skip first 30s of samples after
        # a window change so lagged spot features don't bleed from prior window.
        self._sampler_ticker: str   = ""
        self._sampler_window_start: float = 0.0  # monotonic time of last window switch

        self._log_fh = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_path, "a")

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

        if self._model is not None:
            mtype = type(self._model).__name__
            print(f"  [Preloaded] Using pre-trained {mtype}  "
                  f"R2_hld={self._model.r2_held_out:.3f}  "
                  f"-- ready to trade immediately", flush=True)
        else:
            print(f"  [No model] Run --model-file to load a trained model. "
                  f"Bot will abstain until a model is provided.", flush=True)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._sampler_loop())
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
    # Loop 1: sampler — writes raw ticks to CSV (training data collection)
    # ------------------------------------------------------------------

    async def _sampler_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                cb, kb = self._cb, self._kb

                # Track window transitions to detect when lag features are cold
                cur_ticker = kb.ticker
                if cur_ticker != self._sampler_ticker:
                    self._sampler_ticker = cur_ticker
                    self._sampler_window_start = t0

                # Write raw tick to CSV regardless of feature completeness
                if self._tick_logger and cb.ready and kb.ready:
                    self._tick_logger.log(
                        ts_ns=time.time_ns(),
                        tau_s=kb.tau_s(),
                        btc_microprice=cb.microprice,
                        btc_bid=cb.best_bid,
                        btc_ask=cb.best_ask,
                        cb_microprice=cb.microprice,
                        bn_microprice=cb.microprice,
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
    # Loop 2: decision
    # ------------------------------------------------------------------

    async def _decision_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                await self._tick()
            except Exception as e:
                log.debug("decision tick error: %s", e)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, DECISION_INTERVAL_S - elapsed))

    async def _tick(self) -> None:
        now_ns   = time.time_ns()
        kb       = self._kb
        cb       = self._cb
        model    = self._model

        if not kb.ready or not cb.ready:
            self._log_abstain(now_ns, kb.tau_s(), 0, 0, 0, "data_not_ready")
            return

        tau      = kb.tau_s()
        yes_bid  = kb.yes_bid
        yes_ask  = kb.yes_ask
        yes_mid  = kb.yes_mid

        fv = build_features(cb, kb)

        # ---- Model gate ----
        if model is None or not model.is_fit:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_not_loaded")
            return
        if model.r2_held_out < 0.20:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_low_r2")
            return
        if fv is None or not fv.complete:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "features_incomplete")
            return

        # ---- Feed staleness gates ----
        if cb.stale_ms() > COINBASE_STALE_MS_MAX:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "coinbase_stale")
            return
        if kb.stale_ms() > KALSHI_STALE_MS_MAX:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "kalshi_stale")
            return

        q_pred = model.q_predicted(fv)
        q_set  = model.q_settled(fv)

        if q_pred is None or q_set is None:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_predict_failed")
            return

        if abs(q_pred - yes_mid) > 0.15:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "model_disagrees_market")
            return

        # ---- Edge calculation ----
        edge_up_raw  = q_set - yes_ask
        edge_no_raw  = (1.0 - q_set) - (1.0 - yes_bid)

        fee_up   = _entry_fee(yes_ask)
        fee_no   = _entry_fee(1.0 - yes_bid)

        rough_size = self._wallet * KELLY_TIERS[-1][1]
        slip_up  = _slippage(kb.yes_depth, rough_size)
        slip_no  = _slippage(kb.no_depth, rough_size)

        edge_up_net = edge_up_raw - fee_up - slip_up - EXIT_SLIP_CENTS
        edge_no_net = edge_no_raw - fee_no - slip_no - EXIT_SLIP_CENTS

        if edge_up_net >= edge_no_net:
            edge_signed = edge_up_net
            favored     = "yes"
        else:
            edge_signed = edge_no_net
            favored     = "no"

        edge_mag = abs(edge_signed)

        # ---- Daily loss circuit breaker ----
        if self._halted:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "halted_daily_loss")
            return

        # ---- Open position: run exit logic ----
        if self._pos is not None:
            await self._evaluate_exit(now_ns, tau, yes_bid, yes_ask, yes_mid,
                                       q_pred, q_set, edge_up_raw, edge_up_net,
                                       edge_mag, favored)
            return

        # ---- No position: run entry logic ----
        if yes_mid < DECISION_YES_MID_MIN or yes_mid > DECISION_YES_MID_MAX:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "extreme_probability")
            return

        if tau < FALLBACK_TAU_S:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "tau_too_small")
            return

        if yes_ask - yes_bid > 0.10:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "wide_spread")
            return

        secs_since_entry = time.monotonic() - self._last_entry_t
        if secs_since_entry < MIN_ENTRY_INTERVAL_S:
            self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "entry_rate_limited")
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

        if ENTRY_MODE == "maker":
            entry_price = yes_bid if favored == "yes" else (1.0 - yes_ask)
        else:
            entry_price = yes_ask if favored == "yes" else (1.0 - yes_bid)

        bet_usd   = min(bet_usd, self._max_bet_usd, SIZE_MAX_USD)  # hard cap, dollars
        contracts = bet_usd / max(entry_price, 1e-6)

        # ---- Live order placement (if enabled) ----
        if self._live_orders:
            if favored == "yes":
                action, side, price_cents = "buy", "yes", round(yes_ask * 100)
            else:
                action, side, price_cents = "buy", "no",  round((1.0 - yes_bid) * 100)
            count_int = max(1, int(contracts))
            order = await kalshi_place_order(
                self._session, self._pk, kb.ticker, action, side, price_cents, count_int,
            )
            if not order:
                log.debug("entry order rejected/unfilled, staying flat")
                self._last_entry_t = time.monotonic()
                self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "entry_order_rejected",
                                  edge=edge_mag, side=favored, q_pred=q_pred, q_set=q_set,
                                  edge_raw=edge_up_raw, edge_net=edge_up_net,
                                  attempted_price=price_cents / 100.0)
                return
            filled = int(order.get("filled_count") or 0)
            if filled <= 0:
                log.debug("entry FoK returned 0 fills")
                self._last_entry_t = time.monotonic()
                self._log_abstain(now_ns, tau, yes_bid, yes_ask, yes_mid, "entry_unfilled",
                                  edge=edge_mag, side=favored, q_pred=q_pred, q_set=q_set,
                                  edge_raw=edge_up_raw, edge_net=edge_up_net,
                                  attempted_price=price_cents / 100.0)
                return
            actual_entry_price = price_cents / 100.0
            actual_size_usd    = filled * actual_entry_price
            self._pos = Position(
                side=favored, entry_price=actual_entry_price, entry_tau_s=tau,
                entry_edge=edge_signed, size_usd=actual_size_usd, contracts=float(filled),
                entry_mono_s=time.monotonic(),
            )
            self._last_entry_t = time.monotonic()
            cost_usd = filled * actual_entry_price
            print(
                f"\n  *** ENTRY [{self._asset}] {favored.upper()}  "
                f"{filled} contracts @ {price_cents}¢  cost=${cost_usd:.2f}  "
                f"edge={edge_mag:.3f}  tier={tier}  tau={tau:.0f}s ***",
                flush=True,
            )
        else:
            self._pos = Position(
                side=favored, entry_price=entry_price, entry_tau_s=tau,
                entry_edge=edge_signed, size_usd=bet_usd, contracts=contracts,
                entry_mono_s=time.monotonic(),
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
        pos   = self._pos
        model = self._model
        if pos is None or model is None:
            return

        if pos.side == "yes":
            edge_now   = q_set - yes_ask
            exit_price = yes_bid
        else:
            edge_now   = (1.0 - q_set) - (1.0 - yes_bid)
            exit_price = 1.0 - yes_ask

        reason = None
        hold_s = time.monotonic() - pos.entry_mono_s  # wall-clock hold time
        if edge_now < LAG_CLOSE_THRESHOLD:
            reason = "exit_lag_closed"
        elif STOP_THRESHOLD is not None and edge_now < pos.entry_edge - STOP_THRESHOLD:
            reason = "exit_stopped"
        elif hold_s >= MAX_HOLD_S:
            reason = "exit_max_hold"
        elif tau < FALLBACK_TAU_S:
            reason = "fallback_resolution"

        if reason:
            actual_exit_price = exit_price
            actual_gross      = pos.contracts * exit_price
            if self._live_orders and "resolution" not in reason:
                if pos.side == "yes":
                    action, side = "sell", "yes"
                    price_cents  = round(yes_bid * 100)
                else:
                    action, side = "sell", "no"
                    price_cents  = round((1.0 - yes_ask) * 100)
                count_int = max(1, int(pos.contracts))

                order = await kalshi_place_order(
                    self._session, self._pk, self._kb.ticker,
                    action, side, price_cents, count_int,
                )

                if not order or int(order.get("filled_count") or 0) <= 0:
                    if hold_s >= MAX_HOLD_S * 2:
                        print(
                            f"\n  *** EXIT ABANDONED [{self._asset}] after {hold_s:.0f}s — "
                            f"clearing position, CHECK KALSHI UI ***",
                            flush=True,
                        )
                        self._pos = None
                    else:
                        print(f"  *** EXIT FAILED [{self._asset}] hold={hold_s:.0f}s — retrying next tick ***", flush=True)
                    return

                filled = int(order.get("filled_count") or 0)
                actual_exit_price = price_cents / 100.0
                actual_gross      = filled * actual_exit_price

            entry_fee = _entry_fee(pos.entry_price) * pos.size_usd
            exit_fee  = _fee(actual_exit_price) * actual_gross if "resolution" not in reason else 0.0
            pnl = actual_gross - pos.size_usd - entry_fee - exit_fee

            live_tag = "LIVE " if self._live_orders and "resolution" not in reason else ""
            pnl_sign = "+" if pnl >= 0 else ""
            print(
                f"\n  *** {live_tag}EXIT [{self._asset}] {reason}  "
                f"{pos.side.upper()}  entry={pos.entry_price:.3f} → exit={actual_exit_price:.3f}  "
                f"pnl=${pnl_sign}{pnl:.2f}  hold={hold_s:.0f}s ***",
                flush=True,
            )
            self._pos = None

            if self._live_orders:
                self._realized_pnl += pnl
                if self._realized_pnl < self._daily_loss_cap and not self._halted:
                    self._halted = True
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
            model_r2_hld=model.r2_held_out,
            model_lag_s=model.estimated_lag_s,
            window_ticker=self._kb.ticker,
        )
        self._log_decision(row)
        self._print_tick(row)

    def _log_abstain(self, now_ns, tau, yes_bid, yes_ask, yes_mid, reason,
                     edge=-99.0, side=None, q_pred=None, q_set=None,
                     edge_raw=None, edge_net=None, attempted_price=None) -> None:
        model = self._model
        r2    = model.r2_held_out if model is not None else 0.0
        lag   = model.estimated_lag_s if model is not None else 0.0
        row = DecisionRow(
            ts_ns=now_ns, tau_s=tau, yes_bid=yes_bid, yes_ask=yes_ask, yes_mid=yes_mid,
            q_predicted=q_pred, q_settled=q_set,
            edge_up_raw=edge_raw, edge_up_net=edge_net,
            edge_magnitude=edge, favored_side=side,
            event="abstain", abstention_reason=reason,
            tier=0, would_bet_usd=attempted_price or 0.0,
            has_open=self._pos is not None,
            model_r2_hld=r2,
            model_lag_s=lag,
            window_ticker=self._kb.ticker,
        )
        self._log_decision(row)
        self._print_tick(row)

    # ------------------------------------------------------------------
    # Loop 3: window transition manager
    # ------------------------------------------------------------------

    async def _window_manager_loop(self) -> None:
        import datetime as _dt
        next_mkt: Optional[dict] = None   # full market dict for the next window

        while self._running:
            tau = self._kb.tau_s()

            # Pre-discover next window when < 2 min remaining
            if tau < 120 and next_mkt is None:
                mkt = await _discover_market(self._series)
                if mkt and mkt["ticker"] != self._kb.ticker:
                    next_mkt = mkt
                    log.debug("pre-discovered next window: %s", mkt["ticker"])

            # Switch when current window has closed
            if tau <= 0:
                if next_mkt:
                    self._do_rollover(next_mkt)
                    next_mkt = None
                else:
                    # Window boundary gap — poll until the new market appears.
                    attempt = 0
                    while self._running:
                        await asyncio.sleep(5)
                        attempt += 1
                        mkt = await _discover_market(self._series)
                        if mkt and mkt["ticker"] != self._kb.ticker:
                            self._do_rollover(mkt)
                            next_mkt = None
                            break
                        log.debug("window gap: waiting for next market (attempt %d)", attempt)

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

        log.debug("window rollover -> %s  K=%.2f", ticker, floor_strike)
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
        # Only print feed staleness warnings and non-abstain events — no per-tick spam.
        cb_stale_ms = int((time.time_ns() - self._cb.last_update_ns) / 1e6)
        kb_stale_ms = int((time.time_ns() - self._kb.last_update_ns) / 1e6)

        now_mono = time.monotonic()
        if now_mono - self._last_stale_warn_s >= 30:
            warned = False
            if self._cb.last_update_ns > 0 and cb_stale_ms >= 5000:
                print(f"  *** CB:STALE {cb_stale_ms}ms — Coinbase feed silent ***", flush=True)
                warned = True
            if self._kb.last_update_ns > 0 and kb_stale_ms >= 15000:
                print(f"  *** KB:STALE {kb_stale_ms}ms — Kalshi feed silent ***", flush=True)
                warned = True
            if warned:
                self._last_stale_warn_s = now_mono

        # Print a heartbeat line every 60s so you know it's alive
        now_s = time.monotonic()
        if now_s - self._last_heartbeat_s >= 60:
            self._last_heartbeat_s = now_s
            edge_s = f"{row.edge_magnitude:.4f}" if row.edge_magnitude > -50 else "---"
            pos_str = f"OPEN {self._pos.side}@{self._pos.entry_price:.3f}" if self._pos else "flat"
            print(
                f"  [heartbeat] {self._asset}  tau={row.tau_s:.0f}s  "
                f"bid={row.yes_bid:.3f} ask={row.yes_ask:.3f}  edge={edge_s}  {pos_str}",
                flush=True,
            )
