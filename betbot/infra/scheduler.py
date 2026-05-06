"""
scheduler.py — 10-second decision tick loop.

Reads live state from all feeds, computes features, and writes one decision
row per tick to the Parquet writer. In Phase 1 (DRY_RUN=true) no orders
are placed — every tick either abstains or records a hypothetical entry.

The scheduler also:
  - Detects window rollovers and triggers PolyBook + PolymarketWS resubscription
  - Writes chainlink_ticks rows when K is captured
  - Writes window_outcomes rows at window close
  - Checks all circuit-breaker conditions and sets abstention_reason accordingly
"""

import asyncio
import logging
import math
import time
from typing import Optional

from polybot.infra.config import (
    ASSETS, DECISION_INTERVAL, SAMPLE_INTERVAL, WINDOW_SECONDS,
    BINANCE_STALE_MS_MAX, COINBASE_STALE_MS_MAX, POLY_STALE_MS_MAX, RTDS_STALE_MS_MAX,
    WIDE_SPREAD_THRESHOLD, KILL_SWITCH_PATH, DRY_RUN,
    MODEL_MIN_CV_R2, MODEL_MAX_DISAGREEMENT, MODEL_MAX_STALE_SECONDS,
)
from polybot.infra.parquet_writer import ParquetWriter
from polybot.models.features import build_features
from polybot.models.edge import compute_edge
from polybot.strategy.decision import apply_decision
from polybot.strategy.position import PositionTracker
from polybot.state.spot_book import SpotBook
from polybot.state.coinbase_book import CoinbaseBook
from polybot.state.poly_book import PolyBook
from polybot.state.window import WindowState

# Phase 2 imports — only used when DRY_RUN=False
if not DRY_RUN:
    from polybot.execution.orders import OrderClient
    from polybot.execution.fills import FillTracker, OpenOrder

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Runs the 10-second decision tick for all assets.

    Dependencies are injected so the TaskGroup in main.py can share
    the same state objects across all tasks.
    """

    def __init__(
        self,
        spot_books: dict[str, SpotBook],
        poly_books: dict[str, PolyBook],
        windows: dict[str, WindowState],
        writer: ParquetWriter,
        coinbase_books: dict[str, CoinbaseBook] | None = None,
        model=None,
        order_client=None,    # OrderClient instance (Phase 2 only)
        fill_tracker=None,    # FillTracker instance (Phase 2 only)
        wallet_usd: float = 1000.0,
    ):
        self.spot_books     = spot_books
        self.poly_books     = poly_books
        self.windows        = windows
        self.writer         = writer
        self.coinbase_books = coinbase_books or {}
        self.models         = model or {}
        self.trackers       = {a: PositionTracker(a) for a in ASSETS}
        self.order_client   = order_client
        self.fill_tracker   = fill_tracker
        self.wallet_usd     = wallet_usd
        self._running       = False
        self._tick_count    = 0
        self._last_log_tick = 0

    async def run(self) -> None:
        self._running = True
        logger.info("scheduler started (DRY_RUN=%s, interval=%ds)", DRY_RUN, DECISION_INTERVAL)

        while self._running:
            # Align to 10-second grid
            now = time.time()
            next_tick = math.ceil(now / DECISION_INTERVAL) * DECISION_INTERVAL
            await asyncio.sleep(max(0.0, next_tick - now))

            if not self._running:
                break

            # Kill switch check
            try:
                import os
                if os.path.exists(KILL_SWITCH_PATH):
                    logger.warning("kill switch detected at %s — stopping", KILL_SWITCH_PATH)
                    self._running = False
                    break
            except Exception:
                pass

            tick_ns = time.time_ns()
            for asset in ASSETS:
                try:
                    self._tick(asset, tick_ns)
                except Exception as exc:
                    logger.exception("scheduler tick error asset=%s: %s", asset, exc)

            self._tick_count += 1
            # Log a heartbeat every 60 ticks (~10 minutes) so the terminal isn't silent
            if self._tick_count - self._last_log_tick >= 60:
                self._last_log_tick = self._tick_count
                self._log_heartbeat(tick_ns)

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Spot-driven K capture
    # ------------------------------------------------------------------

    async def run_k_capturer(self) -> None:
        """
        Capture K from the live Binance microprice as soon as a window opens
        and Binance has data. Replaces the Chainlink RTDS-driven K capture,
        which had high latency and gap issues.

        Runs every 1s. For each asset, if the window has open_ts but no K,
        and Binance has a current microprice, install it as K.
        """
        while self._running:
            await asyncio.sleep(1.0)
            for asset in ASSETS:
                win  = self.windows[asset]
                spot = self.spot_books[asset]
                if win.open_ts == 0 or win.K is not None:
                    continue
                if not spot.ready:
                    continue
                mp = spot.microprice_at(0)
                if mp and mp > 0:
                    win.set_k_from_spot(mp)

    # ------------------------------------------------------------------
    # 1-second training sampler
    # ------------------------------------------------------------------

    async def run_sampler(self) -> None:
        """
        Fires every SAMPLE_INTERVAL seconds and writes a lightweight training
        row to the decisions buffer. These rows have event='sample' and are
        used only by the refitter — the scheduler's tick writes the real
        decisions.
        """
        # Per-asset skip counters so we can see why samples are being dropped
        self._sampler_stats: dict[str, dict[str, int]] = {
            asset: {"written": 0, "no_K": 0, "poly_not_ready": 0,
                    "no_spot": 0, "bad_q": 0, "incomplete": 0, "errors": 0}
            for asset in ASSETS
        }
        logger.info("sampler started interval=%.2fs assets=%s",
                    SAMPLE_INTERVAL, list(ASSETS))
        last_report = time.time()
        while self._running:
            await asyncio.sleep(SAMPLE_INTERVAL)
            if not self._running:
                break
            sample_ns = time.time_ns()
            for asset in ASSETS:
                try:
                    self._sample(asset, sample_ns)
                except Exception:
                    self._sampler_stats[asset]["errors"] += 1
                    logger.debug("sampler error asset=%s", asset, exc_info=True)

            # Every 60s, log aggregate sampler stats so we can see drop reasons
            if time.time() - last_report >= 60:
                last_report = time.time()
                for asset in ASSETS:
                    s = self._sampler_stats[asset]
                    logger.info(
                        "sampler %s: wrote=%d no_K=%d poly_not_ready=%d "
                        "no_spot=%d bad_q=%d incomplete=%d errors=%d",
                        asset.upper(), s["written"], s["no_K"], s["poly_not_ready"],
                        s["no_spot"], s["bad_q"], s["incomplete"], s["errors"],
                    )
                    for k in s: s[k] = 0

    def _sample(self, asset: str, sample_ns: int) -> None:
        """Write a training sample row (no decision logic, no OFI drain)."""
        spot    = self.spot_books[asset]
        cb_book = self.coinbase_books.get(asset)
        poly    = self.poly_books[asset]
        win     = self.windows[asset]
        stats   = self._sampler_stats[asset]

        # Skip if minimum state isn't ready
        if not win.K or win.K <= 0 or win.open_ts == 0:
            stats["no_K"] += 1
            return
        if not poly.ready or not poly.up:
            stats["poly_not_ready"] += 1
            # Suppressed verbose per-occurrence log — the aggregated minute-stat
            # line shows the count. Use DEBUG to see token-level details.
            return
        cb_ok = cb_book and cb_book.ready and cb_book.stale_ms() < COINBASE_STALE_MS_MAX
        bn_ok = spot.ready and spot.stale_ms() < BINANCE_STALE_MS_MAX
        if not cb_ok and not bn_ok:
            stats["no_spot"] += 1
            return

        now_s = sample_ns / 1e9
        q_up_ask = float(poly.up.best_ask) if poly.up else None
        if q_up_ask is None or not (0.02 < q_up_ask < 0.98):
            stats["bad_q"] += 1
            return

        fv = build_features(
            spot=spot,
            cb_book=cb_book,
            poly=poly,
            win=win,
            now_s=now_s,
            ofi_l1=0.0,     # don't drain OFI — that's the decision tick's job
            ofi_l5=0.0,
            cross_spot=self.spot_books.get("eth" if asset == "btc" else "btc"),
            cross_win=self.windows.get("eth" if asset == "btc" else "btc"),
        )
        if not fv.complete:
            stats["incomplete"] += 1
            # Only log the first incomplete occurrence per session
            if stats["incomplete"] == 1 and not getattr(self, "_logged_incomplete_" + asset, False):
                setattr(self, "_logged_incomplete_" + asset, True)
                logger.info(
                    "sampler %s incomplete (first occurrence): "
                    "x_now=%s x_15=%s x_30=%s x_60=%s "
                    "tau=%s cb_ready=%s cb_x_now=%s cb_x_60=%s bn_ready=%s",
                    asset.upper(),
                    fv.x_now is not None, fv.x_15 is not None,
                    fv.x_30 is not None, fv.x_60 is not None,
                    fv.tau is not None,
                    cb_book.ready if cb_book else None,
                    fv.cb_x_now is not None, fv.cb_x_60 is not None,
                    spot.ready,
                )
            return

        row = {
            "ts_ns":      sample_ns,
            "asset":      asset,
            "window_ts":  win.open_ts,
            "tau_s":      float(win.tau_s(now_s)),
            "event":      "sample",          # distinguished from real decisions
            "q_up_ask":   q_up_ask,
            # Feature columns (refitter reads these)
            **fv.as_dict(),
            # Nulls for columns the refitter doesn't use
            "K":              win.K,
            "K_uncertain":    win.K_uncertain,
            "abstention_reason": None,
            "circuit_active":    None,
            "edge_signed":    -99.0,
            "edge_magnitude": -99.0,
        }
        self.writer.write_decision(asset, row)
        stats["written"] += 1

    # ------------------------------------------------------------------
    # Single tick per asset
    # ------------------------------------------------------------------

    def _tick(self, asset: str, tick_ns: int) -> None:
        spot     = self.spot_books[asset]
        poly     = self.poly_books[asset]
        win      = self.windows[asset]
        cb_book  = self.coinbase_books.get(asset)

        now_s = tick_ns / 1e9

        # --- Feed staleness ---
        binance_stale_ms   = spot.stale_ms()
        coinbase_stale_ms  = cb_book.stale_ms() if cb_book else 999_999
        poly_stale_ms      = poly.stale_ms()
        chainlink_stale_ms = int((now_s - win.K_capture_ts_ns / 1e9) * 1000) \
                             if win.K_capture_ts_ns > 0 else 999_999

        # --- Basic window info ---
        window_ts = win.open_ts
        tau_s     = win.tau_s(now_s)

        # --- Build the decision row (all fields, nulls where not yet computed) ---
        row: dict = {
            "ts_ns":       tick_ns,
            "asset":       asset,
            "window_ts":   window_ts,
            "tau_s":       float(tau_s),
            # Binance spot
            "S_mid":           spot.mid if spot.ready else None,
            "microprice":      spot.microprice if spot.ready else None,
            "spot_spread":     float(spot.spread) if spot.ready else None,
            "spot_bid_size_l1": float(spot.best_bid_size) if spot.ready else None,
            "spot_ask_size_l1": float(spot.best_ask_size) if spot.ready else None,
            # Coinbase spot (L1 from ticker)
            "cb_mid":       cb_book.mid if cb_book and cb_book.ready else None,
            "cb_microprice": cb_book.microprice if cb_book and cb_book.ready else None,
            # Strike
            "K":           win.K,
            "K_uncertain": win.K_uncertain,
            # Lagged Binance microprice features (computed below)
            "x_now_logKratio": None,
            "x_15_logKratio":  None,
            "x_30_logKratio":  None,
            "x_45_logKratio":  None,
            "x_60_logKratio":  None,
            "x_90_logKratio":  None,
            "x_120_logKratio": None,
            # Lagged Coinbase microprice features (computed below)
            "cb_x_now_logKratio": None,
            "cb_x_15_logKratio":  None,
            "cb_x_30_logKratio":  None,
            "cb_x_60_logKratio":  None,
            # Microstructure
            "ofi_l1":                   None,
            "ofi_l5_weighted":          None,
            "pm_book_imbalance":        None,
            "pm_trade_flow_30s":        None,
            "momentum_30s":             None,
            "momentum_60s":             None,
            "cross_asset_momentum_60s": None,
            "sigma_per_sec_realized":   None,
            # Polymarket book
            "q_up_bid":         None,
            "q_up_ask":         None,
            "q_up_mid":         None,
            "q_down_bid":       None,
            "q_down_ask":       None,
            "q_down_mid":       None,
            "poly_spread_up":   None,
            "poly_spread_down": None,
            "poly_depth_up_l1": None,
            "poly_depth_down_l1": None,
            # Model output (Stage 2B)
            "model_version_id":           None,
            "q_predicted":                None,
            "q_settled":                  None,
            "q_predicted_minus_q_actual": None,
            # Edge (Stage 2C)
            "edge_up_raw":            None,
            "edge_down_raw":          None,
            "fee_up_per_dollar":      None,
            "fee_down_per_dollar":    None,
            "slippage_up_per_dollar": None,
            "slippage_down_per_dollar": None,
            "edge_up_net":    None,
            "edge_down_net":  None,
            "edge_signed":    -99.0,
            "edge_magnitude": -99.0,
            "favored_side":   None,
            # Action
            "event":              "abstain",
            "chosen_side":        None,
            "tier":               0,
            "would_bet_usd":      0.0,
            "bet_price":          None,
            "bet_payout_contracts": None,
            "abstention_reason":  None,
            # Position state
            "has_open_position":     False,
            "position_side":         None,
            "position_entry_tau":    None,
            "position_entry_price":  None,
            "position_edge_at_entry": None,
            # Feed staleness
            "binance_stale_ms":    binance_stale_ms,
            "coinbase_stale_ms":   coinbase_stale_ms,
            "polymarket_stale_ms": poly_stale_ms,
            "chainlink_stale_ms":  chainlink_stale_ms,
            "circuit_active":      None,
        }

        # --- Circuit breakers → abstain reasons ---
        reason = self._circuit_check(spot, poly, win, binance_stale_ms,
                                     coinbase_stale_ms, poly_stale_ms,
                                     chainlink_stale_ms)
        if reason:
            row["abstention_reason"] = reason
            row["circuit_active"]    = reason
            self.writer.write_decision(asset, row)
            return

        # --- Drain OFI before feature computation (destructive — must happen once) ---
        ofi_l1, ofi_l5 = spot.drain_ofi()

        # --- Build full feature vector ---
        cross_asset = "eth" if asset == "btc" else "btc"
        fv = build_features(
            spot=spot,
            cb_book=cb_book,
            poly=poly,
            win=win,
            now_s=now_s,
            ofi_l1=ofi_l1,
            ofi_l5=ofi_l5,
            cross_spot=self.spot_books.get(cross_asset),
            cross_win=self.windows.get(cross_asset),
        )
        row.update(fv.as_dict())

        # --- Polymarket book state (needed for edge calc, logged regardless) ---
        if poly.ready and poly.up and poly.down:
            up, dn = poly.up, poly.down
            row.update({
                "q_up_bid":          float(up.best_bid),
                "q_up_ask":          float(up.best_ask),
                "q_up_mid":          float(up.mid),
                "q_down_bid":        float(dn.best_bid),
                "q_down_ask":        float(dn.best_ask),
                "q_down_mid":        float(dn.mid),
                "poly_spread_up":    float(up.spread),
                "poly_spread_down":  float(dn.spread),
                "poly_depth_up_l1":  float(up.depth_l1),
                "poly_depth_down_l1": float(dn.depth_l1),
            })

            # Wide spread circuit breaker
            if up.spread > WIDE_SPREAD_THRESHOLD or dn.spread > WIDE_SPREAD_THRESHOLD:
                row["abstention_reason"] = "wide_spread"
                row["circuit_active"]    = "wide_spread"
                self.writer.write_decision(asset, row)
                return

        # --- Model predictions ---
        model = self.models.get(asset)
        if model is None or not model.is_fit:
            row["abstention_reason"] = "model_warmup"
            row["event"]             = "abstain"
            self.writer.write_decision(asset, row)
            logger.debug("tick asset=%s tau=%.0f abstain=model_warmup", asset, tau_s)
            return

        # --- Model sanity gates (CLAUDE.md §7) ---
        stale_s = (tick_ns - model.last_refit_ns) / 1e9 if model.last_refit_ns else 9999
        if model.r2_cv_mean < MODEL_MIN_CV_R2:
            row["abstention_reason"] = "model_low_r2"
            self.writer.write_decision(asset, row)
            return
        if stale_s > MODEL_MAX_STALE_SECONDS:
            row["abstention_reason"] = "model_stale"
            self.writer.write_decision(asset, row)
            return

        row["model_version_id"] = model.version_id
        q_pred    = model.q_predicted(fv)
        q_settled = model.q_settled(fv)

        if q_pred is not None:
            row["q_predicted"] = float(q_pred)
        if q_settled is not None:
            row["q_settled"] = float(q_settled)

        q_up_ask = row.get("q_up_ask")
        if q_pred is not None and q_up_ask is not None:
            residual = float(q_pred - q_up_ask)
            row["q_predicted_minus_q_actual"] = residual
            if abs(residual) > MODEL_MAX_DISAGREEMENT:
                row["abstention_reason"] = "model_disagrees_market"
                self.writer.write_decision(asset, row)
                return

        # --- Edge calculation ---
        er = None
        if q_settled is not None and poly.ready:
            er = compute_edge(
                q_settled=q_settled,
                poly=poly,
                wallet_usd=self.wallet_usd,
            )
            if er is not None:
                row.update(er.as_dict())
                row["edge_signed"]    = er.edge_signed
                row["edge_magnitude"] = er.edge_magnitude
                row["favored_side"]   = er.favored_side

        # --- Entry / exit decision ---
        decision = apply_decision(
            asset=asset,
            tau_s=tau_s,
            window_ts=window_ts,
            edge=er,
            tracker=self.trackers[asset],
            q_up_bid=row.get("q_up_bid"),
            q_down_bid=row.get("q_down_bid"),
            q_settled=q_settled,
        )
        row.update(decision)

        event = row.get("event", "abstain")
        reason = row.get("abstention_reason") or ""
        logger.info(
            "decision %s %s%s tau=%.0fs edge=%.4f",
            asset.upper(), event,
            f" ({reason})" if reason else "",
            tau_s, row.get("edge_magnitude", -99),
        )

        # --- Live order execution (Phase 2 only) ---
        if not DRY_RUN and self.order_client and self.fill_tracker:
            self._execute(asset, event, row, win, er)

        self.writer.write_decision(asset, row)

    # ------------------------------------------------------------------
    # Live order execution (DRY_RUN=False only)
    # ------------------------------------------------------------------

    def _execute(self, asset: str, event: str, row: dict, win: WindowState, er) -> None:
        """Place or exit real orders on Polymarket. No-op in dry run."""
        ft  = self.fill_tracker
        oc  = self.order_client

        # Hard stop — refuse all new entries
        if ft.hard_stop_triggered():
            row["abstention_reason"] = "hard_stop"
            row["event"]             = "abstain"
            return

        if event == "entry":
            side      = row.get("chosen_side")
            token_id  = win.up_token_id if side == "up" else win.down_token_id
            if not token_id:
                logger.warning("execute: no token_id for asset=%s side=%s", asset, side)
                return

            price     = float(row.get("bet_price") or 0.5)
            size_usd  = float(row.get("would_bet_usd") or 0.0)
            if size_usd <= 0 or price <= 0:
                return

            # py-clob-client uses "BUY"/"SELL" not "up"/"down"
            order_id = oc.place_order(
                token_id=token_id,
                side="BUY",
                size_usd=size_usd,
                price=price,
            )
            if order_id:
                ft.record_entry(OpenOrder(
                    order_id=order_id,
                    asset=asset,
                    side=side,
                    token_id=token_id,
                    size_usd=size_usd,
                    size_shares=size_usd / price,
                    entry_price=price,
                    placed_ns=time.time_ns(),
                    window_ts=win.open_ts,
                ))
                row["order_id"] = order_id
            else:
                # Order failed — revert the position tracker so we don't
                # think we're in a position we don't actually have
                self.trackers[asset].close()
                row["event"]             = "abstain"
                row["abstention_reason"] = "order_failed"

        elif event in ("exit_lag_closed", "exit_stopped", "fallback_resolution"):
            open_order = ft.open_order_for(asset, win.open_ts)
            if not open_order:
                return

            exit_price = float(row.get("bet_price") or 0.5)
            # Sell: place limit at current bid for fast fill
            exit_order_id = oc.place_order(
                token_id=open_order.token_id,
                side="SELL",
                size_usd=open_order.filled_shares * exit_price
                         if open_order.confirmed else open_order.size_usd,
                price=exit_price,
            )
            shares = open_order.filled_shares or open_order.size_shares
            ft.record_exit(
                order_id=open_order.order_id,
                exit_price=exit_price,
                exit_shares=shares,
                exit_reason=event,
            )
            if exit_order_id:
                row["exit_order_id"] = exit_order_id

    # ------------------------------------------------------------------
    # Heartbeat log (once per ~10 minutes)
    # ------------------------------------------------------------------

    def _log_heartbeat(self, tick_ns: int) -> None:
        now_s = tick_ns / 1e9
        parts = []
        for asset in ASSETS:
            spot    = self.spot_books[asset]
            cb      = self.coinbase_books.get(asset)
            poly    = self.poly_books[asset]
            win     = self.windows[asset]
            model   = self.models.get(asset)

            tau_s   = win.tau_s(now_s)
            cb_rdy  = cb.ready if cb else False
            r2      = f"{model.r2_cv_mean:.3f}" if model and model.is_fit else "unfit"
            diag    = model._last_diagnostics if model and model.is_fit else None
            lag     = f"{diag.estimated_lag_seconds:.0f}s" if diag else "—"

            # Last abstention reason for this asset
            parts.append(
                f"{asset.upper()}: spot={'ok' if spot.ready else 'stale'} "
                f"cb={'ok' if cb_rdy else 'stale'} "
                f"poly={'ok' if poly.ready else 'stale'} "
                f"tau={tau_s:.0f}s "
                f"model_r2={r2} lag={lag}"
            )
        logger.info("heartbeat tick=%d | %s", self._tick_count, " | ".join(parts))

    # ------------------------------------------------------------------
    # Circuit breaker checks
    # ------------------------------------------------------------------

    def _circuit_check(
        self,
        spot: SpotBook,
        poly: PolyBook,
        win: WindowState,
        binance_stale_ms: int,
        coinbase_stale_ms: int,
        poly_stale_ms: int,
        chainlink_stale_ms: int,
    ) -> Optional[str]:
        # Require at least one healthy spot feed (Binance OR Coinbase).
        # Binance is geo-blocked from US IPs; Coinbase is the fallback.
        binance_ok  = spot.ready and binance_stale_ms <= BINANCE_STALE_MS_MAX
        coinbase_ok = coinbase_stale_ms <= COINBASE_STALE_MS_MAX
        if not binance_ok and not coinbase_ok:
            return "data_stale"
        if not poly.ready:
            return "data_stale"
        if poly_stale_ms > POLY_STALE_MS_MAX:
            return "data_stale"
        if win.open_ts == 0:
            return "data_stale"
        if win.K_uncertain:
            return "k_uncertain"
        return None
