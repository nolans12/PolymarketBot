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
    ASSETS, DECISION_INTERVAL, WINDOW_SECONDS,
    BINANCE_STALE_MS_MAX, COINBASE_STALE_MS_MAX, POLY_STALE_MS_MAX, RTDS_STALE_MS_MAX,
    WIDE_SPREAD_THRESHOLD, KILL_SWITCH_PATH, DRY_RUN,
)
from polybot.infra.parquet_writer import ParquetWriter
from polybot.models.features import build_features
from polybot.state.spot_book import SpotBook
from polybot.state.coinbase_book import CoinbaseBook
from polybot.state.poly_book import PolyBook
from polybot.state.window import WindowState

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
        model=None,   # dict[str, RegressionModel] or None until Stage 2B
    ):
        self.spot_books     = spot_books
        self.poly_books     = poly_books
        self.windows        = windows
        self.writer         = writer
        self.coinbase_books = coinbase_books or {}
        self.models         = model or {}   # asset -> RegressionModel
        self._running       = False

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

    def stop(self) -> None:
        self._running = False

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
                                     poly_stale_ms, chainlink_stale_ms)
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
            cross_spot=self.spot_books[cross_asset],
            cross_win=self.windows[cross_asset],
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

        row["model_version_id"] = model.version_id
        q_pred    = model.q_predicted(fv)
        q_settled = model.q_settled(fv)
        if q_pred is not None:
            row["q_predicted"] = float(q_pred)
        if q_settled is not None:
            row["q_settled"] = float(q_settled)
        if q_pred is not None and row.get("q_up_ask") is not None:
            row["q_predicted_minus_q_actual"] = float(q_pred - row["q_up_ask"])

        # Stage 2C/2D will add edge calculation and entry/exit logic here.
        row["event"] = "abstain"
        row["abstention_reason"] = "model_warmup"
        self.writer.write_decision(asset, row)

    # ------------------------------------------------------------------
    # Circuit breaker checks
    # ------------------------------------------------------------------

    def _circuit_check(
        self,
        spot: SpotBook,
        poly: PolyBook,
        win: WindowState,
        binance_stale_ms: int,
        poly_stale_ms: int,
        chainlink_stale_ms: int,
    ) -> Optional[str]:
        if not spot.ready:
            return "data_stale"
        if binance_stale_ms > BINANCE_STALE_MS_MAX:
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
