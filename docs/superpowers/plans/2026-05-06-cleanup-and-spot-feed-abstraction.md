# Cleanup + Pluggable Spot Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute CLAUDE.md WI-1 (delete Polymarket-era dead code) and WI-3 (introduce `SpotFeed` abstraction so Coinbase or Binance can drive the model interchangeably) on the `kalshi` branch, without disturbing the dry run currently running as PID 17128.

**Architecture:** Two sequential commits/PRs. WI-1 is purely subtractive (no live module imports from the dead directories — verified). WI-3 is a rename-plus-extract refactor: `CoinbaseBook` → `SpotBook`, introduce a `SpotFeed` Protocol, add `BinanceFeed`, rename `cb_momentum_*` → `spot_momentum_*`. The CSV log schema (`btc_microprice`, `btc_bid`, `btc_ask`, ...) does not change, so the running dry run's `logs/ticks.csv` remains a valid bootstrap source after restart.

**Tech Stack:** Python 3.11+, asyncio, websockets, scikit-learn (RidgeCV), aiohttp.

---

## Operational constraints (read first)

1. **The dry run must keep running.** PID 17128 is `python scripts/run_kalshi_bot.py --fresh`, last log write 2026-05-06 18:28:32. Editing files on disk does not affect a running Python process — the bot is using already-imported modules. **DO NOT** kill PID 17128 at any point during this plan. The user will restart manually after the refactor is verified.
2. **Do not touch `logs/`.** `logs/ticks.csv` and `logs/decisions.jsonl` are being actively written by the running bot. Don't delete, don't move, don't rename. WI-1's deletion list does not include anything in `logs/`.
3. **Preserve the uncommitted entry-gate change.** Working tree currently has +7 lines in `betbot/kalshi/config.py` (`DECISION_YES_MID_MIN/MAX`) and +9 lines in `betbot/kalshi/scheduler.py` (the gate use). These are commitable on their own; Task 0 below commits them as a clean baseline before refactor work begins.
4. **Branch is `kalshi`.** Commit each WI as its own commit. Do not bundle WI-1 with WI-3.
5. **Smoke-test policy.** Because we cannot start a second `run_kalshi_bot.py` (would conflict with logs and Kalshi REST polling), end-of-task smoke tests are import-only (`python -c "import betbot.kalshi.scheduler"`) plus targeted greps. The full live restart is the user's call after the plan completes.

---

## File structure (what changes, what stays)

**WI-1 — deleted in full:**
- `betbot/clients/` (whole dir)
- `betbot/state/` (whole dir)
- `betbot/strategy/` (whole dir)
- `betbot/execution/` (whole dir)
- `betbot/infra/` (whole dir)
- `betbot/research/` (whole dir)
- `betbot/cli/` (whole dir)
- `betbot/models/` (whole dir)
- `betbot/main.py`
- `scripts/demo_binance.py`
- `scripts/demo_coinbase.py`
- `RUNBOOK.md` (replaced by DRYRUN.md)

**WI-1 — edited:**
- `betbot/__init__.py` (rewrite docstring + bump version to 0.3.0)
- `pyproject.toml` (rename project polybot → betbot, drop polymarket deps)
- `requirements.txt` (mirror pyproject)
- `.env.example` (strip Polymarket section)

**WI-1 — preserved as-is:** everything under `betbot/kalshi/`, `scripts/run_kalshi_bot.py`, `scripts/test_trade.py`, `scripts/check_kalshi_balance.py`, `scripts/backtest.py`, `scripts/analyze_run.py`, `scripts/visualize_market.py`, `CLAUDE.md`, `DRYRUN.md`, `lag_plot_btc.png`, `betbot/tests/__init__.py`, `logs/`.

**WI-3 — new files:**
- `betbot/kalshi/spot_feed.py` (`SpotFeed` Protocol)
- `betbot/kalshi/binance_feed.py` (`BinanceFeed` class)

**WI-3 — edited:**
- `betbot/kalshi/book.py` (`CoinbaseBook` → `SpotBook`, update docstring)
- `betbot/kalshi/coinbase_feed.py` (update import + constructor type)
- `betbot/kalshi/features.py` (`cb_momentum_*` → `spot_momentum_*`, rename param `cb` → `spot`, type hint update)
- `betbot/kalshi/scheduler.py` (type hint, local var names, bootstrap field names)
- `betbot/kalshi/config.py` (add `SPOT_SOURCE`, `BINANCE_WS`, `BINANCE_PRODUCT`)
- `scripts/run_kalshi_bot.py` (branch on `SPOT_SOURCE`, rename local var)
- `scripts/backtest.py` (rename `cb_mom_30/60` → `spot_mom_30/60`)
- `scripts/visualize_market.py` (adapt per §14.9 — share feed implementations with the bot)
- `.env.example` (add `SPOT_SOURCE`)

---

## Task 0: Commit the in-flight entry-gate change as its own baseline

**Files:**
- Modify (commit only — no new edits): `betbot/kalshi/config.py`, `betbot/kalshi/scheduler.py`

The working tree contains a `DECISION_YES_MID_MIN/MAX` gate that the user wants preserved. Committing it now isolates it from the refactor commits.

- [ ] **Step 1: Verify the diff is exactly the entry-gate change**

```bash
git diff betbot/kalshi/config.py betbot/kalshi/scheduler.py
```

Expected: 7 lines added in config.py defining `DECISION_YES_MID_MIN = 0.10` / `DECISION_YES_MID_MAX = 0.90` plus a comment block; 9 lines added in scheduler.py adding `DECISION_YES_MID_MIN/MAX` to the `from betbot.kalshi.config import (...)` block and inserting a yes_mid range gate before the entry logic.

- [ ] **Step 2: Stage and commit those two files only**

```bash
git add betbot/kalshi/config.py betbot/kalshi/scheduler.py
git commit -m "Add DECISION_YES_MID_MIN/MAX entry gate

Restrict new entries to yes_mid in [0.10, 0.90], strictly inside the
training filter [0.05, 0.95] so the model is interpolating rather than
extrapolating at the boundary when we trade. Open positions still run
exit logic regardless."
```

- [ ] **Step 3: Confirm clean working tree (except for `.planning/` if untracked)**

```bash
git status
```

Expected: nothing in `M` state. `.planning/` may remain untracked (it's a tooling directory).

---

## Task 1: WI-1 — Delete dead `betbot/` subdirectories

**Files:**
- Delete: `betbot/clients/`, `betbot/state/`, `betbot/strategy/`, `betbot/execution/`, `betbot/infra/`, `betbot/research/`, `betbot/cli/`, `betbot/models/`

These directories all contain modules whose imports begin with `from polybot.*` — a package that does not exist in this repo. They are not imported by anything under `betbot/kalshi/` or `scripts/` (verified via `grep -r "from betbot\.(clients|state|strategy|execution|infra|research|cli|models|main)"` returning zero matches).

- [ ] **Step 1: Verify no live module imports from any of these dirs**

```bash
git grep -nE "from betbot\.(clients|state|strategy|execution|infra|research|cli|models|main)|import betbot\.(clients|state|strategy|execution|infra|research|cli|models|main)"
```

Expected: zero matches. If any match appears outside the directories about to be deleted, STOP and investigate before deleting.

- [ ] **Step 2: Delete the dead directories**

```bash
git rm -r betbot/clients betbot/state betbot/strategy betbot/execution betbot/infra betbot/research betbot/cli betbot/models
```

- [ ] **Step 3: Confirm deletions and verify live imports still resolve**

```bash
git status --short | grep "^D " | head -40
python -c "import betbot.kalshi.scheduler; import betbot.kalshi.coinbase_feed; import betbot.kalshi.kalshi_rest_feed; print('imports ok')"
```

Expected: many `D ` lines listing the deleted files; `imports ok` printed.

- [ ] **Step 4: Do NOT commit yet** — Tasks 2–6 form the rest of the WI-1 commit.

---

## Task 2: WI-1 — Delete `betbot/main.py` and dead demo scripts

**Files:**
- Delete: `betbot/main.py`, `scripts/demo_binance.py`, `scripts/demo_coinbase.py`

All three import from the non-existent `polybot.*` package and would crash on import. None are referenced by the live entry point or any other live module.

- [ ] **Step 1: Verify nothing references these files**

```bash
git grep -nE "from betbot\.main|import betbot\.main|demo_binance|demo_coinbase"
```

Expected: zero matches outside the files themselves and CLAUDE.md (historical references in §14 are allowed).

- [ ] **Step 2: Delete the files**

```bash
git rm betbot/main.py scripts/demo_binance.py scripts/demo_coinbase.py
```

- [ ] **Step 3: Confirm imports still resolve**

```bash
python -c "import betbot.kalshi.scheduler; print('ok')"
```

Expected: `ok` printed.

---

## Task 3: WI-1 — Rewrite `betbot/__init__.py`

**Files:**
- Modify: `betbot/__init__.py`

Replace the existing two-line docstring and the `__version__ = "0.2.0"` line.

- [ ] **Step 1: Apply the rewrite**

Use Edit with old_string equal to the current file body and new_string equal to:

```python
"""betbot — Kalshi 15-minute BTC lag-arbitrage bot.

See CLAUDE.md for the full design specification.
"""

__version__ = "0.3.0"
```

- [ ] **Step 2: Verify**

```bash
python -c "import betbot; print(betbot.__version__)"
```

Expected: `0.3.0`.

---

## Task 4: WI-1 — Rewrite `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

Drop the polybot name, the polymarket entry points, and the `py-clob-client` / `web3` / `pyarrow` / `duckdb` / `structlog` deps. Keep `requests` in main deps because `scripts/check_kalshi_balance.py` still imports it (acceptable per CLAUDE.md §14.5).

- [ ] **Step 1: Replace the file contents**

Use Write with this content:

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
  "requests>=2.31.0",
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

- [ ] **Step 2: Verify the file is syntactically valid TOML**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml','rb').read().decode())" && echo ok
```

Expected: `ok`.

---

## Task 5: WI-1 — Trim `requirements.txt`

**Files:**
- Modify: `requirements.txt`

Match `pyproject.toml`. Drop `py-clob-client`, `pyarrow`, `duckdb`, `structlog`. Add `aiohttp`. Keep `requests` for `check_kalshi_balance.py`.

- [ ] **Step 1: Replace file contents**

Use Write with:

```
# Generated from pyproject.toml; do not edit by hand.
# Refresh: pip-compile pyproject.toml -o requirements.txt

# Live data
aiohttp>=3.9.0
websockets>=12.0

# Auth
cryptography>=42.0.0
python-dotenv>=1.0.0
requests>=2.31.0

# Numerics + regression
numpy>=1.24.0
pandas>=2.1.0
scikit-learn>=1.3.0
```

- [ ] **Step 2: Confirm**

```bash
cat requirements.txt
```

Expected: file matches the content above.

---

## Task 6: WI-1 — Strip Polymarket from `.env.example` and delete `RUNBOOK.md`

**Files:**
- Modify: `.env.example`
- Delete: `RUNBOOK.md`

The Polymarket variables (`PRIVATE_KEY`, `API_KEY`, `API_SECRET`, `API_PASSPHRASE`, `CLOB_HOST`, `CHAIN_ID`, `FALLBACK_BALANCE`) are never read by the live bot. `RUNBOOK.md`'s first half references the dead Polymarket/Binance stack; DRYRUN.md is the canonical ops doc.

- [ ] **Step 1: Replace `.env.example` contents**

Use Write with:

```
# Copy this file to .env and fill in real values.
# NEVER commit .env to git — it contains private keys.

# =============================================================================
# KALSHI — primary venue (US-legal, runs from a US machine)
# =============================================================================
# Generate an API key on https://kalshi.com (Account -> API Keys). You get:
#   - an Access Key ID (UUID-ish string)
#   - an RSA private key file (download once, store securely)
#
# The bot signs each REST request with RSA-PSS over `timestamp+method+path`.

KALSHI_API_KEY_ID=YOUR_KALSHI_KEY_ID

# Path to the downloaded RSA private key PEM (no passphrase). Tilde-expansion ok.
KALSHI_PRIVATE_KEY_FILE=~/.kalshi/kalshi_rsa.pem

# Or inline (use \n escapes for newlines). Either FILE or PEM is required.
# KALSHI_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"

# =============================================================================
# OPERATIONAL
# =============================================================================

# true  = Phase 1 (default): log all decisions, place NO real orders
# false = Phase 2: place real orders on Kalshi
DRY_RUN=true

# Where decision JSONL and tick CSV are written.
LOG_DIR=logs
```

- [ ] **Step 2: Delete `RUNBOOK.md`**

```bash
git rm RUNBOOK.md
```

- [ ] **Step 3: Confirm no residual references**

```bash
git grep -n "RUNBOOK\.md\|CLOB_HOST\|CHAIN_ID\|FALLBACK_BALANCE\|API_PASSPHRASE" -- ':!CLAUDE.md' ':!docs/'
```

Expected: zero matches (CLAUDE.md is excluded because it discusses the cleanup historically; `docs/` excluded because this plan file mentions them).

---

## Task 7: WI-1 — Final cleanup verification + commit

**Files:** none (verification only).

- [ ] **Step 1: Run the full §14.11 grep battery**

```bash
git grep -n "polybot"               -- ':!CLAUDE.md' ':!docs/'
git grep -nI "polymarket"           -- ':!CLAUDE.md' ':!docs/' ':!README.md'
git grep -n "py-clob-client"
git grep -n "CLOB_HOST"             -- ':!CLAUDE.md' ':!docs/'
git grep -n "CHAIN_ID"              -- ':!CLAUDE.md' ':!docs/'
git grep -n "from polybot\."
git grep -n "import polybot"
```

Expected: all return zero matches. If any match appears outside CLAUDE.md, fix it before continuing. (`docs/` is excluded because this plan mentions the strings descriptively.)

- [ ] **Step 2: Smoke-test core imports**

```bash
python -c "import betbot; import betbot.kalshi.scheduler; import betbot.kalshi.coinbase_feed; import betbot.kalshi.kalshi_rest_feed; import betbot.kalshi.book; import betbot.kalshi.features; import betbot.kalshi.model; import betbot.kalshi.config; print('all imports ok')"
```

Expected: `all imports ok`.

- [ ] **Step 3: Run pytest (will report no tests collected — expected and fine)**

```bash
python -m pytest betbot/tests 2>&1 | tail -3
```

Expected: either "no tests collected" or "0 passed". Either is fine; we have no tests.

- [ ] **Step 4: Commit WI-1 as one logical commit**

```bash
git add -A
git status --short
git commit -m "Remove Polymarket-era dead code and tidy project metadata

Delete betbot/{clients,state,strategy,execution,infra,research,cli,models}/
and betbot/main.py — all imported from a non-existent polybot.* package
and were unreferenced by the live Kalshi stack.

Delete scripts/demo_binance.py and scripts/demo_coinbase.py (same root
cause). Delete RUNBOOK.md — DRYRUN.md is the canonical ops doc.

Rewrite pyproject.toml and requirements.txt: rename project polybot ->
betbot, bump version to 0.3.0, drop py-clob-client / web3 / pyarrow /
duckdb / structlog, add aiohttp. Keep requests for check_kalshi_balance.

Strip Polymarket variables from .env.example."
```

- [ ] **Step 5: Confirm working tree is clean**

```bash
git status
```

Expected: nothing to commit (untracked `.planning/` is fine).

---

## Task 8: WI-3 — Add `SPOT_SOURCE`, `BINANCE_WS`, `BINANCE_PRODUCT` to `config.py`

**Files:**
- Modify: `betbot/kalshi/config.py`

Add the new constants alongside the existing Coinbase ones. `SPOT_SOURCE` defaults to `"coinbase"` (US-friendly). Validate the value at import time so a typo fails fast.

- [ ] **Step 1: Locate the current Coinbase block**

The file has a section header `# Coinbase (primary spot feed — no VPN needed)` at lines 27–32 with `COINBASE_WS` and `COINBASE_PRODUCT`.

- [ ] **Step 2: Replace that block with a generic spot-feed block**

Edit `betbot/kalshi/config.py`. Find:

```python
# ---------------------------------------------------------------------------
# Coinbase (primary spot feed — no VPN needed)
# ---------------------------------------------------------------------------

COINBASE_WS      = "wss://advanced-trade-ws.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"
```

Replace with:

```python
# ---------------------------------------------------------------------------
# Spot feed (Coinbase or Binance — picks via SPOT_SOURCE env var)
# ---------------------------------------------------------------------------
# Coinbase: US-legal, no VPN required, default.
# Binance:  geo-blocked from US IPs, requires VPN if running from US.
# Both feed the same SpotBook and produce identical features — swapping
# is a config flag with no model retrain required.

SPOT_SOURCE = os.getenv("SPOT_SOURCE", "coinbase").strip().lower()
if SPOT_SOURCE not in ("coinbase", "binance"):
    raise ValueError(
        f"SPOT_SOURCE must be 'coinbase' or 'binance', got {SPOT_SOURCE!r}"
    )

COINBASE_WS      = "wss://advanced-trade-ws.coinbase.com"
COINBASE_PRODUCT = "BTC-USD"

BINANCE_WS       = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"
BINANCE_PRODUCT  = "BTCUSDT"
```

- [ ] **Step 3: Verify**

```bash
python -c "from betbot.kalshi.config import SPOT_SOURCE, COINBASE_WS, BINANCE_WS, BINANCE_PRODUCT; print(SPOT_SOURCE, BINANCE_WS, BINANCE_PRODUCT)"
```

Expected: `coinbase wss://stream.binance.com:9443/ws/btcusdt@bookTicker BTCUSDT`.

- [ ] **Step 4: Confirm typo guard works**

```bash
SPOT_SOURCE=bogus python -c "from betbot.kalshi import config" 2>&1 | tail -3
```

Expected: a `ValueError: SPOT_SOURCE must be 'coinbase' or 'binance', got 'bogus'`.

(On Windows PowerShell: `$env:SPOT_SOURCE="bogus"; python -c "from betbot.kalshi import config"; Remove-Item env:SPOT_SOURCE`.)

---

## Task 9: WI-3 — Rename `CoinbaseBook` → `SpotBook` in `book.py`

**Files:**
- Modify: `betbot/kalshi/book.py`

Pure rename. The class internals are already source-agnostic (microprice math is identical for any L1 ticker source).

- [ ] **Step 1: Edit the file header docstring**

Replace:

```
"""
book.py — Live market state for Coinbase (spot) and Kalshi (betting market).

CoinbaseBook: L1 microprice from ticker channel + 5-min ring buffer of 1Hz samples.
KalshiBook:   yes/no order book reconstructed from WS snapshots/deltas + ring buffer.
"""
```

With:

```
"""
book.py — Live market state for spot (Coinbase or Binance) and Kalshi.

SpotBook:   L1 microprice from any ticker source + 5-min ring buffer of 1Hz samples.
            Fed by either CoinbaseFeed or BinanceFeed; both produce identical features.
KalshiBook: yes/no order book reconstructed from REST polls + ring buffer.
"""
```

- [ ] **Step 2: Rename the class declaration block**

Replace:

```python
# ---------------------------------------------------------------------------
# Coinbase
# ---------------------------------------------------------------------------

class CoinbaseBook:
    """Coinbase L1 state from the ticker channel. Computes microprice."""
```

With:

```python
# ---------------------------------------------------------------------------
# Spot (Coinbase or Binance)
# ---------------------------------------------------------------------------

class SpotBook:
    """L1 spot state from a ticker source. Computes microprice."""
```

- [ ] **Step 3: Verify the file imports cleanly (other modules will still be broken until later tasks)**

```bash
python -c "from betbot.kalshi.book import SpotBook, KalshiBook; print('ok')"
```

Expected: `ok`.

```bash
python -c "from betbot.kalshi.book import CoinbaseBook" 2>&1 | tail -2
```

Expected: `ImportError: cannot import name 'CoinbaseBook' from 'betbot.kalshi.book'`.

(This is the breakage we'll repair in Tasks 10–13.)

---

## Task 10: WI-3 — Define the `SpotFeed` Protocol in a new file

**Files:**
- Create: `betbot/kalshi/spot_feed.py`

A minimal Protocol so any spot feed (Coinbase, Binance, or future ones) plugs into the scheduler the same way.

- [ ] **Step 1: Create the file**

Use Write with:

```python
"""
spot_feed.py — Protocol contract for any spot-price feed.

A SpotFeed is anything that runs in the asyncio loop, ingests upstream
ticker data, and pushes (price, bid, ask, bid_size, ask_size) tuples
into a shared SpotBook via book.apply_ticker(). Concrete implementations
live in coinbase_feed.py and binance_feed.py.
"""

from typing import Protocol


class SpotFeed(Protocol):
    """Minimal contract every spot feed must satisfy."""

    async def run(self) -> None:
        """Run the feed loop until stop() is called. Reconnects on disconnect."""
        ...

    def stop(self) -> None:
        """Signal the run() loop to exit on its next iteration."""
        ...
```

- [ ] **Step 2: Verify**

```bash
python -c "from betbot.kalshi.spot_feed import SpotFeed; print('ok')"
```

Expected: `ok`.

---

## Task 11: WI-3 — Update `coinbase_feed.py` to use `SpotBook`

**Files:**
- Modify: `betbot/kalshi/coinbase_feed.py`

Pure type rename; the Coinbase WS protocol logic is unchanged.

- [ ] **Step 1: Update the module docstring (lines 1–6)**

Replace:

```
"""
coinbase_feed.py — Coinbase Advanced Trade WebSocket ticker feed.

Self-contained: no dependency on betbot.clients.coinbase_ws (which has
broken polybot.* imports). Feeds a CoinbaseBook at up to 20 Hz.
"""
```

With:

```
"""
coinbase_feed.py — Coinbase Advanced Trade WebSocket ticker feed.

Self-contained. Implements the SpotFeed Protocol; pushes ticks into a
shared SpotBook at up to 20 Hz. The unauthenticated `ticker` channel is
sufficient for microprice (best_bid, best_ask, best_bid_quantity,
best_ask_quantity all arrive on every tick).
"""
```

- [ ] **Step 2: Update the import**

Replace:

```python
from betbot.kalshi.book import CoinbaseBook
```

With:

```python
from betbot.kalshi.book import SpotBook
```

- [ ] **Step 3: Update the class docstring and constructor signature**

Replace:

```python
class CoinbaseFeed:
    """
    Subscribes to Coinbase Advanced Trade ticker channel for BTC-USD.
    Applies every tick to the shared CoinbaseBook.
    Reconnects automatically on disconnect.
    """

    MAX_HZ = 20.0   # cap ingest rate; Coinbase fires on every trade

    def __init__(self, book: CoinbaseBook,
                 product: str = COINBASE_PRODUCT):
```

With:

```python
class CoinbaseFeed:
    """
    Subscribes to Coinbase Advanced Trade ticker channel for BTC-USD.
    Applies every tick to the shared SpotBook.
    Reconnects automatically on disconnect.
    """

    MAX_HZ = 20.0   # cap ingest rate; Coinbase fires on every trade

    def __init__(self, book: SpotBook,
                 product: str = COINBASE_PRODUCT):
```

- [ ] **Step 4: Verify**

```bash
python -c "from betbot.kalshi.coinbase_feed import CoinbaseFeed; from betbot.kalshi.book import SpotBook; f = CoinbaseFeed(SpotBook()); print('ok')"
```

Expected: `ok`.

---

## Task 12: WI-3 — Add `BinanceFeed` in a new file

**Files:**
- Create: `betbot/kalshi/binance_feed.py`

Subscribes to Binance's `bookTicker` stream, which fires on every top-of-book change with fields `b` (best bid), `a` (best ask), `B` (bid qty), `A` (ask qty). Maps these into `SpotBook.apply_ticker()` with the same semantics as the Coinbase feed.

- [ ] **Step 1: Create the file**

Use Write with:

```python
"""
binance_feed.py — Binance bookTicker stream for BTC spot.

The bookTicker stream emits one event on every top-of-book change with
the fields needed for microprice: b (bid price), a (ask price), B (bid
qty), A (ask qty). No auth required.

NOTE: Binance is geo-blocked from US IPs. Set SPOT_SOURCE=binance only
if running from outside the US or behind a non-US VPN.
"""

import asyncio
import json
import time

import websockets

from betbot.kalshi.book import SpotBook
from betbot.kalshi.config import BINANCE_WS


class BinanceFeed:
    """
    Subscribes to Binance's bookTicker stream for BTCUSDT.
    Applies every top-of-book change to the shared SpotBook.
    Reconnects automatically on disconnect.
    """

    MAX_HZ = 20.0   # cap ingest rate; bookTicker fires very fast on BTCUSDT

    def __init__(self, book: SpotBook):
        self._book    = book
        self._running = False

    async def run(self) -> None:
        self._running = True
        min_interval = 1.0 / self.MAX_HZ
        last_t = 0.0

        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15,
                ) as ws:
                    async for raw in ws:
                        if not self._running:
                            return
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue

                        bid_s = msg.get("b")
                        ask_s = msg.get("a")
                        bsz_s = msg.get("B")
                        asz_s = msg.get("A")
                        if not (bid_s and ask_s and bsz_s and asz_s):
                            continue

                        # Rate-limit
                        now = time.time()
                        if now - last_t < min_interval:
                            continue
                        last_t = now

                        try:
                            bid = float(bid_s)
                            ask = float(ask_s)
                            bsz = float(bsz_s)
                            asz = float(asz_s)
                            if bid <= 0 or ask <= 0 or bid >= ask:
                                continue
                        except (ValueError, TypeError):
                            continue

                        # Use mid as the trade-price proxy — bookTicker
                        # doesn't carry last_trade.
                        price = (bid + ask) / 2.0
                        self._book.apply_ticker(price, bid, ask, bsz, asz)

            except asyncio.CancelledError:
                return
            except Exception:
                if not self._running:
                    return
                await asyncio.sleep(2)

    def stop(self) -> None:
        self._running = False
```

- [ ] **Step 2: Verify the import resolves and the class satisfies the Protocol**

```bash
python -c "
from betbot.kalshi.binance_feed import BinanceFeed
from betbot.kalshi.spot_feed import SpotFeed
from betbot.kalshi.book import SpotBook
f = BinanceFeed(SpotBook())
# Protocol check is structural — verify the methods exist
assert callable(f.run) and callable(f.stop)
print('ok')
"
```

Expected: `ok`.

---

## Task 13: WI-3 — Rename `cb_momentum_*` → `spot_momentum_*` in `features.py`

**Files:**
- Modify: `betbot/kalshi/features.py`

Drop the Coinbase prefix on the source-agnostic momentum features. The lag features (`x_0`..`x_120`) are already source-agnostic. Kalshi-specific features (`kalshi_spread`, `kalshi_momentum_30s`) stay unchanged.

- [ ] **Step 1: Update the import**

Replace:

```python
from betbot.kalshi.book import CoinbaseBook, KalshiBook
```

With:

```python
from betbot.kalshi.book import SpotBook, KalshiBook
```

- [ ] **Step 2: Update `FEATURE_NAMES`**

Replace:

```python
    "cb_momentum_30s",   # log(mp_now / mp_{t-30s}) — recent spot direction
    "cb_momentum_60s",
```

With:

```python
    "spot_momentum_30s",   # log(mp_now / mp_{t-30s}) — recent spot direction (source-agnostic)
    "spot_momentum_60s",
```

- [ ] **Step 3: Update the `FeatureVec` dataclass**

Replace:

```python
    cb_momentum_30s: float
    cb_momentum_60s: float
```

With:

```python
    spot_momentum_30s: float
    spot_momentum_60s: float
```

- [ ] **Step 4: Update `as_array()`**

Replace:

```python
            self.cb_momentum_30s, self.cb_momentum_60s,
```

With:

```python
            self.spot_momentum_30s, self.spot_momentum_60s,
```

- [ ] **Step 5: Update `build_features()` signature, local vars, and constructor call**

Replace:

```python
def build_features(cb: CoinbaseBook, kb: KalshiBook) -> Optional[FeatureVec]:
    """
    Construct a FeatureVec from current live state.
    Returns None if either book isn't ready or K is unknown.
    """
    if not cb.ready or not kb.ready or kb.floor_strike <= 0:
        return None

    K      = kb.floor_strike
    mp_now = cb.microprice
    tau    = kb.tau_s()

    if mp_now <= 0 or K <= 0:
        return None

    # Lagged microprices — fall back to mp_now during ring warmup so model
    # gets a feature vector (complete=False flags it as unreliable for training)
    mp15  = cb.microprice_at(15)
    mp30  = cb.microprice_at(30)
    mp60  = cb.microprice_at(60)
    mp90  = cb.microprice_at(90)
    mp120 = cb.microprice_at(120)
```

With:

```python
def build_features(spot: SpotBook, kb: KalshiBook) -> Optional[FeatureVec]:
    """
    Construct a FeatureVec from current live state.
    Returns None if either book isn't ready or K is unknown.
    """
    if not spot.ready or not kb.ready or kb.floor_strike <= 0:
        return None

    K      = kb.floor_strike
    mp_now = spot.microprice
    tau    = kb.tau_s()

    if mp_now <= 0 or K <= 0:
        return None

    # Lagged microprices — fall back to mp_now during ring warmup so model
    # gets a feature vector (complete=False flags it as unreliable for training)
    mp15  = spot.microprice_at(15)
    mp30  = spot.microprice_at(30)
    mp60  = spot.microprice_at(60)
    mp90  = spot.microprice_at(90)
    mp120 = spot.microprice_at(120)
```

- [ ] **Step 6: Update the momentum lines**

Replace:

```python
    cb_momentum_30s = _log_ratio(mp_now, mp30 or mp_now)
    cb_momentum_60s = _log_ratio(mp_now, mp60 or mp_now)
```

With:

```python
    spot_momentum_30s = _log_ratio(mp_now, mp30 or mp_now)
    spot_momentum_60s = _log_ratio(mp_now, mp60 or mp_now)
```

- [ ] **Step 7: Update the `complete` line and the FeatureVec constructor**

Replace:

```python
    # complete = ring buffer has at least 30s of real history
    complete = (mp30 is not None) and cb.ready and kb.ready

    return FeatureVec(
        x_0=x_0, x_15=x_15, x_30=x_30, x_60=x_60, x_90=x_90, x_120=x_120,
        tau_s=tau, inv_sqrt_tau=inv_sqrt_tau,
        cb_momentum_30s=cb_momentum_30s, cb_momentum_60s=cb_momentum_60s,
        kalshi_spread=kalshi_spread, kalshi_momentum_30s=kalshi_momentum_30s,
        complete=complete,
    )
```

With:

```python
    # complete = ring buffer has at least 30s of real history
    complete = (mp30 is not None) and spot.ready and kb.ready

    return FeatureVec(
        x_0=x_0, x_15=x_15, x_30=x_30, x_60=x_60, x_90=x_90, x_120=x_120,
        tau_s=tau, inv_sqrt_tau=inv_sqrt_tau,
        spot_momentum_30s=spot_momentum_30s, spot_momentum_60s=spot_momentum_60s,
        kalshi_spread=kalshi_spread, kalshi_momentum_30s=kalshi_momentum_30s,
        complete=complete,
    )
```

- [ ] **Step 8: Verify the file is syntactically valid and renamed cleanly**

```bash
python -c "
from betbot.kalshi.features import FEATURE_NAMES, FeatureVec, build_features
assert 'spot_momentum_30s' in FEATURE_NAMES, FEATURE_NAMES
assert 'spot_momentum_60s' in FEATURE_NAMES, FEATURE_NAMES
assert 'cb_momentum_30s' not in FEATURE_NAMES
assert 'cb_momentum_60s' not in FEATURE_NAMES
print('feature rename ok')
"
```

Expected: `feature rename ok`.

---

## Task 14: WI-3 — Update `scheduler.py` (type hint + bootstrap)

**Files:**
- Modify: `betbot/kalshi/scheduler.py`

Update the import, the type hint on `cb`, the docstring example, and rename `cb_mom_30/60` to `spot_mom_30/60` in `_bootstrap_from_history()`.

- [ ] **Step 1: Update the import**

Replace:

```python
from betbot.kalshi.book import CoinbaseBook, KalshiBook
```

With:

```python
from betbot.kalshi.book import SpotBook, KalshiBook
```

- [ ] **Step 2: Update the class docstring (around line 182)**

Find:

```
    Wires CoinbaseBook + KalshiBook + KalshiRegressionModel + TrainingBuffer
```

Replace with:

```
    Wires SpotBook + KalshiBook + KalshiRegressionModel + TrainingBuffer
```

Find:

```
        sched = Scheduler(cb_book, kb_book, kalshi_feed)
```

Replace with:

```
        sched = Scheduler(spot_book, kb_book, kalshi_feed)
```

- [ ] **Step 3: Update the `__init__` signature type hint**

Find:

```python
    def __init__(self, cb: CoinbaseBook, kb: KalshiBook,
```

Replace with:

```python
    def __init__(self, cb: SpotBook, kb: KalshiBook,
```

(Keep the parameter name `cb` — renaming it would touch dozens of call sites inside the class for no benefit. The type hint is what carries the abstraction.)

- [ ] **Step 4: Update the bootstrap momentum field names**

Find (around line 343):

```python
            cb_mom_30    = math.log(mp_now / mp30) if mp30 > 0 else 0.0
            cb_mom_60    = math.log(mp_now / mp60) if mp60 > 0 else 0.0
```

Replace with:

```python
            spot_mom_30  = math.log(mp_now / mp30) if mp30 > 0 else 0.0
            spot_mom_60  = math.log(mp_now / mp60) if mp60 > 0 else 0.0
```

Find (around line 352, where the FeatureVec is constructed):

```python
                cb_mom_30, cb_mom_60,
```

Replace with:

```python
                spot_mom_30, spot_mom_60,
```

- [ ] **Step 5: Verify any FeatureVec keyword args in scheduler.py match the new names**

```bash
git grep -nE "cb_momentum|cb_mom_30|cb_mom_60" -- betbot/kalshi/scheduler.py
```

Expected: zero matches.

- [ ] **Step 6: Smoke test**

```bash
python -c "import betbot.kalshi.scheduler; print('ok')"
```

Expected: `ok`.

If `_bootstrap_from_history()` builds the FeatureVec via positional args (the existing pattern is positional) the rename in Step 4 is sufficient — the field names aren't named at the call site. If the call uses keyword args, additional kwargs need updating; the smoke test above will fail with `unexpected keyword argument` if so.

---

## Task 15: WI-3 — Wire feed selection in `scripts/run_kalshi_bot.py`

**Files:**
- Modify: `scripts/run_kalshi_bot.py`

Branch on `SPOT_SOURCE`. Rename local `cb_book`/`cb_feed` → `spot_book`/`spot_feed` for clarity, and print which source is in use at startup.

- [ ] **Step 1: Update module docstring**

Replace:

```
"""
run_kalshi_bot.py — Entry point for the Kalshi lead-lag arbitrage bot.

Wires together:
  CoinbaseFeed  -> CoinbaseBook
  KalshiRestFeed -> KalshiBook   (REST polling at 1Hz; no WebSocket)
  Scheduler     (sampler + refitter + decision + window manager)
```

With:

```
"""
run_kalshi_bot.py — Entry point for the Kalshi lead-lag arbitrage bot.

Wires together:
  CoinbaseFeed | BinanceFeed -> SpotBook   (selected by SPOT_SOURCE env var)
  KalshiRestFeed             -> KalshiBook (REST polling at 1Hz)
  Scheduler                  (sampler + refitter + decision + window manager)
```

- [ ] **Step 2: Update the imports**

Replace:

```python
from betbot.kalshi.book import CoinbaseBook, KalshiBook
from betbot.kalshi.coinbase_feed import CoinbaseFeed
from betbot.kalshi.kalshi_rest_feed import KalshiRestFeed
from betbot.kalshi.scheduler import Scheduler, _discover_market, _list_active_markets
from betbot.kalshi.config import KALSHI_KEY_ID
```

With:

```python
from betbot.kalshi.book import SpotBook, KalshiBook
from betbot.kalshi.coinbase_feed import CoinbaseFeed
from betbot.kalshi.binance_feed import BinanceFeed
from betbot.kalshi.kalshi_rest_feed import KalshiRestFeed
from betbot.kalshi.scheduler import Scheduler, _discover_market, _list_active_markets
from betbot.kalshi.config import KALSHI_KEY_ID, SPOT_SOURCE
```

- [ ] **Step 3: Add the source line to the startup banner (around line 70)**

Find:

```python
    print("=== Kalshi BTC Lead-Lag Arbitrage Bot ===", flush=True)
    print(f"  Wallet:     ${args.wallet:,.0f} (simulated)", flush=True)
    print(f"  Log:        {args.log}", flush=True)
    print(f"  Ticks:      {args.ticks}", flush=True)
```

Replace with:

```python
    print("=== Kalshi BTC Lead-Lag Arbitrage Bot ===", flush=True)
    print(f"  Wallet:     ${args.wallet:,.0f} (simulated)", flush=True)
    print(f"  Log:        {args.log}", flush=True)
    print(f"  Ticks:      {args.ticks}", flush=True)
    print(f"  Spot feed:  {SPOT_SOURCE}", flush=True)
```

- [ ] **Step 4: Replace the book + feed instantiation block**

Find (around lines 112–120):

```python
    # ---- Book state ----
    cb_book = CoinbaseBook()
    kb_book = KalshiBook()
    kb_book.set_window(ticker, floor_strike, close_time)

    # ---- Feed clients ----
    pk           = load_private_key()
    cb_feed      = CoinbaseFeed(cb_book)
    kalshi_feed  = KalshiRestFeed(kb_book, key_id=KALSHI_KEY_ID, pk=pk)
```

Replace with:

```python
    # ---- Book state ----
    spot_book = SpotBook()
    kb_book   = KalshiBook()
    kb_book.set_window(ticker, floor_strike, close_time)

    # ---- Feed clients ----
    pk          = load_private_key()
    if SPOT_SOURCE == "coinbase":
        spot_feed = CoinbaseFeed(spot_book)
    elif SPOT_SOURCE == "binance":
        spot_feed = BinanceFeed(spot_book)
    else:
        # config.py validates this on import; this branch is unreachable.
        raise RuntimeError(f"Unknown SPOT_SOURCE: {SPOT_SOURCE!r}")
    kalshi_feed = KalshiRestFeed(kb_book, key_id=KALSHI_KEY_ID, pk=pk)
```

- [ ] **Step 5: Update the Scheduler construction and the print line**

Find:

```python
    scheduler = Scheduler(cb_book, kb_book, kalshi_feed,
                          wallet_usd=args.wallet,
                          log_path=log_path,
                          tick_path=tick_path)

    print("  Starting feeds (Coinbase WS + Kalshi REST 1Hz polling)...\n", flush=True)
```

Replace with:

```python
    scheduler = Scheduler(spot_book, kb_book, kalshi_feed,
                          wallet_usd=args.wallet,
                          log_path=log_path,
                          tick_path=tick_path)

    print(f"  Starting feeds ({SPOT_SOURCE} WS + Kalshi REST 1Hz polling)...\n", flush=True)
```

- [ ] **Step 6: Update the TaskGroup + finally block**

Find:

```python
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(cb_feed.run(),     name="coinbase_feed")
            tg.create_task(kalshi_feed.run(), name="kalshi_rest")
            tg.create_task(scheduler.run(),   name="scheduler")
    except* KeyboardInterrupt:
        pass
    except* asyncio.CancelledError:
        pass
    finally:
        cb_feed.stop()
        kalshi_feed.stop()
        scheduler.stop()
```

Replace with:

```python
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(spot_feed.run(),   name=f"{SPOT_SOURCE}_feed")
            tg.create_task(kalshi_feed.run(), name="kalshi_rest")
            tg.create_task(scheduler.run(),   name="scheduler")
    except* KeyboardInterrupt:
        pass
    except* asyncio.CancelledError:
        pass
    finally:
        spot_feed.stop()
        kalshi_feed.stop()
        scheduler.stop()
```

- [ ] **Step 7: Smoke-test the script imports cleanly without running**

```bash
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', 'scripts/run_kalshi_bot.py')
m = importlib.util.module_from_spec(spec)
# Don't execute; just verify imports inside compile by parsing
import ast; ast.parse(open('scripts/run_kalshi_bot.py').read())
print('ok')
"
```

Expected: `ok`.

---

## Task 16: WI-3 — Rename feature columns in `scripts/backtest.py`

**Files:**
- Modify: `scripts/backtest.py`

Rename the two pandas columns and the `FEATURE_COLS` reference. The raw-tick CSV schema does NOT change — only the derived feature columns produced inside `build_features()`.

- [ ] **Step 1: Rename the two derived columns**

Find (around lines 70–72):

```python
        # Spot momentum
        g["cb_mom_30"] = np.log(mp / mp.shift(30).fillna(mp))
        g["cb_mom_60"] = np.log(mp / mp.shift(60).fillna(mp))
```

Replace with:

```python
        # Spot momentum (source-agnostic — same definition for Coinbase or Binance)
        g["spot_mom_30"] = np.log(mp / mp.shift(30).fillna(mp))
        g["spot_mom_60"] = np.log(mp / mp.shift(60).fillna(mp))
```

- [ ] **Step 2: Update `FEATURE_COLS`**

Find:

```python
FEATURE_COLS = (
    ["x_0"] + [f"x_{l}" for l in LAGS]
    + ["tau_s", "inv_sqrt_tau", "cb_mom_30", "cb_mom_60",
       "kalshi_spread", "kalshi_mom_30"]
)
```

Replace with:

```python
FEATURE_COLS = (
    ["x_0"] + [f"x_{l}" for l in LAGS]
    + ["tau_s", "inv_sqrt_tau", "spot_mom_30", "spot_mom_60",
       "kalshi_spread", "kalshi_mom_30"]
)
```

- [ ] **Step 3: Verify**

```bash
python -c "
import ast
src = open('scripts/backtest.py').read()
ast.parse(src)
assert 'spot_mom_30' in src and 'spot_mom_60' in src
assert 'cb_mom_30' not in src and 'cb_mom_60' not in src
print('ok')
"
```

Expected: `ok`.

---

## Task 17: WI-3 — Adapt `scripts/visualize_market.py` to use the shared feed classes

**Files:**
- Modify: `scripts/visualize_market.py`

Per CLAUDE.md §14.9: keep both Coinbase and Binance code paths but route them through the new feed classes so the visualizer and the live bot share one implementation per feed. Default `--spot` to whatever `SPOT_SOURCE` is set to.

- [ ] **Step 1: Read the file to map the inline WS code to the new feed classes**

```bash
wc -l scripts/visualize_market.py
```

Read the file. Note (a) the `--spot` argparse flag default; (b) any inline Coinbase WS subscription / message parsing; (c) any inline Binance WS subscription / message parsing.

- [ ] **Step 2: Replace inline Coinbase WS code with `CoinbaseFeed(spot_book)`**

For each inline Coinbase WS subscription block, replace it with construction of `CoinbaseFeed(spot_book)` using a `SpotBook` instance. Read tick data off `spot_book.microprice`, `spot_book.best_bid`, etc. Drop any duplicated WS connection / parsing logic.

- [ ] **Step 3: Replace inline Binance WS code with `BinanceFeed(spot_book)`**

Same pattern: replace inline Binance WS code with `BinanceFeed(spot_book)`. Read state off the shared `SpotBook`.

- [ ] **Step 4: Default `--spot` to `SPOT_SOURCE`**

Find the `--spot` argparse flag. Add `default=SPOT_SOURCE` after importing `from betbot.kalshi.config import SPOT_SOURCE`. The CLI flag still overrides.

- [ ] **Step 5: Smoke test the file parses**

```bash
python -c "import ast; ast.parse(open('scripts/visualize_market.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Smoke test the script imports without running**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import scripts.visualize_market  # noqa: F401
print('ok')
" 2>&1 | tail -5
```

Expected: `ok`. If imports fail, fix the cause (typically a missing import or a leftover reference to deleted Coinbase WS code) and re-run.

> Note: if `visualize_market.py` is large and has substantial in-line logic, reading it carefully before editing is essential. The §14.9 instruction explicitly preserves the `--spot {coinbase,binance}` UX — DO NOT delete the Binance branch.

---

## Task 18: WI-3 — Document `SPOT_SOURCE` in `.env.example`

**Files:**
- Modify: `.env.example`

Add a new section so operators know how to flip the spot source.

- [ ] **Step 1: Append to the file**

Find the existing `# OPERATIONAL` section (which already has `DRY_RUN` and `LOG_DIR`). Add a new section above or below it:

```
# =============================================================================
# SPOT FEED
# =============================================================================
# Which upstream price feed to use for BTC spot.
#   coinbase  (default) — US-legal, no VPN required
#   binance              — geo-blocked from US IPs, requires VPN if running from US
# Both produce identical features into the regression; pick whichever has
# lower latency in your region.

SPOT_SOURCE=coinbase
```

- [ ] **Step 2: Verify**

```bash
git grep -n "SPOT_SOURCE" -- .env.example
```

Expected: at least one match showing the new line.

---

## Task 19: WI-3 — Final verification + commit

**Files:** none (verification only).

- [ ] **Step 1: §5.6 acceptance grep — no Coinbase-specific naming should leak past the abstraction**

```bash
git grep -nE "CoinbaseBook|cb_book\b|cb_momentum|cb_mom_30|cb_mom_60" -- betbot/ scripts/
```

Expected: zero matches in `betbot/` and `scripts/`. (CLAUDE.md and `docs/` may still reference the old names historically — that's allowed.)

- [ ] **Step 2: Verify `SPOT_SOURCE` is referenced where expected**

```bash
git grep -n "SPOT_SOURCE" -- betbot/ scripts/ .env.example
```

Expected: matches in `betbot/kalshi/config.py`, `scripts/run_kalshi_bot.py`, and `.env.example` at minimum.

- [ ] **Step 3: Verify the SpotFeed Protocol exists and is referenced where intended**

```bash
git grep -n "from betbot.kalshi.spot_feed import\|SpotFeed" -- betbot/ scripts/
```

Expected: at least one match in `betbot/kalshi/spot_feed.py`. (Concrete feeds satisfy the Protocol structurally; explicit `SpotFeed` imports are optional but a good docstring/type-hint target.)

- [ ] **Step 4: Full import smoke**

```bash
python -c "
import betbot.kalshi.scheduler
import betbot.kalshi.coinbase_feed
import betbot.kalshi.binance_feed
import betbot.kalshi.book
import betbot.kalshi.features
import betbot.kalshi.config
import betbot.kalshi.spot_feed
from betbot.kalshi.book import SpotBook, KalshiBook
from betbot.kalshi.coinbase_feed import CoinbaseFeed
from betbot.kalshi.binance_feed import BinanceFeed
sb = SpotBook()
cf = CoinbaseFeed(sb)
bf = BinanceFeed(sb)
print('all good')
"
```

Expected: `all good`.

- [ ] **Step 5: Confirm `SPOT_SOURCE=coinbase` is the default and bot wiring picks the right feed**

```bash
python -c "
from betbot.kalshi.config import SPOT_SOURCE
assert SPOT_SOURCE == 'coinbase', f'expected coinbase, got {SPOT_SOURCE!r}'
print('default ok')
"
```

Expected: `default ok`.

- [ ] **Step 6: Confirm Binance mode parses and instantiates the right feed (does not connect)**

PowerShell:

```powershell
$env:SPOT_SOURCE="binance"; python -c "
from betbot.kalshi.config import SPOT_SOURCE
assert SPOT_SOURCE == 'binance'
from betbot.kalshi.binance_feed import BinanceFeed
from betbot.kalshi.book import SpotBook
f = BinanceFeed(SpotBook())
print('binance mode ok')
"; Remove-Item env:SPOT_SOURCE
```

Bash:

```bash
SPOT_SOURCE=binance python -c "
from betbot.kalshi.config import SPOT_SOURCE
assert SPOT_SOURCE == 'binance'
from betbot.kalshi.binance_feed import BinanceFeed
from betbot.kalshi.book import SpotBook
f = BinanceFeed(SpotBook())
print('binance mode ok')
"
```

Expected: `binance mode ok`.

- [ ] **Step 7: Confirm the running dry run is still alive (we have not killed it)**

```powershell
Get-CimInstance Win32_Process -Filter "ProcessId = 17128" | Select-Object ProcessId,Name | Format-List
```

Expected: ProcessId 17128 still present. If it died for some other reason, that's fine — but it should not be because of our refactor (we did not touch the running process).

- [ ] **Step 8: Commit WI-3**

```bash
git add -A
git status --short
git commit -m "Introduce pluggable SpotFeed abstraction (Coinbase or Binance)

- Rename CoinbaseBook -> SpotBook in betbot/kalshi/book.py; the class
  was already source-agnostic, only the name was Coinbase-specific.
- Add SpotFeed Protocol (betbot/kalshi/spot_feed.py) describing the
  minimal run()/stop() contract every feed must satisfy.
- Add BinanceFeed (betbot/kalshi/binance_feed.py) using Binance's
  bookTicker stream. Geo-blocked from US IPs; document in env.
- Rename cb_momentum_{30s,60s} -> spot_momentum_{30s,60s} in features
  (FEATURE_NAMES, FeatureVec, build_features). Update scheduler
  bootstrap and scripts/backtest.py to match.
- Add SPOT_SOURCE / BINANCE_WS / BINANCE_PRODUCT to config.py with
  fail-fast validation on import.
- Wire SPOT_SOURCE-based feed selection in scripts/run_kalshi_bot.py.
- Adapt scripts/visualize_market.py to share feed implementations
  with the live bot (no duplicate WS code).
- Document SPOT_SOURCE in .env.example.

The CSV schema in logs/ticks.csv is unchanged, so a running dry run's
historical ticks remain valid for bootstrap after restart."
```

- [ ] **Step 9: Show the final commit log**

```bash
git log --oneline -5
```

Expected: top of history shows WI-3 commit, then WI-1 commit, then the Task-0 entry-gate commit.

---

## What's NOT in this plan (deliberately deferred)

- **WI-2** (cleanup of residual Kalshi-WebSocket naming, §14.10): scoped, but not in user's request for this session.
- **WI-4** (forward-projection features, §6.6): depends on this plan landing first; scoped separately.
- **WI-5** (maker-entry execution, §11.1): Phase-2 territory; explicitly out of scope here.
- **`/ultra-review`**: per CLAUDE.md §15.8/§15.9 mandatory after **all five** WIs are merged. Not run after a partial subset.

After this plan, the running dry run continues collecting data unchanged. The user restarts manually (`Ctrl-C` + `python scripts/run_kalshi_bot.py`) when ready to exercise the new code path; on restart, the bootstrap reads the existing `logs/ticks.csv` (schema unchanged) and is trading-ready as soon as the regression refits (~6 minutes of data is the threshold; we already have 3+ hours).

---

## Self-review notes

**Spec coverage:**
- §14.2 dirs deleted: Task 1.
- §14.3 files deleted: Task 2.
- §14.4 `betbot/__init__.py` rewritten: Task 3.
- §14.5 `pyproject.toml` rewritten: Task 4. (Kept `requests` per §14.5's own caveat.)
- §14.6 `requirements.txt` trimmed: Task 5.
- §14.7 `.env.example` stripped: Task 6.
- §14.8 `RUNBOOK.md` deleted: Task 6.
- §14.11 verification greps: Task 7.
- §14.12 final smoke test: Tasks 7 & 19.
- §5.6 #1 `CoinbaseBook` → `SpotBook`: Task 9.
- §5.6 #2 `SpotFeed` Protocol: Task 10.
- §5.6 #3 `CoinbaseFeed` updated: Task 11.
- §5.6 #4 `BinanceFeed` added: Task 12.
- §5.6 #5 `cb_momentum_*` → `spot_momentum_*`: Task 13.
- §5.6 #6 `SPOT_SOURCE` config: Task 8.
- §5.6 #7 wiring in `run_kalshi_bot.py`: Task 15.
- §5.6 #8 `.env.example`: Task 18.
- §5.6 #9 `backtest.py` rename: Task 16.
- §5.6 #10 tick CSV unchanged: noted in plan preamble.
- §14.9 `visualize_market.py` adaptation: Task 17.

**Type/name consistency check:**
- `SpotBook` referenced consistently across Tasks 9–18.
- `spot_book` used as the local variable name in `run_kalshi_bot.py` (Task 15); inside `Scheduler.__init__` the parameter stays `cb` for blast-radius reasons — only the type hint changed (Task 14). Documented in Task 14 Step 3.
- `spot_momentum_30s`/`spot_momentum_60s` used consistently in features.py (Task 13), scheduler bootstrap (Task 14 — the local vars there are `spot_mom_30/60` for brevity, matching the existing `cb_mom_30/60` naming style of that function — and they're positional in the `FeatureVec(...)` call so they don't need to match field names).
- `spot_mom_30`/`spot_mom_60` columns in `backtest.py` (Task 16) align with the in-memory feature names; backtest is a separate code path with its own feature-engineering function, so the column names don't need to match the live `FeatureVec` field names. They just need to be self-consistent within `backtest.py`.

**Placeholder scan:** none.
