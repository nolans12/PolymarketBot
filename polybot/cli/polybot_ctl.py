"""
polybot_ctl.py — Minimal control interface for the running bot.

In Phase 1 (dry run) the only supported operation is checking whether the
process is alive by looking for its PID file or Parquet output.

Usage:
    python -m polybot.cli.polybot_ctl status
    python -m polybot.cli.polybot_ctl kill-switch   # touch /run/polybot/STOP

Phase 2 will add a Unix-domain-socket control channel (pause/resume/status).
"""

import argparse
import os
import sys
from pathlib import Path

from polybot.infra.config import KILL_SWITCH_PATH, PARQUET_DIR


def cmd_status(args: argparse.Namespace) -> None:
    parquet_dir = Path(args.parquet_dir)
    decisions   = list(parquet_dir.glob("decisions/**/*.parquet"))

    if decisions:
        newest = max(decisions, key=lambda p: p.stat().st_mtime)
        import time
        age_s = time.time() - newest.stat().st_mtime
        print(f"OK  — {len(decisions)} decision parquet file(s) found")
        print(f"      Newest: {newest}  ({age_s:.0f}s ago)")
        if age_s > 120:
            print("WARNING: newest file is >120s old — bot may have stopped writing")
    else:
        print("UNKNOWN — no decision parquet files found yet (bot may still be warming up)")

    kill = Path(KILL_SWITCH_PATH)
    if kill.exists():
        print(f"KILL SWITCH ACTIVE: {kill}")
    else:
        print(f"Kill switch: not present ({kill})")


def cmd_kill_switch(args: argparse.Namespace) -> None:
    kill = Path(KILL_SWITCH_PATH)
    try:
        kill.parent.mkdir(parents=True, exist_ok=True)
        kill.touch()
        print(f"Kill switch set: {kill}")
        print("The bot will stop on its next decision tick.")
    except PermissionError:
        print(f"ERROR: cannot write to {kill} (permission denied)")
        print("Try: sudo touch /run/polybot/STOP")
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polybot-ctl", description="PolymarketBot control")
    p.add_argument("--parquet-dir", default=str(PARQUET_DIR))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status",      help="Check if the bot is running")
    sub.add_parser("kill-switch", help="Set the kill switch (graceful stop)")
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    {"status": cmd_status, "kill-switch": cmd_kill_switch}[args.command](args)


if __name__ == "__main__":
    main()
