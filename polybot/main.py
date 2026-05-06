"""
main.py — entrypoint for the polybot service.

This will become the asyncio.TaskGroup orchestrator that owns:
  - Binance WS task (clients/binance_ws.py)
  - Polymarket CLOB book WS task (clients/polymarket_ws.py)
  - Polymarket RTDS WS task (clients/polymarket_rtds.py)
  - Scheduler tick task (infra/scheduler.py)
  - Parquet writer task (infra/parquet_writer.py)
  - Refitter task (infra/refitter.py)

Stage 0 placeholder; Stage 1E wires it up.
"""

import sys


def cli_entry() -> None:
    print("polybot main.py is a Stage 0 placeholder. Stage 1E wires the asyncio TaskGroup.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    cli_entry()
