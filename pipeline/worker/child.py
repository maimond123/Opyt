"""
pipeline/worker/child.py — the one entry point for every process the rail worker launches.

    python -m pipeline.worker.child <module:function>

Every job the worker runs exports a `run_*` that takes no arguments it needs and returns
`{"status": ...}`. What a child does around it is the same for every job, so it is written once,
here:
  1. turn SIGTERM into a normal exit, so every `with CatchupLock(...)` and `finally` on the way
     out runs, and the job's lease is released rather than left to age out;
  2. load the user's keys (`load_rail_env`);
  3. run the job;
  4. print its result as JSON, which is the job's log; a job that raises gets
     `{"status": "error", "error": "<Type>: <message>"}`;
  5. turn its status into the exit code, through the one table (`exit_codes.exit_code_for`).

What it leaves to the job: leases and the model preflight.

The stop signal is set only here, by `main`: only a process's main thread may set a signal
handler, so nothing a job calls may set it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
from contextlib import suppress
from typing import Callable

from dotenv import load_dotenv

from opyt_core.paths import opyt_path

from .exit_codes import EXIT_TERMINATED, exit_code_for


class Terminated(SystemExit):
    """SIGTERM, raised in the main thread. A `SystemExit`, so the jobs' `except Exception`
    fallbacks let it through, and it ends the process without a traceback."""


def _on_sigterm(signum, frame) -> None:
    raise Terminated(EXIT_TERMINATED)


def install_stop_handler() -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)


def load_rail_env() -> None:
    """Load the user's keys into this process before its job runs; a child inherits only the
    worker's environment. `override=True` so a rotated key on disk beats a stale inherited copy.
    An unreadable file is ignored."""
    with suppress(Exception):
        load_dotenv(opyt_path(".env"), override=True)


def _target(spec: str) -> Callable:
    module, _, name = spec.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:function, got {spec!r}")
    return getattr(importlib.import_module(module), name)


def _run(name: str, run: Callable[[], dict]) -> int:
    """The one body every job runs through: run it, print its result, map it to the exit code.
    A job that raises gets the error result printed and exits 1. What a job records in its own
    ledger stays the job's."""
    try:
        res = run()
    except Terminated:
        print(f"[rail-child] {name}: stopped by SIGTERM", file=sys.stderr, flush=True)
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[rail-child] {name}: {error}", file=sys.stderr, flush=True)
        res = {"status": "error", "error": error}
    print(json.dumps(res, indent=2, default=str), flush=True)
    return exit_code_for(res)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one rail-worker job in this process")
    ap.add_argument("run", help="the job's run_* function, as module:function")
    args = ap.parse_args(argv)

    # First, before the imports below can take seconds: a restart may come at any moment.
    install_stop_handler()
    load_rail_env()
    return _run(args.run, _target(args.run))


if __name__ == "__main__":
    raise SystemExit(main())
