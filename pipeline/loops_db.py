"""
pipeline/loops_db.py — `<home>/loops.db`: the leases quota loops hold (`sync_lock`) and what each
loop is doing (`worker/quota_loop/loop_state`).

Its own file, not `opyt.db`, like `circuit_breaker.db` and `x_rate_meter.db`: a lease heartbeat
lands every 10 s on its own thread and must never wait behind a content write holding opyt.db's
lock. Neither table is ever written in the same transaction as content, and nothing reads them
inside one. Resolved at call time, per home: hosted children each run with their own `$OPYT_HOME`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_path, owner_only


def loops_db_path() -> Path:
    """`<home>/loops.db`, for the home this process runs as now."""
    return opyt_path("loops.db")


def connect(ddl: str, db_path: Path | None = None) -> sqlite3.Connection:
    """Open loops.db (or `db_path`) and run the caller's own idempotent `ddl`. Rows come back
    as `sqlite3.Row`."""
    p = Path(db_path) if db_path is not None else loops_db_path()
    # `sqlite3.connect` raises on a missing parent directory, so create the home first.
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    owner_only(p)
    # busy_timeout first: `journal_mode=WAL` takes a brief exclusive lock and would otherwise
    # error immediately when another process holds one.
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(ddl)
    conn.commit()
    return conn
