"""service/usage.py — the per-day read meter, `usage_daily`.

One row per (day, owner, reader, tool), counting reads; a counter, not an event log. The query is
not a column and must never become one.
"""
from __future__ import annotations

import sqlite3

from service import db


def total_reads(conn: sqlite3.Connection, owner: str) -> int:
    """Every read this owner's knowledge base has served; the push gate's demand watermark.
    Not atomic with the caller's write, which costs at most one redundant push."""
    return conn.execute("SELECT COALESCE(SUM(n), 0) FROM usage_daily WHERE owner = ?",
                        (owner,)).fetchone()[0]


def record_usage(owner: str, reader_sha256: str, tool: str, *, zero_results: bool = False) -> None:
    """One increment per served request, aggregated to the day; never the query or the time.
    Called after the read runs, so a failed request spends no allowance."""
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO usage_daily (day, owner, reader, tool, n, zero_results) "
            "VALUES (date('now'), ?, ?, ?, 1, ?) "
            "ON CONFLICT(day, owner, reader, tool) "
            "DO UPDATE SET n = n + 1, zero_results = zero_results + excluded.zero_results",
            (owner, reader_sha256, tool, 1 if zero_results else 0),
        )
        conn.commit()
    finally:
        conn.close()
