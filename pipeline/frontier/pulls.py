"""
pipeline/frontier/pulls.py — the pull ledger: each (query, source) pair's watermark, last status
and last try.

`frontier_query_sources` holds them; the searches source (`searching.py`) is its only writer.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from pipeline.kb.writes import write_tx
from pipeline.timeparse import parse_ts, utc_iso, utc_now


def watermarks(conn) -> dict[tuple[str, str], sqlite3.Row]:
    """Every pair's watermark row, keyed `(query_id, source)`: one read for a whole pass."""
    return {(r["query_id"], r["source"]): r
            for r in conn.execute("SELECT * FROM frontier_query_sources")}


def get_pair(conn, query_id: str, source: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM frontier_query_sources WHERE query_id=? AND source=?",
        (query_id, source)).fetchone()


def record_pull(conn, query_id: str, source: str, *, last_status: str,
                stamp: bool = True, now: datetime | None = None):
    """Write the outcome of one pair's pull.

    `stamp=False` (a failed pull) leaves `last_pulled_at` alone, so a failure never buys a full
    TTL of silence. Every pull stamps `tried_at`, so a failed one waits before the next try.

    Its own write transaction, or a savepoint inside the caller's: the searches source stamps a
    page's pair in the same transaction that stages the page.
    """
    tried_at = utc_iso(now or utc_now())
    stamp_at = tried_at if stamp else None
    with write_tx(conn, site="frontier_query_sources.pull"):
        row = get_pair(conn, query_id, source)
        if row is None:
            conn.execute(
                "INSERT INTO frontier_query_sources "
                "(query_id, source, last_pulled_at, last_status, tried_at) VALUES (?,?,?,?,?)",
                (query_id, source, stamp_at, last_status, tried_at))
        else:
            conn.execute(
                "UPDATE frontier_query_sources SET "
                "  last_pulled_at = COALESCE(?, last_pulled_at),"
                "  last_status    = ?, tried_at = ? "
                "WHERE query_id=? AND source=?",
                (stamp_at, last_status, tried_at, query_id, source))


# The statuses of a try that sent a request: what the searches' hourly allowance counts.
REQUESTED = ("ok", "empty", "error")


def requests_since(conn, since: datetime) -> list[datetime]:
    """When each request since `since` went out: one per pair, since a pair is tried at most once
    an hour."""
    rows = conn.execute(
        "SELECT tried_at FROM frontier_query_sources WHERE tried_at > ? "
        f"AND last_status IN ({','.join('?' * len(REQUESTED))})", (utc_iso(since), *REQUESTED))
    return [at for (raw,) in rows if (at := parse_ts(raw)) is not None]
