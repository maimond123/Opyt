"""
pipeline/worker/quota_loop/loop_state.py — the `scraper_state` table (name kept: it is durable):
what each quota loop is doing now, as of its last beat.

One row per loop, keyed by the quota it owns: what it is doing (running, idle, waiting for its
quota, paused, backing off), why, and its heartbeat. Only the loop writes it; `progress` and
`check_status` read it, for display. It lives in loops.db beside the leases
(`pipeline/loops_db.py`), so a beat never waits on opyt.db's write lock; each call opens its own
connection, so any thread may call it. Unix seconds, like `rail_jobs` and `sync_lock`: it is
liveness data, compared as numbers. Until a loop has run, a read answers as if it never had.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass

from pipeline.kb.writes import write_tx
from pipeline.loops_db import connect, loops_db_path

BEAT_S = 30.0                   # heartbeat while sleeping; also the longest single sleep
STALE_BEAT_S = 120.0            # a loop silent this long is not running

_DDL = """
CREATE TABLE IF NOT EXISTS scraper_state (
  scraper   TEXT PRIMARY KEY,        -- the quota it owns, e.g. 'x:UserOriginalsTimeline'
  state     TEXT NOT NULL,           -- running | idle | waiting | paused | backoff
  cause     TEXT,                    -- why it is not running, e.g. 'x_session'; NULL when it is
  since     REAL NOT NULL,           -- when this state and cause began
  retry_at  REAL,                    -- when it next tries; NULL = when woken
  beat_at   REAL NOT NULL            -- last sign of life
);
"""


@dataclass(frozen=True)
class Status:
    """The loop's state, next try, and when it last stopped running units."""
    state: str
    cause: str | None = None
    until: float | None = None
    since: float | None = None

    def transition(self, state: str, cause: str | None, until: float | None,
                   now: float) -> Status:
        """Waiting states share an idle interval; running starts a new busy interval."""
        since = None if state == "running" else (self.since if self.since is not None else now)
        return Status(state, cause, until, since)

    def in_flight(self) -> Status:
        """A pool still draining fetches is busy even when admission has paused."""
        return Status(self.state, self.cause, self.until)


def set_state(scraper: str, state: str, *, cause: str | None = None,
              retry_at: float | None = None, now: float | None = None) -> None:
    """Record what a scraper is doing, and beat. `since` keeps its value while the state and the
    cause are unchanged, so "paused since 14:02" survives every re-check that finds it still
    paused."""
    now = time.time() if now is None else now
    with closing(connect(_DDL)) as conn, write_tx(conn, site="scraper_state"):
        conn.execute(
            "INSERT INTO scraper_state (scraper, state, cause, since, retry_at, beat_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(scraper) DO UPDATE SET "
            "  since=CASE WHEN scraper_state.state=excluded.state "
            "             AND scraper_state.cause IS excluded.cause "
            "        THEN scraper_state.since ELSE excluded.since END, "
            "  state=excluded.state, cause=excluded.cause, retry_at=excluded.retry_at, "
            "  beat_at=excluded.beat_at",
            (scraper, state, cause, now, retry_at, now))


def beat(scraper: str, *, now: float | None = None) -> None:
    with closing(connect(_DDL)) as conn, write_tx(conn, site="scraper_state"):
        conn.execute("UPDATE scraper_state SET beat_at=? WHERE scraper=?",
                     (time.time() if now is None else now, scraper))


def read_state(scraper: str) -> dict | None:
    """None when this scraper has never run, which is also what a home without loops.db says."""
    try:
        conn = sqlite3.connect(f"file:{loops_db_path()}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM scraper_state WHERE scraper=?", (scraper,)).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return None
        raise
    finally:
        conn.close()
    return dict(rows[0]) if rows else None


def is_beating(beat_at: float | None, now: float) -> bool:
    """Is a loop that last beat at `beat_at` still running? A loop sleeps at most `BEAT_S`
    between beats, so silence past `STALE_BEAT_S` means it stopped."""
    return beat_at is not None and now - beat_at <= STALE_BEAT_S
