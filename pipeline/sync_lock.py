"""
pipeline/sync_lock.py

A heartbeat-lease lock for the bulk catch-up, backed by one row in opyt.db.

The catch-up subprocess holds this while bulk-embedding. If it's SIGKILL'd
(laptop sleeps, OOM) it can never release a normal mutex — every future session
would deadlock. So this is a *lease*, not a mutex: the holder re-stamps a
heartbeat every H seconds, and a would-be acquirer treats the lock as dead once
the stamp is older than the TTL (T = a few × H, so one missed beat from a GC
pause doesn't cause a false eviction).

Three invariants, three mechanisms:
  • no deadlock     — lease expiry: a stale heartbeat is reclaimable.
  • no double-run   — acquire is an atomic compare-and-set (UPDATE … WHERE stale),
                      serialized by SQLite's single writer; exactly one reclaimer wins.
  • no zombie       — epoch fencing: every acquire bumps `epoch`; a heartbeat whose
                      epoch no longer matches the row means we were evicted (a paused
                      holder that woke up) → the worker learns it lost and stops.

Heartbeats are stored as unix seconds (REAL), not ISO text, so staleness is a
clean numeric comparison (ISO strings don't sort correctly when microseconds are
sometimes omitted).
"""

import os
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from pipeline.sqlite_db import default_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_lock (
    name         TEXT PRIMARY KEY,
    holder       TEXT,        -- unique id of the holding acquisition (NULL = free)
    heartbeat_at REAL,        -- unix seconds, bumped every H while alive
    epoch        INTEGER NOT NULL DEFAULT 0  -- fencing token, ++ on every acquire
);
"""

DEFAULT_TTL = 30.0        # seconds of silence before the lease is presumed dead
DEFAULT_HEARTBEAT = 10.0  # seconds between heartbeats (TTL = 3×, tolerates a missed beat)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _new_holder_id() -> str:
    """Unique per acquisition: host + pid + a random tag (pids get reused)."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class CatchupLock:
    """Single-flight heartbeat lease over the named lock row.

    Typical use — the catch-up worker that must not run twice at once::

        with CatchupLock() as lock:
            if not lock.acquired:
                return                 # someone else is catching up — skip (single-flight)
            for unit in work:
                if lock.lost():        # evicted (we stalled past TTL) — stop, don't double-write
                    break
                do_idempotent_work(unit)
    """

    def __init__(self, name: str = "catchup", ttl: float = DEFAULT_TTL,
                 heartbeat: float = DEFAULT_HEARTBEAT, db_path: Path | None = None):
        self.name = name
        self.ttl = ttl
        self.heartbeat = heartbeat
        self._db_path = Path(db_path) if db_path else default_db_path()
        self.holder = _new_holder_id()
        self.epoch: int | None = None
        self.acquired = False
        self._conn = _connect(self._db_path)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lost = threading.Event()

    # ── core operations ───────────────────────────────────────────────────────

    def acquire(self) -> bool:
        """Atomic compare-and-set: take the lock iff it's free or its lease is stale.

        Returns True and records our epoch on success; False if a live holder has it.
        """
        now = time.time()
        cutoff = now - self.ttl
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")  # grab the write lock up front; serializes racing acquirers
        try:
            conn.execute("INSERT OR IGNORE INTO sync_lock(name) VALUES (?)", (self.name,))
            cur = conn.execute(
                "UPDATE sync_lock SET holder=?, heartbeat_at=?, epoch=epoch+1 "
                "WHERE name=? AND (holder IS NULL OR heartbeat_at IS NULL OR heartbeat_at < ?)",
                (self.holder, now, self.name, cutoff),
            )
            if cur.rowcount == 1:
                self.epoch = conn.execute(
                    "SELECT epoch FROM sync_lock WHERE name=?", (self.name,)
                ).fetchone()[0]
                conn.commit()
                self.acquired = True
                return True
            conn.commit()
            return False
        except Exception:
            conn.rollback()
            raise

    def beat(self, conn: sqlite3.Connection | None = None) -> bool:
        """Bump our heartbeat. Returns False if we no longer hold the lock (fenced
        out: holder or epoch changed) — the worker should stop on a False.

        SQLite connections are thread-bound, so the background heartbeat passes
        its own thread-owned connection; direct callers default to self._conn.
        """
        conn = conn or self._conn
        cur = conn.execute(
            "UPDATE sync_lock SET heartbeat_at=? WHERE name=? AND holder=? AND epoch=?",
            (time.time(), self.name, self.holder, self.epoch),
        )
        conn.commit()
        return cur.rowcount == 1

    def release(self) -> None:
        """Free the lock iff it's still ours (epoch-checked, so we never clobber a
        successor that fenced us out)."""
        cur = self._conn.execute(
            "UPDATE sync_lock SET holder=NULL, heartbeat_at=NULL "
            "WHERE name=? AND holder=? AND epoch=?",
            (self.name, self.holder, self.epoch),
        )
        self._conn.commit()
        self.acquired = False
        return cur.rowcount == 1

    def lost(self) -> bool:
        """True once the background heartbeat discovered we were evicted."""
        return self._lost.is_set()

    # ── background heartbeat ──────────────────────────────────────────────────

    def _run_beats(self) -> None:
        # Own the connection in THIS thread — SQLite connections can't cross threads.
        conn = _connect(self._db_path)
        try:
            # wait() returns True when stopped (clean exit) or False on timeout (time to beat)
            while not self._stop.wait(self.heartbeat):
                if not self.beat(conn):
                    self._lost.set()  # fenced out — surface to the worker via lost()
                    return
        finally:
            conn.close()

    def __enter__(self) -> "CatchupLock":
        if self.acquire():
            self._thread = threading.Thread(target=self._run_beats, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.heartbeat + 2)
        if self.acquired:
            self.release()
        self._conn.close()


