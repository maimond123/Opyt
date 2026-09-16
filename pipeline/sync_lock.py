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

All three rest on the heartbeat thread, so that thread must OUTLIVE ITS OWN ERRORS: it writes to
a database shared with ingestion, where `database is locked` is a routine transient outcome, and
a beat that could not be written says nothing about whether we still hold the lock. `_run_beats`
therefore retries and never dies — see the long note there for what it cost when it did not.

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


def _log(message: str) -> None:
    """Log without ever being a reason the heartbeat stops.

    Imported lazily because `pipeline.ingestion.utils` drags in config/YAML machinery this module
    otherwise has no use for, and `sync_lock` is imported by very small tools. Swallowing the
    failure is deliberate: this runs on the thread whose whole job is to keep beating, and a
    logging hiccup taking that down would reintroduce, through the back door, exactly the failure
    the caller of this function exists to prevent."""
    try:
        from pipeline.ingestion.utils import log
        log(message)
    except Exception:
        pass


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
        last_proof = time.time()   # when we last PROVED the row still names us
        failing = False            # log the onset and the recovery, never every tick
        warned = False             # the past-TTL warning is once per outage, not once per beat
        try:
            # wait() returns True when stopped (clean exit) or False on timeout (time to beat)
            while not self._stop.wait(self.heartbeat):
                try:
                    still_ours = self.beat(conn)
                except Exception as exc:
                    # A BEAT THAT FAILED IS NOT A BEAT THAT SAID WE LOST, and the difference is
                    # the whole reason this `except` exists. `beat` writes to a database it shares
                    # with ingestion, whose transactions outlast the 5s `busy_timeout` here, so
                    # `database is locked` is a NORMAL transient outcome of a busy box. This
                    # clause used to be absent: the first such error killed the thread, and with
                    # it every future beat. `_lost` was never set (only an eviction sets it), so
                    # `lost()` answered False forever while the stamp aged past the TTL and every
                    # other process became entitled to reclaim a lease we were still using —
                    # `no double-run` silently off, with a live holder and no way to notice.
                    # Measured on the hosted box 2026-09-16: two of three locks in exactly this
                    # state, `bookmark-catchup` 1094s stale against a 30s TTL with its writer
                    # still paginating X.
                    #
                    # So: retry. Dying is never better than retrying, which is why this catches
                    # `Exception` and not just `sqlite3.Error` — an unforeseen error here costs a
                    # guarantee, and no unforeseen error is worth that. A genuine eviction still
                    # arrives through the normal path, because the NEXT beat that completes reads
                    # rowcount 0 and sets `_lost` then.
                    if not failing:
                        failing = True
                        _log(f"[sync-lock] {self.name}: heartbeat failed, retrying — {exc}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if not warned and time.time() - last_proof > self.ttl:
                        warned = True
                        # Past the TTL the lease is reclaimable by anyone, so single-flight is
                        # no longer guaranteed for as long as this lasts. We do NOT stop the pass
                        # over it: the work is idempotent by this module's contract, a stalled
                        # write is not evidence anyone actually took the lock, and aborting a
                        # healthy pass on a transient stall trades a rare double-run for a common
                        # half-finished one. But it must not be silent.
                        _log(f"[sync-lock] {self.name}: no heartbeat has landed in "
                             f"{time.time() - last_proof:.0f}s (TTL {self.ttl:.0f}s) — another "
                             f"process may reclaim this lease while we are still holding it")
                    continue
                if failing:
                    _log(f"[sync-lock] {self.name}: heartbeat recovered after "
                         f"{time.time() - last_proof:.0f}s")
                    failing = warned = False
                if not still_ours:
                    self._lost.set()  # fenced out — surface to the worker via lost()
                    return
                last_proof = time.time()
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


