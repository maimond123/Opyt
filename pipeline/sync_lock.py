"""
pipeline/sync_lock.py — a heartbeat-lease lock for the bulk catch-up, one row in loops.db
(`pipeline/loops_db.py`, which says why it is not opyt.db).

A lease, not a mutex, because a SIGKILL'd holder can never release a mutex: the holder re-stamps
a heartbeat every H seconds, and an acquirer treats the lock as dead once the stamp is older than
the TTL (a few × H, so one missed beat doesn't evict).

Three invariants, three mechanisms:
  • no deadlock     — lease expiry: a stale heartbeat is reclaimable. So is a lease whose
                      holder names this host and a process that no longer exists: a killed
                      holder's lease is free at once, not after the TTL (see `_holder_is_dead`).
  • no double-run   — acquire is an atomic compare-and-set (UPDATE … WHERE stale),
                      serialized by SQLite's single writer; exactly one reclaimer wins.
  • no zombie       — epoch fencing: every acquire bumps `epoch`; a heartbeat whose
                      epoch no longer matches the row means we were evicted (a paused
                      holder that woke up) → the worker learns it lost and stops.

All three rest on the heartbeat thread, so `_run_beats` retries on any error and never dies: a
beat that could not be written says nothing about whether we still hold the lock.

Heartbeats are stored as unix seconds (REAL), so staleness is a numeric comparison.
"""

import os
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from pipeline.loops_db import connect as connect_loops
from pipeline.loops_db import loops_db_path

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
    return connect_loops(_SCHEMA, db_path)


def _log(message: str) -> None:
    """Log without ever being a reason the heartbeat stops.

    Imported lazily: `pipeline.ingestion.utils` drags in config/YAML machinery that the small
    tools importing `sync_lock` don't need."""
    try:
        from pipeline.ingestion.utils import log
        log(message)
    except Exception:
        pass


def _new_holder_id() -> str:
    """Unique per acquisition: host + pid + a random tag (pids get reused)."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _holder_is_dead(holder: str | None) -> bool:
    """Does `holder` (`host:pid:tag`) name this host and a process that no longer exists?

    Frees a killed holder's lease at once instead of after the TTL. Only a definite "no such
    process" counts: anything uncertain reads as alive, since a false "dead" would let two
    holders run."""
    if not holder:
        return False
    try:
        host, pid_text, _tag = holder.rsplit(":", 2)
        pid = int(pid_text)
    except ValueError:
        return False
    if host != socket.gethostname() or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def is_held(name: str, *, ttl: float = DEFAULT_TTL, db_path: Path | None = None,
            now: float | None = None) -> bool:
    """Is someone holding `name` right now: a holder whose last beat is inside the TTL? A read
    for a caller that only wants to know, never to take it. False when the lock was never made."""
    p = Path(db_path) if db_path else loops_db_path()
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return False
    try:
        row = conn.execute("SELECT holder, heartbeat_at FROM sync_lock WHERE name=?",
                           (name,)).fetchone()
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()
    if row is None or row[0] is None or row[1] is None or _holder_is_dead(row[0]):
        return False
    return (time.time() if now is None else now) - row[1] < ttl


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
        self._db_path = Path(db_path) if db_path else loops_db_path()
        self.holder = _new_holder_id()
        self.epoch: int | None = None
        self.acquired = False
        self._conn = _connect(self._db_path)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lost = threading.Event()

    # ── core operations ───────────────────────────────────────────────────────

    def acquire(self) -> bool:
        """Atomic compare-and-set: take the lock iff it's free, its lease is stale, or its holder
        is a dead process on this host.

        Returns True and records our epoch on success; False if a live holder has it.
        """
        now = time.time()
        cutoff = now - self.ttl
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")  # grab the write lock up front; serializes racing acquirers
        try:
            conn.execute("INSERT OR IGNORE INTO sync_lock(name) VALUES (?)", (self.name,))
            held = conn.execute("SELECT holder FROM sync_lock WHERE name=?",
                                (self.name,)).fetchone()[0]
            dead = held if _holder_is_dead(held) else None
            cur = conn.execute(
                "UPDATE sync_lock SET holder=?, heartbeat_at=?, epoch=epoch+1 "
                "WHERE name=? AND (holder IS NULL OR heartbeat_at IS NULL OR heartbeat_at < ? "
                "OR holder = ?)",
                (self.holder, now, self.name, cutoff, dead),
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
        # Own the connection in this thread — SQLite connections can't cross threads.
        conn = _connect(self._db_path)
        last_proof = time.time()   # when we last proved the row still names us
        failing = False            # log the onset and the recovery, never every tick
        warned = False             # the past-TTL warning is once per outage, not once per beat
        try:
            # wait() returns True when stopped (clean exit) or False on timeout (time to beat)
            while not self._stop.wait(self.heartbeat):
                try:
                    still_ours = self.beat(conn)
                except Exception as exc:
                    # A failed beat is not a lost lease: `database is locked` is a normal transient
                    # on a busy box. Retry on any error; a real eviction still arrives when the
                    # next completed beat reads rowcount 0.
                    if not failing:
                        failing = True
                        _log(f"[sync-lock] {self.name}: heartbeat failed, retrying — {exc}")
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    if not warned and time.time() - last_proof > self.ttl:
                        warned = True
                        # Past the TTL anyone may reclaim the lease. The pass continues (the work
                        # is idempotent), but the risk must not be silent.
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
        # `__exit__` never runs for an `__enter__` that raised, so a failed acquire (a locked
        # store past the busy timeout) closes the connection here or leaks it.
        try:
            acquired = self.acquire()
        except BaseException:
            self._conn.close()
            raise
        if acquired:
            self._thread = threading.Thread(target=self._run_beats, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=self.heartbeat + 2)
            if self.acquired:
                self.release()
        finally:
            self._conn.close()


