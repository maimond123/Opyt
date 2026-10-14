"""
tests/test_sync_lock.py

The heartbeat thread is what makes all three of `CatchupLock`'s invariants true, so the thing
worth pinning is that it SURVIVES. It wrote to a database shared with ingestion, so
`database is locked` was a routine transient outcome (it now has loops.db to itself, pinned at the
bottom, but a beat can still fail) — and the first one used to kill
the thread outright. Nothing set `_lost` (only an eviction does), so `lost()` answered False
forever while the stamp aged past the TTL and every other process became entitled to reclaim a
lease whose holder was still writing. Measured on the hosted box: two of three locks in
exactly that state, `bookmark-catchup` 1094s stale against a 30s TTL.

These tests separate the two outcomes the old code conflated: a beat that FAILED (retry, we still
hold it as far as anyone knows) and a beat that SAID WE LOST (stop).
"""

import sqlite3
import time

import pytest

from pipeline import sync_lock


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    """Poll rather than sleep a fixed span — these tests run a real thread at a 20ms cadence."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _row(db, name: str):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT holder, heartbeat_at, epoch FROM sync_lock WHERE name=?", (name,)
        ).fetchone()
    finally:
        conn.close()


@pytest.fixture
def db(tmp_path):
    return tmp_path / "loops.db"


def test_a_locked_database_does_not_kill_the_heartbeat(db):
    """THE REGRESSION. Three `database is locked` in a row, then the lock frees: the thread must
    still be alive and the stamp must start advancing again. The old code died on the first one."""
    lock = sync_lock.CatchupLock("bookmark-catchup", ttl=1.0, heartbeat=0.02, db_path=db)
    real_beat = lock.beat
    calls = {"n": 0}

    def flaky(conn=None):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise sqlite3.OperationalError("database is locked")
        return real_beat(conn)

    lock.beat = flaky

    with lock:
        assert lock.acquired
        stamp_at_acquire = _row(db, "bookmark-catchup")[1]
        assert _wait_until(lambda: calls["n"] >= 6), "the thread stopped beating"
        assert lock._thread.is_alive(), "the heartbeat thread died on a transient error"
        # Survival alone is not the point — it has to be beating again.
        assert _wait_until(lambda: _row(db, "bookmark-catchup")[1] > stamp_at_acquire), \
            "the heartbeat never landed again after the database freed up"
        assert not lock.lost(), "a failed write was reported as an eviction"


def test_a_failing_heartbeat_is_never_reported_as_an_eviction(db):
    """`lost()` means 'someone else owns the row now'. A beat we could not write is not evidence
    of that, and the worker stops its pass on `lost()` — so conflating them aborts healthy work."""
    lock = sync_lock.CatchupLock("curation-catchup", ttl=0.1, heartbeat=0.02, db_path=db)
    calls = {"n": 0}

    def always_locked(conn=None):
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    lock.beat = always_locked

    with lock:
        # Well past the TTL, still failing every time — and still not "lost".
        assert _wait_until(lambda: calls["n"] >= 10)
        assert not lock.lost()
        assert lock._thread.is_alive()


def test_a_real_eviction_still_sets_lost(db):
    """The control: retrying must not have cost us the detection the retry is wrapped around.
    A beat that COMPLETES and matches no row means we were fenced out — stop."""
    lock = sync_lock.CatchupLock("oracle-refresh", ttl=1.0, heartbeat=0.02, db_path=db)
    with lock:
        assert lock.acquired
        # Another process reclaims the lease: new holder, bumped epoch.
        thief = sqlite3.connect(str(db))
        thief.execute(
            "UPDATE sync_lock SET holder=?, heartbeat_at=?, epoch=epoch+1 WHERE name=?",
            ("someone-else:1:deadbeef", time.time(), "oracle-refresh"),
        )
        thief.commit()
        thief.close()

        assert _wait_until(lock.lost), "the fenced-out holder never learned it was evicted"


def test_an_eviction_discovered_after_a_failed_beat_is_still_caught(db):
    """The two paths in one: the write fails while we are ALSO being evicted. The failures are
    retried, and the first beat that actually completes reports the eviction."""
    lock = sync_lock.CatchupLock("bookmark-catchup", ttl=1.0, heartbeat=0.02, db_path=db)
    real_beat = lock.beat
    calls = {"n": 0}

    def flaky(conn=None):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise sqlite3.OperationalError("database is locked")
        return real_beat(conn)

    lock.beat = flaky

    with lock:
        thief = sqlite3.connect(str(db))
        thief.execute(
            "UPDATE sync_lock SET holder=?, epoch=epoch+1 WHERE name=?",
            ("someone-else:1:deadbeef", "bookmark-catchup"),
        )
        thief.commit()
        thief.close()

        assert _wait_until(lock.lost), "the eviction was swallowed by the retry path"


def test_an_outage_past_the_ttl_says_so_out_loud(db, monkeypatch):
    """Past the TTL the lease is reclaimable by anyone, so single-flight is off for as long as the
    outage lasts. We keep going — the work is idempotent and a stalled write is not proof anyone
    took the lock — but the whole failure this module had was being SILENT, so it must be said."""
    said: list[str] = []
    monkeypatch.setattr(sync_lock, "_log", said.append)

    def always_locked(conn=None):
        raise sqlite3.OperationalError("database is locked")

    lock = sync_lock.CatchupLock("bookmark-catchup", ttl=0.1, heartbeat=0.02, db_path=db)
    lock.beat = always_locked

    with lock:
        assert _wait_until(lambda: any("may reclaim this lease" in m for m in said)), \
            f"no past-TTL warning was logged; got {said}"
        # Once per outage, not once per beat — at a 20ms cadence that would drown the log.
        time.sleep(0.2)
        assert sum("may reclaim this lease" in m for m in said) == 1, \
            f"the past-TTL warning repeated; got {said}"


def test_a_recovered_heartbeat_says_so_too(db, monkeypatch):
    """The onset is logged once and the recovery is logged once, so an outage has both ends in the
    log. Without the recovery line a transient stall and a permanent one read identically."""
    said: list[str] = []
    monkeypatch.setattr(sync_lock, "_log", said.append)

    lock = sync_lock.CatchupLock("curation-catchup", ttl=1.0, heartbeat=0.02, db_path=db)
    real_beat = lock.beat
    calls = {"n": 0}

    def flaky(conn=None):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise sqlite3.OperationalError("database is locked")
        return real_beat(conn)

    lock.beat = flaky

    with lock:
        assert _wait_until(lambda: any("recovered" in m for m in said)), \
            f"a heartbeat that came back never said so; got {said}"
        assert sum("heartbeat failed" in m for m in said) == 1, \
            f"the onset was logged more than once; got {said}"


def test_a_broken_logger_is_swallowed_rather_than_raised(monkeypatch):
    """`_log` is called from the thread whose only job is to keep beating, so it must never be the
    thing that ends it. Tested on the real `_log` — patching `_log` itself would only prove a
    stand-in behaves, which is not what ships."""
    from pipeline.ingestion import utils

    def explode(message):
        raise RuntimeError("no logger here")

    monkeypatch.setattr(utils, "log", explode)
    sync_lock._log("this must not raise")   # the assertion IS that it returns


# ── a holder that died without releasing (lanes design §6, layer 3) ─────────────
def _plant(db, name: str, holder: str) -> None:
    """A lease row as its holder left it: stamped a moment ago, never released."""
    sync_lock._connect(db).close()
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO sync_lock(name, holder, heartbeat_at, epoch) VALUES (?, ?, ?, 1)",
                 (name, holder, time.time()))
    conn.commit()
    conn.close()


def _dead_pid() -> int:
    import subprocess
    import sys
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_dead_holder_on_this_host_frees_its_lease_at_once(db):
    """SIGKILL, a crash, launchd stopping the worker: nothing ran the release, and the beat is
    seconds old. A worker restart lands the next child in that window, where it finds
    the lease apparently live and skips its pass for a whole cycle."""
    import socket
    _plant(db, "bookmark-catchup", f"{socket.gethostname()}:{_dead_pid()}:deadbeef")

    with sync_lock.CatchupLock("bookmark-catchup", db_path=db) as lock:
        assert lock.acquired


@pytest.mark.parametrize("holder", [
    "{host}:{me}:cafef00d",          # alive: this very process
    "elsewhere.example:{dead}:x1",   # another host: its pids mean nothing here
    "not-a-holder-id",               # unparseable
])
def test_a_holder_not_provably_dead_keeps_its_lease_until_the_ttl(db, holder):
    import os
    import socket
    _plant(db, "bookmark-catchup",
           holder.format(host=socket.gethostname(), me=os.getpid(), dead=_dead_pid()))

    with sync_lock.CatchupLock("bookmark-catchup", db_path=db) as lock:
        assert not lock.acquired


# ── the connection is closed whatever happens (store-writes C9) ─────────────────
class _Conn:
    """Wraps the lock's own connection to see it closed and to fail on cue."""

    def __init__(self, real, fail_on: str):
        self.real, self.fail_on, self.closed = real, fail_on, False

    def execute(self, sql, *args):
        if self.fail_on in sql:
            raise sqlite3.OperationalError("database is locked")
        return self.real.execute(sql, *args)

    def commit(self):
        return self.real.commit()

    def rollback(self):
        return self.real.rollback()

    def close(self):
        self.closed = True
        self.real.close()


def test_a_release_that_raises_still_closes_the_connection(db):
    lock = sync_lock.CatchupLock("bookmark-catchup", heartbeat=0.02, db_path=db)
    with pytest.raises(sqlite3.OperationalError):
        with lock:
            assert lock.acquired
            lock._conn = _Conn(lock._conn, fail_on="holder=NULL")
    assert lock._conn.closed


def test_an_acquire_that_raises_still_closes_the_connection(db):
    lock = sync_lock.CatchupLock("bookmark-catchup", db_path=db)
    lock._conn = _Conn(lock._conn, fail_on="BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.OperationalError):
        with lock:
            pass
    assert lock._conn.closed


def test_is_held_reads_a_live_lease_and_takes_nothing(tmp_path):
    db = tmp_path / "lock.db"
    assert sync_lock.is_held("curation-catchup", db_path=db) is False      # never made
    with sync_lock.CatchupLock("curation-catchup", db_path=db) as lock:
        assert lock.acquired
        assert sync_lock.is_held("curation-catchup", db_path=db) is True
        assert sync_lock.is_held("curation-catchup", db_path=db,
                                 now=time.time() + sync_lock.DEFAULT_TTL + 1) is False
    assert sync_lock.is_held("curation-catchup", db_path=db) is False      # released


# ── loops.db: the leases and the loops' states, off opyt.db ──────────────────────

def _tables(path) -> set[str]:
    conn = sqlite3.connect(str(path))
    try:
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()


def test_leases_and_loop_states_live_in_loops_db(tmp_path, monkeypatch):
    from pipeline.kb import schema
    from pipeline.worker.quota_loop import loop_state
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    schema.connect().close()

    with sync_lock.CatchupLock("x:quota") as lock:
        assert lock.acquired
        loop_state.set_state("x:quota", "idle", now=100.0)
        assert _row(tmp_path / "loops.db", "x:quota")[0] == lock.holder
        assert loop_state.read_state("x:quota")["state"] == "idle"
        assert sync_lock.is_held("x:quota")
    assert not {"sync_lock", "scraper_state"} & _tables(tmp_path / "opyt.db")


def test_a_held_opyt_db_write_lock_does_not_block_a_heartbeat(tmp_path, monkeypatch):
    """Photo writes holding opyt.db's lock for 7-9 s failed 9 lease heartbeats."""
    from pipeline.kb import schema
    from pipeline.worker.quota_loop import loop_state
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    store = schema.connect()
    store.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with sync_lock.CatchupLock("x:quota") as lock:
            assert lock.acquired and lock.beat()
            loop_state.set_state("x:quota", "running")
            loop_state.beat("x:quota")
        assert time.monotonic() - started < 1.0
    finally:
        store.rollback()
        store.close()
