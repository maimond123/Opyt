"""`write_tx`: one write transaction, and the three things it does that a bare `commit()` doesn't —
wait its turn past `busy_timeout`, never leave a failed write open, and join a caller's transaction
when nested. The many-writers case is in `test_write_concurrency.py`."""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from pipeline.kb import writes


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "opyt.db"
    c = sqlite3.connect(path)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t (x TEXT PRIMARY KEY)")
    c.commit()
    c.close()
    return path


def _open(path, busy_ms: int = 5000) -> sqlite3.Connection:
    c = sqlite3.connect(path)
    c.execute(f"PRAGMA busy_timeout={busy_ms}")
    return c


def _rows(path) -> list[str]:
    c = sqlite3.connect(path)
    try:
        return sorted(r[0] for r in c.execute("SELECT x FROM t"))
    finally:
        c.close()


@pytest.fixture()
def logged(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(writes, "log", lines.append)
    return lines


def _hold_lock(path, seconds: float) -> threading.Thread:
    """Another connection holds the write lock for `seconds`; returns once it has it."""
    held = threading.Event()

    def run():
        holder = sqlite3.connect(path)
        holder.execute("BEGIN IMMEDIATE")
        held.set()
        time.sleep(seconds)
        holder.commit()
        holder.close()

    t = threading.Thread(target=run)
    t.start()
    held.wait()
    return t


# ── one transaction ──────────────────────────────────────────────────────────────────────

def test_commits_on_a_clean_exit(db, logged):
    c = _open(db)
    with writes.write_tx(c, site="t") as conn:
        assert conn is c
        c.execute("INSERT INTO t VALUES ('a')")
        assert _rows(db) == []                 # not visible until the block ends
    assert _rows(db) == ["a"]
    assert not c.in_transaction
    assert logged == []                        # a quick write says nothing


def test_rolls_back_and_reraises_on_any_exception(db):
    c = _open(db)
    with pytest.raises(RuntimeError):
        with writes.write_tx(c):
            c.execute("INSERT INTO t VALUES ('a')")
            raise RuntimeError("boom")
    assert _rows(db) == []
    assert not c.in_transaction


def test_a_failed_write_leaves_the_connection_usable(db):
    """C3. A failed write left open pins a snapshot, so once anyone else commits, every later
    write on that connection fails at once. `write_tx` rolls back, so the next one just works."""
    c = _open(db)
    with writes.write_tx(c):
        c.execute("INSERT INTO t VALUES ('a')")
    with pytest.raises(sqlite3.IntegrityError):
        with writes.write_tx(c):
            c.execute("INSERT INTO t VALUES ('a')")
    other = _open(db)
    with writes.write_tx(other):
        other.execute("INSERT INTO t VALUES ('b')")
    with writes.write_tx(c):
        c.execute("INSERT INTO t VALUES ('c')")
    assert _rows(db) == ["a", "b", "c"]


def test_commits_a_transaction_left_open_and_says_so(db, logged):
    c = _open(db)
    c.execute("INSERT INTO t VALUES ('leaked')")
    assert c.in_transaction
    with writes.write_tx(c, site="next"):
        c.execute("INSERT INTO t VALUES ('mine')")
    assert _rows(db) == ["leaked", "mine"]
    assert len(logged) == 1 and "next" in logged[0] and "left open" in logged[0]


# ── nested ───────────────────────────────────────────────────────────────────────────────

def test_a_nested_write_tx_joins_the_outer_transaction(db, logged):
    c = _open(db)
    with writes.write_tx(c):
        c.execute("INSERT INTO t VALUES ('outer')")
        with writes.write_tx(c):
            c.execute("INSERT INTO t VALUES ('inner')")
        assert _rows(db) == []                 # the inner block committed nothing
    assert _rows(db) == ["inner", "outer"]
    assert logged == []                        # joining is not a leak


def test_a_nested_failure_undoes_only_the_inner_block(db):
    c = _open(db)
    with writes.write_tx(c):
        c.execute("INSERT INTO t VALUES ('before')")
        with pytest.raises(RuntimeError):
            with writes.write_tx(c):
                c.execute("INSERT INTO t VALUES ('inner')")
                raise RuntimeError("boom")
        c.execute("INSERT INTO t VALUES ('after')")
    assert _rows(db) == ["after", "before"]


def test_a_nested_failure_that_reaches_the_outer_rolls_back_everything(db):
    c = _open(db)
    with pytest.raises(RuntimeError):
        with writes.write_tx(c):
            c.execute("INSERT INTO t VALUES ('outer')")
            with writes.write_tx(c):
                c.execute("INSERT INTO t VALUES ('inner')")
                raise RuntimeError("boom")
    assert _rows(db) == []
    with writes.write_tx(c):                   # and the next one is a fresh outer transaction
        c.execute("INSERT INTO t VALUES ('next')")
    assert _rows(db) == ["next"]


# ── after the commit ─────────────────────────────────────────────────────────────────────

def test_after_commit_outside_a_write_tx_runs_now(db):
    c = _open(db)
    ran = []
    writes.after_commit(c, lambda: ran.append("now"))
    assert ran == ["now"]
    assert not writes.in_write_tx(c)


def test_after_commit_inside_runs_once_the_outermost_commits(db):
    c = _open(db)
    ran = []
    with writes.write_tx(c):
        assert writes.in_write_tx(c)
        c.execute("INSERT INTO t VALUES ('a')")
        with writes.write_tx(c):
            c.execute("INSERT INTO t VALUES ('b')")
            writes.after_commit(c, lambda: ran.append(_rows(db)))
        assert ran == []                       # the inner block's end is not durable
    assert ran == [["a", "b"]]                 # ran after the commit, and saw it
    assert not writes.in_write_tx(c)


def test_after_commit_never_runs_for_a_rollback(db):
    c = _open(db)
    ran = []
    with pytest.raises(RuntimeError):
        with writes.write_tx(c):
            writes.after_commit(c, lambda: ran.append("outer"))
            raise RuntimeError("boom")
    with writes.write_tx(c):
        writes.after_commit(c, lambda: ran.append("kept"))
        with pytest.raises(RuntimeError):
            with writes.write_tx(c):
                writes.after_commit(c, lambda: ran.append("undone"))
                raise RuntimeError("boom")
    assert ran == ["kept"]


# ── waiting for the lock ─────────────────────────────────────────────────────────────────

def test_waits_past_busy_timeout_for_its_turn(db, logged):
    """SQLite gives up at `busy_timeout`; `write_tx` asks again until its deadline (audit A)."""
    c = _open(db, busy_ms=100)
    holder = _hold_lock(db, 0.8)
    try:
        with writes.write_tx(c, site="patient"):
            c.execute("INSERT INTO t VALUES ('a')")
    finally:
        holder.join()
    assert _rows(db) == ["a"]
    assert len(logged) == 1
    assert logged[0].startswith("[writes] patient: waited ") and "busy timeouts" in logged[0]


def test_a_lock_error_that_escapes_is_the_plain_transient_one(db, logged, monkeypatch):
    """Past the deadline the caller gets SQLite's own error, so it leaves its unit owed."""
    monkeypatch.setattr(writes, "LOCK_DEADLINE_S", 0.3)
    c = _open(db, busy_ms=50)
    holder = _hold_lock(db, 1.0)
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            with writes.write_tx(c, site="impatient"):
                pytest.fail("the body must not run without the lock")
    finally:
        holder.join()
    assert writes.is_lock_error(caught.value)
    assert not c.in_transaction
    assert "giving up" in logged[-1]


def test_a_long_hold_is_logged(db, logged, monkeypatch):
    monkeypatch.setattr(writes, "SLOW_S", 0.05)
    c = _open(db)
    with writes.write_tx(c, site="slow"):
        c.execute("INSERT INTO t VALUES ('a')")
        time.sleep(0.1)
    assert len(logged) == 1 and "slow" in logged[0] and "held it" in logged[0]


# ── the decisions ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("refused, waited, pause", [
    (1, 0.0, 0.05),            # first refusal: a short pause
    (2, 5.0, 0.1),             # doubling
    (6, 12.0, 1.0),            # capped at 1 s
    (1, writes.LOCK_DEADLINE_S - 0.01, None),   # the next ask would start past the deadline
    (3, writes.LOCK_DEADLINE_S - 0.1, None),
])
def test_next_try_in(refused, waited, pause):
    assert writes.next_try_in(refused, waited) == pause


@pytest.mark.parametrize("exc, transient", [
    (sqlite3.OperationalError("database is locked"), True),
    (sqlite3.OperationalError("database table is locked"), True),
    (sqlite3.OperationalError("database is busy"), True),
    (sqlite3.OperationalError("no such table: t"), False),
    (sqlite3.IntegrityError("UNIQUE constraint failed: t.x"), False),
    (RuntimeError("database is locked"), False),
])
def test_is_lock_error(exc, transient):
    assert writes.is_lock_error(exc) is transient
