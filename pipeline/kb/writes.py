"""
pipeline/kb/writes.py — the one way to write to the store: `with write_tx(conn, site=...):`.

Three rules hold for every write:
  1. Fetch, then write. No network call, LLM call or sleep inside the `with`.
  2. One unit per transaction: whatever must land together is written inside one `with`.
  3. `database is locked` is transient (`is_lock_error`). It never marks a unit failed, done or
     charged to a breaker; the unit stays owed.

What `write_tx` does:
  • Takes the write lock up front with `BEGIN IMMEDIATE`, so a writer never has to upgrade a read,
    which fails at once with no wait.
  • Keeps asking past `busy_timeout`, up to `LOCK_DEADLINE_S`: SQLite has no writer queue.
  • Rolls back on any exception; a failed write left open breaks every later write on it.
  • Commits, and logs, a transaction it finds leaked open on entry.
  • Logs a lock wait or hold of `SLOW_S` or more, as a log line rather than a store write that
    would compete for the lock it measures.

A page and its done-state commit together when the page's writes run inside the caller's
`write_tx`: each nested `write_tx` joins it. `after_commit` is how a writer says "this is now
durable" when it can't know whether it is the outermost transaction.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from typing import Callable

from pipeline.ingestion.utils import log

# How long a writer keeps asking for the lock before the error escapes to its caller. Each ask
# already waits out the connection's `busy_timeout` (5 s from `schema.connect`).
LOCK_DEADLINE_S = 30.0
# A wait for the lock, or a hold of it, this long or longer gets a log line. An atom write holds
# the lock for milliseconds, and `busy_timeout` gives up at 5 s.
SLOW_S = 0.5

# The connections inside a `write_tx` right now, by `id`, so a nested one joins its caller's
# transaction instead of committing it as a leak. Each holds what `after_commit` queued, to run
# once the outermost transaction commits.
_inside: dict[int, list[Callable[[], None]]] = {}


def in_write_tx(conn: sqlite3.Connection) -> bool:
    """Is a `write_tx` open on this connection? A network call must not run while one is."""
    return id(conn) in _inside


def after_commit(conn: sqlite3.Connection, fn: Callable[[], None]) -> None:
    """Run `fn` once what this connection has written so far is durable. Outside a `write_tx`
    that is now. Inside one, it runs after the outermost commit, and never if the transaction (or
    the nested block that queued it) rolls back."""
    pending = _inside.get(id(conn))
    if pending is None:
        fn()
    else:
        pending.append(fn)


def is_lock_error(exc: BaseException) -> bool:
    """Is this SQLite's transient `database is locked` / `busy`, rather than a real fault?"""
    return isinstance(exc, sqlite3.OperationalError) and (
        "locked" in str(exc) or "busy" in str(exc))


def next_try_in(refused: int, waited: float) -> float | None:
    """Seconds to pause before asking for the lock again, after `refused` refusals and `waited`
    seconds in all; None when the next ask would start past `LOCK_DEADLINE_S`. The pause doubles
    from 50 ms to a 1 s cap, so a connection with a short `busy_timeout` doesn't spin."""
    pause = min(0.05 * 2 ** (refused - 1), 1.0)
    return pause if waited + pause < LOCK_DEADLINE_S else None


@contextmanager
def write_tx(conn: sqlite3.Connection, *, site: str | None = None):
    """One write transaction. `site` names the writer in the log. Commits on a clean exit; on any
    exception, rolls back and re-raises. A lock error that escapes is the plain
    `sqlite3.OperationalError`, so `is_lock_error` still reads it as transient.

    Inside another `write_tx` on the same connection, it joins that transaction as a savepoint:
    nothing commits until the outer one does, an exception undoes what this block wrote, and the
    outer one rolls back the rest if the exception reaches it.

    Never commit inside the `with`: the rest of the block would be written outside the
    transaction. What `after_commit` queued runs after the commit, so an exception it raises
    reaches the caller although the transaction has committed."""
    if id(conn) in _inside:
        with _savepoint(conn, _inside[id(conn)]):
            yield conn
        return
    site = site or "write"
    if conn.in_transaction:
        log(f"[writes] {site}: committed a transaction an earlier write left open")
        conn.commit()
    # ── Write fairness would go here ───────────────────────────────────────────────────────────
    # SQLite has no writer queue. Whichever fairness piece is picked wraps everything
    # from `_begin` to the commit below: a FIFO gate or an OS file lock taken before `_begin` and
    # released after the commit, and/or a ~2 ms pause after the commit. Nothing else here changes.
    waited, refused = _begin(conn, site)
    pending = _inside[id(conn)] = []
    took = time.monotonic()
    try:
        yield conn
        conn.commit()
    except BaseException:
        try:
            conn.rollback()
        except sqlite3.Error as exc:
            log(f"[writes] {site}: rollback failed: {exc}")
        raise
    finally:
        del _inside[id(conn)]
    held = time.monotonic() - took
    if waited >= SLOW_S or held >= SLOW_S:
        log(f"[writes] {site}: waited {waited:.2f}s for the write lock "
            f"({refused} busy timeouts), held it {held:.2f}s")
    for fn in pending:
        fn()


def _begin(conn: sqlite3.Connection, site: str) -> tuple[float, int]:
    """`BEGIN IMMEDIATE`, asked again past `busy_timeout` until `next_try_in` says stop. Returns
    the seconds waited and how many asks were refused."""
    start = time.monotonic()
    refused = 0
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            return time.monotonic() - start, refused
        except sqlite3.OperationalError as exc:
            if not is_lock_error(exc):
                raise
            refused += 1
            waited = time.monotonic() - start
            pause = next_try_in(refused, waited)
            if pause is None:
                log(f"[writes] {site}: no write lock after {waited:.1f}s "
                    f"({refused} busy timeouts); giving up")
                raise
            time.sleep(pause)


@contextmanager
def _savepoint(conn: sqlite3.Connection, pending: list[Callable[[], None]]):
    """A nested `write_tx`: part of the outer transaction, undone alone if it raises, along with
    whatever it queued to run after the commit."""
    queued_before = len(pending)
    conn.execute("SAVEPOINT write_tx")
    try:
        yield
    except BaseException:
        del pending[queued_before:]
        # Some errors (a full disk, an I/O error) make SQLite roll the whole transaction back
        # itself, and then there is no savepoint left to return to.
        if conn.in_transaction:
            conn.execute("ROLLBACK TO write_tx")
            conn.execute("RELEASE write_tx")
        raise
    conn.execute("RELEASE write_tx")
