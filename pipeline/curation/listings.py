"""Durable collection generations, page observations and allowed completion transitions.

A page commits its records, observed identities, successor pages and acceptance together.
Retry policy belongs to the quota loop; exhausted pages remain owed here. Collection readiness
is separate from publication: a caller checks `require_ready`, publishes signals, then calls
`complete`, which checks again inside its own transaction.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from pipeline.kb.writes import write_tx
from pipeline.platforms.collection_page import Incomplete, Page, continuation
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop.source import Changed, Retry

# A walk's states, and the only moves between them. This module is the only writer of `state`.
WALKING, READY, COMPLETE, SUPERSEDED = "walking", "ready", "complete", "superseded"
TRANSITIONS: dict[str, frozenset[str]] = {
    WALKING: frozenset({READY, SUPERSEDED}),      # every page settled; or another account's walk began
    READY: frozenset({COMPLETE, SUPERSEDED}),     # published; or another account's walk began
    COMPLETE: frozenset(),
    SUPERSEDED: frozenset(),
}
OPEN = (WALKING, READY)     # a walk that still owes work

_DDL = """
CREATE TABLE IF NOT EXISTS curation_collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collector TEXT NOT NULL, account TEXT NOT NULL, started_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('walking', 'ready', 'complete', 'superseded')),
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS curation_collection_heads (
    collector TEXT PRIMARY KEY, account TEXT NOT NULL,
    current INTEGER NOT NULL, accepted INTEGER
);
CREATE TABLE IF NOT EXISTS curation_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection INTEGER NOT NULL, operation TEXT NOT NULL, partition TEXT NOT NULL,
    cursor TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
    status TEXT, failures INTEGER, retry_at TEXT, last_error TEXT,
    UNIQUE(collection, operation, partition, cursor)
);
CREATE INDEX IF NOT EXISTS curation_pages_pending ON curation_pages(collection, done, id);
CREATE TABLE IF NOT EXISTS curation_observations (
    collection INTEGER NOT NULL, operation TEXT NOT NULL, partition TEXT NOT NULL,
    key TEXT NOT NULL, data TEXT NOT NULL,
    PRIMARY KEY(collection, operation, partition, key)
);
CREATE TABLE IF NOT EXISTS curation_seen (
    collection INTEGER NOT NULL, operation TEXT NOT NULL, partition TEXT NOT NULL,
    key TEXT NOT NULL,
    PRIMARY KEY(collection, operation, partition, key)
);
"""


@dataclass(frozen=True)
class Walk:
    id: int
    collector: str
    account: str
    started_at: str
    state: str


@dataclass(frozen=True)
class Branch:
    """An endpoint/partition to walk, such as ListMembers for one discovered list ID."""
    operation: str
    partition: str = ""
    cursor: str | None = None


@dataclass(frozen=True)
class Task:
    id: int
    collection: int
    branch: Branch
    retry: Retry


def init_schema(conn: sqlite3.Connection) -> None:
    # executescript commits a caller's transaction, so initialization is entry-only.
    if conn.in_transaction:
        raise RuntimeError("initialize collection tables before starting a transaction")
    conn.executescript(_DDL)


def current(conn: sqlite3.Connection, collector: str) -> Walk | None:
    """An uninitialized home has no collection, and is safe to inspect from a tool."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='curation_collection_heads'").fetchone():
        return None
    row = conn.execute(
        "SELECT c.id,c.collector,c.account,c.started_at,c.state FROM curation_collections c "
        "JOIN curation_collection_heads h ON h.current=c.id WHERE h.collector=?",
        (collector,)).fetchone()
    return Walk(*row) if row else None


def require_current(conn: sqlite3.Connection, walk: Walk) -> Walk:
    active = current(conn, walk.collector)
    if active is None or active.id != walk.id or active.account != walk.account:
        raise Changed("collection generation or connected account changed")
    return active


def _move(conn: sqlite3.Connection, walk: Walk, to: str, **columns) -> None:
    """Change a walk's state along an allowed transition, with any columns that move with it."""
    if to not in TRANSITIONS[walk.state]:
        raise ValueError(f"a {walk.state} walk cannot become {to}")
    sets = "".join(f",{name}=?" for name in columns)
    conn.execute(f"UPDATE curation_collections SET state=?{sets} WHERE id=?",
                 (to, *columns.values(), walk.id))


def _add_page(conn: sqlite3.Connection, collection: int, branch: Branch) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO curation_pages(collection,operation,partition,cursor) "
        "VALUES(?,?,?,?)", (collection, branch.operation, branch.partition, json.dumps(branch.cursor)))


def start(conn: sqlite3.Connection, collector: str, account: str,
          first: Branch, now: datetime) -> Walk:
    """Resume unfinished work; a cadence tick cannot discard failures by starting over.

    A changed account supersedes in-flight work. The accepted generation stays available until
    a replacement completes, but is never reused as another account's aggregate baseline.
    """
    if not collector or not account or not first.operation:
        raise ValueError("a collection requires its collector, account and operation")
    started_at = utc_iso(now)
    with write_tx(conn, site="curation.start"):
        active = current(conn, collector)
        if active and active.account == account and active.state in OPEN:
            return active
        if active and active.state in OPEN:
            _move(conn, active, SUPERSEDED)
        key = conn.execute(
            "INSERT INTO curation_collections(collector,account,started_at,state) "
            "VALUES(?,?,?,?)", (collector, account, started_at, WALKING)).lastrowid
        conn.execute(
            "INSERT INTO curation_collection_heads(collector,account,current) VALUES(?,?,?) "
            "ON CONFLICT(collector) DO UPDATE SET current=excluded.current, "
            "accepted=CASE WHEN account=excluded.account THEN accepted ELSE NULL END, "
            "account=excluded.account", (collector, account, key))
        _add_page(conn, key, first)
    return Walk(key, collector, account, started_at, WALKING)


def pending(conn: sqlite3.Connection, walk: Walk, operation: str | None = None) -> list[Task]:
    require_current(conn, walk)
    rows = conn.execute(
        "SELECT id,collection,operation,partition,cursor,status,failures,retry_at,last_error "
        "FROM curation_pages WHERE collection=? AND done=0 "
        "AND (? IS NULL OR operation=?) ORDER BY id", (walk.id, operation, operation))
    return [Task(row[0], row[1], Branch(row[2], row[3], json.loads(row[4])), Retry(*row[5:]))
            for row in rows]


def _page_continuation(conn: sqlite3.Connection, task: Task, page: Page) -> str | None:
    """Read only this page's evidence, keeping each checkpoint bounded as the walk grows."""
    scope = (task.collection, task.branch.operation, task.branch.partition)
    seen = {key for key in page.observed if conn.execute(
        "SELECT 1 FROM curation_seen WHERE collection=? AND operation=? AND partition=? AND key=?",
        (*scope, key)).fetchone()}
    cursor_seen = conn.execute(
        "SELECT 1 FROM curation_pages WHERE collection=? AND operation=? AND partition=? AND cursor=?",
        (*scope, json.dumps(page.cursor))).fetchone()
    return continuation(page, seen, {page.cursor} if cursor_seen and page.cursor is not None else set())


def accept_page(conn: sqlite3.Connection, walk: Walk, task: Task, page: Page, *,
                branches: tuple[Branch, ...] = ()) -> bool:
    """Checkpoint one page; False means a replay of an already durable response.

    Children (owned lists' member walks) land in this same transaction. A malformed page or a
    stale result writes nothing, and no generation can become ready while a child remains owed.
    """
    records = tuple((key, json.dumps(data, sort_keys=True)) for key, data in page.records)
    if any(not key for key, _ in records) or any(not key for key in page.observed):
        raise Incomplete("collection item lacks an identity")
    if any(not branch.operation for branch in branches):
        raise ValueError("a child collection requires its operation")
    with write_tx(conn, site="curation.page"):
        active = require_current(conn, walk)
        row = conn.execute(
            "SELECT collection,operation,partition,cursor,done FROM curation_pages WHERE id=?",
            (task.id,)).fetchone()
        expected = (walk.id, task.branch.operation, task.branch.partition, json.dumps(task.branch.cursor))
        if task.collection != walk.id or row is None or tuple(row[:4]) != expected:
            raise Changed("page does not belong to this collection")
        if row[4]:
            return False
        if active.state != WALKING:
            raise Changed("collection is no longer walking")
        following = _page_continuation(conn, task, page)
        scope = (walk.id, task.branch.operation, task.branch.partition)
        conn.executemany(
            "INSERT OR IGNORE INTO curation_observations(collection,operation,partition,key,data) "
            "VALUES(?,?,?,?,?)", ((*scope, key, data) for key, data in records))
        conn.executemany(
            "INSERT OR IGNORE INTO curation_seen(collection,operation,partition,key) VALUES(?,?,?,?)",
            ((*scope, key) for key in set(page.observed)))
        if following is not None:
            _add_page(conn, walk.id, Branch(task.branch.operation, task.branch.partition, following))
        for branch in branches:
            _add_page(conn, walk.id, branch)
        conn.execute("UPDATE curation_pages SET done=1,status=NULL,failures=NULL,retry_at=NULL,"
                     "last_error=NULL WHERE id=?", (task.id,))
        if not conn.execute("SELECT 1 FROM curation_pages WHERE collection=? AND done=0 LIMIT 1",
                            (walk.id,)).fetchone():
            _move(conn, active, READY)
    return True


def observations(conn: sqlite3.Connection, walk: Walk, operation: str) -> Iterator[tuple[str, str, dict]]:
    """Stream partition/key/record triples for presence or aggregate publication."""
    for partition, key, data in conn.execute(
            "SELECT partition,key,data FROM curation_observations WHERE collection=? AND operation=? "
            "ORDER BY partition,key", (walk.id, operation)):
        yield partition, key, json.loads(data)


def require_ready(conn: sqlite3.Connection, walk: Walk) -> Walk:
    """Refuse a walk that is not ready to publish: another generation, or a page still owed.
    A walk already complete passes, so a replayed finish repeats its idempotent writes."""
    active = require_current(conn, walk)
    if active.state != COMPLETE and (active.state != READY or pending(conn, walk)):
        raise Incomplete("collection still has unsettled pages")
    return active


def complete(conn: sqlite3.Connection, walk: Walk, now: datetime) -> None:
    """Accept the aggregate only after the whole collection settles; compose with signal writes."""
    completed_at = utc_iso(now)
    with write_tx(conn, site="curation.complete"):
        active = require_ready(conn, walk)
        if active.state == COMPLETE:
            return
        _move(conn, active, COMPLETE, completed_at=completed_at)
        conn.execute("UPDATE curation_collection_heads SET accepted=? WHERE collector=?",
                     (walk.id, walk.collector))


def completed_at(conn: sqlite3.Connection, walk: Walk) -> str | None:
    """When this walk completed; None while it is still owed."""
    row = conn.execute("SELECT completed_at FROM curation_collections WHERE id=?",
                       (walk.id,)).fetchone()
    return row[0] if row else None


def accepted(conn: sqlite3.Connection, collector: str) -> Walk | None:
    if current(conn, collector) is None:
        return None
    row = conn.execute(
        "SELECT c.id,c.collector,c.account,c.started_at,c.state FROM curation_collections c "
        "JOIN curation_collection_heads h ON h.accepted=c.id WHERE h.collector=?",
        (collector,)).fetchone()
    return Walk(*row) if row else None


def prune(conn: sqlite3.Connection, collector: str) -> None:
    """Keep active and last accepted evidence; independent saved-item inputs live elsewhere."""
    with write_tx(conn, site="curation.prune"):
        old = [row[0] for row in conn.execute(
            "SELECT c.id FROM curation_collections c JOIN curation_collection_heads h "
            "ON h.collector=c.collector WHERE c.collector=? "
            "AND c.id!=h.current AND c.id!=COALESCE(h.accepted,-1)", (collector,))]
        for key in old:
            for table in ("curation_pages", "curation_seen", "curation_observations"):
                conn.execute(f"DELETE FROM {table} WHERE collection=?", (key,))
            conn.execute("DELETE FROM curation_collections WHERE id=?", (key,))


class PageRows:
    """The existing quota loop owns retry timing/charging; this adapter owns its page columns."""

    def read(self, task: Task) -> Retry:
        return task.retry

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        with write_tx(conn, site="curation.retry"):
            conn.execute(
                "UPDATE curation_pages SET status=?,failures=?,retry_at=?,last_error=? "
                "WHERE id=? AND done=0 AND collection IN "
                "(SELECT current FROM curation_collection_heads)",
                (retry.status, retry.failures, retry.retry_at, retry.last_error, *key))

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        self.write(conn, key, Retry())


ROWS = PageRows()
