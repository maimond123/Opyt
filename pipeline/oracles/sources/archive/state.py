"""Durable archive walks and their small listing/document units.

A page and the work it discovered commit together. Only a walk with every unit settled may
publish its coverage. Exhausted retries remain unfinished until another explicit ask rearms them.
"""
import json
from dataclasses import dataclass
from datetime import datetime

from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Retry
from pipeline.worker.quota_loop.source import RetryRows

DDL = """
CREATE TABLE IF NOT EXISTS archive_walks (
    key TEXT PRIMARY KEY, canonical_id TEXT NOT NULL, generation TEXT NOT NULL,
    spec TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS archive_units (
    id INTEGER PRIMARY KEY, walk TEXT NOT NULL, generation TEXT NOT NULL,
    kind TEXT NOT NULL, key TEXT NOT NULL, data TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0, status TEXT, failures INTEGER,
    retry_at TEXT, last_error TEXT,
    UNIQUE(walk, generation, kind, key)
);
CREATE INDEX IF NOT EXISTS archive_pending ON archive_units(walk, generation, done, id);
"""


def init_schema(conn):
    conn.executescript(DDL)


@dataclass(frozen=True)
class Walk:
    key: str
    canonical_id: str
    generation: str
    spec: dict


def active(conn, key):
    row = conn.execute("SELECT * FROM archive_walks WHERE key=? AND completed=0", (key,)).fetchone()
    return Walk(row["key"], row["canonical_id"], row["generation"], json.loads(row["spec"])) if row else None


def current(conn, walk):
    return conn.execute("SELECT 1 FROM archive_walks WHERE key=? AND generation=? AND completed=0",
                        (walk.key, walk.generation)).fetchone() is not None


def _add(conn, walk, kind, key, data):
    conn.execute("INSERT OR IGNORE INTO archive_units(walk,generation,kind,key,data) VALUES(?,?,?,?,?)",
                 (walk.key, walk.generation, kind, key, json.dumps(data)))


def start(conn, key, canonical_id, spec, cursor, now: datetime):
    walk = Walk(key, canonical_id, utc_iso(now), spec)
    with write_tx(conn, site="archive.start"):
        if (existing := active(conn, key)) is not None:
            return existing
        conn.execute("DELETE FROM archive_units WHERE walk=?", (key,))
        conn.execute("INSERT OR REPLACE INTO archive_walks(key,canonical_id,generation,spec) VALUES(?,?,?,?)",
                     (key, canonical_id, walk.generation, json.dumps(spec)))
        _add(conn, walk, "page", json.dumps(cursor, sort_keys=True), cursor)
    return walk


def pending(conn, walk):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM archive_units WHERE walk=? AND generation=? AND done=0 ORDER BY id",
        (walk.key, walk.generation))]


def exhausted(conn, key) -> bool:
    """The active walk on `key` has units left, and has given up on every one: nothing more runs
    until an explicit ask rearms them. The walk stays unfinished, so its coverage is not claimed."""
    walk = active(conn, key)
    tasks = pending(conn, walk) if walk else []
    return bool(tasks) and all(retries.given_up(t["failures"]) for t in tasks)


def record_page(conn, walk, task, page):
    with write_tx(conn, site="archive.page"):
        if not current(conn, walk):
            return
        for cursor in page.cursors:
            if conn.execute("SELECT 1 FROM archive_units WHERE walk=? AND generation=? AND kind='page' AND key=?",
                            (walk.key, walk.generation, json.dumps(cursor, sort_keys=True))).fetchone():
                raise RuntimeError("archive listing repeated a previous cursor")
        for key, doc in page.documents:
            _add(conn, walk, "document", key, doc)
        for cursor in page.cursors:
            _add(conn, walk, "page", json.dumps(cursor, sort_keys=True), cursor)
        conn.execute("UPDATE archive_units SET done=1 WHERE id=?", (task["id"],))


def record_document(conn, walk, task, status):
    with write_tx(conn, site="archive.document"):
        if current(conn, walk):
            conn.execute("UPDATE archive_units SET done=1,status=? WHERE id=?", (status, task["id"]))


def complete(conn, walk):
    with write_tx(conn, site="archive.complete"):
        if pending(conn, walk):
            raise RuntimeError("archive still has unsettled units")
        conn.execute("UPDATE archive_walks SET completed=1 WHERE key=? AND generation=?",
                     (walk.key, walk.generation))


class UnitRows(RetryRows):
    def read(self, row):
        return Retry(*(row[k] for k in ("status", "failures", "retry_at", "last_error")))

    def write(self, conn, key, retry):
        with write_tx(conn, site="archive.retry"):
            conn.execute("UPDATE archive_units SET status=?,failures=?,retry_at=?,last_error=? "
                         "WHERE id=? AND done=0", (*vars(retry).values(), *key))

    def clear(self, conn, key):
        self.write(conn, key, Retry())


ROWS = UnitRows()
