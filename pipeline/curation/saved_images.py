"""Saved posts whose photos are still unread, at the input revision their post was saved from.

A saved post's first save renders what the image cache already holds and owes a row here for any
photo it could not describe. The saved-images source (`saved_x_images.SavedXImages`) reads them.
The thread lane waits on an owed row, so a post with both photos and a thread is rewritten once,
not twice. The row carries its own retry columns, apart from the post's, because
an image read has its own ladder and give-up: a post is saved even if its photo never reads.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from pipeline.curation import saved_items
from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop.source import Changed, Retry

_DDL = """
CREATE TABLE IF NOT EXISTS curation_saved_images (
    platform TEXT NOT NULL, item TEXT NOT NULL, revision TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('owed', 'done')),
    owed_at TEXT NOT NULL,
    status TEXT, failures INTEGER, retry_at TEXT, last_error TEXT,
    PRIMARY KEY(platform, item)
);
CREATE INDEX IF NOT EXISTS curation_saved_images_owed
    ON curation_saved_images(platform, state, owed_at);
"""


@dataclass(frozen=True)
class Owed:
    """An owed row, with the saved post it belongs to (its record is the post's input)."""
    saved: saved_items.Saved
    retry: Retry


def init_schema(conn: sqlite3.Connection) -> None:
    # executescript commits a caller's transaction, so initialization is entry-only.
    if conn.in_transaction:
        raise RuntimeError("initialize saved-image tables before starting a transaction")
    conn.executescript(_DDL)


def owe(conn: sqlite3.Connection, saved: saved_items.Saved, now: datetime) -> None:
    """This revision's photos are owed. Joins the caller's transaction, so a post and what it
    still owes land together. Owing the same revision again keeps its retry state."""
    with write_tx(conn, site="saved_images.owe"):
        conn.execute(
            "INSERT INTO curation_saved_images(platform,item,revision,state,owed_at) "
            "VALUES(?,?,?,'owed',?) ON CONFLICT(platform,item) DO UPDATE SET "
            "revision=excluded.revision, state='owed', owed_at=excluded.owed_at, "
            "status=NULL, failures=NULL, retry_at=NULL, last_error=NULL "
            "WHERE curation_saved_images.revision!=excluded.revision "
            "OR curation_saved_images.state='done'",
            (saved.platform, saved.item, saved.revision, utc_iso(now)))


def owed(conn: sqlite3.Connection, platform: str) -> list[Owed]:
    """Owed rows whose saved post is still at the revision they were owed for, oldest first."""
    # The post's own retry state is not this row's, so it is left empty here.
    return [Owed(saved_items.Saved(platform, row[0], row[1], row[2], json.loads(row[3]), Retry()),
                 Retry(*row[4:]))
            for row in conn.execute(
                "SELECT i.item,s.account,i.revision,s.data,i.status,i.failures,i.retry_at,"
                "i.last_error FROM curation_saved_images i JOIN curation_saved_items s "
                "ON s.platform=i.platform AND s.item=i.item AND s.revision=i.revision "
                "WHERE i.platform=? AND i.state='owed' ORDER BY i.owed_at,i.item", (platform,))]


def require_current(conn: sqlite3.Connection, row: Owed) -> None:
    """Refuse a write for a row re-owed at another revision, or settled, meanwhile."""
    got = conn.execute("SELECT revision,state FROM curation_saved_images "
                       "WHERE platform=? AND item=?", (row.saved.platform, row.saved.item)).fetchone()
    if got is None or tuple(got) != (row.saved.revision, "owed"):
        raise Changed(f"saved {row.saved.platform} item {row.saved.item}'s images changed")


def settle(conn: sqlite3.Connection, row: Owed) -> None:
    """This revision's photos are read. Joins the caller's transaction."""
    with write_tx(conn, site="saved_images.settle"):
        require_current(conn, row)
        conn.execute("UPDATE curation_saved_images SET state='done',status=NULL,failures=NULL,"
                     "retry_at=NULL,last_error=NULL WHERE platform=? AND item=?",
                     (row.saved.platform, row.saved.item))


class ImageRows:
    """The quota loop owns retry timing and charging; this adapter owns the row's columns."""

    def read(self, row: Owed) -> Retry:
        return row.retry

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        platform, item, revision = key
        with write_tx(conn, site="saved_images.retry"):
            conn.execute(
                "UPDATE curation_saved_images SET status=?,failures=?,retry_at=?,last_error=? "
                "WHERE platform=? AND item=? AND revision=? AND state='owed'",
                (retry.status, retry.failures, retry.retry_at, retry.last_error,
                 platform, item, revision))

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        self.write(conn, key, Retry())


ROWS = ImageRows()
