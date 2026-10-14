"""Oracle X posts whose photos are still unread.

A post rendered with a photo still unread owes a row here, written in the same transaction as the
post; the Oracle photo source (`oracles/sources/x_photos.py`) reads the photos and renders it
again.

The row carries the post's normalized tweets (`grp`), since atoms store no raw input, and is
deleted when it settles. Its retry columns are its own: a post is kept even if its photo never
reads.

The thread-tail lane waits on a row that is owed and not given up (`WAITING`), so a thread with
photos is rewritten once for both.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso, utc_now
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Changed, Retry

_DDL = """
CREATE TABLE IF NOT EXISTS oracle_x_photos (
    atom_id TEXT PRIMARY KEY,
    who_id TEXT NOT NULL,
    grp TEXT NOT NULL,
    owed_at TEXT NOT NULL,
    status TEXT, failures INTEGER, retry_at TEXT, last_error TEXT
);
CREATE INDEX IF NOT EXISTS oracle_x_photos_owed ON oracle_x_photos(owed_at);
"""

# For a query over `atoms`: the atom's photos are owed and not given up, so its thread tail waits.
WAITING = ("EXISTS (SELECT 1 FROM oracle_x_photos p WHERE p.atom_id=atoms.atom_id "
           f"AND COALESCE(p.failures, 0)<{retries.GIVE_UP_AFTER})")


@dataclass(frozen=True)
class Owed:
    atom_id: str
    who_id: str
    group: list[dict]
    raw: str                       # `grp` as stored: what `require_current` compares
    retry: Retry


def init_schema(conn: sqlite3.Connection) -> None:
    # executescript commits a caller's transaction, so initialization is entry-only.
    if conn.in_transaction:
        raise RuntimeError("initialize the Oracle photo table before starting a transaction")
    conn.executescript(_DDL)


def owe(conn: sqlite3.Connection, atom_id: str, who_id: str, group: list[dict]) -> None:
    """This post's photos are owed, read from `group`. Joins the caller's transaction, so a post
    and what it still owes land together. Owing it again replaces the input and keeps its retry
    state, so a given-up row stays given up."""
    with write_tx(conn, site="footprint_photos.owe"):
        conn.execute(
            "INSERT INTO oracle_x_photos(atom_id,who_id,grp,owed_at) VALUES(?,?,?,?) "
            "ON CONFLICT(atom_id) DO UPDATE SET who_id=excluded.who_id, grp=excluded.grp",
            (atom_id, who_id, json.dumps(group), utc_iso(utc_now())))


def owed(conn: sqlite3.Connection) -> list[Owed]:
    """Every owed row, oldest first."""
    return [Owed(r[0], r[1], json.loads(r[2]), r[2], Retry(*r[3:]))
            for r in conn.execute(
                "SELECT atom_id,who_id,grp,status,failures,retry_at,last_error "
                "FROM oracle_x_photos ORDER BY owed_at,atom_id")]


def require_current(conn: sqlite3.Connection, row: Owed) -> None:
    """Refuse a write for a row settled, or owed again with another input, meanwhile."""
    got = conn.execute("SELECT grp FROM oracle_x_photos WHERE atom_id=?",
                       (row.atom_id,)).fetchone()
    if got is None or got[0] != row.raw:
        raise Changed(f"{row.atom_id}'s owed photos changed")


def settle(conn: sqlite3.Connection, row: Owed) -> None:
    """The post's photos are read and its render is current: the row and its input go. Joins
    the caller's transaction."""
    with write_tx(conn, site="footprint_photos.settle"):
        require_current(conn, row)
        conn.execute("DELETE FROM oracle_x_photos WHERE atom_id=?", (row.atom_id,))


class PhotoRows:
    """The quota loop owns retry timing and charging; this adapter owns the row's columns."""

    def read(self, row: Owed) -> Retry:
        return row.retry

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        with write_tx(conn, site="footprint_photos.retry"):
            conn.execute("UPDATE oracle_x_photos SET status=?,failures=?,retry_at=?,last_error=? "
                         "WHERE atom_id=?",
                         (retry.status, retry.failures, retry.retry_at, retry.last_error, *key))

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        self.write(conn, key, Retry())


ROWS = PhotoRows()
