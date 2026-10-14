"""`service.db` takes the write lock only to write (B4, from the store-writes audit §D).

Every connect used to run the owner-claims seed — an `INSERT OR IGNORE … SELECT` — which takes
the write lock even when it inserts nothing, in rollback-journal mode. Every request connects, so
a read-only auth lookup waited out any write in flight and then failed with "database is locked".
"""
from __future__ import annotations

import sqlite3
import time

from service import db, tokens


def test_an_auth_lookup_does_not_wait_on_a_writer(svc):
    writer = sqlite3.connect(db.db_path())
    writer.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        assert tokens.resolve_token(svc.reader_token)["role"] == "reader"
        assert time.monotonic() - started < 1
    finally:
        writer.rollback()
        writer.close()


def test_service_db_is_in_wal_mode(svc):
    conn = db.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()
