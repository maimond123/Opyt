"""
pipeline/sqlite_db.py — locate and open the machine-canonical ~/.opyt/opyt.db.

`default_db_path()` is the same DB the GUI indexer and Rust reader use; `connect()` opens it (WAL
for writers, read-only URI for readers). dedup_store opens it through
here. The name is not `db_path` because callers use that for their own override argument.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db, owner_only


def default_db_path() -> Path:
    """Same machine-canonical DB the GUI indexer + Rust reader use."""
    return opyt_db()


def connect(read_only: bool = False) -> sqlite3.Connection:
    p = default_db_path()
    if read_only:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p))
        owner_only(p)  # user content is not world-readable
        conn.execute("PRAGMA journal_mode=WAL")  # readers keep reading during rebuild
    conn.row_factory = sqlite3.Row
    return conn
