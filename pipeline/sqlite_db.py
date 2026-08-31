"""
pipeline/sqlite_db.py

Locate and open the machine-canonical ~/.opyt/opyt.db.

`default_db_path()` returns the same DB path the GUI indexer + Rust reader use; `connect()`
opens it (WAL for writers, read-only URI for readers). These two helpers are load-bearing:
sync_lock, dedup_store and circuit_breaker all import them to find/open the shared database.

This module survived the deletion of its one-time fourth importer, `opyt_core/db.py` (gone
2026-08-13 — its only export, `conn_ro`, had zero callers). Do not follow that deletion downward
into this file. The three importers above all open WRITE-CAPABLE, so none of them needs the
database file to pre-exist, which is exactly why `conn_ro` and the bootstrap touch that fed it
could go while this stayed.

Formerly `pipeline/ask/ask_schema.py`, where it also defined the dev-log turn/anchor schema
(turns, turns_fts, turn_embeddings, anchors) + the anchors writer. That DDL went out with the
dev-log, leaving DB helpers stranded in a retrieval package they had nothing to do with — and
that package is the VAULT rail, which is on the deprecation tail. Nothing here is vault, so it
moved up a level to outlive it. `ask_db_path` was renamed in the same move — there has been no
`ask` tool since the interactive /ask was retired under MCP-first. The name is `default_db_path`
and not the shorter `db_path` on purpose: every caller here reads `Path(db_path) if db_path else
...`, where `db_path` is the caller's OWN override parameter, so the bare name would shadow the
import at exactly the sites that need it. The longer name also says the true thing — this is the
fallback when the caller named no database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db


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
        conn.execute("PRAGMA journal_mode=WAL")  # readers keep reading during rebuild
    conn.row_factory = sqlite3.Row
    return conn
