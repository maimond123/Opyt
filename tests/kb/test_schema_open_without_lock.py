"""An existing store must open without ever needing the write lock.

`mcp_server/server.py` forks several detached rails within milliseconds, and every one calls
`schema.connect()` -> `init_kb_schema`. When that path ran backfills that wrote unconditionally,
those connections raced for the write lock over statements that changed zero rows: 28
`database is locked` events over six days, every one a read->write upgrade refusal that no
`busy_timeout` can wait out.

The property is tested by taking the lock away, not by inspecting statement text: holding
`BEGIN IMMEDIATE` on a second connection at `busy_timeout=0` makes any statement that wants the
lock raise immediately.
"""
from __future__ import annotations

import sqlite3

from pipeline.kb import schema


def test_an_existing_store_opens_without_the_write_lock(kb_home, tmp_path):
    path = tmp_path / "opyt.db"
    schema.connect(path).close()
    holder = sqlite3.connect(path)
    holder.execute("PRAGMA busy_timeout=0")
    holder.execute("BEGIN IMMEDIATE")
    reader = sqlite3.connect(path)
    reader.execute("PRAGMA busy_timeout=0")
    try:
        schema.init_kb_schema(reader)      # raises OperationalError if anything wants the lock
    finally:
        reader.close()
        holder.rollback()
        holder.close()
