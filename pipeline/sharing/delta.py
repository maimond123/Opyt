"""
pipeline/sharing/delta.py — the rows that changed, carried from a fresh export into the served copy.

`build` cuts a delta out of a fresh export: the rows a fingerprint diff names, in tables created
from the export's own DDL, the keys to delete, and two digests, of the copy it was cut against
and of the copy it makes. It reads nothing but that export, so `export.py` still decides
everything that crosses. A delta holds rows only, never a search index.

`apply` writes a delta into a served copy in one transaction, and commits only if the copy held
the base digest and the patched copy's digest is the one the owner computed over the whole fresh
export. Anything else is `Stale`, which a full upload fixes, and nothing was written.

Rows are keyed by `rowid`, which an export carries over from the live store; a body's is its
`body_id`. Every table is patched alike: the touched rows are deleted and the delta's inserted.
The served copy's `bodies_fts` follows its `bodies` through the copy's own triggers, which read
the old text from a row as it is deleted and index the new text as it is inserted.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from pipeline.sharing.export import TABLES, copy_ddl
from pipeline.sharing.fingerprint import (STAMP, Changes, as_digest, columns, row_hash,
                                          select_rows, stamped)

# The keys to delete, by table, and the two digests. Present only in a delta.
DELETES = "delta_deletes"
META = "delta_meta"

# Every table and index takes at least one page, so 4 KB pages made an empty delta 119 KB and a
# one-atom delta 201 KB (an own-key store); 1 KB pages, 37 KB and 112 KB.
_DELTA_PAGE = 1024

# Big enough that a patch is held in memory until its commit, so readers keep reading the old
# rows until then instead of waiting on a spill.
_CACHE_KIB = 65_536

# A commit waits for the readers already inside the file; a search takes a second or two.
_WRITE_WAIT_S = 30


class Stale(Exception):
    """The served copy cannot take this delta, and a full upload would replace it. Nothing was
    written."""


class TooLarge(Exception):
    """The patched copy would be larger than the service serves. Nothing was written."""


def build(export_path: Path | str, changes: Changes, base: str, digest: str,
          out_path: Path | str) -> int:
    """Write the delta that turns the copy whose digest is `base` into the export at
    `export_path` (whose digest is `digest`) to `out_path`. Returns its size in bytes."""
    out_path = Path(out_path)
    out_path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(out_path))
    try:
        conn.execute(f"PRAGMA page_size={_DELTA_PAGE}")
        conn.execute("ATTACH ? AS src", (str(export_path),))
        copy_ddl(conn, set(TABLES))
        conn.execute(f"CREATE TABLE {DELETES} (tbl TEXT NOT NULL, key INTEGER NOT NULL)")
        conn.execute(f"CREATE TABLE {META} (base TEXT NOT NULL, digest TEXT NOT NULL)")
        conn.execute("CREATE TEMP TABLE wanted (tbl TEXT NOT NULL, key NOT NULL)")
        conn.executemany("INSERT INTO temp.wanted VALUES (?, ?)",
                         [(t, k) for t, keys in changes.write.items() for k in keys])
        for table in TABLES:
            cols = ", ".join(["rowid", *columns(conn, "src", table)])
            conn.execute(f"INSERT INTO main.{table} ({cols}) SELECT {cols} FROM src.{table} "
                         f"WHERE rowid IN (SELECT key FROM temp.wanted WHERE tbl = ?)", (table,))
        conn.executemany(f"INSERT INTO {DELETES} VALUES (?, ?)",
                         [(t, k) for t, keys in changes.delete.items() for k in keys])
        conn.execute(f"INSERT INTO {META} VALUES (?, ?)", (base, digest))
        conn.commit()
        conn.execute("DETACH src")
    finally:
        conn.close()
    return out_path.stat().st_size


def apply(copy_path: Path | str, delta_path: Path | str, *,
          max_bytes: int | None = None) -> dict:
    """Patch the served copy at `copy_path` with the delta at `delta_path`, in one transaction.
    `{digest, written, deleted}`.

    Raises `Stale` when the copy is missing, records no digest, is not the delta's base, has no
    whole-item search index kept by triggers, cannot be written, or would not come out with the delta's
    digest; `TooLarge` when the patched copy's rows would exceed `max_bytes`. Either way the copy
    is as it was."""
    if not Path(copy_path).exists():
        raise Stale("the service holds no copy of this knowledge base to patch")
    conn = sqlite3.connect(str(copy_path), isolation_level=None, timeout=_WRITE_WAIT_S)
    try:
        conn.execute(f"PRAGMA cache_size=-{_CACHE_KIB}")
        conn.execute("ATTACH ? AS delta", (str(delta_path),))
        base, digest = conn.execute(f"SELECT base, digest FROM delta.{META}").fetchone()
        conn.execute("BEGIN IMMEDIATE")
        try:
            out = _patch(conn, base, digest, max_bytes)
            conn.execute("COMMIT")
        except BaseException:
            # Some errors (a full disk, for one) have already rolled the transaction back.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return out
    except sqlite3.DatabaseError as e:
        raise Stale(f"the served copy could not be patched ({e})") from e
    finally:
        conn.close()


def _patch(conn, base: str, digest: str, max_bytes: int | None) -> dict:
    """The writes and the checks, inside the caller's transaction."""
    held = stamped(conn)
    if held is None:
        raise Stale("the served copy records no digest, so it predates changes-only uploads")
    if held != base:
        raise Stale("the served copy is not the one this delta was cut against")
    if not _index_over_bodies(conn):
        raise Stale("the served copy was made by an older Opyt, before whole-item search; it is "
                    "replaced whole")

    conn.execute("CREATE TEMP TABLE touched (tbl TEXT NOT NULL, key NOT NULL)")
    for table in TABLES:
        conn.execute(f"INSERT INTO temp.touched SELECT ?, rowid FROM delta.{table}", (table,))
    conn.execute(f"INSERT INTO temp.touched SELECT tbl, key FROM delta.{DELETES}")

    total = int(held, 16) - _touched_hash(conn)
    written = 0
    for table in TABLES:
        # The served copy's own columns: the delta's names never reach the SQL. A column the
        # delta lacks fails here, and one it adds is left behind and fails the digest.
        cols = ", ".join(["rowid", *columns(conn, "main", table)])
        conn.execute(f"DELETE FROM main.{table} "
                     f"WHERE rowid IN (SELECT key FROM temp.touched WHERE tbl = ?)", (table,))
        written += conn.execute(
            f"INSERT INTO main.{table} ({cols}) SELECT {cols} FROM delta.{table}").rowcount
    total += _touched_hash(conn)

    if as_digest(total) != digest:
        raise Stale("the patched copy did not come out as the owner's export")
    if max_bytes is not None and _row_bytes(conn) > max_bytes:
        raise TooLarge(
            f"with these changes this knowledge base would be larger than the "
            f"{max_bytes // 1_000_000} MB this service serves per knowledge base.")
    conn.execute(f"UPDATE main.{STAMP} SET digest = ?", (digest,))
    deleted = conn.execute(f"SELECT COUNT(*) FROM delta.{DELETES}").fetchone()[0]
    return {"digest": digest, "written": written, "deleted": deleted}


def _touched_hash(conn) -> int:
    """The sum of the hashes of the served copy's rows under the keys this delta touches."""
    total = 0
    for table in TABLES:
        sql = select_rows(conn, "main", table,
                          "WHERE rowid IN (SELECT key FROM temp.touched WHERE tbl = ?)")
        for row in conn.execute(sql, (table,)):
            total += row_hash(table, row)
    return total


# What a copy must hold for a patch to keep its search index in step with its bodies.
_BODIES_TRIGGERS = ("bodies_ai", "bodies_ad", "bodies_au")


def _index_over_bodies(conn) -> bool:
    """Whether the copy's search index is `bodies_fts`, external-content over `bodies`, with the
    three triggers that keep it in step. A copy
    without them predates whole-item search: patched, its index would describe old text."""
    rows = dict(conn.execute(
        "SELECT name, sql FROM main.sqlite_master WHERE name IN (?, ?, ?, ?)",
        ("bodies_fts", *_BODIES_TRIGGERS)).fetchall())
    index = rows.get("bodies_fts") or ""
    return "content='bodies'" in index and all(t in rows for t in _BODIES_TRIGGERS)


def _row_bytes(conn) -> int:
    """The bytes the copy's pages hold, not counting pages freed and waiting for reuse."""
    pages = conn.execute("PRAGMA main.page_count").fetchone()[0]
    free = conn.execute("PRAGMA main.freelist_count").fetchone()[0]
    size = conn.execute("PRAGMA main.page_size").fetchone()[0]
    return (pages - free) * size
