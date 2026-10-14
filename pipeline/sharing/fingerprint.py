"""
pipeline/sharing/fingerprint.py — what a copy of a knowledge base holds, row by row, as hashes.

A copy is an export file, or the service's patched copy of one. Its fingerprint hashes every row
of every table whose rows an export holds (`export.TABLES`), keyed by table and `rowid`. A body's
rowid is its `body_id`, which the export carries from the live store and an update in place keeps,
so a body keeps its key across pushes. A copy's digest is the sum
of all the row hashes modulo 2^256: equal digests mean equal rows. A sum and not an ordered hash,
so a patch can move a copy's digest by the rows it touched without reading the rest
(`delta.apply`).

A copy records its own digest in a one-row table outside `export.TABLES` (`stamp`), so the
service knows which copy it holds without hashing it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pipeline.sharing.export import TABLES

_MOD = 1 << 256

# The copy's own digest. Not in `export.TABLES`, so it is not part of the digest it records.
STAMP = "served_digest"

def columns(conn, schema: str, table: str) -> list[str]:
    """`table`'s columns in the database attached as `schema`, in its declared order."""
    return [r[1] for r in conn.execute(f"PRAGMA {schema}.table_info({table})")]


def select_rows(conn, schema: str, table: str, where: str = "") -> str:
    """The query that reads `table`'s rows as the fingerprint sees them: `rowid`, then every
    column in the table's own order."""
    cols = ", ".join(["rowid", *columns(conn, schema, table)])
    return f"SELECT {cols} FROM {schema}.{table} {where}"


def row_hash(table: str, row) -> int:
    """One row's hash. Each value is tagged with its type and length, so `1` and `'1'` differ and
    no two rows run together."""
    h = hashlib.sha256(table.encode() + b"\0")
    for v in row:
        if v is None:
            tag, data = b"n", b""
        elif isinstance(v, int):
            tag, data = b"i", str(v).encode()
        elif isinstance(v, float):
            tag, data = b"f", v.hex().encode()
        elif isinstance(v, str):
            tag, data = b"s", v.encode()
        else:
            tag, data = b"b", bytes(v)
        h.update(tag + len(data).to_bytes(8, "big") + data)
    return int.from_bytes(h.digest(), "big")


def as_digest(total: int) -> str:
    """A sum of row hashes, as the 64-hex-digit digest a copy records."""
    return format(total % _MOD, "064x")


@dataclass(frozen=True)
class Fingerprint:
    rows: dict[str, dict[int, int]]           # table → rowid → hash of its row

    @property
    def digest(self) -> str:
        return as_digest(sum(h for table in self.rows.values() for h in table.values()))


def of_copy(path: Path | str) -> Fingerprint:
    """Every row of the copy at `path`, hashed."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows: dict[str, dict[int, int]] = {}
        for table in TABLES:
            rows[table] = {row[0]: row_hash(table, row)
                           for row in conn.execute(select_rows(conn, "main", table))}
        return Fingerprint(rows)
    finally:
        conn.close()


@dataclass(frozen=True)
class Changes:
    """What turns one copy into another, by table: the keys to write (new, or held differently)
    and the keys to delete (held no more)."""
    write: dict[str, list[int]]
    delete: dict[str, list[int]]


def diff(old: Fingerprint, new: Fingerprint) -> Changes:
    write: dict[str, list[int]] = {}
    delete: dict[str, list[int]] = {}
    for table in TABLES:
        was, now = old.rows.get(table, {}), new.rows.get(table, {})
        write[table] = [key for key, h in now.items() if was.get(key) != h]
        delete[table] = [key for key in was if key not in now]
    return Changes(write, delete)


def save(fp: Fingerprint, path: Path) -> None:
    """Write `fp` to `path` as JSON: per table, `[rowid, hash]` pairs, so an int key stays one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({table: [[key, format(h, "064x")] for key, h in rows.items()]
                                for table, rows in fp.rows.items()}))


def load(path: Path) -> Fingerprint | None:
    """The fingerprint saved at `path`, or None when there is none or it cannot be read."""
    try:
        saved = json.loads(path.read_text())
        return Fingerprint({table: {key: int(h, 16) for key, h in pairs}
                            for table, pairs in saved.items()})
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def stamp(path: Path | str, digest: str) -> None:
    """Record in the copy at `path` the digest of the rows it holds."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {STAMP} (digest TEXT NOT NULL)")
        conn.execute(f"DELETE FROM {STAMP}")
        conn.execute(f"INSERT INTO {STAMP} (digest) VALUES (?)", (digest,))
        conn.commit()
    finally:
        conn.close()


def stamped(conn) -> str | None:
    """The digest the copy open on `conn` records, or None for a copy that records none: one
    uploaded by an install older than deltas."""
    if conn.execute("SELECT 1 FROM main.sqlite_master WHERE type = 'table' AND name = ?",
                    (STAMP,)).fetchone() is None:
        return None
    row = conn.execute(f"SELECT digest FROM main.{STAMP}").fetchone()
    return row[0] if row else None
