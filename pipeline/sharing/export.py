"""
pipeline/sharing/export.py — the live store projected into one self-contained, shareable file.

An export is what a second person's Opyt reads: the same `atoms`/`bodies`/`bodies_fts`/`entities`
substrate the local tools run against, minus everything the read path never touches.
It is opened by the same retrieval code.

  • Allow-list, never deny-list: `_CARRY` names every object that crosses, so a new table is
    absent from an export until someone adds it here.
  • A projection, never a second home: an export is rebuilt whole from the live store. A delta
    (`delta.py`) carries only the rows that changed, but it is cut from a fresh export, so this
    file still decides everything that crosses.

The DDL is copied from the live store's own `sqlite_master`, never re-declared. Every row keeps
the live store's `rowid`: ties in a read fall back to rowid order (FTS5's `bm25()` by docid, an
index scan by rowid), and a delta keys its rows by it.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db
from pipeline.kb import schema
from pipeline.retrieval.engine import json_obj

# ── the allow-list ───────────────────────────────────────────────────────────────
# Value = the columns to carry, or None for every column. A column left out is omitted from the
# INSERT (so a NOT NULL column takes its DDL default) rather than written as NULL.
_CARRY: dict[str, tuple[str, ...] | None] = {
    "atoms": None,
    # Each atom's snapshot text, which `open` returns.
    "bodies": None,
    # `identity_links` is dropped: no read tool reads it.
    "entities": ("entity_id", "name", "canonical_id", "profile"),
    # Only the join key and label `aggregate`'s trust-coverage count needs.
    "oracles": ("canonical_id", "name", "confirmed_at"),
    # `peers` must never be added: its `token` column holds the owner's own reader bearer tokens.
}

# Every table whose rows an export holds; its search index is built from `bodies`, so it is not
# one. A delta and the digest that checks it cover exactly these.
TABLES: tuple[str, ...] = tuple(_CARRY)

# ── the free-form JSON columns ───────────────────────────────────────────────────
# `payload` and `profile` are free-form, so each passes a key allow-list: `payload.source_tags`
# feeds `tags=` and `aggregate`, and `profile.handle` is how `resolve_who` covers X.

_PAYLOAD_KEYS = frozenset({
    # `body_state`/`body_basis` tell a reader whether an exported atom's body is whole.
    "source_tags", "body_state", "body_basis",
    # x
    "like_count", "reply_count", "is_quote", "is_thread", "is_article", "has_media",
    "thread_len", "media_substance", "is_reply", "has_link", "text_len",
    # github
    "stars", "forks", "code_language", "license",
    # paper. `external_ids` and `pdf_url` say where its open PDF is, for a page of sources the
    # connector makes from this copy.
    "has_fulltext", "year", "citationCount", "venue", "external_ids", "pdf_url",
    # blog / substack
    "lastmod", "word_count", "paywalled",
})

# `handle` is the only key the read path touches. `bio`/`followers`/`verified` and the screen
# classifier's stamps are scraped third-party text that serving search does not need.
_PROFILE_KEYS = frozenset({"handle"})


def _filtered(blob, allowed: frozenset[str], dropped: dict[str, int]) -> str | None:
    """A stored JSON object column → the same object with only `allowed` keys, re-encoded.
    Counts what it dropped into `dropped`, so a new adapter's key does not vanish silently. NULL
    in, NULL out; a malformed blob becomes `{}`."""
    if blob is None:
        return None
    obj = json_obj(blob)
    kept = {}
    for key, value in obj.items():
        if key in allowed:
            kept[key] = value
        else:
            dropped[key] = dropped.get(key, 0) + 1
    return json.dumps(kept)


def _columns(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA src.table_info({table})")]


def copy_ddl(conn, names: set[str]) -> None:
    """Recreate the named objects, and their indexes, in `main` from the DDL of the database
    attached as `src`. `sql IS NULL` skips SQLite's implicit PRIMARY KEY/UNIQUE indexes."""
    for kind, name, tbl, sql in conn.execute(
            "SELECT type, name, tbl_name, sql FROM src.sqlite_master "
            "WHERE type IN ('table','index') ORDER BY type DESC"):
        if sql is None:
            continue
        if name not in names and not (kind == "index" and tbl in names):
            continue
        conn.execute(sql)


def _copy_table(conn, table: str, carry: tuple[str, ...] | None) -> int:
    """Straight SQL copy of the `rowid` and the carried columns, with no Python round-trip
    (`bodies` is large)."""
    cols = list(carry) if carry is not None else _columns(conn, table)
    names = ", ".join(["rowid", *cols])
    conn.execute(f"INSERT INTO {table} ({names}) SELECT {names} FROM src.{table}")
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _copy_json_table(conn, table: str, carry: tuple[str, ...], column: str,
                     allowed: frozenset[str], dropped: dict[str, int]) -> int:
    """Copy a table whose one free-form JSON column has to pass a key allow-list, so the rows go
    through Python."""
    cols = ["rowid", *carry]
    at = cols.index(column)
    placeholders = ", ".join("?" for _ in cols)
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM src.{table}").fetchall()
    out = []
    for row in rows:
        values = list(row)
        values[at] = _filtered(values[at], allowed, dropped)
        out.append(values)
    conn.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", out)
    return len(out)


def build_export(out_path: Path | str) -> dict:
    """Project the local store (`opyt_db()`; set `$OPYT_HOME` for another) into a self-contained
    export file at `out_path`. Returns a manifest. Built in a sibling temp file and renamed into
    place, so a failed build leaves no finished-looking file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".building")
    tmp.unlink(missing_ok=True)

    # A writable open first applies the schema's in-place migrations, since a reader opens an
    # export read-only and could never fix a stale value.
    src = schema.connect()
    try:
        # A store with no atoms has nothing to share. Checked before the temp file exists, so a
        # refusal leaves no export file behind.
        ingested = src.execute("SELECT 1 FROM atoms LIMIT 1").fetchone()
    finally:
        src.close()
    if not ingested:
        raise ValueError(
            "this store holds nothing yet, so there is no knowledge base to export. Run "
            "`onboard` (or any ingest) first.")

    conn = sqlite3.connect(str(tmp))
    payload_dropped: dict[str, int] = {}
    profile_dropped: dict[str, int] = {}
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        # Attached read-write but only selected from: a read-only URI attach would have to
        # negotiate the live store's WAL.
        conn.execute("ATTACH ? AS src", (str(opyt_db()),))

        copy_ddl(conn, set(_CARRY))

        counts: dict[str, int] = {}
        counts["atoms"] = _copy_json_table(
            conn, "atoms", tuple(_columns(conn, "atoms")), "payload",
            _PAYLOAD_KEYS, payload_dropped)
        counts["entities"] = _copy_json_table(
            conn, "entities", _CARRY["entities"], "profile", _PROFILE_KEYS, profile_dropped)
        for table in ("bodies", "oracles"):
            counts[table] = _copy_table(conn, table, _CARRY[table])
        # After the rows, so the bulk copy is indexed once rather than row by row. The index and
        # its triggers come from `schema.py`, not the source's `sqlite_master`: an export is read
        # by the same search, so it must hold the same index.
        conn.executescript(schema.BODIES_INDEX_DDL)
        schema.rebuild_body_index(conn)
        conn.commit()
        conn.execute("DETACH src")
    finally:
        conn.close()

    os.replace(tmp, out_path)
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "tables": counts,
        # Keys an adapter added that are not on the allow-list, reported rather than lost silently.
        "payload_keys_dropped": payload_dropped,
        "profile_keys_dropped": profile_dropped,
    }
