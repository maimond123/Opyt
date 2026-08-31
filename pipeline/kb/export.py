"""
pipeline/kb/export.py — the live store projected into ONE self-contained, shareable file.

An export is what a second person's Opyt reads when it queries this knowledge base: the same
`atoms`/`chunks`/`chunks_fts`/`entities` substrate the local tools already run against, plus the
snapshot bodies inlined as a table, minus everything the read path never touches. It is opened by
the SAME retrieval code — nothing about `search`/`open`/`aggregate` knows it is reading an export.
Design record: docs/plans/2026-08-26-foreign-kb-export-builder-phase1.md.

Two properties this file exists to hold, and both are enforced by construction rather than by
remembering:

  • ALLOW-LIST, NEVER DENY-LIST. `_CARRY` names every object that crosses. The builder creates
    only those in a fresh file, so a table added to `schema.py` tomorrow is absent from an export
    until someone puts it here — a decision, not an accident. `tests/kb/test_export.py` asserts
    set EQUALITY against this list, so the addition fails a test rather than shipping.

  • A PROJECTION, NEVER A SECOND HOME. An export is rebuilt whole from the live store and never
    patched. There is no sync path here and there must never be one — a function named sync/
    mirror/reconcile in this module would be proof the data had grown two writable homes.

The DDL is COPIED from the live store's own `sqlite_master`, never re-declared, so the export's
schema cannot drift from `schema.py`. `kb_raw` is the one exception: it exists only in an export,
so this module is where it is declared.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db

from . import schema
from .raw_store import read_snapshot
from .retrieve import _json_obj

# ── the allow-list ───────────────────────────────────────────────────────────────
# Value = the columns to CARRY, or None for "every column this table has". A column left out is
# omitted from the INSERT entirely (so a NOT NULL column takes its DDL default) rather than
# written as NULL. Verified 2026-08-26 by reading every SQL statement on the search/open/aggregate
# paths: opyt_core/kb.py, retrieve.py, raw_store.py, embed.stored_dtype.
_CARRY: dict[str, tuple[str, ...] | None] = {
    "atoms": None,
    # `embed_text` is dropped: 10.1 MB on the live store and no query-time reader — its own DDL
    # comment names its two consumers, a human auditor and restrip_embed_surface.py, both offline.
    # It is `text` with OPYT's renderer output stripped, i.e. a strict SUBSET of a column the
    # export already carries, so this removes bytes and never content.
    "chunks": ("chunk_id", "atom_id", "seq", "char_start", "char_end", "text", "vector"),
    # `identity_links` is read only by screen.py and oracles.py — neither is a read tool.
    "entities": ("entity_id", "name", "canonical_id", "profile"),
    # `aggregate`'s trust-coverage count joins atoms → entities → oracles (opyt_core/kb.py). Only
    # the join key and the label are needed; `source`, `ingest_from`, `ingest_to` and `paused` are
    # collection mechanics that tell a reader nothing about the corpus.
    "oracles": ("canonical_id", "name", "confirmed_at"),
    # REQUIRED, not optional: the vector arm reads `storage_dtype` to decode the blobs, and
    # decoding a float16 blob as float32 is silent garbage rather than an error.
    "kb_meta": None,
    # `peers` must NEVER be added here. It is the READER's registry of whose knowledge bases this
    # install may open (pipeline/kb/peers.py) — it says nothing about the corpus, it names paths on
    # the owner's disk, and since 2026-08-27 its `token` column HOLDS THE OWNER'S OWN READER
    # BEARER TOKENS for every remote knowledge base they subscribe to. Carrying it would publish
    # those credentials to everyone the export is served to. It reads like a tidy thing to ship
    # with a KB ("here's who else I read"); it is the opposite.
}

# Copied with an explicit `rowid` — FTS5 breaks `bm25()` ties by docid, so renumbering the docids
# would reorder equal-scoring chunks and change results for a reason unrelated to the projection.
_FTS_TABLE = "chunks_fts"

# The inlined snapshot bodies. On the live store a body is a file under `opyt_home()/kb_raw/`;
# an export has no filesystem beside it, so the bodies travel in the file. `raw_store.read_body`
# is the one reader that knows about both homes.
_KB_RAW_DDL = """
CREATE TABLE kb_raw (
  atom_id TEXT PRIMARY KEY,
  text    TEXT NOT NULL
)
"""

# ── the free-form JSON columns ───────────────────────────────────────────────────
# `payload` and `profile` are documented verbatim passthroughs, so an adapter can put anything in
# them. Both get a KEY allow-list rather than being dropped: `payload.source_tags` is what `tags=`
# filtering and `aggregate`'s topic distribution run json_each over, and `profile.handle` is the
# arm of `resolve_who` that covers X (whose ids are numeric, so the handle lives only here).

_PAYLOAD_KEYS = frozenset({
    # read by code — dropping any of these breaks a filter, not just a display field
    "source_tags", "body_state", "body_basis",
    # x
    "like_count", "reply_count", "is_quote", "is_thread", "is_article", "has_media",
    "thread_len", "media_substance", "keep_reason", "is_reply", "has_link", "text_len",
    # github
    "stars", "forks", "code_language", "license",
    # paper
    "has_fulltext", "year", "citationCount", "venue",
    # blog / substack
    "lastmod", "word_count", "paywalled",
})

# `handle` is the only key the read path touches. `bio`/`followers`/`verified` and the screen
# classifier's stamps are scraped third-party text that serving search does not need.
_PROFILE_KEYS = frozenset({"handle"})


def _filtered(blob, allowed: frozenset[str], dropped: dict[str, int]) -> str | None:
    """A stored JSON object column → the same object with only `allowed` keys, re-encoded.

    Counts what it dropped into `dropped`, because a key a new adapter adds would otherwise
    vanish from every export in silence — which is the failure mode here, not the drop itself.
    NULL in, NULL out; a malformed blob decodes to `{}` via the same helper the hit card uses, so
    an unparseable row is as empty in the export as it already is in a local result."""
    if blob is None:
        return None
    obj = _json_obj(blob)
    kept = {}
    for key, value in obj.items():
        if key in allowed:
            kept[key] = value
        else:
            dropped[key] = dropped.get(key, 0) + 1
    return json.dumps(kept)


def _columns(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA src.table_info({table})")]


def _copy_ddl(conn, names: set[str]) -> None:
    """Recreate the named objects in the export from the SOURCE's own DDL.

    Copying rather than re-declaring is what keeps the export's schema unable to drift from
    `schema.py`. Indexes come along for the same reason. `sql IS NULL` skips the implicit indexes
    SQLite creates for PRIMARY KEY/UNIQUE — recreating those by hand would be a second declaration
    of something the table DDL already carries."""
    for kind, name, tbl, sql in conn.execute(
            "SELECT type, name, tbl_name, sql FROM src.sqlite_master "
            "WHERE type IN ('table','index') ORDER BY type DESC"):
        if sql is None:
            continue
        if name not in names and not (kind == "index" and tbl in names):
            continue
        conn.execute(sql)


def _copy_table(conn, table: str, carry: tuple[str, ...] | None) -> int:
    """Straight SQL copy of the carried columns. No Python round-trip — `chunks.vector` alone is
    66 MB on the live store, and pulling it through the interpreter to hand it back unchanged
    would be the peak-memory cost of the whole build for no gain."""
    cols = list(carry) if carry is not None else _columns(conn, table)
    names = ", ".join(cols)
    conn.execute(f"INSERT INTO {table} ({names}) SELECT {names} FROM src.{table}")
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _copy_json_table(conn, table: str, carry: tuple[str, ...], column: str,
                     allowed: frozenset[str], dropped: dict[str, int]) -> int:
    """Copy a table whose one free-form JSON column has to pass a key allow-list, so the rows go
    through Python."""
    cols = list(carry)
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


def _copy_fts(conn) -> int:
    cols = _columns(conn, _FTS_TABLE)
    names = ", ".join(cols)
    conn.execute(f"INSERT INTO {_FTS_TABLE} (rowid, {names}) "
                 f"SELECT rowid, {names} FROM src.{_FTS_TABLE}")
    return conn.execute(f"SELECT COUNT(*) FROM {_FTS_TABLE}").fetchone()[0]


def _inline_bodies(conn) -> tuple[int, int]:
    """Every atom's snapshot text, from `opyt_home()/kb_raw/` into the file. Returns
    `(written, missing)`; a snapshot the filesystem has lost is skipped and counted, never fatal
    — an export missing one body is worth strictly more than no export at all."""
    written = missing = 0
    batch: list[tuple[str, str]] = []
    for atom_id, raw_ref in conn.execute(
            "SELECT atom_id, raw_ref FROM src.atoms WHERE raw_ref IS NOT NULL AND raw_ref <> ''"):
        text = read_snapshot(raw_ref)
        if text is None:
            missing += 1
            continue
        batch.append((atom_id, text))
        written += 1
        if len(batch) >= 500:
            conn.executemany("INSERT OR REPLACE INTO kb_raw (atom_id, text) VALUES (?, ?)", batch)
            batch.clear()
    if batch:
        conn.executemany("INSERT OR REPLACE INTO kb_raw (atom_id, text) VALUES (?, ?)", batch)
    return written, missing


def build_export(out_path: Path | str) -> dict:
    """Project the local store into a self-contained export file at `out_path`. Returns a manifest.

    ONE parameter, and the source is deliberately not one of them: the database is `opyt_db()` and
    the bodies are under `opyt_home()`. A `src_db=` argument would let a caller name a database
    whose snapshots live under a DIFFERENT home — the half-sandboxed state `opyt_core/paths.py`
    exists to make impossible (one knob, never a per-file override; `OPYT_DB` was deleted for
    exactly this). A caller that wants a different store sets `$OPYT_HOME`.

    Written to a sibling temp file and renamed into place, so a build that dies part-way leaves no
    file that looks like a finished export."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".building")
    tmp.unlink(missing_ok=True)

    # Bring the source up to date FIRST, the same way every reader of this store already does.
    # An export must project the store as the CODE sees it, not as the bytes were last written:
    # `init_kb_schema` carries in-place migrations (the `crawled` → `oracle-footprint` entry-mode
    # rename, the additive columns), and a store not opened writably since one landed still holds
    # the old values. A reader opens an export READ-ONLY, so nothing downstream would ever fix it
    # — the stale value would just be that KB's answer, forever.
    src = schema.connect()
    try:
        # `kb_meta` is written on FIRST INGEST (embed.ensure_kb_meta), not by `init_kb_schema`, so
        # a store nobody has ingested into does not have the table at all — and the allow-list
        # calls it REQUIRED, because it is what tells a reader's vector arm the blob width. Both
        # facts describe one precondition: there is no knowledge base here to share yet. Checked
        # BEFORE the temp file exists, so a refusal leaves nothing behind, and said on the owner's
        # machine rather than as a SQL syntax error mid-build.
        ingested = src.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                               "AND name='kb_meta'").fetchone()
    finally:
        src.close()
    if not ingested:
        raise ValueError(
            "this store has never ingested anything, so there is no knowledge base to export. "
            "Run `onboard` (or any ingest) first — an export carries the embedding identity a "
            "reader needs to query it, and that is written on first ingest.")

    conn = sqlite3.connect(str(tmp))
    payload_dropped: dict[str, int] = {}
    profile_dropped: dict[str, int] = {}
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        # Attached read-write, and only ever SELECTed from. A read-only URI attach would have to
        # negotiate the live store's WAL, which is a fragile dependency for no gain — this is the
        # same access every other reader of the store already takes.
        conn.execute("ATTACH ? AS src", (str(opyt_db()),))

        names = set(_CARRY) | {_FTS_TABLE}
        _copy_ddl(conn, names)
        conn.execute(_KB_RAW_DDL)

        counts: dict[str, int] = {}
        counts["atoms"] = _copy_json_table(
            conn, "atoms", tuple(_columns(conn, "atoms")), "payload",
            _PAYLOAD_KEYS, payload_dropped)
        counts["entities"] = _copy_json_table(
            conn, "entities", _CARRY["entities"], "profile", _PROFILE_KEYS, profile_dropped)
        for table in ("chunks", "oracles", "kb_meta"):
            counts[table] = _copy_table(conn, table, _CARRY[table])
        counts[_FTS_TABLE] = _copy_fts(conn)
        bodies, bodies_missing = _inline_bodies(conn)
        counts["kb_raw"] = bodies

        meta = conn.execute(
            "SELECT embed_model, embed_dim, provider, storage_dtype, strip_version "
            "FROM kb_meta WHERE id = 1").fetchone()
        conn.commit()
        conn.execute("DETACH src")
    finally:
        conn.close()

    os.replace(tmp, out_path)
    return {
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "tables": counts,
        "bodies_missing": bodies_missing,
        # A key an adapter added that nobody put on the allow-list. Reported rather than dropped
        # in silence — silence is what would make the next adapter's field disappear unnoticed.
        "payload_keys_dropped": payload_dropped,
        "profile_keys_dropped": profile_dropped,
        # What a reader must embed their query with to land in the same subspace as these vectors.
        "embed": ({"model": meta[0], "dim": meta[1], "provider": meta[2],
                   "storage_dtype": meta[3], "strip_version": meta[4]} if meta else None),
    }
