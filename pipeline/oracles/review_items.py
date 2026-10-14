"""Every source discovery found for an Oracle, and what became of it: `oracle_review_items`.

Discovery records each finding here (`write_found`): a source it is sure of as `added`, a project
or org as an `also` row. `forget(source=…)` records a removal as `dismissed`. Nothing is asked of
the user (2026-10-14). Any row holds its key: discovery never
records the same source again, so a dismissed source stays dismissed.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline.kb import schema
from pipeline.platforms.url_canon import canonical_identity


ADDED, DISMISSED = "added", "dismissed"
SOURCE, ALSO = "source", "also"


def source_key(source_url: str) -> str:
    """The identity unit used to keep one review choice per actual source.

    `canonical_identity` folds every OpenAlex page to the bare host, which would make all of a
    store's researchers one source; the author or venue id is the identity there."""
    canon = canonical_identity(source_url)
    if canon == "openalex.org":
        return "openalex.org/" + source_url.strip().rstrip("/").rsplit("/", 1)[-1].casefold()
    return canon or source_url.strip().rstrip("/").casefold()


def write_found(conn: sqlite3.Connection, canonical_id: str, items: list[dict]) -> None:
    """Record discovery's findings for one Oracle: each `{source_type, url, kind, status,
    reasons}`. A key already held is left as it is. The caller holds the write transaction."""
    for it in items:
        conn.execute(
            "INSERT INTO oracle_review_items "
            "(canonical_id, source_type, source_url, source_key, kind, reasons, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(canonical_id, source_type, source_key) DO NOTHING",
            (canonical_id, it["source_type"], it["url"], source_key(it["url"]),
             it.get("kind") or SOURCE, json.dumps(list(it.get("reasons") or [])[:2]),
             it["status"]))


def held_source_keys(conn: sqlite3.Connection, canonical_id: str) -> set[tuple[str, str]]:
    """Every source already recorded for this Oracle, whatever became of it: discovery must not
    record or route any of them again.

    Read across every row, not by the stored anchor: a footprint resolve may move a cluster head
    after an item was created, and a dismissal must not disappear exactly when the newly resolved
    source makes that person easier to identify."""
    rows = conn.execute(
        "SELECT canonical_id, source_type, source_key FROM oracle_review_items").fetchall()
    oracle_head = schema.current_canonical(conn, canonical_id)
    return {(row["source_type"], row["source_key"]) for row in rows
            if schema.current_canonical(conn, row["canonical_id"]) == oracle_head}


def dismissed_keys(conn: sqlite3.Connection, canonical_id: str) -> set[tuple[str, str]]:
    """`(source_type, source_key)` of every source the user removed from this Oracle. No pull may
    register one again, even when the Oracle's own profile still links it. A store without the
    table has removed nothing."""
    try:
        rows = conn.execute(
            "SELECT canonical_id, source_type, source_key FROM oracle_review_items WHERE status=?",
            (DISMISSED,)).fetchall()
    except sqlite3.OperationalError:
        return set()
    oracle_head = schema.current_canonical(conn, canonical_id)
    return {(row["source_type"], row["source_key"]) for row in rows
            if schema.current_canonical(conn, row["canonical_id"]) == oracle_head}


def for_oracles(conn: sqlite3.Connection, canonical_ids) -> dict[str, list[dict]]:
    """`head -> [item]`: what discovery found for these Oracles and did not dismiss, oldest
    first, folded to each Oracle's current head. A store without the table has found nothing."""
    want = {schema.current_canonical(conn, c) for c in canonical_ids}
    try:
        rows = conn.execute(
            "SELECT * FROM oracle_review_items WHERE status != ? ORDER BY created_at, review_id",
            (DISMISSED,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, list[dict]] = {}
    for row in rows:
        head = schema.current_canonical(conn, row["canonical_id"])
        if head in want:
            out.setdefault(head, []).append(_row(row))
    return out


def delete_for_oracles(conn: sqlite3.Connection, canonical_ids: list[str]) -> None:
    """Remove review choices when their Oracle subscription is forgotten.

    The subscription deletion already owns the surrounding transaction, so this deliberately does
    not open a nested connection context that could commit its broader roster change early.
    """
    if not canonical_ids:
        return

    heads = {schema.current_canonical(conn, canonical_id) for canonical_id in canonical_ids}
    rows = conn.execute("SELECT review_id, canonical_id FROM oracle_review_items").fetchall()
    conn.executemany("DELETE FROM oracle_review_items WHERE review_id=?",
                     [(row["review_id"],) for row in rows
                      if schema.current_canonical(conn, row["canonical_id"]) in heads])


def _row(row: sqlite3.Row) -> dict:
    item = dict(row)
    try:
        item["reasons"] = json.loads(item.get("reasons") or "[]")
    except ValueError:
        item["reasons"] = []
    return item
