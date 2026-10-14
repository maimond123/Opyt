"""
pipeline/oracles/roster.py — who OPYT follows: the `oracles` table, read and written here.

A roster row keeps the id an Oracle was confirmed under, and a later footprint resolve can move
that person's cluster head. Every question asked of the roster about a person therefore folds
both sides to the current head (`schema.current_canonical`, or `schema.ROSTER_HEADS_SQL` in a
join) before comparing: `on_roster`, `oracle_row`, `heads`, `anchors`, `confirmed_oracles`. A
merge rewrites no roster row; two Oracles it made one person read as one, under the older
confirmation. The table's DDL is `schema`'s.
`end_subscription` is forget's one call: it ends a person's subscription across every oracles
table.
"""

from __future__ import annotations

import sqlite3

from pipeline.kb import derive, schema
from pipeline.kb.writes import write_tx
from pipeline.oracles import review_items


def upsert_oracle(conn: sqlite3.Connection, canonical_id: str, name: str | None = None,
                  source: str = "screen") -> None:
    """Confirm one Oracle (idempotent on canonical_id). Re-confirming refreshes name/source but
    never resets confirmed_at or the reserved `paused` (a re-confirm isn't a re-add)."""
    with write_tx(conn, site="roster.upsert_oracle"):
        conn.execute(
            "INSERT INTO oracles (canonical_id, name, source) VALUES (?, ?, ?) "
            "ON CONFLICT(canonical_id) DO UPDATE SET "
            "name=COALESCE(excluded.name, oracles.name), source=excluded.source",
            (canonical_id, name, source),
        )


def list_oracles(conn: sqlite3.Connection) -> list:
    """Every roster row as stored, newest first."""
    return conn.execute(
        "SELECT canonical_id, name, source, confirmed_at, paused "
        "FROM oracles ORDER BY confirmed_at DESC, canonical_id"
    ).fetchall()


def heads(conn: sqlite3.Connection) -> set[str]:
    """The current head of every Oracle on the roster."""
    return {r["head"] for r in conn.execute(schema.ROSTER_HEADS_SQL)}


def oracle_row(conn: sqlite3.Connection, canonical_id: str):
    """The roster row of this person, under any anchor of their current cluster, or None."""
    head = schema.current_canonical(conn, canonical_id)
    return next((o for o in list_oracles(conn)
                 if schema.current_canonical(conn, o["canonical_id"]) == head), None)


def on_roster(conn: sqlite3.Connection, canonical_id: str) -> bool:
    """Is this Oracle still on the roster, under any anchor of its current cluster?"""
    return oracle_row(conn, canonical_id) is not None


def anchors(conn: sqlite3.Connection, head: str) -> list[str]:
    """Every roster id that folds to `head`: one person can hold several rows from before a
    merge."""
    return [o["canonical_id"] for o in list_oracles(conn)
            if schema.current_canonical(conn, o["canonical_id"]) == head]


def confirmed_oracles(conn) -> list[dict]:
    """Every confirmed Oracle with its cluster members and each member's identity_links."""
    out, seen = [], set()
    # Oldest confirmation first, so two Oracles a merge made one person report once, as the
    # confirmation the user would recognise; then newest first, as `list_oracles` orders them.
    for o in sorted(list_oracles(conn), key=lambda o: (o["confirmed_at"] or "", o["canonical_id"])):
        # The stored canonical_id can be stale after a footprint resolve shifted the cluster head;
        # re-anchor to the current head so members (and the reported id) are correct.
        cid = schema.current_canonical(conn, o["canonical_id"])
        if cid in seen:
            continue
        seen.add(cid)
        members = []
        for r in schema.entities_for_canonical(conn, cid):
            links = r["identity_links"]
            members.append({"entity_id": r["entity_id"], "name": r["name"],
                            "identity_links": links})
        out.append({"canonical_id": cid, "name": o["name"], "source": o["source"],
                    "confirmed_at": o["confirmed_at"], "members": members})
    out.sort(key=lambda o: o["canonical_id"])
    out.sort(key=lambda o: o["confirmed_at"] or "", reverse=True)
    return out


def name_for(conn, canonical_id: str) -> str | None:
    """Best display name for a confirmed canonical: prefer a member with a non-null name, prefer
    the X row (its profile carries the richer name). `derive.best_name` over the cluster."""
    members = [{"entity_id": r["entity_id"], "name": r["name"], "profile": r["profile"]}
               for r in schema.entities_for_canonical(conn, canonical_id)]
    name, _handle = derive.best_name(members)
    return name


def end_subscription(conn: sqlite3.Connection, head: str) -> None:
    """End one person's subscription, inside the caller's transaction: their roster rows under
    every anchor (one person can hold several from before a merge), their review items, their
    registry pairs, and the background scrapers' own record of them — what was asked, what
    discovery found, which milestones were shown, archive walks, recommendation reads. Left
    behind, a later re-add would find discovery done and every found source already pulled, and
    start nothing. Every row is matched by current head: a resolve may have moved it since."""
    anchors_ = anchors(conn, head)
    conn.executemany("DELETE FROM oracles WHERE canonical_id=?", [(a,) for a in anchors_])
    review_items.delete_for_oracles(conn, [*anchors_, head])
    # The registry exists once the scrapers or a tool have seeded it; a fresh store has none.
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='oracle_sources'").fetchone():
        conn.executemany(
            "DELETE FROM oracle_sources WHERE canonical_id=? AND source_type=? AND source_key=?",
            [(r["canonical_id"], r["source_type"], r["source_key"])
             for r in conn.execute("SELECT * FROM oracle_sources")
             if schema.current_canonical(conn, r["canonical_id"]) == head])
    for table in ("oracle_asks", "found_sources", "milestones_shown",
                  "archive_walks", "web_recommendation_reads"):
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone():
            ids = {r[0] for r in conn.execute(f"SELECT DISTINCT canonical_id FROM {table}")}
            removed = [(i,) for i in ids if schema.current_canonical(conn, i) == head]
            if table == "archive_walks":
                conn.executemany("DELETE FROM archive_units WHERE walk IN "
                                 "(SELECT key FROM archive_walks WHERE canonical_id=?)", removed)
            conn.executemany(f"DELETE FROM {table} WHERE canonical_id=?", removed)
