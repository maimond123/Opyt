"""Remove one local atom and its references; preserve sitting events and standing queries.

The predicates below own the removal inventory, shared by preview and execution. Schema
equality tests make a new atom-reference table a decision rather than a dangling row.
"""
from __future__ import annotations

import json

from opyt_core.paths import opyt_home
from . import raw_store, schema


_ATOM_REFERENCES = {
    "atoms": "atom_id = :atom_id",
    "chunks": "atom_id = :atom_id",
    "chunks_fts": "atom_id = :atom_id",
    "sitting_atoms": "atom_id = :atom_id",
    "sitting_claims": "EXISTS (SELECT 1 FROM json_each(atom_ids) WHERE value = :atom_id)",
    "frontier_queries":
        "EXISTS (SELECT 1 FROM json_each(source_atom_ids) WHERE value = :atom_id)",
    "sittings": """
        EXISTS (SELECT 1 FROM json_each(seed_atom_ids) WHERE value = :atom_id)
        OR EXISTS (SELECT 1 FROM json_each(skipped)
                   WHERE json_extract(value, '$.atom_id') = :atom_id)
        OR (seed_kind = 'atoms' AND
            instr(',' || seed_ref || ',', ',' || :atom_id || ',') > 0)
    """,
}


def atom(conn, atom_id: str, *, confirm: bool = False, shared: bool = False) -> dict:
    """Preview or remove one atom. A confirmed removal owns one database transaction.

Unlink while the row still exists and the transaction can roll back on a filesystem error.
SQLite and the filesystem cannot commit atomically; a process crash after unlink can leave
the original row with a missing snapshot, which the existing raw reader reports explicitly.
    """
    with conn:
        if confirm:
            conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM atoms WHERE atom_id = ?", (atom_id,)).fetchone()
        if row is None:
            return {"status": "not_found", "scope": "atom", "atom_id": atom_id}
        params = {"atom_id": atom_id}
        counts = {table: conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0]
            for table, where in _ATOM_REFERENCES.items()}
        affected = {r[0] for r in conn.execute(
            "SELECT sitting_id FROM sitting_atoms WHERE atom_id = ?", (atom_id,))}
        affected.update(r[0] for r in conn.execute(
            f"SELECT sitting_id FROM sittings WHERE {_ATOM_REFERENCES['sittings']}", params))
        caches = sum(conn.execute(
            "SELECT COUNT(*) FROM sitting_lens_outputs WHERE sitting_id = ?", (sid,)
        ).fetchone()[0] for sid in affected)
        snapshot = raw_store.resolve_ref(row["raw_ref"]) if row["raw_ref"] else None
        # The persisted pointer is file input at a destructive boundary. Only KB snapshots
        # belong to this operation, including when an older row spells its path absolutely.
        if snapshot is not None:
            snapshot.resolve().relative_to((opyt_home() / "kb_raw").resolve())
        out = {"status": "preview", "scope": "atom", "atom_id": atom_id,
               **{key: row[key] for key in ("description", "who_id", "when_ts", "source_url")},
               "references": counts, "sittings_affected": len(affected),
               "lens_outputs": caches, "snapshot": snapshot is not None and snapshot.exists(),
               "shared": shared,
               "consent": [
                   f"This removes one atom: {row['description'] or atom_id}.",
                   f"Its {counts['chunks']} chunks, search entries, and raw snapshot are deleted.",
                   f"It removes {counts['sitting_atoms']} sitting memberships and "
                   f"{counts['sitting_claims']} claims citing this atom, cleans references in "
                   f"{counts['sittings']} sitting events and {counts['frontier_queries']} standing "
                   f"queries, and discards {caches} cached lens outputs.",
                   "Sitting events and their read history remain; standing queries keep running.",
                   "There is no undo; saving the source again requires a new ingest.",
               ]}
        if row["source_type"] == "x" and row["entry_mode"] == "user-saved":
            out["consent"].append(
                "If this X post is still in your bookmarks, it will return on an hourly "
                "bookmark pass. Unbookmark it on x.com to keep it removed.")
        elif row["entry_mode"] in {"oracle-footprint", "author_referenced"}:
            out["consent"].append(
                "If its author is still tracked, a due Oracle refresh can restore this atom. "
                "The rail checks every ten minutes; forget(oracle=...) ends that subscription.")
        elif row["entry_mode"] == "frontier":
            out["consent"].append(
                "Its materialized Frontier candidate stays recorded, so that candidate will "
                "not be admitted again automatically.")
        else:
            out["consent"].append("This source has no automatic re-ingest for this deposit.")
        if shared:
            out["consent"].append(
                "This install is configured for sharing, so a served copy may contain this atom. "
                "It disappears from that copy after the next successful push with reader demand; "
                "a reader can still receive the previous copy before then.")
        if not confirm:
            return out

        for table, where in _ATOM_REFERENCES.items():
            if table in {"chunks", "chunks_fts"}:
                continue  # replace_chunks owns both indexes together
            if table == "sittings":
                for sitting in conn.execute(f"SELECT * FROM {table} WHERE {where}", params).fetchall():
                    seeds = [a for a in json.loads(sitting["seed_atom_ids"] or "[]") if a != atom_id]
                    skipped = [s for s in json.loads(sitting["skipped"] or "[]")
                               if s["atom_id"] != atom_id]
                    ref = sitting["seed_ref"]
                    if sitting["seed_kind"] == "atoms" and ref:
                        ref = ",".join(a for a in ref.split(",") if a != atom_id)
                    conn.execute(
                        "UPDATE sittings SET seed_atom_ids=?, seed_ref=?, skipped=?, skipped_dupes=? "
                        "WHERE sitting_id=?",
                        (json.dumps(seeds), ref, json.dumps(skipped), len(skipped), sitting["sitting_id"]))
            elif table == "frontier_queries":
                for query in conn.execute(f"SELECT query_id, source_atom_ids FROM {table} "
                                          f"WHERE {where}", params).fetchall():
                    ids = [a for a in json.loads(query["source_atom_ids"]) if a != atom_id]
                    conn.execute("UPDATE frontier_queries SET source_atom_ids=? WHERE query_id=?",
                                 (json.dumps(ids), query["query_id"]))
            else:
                conn.execute(f"DELETE FROM {table} WHERE {where}", params)
        schema.replace_chunks(conn, atom_id, [])
        for sid in affected:
            conn.execute("DELETE FROM sitting_lens_outputs WHERE sitting_id=?", (sid,))
            conn.execute(
                "UPDATE sittings SET atoms=(SELECT COUNT(*) FROM sitting_atoms WHERE sitting_id=?), "
                "tokens=(SELECT COALESCE(SUM(tokens), 0) FROM sitting_atoms WHERE sitting_id=?) "
                "WHERE sitting_id=?", (sid, sid, sid))
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)
        out["status"] = "forgotten"
        return out
