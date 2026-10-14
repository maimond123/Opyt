"""Remove local atoms and their references.

The predicates below own the removal inventory, shared by preview and execution. Schema
equality tests make a new atom-reference table a decision rather than a dangling row.
"""
from __future__ import annotations

from contextlib import nullcontext

from pipeline.kb.writes import write_tx


_ATOM_REFERENCES = {
    "atoms": "atom_id = :atom_id",
    "bodies": "atom_id = :atom_id",
    "frontier_candidates": "atom_id = :atom_id",
    "thread_walks": "atom_id = :atom_id",
}

# What a preview shows to say which atom it is: description, author, date and source URL.
IDENTITY = ("description", "who_id", "when_ts", "source_url")


def atom(conn, atom_id: str, *, confirm: bool = False, shared: bool = False) -> dict:
    """Preview or remove one atom. A confirmed removal owns one database transaction, which
    takes its row, its body (and, by trigger, its search entry) and every reference together."""
    with write_tx(conn, site="forget.atom") if confirm else nullcontext():
        row = conn.execute("SELECT * FROM atoms WHERE atom_id = ?", (atom_id,)).fetchone()
        if row is None:
            return {"status": "not_found", "scope": "atom", "atom_id": atom_id}
        params = {"atom_id": atom_id}
        counts = {table: conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0]
            for table, where in _ATOM_REFERENCES.items()}
        out = {"status": "preview", "scope": "atom", "atom_id": atom_id,
               **{key: row[key] for key in IDENTITY},
               "references": counts, "snapshot": counts["bodies"] > 0,
               "shared": shared, "consent": consent([row], shared)}
        if not confirm:
            return out

        for table, where in _ATOM_REFERENCES.items():
            if table == "frontier_candidates":
                # The row stays: its `materialized` status keeps the candidate from being
                # admitted again (see `_return_path`). Only the pointer to the atom goes.
                conn.execute(f"UPDATE {table} SET atom_id = NULL WHERE {where}", params)
            else:
                conn.execute(f"DELETE FROM {table} WHERE {where}", params)
        out["status"] = "forgotten"
        return out


def atoms(conn, atom_ids: list[str], *, confirm: bool = False, shared: bool = False) -> dict:
    """Preview or remove several atoms as one: one list of identities, one consent, and on
    confirm one transaction, each atom removed by `atom` so every reference it cleans is
    cleaned. An unknown id refuses the whole set before anything is written.

    The preview sums each atom's reference counts, so a row citing two of them counts twice."""
    ids = list(dict.fromkeys(atom_ids))
    with write_tx(conn, site="forget.atoms") if confirm else nullcontext():
        rows = [conn.execute("SELECT * FROM atoms WHERE atom_id = ?", (atom_id,)).fetchone()
                for atom_id in ids]
        unknown = [atom_id for atom_id, row in zip(ids, rows) if row is None]
        if unknown:
            return {"status": "not_found", "scope": "atoms", "unknown": unknown,
                    "message": f"Not in the store: {', '.join(unknown)}. Nothing was removed; "
                               "preview the set again without the unknown ids."}
        each = [atom(conn, atom_id, confirm=confirm, shared=shared) for atom_id in ids]
    counts = {table: sum(e["references"][table] for e in each) for table in _ATOM_REFERENCES}
    return {"status": "forgotten" if confirm else "preview", "scope": "atoms",
            "atoms": [{key: e[key] for key in ("atom_id", *IDENTITY)} for e in each],
            "shared": shared, "consent": consent(rows, shared)}


def consent(rows: list, shared: bool) -> list[str]:
    """What removing these atoms does, each consequence said once, from their rows. The
    connector's preview calls it too, with the served copy's rows (`service/library_writes.py`)."""
    one = len(rows) == 1
    said = [
        f"This removes one atom: {rows[0]['description'] or rows[0]['atom_id']}." if one
        else f"This removes these {len(rows)} atoms.",
        f"{'Its' if one else 'Their'} text and search entr{'y' if one else 'ies'} are deleted.",
        "There is no undo; saving the source again requires a new ingest.",
    ]
    said += dict.fromkeys(_return_path(row) for row in rows)
    if shared:
        said.append(served_copy(one))
    return said


def served_copy(one: bool) -> str:
    """What a removal adds when this install shares: a served copy may still hold the atoms."""
    them = "this atom" if one else "these atoms"
    return (f"This install is configured for sharing, so a served copy may contain {them}. "
            f"{'It disappears' if one else 'They disappear'} from that copy after the next "
            "successful push with reader demand; a reader can still receive the previous copy "
            "before then.")


def _return_path(row) -> str:
    """How a forgotten atom can come back, by how it entered."""
    if row["source_type"] == "x" and row["entry_mode"] == "user-saved":
        return ("If this X post is still in your bookmarks, it will return on an hourly "
                "bookmark pass. Unbookmark it on x.com to keep it removed.")
    if row["entry_mode"] in {"oracle-footprint", "author_referenced"}:
        return ("If its author is still tracked, a due Oracle refresh can restore this atom. "
                "The rail re-checks on its own schedule; forget(oracle=...) ends that "
                "subscription.")
    if row["entry_mode"] == "frontier":
        return ("Its materialized Frontier candidate stays recorded, so that candidate will "
                "not be admitted again automatically.")
    return "This source has no automatic re-ingest for this deposit."
