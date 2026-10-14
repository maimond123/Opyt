"""
pipeline/forget/footprint.py — the atoms an Oracle's own sources pulled into the library.

Forgetting an Oracle or one of its sources with `delete_atoms` deletes these; what the user saved
themselves is never among them.
"""

from __future__ import annotations

from pipeline.forget import atoms
from pipeline.kb import resolve
from pipeline.oracles import pairs


def atom_ids(conn, members: list[str], head: str) -> list[str]:
    """Every atom a subscription pulled for this cluster (`entry_mode='oracle-footprint'`), but
    not from a source another Oracle also added (`resolve.claimed_elsewhere`): that one stays,
    and is theirs once `head` is gone."""
    kept = resolve.claimed_elsewhere(conn, members, head)
    return [r[0] for eid in members if eid not in kept for r in conn.execute(
        "SELECT atom_id FROM atoms WHERE who_id=? AND entry_mode=? ORDER BY atom_id",
        (eid, pairs.FOOTPRINT_ENTRY_MODE))]


def delete(conn, ids: list[str], *, shared: bool) -> dict:
    """Delete the listed atoms, one transaction each, through `atoms.atom`, so every reference a
    single removal cleans is cleaned. One that fails is counted and the rest still go: the Oracle
    or source is already removed, so what is left can be retried with `forget(atom_id=...)`.
    Returns `atoms_deleted`, and `atoms_not_deleted` with `errors` when any failed."""
    deleted, failed = 0, []
    for atom_id in ids:
        try:
            if atoms.atom(conn, atom_id, confirm=True, shared=shared)["status"] == "forgotten":
                deleted += 1
        except Exception as exc:
            failed.append(f"{atom_id}: {type(exc).__name__}: {exc}")
    out = {"atoms_deleted": deleted}
    if failed:
        out["atoms_not_deleted"] = len(failed)
        out["errors"] = failed[:5]
    return out
