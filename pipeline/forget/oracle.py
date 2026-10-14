"""
pipeline/forget/oracle.py — forget one Oracle: preview, then end the subscription.

Matches the reference to one person on the roster (by name, handle or any id of their cluster),
says what the removal keeps and deletes, and on confirm ends the subscription in one transaction
(`roster.end_subscription`). With `delete_atoms`, the atoms that subscription pulled are then
deleted one by one through `atoms.atom`, so every reference a single removal cleans is cleaned.
"""

from __future__ import annotations

from pipeline.forget import atoms as removal
from pipeline.forget import footprint
from pipeline.kb import resolve
from pipeline.kb import schema
from pipeline.oracles import roster
from pipeline.retrieval.who import resolve_who


# The consent sentences that need nothing but the Oracle's roster row and atom count, so they
# are also true read from a served copy: the connector's preview gives these and leaves out the
# ones below that count refresh registrations or offer `delete_atoms`
# (`service/library_writes.py`).
def stops_tracking(name: str, cid: str) -> str:
    return f"This stops tracking one Oracle: {name} ({cid})."


def atoms_remain(atoms: int) -> str:
    return f"Their {atoms} existing atoms, identity records and trust roots remain."


AFTERWARDS = (
    "Future scheduled refreshes will not select this person. Removal takes effect "
    "at once: whatever a running refresh brings in after it is not kept. Bookmarks and "
    "separately saved links can still arrive.",
    "Adding this Oracle again starts a new subscription.",
)


def forget(conn, reference: str, *, confirm: bool = False, delete_atoms: bool = False,
           shared: bool = False) -> dict:
    """Preview or end one subscription.

    No lease: a scraper unit still running for this Oracle when the removal commits finds it off
    the roster inside its own write transaction, and writes nothing (`roster.on_roster`).

    `delete_atoms` also deletes the cluster's `oracle-footprint` atoms, after the subscription
    ends so nothing re-pulls them. What the user saved themselves stays. `shared` is whether this
    install serves a copy, as for `atoms.atom`.
    """
    out = _forget(conn, reference, confirm=confirm, delete_atoms=delete_atoms, shared=shared)
    if confirm and out["status"] == "forgotten":
        if delete_atoms:
            out.update(footprint.delete(conn, out.pop("footprint_atom_ids"), shared=shared))
        resolve.resolve_entities(conn)     # a source another Oracle also added joins it now
    return out


def _forget(conn, reference: str, *, confirm: bool, delete_atoms: bool = False,
            shared: bool = False) -> dict:
    """End one person's subscription, including pull registrations under old cluster heads.

    Entities, trust roots and existing atoms survive; with `delete_atoms`, the footprint atoms
    are listed here for `forget` to delete once the subscription has ended. The roster and its
    refresh registrations are removed together: the refresh loop reads the registry even
    without an Oracle row.
    """

    with conn:
        if confirm:
            conn.execute("BEGIN IMMEDIATE")
        listed = roster.list_oracles(conn)
        heads = {r["canonical_id"] for r in resolve_who(conn, reference)}
        heads.add(schema.current_canonical(conn, reference))
        matches = [o for o in listed if schema.current_canonical(conn, o["canonical_id"]) in heads
                   or (o["name"] or "").casefold() == reference.casefold()]
        groups = {}
        for o in matches:
            groups.setdefault(schema.current_canonical(conn, o["canonical_id"]), []).append(o)
        if not groups:
            return {"status": "not_found", "scope": "oracle", "oracle": reference}
        if len(groups) > 1:
            return {"status": "ambiguous", "scope": "oracle",
                    "candidates": [{"canonical_id": cid, "name": rows[0]["name"]}
                                   for cid, rows in groups.items()],
                    "message": "Choose one canonical_id and preview it again."}
        cid, matched = next(iter(groups.items()))
        members = [r["entity_id"] for r in schema.entities_for_canonical(conn, cid)]
        atoms = sum(conn.execute("SELECT COUNT(*) FROM atoms WHERE who_id=?", (eid,)
                                 ).fetchone()[0] for eid in members)
        deleting = footprint.atom_ids(conn, members, cid) if delete_atoms else []
        # This registry is initialized by the refresh rail, so a never-refreshed store has none.
        has_registry = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='oracle_sources'").fetchone()
        sources = [dict(r) for r in conn.execute("SELECT * FROM oracle_sources")
                   if schema.current_canonical(conn, r["canonical_id"]) == cid] if has_registry else []
        name = matched[0]["name"] or cid
        kept = atoms - len(deleting)
        if delete_atoms:
            consent = [
                f"This stops tracking one Oracle, {name} ({cid}), and deletes the atoms Opyt "
                "pulled from their own sources.",
                f"Their subscription and {len(sources)} refresh registrations are removed.",
                f"{len(deleting)} atoms from their own sources are deleted, with their text "
                "and search entries. Research jobs' reads of them are removed and "
                "their citations stripped; the claims stay. There is no undo.",
                f"{kept} atoms the user saved themselves (bookmarks, saved links), their identity "
                "records and trust roots remain. Remove those individually with "
                "forget(atom_id=...).",
            ]
        else:
            consent = [
                stops_tracking(name, cid),
                f"Their subscription and {len(sources)} refresh registrations are removed.",
                f"{atoms_remain(atoms)} "
                "Remove unwanted atoms individually with forget(atom_id=...), or preview again "
                "with delete_atoms=True to also delete the atoms pulled from their own sources.",
            ]
        consent += AFTERWARDS
        if delete_atoms and shared:
            consent.append(removal.served_copy(one=False))
        out = {"status": "preview", "scope": "oracle", "canonical_id": cid, "name": name,
               "members": members, "atoms_kept": kept, "sources_removed": len(sources),
               "consent": consent}
        if delete_atoms:
            out["atoms_deleted"] = len(deleting)
        if not confirm:
            return out
        roster.end_subscription(conn, cid)
        out["status"] = "forgotten"
        if delete_atoms:
            out["footprint_atom_ids"] = deleting
        return out
