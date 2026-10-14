"""
pipeline/forget/request.py — one `forget` call: name its target, open the store, remove it.

A target is one atom, several atoms, one Oracle, or one source of an Oracle. A preview reads on a
read-only connection; only a confirmed call writes. Whether this install shares a served copy is
read here once, for every kind of removal.
"""

from __future__ import annotations

import sqlite3

from opyt_core.paths import opyt_db
from pipeline import credentials
from pipeline.forget import atoms
from pipeline.forget import oracle as forget_oracle
from pipeline.forget import source as forget_source
from pipeline.kb import schema


def forget(*, atom_id: str | None = None, atom_ids: list[str] | None = None,
           oracle: str | None = None, source: str | None = None,
           delete_atoms: bool = False, confirm: bool = False) -> dict:
    """Preview or confirm the removal of the one target the call names."""
    target = name_target(atom_id, atom_ids, oracle, source, delete_atoms=delete_atoms)
    if "message" in target:
        return {"status": "error", **target}
    if not opyt_db().exists():
        return {"status": "not_found", "scope": target["scope"]}
    conn = schema.connect(read_only=not confirm)
    try:
        shared = bool(credentials.get_credential("opyt_service"))
        return _remove(conn, target, confirm=confirm, delete_atoms=delete_atoms, shared=shared)
    except (OSError, sqlite3.Error, ValueError) as exc:
        return {"status": "error", "message": str(exc)}
    finally:
        conn.close()


def name_target(atom_id: str | None, atom_ids: list[str] | None, oracle: str | None,
                source: str | None, *, delete_atoms: bool) -> dict:
    """The one target the arguments name, as `{"scope": ..., ...}`, or `{"message": ...}` saying
    why they name none. `oracle` beside `source` only narrows it, so the two count as one target;
    a one-item `atom_ids` is one atom."""
    if (atom_id is not None) + (atom_ids is not None) + (
            oracle is not None or source is not None) != 1:
        return {"message": "Specify exactly one of atom_id, atom_ids, oracle or source."}
    if delete_atoms and (atom_id is not None or atom_ids is not None):
        return {"message": "delete_atoms applies to an oracle or a source; "
                           "an atom_id is always deleted."}
    if atom_ids is not None and len(atom_ids) == 1:
        atom_id, atom_ids = atom_ids[0], None
    named = source if source is not None else atom_id if atom_id is not None else oracle
    if atom_ids == [] or (named is not None and not named.strip()):
        return {"message": "The target must not be empty."}
    if source is not None:
        return {"scope": "source", "source": source.strip(), "oracle": oracle}
    if atom_ids is not None:
        return {"scope": "atoms", "atom_ids": atom_ids}
    if atom_id is not None:
        return {"scope": "atom", "atom_id": atom_id}
    return {"scope": "oracle", "oracle": oracle.strip()}


def _remove(conn, target: dict, *, confirm: bool, delete_atoms: bool, shared: bool) -> dict:
    """Preview or remove the target with the removal for its scope."""
    if target["scope"] == "source":
        return forget_source.forget(conn, target["source"], oracle=target["oracle"],
                                    confirm=confirm, delete_atoms=delete_atoms, shared=shared)
    if target["scope"] == "atoms":
        return atoms.atoms(conn, target["atom_ids"], confirm=confirm, shared=shared)
    if target["scope"] == "atom":
        return atoms.atom(conn, target["atom_id"], confirm=confirm, shared=shared)
    return forget_oracle.forget(conn, target["oracle"], confirm=confirm,
                                delete_atoms=delete_atoms, shared=shared)
