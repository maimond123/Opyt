"""
pipeline/forget/source.py — forget one source of one Oracle: preview, then remove it for good.

Discovery adds the sources it is sure of without asking (one rule for every source type), so this
is how a user takes one back. The source, named by its URL, is matched to the Oracle that pulls it.
The preview says what the removal keeps and deletes. On confirm, one transaction:
  • marks the source `dismissed`, which keeps discovery from adding it again and keeps pair seeding
    from registering it again (`review_items.dismissed_keys`). It stays dismissed even when the
    person's own X bio still names it;
  • deletes its first-pull row and its refresh registration.
Afterwards, with `delete_atoms`, the atoms pulled from it are deleted one by one through
`atoms.atom`, and resolve runs so the source leaves the person's cluster where nothing else ties it.

Their root (the profile they were added by) is not a source to forget: that is `forget(oracle=…)`.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline.forget import atoms as atoms_job
from pipeline.forget import footprint
from pipeline.kb import expand
from pipeline.kb import resolve
from pipeline.kb import schema
from pipeline.kb.writes import write_tx
from pipeline.oracles import coverage
from pipeline.oracles import pairs
from pipeline.oracles import registry
from pipeline.oracles import review_items
from pipeline.oracles import roster


def forget(conn, source: str, *, oracle: str | None = None, confirm: bool = False,
           delete_atoms: bool = False, shared: bool = False) -> dict:
    """Preview or remove one source of one Oracle. `oracle` narrows the match when the same
    source is attached to more than one Oracle. `shared` is whether this install serves a copy,
    as for `atoms.atom`."""
    key = review_items.source_key(source)
    found = _matches(conn, key)
    if oracle:
        want = {schema.current_canonical(conn, oracle.strip())}
        found = {head: m for head, m in found.items()
                 if head in want or (m["name"] or "").casefold() == oracle.strip().casefold()}
    if not found:
        return {"status": "not_found", "scope": "source", "source": source}
    if len(found) > 1:
        return {"status": "ambiguous", "scope": "source", "source": source,
                "candidates": [{"canonical_id": head, "name": m["name"]}
                               for head, m in found.items()],
                "message": "This source belongs to more than one Oracle. Pass oracle= with one "
                           "canonical_id and preview it again."}
    head, match = next(iter(found.items()))
    name = match["name"] or head
    entities = _entities(conn, head, key, match)
    if _is_root(conn, head, key, entities):
        return {"status": "refused", "scope": "source", "source": source, "canonical_id": head,
                "name": name,
                "message": f"{source} is the profile {name} was added by, not one of their other "
                           f"sources. To stop tracking {name}, use forget(oracle='{head}')."}

    atoms = sum(conn.execute("SELECT COUNT(*) FROM atoms WHERE who_id=?", (eid,)).fetchone()[0]
                for eid in entities)
    deleting = footprint.atom_ids(conn, entities, head) if delete_atoms else []
    shown = match["url"] or source
    out = {"status": "preview", "scope": "source", "source": shown, "canonical_id": head,
           "name": name, "atoms": atoms,
           "consent": consent(shown, name, atoms, len(deleting), delete_atoms=delete_atoms,
                              shared=shared)}
    if delete_atoms:
        out["atoms_deleted"] = len(deleting)
    if not confirm:
        return out

    _remove(conn, head, match)
    out["status"] = "forgotten"
    if delete_atoms:
        out.update(footprint.delete(conn, deleting, shared=shared))
    resolve.resolve_entities(conn)
    return out


def consent(shown: str, name: str, atoms: int, deleting: int, *, delete_atoms: bool,
            shared: bool) -> list[str]:
    """What removing the source `shown` from `name` does: `atoms` is how many it wrote, `deleting`
    how many of them `delete_atoms` deletes."""
    if delete_atoms:
        said = [
            f"This removes one source, {shown}, from {name}, and deletes the atoms Opyt pulled "
            f"from it.",
            f"{deleting} atoms from it are deleted, with their text and search "
            "entries. There is no undo.",
            f"{atoms - deleting} atoms the user saved themselves from it remain.",
        ]
    else:
        said = [
            f"This removes one source, {shown}, from {name}.",
            f"Its {atoms} existing atoms remain. Preview again with delete_atoms=True to also "
            "delete the atoms Opyt pulled from it.",
        ]
    said += [
        f"Opyt stops pulling it and will not add it to {name} again, even if their profile still "
        "links it. Their other sources are not affected.",
    ]
    if delete_atoms and shared:
        said.append(atoms_job.served_copy(one=False))
    return said


# ── matching (reads) ─────────────────────────────────────────────────────────────

def _rows(conn, sql: str) -> list:
    """Rows of one of the scrapers' tables, or none on a store that has not made it yet."""
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []


def _matches(conn, key: str) -> dict[str, dict]:
    """`head -> {name, url, review, found, pairs}`: every Oracle on the roster this source is
    attached to, through a review row, a first-pull row or a refresh registration."""
    out: dict[str, dict] = {}

    def slot(anchor: str) -> dict | None:
        head = schema.current_canonical(conn, anchor)
        row = roster.oracle_row(conn, head)
        if row is None:
            return None
        return out.setdefault(head, {"name": row["name"], "url": None, "review": [],
                                     "found": [], "pairs": []})

    for r in _rows(conn, "SELECT * FROM oracle_review_items WHERE status != 'dismissed'"):
        if r["source_key"] == key and (m := slot(r["canonical_id"])) is not None:
            m["review"].append(dict(r))
            m["url"] = m["url"] or r["source_url"]
    for r in _rows(conn, "SELECT * FROM found_sources"):
        if review_items.source_key(r["url"]) == key and (m := slot(r["canonical_id"])) is not None:
            m["found"].append(dict(r))
            m["url"] = m["url"] or r["url"]
    for r in _rows(conn, "SELECT * FROM oracle_sources"):
        url = pairs.pair_url(r["source_type"], r["source_key"])
        if review_items.source_key(url) == key and (m := slot(r["canonical_id"])) is not None:
            m["pairs"].append(dict(r))
            m["url"] = m["url"] or url
    # A declared ORCID is pulled as an `openalex` pair: the researcher entity carrying that ORCID.
    if key.startswith("orcid.org/"):
        for head, m in out.items():
            for eid, (stype, skey) in pairs.pairs_by_entity(conn, head).items():
                if stype == "openalex" and key in _entity_keys(conn, eid):
                    m["pairs"] += [dict(r) for r in _rows(conn, "SELECT * FROM oracle_sources")
                                   if r["source_type"] == "openalex" and r["source_key"] == skey
                                   and schema.current_canonical(conn, r["canonical_id"]) == head]
    return out


def _entity_keys(conn, entity_id: str) -> set[str]:
    """The source keys of one entity's stored `identity_links` (a JSON array)."""
    row = schema.get_entity(conn, entity_id)
    try:
        links = json.loads(row["identity_links"]) if row and row["identity_links"] else []
    except ValueError:
        links = []
    return {review_items.source_key(u) for u in links if isinstance(u, str)}


def _entities(conn, head: str, key: str, match: dict) -> list[str]:
    """The entity ids whose atoms this source wrote: the cluster members its pairs come from,
    and the entity its first pull mints for a website."""
    ids = []
    pulled = {(p["source_type"], p["source_key"]) for p in match["pairs"]}
    for eid, pair in pairs.pairs_by_entity(conn, head).items():
        if pair in pulled or review_items.source_key(pairs.pair_url(*pair)) == key:
            ids.append(eid)
    for r in (*match["review"], *match["found"]):
        url = r.get("source_url") or r.get("url")
        if eid := coverage.result_entity_id({"type": r["source_type"], "url": url}):
            ids.append(eid)
    return list(dict.fromkeys(ids))


def _is_root(conn, head: str, key: str, entities: list[str]) -> bool:
    """Is this the profile the Oracle was added by: their confirmed anchor, or the root
    discovery starts from?"""
    anchors = {o["canonical_id"] for o in roster.list_oracles(conn)
               if schema.current_canonical(conn, o["canonical_id"]) == head}
    if anchors & set(entities):
        return True
    oracle = next((o for o in roster.confirmed_oracles(conn) if o["canonical_id"] == head), None)
    root = expand._root_profile(conn, oracle) if oracle else None
    return bool(root) and review_items.source_key(expand.profile_url(root)) == key


# ── removal (writes) ─────────────────────────────────────────────────────────────

def _remove(conn, head: str, match: dict) -> None:
    """In one transaction: dismiss the source, and delete its first-pull rows and its refresh
    registrations."""
    with write_tx(conn, site="forget.source"):
        _dismiss(conn, head, match)
        for f in match["found"]:
            conn.execute("DELETE FROM found_sources WHERE canonical_id=? AND source_type=? "
                         "AND url=?", (f["canonical_id"], f["source_type"], f["url"]))
        for p in match["pairs"]:
            registry.delete_pair(conn, (p["canonical_id"], p["source_type"], p["source_key"]))


def _dismiss(conn, head: str, match: dict) -> None:
    """Dismiss every review row this source came in by, and hold each of its pairs with a
    dismissed row of its own: a starting pair, or the `openalex` pair behind a declared ORCID,
    has no row under its own key, and seeding reads that key. The caller holds the transaction."""
    for r in match["review"]:
        conn.execute("UPDATE oracle_review_items SET status='dismissed', "
                     "updated_at=datetime('now') WHERE review_id=?", (r["review_id"],))
    held = {(r["source_type"], r["source_key"]) for r in match["review"]}
    extra = [(f["source_type"], f["url"]) for f in match["found"]]
    extra += [(p["source_type"], pairs.pair_url(p["source_type"], p["source_key"]))
              for p in match["pairs"]]
    for stype, url in extra:
        skey = review_items.source_key(url)
        if (stype, skey) in held:
            continue
        held.add((stype, skey))
        conn.execute(
            "INSERT INTO oracle_review_items "
            "(canonical_id, source_type, source_url, source_key, kind, reasons, status) "
            "VALUES (?, ?, ?, ?, 'source', '[]', 'dismissed') "
            "ON CONFLICT(canonical_id, source_type, source_key) DO UPDATE SET "
            "status='dismissed', updated_at=datetime('now')",
            (head, stype, url, skey))
