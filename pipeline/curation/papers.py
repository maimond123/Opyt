"""The authors on paper atoms, as screenable people.

Two producers over the same walk, separated by which `entry_mode` they read and what the signal is
called: the authors of papers the user saved (`save`), and the people a confirmed Oracle publishes
with (`coauthor`). No network, no LLM, no embed: every input is a row written by
`ingest_papers.atomize_paper`. Neither signal is an endorsement type.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline.curation import signals
from pipeline.kb import schema

# The registries an author id can come from, in order. Scholar wins because `atomize_paper`
# already upserts `scholar:{id}` as the entity; minting `openalex:…` too would split the signal.
_REGISTRIES = (("scholar_id", "scholar"), ("openalex_id", "openalex"))


def author_entity(author: dict) -> tuple[str, str] | None:
    """One stored author record → `(entity_id, platform)`, or None when it has no registry id
    (an id-less author is skipped, not name-keyed: a name is not an identity). `platform` is the
    registry that issued the id, matching the entity prefix."""
    for key, platform in _REGISTRIES:
        if ident := (author.get(key) or "").strip():
            return f"{platform}:{ident}", platform
    return None


def _authors_by_entity(conn: sqlite3.Connection, entry_mode: str) -> tuple[dict, int, int]:
    """The full-set re-read: every paper atom in one entry mode, its authors tallied per person.

    Returns `({entity_id: {"name", "platform", "orcid", "papers": [atom_id, …]}}, n_papers,
    n_skipped)`. `entry_mode` is allow-listed by the caller, never deny-listed, so a mode added
    later does not leak in by default.
    """
    by_entity: dict = {}
    n_papers = n_skipped = 0
    for atom_id, payload in conn.execute(
            "SELECT atom_id, payload FROM atoms "
            "WHERE source_type='paper' AND entry_mode=?", (entry_mode,)):
        try:
            authors = (json.loads(payload or "{}") or {}).get("authors") or []
        except (TypeError, ValueError):
            continue
        n_papers += 1
        for a in authors:
            if not isinstance(a, dict):
                continue
            ident = author_entity(a)
            if ident is None:
                n_skipped += 1
                continue
            eid, platform = ident
            rec = by_entity.setdefault(eid, {"name": a.get("name"), "platform": platform,
                                             "orcid": None, "papers": []})
            # Keep the first ORCID seen: one record of a person may carry it and another not.
            rec["orcid"] = rec["orcid"] or (a.get("orcid") or None)
            if atom_id not in rec["papers"]:
                rec["papers"].append(atom_id)
    return by_entity, n_papers, n_skipped


def _papers_missing_an_author_list(conn: sqlite3.Connection) -> int:
    """How many paper atoms carry no `payload.authors`: the detector that tells when
    `scripts/backfill_paper_authors.py` is due (papers are immutable, so no run revisits them)."""
    return conn.execute(
        "SELECT count(*) FROM atoms WHERE source_type='paper' "
        "AND COALESCE(json_extract(payload, '$.authors'), '[]') IN ('[]', 'null')").fetchone()[0]


def papers_by_author(conn: sqlite3.Connection, entry_mode: str = "user-saved") -> dict[str, list]:
    """`{entity_id: [atom_id, …]}`: which papers each person authored. Use this, not
    `atoms.who_id`, which names at most the first author."""
    by_entity, _, _ = _authors_by_entity(conn, entry_mode)
    return {eid: rec["papers"] for eid, rec in by_entity.items()}


def _tally(eid: str, rec: dict, extra: dict | None = None) -> signals.Tally:
    """An author's papers as their count. The ORCID is the only link written: see
    `sync_paper_author_signals`."""
    orcid = rec["orcid"]
    return signals.Tally(signals.Person(eid, name=rec["name"],
                                        site=f"https://orcid.org/{orcid}" if orcid else None),
                         len(rec["papers"]), extra)


def sync_paper_author_signals(conn: sqlite3.Connection) -> dict:
    """Every `user-saved` paper's authors → one entity + one `save` signal each. Idempotent:
    each author's whole total is recomputed and set, never added. Nothing is capped or ranked
    here. The ORCID is the only identity link written."""
    by_entity, n_papers, n_skipped = _authors_by_entity(conn, "user-saved")
    stale = _papers_missing_an_author_list(conn)
    for eid, rec in by_entity.items():
        signals.counts(conn, [_tally(eid, rec)], "save", rec["platform"])
    out = {"source": "paper-authors", "papers": n_papers, "authors": len(by_entity),
           "multi_paper_authors": sum(1 for r in by_entity.values() if len(r["papers"]) > 1),
           # Authors with no registry id cannot become an Oracle, so they are dropped and counted.
           "authors_without_a_registry_id": n_skipped}
    if stale:
        out["needs_backfill"] = stale
        out["remedy"] = ("python scripts/backfill_paper_authors.py — these paper atoms predate "
                         "the author list and papers are immutable under Policy B, so no run "
                         "revisits them and their authors reach nothing")
    return out


def sync_coauthor_signals(conn: sqlite3.Connection, *, min_papers: int = 2) -> dict:
    """The people a confirmed Oracle publishes with → one entity + one `coauthor` signal each.
    Signals only, never Oracles. `min_papers` bounds it: one shared byline is noise, two is a
    working relationship; the number dropped is reported. Oracles themselves are excluded.
    `extra.with` names the Oracles (canonical ids) on the shared papers, so "who has Imas worked
    with" reads one Oracle's coauthors."""
    by_entity, n_papers, n_skipped = _authors_by_entity(conn, "oracle-footprint")
    oracles = signals.oracle_ids(conn)
    oracles_on: dict[str, set[str]] = {}            # paper → the Oracles among its authors
    for eid, rec in by_entity.items():
        head = schema.current_canonical(conn, eid)
        if head in oracles:
            for paper in rec["papers"]:
                oracles_on.setdefault(paper, set()).add(head)
    written = below = 0
    for eid, rec in by_entity.items():
        if len(rec["papers"]) < min_papers:
            below += 1
            continue
        if signals.is_oracle(conn, eid, oracles):
            continue
        with_ = sorted(set().union(*(oracles_on.get(p, set()) for p in rec["papers"])))
        signals.counts(conn, [_tally(eid, rec, {"with": with_})], "coauthor", rec["platform"])
        written += 1
    return {"source": "paper-coauthors", "papers": n_papers, "authors": len(by_entity),
            "signalled": written, "below_min_papers": below,
            "min_papers": min_papers,
            "authors_without_a_registry_id": n_skipped}


def coauthors_of(conn: sqlite3.Connection, oracle_head: str) -> list[dict]:
    """One Oracle's coauthors, most shared papers first: `[{"name", "canonical_id", "papers"}]`,
    read from the `coauthor` signals' `extra.with` (`sync_coauthor_signals`). No network."""
    rows = conn.execute(
        "SELECT s.entity_id, s.count, e.name, COALESCE(e.canonical_id, s.entity_id) AS head "
        "FROM curation_signals s LEFT JOIN entities e ON e.entity_id = s.entity_id "
        "WHERE s.signal_type = 'coauthor' AND EXISTS (SELECT 1 FROM json_each(s.extra, '$.with') "
        "WHERE value = ?) ORDER BY s.count DESC, s.entity_id", (oracle_head,)).fetchall()
    return [{"name": r["name"] or r["entity_id"], "canonical_id": r["head"],
             "papers": int(r["count"] or 0)} for r in rows]
