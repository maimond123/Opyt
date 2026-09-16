"""
pipeline/kb/paper_authors.py — the authors on paper atoms, as screenable people.

One job: turn the author lists on paper atoms already in the store into screenable people. Two
producers over the same walk, separated by which `entry_mode` they read and what the signal is
called: the authors of papers the USER SAVED (`save`), and the people a confirmed ORACLE publishes
with (`coauthor`). No network, no LLM, no embed — every input is a row on disk, written by
`ingest_papers.atomize_paper` at ingest time.

This is the paper-shaped answer to the same question `ingest_curation` answers for X and Substack:
who does the user's own behaviour vouch for? There the answer comes off a platform's follow
primitive; a researcher has none, so authorship of a saved paper is the whole signal. That is why
this is a separate module and not an `ingest_curation` collector — every collector there is a
network pull with a session and a failure mode, and this one reads local rows and cannot fail
that way.

Neither signal is an ENDORSEMENT type (`follow`/`list`/`subscribe`). Saving someone's paper is
reading them, not endorsing them, and a shared byline is weaker still — nobody acted at all.
`screen.Candidate.sort_key` makes that distinction categorically, and `screen.interleave_tiers`
is what stops it burying researchers. Ordering them against X follows is the ranking stage's
problem, not a reason to overstate a signal here.
"""

from __future__ import annotations

import json
import sqlite3

from . import schema

# The registries an author id can come from, most trusted first. A Semantic Scholar id wins when
# both are present, and that is not a quality judgement: `derive.derive_paper` already mints
# `who_id = scholar:{first_author_id}`, and `atomize_paper` upserts THAT id as an entity. Minting
# `openalex:…` for the same person would put a second entity beside it carrying half the signal.
_REGISTRIES = (("scholar_id", "scholar"), ("openalex_id", "openalex"))


def author_entity(author: dict) -> tuple[str, str] | None:
    """One stored author record → `(entity_id, platform)`, or None when it has no registry id.

    An id-less author is SKIPPED, not name-keyed. A name is not an identity — "Frances Arnold"
    matched 16 distinct OpenAlex people on 2026-09-08 — and a candidate minted from one could
    not be acted on anyway: confirming a scholar Oracle pulls their back catalogue by author id,
    so there would be nothing to pull. The count of skipped authors is REPORTED rather than
    swallowed; see `sync_paper_author_signals`.

    `platform` is the registry that issued the id, matching the entity prefix — the same rule
    `x:user:{id}` + platform `x` follows. Two registries knowing one person is then two distinct
    signals once the ORCID merges them, which is honest: two independent registries attested.
    """
    for key, platform in _REGISTRIES:
        if ident := (author.get(key) or "").strip():
            return f"{platform}:{ident}", platform
    return None


def _authors_by_entity(conn: sqlite3.Connection, entry_mode: str) -> tuple[dict, int, int]:
    """The FULL-SET re-read: every paper atom in ONE entry mode, its authors tallied per person.

    Returns `({entity_id: {"name", "platform", "orcid", "papers": [atom_id, …]}}, n_papers,
    n_skipped)`. `entry_mode` is ALLOW-listed by the caller, never deny-listed, per the rule
    `schema.py` states for entry modes — a mode added later must not leak in by default.

    One walk, two producers, because the only thing that differs between "the authors of papers
    the user saved" and "the people an Oracle publishes with" is which mode is read and what the
    resulting signal is called. Written twice first; the second case is what showed the axis.
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
            # COALESCE, not overwrite: one record of a person may carry the ORCID and another may
            # not, and the merge key is worth more than which paper supplied it.
            rec["orcid"] = rec["orcid"] or (a.get("orcid") or None)
            if atom_id not in rec["papers"]:
                rec["papers"].append(atom_id)
    return by_entity, n_papers, n_skipped


def _papers_missing_an_author_list(conn: sqlite3.Connection) -> int:
    """How many paper atoms carry no `payload.authors` — the DETECTOR for the one-time backfill.

    `scripts/backfill_paper_authors.py` exists because papers are immutable under Policy B: every
    atom written before 2026-09-08 has no author list and no run will ever revisit it, so the
    producers above silently yield nothing for a store's existing saves.

    A hand-run pass survives the 2026-09-08 hand-run-entry-point audit only if something can
    DETECT that it is due — `restrip_embed_surface.py` survives on exactly that basis and
    `rechunk.py` was deleted for lacking it ("a maintenance pass whose trigger nothing can detect
    is a pass nobody runs"). This is that trigger, and it rides in the producer's own report, so
    the number appears wherever the pull is read rather than in prose nobody opens.
    """
    return conn.execute(
        "SELECT count(*) FROM atoms WHERE source_type='paper' "
        "AND COALESCE(json_extract(payload, '$.authors'), '[]') IN ('[]', 'null')").fetchone()[0]


def papers_by_author(conn: sqlite3.Connection, entry_mode: str = "user-saved") -> dict[str, list]:
    """`{entity_id: [atom_id, …]}` — which papers each person AUTHORED, in one walk.

    The only correct way to ask it. `atoms.who_id` names the paper's FIRST author and nothing
    else, and 67 of 82 live paper atoms do not even have that (they carry the
    `paper-authors:{paper_id}` placeholder because Semantic Scholar never resolved the work). So
    a `WHERE who_id = …` query answers "which papers did this person lead, when a registry
    happened to know them", which is a different and much smaller question.

    One walk shared with the signal producers rather than a second query with its own idea of
    what authorship means.
    """
    by_entity, _, _ = _authors_by_entity(conn, entry_mode)
    return {eid: rec["papers"] for eid, rec in by_entity.items()}


def sync_paper_author_signals(conn: sqlite3.Connection) -> dict:
    """Every `user-saved` paper's authors → one entity + one `save` signal each. Idempotent.

    `set_signal`, never `add_signal`. This recomputes each author's whole total by walking the
    entire saved set on every run, which is precisely the FULL-SET re-read `set_signal` exists
    for; summing a total into a total is what inflated the live store's `follow/x` from 468 to
    886 in one pass.

    Nothing is truncated and nothing is ranked here. Fifty papers from fifty different labs give
    a few hundred authors all at `count=1`, and capping tied candidates to an arbitrary subset of
    equally tied candidates is not better, it is just arbitrary — so the numbers are REPORTED and
    `screen.Candidate.sort_key` does the ordering. Entities and signals cost no embed and no
    network, so the aggregate is bounded already.

    The ORCID goes in `identity_links` because that is the column `resolve` unions on, and it is
    the ONLY link written: an institution or lab URL would false-merge two researchers who share
    a department, and nobody can claim another person's ORCID.
    """
    by_entity, n_papers, n_skipped = _authors_by_entity(conn, "user-saved")
    stale = _papers_missing_an_author_list(conn)
    for eid, rec in by_entity.items():
        orcid = rec["orcid"]
        schema.upsert_entity(conn, eid, name=rec["name"],
                             identity_links=[f"https://orcid.org/{orcid}"] if orcid else None)
        schema.set_signal(conn, eid, "save", rec["platform"], count=len(rec["papers"]))
    conn.commit()
    out = {"source": "paper-authors", "papers": n_papers, "authors": len(by_entity),
           "multi_paper_authors": sum(1 for r in by_entity.values() if len(r["papers"]) > 1),
           # Named, never silent. An author with no registry id cannot become an Oracle — there
           # is no id to pull a back catalogue by — so they are dropped, and saying how many
           # were dropped is what keeps that a decision rather than a disappearance.
           "authors_without_a_registry_id": n_skipped}
    if stale:
        out["needs_backfill"] = stale
        out["remedy"] = ("python scripts/backfill_paper_authors.py — these paper atoms predate "
                         "the author list and papers are immutable under Policy B, so no run "
                         "revisits them and their authors reach nothing")
    return out


def sync_coauthor_signals(conn: sqlite3.Connection, *, min_papers: int = 2) -> dict:
    """The people a confirmed Oracle publishes WITH → one entity + one `coauthor` signal each.

    A coauthor is a first-degree relationship the Oracle attested by putting their name next to
    this person's. SIGNALS ONLY, never auto-Oracles — the same discipline that disabled the
    second-degree follow scout, and for the same reason: this is the one producer whose input
    grows with the roster rather than with what the user did.

    `min_papers=2` IS that bound, and it is a different decision from the deliberate refusal to
    cap saved-paper authors. There, every author is backed by a user ACTION — the save — so
    capping 300 candidates all at `count=1` would discard evidence indistinguishable from what it
    keeps. Here NO author is backed by a user action; the evidence is entirely inferred from a
    byline. One shared byline is noise (a hyperauthored collaboration, a one-time contribution);
    two is a working relationship. The number dropped is REPORTED, never silent.

    Excluded: the Oracles themselves (a person is not their own candidate) and anyone already
    confirmed. `set_signal`, like every full-set re-read here.
    """
    by_entity, n_papers, n_skipped = _authors_by_entity(conn, "oracle-footprint")
    oracle_ids = {r[0] for r in conn.execute("SELECT canonical_id FROM oracles")}
    written = below = 0
    for eid, rec in by_entity.items():
        if len(rec["papers"]) < min_papers:
            below += 1
            continue
        if eid in oracle_ids or schema.current_canonical(conn, eid) in oracle_ids:
            continue
        orcid = rec["orcid"]
        schema.upsert_entity(conn, eid, name=rec["name"],
                             identity_links=[f"https://orcid.org/{orcid}"] if orcid else None)
        schema.set_signal(conn, eid, "coauthor", rec["platform"], count=len(rec["papers"]))
        written += 1
    conn.commit()
    return {"source": "paper-coauthors", "papers": n_papers, "authors": len(by_entity),
            "signalled": written, "below_min_papers": below,
            "min_papers": min_papers,
            "authors_without_a_registry_id": n_skipped}
