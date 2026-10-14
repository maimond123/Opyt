"""
pipeline/frontier/admit.py — admits one staged candidate into `atoms`.

Drives the existing mint helpers (`ingest_papers.atomize_paper`,
`ingest_github.github_atom_from_url`, `ingest_x.x_atom_from_url`,
`ingest_substack.substack_atom_from_url`, all called with `entry_mode="frontier"`): its atom id
offline, presence, the finder's metadata, library's minter, presence again.

Stage 3 never judges quality: `rejected` means mechanical failure only. It decides on atom
presence before and after each attempt, never on the helper's return value, because the two mint
helpers have inverted dedup contracts and a github refresh can rewrite an atom's entry_mode.
"""

from __future__ import annotations

from typing import NamedTuple

from pipeline.frontier import candidates
from pipeline.ingestion.utils import log
from pipeline.kb import ingest_github, ingest_papers, ingest_substack, ingest_x
from pipeline.kb.schema import stored_atom_id
from pipeline.kb.ingest_common import FETCH_UNDETERMINED
from pipeline.kb.writes import is_lock_error
from pipeline.platforms.papers import lookup

# The mode this rail writes. Never `user-saved`: that is approval, not discovery.
ENTRY_MODE = "frontier"


# ── Identity, computed offline ──────────────────────────────────────────────────
def target_atom_id(kind: str, url: str | None,
                   candidate_id: str = "") -> tuple[str | None, bool]:
    """(atom_id, match_case_insensitively) for a candidate, without touching the network.

    Dispatches on the atom kind, never on `source` (the finder). An unknown kind returns None,
    which `admit_one` turns into `rejected`/`no_atom_id`.
    """
    k = candidates.kind_of(kind)
    if k in ("x", "substack"):
        # Their adapters key a post on its atom's own id (`x:{post id}`, `substack:{post id}`),
        # which a Substack url does not carry.
        return candidate_id or None, False
    if k == "paper":
        paper = ingest_papers.paper_from_url(url or "", enrich=False)      # offline — no S2 call
        return (ingest_papers.paper_atom_id(paper) if paper else None), False
    if k == "repo":
        owner_repo = ingest_github.github_owner_repo(url or "")
        if owner_repo is None:
            return None, True
        # Case-insensitive: github_atom_from_url keys on the API's canonical owner.login, which
        # this offline parse can't know, so an exact-case check could miss an already-stored atom.
        return f"github:{owner_repo[0]}/{owner_repo[1]}", True
    return None, False



# ── The attempt ─────────────────────────────────────────────────────────────────
def known_metadata(row) -> dict:
    """The metadata the finder already supplied, in the field names `paper_from_url(known=…)`
    wants.

    Not an optimization: Semantic Scholar 404s most non-arXiv DOIs, and without this such a paper
    would be stored permanently without title or abstract (papers are immutable under Policy B).
    Source-agnostic: every paper adapter fills `title`, `summary`, `payload["authors"]` and
    `payload["pdf_url"]`.
    """
    payload = candidates.payload_of(row)
    pdf_url = payload.get("pdf_url")
    return {"title": row["title"], "abstract": row["summary"],
            # Library's S2 author shape (`openalexId`, never `authorId`). `who_id` still falls to
            # the `paper-authors:{paper_id}` placeholder when no S2 id resolved, and S2's own
            # authors still win outright when S2 answers.
            "authors": lookup.s2_authors_from_openalex(candidates.payload_authors(payload)),
            "publicationDate": row["published"],
            # The finder's open PDF, in S2's field name, so `pdf.fulltext_pdf_urls` reads it and
            # `_merge_paper` keeps it against an S2 null.
            **({"openAccessPdf": {"url": pdf_url}} if isinstance(pdf_url, str) and pdf_url
               else {})}


def mint(conn, row) -> str | None:
    """Drive the right mint helper. Returns a reason-slug hint; `admit_one` decides on presence."""
    kind = candidates.kind_of(row["kind"])
    url = row["url"]
    if kind == "paper":
        paper = ingest_papers.paper_from_url(url, enrich=True, known=known_metadata(row))
        if paper is None:
            return "no_atom_id"
        ingest_papers.atomize_paper(conn, paper, entry_mode=ENTRY_MODE)
        # Return value discarded on purpose (see module docstring); `_s2_verdict` distinguishes a
        # throttled S2 fetch (retryable) from a generic decline.
        if paper.get(ingest_papers.S2_VERDICT) == FETCH_UNDETERMINED:
            return "blocked_metadata"
        return None
    if kind == "repo":
        ingest_github.github_atom_from_url(conn, url, entry_mode=ENTRY_MODE)
        return None                                   # ditto — presence is the only question
    if kind == "x":
        ingest_x.x_atom_from_url(conn, url, entry_mode=ENTRY_MODE)
        return None                                   # ditto
    if kind == "substack":
        # A paid post is declined (public posts only) and retried until the attempt cap.
        ingest_substack.substack_atom_from_url(conn, url, entry_mode=ENTRY_MODE)
        return None                                   # ditto


class Attempt(NamedTuple):
    """One admission attempt's answer: the candidate's status after it, why, if not admitted, and
    the atom it became, if it was."""
    status: str                    # 'materialized' | 'kept' (failed, retryable) | 'rejected'
                                   # | 'locked' (the store was busy: owed, never charged)
    slug: str | None               # the reason slug; None when materialized
    atom_id: str | None = None     # the stored atom's id, as stored; set only when materialized


def admit_one(conn, row) -> Attempt:
    """One candidate → its `Attempt`. Never raises, never leaves partial state."""
    atom_id, nocase = target_atom_id(row["kind"], row["url"], row["candidate_id"])
    if atom_id is None:
        return Attempt("rejected", "no_atom_id")

    # Presence check before the attempt: avoids re-ingesting (and rewriting entry_mode on) an
    # already-materialized atom, and skips the fetch entirely.
    stored = stored_atom_id(conn, atom_id, nocase=nocase)
    if stored is not None:
        return Attempt("materialized", None, stored)

    hint: str | None = None
    locked = False
    try:
        hint = mint(conn, row)
    except Exception as e:                            # fail-safe: skip, never a partial write
        # A lock that outlasted the write deadline is the store's contention, not this
        # candidate's failure: it stays owed and is never charged an attempt.
        locked = is_lock_error(e)
        hint = "locked" if locked else f"error_{type(e).__name__.lower()}"
        log(f"[frontier-admit] {row['candidate_id']} raised: {type(e).__name__}: {e}")

    # Decide on atom presence, never on the helper's return value (see module docstring).
    if (stored := stored_atom_id(conn, atom_id, nocase=nocase)) is not None:
        return Attempt("materialized", None, stored)
    if locked:
        return Attempt("locked", hint)
    return Attempt("kept", hint or "ingest_declined")
