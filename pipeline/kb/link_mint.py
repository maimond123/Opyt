"""
pipeline/kb/link_mint.py — one classified url → its artifact atom (`link_router` decides the kind).

  • `predicted_atom_id`  — the atom id derivable from the url alone, offline.
  • `mint_artifact`      — url → artifact atom. Vouch bookkeeping stays with the callers.
"""

from __future__ import annotations

from pipeline.kb.schema import stored_atom_id

from . import ingest_common, ingest_github, ingest_papers, ingest_substack
from .ingest_blog import canon_post_url
from .link_router import parse_tweet_id


def predicted_atom_id(url: str, kind: str, *, content_type: str | None = None) -> str | None:
    """The atom_id derivable from the url alone (plus a known `content_type`), for the free
    already-present check before an artifact fetch. None when the url cannot carry it:

      • substack — keys on the post's numeric id, which only the fetch knows.
      • github   — approximate: the real key uses the API's `owner.login` casing. Safe for an
                   "already have it?" check; never safe to write with."""
    if kind == "github":
        gh = ingest_github.github_owner_repo(url)
        return f"github:{gh[0]}/{gh[1]}" if gh else None
    if kind == "paper":
        # Offline id only. `content_type` is omitted rather than passed as None: existing test
        # doubles don't accept it.
        kw = {"content_type": content_type} if content_type else {}
        parsed = ingest_papers.paper_from_url(url, enrich=False, **kw)
        return ingest_papers.paper_atom_id(parsed) if parsed else None
    if kind == "x":
        tid = parse_tweet_id(url)
        return f"x:{tid}" if tid else None
    if kind == "article":
        return canon_post_url(url)
    return None                              # substack — the post id is behind the fetch


def mint_artifact(conn, url: str, kind: str, *, entry_mode: str = "author_referenced",
                  paper_seen: dict | None = None, gh_seen: dict | None = None,
                  sub_seen: dict | None = None, img_cache: dict | None = None,
                  on_written=None, prefetched: dict | None = None,
                  content_type: str | None = None, isolate: bool = True, store=None) -> dict:
    """One GitHub, paper or Substack URL becomes its artifact atom.
    Reference provenance is recorded by the background reference sources.

    `content_type` — the url's real Content-Type when a caller already fetched it; used for 'paper'.
    `store` — the adapter's write when not `store_atom`: one that holds the atom for a caller that
    lands it with its own state.

    Returns `{"status", "atom_id", "used_prefetch"}` where status is:
      • "present"   — a pre-check found it durable; no network spent.
      • "minted"    — the adapter ran and the atom exists, or will once a `store` that held it
                      writes it.
      • "failed"    — nothing was written; do not point an edge at it.

    'article' and 'x' have their own ingest paths and are not routed here. Never raises for a bad
    link."""
    out = {"status": "failed", "atom_id": None, "used_prefetch": False}
    held = {"store": store} if store else {}

    aid = predicted_atom_id(url, kind, content_type=content_type)
    if aid:
        if stored_atom_id(conn, aid):
            # A deposit of a stored github/paper url records its attestation here. No-op under a
            # machine `entry_mode`.
            ingest_common.promote_atom(conn, aid, entry_mode)
            return {**out, "status": "present", "atom_id": aid}
    elif kind == "github":
        # No derivable id means the url is not a repo: bail before the paid enrich. A paper
        # without a derivable id (PubMed, PMC, Zenodo…) still proceeds; its adapter resolves one.
        return out

    if kind == "github":
        out["used_prefetch"] = bool(prefetched)
        minted = ingest_github.github_atom_from_url(   # → canonical id (casing may differ) or None
            conn, url, entry_mode=entry_mode, seen=gh_seen,
            prefetched=prefetched, on_written=on_written, **held,
            **({"isolate": False} if not isolate else {}))
        return {**out, "status": "minted", "atom_id": minted} if minted else out

    if kind == "substack":
        # No url-derived id, so snapshot the hash ledger first to tell a repeat save from a mint.
        if sub_seen is None:
            sub_seen = ingest_substack.schema.load_hashes(conn, "substack")
        before = set(sub_seen)
        minted = ingest_substack.substack_atom_from_url(   # mint / present → post id, or None
            conn, url, entry_mode=entry_mode, seen=sub_seen, img_cache=img_cache,
            on_written=on_written, **held, **({"isolate": False} if not isolate else {}))
        if not minted:
            return out
        return {**out, "status": ("present" if minted in before else "minted"), "atom_id": minted}

    if kind != "paper":
        return out

    # paper (arXiv / DOI / OpenReview / S2 / raw .pdf) — an abstract-only record
    out["used_prefetch"] = bool(prefetched)
    kw = {"content_type": content_type} if content_type else {}          # see predicted_atom_id
    paper = (prefetched["paper"] if prefetched else                      # enrich to mint
             ingest_papers.paper_from_url(url, **kw))
    if not paper:
        return out
    # The id the adapter resolved, for urls whose id was never in the string (PubMed, Zenodo…).
    aid = aid or ingest_papers.paper_atom_id(paper)
    # The pre-check the top could not run for those urls, so a re-save records its attestation.
    if aid and stored_atom_id(conn, aid):
        ingest_common.promote_atom(conn, aid, entry_mode)
        return {**out, "status": "present", "atom_id": aid}
    minted = ingest_papers.atomize_paper(
        conn, paper, entry_mode=entry_mode, seen=paper_seen, on_written=on_written, **held)
    # None means deduped, failed, or blocked; only the dedup case leaves a stored atom.
    if minted is None and not (aid and stored_atom_id(conn, aid)):
        return out
    return {**out, "status": "minted", "atom_id": aid}
