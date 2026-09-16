"""
pipeline/kb/ingest_scholar_footprint.py — a scholar Oracle's own papers into the trusted KB.

One job: an OpenAlex id plus a window and an optional topic filter → that id's papers as
`entry_mode='oracle-footprint'` atoms. The counterpart of `ingest_x_footprint` for someone whose
body of work is published rather than posted.

The id is an AUTHOR (`A…`) or a SOURCE (`S…` — a journal, a preprint repository, a venue), and
this module cannot tell which, because it does not need to: `works_filter` reads the prefix and
picks the `/works` field. A venue Oracle is this same pull with a different letter.

THE TOPIC FILTER IS READ FROM THE PAIR, NOT DECIDED HERE. The caller looks it up on the
`oracle_sources` row and threads it in, which is why a first backlog and a refresh six months
later narrow to exactly the same set.

ABSTRACT-ONLY ON THE PULL, AND THAT IS THE DESIGN, NOT A SHORTCUT. The abstract arrives in the
same `/works` response as the metadata, so a back-catalogue pull of 928 papers fetches no PDFs,
opens no connections beyond the paged works call, and has nothing to parallelise. `atomize_paper`
takes `fulltext=None` — a real resolved value, distinct from its `_UNSET` sentinel — which is
exactly the seam that says "no PDF, use the abstract" without re-paying `resolve_fulltext` for
every paper.

THE BODY ARRIVES LATER, ON DEMAND. Every atom this writes is `body_state=partial`, and
`ingest_papers.upgrade_to_fulltext` promotes one to `complete` the first time a reader actually
opens it (`opyt_core/kb.py::kb_open`). That is why `_paper_from_work` carries `openAccessPdf`
even though nothing on THIS path reads it: the pull is what has the url in hand, and the upgrade
runs long after the `/works` response is gone. A claim that the recent window pulls full text
stood in this docstring from 2026-09-08 to 2026-09-11 and was never true of any code — the
refresh rail (`oracle_refresh._dispatch`) calls this same function, `fulltext=None` and all.

WHY THIS NEEDS NO ELIGIBILITY GATE, THOUGH A PAPER IS MULTI-AUTHOR BY DEFINITION.
`expand._route_source` is the gated door for the two website adapters, and its gate exists because
a multi-author site attributed to one trusted person is trust laundering. A paper looks like the
worst case of that — but `atomize_paper` refuses to launder it at the source: `who_id` is the
PAPER's own author, always, and there is no override parameter. The invariant is enforced in the
atomizer rather than by a gate, so this arm goes DIRECT, the way X does, and for the same reason X
does: an OpenAlex author id names exactly one author by construction.

The Oracle's relationship to a paper they co-wrote is `entry_mode='oracle-footprint'`. `who_id`
still names the paper's first author, which may not be the Oracle.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from . import schema
from .frontier_sources import OpenAlexWorksAdapter, SourceError
from .ingest_common import AtomSink, StageTimer

# How many works a single pull will atomize. Not a politeness bound — the paging already bounds
# requests — but a bound on EMBED spend, which is the only per-work cost on this path.
#
# 2,000 is an order of magnitude past the most prolific real researcher measured (Frances Arnold:
# 928 works, 2026-09-08), so on the AUTHOR path it stops a pathological id and never a real one.
# On the VENUE path it is a live truncation: ChemRxiv is 63,565 works, and an unfiltered venue add
# reaches this on the first pull. That is what `capped` in the returned summary exists to say out
# loud — the count-first ask names the real total, and the pull has to admit when it took less.
MAX_WORKS_PER_PULL = 2000


def _paper_from_work(work: dict) -> dict | None:
    """One OpenAlex work → the normalized Paper shape `atomize_paper` takes (Semantic Scholar's
    field names, which is the one paper vocabulary the whole pipeline already speaks).

    Returns None when the work carries no identifier a paper atom can be keyed on, or no body to
    read. Both are drops, not errors: a work with neither DOI nor landing page has no offline
    route to an atom id, and papers are immutable under Policy B, so writing a contentless one
    freezes it forever.
    """
    from .frontier_sources import _abstract_from_inverted

    loc = work.get("primary_location") or {}
    url = work.get("doi") or loc.get("landing_page_url")
    # The OPEN pdf, carried in S2's field name so `_fulltext_pdf_urls` picks it up with no change
    # of its own. It arrives in the SAME `/works` response as the metadata, so keeping it costs
    # nothing here and is the only reason a later `upgrade_to_fulltext` can reach a body at all:
    # S2 does not index Zenodo or most institutional repositories, so for those works this url is
    # the sole route. `best_oa_location` first — the primary location can be the paywalled
    # publisher copy. Discarded until 2026-09-11, which is what made the footprint corpus
    # permanently abstract-only rather than merely abstract-first.
    pdf_url = ((work.get("best_oa_location") or {}).get("pdf_url") or loc.get("pdf_url"))
    title = " ".join((work.get("title") or "").split())
    abstract = _abstract_from_inverted(work.get("abstract_inverted_index"))[:2000]
    if not url or not (title or abstract):
        return None

    published = (work.get("publication_date") or "")[:10]
    doi = (work.get("doi") or "").rsplit("doi.org/", 1)[-1] if work.get("doi") else None
    return {
        "title": title,
        "abstract": abstract,
        "url": url,
        # `openalexId`, never `authorId` — `derive_paper` reads `authorId` to mint
        # `who_id = scholar:{id}`, and an OpenAlex author id is not a Semantic Scholar one.
        "authors": [{"name": a["name"],
                     **({"openalexId": oid} if (oid := a.get("openalex_id")) else {}),
                     **({"orcid": orc} if (orc := a.get("orcid")) else {}),
                     **({"position": pos} if (pos := a.get("position")) else {})}
                    for a in _work_authors(work)],
        "publicationDate": published or None,
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "venue": ((loc.get("source") or {}).get("display_name") or ""),
        "citationCount": work.get("cited_by_count", 0),
        **({"externalIds": {"DOI": doi}} if doi else {}),
        **({"openAccessPdf": {"url": pdf_url}} if pdf_url else {}),
    }


def _work_authors(work: dict) -> list[dict]:
    from .frontier_sources import _openalex_authors
    return _openalex_authors(work)


def sync_scholar_footprint(conn: sqlite3.Connection, embedder, *, openalex_id: str,
                           author_name: str | None = None, since: datetime | None = None,
                           topics: str | None = None, limit: int = MAX_WORKS_PER_PULL,
                           adapter: OpenAlexWorksAdapter | None = None) -> dict:
    """Pull one scholar Oracle's papers over `since..now` and mint them as trusted atoms.

    `topics` is the stored `oracle_sources.topic_filter` for this pair, threaded in by the caller
    rather than looked up here — this module holds no connection to the refresh registry, and the
    two callers (the first backlog and the refresh rail) each already hold the row.

    Takes a plain `embedder`, matching `sync_x_footprint` and every other adapter here. A zero-arg
    closure — so a pull that finds nothing never constructs one — was considered and dropped: the
    only caller is `oracle_refresh`, which builds its embedder eagerly before walking any pair
    because the X arm needs one regardless. A seam with one value that already holds an embedder
    defers nothing.

    Batched through ONE `AtomSink`. `atomize_paper` is the only mint helper and every caller drives
    it one paper at a time; without a shared sink, N papers cost N embed round-trips. `flush_chunks`
    is sized off the embedder's own batch size, the recipe five adapters already share verbatim.

    Never raises for a per-paper problem. `SourceError` from the adapter DOES propagate — it says
    the host is backing off, and the caller must record that as a blocked pull rather than as an
    author who published nothing.
    """
    from pipeline.ingestion.utils import log

    from . import ingest_papers as ip

    adapter = adapter or OpenAlexWorksAdapter()
    if not adapter.available():
        raise SourceError("openalex breaker open — backing off, retried next run")

    works = adapter.works(openalex_id, since=since, limit=limit, topics=topics,
                          pace_seconds=adapter.min_interval_s)
    # A full return means MORE matched than we took. Read off the limit the caller set rather
    # than reported by the adapter, because the caller is the only one that knows the number it
    # asked for was a bound and not a coincidence.
    capped = bool(limit) and len(works) >= limit
    papers = [p for p in (_paper_from_work(w) for w in works) if p]
    if not papers:
        # Nothing published in the window (or nothing in the chosen topics). A real observation,
        # and the caller advances the cursor on it — not a failure.
        return {"source": "openalex", "openalex_id": openalex_id, "fetched": len(works),
                "atoms": 0, "papers": 0, "topics": topics}

    timer = StageTimer()
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, timer=timer, flush_chunks=8 * bs)
    # Seeded FROM THE DB, not an empty dict. `atomize_paper`'s Policy-B check reads
    # `atom_id in seen` when a caller threads one and only falls back to `_atom_exists` when it
    # does not — so an empty dict silently disables the skip, and a second pull would re-fetch,
    # re-render and re-embed every paper already in the store.
    seen: dict = schema.load_hashes(conn, "paper")
    written: list[str] = []
    submitted = deduped = 0
    for paper in papers:
        # `fulltext=None` is the abstract-only seam: a real resolved value meaning "no PDF, use the
        # abstract", NOT the `_UNSET` sentinel that would send every paper through
        # `resolve_fulltext` and its PDF mirrors.
        #
        # `on_written` fires only for a DURABLY stored atom, so a poison chunk the sink isolates is
        # never counted — with a sink, an atom is not durable when `atomize_paper` returns.
        atom_id = ip.atomize_paper(conn, embedder, paper, entry_mode="oracle-footprint",
                                   seen=seen, sink=sink, fulltext=None,
                                   on_written=written.append)
        if atom_id:
            submitted += 1
        else:
            deduped += 1
    sink.close()

    log(f"[scholar-footprint] {openalex_id} ({author_name or '?'}): {len(works)} work(s) fetched"
        f"{' (CAPPED at the per-pull limit)' if capped else ''}, {submitted} submitted, "
        f"{deduped} already present or unmintable, {len(written)} durable")
    return {"source": "openalex", "openalex_id": openalex_id, "fetched": len(works),
            "papers": len(papers), "submitted": submitted, "deduped": deduped,
            "atoms": len(written), "topics": topics, "capped": capped,
            "stage_seconds": dict(timer.totals)}
