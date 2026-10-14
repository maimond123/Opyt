"""Cursor pages and individually persisted papers; abstracts remain the background body.

ABSTRACT-ONLY, AND THAT IS THE DESIGN. The abstract arrives in the same `/works` response as the
metadata, so a pull fetches no PDFs. Every atom is `body_state=partial`, and `open()` reads one's
full text when a reader opens it (`kb/paper_text.py`), keeping none of it. That is why
`_paper_from_work` carries `openAccessPdf` even though the pull never reads it.

NO ELIGIBILITY GATE, THOUGH A PAPER IS MULTI-AUTHOR BY DEFINITION. `atomize_paper` refuses to
launder it at the source: `who_id` is the PAPER's own author, always. The Oracle's relationship to
a paper they co-wrote is `entry_mode='oracle-footprint'`, including one already stored because
they linked it (`author_referenced`): their works list is what confirms they wrote it.
"""
from pipeline.platforms.papers.openalex import (WORKS_SELECT, OpenAlexClient, openalex_authors,
                                               work_record, works_filter)
from pipeline.platforms.errors import SourceError
from pipeline.kb import ingest_papers
from pipeline.timeparse import parse_ts
from pipeline.oracles.sources.archive.adapter import Adapter, Page


class OpenAlex(Adapter):
    def initial(self, spec):
        return {"cursor": "*"}

    def host(self, spec, kind, data):
        return "api.openalex.org" if kind == "page" else None

    def page(self, spec, cursor, ctx):
        adapter = ctx.cache.setdefault("openalex", OpenAlexClient())
        if not adapter.available():
            raise SourceError("OpenAlex breaker is open")
        payload = adapter._call("/works", {
            "filter": works_filter(spec["url"], topics=spec.get("topics"), since=parse_ts(spec.get("since"))),
            "sort": "publication_date:desc", "select": WORKS_SELECT, "per-page": "200", **cursor})
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise SourceError("OpenAlex listing did not return results")
        records = payload["results"]
        if not all(isinstance(r, dict) and r.get("id") for r in records):
            raise SourceError("OpenAlex listing contained an invalid work")
        following = (payload.get("meta") or {}).get("next_cursor")
        if following == cursor["cursor"]:
            raise SourceError("OpenAlex repeated its cursor")
        return Page([(r["id"], r) for r in records],
                    [{"cursor": following}] if records and following else [])

    def document(self, conn, spec, data, _ctx, hold):
        paper = _paper_from_work(data)
        if paper is None or not paper.get("abstract"):
            return "skipped"
        aid = ingest_papers.paper_atom_id(paper)
        if aid is None:
            return "skipped"
        if conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (aid,)).fetchone():
            # A paper the Oracle linked is theirs once their own works list carries it.
            return aid if ingest_papers.promote_to_footprint(conn, aid, paper) else "present"
        if built := ingest_papers.paper_atom(conn, paper, entry_mode="oracle-footprint",
                                             read_pdf=False):
            hold(*built)
        return aid


def _paper_from_work(work: dict) -> dict | None:
    """One OpenAlex work → the normalized Paper shape `atomize_paper` takes (Semantic Scholar's
    field names). None when the work has no URL to key an atom on, or no title or abstract.
    """
    rec = work_record(work)
    url, pdf_url, title = rec.url, rec.pdf_url, rec.title
    # The open PDF url rides along in S2's field name (`openAccessPdf`) so a later
    # `open()` can reach a body; for repositories S2 does not index it is the only route.
    abstract = rec.abstract[:2000]
    if not url or not (title or abstract):
        return None

    published = (rec.published or "")[:10]
    doi = rec.doi.rsplit("doi.org/", 1)[-1] if rec.doi else None
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
                    for a in openalex_authors(work)],
        "publicationDate": published or None,
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "venue": rec.venue or "",
        "citationCount": work.get("cited_by_count", 0),
        **({"externalIds": {"DOI": doi}} if doi else {}),
        **({"openAccessPdf": {"url": pdf_url}} if pdf_url else {}),
    }
