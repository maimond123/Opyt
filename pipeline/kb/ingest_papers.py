"""
pipeline/kb/ingest_papers.py — the shared paper → atom core every paper source funnels through.

    paper_from_url(url) -> Paper | None          normalize an arXiv / DOI / .pdf / paper-page link
    resolve_fulltext(Paper) -> text | None       download the PDF, extract the full document text
    atomize_paper(conn, Paper, ...)               Paper → one abstract-only atom (source-agnostic)
    paper_from_atom(row, payload) -> Paper        a stored record → the Paper it was built from

A paper is an artifact whose `who_id` is its own author, never the Oracle who referenced it, so
there is no eligibility gate. Dedup is Policy B on `paper:{canonical_id}`: papers are immutable,
so a present atom skips the PDF fetch.
"""

from __future__ import annotations


import json
import re
import sqlite3
from functools import partial
from urllib.parse import urlparse


from . import derive, schema
from .writes import write_tx
from pipeline.ingestion.utils import log
from pipeline.platforms.papers.arxiv import strip_arxiv_version
from pipeline.platforms.papers import landing, lookup, pdf
from pipeline.platforms.papers.registry import MAX_PAYLOAD_AUTHORS
from .ingest_common import (BASIS_OBSERVED, BODY_PARTIAL, FETCH_ABSENT,
                            FETCH_OK, FETCH_UNDETERMINED, body_fields, promote_atom,
                            snapshot_hash, store_atom)

# Key on the Paper dict where `paper_from_url` records whether the S2 metadata fetch answered or
# was blocked. Carried on the Paper because the Paper is what crosses every seam.
S2_VERDICT = "_s2_verdict"

# Keys on the Paper marking a title or date that came from the PDF file, not a registry;
# `atomize_paper` copies them into the payload as `title_source` / `date_source`.
_TITLE_SOURCE = "_title_source"
_DATE_SOURCE = "_date_source"
# Key on the Paper marking an "abstract" that is really the opening of the PDF's text, kept because
# no registry had an abstract (`atomize_paper`); copied into the payload as `abstract_source`.
_ABSTRACT_SOURCE = "_abstract_source"

# ── runaway / quality guards ─────────────────────────────────────────────────────
_PAPER_MAX_CHARS = 500_000        # cap extracted text (mirror x_render._ARTICLE_MAX_CHARS):
                                  # real papers are never this long — this only clips a malicious
                                  # payload so a pathological PDF can't bloat the store.
# A paper with no abstract on any registry keeps this much of its PDF's opening in its place: the
# title block and usually the abstract. 28 of 75 full-text papers in one measured store had none.
_OPENING_CHARS = 3_000


# ══════════════════════════════════════════════════════════════════════════════════
# 1. paper_from_url — link → Paper (the shared helper the 3 link-based sources use)
# ══════════════════════════════════════════════════════════════════════════════════

# arXiv: /abs/{id}, /pdf/{id}[.pdf] or /html/{id} (arXiv's default view for recent papers); id may
# contain a slash (old-style hep-th/9901001).
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(.+?)(?:\.pdf)?(?:[?#].*)?$", re.I)
# The same arXiv id on a reader front-end (Hugging Face papers, alphaXiv). Modern `YYMM.NNNNN` ids
# only: these hosts are not arXiv, so the loose legacy pattern above could turn an unrelated path
# into a paper id.
_ARXIV_MIRROR_RE = re.compile(
    r"(?:huggingface\.co/papers|(?:www\.)?alphaxiv\.org/(?:abs|pdf|overview))/"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
# A DOI in the url's path, on any host (doi.org and publishers' own `/doi/10.…` urls). The leading
# `/` is the guard against false positives. `link_router._DOI_IN_PATH_RE` is the routing half.
_DOI_RE = re.compile(r"/(10\.\d{4,9}/[^\s?#]+)")
# A DOI may contain slashes, so `_DOI_RE` also swallows any view segment a publisher appended
# (`/full`, `/html`); these are trimmed off.
_DOI_VIEW_TAIL_RE = re.compile(
    r"/(?:full|fulltext|full-text|html|pdf|epdf|abstract|meta|references|citations|figures|"
    r"supplemental|supplementary|summary)$", re.I)
# bioRxiv/medRxiv put the version in the url, not the DOI; `.full` and `.full.pdf` ride along.
_BIORXIV_VER_RE = re.compile(r"v\d+(?:\.full(?:-text)?)?(?:\.pdf)?$", re.I)


def _clean_doi(raw: str) -> str:
    """A DOI captured out of a url path → the DOI, with the url's own decoration removed.

    Only a known view word is removed, so a DOI with an unusual final segment is left alone.
    """
    doi = (raw or "").strip().rstrip(".").rstrip("/")
    for _ in range(3):                      # `…/v1.full.pdf` needs more than one pass
        before = doi
        doi = _DOI_VIEW_TAIL_RE.sub("", doi).rstrip("/")
        if doi.lower().startswith("10.1101/"):
            doi = _BIORXIV_VER_RE.sub("", doi)
        if doi == before:
            break
    return doi
# arXiv mints a DOI for every preprint under the 10.48550 prefix. It is an arXiv id wearing a DOI,
# not a second paper — see the DOI branch below for why it is collapsed here.
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)
# SSRN mints every DOI from its abstract id (`abstract_id=3482150` → `10.2139/ssrn.3482150`), so
# the identity needs no request — ssrn.com answers 403 to a server fetch.
_SSRN_RE = re.compile(r"ssrn\.com/\S*?abstract(?:_id)?=(\d+)", re.I)
# Semantic Scholar paper page: /paper/[slug/]{40-hex-hash | corpus-id-digits}.
_S2_RE = re.compile(
    r"semanticscholar\.org/paper/(?:[^/]+/)?(CorpusID:\d+|[0-9a-f]{40}|\d+)", re.I)
# OpenReview: /forum?id=X or /pdf?id=X.
_OPENREVIEW_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([^\s&#]+)", re.I)
# Report archives that mint neither a DOI nor an arXiv id: IACR ePrint (`/2025/2041`, `.pdf`) and
# ECCC (`/report/2026/237/`, `/download`). The capture is the report's landing page, which keys
# the paper and names it (`landing.metadata`). `link_router._PAPER_PATH_RES` is the routing half.
_REPORT_RES = (re.compile(r"(eprint\.iacr\.org/\d{4}/\d+)", re.I),
               re.compile(r"(eccc\.weizmann\.ac\.il/report/\d{4}/\d+)", re.I))


def _parse_paper_url(url: str, *, content_type: str | None = None) -> dict | None:
    """A URL → `{paperId, externalIds, url, openAccessPdf?, landing?, s2_lookup}`, or None if it is
    not a paper link. `paperId` is the deterministic canonical dedup key; `s2_lookup` is the id form the
    Semantic Scholar API accepts (None → skip enrichment).

    `content_type` is the response's real `Content-Type` from a caller that already fetched the
    url, for a PDF whose path has no `.pdf`; a bare url string can never assert it."""
    u = (url or "").strip()
    if not u:
        return None

    m = _ARXIV_RE.search(u)
    if m:
        aid = strip_arxiv_version(m.group(1))
        return {"paperId": f"arXiv:{aid}", "externalIds": {"ArXiv": aid}, "url": u,
                "s2_lookup": f"arXiv:{aid}"}

    m = _ARXIV_MIRROR_RE.search(u)
    if m:
        # Same atom as the arxiv.org link; the user's own url is still what gets stored.
        aid = strip_arxiv_version(m.group(1))
        return {"paperId": f"arXiv:{aid}", "externalIds": {"ArXiv": aid}, "url": u,
                "s2_lookup": f"arXiv:{aid}"}

    m = _DOI_RE.search(u)
    if m:
        raw = _clean_doi(m.group(1))
        # An arXiv DOI is the same preprint as its /abs/ page. Collapsed here, where every link form
        # must dedup to one atom, because OpenAlex, S2 and Crossref all hand back this form.
        am = _ARXIV_DOI_RE.match(raw)
        if am:
            aid = strip_arxiv_version(am.group(1))
            return {"paperId": f"arXiv:{aid}", "externalIds": {"ArXiv": aid}, "url": u,
                    "s2_lookup": f"arXiv:{aid}"}
        doi = raw.lower()                      # DOIs are case-insensitive → canonicalize
        return {"paperId": f"DOI:{doi}", "externalIds": {"DOI": doi}, "url": u,
                "s2_lookup": f"DOI:{doi}"}

    m = _SSRN_RE.search(u)
    if m:
        doi = f"10.2139/ssrn.{m.group(1)}"
        return {"paperId": f"DOI:{doi}", "externalIds": {"DOI": doi}, "url": u,
                "s2_lookup": f"DOI:{doi}"}

    m = _S2_RE.search(u)
    if m:
        sid = m.group(1)
        return {"paperId": sid, "externalIds": {}, "url": u, "s2_lookup": sid}

    m = _OPENREVIEW_RE.search(u)
    if m:
        oid = m.group(1)
        return {"paperId": f"openreview:{oid}", "externalIds": {}, "url": u,
                "openAccessPdf": {"url": f"https://openreview.net/pdf?id={oid}"},
                "s2_lookup": None}

    # A report: keyed on its landing page, so the page and its PDF are one paper.
    for report_re in _REPORT_RES:
        if m := report_re.search(u):
            page = m.group(1).lower()
            return {"paperId": f"url:{page}", "externalIds": {}, "url": u,
                    "landing": f"https://{page}", "s2_lookup": None}

    # A raw hosted PDF: no external id, keyed on the url, which is also the fulltext source.
    p = urlparse(u)
    if p.scheme in ("http", "https") and p.path.lower().endswith(".pdf"):
        canon = f"{(p.netloc or '').lower()}{p.path}"
        return {"paperId": f"url:{canon}", "externalIds": {}, "url": u,
                "openAccessPdf": {"url": u}, "s2_lookup": None}

    # Same raw-PDF case with no `.pdf` in the path, known from the caller's fetch. Keyed on the full
    # url, query included: a download endpoint's query is often the paper's only identity.
    if (p.scheme in ("http", "https") and content_type
            and "application/pdf" in content_type.lower()):
        return {"paperId": f"url:{u}", "externalIds": {}, "url": u,
                "openAccessPdf": {"url": u}, "s2_lookup": None}

    return None


def _merge_paper(minimal: dict, rich: dict) -> dict:
    """S2's rich fields, but keep our deterministic canonical `paperId`, union the external ids,
    and keep a parsed openAccessPdf S2 didn't have."""
    out = dict(rich)
    out["paperId"] = minimal["paperId"]                      # canonical dedup key wins
    ext = dict(rich.get("externalIds") or {})
    ext.update({k: v for k, v in (minimal.get("externalIds") or {}).items() if v})
    out["externalIds"] = ext
    # S2 wins where it has an answer, but its nulls (often `abstract: null`) never erase metadata
    # the caller already held.
    for k, v in minimal.items():
        if v and not out.get(k):
            out[k] = v
    if not (out.get("openAccessPdf") or {}).get("url") and minimal.get("openAccessPdf"):
        out["openAccessPdf"] = minimal["openAccessPdf"]
    return out


def _needs_body(paper: dict) -> bool:
    """Would `atomize_paper` refuse this paper for having nothing to read? Gates the metadata
    resolvers: a title with no abstract is dropped just like a blank paper."""
    return not (paper.get("title") or "").strip() or not (paper.get("abstract") or "").strip()


def _fill_gaps(paper: dict, extra: dict | None) -> None:
    """Merge a resolver's fields into the paper, filling only what it lacks. Mutates. An overwrite
    would swap S2's authors (whose `authorId` mints `who_id`) for ones that cannot."""
    for k, v in (extra or {}).items():
        if not paper.get(k):
            paper[k] = v


# Semantic Scholar's answer, in the library's fetch vocabulary.
_S2_VERDICTS = {lookup.FOUND: FETCH_OK, lookup.ABSENT: FETCH_ABSENT,
                lookup.UNDETERMINED: FETCH_UNDETERMINED}


def paper_from_url(url: str, *, enrich: bool = True, content_type: str | None = None,
                   known: dict | None = None) -> dict | None:
    """Normalize an arXiv / DOI / .pdf / paper-page link → a Paper dict, or None if the URL is not
    a paper. `paperId` is a deterministic canonical id (arXiv > DOI > S2) so every link form of a
    paper dedups to one atom.

    `enrich=True` best-effort fills metadata from Semantic Scholar, then arXiv, OpenAlex and Zenodo,
    then a report's own landing page, while the paper still has no body; failures degrade to the
    parsed Paper, never raise.
    `content_type` passes through to `_parse_paper_url`. `known` is metadata the caller already
    holds, in S2 field names; S2 still wins where it answers.

    Stamps `S2_VERDICT` after the merge (which would drop it), so `frontier/admit.py` can tell a
    throttled attempt from a decided one."""
    parsed = _parse_paper_url(url, content_type=content_type)
    if parsed is None and enrich:
        # Ids that need a lookup to become a DOI (PubMed, PMC, a Zenodo record page); the user's
        # url is kept. Gated on `enrich` so `predicted_atom_id`'s offline pre-check stays free.
        if doi := lookup.looked_up_doi(url or ""):
            parsed = _parse_paper_url(f"https://doi.org/{doi}")
            if parsed is not None:
                parsed["url"] = url
    if parsed is None:
        return None
    paper = {"paperId": parsed["paperId"], "externalIds": dict(parsed.get("externalIds") or {}),
             "url": parsed["url"], "openAccessPdf": parsed.get("openAccessPdf"), "authors": []}
    if known:
        paper.update({k: v for k, v in known.items() if v})   # empties never overwrite the parse
    verdict = FETCH_OK          # `enrich=False` means the caller never asked — not that it was blocked
    if enrich:
        rich, found = lookup.fetch_s2_paper(parsed.get("s2_lookup"))
        verdict = _S2_VERDICTS[found]
        if rich:
            paper = _merge_paper(paper, rich)
        # Fallback resolvers fire only while the paper has no body, the state `atomize_paper`
        # refuses to write. arXiv first: its id is the paper's own.
        if _needs_body(paper):
            _fill_gaps(paper, lookup.arxiv_metadata(paper))
        if _needs_body(paper):
            _fill_gaps(paper, lookup.openalex_metadata(paper))
        # Zenodo last; OpenAlex may subsume it — measure before deleting.
        if _needs_body(paper):
            _fill_gaps(paper, lookup.zenodo_metadata(paper))
        # A report no registry can be asked about by id: its own page names it.
        if _needs_body(paper) and parsed.get("landing"):
            _fill_gaps(paper, landing.metadata(parsed["landing"]))
    paper[S2_VERDICT] = verdict
    return paper


def paper_atom_id(paper: dict) -> str | None:
    """`paper:{canonical_id}` — the atom identity, or None if the paper can't be identified.
    Public so a driver can compute it before calling `atomize_paper` (e.g. to check presence)."""
    pid = derive.canonical_paper_id(paper)
    return f"paper:{pid}" if pid else None


# ══════════════════════════════════════════════════════════════════════════════════
# 2. resolve_fulltext — PDF → full document text
# ══════════════════════════════════════════════════════════════════════════════════

def resolve_fulltext(paper: dict) -> str | None:
    """The full document text (not the abstract) for a Paper, or None → the caller builds an
    abstract-only atom. Open sources first; every failure degrades to None, never raises. Text is
    capped at `_PAPER_MAX_CHARS`.

    Mutates the Paper: fills a missing title (and year, when no registry dated it) from the PDF,
    stamped with `_TITLE_SOURCE` / `_DATE_SOURCE`. This is the only place holding both."""
    # Only when nothing better is known. S2, `known=` and every metadata fallback win.
    found = pdf.fulltext(pdf.PdfSources(
        urls=tuple(pdf.fulltext_pdf_urls(paper)),
        need_title=not (paper.get("title") or "").strip(),
        need_year=not paper.get("publicationDate") and not paper.get("year")))
    if found is None:
        return None
    if found.title:
        paper["title"], paper[_TITLE_SOURCE] = found.title, found.title_source
    if found.year:
        paper["year"], paper[_DATE_SOURCE] = found.year, "pdf_created"
    return found.text[:_PAPER_MAX_CHARS]


# ══════════════════════════════════════════════════════════════════════════════════
# 3. markdown + atomize_paper
# ══════════════════════════════════════════════════════════════════════════════════

def paper_to_markdown_full(paper: dict, full_text: str | None) -> str:
    """Render a Paper (+ optional full body) to the snapshot markdown that is stored and indexed.

    Frontmatter is provenance; the rest is title, authors, venue/year, abstract and, when present,
    the full text. The whole snapshot is searchable, frontmatter included."""
    title = (paper.get("title") or "Untitled").replace("\n", " ").strip()
    abstract = (paper.get("abstract") or "").strip()
    authors = paper.get("authors") or []
    names = [(a.get("name") or "").strip() for a in authors if (a.get("name") or "").strip()]
    authors_str = ", ".join(names[:12]) or "—"
    if len(names) > 12:
        authors_str += f" (+{len(names) - 12} more)"
    year = paper.get("year") or ""
    venue = (paper.get("venue") or "").replace("\n", " ").strip()
    cites = paper.get("citationCount", 0)
    url = paper.get("url") or ""
    ext = paper.get("externalIds") or {}
    arxiv, doi = ext.get("ArXiv"), ext.get("DOI")
    when = derive.derive_paper(paper)["when_ts"]

    fm = ("---\n"
          "source: paper\n"
          f"url: {url}\n"
          f"date: {when}\n"
          "type: paper\n"
          "---\n\n")

    body = f"# {title}\n\n**Authors:** {authors_str}\n\n"
    if venue:
        body += f"**Venue:** {venue}\n\n"
    if year:
        body += f"**Year:** {year}  \n"
    body += f"**Citations:** {cites}\n\n"
    if abstract and paper.get(_ABSTRACT_SOURCE) == "pdf_opening":
        body += f"## Opening of the paper\n\n{abstract}\n\n"
    elif abstract:
        body += f"## Abstract\n\n{abstract}\n\n"
    if full_text:
        body += f"## Full text\n\n{full_text}\n\n"

    links = []
    if url:
        links.append(f"[Semantic Scholar]({url})")
    if arxiv:
        links.append(f"[arXiv](https://arxiv.org/abs/{arxiv})")
    if doi:
        links.append(f"[DOI](https://doi.org/{doi})")
    oa = (paper.get("openAccessPdf") or {}).get("url")
    if oa:
        links.append(f"[PDF]({oa})")
    if links:
        body += "## Links\n\n" + " · ".join(links) + "\n\n"

    return fm + body


def atom_authors(paper: dict) -> list[dict]:
    """A normalized Paper's authors → the list the atom carries, in OPYT's own vocabulary.

    `who_id` records only the first author (often a placeholder), so this is where a paper's
    people live. Registry ids stay under separate keys (`scholar_id`, `openalex_id`) because they
    are different namespaces; the ORCID is what they merge on.
    """
    out = []
    for a in (paper.get("authors") or [])[:MAX_PAYLOAD_AUTHORS]:
        if not isinstance(a, dict):
            continue
        name = (a.get("name") or "").strip()
        if not name:
            continue
        rec = {"name": name}
        if sid := a.get("authorId"):
            rec["scholar_id"] = str(sid)
        if oid := a.get("openalexId"):
            rec["openalex_id"] = str(oid)
        if orcid := a.get("orcid"):
            rec["orcid"] = str(orcid).rsplit("/", 1)[-1]
        if pos := a.get("position"):
            rec["position"] = pos
        out.append(rec)
    return out


def promote_to_footprint(conn: sqlite3.Connection, atom_id: str, paper: dict) -> bool:
    """A paper stored because an Oracle linked it (`author_referenced`), now found in their own
    works list → `oracle-footprint`, with the works list's authors (which carry the registry ids
    `curation.papers` keys coauthors on). The body is unchanged, so nothing is rewritten. False when
    the atom is not `author_referenced` (a saved or footprint paper stays as it is)."""
    with write_tx(conn, site="papers.promote"):
        if not schema.claim_referenced(conn, atom_id=atom_id):
            return False
        conn.execute("UPDATE atoms SET payload = json_set(COALESCE(payload, '{}'), '$.authors', "
                     "json(?)) WHERE atom_id = ?", (json.dumps(atom_authors(paper)), atom_id))
    return True


def _pdf_opening(paper: dict) -> str | None:
    """The opening of the paper's open PDF, standing in for a missing abstract, or None. Reads the
    whole PDF, which can also fill in a missing title and year (`resolve_fulltext`)."""
    text = resolve_fulltext(paper)
    opening = (text or "")[:_OPENING_CHARS].strip()
    return opening or None


def atomize_paper(conn: sqlite3.Connection, paper: dict, *,
                  entry_mode: str = "author_referenced",
                  seen: dict | None = None,
                  on_written=None, read_pdf: bool = True, store=store_atom) -> str | None:
    """Normalized Paper → one abstract-only atom; the caller supplies only `entry_mode`.

    A record keeps the abstract and never the full text: `open()` reads the
    full text, and the KB stays the size of its abstracts. With no abstract on any registry, the
    opening of the open PDF stands in for it, unless the caller says not to read PDFs (`read_pdf`).

    Returns the atom_id, or None if already present (the save is still recorded), unidentifiable,
    bodyless, or the write failed. `who_id` is the paper's own author, never the Oracle;
    there is deliberately no override. `seen` is `{atom_id: raw_hash}` (absent → DB check).
    `on_written(atom_id)` fires once the atom is durable (after a caller's `write_tx` commits).
    `store` is the write (`store_atom`); a caller that lands the atom with its own state later
    passes one that holds it."""
    atom_id = paper_atom_id(paper)
    if atom_id is None:
        return None
    # Policy B — skip an already-ingested (immutable) paper before the PDF fetch.
    already = (atom_id in seen) if seen is not None else bool(schema.stored_atom_id(conn, atom_id))
    if already:
        # Not re-fetched, but the save is recorded (a no-op for the machine entry mode).
        promote_atom(conn, atom_id, entry_mode)
        return None
    built = paper_atom(conn, paper, entry_mode=entry_mode, read_pdf=read_pdf)
    if built is None:
        return None
    atom, md = built
    try:
        store(conn, atom, md, on_written=on_written and partial(on_written, atom_id))
    except Exception as e:                    # write failure → skip (no atom, no seen mark)
        log(f"[footprint] paper atom {atom_id} skipped (write failed): {e}")
        return None
    if seen is not None:
        seen[atom_id] = atom["raw_hash"]
    return atom_id


def paper_atom(conn: sqlite3.Connection, paper: dict, *, entry_mode: str,
               read_pdf: bool = True) -> tuple[dict, str] | None:
    """A Paper → `(atom, snapshot)` ready for `store_atom`, or None when it has no abstract and
    no PDF opening. Registers the paper's author entity and writes nothing else, so a caller
    that lands the atom with other state (the OpenAlex archive) writes it in its own
    transaction."""
    atom_id = paper_atom_id(paper)
    if not (paper.get("abstract") or "").strip() and read_pdf and (opening := _pdf_opening(paper)):
        paper["abstract"], paper[_ABSTRACT_SOURCE] = opening, "pdf_opening"
    # Derived after the PDF read, which can fill in the title and year.
    meta = derive.derive_paper(paper)

    # No abstract and no PDF opening: skip, since Policy B would freeze an empty atom. Asks "is
    # there a body?" rather than why not, so an S2 404 can't write a false `absent/observed`
    # claim. Retries are bounded by stage 3's attempt cap.
    if not (paper.get("abstract") or "").strip():
        log(f"[paper] {atom_id} SKIPPED — no abstract and no PDF opening; nothing "
            f"written (s2_verdict={paper.get(S2_VERDICT)}), will retry until the attempt cap")
        return None

    md = paper_to_markdown_full(paper, None)

    raw_hash = snapshot_hash(md)

    # Register the paper's own author as an entity (display name only — no identity_links, since
    # an S2 author id has nothing to merge on; unlike a blog home, it does not unify with an Oracle).
    schema.upsert_entity(conn, meta["who_id"], name=meta["who_name"])

    atom = {
        "atom_id": atom_id,
        "source_type": "paper",
        "what_kind": "artifact",              # a research artifact (like a repo), not a hot take
        "who_id": meta["who_id"],               # the paper's author, not the Oracle
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": paper.get("url") or "",
        "raw_hash": raw_hash,
        "description": meta["description"],
        "entry_mode": entry_mode,             # 'author_referenced' — Oracle referenced, didn't author
        # Always `partial`: the body is the abstract (the no-body case returned above).
        # `title`/`abstract`/`external_ids`/`pdf_url`/`authors` are carried as data so
        # `paper_from_atom` never parses the snapshot markdown back into fields.
        "payload": {"has_fulltext": False, "year": paper.get("year"),
                    "citationCount": paper.get("citationCount", 0),
                    "venue": paper.get("venue", ""),
                    "title": (paper.get("title") or "").replace("\n", " ").strip(),
                    "abstract": (paper.get("abstract") or "").strip(),
                    "external_ids": paper.get("externalIds") or {},
                    "pdf_url": (paper.get("openAccessPdf") or {}).get("url") or "",
                    # Present only when the PDF file supplied the field, not a registry.
                    **({"title_source": src} if (src := paper.get(_TITLE_SOURCE)) else {}),
                    **({"date_source": src} if (src := paper.get(_DATE_SOURCE)) else {}),
                    **({"abstract_source": src} if (src := paper.get(_ABSTRACT_SOURCE)) else {}),
                    # All authors, capped — not just `who_id`'s first.
                    "authors": atom_authors(paper),
                    **body_fields(BODY_PARTIAL, BASIS_OBSERVED)},
    }
    return atom, md


# ══════════════════════════════════════════════════════════════════════════════════
# 4. paper_from_atom — a stored record back to its Paper
# ══════════════════════════════════════════════════════════════════════════════════
# `open()` reads a record's full text (`kb/paper_text.py`) from the Paper the record was built from,
# rebuilt from its columns and payload, never by parsing the snapshot markdown back into fields.

def paper_from_atom(row: sqlite3.Row, payload: dict) -> dict:
    """The stored atom → the Paper shape `resolve_fulltext` and `paper_to_markdown_full` take.

    Rebuilt from columns and payload only, never by parsing the snapshot markdown. `paperId` is
    pinned to the stored atom id."""
    when_ts = row["when_ts"] or ""
    return {
        "paperId": row["atom_id"].split(":", 1)[1] if ":" in row["atom_id"] else row["atom_id"],
        "title": payload.get("title") or "",
        "abstract": payload.get("abstract") or "",
        "authors": _authors_to_s2(payload.get("authors")),
        "url": row["source_url"] or "",
        "venue": payload.get("venue") or "",
        "year": payload.get("year"),
        "citationCount": payload.get("citationCount") or 0,
        "externalIds": payload.get("external_ids") or {},
        **({"publicationDate": when_ts} if row["when_precision"] == "day" and when_ts else {}),
        **({"openAccessPdf": {"url": u}} if (u := payload.get("pdf_url")) else {}),
    }


def _authors_to_s2(stored: list | None) -> list[dict]:
    """The atom's author records → back into the S2 field names a Paper speaks (`derive_paper` keys
    `who_id` off `authorId`). Lossy only in the capped count of unshown authors."""
    out = []
    for a in (stored or []):
        if not isinstance(a, dict) or not (a.get("name") or "").strip():
            continue
        rec = {"name": a["name"]}
        if sid := a.get("scholar_id"):
            rec["authorId"] = sid
        if oid := a.get("openalex_id"):
            rec["openalexId"] = oid
        if orcid := a.get("orcid"):
            rec["orcid"] = orcid
        if pos := a.get("position"):
            rec["position"] = pos
        out.append(rec)
    return out
