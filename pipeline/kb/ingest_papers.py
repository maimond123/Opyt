"""
pipeline/kb/ingest_papers.py — the shared "paper → atom" core (the reusable primitive).

Papers enter the KB from FOUR sources (X timeline / X bio / personal blog / Radar), but the
differences between them are ONLY: how a paper is *found*, and `entry_mode`. Every source funnels
the paper it found through the SAME three primitives here:

    paper_from_url(url) -> Paper | None          normalize an arXiv / DOI / .pdf / paper-page link
    resolve_fulltext(Paper) -> text | None       download the PDF, extract the FULL document text
    atomize_paper(conn, embedder, Paper, ...)     Paper → ONE full-text atom (source-agnostic)

The one genuinely NEW capability vs every other adapter is `resolve_fulltext`: every other source
hands you a body that is already text, but a paper's metadata (arXiv / Semantic Scholar) gives you
only title + authors + abstract — the BODY lives in a PDF. So the new work is: fetch the PDF and
extract its text, OPEN mirrors first, and fall back to the abstract when none is reachable.

Loop shape mirrors `ingest_blog.sync_blog_footprint` (policy-B dedup, snapshot → chunk → embed →
store, skip-and-count fail-safe). What differs from the footprint adapters:

  • NO eligibility gate. The footprint adapters (`sync_blog_footprint`/`sync_substack_footprint`)
    attribute a person's OWN site to them, so they need the single-author gate to stop
    trust-laundering. A paper's authorship is ATTESTED by arXiv/S2 metadata and `who_id` is the
    paper's OWN author (never the Oracle), so there is nothing to launder — hence no gate, and
    this module is deliberately NOT in `.guards.py`'s footprint-adapter rule.
  • `what_kind="artifact"` — a paper is a research artifact (like a repo), not a hot take.
  • `entry_mode="author_referenced"` — the atom entered because a tracked author (Oracle)
    REFERENCED it (distinct from `user-saved` curation / `oracle-footprint` authored-by-the-Oracle /
    `crawled` radar). It is NOT an authorship claim — `who_id` stays the paper's own author.
  • Dedup = policy B on `atom_id = paper:{canonical_id}`. Papers are IMMUTABLE, so presence →
    skip BEFORE the (paid) PDF fetch + embed. The same paper from two sources → ONE atom.
    `paper_from_url` is the single authority that stamps a
    DETERMINISTIC canonical id (arXiv id > DOI > S2 id) so every link-based source dedups the
    same way — even when the S2 enrichment call fails.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import unicodedata
from datetime import date, timedelta
from urllib.parse import urlparse

import requests

from . import derive, schema
from .embed import assert_model
from pipeline.ingestion.utils import log
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL, FETCH_ABSENT,
                            FETCH_OK, FETCH_UNDETERMINED, body_fields, promote_atom,
                            snapshot_and_hash, store_atom, submit_atom)

# Where `paper_from_url` records whether the S2 metadata fetch ANSWERED or was BLOCKED. Carried on
# the Paper dict rather than returned alongside it, because the Paper is what crosses every seam
# between the two (prefetch pools, caller threading, `atomize_paper`'s public signature).
_S2_VERDICT = "_s2_verdict"

# ── runaway / quality guards ─────────────────────────────────────────────────────
_PAPER_MAX_CHARS = 500_000        # cap extracted text (mirror x_render._ARTICLE_MAX_CHARS):
                                  # real papers are never this long — this only clips a malicious
                                  # payload so a pathological PDF can't blow the embed bill.
_MIN_FULLTEXT_CHARS = 500         # below this there is not enough text to be a BODY at all
                                  # (a cover sheet, a stub) → abstract-only.
_FLOOR_PER_PAGE = 200             # …and below this PER PAGE we got only PART of the document, which
                                  # a total-chars floor cannot see: a 20-page scan whose only text
                                  # layer is a per-page copyright watermark clears 500 while missing
                                  # 99% of the paper. Measured over 505 live PDFs (2026-08-27), the
                                  # distribution is BIMODAL with an empty band between the modes —
                                  # 15 image-only scans at 29–31 chars/page, then NOTHING until
                                  # 787.7, then 490 genuine documents up to 8,089. 200 sits in that
                                  # band: 6.4× above the highest scan, 3.9× below the lowest genuine
                                  # document. The exact value inside the band is not load-bearing —
                                  # anything from 100 to 500 classifies all 505 identically.
_PDF_MAX_BYTES = 60 * 1024 * 1024  # download ceiling on the WIRE (before we ever parse).
_PDF_TIMEOUT = 30

# ── Semantic Scholar single-paper enrichment ─────────────────────────────────────
_S2_BASE = "https://api.semanticscholar.org/graph/v1"
# The Paper SHAPE — reused from the vault ingester's FIELDS so both sides agree on the metadata.
_S2_FIELDS = ("title,abstract,year,url,externalIds,citationCount,"
              "publicationDate,authors,openAccessPdf,venue")
# Headers for PDF downloads — a PDF lives on arXiv, an OA mirror, or some blog's own host, so
# these requests go to hosts that have NOTHING to do with Semantic Scholar. Kept deliberately
# separate from `_s2_headers()`: a credential must never be sent to a host that did not issue it,
# and one shared header dict is exactly how that leak happens.
_PDF_UA = {"User-Agent": "opyt-paper-adapter/1.0"}

# S2 RETRY. Unauthenticated S2 is one shared pool, so a 429 says the POOL is busy, not that this
# caller is over a quota — and the contention is time-varying, measured 2026-09-09 on the same
# endpoint minutes apart: a 20-request burst took 16 429s, and a later 12-paper batch took none.
# One request was therefore a coin flip whose outcome is PERMANENT, because a paper that lands
# without a title can never be repaired (policy-B dedup, `_thin_metadata_warning`).
#
# Retrying is free in the quiet window — no 429, no sleep, not one extra request — and in the busy
# window it is the difference between a good atom and a permanently anonymous one. The same 30
# papers that a single shot lost answered 30/30 when asked again at this interval.
#
# A FIXED interval because S2 gives nothing else to pace on: its 429 carries no `Retry-After`,
# only `x-amzn-ErrorType: TooManyRequestsException` (measured, same date).
_S2_RETRIES = 5
_S2_RETRY_SLEEP = 1.5


def _s2_headers() -> dict:
    """Semantic Scholar request headers, incl. the API key when the user set one.

    A callable, not a module constant: onboarding can write the key to ~/.opyt/.env mid-session,
    and a constant evaluated at import would pin the un-keyed headers for the life of the process.
    """
    from pipeline.credentials import s2_headers
    return s2_headers()


_ARXIV_VER = re.compile(r"v\d+$")


def _strip_arxiv_version(arxiv_id: str) -> str:
    """`2401.00001v2` → `2401.00001`, `hep-th/9901001v1` → `hep-th/9901001`. Version-stripping is
    deliberate: two versions of one arXiv paper are the SAME work, so they dedup to one atom
    (policy B). The original (versioned) url is still preserved in the snapshot's Links section."""
    return _ARXIV_VER.sub("", (arxiv_id or "").strip())


# ══════════════════════════════════════════════════════════════════════════════════
# 1. paper_from_url — link → Paper (the shared helper the 3 link-based sources use)
# ══════════════════════════════════════════════════════════════════════════════════

# arXiv: /abs/{id}, /pdf/{id}[.pdf] or /html/{id}; id may contain a slash (old-style
# hep-th/9901001). `html` is arXiv's rendered view, which it now serves by DEFAULT for recent
# papers — so the url a reader copies out of their browser is increasingly this one, and without
# it here `arxiv.org/html/2402.17764v1` routed to the paper adapter (the host matches) and then
# parsed to None, which is the advertised-but-refuses shape PubMed was in until 2026-09-09.
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(.+?)(?:\.pdf)?(?:[?#].*)?$", re.I)
# The same arXiv id on a READER front-end. Both mirrors put the bare id in the path and neither
# hosts anything else paper-shaped there, so this is the id wearing a different hostname — not a
# second paper. Restricted to the modern `YYMM.NNNNN` form on purpose: these hosts are not arXiv,
# so the loose `.+?` above (which exists for arXiv's own legacy `hep-th/9901001` ids) would let an
# unrelated path become a paper id. Old-style ids predate both sites.
_ARXIV_MIRROR_RE = re.compile(
    r"(?:huggingface\.co/papers|(?:www\.)?alphaxiv\.org/(?:abs|pdf|overview))/"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)
# A DOI in the url's PATH, on any host. Named `doi.org` only until 2026-09-09, which read as
# "a raw DOI path" in this comment and was not: every publisher that prints the DOI in its own url
# — ACS/JACS `/doi/10.1021/…`, Wiley `/doi/10.1002/…`, ACM `/doi/10.1145/…`, Springer
# `/article/10.1007/…`, APS `/abstract/10.1103/…` — parsed to None, so `mint_artifact` returned
# `failed` for a paper whose identity was sitting in the string.
#
# The leading `/` is the whole guard against a false positive, and it also keeps the doi.org form
# matching (`https://doi.org/10.1021/x`), so this one pattern replaced two ideas rather than
# joining them. `link_router._DOI_IN_PATH_RE` is the same shape doing the ROUTING half; both are
# needed, because the parser never runs on a url the router sent to `article`.
_DOI_RE = re.compile(r"/(10\.\d{4,9}/[^\s?#]+)")
# What the pattern above over-captures. A DOI may legitimately contain slashes
# (`10.1088/1748-9326/ab4553`), so it cannot stop at the first one — which means it also swallows
# whatever VIEW segment the publisher appended. Measured over 250 pasted urls (2026-09-11): 5 of
# 85 parsed DOIs came out wrong, and every wrong one resolved to NOTHING in OpenAlex while its
# trimmed form resolved to the real paper:
#
#   frontiersin.org/articles/10.3389/fpsyg.2013.00863/full  -> …00863/full   -> not found
#   degruyter.com/document/doi/10.1515/9783110769043-005/html               -> not found
#   biorxiv.org/content/10.1101/2020.03.22.002386v1         -> …002386v1     -> not found
#
# That last one is the serious case: bioRxiv and medRxiv are first-class paper hosts and that IS
# their canonical url, so the store was minting `paper:DOI:…002386v1` — a wrong id, and immutable,
# so it would never dedup against the same preprint pasted as a clean DOI.
_DOI_VIEW_TAIL_RE = re.compile(
    r"/(?:full|fulltext|full-text|html|pdf|epdf|abstract|meta|references|citations|figures|"
    r"supplemental|supplementary|summary)$", re.I)
# bioRxiv/medRxiv put the VERSION in the url and not in the DOI — exactly the arXiv situation
# `_strip_arxiv_version` already handles, one prefix over. `.full` and `.full.pdf` ride along.
_BIORXIV_VER_RE = re.compile(r"v\d+(?:\.full(?:-text)?)?(?:\.pdf)?$", re.I)


def _clean_doi(raw: str) -> str:
    """A DOI captured out of a url path → the DOI, with the url's own decoration removed.

    Trimmed repeatedly because publishers stack the suffixes (`…v1.full.pdf`). Conservative by
    construction: only a KNOWN view word is ever removed, so a DOI that genuinely ends in an
    unusual segment is left alone.
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
# SSRN mints every paper's DOI from its own abstract id: `abstract_id=3482150` is
# `10.2139/ssrn.3482150`. That makes this the one blocked publisher whose identity needs no
# request — which matters because ssrn.com answers 403 to a server fetch (measured 2026-09-11,
# identical for a browser User-Agent), so the `citation_doi` probe can never reach it.
#
# DERIVED, not declared, which is the line this module otherwise holds — so it was checked rather
# than assumed (2026-09-11): 10 of 10 SSRN works sampled from OpenAlex carry a `10.2139/ssrn.*`
# DOI, and 3 of 3 ids round-tripped through Crossref to a real SSRN paper (3482150 → "The Impact
# of Artificial Intelligence on the Labor Market"). It is a minting SCHEME, not a guess at a
# match, which is the same footing `_ZENODO_DOI_RE` already stands on in reverse.
_SSRN_RE = re.compile(r"ssrn\.com/\S*?abstract(?:_id)?=(\d+)", re.I)
# Semantic Scholar paper page: /paper/[slug/]{40-hex-hash | corpus-id-digits}.
_S2_RE = re.compile(
    r"semanticscholar\.org/paper/(?:[^/]+/)?(CorpusID:\d+|[0-9a-f]{40}|\d+)", re.I)
# PubMed: /{PMID}. The id is NOT a DOI and carries no route to one inside the string, so unlike
# every other branch this needs a lookup — see `_pubmed_doi` for where it happens and why not here.
_PUBMED_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{4,9})", re.I)
# PubMed Central, on both the legacy `www.ncbi.nlm.nih.gov/pmc/articles/…` path and the
# `pmc.ncbi.nlm.nih.gov/articles/…` host NCBI moved it to. Same E-utilities lookup as a PMID with
# `db=pmc`, and the same reason it cannot be a string rule: the PMCID carries no route to a DOI.
#
# Worth its own branch because PMC is the largest open-access full-text archive in biomedicine AND
# it answers 403 to a server fetch (measured 2026-09-11, browser User-Agent included), so the
# `citation_doi` probe cannot rescue it the way it rescues nature.com. Before this, every PMC url
# was refused outright.
_PMC_RE = re.compile(r"ncbi\.nlm\.nih\.gov/(?:pmc/)?articles/(?:PMC)?(\d+)", re.I)
# A Zenodo RECORD page. Zenodo's own api is asked for the DOI rather than deriving it from the
# record number, and the difference is not cosmetic: `zenodo.org/records/20027463` is a CONCEPT id
# ("all versions"), whose current version DOI is `10.5281/zenodo.20027464` — off by one — and
# `records/3509134` (pandas) reports `10.5281/zenodo.21500199`, nowhere near it. Deriving
# `10.5281/zenodo.{n}` would therefore mint a different atom than the same deposit's own DOI url
# does, and papers are immutable, so that split would be permanent. One request buys the id the
# record DECLARES.
# Europe PMC is a second front door onto the same two ids: `/article/MED/{pmid}` and
# `/pmc/articles/PMC{pmcid}`. No new lookup — it reuses the E-utilities calls above.
# Europe PMC addresses an article as `/{article|abstract}/{SOURCE}/{id}`, and the SOURCE segment
# decides who can resolve it. MED is a PMID and PMC is a PMCID — both answerable by NCBI's
# E-utilities above. PPR is a PREPRINT, which NCBI has never heard of; only Europe PMC's own index
# knows it, hence the separate resolver.
#
# The PMC pattern wanted `articles?/PMC` until 2026-09-16 and so matched only the legacy
# `/articles/PMC123` form. Europe PMC's actual url is `/article/PMC/PMC7096066` — source segment,
# THEN the id — which the old shape could not match at any position, so their own PMC view fell
# through to the blog ingester (measured 2026-09-16, and invisible because the host answers 403 to
# the `citation_doi` probe, so nothing ever contradicted the mis-route).
_EUROPEPMC_PMID_RE = re.compile(r"europepmc\.org/(?:article|abstract)/MED/(\d{4,9})", re.I)
_EUROPEPMC_PMC_RE = re.compile(
    r"europepmc\.org/(?:article|abstract)/PMC/PMC(\d+)|europepmc\.org/articles?/PMC(\d+)", re.I)
_EUROPEPMC_PPR_RE = re.compile(r"europepmc\.org/(?:article|abstract)/PPR/(PPR\d+)", re.I)
_EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# An OpenAlex work page. The id names the work outright, so one call gives its DOI.
_OPENALEX_WORK_RE = re.compile(r"openalex\.org/(W\d+)", re.I)
_ZENODO_RECORD_RE = re.compile(r"zenodo\.org/records?/(\d+)", re.I)
_ZENODO_API = "https://zenodo.org/api/records"

# NCBI E-utilities. Keyless, and the free tier is documented at 3 requests/second — well past
# anything one deposit at a time approaches.
_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
# OpenReview: /forum?id=X or /pdf?id=X.
_OPENREVIEW_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([^\s&#]+)", re.I)


def _parse_paper_url(url: str, *, content_type: str | None = None) -> dict | None:
    """A URL → `{paperId, externalIds, url, openAccessPdf?, s2_lookup}`, or None if it is not a
    paper link. `paperId` is the DETERMINISTIC canonical dedup key; `s2_lookup` is the id form the
    Semantic Scholar single-paper API accepts (None → skip enrichment).

    `content_type` is an OPTIONAL, caller-supplied fact — the response's actual `Content-Type`
    header, when a caller already fetched the url (`link_router.classify_link_deep`) — for the one
    case the url's SHAPE alone can't answer: a PDF served with no `.pdf` in its path (a
    `/download?id=123`-style redirect). Trusted only because it required a real fetch to obtain; a
    bare url string can never assert it about itself."""
    u = (url or "").strip()
    if not u:
        return None

    m = _ARXIV_RE.search(u)
    if m:
        aid = _strip_arxiv_version(m.group(1))
        return {"paperId": f"arXiv:{aid}", "externalIds": {"ArXiv": aid}, "url": u,
                "s2_lookup": f"arXiv:{aid}"}

    m = _ARXIV_MIRROR_RE.search(u)
    if m:
        # The reader front-ends. Deliberately identical to the branch above — the whole point is
        # that a Hugging Face papers link and an arxiv.org link are ONE atom — and the user's own
        # url is still what gets stored, as every other branch keeps theirs.
        aid = _strip_arxiv_version(m.group(1))
        return {"paperId": f"arXiv:{aid}", "externalIds": {"ArXiv": aid}, "url": u,
                "s2_lookup": f"arXiv:{aid}"}

    m = _DOI_RE.search(u)
    if m:
        raw = _clean_doi(m.group(1))
        # An arXiv DOI resolves to the same preprint as its /abs/ page, so keying it as a DOI mints
        # a SECOND atom for a paper we already have. Collapsed HERE and not in any one adapter,
        # because this function's contract is that every link form of one paper dedups to one atom
        # — and OpenAlex, Semantic Scholar and Crossref all hand back this form, so a fix inside
        # one of them reintroduces the split for the next. Measured 2026-08-26: the live store held
        # 30 paper atoms, all `paper:arXiv:*` and none DOI-keyed, so no existing atom changes
        # identity under this (papers are immutable under Policy B, so that had to be checked).
        am = _ARXIV_DOI_RE.match(raw)
        if am:
            aid = _strip_arxiv_version(am.group(1))
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

    # A raw hosted PDF (the "arcium view-PDF" blog case): no external id, keyed on the url. The
    # url IS the fulltext source, so metadata stays thin but the full body is still chunk-embedded.
    p = urlparse(u)
    if p.scheme in ("http", "https") and p.path.lower().endswith(".pdf"):
        canon = f"{(p.netloc or '').lower()}{p.path}"
        return {"paperId": f"url:{canon}", "externalIds": {}, "url": u,
                "openAccessPdf": {"url": u}, "s2_lookup": None}

    # Same raw-PDF case with no `.pdf` in the path — the caller already fetched it and knows the
    # response IS a pdf. Keyed on the FULL url (query string included), NOT path-only like the
    # branch above: don't collapse this the same way, it would dedupe two different papers.
    if (p.scheme in ("http", "https") and content_type
            and "application/pdf" in content_type.lower()):
        return {"paperId": f"url:{u}", "externalIds": {}, "url": u,
                "openAccessPdf": {"url": u}, "s2_lookup": None}

    return None


def _eutils_doi(db: str, uid: str) -> str | None:
    """A PubMed or PMC id → the paper's DOI, or None. ONE keyless request to NCBI E-utilities.

    `db` is `"pubmed"` (a PMID) or `"pmc"` (a PMCID). One function rather than two because the
    request, the response shape and every failure mode are identical — only the database name
    differs, and NCBI's `articleids` list answers both the same way.

    NOT in `_parse_paper_url`, and the boundary is load-bearing rather than tidy. That function is
    pure string work, which is the whole contract `link_router.predicted_atom_id` rests on — "the
    atom_id derivable from the url ALONE — no network, no DB" — and Hopper's free already-present
    check reads it once per candidate link. A network call inside it would turn scanning ten search
    results into ten round trips.

    So a PubMed url answers None there, honestly, the same way a Substack post does: the id is
    knowable, just not from the string. `paper_from_url` — which already goes to the network —
    resolves it and re-parses as the DOI, so PubMed mints the SAME atom as the DOI, the publisher
    url, or any other form of that paper. That shared identity is the point; a `pubmed:{pmid}` key
    would have split one immutable paper across two atoms.

    ⚠️ PubMed was advertised and broken for a MONTH, in two distinct ways, and the second one is
    why this warning is long. From 2026-08-13 `pubmed.ncbi.nlm.nih.gov` sat in
    `link_router._PAPER_HOSTS` with no branch here at all, so every PubMed url routed to the paper
    adapter and returned `failed` — silent, because a failed mint writes nothing. Adding this
    function on 2026-09-09 did NOT fix it: `mint_artifact` bailed on a paper with no url-derivable
    id before the adapter ran, so this code was unreachable from Hopper until 2026-09-16. A
    resolver is only half a fix; the caller has to be able to reach it.

    PMC spent longer in a worse version of that state: refused outright, and unreachable by the
    `citation_doi` probe too, because ncbi answers 403 to a server fetch (measured 2026-09-11).

    Fail-safe: any failure returns None, which lands exactly on the old behaviour (unparseable →
    `failed`) rather than on a guess.
    """
    from pipeline.ingestion.utils import log
    try:
        resp = requests.get(_EUTILS, timeout=_PDF_TIMEOUT, headers=_PDF_UA,
                            params={"db": db, "id": uid, "retmode": "json"})
        if resp.status_code != 200:
            return None
        doc = ((resp.json().get("result") or {}).get(uid)) or {}
        for aid in (doc.get("articleids") or []):
            if (aid.get("idtype") or "").lower() == "doi" and aid.get("value"):
                return str(aid["value"]).strip()
    except Exception as e:
        log(f"[papers] {db} id {uid} DOI lookup failed: {type(e).__name__}: {e}")
    return None


def _zenodo_record_doi(record_id: str) -> str | None:
    """A Zenodo record number → the DOI that record DECLARES, or None. ONE keyless request.

    Asked rather than derived. `10.5281/zenodo.{record_id}` looks like a free answer and is not
    one: a record number can be a CONCEPT id standing for every version of a deposit, whose
    current version carries a different DOI (measured 2026-09-11 — `records/20027463` declares
    `…20027464`, and `records/3509134` declares `…21500199`). Minting from the number would give
    that deposit a second, permanent atom id, so the number is only ever used to ASK.

    Fail-safe: any failure returns None and the url stays unparsed, exactly as before.
    """
    from pipeline.ingestion.utils import log
    try:
        resp = requests.get(f"{_ZENODO_API}/{record_id}", timeout=_PDF_TIMEOUT, headers=_PDF_UA)
        if resp.status_code != 200:
            return None
        rec = resp.json()
        doi = (rec.get("doi") or (rec.get("metadata") or {}).get("doi") or "").strip()
        return doi or None
    except Exception as e:
        log(f"[papers] zenodo record {record_id} DOI lookup failed: {type(e).__name__}: {e}")
    return None


# One paced door onto OpenAlex for the whole paper path.
#
# `min_interval_s` is DECLARED on the adapter and APPLIED BY THE CALLER — `frontier_execute` and
# `ingest_scholar_footprint` each do it themselves — and the paper path was never one of those
# callers. It built a fresh adapter per call and fired immediately, which was survivable while the
# only read here was a rare missing-title fallback.
#
# It stopped being survivable on 2026-09-11, when two more reads landed on this path. A 293-url
# run took `api.openalex.org` to HTTP 429; the PERSISTED breaker opened; and every OpenAlex read
# then returned None for the next 15 minutes — including `_openalex_metadata`, which had been
# working. Stored-atom rate fell from 56% to 47% and NOTHING in the logs said why: a breaker-open
# read is indistinguishable from "OpenAlex has never heard of this paper".
#
# So the interval is enforced here, once, for every read on this path. It serializes concurrent
# callers (the X prefetch pool), which is the intent — being slower than the rate limit is the
# only way to keep the shared breaker closed for everyone.
_OPENALEX_GATE = threading.Lock()
_OPENALEX_NEXT_AT = 0.0


def _openalex_read(fn):
    """Run one OpenAlex read on the paper path: breaker-checked, paced, and fail-safe.

    `fn` takes the adapter and returns whatever it reads. Any failure — breaker open, rate limit,
    unparseable JSON — is None, because every caller here runs inside `paper_from_url` or
    `classify_link_deep`, neither of which may raise.
    """
    global _OPENALEX_NEXT_AT
    from .frontier_sources import OpenAlexWorksAdapter
    adapter = OpenAlexWorksAdapter()
    # BEFORE the delay, the way `frontier_execute` does it: a host already known to be down must
    # not also cost a second per caller to rediscover that.
    if not adapter.available():
        return None
    interval = float(getattr(adapter, "min_interval_s", 0.0) or 0.0)
    with _OPENALEX_GATE:
        wait = _OPENALEX_NEXT_AT - time.monotonic()
        if wait > 0:
            time.sleep(min(wait, interval))
        _OPENALEX_NEXT_AT = time.monotonic() + interval
    try:
        return fn(adapter)
    except Exception as e:
        # LOGGED, not swallowed. Every failure here returns None, and None is also what "OpenAlex
        # has never heard of this paper" looks like — so without a line in the log, a spent daily
        # budget is indistinguishable from a genuinely unknown DOI, and the store just quietly
        # gets thinner. That is exactly the shape CLAUDE.md's fail-safe rule exists to prevent, and
        # it is how the 2026-09-11 budget exhaustion stayed invisible until the breaker row was
        # read by hand.
        log(f"[papers] openalex read failed: {type(e).__name__}: {e}")
        return None


def _europepmc_ppr_doi(ppr_id: str) -> str | None:
    """A Europe PMC PREPRINT id (`PPR217527`) → the DOI it declares, or None. ONE keyless request.

    A third resolver rather than a third branch of `_eutils_doi`, because NCBI is the wrong index
    to ask: E-utilities covers PubMed and PMC, and a preprint is in neither. Europe PMC indexes
    them itself and hands back the DOI the preprint server minted (measured 2026-09-16,
    `PPR217527` → `10.21203/rs.3.rs-76053/v1`, a Research Square deposit), so the atom keys on the
    same DOI a reader who followed the link would land on.

    `SRC:PPR` is part of the query on purpose. Europe PMC ids are only unique WITHIN a source, and
    an unqualified `EXT_ID` search is a different question that can answer about another database's
    record with the same number.

    Free and unmetered, unlike the OpenAlex path — worth knowing when the daily budget is the
    binding constraint (`docs/plans/2026-09-12-paper-url-coverage-and-the-openalex-budget.md`).

    Fail-safe: any failure returns None and the url stays unparsed.
    """
    from pipeline.ingestion.utils import log
    try:
        resp = requests.get(_EUROPEPMC_SEARCH, timeout=_PDF_TIMEOUT, headers=_PDF_UA,
                            params={"query": f"EXT_ID:{ppr_id} AND SRC:PPR",
                                    "resultType": "core", "format": "json", "pageSize": 1})
        if resp.status_code != 200:
            return None
        hits = ((resp.json().get("resultList") or {}).get("result")) or []
        return (str(hits[0].get("doi") or "").strip() or None) if hits else None
    except Exception as e:
        log(f"[papers] europepmc preprint {ppr_id} DOI lookup failed: {type(e).__name__}: {e}")
    return None


def _openalex_doi_by_url(url: str) -> str | None:
    """A url no pattern here can read → the DOI of the paper OpenAlex says lives at that url.

    THE LAST RESORT, and the one that covers the largest failure by far. Measured over 250 pasted
    urls (2026-09-11): 96 were refused outright, and this recovers 63 of them — institutional
    repositories (`repositorio.unal.edu.co/handle/unal/81443` is ResNet) and aggregators. Those
    pages are not obscure papers; they are ordinary papers wearing a url we cannot read, and most
    of them ALSO answer 403 to a fetch, so the `citation_doi` probe cannot reach them either.

    What it does NOT cover is the publisher that names an article some private way. This docstring
    listed "Elsevier's PII, MDPI's issue path" until 2026-09-16 and neither was ever true of this
    function — asked directly, `sciencedirect.com/science/article/pii/S0092867420302294`,
    `mdpi.com/2072-6643/13/6/1815` and `dspace.mit.edu/handle/1721.1/7582` all answered None while
    a control url resolved in the same session.

    Still a DECLARED identifier, which is the line this module holds. `locations.landing_page_url`
    is an exact match against a string OpenAlex stores — OpenAlex asserting "this page is a copy of
    that work" — not a bibliographic guess. The contrast is the measurement that set this rule:
    Crossref TITLE search resolved 1 of 3 known papers on 2026-09-09 and returned a different paper
    twice, and a wrong paper atom is immutable.

    REFUSES AMBIGUITY rather than taking the top hit — the same rule `oracles._openalex_root`
    applies to venue names. More than one work claiming a page means we cannot say which paper the
    user meant, and guessing writes a permanent wrong answer.

    Fail-safe: any failure, any ambiguity, anything without a DOI → None, and the url stays exactly
    as unparseable as it was.
    """
    if not (url or "").startswith(("http://", "https://")):
        return None
    works = _openalex_read(lambda a: a.works_by_landing_page(url)) or []
    dois = {d for w in works if (d := (w.get("doi") or "").replace("https://doi.org/", "").lower())}
    if len(dois) != 1:
        return None                      # nothing, or an ambiguity we refuse to resolve by guess
    return dois.pop()


def _looked_up_doi(url: str) -> str | None:
    """A url whose identifier needs a REQUEST to become a DOI → that DOI, or None.

    The three ids that name a paper without carrying a route to one inside the string. Collected
    here so `paper_from_url` keeps ONE network-resolution branch rather than three copies of the
    same re-parse, and so `_parse_paper_url` stays pure string work — the contract
    `link_router.predicted_atom_id` rests on.
    """
    if m := _PUBMED_RE.search(url):
        return _eutils_doi("pubmed", m.group(1))
    if m := _PMC_RE.search(url):
        return _eutils_doi("pmc", m.group(1))
    if m := _EUROPEPMC_PMC_RE.search(url):
        return _eutils_doi("pmc", m.group(1) or m.group(2))
    if m := _EUROPEPMC_PMID_RE.search(url):
        return _eutils_doi("pubmed", m.group(1))
    if m := _EUROPEPMC_PPR_RE.search(url):
        return _europepmc_ppr_doi(m.group(1))
    if m := _ZENODO_RECORD_RE.search(url):
        return _zenodo_record_doi(m.group(1))
    if m := _OPENALEX_WORK_RE.search(url):
        return _openalex_work_doi(m.group(1))
    return None


def _openalex_work_doi(work_id: str) -> str | None:
    """An OpenAlex `W…` id → that work's DOI, or None. Fail-safe, like every lookup here."""
    w = _openalex_read(lambda a: a.work_by_openalex_id(work_id))
    doi = ((w or {}).get("doi") or "").replace("https://doi.org/", "").strip().lower()
    return doi or None


def _fetch_s2_paper(lookup_id: str | None) -> tuple[dict | None, str]:
    """Best-effort Semantic Scholar single-paper fetch → `(data | None, verdict)`.

    Returns a VERDICT, not just None: collapsing "S2 rate-limited us" into a bare None let a 429
    be recorded downstream as "this paper has no abstract" — unauthenticated S2 allows ~1 req/s, so
    bursts 429 routinely.

      FETCH_OK           — S2 answered. Whatever it did or did not include is the truth.
      FETCH_ABSENT       — no lookup id, or a 404. A real answer; retrying changes nothing.
      FETCH_UNDETERMINED — 429 / any other status / transport failure, after `_S2_RETRIES`
                           attempts. We were STOPPED.

    RETRIED, and only the undetermined case is. A 404 is an ANSWER — S2 has no record of this
    paper, and asking again returns the same 404 while spending a request from the pool everyone
    shares. Retrying it would make the busy window worse for the calls that can actually be
    rescued. See `_S2_RETRIES` for the measurement the interval comes from.

    Still never raises: the enrichment itself stays a bonus (fail-safe).
    """
    if not lookup_id:
        return None, FETCH_ABSENT
    for attempt in range(_S2_RETRIES):
        try:
            resp = requests.get(f"{_S2_BASE}/paper/{lookup_id}", params={"fields": _S2_FIELDS},
                                timeout=_PDF_TIMEOUT, headers=_s2_headers())
            if resp.status_code == 404:
                return None, FETCH_ABSENT       # S2 genuinely has no record of this paper
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data, FETCH_OK
        except Exception:
            pass                                # transport failure — indistinguishable from a block
        # Sleep only BETWEEN attempts, never after the last one: a caller that is going to be told
        # UNDETERMINED anyway must not also be made to wait for the privilege.
        if attempt + 1 < _S2_RETRIES:
            time.sleep(_S2_RETRY_SLEEP)
    return None, FETCH_UNDETERMINED             # 429 above all — we were throttled, not answered


def _merge_paper(minimal: dict, rich: dict) -> dict:
    """S2's rich fields, but keep OUR deterministic canonical `paperId` (so dedup stays stable
    across sources) and UNION the external ids + preserve a parsed openAccessPdf S2 didn't have."""
    out = dict(rich)
    out["paperId"] = minimal["paperId"]                      # canonical dedup key wins
    ext = dict(rich.get("externalIds") or {})
    ext.update({k: v for k, v in (minimal.get("externalIds") or {}).items() if v})
    out["externalIds"] = ext
    # S2 wins where it HAS an answer; a null from S2 must never ERASE metadata the caller already
    # held. S2 routinely returns `abstract: null`, and `known=` callers (see `paper_from_url`)
    # arrive with a real title and abstract from their own source — clobbering those with S2's
    # nulls would write the contentless atom this whole seam exists to prevent.
    for k, v in minimal.items():
        if v and not out.get(k):
            out[k] = v
    if not (out.get("openAccessPdf") or {}).get("url") and minimal.get("openAccessPdf"):
        out["openAccessPdf"] = minimal["openAccessPdf"]
    return out


def _needs_body(paper: dict) -> bool:
    """Would `atomize_paper` refuse this paper for having nothing to read?

    The metadata resolvers' gate. Deliberately the SAME question the atomizer asks — "is there a
    body?" — rather than "is there a title?", because a title with no abstract is dropped just as
    completely as a blank paper and reads, from the gate's side, like a paper that needs nothing.
    Full text is not consulted: no PDF has been fetched this early, and an abstract is what makes
    the difference between an atom and a skip when none resolves.
    """
    return not (paper.get("title") or "").strip() or not (paper.get("abstract") or "").strip()


def _fill_gaps(paper: dict, extra: dict | None) -> None:
    """Merge a resolver's fields into the paper, filling ONLY what the paper lacks. Mutates.

    Not `paper.update(extra)`, which the missing-title gate could afford and this one cannot: a
    paper reaching here now usually HAS a title and authors from S2, and an overwrite would swap
    S2's authors — carrying `authorId`, which is what mints `who_id = scholar:{id}` — for
    OpenAlex's, which carry `openalexId` and cannot. The fix would be silent and would only show
    up as papers quietly losing their author identity.
    """
    for k, v in (extra or {}).items():
        if not paper.get(k):
            paper[k] = v


def _openalex_metadata(paper: dict) -> dict | None:
    """A DOI → OpenAlex's record of that work, in the S2 field names, or None.

    WHY, MEASURED 2026-09-11. Paste a days-old ACS or RSC article and S2 has not indexed it yet:
    the DOI parses perfectly, every other field comes back empty, and `atomize_paper`'s no-body
    skip correctly refuses to freeze a contentless atom — so NOTHING is stored. Of 7 such DOIs S2
    could not resolve, OpenAlex had title and abstract for 7. The gap is indexing LAG plus whole
    classes of source S2 does not carry (it resolved 1 of 15 OpenAlex DOIs on 2026-08-26 — Zenodo,
    institutional repositories).

    Gated on a missing title by its caller, which is the whole cost story: over 12 DOIs S2 DID
    resolve, OpenAlex added a pdf S2 lacked in 0 of them. A paper S2 answers for gains nothing
    here, so the common path must never pay for the request.

    METADATA ONLY — never `paperId`, `externalIds` or `url`. OpenAlex will happily report that a
    JACS paper also exists on arXiv; writing that into `externalIds.ArXiv` would move the canonical
    id off the DOI the user actually pasted and mint a DIFFERENT atom, which Policy B then freezes.
    Identity is decided by `_parse_paper_url` and stays there.

    The transport half is `OpenAlexWorksAdapter.work_by_doi`, so this inherits the persisted
    breaker and the courtesy pacing that already guard `api.openalex.org`. `available()` is checked
    FIRST: a host already known to be down must not cost a request to rediscover that.

    Fail-safe: any failure returns None and `paper_from_url` returns exactly what it returns today.
    """
    doi = (paper.get("externalIds") or {}).get("DOI")
    if not doi:
        return None
    # Imported here, not at module scope: `frontier_sources` is the frontier's own module and
    # `ingest_papers` is imported by every paper caller, including offline ones.
    from .frontier_sources import _abstract_from_inverted, _openalex_authors
    work = _openalex_read(lambda a: a.work_by_doi(str(doi).strip()))
    if not isinstance(work, dict) or not work:
        return None
    loc = work.get("primary_location") or {}
    # `best_oa_location` first: the primary location can be the paywalled publisher copy, while the
    # OA one is often the authors' own preprint. That is what recovers part of the paywall case
    # without going near a paywall — and `_fulltext_pdf_urls` already reads this field, so the body
    # flows on to the on-open deepen with no edit of its own.
    pdf_url = (work.get("best_oa_location") or {}).get("pdf_url") or loc.get("pdf_url")
    out = {
        "title": " ".join((work.get("title") or "").split()),
        # `[:2000]` mirrors `ingest_scholar_footprint._paper_from_work`, so an abstract reads the
        # same however it entered the store.
        "abstract": _abstract_from_inverted(work.get("abstract_inverted_index"))[:2000],
        # `openalexId`, never `authorId` — `derive_paper` reads `authorId` to mint
        # `who_id = scholar:{id}`, and an OpenAlex author id is not a Semantic Scholar one.
        "authors": [{"name": a["name"],
                     **({"openalexId": oid} if (oid := a.get("openalex_id")) else {}),
                     **({"orcid": orc} if (orc := a.get("orcid")) else {}),
                     **({"position": pos} if (pos := a.get("position")) else {})}
                    for a in _openalex_authors(work)],
        "publicationDate": (work.get("publication_date") or "")[:10],
        "year": work.get("publication_year"),
        "venue": (loc.get("source") or {}).get("display_name") or "",
        "citationCount": work.get("cited_by_count") or 0,
    }
    if pdf_url:
        out["openAccessPdf"] = {"url": pdf_url}
    # Empties stripped, because the caller merges with `paper.update()`: a null abstract in the
    # record must not erase an abstract a `known=` caller already supplied.
    return {k: v for k, v in out.items() if v} or None


# Zenodo mints every deposit under this prefix; the digits are the id its api takes. Zenodo is
# reached because it is KEYLESS — no email, no token, no setup asked of the user, which is exactly
# what disqualified Unpaywall as a general answer. (`_ZENODO_API` itself lives up with the url
# patterns, since `_zenodo_record_doi` reaches it first.)
_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)$", re.I)


def _zenodo_metadata(paper: dict) -> dict | None:
    """A Zenodo deposit's own record → the S2-shaped fields, or None. ONE request, no key.

    WHY THIS EXISTS AT THE METADATA LAYER. Semantic Scholar resolved 1 of 15 OpenAlex DOIs on
    2026-08-26; the rest 404, and Zenodo is the largest single group of them. A 404 is
    `FETCH_ABSENT`, so nothing raises — the Paper simply comes back with no title, no abstract and
    no pdf url, and every caller that passes no `known=` then mints `# Untitled` with whatever body
    it can find. Under policy-B dedup that is permanent. Two of the three paper callers pass no
    `known=` (`ingest_x_footprint`, `link_router`), and the first of those mints
    `author_referenced` atoms — HUMAN-ATTESTED — so the bad row lands in the tier the KB is built
    on. This is what closes that.

    IT ALSO CARRIES THE BODY, which is why there is no separate url resolver. The pdf lives at a
    path containing the filename the DEPOSITOR chose, so it cannot be derived from the DOI:

        10.5281/zenodo.21921441
          → GET /api/records/21921441
          → files[].links.self = …/files/ClaimKeep%20Paper%20v0.11.pdf/content

    That looks like a reason to make `_fulltext_pdf_urls` impure, and it was built that way first
    (a last-resort `_discover_pdf_urls` beside it, deleted 2026-08-28). It is not. The answer
    arrives in the SAME record as the title, so handing it over as `openAccessPdf` — the field
    `_fulltext_pdf_urls` already reads — costs no extra request, keeps that function pure, and
    removes a whole concept. One request answers both questions because Zenodo is one source of
    truth about one deposit.

    Deliberately NOT a dispatch table for other sources. One source does not reveal the axis of
    variation; if OSF or a repository scraper ever lands, THEN generalize against two real cases.

    Fail-safe: any failure returns None and the caller keeps the metadata it already had."""
    m = _ZENODO_DOI_RE.match(((paper.get("externalIds") or {}).get("DOI") or "").strip())
    if not m:
        return None
    try:
        resp = requests.get(f"{_ZENODO_API}/{m.group(1)}", timeout=_PDF_TIMEOUT, headers=_PDF_UA)
        if resp.status_code != 200:
            return None
        rec = resp.json()
    except Exception:
        return None
    meta = rec.get("metadata") or {}
    out: dict = {}
    if title := (meta.get("title") or "").strip():
        out["title"] = title
    if names := [n for c in (meta.get("creators") or [])
                 if (n := (c.get("name") or "").strip())]:
        out["authors"] = [{"authorId": None, "name": n} for n in names]
    if date := (meta.get("publication_date") or "").strip():
        out["publicationDate"] = date
    if abstract := _html_to_text(meta.get("description") or ""):
        out["abstract"] = abstract
    # A record holds whatever the depositor uploaded — the paper beside its markdown source, a
    # dataset, a zip. Filtering to `.pdf` is also what makes a `type: software` deposit correctly
    # yield no body, so there is no `resource_type` branch to write.
    for f in rec.get("files") or []:
        link = (f.get("links") or {}).get("self")
        if link and str(f.get("key") or "").lower().endswith(".pdf"):
            out["openAccessPdf"] = {"url": link}
            break
    return out or None


def _html_to_text(html: str) -> str:
    """Zenodo ships its abstract as HTML. Mirrors `ingest_curation`'s converter settings so an
    abstract reads the same however it entered the store. Degrades to '' (fail-safe)."""
    if not (html or "").strip():
        return ""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links, h.ignore_images, h.body_width, h.unicode_snob = False, False, 0, True
        return h.handle(html).strip()
    except Exception:
        return ""


def paper_from_url(url: str, *, enrich: bool = True, content_type: str | None = None,
                   known: dict | None = None) -> dict | None:
    """Normalize an arXiv / DOI / .pdf / paper-page link → a Paper dict, or None if the URL is not
    a paper. The Paper carries a DETERMINISTIC canonical `paperId` (arXiv id > DOI > S2 id) so every
    link-based source dedups to ONE atom regardless of which URL form was shared.

    `enrich=True` best-effort fills title/authors/abstract via the Semantic Scholar single-paper
    API; any enrichment failure degrades to the minimal parsed Paper (fail-safe), never raises. A
    driver that already holds full metadata can pass `enrich=False` to stay purely offline.

    `content_type` — see `_parse_paper_url`, whose param this passes straight through: a caller
    that already fetched the url (`link_router.classify_link_deep`) can assert its real
    Content-Type, which recovers a raw PDF the url's own shape can't reveal.

    `known` — metadata the CALLER already holds, in the S2 field names (`title`, `abstract`,
    `authors`, `publicationDate`). S2 still runs and still wins wherever it answers; `known` is
    what survives when it does not. This is what makes a source S2 has never heard of usable:
    measured 2026-08-26, Semantic Scholar resolved 1 of 15 OpenAlex DOIs — the rest were 404s
    (Zenodo, institutional repositories) or 429s. `atomize_paper` skips a paper with no abstract
    and no full text, so WITHOUT `known` those 14 are not written at all; with it they carry the
    finder's own title and abstract and mint normally.

    Stamps `_s2_verdict` — whether the metadata fetch ANSWERED or was BLOCKED. Not read by
    `atomize_paper`, which asks only whether a body resolved; `frontier_admit` reads it to tell a
    throttled attempt (`blocked_metadata`, worth retrying) from a decided one. Stamped AFTER the
    merge: `_merge_paper` rebuilds the dict from S2's response and would drop a key set before
    it."""
    parsed = _parse_paper_url(url, content_type=content_type)
    if parsed is None and enrich:
        # The ids that need a lookup to become a paper id (PubMed, PMC, a Zenodo record page).
        # Re-parsed as the DOI so this mints the same atom every other form of the paper does; the
        # ORIGINAL url is kept, because that is what the user actually saved and every other branch
        # keeps theirs too.
        #
        # GATED ON `enrich`, which already means "may I use the network". `predicted_atom_id`
        # passes `enrich=False` to keep Hopper's already-present pre-check free across ten search
        # results at once, so without this gate that check would cost ten round trips. These urls
        # therefore predict None — knowable, just not from the string, exactly like a Substack post.
        if doi := _looked_up_doi(url or ""):
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
        rich, verdict = _fetch_s2_paper(parsed.get("s2_lookup"))
        if rich:
            paper = _merge_paper(paper, rich)
        # Gated on a MISSING BODY, which is exactly the state `atomize_paper` refuses to write —
        # so the request fires only when it is the difference between an atom and no atom at all.
        # A caller that supplied `known=` (frontier) or an S2 that answered fully pays nothing.
        #
        # The gate read `not title` until 2026-09-11, and the two are NOT the same question. S2
        # answers for plenty of older papers with a title and no abstract; `atomize_paper` then
        # drops them for having no body, and the resolver that could have supplied one was never
        # asked because a title was present. Measured over 129 pasted urls: 12 landed in exactly
        # that state and OpenAlex held an abstract for 9 of them — LeCun's gradient-based learning
        # paper, Tibshirani's lasso paper, the reproducibility-project paper. Cost of asking: 15%
        # of papers that get a title lack an abstract, so that is how often the extra call fires.
        if _needs_body(paper):
            _fill_gaps(paper, _openalex_metadata(paper))
        # Zenodo SECOND, and kept though OpenAlex resolved a Zenodo DOI in testing and probably
        # subsumes it: "probably" is not the standard this repo deletes a working path on. Measure
        # it, then delete.
        if _needs_body(paper):
            _fill_gaps(paper, _zenodo_metadata(paper))
    paper[_S2_VERDICT] = verdict
    return paper


def _canonical_paper_id(paper: dict) -> str | None:
    """The dedup key behind `atom_id = paper:{id}`. Prefer the explicit `paperId`/`id` a source
    supplied (paper_from_url stamps a deterministic one); else derive from external ids (arXiv id,
    version-stripped > DOI). None → the paper cannot be identified (caller returns None)."""
    pid = paper.get("paperId") or paper.get("id")
    if pid:
        return str(pid)
    ext = paper.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    if arxiv:
        return f"arXiv:{_strip_arxiv_version(arxiv)}"
    doi = ext.get("DOI")
    if doi:
        return f"DOI:{str(doi).rstrip('.').lower()}"
    return None


def paper_atom_id(paper: dict) -> str | None:
    """`paper:{canonical_id}` — the atom identity, or None if the paper can't be identified.
    Public so a driver can compute it BEFORE calling `atomize_paper` (e.g. to check presence)."""
    pid = _canonical_paper_id(paper)
    return f"paper:{pid}" if pid else None


# ══════════════════════════════════════════════════════════════════════════════════
# 2. resolve_fulltext — the NEW capability: PDF → full document text
# ══════════════════════════════════════════════════════════════════════════════════

def _fulltext_pdf_urls(paper: dict) -> list[str]:
    """The ordered list of open PDF urls to try, most-reliable first: arXiv's fully-open mirror,
    then `openAccessPdf`. (Future OA mirrors — Unpaywall by DOI, PubMed Central, bioRxiv/medRxiv,
    CORE — slot in HERE behind the same function, no caller change.) Paywalled publisher PDFs are
    deliberately absent — we skip them and rely on the open mirror.

    `openAccessPdf` is whatever the Paper carries, which is S2's when S2 answered and the FINDER's
    when it did not (`paper_from_url(known=…)`). That is the seam OpenAlex uses: S2 does not index
    Zenodo or most institutional repositories, so for those works the finder's url is the only
    route to a body and this function needs no branch to use it."""
    urls: list[str] = []
    ext = paper.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    if arxiv:
        urls.append(f"https://arxiv.org/pdf/{_strip_arxiv_version(arxiv)}")
    oa = (paper.get("openAccessPdf") or {}).get("url")
    if oa:
        urls.append(oa)
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _download_pdf(url: str) -> bytes | None:
    """Stream a PDF with a byte ceiling. None on any failure or a runaway size (fail-safe)."""
    try:
        with requests.get(url, timeout=_PDF_TIMEOUT, stream=True, headers=_PDF_UA) as resp:
            resp.raise_for_status()
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    buf.extend(chunk)
                    if len(buf) > _PDF_MAX_BYTES:   # runaway on the wire → abandon
                        return None
            return bytes(buf)
    except Exception:
        return None


def _extract_pypdf(data: bytes) -> tuple[str, int]:
    """PDF bytes → (raw text, page count). The ONLY extractor. Lazy import so a missing dep
    degrades to abstract-only rather than raising.

    pdfplumber used to sit behind this as a "layout-aware" fallback and was deleted 2026-08-27: it
    measured strictly WORSE on every document tried (−1.3% to −8.9% chars, and it dropped the
    References section on two), and it only ever ran on PDFs pypdf had already found thin — the
    scanned case, where it yields nothing either. It had never once been the extractor that saved
    a document, on any machine.

    The page count is RETURNED rather than discarded because the completeness floor below is
    per-page: total chars alone cannot tell a 20-page scan from a 1-page note."""
    import io
    from pypdf import PdfReader
    pages = PdfReader(io.BytesIO(data)).pages
    return "\n\n".join((page.extract_text() or "") for page in pages), len(pages)


def _repair_extraction(text: str) -> str:
    r"""Repair the two word-level defects pypdf leaves behind. Both make REAL words unreachable by
    search, so this runs before the text is ever chunked or indexed.

    NFKC folds the typographic ligatures a PDF font encodes as single codepoints — `ﬁ` `ﬂ` `ﬀ`
    `ﬃ` — into their letters, so "identiﬁed" becomes findable by a search for "identified".
    Measured over 505 live PDFs: 12% of documents carry them, a median of 67 each. Its OTHER
    rewrites were measured too and are all search-positive in the same way: math-italic `𝑆` → `S`,
    MICRO SIGN → `μ`, NO-BREAK SPACE → space.

    The regex rejoins a word split across a line break — `seg-\nmentation` → `segmentation`. 65%
    of documents carry these, a median of 39 each. Two deliberate narrownesses:

      • LOWERCASE to LOWERCASE only. Widening it to `\w` catches 8% more line-break hyphens, and
        every one of those is a hyphen that BELONGS: `GPT-\n4`, `ERC-\n8004`, `COVID-\n19`.
      • The NAIVE join, not a prefix guard. A `non`/`self`/`pre`/`multi` list preserves 26 real
        compounds per 388 breaks but then wrongly preserves `multi-ple` and `pre-dicted` — it
        trades one error for another and adds a word list to maintain. The naive join's own error
        is milder: a wrong join (`nontumour`) still tokenizes as a plausible word, while an
        unrepaired break tokenizes as two non-words."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)


# Values a PDF's /Title carries when nobody set one: the authoring tool's default, the source
# filename, a typesetter's proof stamp, or a bare accession id. Measured against 120 PDFs whose
# true title was known — the gate costs 2 of 78 present values and is what lifts precision to 91%.
_PDF_TITLE_JUNK = re.compile(r"""^(
      untitled | microsoft\s+word | powerpoint | document\d* | manuscript | main | print
    | layout\s*\d* | slide\s*\d* | paper\d* | template | article\s*\d*
    | .*\.(docx?|tex|pdf|indd|pptx?)\b            # a filename someone forgot to replace
    | \S*_?proof\s+[\d.]+                          # a typesetter proof stamp
    | [\w-]*\d{4,}[\w-]*$                          # an accession id with no words in it
)""", re.I | re.X)


def _pdf_title(data: bytes) -> str | None:
    """The PDF's own `/Title`, when it looks like a real title. None otherwise.

    THE LAST METADATA SOURCE THERE IS. A raw hosted `.pdf` link gets `s2_lookup: None` from
    `_parse_paper_url`, so Semantic Scholar is never even called, and `openAccessPdf` is set to the
    url itself — meaning the body ALWAYS resolves and the title is ALWAYS absent. Every such link
    through `link_router` (a hopper deposit) or `ingest_x_footprint` (an Oracle linking a paper on
    X) therefore minted `# Untitled` with a full body, permanently under policy B. The footprint
    path writes `author_referenced` atoms — human-attested — so the bad row landed in the tier the
    KB is built on. Verified live on a 147,341-character ifo.de working paper.

    PRECISION OVER RECALL, deliberately. Measured over 120 PDFs whose true title was known from
    OpenAlex: `/Title` is present on 65%, survives the junk gate on 63%, and is RIGHT 91% of the
    time it survives. The other 37% keep no title, which is honest — and that is the whole trade.
    A wrong title is a false claim that reads as true and can never be corrected; an absent one is
    a visible gap. Same principle as the `FETCH_ABSENT` fix in `3c26eb63`.

    The first line of the extracted text was measured as an alternative and REJECTED: 100%
    coverage but it grabs the running journal header ("International Journal of Creative and Open
    Research") as often as the title, and a journal name standing in for a paper title is exactly
    the false-claim case above. Coverage is not worth buying with precision here."""
    try:
        import io
        from pypdf import PdfReader
        meta = PdfReader(io.BytesIO(data)).metadata
        title = " ".join((getattr(meta, "title", None) or "").split()) if meta else ""
    except Exception:
        return None
    if len(title) < 12 or _PDF_TITLE_JUNK.match(title):
        return None
    return title


def _pdf_bytes_to_text(data: bytes) -> str | None:
    """PDF bytes → repaired text, or None when what came back is not a WHOLE document.

    TWO floors, because they answer two different questions. `_MIN_FULLTEXT_CHARS` asks whether
    there is enough text to be a body at all. `_FLOOR_PER_PAGE` asks whether we got the whole
    document — which total chars cannot tell, and getting it wrong writes a `body_state: complete`
    that is FALSE and, under policy-B dedup, permanent.

    LOAD-BEARING fail-safe (CLAUDE.md): any extractor failure returns None and the caller falls
    back to an honest abstract-only atom. This never raises."""
    try:
        text, pages = _extract_pypdf(data)
    except Exception:
        return None
    text = _repair_extraction(text).strip()
    if len(text) < _MIN_FULLTEXT_CHARS or len(text) < _FLOOR_PER_PAGE * pages:
        return None
    return text


def resolve_fulltext(paper: dict) -> str | None:
    """The full document text (NOT the abstract) for a Paper, or None → the caller builds an
    abstract-only atom.

    Multi-source, OPEN first: arXiv PDF > `openAccessPdf` > (future mirrors). LOAD-BEARING
    fail-safe (CLAUDE.md): no OA url / download fails / pypdf missing / extraction too thin all
    degrade to None — this NEVER raises. Extracted text is capped at `_PAPER_MAX_CHARS` so a
    pathological PDF can't blow the embed bill (real papers are far shorter and never truncated).

    Its url list stays PURE and offline. A source S2 has never heard of reaches this with a body
    anyway, because whoever resolved the paper's METADATA also filled in `openAccessPdf` — see
    `_openalex_metadata` and `_zenodo_metadata`. Body discovery belongs beside metadata resolution,
    not here.

    FILLS IN A MISSING TITLE from the PDF itself, on the Paper. That is a deliberate mutation, and
    it is the same call the module already makes for `_S2_VERDICT` (see its comment at the top):
    the Paper is what crosses every seam — the prefetch pools, the caller threading, and
    `atomize_paper`'s public signature — so a fact discovered here reaches the minter by riding on
    it. Returning a tuple instead would change that public signature and five call sites to avoid
    one documented write. It happens HERE because this is the only place that holds both the PDF
    bytes and the Paper; `paper_from_url` runs long before a byte is fetched."""
    for url in _fulltext_pdf_urls(paper):
        data = _download_pdf(url)
        if not data:
            continue
        text = _pdf_bytes_to_text(data)
        if text:
            # Only when nothing better is known. S2, `known=` and both metadata fallbacks win.
            if not (paper.get("title") or "").strip() and (t := _pdf_title(data)):
                paper["title"] = t
            return text[:_PAPER_MAX_CHARS]
    return None


# ══════════════════════════════════════════════════════════════════════════════════
# 3. markdown + atomize_paper
# ══════════════════════════════════════════════════════════════════════════════════

def paper_to_markdown_full(paper: dict, full_text: str | None) -> str:
    """Render a Paper (+ optional full body) to the snapshot markdown that gets chunked + embedded.

    Frontmatter is provenance only (`source/url/date/type`) — the chunker strips it, so none of it
    pollutes the routing vector or the FTS snippet (`strip_frontmatter`). The searchable BODY is
    title + authors + venue/year + Abstract + (when present) the Full text. When `full_text` is
    None this is honestly abstract-only — same structure, just no Full-text section."""
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
    if abstract:
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


def _atom_exists(conn: sqlite3.Connection, atom_id: str) -> bool:
    """Policy-B presence check for a lone `atomize_paper` call (no caller-threaded `seen`). Papers
    are immutable, so presence alone means skip — no content-hash comparison needed."""
    return conn.execute("SELECT 1 FROM atoms WHERE atom_id=?", (atom_id,)).fetchone() is not None


_UNSET = object()   # "no full text was supplied" — distinct from None, which means "resolved to none"

# How many of one paper's authors the atom records. HEP papers carry 3,000 and a user saving one
# is not vouching for 3,000 people; the median is 6 and the max 21 across a measured 50 works
# (2026-09-08). A BOUND, not a defended constant — raise it if real saved papers cluster above it
# without being hyperauthored. Matched by `frontier_sources.MAX_PAYLOAD_AUTHORS` upstream.
MAX_ATOM_AUTHORS = 20


def atom_authors(paper: dict) -> list[dict]:
    """A normalized Paper's authors → the list the ATOM carries, in OPYT's own vocabulary.

    The atom needs this because `who_id` records the FIRST author only, and 67 of 82 live paper
    atoms do not even have that — they carry the `paper-authors:{paper_id}` placeholder because
    Semantic Scholar never resolved the work. So the first author is not a usable handle on the
    people behind a saved paper, and the coauthors reach nothing at all.

    Registry ids stay under SEPARATE keys rather than one `id` field. `scholar:2081297` and
    `openalex:A5043841592` are different namespaces over the same person and collapsing them into
    one column would make an id unusable without also knowing which registry issued it. The ORCID
    is the third and is not a namespace at all — it is the one identifier nobody can claim on
    another person's behalf, so it is what those two namespaces MERGE on.
    """
    out = []
    for a in (paper.get("authors") or [])[:MAX_ATOM_AUTHORS]:
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


def atomize_paper(conn: sqlite3.Connection, embedder, paper: dict, *,
                  entry_mode: str = "author_referenced",
                  seen: dict | None = None,
                  sink=None, on_written=None, fulltext=_UNSET) -> str | None:
    """Normalized Paper → ONE full-text atom. SOURCE-AGNOSTIC: the caller supplies only the bit
    that differs per source — `entry_mode`. Returns the
    `atom_id`, or None if deduped / unidentifiable / the embed-write failed / the S2 metadata fetch
    was BLOCKED and no full text resolved (see the skip below — nothing is written, so a later run
    re-attempts). A caller that vouches must break the None ambiguity with a presence check; only
    the dedup case has an atom to point at.

    `who_id` is the PAPER's own author, always — `derive_paper(paper)["who_id"]`, i.e.
    `scholar:{first_author_id}` — and NEVER the Oracle. There is no override parameter: an
    override is the one mechanism by which that invariant could be broken, and no caller
    ever passed one. Policy B: an already-present atom is skipped
    BEFORE the paid PDF fetch + embed (papers are immutable). `seen` is a caller-threaded
    `{atom_id: raw_hash}` for batch dedup across a run; when absent, the DB is checked directly.

    `sink` + `on_written(atom_id)`: join a caller's batch instead of paying an own embed round-trip
    (see `ingest_common.submit_atom`). With a sink the atom is NOT durable when this returns, so the
    embed/write failure branch below cannot fire — a poisoned chunk is isolated by the sink at flush
    and `on_written` simply never runs, which is the same fail-safe outcome by a different route.

    `fulltext` pre-supplies the resolved PDF text so the network pull can happen off this thread.
    It defaults to a SENTINEL, not None, because None is a real resolved value ("no PDF, use the
    abstract") — collapsing the two would make every abstract-only paper re-pay `resolve_fulltext`."""
    atom_id = paper_atom_id(paper)
    if atom_id is None:
        return None
    # Policy B — skip an already-ingested (immutable) paper BEFORE the paid fetch + embed.
    already = (atom_id in seen) if seen is not None else _atom_exists(conn, atom_id)
    if already:
        # The skip discarded the user's save signal entirely until 2026-08-25: a hopper deposit of
        # a paper Frontier had already crawled left the row in the machine lane forever. The paper
        # is still immutable and still not re-fetched — only the attestation is recorded. No-op
        # when `entry_mode` is the machine one, which is how the stage-3 caller passes through.
        promote_atom(conn, atom_id, entry_mode)
        return None

    assert_model(conn, embedder)             # guard the store's embedding identity BEFORE any spend
    meta = derive.derive_paper(paper)
    full_text = resolve_fulltext(paper) if fulltext is _UNSET else fulltext   # None → abstract-only

    # NO BODY → SKIP. No abstract and no full text means the atom would carry nothing to read,
    # and Policy B makes that permanent — no later run revisits a paper that exists.
    #
    # The verdict is deliberately NOT part of this condition, and that is the 2026-08-26 fix.
    # Keyed on `FETCH_UNDETERMINED` alone, an S2 404 (`FETCH_ABSENT`) fell through and wrote
    # `body_state=absent, body_basis=observed` — "WE determined this paper has no body" — when
    # what actually happened is that the one metadata provider we asked has never heard of it.
    # That is a FALSE observed claim, frozen forever. S2 resolved 1 of 15 OpenAlex DOIs on
    # 2026-08-26 (Zenodo, institutional repositories), so it was the common path, not a corner.
    # Asking "is there a body?" instead of "why is there no body?" cannot make that mistake.
    #
    # Measured cost on the live store, same day: 29 of 30 paper atoms are `complete/observed` and
    # unaffected; the single `absent` one is `paper · Untitled` — exactly the row this prevents.
    #
    # The retry is BOUNDED already and needs nothing new here: stage 3 counts a skip as a failed
    # attempt and forces `rejected` at `frontier_admit.ADMIT_MAX_ATTEMPTS`. The hopper deposits
    # once per user action, and the footprint sweep re-fetches referenced papers regardless.
    if not full_text and not (paper.get("abstract") or "").strip():
        log(f"[paper] {atom_id} SKIPPED — no abstract and no full text resolved; nothing "
            f"written (s2_verdict={paper.get(_S2_VERDICT)}), will retry until the attempt cap")
        return None

    md = paper_to_markdown_full(paper, full_text)

    raw_ref, raw_hash = snapshot_and_hash("paper", atom_id, md, seen if seen is not None else {})

    # Register the paper's own author as an entity (display name only — no identity_links, since
    # an S2 author id has nothing to merge on; unlike a blog home, it does not unify with an Oracle).
    schema.upsert_entity(conn, meta["who_id"], name=meta["who_name"])

    atom = {
        "atom_id": atom_id,
        "source_type": "paper",
        "what_kind": "artifact",              # a research artifact (like a repo), not a hot take
        "who_id": meta["who_id"],               # the PAPER's author — NOT the Oracle
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": paper.get("url") or "",
        "raw_ref": raw_ref,
        "raw_hash": raw_hash,
        "description": meta["description"],
        "entry_mode": entry_mode,             # 'author_referenced' — Oracle referenced, didn't author
        # Two states only, because the no-body skip above already returned. Full text = COMPLETE;
        # abstract-only = PARTIAL (OBSERVED — WE tried the PDF mirrors and came back without one).
        # BODY_ABSENT is unreachable for a paper by construction: an atom with neither is never
        # written, so there is no state left for "we stored it with nothing in it".
        # `title`/`abstract`/`external_ids`/`pdf_url` are what `upgrade_to_fulltext` rebuilds the
        # Paper from when a reader opens an abstract-only atom later. They are carried as DATA so
        # that upgrade never has to parse the snapshot markdown back into fields — the same reason
        # `authors` is here rather than re-derived from the rendered name list below. `abstract`
        # duplicates a span of the snapshot, and that is the price of the rule.
        "payload": {"has_fulltext": bool(full_text), "year": paper.get("year"),
                    "citationCount": paper.get("citationCount", 0),
                    "venue": paper.get("venue", ""),
                    "title": (paper.get("title") or "").replace("\n", " ").strip(),
                    "abstract": (paper.get("abstract") or "").strip(),
                    "external_ids": paper.get("externalIds") or {},
                    "pdf_url": (paper.get("openAccessPdf") or {}).get("url") or "",
                    # ALL of them, capped — not just `who_id`'s first author. A reader that wants
                    # the people behind a saved paper has nowhere else to look: the snapshot
                    # markdown renders names for display, and re-deriving them from that would be
                    # parsing a presentation string as data.
                    "authors": atom_authors(paper),
                    **body_fields(BODY_COMPLETE if full_text else BODY_PARTIAL,
                                  BASIS_OBSERVED)},
    }
    try:                                      # chunks + embeds FULL text (batched when `sink` given)
        submit_atom(conn, embedder, sink, atom=atom, snapshot_text=md,
                    on_written=on_written)
    except Exception as e:                    # embed/write failure → SKIP (no atom, no seen mark)
        log(f"[footprint] paper atom {atom_id} skipped (embed/write failed): {e}")
        return None
    if seen is not None:
        seen[atom_id] = raw_hash
    return atom_id


# ══════════════════════════════════════════════════════════════════════════════════
# 4. upgrade_to_fulltext — the abstract-only atom a reader actually opened
# ══════════════════════════════════════════════════════════════════════════════════
# Papers arrive abstract-only from exactly one adapter — `ingest_scholar_footprint`, which pulls a
# scholar Oracle's whole back catalogue and deliberately fetches no PDFs (928 works, one paged call).
# That is the right trade for a corpus nobody has read yet, and the wrong one the moment somebody
# asks about ONE of those papers in depth: `open()` handed back a 2,000-char abstract and there was
# no second act, because Policy B skips a present paper BEFORE the fetch, so no re-ingest could ever
# deepen it. This is that second act, and `kb_open` is its only caller.
#
# THE POLICY-B EXCEPTION, STATED NARROWLY. Papers are immutable: a repeat ingest must never re-pay
# for a body we hold. That is a rule about not spending twice, NOT a claim that a body can never
# improve — `upsert_atom` has always overwritten in place and bumped `version` ("an audit trail of
# how many times this atom was re-observed"), and `replace_chunks` exists precisely because a
# changed snapshot shifts chunk boundaries. So the exception here is ONE-WAY and the narrowest one
# that does the job: `partial → complete`, body columns only, never the reverse and never a
# metadata rewrite. Identity fields are copied from the stored row verbatim rather than re-derived,
# so an upgrade cannot move `who_id`, `when_ts` or the atom id itself no matter what a PDF says.
_FULLTEXT_RETRY_DAYS = 1      # A failed attempt is stamped so that re-opening the same paper does
                              # not re-pay the same failed download — within a working session,
                              # which is where the repetition actually happens.
                              #
                              # ONE DAY, NOT A MONTH, because the stamp cannot tell the two
                              # failures apart: `_download_pdf` returns None for "this paper is
                              # paywalled and always will be" and for "the wifi dropped" alike.
                              # Tuned for the second, since the first costs only one bounded
                              # download a day for a paper somebody keeps opening, while a month
                              # of lockout on a transient blip defeats the entire gesture — the
                              # reader asked for THIS paper in depth, and telling them to come
                              # back in September is not an answer. It doubles as the cheap
                              # version of "a mirror may appear later": we simply re-ask tomorrow.


def _retry_due(stamp: str | None) -> bool:
    """Is a fresh full-text attempt due? No stamp → yes. Unparseable → yes (fail toward trying)."""
    if not stamp:
        return True
    try:
        return date.fromisoformat(str(stamp)) <= date.today() - timedelta(days=_FULLTEXT_RETRY_DAYS)
    except ValueError:
        return True


def _paper_from_atom(row: sqlite3.Row, payload: dict) -> dict:
    """The stored atom → the Paper shape `resolve_fulltext` and `paper_to_markdown_full` take.

    Rebuilt from COLUMNS AND PAYLOAD ONLY. The snapshot markdown holds the same facts in a nicer
    order, and reading them back out of it would be parsing a presentation string as data — the
    thing `atomize_paper` already refuses to do for its author list. `paperId` is pinned to the
    stored atom id, so `derive_paper` re-derives the identity this atom already has instead of
    whatever a freshly-fetched PDF might imply."""
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
    """The atom's author records → back into the S2 field names a Paper speaks.

    `atom_authors` translates ONE way at mint time (`authorId` → `scholar_id`, `openalexId` →
    `openalex_id`), so rebuilding a Paper from the atom has to translate back or hand every
    downstream reader a dict whose keys it does not know. `derive_paper` is the one that would
    quietly misread it — it keys `who_id` off `authorId` — and while an upgrade copies `who_id`
    from the stored row rather than re-deriving it, a Paper that lies about its own shape is a
    trap laid for whoever reaches for this next.

    Lossy in one direction only, and harmlessly: the stored list is capped at `MAX_ATOM_AUTHORS`,
    so a 40-author paper re-renders its "+N more" tail from 20 rather than 40. The names shown are
    the same names; only the count of the unshown remainder shrinks.
    """
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


_UPGRADE_COLS = ("atom_id", "source_type", "what_kind", "who_id", "when_ts", "when_precision",
                 "about_entities", "source_url", "raw_ref", "raw_hash", "description",
                 "payload", "entry_mode")


def upgrade_to_fulltext(conn: sqlite3.Connection, embedder_for, atom_id: str) -> bool:
    """An abstract-only paper atom → the whole document, rewritten in place. True when it grew.

    `embedder_for` is a ZERO-ARG CALLABLE, not an embedder, the same shape
    `frontier_admit.admit_one` takes and for the same reason: most calls resolve no PDF, and
    constructing an embedder needs an API key and a network this read path should not touch until
    there is something to embed.

    FAIL-SAFE, and load-bearing here more than anywhere (CLAUDE.md): this runs inside `open()`, so
    every failure path must return the reader their abstract rather than an error. No open url, a
    download that 404s, a scanned PDF under the per-page floor, a missing pypdf, an embed that
    fails — all return False having written no body. The one thing a failure DOES write is the
    attempt stamp, which is bookkeeping and not a claim that the work finished: it stops the next
    open re-paying the same failed download, and `_FULLTEXT_RETRY_DAYS` lets it be asked again."""
    row = conn.execute(
        f"SELECT {', '.join(_UPGRADE_COLS)} FROM atoms WHERE atom_id = ?", (atom_id,)).fetchone()
    if row is None or row["source_type"] != "paper":
        return False
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return False
    # Only the one transition. `complete` is done, `absent` is unreachable for a paper by
    # construction (`atomize_paper` never writes an atom with neither abstract nor body), and
    # `has_fulltext` is checked too so a payload that disagrees with itself is left alone.
    if payload.get("body_state") != BODY_PARTIAL or payload.get("has_fulltext"):
        return False
    if not _retry_due(payload.get("fulltext_tried_at")):
        return False

    paper = _paper_from_atom(row, payload)
    full_text = resolve_fulltext(paper)
    if not full_text:
        _stamp_attempt(conn, atom_id, payload)
        return False

    md = paper_to_markdown_full(paper, full_text)
    stamped = snapshot_and_hash("paper", atom_id, md, {row["atom_id"]: row["raw_hash"]})
    if stamped is None:                       # byte-identical to what we hold → nothing grew
        _stamp_attempt(conn, atom_id, payload)
        return False
    raw_ref, raw_hash = stamped

    atom = {c: row[c] for c in _UPGRADE_COLS if c not in ("about_entities", "payload")}
    atom["about_entities"] = _json_list(row["about_entities"])
    atom["raw_ref"], atom["raw_hash"] = raw_ref, raw_hash
    atom["payload"] = {k: v for k, v in payload.items() if k != "fulltext_tried_at"}
    atom["payload"].update({"has_fulltext": True,
                            **body_fields(BODY_COMPLETE, BASIS_OBSERVED)})
    try:
        embedder = embedder_for()             # built HERE — only now is there something to embed
        assert_model(conn, embedder)          # same subspace guard `atomize_paper` takes, and for
                                              # the same reason: these chunks join the ones the
                                              # local search arm ranks against.
        # Overwrites the row (bumping `version`) and REPLACES every chunk, so the abstract-only
        # chunks cannot linger beside the full-text ones and be retrieved as a separate hit.
        store_atom(conn, embedder, atom=atom, snapshot_text=md)
    except Exception as e:
        # No stamp on this arm: the body WAS reachable and only the write failed, so the next open
        # should retry immediately rather than wait out the TTL.
        log(f"[paper] {atom_id} full-text upgrade skipped (embed/write failed): {e}")
        return False
    log(f"[paper] {atom_id} upgraded to full text ({len(full_text):,} chars)")
    return True


def _json_list(raw) -> list:
    try:
        out = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


def _stamp_attempt(conn: sqlite3.Connection, atom_id: str, payload: dict) -> None:
    """Record that we tried and found no open PDF. Payload-only: no `version` bump, no body
    column touched, because nothing about the atom's content changed."""
    try:
        conn.execute("UPDATE atoms SET payload = ? WHERE atom_id = ?",
                     (json.dumps({**payload, "fulltext_tried_at": date.today().isoformat()}),
                      atom_id))
        conn.commit()
    except Exception:      # bookkeeping must never break the read it rides on
        pass
