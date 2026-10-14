"""Resolve exact paper identifiers to registry evidence: the DOI behind an id that needs a request
to become one (PubMed, PMC, Europe PMC, Zenodo, OpenAlex), Semantic Scholar's record of a paper,
and the metadata arXiv, OpenAlex and Zenodo hold for it, in Semantic Scholar's field names.

Every lookup is fail-safe and exact: a declared identifier or nothing, never a bibliographic
guess. What the evidence becomes — a Paper's fetch verdict, its provenance, an atom — is the
library's (`pipeline/kb/ingest_papers.py`).
"""
from __future__ import annotations

import re
import time
from typing import NamedTuple

import requests

from pipeline.credentials import s2_headers
from pipeline.host_access import request as host_request
from pipeline.ingestion.utils import log
from pipeline.platforms.papers import arxiv, openalex
from pipeline.platforms.papers.registry import Pace, paced_read

# Every lookup request here goes to a registry that issued no credential: the Semantic Scholar key
# rides only on Semantic Scholar's own requests (`s2_headers`), never on these.
_UA = {"User-Agent": "opyt-paper-adapter/1.0"}
_TIMEOUT = 30

# What Semantic Scholar's answer about one paper was. The library maps these to its own verdicts.
FOUND, ABSENT, UNDETERMINED = "found", "absent", "undetermined"


class LookupResult(NamedTuple):
    """Semantic Scholar's record of one paper (or None) and whether it `FOUND` the paper,
    found it `ABSENT`, or could not tell (`UNDETERMINED`: throttled or unreachable)."""
    data: dict | None
    status: str


# PubMed: /{PMID}. The id is not a DOI and carries no route to one inside the string, so unlike
# every other branch this needs a lookup — see `_pubmed_doi` for where it happens and why not here.
_PUBMED_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{4,9})", re.I)
# PubMed Central, on both the legacy `www.ncbi.nlm.nih.gov/pmc/articles/…` path and the
# `pmc.ncbi.nlm.nih.gov/articles/…` host NCBI moved it to. Same E-utilities lookup as a PMID with
# `db=pmc`, and the same reason it cannot be a string rule: the PMCID carries no route to a DOI.
# PMC answers 403 to a server fetch, so the `citation_doi` probe cannot rescue it.
_PMC_RE = re.compile(r"ncbi\.nlm\.nih\.gov/(?:pmc/)?articles/(?:PMC)?(\d+)", re.I)
# A Zenodo record page; `_zenodo_record_doi` asks Zenodo for the DOI the record declares.
# Europe PMC addresses an article as `/{article|abstract}/{SOURCE}/{id}`: MED is a PMID and PMC a
# PMCID (both resolved by NCBI's E-utilities above); PPR is a preprint only Europe PMC's own index
# knows, hence the separate resolver.
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

# ── Semantic Scholar single-paper enrichment ─────────────────────────────────────
_S2_BASE = "https://api.semanticscholar.org/graph/v1"
# The Paper shape — reused from the vault ingester's FIELDS so both sides agree on the metadata.
_S2_FIELDS = ("title,abstract,year,url,externalIds,citationCount,"
              "publicationDate,authors,openAccessPdf,venue")
# S2 retry. Unauthenticated S2 is one shared pool, so a 429 says the pool is busy, and a paper
# that lands without a title can never be repaired. A fixed interval, because S2's 429 carries no
# `Retry-After`.
_S2_RETRIES = 5
_S2_RETRY_SLEEP = 1.5


def _eutils_doi(db: str, uid: str) -> str | None:
    """A PubMed or PMC id → the paper's DOI, or None. One keyless request to NCBI E-utilities.
    `db` is `"pubmed"` (a PMID) or `"pmc"` (a PMCID).

    Not in `_parse_paper_url`, which must stay pure string work (`link_mint.predicted_atom_id`
    rests on it). `paper_from_url` resolves the DOI and re-parses, so PubMed mints the same atom
    as the DOI. Any failure returns None.
    """
    try:
        resp = host_request(requests.get, _EUTILS, timeout=_TIMEOUT, headers=_UA,
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
    """A Zenodo record number → the DOI that record declares, or None. One keyless request.

    Asked rather than derived: a record number can be a concept id whose current version carries
    a different DOI. Any failure returns None.
    """
    try:
        resp = host_request(requests.get, f"{_ZENODO_API}/{record_id}", timeout=_TIMEOUT, headers=_UA)
        if resp.status_code != 200:
            return None
        rec = resp.json()
        doi = (rec.get("doi") or (rec.get("metadata") or {}).get("doi") or "").strip()
        return doi or None
    except Exception as e:
        log(f"[papers] zenodo record {record_id} DOI lookup failed: {type(e).__name__}: {e}")
    return None


# One paced door per host for the whole paper path. `min_interval_s` is declared on the adapter
# and applied by the caller, so it is enforced here, once, for every read on this path; it
# serializes concurrent callers to keep the shared breaker closed.
_OPENALEX_PACE = Pace()
_ARXIV_PACE = Pace()


def _openalex_read(fn):
    """One OpenAlex read on the paper path — see `registry.paced_read`."""
    return paced_read(openalex.OpenAlexClient(), _OPENALEX_PACE, fn, "openalex")


def _europepmc_ppr_doi(ppr_id: str) -> str | None:
    """A Europe PMC preprint id (`PPR217527`) → the DOI it declares, or None. One keyless request.

    NCBI does not index preprints; Europe PMC hands back the DOI the preprint server minted.
    `SRC:PPR` is in the query because Europe PMC ids are only unique within a source. Any failure
    returns None.
    """
    try:
        resp = host_request(requests.get, _EUROPEPMC_SEARCH, timeout=_TIMEOUT, headers=_UA,
                            params={"query": f"EXT_ID:{ppr_id} AND SRC:PPR",
                                    "resultType": "core", "format": "json", "pageSize": 1})
        if resp.status_code != 200:
            return None
        hits = ((resp.json().get("resultList") or {}).get("result")) or []
        return (str(hits[0].get("doi") or "").strip() or None) if hits else None
    except Exception as e:
        log(f"[papers] europepmc preprint {ppr_id} DOI lookup failed: {type(e).__name__}: {e}")
    return None


def openalex_doi_by_url(url: str) -> str | None:
    """A url no pattern here can read → the DOI of the paper OpenAlex says lives at that url.

    The last resort, mostly for institutional repositories and aggregators. Still a declared
    identifier: `locations.landing_page_url` is an exact match against a string OpenAlex stores,
    not a bibliographic guess. Refuses ambiguity rather than taking the top hit. Any failure, any
    ambiguity, anything without a DOI → None.
    """
    if not (url or "").startswith(("http://", "https://")):
        return None
    return one_declared_doi(_openalex_read(lambda a: a.works_by_landing_page(url)) or [])


def one_declared_doi(works: list[dict]) -> str | None:
    """The one DOI these works declare, or None when they declare none or several.

    Several records carrying the same DOI are one paper; several DOIs are an ambiguity this
    refuses to resolve by guess."""
    dois = {d for w in works if (d := (w.get("doi") or "").replace("https://doi.org/", "").lower())}
    return dois.pop() if len(dois) == 1 else None


def looked_up_doi(url: str) -> str | None:
    """A url whose identifier needs a request to become a DOI → that DOI, or None.

    Collected here so `paper_from_url` keeps one network-resolution branch and `_parse_paper_url`
    stays pure string work.
    """
    found = lookup_identifier(url)
    if found is None:
        return None
    source, uid = found
    if source in ("pubmed", "pmc"):
        return _eutils_doi(source, uid)
    if source == "europepmc_ppr":
        return _europepmc_ppr_doi(uid)
    if source == "zenodo":
        return _zenodo_record_doi(uid)
    return _openalex_work_doi(uid)


def lookup_identifier(url: str) -> tuple[str, str] | None:
    """The registry and exact id a url names, when only a request can turn it into a DOI.

    `("pubmed", pmid)` and `("pmc", pmcid)` are asked of NCBI E-utilities, `("europepmc_ppr", id)`
    of Europe PMC, `("zenodo", record)` of Zenodo and `("openalex", work)` of OpenAlex. Patterns
    are tried in this order and the first that matches wins.
    """
    if m := _PUBMED_RE.search(url):
        return "pubmed", m.group(1)
    if m := _PMC_RE.search(url):
        return "pmc", m.group(1)
    if m := _EUROPEPMC_PMC_RE.search(url):
        return "pmc", m.group(1) or m.group(2)
    if m := _EUROPEPMC_PMID_RE.search(url):
        return "pubmed", m.group(1)
    if m := _EUROPEPMC_PPR_RE.search(url):
        return "europepmc_ppr", m.group(1)
    if m := _ZENODO_RECORD_RE.search(url):
        return "zenodo", m.group(1)
    if m := _OPENALEX_WORK_RE.search(url):
        return "openalex", m.group(1)
    return None


def _openalex_work_doi(work_id: str) -> str | None:
    """An OpenAlex `W…` id → that work's DOI, or None. Fail-safe, like every lookup here."""
    w = _openalex_read(lambda a: a.work_by_openalex_id(work_id))
    doi = ((w or {}).get("doi") or "").replace("https://doi.org/", "").strip().lower()
    return doi or None


def fetch_s2_paper(lookup_id: str | None) -> LookupResult:
    """Best-effort Semantic Scholar single-paper fetch → `LookupResult(data | None, status)`.

    Returns a verdict, so a 429 is never recorded as "this paper has no abstract":

      FOUND        — S2 answered. Whatever it did or did not include is the truth.
      ABSENT       — no lookup id, or a 404. A real answer; retrying changes nothing.
      UNDETERMINED — 429 / any other status / transport failure, after `_S2_RETRIES`
                           attempts.

    Only the undetermined case is retried. Never raises.
    """
    for attempt in range(_S2_RETRIES):
        status, data = _s2_get(lookup_id) if lookup_id else (None, None)
        outcome = s2_outcome(bool(lookup_id), status, data)
        if outcome == FOUND:
            return LookupResult(data, FOUND)
        if outcome == ABSENT:
            return LookupResult(None, ABSENT)
        # Sleep only between attempts, never after the last one.
        if attempt + 1 < _S2_RETRIES:
            time.sleep(_S2_RETRY_SLEEP)
    return LookupResult(None, UNDETERMINED)     # 429 above all — we were throttled, not answered


def _s2_get(lookup_id: str) -> tuple[int | None, object]:
    """One Semantic Scholar request → `(status, decoded body)`; `(None, None)` on a transport or
    decode failure, which is indistinguishable from a block. The body is decoded only on a 200."""
    try:
        resp = host_request(requests.get, f"{_S2_BASE}/paper/{lookup_id}", params={"fields": _S2_FIELDS},
                            timeout=_TIMEOUT, headers=s2_headers())
        return resp.status_code, (resp.json() if resp.status_code == 200 else None)
    except Exception:
        return None, None


def s2_outcome(has_id: bool, status: int | None, data) -> str | None:
    """One Semantic Scholar attempt's verdict: `FOUND`, `ABSENT`, or None to retry.

    No lookup id and a 404 are answers (S2 has no record of this paper); a 200 with an object
    body is the record; anything else — a non-object 200, a 429, any other status, a transport
    failure — is retried."""
    if not has_id or status == 404:
        return ABSENT
    if status == 200 and isinstance(data, dict):
        return FOUND
    return None


def s2_authors_from_openalex(authors: list[dict]) -> list[dict]:
    """OpenAlex author rows (`openalex.openalex_authors`' shape, or a staged candidate's
    `payload["authors"]`) → the library's S2-shaped author records.

    `openalexId`, never `authorId`: an OpenAlex author id is not a Semantic Scholar one, and
    `derive_paper` reads `authorId` to mint `who_id = scholar:{id}`. Writing it there would assert
    a scholar identity that does not exist; under its own key it is an honest second registry id
    that `derive_paper` ignores and `atomize_paper` keeps on the atom. `orcid` and `position` keep
    their names. An empty value is omitted, never written as a null.
    """
    return [{"name": a["name"],
             **({"openalexId": oid} if (oid := a.get("openalex_id")) else {}),
             **({"orcid": orc} if (orc := a.get("orcid")) else {}),
             **({"position": pos} if (pos := a.get("position")) else {})}
            for a in authors]


def openalex_metadata(paper: dict) -> dict | None:
    """A DOI → OpenAlex's record of that work, in the S2 field names, or None.

    Covers papers S2 has not indexed (indexing lag, sources S2 does not carry). Gated on a
    missing title by its caller. Metadata only — never `paperId`, `externalIds` or `url`:
    identity is decided by `_parse_paper_url`. Any failure returns None.
    """
    doi = (paper.get("externalIds") or {}).get("DOI")
    if not doi:
        return None
    work = _openalex_read(lambda a: a.work_by_doi(str(doi).strip()))
    if not isinstance(work, dict) or not work:
        return None
    rec = openalex.work_record(work)
    out = {
        "title": rec.title,
        # `[:2000]` mirrors `oracles/sources/archive/openalex._paper_from_work`, so an abstract reads the
        # same however it entered the store.
        "abstract": rec.abstract[:2000],
        "authors": s2_authors_from_openalex(openalex.openalex_authors(work)),
        "publicationDate": (rec.published or "")[:10],
        "year": work.get("publication_year"),
        "venue": rec.venue or "",
        "citationCount": work.get("cited_by_count") or 0,
    }
    # The open copy (often the authors' preprint); `pdf.fulltext_pdf_urls` reads this field.
    if rec.pdf_url:
        out["openAccessPdf"] = {"url": rec.pdf_url}
    # Empties stripped, because the caller merges with `paper.update()`: a null abstract in the
    # record must not erase an abstract a `known=` caller already supplied.
    return {k: v for k, v in out.items() if v} or None


def arxiv_metadata(paper: dict) -> dict | None:
    """An arXiv id → arXiv's own record of that paper, in the S2 field names, or None.

    For arXiv papers S2 returns empty; OpenAlex cannot cover them (no DOI). An exact lookup, so
    admissible where a title search is not. Metadata only, like OpenAlex. Authors carry no
    `authorId` (arXiv has no author registry). Any failure, or a record for a different id,
    returns None."""
    arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
    if not arxiv_id:
        return None
    bare = arxiv.strip_arxiv_version(str(arxiv_id).strip())
    rec = paced_read(arxiv.ArxivClient(), _ARXIV_PACE, lambda a: a.by_id(bare), "arxiv")
    if rec is None or rec.arxiv_id != bare:
        return None
    out = {"title": rec.title,
           "abstract": rec.summary[:2000],      # the same cut as the OpenAlex abstract above
           "authors": [{"name": n} for n in rec.authors],
           "publicationDate": rec.published or ""}
    return {k: v for k, v in out.items() if v} or None


# Zenodo mints every deposit under this prefix; the digits are the id its api takes. Keyless.
_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)$", re.I)


def zenodo_metadata(paper: dict) -> dict | None:
    """A Zenodo deposit's own record → the S2-shaped fields, or None. One request, no key.

    S2 does not resolve most Zenodo DOIs, which would otherwise mint `# Untitled` permanently.
    The same record also carries the PDF url (the depositor's filename, not derivable from the
    DOI), handed over as `openAccessPdf`. Any failure, or an unexpected shape, returns None."""
    m = _ZENODO_DOI_RE.match(((paper.get("externalIds") or {}).get("DOI") or "").strip())
    if not m:
        return None
    try:
        resp = host_request(requests.get, f"{_ZENODO_API}/{m.group(1)}", timeout=_TIMEOUT, headers=_UA)
        if resp.status_code != 200:
            return None
        rec = resp.json()
    except Exception:
        return None
    if not isinstance(rec, dict):
        return None
    meta = rec.get("metadata") or {}
    files = rec.get("files") or []
    creators = (meta.get("creators") or []) if isinstance(meta, dict) else None
    if not isinstance(meta, dict) or not isinstance(files, list) or not isinstance(creators, list):
        return None
    out: dict = {}
    if title := _text(meta.get("title")):
        out["title"] = title
    if names := [n for c in creators if isinstance(c, dict) and (n := _text(c.get("name")))]:
        out["authors"] = [{"authorId": None, "name": n} for n in names]
    if date := _text(meta.get("publication_date")):
        out["publicationDate"] = date
    if abstract := _html_to_text(_text(meta.get("description"))):
        out["abstract"] = abstract
    # A record holds whatever the depositor uploaded — the paper beside its markdown source, a
    # dataset, a zip. Filtering to `.pdf` is also what makes a `type: software` deposit correctly
    # yield no body, so there is no `resource_type` branch to write.
    for f in files:
        links = f.get("links") if isinstance(f, dict) else None
        link = links.get("self") if isinstance(links, dict) else None
        if link and str(f.get("key") or "").lower().endswith(".pdf"):
            out["openAccessPdf"] = {"url": link}
            break
    return out or None


def _text(value) -> str:
    """A Zenodo string field, stripped; '' for anything that is not a string."""
    return value.strip() if isinstance(value, str) else ""


def _html_to_text(html: str) -> str:
    """Zenodo ships its abstract as HTML. Mirrors `curation/substack_saved`'s converter settings so an
    abstract reads the same however it entered the store. Degrades to '' (fail-safe)."""
    if not (html or "").strip():
        return ""
    try:
        # Imported at call time: html2text is needed only for a Zenodo abstract, and a missing one
        # degrades to no abstract rather than failing the module.
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links, h.ignore_images, h.body_width, h.unicode_snob = False, False, 0, True
        return h.handle(html).strip()
    except Exception:
        return ""
