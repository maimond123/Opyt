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

import re
import sqlite3
import unicodedata
from urllib.parse import urlparse

from . import derive, schema
from .embed import assert_model
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL, FETCH_ABSENT,
                            FETCH_OK, FETCH_UNDETERMINED, body_fields, promote_atom,
                            snapshot_and_hash, submit_atom)

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

# arXiv: /abs/{id} or /pdf/{id}[.pdf]; id may contain a slash (old-style hep-th/9901001).
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?(?:[?#].*)?$", re.I)
# DOI: doi.org / dx.doi.org / a raw DOI path (10.NNNN/suffix).
_DOI_RE = re.compile(r"(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s?#]+)", re.I)
# arXiv mints a DOI for every preprint under the 10.48550 prefix. It is an arXiv id wearing a DOI,
# not a second paper — see the DOI branch below for why it is collapsed here.
_ARXIV_DOI_RE = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)
# Semantic Scholar paper page: /paper/[slug/]{40-hex-hash | corpus-id-digits}.
_S2_RE = re.compile(r"semanticscholar\.org/paper/(?:[^/]+/)?([0-9a-f]{40}|\d+)", re.I)
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

    m = _DOI_RE.search(u)
    if m:
        raw = m.group(1).rstrip(".")
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


def _fetch_s2_paper(lookup_id: str | None) -> tuple[dict | None, str]:
    """Best-effort Semantic Scholar single-paper fetch → `(data | None, verdict)`.

    Returns a VERDICT, not just None: collapsing "S2 rate-limited us" into a bare None let a 429
    be recorded downstream as "this paper has no abstract" — unauthenticated S2 allows ~1 req/s, so
    bursts 429 routinely.

      FETCH_OK           — S2 answered. Whatever it did or did not include is the truth.
      FETCH_ABSENT       — no lookup id, or a 404. A real answer; retrying changes nothing.
      FETCH_UNDETERMINED — 429 / any other status / transport failure. We were STOPPED.

    Still never raises: the enrichment itself stays a bonus (fail-safe).
    """
    if not lookup_id:
        return None, FETCH_ABSENT
    try:
        import requests
        resp = requests.get(f"{_S2_BASE}/paper/{lookup_id}", params={"fields": _S2_FIELDS},
                            timeout=_PDF_TIMEOUT, headers=_s2_headers())
        if resp.status_code == 404:
            return None, FETCH_ABSENT           # S2 genuinely has no record of this paper
        if resp.status_code != 200:
            return None, FETCH_UNDETERMINED     # 429 above all — we were throttled, not answered
        data = resp.json()
        return (data, FETCH_OK) if isinstance(data, dict) else (None, FETCH_UNDETERMINED)
    except Exception:
        return None, FETCH_UNDETERMINED         # transport failure — indistinguishable from a block


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


# Zenodo mints every deposit under this prefix; the digits are the RECORD id, which is all its
# api needs. Zenodo is reached because it is KEYLESS — no email, no token, no setup asked of the
# user, which is exactly what disqualified Unpaywall as a general answer.
_ZENODO_DOI_RE = re.compile(r"^10\.5281/zenodo\.(\d+)$", re.I)
_ZENODO_API = "https://zenodo.org/api/records"


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
        import requests
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
        # Gated on a MISSING TITLE, which is exactly the state that mints an `Untitled` atom — so
        # the request fires only when it is the difference between a good row and a permanent bad
        # one. A caller that supplied `known=` (frontier) or an S2 that answered both pay nothing.
        if not (paper.get("title") or "").strip() and (extra := _zenodo_metadata(paper)):
            paper.update(extra)
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
        import requests
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
    `_zenodo_metadata`. Body discovery belongs beside metadata resolution, not here.

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
            # Only when nothing better is known. S2, `known=` and `_zenodo_metadata` all win.
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


def atomize_paper(conn: sqlite3.Connection, embedder, paper: dict, *,
                  who_id: str | None = None, entry_mode: str = "author_referenced",
                  seen: dict | None = None,
                  sink=None, on_written=None, fulltext=_UNSET) -> str | None:
    """Normalized Paper → ONE full-text atom. SOURCE-AGNOSTIC: the caller supplies only the bit
    that differs per source — `entry_mode`. Returns the
    `atom_id`, or None if deduped / unidentifiable / the embed-write failed / the S2 metadata fetch
    was BLOCKED and no full text resolved (see the skip below — nothing is written, so a later run
    re-attempts). A caller that vouches must break the None ambiguity with a presence check; only
    the dedup case has an atom to point at.

    `who_id` is the PAPER's own author (defaults to `derive_paper(paper)["who_id"]` —
    `scholar:{first_author_id}`), NEVER the Oracle. Policy B: an already-present atom is skipped
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
    resolved_who = who_id or meta["who_id"]
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
        from pipeline.ingestion.utils import log
        log(f"[paper] {atom_id} SKIPPED — no abstract and no full text resolved; nothing "
            f"written (s2_verdict={paper.get(_S2_VERDICT)}), will retry until the attempt cap")
        return None

    md = paper_to_markdown_full(paper, full_text)

    decided = snapshot_and_hash("paper", atom_id, md, seen if seen is not None else {})
    if decided is None:                      # unchanged snapshot (immutable → unreachable) — honor seam
        return None
    raw_ref, raw_hash = decided

    # Register the paper's own author as an entity (display name only — no identity_links, since
    # an S2 author id has nothing to merge on; unlike a blog home, it does not unify with an Oracle).
    schema.upsert_entity(conn, resolved_who, name=meta["who_name"])

    atom = {
        "atom_id": atom_id,
        "source_type": "paper",
        "what_kind": "artifact",              # a research artifact (like a repo), not a hot take
        "who_id": resolved_who,               # the PAPER's author — NOT the Oracle
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
        "payload": {"has_fulltext": bool(full_text), "year": paper.get("year"),
                    "citationCount": paper.get("citationCount", 0),
                    "venue": paper.get("venue", ""),
                    **body_fields(BODY_COMPLETE if full_text else BODY_PARTIAL,
                                  BASIS_OBSERVED)},
    }
    try:                                      # chunks + embeds FULL text (batched when `sink` given)
        submit_atom(conn, embedder, sink, atom=atom, snapshot_text=md,
                    on_written=on_written)
    except Exception as e:                    # embed/write failure → SKIP (no atom, no seen mark)
        from pipeline.ingestion.utils import log
        log(f"[footprint] paper atom {atom_id} skipped (embed/write failed): {e}")
        return None
    if seen is not None:
        seen[atom_id] = raw_hash
    return atom_id
