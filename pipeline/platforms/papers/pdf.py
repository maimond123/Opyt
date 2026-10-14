"""Resolve usable text and metadata evidence from a paper's open PDFs. `fulltext` downloads each
source in order (arXiv's open mirror first, then the open-access copy), keeps the first whose
extracted text is a whole document, and, when its caller still lacks them, reads the file's own
title and creation year.

Evidence only. What the library does with it — which title wins, the provenance markers on the
Paper, the cap on the body it stores — is `pipeline/kb/ingest_papers.resolve_fulltext`'s.
"""
from __future__ import annotations

import difflib
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import date

import requests

from pipeline.host_access import request as host_request
from pipeline.platforms.papers.arxiv import strip_arxiv_version

_MIN_FULLTEXT_CHARS = 500         # below this there is not enough text to be a body at all
                                  # (a cover sheet, a stub) → abstract-only.
_FLOOR_PER_PAGE = 200             # …and below this per page we got only part of the document
                                  # (e.g. a scan whose only text is a per-page watermark). Sits in
                                  # the empty band between image-only scans and real documents.
_PDF_MAX_BYTES = 60 * 1024 * 1024  # download ceiling on the wire (before we ever parse).
_PDF_TIMEOUT = 30

# Headers for PDF downloads, kept separate from `credentials.s2_headers()`: a credential must
# never be sent to a host that did not issue it.
_PDF_UA = {"User-Agent": "opyt-paper-adapter/1.0"}


def fulltext_pdf_urls(paper: dict) -> list[str]:
    """The ordered list of open PDF urls to try, most-reliable first: arXiv's fully-open mirror,
    then `openAccessPdf`. Paywalled publisher PDFs are deliberately absent.

    `openAccessPdf` is whatever the Paper carries: S2's when S2 answered, the finder's when it did
    not (`paper_from_url(known=…)`), which is how OpenAlex-only works get a body."""
    urls: list[str] = []
    ext = paper.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    if arxiv:
        urls.append(f"https://arxiv.org/pdf/{strip_arxiv_version(arxiv)}")
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
        with host_request(requests.get, url, timeout=_PDF_TIMEOUT, stream=True, headers=_PDF_UA) as resp:
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
    """PDF bytes → (raw text, page count). The only extractor; a missing pypdf raises
    `ImportError` here, which `_pdf_bytes_to_text` turns into None. The page count feeds the
    per-page completeness floor."""
    # pypdf is imported at call time so a missing install degrades to abstract-only (pinned).
    from pypdf import PdfReader
    pages = PdfReader(io.BytesIO(data)).pages
    return "\n\n".join((page.extract_text() or "") for page in pages), len(pages)


def _repair_extraction(text: str) -> str:
    r"""Repair the two word-level defects pypdf leaves behind, both of which make real words
    unreachable by search.

    NFKC folds typographic ligatures (`ﬁ` `ﬂ` `ﬀ` `ﬃ`) into their letters. The regex rejoins a
    word split across a line break (`seg-\nmentation` → `segmentation`), lowercase to lowercase
    only, since `GPT-\n4` keeps its hyphen; a naive join, not a prefix word list."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)


# Values a PDF's /Title carries when nobody set one: the authoring tool's default, the source
# filename, a typesetter's proof stamp, or a bare accession id.
_PDF_TITLE_JUNK = re.compile(r"""^(
      untitled | microsoft\s+word | powerpoint | document\d* | manuscript | main | print
    | layout\s*\d* | slide\s*\d* | paper\d* | template | article\s*\d*
    | .*\.(docx?|tex|pdf|indd|pptx?)\b            # a filename someone forgot to replace
    | \S*_?proof\s+[\d.]+                          # a typesetter proof stamp
    | [\w-]*\d{4,}[\w-]*$                          # an accession id with no words in it
)""", re.I | re.X)


def _pdf_title(data: bytes) -> str | None:
    """The PDF's own `/Title`, when it looks like a real title. None otherwise. One of the file's two
    witnesses to its title — `_title_from_pdf` combines it with the first page's.

    A raw hosted `.pdf` link never reaches Semantic Scholar, so the file is its only title
    source. Precision over recall: a wrong title is a false claim that can never be corrected,
    while an absent one is a visible gap."""
    try:
        # Imported at call time, like `_extract_pypdf`: a missing pypdf is a None, not a crash.
        from pypdf import PdfReader
        meta = PdfReader(io.BytesIO(data)).metadata
        title = " ".join((getattr(meta, "title", None) or "").split()) if meta else ""
    except Exception:
        return None
    if len(title) < 12 or _PDF_TITLE_JUNK.match(title):
        return None
    return title


# Large type on a typeset first page that is not the paper's title: a journal's masthead, a
# publisher's or institution's name, a running header.
_PAGE_TITLE_JUNK = re.compile(r"""\b(
      journal | jurnal | revista | issn | doi | vol\.? | volume | issue | abstracts?
    | proceedings | article\s+in\s+press | research\s+article | original\s+(article|research)
    | correspondence | review\s+article | https? | www\. | copyright | bulletin | transactions
    | letters | annals | universit[yàé] | working\s+paper | discussion\s+paper | preprint
    | accepted | received | published | laborator(y|ies) | office | institute | department
    | agency | ministry | cent(er|re) | school | college | faculty
)\b""", re.I | re.X)


def _rotated(m) -> bool:
    return bool(m) and (abs(m[1]) > 1e-6 or abs(m[2]) > 1e-6)


def _pdf_page_title(data: bytes) -> str | None:
    """The first page's title, read by type size: the first block of text set clearly larger than
    the page's body text. None when nothing on the page stands out, or the block fails the gate.

    Not the first line, which grabs a running journal header as often as the title. Rotated runs
    (arXiv's margin stamp) are skipped by their matrix; the block takes everything within 75% of
    the largest size because small-caps titles alternate two sizes."""
    try:
        # Imported at call time, like `_extract_pypdf`: a missing pypdf is a None, not a crash.
        from pypdf import PdfReader
        page = PdfReader(io.BytesIO(data)).pages[0]
        runs: list[tuple[str, float]] = []

        def visit(text, cm, tm, font, size):
            if not (text or "").strip() or _rotated(cm) or _rotated(tm):
                runs.append(("", 0.0))
                return
            runs.append((text, round(size * abs(tm[3] if tm else 1) * abs(cm[3] if cm else 1), 1)))
        page.extract_text(visitor_text=visit)
    except Exception:
        return None
    return page_title(runs)


def page_title(runs: list[tuple[str, float]]) -> str | None:
    """The first page's title from its text runs, each `(text, effective type size)` in reading
    order with `("", 0.0)` for a blank or rotated run: the first block set clearly larger than the
    body text, or None when nothing stands out or the block fails the junk gate."""
    chars: dict[float, int] = {}
    for t, size in runs:
        if size > 0:
            chars[size] = chars.get(size, 0) + len(t.strip())
    if not chars:
        return None
    body = max(chars, key=chars.get)             # the size carrying the most text
    big = max(chars)
    if big < body * 1.15:
        return None                              # nothing stands out: a memo, a court filing
    floor = max(big * 0.75, body * 1.15)
    parts: list[str] = []
    for t, size in runs:
        if size >= floor:
            parts.append(t)
        elif size == 0:
            if parts:
                parts.append(" ")
        elif parts:
            break                                # the first large block ends at the first body run
    title = " ".join(unicodedata.normalize("NFKC", "".join(parts)).split())
    words = title.split()
    if not 3 <= len(words) <= 40 or _PAGE_TITLE_JUNK.search(title) or title[:1].isdigit() \
            or sum(c.isdigit() for c in title) > len(title) * 0.2:
        return None
    return title


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", t or "").lower())


def _title_from_pdf(data: bytes) -> tuple[str | None, str | None]:
    """The PDF's title and where it came from, or `(None, None)`, from the file's two witnesses:
    its `/Title` and its first page. When both give a title and disagree, neither is kept."""
    return choose_title(_pdf_title(data), _pdf_page_title(data))


def choose_title(meta: str | None, page: str | None) -> tuple[str | None, str | None]:
    """The PDF's title and its source from its two witnesses, `/Title` and the first page."""
    if meta and page:
        if difflib.SequenceMatcher(None, _norm_title(meta), _norm_title(page)).ratio() >= 0.9:
            return meta, "pdf_metadata_and_first_page"
        return None, None
    if meta:
        return meta, "pdf_metadata"
    if page:
        return page, "pdf_first_page"
    return None, None


def _pdf_created_year(data: bytes) -> int | None:
    """The year the PDF file was made (`/CreationDate`), or None.

    A year, never a day: the day is a fact about the file, not about when the paper was said."""
    try:
        # Imported at call time, like `_extract_pypdf`: a missing pypdf is a None, not a crash.
        from pypdf import PdfReader
        meta = PdfReader(io.BytesIO(data)).metadata
        raw = str((meta or {}).get("/CreationDate") or "")
    except Exception:
        return None
    return created_year(raw, date.today().year)


def created_year(raw: str, current_year: int) -> int | None:
    """The year in a PDF `/CreationDate` value (`D:YYYY…`), when it is 1990 to `current_year`."""
    m = re.match(r"D:(\d{4})", raw)
    year = int(m.group(1)) if m else None
    return year if year and 1990 <= year <= current_year else None


def _pdf_bytes_to_text(data: bytes) -> str | None:
    """PDF bytes → repaired text, or None when what came back is not a whole document.

    Two floors: `_MIN_FULLTEXT_CHARS` (enough text to be a body) and `_FLOOR_PER_PAGE` (the whole
    document, not part of it). Any extractor failure returns None, so the caller falls back to an
    abstract-only atom; never raises."""
    try:
        text, pages = _extract_pypdf(data)
    except Exception:
        return None
    return usable_text(text, pages)


def usable_text(raw: str, pages: int) -> str | None:
    """Extracted text, repaired, when it is long enough to be a whole document of `pages` pages."""
    text = _repair_extraction(raw).strip()
    if len(text) < _MIN_FULLTEXT_CHARS or len(text) < _FLOOR_PER_PAGE * pages:
        return None
    return text


@dataclass(frozen=True)
class PdfSources:
    """The open PDF urls to try, most reliable first, and which metadata the caller still lacks."""
    urls: tuple[str, ...]
    need_title: bool
    need_year: bool


@dataclass(frozen=True)
class Fulltext:
    """One PDF's whole-document text, and the title and year the file itself gave, when asked."""
    text: str
    title: str | None = None
    title_source: str | None = None
    year: int | None = None


def fulltext(sources: PdfSources) -> Fulltext | None:
    """The first source whose PDF downloads and extracts to a whole document, or None.

    No url, a failed download, a missing pypdf or a too-thin extraction all degrade to None; this
    never raises."""
    for url in sources.urls:
        data = _download_pdf(url)
        if not data:
            continue
        text = _pdf_bytes_to_text(data)
        if text:
            title, source = _title_from_pdf(data) if sources.need_title else (None, None)
            year = _pdf_created_year(data) if sources.need_year else None
            return Fulltext(text, title, source, year)
    return None
