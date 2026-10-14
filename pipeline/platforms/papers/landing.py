"""A report's landing page → what it declares about its paper, in Semantic Scholar's field names.

For report archives that mint neither a DOI nor an arXiv id (IACR ePrint, ECCC), so no registry
can be asked about the paper by id. Their pages carry the `citation_*` meta tags Google Scholar
reads (title, every author, date, venue, PDF) and print the abstract under an "Abstract" label,
which no tag carries. Measured on `eprint.iacr.org/2025/2041` and
`eccc.weizmann.ac.il/report/2026/237/`.

Metadata only, never identity: `ingest_papers._parse_paper_url` keys the paper and names the page.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

import requests

from pipeline.host_access import request as host_request

_UA = {"User-Agent": "opyt-paper-adapter/1.0"}
_TIMEOUT = 30
_ABSTRACT_CHARS = 2000      # the same cut as `lookup`'s OpenAlex and arXiv abstracts


class _Page(HTMLParser):
    """Collects the page's `citation_*` meta tags, and the paragraphs that follow its "Abstract"
    label: the first `<p>` after the label, and each `<p>` that directly follows that one."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: dict[str, list[str]] = {}
        self.abstract: list[str] = []
        self._at = "before"     # before → label → in_p → after_p → done
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            name = (a.get("name") or "").lower()
            if name.startswith("citation_") and (a.get("content") or "").strip():
                self.tags.setdefault(name, []).append(a["content"].strip())
        elif tag == "p" and self._at in ("label", "after_p"):
            self._at, self._text = "in_p", []
        elif self._at == "after_p":
            self._at = "done"

    def handle_endtag(self, tag):
        if tag == "p" and self._at == "in_p":
            self.abstract.append("".join(self._text))
            self._at = "after_p"

    def handle_data(self, data):
        if self._at == "in_p":
            self._text.append(data)
        elif self._at == "before" and data.strip().rstrip(":").lower() == "abstract":
            self._at = "label"


def _paragraphs(blocks: list[str]) -> str:
    """The abstract's paragraphs, each on one line. ePrint prints its paragraphs inside one `<p>`
    split by blank lines; ECCC gives each its own `<p>`."""
    paras = [" ".join(part.split()) for block in blocks for part in re.split(r"\n\s*\n", block)]
    return "\n\n".join(p for p in paras if p)


def declared(page: str) -> dict:
    """What a landing page declares about its paper: title, authors, abstract, year, date, venue
    and open PDF, in S2 field names, leaving out what it does not say. Pure."""
    parsed = _Page()
    parsed.feed(page)
    tags = parsed.tags

    def first(name: str) -> str:
        return (tags.get(name) or [""])[0]

    date = first("citation_publication_date")    # ePrint gives the year alone, ECCC the day
    out = {
        "title": " ".join(first("citation_title").split()),
        "authors": [{"name": name} for name in tags.get("citation_author", [])],
        "abstract": _paragraphs(parsed.abstract)[:_ABSTRACT_CHARS],
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "publicationDate": date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else "",
        "venue": first("citation_journal_title") or first("citation_technical_report_institution"),
        "openAccessPdf": {"url": pdf} if (pdf := first("citation_pdf_url")) else None,
    }
    return {k: v for k, v in out.items() if v}


def metadata(url: str) -> dict | None:
    """The landing page at `url`, read once → `declared`, or None when it could not be read or
    declares nothing. Never raises."""
    try:
        resp = host_request(requests.get, url, timeout=_TIMEOUT, headers=_UA)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    return declared(resp.text) or None
