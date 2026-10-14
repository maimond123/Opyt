"""arXiv's Atom API: a windowed term search and an exact by-id lookup, as `ArxivEntry` records,
and the version-stripping that makes an arXiv id one paper. Transport and records only.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pipeline.platforms.papers import registry as paper_registry
from pipeline.platforms.papers.registry import (
    MAX_PAYLOAD_AUTHORS,
    BreakerBacked,
)
from pipeline.platforms.errors import SourceError

_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_VER = re.compile(r"v\d+$")
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class ArxivEntry:
    """One arXiv record, as the Atom feed gives it: the id version-stripped, title and summary
    whitespace-collapsed (the summary whole; each caller cuts it to its own length), the authors
    capped at `MAX_PAYLOAD_AUTHORS`, the dates cut to the day."""
    arxiv_id: str
    title: str
    summary: str
    authors: list[str] = field(default_factory=list)
    published: str | None = None
    updated: str = ""


class ArxivClient(BreakerBacked):
    """arXiv's Atom API, windowed, and its exact by-id lookup.

    The window is `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` ANDed with the terms, which is
    what makes an incremental pull possible. Sorted by `submittedDate` descending.
    """
    # arXiv enforces a hard 1-request-per-3s limit; violating it costs 429s.
    min_interval_s = 3.0
    # Pacing alone is not enough: once tripped, arXiv applies a timed block on the caller.
    breaker_host = "export.arxiv.org"

    def search_entries(self, query: str, *, since: datetime | None, now: datetime, limit: int = 25) -> list[ArxivEntry]:
        terms = _phrase_terms(query)
        if not terms:
            # A query that reduces to nothing searchable is a stage-1 bug. Failing loudly costs one
            # error row; falling through would send a bare `submittedDate:[...]` and re-pull the
            # entire window of arXiv under one query's name.
            raise SourceError(f"arxiv: no searchable terms in {query!r}")
        if since:
            until = now + timedelta(days=1)
            terms += (f" AND submittedDate:[{since.strftime('%Y%m%d%H%M')}"
                      f" TO {until.strftime('%Y%m%d%H%M')}]")
        url = (f"{_ARXIV_API}?search_query={urllib.parse.quote(terms)}"
               f"&start=0&max_results={int(limit)}&sortBy=submittedDate&sortOrder=descending")
        breaker = self._get_breaker()
        body = paper_registry.call_get(breaker, url)
        if body is None:
            raise SourceError("arxiv request failed")
        return parse_feed(body)

    def by_id(self, arxiv_id: str) -> ArxivEntry | None:
        """One paper by its arXiv id, or None if arXiv lists no such paper.

        An exact lookup (`id_list=`), not a search, so it is admissible as metadata where a title
        search is not. An open breaker, a 429 and a failed request raise `SourceError`."""
        url = f"{_ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}&max_results=1"
        body = paper_registry.call_get(self._get_breaker(), url)
        if body is None:
            raise SourceError("arxiv request failed")
        hits = parse_feed(body)
        return hits[0] if hits else None


def strip_arxiv_version(arxiv_id: str) -> str:
    """`2401.00001v2` → `2401.00001`, `hep-th/9901001v1` → `hep-th/9901001`: two versions of one
    paper are the same work, so they dedup to one atom and one candidate."""
    return _ARXIV_VER.sub("", (arxiv_id or "").strip())


def _phrase_terms(name: str) -> str:
    """The name as one phrase, in the title or the abstract. A watch names one thing (a model, a
    trial, a benchmark), and a phrase there is what keeps a quiet week from returning near-misses
    (measured)."""
    phrase = " ".join(re.sub(r'["\\()]', " ", name or "").split())
    return f'(ti:"{phrase}" OR abs:"{phrase}")' if phrase else ""


def parse_feed(body: bytes) -> list[ArxivEntry]:
    """An Atom feed → its entries."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        raise SourceError(f"arxiv returned unparseable XML: {e}") from None
    out = []
    for e in root.findall(f"{_ATOM_NS}entry"):
        raw_id = (e.findtext(f"{_ATOM_NS}id") or "").strip()
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        if not arxiv_id:
            continue
        out.append(ArxivEntry(
            arxiv_id=strip_arxiv_version(arxiv_id),
            title=" ".join((e.findtext(f"{_ATOM_NS}title") or "").split()),
            summary=" ".join((e.findtext(f"{_ATOM_NS}summary") or "").split()),
            authors=[n for a in e.findall(f"{_ATOM_NS}author")
                     if (n := (a.findtext(f"{_ATOM_NS}name") or "").strip())][:MAX_PAYLOAD_AUTHORS],
            published=(e.findtext(f"{_ATOM_NS}published") or "")[:10] or None,
            updated=(e.findtext(f"{_ATOM_NS}updated") or "")[:10]))
    return out
