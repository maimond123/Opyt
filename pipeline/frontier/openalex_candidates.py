"""
pipeline/frontier/openalex_candidates.py — OpenAlex as a Frontier source: the works search
(`pipeline/platforms/papers/openalex.py`) mapped to `Candidate`s.
"""

from __future__ import annotations

from datetime import datetime

from pipeline.frontier import cadence
from pipeline.frontier.candidates import MAX_SUMMARY_CHARS, Candidate
from pipeline.platforms.papers import openalex
from pipeline.platforms.papers.registry import BreakerBacked


class OpenAlexAdapter(BreakerBacked):
    """OpenAlex works whose title or abstract holds the watch's name, in the order the window
    deserves (`sort_for`): newest first for a pull's window, most cited for a long look.

    The free tier windows only on publication date and OpenAlex indexes late, so the window
    reaches back `min_lookback_days` to catch work published before it and indexed since; the
    repeats are already-seen candidates. The anonymous allowance (~100 requests a day, per IP)
    resets at midnight UTC; the persisted breaker keeps a failing day from burning it.
    """
    slug = "openalex"
    host = openalex.BREAKER_HOST
    min_interval_s = openalex.MIN_INTERVAL_S
    min_lookback_days = 30
    breaker_host = openalex.BREAKER_HOST

    def search(self, query: str, *, since: datetime | None, now: datetime, limit: int = 25,
               context=()) -> list[Candidate]:
        del context  # OpenAlex windows only on the lower publication bound.
        days = None if since is None else (now - since).total_seconds() / 86400.0
        works = openalex.search_works(query, since=since, limit=limit, sort=sort_for(days),
                                      get_breaker=self._get_breaker)
        return from_works(works)


def sort_for(days: float | None) -> str:
    """The order to ask a window's works in: most cited for a landscape
    (`cadence.landscape`), newest first for a pull's window. This is the one source with a
    quality signal to sort on, so it is the only one whose order the window changes.
    """
    return openalex.MOST_CITED if cadence.landscape(days) else openalex.NEWEST


def from_works(works: list[dict]) -> list[Candidate]:
    """An OpenAlex works page → `Candidate`s. A work with no offline route to an atom id (no id,
    or neither a DOI nor a landing page) is not staged."""
    out = []
    for w in works:
        work_id = str(w.get("id") or "").rsplit("/", 1)[-1]
        rec = openalex.work_record(w)
        # `rec.url` is the DOI (an arXiv DOI folds back onto its arXiv id), else the landing page.
        if not work_id or not rec.url:
            continue
        out.append(Candidate(
            candidate_id=f"openalex:{work_id}",
            source="openalex",
            kind="paper",
            title=rec.title,
            url=rec.url,
            published=rec.published,
            summary=rec.abstract[:MAX_SUMMARY_CHARS],
            payload={"authors": openalex.openalex_authors(w),
                     "venue": rec.venue, "type": w.get("type"),
                     "cited_by_count": w.get("cited_by_count"),
                     # The open PDF, so the minter can build a full-document atom.
                     "pdf_url": rec.pdf_url}))
    return out
