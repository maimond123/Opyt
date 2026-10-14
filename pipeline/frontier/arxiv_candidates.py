"""
pipeline/frontier/arxiv_candidates.py — arXiv as a Frontier source: the windowed term search
(`pipeline/platforms/papers/arxiv.py`) mapped to `Candidate`s.
"""

from __future__ import annotations

from datetime import datetime

from pipeline.frontier.candidates import MAX_SUMMARY_CHARS, Candidate
from pipeline.platforms.papers import arxiv


class ArxivAdapter(arxiv.ArxivClient):
    """arXiv's windowed term search (`arxiv.ArxivClient`), as Candidates.

    Sorted by `submittedDate` descending so the first page is the newest — with a window applied,
    relevance ordering would bury the recent work the whole rail exists to catch.
    """
    slug = "arxiv"
    host = "export.arxiv.org"
    min_lookback_days = 0              # arXiv windows on submission date, which it indexes at once

    def search(self, query: str, *, since: datetime | None, now: datetime, limit: int = 25,
               context=()) -> list[Candidate]:
        del context  # a phrase in the title or abstract needs no narrowing
        return from_entries(self.search_entries(query, since=since, limit=limit, now=now))


def from_entries(entries: list[arxiv.ArxivEntry]) -> list[Candidate]:
    """arXiv entries → `Candidate`s, keyed on the version-stripped id."""
    return [Candidate(candidate_id=f"arxiv:{e.arxiv_id}",
                      source="arxiv",
                      kind="paper",
                      title=e.title,
                      url=f"https://arxiv.org/abs/{e.arxiv_id}",
                      published=e.published,
                      summary=e.summary[:MAX_SUMMARY_CHARS],
                      payload={"authors": [{"name": n} for n in e.authors],
                               "updated": e.updated})
            for e in entries]
