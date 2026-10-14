"""OpenAlex author reads: `works` (newest first) and `year_counts` (one `group_by` call), and the
breaker the client shares with the Frontier works adapter.

These test the platform client.
Offline: the client's transport is monkeypatched with byte literals, and the breaker is injected.
"""
from __future__ import annotations

import json

from pipeline.frontier import openalex_candidates
from pipeline.platforms.papers import openalex
from pipeline.platforms.papers import registry as paper_registry


class _PassThrough:
    """A breaker that never trips — the adapter's transport is what these prove, not its backoff."""
    def call(self, fn):
        return fn()

    def allow(self):
        return True


_WORKS = json.dumps({"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/a",
     "title": "Directed  Evolution of Enzymes", "publication_date": "2026-08-01",
     "abstract_inverted_index": {"We": [0], "evolve": [1], "enzymes": [2]},
     "primary_location": {"source": {"display_name": "Nature"},
                          "landing_page_url": "https://nature.com/a"},
     "type": "article", "cited_by_count": 12},
    {"id": "https://openalex.org/W2", "title": "A Second Work", "publication_date": "2026-07-01",
     "abstract_inverted_index": None, "primary_location": {}, "type": "preprint"},
    {"id": "https://openalex.org/W3", "title": "", "abstract_inverted_index": None,
     "primary_location": {}},                       # no title AND no abstract → nothing to read
]}).encode()

_AUTHOR = json.dumps({
    "id": "https://openalex.org/A5043841592", "display_name": "Frances H. Arnold",
    "works_count": 928, "cited_by_count": 120000, "summary_stats": {"h_index": 150},
    "last_known_institutions": [{"display_name": "California Institute of Technology"}],
    # Present in the real response and DELIBERATELY unread — see `describe`.
    "topics": [{"display_name": "Atmospheric chemistry and aerosols"}],
}).encode()

_GROUP_BY = json.dumps({"group_by": [{"key": "2026", "count": 14}, {"key": "2025", "count": 14},
                                     {"key": "2024", "count": 30},
                                     {"key": "bad", "count": 1}]}).encode()


def _adapter(monkeypatch, body, seen=None):
    def _fake_get(url, *, headers=None):
        if seen is not None:
            seen["url"] = url
        return body(url) if callable(body) else body
    monkeypatch.setattr(paper_registry, "get", _fake_get)
    return openalex.OpenAlexClient(breaker=_PassThrough())


def test_recent_works_sorts_by_date_not_relevance(monkeypatch):
    """The opposite of the Frontier works adapter, on purpose. That one searches TERMS, so a work
    competes on match quality; here the filter already names exactly one person, so there is no
    match quality to rank on and recency is the only ordering that means anything."""
    seen = {}
    a = _adapter(monkeypatch, _WORKS, seen)

    works = a.works("A5043841592", limit=25)

    assert "filter=author.id%3AA5043841592" in seen["url"]
    assert "sort=publication_date%3Adesc" in seen["url"]
    assert "per-page=25" in seen["url"]
    assert len(works) == 3


def test_year_counts_returns_the_whole_distribution_in_one_call(monkeypatch):
    """This is what lets the lookback question carry real numbers instead of blind presets. A
    `group_by` returns every year bucket, not a page of them."""
    seen = {}
    a = _adapter(monkeypatch, _GROUP_BY, seen)

    assert a.year_counts("A5043841592") == {2026: 14, 2025: 14, 2024: 30}
    assert "group_by=publication_year" in seen["url"]


def test_the_openalex_breaker_is_shared_with_the_frontier_works_adapter():
    """One host, one budget. Two paths against `api.openalex.org` with independent breakers would
    hammer it, and the anonymous allowance is per IP and resets only at midnight UTC — so a burned
    day stays burned. `CircuitBreaker` is keyed by this string in a persisted table."""
    assert openalex.OpenAlexClient.breaker_host == openalex_candidates.OpenAlexAdapter.breaker_host

