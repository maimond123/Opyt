"""Frontier stage 2 adapters — the transport contracts, proven offline.

The arXiv breaker is the load-bearing one. Pacing alone was measured insufficient: a run firing 22
requests in 12 seconds earned a timed block, after which a fresh probe got 0 of 5 at a 3-second gap
AND 0 of 5 at 5 seconds. Once blocked, the only useful behaviour is to stop asking.
"""
from __future__ import annotations

import json
import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import frontier_sources as fs

_NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)

_ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
 <entry>
  <id>http://arxiv.org/abs/2508.01234v2</id>
  <title>Gated  DeltaNet
   ablations</title>
  <summary>A study.</summary>
  <published>2026-08-09T00:00:00Z</published>
  <updated>2026-08-09T00:00:00Z</updated>
  <author><name>A. Person</name></author>
 </entry>
</feed>"""


def test_arxiv_windows_the_query_and_sorts_by_date(monkeypatch):
    """Without a date window every run re-ranks the same all-time results and "what is new since
    Tuesday" is unanswerable."""
    seen = {}
    monkeypatch.setattr(fs, "_get", lambda url, **kw: seen.setdefault("url", url) and None or _ATOM)
    fs.ArxivAdapter(breaker=_PassThrough()).search("gated DeltaNet", since=_NOW - timedelta(days=7))
    assert "submittedDate%3A%5B20260803" in seen["url"]
    assert "sortBy=submittedDate&sortOrder=descending" in seen["url"]


def test_arxiv_ands_bare_terms_and_never_phrase_quotes(monkeypatch):
    """The regression that mattered. Phrase-quoting made arXiv demand the words consecutively, and
    stage 1 emits topic DESCRIPTORS, not title strings — so `data constrained scaling laws` matched
    ZERO papers in the entire archive while the ANDed form matched 3 in a 14-day window (probed
    2026-08-12). Measured effect: 24 of 26 arXiv pairs recorded `empty`, and GitHub — which passes
    terms bare — supplied 85 of the 111 candidates. One character of punctuation, silently."""
    seen = {}
    monkeypatch.setattr(fs, "_get", lambda url, **kw: seen.setdefault("url", url) and None or _ATOM)
    fs.ArxivAdapter(breaker=_PassThrough()).search("data constrained scaling laws", since=None)
    assert "%22" not in seen["url"], "a quoted phrase is unanswerable for a descriptor query"
    assert "all%3Adata%20AND%20all%3Aconstrained" in seen["url"]


def test_arxiv_refuses_a_query_with_no_searchable_terms(monkeypatch):
    """Fail-safe. Falling through would send a bare `submittedDate:[...]` and re-pull the whole
    window of arXiv under one query's name; a SourceError costs one un-stamped error row."""
    calls = []
    monkeypatch.setattr(fs, "_get", lambda url, **kw: calls.append(url) or _ATOM)
    with pytest.raises(fs.SourceError, match="no searchable terms"):
        fs.ArxivAdapter(breaker=_PassThrough()).search("AND OR", since=_NOW)
    assert calls == []


def test_arxiv_strips_the_version_from_the_id(monkeypatch):
    """v1 and v2 are the same artifact; keying on the versioned id re-surfaces every revision as a
    brand-new candidate."""
    monkeypatch.setattr(fs, "_get", lambda url, **kw: _ATOM)
    c = fs.ArxivAdapter(breaker=_PassThrough()).search("q", since=None)[0]
    assert c.candidate_id == "arxiv:2508.01234" and c.url.endswith("/abs/2508.01234")
    assert c.title == "Gated DeltaNet ablations"        # whitespace collapsed
    assert c.published == "2026-08-09"


def test_a_429_becomes_a_rate_limit_not_a_silent_none():
    """A 404 is about one request; a 429 is about us, and the two need opposite responses."""
    import urllib.error
    def boom(*a, **kw):
        raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)
    import urllib.request
    orig = urllib.request.urlopen
    urllib.request.urlopen = boom
    try:
        with pytest.raises(fs.RateLimited):
            fs._get("http://export.arxiv.org/x")
    finally:
        urllib.request.urlopen = orig


def test_an_open_breaker_fails_the_pair_without_touching_the_network(monkeypatch):
    """Once arXiv has blocked us, sending 21 more requests only deepens the hole."""
    calls = []
    monkeypatch.setattr(fs, "_get", lambda url, **kw: calls.append(url) or _ATOM)
    with pytest.raises(fs.SourceError, match="breaker open"):
        fs.ArxivAdapter(breaker=_Open()).search("q", since=None)
    assert calls == []


def test_github_windows_on_pushed_not_created(monkeypatch):
    """The rail wants a thread that MOVED. A created-date filter hides the two-year-old repo that
    shipped something yesterday, which is exactly the signal."""
    seen = {}

    class C:
        def search_repos(self, q, limit=10, sort="stars"):
            seen.update(q=q, sort=sort)
            return [{"full_name": "o/n", "html_url": "https://github.com/o/n",
                     "pushed_at": "2026-08-09", "description": "d", "stars": 5}]
    c = fs.GitHubAdapter(client=C()).search("kernel", since=_NOW - timedelta(days=3))[0]
    assert "pushed:>2026-08-07" in seen["q"]
    assert c.candidate_id == "repo:o/n" and c.published == "2026-08-09"


def test_github_does_not_sort_on_the_axis_it_filtered_on():
    """`pushed:>DATE` already selects on recency, so `sort=updated` would rank the window by the
    thing the window guarantees — an ordering carrying no information about quality. Measured on
    the first live run: 64 of 73 repos under five stars. Pinned as a test because the fix is one
    keyword and reads as a cosmetic default."""
    seen = {}

    class C:
        def search_repos(self, q, limit=10, sort="stars"):
            seen["sort"] = sort
            return []
    fs.GitHubAdapter(client=C()).search("kernel", since=_NOW - timedelta(days=3))
    assert seen["sort"] == "stars"


def test_only_built_adapters_are_registered():
    """An unbuilt source must be ABSENT rather than stubbed — the loop reports a missing adapter as
    `no_adapter` and counts it, which is honest; a stub returning [] would look like a working
    source that found nothing. The second assertion is the one that would catch a real mistake: an
    adapter no reader can route to is unreachable, and nothing else in the tree says so."""
    from pipeline.kb.reader_core import VALID_SOURCES
    assert set(fs.adapters()) == {"arxiv", "github", "openalex"}
    assert set(fs.adapters()) <= VALID_SOURCES


class _PassThrough:
    def call(self, fn, **kw):
        return fn()


class _Open:
    def call(self, fn, **kw):
        from pipeline.circuit_breaker import CircuitOpenError
        raise CircuitOpenError("cooling down", 900.0)


# ── OpenAlex ────────────────────────────────────────────────────────────────────
# One work of each shape the parser has to tell apart: an arXiv preprint reached by its arXiv DOI,
# a work whose only route is its landing page, and a work with neither.
_OA_BODY = b"""{"results": [
 {"id": "https://openalex.org/W7202193881",
  "doi": "https://doi.org/10.48550/arxiv.2608.09055",
  "title": "Repeated-Game  Security",
  "publication_date": "2026-08-10",
  "abstract_inverted_index": {"restaking": [2], "We": [0], "study": [1]},
  "authorships": [{"author": {"id": "https://openalex.org/A5123233691",
                              "display_name": "Zhenhang Shang"}}],
  "primary_location": {"landing_page_url": "https://arxiv.org/abs/2608.09055",
                       "source": {"display_name": "arXiv (Cornell University)"}},
  "type": "preprint", "cited_by_count": 0, "relevance_score": 34.2},
 {"id": "https://openalex.org/W7164878913", "doi": null, "title": "No DOI, has a page",
  "publication_date": "2026-08-14", "abstract_inverted_index": null, "authorships": [],
  "primary_location": {"landing_page_url": "https://arxiv.org/abs/2608.01755", "source": null},
  "type": "preprint", "cited_by_count": 3},
 {"id": "https://openalex.org/W1", "doi": null, "title": "Unreachable",
  "publication_date": "2026-08-14", "primary_location": {}, "authorships": []}
]}"""


def test_openalex_windows_on_publication_date_and_sorts_by_relevance(monkeypatch):
    """The load-bearing decision in this adapter, asserted directly because both halves of it are
    forced by the same paywall and neither is obvious from the code alone.

    Filtering AND sorting on index date are Premium-only (both return HTTP 429 "Plan upgrade
    required", verified 2026-08-26), so the free tier can only window on PUBLICATION date — and
    OpenAlex indexes late: measured over 400 works in two settled one-week windows, 8-9% are
    indexed more than 7 days after publication. That forces a window far wider than the cursor,
    and a wide window sorted by DATE would return the same newest page every run, leaving the
    late-indexed tail — the entire reason the window is wide — permanently unreachable.
    """
    seen = {}
    monkeypatch.setattr(fs, "_get", lambda url, **kw: seen.setdefault("url", url) and None
                        or _OA_BODY)
    fs.OpenAlexAdapter(breaker=_PassThrough()).search("verifiable inference",
                                                      since=_NOW - timedelta(days=30))
    assert "from_publication_date%3A2026-07-11" in seen["url"]
    assert "sort=relevance_score%3Adesc" in seen["url"]
    assert "publication_date%3Adesc" not in seen["url"]
    # And the select asks for the score itself — `frontier_execute._relevance_cut` reads it, so
    # dropping it from the select would silently disable the intake cut (every payload None).
    assert "relevance_score" in urllib.parse.unquote(seen["url"]).split("select=")[1].split("&")[0]


def test_openalex_declares_a_lookback_the_loop_can_honor():
    """Declared here and applied in `frontier_execute._lookback_floor`, so `window_ok` validates
    the window that is actually sent. An adapter that widened its own `since` silently is the
    exact bug that assertion exists to catch."""
    assert fs.OpenAlexAdapter().min_lookback_days == 30


def test_openalex_folds_an_arxiv_doi_onto_the_arxiv_atom():
    """OpenAlex indexes arXiv heavily, so the same preprint arrives here by DOI and via the arXiv
    adapter by /abs/ page. Both must reach ONE atom — the fold lives in `_parse_paper_url`, and
    this pins that the adapter hands it a url that fold can see."""
    from pipeline.kb.frontier_admit import target_atom_id
    c = fs._parse_openalex(_OA_BODY)[0]
    assert c.kind == "paper" and c.source == "openalex"
    assert target_atom_id(c.kind, c.url)[0] == "paper:arXiv:2608.09055"
    assert target_atom_id("paper", "https://arxiv.org/abs/2608.09055")[0] == "paper:arXiv:2608.09055"


def test_openalex_rebuilds_the_abstract_and_carries_its_authors():
    """The abstract is shipped as a `{word: [positions]}` index, and it is not decoration: its
    LENGTH is the substance signal stage 4 ranks papers by, and it is the body an admitted atom
    gets when no open PDF resolves."""
    c = fs._parse_openalex(_OA_BODY)[0]
    assert c.summary == "We study restaking"
    assert c.title == "Repeated-Game Security"          # whitespace collapsed
    assert c.payload["authors"] == ["Zhenhang Shang"]
    assert c.published == "2026-08-10"
    # The score rides the payload for `frontier_execute._relevance_cut`; a work the API returns
    # without one carries None, which the cut reads as NOT COMPARABLE rather than zero.
    assert c.payload["relevance_score"] == 34.2
    assert fs._parse_openalex(_OA_BODY)[1].payload["relevance_score"] is None


def test_a_work_with_no_offline_route_to_an_atom_is_not_staged():
    """Criterion 3 of the rail: an artifact whose url no minter can parse offline can only ever be
    a `rejected`/`no_atom_id` row. The landing page rescues the DOI-less work; nothing rescues the
    one with neither, so it is not staged at all."""
    cands = fs._parse_openalex(_OA_BODY)
    assert [c.candidate_id for c in cands] == ["openalex:W7202193881", "openalex:W7164878913"]
    assert cands[1].url == "https://arxiv.org/abs/2608.01755"


def test_openalex_refuses_a_query_with_no_searchable_terms(monkeypatch):
    """Same fail-safe as arXiv: a bare filter with no terms re-pulls the whole window under one
    query's name, and costs 10 of a ~100-request daily allowance to do it."""
    calls = []
    monkeypatch.setattr(fs, "_get", lambda url, **kw: calls.append(url) or _OA_BODY)
    with pytest.raises(fs.SourceError, match="no searchable terms"):
        fs.OpenAlexAdapter(breaker=_PassThrough()).search('  ""  ', since=_NOW)
    assert calls == []


def test_an_open_openalex_breaker_costs_no_request(monkeypatch):
    """The breaker is persisted for a reason specific to this loop: a failed pull deliberately does
    not stamp, so a failing source is due again on the very next spawn. Unbounded, that is what
    burns the daily allowance."""
    calls = []
    monkeypatch.setattr(fs, "_get", lambda url, **kw: calls.append(url) or _OA_BODY)
    a = fs.OpenAlexAdapter(breaker=_Open())
    assert a.available() is True or a.available() is False   # never raises
    with pytest.raises(fs.SourceError, match="breaker open"):
        a.search("verifiable inference", since=_NOW)
    assert calls == []


def test_each_adapter_declares_the_kind_its_candidates_mint_as():
    """`source` is the finder and `kind` is the minter, and stage 3 dispatches on the second. An
    adapter that omits `kind` would stage artifacts nothing can materialize — so the dataclass
    gives it no default and this pins what the built adapters declare."""
    with pytest.raises(TypeError):                  # no default — forgetting it cannot compile
        fs.Candidate(candidate_id="x", source="s", title="t", url="u")
    assert {c.kind for c in fs._parse_openalex(_OA_BODY)} == {"paper"}
    assert {c.kind for c in fs._parse_arxiv(_ATOM)} == {"paper"}


def test_openalex_carries_the_open_pdf_url():
    """The full-text seam. `_fulltext_pdf_urls` already reads `openAccessPdf.url`, so capturing
    the OA pdf here is the whole feature — no change to the minter, the merge, or the resolver.

    `best_oa_location` wins over `primary_location` because the primary location can be the
    PAYWALLED publisher copy while best_oa is OpenAlex's own "best open version". Measured over
    68 live results 2026-08-26: 47 carry a pdf here, and 25 of those are works `_fulltext_pdf_urls`
    could not otherwise reach (its list is arXiv's mirror then S2's, and S2 does not index them).
    """
    body = json.dumps({"results": [
        {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.5281/zenodo.1",
         "title": "Open Access Work", "publication_date": "2026-08-01",
         "primary_location": {"landing_page_url": "https://zenodo.org/records/1",
                              "pdf_url": "https://example.test/paywalled.pdf"},
         "best_oa_location": {"pdf_url": "https://example.test/open.pdf"}},
        {"id": "https://openalex.org/W2", "doi": "https://doi.org/10.5281/zenodo.2",
         "title": "Primary Only", "publication_date": "2026-08-01",
         "primary_location": {"pdf_url": "https://example.test/primary.pdf"}},
        {"id": "https://openalex.org/W3", "doi": "https://doi.org/10.5281/zenodo.3",
         "title": "No Open Pdf Anywhere", "publication_date": "2026-08-01",
         "primary_location": {"landing_page_url": "https://zenodo.org/records/3"}},
    ]}).encode()
    got = {c.title: c.payload.get("pdf_url") for c in fs._parse_openalex(body)}
    assert got["Open Access Work"] == "https://example.test/open.pdf"     # best_oa wins
    assert got["Primary Only"] == "https://example.test/primary.pdf"      # falls back
    # 14 of 39 non-arXiv works have no open pdf at all. None, not "", so `_known_metadata` omits
    # the key entirely rather than handing the minter an empty url to try.
    assert got["No Open Pdf Anywhere"] is None
