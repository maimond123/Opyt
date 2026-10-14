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

from pipeline.frontier import (adapters, arxiv_candidates, candidates, github_candidates,
                               openalex_candidates)
from pipeline.platforms.papers import openalex
from pipeline.platforms.papers import arxiv
from pipeline.platforms.papers import registry as paper_registry
from pipeline.platforms.errors import SourceError

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
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: seen.setdefault("url", url) and None or _ATOM)
    arxiv_candidates.ArxivAdapter(breaker=_PassThrough()).search("gated DeltaNet", now=_NOW, since=_NOW - timedelta(days=7))
    assert "submittedDate%3A%5B20260803" in seen["url"]
    assert "TO%20202608110000" in seen["url"]
    assert "sortBy=submittedDate&sortOrder=descending" in seen["url"]


def test_arxiv_searches_the_name_as_a_phrase_in_title_or_abstract(monkeypatch):
    """A watch names one thing. Searched as bare ANDed terms over every field, a name brushed by a
    citation in a paper's body counts; as a phrase in the title or abstract, a quiet week returns
    nothing."""
    seen = {}
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: seen.setdefault("url", url) and None or _ATOM)
    arxiv_candidates.ArxivAdapter(breaker=_PassThrough()).search("Virtual Cell Challenge", now=_NOW, since=None)
    query = urllib.parse.unquote(seen["url"])
    assert '(ti:"Virtual Cell Challenge" OR abs:"Virtual Cell Challenge")' in query
    assert "all:" not in query


def test_arxiv_refuses_a_query_with_no_searchable_terms(monkeypatch):
    """Fail-safe. Falling through would send a bare `submittedDate:[...]` and re-pull the whole
    window of arXiv under one query's name; a SourceError costs one un-stamped error row."""
    calls = []
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: calls.append(url) or _ATOM)
    with pytest.raises(SourceError, match="no searchable terms"):
        arxiv_candidates.ArxivAdapter(breaker=_PassThrough()).search('"()"', now=_NOW, since=_NOW)
    assert calls == []


def test_arxiv_strips_the_version_from_the_id(monkeypatch):
    """v1 and v2 are the same artifact; keying on the versioned id re-surfaces every revision as a
    brand-new candidate."""
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: _ATOM)
    c = arxiv_candidates.ArxivAdapter(breaker=_PassThrough()).search("q", now=_NOW, since=None)[0]
    assert c.candidate_id == "arxiv:2508.01234" and c.url.endswith("/abs/2508.01234")
    assert c.title == "Gated DeltaNet ablations"        # whitespace collapsed
    assert c.published == "2026-08-09"


def test_arxiv_by_id_is_an_exact_lookup(monkeypatch):
    """The paper path's fallback when Semantic Scholar does not answer: arXiv's own record of the
    id the url named. `id_list`, never a search — a title search can return a different paper,
    and a paper atom is permanent."""
    seen = []
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: seen.append(url) or _ATOM)
    c = arxiv.ArxivClient(breaker=_PassThrough()).by_id("2508.01234")
    assert "id_list=2508.01234" in seen[0] and "search_query" not in seen[0]
    assert c.arxiv_id == "2508.01234" and c.title == "Gated DeltaNet ablations"


def test_arxiv_by_id_with_no_such_paper_is_none(monkeypatch):
    """arXiv answers an unknown id with an empty feed (verified on 2202.99999)."""
    empty = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: empty)
    assert arxiv.ArxivClient(breaker=_PassThrough()).by_id("2202.99999") is None


def test_arxiv_by_id_failed_request_refuses(monkeypatch):
    """A failed request is not "no such paper"; it raises, through the breaker."""
    def offline(*_a, **_kw):
        raise ConnectionError("offline")

    monkeypatch.setattr(paper_registry, "host_request", offline)
    monkeypatch.setattr(paper_registry, "log", lambda _msg: None)
    with pytest.raises(SourceError, match="request failed"):
        arxiv.ArxivClient(breaker=_PassThrough()).by_id("2202.99999")


def test_arxiv_by_id_open_breaker_and_429_raise_source_error(monkeypatch):
    """by_id translates CircuitOpenError and RateLimited into SourceError."""
    from types import SimpleNamespace
    with pytest.raises(SourceError, match="breaker open"):
        arxiv.ArxivClient(breaker=_Open()).by_id("2202.99999")
    monkeypatch.setattr(paper_registry, "host_request",
                        lambda *_a, **_kw: SimpleNamespace(status_code=429))
    with pytest.raises(SourceError, match="429"):
        arxiv.ArxivClient(breaker=_PassThrough()).by_id("2202.99999")


def test_a_429_becomes_a_rate_limit_not_a_silent_none(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(paper_registry.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=429))
    with pytest.raises(paper_registry.RateLimited):
        paper_registry.get("http://export.arxiv.org/x")


def test_an_open_breaker_fails_the_pair_without_touching_the_network(monkeypatch):
    """Once arXiv has blocked us, sending 21 more requests only deepens the hole."""
    calls = []
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: calls.append(url) or _ATOM)
    with pytest.raises(SourceError, match="breaker open"):
        arxiv_candidates.ArxivAdapter(breaker=_Open()).search("q", now=_NOW, since=None)
    assert calls == []


def test_github_finds_repos_created_since_that_name_the_watch(monkeypatch):
    """New means created: an old repo with one fresh commit is not new. The
    name is a phrase in the repo's name or description."""
    seen = {}

    class C:
        def search_repos(self, q, limit=10, sort="stars"):
            seen.update(q=q, sort=sort)
            return [{"full_name": "o/n", "html_url": "https://github.com/o/n",
                     "created_at": "2026-08-09", "pushed_at": "2026-08-10", "description": "d",
                     "stars": 5}]
    c = github_candidates.GitHubAdapter(client=C()).search("OpenFold3", now=_NOW, since=_NOW - timedelta(days=3))[0]
    assert seen["q"] == '"OpenFold3" in:name,description created:>2026-08-07'
    assert c.candidate_id == "repo:o/n" and c.published == "2026-08-09"


def test_the_discovery_client_owes_exactly_one_method():
    """Stage 2 asks its GitHub client for exactly one thing. Its per-repo detail reads (`repo`,
    `readme`) had no caller, so the client is a plain object, not an ABC. A plain object implementing only `search_repos`
    must keep working as the adapter's client, which is what breaks the moment somebody makes the adapter need more."""

    class OnlySearch:
        def search_repos(self, q, limit=10, sort="stars"):
            return [{"full_name": "o/n", "html_url": "https://github.com/o/n",
                     "pushed_at": "2026-08-09", "description": "d", "stars": 5}]

    (c,) = github_candidates.GitHubAdapter(client=OnlySearch()).search("kernel", now=_NOW, since=None)
    assert c.candidate_id == "repo:o/n" and c.kind == "repo"


def test_github_does_not_sort_on_the_axis_it_filtered_on():
    """`created:>DATE` already selects on recency, so `sort=updated` would rank the window by the
    thing the window guarantees — an ordering carrying no information about quality. Measured on
    the first live run: 64 of 73 repos under five stars. Pinned as a test because the fix is one
    keyword and reads as a cosmetic default."""
    seen = {}

    class C:
        def search_repos(self, q, limit=10, sort="stars"):
            seen["sort"] = sort
            return []
    github_candidates.GitHubAdapter(client=C()).search("kernel", now=_NOW, since=_NOW - timedelta(days=3))
    assert seen["sort"] == "stars"


def test_github_is_paced_for_the_ANONYMOUS_search_budget():
    """`/search/repositories` allows 10 requests/minute unauthenticated and 30 with a token
    (docs.github.com/en/rest/search/search). Most installs have no token — `GITHUB_TOKEN` is set
    only by `opyt-keys` — so the pacing has to hold at the anonymous number, which is one request
    per 6 seconds.

    A zero delay, justified as "token'd search is 30/min", would be wrong twice over. The token it
    assumes is one nothing in the product can acquire. And
    `MAX_REQUESTS_PER_RUN` is 40, so 40 unpaced requests exceed the TOKENED limit as well — a key
    would have raised the ceiling and left the adapter over it either way.

    Asserted as a FLOOR, not as `== 6.0`: raising the interval is always safe, and pinning the
    literal would make a more conservative value fail for no reason. What must never happen is
    dropping back under the anonymous budget."""
    per_minute_anonymous = 10
    assert github_candidates.GitHubAdapter.min_interval_s >= 60 / per_minute_anonymous


def test_only_built_adapters_are_registered():
    """An unbuilt source must be ABSENT rather than stubbed — the loop reports a missing adapter as
    `no_adapter` and counts it, which is honest; a stub returning [] would look like a working
    source that found nothing."""
    assert set(adapters.adapters()) == {"arxiv", "github", "openalex", "x", "substack"}


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
                              "display_name": "Zhenhang Shang",
                              "orcid": "https://orcid.org/0000-0002-4027-364X"},
                   "author_position": "first"},
                  {"author": {"id": "https://openalex.org/A5000000002",
                              "display_name": "Raw Orcid Only"},
                   "raw_orcid": "https://orcid.org/0000-0001-1111-2222"},
                  {"author": {"display_name": "No Id Here"}}],
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


def test_openalex_filters_on_the_name_in_title_and_abstract_newest_first(monkeypatch):
    """Filtering on index date is Premium-only (HTTP 429 "Plan upgrade required", verified
    with a user's key), so the free tier windows on PUBLICATION
    date, and OpenAlex indexes late: the window is far wider than the cursor. Full-text `search`
    always filled its page with near-misses; a phrase filter on title and abstract returns only
    works that name the thing, few enough that newest-first misses nothing on one page."""
    seen = {}
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: seen.setdefault("url", url) and None
                        or _OA_BODY)
    openalex_candidates.OpenAlexAdapter(breaker=_PassThrough()).search("Tahoe-100M",
                                                      now=_NOW, since=_NOW - timedelta(days=30))
    query = urllib.parse.unquote(seen["url"])
    assert 'filter=title_and_abstract.search:"Tahoe-100M",from_publication_date:2026-07-11' in query
    assert "sort=publication_date:desc" in query
    assert "search=" not in query.replace("title_and_abstract.search", "")


def test_a_long_window_asks_openalex_for_the_most_cited_work_not_the_newest(monkeypatch):
    """The newest 25 matches of a five-year window are a recency slice of minor work, not the
    field. Inside 90 days the opposite: every match has ~0 citations, so
    citations-first would return them tied, in an order OpenAlex does not define."""
    seen = {}
    monkeypatch.setattr(paper_registry, "get",
                        lambda url, **kw: seen.setdefault("url", url) and None or _OA_BODY)
    openalex_candidates.OpenAlexAdapter(breaker=_PassThrough()).search(
        "morpholine", now=_NOW, since=_NOW - timedelta(days=1825))
    assert "sort=cited_by_count:desc" in urllib.parse.unquote(seen["url"])


@pytest.mark.parametrize("days, order", [
    (None, openalex.NEWEST), (7, openalex.NEWEST), (90, openalex.NEWEST),
    (91, openalex.MOST_CITED), (1825, openalex.MOST_CITED)])
def test_the_window_decides_the_order_openalex_is_asked_in(days, order):
    assert openalex_candidates.sort_for(days) == order


def test_openalex_declares_a_lookback_the_loop_can_honor():
    """Declared here and applied in `cadence.lookback_floor`, so `window_ok` validates
    the window that is actually sent. An adapter that widened its own `since` silently is the
    exact bug that assertion exists to catch."""
    assert openalex_candidates.OpenAlexAdapter().min_lookback_days == 30


def test_openalex_folds_an_arxiv_doi_onto_the_arxiv_atom():
    """OpenAlex indexes arXiv heavily, so the same preprint arrives here by DOI and via the arXiv
    adapter by /abs/ page. Both must reach ONE atom — the fold lives in `_parse_paper_url`, and
    this pins that the adapter hands it a url that fold can see."""
    from pipeline.frontier.admit import target_atom_id
    c = openalex_candidates.from_works(openalex.parse_works(_OA_BODY))[0]
    assert c.kind == "paper" and c.source == "openalex"
    assert target_atom_id(c.kind, c.url)[0] == "paper:arXiv:2608.09055"
    assert target_atom_id("paper", "https://arxiv.org/abs/2608.09055")[0] == "paper:arXiv:2608.09055"


def test_openalex_rebuilds_the_abstract_and_carries_its_authors():
    """The abstract is shipped as a `{word: [positions]}` index, and it is not decoration: its
    LENGTH is the substance signal stage 4 ranks papers by, and it is the body an admitted atom
    gets when no open PDF resolves."""
    c = openalex_candidates.from_works(openalex.parse_works(_OA_BODY))[0]
    assert c.summary == "We study restaking"
    assert c.title == "Repeated-Game Security"          # whitespace collapsed
    # The ORCID rides free in this same response (73% of 256 measured authorships carry one) and
    # is the ONLY key an `openalex:` entity can safely merge to a `scholar:` one on. Stored bare.
    assert c.payload["authors"] == [
        {"name": "Zhenhang Shang", "openalex_id": "A5123233691",
         "orcid": "0000-0002-4027-364X", "position": "first"},
        {"name": "Raw Orcid Only", "openalex_id": "A5000000002",
         "orcid": "0000-0001-1111-2222"},               # `raw_orcid` is the fallback source
        {"name": "No Id Here"}]                         # falsy keys omitted, never stored as null
    assert c.published == "2026-08-10"


def test_a_work_with_no_offline_route_to_an_atom_is_not_staged():
    """Criterion 3 of the rail: an artifact whose url no minter can parse offline can only ever be
    a `rejected`/`no_atom_id` row. The landing page rescues the DOI-less work; nothing rescues the
    one with neither, so it is not staged at all."""
    cands = openalex_candidates.from_works(openalex.parse_works(_OA_BODY))
    assert [c.candidate_id for c in cands] == ["openalex:W7202193881", "openalex:W7164878913"]
    assert cands[1].url == "https://arxiv.org/abs/2608.01755"


def test_openalex_refuses_a_query_with_no_searchable_terms(monkeypatch):
    """Same fail-safe as arXiv: a bare filter with no terms re-pulls the whole window under one
    query's name, and costs 10 of a ~100-request daily allowance to do it."""
    calls = []
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: calls.append(url) or _OA_BODY)
    with pytest.raises(SourceError, match="no searchable terms"):
        openalex_candidates.OpenAlexAdapter(breaker=_PassThrough()).search('  ""  ', now=_NOW, since=_NOW)
    assert calls == []


def test_an_open_openalex_breaker_costs_no_request(monkeypatch):
    """The breaker is persisted for a reason specific to this loop: a failed pull deliberately does
    not stamp, so a failing source is due again on the very next spawn. Unbounded, that is what
    burns the daily allowance."""
    calls = []
    monkeypatch.setattr(paper_registry, "get", lambda url, **kw: calls.append(url) or _OA_BODY)
    a = openalex_candidates.OpenAlexAdapter(breaker=_Open())
    assert a.available() is True or a.available() is False   # never raises
    with pytest.raises(SourceError, match="breaker open"):
        a.search("verifiable inference", now=_NOW, since=_NOW)
    assert calls == []


def test_each_adapter_declares_the_kind_its_candidates_mint_as():
    """`source` is the finder and `kind` is the minter, and stage 3 dispatches on the second. An
    adapter that omits `kind` would stage artifacts nothing can materialize — so the dataclass
    gives it no default and this pins what the built adapters declare."""
    with pytest.raises(TypeError):                  # no default — forgetting it cannot compile
        candidates.Candidate(candidate_id="x", source="s", title="t", url="u")
    assert {c.kind for c in openalex_candidates.from_works(openalex.parse_works(_OA_BODY))} == {"paper"}
    assert {c.kind for c in arxiv_candidates.from_entries(arxiv.parse_feed(_ATOM))} == {"paper"}


def test_openalex_carries_the_open_pdf_url():
    """The full-text seam. `_fulltext_pdf_urls` already reads `openAccessPdf.url`, so capturing
    the OA pdf here is the whole feature — no change to the minter, the merge, or the resolver.

    `best_oa_location` wins over `primary_location` because the primary location can be the
    PAYWALLED publisher copy while best_oa is OpenAlex's own "best open version". Measured over
    68 live results: 47 carry a pdf here, and 25 of those are works `_fulltext_pdf_urls`
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
    got = {c.title: c.payload.get("pdf_url") for c in openalex_candidates.from_works(openalex.parse_works(body))}
    assert got["Open Access Work"] == "https://example.test/open.pdf"     # best_oa wins
    assert got["Primary Only"] == "https://example.test/primary.pdf"      # falls back
    # 14 of 39 non-arXiv works have no open pdf at all. None, not "", so `known_metadata` omits
    # the key entirely rather than handing the minter an empty url to try.
    assert got["No Open Pdf Anywhere"] is None


def test_registry_failure_log_names_the_platform(monkeypatch):
    logs = []

    def fail(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(paper_registry, "host_request", fail)
    monkeypatch.setattr(paper_registry, "log", logs.append)
    with pytest.raises(SourceError, match="request failed"):
        paper_registry.call_get(_PassThrough(), "https://example.invalid")
    assert logs[0].startswith("[paper-registry]")


def test_payload_authors_accepts_only_the_current_object_shape():
    assert candidates.payload_authors({"authors": ["legacy", {"name": "Current"}]}) == [
        {"name": "Current"}]


def test_github_client_construction_errors_keep_the_source_error_contract(monkeypatch):
    def unavailable():
        raise RuntimeError("credentials unavailable")

    monkeypatch.setattr(github_candidates, "GitHubApiClient", unavailable)
    with pytest.raises(SourceError, match="github search failed: RuntimeError"):
        github_candidates.GitHubAdapter().search("query", since=None, now=_NOW)
