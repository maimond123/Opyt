"""Behavior pins for the `OpenAlexClient` lookups no other test executes.

Transport is stubbed at `paper_registry.get` (or `paper_registry.host_request` for the breaker
tests), and every client gets a fake breaker, so no CircuitBreaker row is ever persisted. A
stubbed `get` returning None is a 404. A failed or malformed page raises `SourceError` instead
of reading as empty.
"""
from __future__ import annotations

import json
import urllib.parse
from types import SimpleNamespace

import pytest

from pipeline.circuit_breaker import CircuitOpenError
from pipeline.platforms.papers import openalex
from pipeline.platforms.papers import registry as paper_registry
from pipeline.platforms.errors import SourceError


class _Recording:
    """Mirrors `CircuitBreaker.call`: an exception records a failure, a return records success."""

    def __init__(self):
        self.outcomes = []

    def call(self, fn, **_kw):
        try:
            result = fn()
        except Exception:
            self.outcomes.append("failure")
            raise
        self.outcomes.append("success")
        return result


class _Open:
    def call(self, _fn, **_kw):
        raise CircuitOpenError("api.openalex.org", 900.0)


def _client():
    return openalex.OpenAlexClient(breaker=_Recording())


def _serve(monkeypatch, payload):
    """Stub the GET: record each url, answer with `payload` (JSON-encoded, raw bytes, or None)."""
    seen = []
    body = payload if payload is None or isinstance(payload, bytes) else json.dumps(payload).encode()
    monkeypatch.setattr(paper_registry, "get", lambda url, **_kw: seen.append(url) or body)
    return seen


def _fail(monkeypatch):
    """Make the request itself fail, as a timeout or a 5xx does."""
    def offline(*_a, **_kw):
        raise ConnectionError("offline")
    monkeypatch.setattr(paper_registry, "host_request", offline)
    monkeypatch.setattr(paper_registry, "log", lambda _msg: None)


def _split(url):
    parts = urllib.parse.urlsplit(url)
    return (f"{parts.scheme}://{parts.netloc}{parts.path}",
            dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True)))


_AUTHOR_SELECT = ("id,display_name,orcid,works_count,cited_by_count,summary_stats,"
                  "last_known_institutions")
_SOURCE_SELECT = "id,display_name,works_count,type"
_WORK_SELECT = ("title,abstract_inverted_index,primary_location,best_oa_location,"
                "publication_date,publication_year,cited_by_count,authorships")


# ── author_by_orcid ──────────────────────────────────────────────────────────────

def test_author_by_orcid_request(monkeypatch):
    seen = _serve(monkeypatch, {"id": "https://openalex.org/A5043841592"})
    _client().author_by_orcid("0000-0002-4027-364X")
    assert seen == ["https://api.openalex.org/authors/orcid:0000-0002-4027-364X?select="
                    + urllib.parse.quote_plus(_AUTHOR_SELECT)]
    assert _split(seen[0]) == ("https://api.openalex.org/authors/orcid:0000-0002-4027-364X",
                               {"select": _AUTHOR_SELECT})


def test_author_by_orcid_returns_the_decoded_record_unchanged(monkeypatch):
    record = {"id": "https://openalex.org/A5043841592", "display_name": "Frances H. Arnold",
              "orcid": "https://orcid.org/0000-0002-4027-364X", "works_count": 928}
    _serve(monkeypatch, record)
    assert _client().author_by_orcid("0000-0002-4027-364X") == record


def test_author_by_orcid_failed_get_is_none(monkeypatch):
    _serve(monkeypatch, None)
    assert _client().author_by_orcid("0000-0002-4027-364X") is None


@pytest.mark.parametrize("orcid", ["https://orcid.org/0000-0002-4027-364X",
                                   "http://orcid.org/0000-0002-4027-364X/",
                                   " 0000-0002-4027-364X "])
def test_author_by_orcid_asks_for_the_bare_id(monkeypatch, orcid):
    """A full orcid.org URL must be reduced to the bare id, not put into the path as-is."""
    seen = _serve(monkeypatch, {})
    _client().author_by_orcid(orcid)
    assert _split(seen[0])[0] == "https://api.openalex.org/authors/orcid:0000-0002-4027-364X"


# ── topic_counts ─────────────────────────────────────────────────────────────────

def test_topic_counts_request_for_an_author_and_a_source(monkeypatch):
    seen = _serve(monkeypatch, {"group_by": []})
    _client().topic_counts("A5043841592")
    _client().topic_counts("S4393918830")
    assert [_split(u) for u in seen] == [
        ("https://api.openalex.org/works", {"filter": "author.id:A5043841592",
                                            "group_by": "primary_topic.id"}),
        ("https://api.openalex.org/works", {"filter": "primary_location.source.id:S4393918830",
                                            "group_by": "primary_topic.id"}),
    ]


def test_topic_counts_bad_id_raises_before_any_request(monkeypatch):
    seen = _serve(monkeypatch, {"group_by": []})
    with pytest.raises(ValueError):
        _client().topic_counts("W123")
    assert seen == []


def test_topic_counts_decodes_strips_and_skips(monkeypatch):
    _serve(monkeypatch, {"meta": {"count": 9}, "group_by": [
        {"key": "https://openalex.org/T10011", "key_display_name": "Enzyme Catalysis", "count": 40},
        {"key": "https://openalex.org/T10022/", "key_display_name": "Trailing Slash", "count": "7"},
        {"key": "unknown", "key_display_name": "unknown", "count": 99},
        {"key": "https://openalex.org/T10033", "key_display_name": "", "count": 5},
        {"key": "https://openalex.org/T10044", "key_display_name": "No Count"},
        {"key": "https://openalex.org/T10055", "key_display_name": "Null Count", "count": None},
        {"key": "https://openalex.org/T10066", "key_display_name": "Float Count", "count": 3.9},
        {"key": "https://openalex.org/T10077", "key_display_name": "Bigger Later", "count": 500},
        {"key": "https://openalex.org/T", "key_display_name": "Bare Prefix", "count": 1},
    ]})
    assert _client().topic_counts("A5043841592") == [
        {"id": "T10011", "name": "Enzyme Catalysis", "count": 40},
        {"id": "T10022", "name": "Trailing Slash", "count": 7},
        {"id": "T10066", "name": "Float Count", "count": 3},              # truncated, not rounded
        {"id": "T10077", "name": "Bigger Later", "count": 500},          # server order kept, not re-sorted
        {"id": "T", "name": "Bare Prefix", "count": 1},                  # a bare "T" passes the check
    ]


def test_topic_counts_404_is_empty(monkeypatch):
    _serve(monkeypatch, None)
    assert _client().topic_counts("A5043841592") == []


def test_topic_counts_failed_request_refuses(monkeypatch):
    _fail(monkeypatch)
    with pytest.raises(SourceError, match="request failed"):    # not "no topics"
        _client().topic_counts("A5043841592")


def test_topic_counts_malformed_payloads(monkeypatch):
    for payload in ({}, {"group_by": None}, {"error": "Invalid query"}, None):
        _serve(monkeypatch, json.dumps(payload).encode())
        with pytest.raises(SourceError):                     # malformed is not "no topics"
            _client().topic_counts("A5043841592")
    _serve(monkeypatch, {"group_by": ["not-a-dict"]})
    with pytest.raises(AttributeError):                      # a non-dict row is not skipped
        _client().topic_counts("A5043841592")


# ── source ───────────────────────────────────────────────────────────────────────

def test_source_request(monkeypatch):
    seen = _serve(monkeypatch, {})
    _client().source("S4393918830")
    assert _split(seen[0]) == ("https://api.openalex.org/sources/S4393918830",
                               {"select": _SOURCE_SELECT})


def test_source_returns_the_decoded_record_unchanged(monkeypatch):
    record = {"id": "https://openalex.org/S4393918830", "display_name": "ChemRxiv",
              "works_count": 63565, "type": "repository"}
    _serve(monkeypatch, record)
    assert _client().source("S4393918830") == record


def test_source_failed_get_is_none(monkeypatch):
    _serve(monkeypatch, None)
    assert _client().source("S4393918830") is None


# ── sources_by_name ──────────────────────────────────────────────────────────────

def test_sources_by_name_request(monkeypatch):
    seen = _serve(monkeypatch, {"results": []})
    _client().sources_by_name("ChemRxiv")
    assert _split(seen[0]) == ("https://api.openalex.org/sources", {
        "filter": "display_name.search:ChemRxiv", "select": _SOURCE_SELECT, "per-page": "5"})


def test_sources_by_name_puts_a_comma_name_into_the_filter_raw(monkeypatch):
    """OpenAlex reads `,` as its clause separator; the name is not escaped against it."""
    seen = _serve(monkeypatch, {"results": []})
    _client().sources_by_name("Nature, Physics")
    assert _split(seen[0])[1]["filter"] == "display_name.search:Nature, Physics"


def test_sources_by_name_returns_results_as_a_list(monkeypatch):
    hits = [{"id": "https://openalex.org/S1", "display_name": "ChemRxiv"},
            {"id": "https://openalex.org/S2", "display_name": "ChemRxiv Mirror"}]
    _serve(monkeypatch, {"meta": {"count": 2}, "results": hits})
    assert _client().sources_by_name("ChemRxiv") == hits


def test_sources_by_name_failed_or_malformed_refuses(monkeypatch):
    _fail(monkeypatch)
    with pytest.raises(SourceError):                            # not "no venue"
        _client().sources_by_name("ChemRxiv")
    for payload in ({}, {"results": None}, {"error": "x"}, []):
        _serve(monkeypatch, payload)
        with pytest.raises(SourceError):
            _client().sources_by_name("ChemRxiv")


def test_sources_by_name_non_dict_payload_raises_source_error(monkeypatch):
    _serve(monkeypatch, [{"id": "S1"}])
    with pytest.raises(SourceError, match="not an object"):
        _client().sources_by_name("ChemRxiv")


# ── work_by_doi ──────────────────────────────────────────────────────────────────

def test_work_by_doi_request(monkeypatch):
    seen = _serve(monkeypatch, {"results": []})
    _client().work_by_doi("10.1038/s41586-021-03819-2")
    assert _split(seen[0]) == ("https://api.openalex.org/works", {
        "filter": "doi:https://doi.org/10.1038/s41586-021-03819-2",
        "select": _WORK_SELECT, "per-page": "1"})


@pytest.mark.parametrize("doi", ["https://doi.org/10.1/x", "http://dx.doi.org/10.1/x",
                                 "doi:10.1/x", "DOI:10.1/x", "10.1/x"])
def test_work_by_doi_prefixes_the_bare_doi_once(monkeypatch, doi):
    """A DOI already in URL form must not be double-prefixed."""
    seen = _serve(monkeypatch, {"results": []})
    _client().work_by_doi(doi)
    assert _split(seen[0])[1]["filter"] == "doi:https://doi.org/10.1/x"


def test_work_by_doi_returns_the_first_result(monkeypatch):
    first = {"title": "Highly accurate protein structure prediction", "publication_year": 2021}
    _serve(monkeypatch, {"meta": {"count": 2}, "results": [first, {"title": "second"}]})
    assert _client().work_by_doi("10.1038/s41586-021-03819-2") == first


def test_work_by_doi_unknown_is_none_and_failed_or_malformed_refuses(monkeypatch):
    _fail(monkeypatch)
    with pytest.raises(SourceError):                            # not "unknown DOI"
        _client().work_by_doi("10.0/unknown")
    _serve(monkeypatch, {"meta": {"count": 0}, "results": []})
    assert _client().work_by_doi("10.0/unknown") is None
    for payload in ({}, {"results": None}, {"error": "x"}):
        _serve(monkeypatch, payload)
        with pytest.raises(SourceError):
            _client().work_by_doi("10.0/unknown")


# ── work_by_openalex_id ──────────────────────────────────────────────────────────

def test_work_by_openalex_id_request(monkeypatch):
    seen = _serve(monkeypatch, {})
    _client().work_by_openalex_id("W3177828909")
    assert _split(seen[0]) == ("https://api.openalex.org/works/W3177828909",
                               {"select": "doi,title"})


def test_work_by_openalex_id_returns_the_decoded_record_unchanged(monkeypatch):
    record = {"doi": "https://doi.org/10.1038/s41586-021-03819-2",
              "title": "Highly accurate protein structure prediction"}
    _serve(monkeypatch, record)
    assert _client().work_by_openalex_id("W3177828909") == record


def test_work_by_openalex_id_failed_get_is_none(monkeypatch):
    _serve(monkeypatch, None)
    assert _client().work_by_openalex_id("W3177828909") is None


def test_dict_returning_lookups_refuse_json_that_is_not_an_object(monkeypatch):
    """`source`, `author_by_orcid` and `work_by_openalex_id` are typed `dict | None`; a JSON
    array or scalar must not reach the caller."""
    _serve(monkeypatch, [1, 2])
    with pytest.raises(SourceError, match="not an object"):
        _client().work_by_openalex_id("W1")
    _serve(monkeypatch, b"null")
    with pytest.raises(SourceError, match="not an object"):
        _client().source("S1")
    _serve(monkeypatch, b'"oops"')
    with pytest.raises(SourceError, match="not an object"):
        _client().author_by_orcid("0000-0000-0000-0000")


# ── works_by_landing_page ────────────────────────────────────────────────────────

def test_works_by_landing_page_request(monkeypatch):
    seen = _serve(monkeypatch, {"results": []})
    page = "https://www.nber.org/papers/w31161"
    _client().works_by_landing_page(page)
    assert _split(seen[0]) == ("https://api.openalex.org/works", {
        "filter": f"locations.landing_page_url:{page}", "select": "doi,title", "per-page": "5"})


def test_works_by_landing_page_returns_every_match(monkeypatch):
    hits = [{"doi": "https://doi.org/10.3386/w31161", "title": "One"},
            {"doi": None, "title": "Two"}]
    _serve(monkeypatch, {"results": hits})
    assert _client().works_by_landing_page("https://example.test/p") == hits


def test_works_by_landing_page_failed_or_malformed_refuses(monkeypatch):
    _fail(monkeypatch)
    with pytest.raises(SourceError):
        _client().works_by_landing_page("https://example.test/p")
    for payload in ({}, {"results": None}, {"error": "x"}):
        _serve(monkeypatch, payload)
        with pytest.raises(SourceError):
            _client().works_by_landing_page("https://example.test/p")


# ── shared transport through `_call` ─────────────────────────────────────────────

_EVERY_LOOKUP = [
    ("author_by_orcid", "0000-0002-4027-364X"),
    ("topic_counts", "A5043841592"),
    ("source", "S4393918830"),
    ("sources_by_name", "ChemRxiv"),
    ("work_by_doi", "10.1/x"),
    ("work_by_openalex_id", "W1"),
    ("works_by_landing_page", "https://example.test/p"),
]


@pytest.mark.parametrize("method, arg", _EVERY_LOOKUP)
def test_unparseable_json_raises_source_error(monkeypatch, method, arg):
    _serve(monkeypatch, b"<html>not json</html>")
    with pytest.raises(SourceError, match="unparseable JSON"):
        getattr(_client(), method)(arg)


@pytest.mark.parametrize("method, arg", _EVERY_LOOKUP)
def test_an_open_breaker_raises_source_error_without_a_request(monkeypatch, method, arg):
    seen = _serve(monkeypatch, {})
    with pytest.raises(SourceError, match="breaker open"):
        getattr(openalex.OpenAlexClient(breaker=_Open()), method)(arg)
    assert seen == []


@pytest.mark.parametrize("method, arg", _EVERY_LOOKUP)
def test_a_429_raises_source_error_and_counts_as_a_breaker_failure(monkeypatch, method, arg):
    monkeypatch.setattr(paper_registry, "host_request",
                        lambda *_a, **_kw: SimpleNamespace(status_code=429))
    breaker = _Recording()
    with pytest.raises(SourceError, match="429"):
        getattr(openalex.OpenAlexClient(breaker=breaker), method)(arg)
    assert breaker.outcomes == ["failure"]


@pytest.mark.parametrize("method, arg", _EVERY_LOOKUP)
def test_a_500_is_recorded_as_a_breaker_failure(monkeypatch, method, arg):
    def _500(*_a, **_kw):
        def _raise():
            raise RuntimeError("500 Server Error")
        return SimpleNamespace(status_code=500, raise_for_status=_raise)
    monkeypatch.setattr(paper_registry, "host_request", _500)
    monkeypatch.setattr(paper_registry, "log", lambda _msg: None)
    breaker = _Recording()
    with pytest.raises(SourceError, match="500"):   # the failure is not an empty answer
        getattr(openalex.OpenAlexClient(breaker=breaker), method)(arg)
    assert breaker.outcomes == ["failure"]      # the failure raises inside breaker.call


def test_repeated_failed_reads_open_the_breaker_and_a_real_answer_counts_success(monkeypatch):
    """With the real breaker: three failed GETs open it; an empty page and a 404 are
    answers, so they count as successes."""
    from pipeline.circuit_breaker import CircuitBreaker

    def _500(*_a, **_kw):
        def _raise():
            raise RuntimeError("500 Server Error")
        return SimpleNamespace(status_code=500, raise_for_status=_raise)

    monkeypatch.setattr(paper_registry, "log", lambda _msg: None)
    ok = CircuitBreaker("api.openalex.test-ok", threshold=3, cooldown=900.0)
    monkeypatch.setattr(paper_registry, "host_request", lambda *_a, **_kw: SimpleNamespace(
        status_code=200, content=b'{"results": []}', raise_for_status=lambda: None))
    paper_registry.call_get(ok, "https://api.openalex.org/works")
    monkeypatch.setattr(paper_registry, "host_request",
                        lambda *_a, **_kw: SimpleNamespace(status_code=404))
    assert paper_registry.call_get(ok, "https://api.openalex.org/authors/A1") is None
    assert ok.peek()

    down = CircuitBreaker("api.openalex.test-down", threshold=3, cooldown=900.0)
    monkeypatch.setattr(paper_registry, "host_request", _500)
    for _ in range(3):
        with pytest.raises(SourceError):
            paper_registry.call_get(down, "https://api.openalex.org/works")
    assert not down.peek()


# ── work_record: the one decode of a work's location, DOI, title, date and abstract ──

def test_work_record_decodes_the_shared_fields():
    rec = openalex.work_record({
        "doi": "https://doi.org/10.1/x", "title": "  A \n Title ", "publication_date": "2026-08-01",
        "abstract_inverted_index": {"world": [1], "hello": [0]},
        "primary_location": {"landing_page_url": "https://x.org/p", "pdf_url": "https://x.org/p.pdf",
                             "source": {"display_name": "ChemRxiv"}},
        "best_oa_location": {"pdf_url": "https://oa.org/p.pdf"}})
    assert rec == openalex.WorkRecord(
        doi="https://doi.org/10.1/x", landing_page="https://x.org/p",
        pdf_url="https://oa.org/p.pdf", venue="ChemRxiv", title="A Title",
        published="2026-08-01", abstract="hello world")
    assert rec.url == "https://doi.org/10.1/x"


def test_work_record_of_a_bare_work_falls_back_to_the_landing_page_and_primary_pdf():
    rec = openalex.work_record({"primary_location": {"landing_page_url": "https://x.org/p",
                                                     "pdf_url": "https://x.org/p.pdf"}})
    assert (rec.url, rec.pdf_url, rec.venue, rec.title, rec.published, rec.abstract) == (
        "https://x.org/p", "https://x.org/p.pdf", None, "", None, "")


# ── works: one page of an id's corpus ──────────────────────────────────────

def test_works_real_empty_page_is_empty(monkeypatch):
    _serve(monkeypatch, {"meta": {"count": 0}, "results": []})
    assert _client().works("A5043841592", limit=25) == []


@pytest.mark.parametrize("payload", [{}, {"results": None}, {"results": {"id": "W1"}}, [], None])
def test_works_malformed_page_refuses(monkeypatch, payload):
    _serve(monkeypatch, json.dumps(payload).encode())
    with pytest.raises(SourceError):
        _client().works("A5043841592", limit=25)


def test_works_failed_request_refuses(monkeypatch):
    _fail(monkeypatch)
    with pytest.raises(SourceError, match="request failed"):
        _client().works("A5043841592", limit=25)


def test_search_works_malformed_page_refuses(monkeypatch):
    _serve(monkeypatch, {"error": "Invalid query"})
    with pytest.raises(SourceError, match="no 'results' list"):
        openalex.search_works("protein folding", since=None, limit=5, get_breaker=_Recording)
