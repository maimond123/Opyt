"""Behavior pins for the blog read helpers (harvest_links_from, harvest_hub_links, _fetch_article).
Network is stubbed at ``requests.get``; trafilatura's extract/metadata are stubbed for
_fetch_article. These freeze today's output so a pure move that changes it fails here."""
from __future__ import annotations

import sys
from datetime import date
from types import SimpleNamespace

import pytest
from requests.structures import CaseInsensitiveDict

from pipeline.platforms.web import discover as ing
from pipeline.platforms.web import article as web_article

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


class _Resp:
    def __init__(self, status=200, text="", headers=None, url=None):
        self.status_code = status
        self.text = text
        self.headers = headers if headers is not None else {}
        if url is not None:
            self.url = url


def _serve(monkeypatch, pages):
    """``pages`` maps url → _Resp or Exception; an unmapped url is a 404. Returns the request log."""
    seen: list[dict] = []

    def fake_get(url, **kw):
        seen.append({"url": url, **kw})
        r = pages.get(url, _Resp(status=404))
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(ing.requests, "get", fake_get)
    return seen


# ── harvest_links_from ───────────────────────────────────────────────────────

def test_harvest_links_from_tags_via_and_dedups_first_page_wins(monkeypatch):
    seen = _serve(monkeypatch, {
        "https://a.com/": _Resp(text='<a href="/p1">One</a> <a href="https://other.org/x">Ext</a>'),
        "https://a.com/blog": _Resp(text='<a href="p2">Two</a> <a href="https://a.com/p1">Again</a>'),
    })
    out = ing.harvest_links_from(["https://a.com/", "https://a.com/blog"])
    assert out == [
        {"url": "https://a.com/p1", "anchor": "One", "via": "https://a.com/"},
        {"url": "https://other.org/x", "anchor": "Ext", "via": "https://a.com/"},
        {"url": "https://a.com/p2", "anchor": "Two", "via": "https://a.com/blog"},
    ]
    assert [s["url"] for s in seen] == ["https://a.com/", "https://a.com/blog"]
    assert all(s["timeout"] == 15 and s["headers"] == {"User-Agent": _UA} for s in seen)


def test_harvest_links_from_skips_non_200_empty_and_raising_pages(monkeypatch):
    _serve(monkeypatch, {
        "https://a.com/500": _Resp(status=500, text='<a href="/x">X</a>'),
        "https://a.com/empty": _Resp(text=""),
        "https://a.com/boom": ConnectionError("down"),
        "https://a.com/ok": _Resp(text='<a href="/y">Y</a>'),
    })
    out = ing.harvest_links_from(["https://a.com/500", "https://a.com/empty",
                                  "https://a.com/boom", "https://a.com/ok"])
    assert out == [{"url": "https://a.com/y", "anchor": "Y", "via": "https://a.com/ok"}]


def test_harvest_links_from_drops_fragments_and_non_nav_schemes(monkeypatch):
    html = ('<a href="#top">Top</a><a href="mailto:x@y.z">Mail</a><a href="tel:1">Tel</a>'
            '<a href="javascript:void(0)">JS</a><a>none</a><a href="//cdn.b.com/p">Proto</a>'
            '<a href="/n"><span>Nested</span> <b>text</b></a>')
    _serve(monkeypatch, {"https://a.com/x/": _Resp(text=html)})
    assert ing.harvest_links_from(["https://a.com/x/"]) == [
        {"url": "https://cdn.b.com/p", "anchor": "Proto", "via": "https://a.com/x/"},
        {"url": "https://a.com/n", "anchor": "Nested text", "via": "https://a.com/x/"},
    ]


def test_harvest_links_from_resolves_against_each_page(monkeypatch):
    _serve(monkeypatch, {"https://a.com/writings/": _Resp(text='<a href="essay-1">E</a>')})
    out = ing.harvest_links_from(["https://a.com/writings/"])
    assert out == [{"url": "https://a.com/writings/essay-1", "anchor": "E",
                    "via": "https://a.com/writings/"}]


def test_harvest_links_from_empty_page_list(monkeypatch):
    seen = _serve(monkeypatch, {})
    assert ing.harvest_links_from([]) == []
    assert seen == []


# ── harvest_hub_links ────────────────────────────────────────────────────────

def test_harvest_hub_links_probes_homepage_then_index_paths(monkeypatch):
    seen = _serve(monkeypatch, {
        "https://a.com": _Resp(text='<a href="/p1">Home link</a>'),
        "https://a.com/archive": _Resp(text='<a href="/p1">Dup</a><a href="/p2">Archived</a>'),
    })
    out = ing.harvest_hub_links("https://a.com/")
    assert [s["url"] for s in seen] == [
        "https://a.com", "https://a.com/archive", "https://a.com/blog", "https://a.com/posts",
        "https://a.com/writing", "https://a.com/articles", "https://a.com/essays",
        "https://a.com/notes",
    ]
    assert out == [
        {"url": "https://a.com/p1", "anchor": "Home link", "via": "https://a.com"},
        {"url": "https://a.com/p2", "anchor": "Archived", "via": "https://a.com/archive"},
    ]


@pytest.mark.parametrize("base", ["", None, "/", "///"])
def test_harvest_hub_links_blank_base_fetches_nothing(monkeypatch, base):
    seen = _serve(monkeypatch, {})
    assert ing.harvest_hub_links(base) == []
    assert seen == []


def test_harvest_hub_links_all_fail_is_empty(monkeypatch):
    seen = _serve(monkeypatch, {})              # every probe 404s
    assert ing.harvest_hub_links("https://a.com") == []
    assert len(seen) == 1 + len(ing.HUB_INDEX_PATHS)


# ── _fetch_article ───────────────────────────────────────────────────────────

_HTML = '<html><head><meta property="article:published_time" content="2021-03-04T10:00:00Z">' \
        '</head><body><p>Body</p></body></html>'


@pytest.fixture()
def traf(monkeypatch):
    """Stub trafilatura's extract + metadata; record the extract call."""
    rec: dict = {"extract": [], "meta": SimpleNamespace(title="T", author="Ann", date="2020-01-01"),
                 "content": "# Extracted\n\nText"}

    def fake_extract(html, **kw):
        rec["extract"].append({"html": html, **kw})
        return rec["content"]

    monkeypatch.setattr(web_article.trafilatura, "extract", fake_extract)
    monkeypatch.setattr(web_article.trafilatura.metadata, "extract_metadata", lambda _html: rec["meta"])
    return rec


def test_fetch_article_happy_path_shape(monkeypatch, traf):
    hdrs = CaseInsensitiveDict({"Content-Type": "text/html", "cf-mitigated": "challenge"})
    seen = _serve(monkeypatch, {"https://b.com/post": _Resp(
        text=_HTML, headers=hdrs, url="https://b.com/final")})
    out = web_article.fetch_article("https://b.com/post")
    assert out == {
        "url": "https://b.com/post",
        "title": "T",
        "date": "2021-03-04",                    # page's asserted publish date beats meta.date
        "author": "Ann",
        "content": "# Extracted\n\nText",
        "headers": {"Content-Type": "text/html", "cf-mitigated": "challenge"},
        "final_url": "https://b.com/final",
        "declares": {"article": False, "published": True},   # a publish time, no article type
    }
    assert type(out["headers"]) is dict
    assert seen == [{"url": "https://b.com/post", "timeout": 15, "headers": {"User-Agent": _UA}}]
    assert traf["extract"] == [{"html": _HTML, "output_format": "markdown", "include_links": True,
                                "include_images": True, "no_fallback": False}]


@pytest.mark.usefixtures("traf")
def test_fetch_article_final_url_falls_back_to_requested(monkeypatch):
    _serve(monkeypatch, {"https://b.com/a": _Resp(text=_HTML)})              # no .url attr
    assert web_article.fetch_article("https://b.com/a")["final_url"] == "https://b.com/a"
    _serve(monkeypatch, {"https://b.com/a": _Resp(text=_HTML, url="")})      # empty .url
    assert web_article.fetch_article("https://b.com/a")["final_url"] == "https://b.com/a"


@pytest.mark.usefixtures("traf")
def test_fetch_article_meta_date_when_page_asserts_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "htmldate", None)        # rung 1 import fails
    _serve(monkeypatch, {"https://b.com/a": _Resp(text="<p>no dates</p>")})
    assert web_article.fetch_article("https://b.com/a")["date"] == "2020-01-01"


def test_fetch_article_none_meta_gives_blank_title_author(monkeypatch, traf):
    monkeypatch.setitem(sys.modules, "htmldate", None)
    traf["meta"] = None
    _serve(monkeypatch, {"https://b.com/a": _Resp(text="<p>x</p>")})
    out = web_article.fetch_article("https://b.com/a")
    assert (out["title"], out["author"], out["date"]) == ("", "", "")


def test_fetch_article_none_meta_fields_become_blank(monkeypatch, traf):
    traf["meta"] = SimpleNamespace(title=None, author=None, date=None)
    _serve(monkeypatch, {"https://b.com/a": _Resp(text=_HTML)})
    out = web_article.fetch_article("https://b.com/a")
    assert (out["title"], out["author"]) == ("", "")


def test_fetch_article_non_200_is_none_without_extracting(monkeypatch, traf):
    _serve(monkeypatch, {"https://b.com/a": _Resp(status=403, text=_HTML)})
    assert web_article.fetch_article("https://b.com/a") is None
    assert traf["extract"] == []


@pytest.mark.usefixtures("traf")
def test_fetch_article_transport_error_is_none(monkeypatch):
    _serve(monkeypatch, {"https://b.com/a": ConnectionError("down")})
    assert web_article.fetch_article("https://b.com/a") is None


@pytest.mark.parametrize("content", [None, ""])
def test_fetch_article_empty_extraction_is_none(monkeypatch, traf, content):
    traf["content"] = content
    _serve(monkeypatch, {"https://b.com/a": _Resp(text=_HTML)})
    assert web_article.fetch_article("https://b.com/a") is None


@pytest.mark.usefixtures("traf")
@pytest.mark.parametrize("target", ["extract", "metadata"])
def test_fetch_article_parser_error_is_none(monkeypatch, target):
    # B17: trafilatura sat outside the try, so a parser exception escaped instead of None.
    def boom(*_a, **_kw):
        raise ValueError("parser died")

    if target == "extract":
        monkeypatch.setattr(web_article.trafilatura, "extract", boom)
    else:
        monkeypatch.setattr(web_article.trafilatura.metadata, "extract_metadata", boom)
    _serve(monkeypatch, {"https://b.com/a": _Resp(text=_HTML)})
    assert web_article.fetch_article("https://b.com/a") is None


# ── publication_day: one row per case ───────────────────────────────────────────

_TODAY = date(2026, 10, 5)


def test_publication_day_empty_without_an_iso_prefix():
    assert web_article.publication_day("April 21, 2025", _TODAY) == ""


def test_publication_day_empty_for_an_impossible_date():
    assert web_article.publication_day("2025-13-45", _TODAY) == ""


def test_publication_day_empty_for_a_future_date():
    assert web_article.publication_day("2026-10-06T00:00:00Z", _TODAY) == ""


def test_publication_day_is_the_day_of_a_valid_past_or_present_date():
    assert web_article.publication_day("2026-10-05T23:59:59.000Z", _TODAY) == "2026-10-05"


# ── publication_date: one row per case ──────────────────────────────────────────

def test_publication_date_takes_the_oldest_asserted_date():
    assert web_article.publication_date(["2021-03-04", "2019-01-02"], "2024-01-01", "2025-01-01") == (
        "2019-01-02")


def test_publication_date_falls_back_to_the_inferred_date():
    assert web_article.publication_date([], "2018-07-08T10:00", "2025-01-01") == "2018-07-08"


def test_publication_date_falls_back_to_the_metadata_date():
    assert web_article.publication_date([], "", "2017-05-06 00:00:00") == "2017-05-06"


def test_publication_date_empty_when_every_rung_misses():
    assert web_article.publication_date([], "", "") == ""
