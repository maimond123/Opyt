"""robots.txt at the web transport: the RFC 9309 decision on plain values, the once-a-day read,
and a disallowed page skipped exactly as a failed fetch is.

`tests/conftest.py` pins `robots._read` to "no robots.txt" for every test; the tests here that
need a robots.txt patch it again with their own answer.
"""
from __future__ import annotations

import pytest
import requests

from pipeline import host_access, robots
from pipeline.kb import ingest_blog
from pipeline.platforms.web import article as web_article

_REAL_READ = robots._read          # captured at import, before the conftest pin replaces it
_BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_OPYT = "Mozilla/5.0 (compatible; OPYT/1.0)"

# substack.com's own `*` group, as served (abridged): wildcard paths throughout.
_SUBSTACK = """
User-agent: BLEXBot
Disallow: /

User-agent: Twitterbot
Disallow:

User-agent: *
Disallow: /action/
Disallow: /p/*/comment/*
Disallow: /inbox/post/*
Sitemap: https://substack.com/sitemap.xml
"""


def _ok(text, url, ua=_BROWSER):
    return robots.allows(robots.parse(text), url, robots.agents(ua))


# ── deciding ───────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url, allowed", [
    ("https://a.substack.com/p/essay", True),
    ("https://a.substack.com/p/essay/comment/123", False),     # `*` inside the path
    ("https://a.substack.com/action/subscribe", False),
    ("https://a.substack.com/inbox", True),                    # only /inbox/post/* is closed
])
def test_wildcard_paths_follow_the_star_group(url, allowed):
    assert _ok(_SUBSTACK, url) is allowed


@pytest.mark.parametrize("rules, url, allowed", [
    ("User-agent: *\nDisallow: /", "https://s/a", False),
    ("User-agent: *\nDisallow:", "https://s/a", True),                     # empty disallow
    ("User-agent: *\nDisallow: /private", "https://s/private/x", False),   # prefix
    ("User-agent: *\nDisallow: /*.pdf$", "https://s/f.pdf", False),        # `$` ends the path
    ("User-agent: *\nDisallow: /*.pdf$", "https://s/f.pdf?x=1", True),
    ("User-agent: *\nDisallow: /*?", "https://s/a?b=1", False),            # query is matched
    ("User-agent: *\nDisallow: /a\nAllow: /a/b", "https://s/a/b/c", True),  # longest wins
    ("User-agent: *\nAllow: /a\nDisallow: /a", "https://s/a", True),        # allow wins a tie
    ("User-agent: *\nDisallow: /", "https://s/robots.txt", True),           # always readable
    ("# all closed\nUSER-AGENT: *  # every bot\nDISALLOW: /x # x", "https://s/x", False),
])
def test_the_rfc_9309_rules_on_plain_values(rules, url, allowed):
    assert _ok(rules, url) is allowed


_GROUPS = "User-agent: a\nUser-agent: opyt\nDisallow: /x\nUser-agent: b\nDisallow: /"


@pytest.mark.parametrize("rules, ua, path, allowed", [
    # A group naming the fetcher's own token closes it even when `*` is open…
    ("User-agent: opyt\nDisallow: /\n\nUser-agent: *\nAllow: /", _OPYT, "/y", False),
    ("User-agent: OPYT/1.0\nDisallow: /", _OPYT, "/y", False),
    # …and `*` closes it even when its own group is open: a disallow in either refuses.
    ("User-agent: mozilla\nAllow: /\n\nUser-agent: *\nDisallow: /", _BROWSER, "/y", False),
    # Another crawler's group says nothing about Opyt.
    ("User-agent: GPTBot\nDisallow: /", _OPYT, "/y", True),
    # Consecutive user-agent lines share one group; a later user-agent line opens a new one.
    (_GROUPS, _OPYT, "/x", False),
    (_GROUPS, _OPYT, "/y", True),
])
def test_the_fetchers_own_group_and_the_star_group_both_apply(rules, ua, path, allowed):
    assert _ok(rules, f"https://s{path}", ua) is allowed


def test_product_tokens_are_read_from_the_user_agent():
    assert robots.agents(_OPYT) == ("mozilla", "opyt")
    assert robots.agents("Mozilla/5.0 (compatible; OpytBot/1.0; +https://github.com/opyt)") == (
        "mozilla", "opytbot")
    assert robots.agents("opyt-paper-adapter/1.0") == ("opyt-paper-adapter",)
    assert robots.agents("") == ()


@pytest.mark.parametrize("url, governed", [
    ("https://example.com/post", True),
    ("https://arxiv.org/pdf/2401.00001", True),                     # a page, not the API
    ("http://export.arxiv.org/api/query?id_list=1", False),
    ("https://api.openalex.org/works/W1", False),
    ("https://api.github.com/users/x", False),
    ("https://pub.orcid.org/v3.0/0000-0001/person", False),
    ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", False),
    ("https://zenodo.org/api/records/1", False),
    ("https://zenodo.org/records/1", True),
    ("https://www.noahpinion.blog/api/v1/archive", False),          # Substack, custom domain
    ("https://substack.com/inbox", True),
])
def test_documented_apis_are_not_governed(url, governed):
    assert robots.governs(url) is governed


@pytest.mark.parametrize("status, closed, everything_allowed", [
    (200, False, False), (404, False, True), (403, False, True), (401, False, True),
    (429, True, False), (500, True, False), (503, True, False), (None, True, False),
])
def test_what_a_robots_answer_means(status, closed, everything_allowed):
    rules = robots.verdict(status, "User-agent: *\nDisallow: /")
    assert (rules is None) is closed
    assert (rules == robots.ALLOW_ALL) is everything_allowed


# ── doing ──────────────────────────────────────────────────────────────────────────────────────

def _serve_robots(monkeypatch, answers):
    """`answers` maps origin → Rules or None (unreachable). Returns the list of reads."""
    reads = []

    def read(origin, user_agent):
        reads.append((origin, user_agent))
        return answers[origin]
    monkeypatch.setattr(robots, "_read", read)
    return reads


def test_robots_txt_is_read_once_a_day_and_an_unreachable_one_is_asked_again(monkeypatch):
    reads = _serve_robots(monkeypatch, {"https://a": robots.ALLOW_ALL, "https://down": None})
    assert robots.rules_for("https://a", _OPYT, 0.0) is robots.ALLOW_ALL
    assert robots.rules_for("https://a", _OPYT, 3600.0) is robots.ALLOW_ALL
    assert robots.rules_for("https://a", _OPYT, 24 * 3600.0) is robots.ALLOW_ALL
    assert robots.rules_for("https://down", _OPYT, 0.0) is None
    assert robots.rules_for("https://down", _OPYT, 1.0) is None
    assert reads == [("https://a", _OPYT), ("https://a", _OPYT),
                     ("https://down", _OPYT), ("https://down", _OPYT)]


def test_reading_robots_txt_maps_its_status(monkeypatch):
    class Resp:
        def __init__(self, status, text=""):
            self.status_code, self.text = status, text
    answers = {"https://a/robots.txt": Resp(200, "User-agent: *\nDisallow: /x"),
               "https://gone/robots.txt": Resp(404), "https://err/robots.txt": Resp(502)}
    sent = []

    def get(url, **kw):
        sent.append((url, kw["headers"]["User-Agent"]))
        if url not in answers:
            raise requests.ConnectionError("no answer")
        return answers[url]
    monkeypatch.setattr(robots.requests, "get", get)
    assert not robots.allows(_REAL_READ("https://a", _OPYT), "https://a/x", ("opyt",))
    assert _REAL_READ("https://gone", _OPYT) == robots.ALLOW_ALL
    assert _REAL_READ("https://err", _OPYT) is None
    assert _REAL_READ("https://silent", _OPYT) is None
    assert sent[0] == ("https://a/robots.txt", _OPYT)              # asked as the fetcher itself


# ── the seam: a disallowed page is skipped as a failed fetch is ──────────────────────────────

_CLOSED = robots.parse("User-agent: *\nDisallow: /closed")


class _Resp:
    def __init__(self, status=200, headers=None, text="", url=""):
        self.status_code, self.headers, self.text, self.url = status, headers or {}, text, url

    def close(self):
        pass


def test_a_disallowed_url_is_never_sent_inside_or_outside_the_gate(monkeypatch):
    _serve_robots(monkeypatch, {"https://s": _CLOSED})
    sent = []
    send = lambda url, **kw: sent.append(url) or _Resp()
    with pytest.raises(requests.RequestException):
        host_access.request(send, "https://s/closed/page")
    with host_access.scope(host_access.Hosts()):
        with pytest.raises(robots.Refused):
            host_access.request(send, "https://s/closed/page")
        host_access.request(send, "https://s/open")
    assert sent == ["https://s/open"]


def test_inside_the_gate_each_redirect_hop_is_asked(monkeypatch):
    _serve_robots(monkeypatch, {"https://short": robots.ALLOW_ALL, "https://s": _CLOSED})
    sent = []

    def send(url, **kw):
        sent.append(url)
        return _Resp(302, {"Location": "https://s/closed/article"})
    hosts = host_access.Hosts()
    with host_access.scope(hosts), pytest.raises(robots.Refused):
        host_access.request(send, "https://short/x")
    assert sent == ["https://short/x"]
    assert hosts.lock("https://s/").acquire(blocking=False)      # the refusal released its host


def test_an_unreachable_robots_txt_fails_the_read_for_now(monkeypatch):
    _serve_robots(monkeypatch, {"https://down": None})
    with pytest.raises(robots.Refused):
        host_access.request(lambda url, **kw: _Resp(), "https://down/page")


def test_a_documented_api_is_read_without_asking(monkeypatch):
    _serve_robots(monkeypatch, {})                               # any read would KeyError
    response = host_access.request(lambda url, **kw: _Resp(), "http://export.arxiv.org/api/query")
    assert response.status_code == 200


def test_a_feed_feedparser_would_fetch_itself_is_asked_first(monkeypatch):
    reads = _serve_robots(monkeypatch, {"https://s": _CLOSED})
    with pytest.raises(robots.Refused):
        host_access.feed_input("https://s/closed/feed.xml")
    assert host_access.feed_input("https://s/feed.xml") == "https://s/feed.xml"
    assert reads[0][1].startswith("feedparser/")


def test_a_disallowed_article_is_skipped_exactly_like_a_failed_fetch(monkeypatch):
    """`fetch_article` answers None for both, so the document is reported, written nowhere, and
    left owed for the retry schedule a failed fetch already gets."""
    _serve_robots(monkeypatch, {"https://s": _CLOSED})
    sent = []

    def get(url, **kw):
        sent.append(url)
        raise requests.ConnectionError("no answer")
    monkeypatch.setattr(web_article.requests, "get", get)

    assert web_article.fetch_article("https://s/closed/post") is None
    assert sent == []                                            # never requested
    assert web_article.fetch_article("https://s/post") is None   # the failed fetch it mirrors
    assert sent == ["https://s/post"]

    refused = ingest_blog.prepare_article("https://s/closed/post", entry_mode="author_referenced",
                                          seen={})
    failed = ingest_blog.prepare_article("https://s/post", entry_mode="author_referenced", seen={})
    assert refused == {**failed, "url": "https://s/closed/post"}
    assert refused["status"] == "blocked" and refused["res"] is None and refused["atom_id"] is None
