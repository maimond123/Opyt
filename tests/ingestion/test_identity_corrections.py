"""Regressions for the identity review's approved trust and discovery corrections."""

from types import SimpleNamespace

import pytest

from pipeline.platforms.source_classify import classify_source
from pipeline.platforms.url_canon import canonical_identity as ci
from pipeline.oracles.reference import unsupported_root
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x import profile as x_profile

from pipeline.oracles.discovery import backlink
from pipeline.oracles.discovery import probes
from pipeline.oracles.discovery import shaping
from pipeline.oracles.discovery import profile as dp
from pipeline.oracles.discovery.verdict import Candidate, judge


@pytest.mark.parametrize("url", [
    "https://[", "https://[broken/author", "mailto:alice@example.com",
    "javascript:void(0)", "https://example.com:bad",
])
def test_unusable_urls_stop_at_the_identity_boundary(url):
    assert ci(url) == ""
    assert classify_source(url) is None
    assert shaping.normalize_url(url) == ""
    assert probes.probe_blog_profile(url) == ({}, [], [])
    assert backlink.substack_profile(url) == (None, None)
    if url.startswith("https"):
        assert unsupported_root(url)


@pytest.mark.parametrize("href", ["about", "team", "contact.html", "/about", "#bio"])
def test_navigation_links_do_not_create_profiles(href):
    assert classify_source(href) is None
    html = f'<a href="{href}">About</a>'
    assert backlink.facts_from_html("alice.example", html).links_to == frozenset()


def test_blog_root_does_not_declare_platform_roots_or_lookalikes(monkeypatch):
    urls = ["https://substack.com", "https://github.com.evil.example/alice",
            "https://substack.com.evil.example"]
    monkeypatch.setattr(probes.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        status_code=200, text='<title>Alice</title>' + ''.join(
            f'<a href="{url}">link</a>' for url in urls)))
    _, sources, targets = probes.probe_blog_profile("https://alice.example")
    assert targets == []
    assert [s.url for s in sources] == ["https://alice.example"]


def test_a_blog_root_declares_the_authors_other_homes(monkeypatch):
    """The corpus miss. Only x/github/substack counted as a declared identity, so
    an author's own blog on any other host was skipped as a random outbound link — measured on
    karpathy.ai, where his 14 `karpathy.github.io` essays were never considered while a DIFFERENT
    person's repo was auto-trusted for classifying as `github`."""
    urls = ["https://karpathy.github.io/2019/04/25/recipe/",     # his, a deep link
            "https://karpathy.bearblog.dev/blog/",               # his
            "https://www.cs.toronto.edu/~hinton/",               # another researcher
            "https://www.wired.com/2015/01/karpathy/"]           # press about him
    monkeypatch.setattr(probes.requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, text='<title>Andrej Karpathy</title>' + ''.join(
            f'<a href="{u}">link</a>' for u in urls)))

    _, sources, _ = probes.probe_blog_profile("https://karpathy.ai")

    # The HOME of each, never the deep link that revealed it: 14 essays are ONE source, and the
    # blog adapter finds the rest from its feed.
    assert {s.url for s in sources} == {"https://karpathy.ai",
                                        "https://karpathy.github.io",
                                        "https://karpathy.bearblog.dev"}


def test_a_path_match_never_becomes_a_source(monkeypatch):
    """Promoting `cs.stanford.edu/people/karpathy/…` would register all of Stanford CS as his
    site. Path-scoped work is `link_discovery`'s job, as a hub candidate under its own guard."""
    monkeypatch.setattr(probes.requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, text='<title>Andrej Karpathy</title>'
        '<a href="https://cs.stanford.edu/people/karpathy/advice.html">undergrads</a>'))

    _, sources, targets = probes.probe_blog_profile("https://karpathy.ai")

    assert [s.url for s in sources] == ["https://karpathy.ai"]
    assert targets == []


@pytest.mark.parametrize("alice,bob", [
    ("https://scholar.google.com/citations?user=Alice", "https://scholar.google.com/citations?user=Bob"),
    ("https://semanticscholar.org/author/Alice/123", "https://semanticscholar.org/author/Bob/456"),
    ("https://orcid.org/0000-0002-1825-0097", "https://orcid.org/0000-0001-5109-3700"),
    ("https://dblp.org/pid/01/123.html", "https://dblp.org/pid/02/123.html"),
    ("https://researchgate.net/profile/Alice", "https://researchgate.net/profile/Bob"),
    ("https://academia.edu/Alice", "https://academia.edu/Bob"),
    ("https://arxiv.org/a/alice_1", "https://arxiv.org/a/bob_1"),
    ("https://github.com/orgs/alice", "https://github.com/orgs/bob"),
])
def test_accounts_on_the_same_platform_do_not_share_trust(alice, bob):
    assert ci(alice) and ci(bob) and ci(alice) != ci(bob)
    got = {v.key: v.verdict for v in judge(
        [Candidate(ci(alice), alice, declared="on their site"), Candidate(ci(bob), bob)], {},
        oracle_ids=frozenset({"root"}), known=frozenset({"root"}))}
    assert got[ci(alice)] != "drop"
    assert got[ci(bob)] == "drop"


def test_account_aliases_share_the_same_identity():
    assert ci("https://github.com/orgs/openai") == ci("https://github.com/openai")
    assert ci("https://semanticscholar.org/author/Alice/123") == ci("https://semanticscholar.org/author/123")


def test_x_website_and_bio_keep_domains_containing_x_com(monkeypatch):

    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {})
    monkeypatch.setattr(x_local, "auth_headers", lambda *args: {})
    monkeypatch.setattr(x_profile, "fetch_user_profile", lambda *args: {
        "display_name": "Alice", "bio": "", "website": "https://max.com",
        "bio_urls": ["https://relax.com", "https://x.com/alice"],
    })
    _, sources = probes.probe_twitter_bio("alice")
    assert {s.url for s in sources} == {"https://max.com", "https://relax.com"}


@pytest.mark.parametrize("status,body", [(503, "unavailable"), (200, "")])
def test_failed_blog_root_finds_nothing(monkeypatch, status, body):
    monkeypatch.setattr(probes.requests, "get",
                        lambda url, **kwargs: SimpleNamespace(status_code=status, text=body))
    result = dp.discover_profile("https://alice.example", seed_type="blog")
    assert result["sources"] == []
