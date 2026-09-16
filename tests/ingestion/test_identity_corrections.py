"""Regressions for the identity review's approved trust and discovery corrections."""

import importlib
from itertools import permutations
from types import SimpleNamespace

import pytest

from pipeline.ingestion.source_classify import classify_source
from pipeline.ingestion.trust_edges import edges_from_html, profile_links_from_html
from pipeline.ingestion.trust_graph import propagate
from pipeline.ingestion.trust_types import Edge
from pipeline.ingestion.url_canon import canonical_identity as ci
from pipeline.kb.oracles import _unsupported_root

dp = importlib.import_module("pipeline.ingestion.discover_profile")


@pytest.mark.parametrize("url", [
    "https://[", "https://[broken/author", "mailto:alice@example.com",
    "javascript:void(0)", "https://example.com:bad",
])
def test_unusable_urls_stop_at_the_identity_boundary(url):
    assert ci(url) == ""
    assert classify_source(url) is None
    assert dp._normalize_url(url) == ""
    assert dp._sources_from_urls([url]) == []
    assert dp._probe_blog_profile(url) == ({}, [], [])
    assert dp._resolve_substack_user_slug(url) is None
    if url.startswith("https"):
        assert _unsupported_root(url)


@pytest.mark.parametrize("href", ["about", "team", "contact.html", "/about", "#bio"])
def test_navigation_links_do_not_create_profiles(href):
    assert classify_source(href) is None
    html = f'<a href="{href}">About</a>'
    assert profile_links_from_html(html) == []
    assert edges_from_html("alice.example", html, {"about", "contact.html"}) == []


@pytest.mark.parametrize("url", [
    "https://substack.com", "https://orcid.org/signin", "https://dblp.org/db/conf/icml/",
    "https://www.academia.edu/12345/A_paper", "https://alice.academia.edu/12345/A_paper",
    "https://notsemanticscholar.org/author/Alice/123",
    "https://notresearchgate.net/profile/Alice",
    "https://scholar.google.com.evil.example/citations?user=Alice",
])
def test_non_profiles_do_not_surface_from_hubs(url):
    assert profile_links_from_html(f'<a href="{url}">link</a>') == []


def test_blog_root_does_not_declare_platform_roots_or_lookalikes(monkeypatch):
    urls = ["https://substack.com", "https://github.com.evil.example/alice",
            "https://substack.com.evil.example"]
    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        status_code=200, text='<title>Alice</title>' + ''.join(
            f'<a href="{url}">link</a>' for url in urls)))
    monkeypatch.setattr(dp, "_detect_rss_feed", lambda url: None)
    _, sources, targets = dp._probe_blog_profile("https://alice.example")
    assert targets == []
    assert [s.url for s in sources] == ["https://alice.example"]


def test_a_blog_root_declares_the_authors_other_homes(monkeypatch):
    """The corpus miss. Until 2026-09-09 only x/github/substack counted as a declared identity, so
    an author's own blog on any other host was skipped as a random outbound link — measured on
    karpathy.ai, where his 14 `karpathy.github.io` essays were never considered while a DIFFERENT
    person's repo was auto-trusted for classifying as `github`."""
    urls = ["https://karpathy.github.io/2019/04/25/recipe/",     # his, a deep link
            "https://karpathy.bearblog.dev/blog/",               # his
            "https://www.cs.toronto.edu/~hinton/",               # another researcher
            "https://www.wired.com/2015/01/karpathy/"]           # press about him
    monkeypatch.setattr(dp.requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, text='<title>Andrej Karpathy</title>' + ''.join(
            f'<a href="{u}">link</a>' for u in urls)))
    monkeypatch.setattr(dp, "_detect_rss_feed", lambda url: None)

    _, sources, _ = dp._probe_blog_profile("https://karpathy.ai")

    # The HOME of each, never the deep link that revealed it: 14 essays are ONE source, and the
    # blog adapter finds the rest from its feed.
    assert {s.url for s in sources} == {"https://karpathy.ai",
                                        "https://karpathy.github.io",
                                        "https://karpathy.bearblog.dev"}


def test_a_path_match_never_becomes_a_source(monkeypatch):
    """Promoting `cs.stanford.edu/people/karpathy/…` would register all of Stanford CS as his
    site. Path-scoped work is `link_discovery`'s job, as a hub candidate under its own guard."""
    monkeypatch.setattr(dp.requests, "get", lambda *a, **kw: SimpleNamespace(
        status_code=200, text='<title>Andrej Karpathy</title>'
        '<a href="https://cs.stanford.edu/people/karpathy/advice.html">undergrads</a>'))
    monkeypatch.setattr(dp, "_detect_rss_feed", lambda url: None)

    _, sources, targets = dp._probe_blog_profile("https://karpathy.ai")

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
    verdicts = propagate([Edge("root", ci(alice), "identity_declared")], {"root"}, {ci(bob)})
    assert verdicts[ci(alice)].trusted
    assert not verdicts[ci(bob)].trusted


def test_account_aliases_share_the_same_identity():
    assert ci("https://github.com/orgs/openai") == ci("https://github.com/openai")
    assert ci("https://semanticscholar.org/author/Alice/123") == ci("https://semanticscholar.org/author/123")


def test_strongest_evidence_survives_every_duplicate_order():
    edges = [Edge("root", "alice", via=via) for via in
             ("html_link", "identity_declared", "identity_verified")]
    for ordered in permutations(edges):
        verdict = propagate(list(ordered), {"root"})["alice"]
        assert verdict.trusted
        assert verdict.edges[0]["via"] == "identity_verified"


def test_discovery_does_not_overwrite_declared_evidence(monkeypatch):
    monkeypatch.setattr(dp, "fetch_landing_edges", lambda *args: [Edge("root.example", "alice.example", "html_link")])
    monkeypatch.setattr(dp, "fetch_profile_links", lambda *args: [])
    sources = [dp.DiscoveredSource("blog", "https://root.example"),
               dp.DiscoveredSource("blog", "https://alice.example")]
    result = dp._compute_trust("alice", "root.example", [("alice.example", False)], [], sources)
    assert result["alice.example"].trusted
    assert result["alice.example"].edges[0]["via"] == "identity_declared"


def test_host_found_github_with_another_handle_is_a_candidate():
    sources = dp._sources_from_urls(["https://github.com/alicewrites"])
    assert [s.url for s in sources] == ["https://github.com/alicewrites"]
    assert sources[0].trust is None


def test_github_blog_field_keeps_domains_containing_x_com(monkeypatch):
    monkeypatch.setattr(dp.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        status_code=200, raise_for_status=lambda: None,
        json=lambda: {"login": "alice", "blog": "https://max.com"}))
    monkeypatch.setattr(dp, "_detect_rss_feed", lambda url: None)
    assert "https://max.com" in {s.url for s in dp._probe_github("alice")}


def test_x_website_and_bio_keep_domains_containing_x_com(monkeypatch):
    from pipeline.ingestion import x_graphql_core as core

    monkeypatch.setattr(core, "read_x_cookies", lambda: {})
    monkeypatch.setattr(core, "auth_headers", lambda *args: {})
    monkeypatch.setattr(core, "fetch_user_profile", lambda *args: {
        "display_name": "Alice", "bio": "", "website": "https://max.com",
        "bio_urls": ["https://relax.com", "https://x.com/alice"],
    })
    monkeypatch.setattr(dp, "_detect_rss_feed", lambda url: None)
    _, sources = dp._probe_twitter_bio("alice")
    assert {s.url for s in sources} == {"https://max.com", "https://relax.com"}


@pytest.mark.parametrize("status,body", [(503, "unavailable"), (200, "")])
def test_failed_blog_root_is_not_cached(monkeypatch, tmp_path, status, body):
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=status, text=body)
    monkeypatch.setattr(dp.requests, "get", get)
    cfg = SimpleNamespace(state_file=lambda name: tmp_path / f"{name}.json")
    for _ in range(2):
        result = dp.discover_profile("https://alice.example", seed_type="blog", config=cfg)
        assert result["sources"] == []
    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


def test_changed_publication_invalidates_cached_discovery(monkeypatch, tmp_path):
    root = {"url": "https://old.substack.com"}
    monkeypatch.setattr(dp, "_probe_substack_profile", lambda seed: (
        {"display_name": "Alice", "root_url": root["url"]},
        [dp.DiscoveredSource("substack", root["url"])], []))
    monkeypatch.setattr(dp, "_probe_github", lambda seed: [])
    monkeypatch.setattr(dp, "fetch_landing_edges", lambda *args: [])
    monkeypatch.setattr(dp, "fetch_profile_links", lambda *args: [])
    monkeypatch.setattr(dp, "fetch_substack_twitter_handle", lambda *args: None)
    cfg = SimpleNamespace(state_file=lambda name: tmp_path / f"{name}.json")
    dp.discover_profile("newsletter", seed_type="substack", config=cfg)
    root["url"] = "https://new.substack.com"
    result = dp.discover_profile("newsletter", seed_type="substack", config=cfg)
    assert not result.get("from_cache")
    assert [s["url"] for s in result["sources"]] == [root["url"]]
