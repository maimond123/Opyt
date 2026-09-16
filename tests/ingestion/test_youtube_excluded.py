"""Excluded video links cannot become sources, trust evidence, or fetch targets."""

import importlib
from types import SimpleNamespace

import pytest

from pipeline.ingestion import trust_edges, x_graphql_core
from pipeline.ingestion.source_classify import classify_source
from pipeline.ingestion.url_canon import canonical_identity

dp = importlib.import_module("pipeline.ingestion.discover_profile")


@pytest.mark.parametrize("url", [
    "https://youtube.com",
    "https://www.youtube.com/@alice",
    "https://m.youtube.com/channel/UC123",
    "https://music.youtube.com/watch?v=123",
    "https://youtube.com/c/alice",
    "https://youtube.com/user/alice",
    "https://youtu.be/123",
])
def test_youtube_urls_have_no_identity_or_source(url):
    assert canonical_identity(url) == ""
    assert classify_source(url) is None
    assert dp._classify_url(url) is None
    assert dp._sources_from_urls([url]) == []


def test_exclusion_respects_host_boundaries():
    url = "https://notyoutube.com"
    assert canonical_identity(url) == "notyoutube.com"
    assert classify_source(url).is_profile


@pytest.mark.parametrize("url", ["https://m.youtube.com/@alice", "https://youtu.be/123"])
def test_existing_blog_seed_cannot_fetch_an_excluded_host(url):
    """These URLs could be minted as blog roots before platform exclusion was centralized."""
    assert dp._probe_blog_profile(url) == ({}, [], [])


def test_hub_links_do_not_emit_youtube_evidence():
    html = ('<a href="https://youtube.com/@alice">video</a>'
            '<a href="https://youtu.be/123">video</a>'
            '<a href="https://github.com/alice">code</a>')
    relevant = {"youtube.com/@alice", "youtube.com", "github.com/alice"}

    edges = trust_edges.edges_from_html("alice.example", html, relevant)
    assert {e.target for e in edges} == {"github.com/alice"}
    assert [link.url for link in trust_edges.profile_links_from_html(html)] == [
        "https://github.com/alice"]


@pytest.mark.parametrize("seed,seed_type", [("alice", "x"), ("https://alice.example", "blog")])
def test_discovery_drops_video_links_without_fetching_them(monkeypatch, tmp_path, seed, seed_type):
    monkeypatch.setattr(x_graphql_core, "read_x_cookies", lambda: {})
    monkeypatch.setattr(x_graphql_core, "auth_headers", lambda *args: {})
    monkeypatch.setattr(x_graphql_core, "fetch_user_profile", lambda *args: {
        "display_name": "Alice", "bio": "", "website": "https://youtube.com/@alice",
        "bio_urls": ["https://youtu.be/123", "https://alice.example"],
    })
    monkeypatch.setattr(dp, "_probe_substack", lambda username: [])
    monkeypatch.setattr(dp, "_detect_rss_feed", lambda url: None)

    def get(url, **kwargs):
        if url == "https://api.github.com/users/alice":
            return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {"login": "alice", "blog": "youtube.com/@alice"})
        if url == "https://alice.example":
            return SimpleNamespace(status_code=200, text=(
                '<title>Alice</title><a href="https://youtube.com/@alice">video</a>'
                '<a href="https://youtu.be/123">video</a>'))
        pytest.fail(f"Unexpected fetch: {url}")

    monkeypatch.setattr(dp.requests, "get", get)
    cfg = SimpleNamespace(state_file=lambda name: tmp_path / f"{name}.json")
    result = dp.discover_profile(seed, seed_type=seed_type, config=cfg,
                                 extra_source_urls=["https://youtube.com", "https://youtu.be/123"])

    urls = {s["url"] for s in result["sources"]}
    expected = {"https://alice.example"}
    if seed_type == "x":
        expected.add("https://github.com/alice")
    assert urls == expected
