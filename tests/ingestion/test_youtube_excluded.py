"""Excluded video links cannot become sources, trust evidence, or fetch targets."""

from types import SimpleNamespace

import pytest

from pipeline.platforms.source_classify import classify_source
from pipeline.platforms.url_canon import canonical_identity
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x import profile as x_profile

from pipeline.oracles.discovery import backlink
from pipeline.oracles.discovery import probes
from pipeline.oracles.discovery import shaping
from pipeline.oracles.discovery import profile as dp


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
    assert shaping.classify_url(url) is None


def test_exclusion_respects_host_boundaries():
    url = "https://notyoutube.com"
    assert canonical_identity(url) == "notyoutube.com"
    assert classify_source(url).is_profile


@pytest.mark.parametrize("url", ["https://m.youtube.com/@alice", "https://youtu.be/123"])
def test_existing_blog_seed_cannot_fetch_an_excluded_host(url):
    """These URLs could be minted as blog roots before platform exclusion was centralized."""
    assert probes.probe_blog_profile(url) == ({}, [], [])


def test_hub_links_do_not_emit_youtube_evidence():
    html = ('<a href="https://youtube.com/@alice">video</a>'
            '<a href="https://youtu.be/123">video</a>'
            '<a href="https://github.com/alice">code</a>')
    assert backlink.facts_from_html("alice.example", html).links_to == {"github.com/alice"}


@pytest.mark.parametrize("seed,seed_type", [("alice", "x"), ("https://alice.example", "blog")])
def test_discovery_drops_video_links_without_fetching_them(monkeypatch, seed, seed_type):
    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {})
    monkeypatch.setattr(x_local, "auth_headers", lambda *args: {})
    monkeypatch.setattr(x_profile, "fetch_user_profile", lambda *args: {
        "display_name": "Alice", "bio": "", "website": "https://youtube.com/@alice",
        "bio_urls": ["https://youtu.be/123", "https://alice.example"],
    })

    def get(url, **kwargs):
        if url == "https://alice.example":
            return SimpleNamespace(status_code=200, text=(
                '<title>Alice</title><a href="https://youtube.com/@alice">video</a>'
                '<a href="https://youtu.be/123">video</a>'))
        pytest.fail(f"Unexpected fetch: {url}")

    monkeypatch.setattr(probes.requests, "get", get)
    result = dp.discover_profile(seed, seed_type=seed_type)

    assert result["display_name"]
    assert not any("youtu" in v["url"] for v in result["verdicts"])
