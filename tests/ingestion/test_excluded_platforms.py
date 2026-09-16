"""Retired platforms cannot enter discovery through profiles, hubs, or saved roots."""

import importlib

import pytest

from pipeline.ingestion import trust_edges
from pipeline.ingestion.source_classify import classify_source
from pipeline.ingestion.url_canon import canonical_identity
from pipeline.kb.oracles import _unsupported_root

dp = importlib.import_module("pipeline.ingestion.discover_profile")


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/in/alice",
    "https://linkedin.com/company/example",
    "https://uk.linkedin.com/in/alice",
    "https://open.spotify.com/show/123",
    "https://spotify.com/episode/123",
    "https://podcasts.apple.com/us/podcast/example/id123",
    "https://apple.co/123",
    "https://pod.link/123",
    "https://overcast.fm/123",
    "https://pca.st/123",
])
def test_excluded_urls_cannot_be_discovered_or_fetched_as_blogs(url):
    assert canonical_identity(url) == ""
    assert classify_source(url) is None
    assert dp._classify_url(url) is None
    assert dp._sources_from_urls([url]) == []
    assert dp._probe_blog_profile(url) == ({}, [], [])
    assert _unsupported_root(url)
    html = f'<a href="{url}">profile</a><a href="https://alice.example">writing</a>'
    assert [link.url for link in trust_edges.profile_links_from_html(html)] == [
        "https://alice.example"]


@pytest.mark.parametrize("url", [
    "https://notlinkedin.com",
    "https://notspotify.com",
    "https://apple.com",
    "https://podcasts.example.com",
])
def test_exclusion_matches_platform_hosts_only(url):
    assert canonical_identity(url)
    assert classify_source(url).type == "blog"


@pytest.mark.parametrize("url", [
    "https://scholar.google.com/citations?user=alice",
    "https://www.semanticscholar.org/author/Alice/123",
    "https://orcid.org/0000-0002-1825-0097",
    "https://dblp.org/pid/123/456.html",
    "https://alice.academia.edu",
    "https://www.researchgate.net/profile/Alice",
])
def test_academic_profiles_remain_discovery_leads(url):
    profile = classify_source(url)
    assert profile.is_profile
    assert profile.type in {"scholar", "orcid"}
    assert canonical_identity(url)
