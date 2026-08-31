"""Tests for url_canon.canonical_identity — the platform-aware trust unit."""

import pytest

from pipeline.ingestion.url_canon import canonical_identity as ci


@pytest.mark.parametrize("url,expected", [
    # Default: bare host, www stripped, scheme dropped, trailing slash gone.
    ("https://karpathy.ai/", "karpathy.ai"),
    ("http://www.karpathy.ai/blog/post", "karpathy.ai"),
    ("karpathy.ai", "karpathy.ai"),                      # no scheme
    # Substack: subdomain is identity.
    ("https://chamath.substack.com/feed", "chamath.substack.com"),
    ("https://chamath.substack.com/p/some-post", "chamath.substack.com"),
    # GitHub: host + first path segment, lowercased.
    ("https://github.com/Karpathy", "github.com/karpathy"),
    ("https://github.com/karpathy/nanoGPT", "github.com/karpathy"),
    # X / Twitter aliasing + path unit.
    ("https://twitter.com/karpathy", "x.com/karpathy"),
    ("https://x.com/karpathy/status/123", "x.com/karpathy"),
    ("https://mobile.twitter.com/karpathy", "x.com/karpathy"),
    # YouTube channel forms.
    ("https://www.youtube.com/@AndrejKarpathy", "youtube.com/@andrejkarpathy"),
    ("https://youtube.com/channel/UCABC123", "youtube.com/channel/ucabc123"),
    ("https://www.youtube.com/watch?v=xyz", "youtube.com"),   # a video, not a channel
    # Medium.
    ("https://medium.com/@someone/post-title", "medium.com/@someone"),
    ("https://someone.medium.com/post", "someone.medium.com"),
])
def test_canonical_identity(url, expected):
    assert ci(url) == expected


def test_squatter_distinct_from_real_account():
    # The whole point: same host, different handle → different trust nodes.
    assert ci("https://github.com/sama") != ci("https://github.com/Sama2")


def test_empty_and_garbage():
    assert ci("") == ""
    assert ci("   ") == ""
    assert ci(None) == ""  # type: ignore[arg-type]
