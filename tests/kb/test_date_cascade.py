"""The blog-footprint DATE cascade — free (no-LLM) sources tried in decreasing exactness:
htmldate → feed <pubDate> → sitemap lastmod → unknown. These prove the ORDERING, the precision label
that travels with each rung, and the canonical-URL keying that lets a feed link match a sitemap url
for the same post.

A Wayback first-snapshot rung is deliberately NOT wired into the live cascade (flaky under burst +
weak on thin-archive sites — see docs). `_wayback_first_seen` + `_WHEN_APPROX` are kept as tested,
reserved scaffolding for a future off-path background date-backfill; the last block here still proves
that helper parses CDX and fails safe, and one cascade test proves the live path never calls it.
Network is monkeypatched — no live fetch.
"""
from __future__ import annotations

from pipeline.kb import ingest_blog as fp

_URL = "https://simonwillison.net/2024/01/scaling-agents"


# ── the cascade ordering + precision labels ──────────────────────────────────

def test_htmldate_wins_over_everything():
    feed = {fp._canon_post_url(_URL): "2020-05-05"}
    assert fp._resolve_post_date("2024-01-15", _URL, "2019-09-09", feed) == ("2024-01-15", "day")


def test_feed_used_when_htmldate_empty():
    feed = {fp._canon_post_url(_URL): "2020-05-05"}
    assert fp._resolve_post_date("", _URL, "", feed) == ("2020-05-05", "day")


def test_feed_outranks_lastmod():
    """A feed pubDate is a real PUBLISH date; lastmod is a MODIFICATION time — feed must win."""
    feed = {fp._canon_post_url(_URL): "2020-05-05"}
    assert fp._resolve_post_date("", _URL, "2019-09-09", feed) == ("2020-05-05", "day")


def test_lastmod_used_when_no_htmldate_no_feed():
    assert fp._resolve_post_date("", _URL, "2019-09-09", {}) == ("2019-09-09", "day")


def test_unknown_when_all_rungs_miss():
    assert fp._resolve_post_date("", _URL, "", {}) == ("", "unknown")


# ── feed map keys by canonical URL (feed link ≠ sitemap url, same post) ───────

def test_feed_map_matches_across_utm_and_trailing_slash(monkeypatch):
    """The feed link carries `?utm_*` + a trailing slash; the sitemap url is clean. Both must
    resolve to ONE key so the cross-reference actually hits."""
    class _Feed:
        entries = [{"link": _URL + "/?utm_source=rss", "published_parsed": (2020, 5, 5, 0, 0, 0, 0, 0, 0)}]
    import sys, types
    fake = types.ModuleType("feedparser")
    fake.parse = lambda u: _Feed()
    monkeypatch.setitem(sys.modules, "feedparser", fake)
    m = fp._feed_date_map("https://simonwillison.net")
    assert m.get(fp._canon_post_url(_URL)) == "2020-05-05"       # clean sitemap url finds it


def test_feed_map_empty_when_no_feed(monkeypatch):
    class _Empty:
        entries = []
    import sys, types
    fake = types.ModuleType("feedparser")
    fake.parse = lambda u: _Empty()
    monkeypatch.setitem(sys.modules, "feedparser", fake)
    assert fp._feed_date_map("https://nofeed.example") == {}


# ── reserved Wayback helper: CDX parsing + fail-safe (for the future backfill) ─

class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _patch_requests(monkeypatch, resp_or_exc):
    import sys, types
    fake = types.ModuleType("requests")
    def _get(url, **kw):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc
    fake.get = _get
    monkeypatch.setitem(sys.modules, "requests", fake)


