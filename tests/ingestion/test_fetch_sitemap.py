"""_fetch_sitemap_urls resolves RELATIVE <loc> entries to absolute — else the fetch dies with
MissingSchema and a whole archive is unfetchable (guzey.com lists "/balls/", "/links/2019/3/").
Network is stubbed; only the parse/resolve is under test."""
from __future__ import annotations

from pipeline.platforms.web import discover as ing

_SITEMAP = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    '<url><loc>/balls/</loc></url>'                                   # relative, root-absolute path
    '<url><loc>links/2019/3/</loc></url>'                             # relative, no leading slash
    '<url><loc>https://guzey.com/essays/absolute/</loc></url>'        # already absolute
    '</urlset>'
)


class _Resp:
    status_code = 200
    text = _SITEMAP


def test_fetch_sitemap_resolves_relative_locs(monkeypatch):
    monkeypatch.setattr(ing.requests, "get", lambda url, **kw: _Resp())
    out = {e["url"] for e in ing.fetch_sitemap_urls("https://guzey.com")}
    # every stored URL is absolute + fetchable
    assert all(u.startswith("https://guzey.com/") for u in out), out
    assert "https://guzey.com/balls/" in out                          # root-absolute path resolved
    assert "https://guzey.com/links/2019/3/" in out                   # bare relative resolved
    assert "https://guzey.com/essays/absolute/" in out                # absolute passed through


_RSS = (
    '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>'
    '<item><title>a</title><link>https://blog.example/posts/a/</link></item>'
    '<item><title>b</title><link>https://blog.example/posts/b/</link></item>'
    '</channel></rss>'
)


def test_rss_fallback_reads_the_feed_after_sitemaps_fail(monkeypatch):
    # B01: the RSS fallback called an undefined `feed_input`, and its broad except hid the
    # NameError, so a blog with no sitemap never reached its feed.
    class _Missing:
        status_code = 404
        text = ""

    fed = []
    real_parse = ing.feedparser.parse

    def parse(source):
        # Outside managed work `feed_input` hands back the URL itself; serve the feed for it.
        fed.append(source)
        return real_parse(_RSS if source == "https://blog.example/feed.xml" else "")

    monkeypatch.setattr(ing.requests, "get", lambda url, **kw: _Missing())
    monkeypatch.setattr(ing.feedparser, "parse", parse)
    out = [e["url"] for e in ing.fetch_sitemap_urls("https://blog.example")]
    assert fed == ["https://blog.example/feed.xml"]
    assert out == ["https://blog.example/posts/a/", "https://blog.example/posts/b/"]
