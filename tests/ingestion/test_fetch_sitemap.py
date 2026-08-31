"""_fetch_sitemap_urls resolves RELATIVE <loc> entries to absolute — else the fetch dies with
MissingSchema and a whole archive is unfetchable (guzey.com lists "/balls/", "/links/2019/3/").
Network is stubbed; only the parse/resolve is under test."""
from __future__ import annotations

from pipeline.ingestion.sources import blog as ing

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
    out = {e["url"] for e in ing._fetch_sitemap_urls("https://guzey.com")}
    # every stored URL is absolute + fetchable
    assert all(u.startswith("https://guzey.com/") for u in out), out
    assert "https://guzey.com/balls/" in out                          # root-absolute path resolved
    assert "https://guzey.com/links/2019/3/" in out                   # bare relative resolved
    assert "https://guzey.com/essays/absolute/" in out                # absolute passed through
