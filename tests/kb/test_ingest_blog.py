"""ingest_blog — one article's landing url and the atom id it keys on (`prepare_article`).

The archive walk's per-post builder is covered with the walk (`test_archive_adapters.py`).
Offline: the per-post fetch is monkeypatched.
"""
from __future__ import annotations

import pytest

from pipeline.kb import ingest_blog as fp
from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


_BLOG = "https://simonwillison.net"
_POST_URL = "https://simonwillison.net/2024/01/scaling-agents"
_POST_ATOM = "blog:simonwillison.net/2024/01/scaling-agents"
_CONTENT = (
    "Autonomous agents compose small tools into larger systems, and the interesting engineering "
    "question is how you decompose a task into steps an agent can actually execute reliably. This "
    "post walks through a small agent framework I have been building and the tradeoffs it makes "
    "around retries, timeouts, and idempotency. "
    "See [the paper](https://arxiv.org/abs/2401.00001) for the underlying framework, and this "
    "diagram ![chart](https://cdn.example.com/chart.png) for the control flow."
)


def _article(url=_POST_URL, title="Scaling Agents", date="2024-01-15", content=_CONTENT):
    return {"url": url, "title": title, "date": date, "content": content}


def _patch(monkeypatch, entries, articles):
    """Patch the discovery arms + the per-post trafilatura fetch. `entries` = [{url, lastmod}];
    `articles` maps url → article dict (or None → the fetch extracts nothing).

    Discovery now unions the sitemap baseline with a hub-harvest arm (link_discovery). These loop
    tests only care about the baseline, so `harvest_hub_links` is stubbed to `[]` — union discovery
    then degrades to exactly the patched sitemap entries (no network, no LLM triage)."""
    from pipeline.platforms.web import discover as src
    from pipeline.platforms.web import article as web_article
    monkeypatch.setattr(src, "fetch_sitemap_urls", lambda base: [dict(e) for e in entries])
    monkeypatch.setattr(src, "harvest_hub_links", lambda base: [])
    monkeypatch.setattr(web_article, "fetch_article", lambda url: articles.get(url))
    # The date cascade's one live NETWORK arm is part of the offline seam: stub the feed
    # cross-reference so these loop tests never touch a live feed. (The Wayback helper is off-path —
    # the cascade never calls it — so it needs no stub here.)
    monkeypatch.setattr(fp, "_feed_date_map", lambda base: {})


# ── the happy path: one post → one full-body footprint atom ───────────────────────


# ── policy B: the second run skips already-ingested posts BEFORE any re-fetch ──────


# ── challenge / thin bodies are SKIPPED and counted (never a shell atom) ───────────


# ── date cascade end-to-end: a fully-dateless post is stored honestly as `unknown` ──


# ── the `limit` bound caps NEW atoms ──────────────────────────────────────────────


# ── ARC-1 Job A: batch the embed across posts (the long-document win) ──────────────


# ── a hub-harvested post is still authored to its ORIGIN blog ─────────────────────
#
# It also carried a `discovered_via` edge to the hub page (attested=0, a recovery trail). Nothing
# read it, and there is no `edges` table, so only the attribution half exists —
# which is the half that could misattribute someone's writing if it broke.


# ── attribution: resolve unifies the blog author into the Oracle's canonical ───────


# ── prepare_article: the network half, and where a redirect leaves the key ─────────────

@pytest.mark.parametrize("landed_suffix", ["/", ":443"])
def test_a_landing_that_canonicalizes_the_same_does_not_rekey(monkeypatch, landed_suffix):
    """A trailing slash, or the explicit port `otter.ai` → `otter.ai:443/` showed live, is the same
    post under the same id."""
    from pipeline.platforms.web import article as web_article
    from urllib.parse import urlparse
    p = urlparse(_POST_URL)
    landed = (f"{p.scheme}://{p.netloc}:443{p.path}" if landed_suffix == ":443"
              else _POST_URL + "/")
    monkeypatch.setattr(web_article, "fetch_article", lambda url: {**_article(url=url),
                                                             "final_url": landed})
    monkeypatch.setattr(fp, "_feed_date_map", lambda base: {})
    prep = fp.prepare_article(_POST_URL, entry_mode="author_referenced", seen={})
    assert prep["status"] == "ready" and prep["atom_id"] == fp.canon_post_url(_POST_URL)
    assert prep["url"] == _POST_URL


def test_a_same_host_redirect_to_the_real_slug_rekeys(monkeypatch):
    """`/p/123` → `/the-real-slug` is ONE post; keyed by the asked-for url it would sit beside the
    atom a direct link to the slug mints."""
    from pipeline.platforms.web import article as web_article
    monkeypatch.setattr(web_article, "fetch_article", lambda url: {**_article(url=url),
                                                             "final_url": _POST_URL})
    monkeypatch.setattr(fp, "_feed_date_map", lambda base: {})
    asked = _POST_URL.rsplit("/", 1)[0] + "/?p=123"
    prep = fp.prepare_article(asked, entry_mode="author_referenced", seen={})
    assert prep["atom_id"] == fp.canon_post_url(_POST_URL) and prep["url"] == _POST_URL


def test_a_redirect_onto_a_stored_post_is_present(monkeypatch):
    from pipeline.platforms.web import article as web_article
    landed = "https://news.example.com/a-story"
    monkeypatch.setattr(web_article, "fetch_article", lambda url: {**_article(url=url),
                                                             "final_url": landed})
    prep = fp.prepare_article("https://bit.ly/xyz", entry_mode="author_referenced",
                              seen={fp.canon_post_url(landed): "h"})
    assert (prep["status"], prep["atom_id"]) == ("present", "blog:news.example.com/a-story")
