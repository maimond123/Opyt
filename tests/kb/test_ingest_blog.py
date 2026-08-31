"""ingest_blog (Stage-5 footprint) — a confirmed Oracle's OWN blog archive → opinion atoms.

Offline: the sitemap discovery + per-post trafilatura fetch are monkeypatched, so these prove
the WIRING — full-body atom build, `blog:{host}` who_id + path-preserving `blog:{host}{path}`
atom_id, policy-B dedup (skip before the paid fetch), the challenge/thin-body skip-and-count,
the `limit` bound, entry_mode,
and attribution via the upserted blog-home link (which exercises the resolve.py `_SELF_PLATFORMS`
fix) — not the live scrape.
"""
from __future__ import annotations

import pytest

from pipeline.kb import derive, ingest_blog as fp
from pipeline.kb import resolve, schema


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
    from pipeline.ingestion.sources import blog as src
    monkeypatch.setattr(src, "_fetch_sitemap_urls", lambda base: [dict(e) for e in entries])
    monkeypatch.setattr(src, "harvest_hub_links", lambda base: [])
    monkeypatch.setattr(src, "_fetch_article", lambda url: articles.get(url))
    # The date cascade's one live NETWORK arm is part of the offline seam: stub the feed
    # cross-reference so these loop tests never touch a live feed. (The Wayback helper is off-path —
    # the cascade never calls it — so it needs no stub here.)
    monkeypatch.setattr(fp, "_feed_date_map", lambda base: {})


# ── the happy path: one post → one full-body footprint atom ───────────────────────

def test_footprint_builds_full_body_atom(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [{"url": _POST_URL, "lastmod": "2024-01-15"}], {_POST_URL: _article()})
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")
    assert out["added"] == 1 and out["challenge_skipped"] == 0 and out["failed"] == 0

    atom = conn.execute("SELECT * FROM atoms WHERE atom_id=?", (_POST_ATOM,)).fetchone()
    assert atom is not None
    assert atom["source_type"] == "blog"
    assert atom["entry_mode"] == "oracle-footprint"      # NOT user-saved / crawled
    assert atom["who_id"] == "blog:simonwillison.net"    # blog HOME, stable across the archive
    assert atom["what_kind"] == "opinion"
    assert atom["when_ts"] == "2024-01-15"

    # Body is chunked clean (no YAML frontmatter chrome); the prose survives.
    body = " ".join(r["text"] for r in
                    conn.execute("SELECT text FROM chunks WHERE atom_id=?", (_POST_ATOM,)))
    assert "Autonomous agents compose small tools" in body
    assert "source: blog" not in body

    who = conn.execute("SELECT who_id FROM atoms WHERE atom_id=?", (_POST_ATOM,)).fetchone()
    assert who["who_id"] == "blog:simonwillison.net"


# ── policy B: the second run skips already-ingested posts BEFORE any re-fetch ──────

def test_footprint_idempotent_policy_b(conn, fake_embedder, monkeypatch):
    _patch(monkeypatch, [{"url": _POST_URL, "lastmod": "2024-01-15"}], {_POST_URL: _article()})
    fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")

    # Second run: the per-post fetch must NOT be called again (policy B skips before it).
    from pipeline.ingestion.sources import blog as src

    def _boom(url):
        raise AssertionError("article re-fetched for an already-ingested footprint post")

    monkeypatch.setattr(src, "_fetch_article", _boom)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")
    assert out["added"] == 0 and out["skipped"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (_POST_ATOM,)).fetchone()[0] == 1


# ── challenge / thin bodies are SKIPPED and counted (never a shell atom) ───────────

def test_footprint_skips_challenge_and_thin_bodies(conn, fake_embedder, monkeypatch):
    cf_url, thin_url = "https://simonwillison.net/cf", "https://simonwillison.net/thin"
    cf_body = (
        "Just a moment... Checking your browser before accessing simonwillison.net. This process "
        "is automatic. Your browser will redirect to your requested content shortly. Please allow "
        "up to five seconds. Please enable JavaScript and cookies to continue, and verify you are "
        "human before we let you through to the page you requested."
    )  # 200–600 chars WITH markers → exercises the marker branch, not just the length floor
    articles = {
        cf_url: {"url": cf_url, "title": "Just a moment...", "date": "", "content": cf_body},
        thin_url: {"url": thin_url, "title": "Hi", "date": "", "content": "too short"},  # < floor
        _POST_URL: _article(),
    }
    _patch(monkeypatch, [{"url": cf_url, "lastmod": ""}, {"url": thin_url, "lastmod": ""},
                         {"url": _POST_URL, "lastmod": "2024-01-15"}], articles)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG)
    assert out["added"] == 1 and out["challenge_skipped"] == 2 and out["failed"] == 0
    assert {r["atom_id"] for r in conn.execute("SELECT atom_id FROM atoms")} == {_POST_ATOM}


def test_footprint_none_article_is_challenge_skip(conn, fake_embedder, monkeypatch):
    # A fetch that extracts nothing (returns None) → skip-and-count, never a partial atom.
    _patch(monkeypatch, [{"url": _POST_URL, "lastmod": ""}], {_POST_URL: None})
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG)
    assert out["added"] == 0 and out["challenge_skipped"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


# ── date cascade end-to-end: a fully-dateless post is stored honestly as `unknown` ──

def test_footprint_dateless_post_is_unknown_precision(conn, fake_embedder, monkeypatch):
    """A real post with NO on-page date, NO feed date, NO sitemap lastmod. With the Wayback rung
    off-path, the cascade has nothing left → the atom stores an empty `when_ts` at `unknown`
    precision (never a fabricated day). Proves precision travels cascade → article → derive_blog → the stored atom."""
    _patch(monkeypatch, [{"url": _POST_URL, "lastmod": ""}],
           {_POST_URL: _article(date="")})               # htmldate found nothing
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")
    assert out["added"] == 1

    atom = conn.execute("SELECT when_ts, when_precision FROM atoms WHERE atom_id=?",
                        (_POST_ATOM,)).fetchone()
    assert atom["when_ts"] == ""
    assert atom["when_precision"] == "unknown"    # honest: no source, not a fabricated day


def test_long_post_mentioning_captcha_is_not_a_challenge(conn, fake_embedder, monkeypatch):
    # The marker gate is length-bounded: a real long essay that DISCUSSES captchas is kept.
    url = "https://simonwillison.net/2024/02/bot-detection"
    long_body = ("This is a long technical essay about bot detection systems and how a captcha "
                 "actually works under the hood. " + "We discuss agent frameworks and tools. " * 30)
    _patch(monkeypatch, [{"url": url, "lastmod": "2024-02-01"}],
           {url: {"url": url, "title": "How CAPTCHAs Work", "date": "2024-02-01",
                  "content": long_body}})
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG)
    assert out["added"] == 1 and out["challenge_skipped"] == 0


# ── the `limit` bound caps NEW atoms ──────────────────────────────────────────────

def test_footprint_limit_caps_new_atoms(conn, fake_embedder, monkeypatch):
    urls = [f"https://simonwillison.net/p/{i}" for i in range(3)]
    _patch(monkeypatch, [{"url": u, "lastmod": "2024-01-15"} for u in urls],
           {u: _article(url=u, title=f"Post {i}") for i, u in enumerate(urls)})
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, limit=2)
    assert out["added"] == 2
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2


# ── ARC-1 Job A: batch the embed across posts (the long-document win) ──────────────

def test_footprint_batches_embed_across_posts(conn, recording_embedder, monkeypatch):
    # Four posts pool into ONE flush = ONE embed call under the batching sink (per-post store_atom
    # paid four). Fetch stays serial (generic web scrape); only the embed is batched.
    urls = [f"{_BLOG}/2024/01/post{i}" for i in range(4)]
    _patch(monkeypatch, [{"url": u, "lastmod": "2024-01-15"} for u in urls],
           {u: _article(url=u, title=f"Post {i}") for i, u in enumerate(urls)})
    out = fp.sync_blog_footprint(conn, recording_embedder, blog_url=_BLOG, handle="simon")
    assert out["added"] == 4 and out["failed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 4
    assert len(recording_embedder.calls) == 1                      # four posts, ONE flush


# ── a hub-harvested post is still authored to its ORIGIN blog ─────────────────────
#
# It also carried a `discovered_via` edge to the hub page (attested=0, a recovery trail). Nothing
# read it, and the `edges` table was deleted 2026-08-23, so only the attribution half survives —
# which is the half that could misattribute someone's writing if it broke.

def test_footprint_hub_post_is_authored_to_the_origin_blog(conn, fake_embedder, monkeypatch):
    # A post found via HUB-HARVEST (not the sitemap): a dated URL → classify_url STRONG → skips
    # triage → discovery hands it up with via = the hub page. Baseline is empty here so the ONLY
    # atom is the hub-found one.
    from pipeline.ingestion.sources import blog as src
    hub_url = "https://simonwillison.net/2024/03/hub-found-post"
    monkeypatch.setattr(src, "_fetch_sitemap_urls", lambda base: [])                 # empty baseline
    monkeypatch.setattr(src, "harvest_hub_links",
                        lambda base: [{"url": hub_url, "anchor": "Hub Post", "via": _BLOG}])
    monkeypatch.setattr(src, "_fetch_article",
                        lambda url: _article(url=hub_url, title="Hub Found") if url == hub_url else None)
    out = fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")
    assert out["added"] == 1

    atom_id = "blog:simonwillison.net/2024/03/hub-found-post"
    who = conn.execute("SELECT who_id FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()["who_id"]
    assert who == derive.blog_entity_id(_BLOG)                # authored to the ORIGIN blog


# ── attribution: resolve unifies the blog author into the Oracle's canonical ───────

def test_footprint_author_resolves_to_oracle_canonical(conn, fake_embedder, monkeypatch):
    # An X Oracle whose website field attests the blog home (the merge edge). This FAILS without
    # the resolve.py fix (blog link would be attests-only → attests∩attests never merges).
    schema.upsert_entity(conn, "x:user:7", name="Simon", identity_links=[_BLOG])
    _patch(monkeypatch, [{"url": _POST_URL, "lastmod": "2024-01-15"}], {_POST_URL: _article()})
    fp.sync_blog_footprint(conn, fake_embedder, blog_url=_BLOG, author_name="Simon")
    resolve.resolve_entities(conn)

    x_cid = schema.get_entity(conn, "x:user:7")["canonical_id"]
    blog_cid = schema.get_entity(conn, "blog:simonwillison.net")["canonical_id"]
    assert x_cid == blog_cid          # one canonical → the footprint attributes to the Oracle
