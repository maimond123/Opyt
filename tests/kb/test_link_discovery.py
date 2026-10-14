"""link_discovery — the blog/website URL funnel, proven fully offline (no network, no model).

  • classify_url — structure DROPs confident junk; every other owned link is kept.
  • union dedup — a post in both sitemap and hub collapses to one entry; baseline wins (via=None).
  • provenance — hub-sourced entries carry via = the hub page (→ discovered_via edge); baseline None.
  • external candidates survive classify_url (auto-attribute follows external links).
"""
from __future__ import annotations

import pytest

from pipeline.platforms.web import discover as ing
from pipeline.platforms.web import links as web_links
from pipeline.kb import link_discovery as ld


# ── classify_url tiers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    # drop — assets / feeds
    ("https://eugeneyan.com/assets/logo.png", "drop"),
    ("https://eugeneyan.com/style.css", "drop"),
    ("https://eugeneyan.com/feed.xml", "drop"),
    ("https://eugeneyan.com/paper.pdf", "drop"),                 # pdf routing parked → drop
    # drop — nav / mailto / bare homepage
    ("https://eugeneyan.com/about/", "drop"),
    ("https://eugeneyan.com/tags/llm/", "drop"),
    ("mailto:hi@eugeneyan.com", "drop"),
    ("https://eugeneyan.com/", "drop"),
    ("https://someexternal.com", "drop"),                        # bare external homepage
    # keep — dated path, post/section segment, or a marker-less slug structure cannot reject
    ("https://sebastianraschka.com/blog/2024/llm-course.html", "keep"),
    ("https://eugeneyan.com/writing/llm-patterns/", "keep"),
    ("https://newsletter.pragmaticengineer.com/p/scaling-stripe", "keep"),
    ("https://zeroknowledge.fm/the-groth16-episode", "keep"),
    ("https://huyenchip.com/index.php/some-flat-slug", "keep"),
    ("https://chrisbarber.co/resonance", "keep"),          # his whole blog lives at flat slugs
])
def test_classify_url_tiers(url, expected):
    assert ld.classify_url(url) == expected


def test_classify_drop_beats_a_post_marker_on_mixed_path():
    # /blog/ looks like a post section, but /tags/ is nav — a tag index must DROP.
    assert ld.classify_url("https://site.com/blog/tags/crypto/") == "drop"


def test_external_article_candidate_survives_classify():
    # An external post (author's guest essay elsewhere) is NOT dropped for being off-domain —
    # auto-attribute follows external links; only structure (not domain) can drop.
    assert ld.classify_url("https://othersite.com/2024/03/guest-essay") == "keep"
    assert ld.classify_url("https://othersite.com/some-flat-essay-slug") == "keep"


# ── is_nav_path: whole-SEGMENT stop-words, not substring prefixes (the willcb.com bug) ──────────

@pytest.mark.parametrize("path,is_nav", [
    # real nav / feed / admin / asset segments still drop
    ("/feed", True), ("/feed/", True), ("/feed.xml", True), ("/feeds/", True), ("/rss", True),
    ("/about/", True), ("/tags/", True), ("/tag/llm/", True), ("/search/", True),
    ("/page/2", True), ("/robots.txt", True), ("/wp-admin/x", True), ("/blog/tags/x/", True),
    # slugs that merely START with a stop-word must NOT be dropped (the bug)
    ("/feedback-loops/", False),        # 'feed' must NOT eat 'feedback-loops'
    ("/tags-and-tricks/", False),       # 'tags' is a whole segment; this slug is not it
    ("/searching/", False),             # 'search' must NOT eat 'searching'
    ("/authoritative/", False),         # 'author' must NOT eat 'authoritative'
    ("/node.js-guide/", False),         # '.js' must NOT eat 'node.js-guide'
    ("/mypage/2", False),               # '/page/\\d' must NOT eat '/mypage/2'
    ("/2024/03/my-post/", False),
])
def test_is_nav_path_whole_segment(path, is_nav):
    assert ing.is_nav_path(path) is is_nav


def test_classify_url_stopword_prefix_slug_survives():
    # End-to-end: the exact willcb.com casualty — a real post whose slug starts with 'feed'.
    assert ld.classify_url("https://willcb.com/blog/feedback-loops/") == "keep"
    assert ld.classify_url("https://site.com/feedback-loops/") == "keep"            # NOT dropped


# ── shared fixtures: stub the network (baseline + hub) and the LLM ──────────────

@pytest.fixture()
def stub_network(monkeypatch):
    """Stub the two network calls discover_candidate_urls makes (imported lazily FROM
    pipeline.platforms.web.discover inside the function, so patch them on that module)."""
    def _set(baseline, hub):
        monkeypatch.setattr(ing, "fetch_sitemap_urls", lambda base: [dict(e) for e in baseline])
        monkeypatch.setattr(ing, "harvest_hub_links", lambda base: [dict(e) for e in hub])
        # Level-2 crawl is off by default (no real fetch); a test that exercises it overrides this.
        monkeypatch.setattr(ing, "harvest_links_from", lambda pages: [])
    return _set


def _level2(index_to_children):
    """Stub ``harvest_links_from``: for each index page fetched, return its child links tagged with
    ``via`` = that page (mirrors the real harvest's provenance)."""
    def fn(pages):
        out = []
        for p in pages:
            for child in index_to_children.get(p, []):
                out.append({**child, "via": p})
        return out
    return fn


# ── union dedup + provenance ────────────────────────────────────────────────────

def test_union_dedup_baseline_wins(stub_network, monkeypatch):
    # Same post appears in BOTH sitemap and hub → one entry, from the baseline (via=None). The
    # hub's other link is a new candidate that IS added.
    stub_network(
        baseline=[{"url": "https://eugeneyan.com/writing/patterns/", "lastmod": "2024-01-01"}],
        hub=[
            {"url": "https://eugeneyan.com/writing/patterns/", "anchor": "Patterns", "via": "https://eugeneyan.com/"},
            {"url": "https://eugeneyan.com/writing/evals/", "anchor": "Evals", "via": "https://eugeneyan.com/"},
        ],
    )
    out = ld.discover_candidate_urls("https://eugeneyan.com", author_name="Eugene Yan")

    by_url = {e["url"]: e for e in out}
    assert len(out) == 2                                        # dup collapsed, not duplicated
    base = by_url["https://eugeneyan.com/writing/patterns/"]
    assert base["via"] is None and base["source"] == "sitemap"  # baseline wins the dup
    extra = by_url["https://eugeneyan.com/writing/evals/"]
    assert extra["source"] == "hub" and extra["via"] == "https://eugeneyan.com/"


def test_canon_strips_tracking_keeps_identity():
    # Tracking params (utm_*, fbclid, …) are noise → stripped; a real identity param (?p=123) stays.
    from pipeline.kb.ingest_blog import canon_post_url
    assert (canon_post_url("https://zeroknowledge.fm/zksummit14/?utm_source=rss&utm_medium=rss")
            == canon_post_url("https://zeroknowledge.fm/zksummit14/"))
    assert canon_post_url("https://wp.example.com/?p=123") == "blog:wp.example.com?p=123"
    assert canon_post_url("https://x.com/a?fbclid=zz&p=5") == "blog:x.com/a?p=5"


def test_union_dedup_ignores_tracking_params(stub_network, monkeypatch):
    # The exact bug the live dry-run caught: RSS baseline carries `?utm_*`, the hub link is clean.
    # They must collapse to ONE entry (baseline wins), not fetch/store the post twice.
    stub_network(
        baseline=[{"url": "https://zeroknowledge.fm/zksummit14/?utm_source=rss&utm_campaign=x",
                   "lastmod": ""}],
        hub=[{"url": "https://zeroknowledge.fm/zksummit14/", "anchor": "ZK Summit",
              "via": "https://zeroknowledge.fm/blog"}],
    )
    out = ld.discover_candidate_urls("https://zeroknowledge.fm")
    assert len(out) == 1 and out[0]["source"] == "sitemap"             # collapsed, baseline wins


# ── ownership guard + cross-host mirror dedup ───────────────────────────────────

@pytest.mark.parametrize("url,origin,owned", [
    ("https://gajesh.com/2024/post", "gajesh.com", True),                          # same host
    ("https://www.gajesh.com/2024/post", "gajesh.com", True),                      # www stripped
    ("https://magazine.sebastianraschka.com/p/x", "sebastianraschka.com", True),   # subdomain = owned
    ("https://theblock.co/post/1", "gajesh.com", False),                           # press ABOUT them
    ("https://henrydowling.com/x.html", "joinstash.ai", False),                    # someone else
    ("https://notgajesh.com/p", "gajesh.com", False),                              # leading-dot guard: NOT a subdomain
    ("https://evil-gajesh.com/p", "gajesh.com", False),                            # suffix ≠ subdomain
])
def test_is_owned(url, origin, owned):
    assert ld._is_owned(url, origin) is owned


def test_discover_drops_external_press_keeps_owned(stub_network, monkeypatch):
    # gajesh.com links to press ABOUT him (theblock.co) + one of his OWN posts. Auto-attribute must
    # keep only the owned post; the press is dropped, never minted as his opinion atom.
    stub_network(
        baseline=[],
        hub=[{"url": "https://theblock.co/post/gajesh", "anchor": "The Block on Gajesh", "via": "https://gajesh.com"},
             {"url": "https://gajesh.com/2024/my-defi-post", "anchor": "My DeFi Post", "via": "https://gajesh.com"}],
    )
    out = ld.discover_candidate_urls("https://gajesh.com")
    assert {e["url"] for e in out} == {"https://gajesh.com/2024/my-defi-post"}      # press dropped


def test_discover_reaches_the_authors_work_on_a_host_they_do_not_own(stub_network, monkeypatch):
    """The corpus miss this arm exists for. Measured on karpathy.ai: the host rule alone
    kept 3 of his 55 own links, because his essays are on `karpathy.github.io` and his project
    pages under `cs.stanford.edu/people/karpathy/`, and neither is a subdomain of his site.

    The press link is in the same batch on purpose — the arm must widen recall without admitting
    writing ABOUT him."""
    stub_network(
        baseline=[],
        hub=[{"url": "https://cs.stanford.edu/people/karpathy/advice.html", "anchor": "undergrads",
              "via": "https://karpathy.ai"},
             {"url": "https://karpathy.github.io/2019/04/25/recipe/", "anchor": "A Recipe",
              "via": "https://karpathy.ai"},
             {"url": "https://www.wired.com/2015/01/karpathy/", "anchor": "Wired article",
              "via": "https://karpathy.ai"}],
    )

    out = ld.discover_candidate_urls("https://karpathy.ai", author_name="Andrej Karpathy")

    assert {e["url"] for e in out} == {"https://cs.stanford.edu/people/karpathy/advice.html",
                                       "https://karpathy.github.io/2019/04/25/recipe/"}


def test_without_a_name_the_guard_is_exactly_the_host_rule(stub_network, monkeypatch):
    """`tokens` empty ⇒ the host arm alone. The origin host label still yields a token, so this
    pins the case where NEITHER input names the person and the function must not widen."""
    stub_network(
        baseline=[],
        hub=[{"url": "https://cs.stanford.edu/people/karpathy/advice.html", "anchor": "undergrads",
              "via": "https://blog.example.com"}],
    )

    out = ld.discover_candidate_urls("https://blog.example.com")

    assert out == []


def test_discover_keeps_owned_subdomain(stub_network, monkeypatch):
    # A custom subdomain of the author's site (his Substack magazine) IS owned → kept + attributed.
    stub_network(baseline=[],
                 hub=[{"url": "https://magazine.sebastianraschka.com/p/llm-architectures",
                       "anchor": "LLM Architectures", "via": "https://sebastianraschka.com"}])
    out = ld.discover_candidate_urls("https://sebastianraschka.com")
    assert [e["url"] for e in out] == ["https://magazine.sebastianraschka.com/p/llm-architectures"]
    assert out[0]["source"] == "hub"


def test_discover_collapses_cross_host_mirror(stub_network, monkeypatch):
    # Sitemap lists vitalik.ca; the homepage links the vitalik.eth.limo MIRROR of the SAME post
    # (same path, different host) → collapse onto the baseline, don't double-ingest under two who_id.
    stub_network(
        baseline=[{"url": "https://vitalik.ca/general/2024/03/foo.html", "lastmod": ""}],
        hub=[{"url": "https://vitalik.eth.limo/general/2024/03/foo.html", "anchor": "Foo",
              "via": "https://vitalik.eth.limo"}],
    )
    out = ld.discover_candidate_urls("https://vitalik.eth.limo")
    assert [e["url"] for e in out] == ["https://vitalik.ca/general/2024/03/foo.html"]   # one atom
    assert out[0]["source"] == "sitemap"


def test_hub_index_pages_are_not_candidates(stub_network, monkeypatch):
    # A homepage link back to /blog (an index page) must not become an article candidate.
    stub_network(
        baseline=[],
        hub=[{"url": "https://site.com/blog", "anchor": "Blog", "via": "https://site.com/"},
             {"url": "https://site.com/blog/real-post/", "anchor": "Real Post", "via": "https://site.com/"}],
    )
    out = ld.discover_candidate_urls("https://site.com")
    urls = {e["url"] for e in out}
    assert "https://site.com/blog" not in urls                  # the index page itself is excluded
    assert "https://site.com/blog/real-post/" in urls           # a post under it survives


def test_baseline_is_always_kept(stub_network, monkeypatch):
    # Baseline is the author's declared pages — it passes through untouched.
    stub_network(baseline=[{"url": "https://site.com/flat-slug", "lastmod": ""}], hub=[])
    out = ld.discover_candidate_urls("https://site.com")
    assert [e["url"] for e in out] == ["https://site.com/flat-slug"]
    assert out[0]["source"] == "sitemap"


# ── marker-less owned links are kept (chrisbarber.co: 0 sitemap, 52 hub posts) ──────────────

def test_marker_less_owned_hub_links_are_kept(stub_network):
    stub_network(
        baseline=[],
        hub=[{"url": "https://chrisbarber.co/resonance", "anchor": "Resonance",
              "via": "https://chrisbarber.co/start-here"},
             {"url": "https://chrisbarber.co/agi-prep", "anchor": "AGI prep",
              "via": "https://chrisbarber.co/start-here"},
             {"url": "https://chrisbarber.co/about", "anchor": "About",
              "via": "https://chrisbarber.co/start-here"}],
    )
    out = ld.discover_candidate_urls("https://chrisbarber.co")
    assert {e["url"] for e in out} == {"https://chrisbarber.co/resonance",
                                       "https://chrisbarber.co/agi-prep"}
    assert all(e["source"] == "hub" for e in out)


def test_hub_harvest_failure_degrades_to_baseline(stub_network, monkeypatch):
    # If hub-harvest raises, the baseline must still run untouched (degrade to the old sitemap path).
    monkeypatch.setattr(ing, "fetch_sitemap_urls",
                        lambda base: [{"url": "https://site.com/2024/post/", "lastmod": ""}])
    def boom(base):
        raise RuntimeError("homepage 503")
    monkeypatch.setattr(ing, "harvest_hub_links", boom)
    # Level-2 index crawling is a SEPARATE function from `harvest_hub_links` and ran unmocked here,
    # fetching https://site.com/2024/post/ for real on every run (found when the network
    # guard grew a `requests` seam). Stubbing it keeps this test about the level-1 degradation it
    # names — a live level-2 crawl of a nonexistent host only adds latency and nondeterminism.
    monkeypatch.setattr(ing, "harvest_links_from", lambda pages: [])   # patched at its DEFINITION:
    # `link_discovery` imports it inside the function body, so the lookup re-resolves per call.
    out = ld.discover_candidate_urls("https://site.com")
    assert [e["url"] for e in out] == ["https://site.com/2024/post/"]
    assert out[0]["source"] == "sitemap" and out[0]["via"] is None


# ── extract_links / harvest anchor capture (the scraper the funnel rides on) ────

def test_extract_links_captures_anchor_and_all_domains():
    html = ('<a href="/writing/x/">Internal <b>Post</b></a>'
            '<a href="https://other.com/essay">External</a>'
            '<a href="#top">skip anchor</a><a href="mailto:x@y.com">mail</a>')
    links = web_links.extract_links(html, "https://eugeneyan.com/")
    by_url = {l["url"]: l["anchor"] for l in links}
    assert by_url["https://eugeneyan.com/writing/x/"] == "Internal Post"   # relative resolved + nested text
    assert "https://other.com/essay" in by_url                            # external kept
    assert not any(u.startswith(("#", "mailto:")) for u in by_url)        # in-page + mailto skipped


# ── depth-1 second level: _is_index_path + the crawl through the funnel ──────────

@pytest.mark.parametrize("url,is_index", [
    ("https://gajesh.com/writings", True),               # plural section word — the gajesh gap
    ("https://site.com/writing", True),                  # singular
    ("https://site.com/blog/archive", True),             # 2-seg, last is a section word
    ("https://site.com/essays/", True),                  # trailing slash tolerated
    ("https://site.com/writings.html", True),            # .html section page
    ("https://gajesh.com/writings/entropic", False),     # a real essay under the index — NOT an index
    ("https://site.com/2024/03/my-post", False),         # dated article
    ("https://site.com/some-flat-slug", False),          # marker-less slug
    ("https://site.com/", False),                        # homepage
    ("https://site.com/blog/tags/crypto/foo", False),    # too deep (>2 segments)
])
def test_is_index_path(url, is_index):
    assert ld._is_index_path(url) is is_index


def test_level2_crawls_index_children_through_funnel(stub_network, monkeypatch):
    # gajesh case: the sitemap has only the /writings INDEX (index-shaped, owned). Its essays live at
    # /writings/{slug} — reachable from neither sitemap nor level-1 hub. Level-2 crawls /writings and
    # its children flow through the funnel and are kept, tagged via = the index page.
    stub_network(baseline=[{"url": "https://gajesh.com/writings", "lastmod": ""}], hub=[])
    monkeypatch.setattr(ing, "harvest_links_from", _level2({
        "https://gajesh.com/writings": [
            {"url": "https://gajesh.com/writings/entropic", "anchor": "On entropy, curiosity"},
            {"url": "https://gajesh.com/writings/everything", "anchor": "Life Lessons"},
        ]}))
    out = ld.discover_candidate_urls("https://gajesh.com", author_name="Gajesh")
    urls = {e["url"] for e in out}
    assert "https://gajesh.com/writings/entropic" in urls        # essay reached one hop deeper
    assert "https://gajesh.com/writings/everything" in urls
    essay = next(e for e in out if e["url"].endswith("/entropic"))
    assert essay["source"] == "hub" and essay["via"] == "https://gajesh.com/writings"


def test_level2_respects_ownership_guard(stub_network, monkeypatch):
    # A level-2 index links to external press + an owned post. The ownership guard STILL applies to
    # level-2 links (they run the same funnel): the press is dropped, only the owned essay survives.
    stub_network(baseline=[{"url": "https://gajesh.com/writings", "lastmod": ""}], hub=[])
    monkeypatch.setattr(ing, "harvest_links_from", _level2({
        "https://gajesh.com/writings": [
            {"url": "https://forbes.com/gajesh-profile", "anchor": "Forbes on Gajesh"},
            {"url": "https://gajesh.com/writings/real", "anchor": "Real Essay"},
        ]}))
    out = ld.discover_candidate_urls("https://gajesh.com")
    urls = {e["url"] for e in out}
    assert "https://gajesh.com/writings/real" in urls
    assert "https://forbes.com/gajesh-profile" not in urls        # external press still dropped


def test_level2_respects_mirror_dedup(stub_network, monkeypatch):
    # A level-2 index links a cross-host MIRROR of a baseline post → still collapses onto the baseline
    # (mirror dedup applies to level-2 links too), not double-ingested.
    stub_network(
        baseline=[{"url": "https://vitalik.ca/general/2024/03/foo.html", "lastmod": ""},
                  {"url": "https://vitalik.ca/writings", "lastmod": ""}],       # index-shaped seed
        hub=[])
    monkeypatch.setattr(ing, "harvest_links_from", _level2({
        "https://vitalik.ca/writings": [
            {"url": "https://vitalik.eth.limo/general/2024/03/foo.html", "anchor": "Foo mirror"}]}))
    out = ld.discover_candidate_urls("https://vitalik.ca")
    assert [e["url"] for e in out
            if e["url"].endswith("/foo.html")] == ["https://vitalik.ca/general/2024/03/foo.html"]


def test_level2_is_depth_1_not_recursive(stub_network, monkeypatch):
    # Level-2 must NOT recurse: an index-shaped link FOUND at level 2 is a candidate, not a third
    # fetch. harvest_links_from is called exactly ONCE, for the level-1-seeded index pages only.
    calls = []
    def fn(pages):
        calls.append(list(pages))
        return [{"url": "https://gajesh.com/blog/archive", "anchor": "Archive", "via": pages[0]}]
    stub_network(baseline=[{"url": "https://gajesh.com/writings", "lastmod": ""}], hub=[])
    monkeypatch.setattr(ing, "harvest_links_from", fn)
    ld.discover_candidate_urls("https://gajesh.com")
    assert len(calls) == 1 and calls[0] == ["https://gajesh.com/writings"]   # one pass, no recursion


def test_level2_bounded_by_max_index_pages(stub_network, monkeypatch):
    # More index-shaped seeds than the budget → only _MAX_INDEX_PAGES are crawled (bounded fan-out).
    words = ["writings", "thoughts", "words", "talks", "projects", "stories",
             "journal", "reading", "writes", "logs"]                          # 10, none a probe path
    stub_network(baseline=[{"url": f"https://site.com/{w}", "lastmod": ""} for w in words], hub=[])
    seen_pages = []
    def fn(pages):
        seen_pages.extend(pages)
        return []
    monkeypatch.setattr(ing, "harvest_links_from", fn)
    ld.discover_candidate_urls("https://site.com")
    assert len(seen_pages) == ld._MAX_INDEX_PAGES


def test_level2_failure_degrades_to_level1(stub_network, monkeypatch):
    # If the level-2 crawl raises, the level-1 candidates (baseline + hub) survive untouched.
    stub_network(
        baseline=[{"url": "https://gajesh.com/writings", "lastmod": ""},         # triggers level-2
                  {"url": "https://gajesh.com/2024/real-post", "lastmod": ""}],
        hub=[])
    def boom(pages):
        raise RuntimeError("index 503")
    monkeypatch.setattr(ing, "harvest_links_from", boom)
    out = ld.discover_candidate_urls("https://gajesh.com")
    assert "https://gajesh.com/2024/real-post" in {e["url"] for e in out}    # baseline intact


def test_level2_excludes_probe_hubs_not_recrawled(stub_network, monkeypatch):
    # /blog is already crawled at level 1 (it's a probe path) → it must NOT be re-fetched at level 2.
    # Only a NON-probe index-shaped page (/writings) is handed to the level-2 crawl.
    stub_network(
        baseline=[{"url": "https://site.com/blog", "lastmod": ""},        # probe hub — level-1 already
                  {"url": "https://site.com/writings", "lastmod": ""}],   # non-probe index — level-2
        hub=[])
    seen_pages = []
    monkeypatch.setattr(ing, "harvest_links_from", lambda pages: seen_pages.extend(pages) or [])
    ld.discover_candidate_urls("https://site.com")
    assert seen_pages == ["https://site.com/writings"]                   # /blog excluded, /writings crawled
