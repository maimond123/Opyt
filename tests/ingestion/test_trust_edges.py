"""Tests for trust_edges — per-probe edge extraction, bounded to relevant nodes."""

import json

from pipeline.ingestion.trust_edges import (
    PROFILE_SCOPE,
    edges_from_github,
    edges_from_html,
    profile_links_from_html,
    substack_twitter_handle,
    targets_from_text,
)

# The person's own candidate set — edges may only target these.
RELEVANT = {"nav.al", "naval.substack.com", "github.com/naval", "x.com/naval"}


def _pairs(edges):
    return {(e.source, e.target) for e in edges}


def test_targets_from_text_urls_and_handles():
    ids = targets_from_text("see https://nav.al and follow @naval please")
    assert "nav.al" in ids
    assert "x.com/naval" in ids


def test_github_edges_from_bio_mention_and_blog_field():
    edges = edges_from_github("naval", bio="founder. @naval on X.", blog="https://nav.al", relevant=RELEVANT)
    assert _pairs(edges) == {("github.com/naval", "x.com/naval"), ("github.com/naval", "nav.al")}


def test_github_self_reference_is_dropped():
    # A bio that links its own github must not produce a self-loop edge.
    edges = edges_from_github("naval", bio="me: github.com/naval", blog="", relevant=RELEVANT)
    assert all(e.source != e.target for e in edges)


def test_html_footer_links_become_edges_bounded_to_relevant():
    html = """
      <footer>
        <a href="https://naval.substack.com">Newsletter</a>
        <a href="https://x.com/naval">Twitter</a>
        <a href="https://nytimes.com/some-article">NYT</a>   <!-- irrelevant: dropped -->
      </footer>
    """
    edges = edges_from_html("nav.al", html, RELEVANT)
    assert _pairs(edges) == {("nav.al", "naval.substack.com"), ("nav.al", "x.com/naval")}


def test_squatter_html_linking_unrelated_web_yields_nothing():
    # A page that only links the broader web (nothing in the person's set) → no edges.
    html = '<a href="https://random.com">x</a><a href="https://github.com/someoneelse">y</a>'
    assert edges_from_html("squatter.substack.com", html, RELEVANT) == []


# ── Integration: the full discover→edges→propagate wiring graduates a source ──

def test_compute_trust_x_seed_identity_and_bidirectional_collapse(monkeypatch):
    """X-seeded: the person's declared website (an identity target) reaches T1 via Rule 5,
    and — the COLLAPSE decision (2026-07-21) — a source bidirectional with that DECLARED
    site (not the handle itself) also graduates via relaxed Rule 2. Under a keep-restriction
    form the substack would have fallen to needs-review."""
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")  # bypass __init__ shadow
    from pipeline.ingestion.trust_types import Edge
    ci = dp.canonical_identity
    DS = dp.DiscoveredSource

    profile_info = {"display_name": "Naval", "website": "https://nav.al"}
    root_id = ci("x.com/naval")
    identity_targets = [(ci("https://nav.al"), False)]           # website = declared account
    all_sources = [DS("blog", "https://nav.al"), DS("substack", "https://naval.substack.com")]

    def fake_fetch(sid, url, relevant, timeout=12):
        if "nav.al" in url:
            return [Edge("nav.al", "naval.substack.com")]       # site → candidate
        if "substack" in url:
            return [Edge("naval.substack.com", "nav.al")]       # candidate → site (bidirectional)
        return []

    monkeypatch.setattr(dp, "fetch_landing_edges", fake_fetch)
    # Keep hub expansion hermetic — it isn't what this test is about.
    monkeypatch.setattr(dp, "fetch_profile_links", lambda url, timeout=12: [])
    monkeypatch.setattr(dp, "fetch_substack_twitter_handle", lambda url, timeout=12: None)
    verdicts = dp._compute_trust(
        "naval", root_id, identity_targets, profile_info, [], all_sources)

    assert verdicts[ci("https://nav.al")].trusted               # Rule 5 (declared)
    assert "Identity-attested" in verdicts[ci("https://nav.al")].reasons[0]
    assert verdicts[ci("https://naval.substack.com")].trusted   # Rule 2 off the declared site
    assert "Bidirectional" in verdicts[ci("https://naval.substack.com")].reasons[0]


def test_compute_trust_substack_seed_reaches_t1_with_no_x():
    """The de-X-rooting payoff: a Substack-rooted person's declared userLinks (X, site,
    YouTube) all reach T1 via identity edges — no X profile in the root set at all."""
    import importlib
    dp = importlib.import_module("pipeline.ingestion.discover_profile")
    ci = dp.canonical_identity
    DS = dp.DiscoveredSource

    root_id = ci("https://them.substack.com")
    identity_targets = [
        (ci("x.com/them"), True),          # connected account (verified)
        (ci("https://them.com"), False),   # declared website
        (ci("https://youtube.com/@them"), False),
    ]
    all_sources = [DS("youtube", "https://youtube.com/@them"), DS("blog", "https://them.com")]

    # skip_edge_fetch → no landing fetch, no hub expansion, no cold-start: purely the
    # identity edges → Rule 5 → all declared accounts reach T1 with no X in the root set.
    verdicts = dp._compute_trust(
        "them", root_id, identity_targets, {"display_name": "Them"}, [], all_sources,
        skip_edge_fetch=True)

    for tid, _v in identity_targets:
        assert verdicts[tid].trusted, tid


# ── Unbounded profile extraction — GROW the set (opposite of edges_from_html) ──

_HUB_HTML = """
  <a href="https://github.com/willccbb">GitHub</a>
  <a href="https://github.com/willccbb/verifiers">a repo</a>       <!-- artifact -->
  <a href="https://arxiv.org/abs/2310.12345">a paper</a>          <!-- artifact -->
  <a href="https://willcb.substack.com">Newsletter</a>
  <a href="https://willcb.substack.com/p/some-essay">a post</a>   <!-- artifact -->
  <a href="https://someone-else.com/2024/blog-post">a blog post</a> <!-- not a profile -->
"""


def test_profile_links_keeps_a_github_not_in_any_relevant_set():
    # The precise thing edges_from_html REFUSES: a github not in `relevant`.
    assert edges_from_html("hub.com", _HUB_HTML, set()) == []          # bounded → nothing
    pls = profile_links_from_html(_HUB_HTML)                            # unbounded → keeps it
    kept = {(p.type, p.handle) for p in pls}
    assert ("github", "willccbb") in kept
    assert ("substack", "willcb") in kept


def test_profile_links_drops_artifacts_and_out_of_scope():
    pls = profile_links_from_html(_HUB_HTML)
    assert all(p.is_profile for p in pls)                # no repo / post / paper
    assert all(p.type in PROFILE_SCOPE for p in pls)     # no paper
    # exactly the two profiles, deduped
    assert len(pls) == 2


def test_linkedin_is_never_surfaced_as_a_profile():
    # LinkedIn is a walled garden — never an atom source, no corroborating edges —
    # so it was dropped from PROFILE_SCOPE and must not surface from a hub.
    assert "linkedin" not in PROFILE_SCOPE
    html = ('<a href="https://www.linkedin.com/in/someone">LinkedIn</a>'
            '<a href="https://github.com/someone">GitHub</a>')
    types = {p.type for p in profile_links_from_html(html)}
    assert "linkedin" not in types and "github" in types


# ── Substack _preloads → publication X handle (JS-hidden back-edge) ────────────

def _substack_json_parse_html(handle, key="pub"):
    inner = json.dumps({key: {"twitter_screen_name": handle}})
    return f'<html><script>window._preloads = JSON.parse({json.dumps(inner)});</script></html>'


def _substack_raw_object_html(handle, key="publication"):
    obj = json.dumps({key: {"twitter_screen_name": handle}})
    return f'<html><script>window._preloads = {obj};</script></html>'


def test_substack_handle_from_json_parse_form():
    assert substack_twitter_handle(_substack_json_parse_html("tedgioia")) == "tedgioia"


def test_substack_handle_from_raw_object_form_and_publication_key():
    assert substack_twitter_handle(_substack_raw_object_html("GergelyOrosz")) == "GergelyOrosz"


def test_substack_handle_regex_fallback_on_odd_wrapping():
    # Some other wrapping we don't structurally parse — the field is still there.
    html = '<script>var x = window._preloads; foo("twitter_screen_name":"@naval") </script>'
    assert substack_twitter_handle(html) == "naval"


def test_substack_handle_none_on_malformed_or_absent():
    assert substack_twitter_handle('<script>window._preloads = JSON.parse("broken') is None
    assert substack_twitter_handle('<div>no preloads here</div>') is None
    assert substack_twitter_handle('<script>window._preloads = {"pub":{"name":"x"}};</script>') is None
    assert substack_twitter_handle("") is None
