"""Tests for hub expansion — a TRUSTED blog/Substack surfacing the Oracle's other
profiles, re-propagation, and the additive handle-match auto-trust layer.

Design note surfaced while writing these: a *purely* blog-surfaced node gets
exactly ONE inbound trusted edge (the hub) and no edge back to a root, so the
graph rules (2/3) cannot graduate it in a single hop — handle-match is its only
path. Re-propagation's job is (a) to evaluate the new nodes and never regress the
existing verdicts, and (b) to let an edge the expansion recovers (the Substack
_preloads back-edge) participate in the standard rules for *existing* candidates.
The tests below assert exactly that, rather than a Rule-2 graduation of a
blog-only node that can't actually happen.
"""

import importlib

from pipeline.ingestion.source_classify import ProfileLink
from pipeline.ingestion.trust_types import Edge, TrustEvidence
from pipeline.ingestion.url_canon import canonical_identity

dp = importlib.import_module("pipeline.ingestion.discover_profile")  # bypass __init__ shadow
DS = dp.DiscoveredSource


def _neutralize(monkeypatch, *, links=None, sub_handle=None):
    """Silence the network: landing fetches → [], and profile-link / substack
    fetches → whatever the test injects (keyed off the hub url containing 'willcb'
    unless the test overrides)."""
    monkeypatch.setattr(dp, "fetch_landing_edges", lambda *a, **k: [])
    monkeypatch.setattr(dp, "fetch_profile_links", links or (lambda url, timeout=12: []))
    monkeypatch.setattr(dp, "fetch_substack_twitter_handle",
                        sub_handle or (lambda url, timeout=12: None))


def _blog_root(url="https://willcb.com"):
    """A blog the X profile attests → a trusted root, i.e. a trusted hub."""
    return {"website": url}, [DS("blog", url)], [DS("blog", url)]


def _ct(username, info, ts, gh, all_sources, **kw):
    """Call the platform-agnostic _compute_trust with the X-root contract derived
    from an old-style (profile_info + twitter_sources): the X handle is the root, and
    its bio/website links are declared identity targets (via the SAME production
    `_x_identity_targets` the discover_profile X path uses — no logic duplication)."""
    root_id = canonical_identity(f"x.com/{username}")
    targets = dp._x_identity_targets(info or {}, ts)
    return dp._compute_trust(username, root_id, targets, info, gh, all_sources, **kw)


# ── Headline: blog-only github, handle matches the Oracle → auto-trust ─────────

def test_blog_surfaced_github_autotrusted_by_handle_match(monkeypatch):
    gh = ProfileLink("github", True, "willccbb", "personal", "https://github.com/willccbb")
    _neutralize(monkeypatch, links=lambda url, timeout=12: [gh] if "willcb.com" in url else [])
    info, ts, all_sources = _blog_root()

    verdicts = _ct("willccbb", info, ts, [], all_sources)

    ev = verdicts["github.com/willccbb"]
    assert ev.trusted
    assert "willccbb" in ev.reasons[0]                       # names the matched handle
    assert ev.edges and ev.edges[0]["via"] == "handle_match"
    surfaced = [s for s in all_sources if (s.metadata or {}).get("discovered_via") == "blog_hub"]
    assert len(surfaced) == 1 and surfaced[0].source_type == "github"
    assert surfaced[0].metadata["hub"] == "https://willcb.com"


# ── Strictness: a namesake mismatch / generic / org handle never auto-trusts ──

def test_surfaced_github_mismatched_handle_stays_needs_review(monkeypatch):
    gh = ProfileLink("github", True, "randomsquatter", "personal", "https://github.com/randomsquatter")
    _neutralize(monkeypatch, links=lambda url, timeout=12: [gh] if "willcb.com" in url else [])
    info, ts, all_sources = _blog_root()

    verdicts = _ct("willccbb", info, ts, [], all_sources)

    assert not verdicts["github.com/randomsquatter"].trusted     # strict → needs review
    # ...but it is still surfaced + recorded, so a human can confirm in one click.
    assert any((s.metadata or {}).get("discovered_via") == "blog_hub" for s in all_sources)


def test_surfaced_generic_handle_not_autotrusted(monkeypatch):
    gh = ProfileLink("github", True, "blog", "personal", "https://github.com/blog")
    _neutralize(monkeypatch, links=lambda url, timeout=12: [gh] if "willcb.com" in url else [])
    info, ts, all_sources = _blog_root()
    verdicts = _ct("willccbb", info, ts, [], all_sources)
    assert not verdicts["github.com/blog"].trusted              # 'blog' is generic


def test_surfaced_scholar_autotrusted_from_hub(monkeypatch):
    # A research profile the Oracle links from their own trusted blog is trusted on
    # provenance ALONE — no handle/name match. Academic pages can't graduate via the
    # graph (they never link back), so from-hub is the only signal, and it suffices.
    sch = ProfileLink("scholar", True, "abcXYZ", "personal",
                      "https://scholar.google.com/citations?user=abcXYZ")
    _neutralize(monkeypatch, links=lambda url, timeout=12: [sch] if "willcb.com" in url else [])
    info, ts, all_sources = _blog_root()
    verdicts = _ct("willccbb", info, ts, [], all_sources)
    ev = verdicts[canonical_identity(sch.url)]
    assert ev.trusted
    assert "Research profile" in ev.reasons[0]
    assert ev.edges[0]["via"] == "research_hub_link"


def test_surfaced_org_shape_not_autotrusted_even_if_handle_matches(monkeypatch):
    # username == "anthropics" so the handle WOULD match — org shape must still block.
    org = ProfileLink("github", True, "anthropics", "org", "https://github.com/orgs/anthropics")
    _neutralize(monkeypatch, links=lambda url, timeout=12: [org] if "anthropic.com" in url else [])
    info, ts, all_sources = _blog_root("https://anthropic.com")
    verdicts = _ct("anthropics", info, ts, [], all_sources)
    assert not verdicts[canonical_identity(org.url)].trusted    # org can't be "the Oracle"


# ── Re-propagation: the Substack _preloads back-edge participates in the rules ─

def test_substack_preload_edge_graduates_candidate_via_rule3(monkeypatch):
    """A trusted Substack hub's _preloads back-edge is the 2nd trusted pointer to a
    needs-review X candidate → Rule 3 graduates it on re-propagation (no
    handle-match involved). Proves the recovered edge feeds the standard graph."""
    _neutralize(monkeypatch,
                sub_handle=lambda url, timeout=12: "hh" if "s.substack.com" in url else None)

    S = DS("substack", "https://s.substack.com")
    all_sources = [S]
    x_attested = {"x.com/root", "blogt.com", "s.substack.com"}       # S + blogt are roots
    relevant = {"x.com/root", "blogt.com", "s.substack.com", "x.com/hh"}
    edges = [Edge("blogt.com", "x.com/hh", via="seed")]              # 1st trusted pointer to hh
    verdicts = {
        "s.substack.com": TrustEvidence(trusted=True, reasons=["root"]),   # trusted → a hub
        "x.com/hh": TrustEvidence(trusted=False, reasons=["needs review"]),
    }

    out = dp._expand_from_trusted_hubs("root", all_sources, verdicts, relevant, x_attested, edges)

    assert out["x.com/hh"].trusted
    assert "Cited by 2" in out["x.com/hh"].reasons[0]
    # the recovered back-edge is really in the graph now
    assert Edge("s.substack.com", "x.com/hh", via="substack_preload", found_by="substack") in edges


# ── Bounds & fail-safe ────────────────────────────────────────────────────────

def test_only_trusted_hubs_expand(monkeypatch):
    calls = []
    _neutralize(monkeypatch, links=lambda url, timeout=12: calls.append(url) or [])
    # A blog that is NOT attested and uncorroborated → needs-review → not a hub.
    all_sources = [DS("blog", "https://random.com")]
    verdicts = _ct("someuser", {}, [], [], all_sources)
    assert not verdicts["random.com"].trusted
    assert calls == []                                          # never fetched a non-trusted hub


def test_max_new_candidates_respected(monkeypatch):
    monkeypatch.setattr(dp, "MAX_NEW_HUB_CANDIDATES", 1)
    links = [
        ProfileLink("github", True, "gh1", "personal", "https://github.com/gh1"),
        ProfileLink("scholar", True, "sch1", "personal", "https://scholar.google.com/citations?user=sch1"),
    ]
    _neutralize(monkeypatch, links=lambda url, timeout=12: links if "willcb.com" in url else [])
    info, ts, all_sources = _blog_root()
    _ct("willccbb", info, ts, [], all_sources)
    surfaced = [s for s in all_sources if (s.metadata or {}).get("discovered_via") == "blog_hub"]
    assert len(surfaced) == 1                                   # capped, second dropped


def test_already_present_profile_not_re_added(monkeypatch):
    gh = ProfileLink("github", True, "willccbb", "personal", "https://github.com/willccbb")
    _neutralize(monkeypatch, links=lambda url, timeout=12: [gh] if "willcb.com" in url else [])
    info, ts, _ = _blog_root()
    all_sources = [DS("blog", "https://willcb.com"), DS("github", "https://github.com/willccbb")]
    _ct("willccbb", info, ts, [DS("github", "https://github.com/willccbb")],
                      all_sources)
    ghs = [s for s in all_sources if s.source_type == "github"]
    assert len(ghs) == 1                                        # not re-added
    assert not any((s.metadata or {}).get("discovered_via") == "blog_hub" for s in all_sources)


def test_expansion_noop_when_fetch_returns_empty(monkeypatch):
    _neutralize(monkeypatch, links=lambda url, timeout=12: [])   # simulate fetch failure → []
    info, ts, all_sources = _blog_root()
    verdicts = _ct("willccbb", info, ts, [], all_sources)
    assert verdicts["willcb.com"].trusted                       # unchanged root
    assert not any((s.metadata or {}).get("discovered_via") == "blog_hub" for s in all_sources)


def test_skip_edge_fetch_disables_expansion(monkeypatch):
    calls = []
    _neutralize(monkeypatch, links=lambda url, timeout=12: calls.append(url) or [])
    info, ts, all_sources = _blog_root()
    _ct("willccbb", info, ts, [], all_sources,
                      skip_edge_fetch=True)
    assert calls == []                                          # expansion never ran
