"""Tests for the non-X ROOT probe (`_probe_substack_profile`) and the Substack-seeded
`discover_profile` path — the de-X-rooting entry point.

The probe parses the PUBLIC profile payload (`/api/v1/user/{h}/public_profile`) into
(profile_info, sources, identity_targets); the end-to-end path threads that through the
platform-agnostic trust contract so a Substack-rooted person's declared accounts reach T1
with no X in the root set. Both mock the network — no live Substack call.
"""

import importlib

dp = importlib.import_module("pipeline.ingestion.discover_profile")  # bypass __init__ shadow


class _Resp:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


_PAYLOAD = {
    "name": "The Pragmatic Engineer",
    "bio": "Writing about big tech and startups.",
    # Live shape (verified 2026-07-21): user-level subdomainUrl is null; the publication lives
    # in primaryPublication. Substack keeps {subdomain}.substack.com live behind the custom domain.
    "subdomainUrl": None,
    "primaryPublication": {"subdomain": "pragmaticengineer",
                           "custom_domain": "newsletter.pragmaticengineer.com",
                           "name": "The Pragmatic Engineer"},
    "twitterAccount": {"screen_name": "GergelyOrosz", "is_connected_account": True,
                       "display_name": "Gergely Orosz"},
    "userLinks": [
        {"url": "https://pragmaticengineer.com", "is_connected_account": False},
        {"url": "https://www.youtube.com/@ThePragmaticEngineer"},
    ],
}


def _mock_get(monkeypatch, payload=_PAYLOAD, status=200):
    # Serve the profile payload ONLY on the public_profile URL; every other GET (RSS
    # feed-detection on a declared blog link, etc.) misses → the probe degrades gracefully.
    def fake(url, **kw):
        if "public_profile" in url:
            return _Resp(status, payload)
        return _Resp(404, {})
    monkeypatch.setattr(dp.requests, "get", fake)


# ── The probe: payload → (profile_info, sources, identity_targets) ────────────

def test_probe_parses_declared_links_and_root(monkeypatch):
    _mock_get(monkeypatch)
    info, sources, targets = dp._probe_substack_profile("pragmaticengineer")
    ci = dp.canonical_identity

    assert info["display_name"] == "The Pragmatic Engineer"
    # Publication from primaryPublication.subdomain — a UNIQUE canonical node, NOT the bare
    # "substack.com" the /@handle fallback would collapse to, and an ingestable archive host.
    assert info["root_url"] == "https://pragmaticengineer.substack.com"
    assert ci(info["root_url"]) == "pragmaticengineer.substack.com"   # unique, not "substack.com"

    tset = {tid: verified for tid, verified in targets}
    assert tset[ci("https://x.com/GergelyOrosz")] is True        # connected → verified
    assert ci("https://pragmaticengineer.com") in tset          # declared site
    assert ci("https://www.youtube.com/@ThePragmaticEngineer") in tset

    # The root Substack itself is a source (its canonical == root_id → Rule-1 trusted →
    # routes through the Half-A footprint adapter as the person's primary channel).
    stypes = {(s.source_type, ci(s.url)) for s in sources}
    assert ("substack", ci(info["root_url"])) in stypes
    # The connected X is classified `x` (the _classify_url fix), NOT `blog`.
    assert ("x", ci("https://x.com/GergelyOrosz")) in stypes


def test_probe_root_url_falls_back_to_handle_subdomain(monkeypatch):
    # No primaryPublication (e.g. a reader-only account) → the seed handle's subdomain, which
    # is still a UNIQUE canonical node — never the bare "substack.com" collision.
    _mock_get(monkeypatch, payload={"name": "Someone", "twitterAccount": {}, "userLinks": []})
    info, _sources, _targets = dp._probe_substack_profile("someone")
    assert info["root_url"] == "https://someone.substack.com"
    assert dp.canonical_identity(info["root_url"]) == "someone.substack.com"


def test_probe_failsafe_on_non_200(monkeypatch):
    _mock_get(monkeypatch, payload={}, status=404)
    info, sources, targets = dp._probe_substack_profile("nobody")
    assert info["display_name"] == "" and sources == [] and targets == []


# ── Seed → user-slug resolution (the custom-domain-minted member fix) ──────────

def test_resolve_slug_bare_label_is_passthrough_no_network(monkeypatch):
    # A bare slug (the {subdomain}.substack.com-minted case) is used as-is — NO network call.
    monkeypatch.setattr(dp.requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a bare slug must not trigger a network resolution")))
    assert dp._resolve_substack_user_slug("pragmaticengineer") == "pragmaticengineer"


def test_resolve_slug_from_custom_domain_host(monkeypatch):
    # A custom-domain HOST/URL resolves to its primary author's slug via ranked-users.
    monkeypatch.setattr(dp.requests, "get",
                        lambda url, **k: _Resp(200, [{"handle": "owner"}, {"handle": "coauthor"}])
                        if "ranked" in url else _Resp(404, {}))
    assert dp._resolve_substack_user_slug("https://newsletter.foo.com") == "owner"
    assert dp._resolve_substack_user_slug("newsletter.foo.com") == "owner"   # bare host too


def test_resolve_slug_failsafe(monkeypatch):
    monkeypatch.setattr(dp.requests, "get", lambda *a, **k: _Resp(404, {}))
    assert dp._resolve_substack_user_slug("newsletter.foo.com") is None
    assert dp._resolve_substack_user_slug("") is None


def test_probe_resolves_custom_domain_to_user_slug(monkeypatch):
    """A custom-domain-minted member (substack:newsletter.foo.com): the host is NOT the user
    slug, so the probe resolves it to the primary author's slug, then probes that user's
    public_profile — the fix for the custom-domain onboarding 404."""
    def fake(url, **kw):
        if "publication/users/ranked" in url:
            assert "newsletter.foo.com" in url
            return _Resp(200, [{"id": 1, "name": "Foo Bar", "handle": "foobar"}])
        if "user/foobar/public_profile" in url:
            return _Resp(200, {"name": "Foo Bar", "primaryPublication": {"subdomain": "foobar"},
                               "twitterAccount": {"screen_name": "foobar_x", "is_connected_account": True},
                               "userLinks": []})
        return _Resp(404, {})
    monkeypatch.setattr(dp.requests, "get", fake)

    info, sources, targets = dp._probe_substack_profile("newsletter.foo.com")
    ci = dp.canonical_identity
    assert info["display_name"] == "Foo Bar"
    assert info["root_url"] == "https://foobar.substack.com"          # from the RESOLVED slug's pub
    assert ci("https://x.com/foobar_x") in {tid for tid, _ in targets}
    assert ("substack", ci("https://foobar.substack.com")) in {(s.source_type, ci(s.url)) for s in sources}


# ── End-to-end: discover_profile(seed_type="substack") reaches T1 with no X ───

class _Cfg:
    def __init__(self, tmp_path):
        self._tmp = tmp_path

    def state_file(self, name):
        return self._tmp / f"{name}.json"


def test_discover_substack_seed_trusts_declared_accounts(monkeypatch, tmp_path):
    _mock_get(monkeypatch)
    monkeypatch.setattr(dp, "_probe_github", lambda username: [])   # no convention GitHub guess
    ci = dp.canonical_identity

    # skip_edge_fetch → no landing fetch / hub expansion; skip_trust_cache_write → no cache
    # write. Purely: identity edges → Rule 5 → T1.
    result = dp.discover_profile(
        "pragmaticengineer", seed_type="substack", config=_Cfg(tmp_path),
        skip_edge_fetch=True, skip_trust_cache_write=True, probe_scholar=False)

    trusted = {ci(s["url"]) for s in result["sources"] if (s.get("trust") or {}).get("trusted")}
    assert ci("https://pragmaticengineer.substack.com") in trusted    # the Substack root (Rule 1)
    assert ci("https://x.com/GergelyOrosz") in trusted                 # declared X (Rule 5), no X root
    assert ci("https://pragmaticengineer.com") in trusted             # declared site (Rule 5)

    # The discovered X carries source_type `x` so the footprint caller can pull its timeline.
    x_srcs = [s for s in result["sources"] if s["source_type"] == "x"]
    assert len(x_srcs) == 1 and (x_srcs[0].get("trust") or {}).get("trusted")
