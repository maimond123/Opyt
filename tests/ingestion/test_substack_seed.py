"""Tests for the non-X ROOT probe (`_probe_substack_profile`) and the Substack-seeded
`discover_profile` path — the de-X-rooting entry point.

The probe parses the PUBLIC profile payload (`/api/v1/user/{h}/public_profile`) into
(profile_info, sources, identity_targets); the end-to-end path judges them, so a
Substack-rooted person's declared accounts are theirs with no X in the root set. Both mock the
network — no live Substack call.
"""


from pipeline.oracles.discovery import backlink
from pipeline.oracles.discovery import probes
from pipeline.oracles.discovery import profile as dp


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
    # Live shape (verified): user-level subdomainUrl is null; the publication lives
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
        if "ranked" in url:
            return _Resp(200, [{"handle": "someone" if payload.get("name") == "Someone" else "gergelyorosz"}])
        if "public_profile" in url:
            return _Resp(status, payload)
        return _Resp(404, {})
    monkeypatch.setattr(probes.requests, "get", fake)


# ── The probe: payload → (profile_info, sources, identity_targets) ────────────

def test_probe_parses_declared_links_and_root(monkeypatch):
    _mock_get(monkeypatch)
    info, sources, targets = probes.probe_substack_profile("pragmaticengineer")
    ci = dp.canonical_identity

    assert info["display_name"] == "The Pragmatic Engineer"
    # Publication from primaryPublication.subdomain — a UNIQUE canonical node, NOT the bare
    # "substack.com" the /@handle fallback would collapse to, and an ingestable archive host.
    assert info["root_url"] == "https://pragmaticengineer.substack.com"
    assert ci(info["root_url"]) == "pragmaticengineer.substack.com"   # unique, not "substack.com"

    tset = {tid: verified for tid, verified in targets}
    assert tset[ci("https://x.com/GergelyOrosz")] is True        # connected → verified
    assert ci("https://pragmaticengineer.com") in tset          # declared site
    assert set(tset) == {"x.com/gergelyorosz", "pragmaticengineer.com"}
    assert {s.url for s in sources} == {
        "https://pragmaticengineer.substack.com",
        "https://x.com/GergelyOrosz",
        "https://pragmaticengineer.com",
    }

    # The root Substack itself is a source (its canonical == root_id → Rule-1 trusted →
    # routes through the Half-A footprint adapter as the person's primary channel).
    stypes = {(s.source_type, ci(s.url)) for s in sources}
    assert ("substack", ci(info["root_url"])) in stypes
    # The connected X is classified `x` (the _classify_url fix), NOT `blog`.
    assert ("x", ci("https://x.com/GergelyOrosz")) in stypes


def test_probe_without_primary_publication_keeps_the_seed_publication(monkeypatch):
    # The publication seed remains the root when its author has no primaryPublication field.
    _mock_get(monkeypatch, payload={"name": "Someone", "twitterAccount": {}, "userLinks": []})
    info, _sources, _targets = probes.probe_substack_profile("someone")
    assert info["root_url"] == "https://someone.substack.com"
    assert dp.canonical_identity(info["root_url"]) == "someone.substack.com"


def test_probe_failsafe_on_non_200(monkeypatch):
    _mock_get(monkeypatch, payload={}, status=404)
    info, sources, targets = probes.probe_substack_profile("nobody")
    assert info["display_name"] == "" and sources == [] and targets == []


# ── Seed → user-slug resolution (the custom-domain-minted member fix) ──────────

def test_resolve_publication_slug_to_different_author(monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        if "ranked" in url:
            return _Resp(200, [{"handle": "alicewriter"}])
        return _Resp(200, {"name": "Alice", "primaryPublication": None})
    monkeypatch.setattr(probes.requests, "get", get)
    info, sources, _ = probes.probe_substack_profile("newsletter")
    assert calls == [
        "https://newsletter.substack.com/api/v1/publication/users/ranked?public=true",
        "https://substack.com/api/v1/user/alicewriter/public_profile",
    ]
    assert info["root_url"] == "https://newsletter.substack.com"
    assert [s.url for s in sources] == [info["root_url"]]


def test_a_custom_domain_resolves_to_its_owner(monkeypatch):
    # A custom-domain HOST/URL resolves to its top-ranked author's public profile.
    calls = []
    def get(url, **k):
        calls.append(url)
        if "ranked" in url:
            return _Resp(200, [{"handle": "owner"}, {"handle": "coauthor"}])
        return _Resp(200, {"name": "Owner"})
    monkeypatch.setattr(probes.requests, "get", get)
    for seed in ("https://newsletter.foo.com", "newsletter.foo.com"):    # bare host too
        calls.clear()
        data, _final = backlink.substack_profile(seed)
        assert data == {"name": "Owner"}
        assert calls[-1] == "https://substack.com/api/v1/user/owner/public_profile"


def test_resolve_slug_failsafe(monkeypatch):
    monkeypatch.setattr(probes.requests, "get", lambda *a, **k: _Resp(404, {}))
    assert backlink.substack_profile("newsletter.foo.com") == (None, None)
    assert backlink.substack_profile("") == (None, None)


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
    monkeypatch.setattr(probes.requests, "get", fake)

    info, sources, targets = probes.probe_substack_profile("newsletter.foo.com")
    ci = dp.canonical_identity
    assert info["display_name"] == "Foo Bar"
    assert info["root_url"] == "https://foobar.substack.com"          # from the RESOLVED slug's pub
    assert ci("https://x.com/foobar_x") in {tid for tid, _ in targets}
    assert ("substack", ci("https://foobar.substack.com")) in {(s.source_type, ci(s.url)) for s in sources}


# ── End-to-end: discover_profile(seed_type="substack") with no X ───

def test_discover_substack_seed_trusts_declared_accounts(monkeypatch):
    _mock_get(monkeypatch)
    ci = dp.canonical_identity

    # skip_edge_fetch → nothing read: what the root declares is theirs.
    result = dp.discover_profile("pragmaticengineer", seed_type="substack", skip_edge_fetch=True)

    sure = {ci(v["url"]) for v in result["verdicts"] if v["verdict"] == "sure"}
    assert result["root_id"] == ci("https://pragmaticengineer.substack.com")   # the root itself
    assert ci("https://x.com/GergelyOrosz") in sure                    # declared X, no X root
    assert ci("https://pragmaticengineer.com") in sure                # declared site

    # The discovered X carries source_type `x` so register can pull its timeline.
    x_verdicts = [v for v in result["verdicts"] if v["source_type"] == "x"]
    assert len(x_verdicts) == 1 and x_verdicts[0]["verdict"] == "sure"
