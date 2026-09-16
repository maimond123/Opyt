"""add_oracle — the atom-native "add a person".

Proves the load-bearing behavior fully OFFLINE (discovery + the footprint adapters are stubbed;
the X identity fetch is monkeypatched):
  • the two-phase confirm gate — confirm=False PREVIEWS and writes nothing; an unresolvable
    reference is reported, never confirmable (the anti-hallucination guard);
  • the chain — resolve → confirm (oracles row) → ingest → SEED TRUST ROOT → re-resolve;
  • Mode B (network-free local dedup off `profile.handle`) and Mode C (promote a canonical_id);
  • the trust-seed weld: every confirmed Oracle's cluster is a tier-1.0 root, even a rootless one;
  • the lookback windows are reported back (so the host can tell the user how far it pulled).
"""
from __future__ import annotations

import importlib

import pytest

from pipeline.kb import (eligibility, ingest_blog, ingest_github, ingest_substack,
                         ingest_x_footprint, oracles, resolve, schema)

_DP = importlib.import_module("pipeline.ingestion.discover_profile")  # the MODULE (not the fn)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def managed_x_session(monkeypatch):
    """Existing Oracle-ingest tests exercise the connected-X path unless they opt out."""
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)


class _Cfg:
    """Minimal cfg stub: state_file(name) → a tmp path. Mirrors the one in test_trust_cache.py."""
    def __init__(self, tmp_path):
        self._tmp = tmp_path

    def state_file(self, name):
        return self._tmp / f"{name}.json"


def _fake_discover(sources):
    # _ingest_oracle calls discover_profile(seed, seed_type=..., skip_trust_cache_write=True).
    return lambda seed, seed_type="x", **kw: {"username": seed, "sources": sources}


def _src(stype, url, trusted, **meta):
    return {"source_type": stype, "url": url, "metadata": meta,
            "trust": {"trusted": trusted, "reasons": []}}


@pytest.fixture()
def stub_footprint(monkeypatch):
    """Offline footprint engine: the atom-KB adapters + eligibility become no-op recorders, and
    discovery returns an empty source list by default (tests re-patch `_DP.discover_profile` to
    inject sources). Targets the shared submodules, so it covers BOTH `_ingest_oracle` and the
    `onboard_footprint` it calls.

    BOTH eligibility entry points are stubbed, not just `gate`. `oracles._multi_author_refusal`
    calls `classify_authorship` directly — it wants the SITE verdict, not the per-run decision —
    so a stub on `gate` alone leaves the root check reaching a real socket."""
    calls = []

    def mk(name):
        def f(conn, embedder, **kw):
            calls.append((name, kw))
            return {"adapter": name}
        return f

    monkeypatch.setattr(ingest_substack, "sync_substack_footprint", mk("substack"))
    monkeypatch.setattr(ingest_blog, "sync_blog_footprint", mk("blog"))
    monkeypatch.setattr(ingest_github, "sync_github", mk("github"))
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint", mk("x"))
    monkeypatch.setattr(eligibility, "gate",
                        lambda conn, url, **kw: eligibility.GateDecision("ingest", "stub"))
    monkeypatch.setattr(eligibility, "classify_authorship",
                        lambda conn, url: eligibility.AuthorshipVerdict("single", reason="stub"))
    monkeypatch.setattr(_DP, "discover_profile", _fake_discover([]))
    return calls


def _x_ident(uid, handle, name="Person", followers=10):
    return {"user_id": uid, "display_name": name, "bio": "", "site": "",
            "verified": False, "followers": followers, "handle": handle}


def _days_ago(iso: str | None) -> float | None:
    """How many days before NOW an ISO timestamp is — the readable form of a resolved window."""
    from datetime import datetime, timezone
    if iso is None:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 86400


# ── Phase 1: preview (no writes) ────────────────────────────────────────────────

def test_preview_resolves_handle_and_writes_nothing(conn, monkeypatch):
    monkeypatch.setattr(oracles, "_fetch_x_identity",
                        lambda h: _x_ident("42", "kay", name="Kay", followers=1200))
    out = oracles.add_oracle(conn, None, "@kay", confirm=False)
    assert out["confirm_required"] is True and out["mode"] == "new"
    assert out["resolved"]["name"] == "Kay" and out["resolved"]["followers"] == 1200
    assert out["resolved"]["root_entity"] == "x:user:42"
    # No writes: the profile fetch is read-only, nothing is minted or confirmed.
    assert schema.get_entity(conn, "x:user:42") is None
    assert schema.list_oracles(conn) == []


def test_preview_unresolvable_handle_is_reported_not_written(conn, monkeypatch):
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: None)
    out = oracles.add_oracle(conn, None, "@ghost", confirm=False)
    assert out["unresolved"] == "@ghost"                 # nothing to confirm — the anti-hallucination guard
    assert schema.list_oracles(conn) == []


def test_preview_url_reports_the_platform_and_writes_nothing(conn, no_venue):
    """Renamed 2026-09-08: this was `…_network_free`, and a URL preview is no longer that. Rooting
    a site asks OpenAlex once whether the host is a research venue, because a preview that said
    "blog" and then minted a 63,000-work venue would be lying on the consent surface. The
    invariant that survives — and the one this actually tested — is that a preview WRITES
    NOTHING."""
    blog = oracles.add_oracle(conn, None, "https://simonwillison.net", confirm=False)
    assert blog["mode"] == "new" and blog["resolved"]["platform"] == "blog"
    assert blog["resolved"]["root_entity"].startswith("blog:")

    sub = oracles.add_oracle(conn, None, "https://carol.substack.com", confirm=False)
    assert sub["resolved"]["platform"] == "substack"
    assert schema.list_oracles(conn) == []               # still no writes


def test_preview_reports_lookback_windows(conn):
    default = oracles.add_oracle(conn, None, "https://x.dev", confirm=False)
    assert "6-month default" in default["lookback"]["x"]
    assert default["lookback"]["web"] == "full archive" and default["lookback"]["web_since"] is None

    scoped = oracles.add_oracle(conn, None, "https://x.dev", confirm=False,
                                x_lookback="2yr", web_lookback="5yr")
    assert _days_ago(scoped["lookback"]["x_since"]) == pytest.approx(730, abs=1)
    assert _days_ago(scoped["lookback"]["web_since"]) == pytest.approx(1825, abs=1)


def test_report_states_the_x_clamp_instead_of_the_window_asked_for(conn):
    """The report is the CONSENT surface for the pull, so it must say what ran, not what was
    requested. The old collapsed version answered "6 months (default)" for a 5-year request that
    the adapter clamped to 2 years — under-reporting the window by 4x, which is consent to a
    window you were never shown."""
    from datetime import datetime, timedelta, timezone

    five_years = datetime.now(timezone.utc) - timedelta(days=1825)
    rep = oracles._lookback_report(five_years, None)
    assert "CLAMPED" in rep["x"]
    assert _days_ago(rep["x_since"]) == pytest.approx(730, abs=1)   # the 2-year ceiling, not 1825


# ── Phase 2: the chain + the trust-seed weld ────────────────────────────────────

def test_confirm_runs_chain_and_seeds_trust(conn, stub_footprint, monkeypatch):
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia", name="Nia"))
    out = oracles.add_oracle(conn, object(), "@nia", confirm=True)

    assert out["added"]["canonical_id"] == "x:user:7"
    assert out["added"]["source"] == "freeform"
    assert out["added"]["was_already_oracle"] is False
    assert schema.is_oracle(conn, "x:user:7")
    # The X timeline was pulled from the root handle.
    assert any(name == "x" for name, _ in stub_footprint)
    assert "6-month default" in out["lookback"]["x"]


def test_ingest_routes_trusted_offx_source(conn, stub_footprint, monkeypatch):
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://carol.substack.com", True)]))
    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:1"])
    o = next(x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:1")

    r = oracles._ingest_oracle(conn, object(), o)
    assert r["ingested"] >= 1
    assert {name for name, _ in stub_footprint} >= {"substack", "x"}   # off-X routed + X root pulled


# ── the trust boundary: rejected is REPORTED, never dropped ─────────────────────

def _confirmed(conn, cid="x:user:1", *, name="Carol", handle="carol"):
    schema.upsert_entity(conn, cid, name=name, profile={"handle": handle})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[cid])
    return next(x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == cid)


def test_an_untrusted_source_is_reported_for_review_not_dropped(conn, stub_footprint, monkeypatch):
    """⚠️ FIXED 2026-09-04. `_ingest_oracle` pre-filtered the discovered sources to the trusted
    ones before handing them to `onboard_footprint`, which owns the same trust boundary. The
    rejects therefore reached NOTHING: not the adapters (correct) and not the report (the bug).
    A dropped URL is invisible; a rejected one is reviewable, and the whole point of returning
    `needs-review` is that a human can confirm what the graph could not."""
    monkeypatch.setattr(_DP, "discover_profile", _fake_discover([
        _src("blog", "https://maybe-carol.dev", False),
        _src("substack", "https://maybe.substack.com", False),
    ]))
    r = oracles._ingest_oracle(conn, object(), _confirmed(conn))

    reviewed = [x for x in r["results"] if x["action"] == "needs-review"]
    assert {x["url"] for x in reviewed} == {"https://maybe-carol.dev", "https://maybe.substack.com"}
    # …and still NOT ingested: the boundary moved home, it did not move.
    assert {name for name, _ in stub_footprint} == {"x"}


def test_an_untrusted_source_still_never_reaches_an_adapter(conn, stub_footprint, monkeypatch):
    """The half that must not regress while fixing the half above."""
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://squatter.substack.com", False)]))
    oracles._ingest_oracle(conn, object(), _confirmed(conn))
    assert "substack" not in {name for name, _ in stub_footprint}


def test_review_queue_adds_only_the_source_the_user_approved(conn, stub_footprint, monkeypatch):
    """A review approval must not turn `force` into a broad re-ingest override."""
    from pipeline.kb import oracle_reviews

    oracle_reviews.record_outcomes(conn, "x:user:1", [{
        "type": "blog", "url": "https://maybe-carol.dev", "action": "needs-review",
        "detail": "unverified",
    }])
    item = oracle_reviews.list_open(conn)[0]
    _confirmed(conn)

    listed = oracles.review_sources(conn, None)
    assert listed["items"][0]["status"] == "needs_confirmation"
    assert listed["diagnostics"][0]["reason"] == "unverified"

    preview = oracles.review_sources(conn, object(), action="add", review_id=item["review_id"])
    assert preview["status"] == "preview"
    assert oracle_reviews.get(conn, item["review_id"])["status"] == "pending"

    added = oracles.review_sources(conn, object(), action="add", review_id=item["review_id"],
                                   confirm=True)

    assert added["status"] == "added"
    assert oracle_reviews.get(conn, item["review_id"])["status"] == "approved"
    blog_calls = [kw for name, kw in stub_footprint if name == "blog"]
    assert [call["blog_url"] for call in blog_calls] == ["https://maybe-carol.dev"]

    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("blog", "https://maybe-carol.dev", False)]))
    oracles._ingest_oracle(conn, object(), _confirmed(conn))
    assert [kw["blog_url"] for name, kw in stub_footprint if name == "blog"] == [
        "https://maybe-carol.dev"]


def test_review_dismissal_keeps_a_later_trusted_discovery_out(conn, stub_footprint, monkeypatch):
    """"Not this person" is a user exclusion, so a later trust-cache change cannot undo it."""
    from pipeline.kb import oracle_reviews

    oracle_reviews.record_outcomes(conn, "x:user:1", [{
        "type": "blog", "url": "https://maybe-carol.dev", "action": "needs-review",
        "detail": "unverified",
    }])
    item = oracle_reviews.list_open(conn)[0]
    _confirmed(conn)

    out = oracles.review_sources(conn, object(), action="dismiss", review_id=item["review_id"],
                                 confirm=True)
    assert out["status"] == "dismissed"
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("blog", "https://maybe-carol.dev", True)]))

    oracles._ingest_oracle(conn, object(), _confirmed(conn))

    assert oracle_reviews.get(conn, item["review_id"])["status"] == "dismissed"
    assert "blog" not in {name for name, _ in stub_footprint}


def test_review_verification_rechecks_evidence_without_adding_source(conn, stub_footprint,
                                                                       monkeypatch):
    """Evidence can make a source ready, but the user still controls the actual ingestion."""
    from pipeline.kb import oracle_reviews

    oracle_reviews.record_outcomes(conn, "x:user:1", [{
        "type": "blog", "url": "https://maybe-carol.dev", "action": "needs-review",
        "detail": "unverified",
    }])
    item = oracle_reviews.list_open(conn)[0]
    _confirmed(conn)
    seen = {}

    def rechecked(seed, seed_type="x", **kw):
        seen.update(seed=seed, seed_type=seed_type, **kw)
        return {"username": seed, "sources": [_src("blog", "https://maybe-carol.dev", True)]}

    monkeypatch.setattr(_DP, "discover_profile", rechecked)
    out = oracles.review_sources(conn, object(), action="verify", review_id=item["review_id"],
                                 verification_urls=["https://carol.example/about"])

    assert out["status"] == "verified"
    assert oracle_reviews.get(conn, item["review_id"])["status"] == "verified"
    assert seen["reverify"] is True
    assert seen["extra_source_urls"] == ["https://maybe-carol.dev", "https://carol.example/about"]
    assert "blog" not in {name for name, _ in stub_footprint}


def test_forgetting_an_oracle_removes_its_review_queue(conn):
    """Review choices belong to an active Oracle subscription, not retained profile history."""
    from pipeline.kb import oracle_reviews

    _confirmed(conn)
    oracle_reviews.record_outcomes(conn, "x:user:1", [{
        "type": "blog", "url": "https://maybe-carol.dev", "action": "needs-review",
        "detail": "unverified",
    }])

    out = oracles._forget(conn, "x:user:1", confirm=True)

    assert out["status"] == "forgotten"
    assert oracle_reviews.list_open(conn) == []


def test_review_hold_follows_an_oracle_after_its_cluster_head_changes(conn):
    """A dismissal/pending hold cannot vanish when resolve picks a different cluster anchor."""
    from pipeline.kb import oracle_reviews

    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})
    oracle_reviews.record_outcomes(conn, "x:user:1", [{
        "type": "blog", "url": "https://maybe-carol.dev", "action": "needs-review",
        "detail": "unverified",
    }])
    conn.execute("UPDATE entities SET canonical_id=? WHERE entity_id=?", ("blog:carol", "x:user:1"))

    assert ("blog", "maybe-carol.dev") in oracle_reviews.held_source_keys(conn, "blog:carol")

    oracle_reviews.record_outcomes(conn, "blog:carol", [{
        "type": "blog", "url": "https://maybe-carol.dev", "action": "needs-review",
        "detail": "still unverified",
    }])
    items = oracle_reviews.list_open(conn)
    assert len(items) == 1 and items[0]["canonical_id"] == "blog:carol"


def test_a_discovered_x_is_pulled_once_not_recorded_twice(conn, stub_footprint, monkeypatch):
    """⚠️ FIXED 2026-09-04. A Substack-rooted Oracle's discovered X went to `onboard_footprint`
    (which has no X adapter, so it recorded `unsupported`) AND to the timeline pull below (which
    recorded `ingested`) — two contradictory rows for one handle. X is pulled, never routed."""
    schema.upsert_entity(conn, "substack:carol", name="Carol",
                         identity_links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["substack:carol"])
    o = next(x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "substack:carol")
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("x", "https://x.com/carolx", True)]))

    r = oracles._ingest_oracle(conn, object(), o)

    x_rows = [x for x in r["results"] if x["type"] == "x"]
    assert len(x_rows) == 1 and x_rows[0]["action"] == "ingested"
    assert "unsupported" not in {x["action"] for x in r["results"]}


def test_substack_only_ingest_leaves_a_discovered_x_profile_optional(conn, stub_footprint,
                                                                     monkeypatch):
    """Discovering an X identity is not consent to pull it or schedule it for refresh."""
    from pipeline.ingestion import x_graphql
    from pipeline.kb import oracle_refresh_state as rst

    schema.upsert_entity(conn, "substack:carol", name="Carol",
                         identity_links=["https://carol.substack.com"])
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["substack:carol"])
    oracle = next(o for o in oracles.confirmed_oracles(conn)
                  if o["canonical_id"] == "substack:carol")
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("x", "https://x.com/carolx", True)]))
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda *a, **kw: pytest.fail("an unconnected X profile must not pull"))

    result = oracles._ingest_oracle(conn, object(), oracle)

    assert result["available_sources"] == [{"source_type": "x", "url": "https://x.com/carolx"}]
    assert "x" not in {name for name, _ in stub_footprint}
    assert all(row.source_type != "x" for row in rst.list_sources(conn))


def test_unconnected_x_root_creates_no_x_refresh_pair(conn, stub_footprint, monkeypatch):
    """The registry is a pull queue, so it must not retain an unconnected X root."""
    from pipeline.ingestion import x_graphql
    from pipeline.kb import oracle_refresh_state as rst

    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)
    result = oracles._ingest_oracle(conn, object(), _confirmed(conn))

    assert result["available_sources"] == [{"source_type": "x", "url": "https://x.com/carol"}]
    assert "x" not in {name for name, _ in stub_footprint}
    assert all(row.source_type != "x" for row in rst.list_sources(conn))


# ── rooting refuses what it cannot root ────────────────────────────────────────

@pytest.mark.parametrize("url", ["https://github.com/karpathy", "https://x.com/karpathy",
                                 "https://twitter.com/karpathy", "https://youtube.com/@karpathy",
                                 "https://m.youtube.com/@karpathy", "https://youtu.be/video"])
def test_a_platform_profile_url_is_refused_not_minted_as_a_blog(conn, url):
    """⚠️ FIXED 2026-09-04. Any non-Substack http… reference keyed on `blog:{host}`, so
    `https://github.com/karpathy` minted `blog:github.com/karpathy` and rooted discovery on it as
    a personal site. Worst case was an X URL: `blog:x.com/karpathy` for a person who already has
    an `x:user:` identity, i.e. a second, permanently unmergeable copy of them."""
    out = oracles.add_oracle(conn, object(), url, confirm=False)
    assert "error" in out and "cannot add" in out["error"]
    assert schema.get_entity(conn, "blog:github.com/karpathy") is None


def test_an_x_url_says_to_pass_the_handle(conn):
    """The refusal has to be actionable — there IS a right way to add this person."""
    out = oracles.add_oracle(conn, object(), "https://x.com/karpathy", confirm=False)
    assert "@handle" in out["error"]


def test_a_real_personal_site_is_still_rootable(conn, no_venue):
    """The guard names four platform hosts; everything else keeps working — including after the
    venue branch went in front of the blog fallthrough."""
    out = oracles.add_oracle(conn, object(), "https://simonwillison.net", confirm=False)
    assert out["resolved"]["platform"] == "blog"


# ── the X window refuses rather than falling through ──────────────────────────

@pytest.mark.parametrize("preset", ["1y", "6m", "2years", "all"])
def test_an_unknown_x_preset_raises_rather_than_buying_183_days(preset):
    """⚠️ FIXED 2026-09-04. `X_LOOKBACK_PRESETS.get(preset)` returned None for a typo, and on the
    X selector None means the adapter's own 183-day default — so `x_lookback='1y'` silently
    bought the WIDEST pull available. The host supplies this string, so a typo is one call away.
    Same argument the `since_last` refusal was built on, applied to the same failure."""
    with pytest.raises(ValueError, match="unknown x_lookback"):
        oracles._x_since(preset)


def test_the_real_presets_still_resolve():
    assert oracles._x_since(None) is None                    # unset = the adapter's own default
    assert oracles._x_since("6mo") is not None


# ── the trust cache: production OWNS its entries ─────────────────────────────────

def _capture_discovery(monkeypatch, conn, cid="x:user:20", handle="cached"):
    """Confirm one Oracle and capture the kwargs `_ingest_oracle` hands `discover_profile`."""
    seen = {}

    def _capture(seed, seed_type="x", **kw):
        seen.update(kw)
        return {"username": seed, "sources": []}

    monkeypatch.setattr(_DP, "discover_profile", _capture)
    schema.upsert_entity(conn, cid, name="Cached", profile={"handle": handle})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=[cid])
    o = next(x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == cid)
    return seen, o


def test_ingest_now_WRITES_the_trust_cache(conn, stub_footprint, monkeypatch):
    """⚠️ REVERSED 2026-08-16. This path suppressed the cache write for its whole life, so the
    cache was a designed optimization that production could never populate — every re-ingest paid
    a full four-probe walk, forever.

    IDENTITY IS STABLE; CONTENT IS NOT. Re-ingesting an Oracle is about pulling their new posts,
    not re-deriving who they are. The cache key already invalidates on the only things that can
    change a trust verdict from the X side (display name + declared links) and expires on a TTL,
    so caching the identity half across re-ingests is the intended behavior, not a shortcut.
    """
    seen, o = _capture_discovery(monkeypatch, conn)
    oracles._ingest_oracle(conn, object(), o)
    assert not seen.get("skip_trust_cache_write"), "production must own its cache entries"


def test_force_reaches_DISCOVERY_not_just_the_adapters(conn, stub_footprint, monkeypatch):
    """⚠️ THE ESCAPE HATCH THE CACHE WRITE REQUIRES — they ship together or not at all.

    `reverify` is the only way past a cache hit, and it was threaded from NO production caller:
    it existed on `discover_profile` and only the CLI set it. Harmless while production never
    wrote the cache. The moment production writes, it is load-bearing.

    The case that forces it: the snapshot key is display name + declared links ONLY, so a source
    the person created after the last run — or a fix to our own trust rules — leaves it identical
    and replays the stale verdict for the whole TTL. `reverify` is the only way in before that.

    It also makes `force` mean what any caller assumes. Until now `force=True` reached
    `onboard_footprint` and stopped there, so "force re-ingest" quietly did not re-discover.
    """
    seen, o = _capture_discovery(monkeypatch, conn, cid="x:user:21", handle="forced")
    oracles._ingest_oracle(conn, object(), o, force=True)
    assert seen.get("reverify") is True


def test_add_oracle_can_force_too(conn, stub_footprint, monkeypatch):
    """⚠️ CLOSES A HOLE THE CACHE WRITE ITSELF OPENS. `add_oracle` had no `force` at all, which
    was harmless while nothing cached: re-running it always re-discovered. Now it would cache-hit
    and hand back the same stale sources, with no way out from the tool a user reaches for when
    they say "her sources look wrong, add her again".

    The escape hatch existing only on `oracle(action="ingest")` is not good enough — that is the
    refresh tool, and this is the one people re-run."""
    seen = {}

    def _capture(seed, seed_type="x", **kw):
        seen.update(kw)
        return {"username": seed, "sources": []}

    monkeypatch.setattr(_DP, "discover_profile", _capture)
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("23", "dana"))
    oracles.add_oracle(conn, object(), "@dana", confirm=True, force=True)
    assert seen.get("reverify") is True


def test_a_normal_ingest_does_NOT_force_rediscovery(conn, stub_footprint, monkeypatch):
    """The other half of the pair. If `reverify` were always True the cache could never hit, and
    the write above would be pure cost with no benefit."""
    seen, o = _capture_discovery(monkeypatch, conn, cid="x:user:22", handle="normal")
    oracles._ingest_oracle(conn, object(), o)
    assert not seen.get("reverify")


def test_add_oracle_asks_the_host_to_do_the_open_web_search(conn, stub_footprint, monkeypatch):
    """Probe 5's PUSH leg — the hint has to ride a response somebody actually reads.

    ⚠️ THIS REPO HAS FAILED AT CO-ROUTING TWICE, BOTH TIMES THE SAME WAY. v1's frontier pushed its
    hint onto `search_papers_live` — "not the tool anyone actually calls" (frontier_tools.py).
    `_web_search_followup` pushed its hint into a dict that was discarded before any caller saw it.
    Neither idea was wrong; both hints rode a dead carrier. `add_oracle`'s return is a tool the USER
    invoked, so the host reads it by construction. That is the only reason this attempt differs.
    """
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("11", "alice", name="Alice"))
    out = oracles.add_oracle(conn, object(), "@alice", confirm=True)

    fu = out["followup"]
    assert "alice" in fu["instruction"].lower()
    # The call it names must be RE-ENTRANT ON THE SAME PERSON. Naming a bare `add_oracle(<url>)`
    # is what made the original incoherent: a found blog with no link back to the person mints a
    # SECOND Oracle instead of attaching to the first.
    assert "extra_source_urls" in fu["feed_back_via"]
    assert "@alice" in fu["feed_back_via"]


def test_no_followup_when_discovery_was_a_CACHE_HIT(conn, stub_footprint, monkeypatch):
    """⚠️ THE ASK MUST COST SOMETHING TO REPEAT, OR IT REPEATS FOREVER.

    A cache hit means nothing about this person has changed since we last looked — which is
    exactly when re-running an open-web search can only return what we already have. Asking anyway
    burns a host web search per re-ingest and invites the host to resubmit the same URLs, which
    then bypass the cache and force a full re-discovery. So the waste compounds: the followup
    would undo the caching that shipped in the same hour.

    The TTL becomes the natural re-ask cadence. When the cache expires (30 days) or the person's
    profile changes, discovery runs fresh and the ask returns — which is precisely when new
    sources might actually exist.
    """
    monkeypatch.setattr(_DP, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": [],
                                                           "from_cache": True})
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("30", "cachehit"))
    out = oracles.add_oracle(conn, object(), "@cachehit", confirm=True)
    assert "followup" not in out


def test_followup_returns_once_the_cache_goes_stale(conn, stub_footprint, monkeypatch):
    """The other side: a FRESH discovery asks again. A person who starts a blog after their first
    ingest is found on the next uncached run, not never."""
    monkeypatch.setattr(_DP, "discover_profile",
                        lambda seed, seed_type="x", **kw: {"username": seed, "sources": []})
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("31", "freshrun"))
    out = oracles.add_oracle(conn, object(), "@freshrun", confirm=True)
    assert "followup" in out


def test_no_followup_when_there_was_nothing_to_discover(conn, stub_footprint):
    """No rootable profile → discovery never ran. Asking the host to search for someone OPYT
    cannot root is a dead end: `extra_source_urls` is consumed BY discovery, so there is nowhere
    for the answer to go."""
    schema.upsert_entity(conn, "x:user:32", name="NoRoot")
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:32"])
    o = next(x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:32")
    r = oracles._ingest_oracle(conn, object(), o)
    assert "no rootable profile" in r["error"]
    assert not r.get("discovery_ran_fresh")


def test_discover_profile_marks_a_cache_hit(tmp_path, monkeypatch):
    """`from_cache` is the signal the followup gate reads, so it has to actually be set."""
    cfg = _Cfg(tmp_path)
    snap = _DP._snapshot_hash("Alice", ["x.com/alice"])
    _DP._save_cached_trust("alice", snap, {"username": "alice", "sources": []}, cfg)
    got = _DP._get_cached_trust("alice", snap, cfg)
    assert got is not None and "from_cache" not in got, "the STORED copy must stay clean"


def test_the_followup_names_a_parameter_that_actually_exists():
    """⚠️ THE FAILURE THAT KILLED THE LAST TWO ATTEMPTS, PINNED. The original hint named the
    vault-era add-a-person tool's `include_urls` argument, and that tool was DELETED 2026-08-07 —
    so it spent months instructing the host to call a function that did not exist. This string is
    handed to a model as an INSTRUCTION; a stale name here is worse than a stale docstring."""
    import inspect
    sig = inspect.signature(oracles.add_oracle)
    assert "extra_source_urls" in sig.parameters
    sig_dp = inspect.signature(_DP.discover_profile)
    assert "extra_source_urls" in sig_dp.parameters


def test_host_supplied_urls_reach_discovery(conn, stub_footprint, monkeypatch):
    """The RETURN leg, end to end: what the host hands back reaches the trust graph."""
    seen = {}

    def _capture(seed, seed_type="x", **kw):
        seen.update(kw)
        return {"username": seed, "sources": []}

    monkeypatch.setattr(_DP, "discover_profile", _capture)
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("12", "bob", name="Bob"))
    oracles.add_oracle(conn, object(), "@bob", confirm=True,
                       extra_source_urls=["https://bob.dev"])
    assert seen.get("extra_source_urls") == ["https://bob.dev"]


def test_ingest_seeds_trust_even_without_a_rootable_profile(conn, stub_footprint):
    # No handle, no substack/blog member → nothing to root discovery on…
    schema.upsert_entity(conn, "x:user:2", name="NoHandle")
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:2"])
    o = next(x for x in oracles.confirmed_oracles(conn) if x["canonical_id"] == "x:user:2")

    r = oracles._ingest_oracle(conn, object(), o)
    assert "no rootable profile" in r["error"]
    # The confirmation itself stands regardless — an unrootable Oracle is still an Oracle.
    assert schema.is_oracle(conn, "x:user:2")


def test_lookback_threads_since_to_the_x_pull(conn, stub_footprint, monkeypatch):
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="2yr")
    x_kw = next(kw for name, kw in stub_footprint if name == "x")
    assert x_kw["since"] is not None                     # a bounded window reached the X pull
    assert _days_ago(x_kw["since"].isoformat()) == pytest.approx(730, abs=1)


# ── since_last: the cheap top-up ────────────────────────────────────────────────
#
# Every other X preset is a fixed span, identical for everyone. This one is "since I last pulled
# THIS person", so it resolves per-Oracle. It exists because the automatic loop pulls a
# since-last-pull delta while the narrowest window a USER could ask for began at 6 months —
# ~19x the requests for an average poster, to fetch the same handful of posts.

def _seed_last_pull(conn, cid: str, handle: str, hours_ago: float):
    """Register an X pair for `cid` and stamp its last pull `hours_ago` in the past."""
    from datetime import datetime, timedelta, timezone

    from pipeline.kb import oracle_refresh_state as rst
    rst.seed_from_entities(conn, canonical_ids=[cid])
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    conn.execute("UPDATE oracle_sources SET last_pulled_at=? WHERE canonical_id=? "
                 "AND source_type='x'", (when, cid))
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM oracle_sources WHERE canonical_id=? "
                        "AND source_type='x'", (cid,)).fetchone()[0]


def test_since_last_pulls_only_the_gap_plus_the_overlap_sliver(conn, stub_footprint, monkeypatch):
    """A 5-day-old pull asks for 5 days + OVERLAP_HOURS, not 183 days. The sliver is deliberate:
    a tweet can land slightly out of order, and the content-hash dedup absorbs the repeat."""
    from pipeline.kb import oracle_refresh

    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="6mo")
    cid = schema.current_canonical(conn, "x:user:7")
    assert _seed_last_pull(conn, cid, "nia", hours_ago=120.0) == 1     # 5 days

    stub_footprint.clear()
    out = oracles.add_oracle(conn, object(), "@nia", confirm=True,
                             x_lookback=oracles.X_SINCE_LAST)

    assert "error" not in out
    x_kw = next(kw for name, kw in stub_footprint if name == "x")
    expected_days = (120.0 + oracle_refresh.OVERLAP_HOURS) / 24.0
    assert _days_ago(x_kw["since"].isoformat()) == pytest.approx(expected_days, abs=0.05)


def test_since_last_refuses_rather_than_falling_back_to_183_days(conn, stub_footprint,
                                                                 monkeypatch):
    """⚠️ THE trap this preset exists inside. `_x_since(None)` means "the adapter's ~6-month
    default", so a since_last that quietly resolved to None would turn a request for the CHEAPEST
    window into the most expensive pull available — a second full onboarding, unasked and unpriced.
    It must refuse, name the person, and spend nothing."""
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("9", "zed"))
    oracles.add_oracle(conn, object(), "@zed", confirm=True, x_lookback="6mo")
    cid = schema.current_canonical(conn, "x:user:9")
    # Reaching "no basis" takes TWO clears, and that difficulty is itself the good news: a
    # confirmed Oracle almost always carries either a pull stamp or atoms, so this refusal is a
    # rare edge. It is tested anyway because the cost of getting it wrong is a 183-day pull.
    #   1. the pair's own stamp, and 2. its corpus-derived cursor.
    # There used to be a third — `oracles.ingest_to`, which `upsert_source` COALESCEd back in on
    # the next re-seed. That column is gone; coverage is per-pair now and nothing re-seeds it.
    from pipeline.kb import oracle_refresh_state as rst
    rst.seed_from_entities(conn, canonical_ids=[cid])
    cleared = conn.execute("UPDATE oracle_sources SET last_pulled_at=NULL, cursor_ts=NULL "
                           "WHERE canonical_id=? AND source_type='x'", (cid,)).rowcount
    conn.commit()
    assert cleared == 1, "the UPDATE hit no row — the test would prove nothing"
    assert oracles.x_since_last(conn, cid) is None, "setup failed: a window is still derivable"

    stub_footprint.clear()
    out = oracles.add_oracle(conn, object(), "@zed", confirm=True,
                             x_lookback=oracles.X_SINCE_LAST)

    assert "since_last" in out["error"] and "first X pull" in out["error"]
    assert stub_footprint == [], "it refused but still ran the X pull"


def test_x_since_never_silently_resolves_since_last(conn):
    """The batch resolver raises instead of returning None, so a caller that forgets to route
    `since_last` per-Oracle fails loudly rather than buying 183 days for everyone."""
    with pytest.raises(ValueError, match="per-Oracle"):
        oracles._x_since(oracles.X_SINCE_LAST)


def test_the_two_windows_reach_their_own_adapters_and_only_their_own(conn, stub_footprint,
                                                                     monkeypatch):
    """THE regression: `x_lookback="6mo"` + `web_lookback="all"` must reach the X pull with a
    183-day `since` and the web archive with `since=None`.

    Both halves matter and both used to be wrong at once, because ONE `since` went to both
    adapters: asking for a short X window silently truncated the free, durable archive to the same
    6 months, and asking for a deep archive silently pulled 2 years of an ephemeral stream."""
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://nia.substack.com", True)]))
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    oracles.add_oracle(conn, object(), "@nia", confirm=True,
                       x_lookback="6mo", web_lookback="all")

    x_kw = next(kw for name, kw in stub_footprint if name == "x")
    web_kw = next(kw for name, kw in stub_footprint if name == "substack")
    assert _days_ago(x_kw["since"].isoformat()) == pytest.approx(183, abs=1)
    assert web_kw["since"] is None                       # the archive stayed UNBOUNDED


def test_a_deep_web_window_never_widens_the_x_pull(conn, stub_footprint, monkeypatch):
    """The mirror image, and the expensive one: `web_lookback="5yr"` must not hand 1825 days to
    the X adapter (which would clamp it to 2 years of the X stream, 4x the default)."""
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("blog", "https://nia.dev", True)]))
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    oracles.add_oracle(conn, object(), "@nia", confirm=True, web_lookback="5yr")

    x_kw = next(kw for name, kw in stub_footprint if name == "x")
    web_kw = next(kw for name, kw in stub_footprint if name == "blog")
    assert x_kw["since"] is None                         # X falls to its OWN ~6-month default
    assert _days_ago(web_kw["since"].isoformat()) == pytest.approx(1825, abs=1)


def test_the_report_matches_the_datetimes_actually_passed(conn, stub_footprint, monkeypatch):
    """A report that can disagree with the code IS the bug — so assert them against each other,
    not against a hardcoded string."""
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://nia.substack.com", True)]))
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    out = oracles.add_oracle(conn, object(), "@nia", confirm=True,
                             x_lookback="1yr", web_lookback="2yr")

    x_kw = next(kw for name, kw in stub_footprint if name == "x")
    web_kw = next(kw for name, kw in stub_footprint if name == "substack")
    assert out["lookback"]["x_since"] == x_kw["since"].isoformat()
    assert out["lookback"]["web_since"] == web_kw["since"].isoformat()


def _pair(conn, cid, source_type):
    from pipeline.kb import oracle_refresh_state as rst
    return next(r for r in rst.list_sources(conn, canonical_ids=[cid])
                if r.source_type == source_type)


def test_ingest_records_what_each_source_covered(conn, stub_footprint, monkeypatch):
    """Coverage is recorded PER SOURCE, on the `oracle_sources` row, because sources are pulled
    independently and fail independently. `covered_from` is the backward frontier and widens only.
    (The web pair's own stamp needs a real adapter to mint its entity — see
    `test_coverage_joins_a_result_url_back_to_its_registered_pair`, which builds the cluster.)"""
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://nia.substack.com", True)]))
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    oracles.add_oracle(conn, object(), "@nia", confirm=True,
                       x_lookback="6mo", web_lookback="all")

    assert _days_ago(_pair(conn, "x:user:7", "x").covered_from) == pytest.approx(183, abs=1)
    assert _pair(conn, "x:user:7", "x").last_pulled_at is not None

    # A LATER, WIDER run widens the record; it must never shrink back to the narrower window.
    oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="2yr", web_lookback="all")
    assert _days_ago(_pair(conn, "x:user:7", "x").covered_from) == pytest.approx(730, abs=1)

    oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="6mo", web_lookback="all")
    assert _days_ago(_pair(conn, "x:user:7", "x").covered_from) == pytest.approx(730, abs=1)


def test_coverage_joins_a_result_url_back_to_its_registered_pair(conn):
    """The join is DERIVED, not string-matched: an ingest reports outcomes per URL, the registry
    keys pairs off entity ids, and `derive.blog_entity_id` is what makes the two meet. A miss
    would leave the pair unstamped — safe, but it re-pulls a blog every session, and blog is the
    one source type whose spend scales with how often we poll."""
    from datetime import datetime, timedelta, timezone
    from pipeline.kb import oracle_refresh_state as rst

    now = datetime.now(timezone.utc)

    schema.upsert_entity(conn, "x:user:1", name="Will", profile={"handle": "willccbb"},
                         identity_links=["https://willcb.com", "https://github.com/willccbb"])
    schema.upsert_entity(conn, "blog:willcb.com", name="Will",
                         identity_links=["https://willcb.com"])
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:1", "blog:willcb.com": "x:user:1"})
    schema.upsert_oracle(conn, "x:user:1", name="Will")
    rst.seed_from_entities(conn, canonical_ids=["x:user:1"])

    web_since = now - timedelta(days=365)
    oracles._record_coverage(conn, "x:user:1", [
        # A trailing slash and a scheme the registry never stored — the derivation absorbs both.
        {"url": "http://willcb.com/", "type": "blog", "action": "ingested"},
        {"url": "https://x.com/WillCCBB", "type": "x", "action": "ingested"},
        {"url": "https://github.com/willccbb", "type": "github", "action": "blocked"},
    ], x_since=now - timedelta(days=183), web_since=web_since)

    assert _days_ago(_pair(conn, "x:user:1", "blog").covered_from) == pytest.approx(365, abs=1)
    assert _days_ago(_pair(conn, "x:user:1", "x").covered_from) == pytest.approx(183, abs=1)
    # GitHub was BLOCKED — nothing written, nothing marked seen, so nothing stamped.
    assert _pair(conn, "x:user:1", "github").last_pulled_at is None


def test_a_rate_limited_x_pull_defers_and_claims_nothing(conn, stub_footprint, monkeypatch):
    """THE defect this branch exists for. A 429 mid-onboarding used to be recorded as an `error`
    while the coverage write ran anyway, one line below the except that swallowed it — so the X
    pair was stamped as freshly pulled with zero X atoms behind it, `is_stale` said no, and no
    rail ever came back. The web half must still advance; the X half must stay untouched."""
    from pipeline.ingestion import x_graphql_core as core

    def _rate_limited(conn_, embedder, **kw):
        raise core.XRateLimited("UserTweets rate-limited (429) by x.com")

    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://nia.substack.com", True)]))
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint", _rate_limited)

    out = oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="6mo")

    ingest = out.get("ingest") or out
    x_rec = next(r for r in ingest["results"] if r["type"] == "x")
    assert x_rec["action"] == "deferred" and x_rec["resumes"] == "next-scheduled-run"
    assert ingest.get("errors", 0) == 0            # a rate window is the job working, not a fault

    x_pair = _pair(conn, "x:user:7", "x")
    assert x_pair.last_pulled_at is None and x_pair.covered_from is None
    # …while the off-X half of the same run still ran and reported.
    assert next(r for r in ingest["results"] if r["type"] == "substack")["action"] == "ingested"

    # …and the recovery that was disabled now fires: infinitely stale, so it sorts first.
    from pipeline.kb import oracle_refresh_state as rst
    assert rst.is_stale(x_pair) and rst.staleness_hours(x_pair) == float("inf")


def test_an_expired_x_session_requires_reconnect(conn, stub_footprint, monkeypatch):
    """A dead connection is user action, not work the background rail can complete."""
    from pipeline.ingestion.utils import SyncAuthError

    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    monkeypatch.setattr(ingest_x_footprint, "sync_x_footprint",
                        lambda *a, **kw: (_ for _ in ()).throw(SyncAuthError("session expired")))

    out = oracles.add_oracle(conn, object(), "@nia", confirm=True)

    ingest = out["ingest"]
    x_rec = next(r for r in ingest["results"] if r["type"] == "x")
    assert x_rec["action"] == "needs_reconnect"
    assert "deferred" not in ingest
    x_pair = _pair(conn, "x:user:7", "x")
    assert x_pair.last_pulled_at is None and x_pair.covered_from is None


# ── Mode B (local dedup) + Mode C (canonical promote) ───────────────────────────

def test_mode_b_matches_local_roster_without_any_network(conn, stub_footprint, monkeypatch):
    schema.upsert_entity(conn, "x:user:5", name="Dave", profile={"handle": "dave"})
    resolve.resolve_entities(conn)
    # A local match must never hit the network — blow up if the resolver is called.
    monkeypatch.setattr(oracles, "_fetch_x_identity",
                        lambda h: (_ for _ in ()).throw(AssertionError("should not fetch")))
    out = oracles.add_oracle(conn, object(), "@dave", confirm=True)

    assert out["added"]["canonical_id"] == "x:user:5" and out["added"]["source"] == "screen"
    rows = [r[0] for r in conn.execute(
        "SELECT entity_id FROM entities WHERE entity_id LIKE 'x:user:%'").fetchall()]
    assert rows == ["x:user:5"]                          # no duplicate minted


def test_mode_c_promotes_a_canonical_id(conn, stub_footprint):
    schema.upsert_entity(conn, "x:user:9", name="Below Cut", profile={"handle": "bc"})
    resolve.resolve_entities(conn)

    prev = oracles.add_oracle(conn, None, "x:user:9", confirm=False)
    assert prev["mode"] == "existing" and prev["resolved"]["already_oracle"] is False

    out = oracles.add_oracle(conn, object(), "x:user:9", confirm=True)
    assert out["added"]["canonical_id"] == "x:user:9" and out["added"]["source"] == "screen"
    assert schema.is_oracle(conn, "x:user:9")


def test_stale_canonical_id_reference_is_reported(conn):
    out = oracles.add_oracle(conn, None, "x:user:does-not-exist", confirm=False)
    assert "no entity for canonical_id" in out["error"]


def test_re_adding_an_existing_oracle_is_flagged(conn, stub_footprint, monkeypatch):
    schema.upsert_entity(conn, "x:user:5", name="Dave", profile={"handle": "dave"})
    resolve.resolve_entities(conn)
    oracles.confirm(conn, canonical_ids=["x:user:5"])
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: None)  # unused (local match)

    prev = oracles.add_oracle(conn, None, "@dave", confirm=False)
    assert prev["mode"] == "existing" and prev["resolved"]["already_oracle"] is True

    out = oracles.add_oracle(conn, object(), "@dave", confirm=True)
    assert out["added"]["was_already_oracle"] is True


def test_empty_reference_is_rejected(conn):
    assert "add_oracle needs a reference" in oracles.add_oracle(conn, None, "  ")["error"]


# ── MCP wiring ──────────────────────────────────────────────────────────────────

class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_register_exposes_both_tools():
    from mcp_server.oracle_tools import register_oracle_tools
    m = _FakeMCP()
    register_oracle_tools(m)
    assert "oracle" in m.tools and "add_oracle" in m.tools


def test_mcp_add_oracle_preview_builds_no_embedder_and_writes_nothing(kb_home, no_venue):
    """End-to-end through the registered @mcp.tool: a URL preview builds no embedder and writes
    nothing — exercising the real schema.connect() under the OPYT_HOME sandbox.

    It DOES make one free OpenAlex call now (the venue check), which is why `no_venue` is here.
    The embedder is the expensive half: it needs an API key and a model load, and a preview that
    built one would pay for a pull it is not making."""
    from mcp_server.oracle_tools import register_oracle_tools
    m = _FakeMCP()
    register_oracle_tools(m)
    out = m.tools["add_oracle"]("https://simonwillison.net", confirm=False)
    assert out["mode"] == "new" and out["resolved"]["platform"] == "blog"
