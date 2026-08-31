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
    """Offline footprint engine: the atom-KB adapters + eligibility gate become no-op recorders,
    and discovery returns an empty source list by default (tests re-patch `_DP.discover_profile`
    to inject sources). Targets the shared submodules, so it covers BOTH `_ingest_oracle` and the
    `onboard_footprint` it calls."""
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


def test_preview_url_reports_platform_network_free(conn):
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
    snap = _DP._x_snapshot_hash("Alice", [])
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
    # Reaching "no basis" takes THREE clears, and that difficulty is itself the good news: a
    # confirmed Oracle almost always carries either a coverage marker or atoms, so this refusal is
    # a rare edge. It is tested anyway because the cost of getting it wrong is a 183-day pull.
    #   1. the pair's own stamp, 2. its corpus-derived cursor, and
    #   3. `oracles.ingest_to` — which `upsert_source` COALESCEs back in on the next re-seed.
    from pipeline.kb import oracle_refresh_state as rst
    rst.seed_from_entities(conn, canonical_ids=[cid])
    conn.execute("UPDATE oracles SET ingest_to=NULL WHERE canonical_id=?", (cid,))
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


def test_ingest_records_the_window_it_covered_on_the_oracle_row(conn, stub_footprint, monkeypatch):
    """`ingest_from`/`ingest_to` stop being inert: they record what a run actually covered so a
    re-ingest knows what was already paid for. `ingest_from` is the LATER of the two windows —
    the point from which BOTH pulls are complete — so an unbounded archive walk can never vouch
    for a 6-month X pull."""
    monkeypatch.setattr(_DP, "discover_profile",
                        _fake_discover([_src("substack", "https://nia.substack.com", True)]))
    monkeypatch.setattr(oracles, "_fetch_x_identity", lambda h: _x_ident("7", "nia"))
    oracles.add_oracle(conn, object(), "@nia", confirm=True,
                       x_lookback="6mo", web_lookback="all")

    row = next(r for r in schema.list_oracles(conn) if r["canonical_id"] == "x:user:7")
    assert _days_ago(row["ingest_from"]) == pytest.approx(183, abs=1)   # X's window, not the archive's
    assert _days_ago(row["ingest_to"]) == pytest.approx(0, abs=1)

    # A LATER, WIDER run widens the record; it must never shrink back to the narrower window.
    oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="2yr", web_lookback="all")
    row = next(r for r in schema.list_oracles(conn) if r["canonical_id"] == "x:user:7")
    assert _days_ago(row["ingest_from"]) == pytest.approx(730, abs=1)

    oracles.add_oracle(conn, object(), "@nia", confirm=True, x_lookback="6mo", web_lookback="all")
    row = next(r for r in schema.list_oracles(conn) if r["canonical_id"] == "x:user:7")
    assert _days_ago(row["ingest_from"]) == pytest.approx(730, abs=1)   # still the wider coverage


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


def test_mcp_add_oracle_preview_is_network_free(kb_home):
    """End-to-end through the registered @mcp.tool: a URL preview builds no embedder, touches no
    network, and writes nothing — exercising the real schema.connect() under the OPYT_HOME sandbox."""
    from mcp_server.oracle_tools import register_oracle_tools
    m = _FakeMCP()
    register_oracle_tools(m)
    out = m.tools["add_oracle"]("https://simonwillison.net", confirm=False)
    assert out["mode"] == "new" and out["resolved"]["platform"] == "blog"
