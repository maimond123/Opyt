"""add_oracle — the atom-native "add a person".

Proves the load-bearing behavior fully OFFLINE (discovery + the footprint adapters are stubbed;
the X identity fetch is monkeypatched):
  • the two-phase confirm gate — confirm=False PREVIEWS and writes nothing; an unresolvable
    reference is reported, never confirmable (the anti-hallucination guard);
  • confirm=True adds them and records what they are owed, walking nothing inside the call;
  • Mode B (network-free local dedup off `profile.handle`) and Mode C (promote a canonical_id);
  • the lookback windows are recorded separately, each on its own kind.
"""
from __future__ import annotations


import pytest

from pipeline.kb import (resolve, schema)
from pipeline.oracles import windows
from pipeline.platforms import connections
from pipeline.timeparse import utc_now
from pipeline.oracles import roster
from pipeline.oracles import identify
from pipeline.oracles import preview
from pipeline.oracles import coverage
from pipeline.forget import oracle as forget_oracle
from pipeline.oracles import admit
from pipeline.oracles import found
from pipeline.oracles.sources.archive import state as archive_state
from pipeline.oracles.sources.web import archives
from pipeline.oracles.sources import scraper_tables

from pipeline.oracles.discovery import profile as _DP


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


@pytest.fixture(autouse=True)
def managed_x_session(monkeypatch):
    """Existing Oracle-ingest tests exercise the connected-X path unless they opt out."""
    monkeypatch.setattr(connections, "_connected_x", lambda: True)


def _fake_discover(sources):
    # Discovery calls discover_profile(seed, seed_type=...).
    return lambda seed, seed_type="x", **kw: {"username": seed, "sources": sources}


def _src(stype, url, trusted, **meta):
    return {"source_type": stype, "url": url, "metadata": meta,
            "trust": {"trusted": trusted, "reasons": []}}


@pytest.fixture()
def stub_footprint(monkeypatch):
    """Offline footprint engine: discovery returns an empty source list by default (tests re-patch
    `_DP.discover_profile` to inject sources). Nothing is pulled inside a call: a first pull only
    begins a walk the worker runs."""
    monkeypatch.setattr(_DP, "discover_profile", _fake_discover([]))


def _found(conn, cid, rows):
    """Seed review items the way discovery records a source it is sure of."""
    from pipeline.oracles import review_items
    with conn:
        review_items.write_found(conn, cid, [
            {"source_type": r["type"], "url": r["url"], "status": review_items.ADDED,
             "reasons": [r["detail"]] if r.get("detail") else []} for r in rows])


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
    monkeypatch.setattr(identify, "fetch_x_identity",
                        lambda h: _x_ident("42", "kay", name="Kay", followers=1200))
    out = admit.add_oracle(conn, "@kay", confirm=False)
    assert out["confirm_required"] is True and out["mode"] == "new"
    assert out["resolved"]["name"] == "Kay" and out["resolved"]["followers"] == 1200
    assert out["resolved"]["root_entity"] == "x:user:42"
    # No writes: the profile fetch is read-only, nothing is minted or confirmed.
    assert schema.get_entity(conn, "x:user:42") is None
    assert roster.list_oracles(conn) == []


def test_preview_unresolvable_handle_is_reported_not_written(conn, monkeypatch):
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: None)
    out = admit.add_oracle(conn, "@ghost", confirm=False)
    assert out["unresolved"] == "@ghost"                 # nothing to confirm — the anti-hallucination guard
    assert roster.list_oracles(conn) == []


def test_preview_url_reports_the_platform_and_writes_nothing(conn, no_venue):
    """Rooting a site asks OpenAlex once whether the host is a research venue, because a preview
    that said "blog" and then minted a 63,000-work venue would be lying on the consent surface.
    What this pins is that a preview writes nothing."""
    blog = admit.add_oracle(conn, "https://simonwillison.net", confirm=False)
    assert blog["mode"] == "new" and blog["resolved"]["platform"] == "blog"
    assert blog["resolved"]["root_entity"].startswith("blog:")

    sub = admit.add_oracle(conn, "https://carol.substack.com", confirm=False)
    assert sub["resolved"]["platform"] == "substack"
    assert roster.list_oracles(conn) == []               # still no writes


def test_preview_reports_lookback_windows(conn):
    default = admit.add_oracle(conn, "https://x.dev", confirm=False)
    assert "6-month default" in default["lookback"]["x"]
    assert default["lookback"]["web"] == "full archive" and default["lookback"]["web_since"] is None

    scoped = admit.add_oracle(conn, "https://x.dev", confirm=False,
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
    rep = preview._lookback_report(five_years, None)
    assert "CLAMPED" in rep["x"]
    assert _days_ago(rep["x_since"]) == pytest.approx(730, abs=1)   # the 2-year ceiling, not 1825


# ── Phase 2: the chain + the trust-seed weld ────────────────────────────────────

def test_confirm_adds_them_and_walks_nothing(conn, stub_footprint, monkeypatch):
    """The walk left the call (commit 5): the background scrapers pull what the ask records."""
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("7", "nia", name="Nia"))
    out = admit.add_oracle(conn, "@nia", confirm=True)

    assert out["added"]["canonical_id"] == "x:user:7"
    assert out["added"]["source"] == "freeform"
    assert out["added"]["was_already_oracle"] is False
    assert roster.on_roster(conn, "x:user:7")
    assert "ingest" not in out and "lookback" not in out


def _tool(monkeypatch, *args, **kw):
    """The registered `add_oracle` tool against the sandboxed store, waking nothing."""
    from mcp_server.oracle_tools import register_oracle_tools
    from pipeline.worker import liveness
    monkeypatch.setattr("pipeline.worker.job_queue.request_now", lambda *a, **k: True)
    monkeypatch.setattr(liveness, "repair_or_report_cached", lambda **kw: {"ok": None})
    m = _FakeMCP()
    register_oracle_tools(m)
    return m.tools["add_oracle"](*args, **kw)


def _ask(cid):
    from pipeline.oracles import asks
    c = schema.connect()
    try:
        return asks.ask_for(c, cid)
    finally:
        c.close()


def test_the_tool_records_the_ask_and_says_where_they_stand(kb_home, stub_footprint, monkeypatch):
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("7", "nia", name="Nia"))
    out = _tool(monkeypatch, "@nia", confirm=True, x_lookback="2yr")

    assert out["picks"][0]["canonical_id"] == "x:user:7"
    assert out["picks"][0]["newest_posts"] is False
    assert _days_ago(_ask("x:user:7").x_from) == pytest.approx(730, abs=1)
    assert out["picks"][0]["asked"]["x"] == _ask("x:user:7").x_from


def test_a_bad_window_refuses_before_anyone_is_added(kb_home, monkeypatch):
    out = _tool(monkeypatch, "@nia", confirm=True, x_lookback="1y")
    assert "error" in out
    c = schema.connect()
    assert roster.list_oracles(c) == []
    c.close()


# ── the trust boundary: rejected is REPORTED, never dropped ─────────────────────

def _confirmed(conn, cid="x:user:1", *, name="Carol", handle="carol"):
    schema.upsert_entity(conn, cid, name=name, profile={"handle": handle})
    resolve.resolve_entities(conn)
    admit.confirm(conn, canonical_ids=[cid])
    return next(x for x in roster.confirmed_oracles(conn) if x["canonical_id"] == cid)


def test_forgetting_an_oracle_removes_its_review_queue(conn):
    """Review choices belong to an active Oracle subscription, not retained profile history."""
    from pipeline.oracles import review_items

    _confirmed(conn)
    _found(conn, "x:user:1", [{"type": "blog", "url": "https://maybe-carol.dev"}])

    out = forget_oracle._forget(conn, "x:user:1", confirm=True)

    assert out["status"] == "forgotten"
    assert review_items.held_source_keys(conn, "x:user:1") == set()


def test_review_hold_follows_an_oracle_after_its_cluster_head_changes(conn):
    """A hold cannot vanish when resolve picks a different cluster anchor."""
    from pipeline.oracles import review_items

    schema.upsert_entity(conn, "x:user:1", name="Carol", profile={"handle": "carol"})
    _found(conn, "x:user:1", [{"type": "blog", "url": "https://maybe-carol.dev"}])
    conn.execute("UPDATE entities SET canonical_id=? WHERE entity_id=?", ("blog:carol", "x:user:1"))

    assert ("blog", "maybe-carol.dev") in review_items.held_source_keys(conn, "blog:carol")


# ── rooting refuses what it cannot root ────────────────────────────────────────

@pytest.mark.parametrize("url", ["https://github.com/karpathy", "https://x.com/karpathy",
                                 "https://twitter.com/karpathy", "https://youtube.com/@karpathy",
                                 "https://m.youtube.com/@karpathy", "https://youtu.be/video"])
def test_a_platform_profile_url_is_refused_not_minted_as_a_blog(conn, url):
    """Keying every non-Substack http… reference on `blog:{host}` would mint
    `blog:github.com/karpathy` for `https://github.com/karpathy` and root discovery on it as a
    personal site. Worst case is an X URL: `blog:x.com/karpathy` for a person who already has an
    `x:user:` identity, i.e. a second, permanently unmergeable copy of them."""
    out = admit.add_oracle(conn, url, confirm=False)
    assert "error" in out and "cannot add" in out["error"]
    assert schema.get_entity(conn, "blog:github.com/karpathy") is None


def test_an_x_url_says_to_pass_the_handle(conn):
    """The refusal has to be actionable — there IS a right way to add this person."""
    out = admit.add_oracle(conn, "https://x.com/karpathy", confirm=False)
    assert "@handle" in out["error"]


def test_a_real_personal_site_is_still_rootable(conn, no_venue):
    """The guard names four platform hosts; everything else keeps working — including after the
    venue branch went in front of the blog fallthrough."""
    out = admit.add_oracle(conn, "https://simonwillison.net", confirm=False)
    assert out["resolved"]["platform"] == "blog"


# ── the X window refuses rather than falling through ──────────────────────────

@pytest.mark.parametrize("preset", ["1y", "6m", "2years", "all"])
def test_an_unknown_x_preset_raises_rather_than_buying_183_days(preset):
    """`X_LOOKBACK_PRESETS.get(preset)` would return None for a typo, and on the X selector None
    means the adapter's own 183-day default, so `x_lookback='1y'` would silently buy the widest
    pull available. The host supplies this string, so a typo is one call away.
    Same argument the `since_last` refusal was built on, applied to the same failure."""
    with pytest.raises(ValueError, match="unknown x_lookback"):
        windows.since("x", preset, utc_now())


def test_the_real_presets_still_resolve():
    assert windows.since("x", None, utc_now()) is None                    # unset = the adapter's own default
    assert windows.since("x", "6mo", utc_now()) is not None


def test_add_oracle_can_force_too(conn, stub_footprint, monkeypatch):
    """`add_oracle(force=True)` asks discovery to run again for someone already added — the tool a
    user reaches for when they say "her sources look wrong, add her again"."""
    seen = {}

    def _capture(seed, seed_type="x", **kw):
        seen.update(kw)
        return {"username": seed, "sources": []}

    monkeypatch.setattr(_DP, "discover_profile", _capture)
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("23", "dana"))
    _tool(monkeypatch, "@dana", confirm=True, force=True)
    assert _ask("x:user:23").rediscover            # discovery runs again


# ── since_last: the cheap top-up ────────────────────────────────────────────────
#
# Every other X preset is a fixed span, identical for everyone. This one is "since I last pulled
# THIS person", so it resolves per-Oracle. It exists because the automatic loop pulls a
# since-last-pull delta while the narrowest window a USER could ask for began at 6 months —
# ~19x the requests for an average poster, to fetch the same handful of posts.

def _seed_last_pull(conn, cid: str, handle: str, hours_ago: float):
    """Register an X pair for `cid` and stamp its last pull `hours_ago` in the past."""
    from datetime import datetime, timedelta, timezone

    from pipeline.oracles import pairs
    pairs.seed_from_entities(conn, canonical_ids=[cid])
    when = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    conn.execute("UPDATE oracle_sources SET last_pulled_at=? WHERE canonical_id=? "
                 "AND source_type='x'", (when, cid))
    conn.commit()
    return conn.execute("SELECT COUNT(*) FROM oracle_sources WHERE canonical_id=? "
                        "AND source_type='x'", (cid,)).fetchone()[0]


def test_since_last_checks_the_newest_posts_instead_of_asking_again(conn, stub_footprint,
                                                                  monkeypatch):
    """A top-up is a check (`checks.request_check`): the newest posts, at your priority, and
    nothing else asked again."""
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("7", "nia"))
    _tool(monkeypatch, "@nia", confirm=True, x_lookback="6mo")
    cid = schema.current_canonical(conn, "x:user:7")
    assert _seed_last_pull(conn, cid, "nia", hours_ago=120.0) == 1
    asked_at = _ask(cid).asked_at

    out = _tool(monkeypatch, "@nia", confirm=True, x_lookback=windows.X_SINCE_LAST)

    assert out["checking"] == [out["picks"][0]["name"]]
    assert _pair(conn, cid, "x").check_requested_at is not None
    assert _ask(cid).asked_at == asked_at


def test_a_top_up_of_someone_never_pulled_is_their_first_ask(conn, stub_footprint, monkeypatch):
    """It used to refuse, because the window it would fall back to was the adapter's 183 days.
    A check reads the newest page and nothing more, so someone never pulled gets the full first
    ask instead of a refusal or a silent nothing."""
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("9", "zed"))
    _tool(monkeypatch, "@zed", confirm=True, x_lookback=windows.X_SINCE_LAST)
    ask = _ask("x:user:9")
    assert ask.web_from == windows.ALL_TIME
    assert _days_ago(ask.x_from) == pytest.approx(183, abs=1)


def test_x_since_never_silently_resolves_since_last(conn):
    """The batch resolver raises instead of returning None, so a caller that forgets to route
    `since_last` per-Oracle fails loudly rather than buying 183 days for everyone."""
    with pytest.raises(ValueError, match="per-Oracle"):
        windows.since("x", windows.X_SINCE_LAST, utc_now())


def test_the_windows_are_recorded_each_on_its_own_kind(conn, stub_footprint, monkeypatch):
    """THE regression this pair of selectors exists for: ONE `since` once went to both adapters,
    so a short X window truncated the durable archive and a deep archive pulled 2 years of X.
    The ask keeps them apart; the walks read each (`windows.target`, `windows.ask_since`)."""
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("7", "nia"))
    _tool(monkeypatch, "@nia", confirm=True, x_lookback="6mo", web_lookback="all")
    ask = _ask("x:user:7")
    assert _days_ago(ask.x_from) == pytest.approx(183, abs=1)
    assert ask.web_from == windows.ALL_TIME                 # the archive stays UNBOUNDED

    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: _x_ident("8", "ola"))
    _tool(monkeypatch, "@ola", confirm=True, web_lookback="5yr")
    ask = _ask("x:user:8")
    assert _days_ago(ask.x_from) == pytest.approx(183, abs=1)   # never widened by the web's
    assert _days_ago(ask.web_from) == pytest.approx(1825, abs=1)


def _pair(conn, cid, source_type):
    from pipeline.oracles import registry
    return next(r for r in registry.list_sources(conn, canonical_ids=[cid])
                if r.source_type == source_type)


def test_the_asked_window_is_recorded_per_pair(conn):
    """`requested_from` is written where the user's window is already known — alongside the
    coverage stamp — so the background deepen target (`oracle_refresh.target_for`) can honour a
    1-year ask after the foreground run is gone."""
    from datetime import datetime, timedelta, timezone
    from pipeline.oracles import pairs

    now = datetime.now(timezone.utc)
    schema.upsert_entity(conn, "x:user:9", name="Ana", profile={"handle": "ana"})
    schema.set_canonical_ids(conn, {"x:user:9": "x:user:9"})
    roster.upsert_oracle(conn, "x:user:9", name="Ana")
    pairs.seed_from_entities(conn, canonical_ids=["x:user:9"])

    coverage.record_coverage(conn, "x:user:9", [
        {"url": "https://x.com/ana", "type": "x", "action": "ingested",
         # A partial walk: measured reach is shallow, the ASK was a year.
         "covered_from": (now - timedelta(days=40)).isoformat()},
    ], x_since=now - timedelta(days=365), web_since=None)

    pair = _pair(conn, "x:user:9", "x")
    assert _days_ago(pair.covered_from) == pytest.approx(40, abs=1)     # what it reached
    assert _days_ago(pair.requested_from) == pytest.approx(365, abs=1)  # what was asked


def test_a_deferred_pull_still_records_what_was_asked(conn):
    """THE DEFERRED CASE, which is the only one where this column has to survive on its own.

    Measured on a real store: four X timelines asked for a year, x.com rate-limited
    all four, and `requested_from` stayed NULL on every pair — the ask was written only as a side
    effect of a SUCCESSFUL pull, and none succeeded. `oracle_refresh.target_for` reads NULL as
    "nobody asked" and falls back to the 183-day default, so the background deepen would have
    stopped at six months and called itself done.

    The pair must ALSO still read as unpulled: recording the ask is not observing the window, and
    a deferred pair that looks stamped is a pair the refresh rail stops taking."""
    from datetime import datetime, timedelta, timezone
    from pipeline.oracles import pairs

    now = datetime.now(timezone.utc)
    schema.upsert_entity(conn, "x:user:11", name="Ivo", profile={"handle": "ivo"})
    schema.set_canonical_ids(conn, {"x:user:11": "x:user:11"})
    roster.upsert_oracle(conn, "x:user:11", name="Ivo")
    pairs.seed_from_entities(conn, canonical_ids=["x:user:11"])

    coverage.record_coverage(conn, "x:user:11", [
        {"url": "https://x.com/ivo", "type": "x", "action": "deferred",
         "detail": "XRateLimited: x.com cut off both timelines"},
    ], x_since=now - timedelta(days=365), web_since=None)

    pair = _pair(conn, "x:user:11", "x")
    assert _days_ago(pair.requested_from) == pytest.approx(365, abs=1)  # the ask survived
    assert pair.covered_from is None            # nothing was reached, so nothing is claimed
    assert pair.last_pulled_at is None          # and nothing marks it pulled — it stays due


def test_recording_an_ask_never_narrows_an_earlier_wider_one(conn):
    """Widen-only, like `covered_from`. A 6-month top-up after a 1-year ask must not cancel the
    year: the deepen target would quietly retreat and nothing would report that it had."""
    from datetime import datetime, timedelta, timezone
    from pipeline.oracles import pairs

    now = datetime.now(timezone.utc)
    schema.upsert_entity(conn, "x:user:12", name="Gil", profile={"handle": "gil"})
    schema.set_canonical_ids(conn, {"x:user:12": "x:user:12"})
    roster.upsert_oracle(conn, "x:user:12", name="Gil")
    pairs.seed_from_entities(conn, canonical_ids=["x:user:12"])

    deferred = [{"url": "https://x.com/gil", "type": "x", "action": "deferred"}]
    coverage.record_coverage(conn, "x:user:12", deferred,
                             x_since=now - timedelta(days=365), web_since=None)
    coverage.record_coverage(conn, "x:user:12", deferred,
                             x_since=now - timedelta(days=180), web_since=None)

    assert _days_ago(_pair(conn, "x:user:12", "x").requested_from) == pytest.approx(365, abs=1)


def test_coverage_joins_a_result_url_back_to_its_registered_pair(conn):
    """The join is DERIVED, not string-matched: an ingest reports outcomes per URL, the registry
    keys pairs off entity ids, and `derive.blog_entity_id` is what makes the two meet. A miss
    would leave the pair unstamped — safe, but it re-pulls a blog every session, and blog is the
    one source type whose spend scales with how often we poll."""
    from datetime import datetime, timedelta, timezone
    from pipeline.oracles import pairs

    now = datetime.now(timezone.utc)

    schema.upsert_entity(conn, "x:user:1", name="Will", profile={"handle": "willccbb"},
                         identity_links=["https://willcb.com", "https://github.com/willccbb"])
    schema.upsert_entity(conn, "blog:willcb.com", name="Will",
                         identity_links=["https://willcb.com"])
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:1", "blog:willcb.com": "x:user:1"})
    roster.upsert_oracle(conn, "x:user:1", name="Will")
    pairs.seed_from_entities(conn, canonical_ids=["x:user:1"])

    web_since = now - timedelta(days=365)
    coverage.record_coverage(conn, "x:user:1", [
        # A trailing slash and a scheme the registry never stored — the derivation absorbs both.
        {"url": "http://willcb.com/", "type": "blog", "action": "ingested"},
        {"url": "https://x.com/WillCCBB", "type": "x", "action": "ingested"},
        {"url": "https://github.com/willccbb", "type": "github", "action": "blocked"},
    ], x_since=now - timedelta(days=183), web_since=web_since)

    assert _days_ago(_pair(conn, "x:user:1", "blog").covered_from) == pytest.approx(365, abs=1)
    assert _days_ago(_pair(conn, "x:user:1", "x").covered_from) == pytest.approx(183, abs=1)
    # GitHub was BLOCKED — nothing written, nothing marked seen, so nothing stamped.
    assert _pair(conn, "x:user:1", "github").last_pulled_at is None


# ── Mode B (local dedup) + Mode C (canonical promote) ───────────────────────────

def test_mode_b_matches_local_roster_without_any_network(conn, stub_footprint, monkeypatch):
    schema.upsert_entity(conn, "x:user:5", name="Dave", profile={"handle": "dave"})
    resolve.resolve_entities(conn)
    # A local match must never hit the network — blow up if the resolver is called.
    monkeypatch.setattr(identify, "fetch_x_identity",
                        lambda h: (_ for _ in ()).throw(AssertionError("should not fetch")))
    out = admit.add_oracle(conn, "@dave", confirm=True)

    assert out["added"]["canonical_id"] == "x:user:5" and out["added"]["source"] == "screen"
    rows = [r[0] for r in conn.execute(
        "SELECT entity_id FROM entities WHERE entity_id LIKE 'x:user:%'").fetchall()]
    assert rows == ["x:user:5"]                          # no duplicate minted


def test_mode_c_promotes_a_canonical_id(conn, stub_footprint):
    schema.upsert_entity(conn, "x:user:9", name="Below Cut", profile={"handle": "bc"})
    resolve.resolve_entities(conn)

    prev = admit.add_oracle(conn, "x:user:9", confirm=False)
    assert prev["mode"] == "existing" and prev["resolved"]["already_oracle"] is False

    out = admit.add_oracle(conn, "x:user:9", confirm=True)
    assert out["added"]["canonical_id"] == "x:user:9" and out["added"]["source"] == "screen"
    assert roster.on_roster(conn, "x:user:9")


def test_stale_canonical_id_reference_is_reported(conn):
    out = admit.add_oracle(conn, "x:user:does-not-exist", confirm=False)
    assert "no entity for canonical_id" in out["error"]


def test_re_adding_an_existing_oracle_is_flagged(conn, stub_footprint, monkeypatch):
    schema.upsert_entity(conn, "x:user:5", name="Dave", profile={"handle": "dave"})
    resolve.resolve_entities(conn)
    admit.confirm(conn, canonical_ids=["x:user:5"])
    monkeypatch.setattr(identify, "fetch_x_identity", lambda h: None)  # unused (local match)

    prev = admit.add_oracle(conn, "@dave", confirm=False)
    assert prev["mode"] == "existing" and prev["resolved"]["already_oracle"] is True

    out = admit.add_oracle(conn, "@dave", confirm=True)
    assert out["added"]["was_already_oracle"] is True


def test_empty_reference_is_rejected(conn):
    assert "add_oracle needs a reference" in admit.add_oracle(conn, "  ")["error"]


# ── MCP wiring ──────────────────────────────────────────────────────────────────

class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_register_exposes_both_tools():
    from mcp_server.oracle_tools import register_oracle_tools
    m = _FakeMCP()
    register_oracle_tools(m)
    assert "oracle" in m.tools and "add_oracle" in m.tools


def test_mcp_add_oracle_preview_resolves_through_the_registered_tool(kb_home, no_venue):
    """End-to-end through the registered @mcp.tool: a URL preview resolves the platform,
    exercising the real schema.connect() under the OPYT_HOME sandbox.

    It DOES make one free OpenAlex call now (the venue check), which is why `no_venue` is here."""
    from mcp_server.oracle_tools import register_oracle_tools
    m = _FakeMCP()
    register_oracle_tools(m)
    out = m.tools["add_oracle"]("https://simonwillison.net", confirm=False)
    assert out["mode"] == "new" and out["resolved"]["platform"] == "blog"
