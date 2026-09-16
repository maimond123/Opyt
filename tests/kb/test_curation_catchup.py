"""The curation catch-up rail — the spawner, the floor, single-flight, failure isolation.

Same brief as `test_bookmark_catchup.py`: everything here fails SILENTLY in production. A rail whose
candidate list never refreshes is indistinguishable from one with nothing new to add, which is
exactly how the collectors this drives went from 2026-07-21 to 2026-08-12 with no automatic
trigger and nobody noticing. So the wiring gets asserted — a burnt coalesce stamp suppresses the
next hour of spawns, a leaked fd accumulates one per session, a kill switch that does not kill is a
dev machine scraping X on a loop, and a floor that gates on SUCCESS instead of ATTEMPT re-runs a
broken collector on every session open.
"""
from __future__ import annotations

import subprocess

import pytest

from pipeline.kb import curation_catchup as cc
from pipeline.kb import curation_state as cs
from pipeline.kb import ingest_curation as ic
from pipeline.kb import schema


@pytest.fixture()
def rail_home(kb_home, monkeypatch):
    # Consent granted once here because every test using this fixture exercises the rail's
    # MECHANICS — floor, single-flight, collector dispatch — not its gate. A sandboxed home has no
    # atoms, so `_established_store()` is False and an ungranted rail would return `needs_consent`
    # before reaching anything these tests are about. The gate has its own tests below.
    cc.grant_consent()
    # Same reason, for the other gate: with no managed X session the three X collectors are
    # skipped before dispatch, so a mechanics test would assert against one collector instead of
    # four — and the result would depend on whether the machine running the suite has an OPYT X
    # profile. `_platform_reachable`'s own tests set this to False deliberately.
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: True)
    # Since 2026-09-13 Substack gates the SAME way — `readable()` reads the managed session, no
    # user-browser fallback — so the two Substack collectors need this stub for the same reason
    # the X one above exists. The managed-session gate has its own tests below.
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: True)
    return kb_home


@pytest.fixture()
def collectors(monkeypatch):
    """Replace every collector with a recording stub and hand back the call log, so a test can
    assert both what ran and what did NOT. Each stub returns the key its own spec declares, so the
    clock stamps a real `found` — the spec/collector agreement itself is proven in
    `test_ingest_curation.py` against the real collectors."""
    calls: list[str] = []

    def _make(spec):
        def _fn(conn, *, profile=None):
            calls.append(spec.collector)
            return {"source": spec.label, spec.found_key: 1}
        return _fn

    for spec in ic.COLLECTOR_SPECS:
        monkeypatch.setattr(ic, spec.fn_name, _make(spec))
    return calls


def _boom(*a, **kw):
    raise RuntimeError("dead X session")


# ── the floor ───────────────────────────────────────────────────────────────────
def test_a_first_pass_runs_every_collector(rail_home, collectors):
    out = cc.run_curation_catchup()
    assert out["status"] == "ok"
    assert set(collectors) == set(ic.COLLECTORS)
    assert out["skipped_within_floor"] == []
    assert out["errors"] == 0
    assert out["freshness"]["needs_attention"] is False


def test_a_collector_inside_its_floor_is_skipped_without_running(rail_home, collectors):
    """The floor is what makes an hourly spawn cheap: the second pass is a lock acquire, four
    SELECTs and an exit, with no network call at all."""
    cc.run_curation_catchup()
    collectors.clear()

    second = cc.run_curation_catchup()

    assert collectors == []                                    # nothing hit the network
    assert second["ran"] == {}
    assert set(second["skipped_within_floor"]) == set(ic.COLLECTORS)


def test_force_ignores_the_floor(rail_home, collectors):
    cc.run_curation_catchup()
    collectors.clear()
    out = cc.run_curation_catchup(force=True)
    assert set(collectors) == set(ic.COLLECTORS)
    assert out["skipped_within_floor"] == []


def test_the_floor_counts_attempts_not_successes(rail_home, collectors, monkeypatch):
    """PINNED. Gating on `last_ok_at` would remove the floor from exactly the collector that most
    needs one: a dead X session would be retried on EVERY session open, forever."""
    monkeypatch.setattr(ic, "sync_following_signals", _boom)
    first = cc.run_curation_catchup()
    assert "error" in first["ran"]["x_following"]

    second = cc.run_curation_catchup()

    assert "x_following" in second["skipped_within_floor"]
    assert second["ran"] == {}


# ── failure isolation ───────────────────────────────────────────────────────────
def test_one_collector_raising_does_not_stop_the_other_four(rail_home, collectors, monkeypatch):
    monkeypatch.setattr(ic, "sync_likes_signals", _boom)

    out = cc.run_curation_catchup()

    assert out["status"] == "ok" and out["errors"] == 1
    assert "error" in out["ran"]["x_likes"]
    assert set(collectors) == {"x_lists", "x_following", "x_bookmark_signals",
                               "substack_follows", "substack_subscriptions",
                               "substack_saved_signals"}

    conn = schema.connect()
    try:
        assert cs.get_run(conn, "x_likes").last_status == "error"
        assert cs.get_run(conn, "x_likes").last_ok_at is None
        assert cs.get_run(conn, "x_following").last_status == "ok"
    finally:
        conn.close()


def test_it_never_raises(rail_home, monkeypatch):
    """Fail-safe invariant. This runs in a detached child whose only caller is a `-m` entrypoint,
    so a propagated exception is a traceback in a log file nobody opens."""
    monkeypatch.setattr("pipeline.kb.schema.connect", _boom)
    out = cc.run_curation_catchup()
    assert out["status"] == "error" and "dead X session" in out["error"]


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_catchup_skips_while_one_holds_the_lease(rail_home, collectors):
    """A free scrape run twice at once is still twice the requests against a cookie session, and
    that is what gets an account rate-limited."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("curation-catchup") as held:
        assert held.acquired
        out = cc.run_curation_catchup()

    assert out["status"] == "already_running"
    assert collectors == []


def test_force_does_not_bypass_single_flight(rail_home, collectors):
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("curation-catchup") as held:
        assert held.acquired
        assert cc.run_curation_catchup(force=True)["status"] == "already_running"
    assert collectors == []


def test_an_evicted_worker_stops_before_the_next_collector(rail_home, monkeypatch):
    """Lease loss fences this worker out before it starts a second collector."""
    from pipeline import sync_lock

    class _EvictedLock:
        current = None

        def __init__(self, *args, **kwargs):
            self.acquired = True
            self.evicted = False
            type(self).current = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def lost(self):
            return self.evicted

    ran = []

    def run_one(conn, spec):
        ran.append(spec.collector)
        _EvictedLock.current.evicted = True
        return {"found": 0}

    monkeypatch.setattr(sync_lock, "CatchupLock", _EvictedLock)
    monkeypatch.setattr(ic, "run_and_record", run_one)
    monkeypatch.setattr(ic, "resolve_after_pull",
                        lambda *args: pytest.fail("must not resolve after lease loss"))

    out = cc.run_curation_catchup()

    assert out["status"] == "lease_lost"
    assert len(ran) == 1


def test_the_lease_is_not_shared_with_the_bookmark_rail(rail_home, collectors):
    """Different names, different leases. Sharing one would make a long bookmark backfill block
    the free list refresh for its whole duration."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("bookmark-catchup") as held:
        assert held.acquired
        out = cc.run_curation_catchup()
    assert out["status"] == "ok"
    assert set(collectors) == set(ic.COLLECTORS)


# ── the trap: the tiered ladder must never be on this path ──────────────────────
def test_it_never_routes_through_the_tiered_ladder(rail_home, collectors, monkeypatch):
    """⚠️ THE defect this rail was designed around. `curation_pull(tiered=True)` gates on the WHOLE
    STORE's signalled-entity count, not the run's own yield, so on any established store it clears
    `sufficient_at` after Tier 1 and permanently skips following and likes — the exact two
    collectors this rail exists to refresh. It would look like it was working."""
    called: list = []
    monkeypatch.setattr(ic, "curation_pull", lambda *a, **k: called.append(1) or {})

    cc.run_curation_catchup()

    assert called == []
    assert "x_following" in collectors and "x_likes" in collectors


# ── the rail resolves what it minted ────────────────────────────────────────────
#
# ⚠️ THE SECOND CALL SITE, and the reason it is not redundant is the test directly above: this rail
# deliberately never enters `curation_pull`, so it cannot reach the resolve at the end of `_done`.
# Without its own call, the ONLY automatic curation path in the product mints new people and leaves
# every one of them unresolved — and an unresolved person is two candidates carrying one signal each
# where the pre-tick bar is ≥2, so they are dropped before a human ever sees them.

@pytest.fixture()
def one_person_two_platforms(monkeypatch):
    """Patch the FETCH layer, leaving the collectors REAL, so the entities and identity_links
    a merge needs actually get written. One human: an X follow whose bio site is the home of a
    Substack the user subscribes to."""
    site = "https://acme.substack.com"
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_likes, x_lists
    from pipeline.ingestion.sources import substack as sub

    monkeypatch.setattr(core, "read_x_cookies", lambda: {"twid": "u=1"})
    monkeypatch.setattr(core, "viewer_id", lambda cookies: "1")
    monkeypatch.setattr(core, "auth_headers", lambda cookies, referer: {})
    monkeypatch.setattr(core, "fetch_following", lambda c, h, v: [
        {"user_id": "2", "display_name": "Acme Author", "site": site}])
    monkeypatch.setattr(x_lists, "fetch_owned_lists", lambda c, h, v: [])
    monkeypatch.setattr(x_lists, "fetch_list_members", lambda lid, c, h: [])
    monkeypatch.setattr(x_lists, "aggregate_members", lambda owned, by_list, vid: [])
    monkeypatch.setattr(x_likes, "fetch_liked_authors", lambda vid, c, h: [])
    monkeypatch.setattr(x_likes, "aggregate_authors", lambda authors, vid: [])
    monkeypatch.setattr(sub, "read_substack_cookies", lambda profile=None: {"substack.sid": "x"})
    monkeypatch.setattr(sub, "own_user_id", lambda cookies: 7)
    monkeypatch.setattr(sub, "fetch_follows", lambda cookies, uid=None: [
        {"name": "Acme", "url": site}])
    monkeypatch.setattr(sub, "fetch_subscription_list", lambda source, **kw: [
        {"id": 1, "name": "Acme", "url": site, "membership_state": "free_signup",
         "is_favorite": False}])
    # This person saved nothing — but the walk must still be stubbed, or the saved-signal
    # collector goes to the live reader endpoint and spends its 20s of retry backoff.
    monkeypatch.setattr(sub, "saved_source", lambda profile=None: object())
    monkeypatch.setattr(sub, "fetch_saved_posts", lambda src: sub.SavedPosts([], True))
    return site


def test_the_rail_resolves_the_people_it_just_minted(rail_home, one_person_two_platforms):
    from pipeline.kb import schema

    out = cc.run_curation_catchup()

    assert out["resolve"]["cross_platform"] == 1
    conn = schema.connect()
    try:
        canon = [conn.execute("SELECT canonical_id FROM entities WHERE entity_id=?",
                              (eid,)).fetchone()["canonical_id"]
                 for eid in ("x:user:2", "substack:acme")]
    finally:
        conn.close()
    assert canon[0] and canon[0] == canon[1], "the rail left one human as two candidates"


def test_a_pass_that_ran_nothing_does_not_resolve(rail_home, collectors):
    """A pass that took the lease, found every collector inside its floor and made no network call has
    minted nothing, so there is nothing new to merge. Cheap, but this is the pass that happens on
    almost every session open — the floor is 6h and the spawn coalesces hourly."""
    cc.run_curation_catchup()                       # first pass runs everything
    out = cc.run_curation_catchup()                 # second is entirely inside the floor

    assert out["ran"] == {}
    assert out["resolve"] is None


def test_a_resolve_failure_never_sinks_the_rail(rail_home, collectors, monkeypatch):
    """Fail-safe: every signal is committed before resolution runs, so a resolve blowing up must
    degrade to an unmerged store rather than lose the whole pass's report."""
    from pipeline.kb import resolve

    monkeypatch.setattr(resolve, "resolve_entities",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    out = cc.run_curation_catchup()

    assert out["status"] == "ok"
    assert "db locked" in out["resolve"]["error"]
    assert set(collectors) == set(ic.COLLECTORS)     # every collector still ran


# ── the paper derivation ────────────────────────────────────────────────────────
#
# It is not a collector, so none of the collector machinery above applies to it: no floor, no
# `collector_runs` row, no platform reachability, and no gate on whether anything else ran.


def _saved_paper(conn, atom_id, authors):
    """One `user-saved` paper atom, the shape `paper_authors._authors_by_entity` reads."""
    schema.upsert_atom(conn, {
        "atom_id": atom_id, "source_type": "paper", "what_kind": "artifact",
        "who_id": "test:" + atom_id, "when_ts": "2026-01-01", "when_precision": "day",
        "about_entities": [], "source_url": "", "raw_ref": "", "raw_hash": atom_id,
        "description": "", "entry_mode": "user-saved", "payload": {"authors": authors}})
    conn.commit()


def test_a_saved_papers_authors_become_candidates_with_no_collector_running(
        rail_home, collectors):
    """⚠️ THE DEFECT. `derive_paper_signals` lived only in `curation_pull`, which is hand-run, so
    a paper deposited through `hopper` reached the screen only if somebody remembered that CLI.
    Measured on the live store 2026-09-08: zero occurrences of "paper" across the whole
    `curation_catchup.log`, against 11 `user-saved` paper atoms.

    The pass here is the one that happens on almost every session open — every collector inside
    its 6h floor, no network call — and the author still has to land."""
    cc.run_curation_catchup()                       # first pass burns the floor
    conn = schema.connect()
    try:
        _saved_paper(conn, "paper:arXiv:1", [{"name": "A Researcher", "scholar_id": "111"}])
    finally:
        conn.close()

    out = cc.run_curation_catchup()

    assert out["ran"] == {}, "no collector should have run — this is the floor case"
    assert out["paper_authors"]["authors"] == 1
    conn = schema.connect()
    try:
        assert conn.execute("SELECT 1 FROM curation_signals WHERE entity_id='scholar:111'"
                            ).fetchone() is not None
    finally:
        conn.close()


def test_the_derivation_resolves_the_author_it_just_minted(rail_home, collectors):
    """An unresolved person is TWO candidates carrying one signal each, which is below the
    pre-tick bar — they are filtered out before a human sees them. So a pass whose only producer
    was the derivation still has to resolve; gating resolve on `ran` alone left exactly that."""
    conn = schema.connect()
    try:
        _saved_paper(conn, "paper:arXiv:1", [{"name": "A Researcher", "scholar_id": "111"}])
    finally:
        conn.close()
    cc.run_curation_catchup()                       # burn the floor so `ran` is empty next pass

    out = cc.run_curation_catchup()

    assert out["ran"] == {}
    assert out["resolve"] is not None, "the derivation minted an entity and nothing merged it"


def test_a_pass_with_no_papers_and_no_collector_still_does_not_resolve(rail_home, collectors):
    """The derivation runs unconditionally, but finding nothing must not turn every idle pass
    into a resolve — that is the guard `test_a_pass_that_ran_nothing_does_not_resolve` protects,
    and adding a second producer must not quietly remove it."""
    cc.run_curation_catchup()

    out = cc.run_curation_catchup()

    assert out["paper_authors"]["authors"] == 0
    assert out["resolve"] is None


# ── no consent gate, no spend ───────────────────────────────────────────────────
def test_the_rail_spends_no_money(rail_home, collectors, monkeypatch):
    """The structural tell that this rail costs nothing: the two content arms take an `embedder`,
    these four take only `conn` and `profile`. No embedder means no embed, no VLM read, no
    twitterapi call. That is still true, and it is why the rail carries no daily ceiling."""
    made: list = []
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **k: made.append(1))

    out = cc.run_curation_catchup()

    assert out["status"] == "ok"
    assert made == []


# ── the consent gate ────────────────────────────────────────────────────────────
# This file used to assert the opposite — that the rail "asks for nothing" — on the reasoning that
# a consent gate exists to stop money being spent, and there is no money here. A cold-start test on
# 2026-08-20 falsified the premise, not the arithmetic: on a fresh install the rail read the user's
# Chrome cookie jar and hit X and Substack before `onboard` had run. It spends no money and still
# has a cost.
def test_an_unconsented_rail_touches_no_collector(kb_home, collectors):
    """A cold install must refuse BEFORE the cookie jar, not after. Asserting on `collectors`
    rather than just the status is the point: a gate that returns the right word after already
    reading the browser session would pass a status check and still be the bug."""
    out = cc.run_curation_catchup()

    assert out["status"] == "needs_consent"
    assert collectors == []


def test_force_grants_consent_rather_than_bypassing_it(kb_home, collectors):
    """`force=True` must leave the user consented, not sneak past the gate once — a user who
    explicitly asks for this pass has, by asking, opted in. Matches `run_bookmark_catchup`."""
    assert cc.consented() is False

    out = cc.run_curation_catchup(force=True)

    assert out["status"] == "ok"
    assert cc.consented() is True


def test_an_established_store_is_never_re_prompted(kb_home, collectors, monkeypatch):
    """An existing user has had this running for months; introducing a gate must not stop it.
    Consent is implied by content, exactly as in `bookmark_catchup`."""
    monkeypatch.setattr("pipeline.kb.bookmark_catchup._established_store", lambda: True)

    assert cc.consented() is True
    assert cc.run_curation_catchup()["status"] == "ok"


# ── the CLI ─────────────────────────────────────────────────────────────────────
def test_a_successful_run_exits_zero(rail_home, collectors):
    assert cc.main(["--once"]) == 0


def test_the_floor_flag_reaches_the_run(rail_home, collectors):
    cc.run_curation_catchup()
    collectors.clear()
    assert cc.main(["--once", "--floor-hours", "0"]) == 0
    assert set(collectors) == set(ic.COLLECTORS)     # a zero floor makes everything due again


def test_bare_invocation_prints_help_and_exits_two(rail_home, capsys):
    assert cc.main([]) == 2


# ── the platform gate ───────────────────────────────────────────────────────────
#
# X reads ONLY the OPYT-managed profile, so with no session every X collector fails identically
# with `no_viewer_id` — three recorded failures per pass, forever, on a home whose user reads
# Substack. Substack is deliberately NOT gated the same way; see `_platform_reachable`.

def _no_x_session(monkeypatch):
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)


def test_no_x_session_skips_the_x_collectors_and_says_so(rail_home, collectors, monkeypatch):
    _no_x_session(monkeypatch)

    out = cc.run_curation_catchup()

    assert collectors == ["substack_follows", "substack_subscriptions",
                          "substack_saved_signals"]
    assert set(out["skipped_no_session"]) == {"x_lists", "x_following", "x_likes",
                                             "x_bookmark_signals"}


def test_a_skip_for_no_session_is_reported_apart_from_a_skip_for_the_floor(
        rail_home, collectors, monkeypatch):
    """They need opposite actions — connect a session vs wait — so one list cannot carry both."""
    _no_x_session(monkeypatch)
    cc.run_curation_catchup()
    collectors.clear()

    out = cc.run_curation_catchup()

    assert out["skipped_within_floor"] == ["substack_follows", "substack_subscriptions",
                                           "substack_saved_signals"]
    assert set(out["skipped_no_session"]) == {"x_lists", "x_following", "x_likes",
                                             "x_bookmark_signals"}


def test_a_local_home_skips_substack_when_no_managed_session_exists(
        rail_home, collectors, monkeypatch):
    """The asymmetry is GONE as of 2026-09-13. `read_substack_cookies` now reads only the
    OPYT-managed profile — never the user's own browser — so a False `has_managed_substack_session`
    means the collector cannot run, on a local home exactly as on a hosted one. Attempting anyway
    would 401 and record "you follow nobody" where the truth is "Substack is not connected"."""
    _no_x_session(monkeypatch)
    from pipeline.ingestion.sources import substack as sub
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: False)

    out = cc.run_curation_catchup()

    assert collectors == []
    assert set(out["skipped_no_session"]) >= {"substack_follows", "substack_subscriptions"}


def test_a_hosted_home_skips_substack_when_its_profile_holds_no_session(
        rail_home, collectors, monkeypatch):
    """The asymmetry above has no reason on a hosted home: there is no user browser, so the
    managed profile IS the only session. Ungated the collector launches Chrome every pass and
    records `found=0` — "you follow nobody" where the truth is "Substack is not connected"."""
    from pipeline.ingestion import hosted_browser
    from pipeline.ingestion.sources import substack as sub

    _no_x_session(monkeypatch)
    monkeypatch.setattr(hosted_browser, "enabled", lambda: True)
    monkeypatch.setattr(sub, "has_managed_substack_session", lambda: False)

    out = cc.run_curation_catchup()

    assert collectors == []
    assert set(out["skipped_no_session"]) >= {"substack_follows", "substack_subscriptions"}


def test_an_unreadable_session_probe_lets_the_collector_attempt(rail_home, collectors,
                                                                monkeypatch):
    """Fail-safe direction: a broken probe must not silently skip real work. The collector runs
    and records whatever actually happens."""
    from pipeline.ingestion import x_graphql
    monkeypatch.setattr(x_graphql, "has_managed_x_session", _boom)

    out = cc.run_curation_catchup()

    assert set(collectors) == set(ic.COLLECTORS)
    assert out["skipped_no_session"] == []


def test_the_platform_probe_runs_once_per_platform_not_once_per_collector(
        rail_home, collectors, monkeypatch):
    """Three Substack collectors, ONE session probe.

    On a hosted home `_platform_reachable("substack")` is a live `/inbox` fetch, so probing per
    collector spends a duplicate request against the host whose rate limit is the entire
    constraint on this rail. The answer is about a platform's session and cannot change mid-pass.
    """
    probes: list[str] = []
    real = cc._platform_reachable
    monkeypatch.setattr(cc, "_platform_reachable",
                        lambda platform: probes.append(platform) or real(platform))

    cc.run_curation_catchup()

    assert sorted(probes) == ["substack", "x"]
    assert len([s for s in ic.COLLECTOR_SPECS if s.platform == "substack"]) > 1
