"""The curation catch-up rail — the spawner, the floor, single-flight, failure isolation.

Same brief as `test_bookmark_catchup.py`: everything here fails SILENTLY in production. A rail whose
candidate list never refreshes is indistinguishable from one with nothing new to add, which is
exactly how the four collectors this drives went from 2026-07-21 to 2026-08-12 with no automatic
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


@pytest.fixture()
def spawn_env(kb_home, monkeypatch):
    for var in ("OPYT_NO_CURATION_CATCHUP", "OPYT_CURATION_CATCHUP_STAMP",
                "OPYT_CURATION_CATCHUP_LOG", "OPYT_CURATION_CATCHUP_COALESCE"):
        monkeypatch.delenv(var, raising=False)
    # Consent granted once here because every test using this fixture exercises the rail's
    # MECHANICS — floor, single-flight, collector dispatch — not its gate. A sandboxed home has no
    # atoms, so `_established_store()` is False and an ungranted rail would return `needs_consent`
    # before reaching anything these tests are about. The gate has its own tests below.
    cc.grant_consent()
    return kb_home


class _FakePopen:
    """Records the args a spawn was called with, and never actually forks."""

    instances: list = []

    def __init__(self, argv, **kw):
        self.argv, self.kw = argv, kw
        _FakePopen.instances.append(self)


@pytest.fixture()
def popen(monkeypatch):
    _FakePopen.instances = []
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    return _FakePopen


@pytest.fixture()
def collectors(monkeypatch):
    """Replace the four collectors with recording stubs and hand back the call log, so a test can
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


# ── kill switch ─────────────────────────────────────────────────────────────────
def test_kill_switch_forks_nothing(spawn_env, popen, monkeypatch):
    monkeypatch.setenv("OPYT_NO_CURATION_CATCHUP", "1")
    assert cc.spawn_curation_catchup() is False
    assert popen.instances == []
    assert not (spawn_env / "curation_catchup_last_spawn").exists()


# ── the spawn shape ─────────────────────────────────────────────────────────────
def test_spawn_detaches_and_never_inherits_stdout(spawn_env, popen):
    """The MCP server speaks JSON-RPC over stdout. An inherited stdout corrupts the protocol
    stream — the one mistake in this file that breaks the whole server, not just the rail."""
    assert cc.spawn_curation_catchup() is True
    p = popen.instances[0]
    assert p.argv[1:] == ["-m", "pipeline.kb.curation_catchup", "--once"]
    assert p.kw["stdin"] is subprocess.DEVNULL
    assert p.kw["start_new_session"] is True
    assert p.kw["stdout"] is p.kw["stderr"]
    assert p.kw["stdout"] not in (None, subprocess.DEVNULL)     # a real log file, not inherited
    assert (spawn_env / "curation_catchup.log").exists()


def test_the_log_fd_is_closed_in_the_parent(spawn_env, popen):
    """The child has already dup'd it by the time Popen returns, so the parent's copy is pure
    leak — and the MCP server is long-lived."""
    cc.spawn_curation_catchup()
    assert popen.instances[0].kw["stdout"].closed is True


def test_the_stamp_is_touched_only_after_a_successful_fork(spawn_env, monkeypatch):
    """Stamping BEFORE the fork means a failed fork burns the whole coalesce window, and every
    session inside it declines to retry — a silent hour of no catch-up caused by the one attempt
    that failed. `CatchupLock` makes a redundant spawn harmless, so the double-spawn race this
    trades for is the cheaper side."""
    def fork_failed(*a, **kw):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", fork_failed)
    assert cc.spawn_curation_catchup() is False
    assert not (spawn_env / "curation_catchup_last_spawn").exists()


# ── coalescing ──────────────────────────────────────────────────────────────────
def test_second_spawn_inside_the_window_is_suppressed(spawn_env, popen):
    assert cc.spawn_curation_catchup() is True
    assert cc.spawn_curation_catchup() is False
    assert len(popen.instances) == 1


def test_force_ignores_the_coalesce_window(spawn_env, popen):
    cc.spawn_curation_catchup()
    assert cc.spawn_curation_catchup(force=True) is True
    assert len(popen.instances) == 2
    assert popen.instances[1].argv[-1] == "--force"


def test_coalesce_window_is_configurable(spawn_env, popen, monkeypatch):
    monkeypatch.setenv("OPYT_CURATION_CATCHUP_COALESCE", "0")
    cc.spawn_curation_catchup()
    assert cc.spawn_curation_catchup() is True     # window 0 → never coalesces
    assert len(popen.instances) == 2


def test_this_rails_stamp_is_its_own(spawn_env, popen):
    """Each rail owns its spawner AND its stamp. Sharing one would make either rail's spawn
    suppress the other's for an hour."""
    from pipeline.kb import bookmark_catchup as bc

    cc.spawn_curation_catchup()
    assert (spawn_env / "curation_catchup_last_spawn").exists()
    assert bc.spawn_bookmark_catchup() is True     # not coalesced away by ours
    assert len(popen.instances) == 2


# ── the floor ───────────────────────────────────────────────────────────────────
def test_a_first_pass_runs_every_collector(spawn_env, collectors):
    out = cc.run_curation_catchup()
    assert out["status"] == "ok"
    assert set(collectors) == set(ic.COLLECTORS)
    assert out["skipped_within_floor"] == []
    assert out["errors"] == 0
    assert out["freshness"]["needs_attention"] is False


def test_a_collector_inside_its_floor_is_skipped_without_running(spawn_env, collectors):
    """The floor is what makes an hourly spawn cheap: the second pass is a lock acquire, four
    SELECTs and an exit, with no network call at all."""
    cc.run_curation_catchup()
    collectors.clear()

    second = cc.run_curation_catchup()

    assert collectors == []                                    # nothing hit the network
    assert second["ran"] == {}
    assert set(second["skipped_within_floor"]) == set(ic.COLLECTORS)


def test_force_ignores_the_floor(spawn_env, collectors):
    cc.run_curation_catchup()
    collectors.clear()
    out = cc.run_curation_catchup(force=True)
    assert set(collectors) == set(ic.COLLECTORS)
    assert out["skipped_within_floor"] == []


def test_the_floor_counts_attempts_not_successes(spawn_env, collectors, monkeypatch):
    """PINNED. Gating on `last_ok_at` would remove the floor from exactly the collector that most
    needs one: a dead X session would be retried on EVERY session open, forever."""
    monkeypatch.setattr(ic, "sync_following_signals", _boom)
    first = cc.run_curation_catchup()
    assert "error" in first["ran"]["x_following"]

    second = cc.run_curation_catchup()

    assert "x_following" in second["skipped_within_floor"]
    assert second["ran"] == {}


# ── failure isolation ───────────────────────────────────────────────────────────
def test_one_collector_raising_does_not_stop_the_other_three(spawn_env, collectors, monkeypatch):
    monkeypatch.setattr(ic, "sync_likes_signals", _boom)

    out = cc.run_curation_catchup()

    assert out["status"] == "ok" and out["errors"] == 1
    assert "error" in out["ran"]["x_likes"]
    assert set(collectors) == {"x_lists", "x_following", "substack_subs"}

    conn = cs.connect()
    try:
        assert cs.get_run(conn, "x_likes").last_status == "error"
        assert cs.get_run(conn, "x_likes").last_ok_at is None
        assert cs.get_run(conn, "x_following").last_status == "ok"
    finally:
        conn.close()


def test_it_never_raises(spawn_env, monkeypatch):
    """Fail-safe invariant. This runs in a detached child whose only caller is a `-m` entrypoint,
    so a propagated exception is a traceback in a log file nobody opens."""
    monkeypatch.setattr("pipeline.kb.schema.connect", _boom)
    out = cc.run_curation_catchup()
    assert out["status"] == "error" and "dead X session" in out["error"]


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_catchup_skips_while_one_holds_the_lease(spawn_env, collectors):
    """A free scrape run twice at once is still twice the requests against a cookie session, and
    that is what gets an account rate-limited."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("curation-catchup") as held:
        assert held.acquired
        out = cc.run_curation_catchup()

    assert out["status"] == "already_running"
    assert collectors == []


def test_force_does_not_bypass_single_flight(spawn_env, collectors):
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("curation-catchup") as held:
        assert held.acquired
        assert cc.run_curation_catchup(force=True)["status"] == "already_running"
    assert collectors == []


def test_the_lease_is_not_shared_with_the_bookmark_rail(spawn_env, collectors):
    """Different names, different leases. Sharing one would make a long bookmark backfill block
    the free list refresh for its whole duration."""
    from pipeline.sync_lock import CatchupLock

    with CatchupLock("bookmark-catchup") as held:
        assert held.acquired
        out = cc.run_curation_catchup()
    assert out["status"] == "ok"
    assert set(collectors) == set(ic.COLLECTORS)


# ── the trap: the tiered ladder must never be on this path ──────────────────────
def test_it_never_routes_through_the_tiered_ladder(spawn_env, collectors, monkeypatch):
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
    """Patch the FETCH layer, leaving the four collectors REAL, so the entities and identity_links
    a merge needs actually get written. One human: an X follow whose bio site is the home of a
    Substack the user subscribes to."""
    site = "https://acme.substack.com"
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_likes, x_lists
    from pipeline.ingestion.sources import substack as sub

    monkeypatch.setattr(core, "read_x_cookies", lambda profile=None: {"twid": "u=1"})
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
    monkeypatch.setattr(sub, "fetch_subscriptions", lambda cookies, uid: [
        {"name": "Acme", "url": site}])
    return site


def test_the_rail_resolves_the_people_it_just_minted(spawn_env, one_person_two_platforms):
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


def test_a_pass_that_ran_nothing_does_not_resolve(spawn_env, collectors):
    """A pass that took the lease, found all four inside their floor and made no network call has
    minted nothing, so there is nothing new to merge. Cheap, but this is the pass that happens on
    almost every session open — the floor is 6h and the spawn coalesces hourly."""
    cc.run_curation_catchup()                       # first pass runs everything
    out = cc.run_curation_catchup()                 # second is entirely inside the floor

    assert out["ran"] == {}
    assert out["resolve"] is None


def test_a_resolve_failure_never_sinks_the_rail(spawn_env, collectors, monkeypatch):
    """Fail-safe: every signal is committed before resolution runs, so a resolve blowing up must
    degrade to an unmerged store rather than lose the whole pass's report."""
    from pipeline.kb import resolve

    monkeypatch.setattr(resolve, "resolve_entities",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db locked")))

    out = cc.run_curation_catchup()

    assert out["status"] == "ok"
    assert "db locked" in out["resolve"]["error"]
    assert set(collectors) == set(ic.COLLECTORS)     # all four still ran


# ── no consent gate, no spend ───────────────────────────────────────────────────
def test_the_rail_spends_no_money(spawn_env, collectors, monkeypatch):
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
def test_a_successful_run_exits_zero(spawn_env, collectors):
    assert cc.main(["--once"]) == 0


def test_the_floor_flag_reaches_the_run(spawn_env, collectors):
    cc.run_curation_catchup()
    collectors.clear()
    assert cc.main(["--once", "--floor-hours", "0"]) == 0
    assert set(collectors) == set(ic.COLLECTORS)     # a zero floor makes everything due again


def test_bare_invocation_prints_help_and_exits_two(spawn_env, capsys):
    assert cc.main([]) == 2
