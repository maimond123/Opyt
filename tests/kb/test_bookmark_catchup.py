"""The bookmark catch-up rail — the spawner, the consent gate, the seatbelt, single-flight.

Every test here covers something that fails SILENTLY in production, which is the whole reason this
rail exists: its predecessor ran correctly for months and landed its output where nothing read it.
Nothing in a status line catches that, so the wiring gets asserted
instead — a double-spawn double-spends, a leaked fd accumulates one per session, a burnt coalesce
stamp suppresses the next hour of real spawns, a shared consent marker opts you into a loop you
never chose, and a kill switch that does not kill is a dev machine grinding away.
"""
from __future__ import annotations

import subprocess

import pytest

from pipeline.kb import bookmark_catchup as bc


@pytest.fixture()
def rail_home(kb_home, monkeypatch):
    monkeypatch.delenv("OPYT_BOOKMARK_CATCHUP_CONSENT", raising=False)
    return kb_home


@pytest.fixture()
def no_spend(monkeypatch):
    """Stub the two things `run_bookmark_catchup` would otherwise pay for, and hand back the
    call log so a test can assert that NOTHING paid ran."""
    calls = []
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_x.sync_bookmarks",
                        lambda *a, **kw: calls.append(kw) or {"source": "x", "added": 0})
    return calls


# ── consent ─────────────────────────────────────────────────────────────────────
def test_unconsented_returns_needs_consent_and_spends_nothing(rail_home, no_spend):
    """The distributable case: a brand-new user must never have a paid backlog import fire on
    first launch. That is the money-absent + runaway case, which is the only case a consent gate
    is for."""
    out = bc.run_bookmark_catchup()
    assert out["status"] == "needs_consent"
    assert no_spend == []


def test_a_caller_that_granted_consent_then_runs_proceeds(rail_home, no_spend):
    """The path `onboard._apply_consent` actually takes: ask, grant, THEN run.

    Replaces a test of `run_bookmark_catchup(force=True)`, which granted consent on the caller's
    behalf. That made one word mean both "run now" and "the user agreed", and it was redundant —
    the only production caller already calls `grant_consent()` itself before spawning."""
    assert bc.run_bookmark_catchup()["status"] == "needs_consent"
    bc.grant_consent()
    assert bc.consented() is True
    assert bc.run_bookmark_catchup()["status"] == "ok"


def test_an_established_store_is_auto_consented(rail_home, no_spend):
    """Stored atoms imply consent; a retired notes-only store does not."""
    import sqlite3

    from opyt_core.paths import opyt_db
    assert bc.consented() is False                       # no DB at all → brand new
    conn = sqlite3.connect(opyt_db())
    try:
        conn.execute("CREATE TABLE notes (id TEXT)")
        conn.execute("INSERT INTO notes VALUES ('n1')")
        conn.commit()
        assert bc.consented() is False
        conn.execute("CREATE TABLE atoms (atom_id TEXT)")
        assert bc.consented() is False
        conn.execute("INSERT INTO atoms VALUES ('x:1')")
        conn.commit()
    finally:
        conn.close()
    assert bc.consented() is True                        # stored atom → implied
    assert not bc._consent_marker().exists()             # ...without ever writing a marker


def test_granting_consent_here_opts_into_nothing_else(rail_home, monkeypatch):
    """Each paid rail owns its own marker. Opting into the bookmark backlog — a ONE-TIME import —
    must never silently opt you into the Oracle or people refresh, which are RECURRING costs with
    a different shape entirely.

    Asserted over the WHOLE marker namespace rather than against a named list of sibling rails,
    for two reasons. It catches a rail that does not exist yet, which a hardcoded pair cannot. And
    `pipeline.radar` is unimportable from `tests/kb/` by design (`atom-rail-not-welded-to-radar`),
    so naming that rail directly would mean widening a guard allowlist to buy a weaker check."""
    monkeypatch.delenv("OPYT_ORACLE_REFRESH_CONSENT", raising=False)
    from pipeline.kb import oracle_refresh

    bc.grant_consent()

    assert bc._consent_marker() != oracle_refresh._consent_marker()
    assert oracle_refresh.consented() is False
    assert [p.name for p in sorted(rail_home.glob("*consent*"))] == ["bookmark_catchup_consent"]


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_catchup_skips_while_one_holds_the_lease(rail_home, no_spend, monkeypatch):
    """WITHOUT this, the background child and a user's manual `sync` can walk the same bookmarks
    at the same moment. The corpus survives (atoms are idempotent) but the bill does not."""
    from pipeline.sync_lock import CatchupLock

    bc.grant_consent()
    with CatchupLock("bookmark-catchup") as held:
        assert held.acquired
        out = bc.run_bookmark_catchup()

    assert out["status"] == "already_running"
    assert no_spend == []


# ── never raises ────────────────────────────────────────────────────────────────
def test_a_throwing_ingest_is_reported_not_propagated(rail_home, monkeypatch):
    """Fail-safe invariant. This runs in a detached child whose only caller is a `-m` entrypoint,
    so a propagated exception is a traceback in a log file nobody opens — and, worse, it escapes
    before the `finally` that closes the connection."""
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())

    def boom(*a, **kw):
        raise RuntimeError("twitterapi 502")

    monkeypatch.setattr("pipeline.kb.ingest_x.sync_bookmarks", boom)
    bc.grant_consent()
    out = bc.run_bookmark_catchup()
    assert out["status"] == "error"
    assert "twitterapi 502" in out["error"]


# ── a RETURNED failure, which is the shape the adapter actually uses ────────────
# `d7dbcfcf`'s contract: an adapter signals a hard stop by RETURNING a summary carrying `error`,
# never by raising, because a raise would sink the caller's other sources. So the test above — a
# raise — was the only failure path this rail had covered, and the path the adapter really takes
# arrived here as `status: ok`. `run_concurrent` drained the dead walk, `sync_bookmarks` reported
# the counters it honestly reached, and `added: 0, total: 0` is byte-identical to a quiet week.
# A rail that believes it succeeded does not retry, so an expired cookie stopped bookmark imports
# and nothing anywhere said so.

def test_a_returned_error_is_not_reported_as_a_successful_run(rail_home, monkeypatch):
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_x.sync_bookmarks",
                        lambda *a, **kw: {"source": "x", "added": 0, "skipped": 0, "total": 0,
                                          "error": "SyncAuthError: cookie expired"})
    bc.grant_consent()
    out = bc.run_bookmark_catchup()
    assert out["status"] == "error"          # a person must re-authenticate
    assert out["added"] == 0                 # and the counters still ride along


def test_a_returned_rate_limit_is_blocked_not_error(rail_home, monkeypatch):
    """The distinction `d7dbcfcf` refused to collapse: "collapsing them trains the reader to
    ignore errors". x.com's meter resets on its own, so this is `blocked` and retryable; an
    expired cookie needs a person, so it is `error`. Same rule D1 followed for GitHub."""
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_x.sync_bookmarks",
                        lambda *a, **kw: {"source": "x", "added": 3, "total": 3,
                                          "error": "XRateLimited: window spent",
                                          "undetermined": 1})
    bc.grant_consent()
    out = bc.run_bookmark_catchup()
    assert out["status"] == "blocked"
    assert out["added"] == 3                 # the drain: what was fetched still landed


# ── the CLI ─────────────────────────────────────────────────────────────────────
def test_a_successful_run_exits_zero(rail_home, no_spend):
    """`sync_bookmarks` returns a run SUMMARY with no `status` key, so the wrapper has to stamp
    one on. Without it every wholly successful catch-up would exit 1, and the detached child would
    look permanently broken to anything reading exit codes."""
    bc.grant_consent()
    assert bc.main(["--once"]) == 0


def test_the_limit_flag_reaches_sync_bookmarks(rail_home, no_spend):
    bc.grant_consent()
    assert bc.main(["--once", "--limit", "25"]) == 0
    assert no_spend[0]["limit"] == 25
