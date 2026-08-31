"""The bookmark catch-up rail — the spawner, the consent gate, the seatbelt, single-flight.

Every test here covers something that fails SILENTLY in production, which is the whole reason this
rail exists: its predecessor (`mcp_server/hot_feed.py`) ran correctly for months and landed its
output where nothing read it. Nothing in a status line catches that, so the wiring gets asserted
instead — a double-spawn double-spends, a leaked fd accumulates one per session, a burnt coalesce
stamp suppresses the next hour of real spawns, a shared consent marker opts you into a loop you
never chose, and a kill switch that does not kill is a dev machine grinding away.
"""
from __future__ import annotations

import subprocess

import pytest

from pipeline.kb import bookmark_catchup as bc


@pytest.fixture()
def spawn_env(kb_home, monkeypatch):
    for var in ("OPYT_NO_BOOKMARK_CATCHUP", "OPYT_BOOKMARK_CATCHUP_STAMP",
                "OPYT_BOOKMARK_CATCHUP_LOG", "OPYT_BOOKMARK_CATCHUP_COALESCE",
                "OPYT_BOOKMARK_CATCHUP_CONSENT"):
        monkeypatch.delenv(var, raising=False)
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
def no_spend(monkeypatch):
    """Stub the two things `run_bookmark_catchup` would otherwise pay for, and hand back the
    call log so a test can assert that NOTHING paid ran."""
    calls = []
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: object())
    monkeypatch.setattr("pipeline.kb.ingest_x.sync_bookmarks",
                        lambda *a, **kw: calls.append(kw) or {"source": "x", "added": 0})
    return calls


# ── kill switch ─────────────────────────────────────────────────────────────────
def test_kill_switch_forks_nothing(spawn_env, popen, monkeypatch):
    monkeypatch.setenv("OPYT_NO_BOOKMARK_CATCHUP", "1")
    assert bc.spawn_bookmark_catchup() is False
    assert popen.instances == []
    assert not (spawn_env / "bookmark_catchup_last_spawn").exists()


# ── the spawn shape ─────────────────────────────────────────────────────────────
def test_spawn_detaches_and_never_inherits_stdout(spawn_env, popen):
    """The MCP server speaks JSON-RPC over stdout. An inherited stdout corrupts the protocol
    stream — this is the one mistake in this file that breaks the whole server, not just the rail."""
    assert bc.spawn_bookmark_catchup() is True
    p = popen.instances[0]
    assert p.argv[1:] == ["-m", "pipeline.kb.bookmark_catchup", "--once"]
    assert p.kw["stdin"] is subprocess.DEVNULL
    assert p.kw["start_new_session"] is True
    assert p.kw["stdout"] is p.kw["stderr"]
    assert p.kw["stdout"] not in (None, subprocess.DEVNULL)     # a real log file, not inherited
    assert (spawn_env / "bookmark_catchup.log").exists()


def test_the_log_fd_is_closed_in_the_parent(spawn_env, popen):
    """`catchup.spawn_detached` leaks one fd per spawn. The child has already dup'd it by the time
    Popen returns, so the parent's copy is pure leak — and the MCP server is long-lived."""
    bc.spawn_bookmark_catchup()
    assert popen.instances[0].kw["stdout"].closed is True


def test_the_stamp_is_touched_only_after_a_successful_fork(spawn_env, monkeypatch):
    """PINNED because it is the exact defect carried by the earlier spawner this one was
    deliberately NOT copied from (deleted 2026-08-12). That one stamped BEFORE forking, so a
    failed fork still burned
    the whole coalesce window and every session inside it declines to retry — a silent hour of no
    catch-up caused by the one attempt that failed. `CatchupLock` makes a redundant spawn harmless,
    so the double-spawn race this trades for is the cheaper side."""
    def boom(*a, **kw):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", boom)
    assert bc.spawn_bookmark_catchup() is False
    assert not (spawn_env / "bookmark_catchup_last_spawn").exists()


# ── coalescing ──────────────────────────────────────────────────────────────────
def test_second_spawn_inside_the_window_is_suppressed(spawn_env, popen):
    assert bc.spawn_bookmark_catchup() is True
    assert bc.spawn_bookmark_catchup() is False
    assert len(popen.instances) == 1


def test_force_ignores_the_coalesce_window(spawn_env, popen):
    bc.spawn_bookmark_catchup()
    assert bc.spawn_bookmark_catchup(force=True) is True
    assert len(popen.instances) == 2
    assert popen.instances[1].argv[-1] == "--force"


def test_coalesce_window_is_configurable(spawn_env, popen, monkeypatch):
    monkeypatch.setenv("OPYT_BOOKMARK_CATCHUP_COALESCE", "0")
    bc.spawn_bookmark_catchup()
    assert bc.spawn_bookmark_catchup() is True     # window 0 → never coalesces
    assert len(popen.instances) == 2


# ── consent ─────────────────────────────────────────────────────────────────────
def test_unconsented_returns_needs_consent_and_spends_nothing(spawn_env, no_spend):
    """The distributable case: a brand-new user must never have a paid backlog import fire on
    first launch. That is the money-absent + runaway case, which is the only case a consent gate
    is for."""
    out = bc.run_bookmark_catchup()
    assert out["status"] == "needs_consent"
    assert no_spend == []


def test_force_grants_consent(spawn_env, no_spend):
    out = bc.run_bookmark_catchup(force=True)
    assert out["status"] == "ok"
    assert bc._consent_marker().exists()
    assert bc.consented() is True


def test_an_established_store_is_auto_consented(spawn_env, no_spend):
    """A store that already holds content implies consent, and the fallback to `notes` is
    load-bearing rather than legacy tidiness: a pre-migration store has notes and no atoms, so an
    atoms-only check would re-prompt every such user for consent they already granted."""
    import sqlite3

    from opyt_core.paths import opyt_db
    assert bc.consented() is False                       # no DB at all → brand new
    conn = sqlite3.connect(opyt_db())
    try:
        conn.execute("CREATE TABLE notes (id TEXT)")
        conn.execute("INSERT INTO notes VALUES ('n1')")
        conn.commit()
    finally:
        conn.close()
    assert bc.consented() is True                        # notes-only store → implied
    assert not bc._consent_marker().exists()             # ...without ever writing a marker


def test_granting_consent_here_opts_into_nothing_else(spawn_env, monkeypatch):
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
    assert [p.name for p in sorted(spawn_env.glob("*consent*"))] == ["bookmark_catchup_consent"]


# ── the resetting daily seatbelt ────────────────────────────────────────────────
def test_over_budget_pauses_before_taking_the_lease(spawn_env, no_spend, monkeypatch):
    """The budget check must sit OUTSIDE the lock. Held here by another holder: if the ordering
    were reversed this would report `already_running` — a free refusal disguised as contention —
    and a paused run would take a lease a live one could be using."""
    from pipeline.sync_lock import CatchupLock

    bc.grant_consent()
    monkeypatch.setattr("pipeline.llm_client.spend_today_for_rail",
                        lambda *a, **kw: bc.BOOKMARK_CATCHUP_DAILY_USD + 0.01)

    with CatchupLock("bookmark-catchup") as held:
        assert held.acquired
        out = bc.run_bookmark_catchup()

    assert out["status"] == "budget_paused"
    assert no_spend == []


def test_the_ceiling_gates_the_START_and_does_not_cap_the_run(spawn_env, no_spend, monkeypatch):
    """PINNED because the opposite is the natural assumption, and acting on it picks the wrong
    ceiling. This is a START GATE: read once, before the run. `sync_bookmarks` carries no budget
    check, so a run that clears the gate at $0.99 goes on to spend whatever the whole backlog
    costs. `oracle_refresh` DOES re-check mid-loop — copying its number without copying its
    mechanism is exactly the mistake this test exists to make visible."""
    bc.grant_consent()
    monkeypatch.setattr("pipeline.llm_client.spend_today_for_rail",
                        lambda *a, **kw: bc.BOOKMARK_CATCHUP_DAILY_USD - 0.01)

    out = bc.run_bookmark_catchup()

    assert out["status"] == "ok"        # under the ceiling at start → the FULL walk is authorized
    assert len(no_spend) == 1
    assert no_spend[0]["limit"] == 0    # ...unbounded, with nothing to stop it partway


def test_an_unreadable_meter_does_not_block_a_legitimate_run(spawn_env, monkeypatch):
    """Fail-safe: the seatbelt exists to stop a runaway, not to become one more thing that can
    take the rail down."""
    def boom(*a, **kw):
        raise RuntimeError("stats file is a directory")

    monkeypatch.setattr("pipeline.llm_client.spend_today_for_rail", boom)
    assert bc._daily_budget_exhausted() is False


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_catchup_skips_while_one_holds_the_lease(spawn_env, no_spend, monkeypatch):
    """WITHOUT this, the background child and a user's manual `sync` can walk the same bookmarks
    at the same moment. The corpus survives (atoms are idempotent) but the bill does not."""
    from pipeline.sync_lock import CatchupLock

    bc.grant_consent()
    monkeypatch.setattr("pipeline.llm_client.spend_today_for_rail", lambda *a, **kw: 0.0)

    with CatchupLock("bookmark-catchup") as held:
        assert held.acquired
        out = bc.run_bookmark_catchup()

    assert out["status"] == "already_running"
    assert no_spend == []


# ── never raises ────────────────────────────────────────────────────────────────
def test_a_throwing_ingest_is_reported_not_propagated(spawn_env, monkeypatch):
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


# ── the CLI ─────────────────────────────────────────────────────────────────────
def test_a_successful_run_exits_zero(spawn_env, no_spend):
    """`sync_bookmarks` returns a run SUMMARY with no `status` key, so the wrapper has to stamp
    one on. Without it every wholly successful catch-up would exit 1, and the detached child would
    look permanently broken to anything reading exit codes."""
    bc.grant_consent()
    assert bc.main(["--once"]) == 0


def test_the_limit_flag_reaches_sync_bookmarks(spawn_env, no_spend):
    bc.grant_consent()
    assert bc.main(["--once", "--limit", "25"]) == 0
    assert no_spend[0]["limit"] == 25
