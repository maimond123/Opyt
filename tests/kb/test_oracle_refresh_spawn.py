"""The spawner + single-flight — the parts a live run exercises but never ASSERTS.

Every test here covers something that fails SILENTLY in production: a double-spawn double-spends,
a leaked fd accumulates one per session, a burnt coalesce stamp suppresses the next hour of real
spawns, and a kill switch that does not kill is a dev-machine grinding away.
"""
from __future__ import annotations

import subprocess

import pytest

from pipeline.kb import oracle_refresh as orf


@pytest.fixture()
def spawn_env(kb_home, monkeypatch):
    for var in ("OPYT_NO_ORACLE_REFRESH", "OPYT_ORACLE_REFRESH_STAMP",
                "OPYT_ORACLE_REFRESH_LOG", "OPYT_ORACLE_REFRESH_COALESCE"):
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


# ── kill switch ─────────────────────────────────────────────────────────────────
def test_kill_switch_forks_nothing(spawn_env, popen, monkeypatch):
    monkeypatch.setenv("OPYT_NO_ORACLE_REFRESH", "1")
    assert orf.spawn_oracle_refresh() is False
    assert popen.instances == []
    assert not (spawn_env / "oracle_refresh_last_spawn").exists()


# ── the spawn shape ─────────────────────────────────────────────────────────────
def test_spawn_detaches_and_never_inherits_stdout(spawn_env, popen):
    """The MCP server speaks JSON-RPC over stdout. An inherited stdout corrupts the protocol
    stream — this is the one mistake in this file that breaks the whole server, not just the loop."""
    assert orf.spawn_oracle_refresh() is True
    p = popen.instances[0]
    # `--once` is part of the contract, not noise: without it a bare `python -m
    # pipeline.kb.oracle_refresh` fires a PAID rail instead of printing help and exiting 2.
    assert p.argv[1:] == ["-m", "pipeline.kb.oracle_refresh", "--once"]
    assert p.kw["stdin"] is subprocess.DEVNULL
    assert p.kw["start_new_session"] is True
    assert p.kw["stdout"] is p.kw["stderr"]
    assert p.kw["stdout"] not in (None, subprocess.DEVNULL)     # a real log file, not inherited
    assert (spawn_env / "oracle_refresh.log").exists()


def test_the_log_fd_is_closed_in_the_parent(spawn_env, popen):
    """Both existing spawners leak one fd per spawn. The child has already dup'd it by the time
    Popen returns, so the parent's copy is pure leak — and the MCP server is long-lived."""
    orf.spawn_oracle_refresh()
    assert popen.instances[0].kw["stdout"].closed is True


def test_the_stamp_is_touched_only_after_a_successful_fork(spawn_env, monkeypatch):
    """Both existing spawners stamp BEFORE forking, so a failed fork still burns the 600s window
    and suppresses the next ten minutes of legitimate spawns. `CatchupLock` makes a redundant
    spawn harmless, so the double-spawn race this trades for is the cheaper side."""
    def boom(*a, **kw):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", boom)
    assert orf.spawn_oracle_refresh() is False
    assert not (spawn_env / "oracle_refresh_last_spawn").exists()


# ── coalescing ──────────────────────────────────────────────────────────────────
def test_second_spawn_inside_the_window_is_suppressed(spawn_env, popen):
    assert orf.spawn_oracle_refresh() is True
    assert orf.spawn_oracle_refresh() is False
    assert len(popen.instances) == 1


def test_force_ignores_the_coalesce_window(spawn_env, popen):
    orf.spawn_oracle_refresh()
    assert orf.spawn_oracle_refresh(force=True) is True
    assert len(popen.instances) == 2
    assert popen.instances[1].argv[-1] == "--force"


def test_coalesce_window_is_configurable(spawn_env, popen, monkeypatch):
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_COALESCE", "0")
    orf.spawn_oracle_refresh()
    assert orf.spawn_oracle_refresh() is True     # window 0 → never coalesces
    assert len(popen.instances) == 2


# ── single-flight ───────────────────────────────────────────────────────────────
def test_a_second_refresh_skips_while_one_holds_the_lease(kb_home, monkeypatch):
    """WITHOUT this, two triggers can pull the same paid timeline at the same moment — the corpus
    survives (atoms are idempotent) but the bill does not. The lock is the only thing standing
    between them.

    ⚠️ The second trigger CHANGED on 2026-08-15 and the lock did not stop mattering. It used to be
    a user's manual refresh action; that left the tool surface, so it is now the CLI escape hatch
    (`python -m pipeline.kb.oracle_refresh --once --force`) racing a session-open spawner. Fewer
    people will hit it, which makes the lock MORE load-bearing rather than less: a rare double-pull
    nobody is watching for is exactly the bill that arrives unexplained."""
    from pipeline.sync_lock import CatchupLock

    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "consent"))
    orf.grant_consent()
    ran = []
    monkeypatch.setattr(orf, "refresh_all", lambda *a, **kw: ran.append(1) or {"status": "ok"})
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: None)

    with CatchupLock("oracle-refresh") as held:
        assert held.acquired
        out = orf.run_oracle_refresh()

    assert out["status"] == "already_running"
    assert ran == []                                # nothing ran, nothing spent


def test_the_lease_is_released_so_the_next_run_proceeds(kb_home, monkeypatch):
    """A lock that is never released is a loop that never runs again — the failure mode is a
    silent freeze, which is exactly what this whole subsystem exists to end."""
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "consent"))
    orf.grant_consent()
    monkeypatch.setattr(orf, "refresh_all", lambda *a, **kw: {"status": "ok"})
    monkeypatch.setattr("pipeline.kb.embed.get_kb_embedder", lambda *a, **kw: None)

    assert orf.run_oracle_refresh()["status"] == "ok"
    assert orf.run_oracle_refresh()["status"] == "ok"      # not wedged on the previous lease
