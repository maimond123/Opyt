"""`oracle_refresh`'s single-flight lease — the part a live run exercises but never ASSERTS.

Both failures here are silent in production: two overlapping passes double-spend on the same
timelines, and a lease that is never released is a loop that stops forever while every surface
still says it is on.

The spawner tests that used to live beside these are gone with the spawner. What they proved —
detachment, a real log file, the true exit code, one child per home — is now the worker's, and is
proved once in `tests/kb/test_rail_worker.py` instead of eight times here.
"""
from __future__ import annotations

import json

import pytest

from pipeline.kb import oracle_refresh as orf


def test_a_second_refresh_skips_while_one_holds_the_lease(kb_home, monkeypatch):
    """The worker never runs two rails for one home, so the live producer of an overlap is a
    hand-run `--once` beside the worker's own child. Only one may pull a timeline."""
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


def test_the_registry_command_checks_consent_before_running(kb_home, monkeypatch, capsys):
    """The worker launches this rail through a `-c` string, not a module. That string is the
    real entry point, so consent must hold when it is EXECUTED — not merely when `_run` is
    imported by a test that hand-picks the function."""
    from pipeline.kb.rail_worker import RAILS

    monkeypatch.setattr(orf, "load_rail_env", lambda: None)
    monkeypatch.setattr(orf, "refresh_all", lambda **kw: pytest.fail("unconsented refresh ran"))
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "consent"))

    command = RAILS["oracle_refresh"].command
    assert command[0] == "-c"
    with pytest.raises(SystemExit) as exit:
        exec(command[1], {})
    assert exit.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "needs_consent"
    assert not orf.consented()


# ── what counts as a failed pass ────────────────────────────────────────────────
@pytest.mark.parametrize("status, code", [
    ("ok", 0), ("already_running", 0), ("needs_consent", 0),
    # ⚠️ ADDED 2026-09-14, because it is a HEALTHY pass that was recording as a failure. x.com
    # meters timeline reads in 15-minute windows, so the pass right after a first ingest spends
    # the budget and defers the rest: nothing fetched, nothing written, nothing marked pulled,
    # `errors: 0`, retried next pass. Measured on a fresh store — `rate_paused`, `deferred: 3`,
    # `errors: 0` — and it exited 1. Scheduling was unharmed (`finish` applies the cadence
    # whatever the code), so the whole cost was that `exit_code` lied to anyone reading it to ask
    # whether refresh works, which is the one question this rail's repair was about.
    ("rate_paused", 0),
    ("error", 1),
])
def test_only_a_pass_that_actually_failed_exits_nonzero(monkeypatch, capsys, status, code):
    monkeypatch.setattr(orf, "run_oracle_refresh", lambda: {"status": status})
    with pytest.raises(SystemExit) as exit:
        orf._run()
    assert exit.value.code == code
    assert json.loads(capsys.readouterr().out)["status"] == status
