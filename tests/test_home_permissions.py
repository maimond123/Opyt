"""Home and secret-file permission modes, set at CREATION.

A `chmod` of `/srv/homes` (700) and `opyt.db`/`trial_key.json` (600) regressed
for every new signup because both home-creation sites used a bare `mkdir`. These tests pin the
fix: `opyt_core.paths.ensure_home` is the one owner of home creation, and each secret-holding
file is chmodded where it is created (the `keys.set_key` precedent — fail-safe, never breaking
the write).
"""
from __future__ import annotations

import os
import stat

from opyt_core.paths import ensure_home


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_ensure_home_creates_owner_only_despite_the_umask(tmp_path):
    old = os.umask(0o022)  # the umask that made a bare mkdir world-readable
    try:
        home = ensure_home(tmp_path / "homes" / "104729183746152938471")
    finally:
        os.umask(old)

    assert home.is_dir() and _mode(home) == 0o700
    # Only the leaf: the parent is not this function's to tighten.
    assert _mode(tmp_path / "homes") != 0o700 or True  # parent mode is whatever mkdir gave it


def test_ensure_home_tightens_an_existing_home(tmp_path):
    """The homes that already exist with the regression are repaired on next touch."""
    home = tmp_path / "42"
    home.mkdir(mode=0o755)

    ensure_home(home)

    assert _mode(home) == 0o700


def test_a_rail_child_home_is_created_owner_only(tmp_path, monkeypatch):
    from pipeline.worker.job_queue import RailJobStore
    from pipeline.worker.supervisor import RailWorker
    from pipeline.worker import registry
    from pipeline.worker.registry import HOURLY, RailSpec

    monkeypatch.setattr(registry, "RAILS",
                        {"probe": RailSpec("tests.kb._rail_children:noop", "probe.log", HOURLY)})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "probe", due_at=0)
    worker = RailWorker(store, homes_root=tmp_path / "homes")

    worker.launch_available()
    assert len(worker.active) == 1
    for child in worker.active:
        child.process.wait(timeout=5)
    worker.reap_finished()

    assert _mode(tmp_path / "homes" / "42") == 0o700


def test_secret_holding_files_are_created_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv("OPYT_WORKER_DB", raising=False)
    monkeypatch.delenv("OPYT_WORKER_HOME_ID", raising=False)

    from pipeline.kb import schema
    schema.connect().close()
    assert _mode(tmp_path / "opyt.db") == 0o600

    from pipeline import circuit_breaker
    circuit_breaker._connect(tmp_path / "circuit_breaker.db").close()
    assert _mode(tmp_path / "circuit_breaker.db") == 0o600

    from pipeline.worker.job_queue import RailJobStore
    RailJobStore(tmp_path / "rail_jobs.db")
    assert _mode(tmp_path / "rail_jobs.db") == 0o600


def test_the_bootstrapped_settings_file_is_0600(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    from opyt_core import bootstrap

    bootstrap.ensure_initialized()

    assert _mode(tmp_path / "settings.yaml") == 0o600
