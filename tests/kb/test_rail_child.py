"""`rail_child`: the one entry point every worker child goes through, and its one exit table.

A single entry point keeps exit codes consistent across jobs (an `already_running` pass is healthy
in every job) and ensures every job loads the user's keys."""
from __future__ import annotations

import json
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from pipeline.worker import child as rail_child
from pipeline.worker.supervisor import REPO_ROOT
from pipeline.worker import exit_codes
from pipeline.worker.exit_codes import EXIT_TERMINATED, exit_code_for
from tests.kb import _rail_children

# A stand-in job, patched per test: the child's rules are the same for any `module:function`.
JOB = "tests.kb._rail_children:noop"


@pytest.mark.parametrize("status", sorted(exit_codes.HEALTHY))
def test_every_healthy_status_exits_zero(status):
    assert exit_code_for({"status": status}) == 0


@pytest.mark.parametrize("status, code", [
    ("error", 1), ("failed", 1), ("models_unroutable", 1), ("lease_lost", 1),
    # a status nobody listed is a failure until someone lists it
    ("something_new", 1), (None, 1),
])
def test_everything_else_is_named_by_the_table(status, code):
    assert exit_code_for({"status": status}) == code


# ── in this process ─────────────────────────────────────────────────────────────
@pytest.fixture()
def child(monkeypatch):
    """`rail_child.main` without touching this process's own SIGTERM handler, recording the
    order in which keys were loaded and the job ran."""
    order: list[str] = []
    monkeypatch.setattr(rail_child, "install_stop_handler", lambda: order.append("handler"))
    monkeypatch.setattr("pipeline.worker.child.load_rail_env", lambda: order.append("env"))
    return order


def test_keys_are_loaded_before_the_job_runs(child, monkeypatch, capsys):
    """`frontier_admit` embedded and ran the model preflight, and never loaded the keys: a key
    rotated on disk never reached it. Every job's child loads them first."""
    monkeypatch.setattr(_rail_children, "noop",
                        lambda: child.append("run") or {"status": "ok", "considered": 0})

    assert rail_child.main([JOB]) == 0
    assert child == ["handler", "env", "run"]
    assert json.loads(capsys.readouterr().out) == {"status": "ok", "considered": 0}


def test_a_pass_that_found_another_in_flight_is_healthy(child, monkeypatch, capsys):
    """A skipped pass is not a failure."""
    monkeypatch.setattr(_rail_children, "noop", lambda: {"status": "already_running"})

    assert rail_child.main([JOB]) == 0


def _raises():
    raise RuntimeError("database is gone")


@pytest.mark.usefixtures("child")
def test_a_job_that_raises_prints_the_error_result_and_exits_one(monkeypatch, capsys):
    """The error is printed as a result, not escaped as a traceback with no result."""
    monkeypatch.setattr(_rail_children, "noop", _raises)
    assert rail_child.main([JOB]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "error", "error": "RuntimeError: database is gone"}


@pytest.mark.usefixtures("child")
def test_a_stop_still_passes_through_the_catch(monkeypatch, capsys):
    def stopped():
        raise rail_child.Terminated(EXIT_TERMINATED)

    monkeypatch.setattr(_rail_children, "noop", stopped)
    with pytest.raises(rail_child.Terminated):
        rail_child.main([JOB])
    assert capsys.readouterr().out == ""


# ── a real child, really stopped ────────────────────────────────────────────────
def test_sigterm_ends_the_pass_and_releases_its_lease(tmp_path):
    """The worker's restart sends SIGTERM. Without a handler, a child would die holding its
    lease, and the next child would find it apparently live."""
    env = {"PATH": "/usr/bin:/bin", "OPYT_HOME": str(tmp_path), "HOME": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "pipeline.worker.child", "tests.kb._rail_children:holds_a_lease"],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    runs = Path(tmp_path) / "runs"
    deadline = time.time() + 30
    while not runs.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert runs.read_text() == "held\n"

    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)

    assert proc.returncode == EXIT_TERMINATED, err
    assert "stopped by SIGTERM" in err
    conn = sqlite3.connect(str(tmp_path / "loops.db"))
    holder = conn.execute("SELECT holder FROM sync_lock WHERE name='test-lease'").fetchone()[0]
    conn.close()
    assert holder is None
