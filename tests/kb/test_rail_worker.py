from __future__ import annotations

from pipeline.worker.home import LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV, worker_db_path

import sqlite3
import time

import pytest

import pipeline.worker.supervisor as supervisor
from pipeline.worker.job_queue import RailJobStore
from pipeline.worker.supervisor import RailWorker
from pipeline.worker import registry
from pipeline.worker.failure_run import FailureRun
from pipeline.worker.registry import HOURLY, LINE, RAILS, SCRAPERS, RailSpec


def _wait_and_reap(worker: RailWorker) -> None:
    for child in worker.active:
        child.process.wait(timeout=5)
    worker.reap_finished()


def _launched(worker: RailWorker) -> int:
    """How many children one `launch_available` added (it answers nothing). A launch that fails
    is reaped at once, so it is not counted."""
    before = list(worker.active)
    worker.launch_available()
    return sum(all(child is not old for old in before) for child in worker.active)


def _reaped(worker: RailWorker) -> int:
    """How many children one `reap_finished` recorded and released (it answers nothing)."""
    before = len(worker.active)
    worker.reap_finished()
    return before - len(worker.active)


def _spec(job: str, *, log_name: str = "test.log", **kw) -> RailSpec:
    """A spec for one of the stand-in jobs in `tests/kb/_rail_children.py`."""
    return RailSpec(f"tests.kb._rail_children:{job}", log_name, HOURLY, **kw)


def _wait_for(predicate, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# Cadences that are NOT the hourly default, and why each one is deliberate. Kept as data so the
# assertion below can be exhaustive: every other rail must be hourly, and a rail that quietly
# grows its own number fails here instead of pacing itself in private.
_OFF_CADENCE = {
    # The scrapers' process writes its own next start when it exits; its cadence is only the
    # safety net under that (`quota_loop.process.SAFETY_NET_S`).
    SCRAPERS: registry.QUARTER_HOURLY,
}


def test_registry_is_the_fixed_set_of_jobs_each_launched_through_the_one_child_entry_point():
    """The registry holds exactly the scrapers and the line; the scrapers' process does the work
    of every other job. Every job runs its `run_*` through `child`.

    The line is the one job that runs beside the others, because it stays up for as long as the
    worker does."""
    assert set(RAILS) == {SCRAPERS, LINE}
    for name, cadence in _OFF_CADENCE.items():
        assert RAILS[name].cadence == cadence
    assert all(spec.cadence == HOURLY for name, spec in RAILS.items()
               if name not in _OFF_CADENCE)
    for name, run in {
        SCRAPERS: "pipeline.background:run",
        LINE: "mcp_server.line:run",
    }.items():
        assert RAILS[name].command[:3] == ("-m", "pipeline.worker.child", run)
    assert [name for name, spec in RAILS.items() if spec.beside] == [LINE]
    assert len({spec.log_name for spec in RAILS.values()}) == len(RAILS)


def test_child_gets_selected_home_logs_both_streams_and_records_exit(tmp_path, monkeypatch):
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {"probe": _spec("reports_its_home")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "probe", due_at=0)
    worker = RailWorker(store)

    assert _launched(worker) == 1
    _wait_and_reap(worker)

    log = (home / "test.log").read_text()
    assert str(home) in log
    assert "rail stderr" in log
    result = store.get(LOCAL_HOME_ID, "probe")
    assert result.exit_code == 7
    assert result.finished_at is not None
    assert result.due_at >= result.finished_at + HOURLY


def test_failed_rail_does_not_block_the_next_due_job(tmp_path, monkeypatch):
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {
        "fails": _spec("fails", log_name="fails.log"),
        "later": _spec("passes", log_name="later.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "fails", due_at=1)
    store.activate(LOCAL_HOME_ID, "later", due_at=2)
    worker = RailWorker(store)

    assert _launched(worker) == 1
    _wait_and_reap(worker)
    assert store.get(LOCAL_HOME_ID, "fails").exit_code == 9
    assert _launched(worker) == 1
    _wait_and_reap(worker)

    assert store.get(LOCAL_HOME_ID, "later").exit_code == 0
    assert "later pass" in (home / "later.log").read_text()


def test_a_launch_that_fails_releases_its_claim(tmp_path, monkeypatch):
    """If the claim commits and then the launch raises (here `Popen`; also `ensure_home` or
    opening the log), the row must not stay claimed with no child: single-flight would hold back
    that home's every other job until the next lifetime. It is finished as a child that exited
    1 would be, at its base cadence."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(registry, "RAILS", {
        "broken": _spec("passes", log_name="broken.log"),
        "later": _spec("passes", log_name="later.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "broken", due_at=1)
    store.activate(LOCAL_HOME_ID, "later", due_at=2)
    worker = RailWorker(store)
    real_popen = supervisor.subprocess.Popen

    def popen(cmd, **kw):
        if "tests.kb._rail_children:passes" in cmd and kw["stdout"].name.endswith("broken.log"):
            raise FileNotFoundError("the interpreter is gone")
        return real_popen(cmd, **kw)

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    before = time.time()

    assert _launched(worker) == 1
    assert [c.job.rail for c in worker.active] == ["later"]
    broken = store.get(LOCAL_HOME_ID, "broken")
    assert broken.exit_code == 1 and broken.finished_at is not None
    assert broken.due_at >= before + HOURLY
    _wait_and_reap(worker)
    assert store.get(LOCAL_HOME_ID, "later").exit_code == 0


def test_a_failed_launch_retries_a_locked_finish(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(registry, "RAILS", {
        "broken": _spec("passes", log_name="broken.log"),
        "later": _spec("passes", log_name="later.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "broken", due_at=1)
    store.activate(LOCAL_HOME_ID, "later", due_at=2)
    worker = RailWorker(store)
    real_popen, real_finish = supervisor.subprocess.Popen, store.finish
    finishes = []

    def popen(cmd, **kw):
        if kw["stdout"].name.endswith("broken.log"):
            raise FileNotFoundError("the interpreter is gone")
        return real_popen(cmd, **kw)

    def finish(job, code, **kw):
        finishes.append(job.rail)
        if len(finishes) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_finish(job, code, **kw)

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(store, "finish", finish)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        worker.launch_available()

    assert _reaped(worker) == 1
    assert store.get(LOCAL_HOME_ID, "broken").exit_code == 1
    assert _launched(worker) == 1
    _wait_and_reap(worker)
    assert store.get(LOCAL_HOME_ID, "later").exit_code == 0


def test_global_cap_runs_separate_homes_but_never_two_rails_for_one(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "RAILS", {
        "one": _spec("sleeps", log_name="one.log"),
        "two": _spec("sleeps", log_name="two.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    for home_id, rail in (("42", "one"), ("42", "two"), ("77", "one"), ("88", "one")):
        store.activate(home_id, rail, due_at=0)
    worker = RailWorker(store, homes_root=tmp_path / "homes", max_children=3)

    try:
        assert _launched(worker) == 3
        assert {child.job.home_id for child in worker.active} == {"42", "77", "88"}
        assert sum(child.job.home_id == "42" for child in worker.active) == 1
    finally:
        for child in worker.active:
            child.process.terminate()
        _wait_and_reap(worker)


# ── a job that runs beside the others (the line) ────────────────────────────────
def _stop_all(worker: RailWorker) -> None:
    for child in worker.active:
        child.process.terminate()
    for child in worker.active:
        child.process.wait(timeout=30)
    worker.active = []


def test_a_job_beside_the_others_takes_no_slot_and_the_homes_job_still_runs(tmp_path, monkeypatch):
    """The line stays up for as long as the worker does. A local worker has one slot, so a job
    that counted against it would keep the scrapers from ever starting; and a home runs one job
    at a time, so one that counted there would too. It is outside both rules, and they still
    hold for the rest."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(registry, "RAILS", {
        "held": _spec("sleeps", log_name="held.log", beside=True),
        "one": _spec("sleeps", log_name="one.log"),
        "two": _spec("sleeps", log_name="two.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    for rail in ("held", "one", "two"):
        store.activate(LOCAL_HOME_ID, rail, due_at=0)
    worker = RailWorker(store, max_children=1)

    try:
        assert _launched(worker) == 2
        assert {c.job.rail for c in worker.active} == {"held", "one"}
        assert _launched(worker) == 0       # one slot and one job per home, still
    finally:
        _stop_all(worker)


@pytest.mark.parametrize("first, then", [("held", "one"), ("one", "held")])
def test_neither_a_job_beside_the_others_nor_the_homes_job_waits_for_the_other(
        tmp_path, monkeypatch, first, then):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(registry, "RAILS", {
        "held": _spec("sleeps", log_name="held.log", beside=True),
        "one": _spec("sleeps", log_name="one.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, first, due_at=0)
    worker = RailWorker(store, max_children=1)

    try:
        assert _launched(worker) == 1
        store.activate(LOCAL_HOME_ID, then, due_at=0)
        assert _launched(worker) == 1
        assert {c.job.rail for c in worker.active} == {"held", "one"}
    finally:
        _stop_all(worker)


def test_a_job_beside_the_others_is_stopped_by_the_restart_without_waiting_out_the_grace(
        tmp_path, monkeypatch):
    """It would never exit on its own, so the deadline's SIGTERM is what ends it: at once, and
    recorded, not killed when the grace runs out."""
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {
        "held": _spec("sleeps", log_name="held.log", beside=True)})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "held", due_at=0)
    worker = RailWorker(store, poll_interval=0.05, restart_grace=60.0)

    began = time.monotonic()
    worker.run_forever(max_lifetime=1.0)

    assert time.monotonic() - began < 30
    assert store.get(LOCAL_HOME_ID, "held").finished_at is not None
    assert "stopped by SIGTERM" in (home / "held.log").read_text()


_JOBS = {"held": _spec("noop", beside=True), "a": _spec("noop"), "b": _spec("noop")}


@pytest.mark.parametrize("taken, max_children, expected", [
    (0, 1, ("held", "a", "b")),                          # a free slot: every job
    (1, 1, ("held",)),                                   # full: only what needs no slot
    (1, 2, ("held", "a", "b")),
    (8, 8, ("held",)),                                   # hosted is the same rule
])
def test_which_jobs_a_launch_may_claim(taken, max_children, expected):
    assert supervisor.launchable(taken, max_children, _JOBS) == expected


def test_with_no_job_beside_the_others_a_full_worker_claims_nothing():
    assert supervisor.launchable(1, 1, {"a": _spec("noop"), "b": _spec("noop")}) == ()


def test_local_and_hosted_workers_write_only_their_selected_homes(tmp_path, monkeypatch):
    local_home = tmp_path / "local"
    hosted_root = tmp_path / "hosted"
    monkeypatch.setenv("OPYT_HOME", str(local_home))
    monkeypatch.setattr(registry, "RAILS", {"probe": _spec("marks_its_home")})

    local_store = RailJobStore(tmp_path / "local-jobs.db")
    local_store.activate(LOCAL_HOME_ID, "probe", due_at=0)
    local_worker = RailWorker(local_store)
    local_worker.launch_available()
    _wait_and_reap(local_worker)

    hosted_store = RailJobStore(tmp_path / "hosted-jobs.db")
    hosted_store.activate("42", "probe", due_at=0)
    hosted_store.activate("77", "probe", due_at=0)
    hosted_worker = RailWorker(hosted_store, homes_root=hosted_root, max_children=2)
    hosted_worker.launch_available()
    _wait_and_reap(hosted_worker)

    assert (local_home / "selected-home").read_text() == "local"
    assert (hosted_root / "42" / "selected-home").read_text() == "42"
    assert (hosted_root / "77" / "selected-home").read_text() == "77"
    assert not (local_home / "42").exists()
    assert not (hosted_root / LOCAL_HOME_ID).exists()


@pytest.mark.parametrize("max_children, poll_interval, refusal", [
    (1, 1.0, None),                                         # the defaults run
    (0, 1.0, "max_children must be at least one"),
    (1, 0.0, "poll_interval must be greater than zero"),
])
def test_which_settings_the_worker_refuses(max_children, poll_interval, refusal):
    assert supervisor.settings_refusal(max_children, poll_interval) == refusal


def test_a_refused_setting_leaves_no_database(tmp_path):
    """`main` refuses a setting the worker cannot run with before it
    creates the control database, so a refusal leaves nothing behind."""
    db = tmp_path / "worker" / "rail_jobs.db"
    with pytest.raises(ValueError, match="max_children"):
        supervisor.main(["--db", str(db), "--max-children", "0"])
    assert not db.exists()


def test_worker_database_defaults_to_the_local_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv("OPYT_WORKER_DB", raising=False)

    assert worker_db_path() == tmp_path / "rail_jobs.db"


# ── the child is also a producer ────────────────────────────────────────────────
def test_a_rail_child_can_queue_its_successor_for_its_own_home(tmp_path, monkeypatch):
    """A rail is not only work, it is an event: staging candidates makes stage 3 due. The child
    can only say so if the worker tells it which home it is and which control database to write.
    Hosted, a child without the home id falls through to `LOCAL_HOME_ID` and queues against the
    wrong user; without the database path, `worker_db_path()` raises rather than let the row land
    somewhere no worker looks."""
    home = tmp_path / "local-home"
    db = tmp_path / "jobs.db"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    monkeypatch.setattr(registry, "RAILS", {
        "producer": _spec("queues_its_successor"),
        "successor": _spec("noop"),
    })
    store = RailJobStore(db)
    store.activate(LOCAL_HOME_ID, "producer", due_at=0)
    worker = RailWorker(store)

    assert _launched(worker) == 1
    _wait_and_reap(worker)

    assert store.get(LOCAL_HOME_ID, "producer").exit_code == 0     # the queue call succeeded
    queued = store.get(LOCAL_HOME_ID, "successor")
    assert queued is not None and queued.started_at is None


def test_a_hosted_child_queues_against_its_own_home_not_the_local_id(tmp_path, monkeypatch):
    """The isolation this whole boundary exists for: two homes, one worker, one control database,
    and a child that must not be able to schedule work for anybody else."""
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    homes = tmp_path / "homes"
    monkeypatch.setattr(registry, "RAILS", {
        "producer": _spec("queues_its_successor"),
        "successor": _spec("noop"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "producer", due_at=0)
    worker = RailWorker(store, homes_root=homes)

    assert _launched(worker) == 1
    _wait_and_reap(worker)

    assert store.get("42", "producer").exit_code == 0
    assert store.get("42", "successor") is not None
    assert store.get(LOCAL_HOME_ID, "successor") is None
    assert store.get("77", "successor") is None


# ── The bounded lifetime, which is how a published fix reaches the rails ─────────────────────
#
# The LaunchAgent launches `uvx --from opyt@latest opyt-worker`, and uvx resolves `@latest`
# only at process start. A worker that never exits therefore runs its birth build forever;
# the bound makes it exit clean so unconditional KeepAlive relaunches it onto the new build.


def test_a_bounded_lifetime_drains_the_child_and_leaves_its_claim_recoverable(
        tmp_path, monkeypatch):
    """The exit is a drain, not a walk-away, and not a bookkeeping event.

    `_launch` deliberately gives children no new session — the thing that stops the worker
    stops its group. True for launchd/systemd stops, false for a voluntary exit, so the worker
    must terminate its own children or orphan them for the next lifetime to double-launch
    beside. And the interrupted row must be left unfinished: `finish(exit_code=-15)` would
    push the rail a whole cadence out, while an unfinished claim is exactly what the next
    lifetime's `recover_interrupted_claims` re-queues for now. Every child is asked to stop at
    the deadline, so only one still running when the grace runs out is killed this way.
    """
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {"slow": _spec("outlives_a_sigterm")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "slow", due_at=0)
    worker = RailWorker(store, poll_interval=0.05, restart_grace=0.3)

    worker.run_forever(max_lifetime=0.3)  # returns instead of supervising forever

    assert worker.active == []  # the child was stopped and collected, not abandoned
    row = store.get(LOCAL_HOME_ID, "slow")
    assert row.started_at is not None and row.finished_at is None  # interrupted, not finished
    store.recover_interrupted_claims()
    recovered = store.get(LOCAL_HOME_ID, "slow")
    assert recovered.started_at is None and recovered.due_at <= supervisor.time.time()


def test_an_unbounded_worker_stays_the_default_shape(tmp_path, monkeypatch):
    """`max_lifetime=None` must not grow a deadline: hosted runs unbounded under systemd."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(registry, "RAILS", {"noop": _spec("noop")})
    store = RailJobStore(tmp_path / "jobs.db")
    worker = RailWorker(store, poll_interval=0.01)

    # No jobs and no deadline: prove the loop is still a loop by interrupting it.
    calls = {"n": 0}
    real_sleep = supervisor.time.sleep

    def counting_sleep(seconds):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt
        real_sleep(min(seconds, 0.01))

    monkeypatch.setattr(supervisor.time, "sleep", counting_sleep)
    try:
        worker.run_forever(max_lifetime=None)
    except KeyboardInterrupt:
        pass
    assert calls["n"] == 3


def _lifetime_main_resolves_to(monkeypatch, tmp_path, argv, env=None):
    """Run `main` with run_forever stubbed out; return the max_lifetime it was handed."""
    seen = {}
    monkeypatch.setattr(supervisor.RailWorker, "run_forever",
                        lambda self, *, max_lifetime=None: seen.update(m=max_lifetime))
    monkeypatch.delenv("OPYT_WORKER_MAX_LIFETIME", raising=False)
    monkeypatch.delenv("OPYT_HOMES_ROOT", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    assert supervisor.main(["--db", str(tmp_path / "jobs.db"), *argv]) == 0
    return seen["m"]


def test_the_local_default_is_bounded_and_the_hosted_default_is_not(tmp_path, monkeypatch):
    assert _lifetime_main_resolves_to(monkeypatch, tmp_path, []) == 6 * HOURLY
    assert _lifetime_main_resolves_to(
        monkeypatch, tmp_path, ["--homes-root", str(tmp_path / "homes")]) is None


def test_the_lifetime_is_operator_overridable_without_code(tmp_path, monkeypatch):
    assert _lifetime_main_resolves_to(
        monkeypatch, tmp_path, ["--max-lifetime-seconds", "120"]) == 120.0
    assert _lifetime_main_resolves_to(
        monkeypatch, tmp_path, [], env={"OPYT_WORKER_MAX_LIFETIME": "900"}) == 900.0
    # 0 disables the bound entirely — on either side of the default.
    assert _lifetime_main_resolves_to(
        monkeypatch, tmp_path, ["--max-lifetime-seconds", "0"]) is None


# ── a transient store error is survived, not fatal ─────────────────────────────
def test_a_locked_store_never_kills_run_forever(tmp_path, monkeypatch):
    """`database is locked` is a normal outcome of a busy box, and must not reach the top
    of the program and become exit(1). Both the startup `recover_interrupted_claims` and the loop body must retry instead."""
    import sqlite3

    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {"noop": _spec("noop", log_name="noop.log")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "noop", due_at=0)
    worker = RailWorker(store, poll_interval=0.01)

    remaining = {"recover": 2, "claim": 2}

    def flaky(name, real):
        def call(*args, **kwargs):
            if remaining[name]:
                remaining[name] -= 1
                raise sqlite3.OperationalError("database is locked")
            return real(*args, **kwargs)
        return call

    monkeypatch.setattr(store, "recover_interrupted_claims",
                        flaky("recover", store.recover_interrupted_claims))
    monkeypatch.setattr(store, "claim_next", flaky("claim", store.claim_next))

    worker.run_forever(max_lifetime=2.0)

    assert remaining == {"recover": 0, "claim": 0}  # the errors actually fired
    row = store.get(LOCAL_HOME_ID, "noop")
    assert row.exit_code == 0 and row.finished_at is not None


def test_reap_records_partial_progress_and_never_double_finishes(tmp_path, monkeypatch):
    """A `finish` that raises mid-reap must not strand the children already recorded: re-finishing
    them on retry is refused by `finish` as "no longer active", which would wedge the loop on a
    RuntimeError forever."""
    import pytest
    import sqlite3

    monkeypatch.setattr(registry, "RAILS", {"probe": _spec("noop")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "probe", due_at=0)
    store.activate("77", "probe", due_at=0)
    worker = RailWorker(store, homes_root=tmp_path / "homes", max_children=2)
    assert _launched(worker) == 2
    for child in worker.active:
        child.process.wait(timeout=5)

    real_finish = store.finish
    calls = {"n": 0}

    def finish_failing_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(store, "finish", finish_failing_second)

    with pytest.raises(sqlite3.OperationalError):
        worker.reap_finished()  # first child recorded, second raises
    assert len(worker.active) == 1  # the recorded child left; the failed one stayed

    assert _reaped(worker) == 1  # retry finishes ONLY the remaining child
    assert worker.active == []
    assert store.get("42", "probe").exit_code == 0
    assert store.get("77", "probe").exit_code == 0


def test_the_worker_beats_once_per_window_not_once_per_poll(tmp_path, monkeypatch):
    """~30 loop passes land inside BEAT_SECONDS here; a beat per poll would be ~86,400 writes/day
    into the very database whose lock contention run_forever exists to survive."""
    from pipeline.worker.job_queue import worker_is_alive

    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setenv("OPYT_WORKER_DB", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("OPYT_WORKER_HOME_ID", raising=False)
    monkeypatch.setattr(registry, "RAILS", {})
    store = RailJobStore(tmp_path / "jobs.db")
    beats: list[tuple] = []
    real_beat = store.beat

    def recording_beat(scope, worker_id, **kwargs):
        beats.append((scope, worker_id))
        return real_beat(scope, worker_id, **kwargs)

    monkeypatch.setattr(store, "beat", recording_beat)
    worker = RailWorker(store, poll_interval=0.01)

    worker.run_forever(max_lifetime=0.3)

    assert len(beats) == 1
    scope, worker_id = beats[0]
    assert scope == LOCAL_HOME_ID
    assert worker_id.count(":") == 2  # host:pid:uuid, sync_lock's holder shape
    assert worker_is_alive() is True


def test_a_hosted_worker_beats_under_the_shared_scope(tmp_path, monkeypatch):
    from pipeline.worker.job_queue import SHARED_BEAT_SCOPE

    monkeypatch.setattr(registry, "RAILS", {})
    store = RailJobStore(tmp_path / "jobs.db")
    beats: list[str] = []
    monkeypatch.setattr(store, "beat",
                        lambda scope, worker_id, **kw: beats.append(scope))
    worker = RailWorker(store, homes_root=tmp_path / "homes", poll_interval=0.01)

    worker.run_forever(max_lifetime=0.1)

    assert beats == [SHARED_BEAT_SCOPE]


# ── a job the worker does not know ──────────────────────────────────────────────
def test_a_retired_job_is_never_claimed_and_its_row_is_deleted(tmp_path, monkeypatch):
    """A row for a job the worker cannot launch (`oracle_refresh`, `candidate_probe`) holds no
    work, only a wake-up time. Claiming it would raise in `_launch`, and a row left claimed
    blocks every other job for that home. Here one is even left claimed by the previous
    lifetime, and the known job beside it still runs."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(registry, "RAILS", {"noop": _spec("noop")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "oracle_refresh", due_at=0)
    store.activate(LOCAL_HOME_ID, "candidate_probe", due_at=0)
    assert store.claim_next(home_id=LOCAL_HOME_ID).rail in {"oracle_refresh", "candidate_probe"}
    store.activate(LOCAL_HOME_ID, "noop", due_at=0)

    RailWorker(store, poll_interval=0.02).run_forever(max_lifetime=1.5)

    assert [j.rail for j in store.list_jobs()] == ["noop"]
    assert store.get(LOCAL_HOME_ID, "noop").exit_code == 0


def test_a_row_queued_under_a_retired_name_later_is_never_claimed(tmp_path, monkeypatch):
    """A producer still writing a retired name mid-lifetime never gets its row claimed."""
    monkeypatch.setattr(registry, "RAILS", {"noop": _spec("noop")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "oracle_refresh", due_at=0)

    assert _launched(RailWorker(store)) == 0
    assert store.get(LOCAL_HOME_ID, "oracle_refresh").started_at is None


# ── restarting when idle ───────────────────────────────────────────
def _lease_holder(home, name: str):
    db = home / "loops.db"
    if not db.exists():
        return "missing"
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute("SELECT holder FROM sync_lock WHERE name=?", (name,)).fetchone()
    except sqlite3.OperationalError:
        return "missing"
    finally:
        conn.close()
    return "missing" if row is None else row[0]


def _runs(home) -> list[str]:
    path = home / "runs"
    return path.read_text().split() if path.exists() else []


def _rerun(store) -> RailWorker:
    """The next worker lifetime: recover, then launch what is due, and wait for it."""
    worker = RailWorker(store)
    store.drop_retired_jobs(rails=tuple(registry.RAILS))
    store.recover_interrupted_claims()
    job = store.get(LOCAL_HOME_ID, "leased")
    assert job.due_at <= time.time(), "the stopped job must be due again at once"
    assert _launched(worker) == 1
    _wait_and_reap(worker)
    return worker


def test_a_rail_stopped_mid_pass_releases_its_lease_and_reruns_at_once(tmp_path, monkeypatch):
    """A child stopped mid-pass must not die holding its lease: its last beat is seconds old, so
    the next lifetime's child would find it apparently live, return `already_running`, and the
    job would wait a whole cycle. `child` turns SIGTERM into a normal exit, so the lease is released on the way out and the rerun takes it."""
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {"leased": _spec("holds_a_lease")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "leased", due_at=0)
    worker = RailWorker(store)
    assert _launched(worker) == 1
    assert _wait_for(lambda: _runs(home) == ["held"])

    worker.stop_children()                                        # the restart's SIGTERM

    assert _lease_holder(home, "test-lease") is None              # released, not left to age
    _rerun(store)
    assert _runs(home) == ["held", "held"]
    assert store.get(LOCAL_HOME_ID, "leased").exit_code == 0


def test_a_rail_killed_outright_leaves_a_lease_the_rerun_can_still_take(tmp_path, monkeypatch):
    """The backstop, for what no handler can catch (SIGKILL, a crash): the lease still names
    the dead process and its heartbeat is seconds old, and `CatchupLock` frees it because that
    process is gone."""
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {"leased": _spec("holds_a_lease")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "leased", due_at=0)
    worker = RailWorker(store)
    assert _launched(worker) == 1
    assert _wait_for(lambda: _runs(home) == ["held"])

    for child in worker.active:
        child.process.kill()
        child.process.wait()
    worker.active = []

    assert _lease_holder(home, "test-lease") not in (None, "missing")   # left behind
    _rerun(store)
    assert _runs(home) == ["held", "held"]


def test_the_scrapers_stopped_by_a_restart_release_their_leases_and_are_due_at_once(
        tmp_path, monkeypatch):
    """The same restart, for the real runner: told to stop, its scraper releases the quota's
    lease, the runner makes its own row due now and exits healthy, and the worker records that
    without pushing it out a cadence."""
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(registry, "RAILS", {
        SCRAPERS: _spec("runs_an_idle_scraper", log_name="scrapers.log")})
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, SCRAPERS, due_at=0)
    worker = RailWorker(store)
    assert _launched(worker) == 1
    assert _wait_for(lambda: _lease_holder(home, "test:quota") not in (None, "missing"))

    worker.ask_children_to_stop()
    _wait_and_reap(worker)

    assert _lease_holder(home, "test:quota") is None
    row = store.get(LOCAL_HOME_ID, SCRAPERS)
    assert row.exit_code == 0, (home / "scrapers.log").read_text()
    assert row.due_at <= row.finished_at                          # due again at once
    assert store.claim_next(home_id=LOCAL_HOME_ID, rails=(SCRAPERS,)) is not None


# ── the supervisor's decisions, one row each ────────────────────────────────────
@pytest.mark.parametrize("now, deadline, draining_since, running, grace, expected", [
    (5.0, None, None, True, 60.0, "run"),                # no bound
    (5.0, 10.0, None, True, 60.0, "run"),                # before the deadline
    (10.0, 10.0, None, True, 60.0, "start"),             # at it, children running
    (10.0, 10.0, None, False, 60.0, "exit"),             # at it, nothing running
    (10.0, 10.0, None, True, 0.0, "exit"),               # no grace: stop at once
    (30.0, 10.0, 10.0, True, 60.0, "wait"),              # within the grace
    (70.0, 10.0, 10.0, True, 60.0, "exit"),              # the grace is over
    (30.0, 10.0, 10.0, False, 60.0, "exit"),             # everything finished
])
def test_where_a_lifetime_stands_against_its_bound(now, deadline, draining_since, running,
                                                   grace, expected):
    assert supervisor.drain(now, deadline, draining_since, running, grace) == expected


@pytest.mark.parametrize("arg, env, hosted, expected", [
    (120.0, "999", False, 120.0),                        # the argument wins
    (None, "300", True, 300.0),                          # then the environment
    (None, None, False, 6 * HOURLY),                     # local default
    (None, "", False, 6 * HOURLY),                       # an empty variable is unset
    (None, None, True, None),                            # hosted: unbounded
    (0.0, None, False, None),                            # 0 disables
    (None, "-1", False, None),                           # so does a negative
])
def test_how_long_one_lifetime_may_run(arg, env, hosted, expected):
    assert supervisor.lifetime_bound(arg, env, hosted=hosted) == expected


@pytest.mark.parametrize("arg, env, ignored", [
    (None, "6h", True),                                  # unreadable, nothing overrides it
    (120.0, "6h", False),                                # the argument wins: nothing to say
    (None, "300", False),                                # readable
    (None, "", False),                                   # unset
    (None, None, False),
])
def test_when_the_lifetime_variable_is_ignored(arg, env, ignored):
    assert supervisor.ignored_lifetime_env(arg, env) is ignored


@pytest.mark.parametrize("hosted, expected", [(False, 6 * HOURLY), (True, None)])
def test_an_invalid_lifetime_uses_the_platform_default(hosted, expected):
    assert supervisor.lifetime_bound(None, "6h", hosted=hosted) == expected


def test_main_logs_an_invalid_lifetime_once_and_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_WORKER_MAX_LIFETIME", "6h")
    messages, lifetimes = [], []
    monkeypatch.setattr(supervisor, "log", messages.append)
    monkeypatch.setattr(RailWorker, "run_forever",
                        lambda _self, *, max_lifetime: lifetimes.append(max_lifetime))
    assert supervisor.main(["--db", str(tmp_path / "jobs.db")]) == 0
    assert lifetimes == [6 * HOURLY]
    assert len(messages) == 1 and "6h" in messages[0]


def test_a_run_of_failures_is_reported_at_its_start_and_its_end():
    run = FailureRun()
    assert run.recovered(5.0) is None                    # nothing was failing
    assert run.failed(10.0) is True                      # the onset: log it
    assert run.failed(11.0) is False                     # the same run: quiet
    assert run.recovered(40.0) == 30.0                   # the end, and how long it lasted
    assert run.recovered(41.0) is None
    assert run.failed(50.0) is True                      # a new run
