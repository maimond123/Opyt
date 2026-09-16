from __future__ import annotations

import pipeline.kb.rail_worker as rail_worker
from pipeline.kb.rail_jobs import (LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV,
                                   RailJobStore)
from pipeline.kb.rail_worker import HOURLY, RAILS, SIX_HOURLY, RailSpec, RailWorker


def _wait_and_reap(worker: RailWorker) -> None:
    for child in worker.active:
        child.process.wait(timeout=5)
    worker.reap_finished()


def _spec(code: str, *, log_name: str = "test.log") -> RailSpec:
    return RailSpec(("-c", code), log_name, HOURLY)


# Cadences that are NOT the hourly default, and why each one is deliberate. Kept as data so the
# assertion below can be exhaustive: every other rail must be hourly, and a rail that quietly
# grows its own number fails here instead of pacing itself in private.
_OFF_CADENCE = {
    "oracle_refresh": 600.0,             # the shortest — it is the only rail with live pairs
    "substack_saved_catchup": SIX_HOURLY,  # a Cloudflare-guarded reader endpoint + pending retries
}


def test_registry_is_the_fixed_set_of_existing_bounded_commands():
    assert set(RAILS) == {
        "oracle_refresh", "frontier_execute", "frontier_admit", "bookmark_catchup",
        "curation_catchup", "substack_saved_catchup", "candidate_probe", "push_catchup",
        "sitting_scheduler",
    }
    assert RAILS["oracle_refresh"].command == (
        "-c", "from pipeline.kb.oracle_refresh import _run; _run()"
    )
    for name, cadence in _OFF_CADENCE.items():
        assert RAILS[name].cadence == cadence
    assert all(spec.cadence == HOURLY for name, spec in RAILS.items()
               if name not in _OFF_CADENCE)
    for name, module in {
        "frontier_execute": "frontier_execute",
        "frontier_admit": "frontier_admit",
        "bookmark_catchup": "bookmark_catchup",
        "curation_catchup": "curation_catchup",
        "substack_saved_catchup": "substack_saved_catchup",
        "candidate_probe": "probe_catchup",
        "push_catchup": "push_catchup",
        "sitting_scheduler": "sitting_scheduler",
    }.items():
        assert RAILS[name].command == ("-m", f"pipeline.kb.{module}", "--once")


def test_child_gets_selected_home_logs_both_streams_and_records_exit(tmp_path, monkeypatch):
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(rail_worker, "RAILS", {
        "probe": _spec(
            "import os,sys; print(os.environ['OPYT_HOME']); "
            "print('rail stderr', file=sys.stderr); raise SystemExit(7)"
        )
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "probe", due_at=0)
    worker = RailWorker(store)

    assert worker.launch_available() == 1
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
    monkeypatch.setattr(rail_worker, "RAILS", {
        "fails": _spec("print('failed pass'); raise SystemExit(9)", log_name="fails.log"),
        "later": _spec("print('later pass')", log_name="later.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "fails", due_at=1)
    store.activate(LOCAL_HOME_ID, "later", due_at=2)
    worker = RailWorker(store)

    assert worker.launch_available() == 1
    _wait_and_reap(worker)
    assert store.get(LOCAL_HOME_ID, "fails").exit_code == 9
    assert worker.launch_available() == 1
    _wait_and_reap(worker)

    assert store.get(LOCAL_HOME_ID, "later").exit_code == 0
    assert "later pass" in (home / "later.log").read_text()


def test_global_cap_runs_separate_homes_but_never_two_rails_for_one(tmp_path, monkeypatch):
    code = "import time; print('started', flush=True); time.sleep(30)"
    monkeypatch.setattr(rail_worker, "RAILS", {
        "one": _spec(code, log_name="one.log"),
        "two": _spec(code, log_name="two.log"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    for home_id, rail in (("42", "one"), ("42", "two"), ("77", "one"), ("88", "one")):
        store.activate(home_id, rail, due_at=0)
    worker = RailWorker(store, homes_root=tmp_path / "homes", max_children=3)

    try:
        assert worker.launch_available() == 3
        assert {child.job.home_id for child in worker.active} == {"42", "77", "88"}
        assert sum(child.job.home_id == "42" for child in worker.active) == 1
    finally:
        for child in worker.active:
            child.process.terminate()
        _wait_and_reap(worker)


def test_local_and_hosted_workers_write_only_their_selected_homes(tmp_path, monkeypatch):
    local_home = tmp_path / "local"
    hosted_root = tmp_path / "hosted"
    monkeypatch.setenv("OPYT_HOME", str(local_home))
    code = (
        "import os; from pathlib import Path; p=Path(os.environ['OPYT_HOME']); "
        "(p/'selected-home').write_text(p.name)"
    )
    monkeypatch.setattr(rail_worker, "RAILS", {"probe": _spec(code)})

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


def test_worker_database_defaults_to_the_local_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv("OPYT_WORKER_DB", raising=False)

    assert rail_worker.worker_db_path() == tmp_path / "rail_jobs.db"


# ── the child is also a producer ────────────────────────────────────────────────
def test_a_rail_child_can_queue_its_successor_for_its_own_home(tmp_path, monkeypatch):
    """A rail is not only work, it is an EVENT: staging candidates makes stage 3 due. The child
    can only say so if the worker tells it which home it is and which control database to write.
    Hosted, a child without the home id falls through to `LOCAL_HOME_ID` and queues against the
    wrong user; without the database path, `worker_db_path()` raises rather than let the row land
    somewhere no worker looks."""
    home = tmp_path / "local-home"
    db = tmp_path / "jobs.db"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    monkeypatch.setattr(rail_worker, "RAILS", {
        "producer": _spec("from pipeline.kb.rail_jobs import request_now; "
                          "raise SystemExit(0 if request_now('successor') else 1)"),
        "successor": _spec("raise SystemExit(0)"),
    })
    store = RailJobStore(db)
    store.activate(LOCAL_HOME_ID, "producer", due_at=0)
    worker = RailWorker(store)

    assert worker.launch_available() == 1
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
    monkeypatch.setattr(rail_worker, "RAILS", {
        "producer": _spec("from pipeline.kb.rail_jobs import request_now; "
                          "raise SystemExit(0 if request_now('successor') else 1)"),
        "successor": _spec("raise SystemExit(0)"),
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "producer", due_at=0)
    worker = RailWorker(store, homes_root=homes)

    assert worker.launch_available() == 1
    _wait_and_reap(worker)

    assert store.get("42", "producer").exit_code == 0
    assert store.get("42", "successor") is not None
    assert store.get(LOCAL_HOME_ID, "successor") is None
    assert store.get("77", "successor") is None


def test_the_sitting_command_hands_the_child_no_breaker_override(tmp_path):
    """The breaker exists to stop a failing loop from burning money. Nothing in the registry may
    smuggle a caller past it — there is no override to pass, and this asserts the command stays
    the bare `--once` that cannot acquire one."""
    assert RAILS["sitting_scheduler"].command == ("-m", "pipeline.kb.sitting_scheduler", "--once")
    assert not any("breaker" in part or "force" in part
                   for spec in RAILS.values() for part in spec.command)


# ── The bounded lifetime, which is how a published fix reaches the rails ─────────────────────
#
# The LaunchAgent launches `uvx --from opyt@latest opyt-worker`, and uvx resolves `@latest`
# only at process start. A worker that never exits therefore runs its birth build forever;
# the bound makes it exit clean so unconditional KeepAlive relaunches it onto the new build.


def test_a_bounded_lifetime_drains_the_child_and_leaves_its_claim_recoverable(
        tmp_path, monkeypatch):
    """The exit is a DRAIN, not a walk-away, and not a bookkeeping event.

    `_launch` deliberately gives children no new session — the thing that stops the worker
    stops its group. True for launchd/systemd stops, false for a voluntary exit, so the worker
    must terminate its own children or orphan them for the next lifetime to double-launch
    beside. And the interrupted row must be left UNFINISHED: `finish(exit_code=-15)` would
    push the rail a whole cadence out, while an unfinished claim is exactly what the next
    lifetime's `recover_interrupted_claims` re-queues for now.
    """
    home = tmp_path / "local-home"
    monkeypatch.setenv("OPYT_HOME", str(home))
    monkeypatch.setattr(rail_worker, "RAILS", {
        "slow": _spec("import time; time.sleep(60)")
    })
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate(LOCAL_HOME_ID, "slow", due_at=0)
    worker = RailWorker(store, poll_interval=0.05)

    worker.run_forever(max_lifetime=0.3)  # returns instead of supervising forever

    assert worker.active == []  # the child was stopped and collected, not abandoned
    row = store.get(LOCAL_HOME_ID, "slow")
    assert row.started_at is not None and row.finished_at is None  # interrupted, not finished
    assert store.recover_interrupted_claims() == 1
    recovered = store.get(LOCAL_HOME_ID, "slow")
    assert recovered.started_at is None and recovered.due_at <= rail_worker.time.time()


def test_an_unbounded_worker_stays_the_default_shape(tmp_path, monkeypatch):
    """`max_lifetime=None` must not grow a deadline: hosted runs unbounded under systemd."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "local-home"))
    monkeypatch.setattr(rail_worker, "RAILS", {"noop": _spec("pass")})
    store = RailJobStore(tmp_path / "jobs.db")
    worker = RailWorker(store, poll_interval=0.01)

    # No jobs and no deadline: prove the loop is still a loop by interrupting it.
    calls = {"n": 0}
    real_sleep = rail_worker.time.sleep

    def counting_sleep(seconds):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt
        real_sleep(min(seconds, 0.01))

    monkeypatch.setattr(rail_worker.time, "sleep", counting_sleep)
    try:
        worker.run_forever(max_lifetime=None)
    except KeyboardInterrupt:
        pass
    assert calls["n"] == 3


def _lifetime_main_resolves_to(monkeypatch, tmp_path, argv, env=None):
    """Run `main` with run_forever stubbed out; return the max_lifetime it was handed."""
    seen = {}
    monkeypatch.setattr(rail_worker.RailWorker, "run_forever",
                        lambda self, *, max_lifetime=None: seen.update(m=max_lifetime))
    monkeypatch.delenv("OPYT_WORKER_MAX_LIFETIME", raising=False)
    monkeypatch.delenv("OPYT_HOMES_ROOT", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    assert rail_worker.main(["--db", str(tmp_path / "jobs.db"), *argv]) == 0
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
