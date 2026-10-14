from __future__ import annotations

from pipeline.worker.home import LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pipeline.worker.job_queue import DEFAULT_PRIORITY, URGENT_PRIORITY, RailJobStore, request_now
from pipeline.worker.supervisor import WorkerLock


def test_activation_is_unique_and_only_brings_a_schedule_forward(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")

    store.activate("42", "oracle_refresh", due_at=200, priority=0)
    store.activate("42", "oracle_refresh", due_at=100, priority=URGENT_PRIORITY)
    urgent = store.get("42", "oracle_refresh")
    store.activate("42", "oracle_refresh", due_at=300, priority=0)
    unchanged = store.get("42", "oracle_refresh")

    assert len(store.list_jobs()) == 1
    assert urgent.due_at == unchanged.due_at == 100
    assert urgent.priority == unchanged.priority == URGENT_PRIORITY


@pytest.mark.parametrize("home_id", ["", ".", "..", "../42", "42/77", "x" * 129])
def test_activation_rejects_a_home_id_that_cannot_name_one_hosted_directory(
        tmp_path, home_id):
    store = RailJobStore(tmp_path / "jobs.db")

    with pytest.raises(ValueError):
        store.activate(home_id, "oracle_refresh")

    assert store.list_jobs() == []


def test_concurrent_claims_claim_one_job_once(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "oracle_refresh", due_at=10)
    barrier = threading.Barrier(8)

    def claim():
        barrier.wait()
        return store.claim_next(now=10)

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: claim(), range(8)))

    assert sum(job is not None for job in claims) == 1


def test_claims_fill_global_capacity_without_overlapping_one_home(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "bookmark_catchup", due_at=10)
    store.activate("42", "oracle_refresh", due_at=10, priority=URGENT_PRIORITY)
    store.activate("77", "oracle_refresh", due_at=10)
    store.activate("88", "oracle_refresh", due_at=10)

    claims = [store.claim_next(now=10) for _ in range(4)]

    claimed = [job for job in claims if job is not None]
    assert len(claimed) == 3
    assert {job.home_id for job in claimed} == {"42", "77", "88"}
    assert next(job for job in claimed if job.home_id == "42").rail == "oracle_refresh"


def test_due_now_activation_wins_over_the_previous_later_schedule(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "oracle_refresh", due_at=500)
    store.activate("77", "bookmark_catchup", due_at=200)

    store.activate("42", "oracle_refresh", due_at=100, priority=URGENT_PRIORITY)

    claimed = store.claim_next(now=200)
    assert (claimed.home_id, claimed.rail) == ("42", "oracle_refresh")


def test_lifetime_lock_gates_interrupted_claim_recovery(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "oracle_refresh", due_at=10)
    first = WorkerLock(store.db_path)
    second = WorkerLock(store.db_path)

    assert first.acquire() is True
    claimed = store.claim_next(now=10)
    assert claimed is not None
    assert second.acquire() is False
    assert store.claim_next(now=20) is None

    first.release()
    assert second.acquire() is True
    try:
        store.recover_interrupted_claims(now=20)
        assert store.get("42", "oracle_refresh").started_at is None
        recovered = store.claim_next(now=20)
        assert recovered is not None
        assert (recovered.home_id, recovered.rail) == ("42", "oracle_refresh")
    finally:
        second.release()


# ── The producer half: a product action asking for a pass now ───────────────────────────────

def test_a_product_action_queues_a_due_now_job_for_its_own_home(tmp_path, monkeypatch):
    """The one call every converted tool makes, and where it lands with no hosted variables set."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    before = time.time()

    assert request_now("push_catchup") is True

    job = RailJobStore(tmp_path / "rail_jobs.db").get(LOCAL_HOME_ID, "push_catchup")
    assert job is not None, "a local producer must queue into the home the local worker opens"
    assert before <= job.due_at <= time.time()
    assert job.started_at is None and job.priority == DEFAULT_PRIORITY


def test_queueing_brings_a_scheduled_rail_forward_without_a_second_row(tmp_path, monkeypatch):
    """The rail is already active on its hourly cadence; asking for it now must not fork a row."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    store = RailJobStore(tmp_path / "rail_jobs.db")
    store.activate(LOCAL_HOME_ID, "scrapers", due_at=time.time() + 3600)

    assert request_now("scrapers") is True

    assert len(store.list_jobs()) == 1
    assert store.get(LOCAL_HOME_ID, "scrapers").due_at <= time.time()


def test_a_hosted_producer_queues_against_its_own_validated_subject(tmp_path, monkeypatch):
    """The home id comes from the hosted home's environment, never from the tool's arguments."""
    monkeypatch.setenv(WORKER_HOME_ID_ENV, "104729183746152938471")
    monkeypatch.setenv(WORKER_DB_ENV, str(tmp_path / "shared" / "rail_jobs.db"))

    assert request_now("oracle_refresh") is True

    jobs = RailJobStore(tmp_path / "shared" / "rail_jobs.db").list_jobs()
    assert [(j.home_id, j.rail) for j in jobs] == [("104729183746152938471", "oracle_refresh")]


def test_a_misconfigured_deployment_reports_a_failure_instead_of_breaking_the_tool(
        tmp_path, monkeypatch, capsys):
    """`WorkerDbNotConfigured` must reach the log and the response, and never the caller.

    The tool that asked has already written its consent, grant, or claim durably. Raising here
    would turn an operator's missing setting into a failed share; returning False lets the
    response say the pass is not queued, which is the truth.
    """
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setenv(WORKER_HOME_ID_ENV, "104729183746152938471")
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)

    assert request_now("push_catchup") is False

    assert WORKER_DB_ENV in capsys.readouterr().err
    assert not (tmp_path / "rail_jobs.db").exists(), "a hosted job must not land in a home"


# ── the worker heartbeat: one row that answers "is anything claiming?" ──────────

def test_the_heartbeat_answers_liveness_and_counts_restart_churn(tmp_path, monkeypatch):
    import sqlite3

    from pipeline.worker.job_queue import BEAT_STALE_AFTER, worker_is_alive

    db = tmp_path / "jobs.db"
    monkeypatch.setenv("OPYT_WORKER_DB", str(db))
    monkeypatch.delenv("OPYT_WORKER_HOME_ID", raising=False)

    assert worker_is_alive() is False        # no store at all: absent, not unknown
    store = RailJobStore(db)
    assert worker_is_alive() is False        # a store no worker has ever touched

    store.beat("local", "host:1:aaaa", started_at=100.0, now=100.0)
    assert worker_is_alive(now=100.0 + BEAT_STALE_AFTER) is True
    assert worker_is_alive(now=101.0 + BEAT_STALE_AFTER) is False  # stale is a real "down"

    # The same lifetime re-beating is not churn; a NEW worker_id is a restart and increments
    # `lifetimes` with a fresh birth — NRestarts, readable from the store instead of SSH.
    store.beat("local", "host:1:aaaa", started_at=100.0, now=130.0)
    store.beat("local", "host:2:bbbb", started_at=500.0, now=500.0)
    row = sqlite3.connect(str(db)).execute(
        "SELECT worker_id, started_at, beat_at, lifetimes FROM rail_worker_beat").fetchone()
    assert row == ("host:2:bbbb", 500.0, 500.0, 2)


def test_an_unreadable_store_answers_none_not_false(tmp_path, monkeypatch):
    """None is "nothing is known either way"; callers treat it differently from a measured
    stale (onboard keeps RESIDENT on None, drops it on False)."""
    db = tmp_path / "jobs.db"
    monkeypatch.setenv("OPYT_WORKER_DB", str(db))
    monkeypatch.delenv("OPYT_WORKER_HOME_ID", raising=False)
    RailJobStore(db)
    db.chmod(0o000)
    try:
        from pipeline.worker.job_queue import worker_is_alive
        assert worker_is_alive() is None
    finally:
        db.chmod(0o644)


# ── a request made while the job runs ─────────────────────
def test_a_claim_pushes_the_row_out_so_a_request_during_the_run_can_show(tmp_path):
    """A due row's `due_at` is in the past and `activate` keeps the MIN, so a request made while
    the job ran would leave the row exactly as it was. The claim moves it out first."""
    from pipeline.worker.job_queue import CLAIM_HOLD_SECONDS

    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "push_catchup", due_at=10)

    claimed = store.claim_next(now=100)

    assert claimed.due_at == 100 + CLAIM_HOLD_SECONDS
    assert store.get("42", "push_catchup").due_at == claimed.due_at
    store.activate("42", "push_catchup", due_at=150)
    assert store.get("42", "push_catchup").due_at == 150    # the request shows


def test_finish_keeps_a_request_made_during_the_run(tmp_path):
    """Confirming an Oracle while the job runs must not be erased when it finishes."""
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "push_catchup", due_at=10)
    claimed = store.claim_next(now=100)
    store.activate("42", "push_catchup", due_at=150)

    store.finish(claimed, 0, cadence=3600.0, now=200)
    finished = store.get("42", "push_catchup")

    assert finished.due_at == 150 and finished.exit_code == 0


def test_finish_without_a_request_schedules_the_cadence(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "push_catchup", due_at=10)
    claimed = store.claim_next(now=100)

    store.finish(claimed, 0, cadence=3600.0, now=200)
    assert store.get("42", "push_catchup").due_at == 3800


def test_a_wake_the_job_wrote_for_itself_is_kept_but_never_past_the_cadence(tmp_path):
    """The scrapers' process writes its own next start. Earlier than the cadence it stands; a
    later one is capped, so the cadence stays the safety net."""
    from pipeline.worker.registry import SCRAPERS

    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", SCRAPERS, due_at=10)
    claimed = store.claim_next(now=100)
    assert store.rearm("42", SCRAPERS, 500, expected=claimed.due_at)
    store.finish(claimed, 0, cadence=900.0, now=200)
    assert store.get("42", SCRAPERS).due_at == 500

    claimed = store.claim_next(now=600)
    assert store.rearm("42", SCRAPERS, 99_999, expected=claimed.due_at)
    store.finish(claimed, 0, cadence=900.0, now=700)
    assert store.get("42", SCRAPERS).due_at == 1600


def test_rearm_writes_only_over_the_value_the_caller_read(tmp_path):
    """A compare-and-set: a request that landed after the read must not be overwritten."""
    from pipeline.worker.registry import SCRAPERS

    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", SCRAPERS, due_at=10)
    store.claim_next(now=100)
    read = store.get("42", SCRAPERS).due_at
    store.activate("42", SCRAPERS, due_at=120)              # the request, after the read

    assert store.rearm("42", SCRAPERS, 1000, expected=read) is False
    assert store.get("42", SCRAPERS).due_at == 120
    assert store.rearm("42", SCRAPERS, 1000, expected=120) is True
    assert store.get("42", SCRAPERS).due_at == 1000


# ── a job that runs beside the others ───────────────────────────────────────────
def test_a_job_beside_the_others_and_one_other_job_run_together_for_a_home(tmp_path):
    from pipeline.worker.registry import LINE, SCRAPERS

    store = RailJobStore(tmp_path / "jobs.db")
    for rail in (LINE, SCRAPERS, "push_catchup"):
        store.activate("42", rail, due_at=10)

    claims = [store.claim_next(now=10, beside=(LINE,)) for _ in range(3)]

    assert sorted(job.rail for job in claims if job) == sorted([LINE, "push_catchup"])
    assert claims[2] is None                    # still one job besides the line


@pytest.mark.parametrize("running, due", [("scrapers", "line"), ("line", "scrapers")])
def test_neither_the_line_nor_the_scrapers_holds_the_other_back(tmp_path, running, due):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", running, due_at=10)
    assert store.claim_next(now=10, beside=("line",)).rail == running
    store.activate("42", due, due_at=20)

    assert store.claim_next(now=20, beside=("line",)).rail == due


# ── a job a home is given once ──────────────────────────────────────────────────
def test_enrolling_creates_a_due_job_and_never_moves_one_that_exists(tmp_path):
    """What recurs (a publishing pass) enrolls every time. A repeat must leave the schedule
    alone, idle or running: `activate` would bring the job forward each time."""
    store = RailJobStore(tmp_path / "jobs.db")

    store.enroll("42", "line", due_at=100)
    assert store.get("42", "line").due_at == 100

    claimed = store.claim_next(now=100)
    store.enroll("42", "line", due_at=150)
    assert store.get("42", "line").due_at == claimed.due_at     # running: not asked for again
    store.finish(claimed, 0, cadence=3600.0, now=200)
    store.enroll("42", "line", due_at=250)
    assert store.get("42", "line").due_at == 3800               # idle: not brought forward
    assert len(store.list_jobs()) == 1


def test_a_producer_enrolls_its_own_home(tmp_path, monkeypatch):
    from pipeline.worker.job_queue import enroll

    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    before = time.time()

    assert enroll("line") is True
    first = RailJobStore(tmp_path / "rail_jobs.db").get(LOCAL_HOME_ID, "line")
    assert before <= first.due_at <= time.time()
    assert enroll("line") is True
    assert RailJobStore(tmp_path / "rail_jobs.db").get(LOCAL_HOME_ID, "line") == first


def test_a_failed_enrollment_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    """As `request_now`: the publishing pass that enrolls must not fail on the queue."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.setenv(WORKER_HOME_ID_ENV, "104729183746152938471")
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    from pipeline.worker.job_queue import enroll

    assert enroll("line") is False
    assert WORKER_DB_ENV in capsys.readouterr().err


# ── jobs the worker knows ───────────────────────────────────────────────────────
def test_a_claim_limited_to_known_jobs_never_takes_another(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "oracle_refresh", due_at=5, priority=URGENT_PRIORITY)
    store.activate("42", "push_catchup", due_at=10)

    assert store.claim_next(now=10, rails=("push_catchup",)).rail == "push_catchup"
    assert store.claim_next(now=10, rails=("push_catchup",)) is None
    assert store.next_due_at(rails=("push_catchup",)) is None
    assert store.claim_next(now=10, rails=()) is None


def test_recovery_deletes_every_row_of_a_job_the_worker_does_not_know(tmp_path):
    """Even a claimed one: left claimed, it would block every other job for its home."""
    store = RailJobStore(tmp_path / "jobs.db")
    store.activate("42", "oracle_refresh", due_at=5)
    store.claim_next(now=10)
    store.activate("42", "candidate_probe", due_at=5)
    store.activate("77", "push_catchup", due_at=5)

    store.drop_retired_jobs(rails=("push_catchup",))
    store.recover_interrupted_claims(now=20)

    assert [(j.home_id, j.rail) for j in store.list_jobs()] == [("77", "push_catchup")]
