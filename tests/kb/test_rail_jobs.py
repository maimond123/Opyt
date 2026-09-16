from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pipeline.kb import rail_jobs
from pipeline.kb.rail_jobs import (DEFAULT_PRIORITY, LOCAL_HOME_ID, URGENT_PRIORITY,
                                   WORKER_DB_ENV, WORKER_HOME_ID_ENV, RailJobStore,
                                   request_now)
from pipeline.kb.rail_worker import WorkerLock


def test_activation_is_unique_and_only_brings_a_schedule_forward(tmp_path):
    store = RailJobStore(tmp_path / "jobs.db")

    store.activate("42", "oracle_refresh", due_at=200, priority=0)
    urgent = store.activate(
        "42", "oracle_refresh", due_at=100, priority=URGENT_PRIORITY
    )
    unchanged = store.activate("42", "oracle_refresh", due_at=300, priority=0)

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
        assert store.recover_interrupted_claims(now=20) == 1
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
    store.activate(LOCAL_HOME_ID, "sitting_scheduler", due_at=time.time() + 3600)

    assert request_now("sitting_scheduler") is True

    assert len(store.list_jobs()) == 1
    assert store.get(LOCAL_HOME_ID, "sitting_scheduler").due_at <= time.time()


def test_a_producer_that_is_doing_the_pass_itself_schedules_the_repeat_instead(tmp_path,
                                                                               monkeypatch):
    """`request_in` is the other sentence a producer can say — "this work has a repeat" rather
    than "there is work waiting". `sitting_tools._watchlist` needs it because it runs the pull
    in-process: a due-now row would have the worker's child walk the same pairs concurrently,
    and `frontier_execute` has no single-flight lock."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    before = time.time()

    assert rail_jobs.request_in("frontier_execute", 3600.0) is True

    job = RailJobStore(tmp_path / "rail_jobs.db").get(LOCAL_HOME_ID, "frontier_execute")
    assert before + 3600.0 <= job.due_at <= time.time() + 3600.0
    assert job.started_at is None and job.priority == DEFAULT_PRIORITY


def test_scheduling_a_repeat_never_postpones_a_rail_that_is_already_due(tmp_path, monkeypatch):
    """`activate` keeps `MIN(due_at)`. Without that this door could silently push real waiting
    work an hour out, which is the one thing a "just make sure it is scheduled" call must not do."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)
    store = RailJobStore(tmp_path / "rail_jobs.db")
    store.activate(LOCAL_HOME_ID, "frontier_execute", due_at=time.time())

    assert rail_jobs.request_in("frontier_execute", 3600.0) is True

    assert len(store.list_jobs()) == 1
    assert store.get(LOCAL_HOME_ID, "frontier_execute").due_at <= time.time()


def test_a_hosted_producer_queues_against_its_own_validated_subject(tmp_path, monkeypatch):
    """The home id comes from the gateway's environment, never from the tool's arguments."""
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
