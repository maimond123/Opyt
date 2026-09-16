"""
tests/gateway/test_worker_jobs.py — the seam between a disposable child and the resident worker.

The whole hosted integration is two environment variables plus the guarantee that both sides
spell them the same way, so these tests hold the two ends together: `gateway/children.py`
writes `OPYT_WORKER_HOME_ID`, and `pipeline/kb/rail_jobs.py` reads it. Nothing else connects
them, which is the point — the gateway never opens the control database, and the worker never
asks the gateway anything.

Where a test needs a product action, it runs a small Python child under the pool's own
environment rather than `opyt-mcp`. No MCP tool queues a job yet; that is stage 5 of
docs/plans/2026-09-06-persistent-rail-worker-migration.md. The environment dictionary, the
store, the reaper, and the worker are all the real ones.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import pipeline.kb.rail_worker as rail_worker
from gateway import children
from gateway.children import CHILD_LOG, ChildPool
from pipeline.kb.rail_jobs import (LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV,
                                   RailJobStore, WorkerDbNotConfigured, current_home_id,
                                   worker_db_path)
from pipeline.kb.rail_worker import HOURLY, RailSpec, RailWorker

SUBJECT = "104729183746152938471"

# Stands in for a stage-5 product action: resolve this process's own home id and the shared
# control database from the environment alone, then queue one durable job.
_QUEUE_ONE_JOB = (
    "from pipeline.kb.rail_jobs import RailJobStore, current_home_id, worker_db_path\n"
    "RailJobStore(worker_db_path()).activate(current_home_id(), 'oracle_refresh', due_at=0)\n"
)


def _queue_from_child(env: dict[str, str]) -> None:
    """Run the queue program in a real child process carrying exactly the gateway's env."""
    subprocess.run([sys.executable, "-c", _QUEUE_ONE_JOB], env=env,
                   cwd=str(children.REPO_ROOT), check=True, capture_output=True)


def _hosted_env(monkeypatch, homes_root: Path, worker_db: Path,
                subject: str = SUBJECT) -> dict[str, str]:
    """Exactly the environment the gateway hands a child, with the worker database configured."""
    monkeypatch.setenv(WORKER_DB_ENV, str(worker_db))
    return children._child_env(children.home_for(homes_root, subject), subject)


# ── The two variables, and that both sides agree on them ────────────────────────────────────

def test_a_child_is_told_its_own_subject_and_the_shared_control_database(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKER_DB_ENV, "/var/lib/opyt-worker/rail_jobs.db")

    env = children._child_env(tmp_path / "homes" / SUBJECT, SUBJECT)

    assert env[WORKER_HOME_ID_ENV] == SUBJECT
    assert env[WORKER_DB_ENV] == "/var/lib/opyt-worker/rail_jobs.db"
    assert env["OPYT_HOME"] == str(tmp_path / "homes" / SUBJECT)


def test_both_sides_spell_the_worker_variables_the_same_way(monkeypatch):
    """`_GATEWAY_ONLY` is where a future secret gets stripped, and these two must never join it.

    Stripping the control database would send every hosted queue request into a per-home file
    the worker never opens; stripping the home id would leave a child unable to name itself.
    """
    assert WORKER_DB_ENV not in children._GATEWAY_ONLY
    assert WORKER_HOME_ID_ENV not in children._GATEWAY_ONLY

    monkeypatch.setenv(WORKER_HOME_ID_ENV, SUBJECT)
    assert current_home_id() == SUBJECT, "children.py and rail_jobs.py disagree on the name"


def test_a_hosted_process_refuses_to_queue_into_a_per_home_database(monkeypatch, tmp_path):
    """The one deployment mistake this seam can make: the worker unit sets the shared database
    and the gateway unit does not. Silently, every hosted job would land in `$OPYT_HOME`."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "homes" / SUBJECT))
    monkeypatch.setenv(WORKER_HOME_ID_ENV, SUBJECT)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)

    with pytest.raises(WorkerDbNotConfigured):
        worker_db_path()


def test_a_local_process_has_the_one_fixed_home_id_and_its_own_database(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    monkeypatch.delenv(WORKER_HOME_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_DB_ENV, raising=False)

    assert current_home_id() == LOCAL_HOME_ID
    assert worker_db_path() == tmp_path / "rail_jobs.db"


@pytest.mark.parametrize("home_id", ["", "..", "/srv/homes/42", "42 77"])
def test_a_hand_written_home_id_is_rejected_at_the_read(monkeypatch, home_id):
    """The hosted producer is the gateway, which validated. The local producer is an operator's
    service file, which did not — so this read is a second boundary, not a repeated check."""
    monkeypatch.setenv(WORKER_HOME_ID_ENV, home_id)

    with pytest.raises(ValueError):
        current_home_id()


# ── A child produces durable work; the worker consumes it ───────────────────────────────────

def test_a_child_queues_work_only_for_the_home_the_gateway_named(tmp_path, monkeypatch):
    homes_root, worker_db = tmp_path / "homes", tmp_path / "worker" / "rail_jobs.db"
    store = RailJobStore(worker_db)

    _queue_from_child(_hosted_env(monkeypatch, homes_root, worker_db))
    _queue_from_child(_hosted_env(monkeypatch, homes_root, worker_db, subject="77"))

    assert {(job.home_id, job.rail) for job in store.list_jobs()} == {
        (SUBJECT, "oracle_refresh"), ("77", "oracle_refresh")}


@pytest.mark.anyio
async def test_a_reaped_child_leaves_its_scheduled_work_intact(tmp_path, monkeypatch):
    """A real `opyt-mcp` child, really SIGTERMed. Its home is durable and so is its queue.

    This is the property that makes the split worth its second process: the reaper may end a
    child at any idle moment, and the work that child asked for still runs later.
    """
    worker_db = tmp_path / "worker" / "rail_jobs.db"
    pool = ChildPool(tmp_path / "homes", idle_seconds=0, spawn_timeout=60)
    store = RailJobStore(worker_db)
    _queue_from_child(_hosted_env(monkeypatch, pool.homes_root, worker_db))
    queued = store.get(SUBJECT, "oracle_refresh")

    try:
        child = await pool.acquire(SUBJECT)
        pool.release(child)
        assert await pool.reap_once() == [SUBJECT]
        assert not child.alive
    finally:
        await pool.shutdown()

    assert store.get(SUBJECT, "oracle_refresh") == queued
    claimed = store.claim_next(now=queued.due_at)
    assert (claimed.home_id, claimed.rail) == (SUBJECT, "oracle_refresh")


def test_the_worker_runs_a_job_for_a_user_with_no_live_child(tmp_path, monkeypatch):
    """The worker reads the control database and the homes root, and nothing else.

    An inactive user is the normal case for unattended ingest — nobody is connected when the
    hourly pass is due — so the worker must never need the gateway to have spawned anything.
    """
    homes_root = tmp_path / "homes"
    store = RailJobStore(tmp_path / "worker" / "rail_jobs.db")
    monkeypatch.setattr(rail_worker, "RAILS", {"oracle_refresh": RailSpec(
        ("-c", "import os; from pathlib import Path; "
               "(Path(os.environ['OPYT_HOME'])/'rail-ran').write_text('1')"),
        "oracle_refresh.log", HOURLY)})
    store.activate(SUBJECT, "oracle_refresh", due_at=0)

    worker = RailWorker(store, homes_root=homes_root)
    assert worker.launch_available() == 1
    for child in worker.active:
        child.process.wait(timeout=30)
    worker.reap_finished()

    assert store.get(SUBJECT, "oracle_refresh").exit_code == 0
    assert (homes_root / SUBJECT / "rail-ran").exists()
    assert not (homes_root / SUBJECT / CHILD_LOG).exists(), "the worker spawned an MCP child"


# ── the deployed unit must give rail children the hosted environment ─────────────────────────


def _unit_env(name: str) -> dict[str, str]:
    """The `Environment=` assignments in a tracked systemd unit."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "gateway" / "deploy" / name).read_text()
    out = {}
    for line in text.splitlines():
        if line.startswith("Environment="):
            key, _, value = line[len("Environment="):].partition("=")
            out[key] = value
    return out


def _unit_directives(name: str) -> dict[str, str]:
    """The non-`Environment=` directives in a tracked systemd unit."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "gateway" / "deploy" / name).read_text()
    out = {}
    for line in text.splitlines():
        if line.startswith(("#", "[", "Environment=")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key] = value
    return out


@pytest.mark.parametrize("unit", ["opyt-gateway.service", "opyt-worker.service"])
def test_a_memory_ceiling_never_ships_without_an_oom_policy(unit):
    """`MemoryMax=` without `OOMPolicy=` is worse than no ceiling at all, on both units.

    systemd defaults OOMPolicy to `stop`. Both units also set KillMode=control-group, so the
    default turns one OOM-killed process into the loss of every process in the unit: on the
    gateway, every OTHER user's MCP child; on the worker, all rail children. Adding MemoryMax is
    what creates that trigger, at 20G/8G rather than the box's 31 GB, so the ceiling and the
    policy are one change and must never be separated.

    The value is deliberately not asserted — tuning 20G is expected, shipping it bare is not.
    """
    directives = _unit_directives(unit)
    if "MemoryMax" not in directives:
        return
    assert directives.get("OOMPolicy") == "continue", (
        f"{unit} sets MemoryMax without OOMPolicy=continue"
    )


def test_the_signin_cap_ships_with_the_code_that_reads_it():
    """An `Environment=` line no process reads is a dead reference, and this key has already
    been one: ed49516b added it, 73646900 removed the reader, and it came out with it. The pair
    is what makes the 503 refusal page reachable, so pin them together rather than the number.
    """
    from gateway.children import DEFAULT_MAX_SIGNINS
    value = _unit_env("opyt-gateway.service").get("OPYT_GATEWAY_MAX_SIGNINS")
    assert value is not None, "the gateway unit sets no sign-in cap"
    assert int(value) > 0
    assert DEFAULT_MAX_SIGNINS > 0


def test_the_worker_unit_marks_rail_children_as_hosted():
    """A rail child is a hosted process, and only this unit can tell it so.

    `RailWorker._launch` builds a child's environment from `dict(os.environ)`, so the flag has to
    reach the WORKER for a rail to inherit it. `gateway/children.py` sets OPYT_HOSTED_X for MCP
    children, but rail children are a different process tree with a different parent.

    Measured 2026-09-07, which is why this test exists: without it `hosted_x.enabled()` was False
    inside every rail, the X code fell through to the local browser-cookie reader, found no
    browser on a server, and `bookmark_catchup` failed twice with "No OPYT-managed X session
    found" against a home whose X session was live and serving other reads.
    """
    assert _unit_env("opyt-worker.service").get("OPYT_HOSTED_X") == "1"


def test_both_units_agree_on_the_control_database():
    """The worker reads this database and a hosted child writes it, so a mismatch is a silent
    no-op: work queued where nothing looks. DEPLOY.md calls the shared value load-bearing; this
    is the check that makes disagreeing impossible rather than merely discouraged."""
    worker = _unit_env("opyt-worker.service")["OPYT_WORKER_DB"]
    gateway = _unit_env("opyt-gateway.service")["OPYT_WORKER_DB"]
    assert worker == gateway
