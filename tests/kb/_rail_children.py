"""Jobs for the worker tests to launch. A worker child runs a job's `run_*` by `module:function`
through `pipeline.worker.child`, in a fresh process started from the repo root, so a test's
stand-in job has to be importable from there: these are. Each reads its home from `$OPYT_HOME`,
as a real job does."""
from __future__ import annotations

from pipeline.worker.home import current_home_id
from pipeline.worker.quota_loop.retries import Outages, PLATFORM_PAUSES_S
from pipeline.worker.quota_loop.serial import Serial
from pipeline.kb import schema

import os
import sys
import time
from pathlib import Path


def _home() -> Path:
    return Path(os.environ["OPYT_HOME"])


def noop() -> dict:
    return {"status": "ok"}


def reports_its_home() -> dict:
    print(os.environ["OPYT_HOME"], flush=True)
    print("rail stderr", file=sys.stderr, flush=True)
    raise SystemExit(7)


def fails() -> dict:
    print("failed pass", flush=True)
    raise SystemExit(9)


def passes() -> dict:
    print("later pass", flush=True)
    return {"status": "ok"}


def sleeps() -> dict:
    print("started", flush=True)
    time.sleep(60)
    return {"status": "ok"}


def marks_its_home() -> dict:
    (_home() / "selected-home").write_text(_home().name)
    return {"status": "ok"}


def marks_it_ran() -> dict:
    (_home() / "rail-ran").write_text("1")
    return {"status": "ok"}


def queues_its_successor() -> dict:
    from pipeline.worker.job_queue import request_now
    return {"status": "ok" if request_now("successor") else "error"}


def unknown_status() -> dict:
    return {"status": "something_new"}


def holds_a_lease() -> dict:
    """A rail pass under a lease. The first run takes the lease, says so and stays in its pass
    until it is stopped; any later run takes the lease and finishes at once. Every run appends
    what it got to `runs`, so a test can read back whether a successor found the lease free."""
    from pipeline.sync_lock import CatchupLock

    runs = _home() / "runs"
    with CatchupLock("test-lease") as lock:
        first = not runs.exists()
        with runs.open("a") as f:
            f.write(("held" if lock.acquired else "refused") + "\n")
        if not lock.acquired:
            return {"status": "already_running"}
        if first:
            time.sleep(60)
    return {"status": "ok"}


class IdleSource:
    """A source with nothing to do, on a quota of its own."""
    name, quota, klass, floor = "idle", "test:quota", 0, 0

    def next_unit(self, conn, now):
        return None

    def fetch(self, conn, unit, ctx):
        return None

    def record(self, conn, unit, fetched):
        return None

    def next_due(self, conn, now):
        return None


def runs_an_idle_scraper() -> dict:
    """The real runner around one real scraper, which holds the `test:quota` lease while it has
    nothing to do. It never goes idle long enough to exit, so only a stop ends it."""
    from pipeline.worker.quota_loop.process import LoopProcess
    from pipeline.worker.quota_loop.loop import QuotaLoop
    from pipeline.worker.job_queue import RailJobStore

    runner = LoopProcess([QuotaLoop("test:quota", [IdleSource()], executor=Serial(), outages=Outages(PLATFORM_PAUSES_S), connect=schema.connect)], store=RailJobStore(),
                    home_id=current_home_id(), seed=lambda: None, idle_exit_s=3600.0)
    return runner.run()


def outlives_a_sigterm() -> dict:
    """Keeps running through the first SIGTERM (the child turns it into a `SystemExit`), as a
    process still finishing a long unit might, so a restart's grace runs out on it."""
    try:
        time.sleep(60)
    except SystemExit:
        time.sleep(60)
    return {"status": "ok"}


def naps() -> dict:
    time.sleep(1.0)
    return {"status": "ok"}


def holds_the_line_through_a_slow_call() -> dict:
    """The real `line` job against a stand-in service that hands over one `watch` call, whose
    body takes two seconds. `call-began` is written when the body starts, `call-ended` when it
    returns and `call-answered` if its answer is posted, so a test can stop the job mid-call and
    read back what became of the call."""
    from mcp_server import line
    from pipeline.sharing import client
    from pipeline.topics import create

    handed = []

    def line_next(acct, wait):
        if handed:
            time.sleep(60)
        handed.append(True)
        return {"call": {"id": "c1", "tool": "watch", "args": {"action": "list"}}}

    def slow_listing(conn):
        (_home() / "call-began").write_text("1")
        time.sleep(2.0)
        (_home() / "call-ended").write_text("1")
        return {"status": "ok", "topics": []}

    def line_answer(acct, call_id, ok, answer):
        (_home() / "call-answered").write_text("1")
        return {"recorded": True}

    client.line_next, client.line_answer, create.listing = line_next, line_answer, slow_listing
    return line.run()

