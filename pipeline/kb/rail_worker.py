"""Resident scheduler and process supervisor for OPYT's nine bounded rails."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from opyt_core.paths import opyt_home
from pipeline.kb.rail_jobs import (LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV,
                                   RailJob, RailJobStore, worker_db_path)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOURLY = 3600.0
SIX_HOURLY = 6 * HOURLY


@dataclass(frozen=True)
class RailSpec:
    command: tuple[str, ...]
    log_name: str
    cadence: float


RAILS: Mapping[str, RailSpec] = MappingProxyType({
    "oracle_refresh": RailSpec(
        ("-c", "from pipeline.kb.oracle_refresh import _run; _run()"),
        "oracle_refresh.log", 600.0,
    ),
    "frontier_execute": RailSpec(
        ("-m", "pipeline.kb.frontier_execute", "--once"), "frontier_exec.log", HOURLY,
    ),
    "frontier_admit": RailSpec(
        ("-m", "pipeline.kb.frontier_admit", "--once"), "frontier_admit.log", HOURLY,
    ),
    "bookmark_catchup": RailSpec(
        ("-m", "pipeline.kb.bookmark_catchup", "--once"), "bookmark_catchup.log", HOURLY,
    ),
    "curation_catchup": RailSpec(
        ("-m", "pipeline.kb.curation_catchup", "--once"), "curation_catchup.log", HOURLY,
    ),
    # Six-hourly, and the odd one out on purpose. Its steady-state pass is a request to
    # Substack's Cloudflare-guarded reader endpoint plus a re-fetch of every body-blocked stub;
    # `pipeline/kb/substack_saved_catchup.py`'s header has the full reasoning. `request_now` sets
    # `due_at` to now regardless, so consent still imports immediately.
    "substack_saved_catchup": RailSpec(
        ("-m", "pipeline.kb.substack_saved_catchup", "--once"),
        "substack_saved_catchup.log", SIX_HOURLY,
    ),
    "candidate_probe": RailSpec(
        ("-m", "pipeline.kb.probe_catchup", "--once"), "candidate_probe.log", HOURLY,
    ),
    "push_catchup": RailSpec(
        ("-m", "pipeline.kb.push_catchup", "--once"), "push_catchup.log", HOURLY,
    ),
    "sitting_scheduler": RailSpec(
        ("-m", "pipeline.kb.sitting_scheduler", "--once"), "sitting_scheduler.log", HOURLY,
    ),
})


class WorkerAlreadyRunning(RuntimeError):
    pass


class WorkerLock:
    """A kernel-released lifetime lock beside the worker database."""

    def __init__(self, db_path: Path | str) -> None:
        path = Path(db_path)
        self.path = path.with_name(f"{path.name}.lock")
        self._file = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.path, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        self._file = lock_file
        return True

    def release(self) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None

    def __enter__(self) -> "WorkerLock":
        if not self.acquire():
            raise WorkerAlreadyRunning(f"another opyt-worker holds {self.path}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


@dataclass
class ActiveChild:
    job: RailJob
    spec: RailSpec
    process: subprocess.Popen


class RailWorker:
    """Schedule rail children globally while keeping each home single-flight."""

    def __init__(self, store: RailJobStore, *, homes_root: Path | str | None = None,
                 max_children: int = 1, poll_interval: float = 1.0) -> None:
        if max_children < 1:
            raise ValueError("max_children must be at least one")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be greater than zero")
        self.store = store
        self.homes_root = None if homes_root is None else Path(homes_root)
        self.local_home = opyt_home()
        self.max_children = max_children
        self.poll_interval = poll_interval
        self.active: list[ActiveChild] = []

    @property
    def local_mode(self) -> bool:
        return self.homes_root is None

    def home_for(self, home_id: str) -> Path:
        if self.local_mode:
            if home_id != LOCAL_HOME_ID:
                raise ValueError(f"local worker cannot run hosted home {home_id!r}")
            return self.local_home
        # RailJobStore validates home_id before insertion, so path selection trusts that boundary.
        return self.homes_root / home_id

    def _claim_next(self) -> RailJob | None:
        return self.store.claim_next(home_id=LOCAL_HOME_ID if self.local_mode else None)

    def _launch(self, job: RailJob) -> ActiveChild:
        spec = RAILS[job.rail]
        home = self.home_for(job.home_id)
        home.mkdir(parents=True, exist_ok=True)
        log_file = open(home / spec.log_name, "a")
        env = dict(os.environ)
        env["OPYT_HOME"] = str(home)
        # A rail child is also a PRODUCER: a pass that stages candidates, mints candidates, or
        # writes standing queries makes its successor rail due (`rail_jobs.request_now`). These
        # two are what let it do that for its OWN home. Without the home id a hosted child falls
        # through to `LOCAL_HOME_ID` and queues against the wrong home; without the database path
        # a worker started with `--db` would hand its children a store it never opened, which
        # `worker_db_path()` raises on rather than let become a silent no-op.
        env[WORKER_HOME_ID_ENV] = job.home_id
        env[WORKER_DB_ENV] = str(self.store.db_path)
        try:
            # Deliberately no new session: the service control group must terminate rail children
            # with the worker before the next lifetime recovers their claims.
            process = subprocess.Popen(
                [sys.executable, *spec.command],
                cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL,
                stdout=log_file, stderr=log_file,
            )
        finally:
            log_file.close()
        return ActiveChild(job=job, spec=spec, process=process)

    def launch_available(self) -> int:
        """Fill currently free global slots with due jobs."""
        launched = 0
        while len(self.active) < self.max_children:
            job = self._claim_next()
            if job is None:
                break
            child = self._launch(job)
            self.active.append(child)
            launched += 1
        return launched

    def reap_finished(self) -> int:
        """Record every child that has exited, regardless of its result."""
        reaped = 0
        still_running: list[ActiveChild] = []
        for child in self.active:
            exit_code = child.process.poll()
            if exit_code is None:
                still_running.append(child)
                continue
            self.store.finish(child.job, exit_code, cadence=child.spec.cadence)
            reaped += 1
        self.active = still_running
        return reaped

    def stop_children(self, *, grace: float = 30.0) -> None:
        """Terminate every active rail child and wait, WITHOUT recording their exits.

        Deliberately no `store.finish`: a row left with `started_at` set and `finished_at`
        NULL is exactly what the next lifetime's `recover_interrupted_claims` re-queues with
        `due_at` clamped to now. Reaping here instead would call `finish(exit_code=-15)` and
        reschedule the killed rail a whole cadence out — an hour to six of silent delay for
        work the worker itself interrupted.
        """
        for child in self.active:
            if child.process.poll() is None:
                child.process.terminate()
        deadline = time.monotonic() + grace
        for child in self.active:
            remaining = deadline - time.monotonic()
            try:
                child.process.wait(timeout=max(remaining, 0.1))
            except subprocess.TimeoutExpired:
                child.process.kill()
                child.process.wait()
        self.active = []

    def run_forever(self, *, max_lifetime: float | None = None) -> None:
        """Hold the worker lifetime lock and supervise rails until interrupted.

        `max_lifetime` bounds the lifetime in seconds: past it the worker drains its
        children (`stop_children`) and exits 0. This is how a PUBLISHED fix reaches the
        rails on a machine nobody reboots. The launcher is `uvx --from opyt@latest
        opyt-worker`, and uvx resolves `@latest` only at process start — a resident
        worker that never exits runs the build it was born with forever. The LaunchAgent's
        unconditional `KeepAlive` (opyt_core/install_worker.py) restarts a clean exit
        immediately (`ThrottleInterval` is a floor between STARTS, long since satisfied),
        and that relaunch re-resolves `@latest`. The bound lives HERE and not in the plist
        so that shipping it — and ever changing it — needs a publish, not a re-install,
        which is the exact property being built.

        The drain is not optional politeness. `_launch` spawns children with no new
        session, relying on whoever stops the worker to stop its process group — true for
        launchd and systemd stops, false for a VOLUNTARY exit, which would orphan the
        children. The next lifetime's `recover_interrupted_claims` would then clear
        `started_at` on the rows those orphans still hold and relaunch each rail beside
        its own orphan, two processes on one home.
        """
        with WorkerLock(self.store.db_path):
            self.store.recover_interrupted_claims()
            deadline = (None if max_lifetime is None
                        else time.monotonic() + max_lifetime)
            while True:
                self.reap_finished()
                if deadline is not None and time.monotonic() >= deadline:
                    self.stop_children()
                    return
                self.launch_available()
                due_at = self.store.next_due_at(
                    home_id=LOCAL_HOME_ID if self.local_mode else None
                )
                now = time.time()
                delay = self.poll_interval
                if due_at is not None and due_at > now:
                    delay = min(delay, due_at - now)
                time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OPYT's resident rail worker")
    parser.add_argument("--db", type=Path, default=None,
                        help="worker control database (default: $OPYT_WORKER_DB or local home)")
    parser.add_argument("--homes-root", type=Path, default=None,
                        help="hosted per-user homes root (default: $OPYT_HOMES_ROOT)")
    parser.add_argument("--max-children", type=int,
                        default=int(os.environ.get("OPYT_WORKER_MAX_CHILDREN", "1")))
    parser.add_argument("--poll-seconds", type=float,
                        default=float(os.environ.get("OPYT_WORKER_POLL_SECONDS", "1")))
    parser.add_argument("--max-lifetime-seconds", type=float, default=None,
                        help="drain children and exit 0 after this long, so the KeepAlive "
                             "relaunch re-resolves opyt@latest (default: 6h local, "
                             "unbounded hosted; 0 disables; $OPYT_WORKER_MAX_LIFETIME)")
    args = parser.parse_args(argv)

    homes_root = args.homes_root
    if homes_root is None and os.environ.get("OPYT_HOMES_ROOT"):
        homes_root = Path(os.environ["OPYT_HOMES_ROOT"])

    # Bounded on a LOCAL machine, unbounded hosted. The hosted worker updates by
    # `git pull && systemctl restart` on the box and systemd's KillMode=control-group owns its
    # children, so a lifetime bound buys it nothing; the local LaunchAgent runs a uvx-resolved
    # build that only a restart can refresh. 0 (or a negative) means unbounded, so an operator
    # can switch either default off without touching code.
    max_lifetime = args.max_lifetime_seconds
    if max_lifetime is None and os.environ.get("OPYT_WORKER_MAX_LIFETIME"):
        max_lifetime = float(os.environ["OPYT_WORKER_MAX_LIFETIME"])
    if max_lifetime is None and homes_root is None:
        max_lifetime = 6 * HOURLY
    if max_lifetime is not None and max_lifetime <= 0:
        max_lifetime = None

    store = RailJobStore(args.db or worker_db_path())
    worker = RailWorker(store, homes_root=homes_root, max_children=args.max_children,
                        poll_interval=args.poll_seconds)
    try:
        worker.run_forever(max_lifetime=max_lifetime)
    except WorkerAlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
