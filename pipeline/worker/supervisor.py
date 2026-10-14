"""
pipeline/worker/supervisor.py — the worker process: one lifetime that keeps each home's due jobs
running as children.

The resident scheduler and process supervisor for OPYT's background jobs. Two are registered: the
background scrapers' process (`SCRAPERS`), one per home, and the line to the sharing service
(`LINE`), which runs beside it for as long as the worker does.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from opyt_core.paths import ensure_home, opyt_home
from pipeline.ingestion.utils import log
from pipeline.worker import registry
from pipeline.worker.failure_run import FailureRun
from pipeline.worker.home import LOCAL_HOME_ID, WORKER_DB_ENV, WORKER_HOME_ID_ENV
from pipeline.worker.job_queue import BEAT_SECONDS, SHARED_BEAT_SCOPE, RailJob, RailJobStore

REPO_ROOT = Path(__file__).resolve().parents[2]
# A restart at `max_lifetime` waits this long for the jobs running to finish on their own, then
# stops what is left.
RESTART_GRACE = 10 * 60.0


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


def settings_refusal(max_children: int, poll_interval: float) -> str | None:
    """Why a worker cannot run with these settings, or None when it can. The command line asks
    it (`_arguments`) before `main` creates the control database."""
    if max_children < 1:
        return "max_children must be at least one"
    if poll_interval <= 0:
        return "poll_interval must be greater than zero"
    return None


def launchable(taken: int, max_children: int,
               jobs: Mapping[str, registry.RailSpec]) -> tuple[str, ...]:
    """The jobs a launch may claim now, with `taken` of the worker's `max_children` slots in use.
    A job that runs beside the others (`RailSpec.beside`) takes no slot and needs none: it stays
    up for as long as the worker does, so counting it would leave a local worker's one slot
    taken for good."""
    return tuple(name for name, spec in jobs.items() if spec.beside or taken < max_children)


def drain(now: float, deadline: float | None, draining_since: float | None, running: bool,
          grace: float) -> str:
    """Where a lifetime stands against its bound: `run`, `start` the drain, `wait` for it, or
    `exit`. Past the deadline the worker stops claiming and asks every child to stop, and waits
    up to `grace` for them to exit on their own. Only what is still running after that is
    stopped, and left unrecorded."""
    if deadline is None or now < deadline:
        return "run"
    if not running:
        return "exit"
    if draining_since is None:
        return "start" if grace > 0 else "exit"
    return "exit" if now - draining_since >= grace else "wait"


def _lifetime_env(raw: str | None) -> float | None:
    """A numeric lifetime from the environment, or no usable override."""
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def ignored_lifetime_env(arg: float | None, env: str | None) -> bool:
    """Is `$OPYT_WORKER_MAX_LIFETIME` set to a value `lifetime_bound` cannot use? Only when no
    argument overrides it, so the entry point says once that the default applies."""
    return arg is None and bool(env) and _lifetime_env(env) is None


def lifetime_bound(arg: float | None, env: str | None, *, hosted: bool) -> float | None:
    """How long one worker lifetime may run, in seconds; None is unbounded. The argument wins,
    then `$OPYT_WORKER_MAX_LIFETIME`, then the default.

    Bounded locally, where only a restart refreshes the uvx-resolved build; unbounded hosted,
    where the box restarts it on update. 0 (or a negative) means unbounded."""
    bound = arg
    if bound is None:
        bound = _lifetime_env(env)
    if bound is None and not hosted:
        bound = 6 * registry.HOURLY
    if bound is not None and bound <= 0:
        return None
    return bound


@dataclass
class _Lifetime:
    """What one worker lifetime remembers between passes."""
    deadline: float | None              # monotonic; None is unbounded
    beat_scope: str
    worker_id: str
    born_at: float
    last_beat: float | None = None
    claims_recovered: bool = False
    draining_since: float | None = None   # monotonic; when the drain started, `drain`'s timer
    failures: FailureRun = field(default_factory=FailureRun)
    step: str = "drop_retired_jobs"


class _FailedLaunch:
    """A claimed job with no process; reap it with the same retry as an exited child."""

    def poll(self) -> int:
        return 1

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002 — Popen contract
        return 1


@dataclass
class ActiveChild:
    job: RailJob
    spec: registry.RailSpec
    process: subprocess.Popen | _FailedLaunch


class RailWorker:
    """Schedule rail children globally while keeping each home single-flight, and never more
    than `max_children` at once. A job that runs beside the others (`RailSpec.beside`) is outside
    both rules."""

    def __init__(self, store: RailJobStore, *, homes_root: Path | str | None = None,
                 max_children: int = 1, poll_interval: float = 1.0,
                 restart_grace: float = RESTART_GRACE) -> None:
        self.store = store
        self.homes_root = None if homes_root is None else Path(homes_root)
        self.local_home = opyt_home()
        self.max_children = max_children
        self.poll_interval = poll_interval
        self.restart_grace = restart_grace
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

    def _home_scope(self) -> str | None:
        return LOCAL_HOME_ID if self.local_mode else None

    def _launch(self, job: RailJob) -> ActiveChild:
        spec = registry.RAILS[job.rail]
        home = self.home_for(job.home_id)
        ensure_home(home)
        log_file = open(home / spec.log_name, "a")
        env = dict(os.environ)
        env["OPYT_HOME"] = str(home)
        # A rail child is also a producer (`job_queue.request_now`): these two let it queue
        # against its own home in the store this worker opened.
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

    def launch_available(self) -> None:
        """Fill currently free global slots with due jobs, and launch every due job that needs
        no slot (`launchable`).

        A launch that fails after its claim (`ensure_home`, the log, `Popen`) is finished as a
        child that exited 1, at its base cadence: a claim with no child would hold that home's
        every other job back until the next lifetime."""
        beside = tuple(name for name, spec in registry.RAILS.items() if spec.beside)
        while rails := launchable(sum(not child.spec.beside for child in self.active),
                                  self.max_children, registry.RAILS):
            job = self.store.claim_next(home_id=self._home_scope(), rails=rails, beside=beside)
            if job is None:
                break
            try:
                child = self._launch(job)
            except Exception as exc:
                log(f"[rail-worker] could not launch {job.home_id}/{job.rail}: "
                    f"{type(exc).__name__}: {exc}")
                self.active.append(ActiveChild(job, registry.RAILS[job.rail], _FailedLaunch()))
                self.reap_finished()
                continue
            self.active.append(child)

    def reap_finished(self) -> None:
        """Record every child that has exited, regardless of its result.

        Each child leaves ``active`` only once its `finish` has committed. A `finish` that
        raises (a locked store, say) leaves that child and its successors in place, so the
        retrying loop in `run_forever` records them on a later pass instead of re-finishing
        an already-recorded claim — which `finish` refuses as "no longer active".
        """
        for child in list(self.active):
            exit_code = child.process.poll()
            if exit_code is None:
                continue
            self.store.finish(child.job, exit_code, cadence=child.spec.cadence)
            self.active.remove(child)

    def ask_children_to_stop(self) -> None:
        """Tell every running child to stop now. The scrapers' process finishes the unit in
        flight (seconds on X) and releases its leases; the line stops asking. Each writes "due
        now" as its next wake and exits, and the next reap records it; the restart does not have
        to wait for a process that would run for hours, or for good."""
        for child in self.active:
            if child.process.poll() is None:
                child.process.terminate()

    def stop_children(self, *, grace: float = 30.0) -> None:
        """Terminate every active rail child and wait, without recording their exits, so the
        next lifetime's `recover_interrupted_claims` re-queues them due now rather than a whole
        cadence out.
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

        `max_lifetime` bounds the lifetime in seconds: past it the worker drains its children
        and exits 0, so the KeepAlive relaunch re-resolves `opyt@latest`. The drain is required:
        children share the worker's process group, so a voluntary exit would orphan them. Past
        the deadline the worker claims nothing more and SIGTERMs every child at once; one still
        running after `restart_grace` is stopped.

        A transient store error (`database is locked`) must not end the program: each pass
        retries on `Exception`, logging the onset and the recovery, never spinning hotter than
        `poll_interval`.
        """
        with WorkerLock(self.store.db_path):
            life = _Lifetime(
                deadline=None if max_lifetime is None else time.monotonic() + max_lifetime,
                beat_scope=LOCAL_HOME_ID if self.local_mode else SHARED_BEAT_SCOPE,
                # A fresh id per lifetime lets the liveness row count restart churn.
                worker_id=f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}",
                born_at=time.time())
            while True:
                began = time.monotonic()
                try:
                    leave, due_at = self._pass(life)
                    if leave:
                        return
                except Exception as exc:
                    self._failed(life, began, exc)
                    time.sleep(self.poll_interval)
                    continue
                self._recovered(life)
                time.sleep(self._until_next(due_at, time.time()))

    def _pass(self, life: _Lifetime) -> tuple[bool, float | None]:
        """One pass of a lifetime: `(leave, the next due row)`. `life.step` names the step under
        way, for the log when one raises."""
        if not life.claims_recovered:
            life.step = "drop_retired_jobs"
            self.store.drop_retired_jobs(rails=tuple(registry.RAILS))
            life.step = "recover_interrupted_claims"
            self.store.recover_interrupted_claims()
            life.claims_recovered = True
        life.step = "reap_finished"
        self.reap_finished()
        tick = time.monotonic()
        phase = drain(tick, life.deadline, life.draining_since, bool(self.active),
                      self.restart_grace)
        match phase:
            case "start":
                life.draining_since = tick
                self.ask_children_to_stop()
            case "exit":
                self.stop_children()
                return True, None
        life.step = "beat"
        self._beat(life)
        if phase != "run":
            return False, None
        life.step = "launch_available"
        self.launch_available()
        life.step = "next_due_at"
        return False, self.store.next_due_at(home_id=self._home_scope(),
                                             rails=tuple(registry.RAILS))

    def _beat(self, life: _Lifetime) -> None:
        """Stamp this lifetime's heartbeat, throttled to `BEAT_SECONDS`. Part of the pass, so a
        failed beat is retried like any other step."""
        if life.last_beat is None or time.time() - life.last_beat >= BEAT_SECONDS:
            self.store.beat(life.beat_scope, life.worker_id, started_at=life.born_at)
            life.last_beat = time.time()

    def _failed(self, life: _Lifetime, began: float, exc: Exception) -> None:
        """Log a pass that raised, once per run of failures, with the step and how long it took."""
        if life.failures.failed(time.time()):
            log(f"[rail-worker] {life.step} failed after "
                f"{time.monotonic() - began:.1f}s, retrying — {exc!r}")

    def _recovered(self, life: _Lifetime) -> None:
        if (lasted := life.failures.recovered(time.time())) is not None:
            log(f"[rail-worker] recovered after {lasted:.0f}s")

    def _until_next(self, due_at: float | None, now: float) -> float:
        """How long to sleep: until the next due row, at most `poll_interval`."""
        delay = self.poll_interval
        if due_at is not None and due_at > now:
            delay = min(delay, due_at - now)
        return delay


def main(argv: list[str] | None = None) -> int:
    """`opyt-worker`: read the arguments, run one worker lifetime, map the outcome to an exit."""
    args = _arguments(argv)
    worker = RailWorker(RailJobStore(args.db), homes_root=args.homes_root,
                        max_children=args.max_children, poll_interval=args.poll_seconds)
    try:
        worker.run_forever(max_lifetime=args.max_lifetime)
    except WorkerAlreadyRunning as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    """The command line, with each default read from its environment variable. A setting the
    worker would refuse (`settings_refusal`) is refused here, before `main` creates the control
    database. The lifetime is resolved here (`lifetime_bound`), and an environment value it
    ignores is logged once."""
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
    if (refusal := settings_refusal(args.max_children, args.poll_seconds)) is not None:
        raise ValueError(refusal)
    if args.homes_root is None and os.environ.get("OPYT_HOMES_ROOT"):
        args.homes_root = Path(os.environ["OPYT_HOMES_ROOT"])
    raw_lifetime = os.environ.get("OPYT_WORKER_MAX_LIFETIME")
    if ignored_lifetime_env(args.max_lifetime_seconds, raw_lifetime):
        log(f"[rail-worker] invalid OPYT_WORKER_MAX_LIFETIME={raw_lifetime!r}; "
            "using the platform default")
    args.max_lifetime = lifetime_bound(args.max_lifetime_seconds, raw_lifetime,
                                       hosted=args.homes_root is not None)
    return args


if __name__ == "__main__":
    raise SystemExit(main())
