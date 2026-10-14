"""
pipeline/worker/exit_codes.py — the protocol a job's child speaks to the worker: a result's
status, as an exit code.

`fork + exec` discards the memory image, so a child's only report is its exit code, and 0 is
already overloaded to mean "this pass was healthy" (`ok`, `already_running`, `nothing_owed` and the
rest of `HEALTHY`); anything else is 1. `exit_code_for` is the one table every child uses. The
worker records the code, and every exit earns the job's cadence (`RailWorker.reap_finished`).

A child must never `request_now` its own job: `finish` keeps a request made during the run,
so a job that asked for itself would stay permanently due and relaunch in a hot loop. Requests
are for whatever else makes a job due: a tool call, or another job. `scrapers` writes its
own row through `RailJobStore.rearm` and exits once idle. The one exception is a stop: on
SIGTERM `scrapers` and `line` each make their own row due now, so the replacement resumes
immediately.
"""

from __future__ import annotations

import signal

# Exit codes 70, 71 and 72 are burned: shipped workers still map them, so do not reuse them.

# A child stopped by SIGTERM: the shell's convention, 128 + the signal. The worker records no
# exit for a child it stopped itself (`RailWorker.stop_children`), so nothing schedules from it.
EXIT_TERMINATED = 128 + signal.SIGTERM

# The statuses that mean "this pass was healthy", whether or not it did anything. `stopped` is
# the scrapers' process or the line told to stop by the worker's restart.
HEALTHY = frozenset({
    "ok", "skipped", "already_running", "dry-run", "plan",
    "not_shared", "not_published", "no_change", "done", "nothing_owed", "stopped",
})


def exit_code_for(res: dict) -> int:
    """The one table: status → exit code."""
    status = res.get("status")
    return 0 if status in HEALTHY else 1
