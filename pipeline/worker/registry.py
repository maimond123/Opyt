"""
pipeline/worker/registry.py — every job the worker may launch: its entry point, its log file, its
cadence and whether it runs beside the others.

A job is named here once, and the name is durable: it is the `rail` column of every queued row,
so it never changes. Producers queue a job by this name (`job_queue.request_now`); the worker
launches only what is listed here, each through the one child entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

HOURLY = 3600.0
# The scrapers' safety-net wake (`quota_loop.process.SAFETY_NET_S` is this number): the process
# writes its own next wake when it exits, and this is only the ceiling.
QUARTER_HOURLY = 15 * 60.0

# The background scrapers' process: one per home.
SCRAPERS = "scrapers"
# The line to the sharing service, which hands over the tool calls the web and agents ask for:
# one per home that shares.
LINE = "line"


@dataclass(frozen=True)
class RailSpec:
    """One job's entry point, its log, its cadence and whether it runs beside the others.

    `run` is the job's `run_*` as `module:function`, called with no arguments by `child`,
    the one entry point every child goes through, whose result `exit_codes.exit_code_for` turns
    into its exit code. `cadence` is the next run after any exit, unless the job wrote an earlier
    wake for itself while it ran (`RailJobStore.finish`).

    `beside` is for a job that stays up for as long as the worker does (`LINE`). It neither waits
    for its home's other job nor makes that job wait (`RailJobStore.claim_next`), and it takes
    none of the worker's slots (`supervisor.launchable`)."""

    run: str
    log_name: str
    cadence: float
    beside: bool = False

    @property
    def command(self) -> tuple[str, ...]:
        """The child's arguments after the interpreter."""
        return ("-m", "pipeline.worker.child", self.run)


RAILS: Mapping[str, RailSpec] = MappingProxyType({
    # The background scrapers' process (`pipeline/background.py`). It wakes when a tool asks and
    # exits about a minute after going idle, writing its own next wake; the cadence is only the
    # safety net under that.
    SCRAPERS: RailSpec(
        "pipeline.background:run", "scrapers.log", QUARTER_HOURLY,
    ),
    # The line (`mcp_server/line.py`), named by a string so the worker imports no server code. A
    # library with a web or agent writer holds it until the worker restarts. One without asks
    # once and exits, and the cadence asks again; a home that stopped sharing exits before it
    # asks.
    LINE: RailSpec(
        "mcp_server.line:run", "line.log", HOURLY, beside=True,
    ),
})
