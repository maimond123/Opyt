"""
pipeline/background.py — the `scrapers` job: every domain's background work in one process.

The worker owns the engine (the quota loops and the process around them) and imports no domain.
Each domain supplies its own work: oracles its loops and the web sources it owns
(`oracles/sources/loops.py`), curation its sources and loops
(`pipeline/curation/collectors.py`), frontier its search and admission sources
(`pipeline/frontier/searching.py`, `admitting.py`) and sharing its publishing passes
(`pipeline/sharing/publishing.py`). This module is the one place they meet: one loop per quota,
one web pool shared by every domain's web sources, and one connection factory that creates every
domain's tables. A long upload runs on a serial quota of its own, so it holds no web slot.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess

from pipeline.curation import collectors
from pipeline.frontier.admitting import Admissions
from pipeline.frontier.searching import Searches
from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.model_routing import models_unroutable
from pipeline.sharing import publishing
from pipeline.worker.home import current_home_id
from pipeline.worker.job_queue import RailJobStore
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.process import LoopProcess
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.registry import SCRAPERS
from pipeline.oracles.sources import loops


def connect() -> sqlite3.Connection:
    """A loop's connection, with every domain's tables created on it."""
    conn = schema.connect()
    loops.init_schema(conn)
    collectors.init_schema(conn)
    return conn


def scrapers() -> list[QuotaLoop]:
    """One loop per quota. The web pool's outage state is shared by every web source in it, so a
    paused platform holds oracles', curation's and frontier's reads alike."""
    web = Outages(PLATFORM_PAUSES_S)
    extra_web = [*collectors.web_sources(loops.WEB, web), Searches(web), Admissions(web)]
    return [*loops.loops(connect, web, extra_web=extra_web),
            *collectors.loops(connect),
            QuotaLoop(publishing.QUOTA, [publishing.Publications()], executor=Serial(),
                      outages=Outages(PLATFORM_PAUSES_S), connect=connect)]


def keep_awake() -> None:
    """Hold off idle sleep while this process runs: a sleeping Mac stalls every pull. `-w` ties
    the assertion to this process, so it ends by itself when the process exits. It cannot stop
    lid-closed sleep on battery."""
    try:
        subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    except OSError as exc:
        log(f"[scrapers] could not hold off sleep, running anyway: {exc!r}")


def run() -> dict:
    """The `scrapers` job: run every loop until all are idle, or until told to stop.

    When a model has no provider and no fallback, nothing runs, model-free work included."""
    if (reason := models_unroutable(SCRAPERS)) is not None:
        return {"status": "models_unroutable", "message": reason}
    keep_awake()
    return LoopProcess(scrapers(), store=RailJobStore(), home_id=current_home_id(),
                       seed=loops.seed).run()
