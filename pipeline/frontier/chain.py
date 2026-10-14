"""
pipeline/frontier/chain.py — what makes frontier's background work due.

    standing queries added (a topic's watches)                 → the scrapers process wakes
    a search staged a new candidate                            → it waits for the host's check
    the host kept it                                           → the scrapers process wakes
    an admission materialized it                               → nothing: search reads the store

Both stages are sources on the scrapers process (`searching.py`, `admitting.py`), which works out
what is owed from the store each time it asks, so the order between them is the data, not a
call. The one thing the store cannot do is wake a process that has gone idle: that is written
here, once. `job_queue.request_now` is called through the module, so a test that patches it sees
every call.
"""

from __future__ import annotations

from pipeline.worker import job_queue
from pipeline.worker.registry import SCRAPERS


def queries_added() -> bool:
    """Standing queries were just added: wake the scrapers process to search them. True iff the
    request is durable."""
    return job_queue.request_now(SCRAPERS)


def finds_kept() -> bool:
    """The host kept finds: wake the scrapers process to admit them. True iff the request is
    durable."""
    return job_queue.request_now(SCRAPERS)
