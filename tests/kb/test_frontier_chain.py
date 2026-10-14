"""`pipeline/frontier/chain.py` — its one job: standing queries added wake the
scrapers process, whose searches find the new pairs due by themselves. The producers that call it
are pinned in test_rail_activation.py and test_research_flows.py."""
from __future__ import annotations

from pipeline.frontier import chain
from pipeline.worker import job_queue
from pipeline.worker.registry import SCRAPERS


def test_queries_added_wakes_the_scrapers_process(monkeypatch):
    woken = []
    monkeypatch.setattr(job_queue, "request_now", lambda rail: woken.append(rail) or True)
    assert chain.queries_added() is True
    assert woken == [SCRAPERS]


def test_a_wake_that_did_not_land_says_so(monkeypatch):
    monkeypatch.setattr(job_queue, "request_now", lambda rail: False)
    assert chain.queries_added() is False
