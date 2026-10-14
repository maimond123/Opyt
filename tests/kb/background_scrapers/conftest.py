"""Fixtures for the background-scraper tests: a store, the quota loops as oracles assembles them
(`scrapers_job`) with a fake page kit, and the two network seams (`walk_page`, `sample_page`)
serving scripted answers."""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.worker.quota_loop import retries
from pipeline.oracles import registry
from pipeline.oracles.sources.timeline import pages
from pipeline.oracles.sources import loops
from pipeline.oracles.sources import scraper_tables

from ._support import NOW


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    registry.init_state_schema(c)
    scraper_tables.init_schema(c)
    yield c
    c.close()


class Kit:
    """Stands in for `pages.PageKit`."""

    def __init__(self):
        self.forgot = 0

    def user_id(self, conn, cid, handle):
        return "99"

    def session(self):
        return {}

    def forget_session(self):
        self.forgot += 1


@pytest.fixture()
def loop():
    """X's loop, its clock at `NOW`, the instant the walks' tests pin."""
    return loops.x_timeline_loop(meter=lambda: None, clock=lambda: NOW.timestamp())


@pytest.fixture()
def web_loop():
    return loops.web_loop(meter=lambda: None)


@pytest.fixture()
def ctx(loop):
    loop.ctx.cache["page_kit"] = Kit()
    return loop.ctx


@pytest.fixture()
def no_jitter(monkeypatch):
    monkeypatch.setattr(retries, "jitter", lambda: 1.0)


@pytest.fixture()
def walk(monkeypatch):
    """`walk_page` serving scripted pages (or exceptions), recording where each began."""
    served: list = []
    calls: list = []

    def fake(conn, kit, *, user_id, handle, below_id=None, author_name=None):
        calls.append((handle, below_id))
        nxt = served.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(pages, "walk_page", fake)
    return served, calls


@pytest.fixture()
def scout(monkeypatch, conn):
    """One candidate, and `sample_page` serving scripted pages (or exceptions)."""
    from pipeline.oracles.sources.timeline import samples
    cand = {"who_id": "x:user:5", "user_id": "5", "handle": "dana", "canonical_id": "x:user:5",
            "name": "Dana", "distinct_signals": 2}
    monkeypatch.setattr(samples, "candidate_queue", lambda conn, **kw: [cand])
    served: list = []
    calls: list = []

    def fake(conn, session, c, *, below_id=None):
        calls.append(below_id)
        nxt = served.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(samples, "sample_page", fake)
    return served, calls
