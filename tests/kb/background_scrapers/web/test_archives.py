"""The archives (`background_scrapers.web.archives`): a found source's first pull, a pair pulled
again on an ask, keeping pairs current, and what a web failure costs."""
from __future__ import annotations

from datetime import timedelta

import pytest

from pipeline.circuit_breaker import CircuitOpenError
from pipeline.kb import schema
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop import retries
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import windows
from pipeline.oracles import asks
from pipeline.oracles.found import list_found
from pipeline.oracles.sources.web import archives as ar
from pipeline.oracles.sources.web import platform

from .._support import NOW

CID = "x:user:1"


@pytest.fixture()
def ctx(web_loop):
    return web_loop.ctx


@pytest.fixture()
def oracle(conn):
    schema.upsert_entity(conn, CID, name="Alice")
    roster.upsert_oracle(conn, CID, name="Alice")
    asks.record_ask(conn, CID, web_from=windows.ALL_TIME, now=utc_now() - timedelta(minutes=1))
    return CID


def _found(conn, url, source_type="substack", cid=CID):
    conn.execute("INSERT INTO found_sources (canonical_id, source_type, url, metadata, found_at) "
                 "VALUES (?, ?, ?, '{\"trust\": {\"trusted\": true}}', 'then')",
                 (cid, source_type, url))
    conn.commit()


def _pair(conn, source_type, key, **cols):
    registry.upsert_source(conn, registry.SourceRow(CID, source_type, key))
    if cols:
        conn.execute(f"UPDATE oracle_sources SET {', '.join(f'{k}=?' for k in cols)} "
                     "WHERE source_key=?", (*cols.values(), key))
        conn.commit()
    return next(r for r in registry.list_sources(conn) if r.source_key == key)


@pytest.fixture()
def onboard(monkeypatch):
    """`onboard_footprint` answering with scripted records (or raising), recording the stamps."""
    served, stamped = [], []

    def fake(conn, oracle_id, sources, *, isolate=True, **kw):
        assert isolate is False and len(sources) == 1
        nxt = served.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return {"results": [{"url": sources[0]["url"], "type": sources[0]["source_type"], **nxt}]}
    monkeypatch.setattr("pipeline.oracles.sources.first_pull.onboard_footprint", fake)
    monkeypatch.setattr("pipeline.oracles.coverage.stamp_source",
                        lambda conn, cid, rec, **kw: stamped.append(rec["action"]))
    monkeypatch.setattr("pipeline.platforms.connections._connected_x", lambda: True)
    return served, stamped


@pytest.mark.parametrize("stype, url, key", [
    ("substack", "https://www.dwarkesh.com", "substack"),
    ("github", "https://github.com/soren", "github"),
    ("openalex", "A123", "openalex"),
    ("blog", "https://alice.wordpress.com/", "blog:wordpress.com"),
    ("blog", "https://blog.willcb.com", "blog:willcb.com"),
    ("blog", "https://www.foo.co.uk/posts", "blog:foo.co.uk"),
])
def test_a_failure_counts_against_its_platform(stype, url, key):
    assert platform.platform_key(stype, url) == key


# ── a found source's first pull ───────────────────────────────────────────────────

def test_a_first_pull_runs_once_whatever_it_found(conn, web_loop, oracle, ctx, onboard):
    served, stamped = onboard
    _found(conn, "https://a.substack.com")
    _found(conn, "https://b.substack.com")
    served += [{"action": "affiliation"}, {"action": "needs-review"}]
    a = ar.Archives()
    assert [web_loop.run_unit(conn, a, a.next_unit(conn, NOW)).result for _ in range(2)] == [
        "affiliation", "needs-review"]
    assert stamped == ["affiliation", "needs-review"]
    assert {f.url: f.last_status for f in list_found(conn)} == {
        "https://a.substack.com": "affiliation", "https://b.substack.com": "needs-review"}
    assert all(f.pulled_at for f in list_found(conn))
    assert a.next_unit(conn, NOW) is None


def test_a_raise_is_a_blip_and_the_first_pull_stays_owed(conn, web_loop, oracle, ctx, onboard,
                                                          no_jitter):
    served, _stamped = onboard
    _found(conn, "https://willcb.com", "blog")
    served.append(TimeoutError("read timed out"))
    a = ar.Archives()
    web_loop.run_unit(conn, a, a.next_unit(conn, NOW))
    (f,) = list_found(conn)
    assert f.last_status == retries.BLIP and f.failures == 1 and f.pulled_at is None


def test_the_model_provider_being_down_is_never_a_sources_failure(conn, oracle, ctx, onboard):
    served, _stamped = onboard
    _found(conn, "https://a.substack.com")
    served.append(CircuitOpenError("openrouter", 60))
    a = ar.Archives()
    with pytest.raises(CircuitOpenError):
        a.fetch(conn, a.next_unit(conn, NOW), ctx)
    assert list_found(conn)[0].failures is None


# ── a pair, on an ask ─────────────────────────────────────────────────────────────


def test_a_pair_waits_while_a_first_pull_of_its_type_is_owed(conn, oracle, onboard):
    _pair(conn, "substack", "https://a.substack.com")
    _found(conn, "https://a.substack.com")
    unit = ar.Archives().next_unit(conn, NOW)
    assert unit.payload[0] == "found"
    conn.execute("UPDATE found_sources SET pulled_at='now'")
    conn.commit()
    assert ar.Archives().next_unit(conn, NOW).payload[0] == "pair"


# ── keeping pairs current ─────────────────────────────────────────────────────────


def test_the_web_sources_share_one_view_of_each_platform():
    from pipeline.oracles.sources.loops import web_loop
    built = web_loop().sources
    assert [s.name for s in built] == ["discovery", "archives", "linked_documents", "link_cards",
                                       "web_new_posts", "recommendations", "substack_samples"]
    assert built[1].health is built[2].health is built[3].health is built[4].health
    assert {s.quota for s in built} == {"web"}


def test_a_kind_the_ask_left_out_is_never_pulled_unasked(conn, ctx):
    """A research venue confirmed without topics gets no paper ask.
    Its pair is owed nothing on the ask, and is not first-pulled as new posts either: unfiltered,
    it would be the venue's newest 2,000 papers with no sign the rest was cut."""
    roster.upsert_oracle(conn, "openalex:S1", name="ChemRxiv")
    asks.record_ask(conn, "openalex:S1", web_from=windows.ALL_TIME,
                     now=utc_now() - timedelta(minutes=1))
    registry.upsert_source(conn, registry.SourceRow("openalex:S1", "openalex", "S1"))
    row = next(r for r in registry.list_sources(conn) if r.source_key == "S1")
    ask = asks.ask_for(conn, "openalex:S1")

    assert not windows.owed_on_ask(row, ask) and windows.waits_for_an_ask(row, ask)
    assert ar.Archives().next_unit(conn, utc_now()) is None
    assert ar.WebNewPosts().next_unit(conn, utc_now()) is None

    asks.record_ask(conn, "openalex:S1", scholar_from=windows.ALL_TIME)
    assert ar.Archives().next_unit(conn, utc_now()) is not None
