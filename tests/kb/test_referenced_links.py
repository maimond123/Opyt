"""The referenced-links queue: the X walk queues each post's links on the post
(`payload.links_owed`) and fetches none; `fill_referenced_links` captures them afterwards, and
`referenced_links` runs it once every Oracle's posts are in.

Captured inside the walk, one Oracle's links took 378 of a 449-second pull, and every Oracle
after them in the pull waited.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from pipeline.platforms.x import conversation as x_conversation
import pipeline.platforms.x.rate as x_rate
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.quota_loop.retries import Outages, PLATFORM_PAUSES_S, Wait

import pytest

from pipeline.kb import ingest_x_footprint as fp
from pipeline.kb import link_mint, link_router
from pipeline.kb import owed_links
from pipeline.kb import schema
from tests.kb.test_ingest_x_footprint import (_LONG, _NEWS_URL, _other_post, _patch_fetch,
                                              _patch_news, _patch_x_lookup, _payload, _raw,
                                              _sync)
from pipeline.oracles import roster
from pipeline.oracles.sources.references.documents import Documents
from pipeline.oracles.sources.references import documents
from pipeline.oracles.sources.references import state
from pipeline.oracles.sources.references.x_posts import XPosts


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()                       # the default store, so `referenced_links` sees it
    yield c
    c.close()


def _post_linking(tid, *urls):
    t = _raw(tid, text=_LONG)
    t["entities"] = {"urls": [{"expanded_url": u} for u in urls]}
    return t


def _no_link_request(monkeypatch):
    """Every network arm a link capture could take, refused."""
    from pipeline.platforms.web import article as web_article

    def _boom(*a, **k):
        raise AssertionError("the walk made a link request")
    monkeypatch.setattr(link_router, "classify_link_deep", _boom)
    monkeypatch.setattr(link_mint, "mint_artifact", _boom)
    monkeypatch.setattr(web_article, "fetch_article", _boom)
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", _boom)


# ── the walk queues ─────────────────────────────────────────────────────────────

def test_the_walk_fetches_no_link_and_queues_every_one_it_would_capture(conn, monkeypatch):
    quoted = {"id": "9", "text": "my repo", "url": "https://x.com/bob/status/9",
              "author": {"userName": "bob", "id": "7"},
              "entities": {"urls": [{"expanded_url": "https://github.com/bob/thing"}]}}
    post = _post_linking("81", _NEWS_URL, "https://x.com/dan/status/222",
                         "https://arxiv.org/abs/2401.00001",
                         "https://x.com/carol/status/5",            # her own post — never queued
                         "https://gist.github.com/carol/0123")      # not a reference — never queued
    post["quoted_tweet"] = quoted
    _patch_fetch(monkeypatch, [post])
    _no_link_request(monkeypatch)

    out = _sync(conn, handle="carol")

    assert out["added"] == 1 and out["links_owed"] == 4
    owed = _payload(conn, "xprofile:81")["links_owed"]
    assert [e["url"] for e in owed] == [_NEWS_URL, "https://x.com/dan/status/222",
                                        "https://arxiv.org/abs/2401.00001",
                                        "https://github.com/bob/thing"]
    assert owed[0]["ref"] == {"by": "x:user:99", "handle": "carol", "post": "xprofile:81",
                              "url": "https://x.com/carol/status/81", "date": "2026-01-06"}
    assert owed[3]["ref"]["via_quote"]["handle"] == "bob"
    assert {r[0] for r in conn.execute("SELECT atom_id FROM atoms")} == {"xprofile:81"}


def test_a_post_with_no_links_still_records_that_nothing_is_owed(conn, monkeypatch):
    """`[]`, not absent: absent means "stored before the queue", which a re-walk acts on."""
    _patch_fetch(monkeypatch, [_raw("82", text=_LONG)])
    _sync(conn, handle="carol")
    assert _payload(conn, "xprofile:82")["links_owed"] == []


def test_a_post_stored_before_the_queue_has_its_links_queued_without_a_rewrite(
        conn, monkeypatch):
    _patch_news(monkeypatch)
    _patch_fetch(monkeypatch, [_raw("83", text=_LONG, link=_NEWS_URL)])
    _sync(conn, handle="carol")
    conn.execute("UPDATE atoms SET payload = json_remove(payload, '$.links_owed')")
    conn.commit()
    bodies = []
    real = schema.put_body
    monkeypatch.setattr(schema, "put_body",
                        lambda c, atom_id, text: bodies.append(atom_id) or real(c, atom_id, text))

    again = _sync(conn, handle="carol")

    assert again["skipped"] == 1 and again["added"] == 0 and bodies == []
    assert again["links_owed"] == 1
    assert [e["url"] for e in _payload(conn, "xprofile:83")["links_owed"]] == [_NEWS_URL]



@pytest.fixture()
def lanes(conn):
    roster.upsert_oracle(conn, "x:user:99", name="Carol")
    health = Outages(PLATFORM_PAUSES_S)
    x = QuotaLoop("x:TweetResultsByRestIds", [XPosts()], executor=Serial(), meter=lambda: None, outages=Outages(PLATFORM_PAUSES_S), connect=schema.connect)
    web = QuotaLoop("web", [Documents(health)], executor=Serial(), outages=health, connect=schema.connect)
    for lane in (x, web):
        lane.ctx.cache.update(session={})
    return x, web


def _owe(conn, post, *urls, who="x:user:99"):
    entries = [{"url": url, "ref": {"post": post, "by": who}} for url in urls]
    schema.upsert_atom(conn, {"atom_id": post, "source_type": "x", "who_id": who,
                             "entry_mode": "oracle-footprint", "payload": {"links_owed": entries}})
    conn.commit()
    return entries


def test_document_and_referrer_are_durable_before_removing_the_link(conn, lanes, monkeypatch):
    x, web = lanes
    _patch_news(monkeypatch)
    _owe(conn, "xprofile:1", _NEWS_URL)
    assert web.step(conn).result == "captured"
    assert _payload(conn, "xprofile:1")["links_owed"] == []
    target = conn.execute("SELECT payload FROM atoms WHERE source_type='blog'").fetchone()
    assert json.loads(target[0])["referenced_by"] == [{"post": "xprofile:1", "by": "x:user:99"}]
    assert isinstance(web.step(conn), Wait)


def test_reference_write_failure_rolls_back_completion_for_every_referrer(conn, lanes, monkeypatch):
    """The document, its provenance and every referrer's completion commit together: a failed
    reference write takes all of them back, and the link stays owed, uncharged."""
    x, web = lanes
    _patch_news(monkeypatch)
    _owe(conn, "xprofile:1", _NEWS_URL)
    _owe(conn, "xprofile:2", _NEWS_URL)
    original = schema.add_reference
    calls = []

    def fail_second(conn, target, ref):
        calls.append(ref)
        if len(calls) == 2:
            raise sqlite3.OperationalError("database is locked")
        return original(conn, target, ref)

    monkeypatch.setattr(schema, "add_reference", fail_second)
    out = web.step(conn)
    assert isinstance(out, Wait) and out.cause == "transient"
    assert len(state.owed(conn, x=False)) == 2
    assert conn.execute("SELECT 1 FROM atoms WHERE source_type='blog'").fetchone() is None
    assert all(not link.retry.failures for link in state.owed(conn, x=False))
    monkeypatch.setattr(schema, "add_reference", original)
    assert web.step(conn).result == "captured"
    assert state.owed(conn, x=False) == []


def test_new_links_from_a_concurrent_thread_rewrite_survive_completion(conn, lanes, monkeypatch):
    x, web = lanes
    _patch_news(monkeypatch)
    old = _owe(conn, "xprofile:1", _NEWS_URL)
    added = {"url": "https://example.com/new", "ref": {"post": "xprofile:1", "by": "x:user:99"}}
    fetch = documents._fetch

    def rewritten_meanwhile(conn_, *a, **kw):
        out = fetch(conn_, *a, **kw)
        conn_.execute("UPDATE atoms SET payload=json_set(payload, '$.links_owed', json(?)) "
                      "WHERE atom_id='xprofile:1'", (json.dumps([*old, added]),))
        conn_.commit()
        return out
    monkeypatch.setattr(documents, "_fetch", rewritten_meanwhile)
    unit = web.sources[0].next_unit(conn, datetime.now(timezone.utc))
    assert web.sources[0].fetch(conn, unit, web.ctx)[1].status == "captured"
    assert _payload(conn, "xprofile:1")["links_owed"] == [added]


def test_x_batches_deduplicate_ids_across_urls_and_referrers(conn, lanes, monkeypatch):
    x, web = lanes
    roster.upsert_oracle(conn, "x:user:88", name="Dana")
    _owe(conn, "xprofile:1", "https://x.com/bob/status/111")
    _owe(conn, "xprofile:2", "https://twitter.com/bob/status/111?s=20", who="x:user:88")
    _owe(conn, "xprofile:3", "https://x.com/dan/status/222")
    calls = []
    _patch_x_lookup(monkeypatch, [_other_post("111"), _other_post("222", author_id="8", handle="dan")], calls)
    assert x.step(conn).result == "captured"
    assert calls == [["111", "222"]]
    assert {r["post"] for r in _payload(conn, "x:111")["referenced_by"]} == {"xprofile:1", "xprofile:2"}
    assert state.owed(conn, x=True) == []


def test_more_than_one_hundred_ids_yield_after_one_request(conn, lanes, monkeypatch):
    x, web = lanes
    _owe(conn, "xprofile:1", *(f"https://x.com/bob/status/{i}" for i in range(1, 102)))
    calls = []
    _patch_x_lookup(monkeypatch, [_other_post(str(i)) for i in range(1, 102)], calls)
    assert x.step(conn).result == "captured"
    assert len(calls) == 1 and len(calls[0]) == 100
    assert len(state.owed(conn, x=True)) == 1
    assert x.step(conn).result == "captured"
    assert calls[1] == ["101"]


def test_a_missing_x_post_is_final_but_a_refused_request_is_owed(conn, lanes, monkeypatch):
    x, web = lanes
    _owe(conn, "xprofile:1", "https://x.com/bob/status/111")
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", lambda *a: (_ for _ in ()).throw(x_rate.XRateLimited("spent")))
    out = x.step(conn)
    assert isinstance(out, Wait) and out.cause == "quota_spent"
    assert len(state.owed(conn, x=True)) == 1 and state.owed(conn, x=True)[0].retry.failures is None
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", lambda *a: [])
    assert x.step(conn).result == "captured"
    assert state.owed(conn, x=True) == []


def test_a_failed_body_write_leaves_its_link_owed_and_no_bare_atom(conn, lanes, monkeypatch):
    """The atom row and its body land in one transaction with the link's completion: a body
    write that fails takes the atom back too, and the link stays owed, uncharged."""
    x, web = lanes
    _owe(conn, "xprofile:1", "https://x.com/bob/status/111")
    _patch_x_lookup(monkeypatch, [_other_post("111")], [])

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(schema, "put_body", locked)
    out = x.step(conn)
    assert isinstance(out, Wait) and out.cause == "transient"
    assert not conn.execute("SELECT 1 FROM atoms WHERE atom_id='x:111'").fetchone()
    assert not conn.execute("SELECT 1 FROM bodies WHERE atom_id='x:111'").fetchone()
    assert len(state.owed(conn, x=True)) == 1
    assert state.owed(conn, x=True)[0].retry.failures is None


def test_a_lost_lease_writes_no_target(conn, lanes, monkeypatch):
    """Another process took the quota: the fetched post is held, never written, and stays owed."""
    x, web = lanes
    _owe(conn, "xprofile:1", "https://x.com/bob/status/111")
    _patch_x_lookup(monkeypatch, [_other_post("111")], [])
    x.ctx.lease_lost = lambda: True
    x.step(conn)
    assert not conn.execute("SELECT 1 FROM atoms WHERE atom_id='x:111'").fetchone()
    assert not conn.execute("SELECT 1 FROM bodies WHERE atom_id='x:111'").fetchone()
    assert _payload(conn, "xprofile:1")["links_owed"]


def test_forgetting_the_referrer_during_fetch_cannot_create_a_target(conn, lanes, monkeypatch):
    x, web = lanes
    _owe(conn, "xprofile:1", "https://x.com/bob/status/111")

    def fetched(*args):
        conn.execute("DELETE FROM oracles WHERE canonical_id='x:user:99'")
        conn.commit()
        return [_other_post("111")]

    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", fetched)
    x.step(conn)
    assert not conn.execute("SELECT 1 FROM atoms WHERE atom_id='x:111'").fetchone()
    assert _payload(conn, "xprofile:1")["links_owed"]


def test_spent_x_meter_never_calls_transport(conn, lanes, monkeypatch):
    x, web = lanes
    _owe(conn, "xprofile:1", "https://x.com/bob/status/111")
    x.clock = lambda: 1000
    x._meter = lambda: (0, 2000)
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", lambda *a: pytest.fail("spent quota made a request"))
    assert x.step(conn).cause == "quota_spent"


def test_old_link_job_is_retired():
    from pipeline.worker.registry import RAILS
    assert "referenced_links" not in RAILS


def test_a_link_only_shape_is_owed_only_for_a_card_x_did_not_give():
    """A homepage or a jobs page is never fetched as a document. It is owed for its card
    (`kb/link_cards.py`) only when X attached none to the post that carries it: the post itself,
    or the post it quotes."""
    bare = _post_linking("1", "https://supermemory.ai", "https://vercel.com/careers")
    carded = {**_post_linking("2", "https://supermemory.ai"), "card": {"binding_values": [
        {"key": "title", "value": {"string_value": "Supermemory"}}]}}
    quoting = {**_post_linking("3"), "quoted_tweet": {**carded, "author": {"id": "7"}}}

    assert [e["url"] for e in owed_links.owed_links([bare], "x:1")] == [
        "https://supermemory.ai", "https://vercel.com/careers"]
    assert owed_links.owed_links([carded], "x:2") == []
    assert owed_links.owed_links([quoting], "x:3") == []


def test_an_owed_link_that_is_never_a_document_settles_final_without_a_fetch():
    """A video link queued before the rule drains as final, with no network and no store read.
    (A homepage is read for its card instead: `references/cards.py`.)"""
    assert documents._fetch(None, "https://youtube.com/watch?v=1", None, None) == (
        "final", None)
