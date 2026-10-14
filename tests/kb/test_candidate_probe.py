"""The light timeline pull over the Oracle candidate list: the queue, and one page.

Offline throughout — `fetch_user_originals` is monkeypatched. Pinned here: who is due and who is
excluded from the queue at all; which outcomes buy a candidate a TTL's worth of silence; and what
one page stores (the curation filter runs and the substance filter does not, so short posts
survive; an unchanged post is not written again). The walk around the page — how many pages,
how far back, failures, the session, the meter — is the scouting source's
(`background_scrapers/x_timeline/scouting.py`, tested there).
"""
from __future__ import annotations

import pytest

from pipeline.kb import resolve, schema
from pipeline.oracles import probe_store, roster
from pipeline.oracles.sources.timeline import samples as cp
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x import originals as x_originals


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _candidate(conn, uid: str, *, signals=(("follow", "x"),), name=None, handle=None):
    eid = f"x:user:{uid}"
    schema.upsert_entity(conn, eid, name=name or f"P{uid}", profile={"handle": handle or f"p{uid}"})
    for st, pf in signals:
        schema.ensure_signal(conn, eid, st, pf)
    return eid


def _tweet(tid: str, *, uid: str = "11", user: str = "p11", text: str = "on protein folding",
           reply_to: str | None = None, conv: str | None = None) -> dict:
    """A NORMALIZED tweet — the shape `fetch_user_originals` returns."""
    t = {"id": tid, "text": text, "createdAt": "Mon Aug 11 10:00:00 +0000 2026",
         "author": {"userName": user, "name": user, "id": uid, "site": ""},
         "entities": {}, "extendedEntities": {}, "likeCount": 1, "replyCount": 0,
         "url": f"https://x.com/{user}/status/{tid}",
         "conversationId": conv or tid, "isQuote": False, "isRetweet": False,
         "isReply": bool(reply_to), "inReplyToUserId": reply_to or ""}
    return t


def _serve(monkeypatch, by_user: dict):
    """Stub the fetch. A value may be a tweet list, or an Exception to raise. Records the order
    accounts were requested in — that IS the queue order under a bounded budget."""
    asked: list[str] = []

    def _fake(client, user_id, *, after, below_id=None):
        asked.append(str(user_id))
        v = by_user.get(str(user_id), [])
        if isinstance(v, Exception):
            raise v
        return x_originals.OriginalsWalk(list(v), True)

    monkeypatch.setattr(x_originals, "fetch_user_originals", _fake)
    monkeypatch.setattr(x_local, "read_x_cookies", lambda: {"auth_token": "t"})
    monkeypatch.setattr(x_local, "auth_headers", lambda cookies, referer: {})
    return asked


def _sample(conn, monkeypatch, tweets, uid="11"):
    """One page of candidate `uid`, served `tweets`."""
    _serve(monkeypatch, {uid: tweets})
    page = cp.sample_page(conn, {}, {"who_id": f"x:user:{uid}", "user_id": uid,
                                     "handle": f"p{uid}"})
    cp.write_sample(conn, page)
    return page


def test_queue_is_in_screen_rank_order(conn):
    _candidate(conn, "1", signals=(("like", "x"),))                    # 1 distinct
    _candidate(conn, "2", signals=(("save", "x"), ("follow", "x")))    # 2 distinct → first
    assert [c["who_id"] for c in cp.candidate_queue(conn)] == ["x:user:2", "x:user:1"]


def test_queue_honors_min_signals(conn):
    _candidate(conn, "1", signals=(("like", "x"),))
    _candidate(conn, "2", signals=(("save", "x"), ("follow", "x")))
    assert [c["who_id"] for c in cp.candidate_queue(conn, min_signals=2)] == ["x:user:2"]


def test_queue_excludes_confirmed_oracles(conn):
    """Their real footprint is already in `atoms` — probing them would duplicate TRUSTED content
    into the untrusted store."""
    _candidate(conn, "1")
    roster.upsert_oracle(conn, "x:user:1", name="P1", source="screen")
    assert cp.candidate_queue(conn) == []


def test_queue_excludes_candidates_with_no_x_identity(conn):
    """A Substack-only subscription has no timeline on this path — absent, not failed."""
    schema.upsert_entity(conn, "substack:carol", name="Carol")
    schema.ensure_signal(conn, "substack:carol", "subscribe", "substack")
    resolve.resolve_entities(conn)
    assert cp.candidate_queue(conn) == []


# ── one page ──────────────────────────────────────────────────────────────────

def test_content_lands_in_the_probe_store_only(conn, monkeypatch):
    _candidate(conn, "11", handle="p11")
    page = _sample(conn, monkeypatch,
                   [_tweet("1", text="an agent framework for autonomous tools")])
    assert page["written"] == 1
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT atom_id FROM probe_atoms").fetchone()[0] == "xprobe:1"


# ── filtering: the curation filter runs, the substance filter does not ────────

def test_an_inline_article_body_is_complete_while_a_teaser_is_partial():
    full = _tweet("1")
    full["article"] = {"article_results": {"result": {
        "title": "A full essay", "content_state": {"blocks": [
            {"type": "unstyled", "text": "The inline article body is searchable evidence."},
        ]},
    }}}
    teaser = _tweet("2")
    teaser["article"] = {"article_results": {"result": {"title": "Only a teaser"}}}

    full_atom, teaser_atom = cp._render_groups([[full], [teaser]], handle="p11")
    assert "The inline article body is searchable evidence." in full_atom["_markdown"]
    assert full_atom["payload"]["body_state"] == "complete"
    assert teaser_atom["payload"]["body_state"] == "partial"
    assert "source: x-probe" in full_atom["_markdown"]


def test_self_thread_becomes_one_atom_and_replies_to_others_are_dropped(
        conn, monkeypatch):
    _candidate(conn, "11", handle="p11")
    _sample(conn, monkeypatch, [
        _tweet("1", conv="1", text="thread root about protein folding"),
        _tweet("2", conv="1", reply_to="11", text="continuation of my own thread"),
        _tweet("3", reply_to="99", text="replying to a stranger"),
    ])
    rows = [r[0] for r in conn.execute("SELECT atom_id FROM probe_atoms ORDER BY atom_id")]
    assert rows == ["xprobe:1"]                      # ONE atom: root + self-reply, stranger gone


def test_a_one_line_aphorism_is_kept(conn, monkeypatch):
    """The substance filter's 200-char bar must NOT run here. A Proposer asks what field someone is
    in, and a short post answers that — dropping it reintroduces the essayist bias."""
    _candidate(conn, "11", handle="p11")
    _sample(conn, monkeypatch, [_tweet("1", text="AI is biology now.")])
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1


# ── what a page reports ──────────────────────────────────────────────────────

def test_a_filtered_out_page_is_still_an_observation(conn, monkeypatch):
    """A page of nothing but replies-to-others means we observed this candidate and they said
    nothing of their own — a fact, not a failure."""
    _candidate(conn, "11", handle="p11")
    page = _sample(conn, monkeypatch, [_tweet("1", reply_to="99"), _tweet("2", reply_to="98")])
    assert (page["fetched"], page["written"]) == (2, 0)
    assert probe_store.count_probe_atoms(conn) == 0


# ── idempotency ───────────────────────────────────────────────────────────────

def test_an_unchanged_re_pull_writes_nothing(conn, monkeypatch):
    """The hash skip: a post whose snapshot is unchanged is not rewritten, so its version stays
    and the page reports nothing new to store."""
    _candidate(conn, "11", handle="p11")
    tweets = [_tweet("1", text="an agent framework")]
    first = _sample(conn, monkeypatch, tweets)
    assert first["written"] == 1

    page = _sample(conn, monkeypatch, tweets)
    assert page["written"] == 0 and page["atoms"] == []      # observed, nothing new to store
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1
    assert conn.execute("SELECT version FROM probe_atoms WHERE atom_id='xprobe:1'"
                        ).fetchone()[0] == 1


def test_a_changed_post_is_rewritten_in_place(conn, monkeypatch):
    _candidate(conn, "11", handle="p11")
    _sample(conn, monkeypatch, [_tweet("1", text="an agent framework")])
    page = _sample(conn, monkeypatch, [_tweet("1", text="an agent framework, edited")])
    assert page["written"] == 1
    assert probe_store.count_probe_atoms(conn, "x:user:11") == 1
    assert "edited" in probe_store.texts_by_person(conn)["x:user:11"][0]


def test_sampling_a_page_writes_nothing_until_it_is_written(conn, monkeypatch):
    """`sample_page` only renders: the scouting source writes the page with the candidate's
    standing, in one transaction, so a page read but never written leaves no atom behind."""
    _candidate(conn, "11", handle="p11")
    _serve(monkeypatch, {"11": [_tweet("1", text="an agent framework")]})
    page = cp.sample_page(conn, {}, {"who_id": "x:user:11", "user_id": "11", "handle": "p11"})
    assert page["written"] == 1
    assert probe_store.count_probe_atoms(conn) == 0
    cp.write_sample(conn, page)
    assert probe_store.count_probe_atoms(conn) == 1
