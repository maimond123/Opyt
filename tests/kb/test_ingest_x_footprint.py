"""ingest_x_footprint (Stage-5 root footprint) — a confirmed Oracle's OWN X timeline → opinion atoms.

Two layers:
  • PURE filter/stitch unit tests hit `_filter_and_stitch` directly (no DB / network) —
    they encode the LOAD-BEARING order (drop reply-to-others BEFORE grouping by conversationId, or a
    self-thread essay welds to its replies-to-commenters) + the RESOLVED curation filter (drop RTs +
    replies-to-others, keep originals + self-threads, NO length gate). The reply target is
    `inReplyToUserId` vs the AUTHOR's id — `x.records.normalize_tweet` does not emit `inReplyToUsername`
    at all, and so the fixtures mirror the real shape (author.id + inReplyToUserId).
  • Adapter tests stub the X session + timeline walk and the VLM, proving the WIRING: self-thread →
    ONE atom, entry_mode/who_id, thread figure → VLM description in the body, content-hash
    idempotency, and full-window ingestion. Not the live scrape.
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from pipeline.platforms import sessions
from pipeline.platforms.x import conversation as x_conversation
from pipeline.platforms.x import originals as x_originals
from pipeline.platforms.x import profile as x_profile
import pipeline.platforms.x.rate as x_rate
from pipeline.kb import ingest_x_footprint as fp
from pipeline.kb import footprint_photos, link_router, schema, writes
from pipeline.oracles import windows
from tests.kb.reference_support import Capture, drain
from pipeline.oracles import roster
from pipeline.oracles.sources.threads import tail_state
from pipeline.oracles.sources.timeline import pages
from pipeline.oracles.sources.threads import state as thread_state
from pipeline.worker.quota_loop.retries import Wait


# ── PURE: filter + stitch (offline) ───────────────────────────────────────────────

_AUTHOR = "A"     # the timeline owner's numeric author id (uniform: the walk filters to it)


def _tw(tid, *, text="", conv=None, reply_to_uid=None, rt=False, author_id=_AUTHOR):
    """A minimal raw-twitterapi-shaped tweet. `reply_to_uid` = the numeric user id this tweet
    replies to (None = not a reply); reply_to_uid == author_id is a SELF-reply (thread continuation).
    Mirrors the live shape: inReplyToUserId populated, inReplyToUsername absent."""
    return {"id": tid, "text": text, "conversationId": conv or tid,
            "author": {"id": author_id, "userName": "carol"},
            "isRetweet": rt, "isReply": reply_to_uid is not None, "inReplyToUserId": reply_to_uid}


def test_short_aphorism_is_kept():                         # (a) NO length gate — Taleb survives
    groups = fp._filter_and_stitch([_tw("1", text="Skin in the game.")])
    assert [t["id"] for g in groups for t in g] == ["1"]


def test_reply_to_others_dropped_self_reply_kept():        # (b)
    raw = [_tw("1", text="hot take", conv="1"),
           _tw("2", text="@rando you're wrong", conv="9", reply_to_uid="RANDO"),   # reply-to-OTHER → drop
           _tw("3", text="…and another thing", conv="1", reply_to_uid=_AUTHOR)]    # self-reply → keep
    assert {t["id"] for g in fp._filter_and_stitch(raw) for t in g} == {"1", "3"}


def test_self_thread_stitches_to_one_group():              # (c) root + self-replies = ONE group
    raw = [_tw("10", text="1/ thesis", conv="10"),
           _tw("11", text="2/ argument", conv="10", reply_to_uid=_AUTHOR),
           _tw("12", text="3/ conclusion", conv="10", reply_to_uid=_AUTHOR)]
    g = fp._filter_and_stitch(raw)
    assert len(g) == 1 and [t["id"] for t in g[0]] == ["10", "11", "12"]


def test_filter_before_stitch_no_frankenstein():           # THE load-bearing ordering guard
    # A self-thread essay + a reply-to-commenter SHARE conversationId (the live @martin_casado case).
    raw = [_tw("100", text="1/ essay", conv="100"),
           _tw("101", text="2/ essay", conv="100", reply_to_uid=_AUTHOR),   # self-reply → keep
           _tw("200", text="@fan good q", conv="100", reply_to_uid="FAN")]  # reply-to-OTHER, SAME conv
    g = fp._filter_and_stitch(raw)
    assert len(g) == 1 and [t["id"] for t in g[0]] == ["100", "101"]        # reply-to-commenter NOT welded


def test_reply_to_others_dropped_despite_leading_at_mention():
    # A self-continuation carries carried-over @-mentions but replies to the AUTHOR — keep it; a real
    # reply-to-other also leads with @ — drop it. ID comparison separates them; a leading-@ heuristic can't.
    raw = [_tw("1", text="my point", conv="1"),
           _tw("2", text="@x @y Nevermind, I'm an idiot", conv="1", reply_to_uid=_AUTHOR),  # self → keep
           _tw("3", text="@stranger you're wrong", conv="1", reply_to_uid="Z")]             # other → drop
    assert {t["id"] for g in fp._filter_and_stitch(raw) for t in g} == {"1", "2"}


def test_retweets_dropped_and_dedup_by_id():
    raw = [_tw("1", rt=True), _tw("2", text="mine"),
           _tw("2", text="mine")]                          # API repeats across pages → dedup
    g = fp._filter_and_stitch(raw)
    assert {t["id"] for gr in g for t in gr} == {"2"} and sum(len(gr) for gr in g) == 1


# ── ADAPTER: fetch + build (mocked fetch) ─────────────────────────────────────────

@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    footprint_photos.init_schema(c)       # a page owes its unread photos here
    yield c
    c.close()


# A long body, used where the test's point is not the post's text. (There is no length bar on short posts.)
_LONG = ("A substantive take that comfortably clears the two-hundred-character naked-post floor so "
         "the substance filter keeps it as the author's own footprint rather than dropping it as a "
         "thin fragment; this sentence is padded to length here entirely on purpose, yes indeed.")


def _raw(tid, *, text="", conv=None, reply_to_uid=None, rt=False, media=None, quoted=None, link=None):
    """A fuller raw-twitterapi tweet for the adapter path (carries author + createdAt so
    derive_x / tweet_to_markdown render). author.id='99' → who_id x:user:99.

    `quoted` = a nested quoted_tweet dict. Mirrors the LIVE shape: a genuine quote tweet can arrive
    with `isQuote` ABSENT/None while `quoted_tweet` is populated — so a fixture with a quoted object
    leaves isQuote unset, exactly the case a flag-gated render would drop. `link` = an expanded
    outbound url (entities.urls[].expanded_url), the shape the walk carries and the link
    dispatcher classifies."""
    t = {"id": tid, "text": text,
         "author": {"userName": "carol", "name": "Carol", "id": "99"},
         "createdAt": "Mon Jan 06 10:00:00 +0000 2026",
         "conversationId": conv or tid,
         "isRetweet": rt, "isReply": reply_to_uid is not None, "inReplyToUserId": reply_to_uid,
         "likeCount": 5, "replyCount": 1,
         "entities": {"urls": [{"expanded_url": link}]} if link else {},
         "url": f"https://x.com/carol/status/{tid}"}
    if media:
        t["extendedEntities"] = {"media": media}
    if quoted:
        t["quoted_tweet"] = quoted            # NOTE: isQuote deliberately left unset (live shape)
    return t


def _patch_fetch(monkeypatch, tweets, *, articles=False, pinned=None):
    """Stub the X session and the one Originals request, so a page runs offline.

    `fetch_user_originals` is the stub point: it is the one request `walk_page` makes, and
    everything downstream of it (filter, stitch, render, hash, write) is what these tests
    are about. The stubbed page holds the whole timeline and ENDS it, which is what every test
    below is about; paging is the timeline scraper's (`test_partial_x_walks.py`).

    `articles=False` also stubs out article DETECTION, since these fixtures carry no X-Articles and
    the real detector would scan every fixture's entities on every test."""
    from pipeline.ingestion import x_render as xt
    monkeypatch.setattr(sessions, "x_session", lambda referer: object())
    monkeypatch.setattr(x_profile, "fetch_user_profile", lambda client, h: {
        "user_id": "99", "handle": h, "display_name": h, "bio": "", "website": "",
        "bio_urls": [], "verified": False, "followers": 1})
    monkeypatch.setattr(
        x_originals, "fetch_user_originals",
        lambda client, uid, **kw: x_originals.OriginalsWalk(
            [dict(t) for t in tweets] + ([dict(pinned)] if pinned else []), True,
            dict(pinned) if pinned else None))
    if not articles:
        monkeypatch.setattr(xt, "_article_tweet_id", lambda t: None)


def _sync(conn, *, handle="carol", author_name=None) -> dict:
    """One page through the timeline scraper's two halves: `walk_page` reads and renders it,
    `write_page` writes it in one transaction. Returns the page's summary."""
    page = pages.walk_page(conn, pages.PageKit(), user_id="99", handle=handle,
                        author_name=author_name)
    with writes.write_tx(conn, site="test.page"):
        pages.write_page(page)
    return page.summary


def test_self_thread_builds_one_atom(conn, monkeypatch):
    _patch_fetch(monkeypatch, [
        _raw("10", text="1/ agents compose tools", conv="10"),
        _raw("11", text="2/ into larger systems", conv="10", reply_to_uid="99"),  # self-reply
        _raw("12", text="3/ that is the thesis", conv="10", reply_to_uid="99"),
    ])
    out = _sync(conn, handle="carol", author_name="Carol")
    assert out["added"] == 1 and out["threads"] == 1

    atom = conn.execute("SELECT * FROM atoms WHERE atom_id='xprofile:10'").fetchone()
    assert atom is not None
    assert atom["source_type"] == "x" and atom["entry_mode"] == "oracle-footprint"
    assert atom["who_id"] == "x:user:99" and atom["what_kind"] == "opinion"
    body = schema.read_body(conn, "xprofile:10")
    assert "agents compose tools" in body and "into larger systems" in body and "that is the thesis" in body
    who = conn.execute("SELECT who_id FROM atoms WHERE atom_id='xprofile:10'").fetchone()["who_id"]
    assert who == "x:user:99"


# ── thread tails are OWED, not fetched ──────────────
#
# Originals carries a thread's root and one self-reply. The rest is a `TweetDetail` read from a
# bucket shared with the bookmark paths, so the walk writes what it holds, marks the thread owed,
# and the tail queue upgrades it later. Coverage never waits on that bucket.

def _payload(conn, atom_id):
    import json
    return json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id=?",
                                   (atom_id,)).fetchone()["payload"])


def _thread():
    return [_raw("10", text="1/ agents compose tools", conv="10"),
            _raw("11", text="2/ into larger systems", conv="10", reply_to_uid="99")]


def _no_thread_fetch(monkeypatch):
    """The walk must make NO conversation read — that is the queue's job alone."""

    def _boom(*a, **k):
        raise AssertionError("the footprint walk fetched a thread tail inline")

    monkeypatch.setattr(x_conversation, "fetch_thread_page", _boom)
    monkeypatch.setattr(x_conversation, "fetch_conversation", _boom)


def test_a_thread_is_written_partial_and_owed(conn, monkeypatch):
    _patch_fetch(monkeypatch, _thread())
    _no_thread_fetch(monkeypatch)

    out = _sync(conn, handle="carol")

    payload = _payload(conn, "xprofile:10")
    assert payload["body_state"] == "partial"            # has a body; short of its tail
    assert payload["thread_tail"] == tail_state.TAIL_OWED
    assert payload["thread_root"] == "10" and payload["thread_author"] == "99"
    assert payload["thread_ids"] == ["10", "11"]
    assert out["threads_owed"] == 1


def test_a_solo_post_is_complete_and_owes_nothing(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("20", text=_LONG)])
    _no_thread_fetch(monkeypatch)

    out = _sync(conn, handle="carol")

    payload = _payload(conn, "xprofile:20")
    assert payload["body_state"] == "complete"
    assert "thread_tail" not in payload
    assert out["threads_owed"] == 0


def test_the_pin_in_range_with_replies_is_owed_though_it_shows_no_self_reply(conn, monkeypatch):
    """Originals ships the pin alone, so its thread never shows the self-reply
    that marks every other thread, so owe it when it has replies."""
    pin = _raw("30", text=_LONG)                         # replyCount 1 in the fixture
    # An older post on the same page puts the pin inside the page's range, so this page writes it.
    _patch_fetch(monkeypatch, [_raw("20", text=_LONG + " older")], pinned=pin)

    out = _sync(conn, handle="carol")

    assert _payload(conn, "xprofile:30")["thread_tail"] == tail_state.TAIL_OWED
    assert out["threads_owed"] == 1


def test_a_fetched_tail_is_never_downgraded_by_a_walk_that_sees_less(conn, monkeypatch):
    """The forward refresh re-reads a page of overshoot every pass. Without this, every thread in
    it would flip back to owed, shorter, and rewritten — then again after the queue refilled it."""
    import json
    _patch_fetch(monkeypatch, _thread())
    _sync(conn, handle="carol")
    payload = _payload(conn, "xprofile:10")
    payload.update(thread_tail=tail_state.TAIL_FETCHED, thread_ids=["10", "11", "12", "13"],
                   body_state="complete")
    conn.execute("UPDATE atoms SET payload=?, raw_hash='the-full-thread' WHERE atom_id=?",
                 (json.dumps(payload), "xprofile:10"))
    conn.commit()

    out = _sync(conn, handle="carol")

    after = _payload(conn, "xprofile:10")
    assert after["thread_tail"] == tail_state.TAIL_FETCHED and after["body_state"] == "complete"
    assert conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='xprofile:10'"
                        ).fetchone()["raw_hash"] == "the-full-thread"
    assert out["threads_owed"] == 0 and out["added"] == 0


def test_a_self_reply_the_stored_tail_lacks_reopens_it(conn, monkeypatch):
    """Thread growth: the author added a continuation after the tail was fetched."""
    import json
    _patch_fetch(monkeypatch, _thread())
    _sync(conn, handle="carol")
    payload = _payload(conn, "xprofile:10")
    payload.update(thread_tail=tail_state.TAIL_FETCHED, thread_ids=["10"], body_state="complete")
    conn.execute("UPDATE atoms SET payload=?, raw_hash='older' WHERE atom_id=?",
                 (json.dumps(payload), "xprofile:10"))
    conn.commit()

    out = _sync(conn, handle="carol")

    assert _payload(conn, "xprofile:10")["thread_tail"] == tail_state.TAIL_OWED
    assert out["threads_owed"] == 1


# ── the thread-tail queue (S4b) ─────────────────────────────────────────────────

def _owe_a_thread(conn, monkeypatch, root="10"):
    """Walk one thread into the store the way the walk leaves it: root + one self-reply, owed."""
    _patch_fetch(monkeypatch, [
        _raw(root, text="1/ agents compose tools", conv=root),
        _raw(str(int(root) + 1), text="2/ into larger systems", conv=root, reply_to_uid="99")])
    _sync(conn, handle="carol")
    assert _payload(conn, f"xprofile:{root}")["thread_tail"] == tail_state.TAIL_OWED


def _tails_serve(monkeypatch, answers):
    calls = []
    monkeypatch.setattr(sessions, "x_session", lambda url: {})

    def fetch(root, *args, **kwargs):
        calls.append(root)
        answer = answers[root]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(x_conversation, "fetch_thread_page", fetch)
    return calls


def _full_thread(root="10"):
    r = int(root)
    return [{**_raw(str(i), text=f"{i-r+1}/ that is the thesis", conv=root,
                   reply_to_uid="99" if i > r else None), "replyCount": 0,
             "inReplyToStatusId": str(i-1) if i > r else None} for i in range(r, r+3)]


def _thread_loop(conn, **kwargs):
    from pipeline.dedup_store import init_schema
    from pipeline.oracles.sources.loops import x_threads_loop
    init_schema(conn)
    roster.upsert_oracle(conn, "x:user:99", name="Carol")
    loop = x_threads_loop(meter=lambda: None, **kwargs)
    loop.ctx.cache.update(session={})
    return loop


def test_a_filled_tail_rewrites_the_root_complete_once(conn, monkeypatch):
    from pipeline.worker.quota_loop.retries import Wait
    _owe_a_thread(conn, monkeypatch)
    calls = _tails_serve(monkeypatch, {"10": _full_thread()})
    loop = _thread_loop(conn)
    before = conn.execute("SELECT version FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0]
    assert loop.step(conn).result == "fetched"
    payload = _payload(conn, "xprofile:10")
    assert payload["thread_tail"] == tail_state.TAIL_FETCHED and payload["body_state"] == "complete"
    assert payload["thread_ids"] == ["10", "11", "12"] and payload["thread_len"] == 3
    assert "that is the thesis" in schema.read_body(conn, "xprofile:10")
    assert conn.execute("SELECT version FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0] == before + 1
    assert isinstance(loop.step(conn), Wait) and calls == ["10"]


def test_a_filled_tail_and_its_completion_commit_together(conn, monkeypatch):
    """The rewritten thread is a claim the tail is fetched, so it never lands without its
    completion: a completion that fails takes the rewrite back, and the thread stays owed."""
    _owe_a_thread(conn, monkeypatch)
    _tails_serve(monkeypatch, {"10": _full_thread()})
    loop = _thread_loop(conn)
    before = conn.execute("SELECT version, raw_hash FROM atoms WHERE atom_id='xprofile:10'"
                          ).fetchone()

    def refused(*_a, **_kw):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(thread_state, "finish", refused)
    loop.step(conn)

    assert tuple(conn.execute("SELECT version, raw_hash FROM atoms WHERE atom_id='xprofile:10'"
                              ).fetchone()) == tuple(before)
    assert _payload(conn, "xprofile:10")["thread_tail"] == tail_state.TAIL_OWED


def test_a_gone_conversation_is_final_and_never_retried(conn, monkeypatch):
    from pipeline.worker.quota_loop.retries import Wait
    _owe_a_thread(conn, monkeypatch)
    calls = _tails_serve(monkeypatch, {"10": []})
    loop = _thread_loop(conn)
    assert loop.step(conn).result == "gone"
    assert isinstance(loop.step(conn), Wait) and calls == ["10"]
    payload = _payload(conn, "xprofile:10")
    assert payload["thread_tail"] == tail_state.TAIL_GONE and payload["body_state"] == "partial"


def test_a_failed_read_leaves_it_owed_for_the_next_pass(conn, monkeypatch):
    _owe_a_thread(conn, monkeypatch)
    calls = _tails_serve(monkeypatch, {"10": RuntimeError("bad page")})
    loop = _thread_loop(conn)
    loop.step(conn)
    assert calls == ["10"]
    assert _payload(conn, "xprofile:10")["thread_tail"] == tail_state.TAIL_OWED
    assert _payload(conn, "xprofile:10")["thread_retry"]["failures"] == 1


def test_a_spent_window_stops_before_another_thread_request(conn, monkeypatch):
    from pipeline.worker.quota_loop.retries import Wait
    _owe_a_thread(conn, monkeypatch)
    calls = _tails_serve(monkeypatch, {})
    loop = _thread_loop(conn, clock=lambda: 1000)
    loop._meter = lambda: (0, 1900)
    out = loop.step(conn)
    assert isinstance(out, Wait) and out.cause == "quota_spent"
    assert calls == [] and _payload(conn, "xprofile:10")["thread_tail"] == tail_state.TAIL_OWED


def test_a_tail_that_adds_nothing_still_leaves_the_queue(conn, monkeypatch):
    _owe_a_thread(conn, monkeypatch)
    page = [_raw("10", text="1/ agents compose tools", conv="10"),
            {**_raw("11", text="2/ into larger systems", conv="10", reply_to_uid="99"), "replyCount": 0}]
    _tails_serve(monkeypatch, {"10": page})
    loop = _thread_loop(conn)
    before = conn.execute("SELECT version FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0]
    assert loop.step(conn).result == "fetched"
    assert conn.execute("SELECT version FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0] == before
    payload = _payload(conn, "xprofile:10")
    assert payload["thread_tail"] == tail_state.TAIL_FETCHED and payload["body_state"] == "complete"


def test_the_next_walk_does_not_undo_a_filled_tail(conn, monkeypatch):
    _owe_a_thread(conn, monkeypatch)
    _tails_serve(monkeypatch, {"10": _full_thread()})
    loop = _thread_loop(conn)
    assert loop.step(conn).result == "fetched"
    _patch_fetch(monkeypatch, _full_thread()[:2])
    out = _sync(conn, handle="carol")
    assert out["threads_owed"] == 0 and out["added"] == 0
    assert _payload(conn, "xprofile:10")["thread_tail"] == tail_state.TAIL_FETCHED


def test_long_thread_resumes_across_loops_without_intermediate_rewrites(conn, monkeypatch):
    _owe_a_thread(conn, monkeypatch)
    pages = {}
    for i in range(10, 18):
        page = [{**_raw(str(j), text=f"part {j}", conv="10", reply_to_uid="99" if j > 10 else None),
                 "inReplyToStatusId": str(j-1), "replyCount": 1} for j in range(i, min(i+2, 18))]
        pages[str(i)] = page
    calls = _tails_serve(monkeypatch, pages)
    before = conn.execute("SELECT version FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0]
    for i in range(8):
        loop = _thread_loop(conn)
        out = loop.step(conn)
        assert out.result == ("fetched" if i == 7 else "walking")
        version = conn.execute("SELECT version FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0]
        assert version == before + int(i == 7)
    assert calls == [str(i) for i in range(10, 18)]
    assert _payload(conn, "xprofile:10")["thread_ids"] == calls


def test_footprint_concurrent_single_writer(conn, monkeypatch):
    """40 standalone posts through the REAL producer pool. Every one becomes a durable atom (no
    lost/duped atoms), producers ran on MANY threads, and — since a producer touching `conn` would
    raise SQLite's cross-thread error and drop its atom → count short — a full, dup-free count IS the
    single-writer guarantee."""
    import threading

    from pipeline.kb import vision

    _patch_fetch(monkeypatch, [_raw(str(i), text=f"{_LONG} number {i}") for i in range(1, 41)])

    producer_threads: set = set()

    def _attach(norm, cache):
        producer_threads.add(threading.get_ident())      # prove producers ran on >1 thread
        time.sleep(0.005)                                 # long enough for the pool to spread
    monkeypatch.setattr(vision, "attach_cached", _attach)

    out = _sync(conn, handle="carol")

    assert len(producer_threads) > 1                      # concurrency was REAL, not serialized on one
    assert out["added"] == 40 and out["skipped"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE atom_id LIKE 'xprofile:%'").fetchone()[0] == 40
    assert conn.execute("SELECT COUNT(*) FROM bodies WHERE atom_id LIKE 'xprofile:%'"
                        ).fetchone()[0] == 40


def test_rts_and_replies_to_others_excluded(conn, monkeypatch):
    _patch_fetch(monkeypatch, [
        _raw("1", text=_LONG, conv="1"),
        _raw("2", text="@rival wrong", conv="9", reply_to_uid="rival-id"),  # reply-to-OTHER → dropped
        _raw("3", text="a retweet", rt=True),                                # RT → dropped
    ])
    out = _sync(conn, handle="carol")
    assert out["added"] == 1
    assert {r["atom_id"] for r in conn.execute("SELECT atom_id FROM atoms")} == {"xprofile:1"}


def test_a_short_post_is_kept(conn, monkeypatch):
    """No length bar: a short post with a product link is how an
    Oracle recommends a tool. Its homepage is owed only for its card (`kb/link_cards.py`)."""
    _patch_fetch(monkeypatch, [_raw("42", text="Skin in the game."),
                               _raw("44", text="been using it daily", link="https://cursor.com")])
    out = _sync(conn, handle="nntaleb")
    assert out["added"] == 2
    assert {r[0] for r in conn.execute("SELECT atom_id FROM atoms")} == {"xprofile:42",
                                                                         "xprofile:44"}
    assert [e["url"] for e in _payload(conn, "xprofile:44")["links_owed"]] == ["https://cursor.com"]


# ── photos are read on their own loop, never by a page ─────────
#
# A page renders the reads the image cache holds and owes the rest (`footprint_photos`) in its
# own transaction; the Oracle photo source (`oracles/sources/x_photos.py`) reads them and
# renders the post once more.

_CHART = "https://pbs.twimg.com/media/chart.jpg"


def _chart_media():
    return [{"type": "photo", "media_url_https": _CHART}]


def _reads(monkeypatch, read=lambda url: ("a bar chart of benchmark scores", "chart")):
    """Stub the one image reader; `read` returns (text, kind, substance), or None for a failure.
    Returns the urls it was asked for."""
    from pipeline import chart_read
    calls = []

    def fake(url, context=""):
        calls.append(url)
        got = read(url)
        return chart_read.MediaRead(*got) if got else None
    monkeypatch.setattr(chart_read, "read_image", fake)
    return calls


def _photo_loop(conn):
    from pipeline.oracles.sources.loops import x_photos_loop
    roster.upsert_oracle(conn, "x:user:99", name="Carol")
    return x_photos_loop(meter=lambda: None)


def _owed_photos(conn):
    return {r["atom_id"]: dict(r) for r in conn.execute("SELECT * FROM oracle_x_photos")}


def _body(conn, atom_id):
    return schema.read_body(conn, atom_id) or ""


def _version(conn, atom_id):
    row = conn.execute("SELECT version FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()
    return row[0] if row else None


def test_a_page_reads_no_photo_and_owes_them_in_its_own_transaction(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, media=_chart_media())])
    calls = _reads(monkeypatch)
    kit = pages.PageKit()

    page = pages.walk_page(conn, kit, user_id="99", handle="carol")
    with pytest.raises(RuntimeError):
        with writes.write_tx(conn, site="test.page"):
            pages.write_page(page)
            raise RuntimeError("the frontier write failed")
    assert _version(conn, "xprofile:7") is None and _owed_photos(conn) == {}

    page = pages.walk_page(conn, kit, user_id="99", handle="carol")
    with writes.write_tx(conn, site="test.page"):
        pages.write_page(page)
    assert calls == [], "a page never waits on a photo read"
    assert page.summary["added"] == 1 and page.summary["photos_owed"] == 1
    assert "*Image:*" not in _body(conn, "xprofile:7")
    owed = _owed_photos(conn)["xprofile:7"]
    assert owed["who_id"] == "x:user:99" and owed["failures"] is None
    assert [t["id"] for t in json.loads(owed["grp"])] == ["7"]


def test_a_photo_the_cache_holds_is_attached_at_page_time(conn, monkeypatch):
    from opyt_core.paths import opyt_home
    from pipeline import chart_read
    from pipeline.image_cache import save_image_cache
    save_image_cache(opyt_home(), {_CHART: chart_read.MediaRead(
        "a bar chart of benchmark scores", "chart").to_cache()})
    _patch_fetch(monkeypatch, [_raw("7", text="the data speaks for itself", media=_chart_media())])
    calls = _reads(monkeypatch)

    out = _sync(conn, handle="carol")

    assert calls == [] and out["added"] == 1 and out["photos_owed"] == 0
    assert "*Image:* a bar chart of benchmark scores" in _body(conn, "xprofile:7")
    assert _payload(conn, "xprofile:7")["media_substance"] is True
    assert _owed_photos(conn) == {}


def test_the_photo_loop_reads_renders_and_rewrites_once(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, media=_chart_media())])
    calls = _reads(monkeypatch)
    _sync(conn, handle="carol")
    first = _version(conn, "xprofile:7")
    payload_keys = set(_payload(conn, "xprofile:7"))
    loop = _photo_loop(conn)

    assert loop.step(conn).result == "rewritten"
    assert calls == [_CHART]
    assert "*Image:* a bar chart of benchmark scores" in _body(conn, "xprofile:7")
    assert _version(conn, "xprofile:7") == first + 1
    assert _owed_photos(conn) == {}, "a settled row goes, with the posts it carried"
    assert set(_payload(conn, "xprofile:7")) == payload_keys, "nothing new on atoms.payload"
    assert isinstance(loop.step(conn), Wait)

    # The next walk over the same post renders the read from the cache: nothing changes.
    out = _sync(conn, handle="carol")
    assert out["skipped"] == 1 and out["photos_owed"] == 0
    assert _version(conn, "xprofile:7") == first + 1 and calls == [_CHART]


def test_an_oracle_forgotten_while_its_photos_were_read_gets_nothing_written(conn, monkeypatch):
    """The photo loop's write checks the roster in its own transaction.
    A forget that lands while the photos are read refuses the write, uncharged."""
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, media=_chart_media())])
    _sync(conn, handle="carol")
    first = _version(conn, "xprofile:7")
    loop = _photo_loop(conn)

    calls = _reads(monkeypatch)
    from pipeline.kb import vision
    real = vision.read_group_photos

    def forget_during(groups, cache, **kw):
        out = real(groups, cache, **kw)
        conn.execute("DELETE FROM oracles")
        conn.commit()
        return out
    monkeypatch.setattr(vision, "read_group_photos", forget_during)

    assert loop.step(conn).result == "changed"
    assert calls == [_CHART]
    assert _version(conn, "xprofile:7") == first, "nothing lands after the forget"


def _outer_sites(monkeypatch) -> list[str]:
    """The site of every outermost `write_tx`, one per commit, in order."""
    sites, begin = [], writes._begin

    def counted(conn, site):
        sites.append(site)
        return begin(conn, site)
    monkeypatch.setattr(writes, "_begin", counted)
    return sites


def test_the_photo_loop_lands_each_post_alone_and_a_changed_one_skips_only_itself(
        conn, monkeypatch):
    other = "https://pbs.twimg.com/media/other.jpg"
    _patch_fetch(monkeypatch, [
        _raw("7", text=_LONG, media=_chart_media()),
        _raw("9", text=_LONG + " more", media=[{"type": "photo", "media_url_https": other}])])
    _reads(monkeypatch)
    _sync(conn, handle="carol")
    first = _version(conn, "xprofile:7")
    produce = fp._GroupWriter.produce

    def produce_while_a_page_rewrites_9(self, group, **kw):
        """A page rewrites post 9 while the unit renders it, before either post lands."""
        res = produce(self, group, **kw)
        if kw.get("atom_id") == "xprofile:9":
            conn.execute("UPDATE atoms SET raw_hash='rewritten meanwhile' "
                         "WHERE atom_id='xprofile:9'")
            conn.commit()
        return res
    monkeypatch.setattr(fp._GroupWriter, "produce", produce_while_a_page_rewrites_9)
    loop = _photo_loop(conn)
    sites = _outer_sites(monkeypatch)

    assert loop.step(conn).result == "rewritten"
    assert sites.count("oracle_photos.write") == 2, "one transaction per post"
    assert _version(conn, "xprofile:7") == first + 1
    assert conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='xprofile:9'").fetchone()[0] \
        == "rewritten meanwhile", "the changed post is left alone"
    assert set(_owed_photos(conn)) == {"xprofile:9"}, "and stays owed for the next unit"


def test_a_page_still_commits_all_its_posts_in_one_transaction(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, media=_chart_media()),
                               _raw("9", text=_LONG + " more", media=_chart_media())])
    _reads(monkeypatch)
    page = pages.walk_page(conn, pages.PageKit(), user_id="99", handle="carol")
    sites = _outer_sites(monkeypatch)
    with writes.write_tx(conn, site="test.page"):
        pages.write_page(page)
    assert sites == ["test.page"]
    assert page.summary["added"] == 2 and set(_owed_photos(conn)) == {"xprofile:7", "xprofile:9"}


def test_a_failed_read_stays_owed_and_retries_by_the_rules(conn, monkeypatch):
    from pipeline.worker.quota_loop import retries
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, media=_chart_media())])
    _reads(monkeypatch, read=lambda url: None)
    _sync(conn, handle="carol")
    first = _version(conn, "xprofile:7")
    loop = _photo_loop(conn)

    assert loop.step(conn).result == "images_pending"
    owed = _owed_photos(conn)["xprofile:7"]
    assert owed["failures"] == 1 and owed["status"] == retries.BLIP and owed["retry_at"]
    assert _version(conn, "xprofile:7") == first, "the post stays as the page wrote it"
    assert isinstance(loop.step(conn), Wait), "not offered again before its retry"

    conn.execute("UPDATE oracle_x_photos SET failures=?, retry_at=NULL",
                 (retries.GIVE_UP_AFTER,))
    conn.commit()
    assert isinstance(loop.step(conn), Wait), "a given-up row is no longer read"


def test_a_free_install_reads_no_photo_and_charges_nothing(conn, monkeypatch):
    """With no model to read photos (a free install has no key) the loop offers nothing, so a
    row is never charged toward giving up before a key arrives (a free home once failed
    3,260 photo reads in 15 minutes)."""
    _patch_fetch(monkeypatch, [_raw("8", text="look", media=_chart_media())])
    calls = _reads(monkeypatch)
    _sync(conn, handle="carol")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    loop = _photo_loop(conn)

    assert isinstance(loop.step(conn), Wait)
    assert calls == [] and not _owed_photos(conn)["xprofile:8"]["failures"]

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert loop.step(conn).result == "rewritten"


def test_a_short_post_with_an_unread_photo_is_written_then_rendered_with_it(
        conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("8", text="look", media=_chart_media())])
    _reads(monkeypatch)
    out = _sync(conn, handle="carol")
    assert out["added"] == 1 and out["photos_owed"] == 1
    assert "*Image:*" not in _body(conn, "xprofile:8")

    assert _photo_loop(conn).step(conn).result == "rewritten"
    assert _payload(conn, "xprofile:8")["media_substance"] is True
    assert "*Image:* a bar chart of benchmark scores" in _body(conn, "xprofile:8")
    assert _owed_photos(conn) == {}


def test_a_short_post_whose_photo_is_not_a_chart_stays_as_written(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("8", text="look", media=_chart_media())])
    _reads(monkeypatch, read=lambda url: ("", "not_chart"))
    _sync(conn, handle="carol")
    first = _version(conn, "xprofile:8")

    assert _photo_loop(conn).step(conn).result == "settled"
    assert first is not None and _version(conn, "xprofile:8") == first
    assert _owed_photos(conn) == {}


@pytest.mark.parametrize("pending, tail_owed, stored, want", [
    (True, False, True, "images_pending"),
    (True, True, False, "images_pending"),
    (False, True, True, "settled"),        # the tail lane renders the photos
    (False, False, False, "settled"),      # forgotten, or never stored: nothing to render on
    (False, False, True, "render"),
])
def test_what_a_post_comes_to_once_its_photos_were_tried(pending, tail_owed, stored, want):
    from pipeline.oracles.sources import x_photos
    assert x_photos.step(pending, tail_owed, stored) == want


def test_a_thread_with_photos_is_rewritten_once_for_both(conn, monkeypatch):
    """The tail lane waits on the owed photos; the photo loop only fills the cache for a thread
    whose tail is owed, and the tail's rewrite renders them: one rewrite, not two."""
    _patch_fetch(monkeypatch, [
        _raw("10", text="1/ agents compose tools", conv="10", media=_chart_media()),
        _raw("11", text="2/ into larger systems", conv="10", reply_to_uid="99")])
    _reads(monkeypatch)
    _sync(conn, handle="carol")
    first = _version(conn, "xprofile:10")
    thread = _full_thread()
    thread[0]["extendedEntities"] = {"media": _chart_media()}
    calls = _tails_serve(monkeypatch, {"10": thread})
    tails = _thread_loop(conn)
    photos = _photo_loop(conn)

    assert isinstance(tails.step(conn), Wait) and calls == [], "the tail waits for the photos"
    assert photos.step(conn).result == "settled"
    assert _version(conn, "xprofile:10") == first
    assert tails.step(conn).result == "fetched"
    assert "*Image:* a bar chart of benchmark scores" in _body(conn, "xprofile:10")
    assert _version(conn, "xprofile:10") == first + 1
    assert _owed_photos(conn) == {}


def test_quote_tweet_inner_post_is_extracted(conn, monkeypatch):
    # A quote tweet's meaning lives in what it QUOTES — the author's bare reaction is
    # decontextualized (and mis-attributable) without the quoted post. The live profile fetch nulls
    # `isQuote` but populates `quoted_tweet`, so the render must gate on the OBJECT, not the flag.
    quoted = {"id": "555", "text": "Kimi K3 weights will be open in the coming days.",
              "author": {"userName": "yulun_du", "name": "Yulun Du"}, "entities": {}}
    _patch_fetch(monkeypatch, [_raw("30", text="Hurray!! cannot wait to play.", quoted=quoted)])
    out = _sync(conn, handle="carol")
    assert out["added"] == 1
    body = schema.read_body(conn, "xprofile:30")
    # The atom carries BOTH the author's reaction AND the quoted author + quoted substance.
    assert "Hurray" in body
    assert "yulun_du" in body and "Kimi K3 weights will be open" in body
    # …and the structural flag does not lie (derived from the object, not the nulled isQuote).
    import json
    payload = json.loads(conn.execute(
        "SELECT payload FROM atoms WHERE atom_id='xprofile:30'").fetchone()["payload"])
    assert payload["is_quote"] is True


def test_quoted_article_body_arrives_with_the_timeline(conn, monkeypatch):
    """A quote OF an X-Article renders the quoted body, not its teaser.

    No second paid `/twitter/article` call is needed per quoted article. The timeline walk asks for
    `withArticleRichContentState`, and `x.records.normalize_tweet` recurses into
    `quoted_status_result` carrying the whole `article` node with it — so the body is already here
    and no fetch is needed. Losing it would leave the atom holding a reaction plus a title, and the
    long-form it points at would never enter the corpus."""
    quoted = {"id": "888", "text": "https://t.co/x",
              "author": {"userName": "satyanadella", "name": "Satya Nadella"},
              "entities": {"urls": [{"expanded_url": "http://x.com/i/article/999"}]},
              "article": {"article_results": {"result": {
                  "title": "The Reverse Information Paradox",
                  "content_state": {"blocks": [
                      {"type": "unstyled",
                       "text": "Kenneth Arrow described a paradox in the IP market."}]}}}}}
    _patch_fetch(monkeypatch, [_raw("40", text="Protect the loop.", quoted=quoted)], articles=True)
    out = _sync(conn, handle="ccatalini")
    assert out["added"] == 1
    body = schema.read_body(conn, "xprofile:40")
    assert "Protect the loop" in body                                   # the author's reaction
    assert "Kenneth Arrow described a paradox in the IP market" in body  # the quoted BODY


def test_content_hash_idempotent(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("1", text=_LONG, conv="1")])
    _sync(conn, handle="carol")
    out = _sync(conn, handle="carol")   # unchanged body → no rewrite
    assert out["added"] == 0 and out["skipped"] == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:1'").fetchone()[0] == 1


def test_engages_lands_on_write_and_fills_an_older_post_in_place(conn, monkeypatch):
    quoted = {"id": "500", "text": "a claim", "author": {"id": "7", "userName": "bob"}}
    _patch_fetch(monkeypatch, [_raw("1", text=_LONG, quoted=quoted)])
    _sync(conn, handle="carol")

    def engages():
        return json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id='xprofile:1'")
                          .fetchone()[0]).get("engages")

    assert [(e["kind"], e["who"]) for e in engages()] == [("quote", "x:user:7")]
    conn.execute("UPDATE atoms SET payload = json_remove(payload, '$.engages')")
    out = _sync(conn, handle="carol")   # unchanged body: no rewrite, filled in place
    assert out["skipped"] == 1 and [e["handle"] for e in engages()] == ["bob"]


def test_all_qualifying_groups_are_stored(conn, monkeypatch):
    _patch_fetch(monkeypatch, [_raw("1", text=_LONG + " one", conv="1"),
                               _raw("2", text=_LONG + " two", conv="2"),
                               _raw("3", text=_LONG + " three", conv="3")])
    out = _sync(conn, handle="carol")
    assert out["added"] == 3
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 3


# ── Link dispatch: references become artifact atoms attributed to their own authors ──

def _walk_then_drain(conn, handle="carol"):
    """The walk queues its posts' links; the queue captures them. Both summaries."""
    out = _sync(conn, handle=handle)
    return out, drain(conn)


def _minted(conn):
    """Artifact atoms captured by link dispatch."""
    # Keyed on "not an X opinion atom" rather than `entry_mode`, because the paper tests stub
    # `atomize_paper` and their fakes do not stamp one. The Oracle's own tweets are the only
    # `source_type='x'` rows here, so everything else is something the dispatcher captured.
    return {r[0] for r in conn.execute(
        "SELECT atom_id FROM atoms WHERE source_type != 'x'")}


def test_github_reference_dispatched(conn, monkeypatch):
    # An Oracle points at a repo → the repo becomes its OWN artifact atom (attributed to the repo
    # OWNER, entry_mode 'author_referenced').
    _patch_fetch(monkeypatch, [_raw("50", text="best C++ inference repo",
                                    link="https://github.com/ggerganov/llama.cpp")])
    from pipeline.platforms.github import api as gh_ing
    from pipeline.kb import ingest_github as gh
    repo = {"name": "llama.cpp", "owner": {"login": "ggerganov"}, "language": "C++",
            "stargazers_count": 60000, "forks_count": 8000, "description": "LLM inference in C/C++",
            "topics": ["llm"], "pushed_at": "2026-01-05T00:00:00Z", "license": {"spdx_id": "MIT"},
            "html_url": "https://github.com/ggerganov/llama.cpp"}
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: repo)
    monkeypatch.setattr(gh_ing, "fetch_readme", lambda owner, name: "# llama.cpp\ninference")

    out, links = _walk_then_drain(conn)
    assert out["links_owed"] == 1 and links["kinds"] == {"github": 1}
    row = conn.execute("SELECT who_id, what_kind, entry_mode FROM atoms "
                       "WHERE atom_id='github:ggerganov/llama.cpp'").fetchone()
    assert row is not None and row["who_id"] == "github:ggerganov" and row["what_kind"] == "artifact"
    assert row["entry_mode"] == "author_referenced"
    assert "github:ggerganov/llama.cpp" in _minted(conn)      # the referenced repo was captured
    assert conn.execute(                                                     # the reacting tweet still lands
        "SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:50'").fetchone()[0] == 1


def test_paper_reference_dispatched(conn, monkeypatch):
    # An arXiv link → atomize_paper (mocked) mints the paper as author_referenced.
    _patch_fetch(monkeypatch, [_raw("51", text="must read", link="https://arxiv.org/abs/2401.00001")])
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00001", "url": "https://arxiv.org/abs/2401.00001",
             "externalIds": {"ArXiv": "2401.00001"}, "authors": []}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    # The artifact prefetch pulls the PDF, so it must be stubbed independently of atomize_paper
    # — mocking the miner does not shield the fetch.
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")
    seen_modes = []

    def fake_atomize(conn, p, *, who_id=None, entry_mode="author_referenced",
                     seen=None, **kw):                     # **kw: on_written/read_pdf/store
        seen_modes.append(entry_mode)
        schema.upsert_atom(conn, {"atom_id": "paper:arXiv:2401.00001", "source_type": "paper",
                                  "who_id": "scholar:x", "what_kind": "artifact"})
        return "paper:arXiv:2401.00001"
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)

    out, links = _walk_then_drain(conn)
    assert out["links_owed"] == 1 and links["kinds"] == {"paper": 1}
    assert seen_modes == ["author_referenced"]                              # minted as referenced, not authored
    assert "paper:arXiv:2401.00001" in _minted(conn)


def test_substack_reference_dispatched(conn, monkeypatch):
    # A referenced Substack POST → its own opinion atom (same treatment as
    # github/paper: the Oracle pointed at it, who_id is the POST's author, not the Oracle).
    _patch_fetch(monkeypatch, [_raw("52", text="great post", link="https://joe.substack.com/p/x")])
    from pipeline.kb import ingest_substack as isub

    def fake_mint(conn, url, *, entry_mode="author_referenced", seen=None, img_cache=None, **kwargs):
        assert entry_mode == "author_referenced"
        schema.upsert_atom(conn, {"atom_id": "substack:777", "source_type": "substack"})
        return "substack:777"
    monkeypatch.setattr(isub, "substack_atom_from_url", fake_mint)

    out, links = _walk_then_drain(conn)
    assert out["links_owed"] == 1 and links["kinds"] == {"substack": 1}
    assert "substack:777" in _minted(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE atom_id='xprofile:52'").fetchone()[0] == 1


def test_a_link_inside_a_quoted_post_is_captured_and_names_the_quote(conn, monkeypatch):
    """Quoting a post that links something is pointing at it. The repo is minted,
    and its referrer records the quote it came through."""
    import json
    quoted = {"id": "9", "text": "check my repo", "url": "https://x.com/bob/status/9",
              "author": {"userName": "bob", "id": "7"},
              "entities": {"urls": [{"expanded_url": "https://github.com/bob/thing"}]}}
    _patch_fetch(monkeypatch, [_raw("53", text="nice", quoted=quoted)])     # OUTER tweet carries no link
    from pipeline.platforms.github import api as gh_ing
    from pipeline.kb import ingest_github as gh
    monkeypatch.setattr(gh, "_fetch_repo", lambda owner, name: {
        "name": "thing", "owner": {"login": "bob"}, "description": "a thing",
        "html_url": "https://github.com/bob/thing", "pushed_at": "2026-01-01T00:00:00Z"})
    monkeypatch.setattr(gh_ing, "fetch_readme", lambda owner, name: "# thing")

    _, links = _walk_then_drain(conn)
    assert links["kinds"] == {"github": 1}
    refs = json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id='github:bob/thing'")
                      .fetchone()[0])["referenced_by"]
    assert refs == [{"by": "x:user:99", "handle": "carol", "post": "xprofile:53",
                     "url": "https://x.com/carol/status/53", "date": "2026-01-06",
                     "via_quote": {"by": "x:user:7", "handle": "bob",
                                   "url": "https://x.com/bob/status/9"}}]


def test_referenced_pdf_is_minted(conn, monkeypatch):
    # A raw .pdf has no scholarly id and no attested author — IRRELEVANT for a reference: who_id is
    # the artifact's, never the Oracle's, so it mints like any paper (the adapter fetches the full
    # body via the pdf url itself).
    from pipeline.kb import ingest_papers as ip
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: {
        "paperId": "url:acme.com/deck.pdf", "url": url, "openAccessPdf": {"url": url}})
    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: "body")     # the prefetch pulls the PDF now
    seen_modes = []

    def fake_atomize(conn, p, *, who_id=None, entry_mode="author_referenced",
                     seen=None, **kw):                     # **kw: on_written/read_pdf/store
        seen_modes.append(entry_mode)
        schema.upsert_atom(conn, {"atom_id": ip.paper_atom_id(p), "source_type": "paper"})
        return ip.paper_atom_id(p)
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)
    _patch_fetch(monkeypatch, [_raw("55", text="slides", link="https://acme.com/deck.pdf")])

    out, links = _walk_then_drain(conn)
    assert out["links_owed"] == 1 and links["kinds"] == {"paper": 1}
    assert seen_modes == ["author_referenced"]
    assert "paper:url:acme.com/deck.pdf" in _minted(conn)


def test_present_reference_is_counted_without_reminting(conn, monkeypatch):
    # A paper already in the store is counted as captured without fetching or minting it again.
    schema.upsert_atom(conn, {"atom_id": "paper:arXiv:2401.00001", "source_type": "paper"})  # pre-existing
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "arXiv:2401.00001", "url": "https://arxiv.org/abs/2401.00001",
             "externalIds": {"ArXiv": "2401.00001"}}
    monkeypatch.setattr(ip, "paper_from_url", lambda url, enrich=True: dict(paper))
    called = []
    monkeypatch.setattr(ip, "atomize_paper", lambda *a, **k: called.append(1))   # must NOT be called

    d = Capture(conn)
    kinds = d.dispatch([_raw("60", text="ditto", link="https://arxiv.org/abs/2401.00001")])
    assert kinds == {"paper": 1}
    assert not called                                                       # present → no re-mint
    assert "paper:arXiv:2401.00001" in _minted(conn)


def test_unlisted_paper_host_dispatched_via_deep_probe(conn, monkeypatch):
    # A Nature link never matches `_PAPER_HOSTS`, so the FAST classifier returns None. The deep
    # probe (mocked here — the network fetch itself is `classify_link_deep`'s own concern, tested
    # in test_link_router.py) finds a citation_doi meta tag and rewrites to a doi.org url that
    # `paper_from_url` mints normally, same as any other referenced paper.
    monkeypatch.setattr(link_router, "classify_link_deep",
                        lambda u: ("paper", "https://doi.org/10.1038/s41586-021-03819-2", None))
    from pipeline.kb import ingest_papers as ip
    paper = {"paperId": "DOI:10.1038/s41586-021-03819-2",
             "url": "https://doi.org/10.1038/s41586-021-03819-2",
             "externalIds": {"DOI": "10.1038/s41586-021-03819-2"}, "authors": []}
    seen_urls = []

    def fake_paper_from_url(url, enrich=True, content_type=None):
        seen_urls.append(url)                # must be the REWRITTEN doi.org url, not the nature.com one
        return dict(paper)
    monkeypatch.setattr(ip, "paper_from_url", fake_paper_from_url)

    def fake_atomize(conn, p, *, who_id=None, entry_mode="author_referenced",
                     seen=None, **kw):
        schema.upsert_atom(conn, {"atom_id": "paper:DOI:10.1038/s41586-021-03819-2",
                                  "source_type": "paper", "who_id": "scholar:x",
                                  "what_kind": "artifact"})
        return "paper:DOI:10.1038/s41586-021-03819-2"
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)

    d = Capture(conn)
    # A bare link is probed OFF the consumer, so it is queued at dispatch and written at drain.
    assert d.dispatch([_raw("80", text="huge if true", link="https://nature.com/articles/x")]) == {}
    assert d.drain() == {"paper": 1}
    # Called twice (the offline `predicted_atom_id` pre-check, then the mint itself) — same as any
    # other paper reference (see test_paper_reference_dispatched); the point here is
    # every call used the REWRITTEN doi.org url, never the original nature.com one.
    assert seen_urls and all(u == "https://doi.org/10.1038/s41586-021-03819-2" for u in seen_urls)
    assert "paper:DOI:10.1038/s41586-021-03819-2" in _minted(conn)


def test_unlisted_pdf_without_pdf_shaped_url_dispatched_via_content_type_hint(conn, monkeypatch):
    # A `/download?id=123`-style redirect that SERVES a pdf: no host match, no `.pdf` in the path,
    # no citation_doi meta tag either — the ONLY signal is the real Content-Type the deep probe
    # observed. Proves the hint actually reaches `_parse_paper_url`, not just `classify_link_deep`.
    link = "https://acme-journal.example/download?id=123"
    monkeypatch.setattr(link_router, "classify_link_deep",
                        lambda u: ("paper", link, "application/pdf; charset=binary"))
    from pipeline.kb import ingest_papers as ip
    seen_content_types = []

    real_parse = ip._parse_paper_url

    def spying_parse(url, *, content_type=None):
        seen_content_types.append(content_type)
        return real_parse(url, content_type=content_type)
    monkeypatch.setattr(ip, "_parse_paper_url", spying_parse)

    def fake_atomize(conn, p, **kw):
        aid = ip.paper_atom_id(p)
        schema.upsert_atom(conn, {"atom_id": aid, "source_type": "paper",
                                  "who_id": "scholar:x", "what_kind": "artifact"})
        return aid
    monkeypatch.setattr(ip, "atomize_paper", fake_atomize)

    monkeypatch.setattr(ip, "resolve_fulltext", lambda p: None)        # the pdf pull is not the point
    d = Capture(conn)
    assert d.dispatch([_raw("82", text="preprint", link=link)]) == {}
    assert d.drain() == {"paper": 1}
    assert "application/pdf; charset=binary" in seen_content_types
    assert f"paper:url:{link}" in _minted(conn)


_NEWS_URL = "https://www.nytimes.com/2026/01/05/tech/a-story.html"
_NEWS_BODY = ("A reported piece with real paragraphs of substance in it. " * 30).strip()


def _patch_news(monkeypatch, *, calls=None, final_url=None):
    """Stub the article fetch the way Hopper's tests do: `_fetch_article` and the feed-date rung,
    the two network arms of `prepare_article`. The deep paper probe says "not a paper"."""
    from pipeline.platforms.web import article as web_article
    from pipeline.kb import ingest_blog
    monkeypatch.setattr(link_router, "classify_link_deep", lambda u: None)

    def _fetch(url):
        if calls is not None:
            calls.append(url)
        return {"url": url, "title": "A Story", "date": "2026-01-05", "author": "Jane Doe",
                "content": _NEWS_BODY, "final_url": final_url or url}
    monkeypatch.setattr(web_article, "fetch_article", _fetch)
    monkeypatch.setattr(ingest_blog, "_feed_date_map", lambda base: {})


def test_a_news_link_becomes_an_author_referenced_article_naming_its_referrer(conn, monkeypatch):
    """Bare links the footprint never fetched become atoms. It is
    the article's own (host-attributed) atom — never the Oracle's — and it records who linked it."""
    import json
    _patch_news(monkeypatch)
    _patch_fetch(monkeypatch, [_raw("81", text=_LONG, link=_NEWS_URL)])

    _, links = _walk_then_drain(conn)
    assert links["kinds"] == {"article": 1} and links["referrers"] == 1
    row = conn.execute("SELECT who_id, entry_mode, payload FROM atoms WHERE atom_id=?",
                       ("blog:nytimes.com/2026/01/05/tech/a-story.html",)).fetchone()
    assert row["who_id"] == "blog:nytimes.com" and row["entry_mode"] == "author_referenced"
    assert json.loads(row["payload"])["referenced_by"] == [
        {"by": "x:user:99", "handle": "carol", "post": "xprofile:81",
         "url": "https://x.com/carol/status/81", "date": "2026-01-06"}]


def test_a_shortened_link_is_keyed_by_the_page_it_lands_on(conn, monkeypatch):
    """`trib.al/abc` keyed as itself was `blog:trib.al/abc` under `who_id = blog:trib.al`."""
    _patch_news(monkeypatch, final_url=_NEWS_URL)
    _patch_fetch(monkeypatch, [_raw("84", text=_LONG, link="https://trib.al/AbC123")])
    _walk_then_drain(conn)
    ids = {r[0] for r in conn.execute("SELECT atom_id FROM atoms WHERE source_type='blog'")}
    assert ids == {"blog:nytimes.com/2026/01/05/tech/a-story.html"}


@pytest.mark.parametrize("link", [
    "https://gist.github.com/carol/0123456789abcdef",      # not a github repo
    "https://chatgpt.com/share/abc-123",                    # a transcript behind a login
    "https://www.youtube.com/watch?v=abc",
    "https://x.com/i/article/1234567890",                   # an X Article: no post id to resolve
])
def test_links_that_are_not_articles_are_never_fetched(conn, monkeypatch, link):
    monkeypatch.setattr(link_router, "classify_link_deep",
                        lambda u: pytest.fail("a skipped link must cost no request"))
    monkeypatch.setattr("pipeline.platforms.web.article.fetch_article",
                        lambda u: pytest.fail("a skipped link must not be fetched"))
    d = Capture(conn)
    assert d.dispatch([_raw("85", text=_LONG, link=link)]) == {}
    assert d.drain() == {} and d.refs == []


def test_the_oracles_own_blog_is_left_to_its_blog_source(conn, monkeypatch):
    """Minted here it would be `author_referenced`, and the blog crawl would then skip it as
    present — the Oracle's own writing stuck in the wrong lane."""
    from pipeline.oracles import registry
    registry.init_state_schema(conn)
    conn.executemany("INSERT INTO oracle_sources (canonical_id, source_type, source_key) "
                     "VALUES (?, ?, ?)", [("blog:carol.dev", "x", "carol"),
                                          ("blog:carol.dev", "blog", "https://carol.dev")])
    calls = []
    _patch_news(monkeypatch, calls=calls)
    d = Capture(conn)
    d.dispatch([_raw("86", text=_LONG, link="https://www.carol.dev/posts/mine")])
    assert d.drain() == {} and calls == []


def test_a_present_article_records_its_referrer_without_a_fetch(conn, monkeypatch):
    import json
    schema.upsert_atom(conn, {"atom_id": "blog:nytimes.com/2026/01/05/tech/a-story.html",
                              "source_type": "blog", "payload": {"lastmod": ""}})
    monkeypatch.setattr(link_router, "classify_link_deep",
                        lambda u: pytest.fail("a stored article needs no probe"))
    d = Capture(conn)
    assert d.dispatch([_raw("87", text=_LONG, link=_NEWS_URL)]) == {"article": 1}
    d.drain()
    assert d.flush_references() == 1
    payload = json.loads(conn.execute("SELECT payload FROM atoms WHERE source_type='blog'")
                         .fetchone()[0])
    assert payload["lastmod"] == "" and payload["referenced_by"][0]["post"] == "xprofile:87"


def test_a_referrer_lands_once_however_many_walks_see_it(conn, monkeypatch):
    import json
    _patch_news(monkeypatch)
    _patch_fetch(monkeypatch, [_raw("88", text=_LONG, link=_NEWS_URL)])
    _walk_then_drain(conn)
    walk, again = _walk_then_drain(conn)
    assert walk["links_owed"] == 0, "a post the queue already holds is not queued twice"
    assert again["referrers"] == 0
    payload = json.loads(conn.execute("SELECT payload FROM atoms WHERE source_type='blog'")
                         .fetchone()[0])
    assert len(payload["referenced_by"]) == 1


# ── referenced X posts: one batch, written `author_referenced`, never the Oracle's own ─────────

def _other_post(tid, *, author_id="7", handle="bob", text="the post being pointed at"):
    return {"id": tid, "text": text, "url": f"https://x.com/{handle}/status/{tid}",
            "author": {"userName": handle, "name": handle.title(), "id": author_id},
            "createdAt": "Sun Jan 05 09:00:00 +0000 2026", "likeCount": 3, "replyCount": 0,
            "conversationId": tid, "entities": {}}


def _patch_x_lookup(monkeypatch, posts, calls):
    monkeypatch.setattr(sessions, "x_session", lambda page: {})

    def _by_ids(client, ids):
        calls.append(list(ids))
        return [p for p in posts if p["id"] in ids]
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids", _by_ids)


def test_linked_x_posts_are_fetched_in_one_batch_and_written_referenced(conn, monkeypatch):
    import json
    calls = []
    _patch_x_lookup(monkeypatch, [_other_post("111"), _other_post("222", handle="dan",
                                                                  author_id="8")], calls)
    d = Capture(conn)
    d.dispatch([_raw("90", text=_LONG, link="https://x.com/bob/status/111")])
    d.dispatch([_raw("91", text=_LONG, link="https://twitter.com/dan/status/222?s=20")])
    d.dispatch([_raw("92", text=_LONG, link="https://x.com/carol/status/333")])   # her own → skip
    assert d.drain() == {"x": 2}
    assert calls == [["111", "222"]], "one TweetResultsByRestIds call for every queued post"
    d.flush_references()
    row = conn.execute("SELECT who_id, entry_mode, payload FROM atoms WHERE atom_id='x:111'"
                       ).fetchone()
    assert row["who_id"] == "x:user:7" and row["entry_mode"] == "author_referenced"
    assert json.loads(row["payload"])["referenced_by"][0]["post"] == "xprofile:90"
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0, \
        "nobody saved this post — an Oracle pointed at it"
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='x:333'").fetchone()[0] == 0


def test_an_oracle_post_under_a_renamed_handle_is_still_its_own(conn, monkeypatch):
    """The url names an old handle; the fetched author is the Oracle (id 99). Not minted."""
    calls = []
    _patch_x_lookup(monkeypatch, [_other_post("444", author_id="99", handle="carol_old")], calls)
    d = Capture(conn)
    d.dispatch([_raw("93", text=_LONG, link="https://x.com/carol_old/status/444")])
    assert d.drain() == {} and calls == [["444"]]


def test_a_failed_x_lookup_writes_nothing_and_raises_nothing(conn, monkeypatch):
    monkeypatch.setattr(sessions, "x_session", lambda page: {})
    monkeypatch.setattr(x_conversation, "fetch_tweets_by_ids",
                        lambda *a: (_ for _ in ()).throw(x_rate.XRateLimited("429")))
    d = Capture(conn)
    d.dispatch([_raw("94", text=_LONG, link="https://x.com/bob/status/111")])
    assert d.drain() == {} and d.flush_references() == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE entry_mode='author_referenced'").fetchone()[0] == 0


# ── Lookback window: default + the hard 2-year ceiling ────────────────────────

def test_resolve_since_default_is_6mo():
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert windows.x_since(None, now) == now - timedelta(days=windows.X_DEFAULT_DAYS)


def test_resolve_since_clamps_beyond_2yr():
    # A 5-year request is FLOORED at the 2-year ceiling — X is an ephemeral stream, and the
    # timeline walk's own pagination depth runs out well before five years of history anyway.
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert windows.x_since(now - timedelta(days=5 * 365), now) == now - timedelta(days=windows.X_CEILING_DAYS)


def test_resolve_since_passthrough_within_ceiling():
    # A 1-year request is inside the ceiling → honored unchanged.
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    one_yr = now - timedelta(days=365)
    assert windows.x_since(one_yr, now) == one_yr


# ── the profile must account for its own wall clock ─────────────────────────────────

def test_link_queueing_is_timed(conn, monkeypatch):
    """An untimed consumer step scores zero in a profile that otherwise looks complete — and 'zero'
    is indistinguishable from 'free'. Link capture cost 378 of one walk's 449 seconds while it
    lived here; what is left of it on the consumer is the queueing, and it is still measured."""
    _patch_fetch(monkeypatch, [_raw("7", text=_LONG, link="https://github.com/foo/bar")])

    out = _sync(conn, handle="carol")

    assert "link_queue" in out["stage_seconds"] and "link_dispatch" not in out["stage_seconds"]
    assert out["stage_latency"]["link_queue"]["count"] >= 1


def test_residual_is_reported_and_never_negative(conn, monkeypatch):
    """Parent−Σ(children), surfaced so unmeasured work is a NUMBER instead of a silence. Negative
    means a child is being counted outside its declared parent — the bookkeeping lying, which would
    make the residual useless exactly when it matters."""
    _patch_fetch(monkeypatch, [_raw(str(i), text=_LONG + f" n{i}", conv=str(i)) for i in range(1, 4)])

    out = _sync(conn, handle="carol")

    assert set(out["stage_residual"]) == {"produce", "consume"}
    for parent, left in out["stage_residual"].items():
        assert left >= -1e-6, f"{parent} residual {left} < 0 — a child is timed outside its parent"


# ── Batched artifact writes: pending work and durable storage ───────────────────────────────


def test_residual_grows_when_a_child_goes_unregistered():
    """The drift property. `_STAGE_NESTING` is hand-maintained, so it WILL fall behind the code; the
    only question is which way it fails. An unregistered child must inflate its parent's residual
    ('look inside this parent'), never shrink it ('nothing to see') — over-reporting is recoverable,
    under-reporting is the exact bug this whole mechanism exists to catch."""
    totals = {"consume": 10.0, "link_queue": 4.0, "write": 2.0, "brand_new_stage": 3.0}
    assert fp._stage_residual(totals)["consume"] == 4.0     # 10 − 4 − 2, the new child NOT subtracted


# ── An unreadable account is BLOCKED, not ERROR ──────────────────────────────

def test_a_linked_x_article_that_arrived_as_its_teaser_is_not_written():
    """The refusal a hand-saved one gets: a teaser is a fragment that would read as the whole."""
    from pipeline.kb import ingest_x
    teaser = {**_other_post("555"), "article": {"title": "A Long Piece", "preview_text": "…"}}
    assert ingest_x.render_referenced_post(teaser, {}) is None


def test_two_links_that_land_on_one_article_write_it_once(conn, monkeypatch):
    """A shortener and the article itself, in two posts: both are prepared side by side and come
    back ready. One write, two referrers."""
    import json
    _patch_news(monkeypatch, final_url=_NEWS_URL)
    stores = []
    real = schema.put_body
    monkeypatch.setattr(schema, "put_body",
                        lambda conn, atom_id, text: stores.append(atom_id)
                        or real(conn, atom_id, text))
    d = Capture(conn)
    d.dispatch([_raw("95", text=_LONG, link="https://trib.al/AbC123")])
    d.dispatch([_raw("96", text=_LONG, link=_NEWS_URL)])
    assert d.drain() == {"article": 2}
    assert stores == ["blog:nytimes.com/2026/01/05/tech/a-story.html"], \
        "the second landing is present, not a second write"
    d.flush_references()
    refs = json.loads(conn.execute("SELECT payload FROM atoms WHERE source_type='blog'")
                      .fetchone()[0])["referenced_by"]
    assert {r["post"] for r in refs} == {"xprofile:95", "xprofile:96"}


def test_thread_sources_alternate(conn, monkeypatch):
    from tests.kb.saved_x_support import save
    _owe_a_thread(conn, monkeypatch)
    bookmark = _raw("100", text=_LONG)
    save(conn, bookmark)
    calls = _tails_serve(monkeypatch, {"100": [bookmark], "10": _full_thread()})
    loop = _thread_loop(conn)
    assert [loop.step(conn).source for _ in range(2)] == ["bookmark_threads", "thread_tails"]
    assert calls == ["100", "10"]


def test_forgotten_oracle_cannot_receive_a_late_thread_rewrite(conn, monkeypatch):
    _owe_a_thread(conn, monkeypatch)
    loop = _thread_loop(conn)
    before = conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0]

    def forget(*args, **kwargs):
        conn.execute("DELETE FROM oracles WHERE canonical_id='x:user:99'")
        conn.commit()
        return _full_thread()

    monkeypatch.setattr(x_conversation, "fetch_thread_page", forget)
    # Refused inside the save's transaction (`require_current`), uncharged; no purge runs.
    assert loop.step(conn).result == "changed"
    assert conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0] == before


def test_an_explicit_ask_rearms_a_given_up_thread(conn, monkeypatch):
    from pipeline.oracles.sources import scraper_tables as scraper_state
    from pipeline.oracles import asks
    from datetime import datetime, timezone
    _owe_a_thread(conn, monkeypatch)
    loop = _thread_loop(conn)
    conn.execute("UPDATE atoms SET payload=json_set(payload, '$.thread_retry', "
                 "json('{\"status\":\"blip\",\"failures\":10}'))")
    conn.commit()
    now = datetime.now(timezone.utc)
    assert thread_state.next_unit(conn, "tails", now) is None
    scraper_state.init_schema(conn)
    asks.record_ask(conn, "x:user:99", x_from=now, now=now)
    assert thread_state.next_unit(conn, "tails", now) is not None
