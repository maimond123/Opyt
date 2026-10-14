"""The timeline walked one page at a time (`pages.walk_page`): resuming below a post
id, what a page reports, what it loads, that it writes only when its caller does, and threads
whose self-replies sit on another page."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.ingestion import x_render as xt
from pipeline.kb import ingest_x_footprint as fp
from pipeline.kb import footprint_photos, schema, writes
from pipeline.platforms import sessions
from tests.ingestion.x_client_fake import FakeXClient
from pipeline.platforms.x import originals as x_originals
from pipeline.platforms.x import profile as x_profile
from pipeline.oracles import roster
from pipeline.oracles.sources.threads import tail_state
from pipeline.oracles.sources.timeline.pages import PageKit, page_current, walk_page, write_page

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
_LONG = ("A substantive take that comfortably clears the two-hundred-character naked-post floor so "
         "the substance filter keeps it as the author's own footprint rather than dropping it as a "
         "thin fragment; this sentence is padded to length here entirely on purpose, yes indeed.")


def _post(tid, *, at=None, conv=None, reply=False, text=_LONG):
    at = at or T0
    return {"id": str(tid), "text": text,
            "author": {"userName": "carol", "name": "Carol", "id": "99"},
            "createdAt": at.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "conversationId": str(conv or tid), "isRetweet": False, "isReply": reply,
            "inReplyToUserId": "99" if reply else None, "likeCount": 1, "replyCount": 0,
            "entities": {}, "url": f"https://x.com/carol/status/{tid}"}


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    footprint_photos.init_schema(c)       # a page owes its unread photos here
    yield c
    c.close()


@pytest.fixture()
def pages(monkeypatch):
    """`fetch_user_originals` serving scripted pages, one per call, recording where each began."""
    served: list = []
    calls: list = []
    monkeypatch.setattr(sessions, "x_session", lambda referer: {})
    monkeypatch.setattr(xt, "_article_tweet_id", lambda t: None)

    def fake(client, user_id, *, after, below_id=None):
        calls.append({"user_id": user_id, "below_id": below_id})
        tweets, finished, pinned = served.pop(0)
        return x_originals.OriginalsWalk([dict(t) for t in tweets], finished, pinned)

    monkeypatch.setattr(x_originals, "fetch_user_originals", fake)
    return served, calls


@pytest.fixture()
def kit():
    return PageKit()


def _payload(conn, atom_id):
    row = conn.execute("SELECT payload FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()
    return json.loads(row["payload"]) if row else None


# ── resuming below a post ─────────────────────────────────────────────────────────

def test_a_cursor_below_a_post_starts_just_under_it():
    assert x_originals.originals_cursor_below("1000") == x_originals._timeline_cursor(999, "999")


def test_the_walk_sends_the_cursor_below_the_post(monkeypatch):
    sent = []

    def fake_get(op, qid, variables, features, **kw):
        sent.append(variables.get("cursor"))
        return {"data": {"user": {"result": {"timeline_v2": {"timeline": {"instructions": []}}}}}}

    x_originals.fetch_user_originals(FakeXClient(fake_get), "99", after=T0, below_id="5000")
    assert sent == [x_originals.originals_cursor_below("5000")]


def _walk(conn, kit, **kw):
    """One page read and written, as the timeline sources write it."""
    page = walk_page(conn, kit, **kw)
    write_page(page)
    return page


# ── one page ──────────────────────────────────────────────────────────────────────

def test_a_page_is_written_and_says_where_it_stopped(conn, kit, pages):
    served, calls = pages
    served.append(([_post(30, at=T0), _post(20, at=T0 - timedelta(hours=1)),
                    _post(10, at=T0 - timedelta(hours=2))], False, None))

    page = _walk(conn, kit, user_id="99", handle="carol", below_id="40")

    assert calls == [{"user_id": "99", "below_id": "40"}]
    assert (page.oldest_id, page.oldest_at) == ("10", T0 - timedelta(hours=2))
    assert page.complete and not page.ended and page.summary["added"] == 3
    assert {r[0] for r in conn.execute("SELECT atom_id FROM atoms")} == {
        "xprofile:10", "xprofile:20", "xprofile:30"}


def test_a_page_writes_nothing_until_its_caller_writes_it(conn, kit, pages):
    """The page and its caller's frontier commit together: reading a page leaves the store as it
    was, and a transaction that refuses after the write takes the page back with it."""
    served, _calls = pages
    served.append(([_post(30, at=T0)], False, None))
    page = walk_page(conn, kit, user_id="99", handle="carol")
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0

    with pytest.raises(RuntimeError, match="frontier refused"), writes.write_tx(conn):
        write_page(page)
        raise RuntimeError("frontier refused")
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_a_post_rewritten_while_the_page_was_read_makes_the_page_stale(conn, kit, pages):
    served, _calls = pages
    served.append(([_post(30, at=T0)], False, None))
    _walk(conn, kit, user_id="99", handle="carol")
    served.append(([_post(30, at=T0, text=_LONG + " Edited.")], False, None))
    page = walk_page(conn, kit, user_id="99", handle="carol")
    assert page_current(conn, page)

    conn.execute("UPDATE atoms SET raw_hash='rewritten by the tail queue' "
                 "WHERE atom_id='xprofile:30'")
    conn.commit()
    assert not page_current(conn, page)


def test_a_post_stored_by_another_lane_while_the_page_was_read_makes_it_stale(conn, kit, pages):
    served, _calls = pages
    served.append(([_post(30, at=T0)], False, None))
    page = walk_page(conn, kit, user_id="99", handle="carol")
    conn.execute("INSERT INTO atoms (atom_id, source_type, raw_hash) "
                 "VALUES ('xprofile:30', 'x', 'h')")
    conn.commit()
    assert not page_current(conn, page)


def test_the_pin_never_marks_how_far_a_page_reached(conn, kit, pages):
    served, _calls = pages
    pin = _post(5, at=T0 - timedelta(days=400))
    served.append(([_post(30), pin, _post(20)], False, pin))

    page = _walk(conn, kit, user_id="99", handle="carol")

    assert page.oldest_id == "20"
    assert _payload(conn, "xprofile:5") is None    # the pin belongs to the page whose range holds it


def test_a_page_reads_only_its_own_rows_of_the_ledgers(conn, kit, pages, monkeypatch):
    served, _calls = pages
    served.append(([_post(30)], False, None))

    def whole_table(*a, **k):
        raise AssertionError("a page loaded a whole ledger")

    monkeypatch.setattr(schema, "load_hashes", whole_table)
    monkeypatch.setattr(schema, "load_thread_tails", whole_table)
    monkeypatch.setattr(fp, "_links_known", whole_table)
    assert _walk(conn, kit, user_id="99", handle="carol").complete


def test_a_stored_user_id_needs_no_profile_lookup(conn, kit, monkeypatch):
    schema.upsert_entity(conn, "x:user:99", name="Carol", profile={"handle": "carol"})
    roster.upsert_oracle(conn, "x:user:99", name="Carol")
    conn.commit()

    def lookup(*a, **k):
        raise AssertionError("a UserByScreenName request was made for a known account")

    monkeypatch.setattr(x_profile, "fetch_user_profile", lookup)
    assert kit.user_id(conn, "x:user:99", "carol") == "99"


def test_an_unknown_user_id_is_looked_up_once(conn, kit, monkeypatch):
    calls = []
    monkeypatch.setattr(sessions, "x_session", lambda referer: {})
    monkeypatch.setattr(x_profile, "fetch_user_profile",
                        lambda c, handle: calls.append(handle) or {"user_id": "77"})
    assert kit.user_id(conn, "x:user:1", "dave") == "77"
    assert kit.user_id(conn, "x:user:1", "dave") == "77"
    assert calls == ["dave"]


# ── a thread split across pages ───────────────────────────────────────────────────

def test_replies_on_a_newer_page_and_their_root_on_the_next_make_one_atom(conn, kit, pages):
    served, _calls = pages
    served.append(([_post(12, conv=10, reply=True, at=T0 + timedelta(seconds=2)),
                    _post(11, conv=10, reply=True, at=T0 + timedelta(seconds=1))], False, None))
    served.append(([_post(10, at=T0)], False, None))

    _walk(conn, kit, user_id="99", handle="carol")
    _walk(conn, kit, user_id="99", handle="carol", below_id="11")

    assert [r[0] for r in conn.execute("SELECT atom_id FROM atoms")] == ["xprofile:10"]
    payload = _payload(conn, "xprofile:10")
    assert payload["thread_tail"] == tail_state.TAIL_OWED              # the tail queue writes it whole
    assert (payload["thread_root"], payload["thread_author"]) == ("10", "99")


def test_new_replies_to_a_stored_post_leave_its_body_and_owe_its_thread(conn, kit, pages):
    served, _calls = pages
    served.append(([_post(10, at=T0)], False, None))
    _walk(conn, kit, user_id="99", handle="carol")
    before = conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='xprofile:10'").fetchone()[0]
    assert "thread_tail" not in _payload(conn, "xprofile:10")

    served.append(([_post(11, conv=10, reply=True, at=T0 + timedelta(hours=1))], False, None))
    _walk(conn, kit, user_id="99", handle="carol")

    assert conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='xprofile:10'"
                        ).fetchone()[0] == before
    payload = _payload(conn, "xprofile:10")
    assert (payload["thread_tail"], payload["body_state"]) == (tail_state.TAIL_OWED, "partial")
    assert (payload["thread_root"], payload["thread_author"]) == ("10", "99")
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1


def test_replies_a_fetched_thread_already_holds_change_nothing(conn, kit, pages):
    served, _calls = pages
    served.append(([_post(10, at=T0)], False, None))
    _walk(conn, kit, user_id="99", handle="carol")
    payload = _payload(conn, "xprofile:10")
    payload.update(thread_tail=tail_state.TAIL_FETCHED, thread_ids=["10", "11"], body_state="complete")
    conn.execute("UPDATE atoms SET payload=? WHERE atom_id='xprofile:10'", (json.dumps(payload),))
    conn.commit()

    served.append(([_post(11, conv=10, reply=True, at=T0 + timedelta(hours=1))], False, None))
    _walk(conn, kit, user_id="99", handle="carol")

    assert _payload(conn, "xprofile:10")["thread_tail"] == tail_state.TAIL_FETCHED


# ── the image cache ───────────────────────────────────────────────────────────────

def test_a_page_renders_reads_written_after_its_kit_was_made(conn, kit, pages):
    """The photo source fills the cache file while the timeline's kit lives for hours. A page
    rendering a stale copy would drop descriptions its stored atoms already hold, rewrite them,
    and owe the photos again."""
    from opyt_core.paths import opyt_home
    from pipeline import chart_read
    from pipeline.image_cache import save_image_cache

    url = "https://pbs.twimg.com/media/late.jpg"
    save_image_cache(opyt_home(), {url: chart_read.MediaRead("a late chart", "chart")
                                   .to_cache()})
    served, _calls = pages
    served.append(([{**_post(10), "extendedEntities": {"media": [
        {"type": "photo", "media_url_https": url}]}}], True, None))
    _walk(conn, kit, user_id="99", handle="carol")

    assert "a late chart" in schema.read_body(conn, "xprofile:10")
    assert conn.execute("SELECT COUNT(*) FROM oracle_x_photos").fetchone()[0] == 0
