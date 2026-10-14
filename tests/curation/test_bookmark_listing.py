"""The Bookmarks list is walked once: pages feed saved-item work and author signals, and the
counts publish only from a complete walk on the six-hour clock."""

import json
from datetime import datetime, timedelta, timezone

from types import SimpleNamespace

import pytest

from pipeline.curation import bookmark_listing, clock, listings, saved_items, walks, x_walks
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.platforms import sessions
from pipeline.platforms.x import collections as x_collections
from pipeline.platforms.collection_page import Page
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Context, Retry

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)


def bookmarks_source():
    return walks.WalkSource(bookmark_listing.Bookmarks(), x_walks.XAccount())


@pytest.fixture
def conn(tmp_path):
    db = schema.connect(tmp_path / "opyt.db")
    listings.init_schema(db)
    saved_items.init_schema(db)
    yield db
    db.close()


class Feed:
    """X's Bookmarks endpoint as pages keyed by the cursor that asks for them."""

    def __init__(self, monkeypatch, viewer="me", connected=True):
        self.pages: dict = {}
        self.viewer = viewer
        self.requests: list = []
        monkeypatch.setattr(connections, "_connected_x", lambda: connected)
        monkeypatch.setattr(sessions, "x_session",
                            lambda referer: SimpleNamespace(viewer_id=lambda: self.viewer))
        monkeypatch.setattr(x_collections, "bookmark_page", self.page)

    def serve(self, *pages):
        """Each entry is a list of posts; the last page ends the walk."""
        self.pages = {}
        for number, posts in enumerate(pages):
            cursor = None if number == 0 else f"c{number}"
            last = number == len(pages) - 1
            records = tuple((p["id"], p) for p in posts)
            self.pages[cursor] = Page(records, None if last else f"c{number + 1}",
                                      tuple(k for k, _ in records), ended=last)

    def page(self, session, cursor=None):
        self.requests.append(cursor)
        return self.pages[cursor]


def bm(tid, uid, name="A", handle="a", site=None, likes=0):
    return {"id": tid, "likeCount": likes,
            "author": {"id": uid, "userName": handle, "name": name, "site": site}}


def step(conn, source, now=NOW, ctx=None):
    unit = source.next_unit(conn, now)
    if unit is None:
        return None
    return source.record(conn, unit, source.fetch(conn, unit, ctx or Context("x:Bookmarks")))


def walk(conn, source, now=NOW):
    """Run units until the source has nothing owed; the outcome statuses in order."""
    out = []
    while (outcome := step(conn, source, now)) is not None:
        out.append(outcome.status)
    return out


def signals(conn):
    return sorted(tuple(r) for r in conn.execute(
        "SELECT entity_id, signal_type, platform, count FROM curation_signals"))


def entities(conn):
    return sorted((r[0], r[1], json.loads(r[2] or "null"), json.loads(r[3] or "null"))
                  for r in conn.execute(
                      "SELECT entity_id, name, identity_links, profile FROM entities"))


def test_a_complete_walk_publishes_each_authors_save_count(conn, monkeypatch):
    """Pinned against the old whole-feed signal walk until the cutover deleted it."""
    posts = [bm("1", "5", name="Ada", handle="ada", site="https://ada.dev"), bm("2", "5"),
             bm("3", "6", name=None, handle="b"), bm("4", "me")]
    feed = Feed(monkeypatch)
    feed.serve(posts[:2], posts[2:], [])
    assert walk(conn, bookmarks_source()) == \
        ["started", "page", "page", "page", "published"]

    assert signals(conn) == [("x:user:5", "save", "x", 2), ("x:user:6", "save", "x", 1)]
    assert entities(conn) == [("x:user:5", "Ada", ["https://ada.dev"], {"handle": "ada"}),
                              ("x:user:6", "b", None, {"handle": "b"})]
    run = clock.get_run(conn, bookmark_listing.SIGNAL_COLLECTOR)
    assert (run.last_status, run.found, run.stored_after) == ("ok", 2, 2)


def test_each_page_offers_every_saved_post_and_asserts_its_authors(conn, monkeypatch):
    feed = Feed(monkeypatch)
    feed.serve([bm("1", "5"), bm("2", "5"), bm("4", "me")], [bm("3", "6")], [])
    source = bookmarks_source()
    step(conn, source)                                  # start
    step(conn, source)                                  # first page

    assert sorted(s.item for s in saved_items.owed(conn, "x")) == ["1", "2", "4"]
    # Presence only: no count can be claimed before the walk ends.
    assert signals(conn) == [("x:user:5", "save", "x", 1)]
    assert listings.current(conn, bookmark_listing.COLLECTOR).state == "walking"


def test_an_unchanged_bookmark_is_not_owed_again_on_the_next_walk(conn, monkeypatch):
    feed = Feed(monkeypatch)
    feed.serve([bm("1", "5")], [])
    source = bookmarks_source()
    walk(conn, source)
    saved_items.settle(conn, saved_items.owed(conn, "x")[0])

    feed.serve([bm("1", "5", likes=99), bm("2", "6")], [])
    later = NOW + timedelta(hours=1)
    walk(conn, source, later)

    assert [s.item for s in saved_items.owed(conn, "x")] == ["2"]


def test_counts_wait_for_the_six_hour_clock(conn, monkeypatch):
    feed = Feed(monkeypatch)
    feed.serve([bm("1", "5"), bm("2", "5")], [])
    source = bookmarks_source()
    walk(conn, source)

    feed.serve([bm("1", "5"), bm("2", "5"), bm("3", "5"), bm("4", "7")], [])
    assert walk(conn, source, NOW + timedelta(hours=1))[-1] == "complete"
    assert signals(conn) == [("x:user:5", "save", "x", 2), ("x:user:7", "save", "x", 1)]

    assert walk(conn, source, NOW + timedelta(hours=6))[-1] == "published"
    assert signals(conn) == [("x:user:5", "save", "x", 3), ("x:user:7", "save", "x", 1)]


def test_the_next_walk_starts_an_hour_after_the_last_completed(conn, monkeypatch):
    feed = Feed(monkeypatch)
    feed.serve([bm("1", "5")], [])
    source = bookmarks_source()
    walk(conn, source)

    assert source.next_unit(conn, NOW + timedelta(minutes=59)) is None
    assert source.next_due(conn, NOW) == NOW + timedelta(hours=1)
    assert source.next_unit(conn, NOW + timedelta(hours=1)).key.endswith(":start")


def test_a_failed_page_resumes_its_own_cursor_instead_of_starting_over(conn, monkeypatch):
    feed = Feed(monkeypatch)
    feed.serve([bm("1", "5")], [bm("2", "6")], [])
    source = bookmarks_source()
    step(conn, source)
    step(conn, source)
    unit = source.next_unit(conn, NOW)
    task = unit.payload[2]
    listings.ROWS.write(conn, (task.id,),
                        Retry(retries.BLIP, 1, utc_iso(NOW + timedelta(minutes=1)), "boom"))

    assert source.next_unit(conn, NOW) is None
    assert source.next_due(conn, NOW) == NOW + timedelta(minutes=1)
    resumed = source.next_unit(conn, NOW + timedelta(minutes=1))
    assert resumed.payload[2].branch.cursor == "c1"


def test_a_cursor_that_loops_charges_the_page_and_claims_no_total(conn, monkeypatch):
    feed = Feed(monkeypatch)
    looping = Page((("1", bm("1", "5")),), "c1", ("1",))
    feed.pages = {None: looping, "c1": looping}
    source = bookmarks_source()
    statuses = [step(conn, source) for _ in range(3)]

    assert [s.status for s in statuses] == ["started", "page", "incomplete"]
    assert statuses[-1].failure == retries.BLIP
    assert listings.current(conn, bookmark_listing.COLLECTOR).state == "walking"
    assert clock.get_run(conn, bookmark_listing.SIGNAL_COLLECTOR) is None


def test_another_account_supersedes_the_old_accounts_owed_saves(conn, monkeypatch):
    feed = Feed(monkeypatch)
    feed.serve([bm("1", "5")], [])
    source = bookmarks_source()
    step(conn, source)
    step(conn, source)                                  # one page in, item 1 owed

    feed.viewer = "someone-else"
    feed.serve([bm("9", "6")], [])
    walk(conn, source)

    assert [(s.item, s.account) for s in saved_items.owed(conn, "x")] == [("9", "someone-else")]


def test_without_an_x_session_nothing_is_owed_or_due(conn, monkeypatch):
    Feed(monkeypatch, connected=False)
    source = bookmarks_source()
    assert source.next_unit(conn, NOW) is None
    assert source.next_due(conn, NOW) is None


def test_a_session_without_a_viewer_pauses_on_the_session(conn, monkeypatch):
    from pipeline.ingestion.utils import SyncAuthError

    Feed(monkeypatch, viewer=None)
    source = bookmarks_source()
    ctx = Context("x:Bookmarks")
    unit = source.next_unit(conn, NOW)
    with pytest.raises(SyncAuthError):
        source.fetch(conn, unit, ctx)
    assert "session" not in ctx.cache
    assert listings.current(conn, bookmark_listing.COLLECTOR) is None
