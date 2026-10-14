"""Following, Likes and Lists walk to the real end one page per unit, and publish what the old
whole-walk collectors published, only from a complete walk."""

import json
from datetime import datetime, timedelta, timezone

from types import SimpleNamespace

import pytest

from pipeline.curation import clock, listings, walks, x_collectors, x_walks
from pipeline.curation.signals import Person, Tally
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.platforms import sessions
from pipeline.platforms.x import collections as x_collections
from pipeline.platforms.collection_page import Incomplete, Page
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Context, Retry

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
VIEWER = "me"


@pytest.fixture
def conn(tmp_path):
    db = schema.connect(tmp_path / "opyt.db")
    listings.init_schema(db)
    yield db
    db.close()


def user(uid, handle=None, site=None, bio=None):
    return {"user_id": uid, "handle": handle or f"u{uid}", "display_name": f"User {uid}",
            "bio": bio or "", "site": site or "", "followers_count": 10, "verified": False}


def pages(*chunks, key, prefix, value=lambda r: r):
    """Pages keyed by the cursor that asks for them; the last one ends the walk."""
    out = {}
    for number, records in enumerate(chunks):
        last = number == len(chunks) - 1
        recs = tuple((key(r), value(r)) for r in records)
        out[None if number == 0 else f"c{number}"] = Page(
            recs, None if last else f"c{number + 1}",
            tuple(prefix + k for k, _ in recs), ended=last)
    return out


class X:
    """The endpoints both the old collectors and the new sources read."""

    def __init__(self, monkeypatch):
        self.following, self.likes, self.lists, self.members = {}, {}, {}, {}
        self.calls = []
        monkeypatch.setattr(connections, "_connected_x", lambda: True)
        monkeypatch.setattr(sessions, "x_session",
                            lambda referer: SimpleNamespace(viewer_id=lambda: VIEWER))
        monkeypatch.setattr(x_collections, "following_page",
                            lambda c, v, cursor=None: self._serve("Following", self.following, cursor))
        monkeypatch.setattr(x_collections, "likes_page",
                            lambda v, c, cursor=None: self._serve("Likes", self.likes, cursor))
        monkeypatch.setattr(x_collections, "owned_lists_page",
                            lambda c, v, cursor=None: self._serve("Lists", self.lists, cursor))
        monkeypatch.setattr(x_collections, "list_members_page",
                            lambda lid, c, cursor=None: self._serve(f"Members:{lid}",
                                                                       self.members[lid], cursor))

    def _serve(self, what, feed, cursor):
        self.calls.append((what, cursor))
        page = feed[cursor]
        if isinstance(page, Exception):
            raise page
        return page


def walk_source(collection, operation=None, *, ends=True):
    return walks.WalkSource(collection, x_walks.XAccount(), operation, ends=ends)


def lists_sources():
    lists = x_collectors.Lists()
    return [walk_source(lists), walk_source(lists, lists.members, ends=False)]

def drain(conn, sources, now=NOW):
    """Run every owed unit across the sources until none owes one; the statuses in order."""
    out, ctxs = [], {id(s): Context(s.quota) for s in sources}
    while True:
        for source in sources:
            unit = source.next_unit(conn, now)
            if unit is not None:
                out.append(source.record(conn, unit, source.fetch(conn, unit, ctxs[id(source)])).status)
                break
        else:
            return out


def signals(conn, signal_type):
    return sorted((r[0], r[1], json.loads(r[2]) if r[2] else None) for r in conn.execute(
        "SELECT entity_id, count, extra FROM curation_signals WHERE signal_type=?", (signal_type,)))


def entity(u):
    """The entity row a published X person lands as."""
    profile = {"handle": u["handle"], "followers": u["followers_count"],
               **({"bio": u["bio"]} if u["bio"] else {})}
    return (f"x:user:{u['user_id']}", u["display_name"], [u["site"]] if u["site"] else None,
            profile)


def entities(conn):
    return sorted((r[0], r[1], json.loads(r[2] or "null"), json.loads(r[3] or "null"))
                  for r in conn.execute("SELECT entity_id, name, identity_links, profile FROM entities"))


def run_row(conn, collector):
    row = clock.get_run(conn, collector)
    return (row.last_status, row.found, row.stored_after) if row else None


# ── Following ─────────────────────────────────────────────────────────────────

def test_following_publishes_each_followed_person(conn, monkeypatch):
    """Pinned against the old whole-walk collector until the cutover deleted it."""
    x = X(monkeypatch)
    people = [user("1", site="https://one.dev", bio="writes"), user("2"), user("3")]
    x.following = pages(people[:2], people[2:], key=lambda u: u["user_id"], prefix="user-")

    assert drain(conn, [walk_source(x_collectors.Following())]) == ["started", "page", "page", "published"]

    assert signals(conn, "follow") == [
        ("x:user:1", 1, None), ("x:user:2", 1, None), ("x:user:3", 1, None)]
    assert entities(conn) == [entity(u) for u in people]
    assert run_row(conn, "x_following") == ("ok", 3, 3)


def test_following_stamps_the_walks_start_for_retirement(conn, monkeypatch):
    x = X(monkeypatch)
    x.following = pages([user("1"), user("2"), user("3")], key=lambda u: u["user_id"], prefix="user-")
    source = walk_source(x_collectors.Following())
    drain(conn, [source])

    x.following = pages([user("1"), user("2")], key=lambda u: u["user_id"], prefix="user-")
    later = NOW + timedelta(hours=6)
    drain(conn, [source], later)

    row = clock.get_run(conn, "x_following")
    assert (row.found, row.prev_found, row.started_at) == (2, 3, utc_iso(later))
    assert clock.walk_is_trustworthy(row)
    confirmed = conn.execute("SELECT COUNT(*) FROM curation_signals WHERE signal_type='follow' "
                             "AND last_confirmed_at IS NOT NULL").fetchone()[0]
    assert confirmed == 3


def test_a_page_asserts_presence_but_no_count_before_the_walk_ends(conn, monkeypatch):
    x = X(monkeypatch)
    x.following = pages([user("1")], [user("2")], key=lambda u: u["user_id"], prefix="user-")
    source = walk_source(x_collectors.Following())
    ctx = Context(source.quota)
    for _ in range(2):                                  # start, first page
        unit = source.next_unit(conn, NOW)
        source.record(conn, unit, source.fetch(conn, unit, ctx))

    assert signals(conn, "follow") == [("x:user:1", 1, None)]
    assert conn.execute("SELECT last_confirmed_at FROM curation_signals").fetchone()[0] is None
    assert clock.get_run(conn, "x_following") is None


def test_the_next_walk_waits_for_the_six_hour_floor(conn, monkeypatch):
    x = X(monkeypatch)
    x.following = pages([user("1")], key=lambda u: u["user_id"], prefix="user-")
    source = walk_source(x_collectors.Following())
    drain(conn, [source])

    assert source.next_unit(conn, NOW + timedelta(hours=5, minutes=59)) is None
    assert source.next_due(conn, NOW) == NOW + timedelta(hours=6)
    assert source.next_unit(conn, NOW + timedelta(hours=6)).payload[0] == "start"


def test_a_walk_continues_past_the_old_hundred_page_cap(conn, monkeypatch):
    x = X(monkeypatch)
    x.following = pages(*[[user(str(n))] for n in range(105)], key=lambda u: u["user_id"],
                        prefix="user-")

    assert drain(conn, [walk_source(x_collectors.Following())])[-1] == "published"
    assert run_row(conn, "x_following") == ("ok", 105, 105)


def test_a_failing_page_keeps_the_walk_unpublished(conn, monkeypatch):
    x = X(monkeypatch)
    x.following = pages([user("1")], [user("2")], key=lambda u: u["user_id"], prefix="user-")
    source = walk_source(x_collectors.Following())
    ctx = Context(source.quota)
    for _ in range(2):
        unit = source.next_unit(conn, NOW)
        source.record(conn, unit, source.fetch(conn, unit, ctx))
    unit = source.next_unit(conn, NOW)
    listings.ROWS.write(conn, (unit.payload[2].id,),
                        Retry(retries.BLIP, 10, None, "gave up"))          # retries exhausted

    assert source.next_unit(conn, NOW + timedelta(days=1)) is None
    assert clock.get_run(conn, "x_following") is None


# ── Likes ─────────────────────────────────────────────────────────────────────

def test_likes_count_distinct_liked_posts(conn, monkeypatch):
    x = X(monkeypatch)
    liked = [("t1", user("5")), ("t2", user("5")), ("t3", user("6")), ("t4", user(VIEWER))]
    x.likes = pages(liked[:2], liked[2:], key=lambda r: r[0], value=lambda r: r[1],
                    prefix="tweet-")

    assert drain(conn, [walk_source(x_collectors.Likes())])[-1] == "published"

    assert signals(conn, "like") == [("x:user:5", 2, None), ("x:user:6", 1, None)]
    assert entities(conn) == [entity(user("5")), entity(user("6"))]
    assert run_row(conn, "x_likes") == ("ok", 2, 2)


# ── Lists ─────────────────────────────────────────────────────────────────────

def owned(*lists):
    return pages(list(lists), key=lambda lst: lst["id"], prefix="list-")


def lst(lid, name):
    return {"id": lid, "name": name, "mode": "Private", "member_count": 0}


def test_lists_publish_memberships(conn, monkeypatch):
    x = X(monkeypatch)
    x.lists = owned(lst("L1", "AI"), lst("L2", "Crypto"))
    x.members = {"L1": pages([user("7"), user(VIEWER)], [user("8")], key=lambda u: u["user_id"],
                             prefix="user-"),
                 "L2": pages([user("7")], key=lambda u: u["user_id"], prefix="user-")}

    assert drain(conn, lists_sources())[-1] == "published"

    assert signals(conn, "list") == [
        ("x:user:7", 2, {"list_names": ["AI", "Crypto"]}),
        ("x:user:8", 1, {"list_names": ["AI"]})]
    assert entities(conn) == [entity(user("7")), entity(user("8"))]
    assert run_row(conn, "x_lists") == ("ok", 2, 2)


def test_lists_finish_only_after_every_member_walk_settles(conn, monkeypatch):
    x = X(monkeypatch)
    x.lists = owned(lst("L1", "AI"), lst("L2", "Crypto"))
    x.members = {"L1": pages([user("7")], key=lambda u: u["user_id"], prefix="user-"),
                 "L2": pages([user("8")], key=lambda u: u["user_id"], prefix="user-")}
    enumeration, members = lists_sources()
    ctx = Context(enumeration.quota)
    for _ in range(2):                                  # start, the one enumeration page
        unit = enumeration.next_unit(conn, NOW)
        enumeration.record(conn, unit, enumeration.fetch(conn, unit, ctx))

    assert enumeration.next_unit(conn, NOW) is None     # members owed: no finish yet
    unit = members.next_unit(conn, NOW)
    assert unit.payload[2].branch.partition == "L1"
    listings.ROWS.write(conn, (unit.payload[2].id,), Retry(retries.BLIP, 10, None, "gave up"))
    drain(conn, [enumeration, members])

    assert listings.current(conn, "x_lists").state == "walking"
    assert clock.get_run(conn, "x_lists") is None
    assert signals(conn, "list") == [("x:user:8", 1, None)]   # presence from L2's settled walk


def test_the_member_source_never_starts_or_finishes_a_walk(conn, monkeypatch):
    X(monkeypatch)
    _, members = lists_sources()
    assert members.next_unit(conn, NOW) is None
    assert members.next_due(conn, NOW) is None
    assert members.quota == "x:ListMembers"


def test_a_finish_on_an_unsettled_walk_publishes_nothing(conn):
    """B-C8: the finish checks the walk is ready, with no page owed, before it publishes: counts
    and the clock's `ok` from an unsettled walk would stand even though completing it refuses."""
    following = x_collectors.Following()
    walk = listings.start(conn, following.collector, "me", listings.Branch(following.operation), NOW)
    tallies = [Tally(Person("x:user:1", name="User 1"), 1)]
    with pytest.raises(Incomplete):
        walk_source(following)._finish(conn, walk, NOW, tallies)
    assert signals(conn, "follow") == []
    assert clock.get_run(conn, following.collector) is None
