"""Substack's follows, subscriptions and saved posts walk one page per unit in the web pool, and
publish what the old whole-list collectors published, only from a complete walk."""

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.curation import (clock, listings, saved_items, substack_people,
                               substack_saved_listing, substack_walks, walks)
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.platforms import sessions
from pipeline.platforms.collection_page import Incomplete
from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.source import Context

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
ACCOUNT = 42


@pytest.fixture
def conn(tmp_path):
    db = schema.connect(tmp_path / "opyt.db")
    listings.init_schema(db)
    saved_items.init_schema(db)
    yield db
    db.close()


def saved_item(pid, pub="carol", handle=None):
    return {"type": "post",
            "post": {"id": pid, "title": f"Post {pid}", "slug": f"p{pid}",
                     "canonical_url": f"https://{pub}.substack.com/p/p{pid}",
                     "post_date": "2026-09-01T00:00:00Z", "audience": "everyone"},
            "publication": {"name": pub.title(), "base_url": f"https://{pub}.substack.com",
                            "author_handle": handle or pub, "author_name": pub.title()}}


SUBS = {"subscriptions": [{"id": 11, "publication_id": 101, "membership_state": "free_signup",
                           "is_favorite": False},
                          {"id": 12, "publication_id": 102, "membership_state": "subscribed",
                           "is_favorite": True}],
        "publications": [{"id": 101, "name": "Free Pub", "subdomain": "freepub"},
                         {"id": 102, "name": "Paid Pub", "subdomain": "paidpub",
                          "custom_domain": "letters.example.com"}]}


class Substack:
    """The account endpoints both the old collectors and the new walks read."""

    def __init__(self, monkeypatch):
        self.follows = [{"name": "A", "url": "https://a.substack.com"},
                        {"name": "B", "url": "https://b.example.com"}]
        self.subs = {None: SUBS}
        self.saved = {None: {"items": [saved_item(1), saved_item(2)], "nextCursor": "c2"},
                      "c2": {"items": [saved_item(3, "dave")]}}
        self.account = ACCOUNT
        self.connected = True
        self.calls = []
        test = self

        class Follows:
            def follows(self, user_id=None):
                test.calls.append(("follows", user_id))
                return [dict(f) for f in test.follows]

        class Pages:
            def __init__(self, what, feed):
                self.what, self.feed = what, feed

            def page(self, cursor):
                test.calls.append((self.what, cursor))
                page = self.feed()[cursor]
                if isinstance(page, Exception):
                    raise page
                return page

        monkeypatch.setattr(connections, "_connected_substack", lambda: test.connected)
        monkeypatch.setattr(sessions, "own_account_id", self._account)
        monkeypatch.setattr(sessions, "follow_source", lambda profile=None: Follows())
        monkeypatch.setattr(sessions, "subscription_list_source",
                            lambda profile=None: Pages("subs", lambda: test.subs))
        monkeypatch.setattr(sessions, "saved_source",
                            lambda profile=None: Pages("saved", lambda: test.saved))
        monkeypatch.setattr(time, "sleep", lambda *_: None)

    def _account(self):
        self.calls.append(("account", None))
        if isinstance(self.account, Exception):
            raise self.account
        return self.account


def people_sources(via):
    return [walks.WalkSource(substack_people.Follows(), via),
            walks.WalkSource(substack_people.Subscriptions(), via)]


def saved_source(via):
    return walks.WalkSource(substack_saved_listing.SavedPosts(), via)

def account(clock=lambda: NOW - timedelta(hours=1)):
    return substack_walks.SubstackAccount("web", Outages(PLATFORM_PAUSES_S), clock)


def drain(conn, sources, now=NOW):
    """Run every owed unit across the sources until none owes one; the statuses in order."""
    out, ctxs = [], {id(s): Context(s.quota) for s in sources}
    while True:
        for source in sources:
            unit = source.next_unit(conn, now)
            if unit is not None:
                out.append(source.record(conn, unit,
                                         source.fetch(conn, unit, ctxs[id(source)])).status)
                break
        else:
            return out


def signals(conn, signal_type):
    return sorted((r[0], r[1], json.loads(r[2]) if r[2] else None) for r in conn.execute(
        "SELECT entity_id, count, extra FROM curation_signals WHERE signal_type=?", (signal_type,)))


def entities(conn):
    """A walk writes an entity at presence and again at its count, and `upsert_entity`'s second
    write turns a NULL profile into `{}`: the same nothing to every reader."""
    return sorted((r[0], r[1], json.loads(r[2] or "null"), json.loads(r[3] or "null") or None)
                  for r in conn.execute("SELECT entity_id, name, identity_links, profile FROM entities"))


def run_row(conn, collector):
    row = clock.get_run(conn, collector)
    return (row.last_status, row.found, row.stored_after) if row else None


# ── what each walk publishes (pinned against the old collectors, deleted at the cutover) ──

def test_the_follow_walk_publishes_each_followed_publication(conn, monkeypatch):
    Substack(monkeypatch)
    follows = people_sources(account())[0]

    assert drain(conn, [follows]) == ["started", "page", "published"]
    assert signals(conn, "follow") == [("substack:a", 1, None),
                                       ("substack:b.example.com", 1, None)]
    assert entities(conn) == [("substack:a", "A", ["https://a.substack.com"], None),
                              ("substack:b.example.com", "B", ["https://b.example.com"], None)]
    assert run_row(conn, "substack_follows") == ("ok", 2, 2)


def test_the_subscription_walk_lands_the_paid_flag_the_follow_read_never_could(conn,
                                                                               monkeypatch):
    """`membership_state` supplies `is_paid`, the first code path that makes a claim about the
    user's money; the raw state rides along so the inferred `subscribed` arm can be audited."""
    Substack(monkeypatch)
    subs = people_sources(account())[1]

    assert drain(conn, [subs]) == ["started", "page", "published"]
    assert signals(conn, "subscribe") == [
        ("substack:freepub", 1, {"is_paid": False, "membership_state": "free_signup",
                                 "is_favorite": False}),
        ("substack:letters.example.com", 1, {"is_paid": True, "membership_state": "subscribed",
                                             "is_favorite": True})]
    assert run_row(conn, "substack_subscriptions") == ("ok", 2, 2)


def test_an_unrecognised_membership_state_never_invents_a_payment():
    """Three-way, and the None arm is why: only `subscribed` may print "(paid)"."""
    assert substack_people.is_paid("subscribed") is True
    assert substack_people.is_paid("free_signup") is False
    assert substack_people.is_paid("unsubscribed") is None
    assert substack_people.is_paid("") is None
    assert substack_people.is_paid("some_state_substack_adds_later") is None


def test_the_saved_list_counts_saves_per_publication(conn, monkeypatch):
    Substack(monkeypatch)

    assert drain(conn, [saved_source(account())]) == \
        ["started", "page", "page", "published"]
    assert signals(conn, "save") == [("substack:carol", 2, None), ("substack:dave", 1, None)]
    assert run_row(conn, "substack_saved_signals") == ("ok", 2, 2)


def test_a_saved_publication_merges_with_the_subscription_that_keys_on_its_url(conn,
                                                                               monkeypatch):
    """⚠️ THE KEYING SPLIT. The saved list carries a HANDLE and the account reads do not, so a
    publication the user both saves from and subscribes to mints TWO ids, each below the
    corroboration bar. Stage-3 closes it on the publication URL, which this side must write."""
    from pipeline.curation import maintenance

    feed = Substack(monkeypatch)
    feed.subs = {None: {"subscriptions": [{"id": 1, "publication_id": 7,
                                           "membership_state": "free_signup"}],
                        "publications": [{"id": 7, "name": "Carol", "subdomain": "carol"}]}}
    feed.saved = {None: {"items": [saved_item(1, "carol", handle="carolwrites")]}}
    drain(conn, [people_sources(account())[1], saved_source(account())])
    maintenance.init_schema(conn)
    source = maintenance.Maintenance()
    while (unit := source.next_unit(conn, NOW)) is not None:
        source.record(conn, unit, source.fetch(conn, unit, Context(source.quota)))

    heads = {r[0]: r[1] for r in conn.execute(
        "SELECT entity_id, canonical_id FROM entities WHERE entity_id IN "
        "('substack:carol', 'substack:carolwrites')")}
    assert heads["substack:carol"] and heads["substack:carol"] == heads["substack:carolwrites"]


def test_a_subscription_and_a_save_corroborate_one_publication(conn, monkeypatch):
    """THE ACCEPTANCE SHAPE: two distinct (signal_type, platform) pairs on one entity."""
    feed = Substack(monkeypatch)
    feed.subs = {None: {"subscriptions": [{"id": 1, "publication_id": 7,
                                           "membership_state": "free_signup"}],
                        "publications": [{"id": 7, "name": "Carol", "subdomain": "carol"}]}}
    feed.saved = {None: {"items": [saved_item(1, "carol", handle="")]}}
    drain(conn, [people_sources(account())[1], saved_source(account())])

    kinds = {r[0] for r in conn.execute(
        "SELECT signal_type FROM curation_signals WHERE entity_id='substack:carol'")}
    assert kinds == {"subscribe", "save"}


# ── the walk on Substack ───────────────────────────────────────────────────────

def test_the_account_is_read_once_per_walk_and_the_follow_read_reuses_it(conn, monkeypatch):
    """Each account read is a request to the Cloudflare-guarded reader page."""
    feed = Substack(monkeypatch)
    drain(conn, people_sources(account()))

    assert feed.calls.count(("account", None)) == 2       # one start per graph
    assert ("follows", ACCOUNT) in feed.calls


def test_units_run_in_the_web_pool_on_substack_com(conn, monkeypatch):
    Substack(monkeypatch)
    source = saved_source(account())
    unit = source.next_unit(conn, NOW)

    assert source.quota == "web" and unit.host == "substack.com"
    source.record(conn, unit, source.fetch(conn, unit, Context("web")))
    page = source.next_unit(conn, NOW)
    assert page.item.platform == "substack" and page.host == "substack.com"


def test_each_saved_page_offers_its_posts_and_asserts_their_authors(conn, monkeypatch):
    Substack(monkeypatch)
    source = saved_source(account())
    drain(conn, [source])

    assert [(s.item, s.account) for s in saved_items.owed(conn, "substack")] == \
        [("1", "42"), ("2", "42"), ("3", "42")]
    assert saved_items.owed(conn, "substack")[0].data["slug"] == "p1"


def test_presence_lands_before_the_walk_ends(conn, monkeypatch):
    feed = Substack(monkeypatch)
    feed.saved["c2"] = substack_contracts.SubstackListingError("refused")
    source = saved_source(account())
    ctx = Context("web")
    for _ in range(2):
        unit = source.next_unit(conn, NOW)
        source.record(conn, unit, source.fetch(conn, unit, ctx))

    assert signals(conn, "save") == [("substack:carol", 1, None)]   # presence, not a count
    assert run_row(conn, "substack_saved_signals") is None


@pytest.mark.parametrize("bad, raised", [(None, substack_contracts.SubstackListingError),
                                          ({"error": "Cloudflare"}, Incomplete)])
def test_a_refused_page_raises_so_the_loop_charges_it_and_the_walk_resumes_there(
        conn, monkeypatch, bad, raised):
    feed = Substack(monkeypatch)
    feed.saved["c2"] = bad
    source = saved_source(account())
    ctx = Context("web")
    for _ in range(2):
        unit = source.next_unit(conn, NOW)
        source.record(conn, unit, source.fetch(conn, unit, ctx))
    unit = source.next_unit(conn, NOW)
    with pytest.raises(raised) as failure:
        source.fetch(conn, unit, ctx)
    assert unit.item is not None and retries.kind_of(failure.value) == retries.BLIP

    feed.saved["c2"] = {"items": [saved_item(3, "dave")]}
    assert drain(conn, [source]) == ["page", "published"]
    assert feed.calls.count(("saved", None)) == 1        # page 1 was not read again


def test_a_missing_session_never_pauses_the_web_pool_on_the_x_session_rule(conn, monkeypatch):
    """`SyncAuthError` is the X session's: the loop pauses the whole quota on it, and here that
    would stop every web unit. A Substack walk raises an ordinary failure instead."""
    feed = Substack(monkeypatch)
    feed.account = None
    source = people_sources(account())[0]
    unit = source.next_unit(conn, NOW)
    with pytest.raises(substack_walks.NotSignedIn):
        source.fetch(conn, unit, Context("web"))

    feed.account = SyncAuthError("no managed profile")
    with pytest.raises(substack_walks.NotSignedIn) as raised:
        source.fetch(conn, unit, Context("web"))
    assert retries.kind_of(raised.value) == retries.BLIP


def test_without_a_session_nothing_is_owed_or_due(conn, monkeypatch):
    Substack(monkeypatch).connected = False
    for source in [*people_sources(account()), saved_source(account())]:
        assert source.next_unit(conn, NOW) is None
        assert source.next_due(conn, NOW) is None


def test_a_paused_substack_holds_its_reads_until_the_pause_ends(conn, monkeypatch):
    Substack(monkeypatch)
    via = account()
    until = via.outages.pause("substack", NOW, 1.0)
    source = saved_source(via)

    assert source.next_unit(conn, NOW) is None
    assert source.next_due(conn, NOW) == until
    assert source.next_unit(conn, until + timedelta(seconds=1)) is not None


def test_reads_are_paced_one_second_apart(conn, monkeypatch):
    Substack(monkeypatch)
    via = account(clock=lambda: NOW)
    source = saved_source(via)
    unit = source.next_unit(conn, NOW)
    source.record(conn, unit, source.fetch(conn, unit, Context("web")))

    assert source.next_unit(conn, NOW) is None
    assert source.next_due(conn, NOW) == NOW + substack_walks.PACE
    assert source.next_unit(conn, NOW + substack_walks.PACE) is not None


def test_another_account_at_the_next_start_supersedes_the_old_accounts_owed_saves(conn,
                                                                                  monkeypatch):
    feed = Substack(monkeypatch)
    source = saved_source(account())
    drain(conn, [source])
    feed.account = 7
    feed.saved = {None: {"items": [saved_item(9)]}}

    drain(conn, [source], NOW + substack_saved_listing.REFRESH)

    assert [(s.item, s.account) for s in saved_items.owed(conn, "substack")] == [("9", "7")]


def test_the_account_graphs_wait_for_the_six_hour_floor_and_the_list_for_six_hours(conn,
                                                                                   monkeypatch):
    Substack(monkeypatch)
    sources = [*people_sources(account()), saved_source(account())]
    drain(conn, sources)

    assert drain(conn, sources, NOW + timedelta(hours=5)) == []
    assert len(drain(conn, sources, NOW + timedelta(hours=6, minutes=1))) == 10


def test_a_page_of_more_than_the_old_cap_still_walks_to_the_end(conn, monkeypatch):
    """The saved drain stopped at 200 pages; the walk has no cap."""
    feed = Substack(monkeypatch)
    feed.saved = {(None if n == 0 else f"c{n}"):
                  {"items": [saved_item(n + 1)], **({"nextCursor": f"c{n + 1}"} if n < 204 else {})}
                  for n in range(205)}
    statuses = drain(conn, [saved_source(account())])

    assert statuses.count("page") == 205 and statuses[-1] == "published"
    assert signals(conn, "save") == [("substack:carol", 205, None)]
