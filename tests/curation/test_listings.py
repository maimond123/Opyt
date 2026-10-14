"""Collection checkpoints survive restarts and refuse incomplete or obsolete evidence."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.curation import listings
from pipeline.kb.writes import write_tx
from pipeline.platforms.collection_page import Incomplete, Page
from pipeline.worker.quota_loop.source import Changed, Retry

NOW = datetime(2026, 10, 4, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path):
    db = sqlite3.connect(tmp_path / "collections.db")
    listings.init_schema(db)
    yield db
    db.close()


def start(conn, *, collector="x_likes", account="me", operation="Likes", now=NOW):
    return listings.start(conn, collector, account, listings.Branch(operation), now)


def page(key="tweet-1", cursor=None, **kwargs):
    return Page(((key, {"user_id": "author", "item": key}),), cursor, (key,), **kwargs)


def test_fresh_home_is_readable_without_creating_tables():
    with sqlite3.connect(":memory:") as db:
        assert listings.current(db, "x_likes") is None
        assert listings.accepted(db, "x_likes") is None
        assert list(db.execute("SELECT name FROM sqlite_master")) == []


def test_restart_after_each_page_goes_beyond_the_old_cap_and_deduplicates(tmp_path):
    path = tmp_path / "restart.db"
    with sqlite3.connect(path) as db:
        listings.init_schema(db)
        walk = start(db)
    for number in range(105):
        # A new connection for every page: there is no in-memory cursor or seen set to rely on.
        db = sqlite3.connect(path)
        walk = listings.current(db, "x_likes")
        task, = listings.pending(db, walk)
        assert task.branch.cursor == (f"c{number}" if number else None)
        result = page(f"tweet-{number}", f"c{number + 1}")
        assert listings.accept_page(db, walk, task, result)
        assert not listings.accept_page(db, walk, task, result)
        db.close()
    with sqlite3.connect(path) as db:
        task, = listings.pending(db, walk)
        listings.accept_page(db, walk, task, Page((), "end", (), ended=True))
        assert listings.current(db, "x_likes").state == "ready"
        assert listings.accepted(db, "x_likes") is None
        assert len(list(listings.observations(db, walk, "Likes"))) == 105
        listings.complete(db, walk, NOW)
        assert listings.accepted(db, "x_likes").id == walk.id


def test_cursor_cycle_leaves_page_unaccepted_and_preserves_prior_aggregate(conn):
    old = start(conn)
    listings.accept_page(conn, old, listings.pending(conn, old)[0], page())
    listings.complete(conn, old, NOW)
    walk = start(conn, now=NOW + timedelta(hours=6))
    listings.accept_page(conn, walk, listings.pending(conn, walk)[0], page("one", "c1"))
    listings.accept_page(conn, walk, listings.pending(conn, walk)[0], page("two", "c2"))
    task, = listings.pending(conn, walk)
    with pytest.raises(Incomplete, match="cursor repeated"):
        listings.accept_page(conn, walk, task, page("three", "c1"))
    with pytest.raises(Incomplete, match="unsettled"):
        listings.complete(conn, walk, NOW)
    assert listings.pending(conn, walk) == [task]
    assert [key for _, key, _ in listings.observations(conn, walk, "Likes")] == ["one", "two"]
    assert listings.accepted(conn, "x_likes").id == old.id


def test_filtered_observations_count_toward_end_detection_without_becoming_signals(conn):
    walk = start(conn)
    task, = listings.pending(conn, walk)
    listings.accept_page(conn, walk, task, Page((), "c1", ("unavailable",), repeat_is_end=True))
    assert listings.current(conn, "x_likes").state == "walking"
    task, = listings.pending(conn, walk)
    listings.accept_page(conn, walk, task, Page((), "c2", ("unavailable",), repeat_is_end=True))
    assert listings.current(conn, "x_likes").state == "ready"
    assert list(listings.observations(conn, walk, "Likes")) == []


def test_list_members_are_separate_partitions_and_all_must_finish(conn):
    walk = start(conn, collector="x_lists", operation="ListsManagementPageTimeline")
    task, = listings.pending(conn, walk)
    branches = (listings.Branch("ListMembers", "L1"), listings.Branch("ListMembers", "L2"))
    listings.accept_page(conn, walk, task, page("list", None), branches=branches)
    first, second = listings.pending(conn, walk, "ListMembers")
    listings.accept_page(conn, walk, first, page("same-user", "c1", repeat_is_end=True))
    listings.accept_page(conn, walk, second, page("same-user", "c1", repeat_is_end=True))
    # The same user in another list is still a new observation, so both advance to c1.
    first_end, second_end = listings.pending(conn, walk, "ListMembers")
    listings.accept_page(conn, walk, first_end, Page((), None, (), ended=True))
    listings.ROWS.write(conn, (second_end.id,), Retry("error", 5, None, "unavailable"))
    with pytest.raises(Incomplete):
        listings.complete(conn, walk, NOW)
    assert listings.accepted(conn, "x_lists") is None
    assert listings.current(conn, "x_lists").state == "walking"
    listings.accept_page(conn, walk, second_end, Page((), None, (), ended=True))
    listings.complete(conn, walk, NOW)
    assert [(part, key) for part, key, _ in listings.observations(conn, walk, "ListMembers")] == [
        ("L1", "same-user"), ("L2", "same-user")]


def test_failed_checkpoint_rolls_back_records_children_and_acceptance(conn):
    walk = start(conn)
    task, = listings.pending(conn, walk)
    conn.execute("CREATE TRIGGER refuse_child BEFORE INSERT ON curation_pages "
                 "WHEN NEW.partition='refused' BEGIN SELECT RAISE(ABORT,'disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="disk failure"):
        listings.accept_page(conn, walk, task, page(cursor="next"),
                             branches=(listings.Branch("ListMembers", "refused"),))
    assert listings.pending(conn, walk) == [task]
    assert list(listings.observations(conn, walk, "Likes")) == []
    assert list(conn.execute("SELECT * FROM curation_seen")) == []
    assert not conn.in_transaction


def test_account_switch_and_switch_back_invalidate_inflight_results(conn):
    first = start(conn)
    old_task, = listings.pending(conn, first)
    second = start(conn, account="other")
    third = start(conn)  # Identical timestamp and account still means a different generation.
    assert len({first.id, second.id, third.id}) == 3
    with pytest.raises(Changed):
        listings.accept_page(conn, first, old_task, page())
    with pytest.raises(Changed):
        listings.complete(conn, first, NOW)
    listings.ROWS.write(conn, (old_task.id,), Retry("late-error", 3, None, "old request"))
    assert conn.execute("SELECT failures FROM curation_pages WHERE id=?", (old_task.id,)).fetchone() == (None,)
    assert listings.accepted(conn, "x_likes") is None


def test_account_switch_does_not_reuse_other_accounts_accepted_baseline(conn):
    old = start(conn)
    listings.accept_page(conn, old, listings.pending(conn, old)[0], page())
    listings.complete(conn, old, NOW)
    start(conn, account="other")
    assert listings.accepted(conn, "x_likes") is None


def test_cadence_does_not_reset_exhausted_work(conn):
    walk = start(conn)
    task, = listings.pending(conn, walk)
    exhausted = Retry("network", 5, None, "failed")
    listings.ROWS.write(conn, (task.id,), exhausted)
    assert start(conn, now=NOW + timedelta(days=1)).id == walk.id
    assert listings.pending(conn, walk)[0].retry == exhausted


def test_publication_and_completion_can_share_one_transaction(conn):
    conn.execute("CREATE TABLE published (count INTEGER)")
    walk = start(conn)
    listings.accept_page(conn, walk, listings.pending(conn, walk)[0], page())
    with pytest.raises(RuntimeError), write_tx(conn, site="test.publish"):
        conn.execute("INSERT INTO published VALUES(1)")
        listings.complete(conn, walk, NOW)
        raise RuntimeError("publication failed")
    assert listings.current(conn, "x_likes").state == "ready"
    assert listings.accepted(conn, "x_likes") is None
    assert list(conn.execute("SELECT * FROM published")) == []


def test_retention_keeps_active_and_last_accepted_then_prunes_replaced_evidence(conn):
    old = start(conn)
    listings.accept_page(conn, old, listings.pending(conn, old)[0], page())
    listings.complete(conn, old, NOW)
    fresh = start(conn)
    listings.prune(conn, "x_likes")
    assert list(listings.observations(conn, old, "Likes"))
    assert listings.pending(conn, fresh)
    listings.accept_page(conn, fresh, listings.pending(conn, fresh)[0], page("new"))
    listings.complete(conn, fresh, NOW)
    listings.prune(conn, "x_likes")
    assert list(listings.observations(conn, old, "Likes")) == []
    assert conn.execute("SELECT id FROM curation_collections").fetchall() == [(fresh.id,)]
    assert start(conn).id > fresh.id


def test_foreign_task_cannot_be_attached_to_current_generation(conn):
    walk = start(conn)
    other = start(conn, collector="following", operation="Following")
    task, = listings.pending(conn, other)
    with pytest.raises(Changed, match="does not belong"):
        listings.accept_page(conn, walk, task, page())
    assert list(listings.observations(conn, walk, "Likes")) == []


@pytest.mark.parametrize("frm, to", [
    (listings.COMPLETE, listings.READY),
    (listings.SUPERSEDED, listings.WALKING),
    (listings.WALKING, listings.COMPLETE),
])
def test_a_walk_moves_only_along_its_transitions(conn, frm, to):
    walk = start(conn)
    with pytest.raises(ValueError):
        listings._move(conn, listings.Walk(walk.id, walk.collector, walk.account,
                                           walk.started_at, frm), to)
