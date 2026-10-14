"""Saved-content work outlives its listing and refuses to settle a stale revision."""

import sqlite3
from datetime import datetime, timezone

import pytest

from pipeline.curation import saved_items
from pipeline.worker.quota_loop.source import Changed, Retry

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path):
    db = sqlite3.connect(tmp_path / "saved.db")
    saved_items.init_schema(db)
    yield db
    db.close()


def post(tid="1", text="hello", likes=0, replies=0):
    return {"id": tid, "text": text, "likeCount": likes, "replyCount": replies}


def offer(conn, *posts, account="me"):
    saved_items.offer(conn, "x", account,
                      [(p["id"], saved_items.x_revision(p), p) for p in posts], NOW)


def state(conn, item="1"):
    return conn.execute("SELECT state, failures FROM curation_saved_items WHERE item=?",
                        (item,)).fetchone()


def test_a_fresh_home_owes_nothing_without_creating_tables():
    with sqlite3.connect(":memory:") as db:
        assert saved_items.owed(db, "x") == []
        assert list(db.execute("SELECT name FROM sqlite_master")) == []


def test_a_like_count_is_not_a_new_revision_but_a_reply_is():
    base = saved_items.x_revision(post())
    assert saved_items.x_revision(post(likes=40)) == base
    assert saved_items.x_revision(post(replies=1)) != base
    quoted = {**post(), "quoted_tweet": post("2")}
    assert saved_items.x_revision({**quoted, "quoted_tweet": post("2", likes=9)}) == \
        saved_items.x_revision(quoted)


def test_reoffering_the_same_revision_keeps_its_backoff_and_its_done_state(conn):
    offer(conn, post("1"), post("2"))
    saved_items.ROWS.write(conn, ("x", "1", saved_items.x_revision(post("1"))),
                           Retry("blip", 3, "2026-10-05T01:00:00Z", "boom"))
    saved_items.settle(conn, next(s for s in saved_items.owed(conn, "x") if s.item == "2"))

    offer(conn, post("1", likes=5), post("2", likes=5))

    assert tuple(state(conn, "1")) == ("owed", 3)
    assert tuple(state(conn, "2")) == ("done", None)


def test_a_re_offer_keeps_the_rows_re_render_inputs(conn):
    offer(conn, post())
    saved_items.record_chain(conn, "x", "1", [post("0"), post("1")])
    saved_items.record_rendered(conn, "x", "1", teaser=True)

    offer(conn, post(text="edited"))

    kept = saved_items.rendering(conn, "x", "1")
    assert kept.data["text"] == "edited"
    assert [t["id"] for t in kept.chain] == ["0", "1"] and kept.rendered_from_teaser
    assert saved_items.rendering(conn, "x", "2") is None


def test_a_changed_post_is_owed_again_with_fresh_retries(conn):
    offer(conn, post())
    saved_items.ROWS.write(conn, ("x", "1", saved_items.x_revision(post())),
                           Retry("blip", 10, None, "gave up"))
    offer(conn, post(text="edited"))

    saved, = saved_items.owed(conn, "x")
    assert saved.data["text"] == "edited" and saved.retry == Retry()


def test_an_older_revision_cannot_settle_a_newer_offer(conn):
    offer(conn, post())
    old, = saved_items.owed(conn, "x")
    offer(conn, post(text="edited"))

    with pytest.raises(Changed):
        saved_items.settle(conn, old)
    assert state(conn)[0] == "owed"


def test_a_charge_for_an_older_revision_does_not_touch_the_newer_one(conn):
    offer(conn, post())
    old, = saved_items.owed(conn, "x")
    offer(conn, post(text="edited"))

    saved_items.ROWS.write(conn, ("x", "1", old.revision), Retry("blip", 1, None, "late"))

    assert state(conn)[1] is None


def test_another_account_supersedes_owed_work_and_a_later_offer_revives_it(conn):
    offer(conn, post("1"), post("2"), account="old")
    saved_items.settle(conn, saved_items.owed(conn, "x")[1])

    saved_items.supersede(conn, "x", "new")

    assert saved_items.owed(conn, "x") == []
    assert tuple(state(conn, "2")) == ("done", None)
    offer(conn, post("1"), account="new")
    saved, = saved_items.owed(conn, "x")
    assert (saved.item, saved.account) == ("1", "new")


def test_an_item_without_an_identity_offers_nothing(conn):
    with pytest.raises(ValueError):
        saved_items.offer(conn, "x", "me", [("", "rev", {})], NOW)
    assert saved_items.owed(conn, "x") == []
