"""SavedXPosts saves a batch of offered posts per unit, one post per transaction: each post's atom,
checked conversation and done-state land together, a post refused for its own reason stays owed,
and its unread photos are owed to SavedXImages, which reads them and re-saves the post."""

import json
from datetime import datetime, timezone

import pytest

from pipeline.curation import bookmarks, conversations, saved_images, saved_items, saved_x_images
from pipeline.kb import derive, schema
from pipeline.chart_read import MediaRead
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Changed, Context
from tests.kb.saved_x_support import read_images

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
LATER = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)   # past any retry's wait


def _norm(tid="100", *, photos=0, reply_count=0):
    return {"id": tid, "text": f"the saved take {tid}", "replyCount": reply_count,
            "isReply": False, "author": {"id": "1", "name": "U", "userName": "u"},
            "url": f"https://x.com/u/status/{tid}", "createdAt": "2026-08-01T12:00:00Z",
            "entities": {"urls": []}, "extendedEntities": {"media": [
                {"type": "photo", "media_url_https": f"https://pbs/{tid}-{i}.jpg"}
                for i in range(photos)]}}


@pytest.fixture()
def conn(kb_home, monkeypatch):
    monkeypatch.setattr(derive, "derive_x", lambda n: {
        "who_id": "x:user:1", "who_name": "U", "who_handle": "u", "who_site": None,
        "when_ts": "2026-08-01T12:00:00Z", "when_precision": "second",
        "source_tags": [], "about_entities": [], "description": "d"})
    c = schema.connect()
    saved_items.init_schema(c)
    saved_images.init_schema(c)
    yield c
    c.close()


def offer(conn, *posts):
    saved_items.offer(conn, "x", "me", [(p["id"], saved_items.x_revision(p), p) for p in posts], NOW)


def run(conn, ctx=None):
    """One unit: the source's outcome, or None when nothing is owed."""
    source = bookmarks.SavedXPosts()
    unit = source.next_unit(conn, NOW)
    if unit is None:
        return None
    ctx = ctx or Context("saved:x")
    return source.record(conn, unit, source.fetch(conn, unit, ctx))


def md(conn, atom_id="x:100"):
    return " ".join(r[0] for r in conn.execute("SELECT text FROM bodies WHERE atom_id=?", (atom_id,)))


def work_keys(conn, atom_id="x:100"):
    """The saved-post work keys found in an atom's payload (they belong on its saved row)."""
    payload = json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id=?",
                                      (atom_id,)).fetchone()[0])
    return {"bookmark_input", "bookmark_chain"} & set(payload)


def checked(conn):
    return {r[0] for r in conn.execute(
        "SELECT item_id FROM sync_dedup WHERE namespace='x_convo_checked'")}


def test_a_standalone_post_lands_checked_and_done_and_its_photo_is_read_after(
        conn, ocr):
    offer(conn, _norm(photos=1))

    outcome = run(conn)

    assert (outcome.status, outcome.failure) == ("saved", None)
    assert "*Image:*" not in md(conn) and ocr.calls == []
    assert work_keys(conn) == set()
    assert [r.saved.item for r in saved_images.owed(conn, "x")] == ["100"]
    assert read_images(conn) == ["resaved"]
    assert "*Image:*" in md(conn) and len(ocr.calls) == 1
    assert work_keys(conn) == set()
    assert saved_images.owed(conn, "x") == []
    assert checked(conn) == {"100"}
    assert saved_items.owed(conn, "x") == []
    assert conn.execute("SELECT count FROM curation_signals WHERE entity_id='x:user:1' "
                        "AND signal_type='save'").fetchone()[0] == 1
    assert run(conn) is None


def test_a_post_with_replies_is_done_but_owes_its_conversation(conn):
    offer(conn, _norm(reply_count=3))

    assert run(conn).status == "saved"

    assert checked(conn) == set()
    assert conversations.owed_bookmarks(conn) == ["x:100"]
    assert saved_items.owed(conn, "x") == []


def test_an_unread_photo_leaves_the_post_saved_and_its_photo_owed_until_it_lands(
        conn, ocr):
    ocr.respond(lambda url, context: None)
    offer(conn, _norm(photos=1))
    assert run(conn).status == "saved"
    assert saved_items.owed(conn, "x") == []

    assert read_images(conn) == ["images_pending"]
    row, = saved_images.owed(conn, "x")
    assert (row.retry.status, row.retry.failures) == (retries.BLIP, 1)
    assert "the saved take 100" in md(conn) and "*Image:*" not in md(conn)

    ocr.respond(lambda url, context: MediaRead("a chart of CPI", "chart"))
    assert read_images(conn, now=LATER) == ["resaved"]
    assert "a chart of CPI" in md(conn)
    assert saved_images.owed(conn, "x") == []


def test_a_free_install_leaves_the_photo_owed_and_uncharged_until_a_key_arrives(
        conn, ocr, monkeypatch):
    """With no model to read photos (a free install has no key) the source offers nothing, so a
    photo is never charged toward giving up before a key arrives (a free home once failed
    3,260 photo reads in 15 minutes)."""
    ocr.respond(lambda url, context: MediaRead("a chart of CPI", "chart"))
    offer(conn, _norm(photos=1))
    assert run(conn).status == "saved"
    monkeypatch.delenv("OPENROUTER_API_KEY")

    assert read_images(conn) == []
    assert saved_x_images.SavedXImages().next_due(conn, NOW) is None
    row, = saved_images.owed(conn, "x")
    assert not row.retry.failures

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert read_images(conn) == ["resaved"]
    assert "a chart of CPI" in md(conn)


def test_a_photo_already_read_is_rendered_on_the_first_save(conn, ocr):
    offer(conn, _norm("100", photos=1))
    run(conn)
    read_images(conn)
    offer(conn, {**_norm("200"), "extendedEntities": _norm("100", photos=1)["extendedEntities"]})

    assert run(conn).status == "saved"
    assert "*Image:*" in md(conn, "x:200") and saved_images.owed(conn, "x") == []
    assert len(ocr.calls) == 1


def test_a_batch_is_saved_in_one_unit_post_by_post(conn):
    offer(conn, *[_norm(str(100 + i)) for i in range(3)])

    assert run(conn).status == "saved"

    assert saved_items.owed(conn, "x") == []
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 3     # each with its body


def test_a_rider_that_fails_is_left_owed_and_uncharged(conn):
    """A rider refused for its own reason (re-offered while the unit ran) does not cost the head
    its save, and is left owed, uncharged, for its newer revision."""
    offer(conn, _norm("100"), _norm("101"))
    source = bookmarks.SavedXPosts()
    unit = source.next_unit(conn, NOW)
    offer(conn, _norm("100"), {**_norm("101"), "text": "the edited take 101"})

    outcome = source.record(conn, unit, source.fetch(conn, unit, Context("saved:x")))

    assert outcome.status == "saved"            # the head; its outcome is the unit's
    rider, = saved_items.owed(conn, "x")
    assert rider.item == "101" and rider.retry.failures is None
    assert not conn.execute("SELECT 1 FROM atoms WHERE atom_id='x:101'").fetchone()


def test_a_lost_lease_writes_nothing(conn):
    offer(conn, _norm())
    ctx = Context("saved:x", lease_lost=lambda: True)

    with pytest.raises(Changed):
        run(conn, ctx)

    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert len(saved_items.owed(conn, "x")) == 1


def test_a_newer_offer_during_the_unit_refuses_the_older_write(conn, monkeypatch):
    offer(conn, _norm())
    source = bookmarks.SavedXPosts()
    unit = source.next_unit(conn, NOW)
    offer(conn, {**_norm(), "text": "the edited take"})

    fetched = source.fetch(conn, unit, Context("saved:x"))

    assert source.record(conn, unit, fetched).status == "changed"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert run(conn).status == "saved"
    assert "the edited take" in md(conn)


def test_a_saved_post_already_in_the_store_is_promoted_and_done(conn):
    offer(conn, _norm())
    run(conn)
    conn.execute("UPDATE atoms SET entry_mode='frontier' WHERE atom_id='x:100'")
    conn.execute("UPDATE curation_saved_items SET state='owed'")
    conn.commit()

    assert run(conn).status == "present"

    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='x:100'").fetchone()[0] == \
        "user-saved"
    assert saved_items.owed(conn, "x") == []


@pytest.mark.parametrize("stored, checked, rerender, chain_missing, step", [
    (True, True, False, False, bookmarks.PRESENT),
    (True, True, False, True, bookmarks.PRESENT),
    (True, False, False, True, bookmarks.RECOVER),
    (True, True, True, True, bookmarks.RECOVER),
    (True, False, False, False, bookmarks.RENDER),
    (True, True, True, False, bookmarks.RENDER),
    (False, False, False, False, bookmarks.RENDER),
    (False, True, False, True, bookmarks.RENDER),
])
def test_what_to_save_table(stored, checked, rerender, chain_missing, step):
    assert bookmarks.what_to_save(stored=stored, checked=checked, rerender=rerender,
                                  chain_missing=chain_missing) == step


def test_a_refused_presence_hit_does_not_promote_the_stored_post(conn):
    """A post already stored is promoted to `user-saved` only once its refusals pass."""
    offer(conn, _norm())
    assert run(conn).status == "saved"
    conn.execute("UPDATE atoms SET entry_mode='frontier' WHERE atom_id='x:100'")
    conn.commit()
    offer(conn, {**_norm(), "text": "the edited take"})
    ctx = Context("saved:x", lease_lost=lambda: True)

    with pytest.raises(Changed):
        run(conn, ctx)

    assert conn.execute("SELECT entry_mode FROM atoms WHERE atom_id='x:100'").fetchone()[0] == \
        "frontier"


def test_a_refused_write_leaves_no_author_behind(conn):
    """The author's entity and `save` presence land only with the post's atom."""
    offer(conn, _norm())
    source = bookmarks.SavedXPosts()
    unit = source.next_unit(conn, NOW)
    offer(conn, {**_norm(), "text": "the edited take"})

    assert source.record(conn, unit, source.fetch(conn, unit, Context("saved:x"))).status == \
        "changed"
    assert conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM curation_signals").fetchone()[0] == 0


def _save_signals(conn):
    return [r[0] for r in conn.execute(
        "SELECT entity_id FROM curation_signals WHERE signal_type='save' AND platform='x'")]


def test_a_saved_post_of_the_viewers_own_is_saved_without_making_them_a_candidate(
        conn):
    """The Bookmarks listing never counts the viewer, and neither does saving the post."""
    offer(conn, {**_norm(), "author": {"id": "me", "name": "Me", "userName": "me"}})

    assert run(conn).status == "saved"
    assert _save_signals(conn) == []


def test_a_saved_post_names_the_author_its_atom_records(conn):
    offer(conn, _norm())

    assert run(conn).status == "saved"
    assert _save_signals(conn) == ["x:user:1"]
