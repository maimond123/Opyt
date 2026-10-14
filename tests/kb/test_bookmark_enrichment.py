"""The saved-post import (`SavedXPosts`), its photos (`SavedXImages`) and the TweetDetail lane
converge without losing images or saved conversations, and a post with both is written at most
twice."""
import json
from datetime import datetime, timezone

import pytest

from pipeline.platforms import sessions
from pipeline.platforms.x import conversation as x_conversation
import pipeline.platforms.x.rate as x_rate
from pipeline.ingestion.utils import SyncAuthError
from pipeline.curation import bookmarks, saved_images, saved_items
from pipeline.forget import atoms as forget
from pipeline.kb import derive, footprint_photos, schema
from pipeline.chart_read import MediaRead
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import Wait
from pipeline.worker.quota_loop.source import Context
from tests.kb.saved_x_support import read_images
from pipeline.oracles.sources.threads import state
from pipeline.oracles.sources.loops import x_threads_loop
from pipeline.retrieval import reads

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)
LATER = datetime(2026, 10, 6, tzinfo=timezone.utc)   # past any retry's wait


def _norm(tid="100", *, photos=0, reply_count=0):
    return {"id": tid, "text": f"the saved take {tid}", "replyCount": reply_count,
            "isReply": False, "author": {"id": "1", "name": "U", "userName": "u"},
            "url": f"https://x.com/u/status/{tid}", "createdAt": "2026-08-01T12:00:00Z",
            "entities": {"urls": []}, "extendedEntities": {"media": [
                {"type": "photo", "media_url_https": f"https://pbs/{tid}-{i}.jpg"}
                for i in range(photos)]}}


@pytest.fixture()
def walk(kb_home, monkeypatch):
    norm = _norm(photos=1, reply_count=3)
    pages, calls = [], []

    def fetch(tid, *args, **kwargs):
        calls.append(tid)
        page = pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page

    monkeypatch.setattr(x_conversation, "fetch_thread_page", fetch)
    monkeypatch.setattr(sessions, "x_session", lambda *args: {})
    monkeypatch.setattr(derive, "derive_x", lambda n: {
        "who_id": "x:user:1", "who_name": "U", "who_handle": "u", "who_site": None,
        "when_ts": "2026-08-01T12:00:00Z", "when_precision": "second",
        "source_tags": [], "about_entities": [], "description": "d"})
    conn = schema.connect()
    footprint_photos.init_schema(conn)          # the tail source on this loop reads it
    loop = x_threads_loop(meter=lambda: None)
    loop.ctx.cache.update(session={})
    yield conn, norm, pages, calls, loop
    conn.close()


def _import(conn, norm, *, images=True):
    """The Bookmarks listing offers the post, one saved-X unit saves it, and (unless `images` is
    off) the saved-images source reads its photos: what the save came to, or None when nothing
    was owed (an unchanged post already saved)."""
    saved_items.init_schema(conn)
    saved_images.init_schema(conn)
    saved_items.offer(conn, "x", "me", [(norm["id"], saved_items.x_revision(norm), norm)], NOW)
    source = bookmarks.SavedXPosts()
    unit = source.next_unit(conn, NOW)
    if unit is None:
        return None
    ctx = Context(bookmarks.QUOTA)
    status = source.record(conn, unit, source.fetch(conn, unit, ctx)).status
    if images:
        read_images(conn)
    return status


def _md(conn):
    return " ".join(r[0] for r in conn.execute("SELECT text FROM bodies WHERE atom_id='x:100'"))


def _atom(conn):
    return dict(conn.execute("SELECT * FROM atoms WHERE atom_id='x:100'").fetchone())


def _chain(norm):
    return [norm, {**_norm("101"), "text": "my continuation", "isReply": True,
                   "inReplyToStatusId": "100", "inReplyToUserId": "1"}]


def _work_keys(conn):
    """The saved-post work inputs that once lived on the atom and now live on its saved row."""
    payload = json.loads(_atom(conn)["payload"])
    return {"bookmark_input", "bookmark_chain"} & set(payload)


def _kept(conn):
    return saved_items.rendering(conn, "x", "100")


def test_the_save_reads_no_image_and_the_thread_waits_for_the_image_read(walk, ocr):
    conn, norm, pages, calls, loop = walk
    assert _import(conn, norm, images=False) == "saved"
    assert ocr.calls == [] and "*Image:*" not in _md(conn)
    assert state.owed(conn, "bookmarks") == []                 # waiting on its photo
    assert read_images(conn) == ["settled"]     # thread-owed: the cache only
    assert len(ocr.calls) == 1 and "*Image:*" not in _md(conn)
    assert len(state.owed(conn, "bookmarks")) == 1 and calls == []


def test_a_post_with_a_photo_and_a_thread_is_written_twice(walk, ocr):
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)
    assert _work_keys(conn) == set()
    pages.append(_chain(norm))
    assert loop.step(conn).result == "fetched"
    assert _atom(conn)["version"] == 2                          # the save, then the thread's
    assert "my continuation" in _md(conn) and "*Image:*" in _md(conn)
    assert _work_keys(conn) == set()
    assert [t["id"] for t in _kept(conn).chain] == ["100", "101"]


def test_the_kept_chain_is_rendered_when_a_full_article_re_renders_the_post(walk):
    conn, norm, pages, calls, loop = walk
    node = {"title": "Article", "preview_text": "Preview"}
    norm["article"] = {"article_results": {"result": node}}
    norm["extendedEntities"] = {"media": []}
    _import(conn, norm)
    pages.append(_chain(norm))
    assert loop.step(conn).result == "fetched"
    assert _kept(conn).rendered_from_teaser
    node["content_state"] = {"blocks": [{"type": "unstyled", "text": "The full article body."}]}
    assert _import(conn, norm) == "saved"       # the flag forces the re-render
    assert "The full article body." in _md(conn) and "my continuation" in _md(conn)
    assert not _kept(conn).rendered_from_teaser
    assert _work_keys(conn) == set() and calls == ["100"]


def test_a_photo_given_up_no_longer_holds_the_thread(walk, ocr):
    conn, norm, pages, calls, loop = walk
    ocr.respond(lambda url, context: None)
    _import(conn, norm, images=False)
    conn.execute("UPDATE curation_saved_images SET failures=?", (retries.GIVE_UP_AFTER,))
    conn.commit()
    assert len(state.owed(conn, "bookmarks")) == 1


def test_lane_remints_once_and_repeat_import_preserves_chain(walk, ocr):
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)
    bare = _atom(conn)
    pages.append(_chain(norm))
    assert loop.step(conn).result == "fetched"
    updated = _atom(conn)
    assert updated["version"] == bare["version"] + 1
    assert updated["first_seen"] == bare["first_seen"]
    assert "my continuation" in _md(conn) and "*Image:*" in _md(conn)
    assert conn.execute("SELECT COUNT(*) FROM bodies_fts WHERE bodies_fts MATCH 'continuation'").fetchone()[0]
    assert _import(conn, norm) is None          # unchanged: nothing is owed
    assert state.owed(conn, "bookmarks") == []
    assert _atom(conn)["version"] == updated["version"]
    assert isinstance(loop.step(conn), Wait)
    assert calls == ["100"] and len(ocr.calls) == 1


def test_image_retry_preserves_a_completed_thread(walk, ocr):
    conn, norm, pages, calls, loop = walk
    ocr.respond(lambda url, context: None)
    assert _import(conn, norm) == "saved"       # its photo failed: owed, charged
    assert state.owed(conn, "bookmarks") == []                 # the thread waits for it
    ocr.respond(lambda url, context: MediaRead("a chart of CPI", "chart"))
    assert read_images(conn, now=LATER) == ["settled"]
    pages.append(_chain(norm))
    assert loop.step(conn).result == "fetched"
    assert state.owed(conn, "bookmarks") == []
    assert "a chart of CPI" in _md(conn) and "my continuation" in _md(conn)
    assert _atom(conn)["version"] == 2 and calls == ["100"]


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("unavailable"),
                                     x_rate.XRateLimited("spent"), SyncAuthError("rejected")])
def test_failed_conversation_keeps_import_and_owed_work(walk, failure):
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)
    before = _atom(conn)
    pages.append(failure)
    loop.step(conn)
    assert _atom(conn)["raw_hash"] == before["raw_hash"]
    assert len(state.owed(conn, "bookmarks")) == 1
    assert conn.execute("SELECT COUNT(*) FROM sync_dedup WHERE namespace='x_convo_checked'").fetchone()[0] == 0
    # A new loop after the retry/reset reads this atom; it never needs the bookmark feed again.
    conn.execute("UPDATE atoms SET payload=json_remove(payload, '$.thread_retry')")
    conn.commit()
    pages.append([norm])
    unit = loop.sources[0].next_unit(conn, datetime.now(timezone.utc))
    assert loop.run_unit(conn, loop.sources[0], unit).result == "fetched"
    assert state.owed(conn, "bookmarks") == []
    assert calls == ["100", "100"]


def test_a_failed_write_retries_saved_response_without_another_request(walk, monkeypatch):
    """The thread atom's write fails after the conversation was read: the read response stays
    with the owed work, so the retry writes it without another TweetDetail request."""
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)
    pages.append(_chain(norm))
    real = schema.put_body

    def locked(*a, **k):
        raise RuntimeError("store refused the body")
    monkeypatch.setattr(schema, "put_body", locked)
    unit = loop.sources[0].next_unit(conn, datetime.now(timezone.utc))
    loop.run_unit(conn, loop.sources[0], unit)
    assert len(state.owed(conn, "bookmarks")) == 1
    assert state.read(conn, "x:100").walk.focused == ("100",)
    assert "my continuation" not in _md(conn)
    monkeypatch.setattr(schema, "put_body", real)
    conn.execute("UPDATE atoms SET payload=json_remove(payload, '$.thread_retry')")
    conn.commit()
    unit = loop.sources[0].next_unit(conn, datetime.now(timezone.utc))
    assert loop.run_unit(conn, loop.sources[0], unit).result == "fetched"
    assert calls == ["100"]


def _stall_after_read(walk, monkeypatch):
    """A saved post whose conversation was read and whose write then failed: its walk is kept."""
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)
    pages.append(_chain(norm))

    def locked(*a, **k):
        raise RuntimeError("store refused the body")
    monkeypatch.setattr(schema, "put_body", locked)
    unit = loop.sources[0].next_unit(conn, datetime.now(timezone.utc))
    loop.run_unit(conn, loop.sources[0], unit)
    return conn


def test_a_walk_in_progress_is_not_in_what_search_and_open_return(walk, monkeypatch):
    """The posts a walk has read so far are working state, tens of thousands of characters on a
    long thread. They stay out of the atom's payload, which a hit and `open` return whole."""
    conn = _stall_after_read(walk, monkeypatch)
    assert any(t["text"] == "my continuation" for t in state.read(conn, "x:100").walk.tweets)
    assert "my continuation" not in _atom(conn)["payload"]
    assert "my continuation" not in json.dumps(reads.kb_open("x:100")["payload"])


def test_forgetting_a_post_takes_its_walk(walk, monkeypatch):
    conn = _stall_after_read(walk, monkeypatch)
    forget.atom(conn, "x:100", confirm=True)
    assert conn.execute("SELECT COUNT(*) FROM thread_walks").fetchone()[0] == 0


def test_a_walk_an_older_version_left_in_the_payload_moves_to_its_table(walk):
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)
    conn.execute("UPDATE atoms SET payload=json_set(payload, '$.thread_walk', json(?)) "
                 "WHERE atom_id='x:100'",
                 (json.dumps({"tweets": _chain(norm), "focused": ["100"], "gone": False}),))
    conn.commit()
    state.move_payload_walks(conn)
    assert state.read(conn, "x:100").walk.focused == ("100",)
    assert "my continuation" not in _atom(conn)["payload"]


def test_deleted_parent_is_not_resurrected(walk, monkeypatch):
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)

    def deleted(*args, **kwargs):
        forget.atom(conn, "x:100", confirm=True)
        return _chain(norm)

    monkeypatch.setattr(x_conversation, "fetch_thread_page", deleted)
    assert loop.step(conn).result == "changed"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sync_dedup WHERE namespace='x_convo_checked'").fetchone()[0] == 0


def test_lost_lease_keeps_an_unchanged_conversation_owed(walk, monkeypatch):
    conn, norm, pages, calls, loop = walk
    _import(conn, norm)

    def lose_lease(*args, **kwargs):
        loop.ctx.lease_lost = lambda: True
        return [norm]

    monkeypatch.setattr(x_conversation, "fetch_thread_page", lose_lease)
    assert loop.step(conn).result == "lease_lost"
    assert len(state.owed(conn, "bookmarks")) == 1


def test_failed_atom_write_does_not_mark_a_standalone_bookmark_complete(walk, monkeypatch):
    conn, norm, pages, calls, loop = walk
    norm["replyCount"] = 0

    def fail(*args, **kwargs):
        raise RuntimeError("store refused the atom")

    monkeypatch.setattr(schema, "put_body", fail)
    with pytest.raises(RuntimeError, match="store refused"):
        _import(conn, norm)
    assert [s.item for s in saved_items.owed(conn, "x")] == ["100"]
    assert conn.execute("SELECT COUNT(*) FROM sync_dedup WHERE namespace='x_convo_checked'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0


def test_a_resolved_conversation_cannot_hide_an_article_still_waiting_for_its_body(walk):
    conn, norm, pages, calls, loop = walk
    node = {"title": "Article", "preview_text": "Preview"}
    norm["article"] = {"article_results": {"result": node}}
    _import(conn, norm)
    pages.append([norm])
    assert loop.step(conn).result == "fetched"
    node["content_state"] = {"blocks": [{"type": "unstyled", "text": "The full article body."}]}
    assert _import(conn, norm) == "saved"       # a new revision of the post
    assert "The full article body." in _md(conn)
    assert state.owed(conn, "bookmarks") == [] and calls == ["100"]
