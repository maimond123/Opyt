"""`fetch_self_thread` and `threads_owed` — a thread's whole tail, offline.

The shapes are the ones inspected live (see `fetch_self_thread`'s docstring): focused
on a root, TweetDetail ships the root and one `conversationthread-` module per direct reply, with
the author's linear chain inline in the first; a self-reply to a self-reply that X did not inline
appears only when its parent is the focal.
"""
from __future__ import annotations

import pytest

from pipeline.platforms.x import conversation as x_conversation
from pipeline.platforms.x import records as x_records
import pipeline.platforms.x.rate as x_rate
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb.ingest_x_footprint import _filter_and_stitch
from pipeline.oracles.sources.threads.tail_state import threads_owed
from tests.ingestion.x_client_fake import FakeXClient

CLIENT = FakeXClient()

_ME = "100"
_ROOT = "1000"


def _raw(tid: str, *, uid: str = _ME, parent: str | None = None, parent_uid: str | None = None,
         conv: str = _ROOT, replies: int = 0) -> dict:
    legacy = {"id_str": tid, "full_text": f"t{tid}", "created_at": "Mon Sep 07 10:00:00 +0000 2026",
              "conversation_id_str": conv, "reply_count": replies, "entities": {}}
    if parent:
        legacy["in_reply_to_status_id_str"] = parent
        legacy["in_reply_to_user_id_str"] = parent_uid or _ME
    return {"rest_id": tid, "legacy": legacy,
            "core": {"user_results": {"result": {
                "rest_id": uid, "core": {"screen_name": f"u{uid}", "name": f"U{uid}"}}}}}


def _top(raw: dict) -> dict:
    return {"entryId": f"tweet-{raw['rest_id']}",
            "content": {"itemContent": {"tweet_results": {"result": raw}}}}


def _module(*raws: dict) -> dict:
    return {"entryId": f"conversationthread-{raws[0]['rest_id']}",
            "content": {"items": [{"item": {"itemContent": {"tweet_results": {"result": r}}}}
                                  for r in raws]}}


def _detail(*entries: dict) -> dict:
    return {"data": {"threaded_conversation_with_injections_v2": {"instructions": [
        {"type": "TimelineAddEntries",
         "entries": list(entries) + [{"entryId": "cursor-bottom-x",
                                      "content": {"cursorType": "Bottom", "value": "x"}}]}]}}}


def _serve(monkeypatch, by_focal: dict, *, raise_on: dict | None = None) -> list[dict]:
    """TweetDetail keyed by focal id. Returns the call log."""
    calls: list[dict] = []

    def _fake_get(op, qid, variables, features, **kw):
        focal = variables["focalTweetId"]
        calls.append({"op": op, "focal": focal, **kw})
        if raise_on and focal in raise_on:
            raise raise_on[focal]
        return by_focal[focal]

    monkeypatch.setattr(CLIENT, "get", _fake_get)
    return calls


def _walk(root, author, client):
    from pipeline.oracles.sources.threads.walk import Walk
    from pipeline.oracles.sources.threads.walk import advance
    walk = Walk()
    while focal := walk.next_focal(root):
        walk = advance(walk, root, author, focal,
                       x_conversation.fetch_thread_page(focal, client))
    return list(walk.tweets)


def _ids(tweets) -> list[str]:
    return [t["id"] for t in tweets]


ROOT = _raw(_ROOT, replies=40)


# ── the first screen ──────────────────────────────────────────────────────────

def test_a_linear_chain_arrives_inline_in_one_request(monkeypatch):
    """@BrettHarrison's shape: the chain is the first module, whole. Its end has no replies, so
    there is nowhere further to look."""
    calls = _serve(monkeypatch, {_ROOT: _detail(
        _top(ROOT),
        _module(_raw("1001", parent=_ROOT), _raw("1002", parent="1001"), _raw("1003", parent="1002")),
        _module(_raw("2001", uid="7", parent=_ROOT, parent_uid=_ME)))})

    tail = _walk(_ROOT, _ME, CLIENT)

    assert _ids(tail) == [_ROOT, "1001", "1002", "1003"]
    assert [c["focal"] for c in calls] == [_ROOT]
    assert calls[0]["op"] == x_conversation.TWEETDETAIL_OP
    assert calls[0]["field_toggles"]["withArticleRichContentState"] is True   # a root can be an article


def test_other_authors_and_replies_to_commenters_are_dropped(monkeypatch):
    _serve(monkeypatch, {_ROOT: _detail(
        _top(ROOT),
        _module(_raw("1001", parent=_ROOT)),
        _module(_raw("2001", uid="7", parent=_ROOT, parent_uid=_ME),       # a commenter
                _raw("2002", parent="2001", parent_uid="7")),               # my reply to them
        _module(_raw("3001", uid="8", parent="2001", parent_uid="7")))})   # chatter

    assert _ids(_walk(_ROOT, _ME, CLIENT)) == [_ROOT, "1001"]


def test_a_self_reply_from_another_conversation_is_not_this_thread(monkeypatch):
    _serve(monkeypatch, {_ROOT: _detail(
        _top(ROOT), _module(_raw("1001", parent=_ROOT)),
        _module(_raw("5001", parent="5000", conv="5000")))})
    assert _ids(_walk(_ROOT, _ME, CLIENT)) == [_ROOT, "1001"]


# ── past the first screen: down the author's tree ─────────────────────────────

def test_a_self_reply_to_a_self_reply_is_found_by_refocusing_on_its_parent(monkeypatch):
    """@VictorTaelin's 2096268442970010010: it replies to his own reply, which has replies of its
    own, and is on no page of the root's conversation."""
    a = _raw("1001", parent=_ROOT, replies=26)
    calls = _serve(monkeypatch, {
        _ROOT: _detail(_top(ROOT), _module(a), _module(_raw("2001", uid="7", parent=_ROOT))),
        "1001": _detail(_top(ROOT), _top(a),
                        _module(_raw("1002", parent="1001")),                 # the hidden one
                        _module(_raw("2002", uid="7", parent="1001", parent_uid=_ME)))})

    tail = _walk(_ROOT, _ME, CLIENT)

    assert _ids(tail) == [_ROOT, "1001", "1002"]
    assert [c["focal"] for c in calls] == [_ROOT, "1001"]    # 1002 has no replies: done


def test_only_the_ends_of_the_authors_chains_are_refocused(monkeypatch):
    """The middle of a chain is already followed by the author's own next tweet — re-asking for
    it would only re-read what is held."""
    calls = _serve(monkeypatch, {
        _ROOT: _detail(_top(ROOT), _module(_raw("1001", parent=_ROOT, replies=5),
                                           _raw("1002", parent="1001", replies=3))),
        "1002": _detail(_top(ROOT), _top(_raw("1001", parent=_ROOT, replies=5)),
                        _top(_raw("1002", parent="1001", replies=3)),
                        _module(_raw("1003", parent="1002")))})

    tail = _walk(_ROOT, _ME, CLIENT)

    assert _ids(tail) == [_ROOT, "1001", "1002", "1003"]
    assert [c["focal"] for c in calls] == [_ROOT, "1002"]


def test_walk_continues_past_five_requests(monkeypatch):
    """A chain that keeps going, one hidden tweet per focal."""
    pages = {_ROOT: _detail(_top(ROOT), _module(_raw("1001", parent=_ROOT, replies=1)))}
    for i in range(1001, 1011):
        pages[str(i)] = _detail(_top(_raw(str(i), parent=str(i - 1) if i > 1001 else _ROOT,
                                          replies=1)),
                                _module(_raw(str(i + 1), parent=str(i), replies=int(i < 1010))))
    calls = _serve(monkeypatch, pages)

    tail = _walk(_ROOT, _ME, CLIENT)

    assert len(calls) == 11
    assert _ids(tail) == [str(i) for i in range(1000, 1012)]


# ── final vs retryable ────────────────────────────────────────────────────────

def test_a_root_missing_from_its_own_conversation_is_gone(monkeypatch):
    _serve(monkeypatch, {_ROOT: _detail()})
    assert _walk(_ROOT, _ME, CLIENT) == []


def test_a_root_by_someone_else_is_not_kept(monkeypatch):
    """A thread the author started under somebody else's post: their post is not the author's
    content, and kept here it would weld onto the thread as if it were."""
    theirs = _raw(_ROOT, uid="7", replies=3)
    _serve(monkeypatch, {_ROOT: _detail(_top(theirs),
                                        _module(_raw("1001", parent=_ROOT, parent_uid="7"),
                                                _raw("1002", parent="1001")))})
    assert _ids(_walk(_ROOT, _ME, CLIENT)) == ["1002"]


@pytest.mark.parametrize("exc", [x_rate.XRateLimited("spent", op="TweetDetail", reset_at=1.0),
                                 SyncAuthError("dead session")])
def test_a_spent_bucket_or_a_dead_session_propagates(monkeypatch, exc):
    a = _raw("1001", parent=_ROOT, replies=2)
    _serve(monkeypatch, {_ROOT: _detail(_top(ROOT), _module(a))}, raise_on={"1001": exc})
    with pytest.raises(type(exc)):
        _walk(_ROOT, _ME, CLIENT)


def test_other_failures_propagate_without_completing_the_walk(monkeypatch):
    _serve(monkeypatch, {}, raise_on={_ROOT: RuntimeError("TweetDetail GraphQL 404")})
    with pytest.raises(RuntimeError):
        _walk(_ROOT, _ME, CLIENT)


# ── what the caller does with it ──────────────────────────────────────────────

def test_the_tail_stitches_onto_the_root_through_the_shared_filter(monkeypatch):
    _serve(monkeypatch, {_ROOT: _detail(_top(ROOT), _module(_raw("1001", parent=_ROOT),
                                                            _raw("1002", parent="1001")))})
    walked = [x_records.normalize_tweet(ROOT), x_records.normalize_tweet(_raw("1001", parent=_ROOT)),
              x_records.normalize_tweet(_raw("4000", conv="4000"))]
    tail = _walk(_ROOT, _ME, CLIENT)

    groups = _filter_and_stitch(walked + tail)

    assert sorted(sorted(_ids(g)) for g in groups) == [[_ROOT, "1001", "1002"], ["4000"]]


def test_threads_owed_picks_only_conversations_holding_a_self_reply():
    tweets = [x_records.normalize_tweet(r) for r in (
        _raw("1000", conv="1000"), _raw("1001", parent="1000", conv="1000"),        # owed
        _raw("2000", conv="2000"),                                                   # solo post
        _raw("3001", parent="3000", parent_uid="7", conv="3000"),                    # reply to other
        _raw("4001", uid="7", parent="4000", parent_uid="7", conv="4000"),           # someone else's
    )]
    assert threads_owed(tweets, _ME) == {"1000"}


def test_malformed_response_cannot_complete_a_thread(monkeypatch):
    _serve(monkeypatch, {_ROOT: {"data": {}, "errors": [{"message": "_Missing"}]}})
    with pytest.raises(x_records.XUnrecognizedPage):
        x_conversation.fetch_thread_page(_ROOT, CLIENT)
