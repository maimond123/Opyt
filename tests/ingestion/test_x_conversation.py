"""fetch_conversation — TweetDetail chain reconstruction (ancestors + focal + the
author's self-thread). The client is a fake answering with a synthetic conversation of the shape
confirmed live, so this pins the PARSER without a network call."""
from __future__ import annotations

from tests.ingestion.x_client_fake import FakeXClient
from pipeline.platforms.x import conversation as x_conversation

CLIENT = FakeXClient()


def _raw(tid, author_id, screen, text, reply_to=None):
    """A minimal raw X tweet `result` that x.records.normalize_tweet can parse. `reply_to`: the
    (tweet id, user id) it replies to."""
    replying = ({"in_reply_to_status_id_str": reply_to[0], "in_reply_to_user_id_str": reply_to[1]}
                if reply_to else {})
    return {
        "rest_id": tid,
        "core": {"user_results": {"result": {
            "rest_id": author_id,
            "core": {"screen_name": screen, "name": screen.title()},
            "legacy": {"screen_name": screen, "name": screen.title()},
        }}},
        "legacy": {"id_str": tid, "full_text": text, "created_at": "", "entities": {},
                   **replying},
    }


def _entry_tweet(tid, raw):
    return {"entryId": f"tweet-{tid}", "content": {"itemContent": {"tweet_results": {"result": raw}}}}


def _entry_module(mod_id, raws):
    items = [{"item": {"itemContent": {"tweet_results": {"result": r}}}} for r in raws]
    return {"entryId": f"conversationthread-{mod_id}", "content": {"items": items}}


def _convo(entries):
    return {"data": {"threaded_conversation_with_injections_v2": {
        "instructions": [{"type": "TimelineAddEntries", "entries": entries}]}}}


def _patch(monkeypatch, payload):
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: payload)


def test_reply_chain_is_ancestors_then_focal(monkeypatch):
    # parent (by @alice) → focal reply (by @bob). The debate context is the parent.
    payload = _convo([
        _entry_tweet("1", _raw("1", "100", "alice", "what's a good project?")),
        _entry_tweet("2", _raw("2", "200", "bob", "@alice here's my take")),
        {"entryId": "cursor-bottom-x", "content": {}},
    ])
    _patch(monkeypatch, payload)
    chain = x_conversation.fetch_conversation("2", CLIENT)
    assert [t["id"] for t in chain] == ["1", "2"]
    assert chain[0]["author"]["userName"] == "alice" and chain[1]["author"]["userName"] == "bob"


def test_self_thread_included_replies_to_others_excluded(monkeypatch):
    # focal (bob) with a self-thread continuation (bob again), someone else's reply (carol), and
    # bob's answer to carol.
    payload = _convo([
        _entry_tweet("1", _raw("1", "100", "alice", "root question")),
        _entry_tweet("2", _raw("2", "200", "bob", "@alice part 1", ("1", "100"))),
        _entry_module("9", [
            _raw("3", "200", "bob", "part 2 of my thread", ("2", "200")),   # self-thread
            _raw("4", "300", "carol", "nice one bob", ("3", "200")),        # other author
            _raw("5", "200", "bob", "@carol thanks!", ("4", "300")),        # bob to a commenter
        ]),
    ])
    _patch(monkeypatch, payload)
    chain = x_conversation.fetch_conversation("2", CLIENT)
    assert [t["id"] for t in chain] == ["1", "2", "3"]          # carol's #4 and bob's #5 dropped


def test_no_context_returns_empty(monkeypatch):
    # focal alone (a standalone tweet with no ancestors/continuation) → [] (render solo).
    payload = _convo([_entry_tweet("2", _raw("2", "200", "bob", "solo take"))])
    _patch(monkeypatch, payload)
    assert x_conversation.fetch_conversation("2", CLIENT) == []


def test_focal_not_found_returns_empty(monkeypatch):
    payload = _convo([_entry_tweet("1", _raw("1", "100", "alice", "unrelated"))])
    _patch(monkeypatch, payload)
    assert x_conversation.fetch_conversation("999", CLIENT) == []
