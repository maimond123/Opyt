"""fetch_conversation — TweetDetail chain reconstruction (ancestors + focal + same-author
self-continuation). graphql_get is monkeypatched with a synthetic conversation of the shape
confirmed live 2026-07-16, so this pins the PARSER without a network call."""
from __future__ import annotations

from pipeline.ingestion import x_graphql_core as core


def _raw(tid, author_id, screen, text):
    """A minimal raw X tweet `result` that x_graphql._normalize can parse."""
    return {
        "rest_id": tid,
        "core": {"user_results": {"result": {
            "rest_id": author_id,
            "core": {"screen_name": screen, "name": screen.title()},
            "legacy": {"screen_name": screen, "name": screen.title()},
        }}},
        "legacy": {"id_str": tid, "full_text": text, "created_at": "", "entities": {}},
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
    monkeypatch.setattr(core, "resolve_query_id", lambda *a, **k: "QID")
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: payload)


def test_reply_chain_is_ancestors_then_focal(monkeypatch):
    # parent (by @alice) → focal reply (by @bob). The debate context is the parent.
    payload = _convo([
        _entry_tweet("1", _raw("1", "100", "alice", "what's a good project?")),
        _entry_tweet("2", _raw("2", "200", "bob", "@alice here's my take")),
        {"entryId": "cursor-bottom-x", "content": {}},
    ])
    _patch(monkeypatch, payload)
    chain = core.fetch_conversation("2", {}, {})
    assert [t["id"] for t in chain] == ["1", "2"]
    assert chain[0]["author"]["userName"] == "alice" and chain[1]["author"]["userName"] == "bob"


def test_same_author_self_continuation_included_others_excluded(monkeypatch):
    # focal (bob) with a self-thread continuation (bob again) AND someone else's reply (carol).
    payload = _convo([
        _entry_tweet("1", _raw("1", "100", "alice", "root question")),
        _entry_tweet("2", _raw("2", "200", "bob", "@alice part 1")),
        _entry_module("9", [
            _raw("3", "200", "bob", "part 2 of my thread"),     # same author → included
            _raw("4", "300", "carol", "nice one bob"),          # other author → excluded
        ]),
    ])
    _patch(monkeypatch, payload)
    chain = core.fetch_conversation("2", {}, {})
    assert [t["id"] for t in chain] == ["1", "2", "3"]          # carol's #4 dropped


def test_no_context_returns_empty(monkeypatch):
    # focal alone (a standalone tweet with no ancestors/continuation) → [] (render solo).
    payload = _convo([_entry_tweet("2", _raw("2", "200", "bob", "solo take"))])
    _patch(monkeypatch, payload)
    assert core.fetch_conversation("2", {}, {}) == []


def test_focal_not_found_returns_empty(monkeypatch):
    payload = _convo([_entry_tweet("1", _raw("1", "100", "alice", "unrelated"))])
    _patch(monkeypatch, payload)
    assert core.fetch_conversation("999", {}, {}) == []
