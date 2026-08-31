"""The FREE UserTweets cookie-GraphQL path — the candidate-list Proposer's light content pull.

Pins the three things that can silently lose posts or never terminate, all offline (the network is
monkeypatched; the live scrape is a separate, cookie-dependent check):

  1. `_normalize` emits the two fields the CURATION FILTER decides on. Without them every RT reads
     as an original and every self-thread continuation is dropped as a reply to an unknown target.
  2. `_parse_user_timeline` reads all three entry shapes — standalone, self-thread module, pin.
  3. `fetch_user_tweets` STOPS. X hands back a fresh bottom cursor forever, so a walk that trusts
     the cursor runs until it is rate-limited.
"""
from __future__ import annotations

import pytest

from pipeline.ingestion import x_graphql as xg
from pipeline.ingestion import x_graphql_core as core


# ── fixtures: the raw X GraphQL `result` shape ────────────────────────────────

def _result(tid: str, *, user: str = "alice", uid: str = "11", text: str = "hi",
            retweet: bool = False, reply_to_uid: str | None = None,
            conv: str | None = None) -> dict:
    legacy = {
        "id_str": tid,
        "full_text": text,
        "created_at": "Mon Aug 11 10:00:00 +0000 2026",
        "conversation_id_str": conv or tid,
        "favorite_count": 1,
        "reply_count": 0,
        "entities": {},
    }
    if retweet:
        legacy["retweeted_status_result"] = {"result": {"rest_id": "999"}}
    if reply_to_uid:
        legacy["in_reply_to_status_id_str"] = "555"
        legacy["in_reply_to_user_id_str"] = reply_to_uid
    return {
        "rest_id": tid,
        "legacy": legacy,
        "core": {"user_results": {"result": {
            "rest_id": uid, "legacy": {"screen_name": user, "name": user}}}},
    }


def _entry(tid: str, **kw) -> dict:
    return {"entryId": f"tweet-{tid}",
            "content": {"itemContent": {"tweet_results": {"result": _result(tid, **kw)}}}}


def _module(conv: str, tids: list[str], **kw) -> dict:
    """A `profile-conversation-` module — how X delivers a SELF-THREAD."""
    return {"entryId": f"profile-conversation-{conv}",
            "content": {"items": [
                {"item": {"itemContent": {"tweet_results": {"result": _result(t, conv=conv, **kw)}}}}
                for t in tids]}}


def _cursor(value: str) -> dict:
    return {"entryId": f"cursor-bottom-{value}",
            "content": {"cursorType": "Bottom", "value": value}}


def _page(entries: list[dict], *, pinned: dict | None = None) -> dict:
    ins: list[dict] = [{"type": "TimelineAddEntries", "entries": entries}]
    if pinned is not None:
        ins.append({"type": "TimelinePinEntry", "entry": pinned})
    return {"data": {"user": {"result": {"timeline_v2": {"timeline": {"instructions": ins}}}}}}


# ── 1. the normalizer carries the filter's two decision fields ────────────────

def test_normalize_marks_a_retweet():
    assert xg._normalize(_result("1", retweet=True))["isRetweet"] is True


def test_normalize_does_not_infer_a_retweet_from_the_text_prefix():
    # "RT @" is a rendering convention, not a fact — anyone can type it.
    assert xg._normalize(_result("1", text="RT @someone: stolen take"))["isRetweet"] is False


def test_normalize_carries_the_numeric_reply_target():
    norm = xg._normalize(_result("2", uid="11", reply_to_uid="77"))
    assert norm["isReply"] is True
    assert norm["inReplyToUserId"] == "77"      # the filter compares THIS to the author's own id


def test_normalize_leaves_reply_target_empty_on_an_original():
    assert xg._normalize(_result("3"))["inReplyToUserId"] == ""


def test_curation_filter_keeps_a_self_thread_off_normalized_output():
    """The point of the two new fields: the shipped filter must work on THIS shape unchanged."""
    from pipeline.kb.ingest_x_footprint import _filter_and_stitch

    tweets = [xg._normalize(t) for t in (
        _result("1", uid="11", conv="1"),                        # original
        _result("2", uid="11", reply_to_uid="11", conv="1"),     # SELF-reply → thread continuation
        _result("3", uid="11", reply_to_uid="99"),               # reply to someone else → dropped
        _result("4", uid="11", retweet=True),                    # RT → dropped by the filter
    )]
    groups = _filter_and_stitch(tweets)
    assert [sorted(t["id"] for t in g) for g in groups] == [["1", "2"]]


# ── 2. every entry shape is read ──────────────────────────────────────────────

def test_parse_reads_standalone_thread_and_pinned_entries():
    data = _page([_entry("1"), _module("2", ["2", "3"]), _cursor("c1")],
                 pinned=_entry("9"))
    timeline, unavailable = core._user_timeline_root(data)
    assert unavailable is None
    results, cursor = core._parse_user_timeline(timeline)
    assert [r["rest_id"] for r in results] == ["1", "2", "3", "9"]
    assert cursor == "c1"


def test_parse_reads_the_legacy_timeline_key():
    """X has shipped this under both `timeline_v2` and `timeline`. A rotation between them presents
    as an empty timeline, which is indistinguishable from an account that posts nothing."""
    data = {"data": {"user": {"result": {"timeline": {"timeline": {
        "instructions": [{"entries": [_entry("1")]}]}}}}}}
    timeline, _ = core._user_timeline_root(data)
    assert [r["rest_id"] for r in core._parse_user_timeline(timeline)[0]] == ["1"]


def test_unavailable_user_is_not_an_empty_timeline():
    data = {"data": {"user": {"result": {"__typename": "UserUnavailable", "reason": "Suspended"}}}}
    _timeline, unavailable = core._user_timeline_root(data)
    assert unavailable == "Suspended"


# ── 3. the walk terminates ────────────────────────────────────────────────────

def _stub_pages(monkeypatch, pages: list[dict]) -> dict:
    """Serve `pages` in order, repeating the LAST one forever — X's real behavior at the end of a
    timeline, and the shape that makes a cursor-trusting loop run until it is rate-limited."""
    calls = {"n": 0}
    monkeypatch.setattr(core, "resolve_query_id", lambda *a, **k: "qid")

    def _fake_get(op, qid, variables, features, headers, **kw):
        i = min(calls["n"], len(pages) - 1)
        calls["n"] += 1
        return pages[i]

    monkeypatch.setattr(core, "graphql_get", _fake_get)
    return calls


def test_one_page_is_one_request(monkeypatch):
    calls = _stub_pages(monkeypatch, [_page([_entry("1"), _entry("2"), _cursor("c1")])])
    out = core.fetch_user_tweets({}, {}, "11")
    assert [t["id"] for t in out] == ["1", "2"]
    assert calls["n"] == 1                       # the design default: ONE request per candidate


def test_walk_stops_when_a_page_adds_nothing_new(monkeypatch):
    # Page 2 repeats page 1's tweets under a FRESH cursor — the forever-advancing-cursor trap.
    p1 = _page([_entry("1"), _cursor("c1")])
    p2 = _page([_entry("1"), _cursor("c2")])
    calls = _stub_pages(monkeypatch, [p1, p2])
    out = core.fetch_user_tweets({}, {}, "11", pages=10)
    assert [t["id"] for t in out] == ["1"]       # deduped
    assert calls["n"] == 2                       # stopped on the first no-new-items page


def test_walk_stops_on_a_repeated_cursor(monkeypatch):
    p1 = _page([_entry("1"), _cursor("c1")])
    p2 = _page([_entry("2"), _cursor("c1")])     # new tweet, SAME cursor
    calls = _stub_pages(monkeypatch, [p1, p2])
    core.fetch_user_tweets({}, {}, "11", pages=10)
    assert calls["n"] == 2


def test_page_ceiling_is_clamped_not_trusted(monkeypatch):
    """Each page brings a new tweet under a new cursor, so ONLY the hard cap can stop this."""
    pages = [_page([_entry(str(i)), _cursor(f"c{i}")]) for i in range(1, 200)]
    calls = _stub_pages(monkeypatch, pages)
    core.fetch_user_tweets({}, {}, "11", pages=10_000)
    assert calls["n"] == core._USERTWEETS_MAX_PAGES


def test_empty_timeline_returns_no_tweets(monkeypatch):
    _stub_pages(monkeypatch, [_page([])])
    assert core.fetch_user_tweets({}, {}, "11") == []


def test_unavailable_account_raises_instead_of_returning_empty(monkeypatch):
    _stub_pages(monkeypatch, [
        {"data": {"user": {"result": {"__typename": "UserUnavailable", "reason": "Suspended"}}}}])
    with pytest.raises(core.XUserUnavailable):
        core.fetch_user_tweets({}, {}, "11")
