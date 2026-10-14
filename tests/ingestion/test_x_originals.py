"""`fetch_user_originals` — one page of an account's originals, offline.

The read every Oracle's X history goes through. What these pin, each of which loses tweets or claims coverage silently if it breaks:

  1. The built cursor is X's own format, byte for byte — a wrong byte starts the read at the top
     (or nowhere) and nothing says so.
  2. The PINNED tweet, re-delivered on every page, is kept but never decides the end.
  3. The end of the timeline: no cursor, or nothing older than `after`.
  4. An unreadable page raises rather than reading as an empty one.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.platforms.x import originals as x_originals
from pipeline.platforms.x import records as x_records
from tests.ingestion.x_client_fake import FakeXClient

CLIENT = FakeXClient()

_UID = "272239060"


def _at(y: int, m: int, d: int, h: int = 12) -> datetime:
    return datetime(y, m, d, h, tzinfo=timezone.utc)


# ── fixtures: the raw X GraphQL shapes Originals ships ────────────────────────

def _result(tid: str, when: datetime, *, uid: str = _UID, reply_to_uid: str | None = None,
            conv: str | None = None) -> dict:
    legacy = {"id_str": tid, "full_text": f"t{tid}",
              "created_at": when.strftime("%a %b %d %H:%M:%S +0000 %Y"),
              "conversation_id_str": conv or tid, "favorite_count": 1, "reply_count": 0,
              "entities": {}}
    if reply_to_uid:
        legacy["in_reply_to_status_id_str"] = conv or "555"
        legacy["in_reply_to_user_id_str"] = reply_to_uid
    return {"rest_id": tid, "legacy": legacy,
            "core": {"user_results": {"result": {
                "rest_id": uid, "core": {"screen_name": "victor", "name": "Victor"}}}}}


def _entry(tid: str, when: datetime, **kw) -> dict:
    return {"entryId": f"tweet-{tid}",
            "content": {"itemContent": {"tweet_results": {"result": _result(tid, when, **kw)}}}}


def _page(entries: list[dict], cursor: str | None, *, pin: dict | None = None) -> dict:
    ins: list[dict] = [{"type": "TimelineClearCache"}]
    if pin is not None:
        ins.append({"type": "TimelinePinEntry", "entry": pin})
    if cursor is not None:
        entries = entries + [{"entryId": f"cursor-bottom-{cursor}",
                              "content": {"cursorType": "Bottom", "value": cursor}}]
    ins.append({"type": "TimelineAddEntries", "entries": entries})
    return {"data": {"user": {"result": {"__typename": "User",
                                         "timeline": {"timeline": {"instructions": ins}}}}}}


def _serve(monkeypatch, pages: list) -> list[dict]:
    """Serve `pages` in order, repeating the LAST forever (X at the end of a timeline). Returns
    the log of calls — op, variables, field toggles — so a test can read what was asked."""
    calls: list[dict] = []

    def _fake_get(op, qid, variables, features, **kw):
        calls.append({"op": op, "variables": dict(variables), **kw})
        return pages[min(len(calls) - 1, len(pages) - 1)]

    monkeypatch.setattr(CLIENT, "get", _fake_get)
    return calls


def _ids(walk) -> list[str]:
    return sorted(t["id"] for t in walk.tweets)


# ── 1. the built cursor ───────────────────────────────────────────────────────

# Two real bottom cursors off @VictorTaelin's Originals, captured live, with the two values
# each one carries: the timeline's entry sort index (NOT a tweet time) and the page's oldest tweet
# id minus one.
_REAL_CURSORS = [
    (2105450626518351849, "2102118439892762935",
     "DAAHCgABHTgQMI2__-kLAAIAAAATMjEwMjExODQzOTg5Mjc2MjkzNQgAAwAAAAIAAA"),
    (2105450626518351828, "2098804747666907374",
     "DAAHCgABHTgQMI2__9QLAAIAAAATMjA5ODgwNDc0NzY2NjkwNzM3NAgAAwAAAAIAAA"),
]


@pytest.mark.parametrize("sort_key,tweet_id,real", _REAL_CURSORS)
def test_the_cursor_encoding_matches_a_real_x_cursor_byte_for_byte(sort_key, tweet_id, real):
    assert x_originals._timeline_cursor(sort_key, tweet_id) == real


# ── the request ───────────────────────────────────────────────────────────────

def test_one_request_asks_for_articles_whole(monkeypatch):
    calls = _serve(monkeypatch, [_page([_entry("10", _at(2025, 6, 20))], "c1")])

    x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))

    assert len(calls) == 1
    first = calls[0]
    assert first["op"] == x_originals.ORIGINALS_OP
    assert first["variables"]["userId"] == _UID
    assert first["field_toggles"] == {"withArticleRichContentState": True,
                                      "withArticlePlainText": False}
    assert first["tolerate_errors"] is True


def test_with_no_before_the_walk_starts_at_the_top(monkeypatch):
    calls = _serve(monkeypatch, [_page([_entry("1", _at(2025, 5, 1))], "c1")])
    x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))
    assert "cursor" not in calls[0]["variables"]


# ── 2. the pin ────────────────────────────────────────────────────────────────

def test_the_walk_names_the_pinned_tweet(monkeypatch):
    """Originals ships the pin only in its `TimelinePinEntry`, never in its chronological slot, so
    a thread rooted at it never shows a self-reply; the caller owes it by name instead."""
    pin = _entry("900", _at(2025, 6, 15))
    _serve(monkeypatch, [_page([_entry("1", _at(2025, 6, 20))], "c1", pin=pin),
                         _page([_entry("0", _at(2025, 5, 1))], "c2", pin=pin)])

    walk = x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))

    assert walk.pinned is not None and walk.pinned["id"] == "900"


def test_no_pin_is_none(monkeypatch):
    _serve(monkeypatch, [_page([_entry("1", _at(2025, 6, 20))], None)])
    assert x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1)).pinned is None


def test_a_page_of_only_the_pin_is_finished_and_keeps_it(monkeypatch):
    """A 2019 pin on an otherwise empty page: kept as content, and never what decides the end."""
    _serve(monkeypatch, [_page([], "c1", pin=_entry("900", _at(2019, 1, 1)))])
    walk = x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))
    assert _ids(walk) == ["900"]
    assert walk.finished is True


# ── 3. the end of the timeline ────────────────────────────────────────────────

def test_no_cursor_is_the_end_of_the_timeline(monkeypatch):
    calls = _serve(monkeypatch, [_page([_entry("1", _at(2025, 6, 20))], None)])
    walk = x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))
    assert len(calls) == 1
    assert walk.finished is True


def test_other_authors_tweets_are_never_returned(monkeypatch):
    _serve(monkeypatch, [_page([_entry("1", _at(2025, 6, 20)),
                                _entry("x", _at(2025, 6, 19), uid="77")], None)])
    walk = x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))
    assert _ids(walk) == ["1"]


@pytest.mark.parametrize(("fresh_dates", "cursor", "expected"), [
    ([], "c1", True),                       # nothing but the pin, or nothing at all
    ([10.0, 20.0], "c1", True),             # every dated post is older than the floor
    ([None], "c1", False),                  # an undated post never ends the timeline
    ([10.0, 50.0], "c1", False),            # a self-reply newer than its root keeps it in range
    ([50.0], None, True),                   # no cursor
    ([50.0], "c1", False),
])
def test_originals_finished_cases(fresh_dates, cursor, expected):
    assert x_originals.originals_finished(fresh_dates, 30.0, cursor) is expected


# ── 4. unreadable vs empty ────────────────────────────────────────────────────

def test_an_unavailable_account_raises(monkeypatch):
    _serve(monkeypatch, [{"data": {"user": {"result": {"__typename": "UserUnavailable",
                                                       "reason": "Suspended"}}}}])
    with pytest.raises(x_originals.XUserUnavailable):
        x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))


@pytest.mark.parametrize("body", [
    {"errors": [{"message": "Internal error"}]},                         # errors only
    {"data": {"user": {"result": {"__typename": "User",
                                  "timeline_v3": {"timeline": {"instructions": []}}}}}},  # renamed
])
def test_a_200_with_no_timeline_raises_never_an_empty_page(monkeypatch, body):
    """Read as an empty page, it would end the walk, and a finished walk claims everything below
    it."""
    _serve(monkeypatch, [body])
    with pytest.raises(x_records.XUnrecognizedPage):
        x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))


def test_an_empty_timeline_is_finished_with_no_tweets(monkeypatch):
    _serve(monkeypatch, [_page([], "c1")])
    walk = x_originals.fetch_user_originals(CLIENT, _UID, after=_at(2025, 6, 1))
    assert walk == x_originals.OriginalsWalk([], True, None)


def test_originals_page_classifies_refusals_and_accepts_own_new_records():
    unavailable = {"data": {"user": {"result": {"__typename": "UserUnavailable",
                                                    "reason": "Suspended"}}}}
    decision = x_originals.originals_page(unavailable, _UID)
    assert decision.action == "unavailable"
    assert decision.refusal == f"x:user:{_UID} timeline unreadable: Suspended"

    unrecognized = x_originals.originals_page({"errors": [{"message": "drift"}]}, _UID)
    assert unrecognized.action == "unrecognized"
    assert "errors" in (unrecognized.refusal or "")

    pin = _entry("9", _at(2025, 6, 10))
    data = _page([_entry("4", _at(2025, 6, 20)),
                  _entry("3", _at(2025, 6, 19), uid="someone-else")], "next", pin=pin)
    accepted = x_originals.originals_page(data, _UID)
    assert accepted.action == "accepted"
    assert accepted.cursor == "next"
    assert accepted.pinned_id == "9"
    assert accepted.pinned["id"] == "9"
    assert [record["id"] for record in accepted.new] == ["9", "4"]


# ── normalize + the shared timeline parser ──
#
# These pin the two pieces every user timeline still shares. Helpers prefixed `_tl`
# so they cannot shadow this file's Originals fixtures.

def _tl_result(tid: str, *, user: str = "alice", uid: str = "11", text: str = "hi",
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


def _tl_entry(tid: str, **kw) -> dict:
    return {"entryId": f"tweet-{tid}",
            "content": {"itemContent": {"tweet_results": {"result": _tl_result(tid, **kw)}}}}


def _tl_module(conv: str, tids: list[str], **kw) -> dict:
    """A `profile-conversation-` module — how X delivers a SELF-THREAD."""
    return {"entryId": f"profile-conversation-{conv}",
            "content": {"items": [
                {"item": {"itemContent": {"tweet_results": {"result": _tl_result(t, conv=conv, **kw)}}}}
                for t in tids]}}


def _tl_cursor(value: str) -> dict:
    return {"entryId": f"cursor-bottom-{value}",
            "content": {"cursorType": "Bottom", "value": value}}


def _tl_page(entries: list[dict], *, pinned: dict | None = None) -> dict:
    ins: list[dict] = [{"type": "TimelineAddEntries", "entries": entries}]
    if pinned is not None:
        ins.append({"type": "TimelinePinEntry", "entry": pinned})
    return {"data": {"user": {"result": {"timeline_v2": {"timeline": {"instructions": ins}}}}}}


# the normalizer carries the filter's two decision fields

def test_normalize_marks_a_retweet():
    assert x_records.normalize_tweet(_tl_result("1", retweet=True))["isRetweet"] is True


def test_normalize_does_not_infer_a_retweet_from_the_text_prefix():
    # "RT @" is a rendering convention, not a fact — anyone can type it.
    assert x_records.normalize_tweet(_tl_result("1", text="RT @someone: stolen take"))["isRetweet"] is False


def test_normalize_carries_the_numeric_reply_target():
    norm = x_records.normalize_tweet(_tl_result("2", uid="11", reply_to_uid="77"))
    assert norm["isReply"] is True
    assert norm["inReplyToUserId"] == "77"      # the filter compares THIS to the author's own id


def test_normalize_leaves_reply_target_empty_on_an_original():
    assert x_records.normalize_tweet(_tl_result("3"))["inReplyToUserId"] == ""


def test_curation_filter_keeps_a_self_thread_off_normalized_output():
    """The point of the two new fields: the shipped filter must work on THIS shape unchanged."""
    from pipeline.kb.ingest_x_footprint import _filter_and_stitch

    tweets = [x_records.normalize_tweet(t) for t in (
        _tl_result("1", uid="11", conv="1"),                        # original
        _tl_result("2", uid="11", reply_to_uid="11", conv="1"),     # SELF-reply → thread continuation
        _tl_result("3", uid="11", reply_to_uid="99"),               # reply to someone else → dropped
        _tl_result("4", uid="11", retweet=True),                    # RT → dropped by the filter
    )]
    groups = _filter_and_stitch(tweets)
    assert [sorted(t["id"] for t in g) for g in groups] == [["1", "2"]]


# every entry shape is read

def test_parse_reads_standalone_thread_and_pinned_entries():
    data = _tl_page([_tl_entry("1"), _tl_module("2", ["2", "3"]), _tl_cursor("c1")],
                 pinned=_tl_entry("9"))
    timeline, unavailable = x_originals._user_timeline_root(data)
    assert unavailable is None
    results, cursor, pinned_id = x_originals._parse_user_timeline(timeline)
    assert [r["rest_id"] for r in results] == ["1", "2", "3", "9"]
    assert cursor == "c1"
    assert pinned_id == "9"         # named, so a walk can keep its arbitrary date out of `reached`


def test_parse_reads_the_legacy_timeline_key():
    """X has shipped this under both `timeline_v2` and `timeline`. A rotation between them presents
    as an empty timeline, which is indistinguishable from an account that posts nothing."""
    data = {"data": {"user": {"result": {"timeline": {"timeline": {
        "instructions": [{"entries": [_tl_entry("1")]}]}}}}}}
    timeline, _ = x_originals._user_timeline_root(data)
    assert [r["rest_id"] for r in x_originals._parse_user_timeline(timeline)[0]] == ["1"]


def test_unavailable_user_is_not_an_empty_timeline():
    data = {"data": {"user": {"result": {"__typename": "UserUnavailable", "reason": "Suspended"}}}}
    _timeline, unavailable = x_originals._user_timeline_root(data)
    assert unavailable == "Suspended"
