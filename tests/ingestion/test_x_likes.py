"""
tests/ingestion/test_x_likes.py

Offline tests for the X Likes candidate-signal scraper. No live network: the fixtures
below are the STANDARD X user-timeline shape (Bookmarks' tweet-timeline family), which is
what the Likes op returns. They pin the parts that must hold regardless of X's session
state and regardless of a not-yet-done live capture:
  - tweet extraction + bottom-cursor parse out of `data.user.result.timeline.timeline`,
  - author extraction incl. the bio outbound URL (the entity-resolution attestation seed)
    AND the TweetWithVisibilityResults unwrap,
  - aggregation: dedup authors by id, count `liked_count` per liked tweet, viewer excluded,
    sort by (-liked_count, handle),
  - pagination TERMINATION on an advancing-cursor-forever mock, deduping by TWEET id (the
    regression that bit Lists — X hands back a fresh, non-empty bottom cursor forever).
"""

from __future__ import annotations

from pipeline.ingestion import x_likes as xlk

VIEWER = "1861260702494957568"   # David's rest_id


# ── Fixtures: standard tweet-timeline shape ──────────────────────────────────

def _user_result(uid, handle, name, bio, site, following, blue):
    legacy = {"description": bio, "followers_count": 1000, "entities": {}}
    if site:
        legacy["entities"] = {"url": {"urls": [{"expanded_url": site}]}}
    return {"__typename": "User", "rest_id": uid, "is_blue_verified": blue,
            "core": {"screen_name": handle, "name": name},
            "legacy": legacy,
            "profile_bio": {"description": bio},
            "relationship_perspectives": {"following": following}}


def _tweet_result(tid, author, visibility_wrapped=False):
    tweet = {"__typename": "Tweet", "rest_id": tid,
             "legacy": {"id_str": tid, "full_text": f"liked tweet {tid}"},
             "core": {"user_results": {"result": author}}}
    if visibility_wrapped:
        return {"__typename": "TweetWithVisibilityResults", "tweet": tweet}
    return tweet


def _tweet_entry(tid, author, visibility_wrapped=False):
    return {"entryId": f"tweet-{tid}", "content": {
        "__typename": "TimelineTimelineItem",
        "itemContent": {"__typename": "TimelineTweet",
                        "tweet_results": {"result": _tweet_result(tid, author,
                                                                  visibility_wrapped)}}}}


def _likes_timeline() -> dict:
    """data.user.result.timeline.timeline — four liked tweets:
      - two by NousResearch (→ liked_count 2),
      - one by karpathy (no site → empty, not a crash),
      - one by the VIEWER themselves (must be excluded), wrapped in
        TweetWithVisibilityResults (must be unwrapped) — plus bottom/top cursors."""
    nous = _user_result("1318419526132862976", "NousResearch", "Nous Research",
                        "open source AI", "http://hermes-agent.nousresearch.com", True, True)
    kp = _user_result("33836629", "karpathy", "Andrej Karpathy",
                      "I like training large deep neural nets.", "", True, True)
    me = _user_result(VIEWER, "davidself", "David", "me", "", False, True)
    return {"data": {"user": {"result": {"__typename": "User",
        "timeline": {"timeline": {"instructions": [{"type": "TimelineAddEntries", "entries": [
            _tweet_entry("t1", nous),
            _tweet_entry("t2", nous),
            _tweet_entry("t3", kp),
            _tweet_entry("t4", me, visibility_wrapped=True),
            {"entryId": "cursor-bottom-y", "content": {
                "__typename": "TimelineTimelineCursor", "cursorType": "Bottom", "value": "0|123"}},
            {"entryId": "cursor-top-z", "content": {
                "__typename": "TimelineTimelineCursor", "cursorType": "Top", "value": "-1|456"}},
        ]}]}}}}}}


# ── parse: tweets + cursor out of the Likes root ─────────────────────────────

def test_parse_timeline_extracts_tweets_and_bottom_cursor():
    tweets, cursor = xlk._parse_timeline(_likes_timeline())
    assert [xlk._tweet_id(t) for t in tweets] == ["t1", "t2", "t3", "t4"]
    assert cursor == "0|123"                     # bottom cursor, not the top one


def test_parse_timeline_empty_root_is_safe():
    # A malformed/absent root → empty result, not a KeyError (fail-safe).
    tweets, cursor = xlk._parse_timeline({"data": {}})
    assert tweets == [] and cursor is None


# ── author extraction incl. visibility unwrap + attestation site ─────────────

def test_tweet_author_extraction_and_site():
    nous = _user_result("n1", "NousResearch", "Nous Research", "AI",
                        "http://hermes-agent.nousresearch.com", True, True)
    kp = _user_result("k1", "karpathy", "Andrej Karpathy", "nets", "", False, True)
    a_nous = xlk._tweet_author(_tweet_result("t1", nous))
    a_kp = xlk._tweet_author(_tweet_result("t2", kp))
    assert a_nous["handle"] == "NousResearch"
    assert a_nous["site"] == "http://hermes-agent.nousresearch.com"   # attestation seed
    assert a_nous["i_follow"] is True
    assert a_kp["site"] == ""                    # no url → empty, not a crash
    assert a_kp["i_follow"] is False


def test_tweet_author_unwraps_visibility_results():
    me = _user_result("m1", "davidself", "David", "me", "", False, True)
    author = xlk._tweet_author(_tweet_result("t9", me, visibility_wrapped=True))
    assert author is not None                    # reached the author THROUGH the wrapper
    assert author["handle"] == "davidself"
    assert xlk._tweet_id(_tweet_result("t9", me, visibility_wrapped=True)) == "t9"


# ── aggregate: liked_count, dedup, self-exclusion, ordering ──────────────────

def test_aggregate_authors_counts_liked_and_excludes_self():
    def a(uid, handle):
        return {"user_id": uid, "handle": handle, "display_name": handle, "bio": "",
                "site": "", "followers_count": 1, "verified": False, "i_follow": False}
    # NousResearch liked twice, karpathy once, the viewer once (must drop)
    authors = [a("nous", "NousResearch"), a("kp", "karpathy"),
               a("nous", "NousResearch"), a(VIEWER, "me")]
    out = xlk.aggregate_authors(authors, VIEWER)
    assert VIEWER not in [c["user_id"] for c in out]        # self excluded
    assert [c["handle"] for c in out] == ["NousResearch", "karpathy"]  # 2 likes ranks first
    assert out[0]["liked_count"] == 2
    assert out[1]["liked_count"] == 1


def test_aggregate_authors_sorts_ties_by_handle():
    def a(uid, handle):
        return {"user_id": uid, "handle": handle, "display_name": handle, "bio": "",
                "site": "", "followers_count": 1, "verified": False, "i_follow": False}
    out = xlk.aggregate_authors([a("z", "Zebra"), a("a", "Apple")], VIEWER)
    assert [c["handle"] for c in out] == ["Apple", "Zebra"]  # equal count → handle asc


# ── end-to-end parse→aggregate over the fixture ──────────────────────────────

def test_fixture_end_to_end_author_counts():
    tweets, _ = xlk._parse_timeline(_likes_timeline())
    authors = [xlk._tweet_author(t) for t in tweets]
    out = xlk.aggregate_authors([a for a in authors if a], VIEWER)
    by_handle = {c["handle"]: c for c in out}
    assert set(by_handle) == {"NousResearch", "karpathy"}   # viewer's own like excluded
    assert by_handle["NousResearch"]["liked_count"] == 2    # two liked tweets by them


# ── pagination termination + tweet-dedup (the Lists regression) ──────────────

def _likes_response(cursor_value: str, tweet_ids: list[str]) -> dict:
    def _auth(uid):
        return _user_result(uid, f"u{uid}", f"u{uid}", "", "", False, False)
    entries = [_tweet_entry(tid, _auth(f"author-{tid}")) for tid in tweet_ids]
    entries.append({"entryId": f"cursor-bottom-{cursor_value}", "content": {
        "__typename": "TimelineTimelineCursor", "cursorType": "Bottom", "value": cursor_value}})
    return {"data": {"user": {"result": {"timeline": {"timeline": {
        "instructions": [{"type": "TimelineAddEntries", "entries": entries}]}}}}}}


def test_fetch_liked_authors_terminates_and_dedups_tweets(monkeypatch):
    calls = {"n": 0}
    def fake_get(op, qid, variables, features, headers, **kw):
        calls["n"] += 1
        # X repeats the SAME liked tweets with an ever-advancing cursor → dedup + stop
        return _likes_response(f"cursor-{calls['n']}", ["t1", "t2", "t3"])
    monkeypatch.setattr(xlk.core, "resolve_query_id", lambda *a, **k: "qid")
    monkeypatch.setattr(xlk.core, "graphql_get", fake_get)

    authors = xlk.fetch_liked_authors(VIEWER, {}, {})
    assert len(authors) == 3         # deduped by tweet id across pages, not 3×N
    assert calls["n"] <= 2           # page1: 3 new → page2: 0 new → stop (never a 429)


def test_fetch_liked_authors_counts_repeat_authors_across_distinct_tweets(monkeypatch):
    # Same author on DIFFERENT tweet ids must NOT be deduped away — that IS the like count.
    shared = _user_result("shared", "sharedauthor", "Shared", "", "", False, False)
    def _resp(cursor_value, tweet_ids):
        entries = [_tweet_entry(tid, shared) for tid in tweet_ids]
        entries.append({"entryId": f"cursor-bottom-{cursor_value}", "content": {
            "__typename": "TimelineTimelineCursor", "cursorType": "Bottom", "value": cursor_value}})
        return {"data": {"user": {"result": {"timeline": {"timeline": {
            "instructions": [{"type": "TimelineAddEntries", "entries": entries}]}}}}}}
    calls = {"n": 0}
    def fake_get(op, qid, variables, features, headers, **kw):
        calls["n"] += 1
        # page1: two distinct tweets by the SAME author; page2 repeats → 0 new → stop
        return _resp(f"c{calls['n']}", ["ta", "tb"])
    monkeypatch.setattr(xlk.core, "resolve_query_id", lambda *a, **k: "qid")
    monkeypatch.setattr(xlk.core, "graphql_get", fake_get)

    authors = xlk.fetch_liked_authors(VIEWER, {}, {})
    assert len(authors) == 2         # ta + tb → two author contributions, both counted
    assert all(a["handle"] == "sharedauthor" for a in authors)
    # and aggregation folds them into one candidate liked twice
    agg = xlk.aggregate_authors(authors, VIEWER)
    assert len(agg) == 1 and agg[0]["liked_count"] == 2


# ── fail-safe: no viewer id → skip, no crash, no write ───────────────────────

def test_sync_likes_skips_without_viewer_id(monkeypatch):
    monkeypatch.setattr(xlk.core, "read_x_cookies", lambda **k: {"auth_token": "x"})
    monkeypatch.setattr(xlk.core, "viewer_id", lambda c: None)
    out = xlk.sync_likes(dry_run=True)
    assert out["skipped"] == "no_viewer_id"
    assert out["candidates"] == 0
