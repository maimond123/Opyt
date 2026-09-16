"""The two operations wired for the twitterapi.io cutover, plus the replies timeline's sharp edge.

Offline: every test drives `graphql_get` from a recorded response shape. What is worth pinning
here is not the transport (curl_cffi's job) but the four things a live probe had to discover and
that a fixture written from memory would get wrong:

  1. `UserByScreenName` has NO `legacy` block — the classic paths yield None on a 200 response.
  2. The replies timeline returns OTHER PEOPLE'S tweets, 45% of them in the measured walk.
  3. A page of nothing but conversation partners is not an exhausted cursor.
  4. `TweetResultsByRestIds` answers positionally, padding a missing tweet with `{}`.
"""
from __future__ import annotations

import pytest

from pipeline.ingestion import x_graphql_core as core


@pytest.fixture(autouse=True)
def _no_qid_lookup(monkeypatch):
    """queryId resolution reaches the network (and the on-disk cache). Every test here is about
    what happens to a RESPONSE, so short-circuit it."""
    monkeypatch.setattr(core, "resolve_query_id", lambda op, *a, **k: f"qid-{op}")


# ── UserByScreenName ─────────────────────────────────────────────────────────

def _user_response(**over) -> dict:
    """The live 2026-08-30 shape, trimmed to the fields that are read. Note what is ABSENT: there
    is no `legacy` block anywhere in it, which is the whole point."""
    user = {
        "__typename": "User",
        "rest_id": "975243637",
        "is_blue_verified": True,
        "core": {"name": "Soren Larson", "screen_name": "hypersoren",
                 "created_at": "Wed Nov 28 03:00:45 +0000 2012"},
        "verification": {"verified": False},
        "relationship_counts": {"followers": 5625, "following": 1740},
        "website": {"url": "https://t.co/q5bBvDIUdp"},
        "profile_bio": {
            "description": "applied cybernetics",
            "entities": {
                "url": {"urls": [{"url": "https://t.co/q5bBvDIUdp",
                                  "expanded_url": "http://hypersoren.xyz"}]},
                "description": {"urls": [{"url": "https://t.co/aaa",
                                          "expanded_url": "https://example.com/pod"}]},
            },
        },
    }
    user.update(over)
    return {"data": {"user": {"result": user}}}


def test_the_profile_maps_off_core_and_profile_bio_not_legacy(monkeypatch):
    """⚠️ THE trap. A first pass written from memory reaches for `legacy.followers_count` and
    `legacy.description`; the response has no `legacy` block, so every field comes back None on a
    request that returned 200. This pins the live paths."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: _user_response())
    p = core.fetch_user_profile({}, {}, "@hypersoren")
    assert p["user_id"] == "975243637"          # rest_id — what `x:user:{id}` entities key on
    assert p["handle"] == "hypersoren"          # core.screen_name
    assert p["display_name"] == "Soren Larson"  # core.name
    assert p["bio"] == "applied cybernetics"    # profile_bio.description
    assert p["followers"] == 5625               # relationship_counts.followers


def test_the_website_is_expanded_past_tco(monkeypatch):
    """`website.url` is the SHORTENED form; the expansion lives only in the entities block beside
    it. Reading `website.url` alone hands every caller a t.co link."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: _user_response())
    p = core.fetch_user_profile({}, {}, "hypersoren")
    assert p["website"] == "http://hypersoren.xyz"
    assert p["bio_urls"] == ["https://example.com/pod"]


def test_blue_verification_counts_as_verified(monkeypatch):
    """Two separate flags, and the consumers only ever wanted "has a check". `verification.verified`
    is legacy blue and false for nearly everyone now, so reading it alone reports almost every
    account as unverified."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: _user_response())
    assert core.fetch_user_profile({}, {}, "hypersoren")["verified"] is True


@pytest.mark.parametrize("response, why", [
    ({"data": {}}, "handle does not exist — no `user` key at all"),
    ({"data": {"user": {"result": {"__typename": "UserUnavailable",
                                   "reason": "Suspended"}}}}, "suspended"),
    ({"data": {"user": {"result": {"__typename": "User"}}}}, "no rest_id to key on"),
])
def test_an_unreadable_account_is_none_not_an_exception(monkeypatch, response, why):
    """Returns None rather than raising `XUserUnavailable` like `fetch_user_tweets` does, because
    neither caller distinguishes: both report "unresolved" for a suspended account and a network
    failure alike."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: response)
    assert core.fetch_user_profile({}, {}, "someone") is None, why


def test_a_blank_handle_never_reaches_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not have made a request")
    monkeypatch.setattr(core, "graphql_get", boom)
    assert core.fetch_user_profile({}, {}, "  @  ".strip("@ ")) is None


# ── TweetResultsByRestIds ────────────────────────────────────────────────────

def _tweet_node(tid: str) -> dict:
    return {"__typename": "Tweet", "rest_id": tid,
            "core": {"user_results": {"result": {
                "rest_id": "1", "core": {"name": "A", "screen_name": "a"},
                "legacy": {"name": "A", "screen_name": "a"}}}},
            "legacy": {"full_text": f"post {tid}", "created_at": "Mon Jul 27 00:24:29 +0000 2026",
                       "favorite_count": 1, "conversation_id_str": tid, "entities": {}}}


def test_a_missing_tweet_is_dropped_not_returned_as_a_placeholder(monkeypatch):
    """X answers positionally: ask for three ids where one is deleted and the middle slot comes
    back `{}`. Passing that through would make a caller's index silently mean a different tweet."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: {"data": {"tweetResult": [
        {"result": _tweet_node("111")}, {}, {"result": _tweet_node("333")}]}})
    got = core.fetch_tweets_by_ids({}, {}, ["111", "222", "333"])
    assert [t["id"] for t in got] == ["111", "333"]


def test_an_empty_id_list_never_reaches_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not have made a request")
    monkeypatch.setattr(core, "graphql_get", boom)
    assert core.fetch_tweets_by_ids({}, {}, []) == []
    assert core.fetch_tweets_by_ids({}, {}, ["", "  "]) == []


def test_past_the_per_request_maximum_it_raises_rather_than_truncating(monkeypatch):
    """No chunking loop is written, because no caller batches. The alternative to raising is
    letting X drop the overflow, and losing tweets quietly is the one failure this must not have."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: {"data": {"tweetResult": []}})
    with pytest.raises(ValueError, match="at most 100 ids"):
        core.fetch_tweets_by_ids({}, {}, [str(i) for i in range(101)])


# ── The replies timeline ─────────────────────────────────────────────────────

def _timeline_page(entries: list, cursor: str | None) -> dict:
    items = list(entries)
    if cursor:
        items.append({"entryId": "cursor-bottom-0",
                      "content": {"cursorType": "Bottom", "value": cursor}})
    return {"data": {"user": {"result": {"timeline_v2": {"timeline": {
        "instructions": [{"entries": items}]}}}}}}


def _conversation_entry(*tweets: dict) -> dict:
    """How the replies timeline ships EVERYTHING — as a `profile-conversation-` module holding the
    whole exchange, the tweet replied to as well as the reply."""
    return {"entryId": "profile-conversation-1", "content": {"items": [
        {"item": {"itemContent": {"tweet_results": {"result": t}}}} for t in tweets]}}


def _by(uid: str, tid: str) -> dict:
    n = _tweet_node(tid)
    n["core"]["user_results"]["result"]["rest_id"] = uid
    return n


def test_the_replies_walk_drops_other_peoples_tweets(monkeypatch):
    """⚠️ Measured 2026-08-30: a raw @hypersoren replies walk returned 321 tweets from 92 DISTINCT
    authors, 145 of them (45%) not his. `_filter_and_stitch` would NOT have caught them — it drops
    retweets and replies-to-others, and another author's plain original is neither."""
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: _timeline_page(
        [_conversation_entry(_by("999", "partner-1"), _by("42", "mine-1"))], None))
    got = core.fetch_user_tweets({}, {}, "42", pages=1, timeline="replies")
    assert [t["id"] for t in got] == ["mine-1"]


def test_a_page_of_only_conversation_partners_does_not_end_the_walk(monkeypatch):
    """On a replies walk that is an ordinary page, not an exhausted cursor. Stopping on "no new
    tweet by this author" would end the walk the first time every reply on a page sat under
    someone else's post — which is exactly what lost 2 tweets in the live comparison."""
    pages = [
        _timeline_page([_conversation_entry(_by("999", "partner-1"))], "c1"),
        _timeline_page([_conversation_entry(_by("999", "partner-2"), _by("42", "mine-1"))], None),
    ]
    monkeypatch.setattr(core, "graphql_get", lambda *a, **k: pages.pop(0))
    got = core.fetch_user_tweets({}, {}, "42", pages=5, timeline="replies")
    assert [t["id"] for t in got] == ["mine-1"]


def test_a_page_that_is_entirely_re_seen_does_end_the_walk(monkeypatch):
    """The terminator must still fire, or X's forever-advancing cursor makes this run to the cap.
    Re-seen ids count as neither new nor other — the page added nothing at all."""
    calls = []

    def fake(op, qid, variables, *a, **k):
        calls.append(variables.get("cursor"))
        return _timeline_page([_conversation_entry(_by("999", "p1"), _by("42", "m1"))], "c1")

    monkeypatch.setattr(core, "graphql_get", fake)
    got = core.fetch_user_tweets({}, {}, "42", pages=10, timeline="replies")
    assert [t["id"] for t in got] == ["m1"]
    assert len(calls) == 2, "should have stopped on the first fully-re-seen page"


def test_the_posts_timeline_is_the_default_and_unchanged(monkeypatch):
    seen = {}

    def fake(op, *a, **k):
        seen["op"] = op
        return _timeline_page(
            [{"entryId": "tweet-1", "content": {"itemContent": {
                "tweet_results": {"result": _by("42", "t1")}}}}], None)

    monkeypatch.setattr(core, "graphql_get", fake)
    assert [t["id"] for t in core.fetch_user_tweets({}, {}, "42")] == ["t1"]
    assert seen["op"] == "UserTweets"


def test_an_unknown_timeline_raises_rather_than_silently_walking_posts(monkeypatch):
    with pytest.raises(ValueError, match="timeline must be one of"):
        core.fetch_user_tweets({}, {}, "42", timeline="likes")


# ── The per-operation request budget ─────────────────────────────────────────
# Every X read passes through `graphql_get`, so this is where the budget belongs. What is pinned
# here is that it reads the SERVER's meter rather than a constant, that each operation gets its
# own bucket, and that a spent bucket is refused LOCALLY — the request X would 429 is never
# issued, so it never counts against the window we are protecting.

class _Resp:
    def __init__(self, status=200, headers=None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload if payload is not None else {"data": {}}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


@pytest.fixture(autouse=True)
def _clear_rate_state():
    core._RATE_STATE.clear()
    yield
    core._RATE_STATE.clear()


def _drive(monkeypatch, responses):
    """Point `graphql_get` at a canned response sequence and hand back the issued-op log."""
    issued = []

    class _Cffi:
        @staticmethod
        def get(url, **kw):
            issued.append(url.rsplit("/", 1)[-1])
            return responses.pop(0)

    import sys, types
    mod = types.ModuleType("curl_cffi")
    mod.requests = _Cffi
    monkeypatch.setitem(sys.modules, "curl_cffi", mod)
    return issued


def test_the_budget_is_read_off_the_servers_own_headers(monkeypatch):
    """Self-calibrating: nothing here encodes X's numbers, so a change to them needs no code
    change. An absent header must leave the state untouched rather than record a zero — an
    unmetered response and a spent bucket are opposite facts."""
    issued = _drive(monkeypatch, [
        _Resp(headers={"x-rate-limit-remaining": "37", "x-rate-limit-reset": "9999999999"}),
        _Resp(headers={}),
    ])
    core.graphql_get("UserTweets", "q", {}, {}, {})
    assert core.rate_budget("UserTweets") == (37, 9999999999.0)
    core.graphql_get("UserTweets", "q", {}, {}, {})
    assert core.rate_budget("UserTweets") == (37, 9999999999.0)   # unmetered != spent
    assert issued == ["UserTweets", "UserTweets"]


def test_a_spent_bucket_refuses_before_the_request_goes_out(monkeypatch):
    """THE assertion. Once the meter reads 0 before the reset instant, no request is ISSUED —
    refusing locally is what keeps the 429 (which X counts) from ever being sent."""
    import time
    reset = time.time() + 600
    issued = _drive(monkeypatch, [
        _Resp(headers={"x-rate-limit-remaining": "0", "x-rate-limit-reset": str(reset)}),
    ])
    core.graphql_get("UserTweets", "q", {}, {}, {})
    assert issued == ["UserTweets"]

    with pytest.raises(core.XRateLimited) as e:
        core.graphql_get("UserTweets", "q", {}, {}, {})
    assert issued == ["UserTweets"]                  # nothing new went out
    assert e.value.op == "UserTweets" and e.value.reset_at == pytest.approx(reset)


def test_each_operation_holds_its_own_bucket(monkeypatch):
    """A single global pacer is wrong in both directions — 10x too slow for a 500/window
    operation, and no protection at all against a 51-request burst on a 50/window one."""
    import time
    reset = time.time() + 600
    issued = _drive(monkeypatch, [
        _Resp(headers={"x-rate-limit-remaining": "0", "x-rate-limit-reset": str(reset)}),
        _Resp(headers={"x-rate-limit-remaining": "480", "x-rate-limit-reset": str(reset)}),
        _Resp(headers={"x-rate-limit-remaining": "479", "x-rate-limit-reset": str(reset)}),
    ])
    core.graphql_get("UserTweets", "q", {}, {}, {})
    core.graphql_get("TweetResultsByRestIds", "q", {}, {}, {})

    with pytest.raises(core.XRateLimited):
        core.graphql_get("UserTweets", "q", {}, {}, {})
    core.graphql_get("TweetResultsByRestIds", "q", {}, {}, {})   # its own bucket: unaffected
    assert issued == ["UserTweets", "TweetResultsByRestIds", "TweetResultsByRestIds"]


def test_the_bucket_reopens_once_its_reset_instant_passes(monkeypatch):
    """`x-rate-limit-reset` is a fixed wall-clock instant, not a rolling window, so a spent
    bucket has a known end. Refusing past it would be a deadlock: only a request can refill the
    meter, and the refusal is what prevents the request."""
    import time
    issued = _drive(monkeypatch, [
        _Resp(headers={"x-rate-limit-remaining": "0", "x-rate-limit-reset": str(time.time() - 1)}),
        _Resp(headers={"x-rate-limit-remaining": "49", "x-rate-limit-reset": "9999999999"}),
    ])
    core.graphql_get("UserTweets", "q", {}, {}, {})
    core.graphql_get("UserTweets", "q", {}, {}, {})               # reset passed → allowed
    assert issued == ["UserTweets", "UserTweets"]
    assert core.rate_budget("UserTweets") == (49, 9999999999.0)


def test_a_429_still_records_its_reset_instant(monkeypatch):
    """The one response whose reset we most need. Reading the headers only on success is how a
    spent bucket stays invisible and the next call issues another 429."""
    import time
    reset = time.time() + 300
    _drive(monkeypatch, [
        _Resp(status=429, headers={"x-rate-limit-remaining": "0",
                                   "x-rate-limit-reset": str(reset)}),
    ])
    with pytest.raises(core.XRateLimited) as e:
        core.graphql_get("UserTweets", "q", {}, {}, {})
    assert e.value.reset_at == pytest.approx(reset)
    assert core.rate_budget("UserTweets") == (0, pytest.approx(reset))
