"""The two operations wired for the twitterapi.io cutover, and the per-operation request budget.

Offline: every test drives `XClient.graphql_get` from a recorded response shape. What is worth pinning
here is not the transport (curl_cffi's job) but the things a live probe had to discover and that
a fixture written from memory would get wrong:

  1. `UserByScreenName` has NO `legacy` block — the classic paths yield None on a 200 response.
  2. `TweetResultsByRestIds` answers positionally, padding a missing tweet with `{}`.

"""
from __future__ import annotations

import pytest

from pipeline.platforms.x import conversation as x_conversation
import pipeline.platforms.x.rate as x_rate
from pipeline.platforms.x import profile as x_profile
from pipeline.platforms.x.local import LocalXTransport
from pipeline.platforms.x.request import XClient
from tests.ingestion.x_client_fake import FakeXClient

# queryId resolution reaches the network (and the on-disk cache). Every endpoint test here is
# about what happens to a RESPONSE, so the client is a fake that answers it.
CLIENT = FakeXClient()
# The budget tests drive the real client over the real local transport, with curl_cffi faked.
LOCAL = XClient(LocalXTransport({}, {}))


# ── UserByScreenName ─────────────────────────────────────────────────────────

def _user_response(**over) -> dict:
    """A live shape, trimmed to the fields that are read. Note what is ABSENT: there
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
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: _user_response())
    p = x_profile.fetch_user_profile(CLIENT, "@hypersoren")
    assert p["user_id"] == "975243637"          # rest_id — what `x:user:{id}` entities key on
    assert p["handle"] == "hypersoren"          # core.screen_name
    assert p["display_name"] == "Soren Larson"  # core.name
    assert p["bio"] == "applied cybernetics"    # profile_bio.description
    assert p["followers"] == 5625               # relationship_counts.followers


def test_the_website_is_expanded_past_tco(monkeypatch):
    """`website.url` is the SHORTENED form; the expansion lives only in the entities block beside
    it. Reading `website.url` alone hands every caller a t.co link."""
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: _user_response())
    p = x_profile.fetch_user_profile(CLIENT, "hypersoren")
    assert p["website"] == "http://hypersoren.xyz"
    assert p["bio_urls"] == ["https://example.com/pod"]


def test_blue_verification_counts_as_verified(monkeypatch):
    """Two separate flags, and the consumers only ever wanted "has a check". `verification.verified`
    is legacy blue and false for nearly everyone now, so reading it alone reports almost every
    account as unverified."""
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: _user_response())
    assert x_profile.fetch_user_profile(CLIENT, "hypersoren")["verified"] is True


@pytest.mark.parametrize("response, why", [
    ({"data": {}}, "handle does not exist — no `user` key at all"),
    ({"data": {"user": {"result": {"__typename": "UserUnavailable",
                                   "reason": "Suspended"}}}}, "suspended"),
    ({"data": {"user": {"result": {"__typename": "User"}}}}, "no rest_id to key on"),
])
def test_an_unreadable_account_is_none_not_an_exception(monkeypatch, response, why):
    """Returns None rather than raising `XUserUnavailable` like `fetch_user_originals` does, because
    neither caller distinguishes: both report "unresolved" for a suspended account and a network
    failure alike."""
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: response)
    assert x_profile.fetch_user_profile(CLIENT, "someone") is None, why


def test_a_blank_handle_never_reaches_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not have made a request")
    monkeypatch.setattr(CLIENT, "get", boom)
    assert x_profile.fetch_user_profile(CLIENT, "  @  ".strip("@ ")) is None


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
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: {"data": {"tweetResult": [
        {"result": _tweet_node("111")}, {}, {"result": _tweet_node("333")}]}})
    got = x_conversation.fetch_tweets_by_ids(CLIENT, ["111", "222", "333"])
    assert [t["id"] for t in got] == ["111", "333"]


def test_an_empty_id_list_never_reaches_the_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not have made a request")
    monkeypatch.setattr(CLIENT, "get", boom)
    assert x_conversation.fetch_tweets_by_ids(CLIENT, []) == []
    assert x_conversation.fetch_tweets_by_ids(CLIENT, ["", "  "]) == []


def test_past_the_per_request_maximum_it_raises_rather_than_truncating(monkeypatch):
    """No chunking loop is written, because no caller batches. The alternative to raising is
    letting X drop the overflow, and losing tweets quietly is the one failure this must not have."""
    monkeypatch.setattr(CLIENT, "get", lambda *a, **k: {"data": {"tweetResult": []}})
    with pytest.raises(ValueError, match="at most 100 ids"):
        x_conversation.fetch_tweets_by_ids(CLIENT, [str(i) for i in range(101)])


# ── The per-operation request budget ─────────────────────────────────────────
# Every X read passes through `XClient.graphql_get`, so this is where the budget belongs. What is pinned
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
    x_rate._RATE_STATE.clear()
    yield
    x_rate._RATE_STATE.clear()


def _drive(monkeypatch, responses):
    """Point the local transport at a canned response sequence and hand back the issued-op log."""
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
    LOCAL.graphql_get("UserTweets", "q", {}, {})
    assert x_rate.rate_budget("UserTweets") == (37, 9999999999.0)
    LOCAL.graphql_get("UserTweets", "q", {}, {})
    assert x_rate.rate_budget("UserTweets") == (37, 9999999999.0)   # unmetered != spent
    assert issued == ["UserTweets", "UserTweets"]


def test_a_spent_bucket_refuses_before_the_request_goes_out(monkeypatch):
    """THE assertion. Once the meter reads 0 before the reset instant, no request is ISSUED —
    refusing locally is what keeps the 429 (which X counts) from ever being sent."""
    import time
    reset = time.time() + 600
    issued = _drive(monkeypatch, [
        _Resp(headers={"x-rate-limit-remaining": "0", "x-rate-limit-reset": str(reset)}),
    ])
    LOCAL.graphql_get("UserTweets", "q", {}, {})
    assert issued == ["UserTweets"]

    with pytest.raises(x_rate.XRateLimited) as e:
        LOCAL.graphql_get("UserTweets", "q", {}, {})
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
    LOCAL.graphql_get("UserTweets", "q", {}, {})
    LOCAL.graphql_get("TweetResultsByRestIds", "q", {}, {})

    with pytest.raises(x_rate.XRateLimited):
        LOCAL.graphql_get("UserTweets", "q", {}, {})
    LOCAL.graphql_get("TweetResultsByRestIds", "q", {}, {})   # its own bucket: unaffected
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
    LOCAL.graphql_get("UserTweets", "q", {}, {})
    LOCAL.graphql_get("UserTweets", "q", {}, {})               # reset passed → allowed
    assert issued == ["UserTweets", "UserTweets"]
    assert x_rate.rate_budget("UserTweets") == (49, 9999999999.0)


def test_a_429_still_records_its_reset_instant(monkeypatch):
    """The one response whose reset we most need. Reading the headers only on success is how a
    spent bucket stays invisible and the next call issues another 429."""
    import time
    reset = time.time() + 300
    _drive(monkeypatch, [
        _Resp(status=429, headers={"x-rate-limit-remaining": "0",
                                   "x-rate-limit-reset": str(reset)}),
    ])
    with pytest.raises(x_rate.XRateLimited) as e:
        LOCAL.graphql_get("UserTweets", "q", {}, {})
    assert e.value.reset_at == pytest.approx(reset)
    assert x_rate.rate_budget("UserTweets") == (0, pytest.approx(reset))


def test_a_blank_handle_resolves_no_query_id(monkeypatch):
    """B06: the handle check ran after `resolve_query_id`, so a blank handle could discover and
    cache a query id before refusing."""
    def boom(*a, **k):
        raise AssertionError("resolved a query id for a blank handle")
    monkeypatch.setattr(CLIENT, "resolve_query_id", boom)
    assert x_profile.fetch_user_profile(CLIENT, "  @ ") is None
