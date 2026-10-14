"""
tests/ingestion/test_x_lists.py

Offline tests for the X Lists candidate-signal scraper. No live network: the two
GraphQL ops were captured live and the fixtures below are trimmed-but-
faithful copies of those real responses. We pin the parts that must hold regardless of
X's session state:
  - owned-only filter (skip the ListToFollow/suggested module AND any subscribed list
    whose owner != the viewer),
  - member extraction incl. the bio outbound URL (the entity-resolution attestation
    seed),
  - cross-list dedup with UNION of list names, and viewer self-exclusion,
  - `XClient.graphql_get` tolerating a partial `errors` array when `data` is present.
"""

from __future__ import annotations

import sys
import types

import pytest

import pipeline.platforms.x.records as x_records
from pipeline.curation import x_collectors
from pipeline.platforms.x import collections as xl
from pipeline.platforms.x.local import LocalXTransport
from pipeline.platforms.x.request import XClient
from tests.ingestion.page_walk import walk
from tests.ingestion.x_client_fake import FakeXClient

VIEWER = "1861260702494957568"   # the viewer's rest_id (owner of the "Research" list)


# ── Fixtures: trimmed real captures ──────────────────────────────────────────

def _mgmt_timeline() -> dict:
    """data.viewer.list_management_timeline.timeline — a ListToFollow (suggested)
    module, an OwnedSubscribedList module holding one OWNED list (Research) and one
    SUBSCRIBED list (owner != viewer), then a bottom cursor."""
    def _list(id_str, name, mode, count, owner_id):
        return {"item": {"itemContent": {"__typename": "TimelineTwitterList", "list": {
            "id_str": id_str, "name": name, "mode": mode, "member_count": count,
            "user_results": {"result": {"__typename": "User", "rest_id": owner_id}}}}}}

    return {"instructions": [{"type": "TimelineAddEntries", "entries": [
        {"entryId": "list-to-follow-module-1", "content": {
            "__typename": "TimelineTimelineModule",
            "clientEventInfo": {"component": "suggest_list_to_follow",
                                "details": {"timelinesDetails": {"injectionType": "ListToFollow"}}},
            "items": [_list("57801307", "NYC-News-and-info", "Public", 94, "821050")]}},
        {"entryId": "owned-subscribed-list-module-0", "content": {
            "__typename": "TimelineTimelineModule",
            "clientEventInfo": {"component": "suggest_owned_subscribed_list",
                                "details": {"timelinesDetails": {"injectionType": "OwnedSubscribedList"}}},
            "items": [
                _list("1000000000000000001", "Research", "Private", 6, VIEWER),
                _list("555000", "Someone Else's List", "Public", 10, "999888"),
            ]}},
        {"entryId": "cursor-bottom-x", "content": {
            "__typename": "TimelineTimelineCursor", "cursorType": "Bottom",
            "value": "CURSOR_MGMT_NEXT"}},
    ]}]}


def _members_timeline() -> dict:
    """data.list.members_timeline.timeline — three real members + bottom/top cursors."""
    def _user(uid, handle, name, bio, site, following, blue):
        legacy = {"description": bio, "followers_count": 1000, "entities": {}}
        if site:
            legacy["entities"] = {"url": {"urls": [{"expanded_url": site}]}}
        return {"entryId": f"user-{uid}", "content": {
            "__typename": "TimelineTimelineItem",
            "itemContent": {"__typename": "TimelineUser", "user_results": {"result": {
                "__typename": "User", "rest_id": uid, "is_blue_verified": blue,
                "core": {"screen_name": handle, "name": name},
                "legacy": legacy,
                "profile_bio": {"description": bio},
                "relationship_perspectives": {"following": following}}}}}}

    return {"instructions": [{"type": "TimelineAddEntries", "entries": [
        _user("1318419526132862976", "NousResearch", "Nous Research",
              "open source AI", "http://hermes-agent.nousresearch.com", True, True),
        _user("33836629", "karpathy", "Andrej Karpathy",
              "I like training large deep neural nets.", "", True, True),
        _user("1605", "sama", "Sam Altman", "AI is cool i guess",
              "http://blog.samaltman.com", False, True),
        {"entryId": "cursor-bottom-y", "content": {
            "__typename": "TimelineTimelineCursor", "cursorType": "Bottom",
            "value": "0|123"}},
        {"entryId": "cursor-top-z", "content": {
            "__typename": "TimelineTimelineCursor", "cursorType": "Top",
            "value": "-1|456"}},
    ]}]}


# ── owned-only filter ────────────────────────────────────────────────────────

def test_parse_owned_lists_keeps_only_viewer_owned():
    lists, _observed = xl._parse_owned_lists(_mgmt_timeline(), VIEWER)
    assert [l["id"] for l in lists] == ["1000000000000000001"]   # only Research
    assert lists[0]["name"] == "Research"
    assert lists[0]["mode"] == "Private"                          # private is included


def test_parse_owned_lists_excludes_suggested_and_subscribed():
    lists, _observed = xl._parse_owned_lists(_mgmt_timeline(), VIEWER)
    names = {l["name"] for l in lists}
    assert "NYC-News-and-info" not in names        # suggested (ListToFollow) dropped
    assert "Someone Else's List" not in names       # subscribed (owner != viewer) dropped


def test_parse_owned_lists_no_owned_returns_empty():
    # A viewer who owns none of these lists gets nothing (not a crash).
    lists, _observed = xl._parse_owned_lists(_mgmt_timeline(), "000nonexistent")
    assert lists == []


# ── member extraction ────────────────────────────────────────────────────────

def test_parse_members_extracts_users_and_site():
    members = x_records.parse_members(_members_timeline())
    by_handle = {m["handle"]: m for m in members}
    assert set(by_handle) == {"NousResearch", "karpathy", "sama"}
    # bio outbound URL is the attestation seed for entity resolution
    assert by_handle["NousResearch"]["site"] == "http://hermes-agent.nousresearch.com"
    assert by_handle["karpathy"]["site"] == ""          # no url → empty, not a crash
    assert by_handle["NousResearch"]["i_follow"] is True
    assert by_handle["sama"]["i_follow"] is False
    assert by_handle["sama"]["display_name"] == "Sam Altman"


def _modern_user_result() -> dict:
    """The shape X actually returns — captured live from `Following`.
    There is NO `legacy` blob: it was split into typed sub-objects. Trimmed to the keys
    the parser reads, with every value verbatim from the capture."""
    return {
        "__typename": "User", "rest_id": "4593727300", "is_blue_verified": True,
        "core": {"screen_name": "a1zhang", "name": "alex zhang",
                 "created_at": "Thu Dec 24 22:30:58 +0000 2015"},
        "verification": {"verified": False},
        "privacy": {"protected": False},
        "relationship_counts": {"followers": 37190, "following": 983},
        "tweet_counts": {"tweets": 1171, "media_tweets": 168},
        "action_counts": {"favorites_count": 5532},
        "pinned_items": {"tweet_ids_str": ["2079203524395573442"]},
        "location": {"location": "USA"},
        "profile_bio": {
            "description": "phd student @mit_csail @nlp_mit",
            "entities": {"description": {}, "url": {"urls": [
                {"display_url": "alexzhang13.github.io/blog/2025/rlm",
                 "expanded_url": "https://alexzhang13.github.io/blog/2025/rlm",
                 "url": "https://t.co/u0X4GbxJ9K"}]}}},
        "relationship_perspectives": {"following": True},
    }


def test_normalize_user_reads_the_modern_schema_not_legacy():
    """REGRESSION: X dropped the flat `legacy` blob. `site` and `followers_count` had no
    other source, so both silently degraded to ""/0 for EVERY account — no exception, no
    log line. These two assertions are the tripwire: if X moves the fields again, this
    fails loudly instead of the pipeline quietly ingesting blanks."""
    m = x_records.normalize_user(_modern_user_result())
    assert m["site"] == "https://alexzhang13.github.io/blog/2025/rlm", "bio link went blank"
    assert m["followers_count"] == 37190, "follower count went to 0"
    assert m["handle"] == "a1zhang"
    assert m["display_name"] == "alex zhang"
    assert m["bio"] == "phd student @mit_csail @nlp_mit"
    assert m["verified"] is True          # blue-verified, though verification.verified False
    assert m["i_follow"] is True


def test_normalize_user_still_reads_legacy_when_present():
    """The legacy path stays as a fallback — other surfaces may still return it, and a
    partial rollback must not re-break the fields this commit just fixed."""
    m = x_records.normalize_user({
        "__typename": "User", "rest_id": "1605",
        "core": {"screen_name": "sama", "name": "Sam Altman"},
        "legacy": {"description": "AI is cool i guess", "followers_count": 4242,
                   "entities": {"url": {"urls": [{"expanded_url": "http://blog.samaltman.com"}]}}},
    })
    assert m["site"] == "http://blog.samaltman.com"
    assert m["followers_count"] == 4242


# ── cross-list dedup + union ─────────────────────────────────────────────────

def test_aggregate_dedup_unions_list_names_and_excludes_self():
    owned = [{"id": "L1", "name": "AI"}, {"id": "L2", "name": "Founders"}]
    def u(uid, handle):
        return {"user_id": uid, "handle": handle, "display_name": handle, "bio": "",
                "site": "", "followers_count": 1, "verified": False, "i_follow": False}
    members_by_list = {
        "L1": [u("nous", "NousResearch"), u("kp", "karpathy"), u(VIEWER, "me")],
        "L2": [u("kp", "karpathy"), u("sama", "sama")],
    }
    out = x_collectors.aggregate_members(owned, members_by_list, VIEWER)
    handles = [c["handle"] for c in out]
    assert VIEWER not in [c["user_id"] for c in out]     # self excluded
    # karpathy is in BOTH lists → ranks first (breadth), with both names unioned
    assert handles[0] == "karpathy"
    kp = out[0]
    assert kp["list_names"] == ["AI", "Founders"]
    assert kp["list_ids"] == ["L1", "L2"]
    assert set(handles) == {"karpathy", "NousResearch", "sama"}   # 3 distinct


# ── graphql_get tolerates partial errors alongside data ──────────────────────

class _FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload
        self.text = "ok"
        self.headers = {}          # every real response carries them; `graphql_get` meters off it
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


def _install_fake_cffi(monkeypatch, payload):
    fake = types.ModuleType("curl_cffi")
    req = types.ModuleType("curl_cffi.requests")
    req.get = lambda *a, **k: _FakeResp(payload)
    fake.requests = req
    monkeypatch.setitem(sys.modules, "curl_cffi", fake)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", req)


def test_graphql_get_tolerates_partial_errors(monkeypatch):
    payload = {"data": {"viewer": {"ok": True}},
               "errors": [{"code": 214, "message": "decode fail on a suggested list"}]}
    _install_fake_cffi(monkeypatch, payload)
    out = XClient(LocalXTransport({}, {})).graphql_get("ListsManagementPageTimeline", "qid", {},
                                                       {}, tolerate_errors=True)
    assert out["data"]["viewer"]["ok"] is True          # data kept despite errors


def test_graphql_get_raises_on_errors_when_not_tolerated(monkeypatch):
    payload = {"data": None, "errors": [{"code": 32, "message": "bad auth"}]}
    _install_fake_cffi(monkeypatch, payload)
    with pytest.raises(RuntimeError):
        XClient(LocalXTransport({}, {})).graphql_get("Bookmarks", "qid", {}, {},
                                                     tolerate_errors=False)


# ── pagination termination (the bug the live run caught: X hands back a fresh, non-
#    empty bottom cursor forever, so a loop that only stops on empty/repeated cursor
#    runs until a 429). These exercise the LOOP, which the parse tests above do not. ──

def _mgmt_response(cursor_value: str, include_owned: bool) -> dict:
    entries = []
    if include_owned:
        entries.append({"entryId": "owned-module", "content": {
            "__typename": "TimelineTimelineModule",
            "clientEventInfo": {"details": {"timelinesDetails": {"injectionType": "OwnedSubscribedList"}}},
            "items": [{"item": {"itemContent": {"list": {
                "id_str": "1000000000000000001", "name": "Research", "mode": "Private",
                "member_count": 6, "user_results": {"result": {"rest_id": VIEWER}}}}}}]}})
    entries.append({"entryId": f"cursor-bottom-{cursor_value}", "content": {
        "__typename": "TimelineTimelineCursor", "cursorType": "Bottom", "value": cursor_value}})
    return {"data": {"viewer": {"list_management_timeline": {
        "timeline": {"instructions": [{"type": "TimelineAddEntries", "entries": entries}]}}}}}


def _members_response(cursor_value: str, user_ids: list[str]) -> dict:
    entries = [{"entryId": f"user-{uid}", "content": {
        "itemContent": {"user_results": {"result": {
            "__typename": "User", "rest_id": uid,
            "core": {"screen_name": f"u{uid}", "name": f"u{uid}"},
            "legacy": {"description": "", "followers_count": 1, "entities": {}},
            "relationship_perspectives": {"following": False}}}}}} for uid in user_ids]
    entries.append({"entryId": f"cursor-bottom-{cursor_value}", "content": {
        "__typename": "TimelineTimelineCursor", "cursorType": "Bottom", "value": cursor_value}})
    return {"data": {"list": {"members_timeline": {
        "timeline": {"instructions": [{"type": "TimelineAddEntries", "entries": entries}]}}}}}


def test_owned_lists_pages_terminate_on_advancing_cursor(monkeypatch):
    calls = {"n": 0}
    def fake_get(op, qid, variables, features, **kw):
        calls["n"] += 1
        # owned lists arrive only on page 1; every page returns a NEW cursor forever
        return _mgmt_response(f"cursor-{calls['n']}", include_owned=(calls["n"] == 1))
    client = FakeXClient(fake_get)

    lists = walk(lambda cursor: xl.owned_lists_page(client, VIEWER, cursor))
    assert [l["id"] for l in lists] == ["1000000000000000001"]
    assert calls["n"] <= 3          # terminates almost immediately, not ~200 → 429


def test_list_members_pages_terminate_and_dedup(monkeypatch):
    calls = {"n": 0}
    def fake_get(op, qid, variables, features, **kw):
        calls["n"] += 1
        # X repeats the SAME members with an ever-advancing cursor → must dedup + stop
        return _members_response(f"cursor-{calls['n']}", ["1318419526132862976", "33836629", "1605"])
    client = FakeXClient(fake_get)

    members = walk(lambda cursor: xl.list_members_page("L1", client, cursor))
    assert len(members) == 3         # deduped across pages, not 3×N
    assert calls["n"] <= 2           # page1: 3 new → page2: 0 new → stop


# ── Query ids: no built-in List seed (B07) ───────────────────────────────────

class _Sent(Exception):
    pass


def _qid_client(monkeypatch, discovered="fresh-id"):
    """A real `XClient` whose discovery answers `discovered` and whose request records the query id
    it was given, then stops the read."""
    import pipeline.platforms.x.query_ids as query_ids
    monkeypatch.setattr(query_ids, "qid_cache_get", lambda _op: None)
    monkeypatch.setattr(query_ids, "qid_cache_put", lambda *_a: None)
    client = XClient(types.SimpleNamespace(discover_query_id=lambda op, page_url: discovered))
    sent: list[str] = []

    def graphql_get(op, query_id, *a, **k):
        sent.append(query_id)
        raise _Sent()

    monkeypatch.setattr(client, "graphql_get", graphql_get)
    return client, sent


@pytest.mark.parametrize("read", [lambda c: xl.owned_lists_page(c, VIEWER),
                                  lambda c: xl.list_members_page("2075260793176592514", c)])
def test_list_reads_use_a_discovered_query_id_not_a_built_in_seed(monkeypatch, read):
    """B07: a fixed seed beat the cache forever, so clearing a rejected id on a 400/404 could never
    replace it. With no override, the id comes from discovery (and so from the cache after it)."""
    monkeypatch.delenv("X_LISTS_MGMT_QUERY_ID", raising=False)
    monkeypatch.delenv("X_LIST_MEMBERS_QUERY_ID", raising=False)
    client, sent = _qid_client(monkeypatch)
    with pytest.raises(_Sent):
        read(client)
    assert sent == ["fresh-id"]


@pytest.mark.parametrize("env, read", [
    ("X_LISTS_MGMT_QUERY_ID", lambda c: xl.owned_lists_page(c, VIEWER)),
    ("X_LIST_MEMBERS_QUERY_ID", lambda c: xl.list_members_page("2075260793176592514", c))])
def test_an_explicit_list_query_id_override_stays_authoritative(monkeypatch, env, read):
    monkeypatch.setenv(env, "override-id")
    client, sent = _qid_client(monkeypatch)
    with pytest.raises(_Sent):
        read(client)
    assert sent == ["override-id"]
