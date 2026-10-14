"""One request per page, trustworthy end evidence, and complete foreground walks."""
import pytest

from pipeline.platforms.x import collections as x_collections
from pipeline.platforms.collection_page import Incomplete, Page, continuation, error_inside_entry
from tests.ingestion.page_walk import walk
from tests.ingestion.x_client_fake import FakeXClient
from tests.ingestion.test_x_likes import _likes_response
from tests.ingestion.test_x_lists import VIEWER, _members_response, _mgmt_response


CLIENT = FakeXClient(query_id="query-id")


def following_response(cursor, ids):
    timeline = _members_response(cursor, ids)["data"]["list"]["members_timeline"]["timeline"]
    return {"data": {"user": {"result": {"timeline": {"timeline": timeline}}}}}


def bookmark_response(cursor, ids):
    timeline = _likes_response(cursor, ids)["data"]["user"]["result"]["timeline"]["timeline"]
    return {"data": {"bookmark_timeline_v2": {"timeline": timeline}}}


@pytest.mark.parametrize("fetch,response,operation,extra", [
    (lambda cursor: x_collections.following_page(CLIENT, VIEWER, cursor),
     lambda: following_response("next", ["u1"]), "Following", {"userId": VIEWER}),
    (lambda cursor: x_collections.likes_page(VIEWER, CLIENT, cursor),
     lambda: _likes_response("next", ["t1"]), "Likes", {"userId": VIEWER}),
    (lambda cursor: x_collections.owned_lists_page(CLIENT, VIEWER, cursor),
     lambda: _mgmt_response("next", True), "ListsManagementPageTimeline", {}),
    (lambda cursor: x_collections.list_members_page("L1", CLIENT, cursor),
     lambda: _members_response("next", ["u1"]), "ListMembers", {"listId": "L1"}),
    (lambda cursor: x_collections.bookmark_page(CLIENT, cursor),
     lambda: bookmark_response("next", ["t1"]), "Bookmarks", {}),
])
def test_each_page_issues_one_operation_and_exposes_its_successor(
        monkeypatch, fetch, response, operation, extra):
    calls = []

    def get(op, _qid, variables, _features, **_kwargs):
        calls.append((op, variables))
        return response()

    monkeypatch.setattr(CLIENT, "get", get)
    result = fetch("resume")
    assert result.cursor == "next"
    assert not result.ended
    assert len(result.records) == 1
    assert len(calls) == 1
    assert calls[0][0] == operation
    assert calls[0][1]["cursor"] == "resume"
    assert all(calls[0][1][key] == value for key, value in extra.items())


@pytest.mark.parametrize("fetch", [
    lambda: x_collections.following_page(CLIENT, VIEWER),
    lambda: x_collections.likes_page(VIEWER, CLIENT),
    lambda: x_collections.owned_lists_page(CLIENT, VIEWER),
    lambda: x_collections.list_members_page("L1", CLIENT),
    lambda: x_collections.bookmark_page(CLIENT),
])
@pytest.mark.parametrize("response", [{}, {"data": {}}, {"data": None}])
def test_missing_timeline_never_looks_like_a_complete_empty_collection(monkeypatch, fetch, response):
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: response)
    with pytest.raises(Incomplete):
        fetch()


def test_graphql_partial_errors_cannot_establish_an_aggregate(monkeypatch):
    result = _likes_response("next", ["t1"])
    result["errors"] = [{"message": "partial response"}]
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: result)
    with pytest.raises(Incomplete, match="GraphQL errors"):
        x_collections.likes_page(VIEWER, CLIENT)


def test_following_page_skips_clear_cache_and_top_termination(monkeypatch):
    result = following_response("next", ["u1", "u2"])
    timeline = result["data"]["user"]["result"]["timeline"]["timeline"]
    timeline["instructions"][:0] = [{"type": "TimelineClearCache"},
                                    {"type": "TimelineTerminateTimeline", "direction": "Top"}]
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: result)
    page = x_collections.following_page(CLIENT, VIEWER)
    assert [key for key, _ in page.records] == ["u1", "u2"]
    assert page.cursor == "next" and not page.ended
    assert continuation(page, set(), set()) == "next"


LISTS_ROOT = ["viewer", "list_management_timeline", "timeline"]
BANNER = ["content", "items", 0, "item", "itemContent", "list",
          "default_banner_media_results", "result"]


def lists_page_with_errors(monkeypatch, *paths):
    result = _mgmt_response("next", True)
    timeline = result["data"]["viewer"]["list_management_timeline"]["timeline"]
    timeline["instructions"][:0] = [{"type": "TimelineClearCache"},
                                    {"type": "TimelineTerminateTimeline", "direction": "Top"}]
    result["errors"] = [{"message": "BadRequest: DecodeException", "code": 214,
                         **({} if path is None else {"path": path})} for path in paths]
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: result)
    return x_collections.owned_lists_page(CLIENT, VIEWER)


def test_lists_page_accepts_errors_inside_an_entry(monkeypatch):
    page = lists_page_with_errors(monkeypatch, [*LISTS_ROOT, "instructions", 2, "entries", 0, *BANNER])
    assert len(page.records) == 1
    assert page.cursor == "next"


@pytest.mark.parametrize("path", [
    None,
    [*LISTS_ROOT, "instructions", 2, "entries", 0],
    [*LISTS_ROOT, "instructions", 2, "entries"],
    [*LISTS_ROOT, "instructions"],
    ["viewer", "list_management_timeline"],
])
def test_lists_page_refuses_an_error_that_could_hide_an_entry(monkeypatch, path):
    inside = [*LISTS_ROOT, "instructions", 2, "entries", 0, *BANNER]
    with pytest.raises(Incomplete, match="GraphQL errors"):
        lists_page_with_errors(monkeypatch, inside, path)


ROOT = ("data", "user", "result", "timeline", "timeline")
INSTRUCTIONS = ["user", "result", "timeline", "timeline", "instructions"]


@pytest.mark.parametrize("error,expected", [
    ({"path": [*INSTRUCTIONS, 0, "entries", 3, "content", "itemContent"]}, True),
    ({"path": [*INSTRUCTIONS, 0, "entries", 3]}, False),
    ({"path": [*INSTRUCTIONS, 0, "entries"]}, False),
    ({"path": [*INSTRUCTIONS, 0, "entry", "content", "itemContent"]}, False),
    ({"path": [*INSTRUCTIONS, "0", "entries", 3, "content"]}, False),
    ({"path": ["data", *INSTRUCTIONS, 0, "entries", 3, "content"]}, False),
    ({"path": ["user", "result", "legacy", "name"]}, False),
    ({"message": "no path"}, False),
    ({"path": None}, False),
    ("not an error object", False),
])
def test_error_inside_entry_table(error, expected):
    assert error_inside_entry(error, ROOT) is expected


@pytest.mark.parametrize("instructions", [
    None,
    [None],
    [{"type": "TimelineAddEntries"}],
    [{"type": "UnknownInstruction"}],
    [{"type": "TimelineTerminateTimeline", "direction": "Top"}],
    [{"type": "TimelineAddEntries", "entries": [{"entryId": "tweet-1"}]}],
    [{"type": "TimelineAddEntries", "entries": [{"entryId": "cursor-bottom", "content": {
        "cursorType": "Bottom", "value": 123}}]}],
])
def test_malformed_instructions_do_not_complete_a_bookmark_collection(monkeypatch, instructions):
    response = {"data": {"bookmark_timeline_v2": {"timeline": {"instructions": instructions}}}}
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: response)
    with pytest.raises(Incomplete):
        x_collections.bookmark_page(CLIENT)


def test_unavailable_user_still_advances_pagination(monkeypatch):
    result = following_response("next", ["gone"])
    timeline = result["data"]["user"]["result"]["timeline"]["timeline"]
    timeline["instructions"][0]["entries"][0]["content"]["itemContent"]["user_results"]["result"] = {
        "__typename": "UserUnavailable"}
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: result)
    page = x_collections.following_page(CLIENT, VIEWER)
    assert page.records == ()
    assert page.observed == ("user-gone",)
    assert continuation(page, set(), set()) == "next"


@pytest.mark.parametrize("reader,response,root", [
    (lambda: x_collections.bookmark_page(CLIENT), bookmark_response,
     lambda data: data["data"]["bookmark_timeline_v2"]["timeline"]),
    (lambda: x_collections.likes_page(VIEWER, CLIENT), _likes_response,
     lambda data: data["data"]["user"]["result"]["timeline"]["timeline"]),
])
def test_unreadable_tweet_and_non_post_rows_are_not_an_empty_page(monkeypatch, reader, response, root):
    result = response("next", ["gone"])
    rows = root(result)["instructions"][0]["entries"]
    rows[0]["content"]["itemContent"]["tweet_results"]["result"] = {"__typename": "TweetTombstone"}
    rows.insert(0, {"entryId": "message-1", "content": {"__typename": "TimelineMessage"}})
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: result)
    page = reader()
    assert page.records == ()
    assert page.observed == ("message-1", "tweet-gone")
    assert not page.repeat_is_end
    assert continuation(page, set(), set()) == "next"


def test_owned_lists_use_item_identity_even_when_module_id_repeats(monkeypatch):
    calls = []

    def get(*_args, **_kwargs):
        number = len(calls)
        calls.append(number)
        response = _mgmt_response(f"c{number}", number < 2)
        if number < 2:
            module = response["data"]["viewer"]["list_management_timeline"]["timeline"]["instructions"][0]["entries"][0]
            module["content"]["items"][0]["item"]["itemContent"]["list"]["id_str"] = str(number)
        return response

    monkeypatch.setattr(CLIENT, "get", get)
    lists = walk(lambda cursor: x_collections.owned_lists_page(CLIENT, VIEWER, cursor))
    assert [item["id"] for item in lists] == ["0", "1"]
    assert len(calls) == 3


def test_subscribed_lists_are_observed_before_filtering_and_suggestions_end_the_walk(monkeypatch):
    response = _mgmt_response("c1", True)
    module = response["data"]["viewer"]["list_management_timeline"]["timeline"]["instructions"][0]["entries"][0]
    module["content"]["items"][0]["item"]["itemContent"]["list"]["user_results"]["result"]["rest_id"] = "other"
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: response)
    page = x_collections.owned_lists_page(CLIENT, VIEWER)
    assert not page.records
    assert page.observed
    assert continuation(page, set(), set()) == "c1"
    module["content"]["clientEventInfo"]["details"]["timelinesDetails"]["injectionType"] = "ListToFollow"
    assert x_collections.owned_lists_page(CLIENT, VIEWER).ended


@pytest.mark.parametrize("reader,response,root", [
    (lambda: x_collections.following_page(CLIENT, VIEWER), following_response,
     lambda data: data["data"]["user"]["result"]["timeline"]["timeline"]),
    (lambda: x_collections.likes_page(VIEWER, CLIENT), _likes_response,
     lambda data: data["data"]["user"]["result"]["timeline"]["timeline"]),
    (lambda: x_collections.bookmark_page(CLIENT), bookmark_response,
     lambda data: data["data"]["bookmark_timeline_v2"]["timeline"]),
])
def test_replacement_entries_are_normalized_as_well_as_observed(monkeypatch, reader, response, root):
    result = response("next", ["1"])
    timeline = root(result)
    entry = timeline["instructions"][0]["entries"].pop(0)
    timeline["instructions"].append({"type": "TimelineReplaceEntry", "entry": entry})
    monkeypatch.setattr(CLIENT, "get", lambda *_a, **_k: result)
    page = reader()
    assert len(page.records) == len(page.observed) == 1
    assert continuation(page, set(), set()) == "next"


@pytest.mark.parametrize("page,seen,cursors,expected", [
    (Page((), "end", (), ended=True), set(), set(), None),
    (Page((), None, ()), set(), set(), None),
    (Page((), "next", ("filtered",)), set(), set(), "next"),
    (Page((), "next", ("old",), repeat_is_end=True), {"old"}, {"prior"}, None),
    (Page((), "next", ("old",)), {"old"}, set(), Incomplete),
    (Page((), "prior", ("new",)), set(), {"prior"}, Incomplete),
    (Page((), "prior", ("old",), repeat_is_end=True), {"old"}, {"prior"}, Incomplete),
])
def test_continuation_table(page, seen, cursors, expected):
    if expected is Incomplete:
        with pytest.raises(Incomplete):
            continuation(page, seen, cursors)
    else:
        assert continuation(page, seen, cursors) == expected
