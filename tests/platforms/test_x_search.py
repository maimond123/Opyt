"""X search (SearchTimeline): the POST beside the GET in the one X request path, and the page read
off a recorded response. Offline: the transport is a fake `curl_cffi`, the response a fixture
modeled on a live one."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import pipeline.platforms.x.query_ids as query_ids
import pipeline.platforms.x.rate as x_rate
from pipeline.platforms.collection_page import Incomplete
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x import search as x_search
from pipeline.platforms.x.contracts import GRAPHQL_ROOT
from pipeline.platforms.x.request import XClient

RECORDED = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "x"
                       / "search_timeline.json").read_text())


@pytest.fixture()
def metered(monkeypatch):
    """The rate meter, observed instead of written."""
    calls = []
    monkeypatch.setattr(x_rate, "refuse_if_spent", lambda op: calls.append(("budget", op)))
    monkeypatch.setattr(x_rate, "record_rate_values",
                        lambda op, remaining, reset: calls.append(("rate", op, remaining)))
    return calls


def _fake_cffi(monkeypatch, sent: list, body: dict):
    class _Resp:
        status_code, text = 200, "{}"
        headers = {"x-rate-limit-remaining": "186", "x-rate-limit-reset": "1791865630"}

        def raise_for_status(self):
            pass

        def json(self):
            return body

    cffi = types.ModuleType("curl_cffi")
    cffi.requests = types.SimpleNamespace(
        get=lambda url, **kw: sent.append(("GET", url, kw)) or _Resp(),
        post=lambda url, **kw: sent.append(("POST", url, kw)) or _Resp())
    monkeypatch.setitem(sys.modules, "curl_cffi", cffi)


def test_a_post_sends_the_web_clients_json_body_under_the_shared_policy(monkeypatch, metered):
    """SearchTimeline is a POST: its arguments go in a JSON body, unencoded, and the request is
    budgeted and metered like every GET."""
    sent = []
    _fake_cffi(monkeypatch, sent, RECORDED)
    client = XClient(x_local.LocalXTransport({}, {"Cookie": "managed"}))
    data = client.graphql_post("SearchTimeline", "qid", {"rawQuery": "q"}, {"f": True})
    [(method, url, kw)] = sent
    assert (method, url) == ("POST", f"{GRAPHQL_ROOT}/qid/SearchTimeline")
    assert kw["json"] == {"variables": {"rawQuery": "q"}, "features": {"f": True},
                          "queryId": "qid"}
    assert kw["headers"] == {"Cookie": "managed"} and "params" not in kw
    assert data == RECORDED
    assert metered == [("budget", "SearchTimeline"), ("rate", "SearchTimeline", 186)]


def test_a_get_still_sends_its_arguments_in_the_query_string(monkeypatch, metered):
    sent = []
    _fake_cffi(monkeypatch, sent, {"data": {}})
    XClient(x_local.LocalXTransport({}, {})).graphql_get("Bookmarks", "qid", {"a": 1}, {})
    [(method, _url, kw)] = sent
    assert method == "GET" and kw["params"]["variables"] == '{"a":1}'


def test_a_search_page_reads_its_posts_in_order_and_skips_the_rest():
    """Cursors and the relevance prompt carry no post; a post behind a visibility wrapper is
    still a post."""
    posts = x_search.parse_posts(RECORDED)
    assert [(p["id"], p["author"]["userName"]) for p in posts] == [
        ("2108556992570638761", "fixturelab"), ("2107926730329534743", "second")]
    assert posts[0]["url"] == "https://x.com/fixturelab/status/2108556992570638761"
    assert posts[0]["createdAt"] == "Fri Oct 09 13:55:36 +0000 2026"


def test_a_response_without_the_search_timeline_is_refused_not_read_as_a_quiet_day():
    with pytest.raises(Incomplete):
        x_search.parse_posts({"data": {}})


def test_search_asks_the_top_tab_for_one_page_of_the_raw_query(monkeypatch):
    asked = {}

    class _Client:
        def resolve_query_id(self, op, **kw):
            asked["resolve"] = (op, kw["env_var"])
            return "qid"

        def graphql_post(self, op, qid, variables, features, **kw):
            asked["post"] = (op, qid, variables)
            return RECORDED

    raw = '"Boltz-2" lang:en -filter:replies since:2026-10-06'
    assert len(x_search.search_posts(_Client(), raw)) == 2
    assert asked["resolve"] == ("SearchTimeline", "X_SEARCHTIMELINE_QUERY_ID")
    assert asked["post"] == ("SearchTimeline", "qid", {
        "rawQuery": raw, "count": 20, "querySource": "typed_query", "product": "Top"})


def test_search_is_available_only_signed_into_x(monkeypatch):
    monkeypatch.setattr(x_search.x_local, "has_session", lambda: True)
    assert x_search.available() is True
    monkeypatch.setattr(x_search.x_local, "has_session", lambda: False)
    assert x_search.available() is False


def test_the_query_id_is_discovered_never_baked():
    """A baked id survives X's rotation as a permanent 404 (x/profile.py says why)."""
    assert x_search.DEFAULT_SEARCH_QID == ""
    assert query_ids.query_id_choice(None, x_search.DEFAULT_SEARCH_QID, None) is None
