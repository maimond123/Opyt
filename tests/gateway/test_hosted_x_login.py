"""What is true only of X in the hosted browser boundary.

Its profile, desktop, relay and shared Chrome moved to `test_hosted_browser_login.py` on
2026-09-07 with the module they cover. What stays here is X's transport: the bearer, the
queryId discovery, and the viewer request that proves a profile is signed in.
"""
from __future__ import annotations

import json
import re

import pytest

from gateway import children
from pipeline.ingestion import hosted_browser, hosted_x


class _Runner:
    """A `ChromeRequestRunner` stand-in that records what X asked its page to run."""

    def __init__(self, value=None):
        self.value, self.scripts, self.urls = value, [], []

    def evaluate(self, home_url, expression):
        self.scripts.append(expression)
        return self.value

    def fetch_json(self, home_url, url, headers_js="{}"):
        self.urls.append((home_url, url, headers_js))
        return hosted_browser.ChromeRequestResult(hosted_browser.ChromeRequestStatus.OK, {})


def test_hosted_child_does_not_inherit_an_operator_x_bearer(monkeypatch, tmp_path):
    monkeypatch.setenv("X_WEB_BEARER", "operator-value")
    env = children._child_env(
        tmp_path / "hosted-home", "42", interaction_registration_url="http://gateway/register",
        interaction_url="https://gateway.example", interaction_key="child-key")
    assert "X_WEB_BEARER" not in env


def test_the_hosted_transport_never_reads_a_local_cookie(monkeypatch, tmp_path):
    from pipeline.ingestion import x_graphql_core as core

    monkeypatch.setenv("OPYT_HOSTED_X", "1")
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "hosted-home"))
    monkeypatch.setattr(core, "read_x_cookies", lambda: (_ for _ in ()).throw(
        AssertionError("hosted transport read a local cookie")))
    monkeypatch.setattr(hosted_x.HostedXSession, "graphql_get",
                        lambda self, *args, **kwargs: {"data": {"safe": True}})

    session = core.x_session("https://x.com/home")
    assert core.graphql_get("Bookmarks", "query", {}, {}, session) == {"data": {"safe": True}}


def test_x_requests_run_from_an_x_page_and_carry_the_csrf_read_in_it():
    """The csrf value lives in `document.cookie`, so it can only be computed inside the page —
    which is also why the header set is JavaScript source rather than a dict."""
    runner = _Runner()
    hosted_x.graphql(runner, "Bookmarks", "iblrFnKr6PZUR-dWpfXG6g", {}, {})

    home_url, url, headers_js = runner.urls[0]
    assert home_url == hosted_x._HOME_URL
    assert "iblrFnKr6PZUR-dWpfXG6g/Bookmarks" in url
    assert "document.cookie" in headers_js and "x-csrf-token" in headers_js


# ── login completion through GraphQL ───────────────────────────────────────────

def _viewer_payload(rest_id):
    user = {"rest_id": rest_id} if rest_id is not None else {}
    return hosted_browser.ChromeRequestResult(
        hosted_browser.ChromeRequestStatus.OK,
        {"data": {"viewer": {"user_results": {"result": user}}}})


@pytest.fixture
def viewer(monkeypatch):
    """Answer `validate` with one canned viewer payload, recording what it asked for."""
    asked: list[tuple[str, str]] = []

    def install(payload, *, query_id="qid123"):
        monkeypatch.setattr(hosted_x, "resolve_query_id",
                            lambda runner, op, page_url: query_id)

        def graphql(runner, op, qid, variables, features, **kwargs):
            asked.append((op, qid))
            return payload

        monkeypatch.setattr(hosted_x, "graphql", graphql)
        return asked

    return install


def test_validate_reports_the_viewer_id_from_the_graphql_response(viewer):
    asked = viewer(_viewer_payload("1861260702494957568"))
    result = hosted_x.validate(_Runner())

    assert result.status is hosted_browser.ChromeRequestStatus.OK
    assert result.data == {"id": "1861260702494957568"}
    assert asked == [(hosted_x._VIEWER_OP, "qid123")]


def test_validate_reports_no_id_when_the_response_carries_no_viewer(viewer):
    viewer(_viewer_payload(None))
    assert hosted_x.validate(_Runner()).data == {}


def test_validate_gives_up_when_the_operation_id_cannot_be_resolved(viewer):
    viewer(_viewer_payload("1"), query_id=None)
    assert hosted_x.validate(_Runner()).status is hosted_browser.ChromeRequestStatus.UNAVAILABLE


def test_validate_resolves_its_operation_id_without_retaking_the_profile_lock(
        monkeypatch, viewer):
    """`_profile_lock` is not reentrant, so a second acquire inside `shared_chrome()` blocks
    until it times out. `validate` takes the runner it is given rather than reaching for one."""
    class _LockedRunner:
        def start(self):
            self.started = True

        def close(self):
            pass

        def alive(self):
            return True

    viewer(_viewer_payload("77"))
    monkeypatch.setattr(hosted_browser, "ChromeRequestRunner", _LockedRunner)
    monkeypatch.setattr(hosted_browser, "_shared_chrome", hosted_browser._SharedChrome())
    monkeypatch.setattr(hosted_browser, "_profile_lock", hosted_browser.threading.Lock())
    monkeypatch.setattr(hosted_browser, "_PROFILE_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(hosted_browser.Path, "exists", lambda self: True)

    assert hosted_x.HostedXSession(hosted_x._HOME_URL).viewer_id() == "77"


# ── queryId discovery ──────────────────────────────────────────────────────────

def test_bundle_urls_are_matchable_by_the_discovery_pattern():
    runner = _Runner()
    hosted_x.discover_query_id(runner, "Viewer", hosted_x._HOME_URL)

    pattern = re.search(r"page\.matchAll\((/.+?/)g\)", runner.scripts[0]).group(1)[1:-1]
    real_url = ("https://abs.twimg.com/responsive-web/client-web/"
                "shared~bundle.LoggedInMain~ondemand.HoverCard.7f3a91c2.js")
    assert re.search(pattern, f'<script src="{real_url}"></script>')


def test_discovery_refuses_a_page_that_is_not_on_x():
    """The scan fetches whatever page it is pointed at, so the origin is a boundary check."""
    assert hosted_x.discover_query_id(_Runner(), "Viewer", "https://substack.com/inbox") is None


def test_a_base64url_query_id_survives_validation():
    """X ships base64url query ids, so roughly half of them contain a dash.

    Measured live 2026-09-07: `UserByScreenName` is `Gb-d6r0vxPOADdG62OEBpQ` and `Bookmarks`
    is `iblrFnKr6PZUR-dWpfXG6g`. Validating both against the operation-NAME alphabet discarded
    a correctly discovered id and made those operations unreachable.
    """
    found = _Runner({"qid": "Gb-d6r0vxPOADdG62OEBpQ"})
    assert hosted_x.discover_query_id(found, "Bookmarks", hosted_x._HOME_URL) \
        == "Gb-d6r0vxPOADdG62OEBpQ"

    sent = _Runner()
    hosted_x.graphql(sent, "Bookmarks", "iblrFnKr6PZUR-dWpfXG6g", {}, {})
    assert "iblrFnKr6PZUR-dWpfXG6g/Bookmarks" in sent.urls[0][1]


def test_a_query_id_that_could_escape_the_url_path_is_still_refused():
    """The validator is a URL-path boundary guard, so widening its alphabet must not open it."""
    for hostile in ("../../evil", "abc?x=1", "abc#frag", "abc/def"):
        assert hosted_x.graphql(_Runner(), "Bookmarks", hostile, {}, {}).status is (
            hosted_browser.ChromeRequestStatus.REJECTED)


def test_the_webpack_runtime_map_is_matchable_by_the_hosted_pattern():
    """Without this match `Bookmarks` has no discovery path at all on the hosted transport.

    X builds a route chunk's URL at call time from the `p.u` map, so the URL is never a literal
    in the page and the entry-bundle scan cannot reach it. Measured live 2026-09-07: the
    `Bookmarks` id is in none of the three bundles `x.com` links. The pattern is a JavaScript
    literal, so it is checked the way the bundle-URL pattern is — lifted out of the emitted
    script and run here, with JS named-group syntax translated to Python's.
    """
    runner = _Runner()
    hosted_x.discover_query_id(runner, "Bookmarks", hosted_x._HOME_URL)

    js = re.search(r"new RegExp\((\"\\\\\.u=.+?\")\);", runner.scripts[0]).group(1)
    pattern = re.sub(r"\\k<(\w+)>", r"(?P=\1)", json.loads(js).replace("(?<", "(?P<"))
    page = ('p.u=e=>""+(({10:"bundle.Bookmarks",20:"shared~bundle.Bookmarks"}'
            ')[e]||e)+"."+({10:"abc123",20:"def456"})[e]+"a.js"')

    found = re.search(pattern, page)

    assert found is not None
    assert dict(re.findall(r'(\d+):"([^"]+)"', found.group("names"))) == {
        "10": "bundle.Bookmarks", "20": "shared~bundle.Bookmarks"}
