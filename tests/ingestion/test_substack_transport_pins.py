"""Stubbed HTTP pins for Substack account and public reads."""

from __future__ import annotations

import time
import sys
import types

import pytest

from pipeline.platforms.substack import contracts as substack_contracts
from pipeline.platforms.substack import http as substack_http
from pipeline.platforms.substack import local as substack_local
from pipeline.platforms.substack import publication as substack_publication


class Response:
    def __init__(self, *, status=200, text="ok", body=None):
        self.status_code = status
        self.text = text
        self.body = body
        self.calls = 0

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        self.calls += 1
        return self.body


def install_curl_get(monkeypatch, get):
    curl = types.ModuleType("curl_cffi")
    requests = types.ModuleType("curl_cffi.requests")
    requests.get = get
    curl.requests = requests
    monkeypatch.setitem(sys.modules, "curl_cffi", curl)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", requests)
    monkeypatch.setattr(substack_http, "host_request", lambda fn, *args, **kwargs: fn(*args, **kwargs))


def test_authenticated_get_uses_the_managed_cookie_and_json_response(monkeypatch):
    response = Response(body={"items": [1]})
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    install_curl_get(monkeypatch, get)
    assert substack_http.authed_get_json("https://substack.com/list", {"cursor": "c"},
                                {"substack.sid": "session"}) == {"items": [1]}

    url, kwargs = calls[0]
    assert url == "https://substack.com/list"
    assert kwargs["params"] == {"cursor": "c"}
    assert kwargs["headers"]["Cookie"] == "substack.sid=session"
    assert kwargs["impersonate"] == "chrome120"
    assert response.calls == 1


def test_authenticated_retry_sleeps_between_refusals_and_keeps_the_referer(monkeypatch):
    responses = [Response(status=403, text="blocked"), Response(body={"ok": True})]
    calls, sleeps = [], []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return responses.pop(0)

    install_curl_get(monkeypatch, get)
    monkeypatch.setattr(time, "sleep", sleeps.append)

    result = substack_http.authed_get_json_retry(
        "https://pub.substack.com/api/v1/posts/a", {}, {"substack.sid": "s"},
        referer="https://pub.substack.com/", retries=3, backoff=0.25)

    assert result == {"ok": True}
    assert [url for url, _ in calls] == [
        "https://pub.substack.com/api/v1/posts/a",
        "https://pub.substack.com/api/v1/posts/a",
    ]
    assert calls[0][1]["headers"]["Referer"] == "https://pub.substack.com/"
    assert calls[0][1]["headers"]["Cookie"] == "substack.sid=s"
    assert sleeps == [0.25]


def test_public_get_sends_no_session_cookie_and_keeps_an_optional_referer(monkeypatch):
    response = Response(body={"public": True})
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    install_curl_get(monkeypatch, get)
    assert substack_http.public_get_json("https://substack.com/public", {"limit": 1},
                                referer="https://substack.com/") == {"public": True}

    url, kwargs = calls[0]
    assert url == "https://substack.com/public"
    assert kwargs["params"] == {"limit": 1}
    assert kwargs["headers"]["Referer"] == "https://substack.com/"
    assert "Cookie" not in kwargs["headers"]


def test_public_get_refusal_keeps_its_public_read_error(monkeypatch):
    install_curl_get(monkeypatch, lambda *_args, **_kwargs: Response(status=403, text="blocked"))

    with pytest.raises(substack_contracts.SubstackPublicReadError, match="HTTP 403"):
        substack_http.public_get_json("https://substack.com/public", {})


@pytest.mark.parametrize("html, expected", [
    ('<script>_preloads = JSON.parse("{\\"user\\":{\\"id\\":42}}")</script>', 42),
    ('<script>_preloads = JSON.parse("{\\"user\\":{}}")</script>', None),
])
def test_own_user_id_reads_only_the_embedded_account_id(monkeypatch, html, expected):
    response = Response(text=html)
    install_curl_get(monkeypatch, lambda *_args, **_kwargs: response)

    assert substack_local.own_user_id({"substack.sid": "s"}) == expected


@pytest.mark.parametrize("html", [
    "<html><title>Error - Substack</title></html>",                      # no preload marker
    "<script>window._preloads = {}</script>",                             # marker, no JSON.parse
    '<script>_preloads = JSON.parse("[1]")</script>',                     # root not an object
    '<script>_preloads = JSON.parse("{\\"user\\":\\"x\\"}")</script>',       # user not an object
    '<script>_preloads = JSON.parse("{not json")</script>',              # undecodable
])
def test_own_user_id_refuses_a_page_in_an_unknown_shape(monkeypatch, html):
    """B13: a page without the supported preloads read as signed out (or leaked AttributeError).
    It says nothing about the sign-in, so it is unavailable."""
    install_curl_get(monkeypatch, lambda *_args, **_kwargs: Response(text=html))
    with pytest.raises(substack_contracts.SubstackListingError):
        substack_local.own_user_id({"substack.sid": "s"})


def test_own_user_id_reads_a_null_user_as_signed_out(monkeypatch):
    html = '<script>_preloads = JSON.parse("{\\"user\\":null}")</script>'
    install_curl_get(monkeypatch, lambda *_args, **_kwargs: Response(text=html))
    assert substack_local.own_user_id({"substack.sid": "s"}) is None


def test_fetch_follows_uses_the_supplied_account_id_and_shared_parser(monkeypatch):
    calls = []
    payload = {"subscriberLists": [{"groups": [{"users": [
        {"primary_publication": {"name": "Pub", "subdomain": "pub"}},
    ]}]}]}
    monkeypatch.setattr(substack_http, "authed_get_json",
                        lambda url, params, cookies: calls.append((url, params, cookies)) or payload)

    assert substack_local.fetch_follows({"substack.sid": "s"}, user_id=42) == [
        {"name": "Pub", "url": "https://pub.substack.com"}]
    assert calls == [(
        "https://substack.com/api/v1/user/42/subscriber-lists",
        {"lists": "following"}, {"substack.sid": "s"})]


def test_local_subscription_and_saved_pages_read_cookies_lazily_once(monkeypatch):
    cookies = {"substack.sid": "s"}
    reads, calls = [], []
    monkeypatch.setattr(substack_local, "read_substack_cookies",
                        lambda: reads.append("read") or cookies)
    monkeypatch.setattr(substack_http, "authed_get_json_retry",
                        lambda url, params, seen_cookies, **kwargs:
                        calls.append((url, params, seen_cookies, kwargs)) or {"items": []})

    subscriptions = substack_local.LocalSubscriptionList()
    assert subscriptions.page("cursor-a") == {"items": []}
    assert subscriptions.page(None) == {"items": []}
    saved = substack_local.LocalSaved()
    assert saved.page("cursor-b") == {"items": []}

    assert reads == ["read", "read"]                  # once per source, on first use
    assert calls == [
        (substack_local._SUBS_ENDPOINT, {"cursor": "cursor-a"}, cookies,
         {"referer": "https://substack.com/inbox"}),
        (substack_local._SUBS_ENDPOINT, {}, cookies, {"referer": "https://substack.com/inbox"}),
        (substack_local._SAVED_ENDPOINT, {"filter": "all", "cursor": "cursor-b"}, cookies,
         {"referer": "https://substack.com/saved"}),
    ]


def test_fetch_full_post_uses_the_publication_origin_and_session(monkeypatch):
    calls = []
    monkeypatch.setattr(substack_http, "authed_get_json_retry",
                        lambda url, params, cookies, **kwargs:
                        calls.append((url, params, cookies, kwargs)) or {"body_html": "full"})

    # B14: was a custom host, which got the session too; that case is now the refusal below.
    assert substack_publication.fetch_full_post("https://pub.substack.com/", "post-slug",
                                                {"substack.sid": "s"}) == {"body_html": "full"}
    assert calls == [(
        "https://pub.substack.com/api/v1/posts/post-slug", {}, {"substack.sid": "s"},
        {"referer": "https://pub.substack.com/"})]


def test_publication_id_and_recommendations_use_public_reads(monkeypatch):
    calls = []
    responses = [[{"publication_id": 23}], [{
        "recommendedPublication": {
            "name": "Recommended", "subdomain": "recommended",
            "author": {"bio": "A short bio"},
        },
    }]]

    def public(url, params, *, referer=None):
        calls.append((url, params, referer))
        return responses.pop(0)

    monkeypatch.setattr(substack_http, "public_get_json", public)
    assert substack_publication.fetch_publication_id("https://pub.substack.com/") == 23
    assert substack_publication.fetch_recommendations(23) == [{
        "url": "https://recommended.substack.com", "name": "Recommended",
        "bio": "A short bio", "payments_state": "",
    }]
    assert calls == [
        ("https://pub.substack.com/api/v1/archive", {"sort": "new", "limit": 1}, None),
        (f"{substack_publication._RECOMMENDATIONS_ENDPOINT}/23", {"limit": 99}, "https://substack.com/"),
    ]


@pytest.mark.parametrize("body, expected", [
    ([], []),
    ([{"recommendedPublication": {"name": "R", "subdomain": "r"}}],
     [{"url": "https://r.substack.com", "name": "R", "bio": "", "payments_state": ""}]),
])
def test_recommendations_list_bodies_normalize(monkeypatch, body, expected):
    monkeypatch.setattr(substack_http, "public_get_json", lambda *_a, **_k: body)
    assert substack_publication.fetch_recommendations(23) == expected


@pytest.mark.parametrize("body", [{"error": "Invalid value"}, None, "oops"])
def test_a_recommendations_body_that_is_not_a_list_refuses(monkeypatch, body):
    """B12: a malformed HTTP-200 body parsed to a successful empty list."""
    monkeypatch.setattr(substack_http, "public_get_json", lambda *_a, **_k: body)
    with pytest.raises(substack_contracts.SubstackPublicReadError):
        substack_publication.fetch_recommendations(23)


_HOSTS = [("https://substack.com", True), ("https://pub.substack.com/", True),
          ("https://www.custom-pub.example", False), ("https://substack.com.evil.test", False),
          ("https://evilsubstack.com", False)]


@pytest.mark.parametrize("base, honored", _HOSTS)
def test_a_full_post_carries_the_session_only_to_a_host_that_honors_it(monkeypatch, base, honored):
    """B14: the session cookie went to custom and lookalike hosts too."""
    sent = []
    monkeypatch.setattr(substack_http, "authed_get_json_retry",
                        lambda url, params, cookies, **kw: sent.append(cookies) or {"ok": 1})
    assert substack_publication.fetch_full_post(base, "a-post", {"substack.sid": "s"}) == {"ok": 1}
    assert sent == [{"substack.sid": "s"} if honored else {}]


@pytest.mark.parametrize("base, honored", _HOSTS)
def test_an_archive_walk_carries_the_session_only_to_a_host_that_honors_it(monkeypatch, base,
                                                                          honored):
    """B14, the archive list."""
    headers_seen = []

    class _Session:
        def get(self, url, *, params, headers, timeout):
            headers_seen.append(headers)
            return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: [])

        def close(self):
            pass

    monkeypatch.setattr(substack_publication.requests, "Session", _Session)
    assert substack_publication.fetch_all_posts(base, cookies={"substack.sid": "s"}) == []
    assert ("Cookie" in headers_seen[0]) is honored
