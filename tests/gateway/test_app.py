"""
tests/gateway/test_app.py — the edge: who gets in, and what reaches the child.

Google is never contacted. `verify_token` is monkeypatched on the provider class, which is the
one seam a test needs and the one the production code already exposes (`AuthProvider`'s public
API). Everything downstream of it — subject extraction, spawn, proxy, stream — is real, and the
round-trip test runs a real `opyt-mcp --http` child in a temporary home.
"""
from __future__ import annotations

import json

import pytest
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.providers.google import GoogleProvider
from starlette.testclient import TestClient

from gateway.app import build_app

BASE = "https://gw.example.com"

INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "test", "version": "1"}},
}

# token → the `sub` claim it carries. "no-sub" and anything unlisted are rejected upstream.
_TOKENS = {"good": "42", "traversal": "../etc", "no-sub": None}


@pytest.fixture
def app(tmp_path, monkeypatch):
    async def fake_verify(self, token: str) -> AccessToken | None:
        if token not in _TOKENS:
            return None
        sub = _TOKENS[token]
        return AccessToken(token=token, client_id="test-client", scopes=["openid", "email"],
                           expires_at=None, claims=({"sub": sub} if sub else {}))

    monkeypatch.setattr(GoogleProvider, "verify_token", fake_verify)
    return build_app(base_url=BASE, client_id="cid", client_secret="secret",
                     homes_root=tmp_path, idle_seconds=9999)


# ── The edge ────────────────────────────────────────────────────────────────────────────────

def test_no_token_is_401_and_says_where_to_authenticate(app):
    """A bare 401 is a dead end. RFC 9728's `resource_metadata` is how an MCP client discovers
    which authorization server to send the user to, so the header is the point of the reply."""
    with TestClient(app) as client:
        r = client.post("/mcp", json=INITIALIZE)

    assert r.status_code == 401
    assert f'resource_metadata="{BASE}/.well-known/oauth-protected-resource/mcp"' \
        in r.headers["www-authenticate"]


@pytest.mark.parametrize("header", ["", "Basic abc", "Bearer", "bearer "])
def test_malformed_authorization_headers_are_401(app, header):
    with TestClient(app) as client:
        r = client.post("/mcp", json=INITIALIZE, headers={"Authorization": header})
    assert r.status_code == 401


def test_a_rejected_token_is_401(app):
    with TestClient(app) as client:
        r = client.post("/mcp", json=INITIALIZE, headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_a_token_without_a_subject_is_401(app):
    """The subject is the only claim this design uses — it names the home. No subject, no
    routing, and nothing should be spawned."""
    with TestClient(app) as client:
        r = client.post("/mcp", json=INITIALIZE, headers={"Authorization": "Bearer no-sub"})
    assert r.status_code == 401


def test_a_subject_that_could_escape_the_homes_root_is_403(app, tmp_path):
    """Authenticated but unusable: the token is real, its subject cannot name a directory.
    That is a different answer from 401 because re-authenticating would not help."""
    with TestClient(app) as client:
        r = client.post("/mcp", json=INITIALIZE, headers={"Authorization": "Bearer traversal"})

    assert r.status_code == 403
    assert list(tmp_path.iterdir()) == [], "a rejected subject created a directory"


def test_the_oauth_routes_are_mounted(app):
    """The gateway is the authorization server toward claude.ai, so these paths are the
    contract: discovery, the two code-exchange endpoints, and dynamic client registration."""
    paths = {r.path for r in app.routes}

    assert {"/authorize", "/token", "/register", "/auth/callback",
            "/.well-known/oauth-authorization-server",
            "/.well-known/oauth-protected-resource/mcp", "/mcp"} <= paths


def test_the_vendored_novnc_client_is_served_by_the_gateway(app):
    with TestClient(app) as client:
        response = client.get("/static/novnc/core/rfb.js")
        module = client.get("/static/desktop.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "class RFB" in response.text
    # The pages import this one, and it imports the above. A sign-in page whose module 404s
    # shows a dead grey rectangle and no error, so serving it is part of serving noVNC.
    assert module.status_code == 200
    assert "javascript" in module.headers["content-type"]


def test_healthz_reports_an_empty_pool_before_any_request(app):
    with TestClient(app) as client:
        r = client.get("/healthz")

    # `trial` is part of the contract, not incidental: a local install reads it to find out
    # whether there is an allowance to claim BEFORE it opens a browser tab here. False is the
    # right answer for a gateway with no management key, which is every test gateway.
    assert r.status_code == 200
    assert r.json() == {"ok": True, "children": [], "trial": False}


# ── Through to a real child ─────────────────────────────────────────────────────────────────

def test_an_authenticated_call_reaches_the_users_own_child(app, tmp_path):
    """The whole path: verify, resolve `sub` to a home, spawn, proxy, stream the answer back.

    `initialize` is the right probe because it is what claude.ai sends first, and its reply
    carries the `mcp-session-id` the gateway must relay untouched for the session to continue.
    """
    with TestClient(app) as client:
        r = client.post("/mcp", json=INITIALIZE,
                        headers={"Authorization": "Bearer good",
                                 "Accept": "application/json, text/event-stream"})

        assert r.status_code == 200, r.text
        assert r.headers.get("mcp-session-id"), "the session id was not relayed"
        initialized = json.loads(r.text.split("data: ", 1)[1])["result"]
        assert initialized["serverInfo"]["name"] == "Opyt"
        assert initialized["serverInfo"]["icons"] == [{
            "src": "https://mcp.useopyt.com/icon.png",
            "mimeType": "image/png",
            "sizes": ["512x512"],
        }]

        # The home is named by the token's subject, and it is the only one that exists.
        assert (tmp_path / "42").is_dir()
        assert [p.name for p in tmp_path.iterdir()] == ["42"]

        health = client.get("/healthz").json()
        assert [c["subject"] for c in health["children"]] == ["42"]
        assert health["children"][0]["inflight"] == 0, "the stream never released the child"


def test_two_subjects_get_two_children_and_two_homes(app, tmp_path):
    """The isolation the whole design rests on: one `$OPYT_HOME` per child, so a child's
    module-level state cannot be another user's."""
    _TOKENS["second"] = "77"
    try:
        with TestClient(app) as client:
            for token in ("good", "second"):
                r = client.post("/mcp", json=INITIALIZE,
                                headers={"Authorization": f"Bearer {token}",
                                         "Accept": "application/json, text/event-stream"})
                assert r.status_code == 200, r.text

            subjects = {c["subject"] for c in client.get("/healthz").json()["children"]}
            ports = {c["port"] for c in client.get("/healthz").json()["children"]}

        assert subjects == {"42", "77"}
        assert len(ports) == 2
        assert {p.name for p in tmp_path.iterdir()} == {"42", "77"}
    finally:
        _TOKENS.pop("second", None)


def test_the_child_answers_tools_list_through_the_gateway(app):
    """Proves the proxy relays a real session, not just the handshake: `tools/list` only
    succeeds if the `mcp-session-id` from `initialize` reached the same child again."""
    with TestClient(app) as client:
        accept = "application/json, text/event-stream"
        init = client.post("/mcp", json=INITIALIZE,
                           headers={"Authorization": "Bearer good", "Accept": accept})
        session = init.headers["mcp-session-id"]

        headers = {"Authorization": "Bearer good", "Accept": accept,
                   "mcp-session-id": session, "MCP-Protocol-Version": "2025-06-18"}
        client.post("/mcp", headers=headers,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"})
        listed = client.post("/mcp", headers=headers,
                             json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    assert listed.status_code == 200, listed.text
    payload = json.loads(listed.text.split("data: ", 1)[1])
    names = {t["name"] for t in payload["result"]["tools"]}
    assert "search" in names, names
