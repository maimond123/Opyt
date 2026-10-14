"""The connector's sign-in, held to what Claude's connector requirements ask of it: the consent page
reads as Opyt; revoking a token ends the sign-in it came from (both of the client's tokens are
refused afterwards, and Google is asked to revoke the grant); a refresh rotates the refresh token,
and a refresh token that is used up, expired or revoked gets `invalid_grant`; and Claude Code,
which names itself by its Client ID Metadata Document and listens on a loopback port of its own,
signs in on that port.

The connector's real app and sign-in run here, over HTTP: client registration, `/authorize`, the
consent page, Google's return, the token endpoint, `/revoke` and `/mcp`. Google is the stand-in,
at its calls: the code exchange on its return and the refresh (FastMCP's upstream client), the
token check (`GoogleTokenVerifier.verify_token`) and the revocation
(`google_signin.revoke_at_google`). It revokes nothing unless a test says so, so a refusal here is
this service's own. claude.ai is the stand-in for Claude Code's metadata document. Most tests skip
the Google sign-in page by writing the code FastMCP stores on Google's return. Token requests and
revocations are sent as a public client sends them: form-encoded, `client_id`, no secret.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import fastmcp
import pytest
from authlib.integrations.httpx_client import OAuthError
from fastmcp.server.auth import cimd
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.oauth_proxy import proxy
from fastmcp.server.auth.oauth_proxy.models import ClientCode
from fastmcp.server.auth.providers.google import GoogleTokenVerifier
from fastmcp.server.auth.ssrf import SSRFFetchResponse
from mcp.server.auth.handlers import token as token_handler
from starlette.testclient import TestClient

from service import connector, db, google_signin, line

SUB = "google-123"
SCOPES = ["openid"]
GOOGLE_ACCESS, GOOGLE_REFRESH = "google-access", "google-refresh"
GOOGLE_ANSWER = {"access_token": GOOGLE_ACCESS, "refresh_token": GOOGLE_REFRESH,
                 "expires_in": 3600, "token_type": "Bearer", "scope": " ".join(SCOPES)}
REDIRECT = "http://localhost:7777/callback"
VERIFIER = "v" * 64
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).decode().rstrip("=")

# Claude Code's Client ID Metadata Document, as claude.ai serves it. Claude Code
# redirects to one of these with the port it listens on added.
CLAUDE_CODE = "https://claude.ai/oauth/claude-code-client-metadata"
CLAUDE_CODE_DOCUMENT = {
    "client_id": CLAUDE_CODE, "client_name": "Claude Code", "client_uri": "https://claude.ai",
    "redirect_uris": ["http://localhost/callback", "http://127.0.0.1/callback"],
    "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
    "token_endpoint_auth_method": "none"}
LOOPBACKS = ["http://localhost:54321/callback", "http://127.0.0.1:54321/callback"]


class Google:
    """Google after sign-in: what it was asked to revoke, and whether the user still allows
    Opyt (`False` once they remove it in their Google account)."""

    def __init__(self):
        self.revoked: list[str] = []
        self.allowed = True

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != GOOGLE_ACCESS:
            return None
        return AccessToken(token=token, client_id=SUB, scopes=SCOPES,
                           expires_at=int(time.time()) + 3600, claims={"sub": SUB})

    async def refresh(self, refresh_token: str) -> dict:
        assert refresh_token == GOOGLE_REFRESH
        if not self.allowed:
            raise OAuthError(error="invalid_grant", description="Token has been expired or revoked.")
        return GOOGLE_ANSWER

    async def revoke(self, token: str) -> None:
        self.revoked.append(token)


@pytest.fixture()
def google(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "service"))
    g = Google()

    async def verify_token(_verifier, token):
        return await g.verify_token(token)

    async def refresh_token(_client, url, refresh_token=None, **_params):
        return await g.refresh(refresh_token)

    async def fetch_token(_client, url=None, **_params):
        return GOOGLE_ANSWER
    monkeypatch.setattr(GoogleTokenVerifier, "verify_token", verify_token)
    monkeypatch.setattr(proxy.AsyncOAuth2Client, "refresh_token", refresh_token)
    monkeypatch.setattr(proxy.AsyncOAuth2Client, "fetch_token", fetch_token)
    monkeypatch.setattr(google_signin, "revoke_at_google", g.revoke)
    return g


@pytest.fixture()
def http(google, monkeypatch, tmp_path):
    """The connector as the service builds it, its sign-in state under a temporary home."""
    monkeypatch.setenv(connector.BASE_URL_ENV, "http://localhost")
    monkeypatch.setenv(connector.CLIENT_ID_ENV, "web-client.apps.googleusercontent.com")
    monkeypatch.setenv(connector.CLIENT_SECRET_ENV, "web-secret")
    monkeypatch.setattr(fastmcp.settings, "home", tmp_path / "fastmcp")
    app = connector.asgi_app()
    with TestClient(app, base_url="http://localhost") as client:
        client.provider = app.state.fastmcp_server.auth
        yield client


@pytest.fixture()
def claude_code(monkeypatch):
    """claude.ai serving Claude Code's metadata document, the one fetch FastMCP makes for it."""
    async def fetch(url, **_kwargs):
        assert url == CLAUDE_CODE
        return SSRFFetchResponse(content=json.dumps(CLAUDE_CODE_DOCUMENT).encode(),
                                 status_code=200, headers={})
    monkeypatch.setattr(cimd, "ssrf_safe_fetch_response", fetch)


def signed_in(http) -> dict:
    """A client registered here and holding tokens, as after a real Google sign-in."""
    client_id = http.post("/register", json={
        "redirect_uris": [REDIRECT], "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
    }).json()["client_id"]
    now = time.time()
    code = ClientCode(code="code-1", client_id=client_id, redirect_uri=REDIRECT,
                      code_challenge=CHALLENGE, code_challenge_method="S256", scopes=SCOPES,
                      idp_tokens=GOOGLE_ANSWER, expires_at=now + 300, created_at=now)
    http.portal.call(lambda: http.provider._code_store.put(key="code-1", value=code))
    tokens = http.post("/token", data={
        "grant_type": "authorization_code", "code": "code-1", "redirect_uri": REDIRECT,
        "client_id": client_id, "code_verifier": VERIFIER}).json()
    return {"client_id": client_id, **tokens}


def reads(http, access_token: str) -> int:
    """The status of one `status` call to the connector with this token."""
    return http.post(connector.MCP_PATH, headers={
        "Authorization": f"Bearer {access_token}", "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"},
        content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "status", "arguments": {}}})).status_code


def refresh(http, tokens: dict):
    return http.post("/token", data={
        "grant_type": "refresh_token", "client_id": tokens["client_id"],
        "refresh_token": tokens["refresh_token"]})


def authorize(http, client_id: str, redirect_uri: str):
    """An authorization request as Claude sends it, and FastMCP's answer, not followed."""
    return http.get("/authorize", follow_redirects=False, params={
        "response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri,
        "code_challenge": CHALLENGE, "code_challenge_method": "S256", "state": "s",
        "scope": " ".join(SCOPES), "resource": f"http://localhost{connector.MCP_PATH}"})


def consents(http, to_consent) -> str:
    """Allow on the consent page, sign in at Google and come back: where the browser is sent."""
    consent = to_consent.headers["location"]
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', http.get(consent).text)[1]
    to_google = http.post(consent, follow_redirects=False, data={
        "txn_id": parse_qs(urlparse(consent).query)["txn_id"][0], "csrf_token": csrf,
        "action": "approve"})
    state = parse_qs(urlparse(to_google.headers["location"]).query)["state"][0]
    return http.get("/auth/callback", follow_redirects=False,
                    params={"code": "google-code", "state": state}).headers["location"]


def test_the_server_advertises_revocation(http):
    meta = http.get("/.well-known/oauth-authorization-server").json()
    assert meta["revocation_endpoint"] == "http://localhost/revoke"


def test_a_live_sign_in_reads_and_refreshes(http):
    """What the revocation test below takes away."""
    tokens = signed_in(http)
    assert reads(http, tokens["access_token"]) == 200
    assert refresh(http, tokens).status_code == 200


@pytest.mark.parametrize("which", ["access_token", "refresh_token"])
def test_revoking_either_token_refuses_both(http, google, which):
    tokens = signed_in(http)
    assert reads(http, tokens["access_token"]) == 200

    got = http.post("/revoke", data={"token": tokens[which], "client_id": tokens["client_id"]})
    assert got.status_code == 200, got.text

    assert reads(http, tokens["access_token"]) == 401
    assert refresh(http, tokens).json()["error"] == "invalid_grant"
    assert google.revoked == [GOOGLE_REFRESH]


def test_another_client_cannot_revoke_this_one(http, google):
    tokens = signed_in(http)
    other = http.post("/register", json={
        "redirect_uris": [REDIRECT], "token_endpoint_auth_method": "none"}).json()["client_id"]

    http.post("/revoke", data={"token": tokens["access_token"], "client_id": other})
    assert reads(http, tokens["access_token"]) == 200
    assert google.revoked == []


def test_the_consent_page_before_google_reads_as_opyt(http):
    """FastMCP asks the user to allow a newly registered client before sending them to Google.
    Its title and help text are FastMCP's; the server it names is Opyt, linked to the site, under
    Opyt's icon."""
    client_id = http.post("/register", json={
        "redirect_uris": [REDIRECT], "token_endpoint_auth_method": "none"}).json()["client_id"]
    page = http.get(authorize(http, client_id, REDIRECT).headers["location"]).text
    assert f'href="{connector.SITE_URL}"' in page and ">Opyt</a>" in page
    assert connector.ICON_URL in page


def test_a_refresh_rotates_the_refresh_token(http):
    """Claude's clients are public, so the refresh token is single-use: the form-encoded refresh
    is answered with a new one, which refreshes in turn."""
    tokens = signed_in(http)
    got = refresh(http, tokens)
    assert got.status_code == 200, got.text
    assert got.request.headers["content-type"] == "application/x-www-form-urlencoded"
    rotated = {**tokens, **got.json()}
    assert rotated["refresh_token"] != tokens["refresh_token"]
    assert reads(http, rotated["access_token"]) == 200
    assert refresh(http, rotated).status_code == 200


@pytest.mark.parametrize("dead", ["used", "expired", "removed_at_google"])
def test_a_dead_refresh_token_gets_invalid_grant(http, google, monkeypatch, dead):
    """`invalid_grant` is what tells Claude to sign in again: for a refresh token already
    rotated away, one past its lifetime (two years on), and one whose Google grant the user
    removed. A revoked one is `test_revoking_either_token_refuses_both`."""
    tokens = signed_in(http)
    if dead == "used":
        assert refresh(http, tokens).status_code == 200
    elif dead == "expired":
        later = time.time() + 2 * 365 * 24 * 3600
        monkeypatch.setattr(token_handler, "time", SimpleNamespace(time=lambda: later))
    else:
        google.allowed = False
    got = refresh(http, tokens)
    assert got.json()["error"] == "invalid_grant", got.text


@pytest.mark.parametrize("redirect", LOOPBACKS)
def test_claude_code_signs_in_on_its_loopback_port(http, claude_code, redirect):
    """Claude Code's document declares its loopback callbacks without a port and it listens on
    one of its own: the declared URI matches with the port ignored, the consent page and Google's
    return keep the port, and the code it brings back there buys tokens that refresh."""
    to_consent = authorize(http, CLAUDE_CODE, redirect)
    assert to_consent.status_code == 302, to_consent.text
    assert urlparse(to_consent.headers["location"]).path == "/consent"

    back = urlparse(consents(http, to_consent))
    assert f"{back.scheme}://{back.netloc}{back.path}" == redirect
    got = http.post("/token", data={
        "grant_type": "authorization_code", "code": parse_qs(back.query)["code"][0],
        "redirect_uri": redirect, "client_id": CLAUDE_CODE, "code_verifier": VERIFIER})
    assert got.status_code == 200, got.text
    assert refresh(http, {"client_id": CLAUDE_CODE, **got.json()}).status_code == 200


def test_claude_code_is_refused_a_callback_its_document_does_not_declare(http, claude_code):
    assert authorize(http, CLAUDE_CODE, "http://localhost:54321/elsewhere").status_code == 400


@pytest.mark.parametrize("redirect", LOOPBACKS)
def test_a_client_registered_with_a_portless_loopback_signs_in_on_a_port(http, redirect):
    """The same for a client that registers (FastMCP runs with no redirect allow-list)."""
    client_id = http.post("/register", json={
        "redirect_uris": CLAUDE_CODE_DOCUMENT["redirect_uris"],
        "token_endpoint_auth_method": "none"}).json()["client_id"]
    to_consent = authorize(http, client_id, redirect)
    assert to_consent.status_code == 302, to_consent.text
    assert urlparse(to_consent.headers["location"]).path == "/consent"


def test_a_signed_in_call_is_told_once_what_ended_since_the_account_last_heard(http):
    """Over the real `/mcp` path, with a real token: the hook every tool's answer passes
    (`connector.Earlier`) finds the caller's library from the same verified sign-in the tools
    read, so the outcome of a change nobody waited for rides the account's next answer, once."""
    tokens = signed_in(http)
    conn = db.connect()
    conn.execute("INSERT INTO accounts (google_sub, owner) VALUES (?, 'a-library')", (SUB,))
    conn.commit()
    conn.close()
    asyncio.run(line.call("a-library", "hopper", {"reference": "https://example.com/a",
                                                  "confirm": True}))
    handed = asyncio.run(line.next_call("a-library", 0))["call"]
    assert line.answer("a-library", handed["id"], True, {"status": "saved"})

    def status() -> dict:
        got = http.post(connector.MCP_PATH, headers={
            "Authorization": f"Bearer {tokens['access_token']}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"},
            content=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                "params": {"name": "status", "arguments": {}}}))
        return got.json()["result"]["structuredContent"]

    first = status()
    assert first["status"] == "not_arrived", "this library has published nothing yet"
    assert first["earlier"] == ["https://example.com/a is saved in the library."]
    assert "earlier" not in status()
