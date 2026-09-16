"""
tests/gateway/test_trial_end_to_end.py

The local trial flow over REAL HTTP, end to end, with only the two external parties stubbed.

WHY THIS EXISTS. Every other test here is hermetic and mocks at a seam inside one module, so a
whole class of defect survives them all: a route registered at the wrong path, a redirect built
with the wrong query encoding, an env var the client reads and the gateway never sets, a callback
URL the far end cannot parse. Those only appear when the pieces are wired together and bytes move
between them over a socket. They are also exactly what a first live deploy surfaces, at the worst
possible time — in front of the first real user.

WHAT IS REAL HERE: the gateway app on a real port, its routes, the redirect chain, PKCE across
both halves, the loopback `Capture`, the ledger on disk, the marker, `keys.set_key`, and the
`readiness` state that comes out the other side.

WHAT IS STUBBED, and nothing else: Google's token endpoint and OpenRouter's key endpoint. Those
are the two parties we cannot stand up, and they are stubbed at the HTTP client rather than at our
own code, so our request to them is built for real and asserted on.

WHAT THIS STILL DOES NOT PROVE, and only a live run can: that Google accepts the redirect URI
registered for this client, and that OpenRouter accepts the management key and honours `limit` /
`expires_at` as documented. See docs/plans/2026-09-11-starter-allowance.md.
"""
from __future__ import annotations

import base64
import json
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import uvicorn
from fastmcp.server.auth.providers.google import GoogleProvider

from gateway import trial
from gateway.app import build_app
from opyt_core import keys, local_auth, readiness
from opyt_core import trial as client_trial

MINTED_KEY = "sk-or-v1-minted-for-this-test"

# Every test here talks to a gateway IT STARTED on 127.0.0.1. That is what the `loopback` marker
# is for, and it is deliberately NOT `live_llm`: nothing here is real or paid, and hiding these
# from a `-m "not live_llm"` run would hide the only integration coverage the trial flow has.
pytestmark = pytest.mark.loopback


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _id_token(sub: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub}).encode()).rstrip(b"=").decode()
    return f"header.{payload}.signature"


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    """A real gateway on a real loopback port, with Google and OpenRouter stubbed at the wire."""
    async def no_token(self, token):
        return None
    monkeypatch.setattr(GoogleProvider, "verify_token", no_token)
    monkeypatch.setenv(trial.MANAGEMENT_KEY_ENV, "sk-or-v1-management")
    monkeypatch.setenv("OPYT_TRIAL_LIMIT_USD", "0.25")

    sent: dict = {}
    real_post = httpx.AsyncClient.post

    async def fake_post(self, url, **kwargs):
        if str(url) == trial.GOOGLE_TOKEN:
            sent["google"] = kwargs.get("data")
            return httpx.Response(200, json={"id_token": _id_token("google-sub-42")})
        if str(url) == trial.KEYS_ENDPOINT:
            sent["mint"] = kwargs.get("json")
            sent["mint_auth"] = kwargs.get("headers", {}).get("Authorization")
            return httpx.Response(200, json={"key": MINTED_KEY, "data": {"hash": "hash-42"}})
        return await real_post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    port = _free_port()
    app = build_app(base_url=f"http://127.0.0.1:{port}", client_id="cid", client_secret="secret",
                    homes_root=tmp_path / "homes", idle_seconds=9999)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "the gateway never came up"
    try:
        yield f"http://127.0.0.1:{port}", sent
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture()
def install(tmp_path, monkeypatch, gateway):
    """A local OPYT install pointed at that gateway, with its own empty home."""
    base, sent = gateway
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv(client_trial.GATEWAY_ENV, base)
    monkeypatch.delenv(client_trial.HOSTED_TRIAL_URL_ENV, raising=False)
    return base, sent


def _drive_browser(url: str) -> None:
    """Stand in for the user's browser: follow the chain, substituting Google for a click.

    Deliberately NOT `follow_redirects=True` over the whole chain. Each hop is asserted, because
    the hops are the thing under test — and the Google hop has to be replaced by hand, since the
    one party we cannot stand up sits in the middle of it.
    """
    with httpx.Client(timeout=10) as browser:
        start = browser.get(url)
        assert start.status_code == 302, start.text
        to_google = urlparse(start.headers["location"])
        assert to_google.netloc == "accounts.google.com"
        params = parse_qs(to_google.query)
        # The redirect URI Google would have to accept. A live run needs exactly this string
        # registered for this client, and this assertion is what pins its shape.
        assert params["redirect_uri"][0].endswith("/trial/google")
        assert params["scope"] == ["openid"]

        # The user picks an account; Google bounces back to us with a code.
        back = browser.get(params["redirect_uri"][0],
                           params={"state": params["state"][0], "code": "google-auth-code"})
        assert back.status_code == 302, back.text
        loopback = back.headers["location"]
        assert urlparse(loopback).hostname in ("localhost", "127.0.0.1")

        # The final hop is the one that hands the code to the waiting loopback listener.
        assert browser.get(loopback).status_code == 200


def test_a_local_install_claims_an_allowance_over_real_http(install, monkeypatch):
    base, sent = install
    monkeypatch.setattr(local_auth, "open_browser",
                        lambda url: (_drive_browser(url), True)[1])

    out = client_trial.acquire(timeout=20)

    assert out["status"] == "stored"
    assert MINTED_KEY not in json.dumps(out), "a credential reached the caller"

    # The key landed where every other part of OPYT looks for it, by the same writer.
    assert MINTED_KEY in keys.env_path().read_text()
    # ...and it is marked as OPYT's, which is what makes its end tell the right story.
    assert client_trial.is_trial() is True
    assert client_trial.read_marker()["hash"] == "hash-42"

    # The mint we actually sent OpenRouter — asserted on the wire, not on our own arguments.
    assert sent["mint"]["limit"] == 0.25
    assert sent["mint"]["expires_at"]
    assert "limit_reset" not in sent["mint"], "a resetting cap is a permanent free tier"
    assert "creator_user_id" not in sent["mint"]      # rejected on a personal account
    assert sent["mint"]["name"].endswith("google-sub-42")
    assert sent["mint_auth"] == "Bearer sk-or-v1-management"
    # And the subject came from Google, never from anything the client could choose.
    assert sent["google"]["grant_type"] == "authorization_code"


def test_the_same_person_cannot_claim_twice(install, monkeypatch):
    """The ledger across two full flows, on disk, through the real routes — the property that
    stops a reinstall being the way to get another allowance."""
    base, _ = install
    monkeypatch.setattr(local_auth, "open_browser",
                        lambda url: (_drive_browser(url), True)[1])
    assert client_trial.acquire(timeout=20)["status"] == "stored"

    # A second install on the same machine, same person: a fresh home with no marker.
    client_trial.clear()
    assert client_trial.available() is True          # this home has no record of its own

    out = client_trial.acquire(timeout=20)

    assert out["status"] == "unavailable"
    assert "already_claimed" in out["reason"]
    assert client_trial.is_trial() is False, "a refused claim must leave no marker"


def test_the_spent_key_ends_where_the_oauth_flow_begins(install, monkeypatch):
    """The handoff the whole design exists for, exercised against a key that really was minted:
    OpenRouter starts answering 402, and `readiness` says `trial_over` rather than `unfunded`."""
    monkeypatch.setattr(local_auth, "open_browser",
                        lambda url: (_drive_browser(url), True)[1])
    assert client_trial.acquire(timeout=20)["status"] == "stored"

    monkeypatch.setattr(readiness, "_credential", lambda s: MINTED_KEY)
    monkeypatch.setattr(readiness, "_ping", lambda k: (False, "HTTP 402", 402))

    state = readiness.openrouter()

    assert state["state"] == "trial_over"
    assert readiness.TOPUP_URL not in state["message"]


def test_a_gateway_that_cannot_mint_is_not_advertised(install, monkeypatch):
    """Fail-safe over the wire: with the management key gone the gateway reports `trial: false`
    on /healthz, and the client falls through WITHOUT opening a browser tab at it."""
    monkeypatch.delenv(trial.MANAGEMENT_KEY_ENV, raising=False)
    opened = []
    monkeypatch.setattr(local_auth, "open_browser", lambda url: opened.append(url) or True)

    out = client_trial.acquire(timeout=5)

    assert out["status"] == "unavailable"
    assert opened == [], "a tab was opened at a gateway that had nothing to give"
