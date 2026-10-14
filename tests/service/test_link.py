"""Tying a library to a Google account: `POST /v1/link` drives Google's device sign-in.

Google is the one stand-in, at `requests.post` in `service/google_device.py`. The route, the
owner-token check and both tables are the real ones.
"""
from __future__ import annotations

import base64
import json

import pytest
import requests
from fastapi.testclient import TestClient

from service import accounts, db, google_device, tokens
from service.app import app

CLIENT_ID = "device-client.apps.googleusercontent.com"


def id_token(claims: dict) -> str:
    """A JWT as Google's token endpoint returns it; the signature is not read."""
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


GOOGLE_ID = id_token({"sub": "google-123", "aud": CLIENT_ID, "iss": "https://accounts.google.com"})


class Google:
    """Google's two device endpoints. `answers` is what the token endpoint says, in order."""

    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.codes = 0

    def post(self, url, data=None, timeout=None):
        if url == google_device.DEVICE_CODE_URL:
            self.codes += 1
            return _Reply({"device_code": f"dev-{self.codes}", "user_code": f"CODE-{self.codes}",
                           "verification_url": "https://www.google.com/device",
                           "expires_in": 1800, "interval": 5})
        assert url == google_device.TOKEN_URL and data["client_secret"] == "device-secret"
        return _Reply(self.answers.pop(0))


class _Reply:
    def __init__(self, body: dict):
        self._body = body

    def json(self):
        return self._body


@pytest.fixture()
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "service"))
    monkeypatch.setenv(google_device.CLIENT_ID_ENV, CLIENT_ID)
    monkeypatch.setenv(google_device.CLIENT_SECRET_ENV, "device-secret")
    key, token = tokens.register_owner()
    return key, {"Authorization": f"Bearer {token}"}


def google(monkeypatch, *answers: dict) -> Google:
    g = Google(*answers)
    monkeypatch.setattr(google_device.requests, "post", g.post)
    return g


def test_a_first_call_shows_a_code_and_links_nothing(owner, monkeypatch):
    key, hdr = owner
    google(monkeypatch)
    got = TestClient(app).post("/v1/link", headers=hdr).json()
    assert got == {"status": "waiting", "user_code": "CODE-1",
                   "verification_url": "https://www.google.com/device"}
    assert accounts.owner_of("google-123") is None


def test_an_approved_sign_in_ties_the_account_to_the_library(owner, monkeypatch):
    key, hdr = owner
    google(monkeypatch, {"error": "authorization_pending"}, {"id_token": GOOGLE_ID})
    client = TestClient(app)
    client.post("/v1/link", headers=hdr)
    assert client.post("/v1/link", headers=hdr).json()["status"] == "waiting"
    assert client.post("/v1/link", headers=hdr).json() == {"status": "linked"}
    assert accounts.owner_of("google-123") == key
    conn = db.connect()
    assert conn.execute("SELECT COUNT(*) FROM account_links").fetchone()[0] == 0
    conn.close()


def test_the_owner_sees_the_link_and_cuts_it_off_alone(owner, monkeypatch):
    """`GET /v1/tokens` says since when a Google account reaches the library, and
    `POST /v1/unlink` ends that and nothing else: the token and its readers stay."""
    key, hdr = owner
    client = TestClient(app)
    assert client.get("/v1/tokens", headers=hdr).json()["google_linked_at"] is None
    google(monkeypatch, {"id_token": GOOGLE_ID})
    client.post("/v1/link", headers=hdr)
    assert client.post("/v1/link", headers=hdr).json() == {"status": "linked"}
    assert client.get("/v1/tokens", headers=hdr).json()["google_linked_at"]

    assert client.post("/v1/unlink", headers=hdr).json() == {"unlinked": True}
    assert accounts.owner_of("google-123") is None
    listed = client.get("/v1/tokens", headers=hdr).json()
    assert listed["google_linked_at"] is None and listed["owner"] == key
    assert client.post("/v1/unlink", headers=hdr).json() == {"unlinked": False}


def test_a_denied_sign_in_links_nothing_and_the_next_call_starts_over(owner, monkeypatch):
    key, hdr = owner
    g = google(monkeypatch, {"error": "access_denied"})
    client = TestClient(app)
    client.post("/v1/link", headers=hdr)
    assert client.post("/v1/link", headers=hdr).json()["status"] == "failed"
    assert accounts.owner_of("google-123") is None
    assert client.post("/v1/link", headers=hdr).json()["user_code"] == "CODE-2"
    assert g.codes == 2


def test_an_expired_code_is_replaced_without_asking_google_about_it(owner, monkeypatch):
    key, hdr = owner
    g = google(monkeypatch)
    accounts.link(key, now=0.0)                       # expires at 1800
    assert accounts.link(key, now=5000.0)["user_code"] == "CODE-2"
    assert g.codes == 2


def test_a_later_link_from_the_same_account_wins(owner, monkeypatch):
    key, hdr = owner
    google(monkeypatch, {"id_token": GOOGLE_ID}, {"id_token": GOOGLE_ID})
    other, _token = tokens.register_owner()
    for library in (key, other):
        accounts.link(library, now=0.0)
        assert accounts.link(library, now=1.0) == {"status": "linked"}
    assert accounts.owner_of("google-123") == other


def test_google_unreachable_is_a_503_and_keeps_the_code(owner, monkeypatch):
    key, hdr = owner
    g = google(monkeypatch, {"id_token": GOOGLE_ID})
    client = TestClient(app)
    client.post("/v1/link", headers=hdr)

    def down(*a, **kw):
        raise requests.ConnectionError("no route")
    monkeypatch.setattr(google_device.requests, "post", down)
    assert client.post("/v1/link", headers=hdr).status_code == 503
    monkeypatch.setattr(google_device.requests, "post", g.post)
    assert client.post("/v1/link", headers=hdr).json() == {"status": "linked"}
    assert g.codes == 1


def test_no_device_client_here_is_a_503(owner, monkeypatch):
    key, hdr = owner
    monkeypatch.delenv(google_device.CLIENT_SECRET_ENV)
    assert TestClient(app).post("/v1/link", headers=hdr).status_code == 503


def test_a_reader_token_cannot_link(owner, monkeypatch):
    key, _hdr = owner
    reader = tokens.mint_token(key, "reader")
    got = TestClient(app).post("/v1/link", headers={"Authorization": f"Bearer {reader}"})
    assert got.status_code == 403


@pytest.mark.parametrize("body, status", [
    ({"id_token": GOOGLE_ID}, "approved"),
    ({"error": "authorization_pending"}, "waiting"),
    ({"error": "slow_down"}, "waiting"),
    ({"error": "access_denied"}, "failed"),
    ({"error": "expired_token"}, "failed"),
    ({"id_token": id_token({"sub": "x", "aud": "another-client",
                            "iss": "https://accounts.google.com"})}, "failed"),
    ({"id_token": id_token({"sub": "x", "aud": CLIENT_ID, "iss": "https://evil.example"})},
     "failed"),
    ({"id_token": "not-a-jwt"}, "failed"),
])
def test_reading_googles_answer(body, status):
    assert google_device.read_answer(body, CLIENT_ID).status == status
