"""Connecting X on an install with no browser a person can see, through Opyt's service.

The service is the one stand-in, at `client.x_signin` and `client.x_signin_collect`. The home,
`.env` and the session reader the X collectors use are the real ones.
"""
from __future__ import annotations

import os
import stat

import pytest

from opyt_core import keys
from pipeline.onboarding import connect, service_signin
from pipeline.platforms.browser import cookies
from pipeline.sharing import client

SESSION = {"auth_token": "a" * 40, "ct0": "c" * 32, "twid": "u%3D42"}
BASE = "https://api.useopyt.com"


class Service:
    """The two routes. `replies` is what collect says, in order; the last one repeats."""

    def __init__(self, *replies: dict, down: bool = False):
        self.replies, self.down, self.minted = list(replies), down, 0

    def x_signin(self, url):
        if self.down:
            raise client.Failed("unreachable", "could not reach the service")
        self.minted += 1
        return {"signin": f"token-{self.minted}", "path": f"/x-signin/nonce-{self.minted}",
                "expires_in": 600}

    def x_signin_collect(self, url, signin):
        if self.down:
            raise client.Failed("unreachable", "could not reach the service")
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    # Set, then removed: the monkeypatch then undoes what `_save` writes into this process.
    for name in (service_signin.SESSION_ENV, service_signin.TIMEOUT_ENV):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(service_signin.account, "service_url", lambda: BASE)
    monkeypatch.setattr(service_signin, "_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(service_signin, "_POLL_SECONDS", 0.0)
    return tmp_path


def serve(monkeypatch, service: Service) -> Service:
    monkeypatch.setattr(client, "x_signin", service.x_signin)
    monkeypatch.setattr(client, "x_signin_collect", service.x_signin_collect)
    return service


@pytest.mark.parametrize("platform, source, route", [
    ("linux", "x", "service"),
    ("linux", "substack", "local"),
    ("darwin", "x", "local"),
])
def test_which_sign_in_a_source_gets(platform, source, route):
    assert connect.route(platform, source) == route


def test_x_off_macos_goes_to_the_service(monkeypatch, home):
    serve(monkeypatch, Service({"status": "waiting"}))
    monkeypatch.setattr(connect.sys, "platform", "linux")
    assert connect.start("x")["login_url"] == f"{BASE}/x-signin/nonce-1"


def test_the_first_call_hands_the_user_a_link_and_keeps_the_token_private(monkeypatch, home):
    serve(monkeypatch, Service({"status": "waiting"}))
    answer = service_signin.connect()
    assert answer["status"] == "awaiting_login" and answer["login_url"].endswith("/nonce-1")
    assert "**Connect X to Opyt**" in answer["message"]
    assert "token-1" not in str(answer)
    pending = home / "x-signin.json"
    assert stat.S_IMODE(pending.stat().st_mode) == 0o600


def test_a_call_while_the_user_signs_in_keeps_the_same_link(monkeypatch, home):
    service = serve(monkeypatch, Service({"status": "waiting"}))
    first = service_signin.connect()
    again = service_signin.connect()
    assert again["status"] == "awaiting_login" and again["login_url"] == first["login_url"]
    assert service.minted == 1


def test_once_the_user_is_in_the_session_lands_where_the_x_collectors_read_it(monkeypatch,
                                                                           home):
    serve(monkeypatch, Service({"status": "connected", "session": SESSION}))
    service_signin.connect()
    answer = service_signin.connect()
    assert answer["status"] == "connected"
    assert cookies.env_session(["x.com"]) == SESSION
    saved = keys._read()
    assert saved[service_signin.SESSION_ENV] == os.environ[service_signin.SESSION_ENV]
    assert saved[service_signin.TIMEOUT_ENV] == "180"
    assert not (home / "x-signin.json").exists()


def test_a_timeout_already_set_is_kept(monkeypatch, home):
    monkeypatch.setenv(service_signin.TIMEOUT_ENV, "60")
    serve(monkeypatch, Service({"status": "connected", "session": SESSION}))
    service_signin.connect()
    service_signin.connect()
    assert service_signin.TIMEOUT_ENV not in keys._read()


@pytest.mark.parametrize("ended", ["failed", "expired"])
def test_a_sign_in_that_ended_gets_a_fresh_link(monkeypatch, home, ended):
    service = serve(monkeypatch, Service({"status": ended}))
    service_signin.connect()
    answer = service_signin.connect()
    assert answer["status"] == "awaiting_login" and answer["login_url"].endswith("/nonce-2")
    assert service.minted == 2


def test_an_unreachable_service_is_an_answer_and_keeps_the_pending_sign_in(monkeypatch, home):
    service = serve(monkeypatch, Service({"status": "waiting"}))
    service_signin.connect()
    service.down = True
    assert service_signin.connect()["status"] == "needs_login"
    assert (home / "x-signin.json").exists()
    assert service_signin.SESSION_ENV not in keys._read()
