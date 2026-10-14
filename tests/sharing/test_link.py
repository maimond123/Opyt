"""Connecting this library to Google from the builder: `pipeline/sharing/link.py`.

The service is the stand-in, at `client.register` and `client.link`; the account files, the push
debt and the worker wake are the real ones.
"""
from __future__ import annotations

import pytest

from pipeline.sharing import account, client, desktop_only, link, push_gate, share
from pipeline.worker import job_queue
from pipeline.worker.registry import LINE, RAILS, SCRAPERS


class Service:
    """The two routes the flow calls. `answers` is what `/v1/link` says, in order."""

    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.registered = 0
        self.linked_with: list[str] = []

    def register(self, url, label):
        self.registered += 1
        return {"owner": "k1", "token": f"tok-{self.registered}"}

    def link(self, acct):
        self.linked_with.append(acct.token)
        return self.answers.pop(0)


WAITING = {"status": "waiting", "user_code": "ABCD-EFGH",
           "verification_url": "https://www.google.com/device"}


@pytest.fixture()
def service(monkeypatch):
    def make(*answers):
        svc = Service(*answers)
        monkeypatch.setattr(client, "register", svc.register)
        monkeypatch.setattr(client, "link", svc.link)
        return svc
    return make


@pytest.fixture()
def woken(monkeypatch):
    calls = []
    monkeypatch.setattr(job_queue, "request_now", lambda name: calls.append(name) or True)
    return calls


def test_a_first_call_shows_the_code_and_publishes_nothing(service, woken):
    svc = service(WAITING)
    got = link.run()
    assert got["status"] == "waiting" and "ABCD-EFGH" in got["message"]
    assert got["open_this_url"] == "https://www.google.com/device"
    # Registered, but the token is held aside: the publishing pass sees no account yet.
    assert account.load().token is None and account.pending_token() == "tok-1"
    assert push_gate.owed_at() is None and woken == []


def test_approval_makes_the_held_token_the_account_and_owes_the_first_push(service, woken):
    svc = service(WAITING, {"status": "linked"})
    link.run()
    got = link.run()
    assert got["status"] == "ok"
    assert svc.registered == 1 and svc.linked_with == ["tok-1", "tok-1"]
    assert account.load().token == "tok-1" and account.pending_token() is None
    # The scrapers, whose publishing pass uploads the library, and the line, which from now on
    # has a web writer to hold it for: woken at once, not at its next hourly ask.
    assert push_gate.owed_at() is not None and woken == [SCRAPERS, LINE]
    assert push_gate.fast_since() is not None


def test_every_job_the_connection_wakes_is_one_the_worker_can_launch(service, woken):
    """A name outside the registry is a row nothing ever dispatches, and nothing says so."""
    service({"status": "linked"})
    link.run()
    assert woken and set(woken) <= set(RAILS)


def test_approval_says_how_to_read_the_library_from_claude_and_chatgpt(service, woken):
    service({"status": "linked"})
    said = link.run()["message"]
    assert said.count("https://api.useopyt.com/mcp") == 2
    for step in ("Add custom connector", "Allow Access", "Developer mode", "OAuth",
                 "this same Google account"):
        assert step in said
    assert "Muse" not in said      # not in Meta's connector store until it is approved


def test_a_denied_sign_in_publishes_nothing_and_a_retry_reuses_the_registration(service, woken):
    svc = service(WAITING, {"status": "failed", "message": "access_denied"}, WAITING)
    link.run()
    got = link.run()
    assert got["status"] == "error" and "access_denied" in got["message"]
    assert account.load().token is None
    link.run()
    assert svc.registered == 1 and svc.linked_with == ["tok-1"] * 3


def test_an_install_that_already_shares_links_its_own_library(service, woken):
    account.save_token(account.load(), "shared-token")
    svc = service({"status": "linked"})
    assert link.run()["status"] == "ok"
    assert svc.registered == 0 and svc.linked_with == ["shared-token"]
    assert account.load().token == "shared-token"


def test_an_unreachable_service_is_a_sentence(monkeypatch, woken):
    def down(url, label):
        raise client.Failed("unreachable", "could not reach https://api.useopyt.com")
    monkeypatch.setattr(client, "register", down)
    got = link.run()
    assert got["status"] == "error" and "could not reach" in got["message"]
    assert account.pending_token() is None



# ── reached through `share`, which is the one way in ─────────────────────────────

def test_share_for_the_users_own_claude_previews_and_then_runs_this_sign_in(service, woken):
    """`to="connector"`: the preview is the consent, on a library that may still be empty, and
    it says what only the desktop app does; the confirm is one step of the sign-in."""
    preview = share.run(False, None, None, "connector")
    assert preview["status"] == "preview" and desktop_only.SENTENCE in preview["consent"]
    assert "who_is_it_for" not in preview
    assert account.pending_token() is None           # a preview registers nothing

    service(WAITING)
    got = share.run(True, None, None, "connector")
    assert got["status"] == "waiting" and "ABCD-EFGH" in got["message"]
    assert "share(to='connector', confirm=True)" in got["message"]


def test_once_connected_the_user_is_told_what_only_the_desktop_app_does(service, woken):
    service(WAITING, {"status": "linked"})
    link.run()
    assert desktop_only.SENTENCE in link.run()["message"]


def test_share_refuses_a_word_it_does_not_know():
    got = share.run(True, None, None, "everyone")
    assert got["status"] == "error" and "`person`, `agent` or `connector`" in got["message"]
