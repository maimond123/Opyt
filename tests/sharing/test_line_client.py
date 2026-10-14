"""The line's two routes as an install calls them: `client.line_next` and `client.line_answer`.

The service is the stand-in, at the two verbs `service_http` sends with, so what is asserted is
the request that leaves the install: its address, its token, its body and its timeout. The
service's own half of the wire is tested with the service.
"""
from __future__ import annotations

import pytest
from requests import ConnectionError as NoConnection

from opyt_core import service_http
from pipeline.sharing import account, client

ACCT = account.Account("https://svc.test", "tok-owner")


class _Answer:
    def __init__(self, status: int, body: dict):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


@pytest.fixture()
def service(monkeypatch):
    """Every request the install sends, answered with `status` and `body`."""
    sent = []

    def make(status: int = 200, body: dict | None = None):
        def send(verb):
            def call(url, **kw):
                sent.append((verb, url, kw))
                return _Answer(status, body if body is not None else {})
            return call
        monkeypatch.setattr(service_http.requests, "get", send("get"))
        monkeypatch.setattr(service_http.requests, "post", send("post"))
        return sent
    return make


def test_asking_is_a_get_that_names_its_wait_and_outlasts_it(service):
    """The service holds the question open for the wait, so a read that timed out first would
    drop every line just before its answer."""
    call = {"id": "ab12", "tool": "hopper", "args": {"reference": "https://arxiv.org/abs/1"}}
    sent = service(body={"call": call})

    assert client.line_next(ACCT, 50) == {"call": call}

    (verb, url, kw), = sent
    assert (verb, url) == ("get", "https://svc.test/v1/line/next")
    assert kw["params"] == {"wait": 50} and kw["timeout"] > 50
    assert kw["headers"] == {"Authorization": "Bearer tok-owner"}


@pytest.mark.parametrize("body", [{"call": None}, {"call": None, "retry_after": 3600}])
def test_asking_answers_what_the_service_said_when_nothing_was_handed_over(service, body):
    service(body=body)
    assert client.line_next(ACCT, 50) == body


def test_answering_posts_whether_the_call_ran_and_the_tools_own_answer(service):
    sent = service(body={"recorded": True})

    got = client.line_answer(ACCT, "ab12", True, {"status": "saved", "atom_id": "paper:1"})

    (verb, url, kw), = sent
    assert got == {"recorded": True}
    assert (verb, url) == ("post", "https://svc.test/v1/line/ab12/answer")
    assert kw["json"] == {"ok": True, "answer": {"status": "saved", "atom_id": "paper:1"}}
    assert kw["headers"] == {"Authorization": "Bearer tok-owner"}


def test_an_answer_the_service_has_no_call_for_is_not_a_failure(service):
    """No running call of this library has that id: the install drops the answer."""
    service(body={"recorded": False})
    assert client.line_answer(ACCT, "gone", False, {"status": "error"}) == {"recorded": False}


@pytest.mark.parametrize("ask", [
    lambda acct: client.line_next(acct, 50),
    lambda acct: client.line_answer(acct, "ab12", True, {}),
])
def test_a_401_on_the_line_drops_the_token_as_on_every_owner_route(service, ask, monkeypatch):
    """The library is gone, and the install stops: nothing asks, publishes or links with it."""
    acct = account.save_token(account.load(), "tok-dead")
    monkeypatch.setenv("OPYT_SERVICE_TOKEN", "tok-dead")
    service(401, {"detail": "unknown or revoked token"})

    with pytest.raises(client.Failed) as e:
        ask(acct)

    assert (e.value.kind, e.value.status) == ("revoked", 401)
    assert account.load().token is None


def test_a_service_that_is_away_keeps_the_token_and_says_which_way(service, monkeypatch):
    acct = account.save_token(account.load(), "tok-live")
    service(503, {"detail": "upstream down"})
    with pytest.raises(client.Failed) as e:
        client.line_next(acct, 50)
    assert (e.value.kind, e.value.status) == ("refused", 503)

    def dead(url, **kw):
        raise NoConnection("connection refused")
    monkeypatch.setattr(service_http.requests, "get", dead)
    with pytest.raises(client.Failed) as e:
        client.line_next(acct, 50)
    assert (e.value.kind, e.value.status) == ("unreachable", None)
    assert account.load().token == "tok-live"
