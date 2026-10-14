"""A token the service no longer knows: `client.py` drops it, and the builder stops publishing.

The service answers 401 to a token it holds nothing for, after `unshare` or a deletion done on the
service when a user asks by email. The service is the stand-in here, at `service_http.request`;
the account files, the push markers and the publishing pass are the real ones.
"""
from __future__ import annotations

import pytest

from opyt_core import service_http
from pipeline.sharing import account, client, publishing, push_gate
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop.source import Unit


@pytest.fixture()
def answers(monkeypatch):
    """Every request the builder sends, refused with `status`."""
    sent = []

    def make(status: int, detail: str = "unknown or revoked token"):
        def refuse(method, url, **kw):
            sent.append(url)
            raise service_http.Refused(status, detail)
        monkeypatch.setattr(service_http, "request", refuse)
        return sent
    return make


@pytest.fixture()
def shared(monkeypatch):
    """An install that shared: its token in `.env`, and the copy `opyt_core` put in this process's
    environment at import, which is what a running worker still reads once `.env` lost it. A push
    is owed, so the pass asks the service (an unchanged store asks it nothing)."""
    account.save_token(account.load(), "tok-dead")
    monkeypatch.setenv("OPYT_SERVICE_TOKEN", "tok-dead")
    push_gate.owe()


def _pass() -> dict:
    src, unit = publishing.Publications(), Unit("publishing:pass", utc_now())
    fetched = src.fetch(None, unit, None)
    src.record(None, unit, fetched)
    return fetched.result


def test_a_refused_token_is_dropped_and_no_pass_is_due_again(shared, answers):
    sent = answers(401)
    got = _pass()

    assert got["status"] == "revoked" and "no longer holds" in got["message"]
    assert account.load().token is None
    assert publishing.Publications().next_due(None, utc_now()) is None
    assert len(sent) == 1, "the pass registered or uploaded after the refusal"


def test_a_held_token_goes_with_it(shared, answers):
    account.hold_token("tok-held")
    answers(401)
    with pytest.raises(client.Failed) as e:
        client.grant(account.load(), None)
    assert e.value.kind == "revoked"
    assert account.load().token is None and account.pending_token() is None


def test_a_server_error_keeps_the_token(shared, answers):
    answers(503, "upstream down")
    assert _pass()["status"] == "refused"
    assert account.load().token == "tok-dead"
