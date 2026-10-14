"""`unshare` for the whole knowledge base, on this install's side: `pipeline/sharing/unshare.py`.

The service is the stand-in, at `client.unpublish`; the account files and the push markers are the
real ones. What the service itself deletes is held by the service's own tests.
"""
from __future__ import annotations

import pytest

from pipeline.sharing import account, client, push_gate, unshare

DELETED = {"owner": "k1", "readers_revoked": 2, "export_deleted": True, "deleted": True}


@pytest.fixture()
def shared(monkeypatch):
    """An install that shared and published: its token in `.env` and in this process's
    environment, a watermark, and a push owed."""
    account.save_token(account.load(), "tok-1")
    monkeypatch.setenv("OPYT_SERVICE_TOKEN", "tok-1")
    push_gate.write_watermark({"atoms_at": "2026-10-12", "chunk_id": 1, "atom_count": 1})
    push_gate.owe()


def test_a_confirmed_unshare_drops_the_token_and_says_what_is_gone(shared, monkeypatch):
    monkeypatch.setattr(client, "unpublish", lambda acct: DELETED)
    got = unshare.run(confirm=True, reader=None)

    assert got["status"] == "unshared"
    assert account.load().token is None
    assert push_gate.read_watermark() is None and push_gate.owed_at() is None
    for said in ("Google account", "claude.ai and ChatGPT", "share link", "this computer"):
        assert said in got["message"]
    assert unshare.run(confirm=True, reader=None)["status"] == "not_shared"


def test_a_failed_unshare_deletes_nothing_here(shared, monkeypatch):
    def down(acct):
        raise client.Failed("unreachable", "could not reach https://api.useopyt.com")
    monkeypatch.setattr(client, "unpublish", down)
    got = unshare.run(confirm=True, reader=None)

    assert got["status"] == "failed" and "could not reach" in got["message"]
    assert account.load().token == "tok-1"
    assert push_gate.read_watermark() is not None and push_gate.owed_at() is not None
