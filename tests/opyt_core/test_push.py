"""`opyt-push` — build the export and replace what the service serves.

The owner's whole publish loop is this one command, so the tests assert on what a publish is
supposed to leave behind: the service now serves what THIS store holds (not the file it was
seeded with), the bytes that arrived hash to what was sent, and each of the two settings the
command needs is named — by name, and by where it is set — when it is absent.
"""
from __future__ import annotations

import hashlib
import sqlite3
from types import SimpleNamespace

from opyt_core import config, push
from opyt_core.paths import opyt_path
from pipeline.kb import schema
from service import uploads
from tests.kb.test_export import _add
from tests.opyt_core.conftest import URL

NEW = "github:pushed/after-the-seed"


def _ingest_one_more(emb) -> None:
    """An atom the seeded export the service already holds does not have — so "the service now
    serves what this store holds" is observable rather than a byte comparison of two identical
    files."""
    conn = schema.connect()
    _add(conn, emb, NEW, "github", "artifact", "github:pushed", ["ai-agents"],
         "a library of autonomous agent tools added after the first upload")
    conn.close()


def _served_ids(publisher) -> set[str]:
    with publisher.on_service():
        served = uploads.export_path(publisher.svc.owner)
        conn = sqlite3.connect(f"file:{served}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT atom_id FROM atoms")}
    finally:
        conn.close()


def _served_sha(publisher) -> str:
    with publisher.on_service():
        return hashlib.sha256(uploads.export_path(publisher.svc.owner).read_bytes()).hexdigest()


def test_push_replaces_what_the_service_serves(publisher, capsys):
    assert NEW not in _served_ids(publisher)      # the seeded upload predates this atom
    _ingest_one_more(publisher.emb)

    assert push.main([]) == 0

    assert NEW in _served_ids(publisher)
    out = capsys.readouterr().out
    assert _served_sha(publisher)[:12] in out     # what arrived hashes to what was sent
    assert not opyt_path("tmp", "export-push.db").exists()


def test_a_sha_mismatch_fails_loudly(publisher, capsys, monkeypatch):
    """The bytes that arrived are not the bytes that were sent. The exit code is the whole point
    — a publish that half-worked must not look like one that worked."""
    real_post = push.requests.post

    def corrupt(url, **kw):
        r = real_post(url, **kw)
        body = {**r.json(), "sha256": "0" * 64}
        return SimpleNamespace(status_code=r.status_code, json=lambda: body, text=r.text)

    monkeypatch.setattr(push.requests, "post", corrupt)

    assert push.main([]) == 1
    err = capsys.readouterr().err
    assert "0" * 64 in err and _served_sha(publisher)[:12] in err
    assert not opyt_path("tmp", "export-push.db").exists()   # the finally runs on this path too


def test_a_missing_token_names_the_env_var(publisher, capsys, monkeypatch):
    monkeypatch.setattr(push, "get_credential", lambda service: None)
    assert push.main([]) == 1
    err = capsys.readouterr().err
    assert "OPYT_SERVICE_TOKEN" in err and "opyt-keys" in err


def test_an_unset_service_url_falls_back_to_the_hosted_service(publisher):
    """Sharing must not begin with editing a config file, so `service_url` has a default rather
    than an error. The key still WINS when it is set — which is what keeps a self-hosted service
    and this whole fixture possible."""
    assert config.service_url() == URL

    publisher.settings.write_text(publisher.template)   # the same file, without the one line
    assert config.service_url() == config.DEFAULT_SERVICE_URL
