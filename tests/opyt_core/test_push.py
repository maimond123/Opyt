"""`push.publish` — build the export and replace what the service serves.

The owner's whole publish loop is this one function, so the tests assert on what a publish is
supposed to leave behind: the service now serves what THIS store holds (not the file it was
seeded with), and the bytes that arrived hash to what was sent.

They call `publish()` DIRECTLY. Until 2026-09-05 they reached it through `opyt-push`'s `main()`
and read its stdout; the command is gone and the function is what `pipeline/kb/push_catchup.py`
calls, so the returned dict is now both the real contract and the stronger assertion — a whole
sha256 rather than the twelve characters a print statement happened to show.
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


def test_push_replaces_what_the_service_serves(publisher):
    assert NEW not in _served_ids(publisher)      # the seeded upload predates this atom
    _ingest_one_more(publisher.emb)

    res = push.publish(publisher.svc.owner_token, URL)

    assert res["status"] == "ok"
    assert NEW in _served_ids(publisher)
    assert res["sha256"] == _served_sha(publisher)   # what arrived hashes to what was sent
    assert not opyt_path("tmp", "export-push.db").exists()


def test_a_sha_mismatch_fails_loudly(publisher, monkeypatch):
    """The bytes that arrived are not the bytes that were sent. The STATUS is the whole point —
    a publish that half-worked must not report ok. `push_catchup` writes its watermark only on
    `status == "ok"` (push_catchup.py:149), so a soft failure here would mark the store published
    against a damaged copy and no later pass would ever correct it."""
    real_post = push.requests.post

    def corrupt(url, **kw):
        r = real_post(url, **kw)
        body = {**r.json(), "sha256": "0" * 64}
        return SimpleNamespace(status_code=r.status_code, json=lambda: body, text=r.text)

    monkeypatch.setattr(push.requests, "post", corrupt)

    res = push.publish(publisher.svc.owner_token, URL)

    assert res["status"] == "corrupt"
    assert "0" * 64 in res["message"] and _served_sha(publisher) in res["message"]
    assert not opyt_path("tmp", "export-push.db").exists()   # the finally runs on this path too


def test_an_unset_service_url_falls_back_to_the_hosted_service(publisher):
    """Sharing must not begin with editing a config file, so `service_url` has a default rather
    than an error. The key still WINS when it is set — which is what keeps a self-hosted service
    and this whole fixture possible."""
    assert config.service_url() == URL

    publisher.settings.write_text(publisher.template)   # the same file, without the one line
    assert config.service_url() == config.DEFAULT_SERVICE_URL
