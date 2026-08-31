"""`opyt-push` — build the export and replace what the service serves.

The owner's whole publish loop is this one command, so the tests assert on what a publish is
supposed to leave behind: the service now serves what THIS store holds (not the file it was
seeded with), the bytes that arrived hash to what was sent, and each of the two settings the
command needs is named — by name, and by where it is set — when it is absent.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from opyt_core import config, push
from opyt_core.paths import opyt_path
from pipeline.kb import schema
from service import uploads
from tests.kb.test_export import _add

URL = "https://svc.test"
NEW = "github:pushed/after-the-seed"


@pytest.fixture()
def cli(svc, tmp_path, monkeypatch, emb):
    """push's transport pointed at the real service, with the two homes SEPARATED the way a real
    publish has them: the OWNER's home holds the live store the export is projected from and the
    settings file naming the service, and anything crossing the shim runs under the SERVICE's,
    which holds `service.db`, `exports/` and the peers registry."""
    owner_home = tmp_path / "live"          # the store `export_file` seeded
    service_home = str(svc.home)
    monkeypatch.setenv("OPYT_HOME", str(owner_home))
    monkeypatch.setenv("OPYT_SERVICE_TOKEN", svc.owner_token)
    # What an owner's settings.yaml looks like once they add the key: the shipped template, plus
    # the one line. Written under the owner's home, which is where `config_path()` looks.
    template = (config.REPO_ROOT / "config" / "settings.example.yaml").read_text()
    settings = owner_home / "settings.yaml"
    settings.write_text(f"{template}\nservice_url: {URL}\n")

    @contextmanager
    def on_service():
        os.environ["OPYT_HOME"] = service_home
        try:
            yield
        finally:
            os.environ["OPYT_HOME"] = str(owner_home)

    def _shim(verb):
        def call(url, **kw):
            kw.pop("timeout", None)   # meaningless in-process; TestClient deprecates it per-request
            if "data" in kw:          # httpx names the raw body `content`; requests names it `data`
                kw["content"] = kw.pop("data")
            with on_service():
                return verb(url, **kw)
        return call

    monkeypatch.setattr(push, "requests", SimpleNamespace(
        get=_shim(svc.client.get), post=_shim(svc.client.post)))
    return SimpleNamespace(svc=svc, on_service=on_service, home=owner_home,
                           settings=settings, template=template, emb=emb)


def _ingest_one_more(emb) -> None:
    """An atom the seeded export the service already holds does not have — so "the service now
    serves what this store holds" is observable rather than a byte comparison of two identical
    files."""
    conn = schema.connect()
    _add(conn, emb, NEW, "github", "artifact", "github:pushed", ["ai-agents"],
         "a library of autonomous agent tools added after the first upload")
    conn.close()


def _served_ids(cli) -> set[str]:
    with cli.on_service():
        served = uploads.export_path(cli.svc.owner)
        conn = sqlite3.connect(f"file:{served}?mode=ro", uri=True)
    try:
        return {r[0] for r in conn.execute("SELECT atom_id FROM atoms")}
    finally:
        conn.close()


def _served_sha(cli) -> str:
    with cli.on_service():
        return hashlib.sha256(uploads.export_path(cli.svc.owner).read_bytes()).hexdigest()


def test_push_replaces_what_the_service_serves(cli, capsys):
    assert NEW not in _served_ids(cli)      # the seeded upload predates this atom
    _ingest_one_more(cli.emb)

    assert push.main([]) == 0

    assert NEW in _served_ids(cli)
    out = capsys.readouterr().out
    assert _served_sha(cli)[:12] in out     # what arrived hashes to what was sent
    assert not opyt_path("tmp", "export-push.db").exists()


def test_a_sha_mismatch_fails_loudly(cli, capsys, monkeypatch):
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
    assert "0" * 64 in err and _served_sha(cli)[:12] in err
    assert not opyt_path("tmp", "export-push.db").exists()   # the finally runs on this path too


def test_a_missing_token_names_the_env_var(cli, capsys, monkeypatch):
    monkeypatch.setattr(push, "get_credential", lambda service: None)
    assert push.main([]) == 1
    err = capsys.readouterr().err
    assert "OPYT_SERVICE_TOKEN" in err and "opyt-keys" in err


def test_a_missing_service_url_names_the_config_key(cli, capsys):
    cli.settings.write_text(cli.template)   # the same file, without the one line
    assert push.main([]) == 1
    err = capsys.readouterr().err
    assert "service_url" in err and str(cli.settings) in err
