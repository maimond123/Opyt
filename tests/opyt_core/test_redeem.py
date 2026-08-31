"""`opyt-redeem` — a grant code in, a queryable peer out.

The command is the reader's entire setup: it exchanges the one-time code for a reader token and
writes the peer row the `kb=` entry points read. So the tests assert on those two artifacts —
the row (location, token, name) and the printed instruction — and on the two failure facts: a
spent code registers nothing, and the install id is minted once, ever.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from opyt_core import redeem
from opyt_core.paths import opyt_path
from pipeline.kb import peers
from service import store

URL = "https://svc.test"


@pytest.fixture()
def cli(svc, tmp_path, monkeypatch):
    """redeem's transport pointed at the real service, with the two homes SEPARATED: client-side
    code (the install id, the peer row) writes under a fresh reader home, and anything crossing
    the shim runs under the service's — the same two-machine split a real redeem has. B4's tests
    could share one home because the reader registered a name the service never opens; redeem's
    default name IS the owner's, and under a shared home the new row would overwrite the
    service's own peers row for the export it serves."""
    reader_home = tmp_path / "reader"
    reader_home.mkdir()
    service_home = str(svc.home)
    monkeypatch.setenv("OPYT_HOME", str(reader_home))

    @contextmanager
    def on_service():
        os.environ["OPYT_HOME"] = service_home
        try:
            yield
        finally:
            os.environ["OPYT_HOME"] = str(reader_home)

    def _post(url, **kw):
        kw.pop("timeout", None)   # meaningless in-process; TestClient deprecates it per-request
        with on_service():
            return svc.client.post(url, **kw)

    monkeypatch.setattr(redeem, "requests", SimpleNamespace(post=_post))

    def mint():
        with on_service():
            return svc.client.post("/v1/grant", json={"label": "a friend"},
                                   headers=svc.owner_hdr).json()["code"]

    return SimpleNamespace(svc=svc, mint=mint, on_service=on_service, home=reader_home)


def test_redeem_writes_the_peer_row_and_says_how_to_use_it(cli, capsys):
    rc = redeem.main([URL, cli.mint()])
    assert rc == 0

    row = peers.get(cli.svc.owner)   # the name defaults to the owner's
    assert row["location"] == f"{URL}/v1/kb/{cli.svc.owner}"
    with cli.on_service():
        resolved = store.resolve_token(row["token"])
    assert resolved and resolved["role"] == "reader" and resolved["owner"] == cli.svc.owner

    out = capsys.readouterr().out
    assert f"kb='{cli.svc.owner}'" in out


def test_the_default_peer_name_makes_the_foreign_kb_hint_resolve(cli):
    """Search notices tell the host to pass `kb='<owner>'` back to `open()` — the OWNER'S name,
    because the envelope crosses verbatim. Defaulting the peer name to the owner is what makes
    that hint resolve on this install, so the default is a contract, not a convenience."""
    assert redeem.main([URL, cli.mint()]) == 0
    assert peers.get(cli.svc.owner) is not None


def test_name_overrides_the_default(cli):
    assert redeem.main([URL, cli.mint(), "--name", "colleague"]) == 0
    assert peers.get("colleague")["location"] == f"{URL}/v1/kb/{cli.svc.owner}"
    assert peers.get(cli.svc.owner) is None


def test_a_spent_code_fails_loudly_and_registers_nothing(cli, capsys):
    code = cli.mint()
    assert redeem.main([URL, code]) == 0
    _conn = peers.schema.connect()               # forget the peer so "registers
    _conn.execute("DELETE FROM peers WHERE name = ?", (cli.svc.owner,))
    _conn.commit(); _conn.close()               # nothing" is observable below
    with cli.on_service():        # the server's own sentence for this exact failure
        sentence = cli.svc.client.post("/v1/redeem", json={"code": code}).json()["detail"]
    capsys.readouterr()

    rc = redeem.main([URL, code])
    assert rc == 1
    assert sentence in capsys.readouterr().err
    assert peers.get(cli.svc.owner) is None


def test_install_id_is_minted_once_and_reused(cli):
    assert redeem.main([URL, cli.mint()]) == 0
    iid = opyt_path("install_id").read_text().strip()
    assert iid

    assert redeem.main([URL, cli.mint(), "--name", "second"]) == 0
    assert opyt_path("install_id").read_text().strip() == iid
    with cli.on_service():
        rows = store.list_tokens(cli.svc.owner)
    assert len([t for t in rows if t["install_id"] == iid]) == 2
