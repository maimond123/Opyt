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
from tests.service.conftest import LABEL

URL = "https://svc.test"


@pytest.fixture()
def cli(svc, tmp_path, monkeypatch):
    """redeem's transport pointed at the real service, with the two homes SEPARATED: client-side
    code (the install id, the peer row) writes under a fresh reader home, and anything crossing
    the shim runs under the service's — the same two-machine split a real redeem has. The split
    is not a nicety: the service's OWN peers registry lives in its home and names the export it
    serves, so a reader row written under that home would land in the same table."""
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

    row = peers.get(LABEL)   # the name the OWNER registered under, from `suggested_name`
    assert row["location"] == f"{URL}/v1/kb/{cli.svc.owner}"
    with cli.on_service():
        resolved = store.resolve_token(row["token"])
    assert resolved and resolved["role"] == "reader" and resolved["owner"] == cli.svc.owner

    out = capsys.readouterr().out
    assert f"kb='{LABEL}'" in out


def test_the_peer_name_is_the_readers_and_the_routing_key_stays_in_the_url(cli):
    """R4. The name in the URL is a ROUTING key — after R5 an opaque hex string nobody types —
    and the reader never registers a peer under it. What they register under comes from
    `suggested_name`, and every request then sends `as_kb`, so the notices that tell the host to
    pass `kb='...'` back to `open()` name a peer THIS install resolves.

    Before R4 this test asserted the opposite: the peer name had to equal the owner's, because
    the envelope crossed the service verbatim carrying the owner's name and nothing else would
    have resolved."""
    assert redeem.main([URL, cli.mint()]) == 0

    assert peers.get(LABEL)["location"] == f"{URL}/v1/kb/{cli.svc.owner}"
    assert peers.get(cli.svc.owner) is None


def test_name_overrides_the_suggestion(cli):
    assert redeem.main([URL, cli.mint(), "--name", "colleague"]) == 0
    assert peers.get("colleague")["location"] == f"{URL}/v1/kb/{cli.svc.owner}"
    assert peers.get(LABEL) is None


def test_a_second_owner_suggesting_a_taken_name_is_suffixed_not_overwritten(cli, capsys):
    """R4a meeting R4: suggestions are not unique — two owners may both call themselves the same
    thing, and after R4 nothing stops them, because uniqueness moved onto the routing key. The
    first reader token must survive, since the registry holds the only copy of it."""
    assert redeem.main([URL, cli.mint()]) == 0
    first = peers.get(LABEL)["token"]

    with cli.on_service():
        hdr = {"Authorization": f"Bearer {store.mint_token('other-key', 'owner', label=LABEL)}"}
        code = cli.svc.client.post("/v1/grant", json={}, headers=hdr).json()["code"]
    capsys.readouterr()
    assert redeem.main([URL, code]) == 0

    assert peers.get(LABEL)["token"] == first
    assert peers.get(f"{LABEL}-2")["location"] == f"{URL}/v1/kb/other-key"
    assert f"kb='{LABEL}-2'" in capsys.readouterr().out


def test_a_spent_code_fails_loudly_and_registers_nothing(cli, capsys):
    code = cli.mint()
    assert redeem.main([URL, code]) == 0
    _conn = peers.schema.connect()               # forget the peer so "registers
    _conn.execute("DELETE FROM peers WHERE name = ?", (LABEL,))
    _conn.commit(); _conn.close()               # nothing" is observable below
    with cli.on_service():        # the server's own sentence for this exact failure
        sentence = cli.svc.client.post("/v1/redeem", json={"code": code}).json()["detail"]
    capsys.readouterr()

    rc = redeem.main([URL, code])
    assert rc == 1
    assert sentence in capsys.readouterr().err
    assert peers.get(LABEL) is None


def test_install_id_is_minted_once_and_reused(cli):
    assert redeem.main([URL, cli.mint()]) == 0
    iid = opyt_path("install_id").read_text().strip()
    assert iid

    assert redeem.main([URL, cli.mint(), "--name", "second"]) == 0
    assert opyt_path("install_id").read_text().strip() == iid
    with cli.on_service():
        rows = store.list_tokens(cli.svc.owner)
    assert len([t for t in rows if t["install_id"] == iid]) == 2
