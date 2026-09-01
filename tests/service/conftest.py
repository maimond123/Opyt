"""Fixtures for the foreign-KB service — a real export, served over a real HTTP client.

Nothing here is a mock of the thing under test. `export_file` is built by `build_export` from a
seeded live store, it is uploaded through the actual endpoint, and the reader token is obtained by
minting a grant and redeeming it — the same three calls a real owner and reader make. The only
stand-in is the EMBEDDER, and only because the hosted one costs money and the test suite blocks
sockets; the vectors it produces are stored, exported and ranked by the unmodified code.

Two homes, and keeping them apart is the point: the OWNER's `$OPYT_HOME` holds the live store the
export is projected from, and the SERVICE's holds the peers registry, `service.db`, and
`exports/`. A test that confused them would prove nothing about a hop.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pipeline.kb import export, schema
from service import store
from service.app import app
from tests.kb.conftest import FakeEmbedder
from tests.kb.test_export import _corpus

OWNER = "david"
LABEL = "David's KB"


@pytest.fixture()
def emb():
    """The same vocabulary `tests/kb/test_export.py` seeds its corpus with, so the queries that
    file already exercises mean the same thing here."""
    return FakeEmbedder(["agent", "framework", "autonomous", "tools", "library",
                         "crypto", "rollup", "proof", "react", "dashboard", "web"])


@pytest.fixture()
def export_file(tmp_path, monkeypatch, emb):
    """A real export, built by `build_export` from a seeded live store under its own home."""
    live = tmp_path / "live"
    monkeypatch.setenv("OPYT_HOME", str(live))
    conn = schema.connect()
    _corpus(conn, emb)
    conn.close()
    out = tmp_path / "export.db"
    export.build_export(out)
    return out


@pytest.fixture()
def service_home(tmp_path, monkeypatch, export_file):
    """The SERVICE's `$OPYT_HOME`. Set after `export_file` has finished with the owner's."""
    home = tmp_path / "service"
    home.mkdir()
    monkeypatch.setenv("OPYT_HOME", str(home))
    return home


@pytest.fixture()
def svc(service_home, export_file):
    """A service with one owner token, one uploaded export, and one redeemed reader token.

    The owner token is minted directly against `service.db` rather than through
    `POST /v1/register`, because these fixtures need a KNOWN routing key: `OWNER` is what every
    URL in these tests is built from, and register assigns an opaque one. `test_register.py`
    exercises the self-service path against the same app.
    """
    client = TestClient(app)
    owner_token = store.mint_token(OWNER, "owner", label=LABEL)

    owner_hdr = {"Authorization": f"Bearer {owner_token}"}
    up = client.post(f"/v1/upload/{OWNER}", content=export_file.read_bytes(), headers=owner_hdr)
    assert up.status_code == 200, up.text

    code = client.post("/v1/grant", json={"label": "a reader"}, headers=owner_hdr).json()["code"]
    redeemed = client.post("/v1/redeem", json={"code": code, "install_id": "install-1"}).json()

    return SimpleNamespace(
        client=client,
        owner=OWNER,
        owner_token=owner_token,
        owner_hdr=owner_hdr,
        reader_token=redeemed["token"],
        reader_hdr={"Authorization": f"Bearer {redeemed['token']}"},
        export=export_file,
        upload=up.json(),
        home=service_home,
    )


def query_vector(emb, query: str) -> list[float]:
    """What a READER computes on their own machine and sends: the query embedded with the model
    the export's `kb_meta` names, as plain floats. The server never calls an embedder for it."""
    return [float(x) for x in np.asarray(emb.embed([query], role="query")[0])]
