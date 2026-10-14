"""Fixtures for the foreign-KB service — a real export, served over a real HTTP client.

Nothing here is a mock of the thing under test. `export_file` is built by `build_export` from a
seeded live store, it is uploaded through the actual endpoint, and the reader token is obtained by
minting a grant and redeeming it — the same three calls a real owner and reader make.

Two homes, and keeping them apart is the point: the OWNER's `$OPYT_HOME` holds the live store the
export is projected from, and the SERVICE's holds the peers registry, `service.db`, and
`exports/`. A test that confused them would prove nothing about a hop.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pipeline.kb import schema
from pipeline.sharing import export
from service import line, tokens
from service.app import app
from tests.kb.test_export import _corpus

OWNER = "david"
LABEL = "David's KB"


@pytest.fixture()
def export_file(tmp_path, monkeypatch):
    """A real export, built by `build_export` from a seeded live store under its own home. The
    corpus is `tests/kb/test_export.py`'s, so the queries that file exercises mean the same thing
    here."""
    live = tmp_path / "live"
    monkeypatch.setenv("OPYT_HOME", str(live))
    conn = schema.connect()
    _corpus(conn)
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
    owner_token = tokens.mint_token(OWNER, "owner", label=LABEL)

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


def agent_key(svc, label: str | None = "Instinct") -> str:
    """An agent token for `svc`'s library, got the way a real agent gets one: the owner mints an
    agent invite and the agent redeems it."""
    code = svc.client.post("/v1/grant", json={"label": label, "agent": True},
                           headers=svc.owner_hdr).json()["code"]
    return svc.client.post("/v1/redeem", json={"code": code}).json()["token"]


class Mac:
    """A Mac on the line, in a thread: it asks for calls through the real route with the
    library's owner token, and posts `answer(call)` for each one it is handed. With no `answer`
    it takes each call and never answers, like a Mac that stopped mid-call. `taken` is every call
    it was handed. Entering waits until the service has seen it ask."""

    def __init__(self, svc, answer=None, *, ok: bool = True):
        self.owner, self.headers, self.answer, self.ok = svc.owner, svc.owner_hdr, answer, ok
        self.taken: list[dict] = []
        self._off = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        client = TestClient(app)
        while not self._off.is_set():
            call = client.get("/v1/line/next", params={"wait": 0.2},
                              headers=self.headers).json()["call"]
            if call is None:
                continue
            self.taken.append(call)
            if self.answer is not None:
                client.post(f"/v1/line/{call['id']}/answer", headers=self.headers,
                            json={"ok": self.ok, "answer": self.answer(call)})

    def __enter__(self) -> "Mac":
        self._thread.start()
        deadline = time.monotonic() + 5
        while not line.held(self.owner):
            assert time.monotonic() < deadline, "the Mac never asked"
            time.sleep(0.01)
        return self

    def __exit__(self, *exc) -> None:
        self._off.set()
        self._thread.join(timeout=5)
