"""Fixtures for the remote-peer transport — the real service, reached through the seam.

The service fixtures are reused verbatim from `tests/service/conftest.py`: a real export built
by `build_export`, uploaded through the actual endpoint, read with a token from a redeemed
grant. What this directory adds is the READER'S half — the same service registered as a peer
whose location is a URL, with `kb_remote`'s transport pointed at `TestClient` so every request
crosses the real FastAPI app in-process. Nothing about `kb_remote` is mocked; the one
substitution is the socket.

`publisher` is the OWNER's half of the same wiring — the store an export is projected from, the
settings naming the service, and `push`'s transport shimmed. Both `opyt-push` and the
`push_catchup` rail run through it, which is the point: they share one implementation, so they
have to share one fixture or the tests stop proving that.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from opyt_core import config, push
from pipeline.kb import peers

# Re-exported so this directory's tests run against the same served knowledge base the service
# tests do. pytest collects fixtures by name from a conftest's namespace, so an import IS a
# registration.
from tests.service.conftest import (  # noqa: F401
    emb,
    export_file,
    service_home,
    svc,
)

# The service both halves of this directory talk to. `TestClient` routes every host to the app,
# so the string only has to be a well-formed absolute URL.
URL = "https://svc.test"

# Deliberately NOT the owner's name. After R4 that is the whole point rather than an accident:
# the reader sends `as_kb` and the service labels the envelope with THIS name, so every `kb`
# field that comes back says "x" and not the owner's routing key. Before R4 the assertion here
# was the opposite one — that this name appeared nowhere in the envelope.
PEER = "x"


@pytest.fixture()
def remote(svc, monkeypatch):
    """The svc service, registered as remote peer 'x' with the reader's token, transport shimmed.

    The shim satisfies exactly the surface `kb_remote` uses — `.post`/`.get` — and `TestClient`
    accepts absolute URLs because its transport routes every host to the app. So `kb_remote`
    builds real URLs and real Authorization headers, and none of it is special-cased for tests."""
    from opyt_core import kb_remote

    def _verb(send):
        def call(url, **kw):
            kw.pop("timeout", None)   # meaningless in-process; TestClient deprecates it per-request
            return send(url, **kw)
        return call

    location = f"https://svc.test/v1/kb/{svc.owner}"
    peers.add(PEER, location, "A served KB", token=svc.reader_token)
    monkeypatch.setattr(kb_remote, "requests",
                        SimpleNamespace(post=_verb(svc.client.post), get=_verb(svc.client.get)))
    kb_remote._META_CACHE.clear()
    yield SimpleNamespace(name=PEER, location=location, svc=svc)
    kb_remote._META_CACHE.clear()


@pytest.fixture()
def publisher(svc, tmp_path, monkeypatch, emb):
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
