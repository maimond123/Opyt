"""Fixtures for the remote-peer transport — the real service, reached through the seam.

The service fixtures are reused verbatim from `tests/service/conftest.py`: a real export built
by `build_export`, uploaded through the actual endpoint, read with a token from a redeemed
grant. What this directory adds is the READER'S half — the same service registered as a peer
whose location is a URL, with `kb_remote`'s transport pointed at `TestClient` so every request
crosses the real FastAPI app in-process. Nothing about `kb_remote` is mocked; the one
substitution is the socket.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

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

# Deliberately NOT the owner's name: the envelope must cross verbatim, so the name the READER
# registered the peer under appears nowhere in it, and the fidelity assertion proves that.
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
