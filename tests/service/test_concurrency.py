"""Requests that overlap must not fail, and the reason this file exists is that they did.

`TestClient` drives one request at a time, so every other test in this directory exercises a
service whose worker threadpool has one hot thread and reuses it. That hid a real defect: the
first version passed a `sqlite3` connection into each handler through a FastAPI yield dependency,
and FastAPI resolves dependencies and runs sync handlers on DIFFERENT worker threads. `sqlite3`
refuses a connection used off the thread that created it, so the service returned 500 on the
first genuinely concurrent request and 200 on any sequential rate — measured against a real
uvicorn, not found by the suite.

So this runs a real server in a real process and overlaps real requests. The fix was to let
`service/store.py` own its connections the way `pipeline/kb/peers.py` does, which removes the
handoff rather than permitting it with `check_same_thread=False`.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from service import store

# The scoped exemption, not a blanket one: this module starts a server on 127.0.0.1 and talks to
# it, which is the case `loopback` exists for (see tests/conftest.py). A request to a routable
# address from here still fails the guard, and that is the property that matters. `live_llm` would
# have been a lie — nothing here is paid — and would have hidden the module from ordinary runs.
pytestmark = pytest.mark.loopback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def live_server(svc):
    """A real uvicorn against the same home `svc` set up. Bound to 127.0.0.1 explicitly rather
    than to "localhost", which resolves to ::1 on this platform and would hang."""
    port = _free_port()
    env = {**os.environ, "OPYT_HOME": str(svc.home), "PYTHONPATH": REPO}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if proc.poll() is not None:
                pytest.fail(f"the server died: {proc.stdout.read().decode()[-2000:]}")
            try:
                urllib.request.urlopen(f"{base}/v1/tokens", timeout=0.5)
            except urllib.error.HTTPError:
                break                       # 401 — it is up and answering
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail("the server never came up")
        yield base, proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _search(base, token, i):
    """The status and body, with an HTTP error turned into a status rather than an exception —
    so a partial failure reports "three of eight were 500" instead of one raised HTTPError that
    says nothing about the other seven."""
    req = urllib.request.Request(
        f"{base}/v1/kb/david/search", method="POST",
        data=json.dumps({"query": f"agent framework {i}", "k": 3}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {"hits": [], "detail": e.read().decode()[:200]}


def test_overlapping_reads_all_succeed(live_server, svc):
    """Eight at once against a threadpool that will hand them different threads. Every one must
    be a 200 — a single 500 here is the cross-thread bug back."""
    base, proc = live_server
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda i: _search(base, svc.reader_token, i), range(8)))
    assert [s for s, _ in results] == [200] * 8
    assert all(isinstance(b["hits"], list) for _, b in results)


def test_every_overlapping_read_is_counted_exactly_once(live_server, svc):
    """The meter writes to `service.db` from several worker threads at once. Concurrent writers
    are where a count silently goes wrong, so it is asserted under load rather than in isolation.
    """
    base, _ = live_server
    before = store.usage_total(svc.owner, store.token_hash(svc.reader_token))
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: _search(base, svc.reader_token, i), range(8)))
    after = store.usage_total(svc.owner, store.token_hash(svc.reader_token))
    assert after - before == 8
