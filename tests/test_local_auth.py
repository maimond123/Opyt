"""
tests/test_local_auth.py

The one-shot loopback credential channel. These tests pin the SECURITY properties, not the
HTML: loopback-only binding, an unguessable nonce, a bad nonce that does not kill the capture,
and a hard timeout. Everything here runs against real sockets on ephemeral ports — the module
is 60 lines of socket handling and stubbing it would test nothing.
"""

import threading
import urllib.error
import urllib.request

import pytest

from opyt_core import local_auth

# Every test here starts a server on 127.0.0.1 and talks to it. See the `loopback`
# marker in tests/conftest.py — a SCOPED exemption, not a blanket one.
pytestmark = pytest.mark.loopback


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode()


def test_captures_a_query_param_and_shuts_down():
    cap = local_auth.Capture(route="cb", timeout=10)
    with cap:
        url = f"{cap.url}?code=abc123"
        threading.Timer(0.1, lambda: _get(url)).start()
        got = cap.wait()
    assert got["params"]["code"] == "abc123"


def test_reachable_on_both_ipv4_and_ipv6_literals():
    """⚠️ macOS resolves `localhost` to ::1. An IPv4-only bind makes the browser redirect hang
    with no error and no log line — the exact failure this dual bind exists to prevent."""
    cap = local_auth.Capture(route="cb", timeout=10)
    with cap:
        for host in ("127.0.0.1", "[::1]"):
            status, _ = _get(f"http://{host}:{cap.port}{cap.path}?code=x")
            assert status == 200


def test_wrong_nonce_is_404_and_does_not_shut_down():
    cap = local_auth.Capture(route="cb", timeout=10)
    with cap:
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(f"http://127.0.0.1:{cap.port}/cb/not-the-nonce?code=x")
        assert e.value.code == 404
        # still alive: a real capture must still work
        threading.Timer(0.1, lambda: _get(f"{cap.url}?code=real")).start()
        assert cap.wait()["params"]["code"] == "real"


def test_timeout_returns_none_rather_than_hanging():
    cap = local_auth.Capture(route="cb", timeout=1)
    with cap:
        assert cap.wait() is None


def test_never_binds_a_routable_address():
    """A wildcard bind would put a credential form on the LAN."""
    cap = local_auth.Capture(route="cb", timeout=5)
    with cap:
        assert {s.server_address[0] for s in cap._servers} == {"127.0.0.1", "::1"}


def test_post_body_is_captured_for_the_paste_form():
    cap = local_auth.Capture(route="paste", timeout=10)
    with cap:
        def _post():
            urllib.request.urlopen(cap.url, data=b"value=sk-test-123", timeout=5)
        threading.Timer(0.1, _post).start()
        assert cap.wait()["form"]["value"] == "sk-test-123"


def test_the_loopback_exemption_is_scoped_not_blanket():
    """⚠️ THIS TEST GUARDS THE GUARD. `loopback` lifts the live-network block ONLY for
    127.0.0.1/::1/localhost. If it ever degrades into a second `live_llm`, this module would
    silently gain permission to reach the internet — so prove a routable address still fails."""
    with pytest.raises(BaseException) as e:      # pytest.fail raises an OutcomeException
        urllib.request.urlopen("https://example.com", timeout=5)
    assert "LIVE NETWORK CALL" in str(e.value)
