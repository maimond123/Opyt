"""Pins for cdp's endpoint wait, its stdlib WebSocket client and the Page wrapper.

The WebSocket side runs over `socket.socketpair()`: `create_connection` hands the client one
end and the test plays DevTools on the other, so the bytes on the wire are what is asserted.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading

import pytest

from pipeline.platforms.browser import cdp

_URL = "ws://127.0.0.1:9222/devtools/browser/abc"
_ACCEPT = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n\r\n"


class _Proc:
    def __init__(self, code=None):
        self.returncode = code

    def poll(self):
        return self.returncode


def _frame(payload: bytes, opcode: int = 0x1, fin: bool = True) -> bytes:
    """A server frame: unmasked, as DevTools sends them."""
    b0 = (0x80 if fin else 0) | opcode
    n = len(payload)
    if n < 126:
        head = struct.pack("!BB", b0, n)
    elif n < 65536:
        head = struct.pack("!BBH", b0, 126, n)
    else:
        head = struct.pack("!BBQ", b0, 127, n)
    return head + payload


def _msg(obj: dict) -> bytes:
    return _frame(json.dumps(obj).encode())


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        assert chunk, "client closed early"
        out += chunk
    return out


def _read_client_frame(sock: socket.socket) -> tuple[int, int, int, bytes, bytes]:
    """(b0, b1, length-form byte, mask, unmasked payload) of one frame the client sent."""
    b0, b1 = _recv_exact(sock, 2)
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack("!H", _recv_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack("!Q", _recv_exact(sock, 8))[0]
    mask = _recv_exact(sock, 4)
    raw = _recv_exact(sock, n)
    return b0, b1, b1 & 0x7F, mask, bytes(b ^ mask[i % 4] for i, b in enumerate(raw))


def _read_request(sock: socket.socket) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(4096)
    return buf


@pytest.fixture
def wire(monkeypatch):
    """(connect, peer, calls): `connect(prefill)` builds a `_Socket` whose DevTools side is
    `peer`, with `prefill` already waiting on the wire."""
    client, peer = socket.socketpair()
    calls = []

    def _create_connection(address, timeout=None):
        calls.append((address, timeout))
        return client

    monkeypatch.setattr(cdp.socket, "create_connection", _create_connection)

    def connect(prefill: bytes = _ACCEPT, url: str = _URL):
        peer.sendall(prefill)
        s = cdp._Socket(url)
        _read_request(peer)
        return s

    yield connect, peer, calls
    client.close()
    peer.close()


# ── _wait_for_endpoint ───────────────────────────────────────────────────────────

def test_wait_for_endpoint_returns_the_ws_url_from_the_port_file(tmp_path):
    (tmp_path / cdp._PORT_FILE).write_text("9222\n/devtools/browser/abc\nignored\n")
    got = cdp._wait_for_endpoint(tmp_path, _Proc())
    assert got == "ws://127.0.0.1:9222/devtools/browser/abc"


def test_wait_for_endpoint_polls_through_missing_and_partial_files(monkeypatch, tmp_path):
    path = tmp_path / cdp._PORT_FILE
    contents = iter(["9222\n", "9222\n/devtools/browser/late\n"])
    sleeps = []

    def _sleep(seconds):
        sleeps.append(seconds)
        path.write_text(next(contents))

    monkeypatch.setattr(cdp.time, "sleep", _sleep)
    got = cdp._wait_for_endpoint(tmp_path, _Proc())
    assert got == "ws://127.0.0.1:9222/devtools/browser/late"
    assert sleeps == [0.1, 0.1]


def test_wait_for_endpoint_raises_when_the_browser_exits_even_with_a_port_file(tmp_path):
    (tmp_path / cdp._PORT_FILE).write_text("9222\n/devtools/browser/abc\n")
    with pytest.raises(cdp.CDPError) as ei:
        cdp._wait_for_endpoint(tmp_path, _Proc(code=1))
    assert str(ei.value) == "browser exited (code 1) before opening DevTools"


def test_wait_for_endpoint_times_out_on_an_unparseable_port(monkeypatch, tmp_path):
    (tmp_path / cdp._PORT_FILE).write_text("not-a-port\n/devtools/browser/abc\n")
    clock = [1000.0]
    monkeypatch.setattr(cdp.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cdp.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    with pytest.raises(cdp.CDPError) as ei:
        cdp._wait_for_endpoint(tmp_path, _Proc(), timeout=2.0)
    assert str(ei.value) == "browser did not publish DevToolsActivePort within 2s"


def test_wait_for_endpoint_default_timeout_is_thirty_seconds(monkeypatch, tmp_path):
    clock = [0.0]
    monkeypatch.setattr(cdp.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cdp.time, "sleep", lambda _s: clock.__setitem__(0, clock[0] + 1.0))
    with pytest.raises(cdp.CDPError, match="within 30s"):
        cdp._wait_for_endpoint(tmp_path, _Proc())


# ── _Socket handshake ────────────────────────────────────────────────────────────

def test_socket_handshake_sends_the_upgrade_request(monkeypatch):
    client, peer = socket.socketpair()
    calls = []
    monkeypatch.setattr(cdp.socket, "create_connection",
                        lambda address, timeout=None: calls.append((address, timeout)) or client)
    peer.sendall(_ACCEPT)
    try:
        cdp._Socket(_URL)
        request = _read_request(peer)
    finally:
        client.close()
        peer.close()

    assert calls == [(("127.0.0.1", 9222), 30.0)]
    head, _, body = request.partition(b"\r\n\r\n")
    lines = head.decode().split("\r\n")
    assert body == b""
    assert lines[0] == "GET /devtools/browser/abc HTTP/1.1"
    assert lines[1] == "Host: 127.0.0.1:9222"
    assert lines[2:4] == ["Upgrade: websocket", "Connection: Upgrade"]
    assert lines[4].startswith("Sec-WebSocket-Key: ")
    assert len(base64.b64decode(lines[4].split(": ", 1)[1])) == 16
    assert lines[5] == "Sec-WebSocket-Version: 13"
    assert len(lines) == 6


def test_socket_passes_a_custom_timeout_to_the_connection(monkeypatch):
    client, peer = socket.socketpair()
    calls = []
    monkeypatch.setattr(cdp.socket, "create_connection",
                        lambda address, timeout=None: calls.append((address, timeout)) or client)
    peer.sendall(_ACCEPT)
    try:
        cdp._Socket("ws://localhost:1234/x", timeout=5.0)
    finally:
        client.close()
        peer.close()
    assert calls == [(("localhost", 1234), 5.0)]


def test_socket_keeps_bytes_that_arrived_with_the_handshake(wire):
    connect, _peer, _calls = wire
    s = connect(_ACCEPT + _msg({"method": "Early.event"}))
    assert json.loads(s._recv_message()) == {"method": "Early.event"}


def test_socket_refused_upgrade_raises_with_the_status_line(wire):
    connect, _peer, _calls = wire
    with pytest.raises(cdp.CDPError) as ei:
        connect(b"HTTP/1.1 403 Forbidden\r\nX: y\r\n\r\n")
    assert str(ei.value) == "DevTools refused the upgrade: b'HTTP/1.1 403 Forbidden'"


def test_socket_closed_during_handshake_raises(monkeypatch):
    client, peer = socket.socketpair()
    monkeypatch.setattr(cdp.socket, "create_connection", lambda _address, **_: client)
    peer.sendall(b"HTTP/1.1 101 Switching")
    peer.shutdown(socket.SHUT_WR)
    try:
        with pytest.raises(cdp.CDPError) as ei:
            cdp._Socket(_URL)
    finally:
        client.close()
        peer.close()
    assert str(ei.value) == "DevTools closed the connection during the handshake"


# ── frames out ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size, length_byte", [(5, 5), (125, 125), (126, 126), (65535, 126),
                                               (65536, 127)])
def test_send_frame_is_a_masked_final_frame_in_all_three_length_forms(wire, size, length_byte):
    connect, peer, _calls = wire
    s = connect()
    payload = bytes(i % 251 for i in range(size))
    sender = threading.Thread(target=s._send_frame, args=(payload,))
    sender.start()
    b0, b1, form, mask, got = _read_client_frame(peer)
    sender.join(5)
    assert b0 == 0x81
    assert b1 & 0x80
    assert form == length_byte
    assert len(mask) == 4
    assert got == payload


def test_send_frame_carries_the_given_opcode(wire):
    connect, peer, _calls = wire
    s = connect()
    s._send_frame(b"bye", opcode=0x8)
    b0, _b1, _form, _mask, got = _read_client_frame(peer)
    assert b0 == 0x88 and got == b"bye"


# ── frames in ────────────────────────────────────────────────────────────────────

def test_recv_message_reassembles_continuation_frames(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_frame(b"ab", fin=False) + _frame(b"cd", opcode=0x0, fin=False)
                 + _frame(b"ef", opcode=0x0))
    assert s._recv_message() == b"abcdef"


@pytest.mark.parametrize("size", [126, 65536 + 10])
def test_recv_message_reads_extended_lengths(wire, size):
    connect, peer, _calls = wire
    s = connect()
    payload = bytes(i % 7 for i in range(size))
    writer = threading.Thread(target=peer.sendall, args=(_frame(payload),))
    writer.start()
    got = s._recv_message()
    writer.join(5)
    assert got == payload


def test_recv_message_answers_a_ping_with_a_masked_pong_and_keeps_waiting(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_frame(b"hi", opcode=0x9) + _frame(b"after"))
    assert s._recv_message() == b"after"
    b0, b1, _form, _mask, got = _read_client_frame(peer)
    assert b0 == 0x8A and b1 & 0x80 and got == b"hi"


def test_recv_message_skips_a_pong(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_frame(b"x", opcode=0xA) + _frame(b"real"))
    assert s._recv_message() == b"real"


def test_recv_message_raises_on_a_close_frame(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_frame(b"", opcode=0x8))
    with pytest.raises(cdp.CDPError) as ei:
        s._recv_message()
    assert str(ei.value) == "DevTools sent a close frame"


def test_recv_message_raises_when_the_peer_closes_mid_frame(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_frame(b"truncated")[:5])
    peer.shutdown(socket.SHUT_WR)
    with pytest.raises(cdp.CDPError) as ei:
        s._recv_message()
    assert str(ei.value) == "DevTools closed the connection"


# ── call ─────────────────────────────────────────────────────────────────────────

def test_call_sends_numbered_commands_and_returns_the_result(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_msg({"id": 1, "result": {"product": "Chrome"}}) + _msg({"id": 2}))
    assert s.call("Browser.getVersion") == {"product": "Chrome"}
    assert s.call("Target.attachToTarget", {"targetId": "T"}, session_id="S1") == {}
    first = json.loads(_read_client_frame(peer)[4])
    second = json.loads(_read_client_frame(peer)[4])
    assert first == {"id": 1, "method": "Browser.getVersion", "params": {}}
    assert second == {"id": 2, "method": "Target.attachToTarget",
                      "params": {"targetId": "T"}, "sessionId": "S1"}


def test_call_skips_other_replies_and_buffers_events(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_msg({"id": 99, "result": {"stale": True}})
                 + _msg({"method": "Page.loadEventFired", "sessionId": "S1"})
                 + _msg({"id": 1, "result": {"ok": 1}}))
    assert s.call("Page.navigate", {"url": "https://x.com"}) == {"ok": 1}
    assert s.wait_for_event("Page.loadEventFired", timeout=0) == {
        "method": "Page.loadEventFired", "sessionId": "S1"}


def test_call_raises_with_the_method_and_the_error_json(wire):
    connect, peer, _calls = wire
    s = connect()
    error = {"code": -32601, "message": "'Nope.method' wasn't found"}
    peer.sendall(_msg({"id": 1, "error": error}))
    with pytest.raises(cdp.CDPError) as ei:
        s.call("Nope.method")
    assert str(ei.value) == f"Nope.method failed: {json.dumps(error)}"


# ── wait_for_event ───────────────────────────────────────────────────────────────

def test_wait_for_event_returns_a_buffered_event_without_reading(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_msg({"method": "A", "sessionId": "S1"})
                 + _msg({"method": "A", "sessionId": "S2"})
                 + _msg({"id": 1, "result": {}}))
    s.call("Noop")
    assert s.wait_for_event("A", session_id="S2", timeout=0) == {"method": "A", "sessionId": "S2"}
    assert s.wait_for_event("A", timeout=0) == {"method": "A", "sessionId": "S1"}
    assert s.wait_for_event("A", timeout=0) is None


def test_wait_for_event_buffers_non_matching_events_and_drops_replies(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_msg({"method": "Other", "params": {"n": 1}})
                 + _msg({"id": 7, "result": {}})
                 + _msg({"method": "Want", "sessionId": "S9"})
                 + _msg({"method": "Want", "sessionId": "S1", "params": {"n": 2}}))
    got = s.wait_for_event("Want", session_id="S1", timeout=5)
    assert got == {"method": "Want", "sessionId": "S1", "params": {"n": 2}}
    assert s.wait_for_event("Want", timeout=0) == {"method": "Want", "sessionId": "S9"}
    assert s.wait_for_event("Other", timeout=0) == {"method": "Other", "params": {"n": 1}}


def test_wait_for_event_returns_none_on_timeout_and_restores_the_socket_timeout(wire):
    connect, _peer, _calls = wire
    s = connect()
    before = s._sock.gettimeout()
    assert s.wait_for_event("Never.fires", timeout=0.05) is None
    assert s._sock.gettimeout() == before


def test_wait_for_event_default_timeout_is_the_call_timeout(wire, monkeypatch):
    connect, _peer, _calls = wire
    s = connect()
    clock = iter([100.0, 100.0 + cdp._CALL_TIMEOUT])
    monkeypatch.setattr(cdp.time, "monotonic", lambda: next(clock))
    assert s.wait_for_event("Never.fires") is None


# ── close ────────────────────────────────────────────────────────────────────────

def test_close_closes_the_connection(wire):
    connect, peer, _calls = wire
    s = connect()
    s.close()
    peer.settimeout(1)
    assert peer.recv(1) == b""


def test_close_swallows_os_error(monkeypatch):
    client, peer = socket.socketpair()

    class _Conn:
        def __getattr__(self, name):
            return getattr(client, name)

        def close(self):
            raise OSError("already gone")

    monkeypatch.setattr(cdp.socket, "create_connection", lambda _address, **_: _Conn())
    peer.sendall(_ACCEPT)
    try:
        s = cdp._Socket(_URL)
        s.close()
    finally:
        client.close()
        peer.close()


# ── get_cookies and Page ─────────────────────────────────────────────────────────

class _Browser:
    def __init__(self, reply=None):
        self.calls = []
        self.waits = []
        self.reply = reply if reply is not None else {}

    def call(self, method, params=None, *, session_id=None):
        self.calls.append((method, params, session_id))
        return self.reply

    def wait_for_event(self, method, *, session_id=None, timeout=None):
        self.waits.append((method, session_id, timeout))
        return {"method": method}


def test_get_cookies_asks_the_browser_target_for_storage_cookies():
    cookies = [{"name": "auth_token", "value": "t", "domain": ".x.com"}]
    browser = _Browser({"cookies": cookies})
    assert cdp.get_cookies(browser) == cookies
    assert browser.calls == [("Storage.getCookies", None, None)]


def test_get_cookies_raises_key_error_when_the_reply_has_no_cookies():
    with pytest.raises(KeyError):
        cdp.get_cookies(_Browser({}))


def test_get_cookies_over_the_wire(wire):
    connect, peer, _calls = wire
    s = connect()
    peer.sendall(_msg({"id": 1, "result": {"cookies": [{"name": "a", "value": "1"}]}}))
    assert cdp.get_cookies(s) == [{"name": "a", "value": "1"}]
    assert json.loads(_read_client_frame(peer)[4]) == {
        "id": 1, "method": "Storage.getCookies", "params": {}}


def test_page_call_and_wait_carry_the_page_session():
    browser = _Browser({"ok": True})
    page = cdp.Page(browser, "S1")
    assert page.call("Runtime.evaluate", {"expression": "1"}) == {"ok": True}
    assert page.call("Page.enable") == {"ok": True}
    assert page.wait_for_event("Page.loadEventFired") == {"method": "Page.loadEventFired"}
    assert page.wait_for_event("Page.screencastFrame", timeout=2.5) == {
        "method": "Page.screencastFrame"}
    assert browser.calls == [("Runtime.evaluate", {"expression": "1"}, "S1"),
                             ("Page.enable", None, "S1")]
    assert browser.waits == [("Page.loadEventFired", "S1", 30.0),
                             ("Page.screencastFrame", "S1", 2.5)]


def test_page_over_the_wire_filters_events_by_its_session(wire):
    connect, peer, _calls = wire
    s = connect()
    page = cdp.Page(s, "S1")
    peer.sendall(_msg({"method": "Page.loadEventFired", "sessionId": "S2"})
                 + _msg({"id": 1, "result": {}})
                 + _msg({"method": "Page.loadEventFired", "sessionId": "S1"}))
    assert page.call("Page.enable") == {}
    assert json.loads(_read_client_frame(peer)[4])["sessionId"] == "S1"
    assert page.wait_for_event("Page.loadEventFired", timeout=5) == {
        "method": "Page.loadEventFired", "sessionId": "S1"}
    assert s.wait_for_event("Page.loadEventFired", timeout=0) == {
        "method": "Page.loadEventFired", "sessionId": "S2"}
