"""Measure input-to-frame latency of a login desktop, one x11vnc flag at a time.

Runs ON the hosted box against a SCRATCH Xvfb display and a SCRATCH Chrome profile. It never
attaches to the live desktop and never opens a debugging port: the login desktop's Chrome must
stay non-CDP or Google refuses SSO (docs/plans/2026-09-07-hosted-x-google-sso-block-context.md).

The number is the wall time from an RFB KeyEvent leaving the client to the FramebufferUpdate
carrying the resulting character. It speaks to "slow with accepting our edits" in a way that
throughput and bandwidth do not.

Faithfulness to the real client, which is the vendored noVNC at gateway/static/novnc:
  * the same pixel format (32bpp, depth 24, little-endian, true colour, shifts 0/8/16),
  * the same encoding preference (Tight) and the same quality 6 / compression 2,
  * one incremental FramebufferUpdateRequest outstanding at all times.
Rect payloads are skipped, not decoded -- the timing needs the message boundary, not the pixels.

Two regimes, and `stray_updates` in the report tells them apart. Below roughly 130 keys the typed
text fits the window and each key dirties one glyph (~650 B, strays near zero). Past that the
field scrolls sideways and every key repaints a whole line (~10.8 kB, strays in the dozens). Both
are real -- the second is closer to a page with images -- but only compare arms run at the same
key count.

The workload page hides the text caret. A blinking caret is a framebuffer change no keystroke
caused, so hiding it makes every update attributable; the silence check before each arm proves
the display is otherwise still.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from dataclasses import dataclass, field

# Long enough that a caret blink (500 ms in Chrome) would be seen several times over.
_QUIET_SECONDS = 2.0
# A person types at roughly this rate. Spacing the keystrokes keeps each one its own update
# instead of measuring a saturated pipeline.
_KEY_GAP_SECONDS = 0.15
# Long enough to catch a second update for the same keystroke, short enough to stay inside the
# gap between keystrokes.
_SWEEP_SECONDS = 0.05
_READ_TIMEOUT_SECONDS = 10.0
_CHROME_PAINT_SECONDS = 6.0

_ENC_RAW = 0
_ENC_COPYRECT = 1
_ENC_TIGHT = 7
_ENC_LAST_RECT = -224
# Requested so the harness can report whether the real noVNC would be offered continuous
# updates. It is never enabled -- the server only pushes unrequested frames if a client asks.
_PSEUDO_CONTINUOUS_UPDATES = -313
_PSEUDO_LAST_RECT = -224
_PSEUDO_QUALITY_6 = -26
_PSEUDO_COMPRESS_2 = -254

_TIGHT_MIN_TO_COMPRESS = 12


class HarnessError(RuntimeError):
    """Anything that would make a number wrong. Never recovered from -- the run stops."""


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, so every reported number is an observation that happened."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-pct * len(ordered) // 100))))
    return ordered[rank - 1]


def _write_png(path: str, width: int, height: int, scanlines: bytes) -> None:
    """Minimal PNG writer -- the box has no image library and needs none for this."""
    import zlib

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", header))
        handle.write(chunk(b"IDAT", zlib.compress(scanlines, 6)))
        handle.write(chunk(b"IEND", b""))


class RfbClient:
    """An RFB 3.8 client that times updates and skips their pixels."""

    def __init__(self, port: int) -> None:
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=_READ_TIMEOUT_SECONDS)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.width = 0
        self.height = 0
        self.server_offers_continuous_updates = False
        self._handshake()

    def _recv(self, count: int) -> bytes:
        chunks = []
        remaining = count
        while remaining:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise HarnessError("the VNC server closed the connection mid-message")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _handshake(self) -> None:
        version = self._recv(12)
        if not version.startswith(b"RFB 003."):
            raise HarnessError(f"unexpected protocol version {version!r}")
        self._sock.sendall(b"RFB 003.008\n")
        count = self._recv(1)[0]
        if count == 0:
            reason_len = struct.unpack(">I", self._recv(4))[0]
            raise HarnessError(f"server refused: {self._recv(reason_len).decode()}")
        types = set(self._recv(count))
        if 1 not in types:
            raise HarnessError(f"server wants authentication; offered types {sorted(types)}")
        self._sock.sendall(bytes([1]))
        result = struct.unpack(">I", self._recv(4))[0]
        if result != 0:
            raise HarnessError("security handshake failed")
        self._sock.sendall(bytes([1]))  # ClientInit, shared
        # ServerInit is 24 bytes: 2+2 size, 16 pixel format, then the name length.
        header = self._recv(24)
        self.width, self.height = struct.unpack(">HH", header[:4])
        name_len = struct.unpack(">I", header[20:24])[0]
        self._recv(name_len)
        self._set_pixel_format()
        self._set_encodings()

    def _set_pixel_format(self) -> None:
        self._sock.sendall(struct.pack(
            ">BBBBBBBBHHHBBBBBB", 0, 0, 0, 0, 32, 24, 0, 1, 255, 255, 255, 0, 8, 16, 0, 0, 0))

    def _set_encodings(self) -> None:
        encodings = [_ENC_COPYRECT, _ENC_TIGHT, _ENC_RAW, _PSEUDO_QUALITY_6, _PSEUDO_COMPRESS_2,
                     _PSEUDO_LAST_RECT, _PSEUDO_CONTINUOUS_UPDATES]
        body = struct.pack(">BBH", 2, 0, len(encodings))
        body += b"".join(struct.pack(">i", enc) for enc in encodings)
        self._sock.sendall(body)

    def request_update(self) -> None:
        self._sock.sendall(struct.pack(">BBHHHH", 3, 1, 0, 0, self.width, self.height))

    def send_key(self, keysym: int) -> None:
        self._sock.sendall(struct.pack(">BBHI", 4, 1, 0, keysym))
        self._sock.sendall(struct.pack(">BBHI", 4, 0, 0, keysym))

    def click(self, x: int, y: int) -> None:
        self._sock.sendall(struct.pack(">BBHH", 5, 1, x, y))
        self._sock.sendall(struct.pack(">BBHH", 5, 0, x, y))

    def read_update(self, timeout: float) -> tuple[int, int]:
        """Block for one FramebufferUpdate; return (bytes consumed, rectangles).

        Other server messages are consumed and ignored -- only a framebuffer update returns.
        """
        deadline = time.monotonic() + timeout
        while True:
            self._sock.settimeout(max(0.001, deadline - time.monotonic()))
            kind = self._recv(1)[0]
            if kind == 0:
                return self._read_framebuffer_update()
            self._skip_other_message(kind)

    def _skip_other_message(self, kind: int) -> None:
        if kind == 2:  # Bell
            return
        if kind == 3:  # ServerCutText
            self._recv(3)
            length = struct.unpack(">I", self._recv(4))[0]
            self._recv(length)
            return
        if kind == 150:  # EndOfContinuousUpdates
            self.server_offers_continuous_updates = True
            return
        raise HarnessError(f"unhandled server message type {kind}")

    def _read_framebuffer_update(self) -> tuple[int, int]:
        consumed = 4
        header = self._recv(3)
        count = struct.unpack(">H", header[1:])[0]
        rectangles = 0
        for _ in range(count):
            x, y, w, h, encoding = struct.unpack(">HHHHi", self._recv(12))
            consumed += 12
            if encoding == _ENC_LAST_RECT:
                break
            rectangles += 1
            if encoding == _ENC_RAW:
                consumed += self._skip(w * h * 4)
            elif encoding == _ENC_COPYRECT:
                consumed += self._skip(4)
            elif encoding == _ENC_TIGHT:
                consumed += self._skip_tight(w, h)
            else:
                raise HarnessError(f"unhandled rect encoding {encoding} at {x},{y} {w}x{h}")
        return consumed, rectangles

    def _skip(self, count: int) -> int:
        if count:
            self._recv(count)
        return count

    def _read_compact_length(self) -> tuple[int, int]:
        """Tight's 1-3 byte length: seven bits each, high bit continues."""
        value = 0
        for index in range(3):
            byte = self._recv(1)[0]
            value |= (byte & 0x7F) << (7 * index)
            if not byte & 0x80:
                return value, index + 1
        return value, 3

    def _skip_tight(self, w: int, h: int) -> int:
        control = self._recv(1)[0]
        consumed = 1
        mode = control & 0xF0
        if mode == 0x80:  # fill: one TPIXEL, three bytes at depth 24
            return consumed + self._skip(3)
        if mode == 0x90:  # jpeg
            length, header_bytes = self._read_compact_length()
            return consumed + header_bytes + self._skip(length)
        if mode >= 0xA0:
            raise HarnessError(f"unhandled tight compression control 0x{control:02x}")
        filter_id = 0
        if control & 0x40:
            filter_id = self._recv(1)[0]
            consumed += 1
        if filter_id == 1:  # palette
            colours = self._recv(1)[0] + 1
            consumed += 1 + self._skip(colours * 3)
            row = (w + 7) // 8 if colours <= 2 else w
            raw_length = row * h
        elif filter_id == 2:  # gradient
            raw_length = w * h * 3
        elif filter_id == 0:  # copy
            raw_length = w * h * 3
        else:
            raise HarnessError(f"unhandled tight filter {filter_id}")
        if raw_length < _TIGHT_MIN_TO_COMPRESS:
            return consumed + self._skip(raw_length)
        length, header_bytes = self._read_compact_length()
        return consumed + header_bytes + self._skip(length)

    def screenshot(self, path: str) -> None:
        """Write the whole framebuffer to a PNG. Use `capture()`, not this, while measuring.

        The only way to see this desktop. CDP is banned on it -- a debugging port sets
        navigator.webdriver and Google then refuses SSO -- so the pixels have to come back over
        the same RFB connection a user's browser uses.

        MEASURED 2026-09-09: this ends the connection's usefulness as an instrument. The full
        update it asks for makes x11vnc drop the incremental request that was already
        outstanding, and every later keystroke then waits forever for a frame nobody asked for.
        """
        self._sock.sendall(struct.pack(">BBH", 2, 0, 1) + struct.pack(">i", _ENC_RAW))
        self._sock.sendall(struct.pack(">BBHHHH", 3, 0, 0, 0, self.width, self.height))
        rows = bytearray(b"\x00" * (self.height * (self.width * 3 + 1)))
        self._sock.settimeout(_READ_TIMEOUT_SECONDS)
        while True:
            kind = self._recv(1)[0]
            if kind == 0:
                break
            self._skip_other_message(kind)
        count = struct.unpack(">H", self._recv(3)[1:])[0]
        for _ in range(count):
            x, y, w, h, encoding = struct.unpack(">HHHHi", self._recv(12))
            if encoding != _ENC_RAW:
                raise HarnessError(f"screenshot got encoding {encoding}, expected raw")
            pixels = self._recv(w * h * 4)
            # Slice-swizzle BGRX to RGB; a per-pixel Python loop over a megapixel is minutes.
            rgb = bytearray(w * h * 3)
            rgb[0::3] = pixels[2::4]
            rgb[1::3] = pixels[1::4]
            rgb[2::3] = pixels[0::4]
            for row in range(h):
                base = (y + row) * (self.width * 3 + 1) + 1 + x * 3
                rows[base:base + w * 3] = rgb[row * w * 3:(row + 1) * w * 3]
        _write_png(path, self.width, self.height, bytes(rows))

    def is_quiet(self, seconds: float) -> bool:
        """True when no update arrives for `seconds` with a request already outstanding."""
        try:
            self.read_update(seconds)
        except (socket.timeout, TimeoutError):
            return True
        return False

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>rfb latency workload</title>
<style>
  html, body { margin: 0; height: 100%; background: #ffffff; }
  /* A blinking caret is a framebuffer change no keystroke caused. Hiding it makes every
     update in this measurement attributable to a key. */
  textarea {
    caret-color: transparent; box-sizing: border-box; width: 100%; height: 100%;
    border: 0; outline: none; resize: none; padding: 24px;
    font: 15px/1.5 monospace; color: #111111; background: #ffffff;
    /* No wrapping: a wrap repaints a whole line, and that outlier would become the p95
       instead of a keystroke. 100 keys at ~9 px each stay well inside the window's width. */
    white-space: pre;
  }
</style>
<textarea id="t" autofocus spellcheck="false"></textarea>
<script>
  const box = document.getElementById('t');
  box.focus();
  box.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') { return; }
    event.preventDefault();
    fetch('/typed?value=' + encodeURIComponent(box.value))
      .then(() => { location.reload(); });
  });
</script>
"""

# Letters only, deliberately. A space typed at the end of the text moves an invisible caret and
# changes no pixels, so there is no frame for it to arrive in -- measured 2026-09-09, the run
# stalled forever on the space after "brown". A key that paints nothing has no latency to report.
_PHRASE = "thequickbrownfoxjumpsoverthelazydog"
_KEYSYM_RETURN = 0xFF0D


class _Workload(http.server.BaseHTTPRequestHandler):
    """Serves the typing page and receives back what actually got typed into it."""

    def do_GET(self) -> None:  # noqa: N802 - http.server's interface
        path = urllib.parse.urlparse(self.path)
        if path.path == "/typed":
            value = urllib.parse.parse_qs(path.query).get("value", [""])[0]
            self.server.typed.append(value)
            body = b"ok"
        else:
            body = _PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


def _serve_page() -> tuple[http.server.ThreadingHTTPServer, str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Workload)
    server.typed = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _start_display(size: str) -> tuple[str, subprocess.Popen]:
    """Start a scratch Xvfb the same way the login desktop does, and read back its number."""
    read_fd, write_fd = os.pipe()
    try:
        proc = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_fd), "-screen", "0", size, "-nolisten", "tcp"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            pass_fds=(write_fd,))
    finally:
        os.close(write_fd)
    number = os.read(read_fd, 32).decode().strip()
    os.close(read_fd)
    if not number:
        raise HarnessError("Xvfb did not report a display number")
    return f":{number}", proc


def _cpu_seconds(pid: int) -> float:
    with open(f"/proc/{pid}/stat", "rb") as handle:
        fields = handle.read().rsplit(b")", 1)[1].split()
    ticks = int(fields[11]) + int(fields[12])
    return ticks / os.sysconf("SC_CLK_TCK")


def capture(port: int, path: str) -> None:
    """Photograph the desktop over a connection that is thrown away afterwards.

    x11vnc runs `-shared`, so this costs one extra client for the length of one frame and
    leaves the measuring client's outstanding request untouched.
    """
    client = RfbClient(port)
    try:
        client.screenshot(path)
    finally:
        client.close()


@dataclass
class Arm:
    name: str
    extra: list[str] = field(default_factory=list)


@dataclass
class Result:
    arm: str
    argv: list[str]
    latencies_ms: list[float]
    update_bytes: list[int]
    rectangles: list[int]
    vnc_cpu_seconds: float
    first_frame_ms: float
    first_frame_bytes: int
    stray_updates: int
    typed_back: str
    continuous_updates_offered: bool

    def summary(self) -> dict:
        return {
            "arm": self.arm,
            "n": len(self.latencies_ms),
            "p50_ms": round(_percentile(self.latencies_ms, 50), 1),
            "p95_ms": round(_percentile(self.latencies_ms, 95), 1),
            "p99_ms": round(_percentile(self.latencies_ms, 99), 1),
            "min_ms": round(min(self.latencies_ms), 1),
            "max_ms": round(max(self.latencies_ms), 1),
            "mean_bytes_per_key": round(sum(self.update_bytes) / len(self.update_bytes)),
            "total_bytes": sum(self.update_bytes),
            "mean_rects_per_key": round(
                sum(self.rectangles) / len(self.rectangles), 2),
            # All the CPU x11vnc burns across the typing window -- polling included, not just
            # encoding -- divided by the keys typed in it. This is the budget a larger display
            # spends, so it is reported next to the latency it competes with.
            "vnc_cpu_ms_per_key_window": round(
                1000 * self.vnc_cpu_seconds / len(self.latencies_ms), 2),
            "keys_landed": len(self.typed_back),
            "stray_updates": self.stray_updates,
            "first_frame_ms": round(self.first_frame_ms, 1),
            "first_frame_bytes": self.first_frame_bytes,
        }


def _launch_vnc(display: str, port: int, extra: list[str], log_path: str
                ) -> tuple[subprocess.Popen, list[str]]:
    """Launch x11vnc with the login desktop's own argv plus this arm's flag."""
    argv = ["x11vnc", "-display", display, "-localhost", "-nopw", "-forever", "-shared",
            "-rfbport", str(port), *extra]
    log = open(log_path, "wb")
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise HarnessError(
                f"x11vnc exited {proc.returncode} for {' '.join(extra) or 'baseline'}; "
                f"see {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc, argv
        except OSError:
            time.sleep(0.1)
    raise HarnessError(f"x11vnc never accepted on {port}; see {log_path}")


def _assert_flag_accepted(log_path: str, extra: list[str]) -> None:
    """A flag x11vnc 0.9.16 does not know must fail here, not in front of a user.

    The login desktop turns any x11vnc failure into the refusal page, so an unknown flag would
    make every sign-in unreachable. Verify against the installed build, not the current manual.
    """
    with open(log_path, "rb") as handle:
        text = handle.read().decode("utf-8", "replace")
    for line in text.splitlines():
        lowered = line.lower()
        if "invalid" in lowered or "unrecognized" in lowered or "unknown option" in lowered:
            raise HarnessError(f"x11vnc rejected {extra}: {line}")


def run_arm(arm: Arm, display: str, keys: int, log_dir: str, typed_sink: list) -> Result:
    port = _free_port()
    log_path = os.path.join(log_dir, f"x11vnc-{arm.name}.log")
    vnc, argv = _launch_vnc(display, port, arm.extra, log_path)
    try:
        _assert_flag_accepted(log_path, arm.extra)
        client = RfbClient(port)
        try:
            client.click(client.width // 2, client.height // 2)
            # The first update is the whole screen. It is the one cost that does scale with the
            # display size, and a user pays it once, while the page is opening.
            first_start = time.perf_counter()
            client.request_update()
            first_bytes, _ = client.read_update(_READ_TIMEOUT_SECONDS)
            first_ms = 1000 * (time.perf_counter() - first_start)
            client.request_update()
            for _ in range(15):
                if client.is_quiet(_QUIET_SECONDS):
                    break
                client.request_update()
            else:
                raise HarnessError("the display never went quiet; something is animating")

            capture(port, os.path.join(log_dir, f"{arm.name}-before.png"))
            typed_sink.clear()
            cpu_before = _cpu_seconds(vnc.pid)
            strays = 0
            latencies: list[float] = []
            update_bytes: list[int] = []
            rectangles: list[int] = []
            for index in range(keys):
                # Start the clock on an empty socket with exactly one request outstanding, the
                # state a noVNC client sits in between keystrokes. A frame left in the buffer
                # would otherwise be read as this key's, and recorded at ~0 ms.
                while not client.is_quiet(_SWEEP_SECONDS):
                    strays += 1
                    client.request_update()
                keysym = ord(_PHRASE[index % len(_PHRASE)])
                start = time.perf_counter()
                client.send_key(keysym)
                try:
                    consumed, rects = client.read_update(_READ_TIMEOUT_SECONDS)
                except (socket.timeout, TimeoutError) as error:
                    waited = time.perf_counter() - start
                    shot = os.path.join(log_dir, f"{arm.name}-stalled.png")
                    try:
                        capture(port, shot)
                    except Exception:  # the socket may be unusable; the stall matters more
                        shot = "(no screenshot)"
                    raise HarnessError(
                        f"no frame for key {index} after {waited:.1f}s; screen at {shot}"
                    ) from error
                latencies.append(1000 * (time.perf_counter() - start))
                update_bytes.append(consumed)
                rectangles.append(rects)
                client.request_update()
                time.sleep(_KEY_GAP_SECONDS)
            cpu_after = _cpu_seconds(vnc.pid)
            capture(port, os.path.join(log_dir, f"{arm.name}-after.png"))

            client.send_key(_KEYSYM_RETURN)
            deadline = time.monotonic() + 10.0
            while not typed_sink and time.monotonic() < deadline:
                time.sleep(0.05)
            if not typed_sink:
                raise HarnessError("the page never reported what was typed into it")
            return Result(arm.name, argv, latencies, update_bytes, rectangles,
                          cpu_after - cpu_before, first_ms, first_bytes, strays,
                          typed_sink[-1],
                          client.server_offers_continuous_updates)
        finally:
            client.close()
    finally:
        vnc.terminate()
        try:
            vnc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            vnc.kill()
        time.sleep(1.0)


_ARMS = {
    "baseline": [],
    "threads": ["-threads"],
    "wait": ["-wait", "10"],
    "defer": ["-defer", "10"],
    "wait5": ["-wait", "5"],
    "tuned": ["-threads", "-wait", "10", "-defer", "10"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="baseline,threads,wait,defer,baseline")
    parser.add_argument("--keys", type=int, default=100)
    parser.add_argument("--size", default="1280x900x24")
    parser.add_argument("--json", default="")
    parser.add_argument("--chrome-window", default="",
                        help="WxH for Chrome's window; without it Chrome picks its own size, "
                             "which is what the login desktop does today")
    parser.add_argument("--shot-only", action="store_true",
                        help="start the workload, save one PNG of it, and exit")
    args = parser.parse_args()

    arms = []
    seen: dict[str, int] = {}
    for name in args.arms.split(","):
        if name not in _ARMS:
            raise SystemExit(f"unknown arm {name}; known: {', '.join(_ARMS)}")
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}#{seen[name]}"
        arms.append(Arm(label, list(_ARMS[name])))

    log_dir = tempfile.mkdtemp(prefix="rfb-latency-")
    profile = os.path.join(log_dir, "profile")
    server, url = _serve_page()
    display, xvfb = _start_display(args.size)
    window = []
    if args.chrome_window:
        width, height = args.chrome_window.split("x")
        window = [f"--window-size={width},{height}", "--window-position=0,0"]
    chrome = subprocess.Popen(
        ["google-chrome", f"--user-data-dir={profile}", "--no-first-run",
         "--no-default-browser-check", *window, url],
        env=dict(os.environ, DISPLAY=display),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: list[Result] = []
    try:
        time.sleep(_CHROME_PAINT_SECONDS)
        if chrome.poll() is not None:
            raise HarnessError(f"Chrome exited {chrome.returncode} before any measurement")
        if args.shot_only:
            port = _free_port()
            vnc, _ = _launch_vnc(display, port, [], os.path.join(log_dir, "x11vnc-shot.log"))
            client = RfbClient(port)
            client.click(client.width // 2, client.height // 2)
            client.request_update()
            client.is_quiet(_QUIET_SECONDS)
            for keysym in b"hello":
                client.send_key(keysym)
                time.sleep(0.2)
            time.sleep(1.0)
            capture(port, os.path.join(log_dir, "shot.png"))
            client.close()
            vnc.terminate()
            print(os.path.join(log_dir, "shot.png"))
            return 0
        for arm in arms:
            print(f"# arm {arm.name}: {' '.join(arm.extra) or '(no extra flags)'}",
                  file=sys.stderr, flush=True)
            results.append(run_arm(arm, display, args.keys, log_dir, server.typed))
            print(json.dumps(results[-1].summary()), file=sys.stderr, flush=True)
    finally:
        for proc in (chrome, xvfb):
            proc.terminate()
        server.shutdown()
        shutil.rmtree(profile, ignore_errors=True)

    report = {
        "display_size": args.size,
        "chrome_window": args.chrome_window or "chrome default",
        "keys_per_arm": args.keys,
        "continuous_updates_offered": any(r.continuous_updates_offered for r in results),
        "arms": [r.summary() for r in results],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as handle:
            handle.write(text + "\n")
    for result in results:
        expected = "".join(_PHRASE[i % len(_PHRASE)] for i in range(args.keys))
        if result.typed_back != expected:
            print(f"WARNING {result.arm}: the page received {result.typed_back!r}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
