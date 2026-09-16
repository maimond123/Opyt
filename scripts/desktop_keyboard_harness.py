"""What the hosted sign-in desktop does when a phone's keyboard takes half the screen.

Two questions no unit test can answer, because both are layout:

  1. Is the desktop still readable while the user types into it?
  2. Does the part of it they tapped end up in the strip the keyboard leaves?

This serves the REAL login page and the REAL module against a fake RFB server that answers the
handshake and nothing after it, so the framebuffer size is a parameter rather than a live
sign-in. The keyboard is simulated the only way it ever reaches the page: `--visual-height`
shrinking, which is exactly what `visualViewport` makes `trackSoftKeyboard` do.

    python scripts/desktop_keyboard_harness.py

Needs `playwright` and `websockets`. Design record:
docs/plans/2026-09-09-hosted-signin-from-a-phone.md.
"""
from __future__ import annotations

import asyncio
import http.server
import os
import socketserver
import struct
import sys
import threading
from pathlib import Path

import websockets
from playwright.sync_api import sync_playwright

GATEWAY = Path(__file__).resolve().parent.parent / "gateway"
sys.path.insert(0, str(GATEWAY.parent))

from gateway import app as gateway_app  # noqa: E402

HTTP_PORT, WS_PORT = 8897, 8898
# A 500x959 viewport at scale 2: what `clamp_desktop_size` builds for a phone today.
FB_WIDTH, FB_HEIGHT = 1000, 1918
# A 390x844 phone, and what is left of it once an iPhone keyboard and its strip are up.
PHONE_HEIGHT, KEYBOARD_UP = 844, 430

PAGE = gateway_app._login_html("x", "s", "c").replace(
    '"/login/session/s"', f"'ws://127.0.0.1:{WS_PORT}/stream'")

REPORT = """() => {
  const box = (sel) => { const e = document.querySelector(sel);
    if (!e) { return null; }
    const r = e.getBoundingClientRect();
    return `${Math.round(r.width)}x${Math.round(r.height)}`; };
  const stage = document.querySelector('.stage');
  return {stage: box('.stage'), screen: box('#screen'), canvas: box('#screen canvas'),
          scrolled: stage.scrollTop};
}"""


class Handler(http.server.SimpleHTTPRequestHandler):
    """The page, its static module, and the one POST it makes before connecting."""

    def do_GET(self):  # noqa: N802
        if self.path == "/login":
            self._send(PAGE.encode(), "text/html")
            return
        super().do_GET()

    def do_POST(self):  # noqa: N802
        self._send(b'{"status":"started","width":%d,"height":%d,"expires_in":600}'
                   % (FB_WIDTH, FB_HEIGHT), "application/json")

    def _send(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


async def rfb(socket):
    """Just enough RFB 3.8 for noVNC to say `connect`, and nothing at all after it."""
    await socket.send(b"RFB 003.008\n")
    await socket.recv()
    await socket.send(bytes([1, 1]))
    await socket.recv()
    await socket.send(struct.pack(">I", 0))
    await socket.recv()
    name = b"harness"
    await socket.send(struct.pack(">HH", FB_WIDTH, FB_HEIGHT)
                      + struct.pack(">BBBBHHHBBBxxx", 32, 24, 0, 1, 255, 255, 255, 16, 8, 0)
                      + struct.pack(">I", len(name)) + name)
    try:
        async for _message in socket:
            pass
    except Exception:
        pass


async def ws_server(ready):
    async with websockets.serve(rfb, "127.0.0.1", WS_PORT):
        ready.set()
        await asyncio.Future()


def main() -> None:
    os.chdir(GATEWAY)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", HTTP_PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    threading.Thread(target=lambda: loop.run_until_complete(ws_server(ready)),
                     daemon=True).start()
    ready.wait(5)

    def keyboard(page, height):
        page.evaluate("(h) => document.documentElement.style.setProperty("
                      "'--visual-height', h + 'px')", height)
        page.wait_for_timeout(900)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 390, "height": PHONE_HEIGHT},
                                   device_scale_factor=3, is_mobile=True,
                                   has_touch=True).new_page()
        page.on("pageerror", lambda error: print(f"  [pageerror] {error}"))
        page.goto(f"http://127.0.0.1:{HTTP_PORT}/login")
        page.wait_for_selector("#screen canvas", timeout=10000)
        page.wait_for_timeout(600)
        before = page.evaluate(REPORT)
        print(f"1. keyboard down          {before}")

        box = page.locator("#screen").bounding_box()
        tapped = box["height"] * 0.66
        page.locator("#screen").tap(position={"x": box["width"] / 2, "y": tapped})
        keyboard(page, KEYBOARD_UP)
        during = page.evaluate(REPORT)
        print(f"2. tapped 66% down, up    {during}")
        print(f"   the desktop kept its size: {during['canvas'] == before['canvas']}")
        top, bottom = during["scrolled"], during["scrolled"] + int(during["stage"].split("x")[1])
        print(f"   the tap at {tapped:.0f} is in view {top}..{bottom}: "
              f"{top <= tapped <= bottom}")

        page.locator("#keyboard").click()
        keyboard(page, PHONE_HEIGHT)
        after = page.evaluate(REPORT)
        print(f"3. keyboard hidden        {after}")
        print(f"   the whole desktop is back: {after['canvas'] == before['canvas']} "
              f"at scroll {after['scrolled']}")
        browser.close()


main()
