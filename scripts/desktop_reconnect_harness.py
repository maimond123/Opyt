"""Test `gateway/static/desktop.js`'s reconnect against a VNC server that can be dropped at will.

    python scripts/desktop_reconnect_harness.py gateway

Needs `playwright` (with its Chromium) and `websockets`, neither of which is a dependency of the
project. This is an instrument, run by hand, like `scripts/rfb_latency.py`.

The live gateway cannot answer this question. Killing its socket from the client mangles the
binary relay, and killing it from the server takes the desktop with it. So this stands up the
smallest thing noVNC will call "connected", an RFB 3.8 handshake and nothing after it, and
serves the real module from the repo, unmodified.

Three drops, each the shape of a real one:
  1. the server closes the connection (a suspended tab's socket being reaped)
  2. the page returns from the background with the socket still open (the silent-death case: a
     phone that changed network holds a socket that is dead without being closed)
  3. the page returns after a glance (must NOT reconnect)

Measured 2026-09-09, all three as expected:
docs/plans/2026-09-09-hosted-signin-from-a-phone.md.
"""
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

GATEWAY = Path(sys.argv[1])          # the gateway/ directory, so /static/... resolves
HTTP_PORT = 8891
WS_PORT = 8892

HARNESS = """<!doctype html>
<html><head><meta charset=utf-8><title>reconnect harness</title></head><body>
<div class=stage style="position:relative;width:390px;height:600px">
<textarea id=keys style="position:absolute;width:1px;height:1px;opacity:0"></textarea>
<div id=screen style="width:390px;height:600px"></div>
</div>
<button id=keyboard aria-pressed=false>Keyboard</button>
<script type=module>
import {attachDesktop} from '/static/desktop.js';
window.seen = [];
attachDesktop({
  screen: document.getElementById('screen'),
  keys: document.getElementById('keys'),
  button: document.getElementById('keyboard'),
  streamPath: 'ws://127.0.0.1:__WS__/stream',
  expiresIn: 600,
  onStatus: (state) => { window.seen.push(state); console.log('status:' + state); },
});
</script></body></html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/":
            body = HARNESS.replace("__WS__", str(WS_PORT)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *_args):
        pass


def serve_files():
    os.chdir(GATEWAY)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", HTTP_PORT), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


handshakes = []
live_sockets = []


async def rfb(socket):
    """Just enough RFB 3.8 for noVNC to say `connect`, and nothing at all after it."""
    handshakes.append(socket)
    live_sockets.append(socket)
    print(f"   [ws] connection {len(handshakes)}")
    await socket.send(b"RFB 003.008\n")
    await socket.recv()                                    # the client's version
    await socket.send(bytes([1, 1]))                       # one security type: None
    await socket.recv()                                    # the client's choice
    await socket.send(struct.pack(">I", 0))                # SecurityResult: OK
    await socket.recv()                                    # ClientInit
    name = b"harness"
    await socket.send(struct.pack(">HH", 480, 690)
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


def main():
    serve_files()
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    threading.Thread(target=lambda: loop.run_until_complete(ws_server(ready)),
                     daemon=True).start()
    ready.wait(5)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 390, "height": 844},
                                   device_scale_factor=3, is_mobile=True,
                                   has_touch=True).new_page()
        page.on("pageerror", lambda e: print(f"  [pageerror] {e}"))
        page.on("console", lambda m: print(f"  [console:{m.type}] {m.text}"))
        page.goto(f"http://127.0.0.1:{HTTP_PORT}/")
        page.wait_for_function("window.seen && window.seen.includes('live')", timeout=10000)
        print(f"1. connected. handshakes={len(handshakes)} states={page.evaluate('window.seen')}")

        print("2. the server drops the connection")
        asyncio.run_coroutine_threadsafe(live_sockets[-1].close(), loop).result(5)
        page.wait_for_function("window.seen.filter(s => s === 'live').length >= 2", timeout=15000)
        print(f"   reconnected. handshakes={len(handshakes)} states={page.evaluate('window.seen')}")

        print("3. away for 8 s with the socket still open, then back")
        page.evaluate("""() => {
          Object.defineProperty(document, 'visibilityState', {configurable: true,
                                                             get: () => 'hidden'});
          document.dispatchEvent(new Event('visibilitychange'));
        }""")
        page.wait_for_timeout(8000)
        before = len(handshakes)
        page.evaluate("""() => {
          Object.defineProperty(document, 'visibilityState', {configurable: true,
                                                             get: () => 'visible'});
          document.dispatchEvent(new Event('visibilitychange'));
        }""")
        page.wait_for_function(f"window.seen.filter(s => s === 'live').length >= 3",
                               timeout=15000)
        print(f"   replaced a live-looking socket: {len(handshakes) - before} new handshake(s)")

        print("4. away for 1 s, then back")
        for state in ("hidden", "visible"):
            if state == "visible":
                page.wait_for_timeout(1000)
            page.evaluate(f"""() => {{
              Object.defineProperty(document, 'visibilityState', {{configurable: true,
                                                                  get: () => '{state}'}});
              document.dispatchEvent(new Event('visibilitychange'));
            }}""")
        page.wait_for_timeout(3000)
        print(f"   handshakes now {len(handshakes)} (a glance must not reconnect)")
        print(f"   final states: {page.evaluate('window.seen')}")
        browser.close()


main()
