"""
pipeline/hosted_signin/display.py — the screen a hosted sign-in runs on: a private X display
(Xvfb), a loopback VNC server onto it (x11vnc), the process handling both need, and the pump that
carries the screen to a visitor's WebSocket.

The sharing service's X sign-in (`service/x_signin_desktop.py`) sits on this. What runs on the
screen, and when a sign-in is done, is its.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import select
import shutil
import socket
import subprocess
import time

from pipeline.platforms.browser.profile import HostedBrowserError

_DESKTOP_START_SECONDS = 15.0
# How long a desktop watches its browser before calling it open.
_CHROME_SETTLE_SECONDS = 1.0


def binary(name: str) -> str:
    """One required program in the hosted login desktop, named by its Ubuntu package."""
    path = shutil.which(name)
    if path:
        return path
    raise HostedBrowserError(
        f"A hosted sign-in needs {name}. Install the server display stack before starting the "
        "service.")


def reap(proc: subprocess.Popen) -> None:
    """End and collect one desktop process this login owns, escalating only when needed."""
    if proc.poll() is not None:
        with contextlib.suppress(OSError):
            proc.wait()
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        # A kill ends a process but does not collect it. The session that started it is the
        # lifecycle owner, so collection cannot be left to its parent or a later restart.
        with contextlib.suppress(OSError):
            proc.wait()


def start_xvfb(width: int, height: int) -> tuple[str, subprocess.Popen]:
    """Start one private X display and read its collision-free number from Xvfb itself."""
    read_fd, write_fd = os.pipe()
    try:
        try:
            proc = subprocess.Popen(
                [binary("Xvfb"), "-displayfd", str(write_fd), "-screen", "0",
                 f"{width}x{height}x24", "-nolisten", "tcp"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,),
            )
        finally:
            os.close(write_fd)
    except Exception:
        os.close(read_fd)
        raise
    try:
        number = read_display_number(read_fd, time.monotonic() + _DESKTOP_START_SECONDS)
    except Exception:
        # The caller cannot own this Popen until this function returns its display number, so
        # every failed startup path has to collect it here.
        reap(proc)
        raise
    finally:
        os.close(read_fd)
    if proc.poll() is not None or not number.isdecimal():
        reap(proc)
        raise HostedBrowserError("Xvfb exited before it allocated a usable display")
    return f":{number}", proc


def read_display_number(read_fd: int, deadline: float) -> str:
    """The display number Xvfb writes to its `-displayfd`, read through its newline.

    Read to the newline, not just what is there first: Xvfb 21.1.12-1ubuntu1.8 writes the number
    and the newline separately, and exits ("Cannot write display number to fd") when the pipe is
    already closed for the second write"""
    data = b""
    while not data.endswith(b"\n"):
        ready, _, _ = select.select([read_fd], [], [], max(0.0, deadline - time.monotonic()))
        if not ready:
            raise HostedBrowserError(
                "Xvfb did not allocate a display before the login timed out")
        chunk = os.read(read_fd, 32)
        if not chunk:                     # Xvfb closed its end
            break
        data += chunk
    return data.decode().strip()


def require_alive(proc: subprocess.Popen) -> None:
    """Fail a sign-in whose Chrome died, instead of serving a desktop with nothing drawn on it.
    Watches for a settle period, since a Chrome refused a held profile exits only after VNC is up.
    """
    deadline = time.monotonic() + _CHROME_SETTLE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise HostedBrowserError("Chrome could not open the sign-in desktop")
        time.sleep(0.05)


def free_loopback_port() -> int:
    """A currently-unused loopback port for this session's VNC listener."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def wait_for_vnc(proc: subprocess.Popen, port: int) -> None:
    """Wait for the owned VNC subprocess to accept locally, or report its failed start."""
    deadline = time.monotonic() + _DESKTOP_START_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise HostedBrowserError(
                "x11vnc exited before accepting the login desktop")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise HostedBrowserError(
        "x11vnc did not accept the login desktop before the timeout")


def vnc_argv(display: str, port: int, *, threads: bool) -> list[str]:
    """x11vnc on `display`, reachable only from this machine, without a password: the relay in
    front of it is what checks who may watch. `threads` gives each client its own input and output
    thread, which cuts input-to-frame latency; on the service box it stopped x11vnc
    accepting connections, so the service desktop runs without it."""
    return ([binary("x11vnc"), "-display", display, "-localhost", "-nopw", "-forever", "-shared"]
            + (["-threads"] if threads else []) + ["-rfbport", str(port)])


async def relay_vnc(websocket, reader: asyncio.StreamReader,
                    writer: asyncio.StreamWriter) -> None:
    """Pump raw RFB bytes between a visitor's WebSocket (Starlette's) and the desktop's VNC
    server, until either side ends."""
    async def browser_to_vnc() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("bytes")
            if payload is None:
                await websocket.close(code=1003)
                return
            writer.write(payload)
            await writer.drain()

    async def vnc_to_browser() -> None:
        while payload := await reader.read(65536):
            await websocket.send_bytes(payload)

    tasks = {asyncio.create_task(browser_to_vnc()), asyncio.create_task(vnc_to_browser())}
    _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
