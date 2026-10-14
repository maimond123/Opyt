"""Find the Chrome a server-side sign-in launches, and name the error a hosted browser raises.

The service's X sign-in (`service/x_signin_desktop.py`) runs Chrome on the server, and its screen
(`pipeline/hosted_signin/display.py`) raises `HostedBrowserError` when the server lacks a program
it needs.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


class HostedBrowserError(RuntimeError):
    """A server-side browser could not be prepared or used.

    Every message raised with it is an authored literal, never an interpolated profile path, URL,
    cookie or environment value, so a log that keeps the message verbatim stays safe to keep.
    """


def chrome_binary() -> Path:
    configured = os.environ.get("OPYT_HOSTED_CHROME")
    candidates = [configured] if configured else [
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise HostedBrowserError(
        "Hosted Opyt needs Google Chrome or Chromium on the server. Install it, or set "
        "OPYT_HOSTED_CHROME to its executable path.")
