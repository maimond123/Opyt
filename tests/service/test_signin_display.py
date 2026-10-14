"""The screen both sign-in desktops run on (`pipeline/hosted_signin/display.py`)."""
from __future__ import annotations

import os
import threading
import time

import pytest

from pipeline.hosted_signin import display
from pipeline.platforms.browser.profile import HostedBrowserError


def test_the_display_number_is_read_through_its_newline_across_two_writes():
    """Xvfb 21.1.12-1ubuntu1.8 writes "0" and "\\n" separately, and dies if the pipe is closed
    between them."""
    read_fd, write_fd = os.pipe()

    def xvfb():
        os.write(write_fd, b"0")
        time.sleep(0.05)
        os.write(write_fd, b"\n")
        os.close(write_fd)

    writer = threading.Thread(target=xvfb)
    writer.start()
    try:
        assert display.read_display_number(read_fd, time.monotonic() + 5) == "0"
    finally:
        writer.join()
        os.close(read_fd)


def test_a_display_number_that_never_comes_times_out():
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(HostedBrowserError):
            display.read_display_number(read_fd, time.monotonic() + 0.1)
    finally:
        os.close(read_fd)
        os.close(write_fd)
