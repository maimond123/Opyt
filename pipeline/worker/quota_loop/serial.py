"""
pipeline/worker/quota_loop/serial.py — the serial executor (`Serial`): one unit at a time, on the
loop's own thread and connection.

Every X quota runs this way: one request at a time is what the meter and X's windows allow. The
loop admits and settles each unit (`QuotaLoop.step`); this executor only repeats it, backs off a
raise before any unit was picked, and sleeps on a wait.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Iterable

from .loop import QuotaLoop
from .retries import Wait
from .source import Source, Unit


class Serial:
    """The `Executor` that runs one unit at a time."""

    def units(self, source: Source, conn: sqlite3.Connection, now: datetime) -> Iterable[Unit]:
        """One unit runs at a time, so a source offers only its most urgent one."""
        unit = source.next_unit(conn, now)
        return [] if unit is None else [unit]

    def execute(self, loop: QuotaLoop, conn: sqlite3.Connection) -> None:
        """Step until the loop is stopping: a raise before any unit was picked backs off, and a
        wait is slept on."""
        while not loop.stopping:
            loop.take_wake()
            try:
                out = loop.step(conn)
            except Exception as exc:
                out = loop.record_backoff(conn, exc)
            if isinstance(out, Wait):
                loop.sleep_until(out.until)
