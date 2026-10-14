"""
pipeline/worker/failure_run.py — a run of failures, logged once when it starts and once when it
ends.

A loop that retries every second must not log every second. The supervisor's pass and the loops'
process reading its row both say when a run of failures begins and when it is over, and nothing
in between; each keeps its own words and its own `log`.
"""

from __future__ import annotations


class FailureRun:
    """Whether a run of failures is under way, and since when."""

    def __init__(self) -> None:
        self.since: float | None = None

    def failed(self, now: float) -> bool:
        """Count a failure. True for the first of a run, the one to log."""
        if self.since is not None:
            return False
        self.since = now
        return True

    def recovered(self, now: float) -> float | None:
        """Count a success. How long the run lasted when this one ends it; None otherwise."""
        if self.since is None:
            return None
        lasted, self.since = now - self.since, None
        return lasted
