"""
pipeline/frontier/allowance.py — how much frontier may spend in an hour, and when a spent hour
frees.

Decisions only. The caps apply to any rolling hour.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

HOUR = timedelta(hours=1)
SEARCHES_PER_HOUR = int(os.environ.get("OPYT_FRONTIER_MAX_REQUESTS", 40))
ADMISSIONS_PER_HOUR = int(os.environ.get("OPYT_ADMIT_MAX_PER_RUN", 25))


def frees_at(spent: list[datetime], per_hour: int, now: datetime) -> datetime | None:
    """None while the last hour has room for another; else when the use that fills it leaves the
    hour. `spent` is when each use went out (older ones are ignored)."""
    recent = sorted(t for t in spent if t > now - HOUR)
    if len(recent) < per_hour:
        return None
    return recent[len(recent) - per_hour] + HOUR
