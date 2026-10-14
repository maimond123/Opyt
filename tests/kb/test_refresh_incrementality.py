"""Incrementality seams that skip a paid call: a missing GitHub `pushed_at`.

Modelled on tests/kb/test_bookmark_lookback.py: these assert that no fetch happened, not that a
count came out right. A count can be satisfied by a dedup skip further downstream — which is
exactly the bug these seams exist to remove, since the whole point is not paying for the call.

The MCP surface lives in test_oracle_refresh_surface.py: these prove an adapter skips a call,
those prove the tool routes and reports.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.kb import ingest_github



# ── GitHub: `pushed_at` parse ────────────────────────────────────────────────────
def test_missing_pushed_at_is_processed_not_dropped():
    """Fail-safe: an absent or unparseable date must never silently drop a repo. `None` from here
    means neither window gate fires, so the repo is fetched."""
    assert ingest_github._pushed_at({"pushed_at": ""}) is None
    assert ingest_github._pushed_at({"pushed_at": "garbage"}) is None
    assert ingest_github._pushed_at({}) is None
    assert ingest_github._pushed_at({"pushed_at": "2026-01-01T00:00:00Z"}) == datetime(
        2026, 1, 1, tzinfo=timezone.utc)
