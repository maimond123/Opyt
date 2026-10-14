"""
pipeline/oracles/sources/scraper_tables.py — every table the scrapers keep, created in one call,
so a tool that is the first writer on a fresh store makes them all. Every table is additive: a
server and a worker on different versions share one store for hours, so nothing is ever dropped
or renamed.
"""

from __future__ import annotations

import sqlite3

from pipeline.kb import footprint_photos
from pipeline.oracles import asks, found, told
from pipeline.oracles.sources.archive import state as archive_state
from pipeline.oracles.sources.web import recommendations


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL for every table the scrapers keep, safe on every writable open."""
    archive_state.init_schema(conn)
    recommendations.init_schema(conn)
    asks.init_schema(conn)
    found.init_schema(conn)
    told.init_schema(conn)
    footprint_photos.init_schema(conn)
