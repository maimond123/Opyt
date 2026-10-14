"""
pipeline/oracles/sources/loops.py — oracles' part of the worker's `scrapers` job: one loop per
quota over this package's sources, with its outage ladder, this package's tables, and the hourly
seeding of the pairs Oracles gained. The one web pool is built here; another domain's web sources
join it (`loops(extra_web=...)`).
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from pipeline.dedup_store import init_schema as init_dedup_schema
from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.pool import WebPool
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.quota_loop.retries import OUTAGE_PAUSES_S, PLATFORM_PAUSES_S, Outages
from pipeline.oracles import pairs
from pipeline.oracles.sources.threads.bookmarks import Bookmarks
from pipeline.oracles.sources.threads.state import QUOTA as X_THREADS
from pipeline.oracles.sources.threads.state import move_payload_walks
from pipeline.oracles.sources.threads.tails import Tails
from pipeline.oracles.sources.references.cards import Cards
from pipeline.oracles.sources.references.documents import Documents
from pipeline.oracles.sources.references.x_posts import QUOTA as X_REFERENCES
from pipeline.oracles.sources.references.x_posts import XPosts
from pipeline.oracles.sources.timeline.scouting import Scouting
from pipeline.oracles.sources.timeline.sources import Check, NewPosts, Picks
from pipeline.oracles.sources.timeline.walks import QUOTA as X_TIMELINE
from pipeline.oracles.sources.web.archives import Archives
from pipeline.oracles.sources.web.archives import WebNewPosts
from pipeline.oracles.sources.web.platform import QUOTA as WEB
from pipeline.oracles.sources.web.discovery import Discovery
from pipeline.oracles.sources.web.recommendations import Recommendations
from pipeline.oracles.sources.web.substack_samples import SubstackSamples
from pipeline.oracles.sources.x_photos import QUOTA as X_PHOTOS
from pipeline.oracles.sources.x_photos import OraclePhotos

from pipeline.oracles.sources import scraper_tables


def x_timeline_sources() -> list:
    """The sources on X's `UserOriginalsTimeline` quota. Within a class they take turns in this
    order."""
    return [Picks(), Check(), NewPosts(), Scouting()]


def init_schema(conn: sqlite3.Connection) -> None:
    """These sources' tables, on a loop's connection: the loop creates only its own, and
    `told.mark_shown` on a store with no ask yet would log and repeat. A thread walk an older
    version left in an atom's payload moves to its table here, before any loop reads one."""
    scraper_tables.init_schema(conn)
    init_dedup_schema(conn)
    move_payload_walks(conn)


def _connect():
    conn = schema.connect()
    init_schema(conn)
    return conn


def x_timeline_loop(*, connect=_connect, **kw) -> QuotaLoop:
    """X's loop: every unit is one platform, `x`, so an outage stops the whole loop."""
    return QuotaLoop(X_TIMELINE, x_timeline_sources(), executor=Serial(),
                     outages=Outages(OUTAGE_PAUSES_S, whole_quota=True),
                     connect=connect, **kw)


def web_loop(outages: Outages | None = None, extra: list | tuple = (), *, connect=_connect,
             **kw) -> QuotaLoop:
    """The web's loop: an outage pauses one platform, whose units its sources then skip.
    `extra`: another domain's web sources, which share `outages`."""
    outages = outages if outages is not None else Outages(PLATFORM_PAUSES_S)
    return QuotaLoop(WEB, [Discovery(), Archives(outages), Documents(outages), Cards(outages),
                           WebNewPosts(outages), Recommendations(), SubstackSamples(), *extra],
                     executor=WebPool(), outages=outages,
                     connect=connect, **kw)


def x_threads_loop(*, connect=_connect, **kw) -> QuotaLoop:
    """Bookmarks and Oracle tails share TweetDetail, one request each turn."""
    return QuotaLoop(X_THREADS, [Bookmarks(), Tails()], executor=Serial(),
                     outages=Outages(OUTAGE_PAUSES_S, whole_quota=True),
                     connect=connect, **kw)


def x_references_loop(*, connect=_connect, **kw) -> QuotaLoop:
    return QuotaLoop(X_REFERENCES, [XPosts()], executor=Serial(),
                     outages=Outages(OUTAGE_PAUSES_S, whole_quota=True),
                     connect=connect, **kw)


def x_photos_loop(*, connect=_connect, **kw) -> QuotaLoop:
    """Oracles' X photos, read on their own loop so a timeline page or a thread tail never waits
    on one. It spends no X request."""
    return QuotaLoop(X_PHOTOS, [OraclePhotos()], executor=Serial(),
                     outages=Outages(PLATFORM_PAUSES_S),
                     connect=connect, **kw)


def loops(connect: Callable[[], sqlite3.Connection], web_outages: Outages, *,
          extra_web: list | tuple = ()) -> list[QuotaLoop]:
    """Oracles' loops, one per quota, on the composition's connection factory."""
    return [x_timeline_loop(connect=connect), x_threads_loop(connect=connect),
            x_references_loop(connect=connect), x_photos_loop(connect=connect),
            web_loop(web_outages, extra_web, connect=connect)]


def seed() -> None:
    """Register the pairs Oracles have gained since the last seeding: X connected after Oracles
    were added, an identity `resolve` merged into an Oracle, an Oracle's links changing. Seeding
    only adds rows. Never raises: a failed seeding is retried at the next one."""
    try:
        conn = schema.connect()
        try:
            pairs.seed_from_entities(
                conn, include_x=connections._connected_x())
        finally:
            conn.close()
    except Exception as exc:
        log(f"[scrapers] seeding failed, retried in an hour: {type(exc).__name__}: {exc}")
