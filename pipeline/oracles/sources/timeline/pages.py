"""
pipeline/oracles/sources/timeline/pages.py — one page of an Oracle's Originals timeline: the
kit a scraper loads once (`PageKit`), the request rendered (`walk_page`), and the
write inside the caller's transaction (`write_page`, after `page_current`). Library's writer
(`ingest_x_footprint._GroupWriter`) renders and saves each group, handed the thread lane's tail
rules.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from opyt_core.paths import opyt_home
from pipeline.image_cache import load_image_cache
from pipeline.ingestion import x_render as xt
from pipeline.kb.ingest_common import StageTimer
from pipeline.kb.ingest_common import run_concurrent
from pipeline.kb.ingest_x_footprint import _GroupWriter, _INGEST_WORKERS, _filter_and_stitch
from pipeline.oracles import pairs
from pipeline.oracles.sources.threads import tail_state
from pipeline.platforms import sessions
from pipeline.platforms.x import originals as x_originals
from pipeline.platforms.x import profile as x_profile
from pipeline.timeparse import utc_now

# ── one page at a time ─────────────────────────
#
# The timeline scraper spends one request per unit, so it walks a timeline one page per call and
# keeps its place in the store between calls. What a whole-timeline walk paid per call is paid
# here once per scraper: the X session and each pair's user id live in a
# `PageKit`. Every stored X hash, every thread's tail state and every post's link state is looked
# up per page, for the twenty or so atoms the page touches, and so is the image cache, which the
# photo source fills meanwhile: a page rendering from a stale copy would drop descriptions the
# stored atoms already hold, and rewrite them.

X_FIRST_DAY = datetime(2006, 3, 21, tzinfo=timezone.utc)   # no page lies below it


class PageKit:
    """What a background scraper loads once and reuses for every page it walks."""

    def __init__(self):
        self._session = None
        self._user_ids: dict[tuple[str, str], str] = {}

    def session(self):
        if self._session is None:
            self._session = sessions.x_session("https://x.com/home")
        return self._session

    def forget_session(self) -> None:
        """Drop the session after it died, so the next page reads the cookies again: the user may
        have reconnected X in the meantime."""
        self._session = None

    def user_id(self, conn: sqlite3.Connection, canonical_id: str, handle: str) -> str | None:
        """The numeric X id behind a handle: the stored `x:user:{id}` entity of this Oracle's
        cluster when there is one, else one `UserByScreenName` lookup. None when X cannot say
        (suspended, protected or renamed). Remembered for the scraper's lifetime."""
        key = (canonical_id, handle.lower())
        if key not in self._user_ids:
            uid = stored_x_user_id(conn, canonical_id, handle)
            if uid is None:
                profile = x_profile.fetch_user_profile(self.session(), handle)
                uid = str(profile["user_id"]) if profile else None
            if uid is None:
                return None
            self._user_ids[key] = uid
        return self._user_ids[key]


def stored_x_user_id(conn: sqlite3.Connection, canonical_id: str, handle: str) -> str | None:
    """The `x:user:{id}` entity in this Oracle's cluster whose stored handle is `handle`."""

    for entity_id, (stype, key) in pairs.pairs_by_entity(conn, canonical_id).items():
        if stype == "x" and key.lower() == handle.lower() and entity_id.startswith("x:user:"):
            return entity_id[len("x:user:"):]
    return None


def _conv(tweet: dict) -> str:
    return str(tweet.get("conversationId") or tweet.get("id") or "")


def page_ledgers(conn: sqlite3.Connection, atom_ids) -> tuple[dict, dict, set]:
    """The three ledgers a writer reads — content hashes, thread-tail state, which posts' links
    are known — for the atoms one page touches, in one query."""
    ids = sorted(atom_ids)
    if not ids:
        return {}, {}, set()
    seen: dict[str, str] = {}
    tails: dict[str, dict] = {}
    links: set[str] = set()
    for row in conn.execute(
            "SELECT atom_id, raw_hash, json_extract(payload, '$.thread_tail') AS tail, "
            "json_extract(payload, '$.thread_ids') AS tids, "
            "json_type(payload, '$.links_owed') AS links FROM atoms "
            f"WHERE atom_id IN ({', '.join('?' for _ in ids)})", ids):
        seen[row["atom_id"]] = row["raw_hash"]
        if row["tail"]:
            try:
                tids = json.loads(row["tids"]) if row["tids"] else []
            except (TypeError, ValueError):
                tids = []
            tails[row["atom_id"]] = {"thread_tail": row["tail"], "thread_ids": tids}
        if row["links"] is not None:
            links.add(row["atom_id"])
    return seen, tails, links


@dataclass(frozen=True)
class Page:
    """What one `walk_page` call read, ready to write: its atoms rendered, and the tail and link
    states it decided. `write_page` writes it inside the caller's transaction, so the page and
    the caller's frontier commit together or not at all.

    `read_at`   — when the request went out. A page read from the top of the timeline saw
                  everything posted before this instant, which is what a forward walk stamps.
    `oldest_id` — the oldest post on the page, the pin excluded: where the next page resumes.
    `oldest_at` — that post's time. None when the page held nothing but the pin.
    `ended`     — nothing lies below this page: X sent no further cursor, or nothing new.
    `complete`  — every atom the page owes was rendered and goes out with `write_page`. A caller
                  advances its frontier only then, since a frontier is a claim that what lies
                  above it is stored.
    `added`     — the atoms `write_page` writes; `submitted`, the atoms the page owed.
    `seen`      — the stored hash of each of the page's atoms when it was read (None: not
                  stored), which `page_current` checks again before the write."""
    read_at: datetime
    fetched: int
    oldest_id: str | None
    oldest_at: datetime | None
    ended: bool
    complete: bool
    added: int
    submitted: int
    seen: dict = field(default_factory=dict)
    writer: "_GroupWriter | None" = None

    @property
    def summary(self) -> dict:
        """The writer's counters: after `write_page` commits, what the page wrote."""
        if self.writer is None:
            return {"added": self.added, "submitted": self.submitted}
        return {"fetched": self.fetched, "submitted": self.submitted,
                "threads_owed": self.writer.owed_held, **self.writer.summary()}


def page_current(conn: sqlite3.Connection, page: Page) -> bool:
    """Do the page's atoms still hold the hashes the page was read against? Another lane (the
    thread tails) may have rewritten one meanwhile, and the page would write over it."""
    stored = page_ledgers(conn, page.seen)[0]
    return all(stored.get(atom_id) == raw_hash for atom_id, raw_hash in page.seen.items())


def write_page(page: Page) -> None:
    """Write what `walk_page` read, on the connection it read with: its atoms and the tail and
    link states it decided. Inside the caller's `write_tx` they commit with whatever else the
    caller writes there."""
    if page.writer is not None:
        page.writer.write()


def walk_page(conn: sqlite3.Connection, kit: PageKit, *, user_id: str, handle: str,
              below_id: str | None = None, author_name: str | None = None) -> Page:
    """ONE request of an Oracle's Originals timeline — from the top, or just below `below_id` —
    rendered as opinion atoms from the photo reads already cached. Reads no photo
    and writes nothing: the caller writes the page with `write_page`, with its frontier, and the
    photos it still owes go with it.

    Raises what the request raises: `XRateLimited`, `SyncAuthError` and `XUserUnavailable` are
    the caller's to handle."""

    read_at = utc_now()
    session = kit.session()
    walk = x_originals.fetch_user_originals(session, str(user_id), after=X_FIRST_DAY,
                                     below_id=below_id)
    pin = walk.pinned
    pin_id = str(pin["id"]) if pin else None
    fresh = [t for t in walk.tweets if str(t["id"]) != pin_id]
    oldest = min(fresh, key=lambda t: int(t["id"])) if fresh else None
    oldest_at = xt.parse_twitter_date(oldest.get("createdAt", "")) if oldest else None
    # The pin arrives with every page. It is written by the page whose range holds it, and only
    # that page owes its thread — `tail_state.threads_owed`.
    pin_here = (pin is not None and oldest is not None and int(pin_id) > int(oldest["id"])
                and (below_id is None or int(pin_id) < int(below_id)))
    tweets = fresh + ([pin] if pin_here else [])

    groups = _filter_and_stitch(tweets)
    ids = {f"xprofile:{_conv(g[0])}" for g in groups}
    seen, tails, links = page_ledgers(conn, ids)
    before = {atom_id: seen.get(atom_id) for atom_id in ids}
    timer = StageTimer()
    writer = _GroupWriter(conn, img_cache=load_image_cache(opyt_home()), seen=seen,
                          timer=timer, author_name=author_name,
                          owed_convs=tail_state.threads_owed(tweets, str(user_id),
                                                             pinned=pin if pin_here else None),
                          tails=tails, links_known=links, tail_rules=tail_state)
    with timer.stage("process"):
        failures = run_concurrent(groups, writer.work, writer.consume, workers=_INGEST_WORKERS)
    complete = not failures.get("producer_failed") and not failures.get("consumer_failed")
    return Page(read_at=read_at, fetched=len(walk.tweets),
                oldest_id=str(oldest["id"]) if oldest else None, oldest_at=oldest_at,
                ended=walk.finished, complete=complete, added=writer.submitted,
                submitted=writer.submitted, seen=before, writer=writer)
