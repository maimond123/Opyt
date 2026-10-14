"""A collection walked as quota-loop units: start, one page per unit, finish.

Every collection curation reads from the user's own accounts runs through `WalkSource`: X's
Bookmarks, Following, Likes and Lists, and Substack's follows, subscriptions and saved posts. What
differs between collections is written down once, as `Collection`, and each implementation lives
with its rules (`bookmark_listing`, `x_collectors`, `substack_people`, `substack_saved_listing`).
What differs between platforms (the quota a page spends, how the signed-in account is read, when
the platform may be read next) is `Account`, one per platform (`x_walks`, `substack_walks`).

    start   read the account → begin a walk, or resume the unfinished one
    page    read one page → its observations, child walks and the collection's own writes in
            one transaction, then presence for the people it names; another signed-in account,
            where the platform checks for one before a page, instead supersedes the walk and
            starts the new account's
    finish  tally the walk → publish → complete

A collection whose pages span two operations (X's Lists, then each list's members) runs as two
sources, one per quota, over the same walk; the one that starts the walk also finishes it, once
every page of both operations is settled.

Signal writes commit per person (see `signals`), so a page asserts presence after its
transaction, and the finish checks the walk is ready, publishes, then completes: a crash between
the publish and the completion repeats the publish.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from pipeline.curation import clock, listings, saved_items, signals
from pipeline.kb.writes import write_tx
from pipeline.platforms.collection_page import Incomplete, Page
from pipeline.timeparse import parse_ts, utc_iso
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import YOURS, Context, Item, Outcome, Unit


class Collection(Protocol):
    collector: str      # the walk's key in `listings`
    operation: str      # the operation of its first page
    platform: str       # the platform its people's signals are on
    signal: str         # the signal type its people carry

    def read(self, session: Any, account: str, branch: listings.Branch) -> Page:
        """One page of `branch` (doing). `session` is what the platform's `Account` hands it."""

    def children(self, branch: listings.Branch, page: Page) -> tuple[listings.Branch, ...]:
        """Walks this page discovers, owed before the collection can finish."""

    def accepted(self, conn: sqlite3.Connection, walk: listings.Walk, page: Page,
                 now: datetime) -> None:
        """This collection's own writes for an accepted page, inside the page's transaction."""

    def people(self, walk: listings.Walk, branch: listings.Branch,
               page: Page) -> list[signals.Person]:
        """Who this page shows to hold the signal, for presence."""

    def tally(self, conn: sqlite3.Connection, walk: listings.Walk) -> list[signals.Tally]:
        """Each person's total over the whole walk's observations."""

    def publish(self, conn: sqlite3.Connection, walk: listings.Walk,
                tallies: list[signals.Tally], now: datetime) -> str:
        """Publish a complete walk's tallies; the outcome's status word."""

    def refresh_at(self, conn: sqlite3.Connection, walk: listings.Walk | None,
                   now: datetime) -> datetime:
        """When the next walk may start, given the last one (complete, or None)."""

    def reconnected(self, conn: sqlite3.Connection, account: str) -> None:
        """Another account replaced the walk's; inside the new walk's transaction."""


class Account(Protocol):
    """How a walk reaches the signed-in account on one platform."""
    platform: str        # the outage streak a page's failures count toward
    host: str | None     # the web pool's admission host; None on a quota of its own

    def quota(self, operation: str) -> str:
        """The quota a page of `operation` spends."""

    def connected(self) -> bool:
        """Is a session present at all (doing). Asked only when a unit would be offered."""

    def held(self, now: datetime) -> datetime | None:
        """Until when this platform must not be read (a pause, the endpoint's pace); None: now."""

    def session(self, ctx: Context, collection: Collection) -> Any:
        """What `collection.read` is handed (doing). Each platform's collections declare what
        this needs of them (`x_walks.XCollection`, `substack_walks.SubstackCollection`)."""

    def signed_in(self, ctx: Context, collection: Collection) -> str:
        """The signed-in account a walk starts under, or raise (doing)."""

    def switched(self, ctx: Context, collection: Collection) -> str | None:
        """Before a page: the signed-in account, where this platform checks it; None: unchecked."""

    def failed(self, ctx: Context, exc: Exception) -> Exception:
        """A read raised: drop whatever the next read should not reuse, and say what the loop
        should see (`exc` itself, or what it means on this platform)."""

    def read(self) -> None:
        """A request to the platform went out."""


@dataclass(frozen=True)
class Reconnected:
    """A page unit found another account signed in instead of the walk's."""
    account: str


def publish_counts(conn: sqlite3.Connection, collection: Collection, walk: listings.Walk,
                   tallies: list[signals.Tally], now: datetime,
                   clock_key: str | None = None) -> None:
    """Replace the people's counts and stamp the collector's clock with the walk's start, which
    is what absence-based retirement compares confirmations against."""
    signals.counts(conn, tallies, collection.signal, collection.platform)
    clock.record_run(conn, clock_key or collection.collector,
                              status=clock.STATUS_OK, now=utc_iso(now),
                              found=len(tallies),
                              stored_after=signals.stored(conn, collection.signal,
                                                          collection.platform),
                              started_at=walk.started_at)


def clock_refresh(conn: sqlite3.Connection, collector: str, now: datetime) -> datetime:
    """The collector clock's floor: six hours after its last attempt, or now."""
    run = clock.get_run(conn, collector)
    return clock.due_at(run, floor_hours=clock.FLOOR_HOURS) or now


def completed_refresh(conn: sqlite3.Connection, walk: listings.Walk | None, now: datetime,
                      interval: timedelta) -> datetime:
    """`interval` after the last walk completed; now for a home that never walked."""
    done = parse_ts(listings.completed_at(conn, walk)) if walk is not None else None
    return done + interval if done else now


class Clocked:
    """A people-only collection: its walk starts on the collector clock's six-hour floor, its
    counts publish from every complete walk, and nothing else is owed. A subclass names its
    `collector`, `operation`, `signal` and `platform`, and reads, people and tallies its pages."""

    def children(self, branch, page):
        return ()

    def accepted(self, conn, walk, page, now):
        pass

    def publish(self, conn, walk, tallies, now):
        publish_counts(conn, self, walk, tallies, now)
        return "published"

    def refresh_at(self, conn, walk, now):
        return clock_refresh(conn, self.collector, now)

    def reconnected(self, conn, account):
        pass


class SavedList:
    """A list of the user's saved posts, walked for its two consumers: each accepted page offers
    its posts as owed saved work (`saved_items`), and the posts' authors carry `save`. The list
    is walked every `refresh`; its counts publish on the six-hour `signal_collector` clock, and
    between those a complete walk only re-asserts presence (a page's own assertion may have been
    lost to a crash after it committed). A subclass names its `collector`, `operation`,
    `platform`, `signal_collector` and `refresh`, and `revision(record)`."""
    signal = "save"

    def children(self, branch, page):
        return ()

    def accepted(self, conn, walk, page, now):
        saved_items.offer(conn, self.platform, walk.account,
                          [(key, self.revision(record), record) for key, record in page.records],
                          now)

    def publish(self, conn, walk, tallies, now):
        run = clock.get_run(conn, self.signal_collector)
        if clock.is_due(run, floor_hours=clock.FLOOR_HOURS, now=now):
            publish_counts(conn, self, walk, tallies, now, clock_key=self.signal_collector)
            return "published"
        signals.presence(conn, [t.person for t in tallies], self.signal, self.platform)
        return "complete"

    def refresh_at(self, conn, walk, now):
        return completed_refresh(conn, walk, now, self.refresh)

    def reconnected(self, conn, account):
        saved_items.supersede(conn, self.platform, account)


START, FINISH, PAGES = "start", "finish", "pages"


def next_step(state: str | None, ends: bool) -> str | None:
    """Which kind of unit a walk owes next, from its state alone (deciding). `state` is None
    when no walk exists; `ends`: the source starts and finishes the walk.

    | Walk state           | `ends` | Step     |
    |----------------------|--------|----------|
    | none, or complete    | yes    | `START`  (once the collection's refresh is due) |
    | none, or complete    | no     | None     |
    | ready                | yes    | `FINISH` |
    | ready                | no     | None     |
    | walking              | either | `PAGES`  |
    """
    if state is None or state == listings.COMPLETE:
        return START if ends else None
    if state == listings.READY:
        return FINISH if ends else None
    return PAGES


class WalkSource:
    """The units of one collection's walk on one quota. `ends`: this source starts and finishes
    the walk; a companion source (`ends=False`) only reads its own operation's pages."""

    klass, floor = YOURS, 0

    def __init__(self, collection: Collection, via: Account, operation: str | None = None, *,
                 ends: bool = True):
        self.collection = collection
        self.via = via
        self.operation = operation or collection.operation
        self.ends = ends
        self.quota = via.quota(self.operation)
        self.name = f"{collection.collector}:{self.operation}"

    def _step(self, conn) -> tuple[listings.Walk | None, str | None]:
        """The current walk, and the kind of unit it owes next."""
        walk = listings.current(conn, self.collection.collector)
        return walk, next_step(walk.state if walk else None, self.ends)

    def _owed(self, conn, walk, step, now) -> list[Unit]:
        """The units the store says are owed, before asking the platform anything."""
        if step is None:
            return []
        if step == START:
            due = self.collection.refresh_at(conn, walk, now) <= now
            return [Unit(f"{self.name}:start", (START, now), host=self.via.host)] if due else []
        if step == FINISH:
            return [Unit(f"{self.name}:finish", (FINISH, walk, now), host=self.via.host)]
        return [Unit(f"{self.name}:page:{task.id}", (PAGES, walk, task, now),
                     item=Item(listings.ROWS, (task.id,), task.retry, platform=self.via.platform),
                     host=self.via.host)
                for task in listings.pending(conn, walk, self.operation)
                if retries.may_run(task.retry.retry_at, task.retry.failures, now)]

    def _due(self, conn, walk, step, now) -> datetime | None:
        """When the unit the walk owes may next run, before the platform's gate."""
        if step is None:
            return None
        if step == START:
            return self.collection.refresh_at(conn, walk, now)
        if step == FINISH:
            return now
        tasks = listings.pending(conn, walk, self.operation)
        if any(retries.may_run(t.retry.retry_at, t.retry.failures, now) for t in tasks):
            return now
        return retries.earliest_retry(t.retry for t in tasks)

    def _held_until(self, step, now) -> datetime | None:
        """Until when the platform holds this step: never a finish, which reads nothing from it."""
        return None if step == FINISH else self.via.held(now)

    def candidates(self, conn, now):
        walk, step = self._step(conn)
        owed = self._owed(conn, walk, step, now)
        if not owed or self._held_until(step, now) is not None:
            return []
        return owed if self.via.connected() else []

    def next_unit(self, conn, now):
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit, ctx):
        kind, *args = unit.payload
        if kind == FINISH:
            walk, _ = args
            return self.collection.tally(conn, walk)
        try:
            if kind == START:
                return self.via.signed_in(ctx, self.collection)
            walk, task, _ = args
            account = self.via.switched(ctx, self.collection)
            if account and account != walk.account:
                return Reconnected(account)
            return self.collection.read(self.via.session(ctx, self.collection), walk.account,
                                        task.branch)
        except Exception as exc:
            seen = self.via.failed(ctx, exc)
            if seen is exc:
                raise
            raise seen from exc
        finally:
            self.via.read()

    def record(self, conn, unit, fetched):
        kind, *args = unit.payload
        if kind == START:
            return self._start(conn, fetched, *args)
        if kind == FINISH:
            return self._finish(conn, *args, fetched)
        walk, task, now = args
        if isinstance(fetched, Reconnected):
            return self._start(conn, fetched.account, now)
        return self._page(conn, walk, task, now, fetched)

    def _start(self, conn, account: str, now: datetime) -> Outcome:
        with write_tx(conn, site="curation.walk.start"):
            prior = listings.current(conn, self.collection.collector)
            listings.start(conn, self.collection.collector, account,
                           listings.Branch(self.collection.operation), now)
            if prior is not None and prior.account != account:
                self.collection.reconnected(conn, account)
        return Outcome("started")

    def _page(self, conn, walk: listings.Walk, task: listings.Task, now: datetime,
              page: Page) -> Outcome:
        try:
            with write_tx(conn, site="curation.walk.page"):
                accepted = listings.accept_page(
                    conn, walk, task, page,
                    branches=self.collection.children(task.branch, page))
                if accepted:
                    self.collection.accepted(conn, walk, page, now)
        except Incomplete as exc:
            # A cursor that loops without an end: the page stays owed and is charged, so the
            # walk ends incomplete rather than claiming a total.
            return Outcome("incomplete", failure=retries.BLIP, detail=str(exc))
        if not accepted:
            return Outcome("replayed")
        signals.presence(conn, self.collection.people(walk, task.branch, page),
                         self.collection.signal, self.collection.platform)
        return Outcome("page")

    def _finish(self, conn, walk: listings.Walk, now: datetime,
                tallies: list[signals.Tally]) -> Outcome:
        listings.require_ready(conn, walk)
        status = self.collection.publish(conn, walk, tallies, now)
        listings.complete(conn, walk, now)
        listings.prune(conn, self.collection.collector)
        return Outcome(status)

    def next_due(self, conn, now):
        walk, step = self._step(conn)
        due = self._due(conn, walk, step, now)
        if due is None or not self.via.connected():
            return None             # connecting the platform wakes the loop
        held = self._held_until(step, now)
        return max(due, held) if held is not None else due
