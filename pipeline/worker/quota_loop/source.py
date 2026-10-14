"""
pipeline/worker/quota_loop/source.py — the contract every source of a quota loop meets.

A source is one kind of owed work on one quota: Oracles' timelines, a candidate's sample, an
Oracle's archives. The quota loop (`loop.py`) asks each of its sources what it owes, runs the most
urgent unit, and asks again. Which class a source is in decides when it runs (YOURS › NEW_POSTS ›
SCOUTING); which of its own units is most urgent is the source's business. The implementations
live with the unit whose work they fetch (oracles' `oracles/sources`, curation's
`collectors`).

A unit runs in two halves, and the loop does everything between and after them: the source fetches and saves (`fetch`), the loop removes what landed
for an Oracle forgotten meanwhile, the source records its own state (`record`) and says what the
unit came to (`Outcome`), and the loop charges a failure to the unit's item or clears it, and
keeps the outage streaks. A source never charges, retries, gives up or purges on its own.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Protocol

YOURS, NEW_POSTS, SCOUTING = 0, 1, 2


class Changed(RuntimeError):
    """The work's input/owner changed, or the executor lost its lease before publication."""


# The same thing reported as an `Outcome` status rather than raised. Either way the loop settles
# nothing: no charge, no clear, and the platform has not answered.
CHANGED = "changed"


@dataclass(frozen=True)
class Retry:
    """An item's retry state as its own row stores it, values as stored: a NULL stays None, so a
    charge given back restores the row exactly."""
    status: str | None = None       # the row's last status: the failure kind while failing
    failures: int | None = None     # failures in a row of that kind
    retry_at: str | None = None     # not offered before this instant; None once given up
    last_error: str | None = None


class RetryRows(Protocol):
    """One table's retry columns. Each source that charges keeps its items' retry state on its
    own rows; this is how the loop reads and writes them, with no schema of its own."""

    def read(self, row: Any) -> Retry:
        """The retry state of a row the source already loaded."""

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        """A charge, or a charge given back."""

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        """After a success."""


@dataclass(frozen=True)
class Item:
    """What a unit's failure is charged to: one row of one source's table."""
    rows: RetryRows
    key: tuple                      # the row's key in its table
    prior: Retry                    # its retry state when the unit was picked
    platform: str | None = None     # the outage streak its failures count toward; None: none
    kind: str | None = None         # every failure is charged as this kind; None: as reported
    # The unit reads a site that is its owner's, not its platform's (a Substack on its own
    # domain). A request to it that gets no answer is that site failing, so it is charged to the
    # item and never counts toward the platform's outage (`retries.unanswered`).
    own_site: bool = False


@dataclass
class Footprint:
    """The Oracle whose atoms a unit saves, for the loop's forget cleanup. A source adds each
    author it writes as, beyond the Oracle's cluster, before it saves under it."""
    oracle: str
    authors: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Forget:
    """How the loop honors a forget that lands while a unit runs, handed in by the unit that owns
    the roster: where the store's atoms end now (`mark`), and `purge(conn, oracle, mark,
    authors)`, which removes what arrived since for an Oracle no longer on the roster and says
    whether it was."""
    mark: Callable[[sqlite3.Connection], int]
    purge: Callable[[sqlite3.Connection, str, int, Iterable[str]], bool]


@dataclass(frozen=True)
class Unit:
    """One request's worth of owed work. `key` names it within its source (e.g. a pair and a
    direction), and the backstop backoff is keyed on it. `payload` is whatever the source needs
    to run it."""
    key: str
    payload: Any = None
    item: Item | None = None            # what a failure is charged to; None: the backstop's
    footprint: Footprint | None = None  # the Oracle whose atoms it saves; None: it saves none
    host: str | None = None             # admission host for web work; request hosts are gated separately


@dataclass(frozen=True)
class Outcome:
    """What one unit came to, as its source reports it after the save. The stores' differences
    are carried as data: what kind a failure is, whether it counts toward its platform's outage,
    and whether the platform answered at all."""
    status: str                     # the word for the log; for a failure, its kind
    failure: str | None = None      # the kind charged to the item; None: nothing to charge
    detail: str = ""                # the item's last error, for a failure
    outage: bool = False            # the failure counts toward its platform's outage streak
    answered: bool = True           # the platform answered: its streak and pause end


@dataclass
class Context:
    """What a loop hands every unit it runs.

    `cache` lives as long as the loop: a source loads what it needs once (a model check, the
    image cache) instead of once per page. `lease_lost` turns True if another process took this
    quota's lease, and a source checks it before it writes."""
    quota: str
    cache: dict = field(default_factory=dict)
    lease_lost: Callable[[], bool] = lambda: False


class Source(Protocol):
    name: str            # "picks", "check", "new_posts", "scouting", …
    quota: str           # the quota its units spend, e.g. "x:UserOriginalsTimeline"
    klass: int           # YOURS, NEW_POSTS or SCOUTING
    floor: int           # runs only while the meter shows more than this left (0: no floor)

    def next_unit(self, conn: sqlite3.Connection, now: datetime) -> Unit | None:
        """The most urgent unit this source owes, computed from the store. Never cached."""

    def fetch(self, conn: sqlite3.Connection, unit: Unit, ctx: Context) -> Any:
        """Fetch one unit and save what it brought. Raises what failed; the loop decides what
        that costs."""

    def record(self, conn: sqlite3.Connection, unit: Unit, fetched: Any) -> Outcome:
        """This source's own state writes for a fetched unit (through `writes.write_tx`), and
        what the unit came to."""

    def next_due(self, conn: sqlite3.Connection, now: datetime) -> datetime | None:
        """When something next becomes owed, for a loop with nothing to do. None: never, as
        far as this source knows."""


class PoolSource(Source, Protocol):
    def candidates(self, conn: sqlite3.Connection, now: datetime) -> Iterable[Unit]:
        """Priority-ordered ready units, so a busy host cannot hide this source's other work."""
