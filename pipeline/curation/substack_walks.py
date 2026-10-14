"""How a walk reaches the signed-in Substack account: in the web pool, one unit at a time on
`substack.com`, behind the shared Substack outage pause and the reader endpoint's pace.

The account is read once, when a walk starts. Unlike X's viewer, which a local session answers
from a cookie, Substack's account is a request to the Cloudflare-guarded reader page, so reading
it before every page would double what a walk asks of the host. A switch to another account
mid-walk is caught by the next walk's start, which supersedes the old account's owed work; no
Substack signal is retired for being absent from a walk, so the pages read in between cost
nothing a switch between walks would not.

A missing session never reaches the quota loop as `SyncAuthError`, which would pause the whole
web pool on the X session's rule: a walk that cannot name its account raises `NotSignedIn`, an
ordinary failure the loop backs off.
"""

from datetime import datetime, timedelta
from typing import Any, Callable, Protocol

from pipeline.curation import walks
from pipeline.ingestion.utils import SyncAuthError
from pipeline.platforms import connections
from pipeline.platforms import sessions
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop.retries import Outages

PLATFORM = "substack"
HOST = "substack.com"
# Between two reads of the account lists: the reader endpoint is Cloudflare-guarded and hostile
# to bursts. Held in memory, like the outage streaks:
# a restart forgets it at the cost of one unpaced read.
PACE = timedelta(seconds=1)


class SubstackCollection(walks.Collection, Protocol):
    """A Substack collection: the transport its reads go over (doing)."""

    def transport(self) -> Any:
        """A fresh transport carrying the managed session's cookies."""


class NotSignedIn(RuntimeError):
    """The Substack session names no signed-in account."""


def connected() -> bool:
    """Is a managed Substack session present (doing)."""
    return connections._connected_substack()


def paused_until(outages: Outages, now: datetime) -> datetime | None:
    """Until when the shared Substack outage pauses every Substack unit; None when it does not."""
    return outages.until[PLATFORM] if outages.paused(PLATFORM, now) else None


def not_signed_in(exc: SyncAuthError) -> NotSignedIn:
    """What a missing Substack session means to the loop. Never `SyncAuthError`: the loop pauses
    a whole quota on that, and in the web pool it would stop every web unit."""
    return NotSignedIn(f"no managed Substack session: {exc}")


class SubstackAccount:
    platform, host = PLATFORM, HOST

    def __init__(self, quota: str, outages: Outages,
                 clock: Callable[[], datetime] = utc_now) -> None:
        self._quota = quota
        self.outages = outages
        self.clock = clock
        self._read_at: datetime | None = None

    def quota(self, operation):
        return self._quota

    def connected(self):
        return connected()

    def held(self, now):
        waits = [self._read_at + PACE if self._read_at is not None else None,
                 paused_until(self.outages, now)]
        later = [at for at in waits if at is not None and at > now]
        return max(later) if later else None

    def session(self, ctx, collection: SubstackCollection):
        key = (PLATFORM, collection.collector)
        if key not in ctx.cache:
            ctx.cache[key] = collection.transport()
        return ctx.cache[key]

    def signed_in(self, ctx, collection):
        account = sessions.own_account_id()
        if not account:
            raise NotSignedIn("the Substack session names no signed-in account")
        return str(account)

    def switched(self, ctx, collection):
        return None                 # read at start only; see the module docstring

    def failed(self, ctx, exc):
        # A transport caches the session's cookies; a refusal may be a session that changed.
        for key in [k for k in ctx.cache if isinstance(k, tuple) and k[0] == PLATFORM]:
            ctx.cache.pop(key)
        if isinstance(exc, SyncAuthError):
            return not_signed_in(exc)
        return exc

    def read(self):
        self._read_at = self.clock()
