"""
pipeline/worker/quota_loop/retries.py — what a failure costs: every wait the quota takes after a
failure or a spent window (`Wait`), an item's retry ladder and give-up, a platform's outage
streak, and the charge and clear writes. The one retry policy every source uses.

An item is whatever a source walks: an Oracle's pair, a scouting candidate. When a unit fails,
the item waits before its next try, for longer the more often that kind of failure repeats:

  | kind      | what failed                                         | waits, each ±20%             |
  |-----------|-----------------------------------------------------|------------------------------|
  | `blip`    | a timeout, a dropped connection, a 5xx, a 200 that  | 1, 2, 5, 15, 30, then 60 min |
  |           | carries no timeline, queryId drift                  |                              |
  | `account` | the account is suspended, protected or unknown      | 1 h, 6 h, then daily         |
  | `unsaved` | the page came back but would not save (ours)        | as a blip; after 3 in a row  |
  |           |                                                     | the item is reported stuck   |

The count is durable, on the item's own row, and a success clears it.

Giving up. The tenth failure in a row of one kind clears the item's `retry_at`, and the item is no
longer offered (`given_up`): about 5 hours of trying for a blip or an unsaved page, about a week
for an account or a blocked host. Only the user's next ask for that Oracle re-arms it
(`oracles.asks.record_ask` resets the counts); a check never does, and a scouting candidate never is.

Outages (`Outages`, one per loop). Blips on three different items of one platform in a row, with
nothing answering between, are the platform failing rather than them: the three get back what
the streak charged them, the platform pauses, and nothing is charged to anyone until it answers
again: the first unit after each pause is the probe. A site that is its owner's own (a Substack
on its own domain) and gives no answer is charged to its item and counts toward nothing
(`counts_toward_outage`), so dead domains cannot pause the platform they sit on. The loop holds a paused platform's units at
admission (`QuotaLoop.admit`), for every source, so no source has to remember the pause. The
streak lives in memory, because it is about the present moment. On X every unit is one platform, `x`, so a pause stops the whole loop
(30 s, 1, 2, 5, 10, then 15 min). On the web most hosts are independent: a platform is Substack,
GitHub, OpenAlex or a blog's registrable domain, blips and blocks both count, and a pause skips
only that platform's units (1 h, 6 h, then daily) while everyone else's run on.

What is not an item's failure is never charged to one: the window spent, the session dead, the
model provider down, a lock on the store. Those are the quota's
(`quota_wait`), and the loop waits on them.

The web. An archive walk's document that would not save comes back `unsaved`
(`oracles/sources/archive/run.py`), and a raise is a blip.

One entry point. The loop charges every failure through `charge` and clears every success through
`succeeded`, writing through the item's own table (`source.RetryRows`).
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from pipeline.circuit_breaker import CircuitOpenError
from pipeline.ingestion.utils import SyncAuthError, log
from pipeline.platforms.x.originals import XUserUnavailable
from pipeline.kb.writes import is_lock_error, write_tx
from pipeline.platforms.x.rate import X_RATE_WINDOW, XRateLimited
from pipeline.timeparse import parse_ts, utc_iso

from .source import Item, Outcome, Retry

BLIP, ACCOUNT, UNSAVED = "blip", "account", "unsaved"
BLIP_WAITS_S = (60.0, 120.0, 300.0, 900.0, 1800.0, 3600.0)
ACCOUNT_WAITS_S = (3600.0, 6 * 3600.0, 24 * 3600.0)
JITTER = 0.2
STUCK_AFTER = 3                            # `unsaved` failures in a row before an item is stuck
GIVE_UP_AFTER = 10                         # failures in a row before an item is no longer tried
OUTAGE_STREAK = 3                          # different items failing in a row: X is down
OUTAGE_PAUSES_S = (30.0, 60.0, 120.0, 300.0, 600.0, 900.0)   # X: the whole loop waits
PLATFORM_PAUSES_S = ACCOUNT_WAITS_S                           # a web platform's units wait

RESET_GRACE_S = 5.0             # after X's announced reset, before the first request
PAUSED_RECHECK_S = 15 * 60.0    # a paused loop tries again on its next wake, or after this
TRANSIENT_BACKOFF_S = (1.0, 2.0, 5.0, 15.0, 30.0)
TRANSIENT = "transient"         # the cause of a wait on the transient ladder (`transient_wait`)

@dataclass(frozen=True)
class Wait:
    """How long the quota waits, and why; None means until woken."""
    until: float | None
    state: str
    cause: str | None = None


def spent_wait(reset: float, now: float) -> Wait:
    """The window is spent: wait for its reset, plus `RESET_GRACE_S`. A reset already past still
    gets the grace, so a stale reading cannot make the loop spin."""
    return Wait(max(reset, now) + RESET_GRACE_S, "waiting", "quota_spent")


def transient_wait(now: float, transient: int) -> Wait:
    """The `transient`-th rung, from 1, of the backoff for a failure that passes on its own (a
    lock on the store)."""
    delay = TRANSIENT_BACKOFF_S[min(transient, len(TRANSIENT_BACKOFF_S)) - 1]
    return Wait(now + delay, "backoff", TRANSIENT)


def outage_wait(until: float, platform: str) -> Wait:
    """A platform that is the whole quota is down: the quota pauses until `until`."""
    return Wait(until, "paused", f"{platform}_outage")


def _rate_wait(exc: XRateLimited, now: float, _transient: int) -> Wait:
    return spent_wait(exc.reset_at if exc.reset_at else now + X_RATE_WINDOW, now)


def _session_wait(_exc: BaseException, now: float, _transient: int) -> Wait:
    return Wait(now + PAUSED_RECHECK_S, "paused", "x_session")


def _model_wait(exc: CircuitOpenError, now: float, _transient: int) -> Wait:
    return Wait(now + max(float(exc.retry_after), 1.0), "paused", f"model:{exc.service}")


# Ordered once: these failures belong to the quota, never the item.
_WAIT_RULES = ((XRateLimited, _rate_wait), (SyncAuthError, _session_wait),
               (CircuitOpenError, _model_wait))


def kind_of(exc: BaseException) -> str | None:
    """The item's failure kind, or None for exactly the failures `quota_wait` waits on."""
    if any(isinstance(exc, cls) for cls, _ in _WAIT_RULES) or is_lock_error(exc):
        return None
    if isinstance(exc, sqlite3.OperationalError):
        return UNSAVED
    if isinstance(exc, XUserUnavailable):
        return ACCOUNT
    return BLIP


# By class name: the transports (curl_cffi, requests) are optional imports and share these names.
_NO_ANSWER = frozenset({"ConnectionError", "Timeout", "TimeoutError"})


def unanswered(exc: BaseException) -> bool:
    """The request got no HTTP answer at all: no connection, no TLS, no name, or a timeout. A
    refusal (a 403, a 429) is an answer."""
    return any(cls.__name__ in _NO_ANSWER for cls in type(exc).__mro__)


def counts_toward_outage(kind: str, exc: BaseException, item: Item) -> bool:
    """Does this failure say the item's platform is failing? A blip does, unless the item is
    its owner's own site and it never answered: a dead domain is not Substack being down. Three
    of those in a row would pause all of Substack."""
    return kind == BLIP and not (item.own_site and unanswered(exc))


def quota_wait(exc: BaseException, now: float, transient: int = 1) -> Wait | None:
    """The quota's wait; None leaves a unit's own failure to the backstop.

    `transient` counts this failure, starting at 1 (the default). No reset header means a whole window;
    a reset already past still gets the grace, so a stale header cannot make the loop spin.
    """
    for cls, build in _WAIT_RULES:
        if isinstance(exc, cls):
            return build(exc, now, transient)
    if is_lock_error(exc):
        return transient_wait(now, transient)
    return None


def jitter() -> float:
    return random.uniform(1 - JITTER, 1 + JITTER)


def wait_for(kind: str, failures: int) -> float:
    """The base wait after the `failures`-th failure in a row of `kind`, before jitter."""
    ladder = ACCOUNT_WAITS_S if kind == ACCOUNT else BLIP_WAITS_S
    return ladder[min(max(failures, 1), len(ladder)) - 1]


def failures_after(prior_kind: str | None, prior_failures: int | None, kind: str) -> int:
    """The count after one more failure of `kind`: another kind starts its own count."""
    return (prior_failures or 0) + 1 if prior_kind == kind else 1


def retry_at(kind: str, failures: int, now: datetime, jitter: float) -> datetime | None:
    """When the item may run again after its `failures`-th failure in a row of `kind`; None once
    that is the one it gives up on."""
    if given_up(failures):
        return None
    return now + timedelta(seconds=jitter * wait_for(kind, failures))


def given_up(failures: int | None) -> bool:
    """`GIVE_UP_AFTER` failures in a row: the item is no longer offered until the user asks
    again. One condition, read by every may-this-run-now check."""
    return (failures or 0) >= GIVE_UP_AFTER


def may_run(retry_at: str | None, failures: int | None, now: datetime) -> bool:
    """May the item run now: no `retry_at` ahead of `now` (as its row stores it), and not given
    up on. The one rule every source applies to its own items."""
    retry = parse_ts(retry_at)
    return not (retry is not None and retry > now) and not given_up(failures)


def earliest_retry(waits: Iterable[Retry]) -> datetime | None:
    """The earliest `retry_at` among rows that wait on one; None when none waits."""
    times = [parse_ts(r.retry_at) for r in waits if r.retry_at is not None]
    return min((at for at in times if at is not None), default=None)


def next_failure(prior_kind: str | None, prior_failures: int | None, kind: str,
                 now: datetime, jitter: float) -> tuple[int, datetime | None]:
    """What one more failure of `kind` costs an item whose row last recorded `prior_kind`
    `prior_failures` times: its new count, and when it may run again (None once given up)."""
    failures = failures_after(prior_kind, prior_failures, kind)
    return failures, retry_at(kind, failures, now, jitter)


def stuck(last_kind: str | None, failures: int | None) -> bool:
    """The item's pages come back but will not save, `STUCK_AFTER` times running: report it,
    with its last error as the cause. Retries go on until the item gives up."""
    return last_kind == UNSAVED and (failures or 0) >= STUCK_AFTER


class Outages:
    """Whether a platform is failing, for every item of one loop. Held in memory for the loop's
    lifetime: a restart that forgets it costs at most three more failed requests.

    `ladder` is the pauses, longest last; `whole_quota` says a pause stops the whole loop (X,
    where every unit is one platform) rather than only that platform's units (the web)."""

    def __init__(self, ladder: tuple[float, ...], *, whole_quota: bool = False) -> None:
        self.ladder = ladder
        self.whole_quota = whole_quota
        self.streaks: dict[str, dict[tuple, Item]] = {}   # platform → items charged in a row
        self.until: dict[str, datetime] = {}              # platform → paused until
        self.pauses: dict[str, int] = {}                  # platform → pauses since it answered

    def paused(self, platform: str, now: datetime) -> bool:
        until = self.until.get(platform)
        return until is not None and until > now

    def next_resume(self, now: datetime) -> datetime | None:
        later = [t for t in self.until.values() if t > now]
        return min(later) if later else None

    def answered(self, platform: str) -> None:
        """A unit on `platform` came back: the platform is answering."""
        self.streaks.pop(platform, None)
        self.until.pop(platform, None)
        self.pauses.pop(platform, None)

    def state(self, platform: str | None) -> OutageState:
        """Copy the platform's streak so planning cannot change the live tracker. It is in
        outage when paused since it last answered: its next counted failure is a failed probe."""
        return OutageState(tuple(self.streaks.get(platform, {}).values()),
                           platform in self.until)

    def apply(self, platform: str, plan: ChargePlan, now: datetime,
              jitter: float) -> datetime | None:
        """What a charge's plan does to the platform: it answered, its streak, and its pause.
        The one method that changes a platform's state for a charge. Returns when the pause
        ends, if the plan paused it.

        A failure while the pause still holds is not a probe: a unit already in flight when the
        platform paused, or one admitted beside the probe. It leaves the pause as it is, so the
        ladder climbs one rung a pause, however many units fail."""
        if plan.reset:
            self.answered(platform)
        if plan.streak:
            self.streaks[platform] = {streak_key(prior): prior for prior in plan.streak}
        else:
            self.streaks.pop(platform, None)
        if not plan.pause or self.paused(platform, now):
            return None
        return self.pause(platform, now, jitter)

    def pause(self, platform: str, now: datetime, jitter: float) -> datetime:
        """The next pause on the ladder; returns when it ends."""
        n = self.pauses.get(platform, 0)
        self.until[platform] = now + timedelta(
            seconds=jitter * self.ladder[min(n, len(self.ladder) - 1)])
        self.pauses[platform] = n + 1
        return self.until[platform]


@dataclass(frozen=True)
class Charged:
    """What a charge came to: the word for the log, and when the platform's pause ends if the
    charge paused it."""
    word: str
    paused_until: datetime | None = None


@dataclass(frozen=True)
class OutageState:
    streak: tuple[Item, ...] = ()
    in_outage: bool = False


def streak_key(item: Item) -> tuple:
    """An item's place in its platform's streak. A found source and a pair can share a URL, but
    never a table."""
    return (type(item.rows), item.key)


@dataclass(frozen=True)
class ChargePlan:
    word: str
    retry: Retry | None
    givebacks: tuple[Item, ...]
    streak: tuple[Item, ...]
    reset: bool
    pause: bool


def plan_charge(item: Item, outcome: Outcome, outage_state: OutageState,
                now: datetime, jitter: float) -> ChargePlan:
    """Decide a charge, its give-backs and whether the platform pauses, from values alone."""
    reset = outcome.answered and item.platform is not None
    state = OutageState() if reset else outage_state
    counted = outcome.outage and item.platform is not None
    if counted and state.in_outage:
        return ChargePlan("paused", None, (), state.streak, reset, True)
    kind = item.kind or outcome.failure
    failures, when = next_failure(item.prior.status, item.prior.failures, kind, now, jitter)
    retry = Retry(kind, failures, utc_iso(when) if when else None, outcome.detail[:500])
    streak = {streak_key(prior): prior for prior in state.streak}
    if counted:
        streak.setdefault(streak_key(item), item)
    pause = counted and len(streak) >= OUTAGE_STREAK
    items = tuple(streak.values())
    return ChargePlan(outcome.status, retry, items if pause else (),
                      () if pause else items, reset, pause)


def charge(conn: sqlite3.Connection, item: Item, outcome: Outcome, outages: Outages,
           now: datetime) -> Charged:
    """Gather the platform's streak, plan once, commit all rows, then apply the plan to the
    tracker (`Outages.apply`).

    The pause decision is made before writes. Applying it advances the platform's pause ladder
    with the same jitter sample as the item's retry. A failed write leaves the tracker intact.
    """
    sample = jitter()
    plan = plan_charge(item, outcome, outages.state(item.platform), now, sample)
    with write_tx(conn, site="quota_loop.charge"):
        if plan.retry is not None:
            item.rows.write(conn, item.key, plan.retry)
        for charged in plan.givebacks:
            charged.rows.write(conn, charged.key, charged.prior)
    until = (outages.apply(item.platform, plan, now, sample) if item.platform is not None
             else None)
    if until is not None:
        log(f"[scrapers] {item.platform}: outage — {len(plan.givebacks)} charges given back; "
            f"paused until {until:%H:%M:%S}")
    elif plan.retry is not None:
        retry = plan.retry
        when = parse_ts(retry.retry_at)
        log(f"[scrapers] {'/'.join(map(str, item.key))}: {retry.status} #{retry.failures}: "
            f"{outcome.detail} — " + (f"next try {when:%H:%M:%S}" if when else "giving up"))
    return Charged(plan.word, until)


def succeeded(conn: sqlite3.Connection, item: Item, outcome: Outcome, outages: Outages) -> None:
    """A unit that came back: its item's retry state is cleared, if there is any, and then its
    platform answered, if it did. The write comes first, as in `charge`, so a failed clear
    leaves the tracker as it was."""
    prior = item.prior
    if prior.failures or prior.retry_at or prior.last_error:
        item.rows.clear(conn, item.key)
    if outcome.answered and item.platform is not None:
        outages.answered(item.platform)
