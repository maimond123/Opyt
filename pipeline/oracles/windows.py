"""
pipeline/oracles/windows.py — how far back and how often an Oracle's sources are pulled.

Pure decisions, with `now` passed in: the host's lookback presets and what each resolves to, X's
six-month default and two-year ceiling, how far back an X pair is owed, the forward window of the
next pull, the flat jittered TTL that paces a pair, and the connectivity rule that keeps a walk
from claiming a range nobody read. Nothing here reads or writes the store.

X (an ephemeral stream: free to fetch, metered per post to ingest) and the web archive (a durable
corpus) use separate selectors. A single symmetric knob can only be wrong in one direction:
over-pull X, or truncate the archive. Each preset resolves against its own dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pipeline import jitter
from pipeline.timeparse import parse_ts, utc_iso
from pipeline.worker.quota_loop import retries

# ── Lookback presets ────────────────────────────────────────────────────────────
# Surfaced at onboarding. X is an ephemeral stream: short default, hard 2-year ceiling (the
# largest preset below; an ask can name no other), bounded by wall-clock and one session's rate
# limits.
# Substack/blog are a durable corpus: default to the whole archive. A preset maps to
# days-before-now; None = 'all' (no lower bound). Substack presets govern blog too.
X_LOOKBACK_PRESETS: dict[str, int] = {"6mo": 183, "1yr": 365, "2yr": 730}
WEB_LOOKBACK_PRESETS: dict[str, int | None] = {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
# Papers follow the web shape: a published corpus is durable, so no short default and no ceiling.
# 10yr is the extra rung a researcher's career needs.
SCHOLAR_LOOKBACK_PRESETS: dict[str, int | None] = {"2yr": 730, "5yr": 1825, "10yr": 3653,
                                                   "all": None}
_PRESETS = {"x": X_LOOKBACK_PRESETS, "web": WEB_LOOKBACK_PRESETS,
            "scholar": SCHOLAR_LOOKBACK_PRESETS}

# The one X selector that is not a fixed span: it answers "since I last pulled this person",
# a different date per Oracle, rather than a uniform lookback.
X_SINCE_LAST = "since_last"

# What an ask names when the host names nothing: six months of X, the whole web archive, every
# paper.
DEFAULT_X, DEFAULT_WEB, DEFAULT_PAPERS = "6mo", "all", "all"

# X's window when nobody asked for more: onboarding's own default. It is not the ceiling: two
# years belongs to a user who asks for it — `oracle(action='ingest', x_lookback='2yr')` — and an
# ask widens the target per pair (`target_for`).
X_DEFAULT_DAYS = 183
# The hard ceiling on any X window: `x_since` clamps rather than honors a longer ask.
X_CEILING_DAYS = 730


def since(kind: str, preset: str | None, now: datetime) -> datetime | None:
    """A host preset of `kind` ('x', 'web', 'scholar') → its `since`, or None.

    On X, None means "the adapter's own ~6-month default", so two inputs refuse rather than
    resolving: any silent fallthrough would turn a request for a narrow window into the widest
    pull available. `since_last` is not a window at all but a check of the newest posts
    (`oracles.admit`), and an unrecognised preset — the host supplies this string, so
    `'1y'` for `'1yr'` is one typo away — is a caller bug, not a request for six months.

    On the web and papers selectors None means "no lower bound", so they are lenient: `'all'`
    and an unknown preset both widen to the whole corpus, which is the safe direction there."""
    if not preset:
        return None
    presets = _PRESETS[kind]
    if kind == "x":
        if preset == X_SINCE_LAST:
            raise ValueError(f"{X_SINCE_LAST!r} is a per-Oracle check, not a window: "
                             f"`windows.parse` reads it")
        if preset not in presets:
            raise ValueError(f"unknown x_lookback {preset!r} — expected one of "
                             f"{sorted(presets)} or {X_SINCE_LAST!r}")
    days = presets.get(preset)
    return now - timedelta(days=days) if days else None


def x_since(since: datetime | None, now: datetime) -> datetime:
    """The lower bound an X pull actually uses: `since` (default ~6mo), floored at the hard
    2-year ceiling. However far back a caller asks, the window never exceeds `X_CEILING_DAYS` —
    a longer request is clamped, not honored. The consent report reads this too, so what the
    user is told cannot drift from the clamp."""
    since = since or (now - timedelta(days=X_DEFAULT_DAYS))
    return max(since, now - timedelta(days=X_CEILING_DAYS))


# ── An ask's stored window ──────────────────────────────────────────────────────
# The lower bound of an ask with none: 'all' on the web or in papers. An instant rather than NULL,
# because NULL already means "nothing was asked for this kind", and because the widen-only MIN in
# `record_ask` then needs no special case. 1900, not year 1: callers subtract overlaps from these
# instants, and `datetime.min` minus anything overflows.
ALL_TIME = utc_iso(datetime(1900, 1, 1, tzinfo=timezone.utc))


def window(instant: str | None) -> datetime | None:
    """An ask's stored window → an adapter's `since`. Everything (`ALL_TIME`) and nothing
    asked are both None: the adapters read None as the whole archive."""
    return None if instant in (None, ALL_TIME) else parse_ts(instant)


# ── One call's ask ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Windows:
    """What one call asks for, as `asks.record_ask` stores it. None: nothing asked of that kind."""
    x_from: datetime | None
    web_from: str | None
    scholar_from: str | None
    check: bool = False                 # `since_last`: the newest posts on X, as a check

    @property
    def asks(self) -> bool:
        return any(v is not None for v in (self.x_from, self.web_from, self.scholar_from))


def parse(x_lookback: str | None = None, web_lookback: str | None = None,
          scholar_lookback: str | None = None, *, now: datetime) -> Windows:
    """The host's presets as instants, read before anything is written so a bad preset refuses
    the whole call. Absent, they are the defaults: six months of X, the whole web archive, every
    paper. "All" is `ALL_TIME`, never None, which means nothing was asked of that kind. Raises
    `ValueError` on an X preset it does not know: on that selector a silent fallthrough would
    turn a narrow window into the widest pull. A top-up (`since_last`) asks for no other kind it
    does not name."""
    check = x_lookback == X_SINCE_LAST
    x_from = None if check else since("x", x_lookback or DEFAULT_X, now)

    def stored(kind: str, preset: str | None, default: str) -> str | None:
        if preset is None and check:
            return None
        start = since(kind, preset or default, now)
        return ALL_TIME if start is None else utc_iso(start)

    return Windows(x_from, stored("web", web_lookback, DEFAULT_WEB),
                   stored("scholar", scholar_lookback, DEFAULT_PAPERS), check)


# ── How far back an X pair is owed ──────────────────────────────────────────────
def deepen_target(now: datetime) -> datetime:
    """The instant an X pair is deepened to when nobody asked for more. One function so the
    selector, the ordering and the walk cannot disagree about what "deep enough" means."""
    return now - timedelta(days=X_DEFAULT_DAYS)


def target_for(row, now: datetime) -> datetime:
    """This pair's deepen target (`row` is a registry `SourceRow`): the clock default, widened
    by a window someone actually asked for (`requested_from`), never narrowed by one."""
    default = deepen_target(now)
    requested = parse_ts(row.requested_from)
    if requested is None:
        return default
    return min(requested, default)


def target(row, ask, now: datetime) -> datetime:
    """How far back this pair is owed: the oldest of its own ask, its Oracle's ask (`ask`, the
    Oracle's recorded ask or None) and six months, never past X's two-year ceiling."""
    t = target_for(row, now)
    asked = parse_ts(ask.x_from) if ask is not None else None
    if asked is not None and asked < t:
        t = asked
    return max(t, now - timedelta(days=X_CEILING_DAYS))


def owes_history(row, ask, now: datetime) -> bool:
    covered = parse_ts(row.covered_from)
    return covered is None or covered > target(row, ask, now)


# ── The forward window ──────────────────────────────────────────────────────────
# Re-ask a sliver behind the last pull; dedup absorbs the overlap.
OVERLAP = timedelta(hours=6)
# How far above `covered_from` a backward X walk starts. Tweet times are whole seconds and
# tool-posted threads can be sub-second apart; resuming exactly at the boundary could skip a
# same-second tweet. Read by `oracles/sources/timeline/walks.py`.
X_RESUME_OVERLAP = timedelta(minutes=10)


def since_for(row) -> datetime | None:
    """The lower bound to ask this pair for: its last successful pull, minus an overlap sliver.

    Falls back to `cursor_ts` (the newest atom we hold) when the pair has never been refreshed.
    None when we have neither."""
    base = parse_ts(row.last_pulled_at) or parse_ts(row.cursor_ts)
    if base is None:
        return None
    return base - OVERLAP


def leaves_a_gap(*, started_at_top: bool, partial: bool, reached: str | None,
                 last_pulled_at: str | None) -> bool:
    """Would recording this walk lose a range nobody read? — the connectivity rule.

    A walk that starts at the top and is cut short at R covered `[R, now]`. If the pair's previous
    top (`last_pulled_at`) is older than R, `[last_pulled_at, R)` was never read — yet
    `record_pull` takes the MIN of `covered_from`, so the old deep frontier would stand, and the
    stamp would move the forward window past the gap: lost for good. Such a walk keeps its atoms
    and neither stamps nor claims. `coverage.record_coverage` applies it too."""
    if not (started_at_top and partial):
        return False
    stopped, last = parse_ts(reached), parse_ts(last_pulled_at)
    return stopped is not None and last is not None and stopped > last


# ── What an Oracle's ask owes a web or papers pair ──────────────────────────────
def walking(row) -> bool:
    """A GitHub sweep that spent its ceiling and continues on its next unit."""
    return row.source_type == "github" and row.fwd_started_at is not None


def asked_from(row, ask) -> str | None:
    """The stored instant the ask names for this pair's kind: papers for OpenAlex, the web for
    the rest. None when nothing was asked of this kind."""
    if ask is None:
        return None
    return ask.scholar_from if row.source_type == "openalex" else ask.web_from


def owed_on_ask(row, ask) -> bool:
    """Not pulled since the Oracle's last ask, or a sweep for it still under way. Never for a
    kind the ask left out."""
    if asked_from(row, ask) is None:
        return False
    if walking(row):
        return True
    last, asked = parse_ts(row.last_pulled_at), parse_ts(ask.asked_at)
    return last is None or (asked is not None and last < asked)


def waits_for_an_ask(row, ask) -> bool:
    """Never pulled, on an Oracle whose ask left this kind out: its first pull is an ask's to
    make, not new posts'."""
    return ask is not None and row.last_pulled_at is None and asked_from(row, ask) is None


def first_pulls_done(rows, ask, now: datetime, exhausted: frozenset = frozenset()) -> bool:
    """Has every starting pair of one Oracle finished its first pull? `rows` are that Oracle's
    registered pairs (paused ones left out), `ask` its recorded ask, `exhausted` the
    `(source_type, source_key)` of pairs whose archive walk has given up on every unit it has left.
    Discovery's links pass waits for this, so it reads the Oracle's own posts once they are all in.

    | pair                                              | finished            |
    |---------------------------------------------------|---------------------|
    | given up, or stuck on pages that will not save    | yes                 |
    | its archive gave up on every post it has left     | yes                 |
    | X                                                 | reached its window  |
    | a web or papers pair                              | not owed on the ask |

    A post given up on stays owed in its walk, so the pair's own row never gives up: without the
    second row, one refused post held an Oracle's site discovery back for good.

    No pairs at all is finished: there is nothing to wait for. A paused X loop (a dead session)
    leaves its pair owing history, so it is not finished: the X probe needs that session too."""
    for row in rows:
        if retries.given_up(row.failures) or retries.stuck(row.last_status, row.failures):
            continue
        if (row.source_type, row.source_key) in exhausted:
            continue
        if row.source_type == "x":
            if owes_history(row, ask, now):
                return False
        elif owed_on_ask(row, ask):
            return False
    return True


def ask_since(row, ask) -> datetime | None:
    """The window a pull for the ask covers: the web window for a Substack or blog, the papers
    window for OpenAlex. GitHub has no window: its refresh window lets the `pushed_at` gate skip
    every repo untouched since the last pull, for free."""
    if row.source_type == "github":
        return since_for(row)
    return window(asked_from(row, ask))


# ── How often: flat per-type TTLs (hours) ───────────────────────────────────────
# x       — the freshest channel; a short TTL costs little.
# substack— a newsletter is bursty and low-frequency; the listing check itself is free.
# blog    — a fixed sitemap + hub-page crawl per refresh, so this is the one type whose request
#           count scales directly with how often we poll.
# github  — repos change slowly; ~2 API calls per refresh once the `pushed_at` gate is in.
FLAT_TTL_HOURS: dict[str, float] = {"x": 72.0, "substack": 168.0, "blog": 336.0, "github": 336.0}
DEFAULT_TTL_HOURS = 168.0          # an unknown source_type falls back to a week

# ±10% per-pair spread on the flat TTL, so pairs stamped together do not fall due together every
# cycle. Derived from the pair key, never drawn per call, so `is_stale` stays a pure function of
# stored state.
TTL_JITTER = 0.10


def ttl_hours(source_type: str) -> float:
    """The base TTL for a source type, before per-pair jitter. `pair_ttl_hours` is what gates."""
    return FLAT_TTL_HOURS.get(source_type, DEFAULT_TTL_HOURS)


def jitter_factor(canonical_id: str, source_type: str, source_key: str) -> float:
    """A stable multiplier in [1-TTL_JITTER, 1+TTL_JITTER], derived from the pair key alone.

    The body lives in `pipeline/jitter.py` (shared with the candidate probe); the NUL-joined pair
    key stays local since it's the registry's own identity."""
    return jitter.stable_factor(f"{canonical_id}\x00{source_type}\x00{source_key}", TTL_JITTER)


def pair_ttl_hours(row) -> float:
    """This pair's effective TTL: the flat per-type base, spread by its own stable jitter."""
    return ttl_hours(row.source_type) * jitter_factor(
        row.canonical_id, row.source_type, row.source_key)


def is_stale(row, now: datetime) -> bool:
    """Is this pair due? Never-pulled (or an unparseable stamp) is always stale; otherwise stale
    iff the hours since the last pull meet or exceed this pair's effective TTL."""
    last = parse_ts(row.last_pulled_at)
    if last is None:
        return True
    return (now - last).total_seconds() / 3600.0 >= pair_ttl_hours(row)


def staleness_hours(row, now: datetime) -> float:
    """How overdue a pair is, in hours past its TTL. Never-pulled sorts first (infinite). Used
    only to order the run so an interrupted pass drains the worst backlog — `is_stale` gates."""
    last = parse_ts(row.last_pulled_at)
    if last is None:
        return float("inf")
    return (now - last).total_seconds() / 3600.0 - pair_ttl_hours(row)


def x_question() -> dict:
    """The X window question the host asks after the picks, read off the preset dict itself so a
    new preset shows up in the question without anyone remembering to update a prompt.

    Only X is asked: it is the window that silently truncates. The web archive and papers pull in
    full by default. There is no bookmark selector: no tool takes one, and X has no bookmark
    timestamp to filter on."""
    spoken = [_spoken(p) for p in X_LOOKBACK_PRESETS]
    choices = f"{', '.join(spoken[:-1])}, or {spoken[-1]}" if len(spoken) > 1 else spoken[0]
    return {"presets": list(X_LOOKBACK_PRESETS), "default": DEFAULT_X,
            "ask": (f"How far back should I pull their X posts: {choices}? "
                    f"({_spoken(DEFAULT_X).capitalize()} is quickest.)")}


def _spoken(preset: str) -> str:
    """A preset as the user hears it: "6mo" → "6 months", "1yr" → "1 year"."""
    n, unit = int(preset[:-2]), {"mo": "month", "yr": "year"}[preset[-2:]]
    return f"{n} {unit}{'' if n == 1 else 's'}"
