"""
pipeline/kb/oracle_refresh.py — keep every confirmed Oracle's sources GROWING after onboarding.

Before this module, an Oracle's footprint was fetched once at onboarding and never re-pulled,
so `engagements` rows (written only inside the X pull) stopped accruing after day one. This
module is the loop: it runs from its own detached subprocess, refreshes stale pairs worst-lag-
first, and reports a bounded spend.
full background.

Four composed safety mechanisms, each catching a different failure:
  • STALENESS gate    — a pair inside its flat TTL is skipped for free (`oracle_refresh_state`).
  • WINDOW assertion  — a METERED pair whose computed `since` is absurdly old (or absent) is
                        refused before any spend; guards against a threading bug reintroducing
                        the adapter's 183-day default on every pull. The X FETCH is free since
                        2026-08-30, but a wide window still lands hundreds of atoms and every one
                        of them is OCR-VLM'd and embedded — so the window still bounds real
                        spend, derived rather than metered.
  • Daily ceiling     — a generous, resetting seatbelt for this, the first unattended spending
                        loop on the atom rail (consent otherwise lives at deposit, not per-use).
  • CIRCUIT breaker   — 3 consecutive errors open a 7-day breaker per pair, guarding against a
                        BROKEN endpoint (dead/private/renamed handle), not against spend.

Deliberately NOT here: the legacy empty-backoff, which throttled a working endpoint by how much
it produced. An empty X pull costs nothing real on the atom rail, so backing off would only cost
freshness for no saving.

Consent: a dedicated `oracle_refresh_consent` marker. Marker-only, NO auto-grant — refresh is a
RECURRING cost, not a one-time backlog import, so an established store must still opt in once.

Never raises. A refresh failure is reported, never propagated.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pipeline.timeparse import utc_now
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb.rail_runtime import (load_rail_env, models_unroutable,
                                      rail_budget_exhausted, spawn_rail)
from pipeline import llm_client
from pipeline.circuit_breaker import CircuitBreaker
from pipeline.dedup_store import record_health
from pipeline.ingestion.utils import log

from . import ingest_common
from . import oracle_refresh_state as st

# ── Tuning knobs ────────────────────────────────────────────────────────────────
OVERLAP_HOURS = 6.0                  # re-ask a sliver behind the cursor; dedup absorbs the overlap
MAX_REFRESH_WINDOW_DAYS = 45         # the window assertion's ceiling (METERED sources only)
ORACLE_REFRESH_DAILY_USD = 1.00      # resetting daily seatbelt — ~30x expected spend at 25 Oracles

# The rail label this ceiling governs. ONE constant, read by BOTH the `@llm_client.rail` decorator
# on `run_oracle_refresh` and by `_daily_budget_exhausted` below. Two matching string literals
# would drift, and a drifted pair fails SILENTLY: spend fills the meter under one name while the
# ceiling reads an empty meter under the other, so the gate never trips and looks fine.
RAIL = "oracle_refresh"
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 7 * 24 * 3600   # 7d — a dead handle is re-trialed weekly, not per-session
MAX_PAIRS_PER_RUN = 8                # see refresh_all: a backlog drain, not a safety afterthought

# Fraction of tracked pairs overdue above which the freshness notice fires. A ratio rather than
# an absolute count so it means the same thing at 8 Oracles or 500; 0.5 means the refresh cycle
# has stretched past twice its target TTL.
STALE_FRACTION_ATTENTION = 0.5

# Bound that replaces the window assertion when `force` waives it (since `since` may then be
# None, which would otherwise reach the adapter as its 183-day default). Bounds the DERIVED
# per-atom spend (OCR-VLM, artifact fetches, embeddings); the fetch itself is free and
# `_FETCH_CAP` bounds its volume. 300 ≈ one onboarding footprint, so a forced pull costs at most a
# repeat of it.
FORCED_X_ATOM_LIMIT = 300

# The window assertion applies ONLY to X, and the reason CHANGED on 2026-08-30 without the rule
# changing. It used to be that X was billed per tweet returned, so the window WAS the bill. The
# X fetch is free now — but a wide window still lands hundreds of new atoms, and every one of them
# is OCR-VLM'd and embedded. So the window still bounds real spend; it is just derived spend now.
#
# Substack/blog/github stay out for the same reason as before: a wide `since` costs them nothing
# extra, because their bound is the snapshot-hash skip rather than the window. Refusing them would
# be a livelock with no saving behind it — the pair could never advance `last_pulled_at`, so it
# could never stop being refused.
METERED_SOURCES = frozenset({"x"})

# Statuses that spent nothing and therefore must NOT consume one of `MAX_PAIRS_PER_RUN`.
# Without this a single permanently-refused pair sorts first (worst-lag-first) every run and
# starves the whole roster behind it.
_FREE_STATUSES = frozenset({"fresh", "breaker_open", "window_refused"})


# ── Consent ─────────────────────────────────────────────────────────────────────
def _consent_marker() -> Path:
    """Resolve the marker path at call time so it honors `$OPYT_HOME` (Distributable: derive
    paths at runtime). A path bound at import points at the wrong home under a sandboxed
    `$OPYT_HOME` and, in tests, at the real one."""
    return Path(os.environ.get("OPYT_ORACLE_REFRESH_CONSENT",
                               opyt_path("oracle_refresh_consent")))


def consented() -> bool:
    """Has the user opted into paid auto-refresh of their Oracles? Marker-only. Unlike the
    bookmark-catchup handshake we do NOT auto-consent an established store: this is a
    recurring cost, not a one-time backlog import."""
    return _consent_marker().exists()


def grant_consent() -> None:
    try:
        marker = _consent_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass


def revoke_consent() -> None:
    """Opt back out. A toggle should toggle both ways; afterwards the loop returns
    `needs_consent` and spends nothing."""
    try:
        _consent_marker().unlink(missing_ok=True)
    except OSError:
        pass


# ── Spend accounting ────────────────────────────────────────────────────────────
def _spend_probe() -> float:
    """This rail's recorded spend today. Fail-safe: an unreadable stats module reads 0.

    It used to return a PAIR, the second half being twitterapi.io's lifetime recorded spend, so a
    bracketing read could subtract that provider's per-REQUEST guardrail estimate back out and
    substitute its real per-TWEET price. The provider was removed on 2026-08-30, the X fetch is
    free, and both halves of that correction now compute zero — so the correction went with it.

    Per-rail rather than the global total, and not only to satisfy the guard. `run_oracle_refresh`
    can be called IN-PROCESS by the MCP server, where `hopper` may spend during the same window —
    against `spend_today()` that spend would land in this pull's delta and be reported as the cost
    of refreshing an Oracle. In a detached child the two figures are identical, so this is free."""
    try:
        return llm_client.spend_today_for_rail(RAIL)
    except Exception:
        return 0.0


def _daily_budget_exhausted() -> bool:
    """Has this rail's recorded spend today reached its ceiling? See
    `rail_runtime.rail_budget_exhausted` for why it is never the global total."""
    return rail_budget_exhausted(RAIL, ORACLE_REFRESH_DAILY_USD)


# ── The window ──────────────────────────────────────────────────────────────────
def since_for(row: st.SourceRow, now: datetime | None = None) -> datetime | None:
    """The lower bound to ask this pair for: its last successful pull, minus an overlap sliver.

    Falls back to `cursor_ts` (the newest atom we hold) when the pair has never been refreshed —
    which is the normal state for an Oracle onboarded before `set_oracle_window` went live, and
    is still a sane, corpus-derived window. None when we have neither, i.e. we know nothing about
    what has already been paid for."""
    base = st.parse_ts(row.last_pulled_at) or st.parse_ts(row.cursor_ts)
    if base is None:
        return None
    return base - timedelta(hours=OVERLAP_HOURS)


def never_pulled(row: st.SourceRow) -> bool:
    """Has this pair NEVER been pulled — no coverage stamp, no atoms, no recorded attempt?

    All three, because any one alone is ambiguous. `last_pulled_at` is NULL on a pair whose pulls
    all failed; `cursor_ts` is NULL for an Oracle who has genuinely posted nothing; `last_status`
    is NULL only before the loop has ever reached this pair at all. Together they mean the pair
    has no history of any kind, which is a different fact from "its history is missing"."""
    return (row.last_pulled_at is None and row.cursor_ts is None and row.last_status is None)


def window_ok(row: st.SourceRow, since: datetime | None, now: datetime) -> bool:
    """Is this window safe to spend on without an explicit force?

    Unmetered sources always pass (see `METERED_SOURCES`). A metered source passes with a window
    that both EXISTS and is inside `MAX_REFRESH_WINDOW_DAYS`.

    ⚠️ A never-pulled pair also passes, and leaving it out was a DEADLOCK. `since_for` returns
    None when there is no `last_pulled_at` and no `cursor_ts`; refusing on that meant the pair was
    refused every run, forever, because a pull is the only thing that writes either field and the
    refusal is what prevented the pull. It is reachable without anything exotic — confirm an
    Oracle whose first X pull fails outright (suspended account, expired cookie, an interrupted
    onboarding) and the rail can never pick it up again. Note `refresh_all` seeds `oracle_sources`
    for every confirmed Oracle, so the ROW was always there; it was the window that was missing.

    The assertion still does its job on every pair that HAS history. It guards against a threading
    bug dropping `since`, which silently becomes the adapter's 183-day default instead of an
    error. A pair with no history has no incremental window to lose, so there the default IS the
    right answer. `since is None` therefore means two different things, and only one is a fault:
    "start from the beginning" on a first pull, "something dropped my window" on any other.

    A first pull is not unbounded, either. `refresh_pair` treats it exactly like the `--force`
    waiver, swapping the window assertion for `FORCED_X_ATOM_LIMIT` — with the window guard
    waived, the atom cap is what is left holding the derived spend."""
    if row.source_type not in METERED_SOURCES:
        return True
    if since is None:
        return never_pulled(row)
    return (now - since) <= timedelta(days=MAX_REFRESH_WINDOW_DAYS)


# ── Dispatch ────────────────────────────────────────────────────────────────────
def _dispatch(conn, embedder, row: st.SourceRow, since, author_name, *, x_limit: int = 0):
    """Run ONE pair's adapter. Returns `(summary, outcome, note)` where `outcome` is one of
    `ingest_common.RUN_*` or the string `"skipped"` (the eligibility gate refused).

    X goes DIRECTLY to `sync_x_footprint` — an X timeline is single-author by construction, so
    there is no gate to route through. The other three go through `expand._route_source`, which
    is the single gated door for the two website adapters (free on a cache hit: `source_authorship`
    is cached forever) and which already classifies the adapter's returned summary.

    `x_limit` (0 = unbounded) is the forced-wide-window bound; see `FORCED_X_ATOM_LIMIT`. It is
    passed ONLY on the X path because only X is window-asserted, so only X can be widened."""
    from . import expand

    if row.source_type == "x":
        summ = ingest_x_footprint_sync(conn, embedder, handle=row.source_key,
                                       author_name=author_name, since=since, limit=x_limit)
        return summ, ingest_common.classify_run(summ), None

    url = (f"https://github.com/{row.source_key}" if row.source_type == "github"
           else row.source_key)
    entry = expand._route_source(
        conn, embedder, {"source_type": row.source_type, "url": url},
        author_name=author_name, limit=0,
        web_since=since if row.source_type in ("substack", "blog") else None,
        github_since=since if row.source_type == "github" else None,
    )
    if entry.get("skipped"):
        return {}, "skipped", f"{entry.get('skipped')}: {entry.get('reason') or ''}".strip()
    if entry.get("blocked") is not None:
        return entry["blocked"], ingest_common.RUN_BLOCKED, entry.get("reason")
    if entry.get("error"):
        return {}, ingest_common.RUN_ERROR, str(entry["error"])
    return entry.get("ingested") or {}, ingest_common.RUN_INGESTED, None


def ingest_x_footprint_sync(conn, embedder, **kw):
    """Indirection so a test can fake the X pull without importing the (heavy) adapter, and so the
    call site above stays readable. Imported lazily for the same reason every other adapter here
    is: `pipeline.kb.oracle_refresh` is imported by the MCP tool surface at request time."""
    from . import ingest_x_footprint
    return ingest_x_footprint.sync_x_footprint(conn, embedder, **kw)


# ── The unit: refresh ONE pair ──────────────────────────────────────────────────
def refresh_pair(conn, embedder, row: st.SourceRow, *, force: bool = False,
                 now: datetime | None = None) -> dict:
    """staleness → window → breaker → pull → classify → persist, for one (Oracle, source) pair.

    The four write-outcomes differ only in what they persist:
      ingested/empty — a real observation: advance the cursor, stamp `last_pulled_at`, TTL restarts.
      blocked        — a host stopped us: nothing was written and nothing marked seen, so neither
                       the cursor nor the stamp moves. A Cloudflare shell is NOT an author who went
                       quiet, and stamping would buy one bad night a full TTL of silence.
      error          — the pull raised: breaker records it, no advance, no stamp; retried next run.
      fresh / window_refused / breaker_open — nothing ran, nothing spent.

    `force` here overrides BOTH the staleness gate and the window assertion — this is the unit,
    and a caller that names one pair explicitly has said what it wants. `refresh_all` deliberately
    does NOT widen its selection the same way; see its docstring for why.

    A waived window does not become an unbounded one: the waiver SWAPS the window assertion for
    `FORCED_X_ATOM_LIMIT`, so the widest possible pull still lands on a bound the user has already
    paid once. Forcing a pair whose window was fine changes nothing — it is not the widened case,
    and capping it would only disable the media prefetch for no saving.

    A pair's FIRST pull takes that same waiver and that same bound, without needing `force`. It
    has no window to assert against, and refusing it was a deadlock — see `window_ok`.
    """
    now = now or utc_now()

    if not force and not st.is_stale(row, now):
        return {"status": "fresh", **_ident(row)}

    since = since_for(row, now)
    # A first pull runs on the adapter's default window, which nothing here chose — so it takes
    # the same atom bound `--force` takes, for the same reason. `wide` is what carries that bound
    # down to `x_limit` below; it is not a claim that anything was refused.
    first = never_pulled(row) and row.source_type in METERED_SOURCES
    wide = first or not window_ok(row, since, now)
    if wide and not force and not first:
        detail = ("no recorded coverage window" if since is None
                  else f"since {since:%Y-%m-%d} is older than the "
                       f"{MAX_REFRESH_WINDOW_DAYS}-day refresh ceiling")
        st.record_pull(conn, row, last_status="window_refused", stamp=False)
        log(f"[oracle-refresh] {row.source_type}:{row.source_key} REFUSED — {detail}")
        return {"status": "window_refused", **_ident(row), "reason": detail,
                "remedy": "The wider-window override is CLI-only since 2026-08-15: "
                          "python -m pipeline.kb.oracle_refresh --once --force. To pull "
                          "just this Oracle now, call oracle(action='ingest', "
                          "canonical_ids=[...], x_lookback='6mo')."}

    x_limit = FORCED_X_ATOM_LIMIT if wide else 0
    bound = {"atom_limit": x_limit} if x_limit else {}
    if x_limit:
        why = "FIRST pull, no prior coverage" if first else "FORCED past the window assertion"
        log(f"[oracle-refresh] {row.source_type}:{row.source_key} {why} — bounded to "
            f"{x_limit} new atoms")

    service = f"oracle-refresh:{row.canonical_id}:{row.source_type}"
    breaker = CircuitBreaker(service, threshold=BREAKER_THRESHOLD, cooldown=BREAKER_COOLDOWN_S)

    # `breaker.allow()` + explicit outcome recording, NOT `breaker.call(...)`.
    # `breaker.call` counts a failure only when the callable RAISES — and these adapters signal a
    # hard stop by RETURNING a summary carrying `error` (a raise would sink the caller's other
    # sources). Wrapped in `call`, a handle that is 403-ing on every single pull would be recorded
    # as three consecutive SUCCESSES and the breaker would never open, which is precisely the
    # half-of-the-contract blindness `classify_run` was written to end.
    if not breaker.allow():
        return {"status": "breaker_open", **_ident(row),
                "retry_after_s": round(breaker.retry_after(), 1)}

    t0 = _spend_probe()
    try:
        summary, outcome, note = _dispatch(conn, embedder, row, since, row.name,
                                           x_limit=x_limit)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        breaker.record_failure(detail)
        record_health(service, ok=False, detail=detail)
        st.record_pull(conn, row, last_status="error", stamp=False)
        log(f"[oracle-refresh] {row.source_type}:{row.source_key} pull FAILED: {detail}")
        return {"status": "error", **_ident(row), "error": detail, **_cost(t0, {})}

    added = int(summary.get("added") or 0)
    costs = _cost(t0, summary)

    if outcome in (ingest_common.RUN_BLOCKED, ingest_common.RUN_ERROR):
        detail = note or str(summary.get("error"))
        # BLOCKED counts toward the breaker as much as ERROR does. Both mean "this endpoint is not
        # serving us"; the breaker's job is to stop calling a broken one, and a Cloudflare wall
        # that turns us away every night is the textbook case for a 7-day cooldown.
        breaker.record_failure(detail)
        record_health(service, ok=False, detail=detail)
        blocked = outcome == ingest_common.RUN_BLOCKED
        st.record_pull(conn, row, last_status="blocked" if blocked else "error", stamp=False)
        return {"status": "blocked" if blocked else "error", **_ident(row),
                ("reason" if blocked else "error"): detail, **costs}

    breaker.record_success()
    record_health(service, ok=True)
    if outcome == "skipped":
        # The eligibility gate refused (multi-author / needs-review). A real, cheap observation —
        # stamp it so the TTL paces the re-check rather than re-asking every single run.
        st.record_pull(conn, row, last_status="skipped", stamp=True, now=st._now())
        return {"status": "skipped", **_ident(row), "reason": note, **costs}

    if x_limit and added >= x_limit:
        # Groups reach the consumer newest-first, so a truncated pull keeps the RECENT tail of the
        # window and drops the old end — the right truncation for a loop whose job is currency, and
        # the reason this is reported rather than resumed. Refresh keeps an Oracle CURRENT; the
        # backfill tool is `oracle(action='ingest')`, which is a deliberate one-time spend. Saying
        # so is the whole point: a silent cap reads identically to a window that simply held 300.
        bound["truncated"] = True
        bound["note"] = (f"hit the {x_limit}-atom forced-pull bound — the oldest part of this "
                         f"window was not ingested; use oracle(action='ingest') to backfill")
        log(f"[oracle-refresh] {row.source_type}:{row.source_key} TRUNCATED at {x_limit} atoms")

    status = "ingested" if added > 0 else "empty"
    _, who_ids = st.pairs_for_oracle(conn, row.canonical_id)
    cursor = st.latest_atom_ts(conn, row.source_type, who_ids) or row.cursor_ts
    st.record_pull(conn, row, last_status=status, cursor_ts=cursor, stamp=True, now=st._now())
    # The adapter's own counters ride along. Without them the incrementality seams are INVISIBLE
    # in a normal run: `stale: 258` (the GitHub calls the `pushed_at` gate avoided) and `fetched`
    # (the tweets we were actually billed for) both stop at the adapter otherwise, and a report
    # that cannot show the saving cannot show a regression in it either.
    return {"status": status, **_ident(row), "new_atoms": added,
            "engagements": int(summary.get("engagements") or 0),
            "cursor_ts": cursor, "stats": ingest_common.run_stats(summary), **costs, **bound}


def _ident(row: st.SourceRow) -> dict:
    return {"canonical_id": row.canonical_id, "name": row.name,
            "source_type": row.source_type, "source_key": row.source_key}


def _cost(t0: float, summary: dict) -> dict:
    """The pair's spend: the api_stats delta over the pull.

    Both keys are kept and both now carry the same number. `cost_usd` used to differ — it swapped
    twitterapi.io's per-REQUEST guardrail estimate back out for the real per-TWEET price, because
    the guardrail over-counted a short page by up to 20x. With the X fetch free, every dollar in
    this delta is derived spend (OCR-VLM, artifact fetches, embeddings) that the meter already
    records honestly, so there is nothing left to correct. They stay two keys because consumers
    read them, and collapsing them is a surface change, not this change."""
    recorded = max(0.0, _spend_probe() - t0)
    return {"cost_usd": round(recorded, 6), "cost_usd_recorded": round(recorded, 6)}


# ── The loop ────────────────────────────────────────────────────────────────────
def refresh_all(conn, embedder, *, force: bool = False, only=None,
                max_pairs: int = MAX_PAIRS_PER_RUN, now: datetime | None = None) -> dict:
    """Refresh every stale pair, worst-lag-first, bounded. Never raises.

    `max_pairs` is LOAD-BEARING, not a safety afterthought. On the very first run after seeding,
    every pair's `last_pulled_at` comes from the Oracle's onboarding coverage marker — so any
    Oracle onboarded more than a TTL ago is stale immediately and the WHOLE roster comes due at
    once. The bound turns that burst into a backlog that drains a few pairs per session, and
    `deferred` reports the remainder so it is visible rather than silently truncated.

    It bounds paid attempts, not iterations: a pair that returns fresh / breaker_open /
    window_refused spent nothing, so it does not consume a slot. Otherwise one permanently-refused
    pair would sort first every run and starve everything behind it.

    `force` authorizes a wider window, it does NOT bypass the TTL. Selection stays
    staleness-gated even under force, because those are two different asks and only one of them
    is worth money: re-pulling a pair refreshed an hour ago buys nothing (it is current by
    definition), while unblocking a window-refused pair is the entire reason the flag exists. The
    distinction matters because `force=True` is also how consent is granted — so if it meant
    "ignore every gate", the very first opt-in call would re-pull the whole roster over its
    widest possible window, which is the one moment a user has least idea what they just bought.
    """
    now = now or utc_now()
    st.seed_from_entities(conn, canonical_ids=only)

    rows = [r for r in st.list_sources(conn, canonical_ids=only) if r.status != "paused"]
    stale = [r for r in rows if st.is_stale(r, now)]
    stale.sort(key=lambda r: st.staleness_hours(r, now), reverse=True)   # worst lag first

    agg = {"status": "ok", "registered": len(rows), "considered": len(stale),
           "refreshed": 0, "empty": 0, "blocked": 0, "errors": 0, "breaker_open": 0,
           "window_refused": 0, "skipped": 0, "deferred": 0,
           "new_atoms": 0, "engagements": 0,
           "cost_usd": 0.0, "cost_usd_recorded": 0.0, "by_oracle": {}, "results": []}
    attempted = 0

    for row in stale:
        if attempted >= max_pairs:
            agg["deferred"] += 1
            continue
        if _daily_budget_exhausted():
            agg["status"] = "budget_paused"
            agg["deferred"] += 1
            continue
        try:
            r = refresh_pair(conn, embedder, row, force=force, now=now)
        except Exception as e:                  # a pair must never sink the run
            detail = f"{type(e).__name__}: {e}"
            log(f"[oracle-refresh] {row.source_type}:{row.source_key} unhandled: {detail}")
            r = {"status": "error", **_ident(row), "error": detail}

        status = r["status"]
        if status not in _FREE_STATUSES:
            attempted += 1
        agg["results"].append(r)
        agg["new_atoms"] += r.get("new_atoms", 0)
        agg["engagements"] += r.get("engagements", 0)
        agg["cost_usd"] += r.get("cost_usd", 0.0)
        agg["cost_usd_recorded"] += r.get("cost_usd_recorded", 0.0)
        key = {"ingested": "refreshed", "empty": "empty", "blocked": "blocked",
               "error": "errors", "breaker_open": "breaker_open",
               "window_refused": "window_refused", "skipped": "skipped"}.get(status)
        if key:
            agg[key] += 1
        bucket = agg["by_oracle"].setdefault(row.canonical_id,
                                             {"name": row.name, "sources": {}})
        bucket["sources"][f"{row.source_type}:{row.source_key}"] = status

    agg["cost_usd"] = round(agg["cost_usd"], 6)
    agg["cost_usd_recorded"] = round(agg["cost_usd_recorded"], 6)
    if agg["status"] == "budget_paused":
        agg["message"] = (f"stopped after ${agg['cost_usd_recorded']:.4f} — today's recorded API "
                          f"spend reached the ${ORACLE_REFRESH_DAILY_USD:.2f} daily refresh "
                          f"ceiling. It resets at UTC midnight; the backlog drains next run.")
    record_health("oracle-refresh", ok=(agg["errors"] == 0),
                  detail=None if agg["errors"] == 0 else f"{agg['errors']} pair(s) errored")
    log(f"[oracle-refresh] {agg['considered']} stale, {agg['refreshed']} refreshed, "
        f"{agg['empty']} empty, {agg['deferred']} deferred, {agg['new_atoms']} new atoms, "
        f"${agg['cost_usd']:.4f} real (${agg['cost_usd_recorded']:.4f} recorded)")
    return agg


# ── Status ──────────────────────────────────────────────────────────────────────
def status_summary(conn=None, canonical_ids=None) -> dict:
    """A read-only freshness snapshot — per-Oracle, per-source. All DERIVED: this loop writes no
    log of its own, `oracle_sources` + `circuit_breaker` ARE the log.

    A FROZEN Oracle must be visible HERE. That bug — nothing ever re-pulling an Oracle — was
    invisible for months precisely because no surface reported per-source staleness; this is
    where that stops. Never raises: a missing table degrades to an empty snapshot."""
    own = conn is None
    if own:
        conn = st.connect()
    try:
        rows = st.list_sources(conn, canonical_ids=canonical_ids)
    except Exception as e:
        return {"consented": consented(), "error": f"{type(e).__name__}: {e}",
                "tracked_pairs": 0, "oracles": []}
    finally:
        if own:
            conn.close()

    try:
        from pipeline.circuit_breaker import status as breaker_status
        open_breakers = {b["service"] for b in breaker_status()
                         if b["service"].startswith("oracle-refresh:") and b["state"] != "closed"}
    except Exception:
        open_breakers = set()

    now = utc_now()
    by_oracle: dict = {}
    overdue = 0
    for r in rows:
        lag = st.staleness_hours(r, now)
        stale = st.is_stale(r, now)
        overdue += 1 if stale else 0
        entry = by_oracle.setdefault(r.canonical_id,
                                     {"canonical_id": r.canonical_id, "name": r.name,
                                      "sources": [], "stale_sources": 0})
        entry["sources"].append({
            "source_type": r.source_type, "source_key": r.source_key,
            "last_pulled_at": r.last_pulled_at,
            # The EFFECTIVE ttl (base × this pair's stable jitter), not the base — otherwise the
            # report and the gate disagree, and `hours_overdue` reads off by up to 10%.
            "ttl_hours": round(st.pair_ttl_hours(r), 1),
            "hours_overdue": None if lag == float("inf") else round(lag, 1),
            "never_refreshed": r.last_pulled_at is None,
            "last_status": r.last_status, "stale": stale,
            "breaker_open": f"oracle-refresh:{r.canonical_id}:{r.source_type}" in open_breakers,
        })
        entry["stale_sources"] += 1 if stale else 0

    out = {
        "consented": consented(),
        "tracked_pairs": len(rows),
        "stale_pairs": overdue,
        "last_refreshed_at": max((r.last_pulled_at for r in rows if r.last_pulled_at),
                                 default=None),
        "ttl_hours": dict(st.FLAT_TTL_HOURS),
        "oracles": sorted(by_oracle.values(), key=lambda e: -e["stale_sources"]),
    }
    out["stale_fraction"] = round(overdue / len(rows), 3) if rows else 0.0

    # The verdict lives here, not at the call sites — the same rule `curation_state.status_summary`
    # follows, for the reason it states: re-deriving "stale" per surface is how two surfaces end up
    # disagreeing about the same store on the same day.
    #
    # Two conditions, both scale-free, so they mean the same thing at 8 Oracles and at 500:
    #   • never consented  — the loop has no ENTRANCE. This is not a slow loop, it is a closed one,
    #     and it is the defect that hid for months: the spawner fired every session, the rail
    #     refused, and the refusal went to a log nobody reads.
    #   • the cycle stretched past 2x TTL — the loop is running and losing ground.
    # Guarded on `tracked_pairs`: a store with no Oracles yet must stay silent. Telling a fresh
    # install that Oracle refresh is off is noise on the first surface a new user ever sees.
    out["needs_attention"] = bool(
        rows and (not out["consented"]
                  or out["stale_fraction"] > STALE_FRACTION_ATTENTION))

    if not out["consented"] and rows:
        out["note"] = ("Automatic Oracle refresh is OFF — nothing re-pulls your Oracles, so their "
                       "content is frozen at the day each was added. Run `onboard` to opt in; it "
                       "asks once and explains what the recurring cost is.")
    elif out["needs_attention"]:
        out["note"] = (f"{overdue} of {len(rows)} Oracle sources are overdue — your refresh cycle "
                       f"has stretched past twice its target. The loop drains "
                       f"{MAX_PAIRS_PER_RUN} sources per session open, so opening a session more "
                       f"often is the free fix. To pull one person now, call "
                       f"oracle(action='ingest', canonical_ids=[...], x_lookback='since_last').")
    return out


# ── The entrypoint ──────────────────────────────────────────────────────────────
@llm_client.rail(RAIL)
def run_oracle_refresh(force: bool = False, *, only=None,
                       max_pairs: int = MAX_PAIRS_PER_RUN) -> dict:
    """Load creds → gate on consent → single-flight → refresh. Never raises.

    The rail label goes on `run_*` and NOT on `main()` — `main()` only wraps this for the
    `--once` child, so labelling it would miss every in-process call the MCP side makes directly.

    `force=True` grants consent and runs now (the manual escape hatch, matching
    `run_people_refresh`). An un-consented, non-forced call returns `needs_consent` having spent
    nothing — before opt-in the background spawn is a free no-op."""
    load_rail_env()

    if force:
        grant_consent()
    elif not consented():
        return {"status": "needs_consent",
                "message": ("Automatic refresh of your Oracles uses API credits. The X pull "
                            "itself is free — it runs on your own browser session — but reading "
                            "each new post costs: the content-quality gate, image descriptions, "
                            "and embeddings. Run `onboard` once to opt in — it asks for this "
                            "alongside the backlog import and explains that this half is "
                            "recurring. After that it keeps them current in the background.")}

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock
    try:
        with CatchupLock("oracle-refresh") as lock:
            if not lock.acquired:
                # Another session's refresh holds the lease. Skipping is correct, not a failure:
                # the work is identical and the other holder is already doing it.
                return {"status": "already_running",
                        "message": "another Oracle refresh is in flight — skipped (single-flight)."}
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            conn = st.connect()
            try:
                return refresh_all(conn, embedder, force=force, only=only, max_pairs=max_pairs)
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[oracle-refresh] run_oracle_refresh errored: {detail}")
        return {"status": "error", "error": detail}


# ── The detached spawn ──────────────────────────────────────────────────────────
def spawn_oracle_refresh(force: bool = False, coalesce_window: float = 600.0) -> bool:
    """Fire the refresh as a detached, non-blocking child and return immediately.

    This owns its own spawner rather than hooking into another rail's, so its lifetime is never
    welded to an unrelated module's — each rail owns its spawner (see the
    `atom-rail-not-welded-to-catchup` guard).

    Trigger rate is not pull rate: the 600s coalesce governs how often the loop ASKS whether
    anything is due, the per-pair TTL governs whether anything actually pulls. A session opened
    minutes after another forks nothing; one opened hours later forks, finds nothing due, and
    exits having spent $0.

    The log fd is closed in the parent right after `Popen` (the child has already dup'd it), and
    the stamp is touched only after `Popen` succeeds so a failed fork doesn't burn the 600s window.
    """
    return spawn_rail("pipeline.kb.oracle_refresh", slug="oracle_refresh",
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    """The rail's CLI, matching the flag shape of the other rails: `--once` required (no bare
    invocation fires a spending rail), `--force` is a real flag rather than a positional word. See
    """
    import argparse
    ap = argparse.ArgumentParser(
        description="Oracle refresh — bring each confirmed Oracle's sources current "
                    "(the X pull is free; the content-quality gate and embeds are metered)")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--force", action="store_true",
                    help="grant consent and ignore the per-pair TTL. Still single-flight.")
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_RUN,
                    help=f"cap on source pairs refreshed per run (default {MAX_PAIRS_PER_RUN})")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_oracle_refresh(force=args.force, max_pairs=args.max_pairs)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "already_running", "needs_consent"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
