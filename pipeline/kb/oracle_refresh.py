"""
pipeline/kb/oracle_refresh.py — keep every confirmed Oracle's sources GROWING after onboarding.

Before this module, an Oracle's footprint was fetched once at onboarding and never re-pulled,
so `engagements` rows (written only inside the X pull) stopped accruing after day one. This
module is the loop: it runs from its own detached subprocess, refreshes stale pairs worst-lag-
first, and reports a bounded amount of work.

Four composed safety mechanisms, each catching a different failure:
  • STALENESS gate    — a pair inside its flat TTL is skipped for free (`oracle_refresh_state`).
  • WINDOW assertion  — a METERED pair whose computed `since` is absurdly old (or absent) is
                        refused before any work; guards against a threading bug reintroducing
                        the adapter's 183-day default on every pull. The X FETCH is free since
                        2026-08-30, but a wide window still lands hundreds of atoms and every one
                        of them is OCR-VLM'd and embedded — so the window still bounds real
                        work.
  • CIRCUIT breaker   — 3 consecutive errors open a 7-day breaker per pair, guarding against a
                        BROKEN endpoint (dead/private/renamed handle), not against repeated requests.

Deliberately NOT here: the legacy empty-backoff, which throttled a working endpoint by how much
it produced. An empty X pull has no new content, so backing off would only delay freshness.

One thing here is not a source refresh at all: `_recommendation_pass` reads each Substack
Oracle's PUBLISHED recommendations and lands `curation_signals`, not atoms. It sits on this rail
rather than `curation_catchup` because its input grows with the Oracle roster, which is what this
rail is about; it runs after both loops, on its own 24-hour clock, and spends no session.

Consent: a dedicated `oracle_refresh_consent` marker. Marker-only, NO auto-grant — refresh reads
the user's connected sessions in the background, so an established store must still opt in once.

Never raises. A refresh failure is reported, never propagated.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pipeline.timeparse import utc_now
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb.rail_runtime import load_rail_env, models_unroutable
from pipeline.circuit_breaker import CircuitBreaker, CircuitOpenError
from pipeline.ingestion.utils import log

from pipeline.ingestion.utils import SyncAuthError

from . import ingest_common
from .frontier_sources import SourceError
from . import oracle_refresh_state as st


def _core():
    """`x_graphql_core`, imported lazily. This module is loaded by the MCP tool surface at request
    time, and the transport pulls in `curl_cffi` and the browser-cookie readers."""
    from pipeline.ingestion import x_graphql_core
    return x_graphql_core

# ── Tuning knobs ────────────────────────────────────────────────────────────────
OVERLAP_HOURS = 6.0                  # re-ask a sliver behind the cursor; dedup absorbs the overlap
MAX_REFRESH_WINDOW_DAYS = 45         # the window assertion's ceiling (METERED sources only)
RAIL = "oracle_refresh"
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 7 * 24 * 3600   # 7d — a dead handle is re-trialed weekly, not per-session
MAX_PAIRS_PER_RUN = 8                # see refresh_all: a backlog drain, not a safety afterthought

# Fraction of tracked pairs overdue above which the freshness notice fires. A ratio rather than
# an absolute count so it means the same thing at 8 Oracles or 500; 0.5 means the refresh cycle
# has stretched past twice its target TTL.
STALE_FRACTION_ATTENTION = 0.5

# ── Breadth, then depth ─────────────────────────────────────────────────────────
# The window a pair's FIRST rail pull covers. Deliberately shallow: one or two `UserTweets`
# requests, so a roster of fifteen never-pulled pairs uses ~15 requests to give EVERY Oracle
# recent coverage, not ~200 to give three of them complete coverage and twelve of them nothing.
#
# Uniform incompleteness is the point, and it is reportable: "I have all fifteen back to August"
# is a sentence the store can say, and `covered_from` is what lets it say it. The tradeoff, which the
# copy must surface rather than hide, is that for the first hours a question about someone's older
# position returns nothing — and nothing distinguishes that from their never having said it.
BREADTH_WINDOW_DAYS = 30

# How far back `backfill_pass` deepens a pair. 183 days = onboarding's own X default, so the rail
# finishes the job onboarding started and stops. It is NOT the 2-year ceiling: two years is ~70
# minutes of pulling for fifteen Oracles and belongs to a user who asks for it — 
# `oracle(action='ingest', x_lookback='2yr')` — not to a background pass nobody watched start.
BACKFILL_TARGET_DAYS = 183

# NO PRE-EMPTIVE RESERVE. `BACKFILL_MIN_BUDGET = 25` stood here until 2026-09-14, read by this
# module and by a hand-copied twin in `oracles.py`. Both existed for ONE reason —
# `_pull_own_timeline` was all-or-nothing, so a walk that ran dry mid-way spent requests and wrote
# nothing, and paying for a doomed walk was worse than refusing before it. Durable partial walks
# removed the premise, so both copies went together: keeping one would have recreated exactly the
# two-answers-to-one-question split that `test_the_reserve_is_the_refresh_rail_s_own_number` was
# written to prevent.
#
# `x_graphql_core._refuse_if_spent` is the bound and always was. It refuses a request x.com has
# already said it will 429 — evidence rather than prediction — and its absence up here is what
# finally makes `backfill_pass` match its own docstring: "until the meter runs dry".
#
# The reserve was also never a worst case. 25 is a TYPICAL walk (measured 12-25 requests for 183
# days, and 117 pages for one 6-month pull of a prolific account), so a deep pull on a high-volume
# handle could always start and still run dry — which is now simply the ordinary partial.

# The window assertion applies ONLY to X, and the reason CHANGED on 2026-08-30 without the rule
# changing. A wide X window can still land hundreds of new atoms, each of which is OCR-VLM'd and
# embedded, so the window remains a real work bound.
#
# Substack/blog/github stay out because their bound is the snapshot-hash skip rather than the
# window. Refusing them would be a livelock with no benefit — the pair could never advance `last_pulled_at`, so it
# could never stop being refused.
METERED_SOURCES = frozenset({"x"})

# Which sources a BACKWARD pass deepens. Deliberately NOT `METERED_SOURCES` above, and the two
# must not be merged: that set answers "does the window assertion apply", this one answers "can a
# pull reach further back than the last one did". GitHub answers no to the first (its bound is
# the snapshot-hash skip, and refusing it on window width would livelock the pair) and yes to the
# second, because `ingest_github` reads `pushed_at` out of the repo LIST and so can bound a sweep
# from both ends.
#
# Substack and blog stay out because neither adapter reports a frontier: their sweep is
# hash-bounded, not window-bounded, so there is nothing for a resume to pick up from.
DEEPENED_SOURCES = frozenset({"x", "github"})

# Statuses that made no request and therefore must NOT consume one of `MAX_PAIRS_PER_RUN`.
# Without this a single permanently-refused pair sorts first (worst-lag-first) every run and
# starves the whole roster behind it.
#
# `fresh` is deliberately absent: `refresh_all` selects on `st.is_stale(row, now)` and passes that
# same `now` down, so a pair that reaches `refresh_pair` from the loop can never come back fresh.
# Listing it would suggest the loop has a case it does not have.
_FREE_STATUSES = frozenset({"breaker_open", "window_refused"})


# ── Consent ─────────────────────────────────────────────────────────────────────
def _consent_marker() -> Path:
    """Resolve the marker path at call time so it honors `$OPYT_HOME` (Distributable: derive
    paths at runtime). A path bound at import points at the wrong home under a sandboxed
    `$OPYT_HOME` and, in tests, at the real one."""
    return Path(os.environ.get("OPYT_ORACLE_REFRESH_CONSENT",
                               opyt_path("oracle_refresh_consent")))


def consented() -> bool:
    """Has the user opted into automatic refresh of their Oracles? Marker-only. Unlike the
    bookmark-catchup handshake we do NOT auto-consent an established store: this is a recurring
    background loop, not a one-time backlog import."""
    return _consent_marker().exists()


def grant_consent() -> None:
    """Opt in. A failure here is LOUD on purpose: swallowing it leaves the user believing they
    consented while `consented()` keeps returning False and the loop keeps refusing to run."""
    marker = _consent_marker()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def revoke_consent() -> None:
    """Opt back out. A toggle should toggle both ways; afterwards the loop returns
    `needs_consent` before work begins. Loud for the same reason as `grant_consent`, and more
    so: a swallowed failure here leaves a background loop running that the user just switched
    off."""
    _consent_marker().unlink(missing_ok=True)


# ── The window ──────────────────────────────────────────────────────────────────
def since_for(row: st.SourceRow) -> datetime | None:
    """The lower bound to ask this pair for: its last successful pull, minus an overlap sliver.

    Falls back to `cursor_ts` (the newest atom we hold) when the pair has never been refreshed —
    which is the normal state for an Oracle onboarded before `set_oracle_window` went live, and
    is still a sane, corpus-derived window. None when we have neither."""
    base = st.parse_ts(row.last_pulled_at) or st.parse_ts(row.cursor_ts)
    if base is None:
        return None
    return base - timedelta(hours=OVERLAP_HOURS)


def window_ok(row: st.SourceRow, since: datetime | None, now: datetime) -> bool:
    """Is this window within the automatic refresh ceiling?

    Unmetered sources always pass (see `METERED_SOURCES`). A metered source passes with a window
    inside `MAX_REFRESH_WINDOW_DAYS`. The assertion guards against a threading bug dropping
    `since`, which silently becomes the adapter's 183-day default instead of an error.

    ⚠️ CALLER CONTRACT: a metered row must arrive with a concrete `since`. `refresh_pair` gives a
    never-pulled pair a `BREADTH_WINDOW_DAYS` window before calling here, so `None` reaches this
    only on an unmetered row, which returns above.

    That contract is load-bearing and it replaced a DEADLOCK. Refusing a metered pair with no
    window meant refusing it every run, forever, because a pull is the only thing that writes the
    fields the window comes from and the refusal is what prevented the pull. Confirming an Oracle
    whose first X pull failed — suspended account, expired cookie, an interrupted onboarding — was
    enough to reach it. The escape used to live HERE, as a `never_pulled(row)` branch that also
    required `last_status IS NULL`; one recorded error destroyed that and the deadlock came back.
    Giving the first pull a real window at the caller is what fixed it for good, which is why this
    predicate no longer has a None arm to get wrong."""
    if row.source_type not in METERED_SOURCES:
        return True
    return (now - since) <= timedelta(days=MAX_REFRESH_WINDOW_DAYS)


# ── Dispatch ────────────────────────────────────────────────────────────────────
def _dispatch(conn, embedder, row: st.SourceRow, since, author_name, before=None):
    """Run ONE pair's adapter. Returns `(summary, outcome, note)` where `outcome` is one of
    `ingest_common.RUN_*` or the string `"skipped"` (the eligibility gate refused).

    `before` is the BACKWARD bound and reaches GitHub only. X's pagination carries no upper
    bound — reaching an older instant means re-walking everything newer, which is what
    `backfill_pair` documents — so there is nothing to hand it.

    X and OpenAlex go DIRECTLY to their adapters, with no eligibility gate. The other three go
    through `expand._route_source`, which is the single gated door for the two website adapters
    (free on a cache hit: `source_authorship` is cached forever) and which already classifies the
    adapter's returned summary.

    WHY OPENALEX NEEDS NO GATE, and the reason is the atomizer, not the source. The gate exists to
    stop a multi-author source being attributed to one trusted person. A paper is multi-author by
    definition, and an OpenAlex SOURCE id (a journal, a preprint repository) is a whole venue of
    other people's work — so on the face of it this is the worst case the gate was built for. It
    is not, because `atomize_paper` refuses to launder it at the source: `who_id` is the PAPER's
    own author, always, and there is no override parameter. The invariant is enforced in the
    atomizer, which is why it survives a venue.

    ⚠️ This said "an OpenAlex author id is single-author BY CONSTRUCTION" until 2026-09-08, when
    venue roots made that false. The atomizer sentence was always the load-bearing one.

    The topic filter rides off `row.topic_filter` and nowhere else. That is the whole point of
    storing it: this loop re-pulls a scholar pair forever from the row, so a filter the ingest
    call knew about but the row did not would widen back to the full corpus on the next TTL."""
    from . import expand

    if row.source_type == "x":
        summ = ingest_x_footprint_sync(conn, embedder, handle=row.source_key,
                                       author_name=author_name, since=since)
        return summ, ingest_common.classify_run(summ), None

    if row.source_type == "openalex":
        try:
            summ = ingest_scholar_footprint_sync(conn, embedder, openalex_id=row.source_key,
                                                 author_name=author_name, since=since,
                                                 topics=row.topic_filter)
        except SourceError as e:
            # The HOST is backing off, which is not an author who published nothing. Reported
            # BLOCKED so nothing stamps and nothing advances — the next pass retries.
            return {}, ingest_common.RUN_BLOCKED, str(e)
        return summ, ingest_common.classify_run(summ), None

    url = (f"https://github.com/{row.source_key}" if row.source_type == "github"
           else row.source_key)
    entry = expand._route_source(
        conn, embedder, {"source_type": row.source_type, "url": url},
        author_name=author_name, limit=0,
        web_since=since if row.source_type in ("substack", "blog") else None,
        github_since=since if row.source_type == "github" else None,
        github_before=before if row.source_type == "github" else None,
    )
    if entry.get("skipped"):
        return {}, "skipped", f"{entry.get('skipped')}: {entry.get('reason') or ''}".strip()
    if entry.get("blocked") is not None:
        return entry["blocked"], ingest_common.RUN_BLOCKED, entry.get("reason")
    if entry.get("error"):
        return {}, ingest_common.RUN_ERROR, str(entry["error"])
    return entry.get("ingested") or {}, ingest_common.RUN_INGESTED, None


def ingest_scholar_footprint_sync(conn, embedder, **kw):
    """One named seam for the OpenAlex author pull, lazily imported for the same reason the X one
    is: this module is loaded by the MCP tool surface at REQUEST time."""
    from . import ingest_scholar_footprint
    return ingest_scholar_footprint.sync_scholar_footprint(conn, embedder, **kw)


def ingest_x_footprint_sync(conn, embedder, **kw):
    """One named seam for the X pull, imported lazily for the same reason every other adapter
    here is: `pipeline.kb.oracle_refresh` is loaded by the MCP tool surface at REQUEST time, and
    the adapter pulls in the whole X transport. Tests fake the pull by replacing this name, which
    is a consequence of the seam, not the reason for it."""
    from . import ingest_x_footprint
    return ingest_x_footprint.sync_x_footprint(conn, embedder, **kw)


# ── The unit: refresh ONE pair ──────────────────────────────────────────────────
def refresh_pair(conn, embedder, row: st.SourceRow, *,
                 now: datetime | None = None) -> dict:
    """Refresh one stale pair within its automatic window.

    The forward window starts at the last pull or stored cursor. A pair with no prior
    coverage gets `BREADTH_WINDOW_DAYS`; `backfill_pair` later deepens that coverage.
    Wider forward windows require explicit Oracle ingestion.
    """
    now = now or utc_now()

    if not st.is_stale(row, now):
        return {"status": "fresh", **_ident(row)}

    since = since_for(row)
    if since is None and row.source_type in METERED_SOURCES:
        # The BREADTH pull. No stamp and no atoms means no window to pull forward from, and the
        # answer is a shallow recent slice — one or two requests — not the adapter's 183-day
        # default. Depth is `backfill_pass`'s job, ordered and budgeted.
        since = now - timedelta(days=BREADTH_WINDOW_DAYS)
    if not window_ok(row, since, now):
        # Only one way to be wide: `window_ok` refuses a metered row whose window predates the
        # ceiling, and a metered row always has a window by here (see the contract above).
        detail = (f"since {since:%Y-%m-%d} is older than the "
                  f"{MAX_REFRESH_WINDOW_DAYS}-day refresh ceiling")
        st.record_pull(conn, row, last_status="window_refused", stamp=False)
        log(f"[oracle-refresh] {row.source_type}:{row.source_key} REFUSED — {detail}")
        return {"status": "window_refused", **_ident(row), "reason": detail,
                "remedy": "To pull this Oracle over a wider window, call oracle(action='ingest', "
                          "canonical_ids=[...], x_lookback='6mo')."}

    return _pull_pair(conn, embedder, row, since=since)


def _pull_pair(conn, embedder, row: st.SourceRow, *, since, before=None) -> dict:
    """breaker → pull → classify → persist, for one (Oracle, source) pair over ONE window.

    The window is the CALLER's decision — `refresh_pair` derives it from the cursor (forward),
    `backfill_pair` from the coverage frontier (backward). Everything after that is identical, and
    it is identical here rather than twice because the persistence rules below are the part that
    is easy to get subtly wrong in a copy.

    The four write-outcomes differ only in what they persist:
      ingested/empty — a real observation: advance the cursor, widen `covered_from` to the window
                       this pull reached, stamp `last_pulled_at`, TTL restarts.
      blocked        — a host stopped us: nothing was written and nothing marked seen, so neither
                       the cursor nor the stamp nor the frontier moves. A Cloudflare shell is NOT
                       an author who went quiet, and stamping would buy one bad night a full TTL
                       of silence.
      error          — the pull raised: breaker records it, no advance, no stamp; retried next run.
      breaker_open   — nothing ran.
    """
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

    try:
        summary, outcome, note = _dispatch(conn, embedder, row, since, row.name, before=before)
    except (_core().XRateLimited, SyncAuthError, CircuitOpenError):
        # SESSION-WIDE, not this pair's fault — every later X request fails identically until the
        # window resets or the user re-logs in. Propagate so the LOOP decides, exactly as
        # `candidate_probe.probe_candidate` does. Recording it here would charge the breaker for
        # somebody else's outage: three rate-limited sessions in a row would open a 7-day cooldown
        # on three perfectly healthy handles.
        #
        # ⚠️ `CircuitOpenError` JOINED THIS SET 2026-09-15, AND IT IS THE SAME SENTENCE ONE LAYER
        # OUT. It can only come from a MODEL-PROVIDER circuit here — this pair's own breaker is
        # consulted with `allow()` above and never raises — so it means the embedder or the
        # classifier is down, which is as session-wide as a spent X window and as little the
        # handle's fault. Measured on the box: a user's starter allowance hit its ceiling,
        # OpenRouter answered `403 Key limit exceeded`, the `openrouter-embed` circuit opened, and
        # the next three X pulls each failed with `CircuitOpenError` and were charged HERE. Three
        # strikes each, and three perfectly healthy X handles sat under a SEVEN-DAY cooldown whose
        # cause was a missing cent of credit. Adding credit did not clear it, because nothing the
        # user could do touches this table — which is the precise failure the comment above was
        # already written to prevent, for the one outage nobody had thought of yet.
        raise
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        breaker.record_failure(detail)
        st.record_pull(conn, row, last_status="error", stamp=False)
        log(f"[oracle-refresh] {row.source_type}:{row.source_key} pull FAILED: {detail}")
        return {"status": "error", **_ident(row), "error": detail}

    added = int(summary.get("added") or 0)

    if outcome in (ingest_common.RUN_BLOCKED, ingest_common.RUN_ERROR):
        detail = note or str(summary.get("error"))
        # BLOCKED counts toward the breaker as much as ERROR does. Both mean "this endpoint is not
        # serving us"; the breaker's job is to stop calling a broken one, and a Cloudflare wall
        # that turns us away every night is the textbook case for a 7-day cooldown.
        breaker.record_failure(detail)
        blocked = outcome == ingest_common.RUN_BLOCKED
        st.record_pull(conn, row, last_status="blocked" if blocked else "error", stamp=False)
        return {"status": "blocked" if blocked else "error", **_ident(row),
                ("reason" if blocked else "error"): detail}

    breaker.record_success()
    if outcome == "skipped":
        # The eligibility gate refused (multi-author / needs-review). A real, cheap observation —
        # stamp it so the TTL paces the re-check rather than re-asking every single run.
        st.record_pull(conn, row, last_status="skipped", stamp=True, now=st._now())
        return {"status": "skipped", **_ident(row), "reason": note}

    status = "ingested" if added > 0 else "empty"
    _, who_ids = st.pairs_for_oracle(conn, row.canonical_id)
    cursor = st.latest_atom_ts(conn, row.source_type, who_ids) or row.cursor_ts
    # An adapter that MEASURES its own reach is the only thing that knows where it stopped, so it
    # reports the frontier and that report wins. `sync_github` does: its ceiling is repos, not
    # time, so the instant it reached is not derivable from the window it was asked for. X does
    # too, since 2026-09-14 — its walk keeps what it got, so reaching `since` stopped being
    # implied by returning at all.
    #
    # ⚠️ PRESENCE, NOT TRUTHINESS, and this line read `summary.get(...) or since` until that same
    # day. A partial X walk reports `covered_from: None` DELIBERATELY — one timeline finished and
    # the other did not, which is a biased sample and no defensible frontier (`_walk_frontier`) —
    # and `or` sent exactly that case down the fallback, stamping the window we asked for onto a
    # pull that never covered it. `covered_from` only ever widens, so nothing would revisit it.
    # `_record_coverage` reads the same rule on the foreground path.
    if isinstance(summary, dict) and "covered_from" in summary:
        reached = summary["covered_from"]
    else:
        reached = since.isoformat() if since else None
    st.record_pull(conn, row, last_status=status, cursor_ts=cursor, covered_from=reached,
                   stamp=True, now=st._now())
    # The adapter's own counters ride along. Without them the incrementality seams are INVISIBLE
    # in a normal run: `stale: 258` (the GitHub calls the `pushed_at` gate avoided) and `fetched`
    # (the tweets actually fetched) both stop at the adapter otherwise, and a report that cannot
    # show the avoided work cannot show a regression in it either.
    return {"status": status, **_ident(row), "new_atoms": added,
            "engagements": int(summary.get("engagements") or 0),
            "cursor_ts": cursor, "stats": ingest_common.run_stats(summary)}


def _ident(row: st.SourceRow) -> dict:
    return {"canonical_id": row.canonical_id, "name": row.name,
            "source_type": row.source_type, "source_key": row.source_key}


# ── The loop ────────────────────────────────────────────────────────────────────
def refresh_all(conn, embedder, *, max_pairs: int = MAX_PAIRS_PER_RUN, now: datetime | None = None,
                should_stop=None) -> dict:
    """Refresh every stale pair, worst-lag-first, bounded. Never raises.

    `max_pairs` is LOAD-BEARING, not a safety afterthought. On the very first run after seeding
    the enabled sources, every pair's `last_pulled_at` comes from the Oracle's onboarding coverage
    marker — so any
    Oracle onboarded more than a TTL ago is stale immediately and the WHOLE roster comes due at
    once. The bound turns that burst into a backlog that drains a few pairs per session, and
    `deferred` reports the remainder so it is visible rather than silently truncated.

    It bounds attempted pulls, not iterations: a pair that returns fresh / breaker_open /
    window_refused made no request, so it does not consume a slot. Otherwise one permanently-refused
    pair would sort first every run and starve everything behind it.

    """
    from pipeline.ingestion.x_graphql import has_managed_x_session

    now = now or utc_now()
    x_connected = has_managed_x_session()
    st.seed_from_entities(conn, include_x=x_connected)

    rows = [r for r in st.list_sources(conn)
            if r.status != "paused" and (x_connected or r.source_type != "x")]
    stale = [r for r in rows if st.is_stale(r, now)]
    stale.sort(key=lambda r: st.staleness_hours(r, now), reverse=True)   # worst lag first

    agg = {"status": "ok", "registered": len(rows), "considered": len(stale),
           "refreshed": 0, "empty": 0, "blocked": 0, "errors": 0, "breaker_open": 0,
           "window_refused": 0, "skipped": 0, "deferred": 0,
           "new_atoms": 0, "engagements": 0, "by_oracle": {}, "results": []}
    attempted = 0

    for row in stale:
        if should_stop and should_stop():
            agg["status"] = "lease_lost"
            agg["deferred"] += len(stale) - stale.index(row)
            break
        if attempted >= max_pairs:
            agg["deferred"] += 1
            continue
        try:
            r = refresh_pair(conn, embedder, row, now=now)
        except _core().XRateLimited as e:
            # Session-wide: the remaining pairs would fail identically, so stop rather than burn
            # through the queue marking healthy handles broken. An unstamped pair is infinitely
            # stale and sorts first on the next scheduled run.
            agg["status"] = "rate_paused"
            agg["deferred"] += len(stale) - stale.index(row)
            agg["resumes"] = "next-scheduled-run"
            log(f"[oracle-refresh] session-wide stop, {agg['deferred']} deferred: "
                f"{type(e).__name__}: {e}")
            break
        except SyncAuthError as e:
            # The source was connected when this run started but is no longer usable. This needs
            # a new login, not a promise that the worker will fix it on its own.
            agg["status"] = "needs_reconnect"
            agg["needs_reconnect"] = "x"
            log(f"[oracle-refresh] X session needs reconnect: {e}")
            break
        except CircuitOpenError as e:
            # The model provider is down or out of credit. Every remaining pair would fail the
            # same way, so stop: an unstamped pair is infinitely stale and sorts first next run,
            # and no handle is marked broken for an outage that is not its own.
            agg["status"] = "provider_down"
            agg["deferred"] += len(stale) - stale.index(row)
            agg["resumes"] = "next-scheduled-run"
            agg["provider_down"] = str(e)
            log(f"[oracle-refresh] session-wide stop, {agg['deferred']} deferred: "
                f"model provider unavailable: {e}")
            break
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
        key = {"ingested": "refreshed", "empty": "empty", "blocked": "blocked",
               "error": "errors", "breaker_open": "breaker_open",
               "window_refused": "window_refused", "skipped": "skipped"}.get(status)
        if key:
            agg[key] += 1
        bucket = agg["by_oracle"].setdefault(row.canonical_id,
                                             {"name": row.name, "sources": {}})
        bucket["sources"][f"{row.source_type}:{row.source_key}"] = status

    log(f"[oracle-refresh] {agg['considered']} stale, {agg['refreshed']} refreshed, "
        f"{agg['empty']} empty, {agg['deferred']} deferred, {agg['new_atoms']} new atoms")
    return agg


# ── The backward job: deepen coverage toward the target window ──────────────────
def coverage_gap(row: st.SourceRow, target: datetime) -> bool:
    """Does this pair need deepening — is its backward frontier LATER than `target`?

    A NULL `covered_from` on a pulled pair counts as a gap. It means no pull has reported a lower
    bound (a pre-`covered_from` store, or a pull that truncated), and re-pulling to the target is
    the safe direction: over-asking is absorbed by dedup, under-asking loses history silently."""
    if row.source_type not in DEEPENED_SOURCES:
        return False                      # see that set for which sources are walked backward
    if row.last_pulled_at is None:
        return False                      # never pulled at all — that is the BREADTH pull's job
    frontier = st.parse_ts(row.covered_from)
    return frontier is None or frontier > target


def deepen_target(now: datetime | None = None) -> datetime:
    """The instant a backfill pass pulls back to. One function so the selector, the ordering and
    the pull cannot disagree about what "deep enough" means."""
    return (now or utc_now()) - timedelta(days=BACKFILL_TARGET_DAYS)


def backfill_pair(conn, embedder, row: st.SourceRow, *, target: datetime,
                  now: datetime | None = None) -> dict:
    """Pull ONE pair back to `target`. The backward sibling of `refresh_pair`.

    Two things differ from a refresh, and both follow from the direction:

    • The window is the TARGET, not the cursor. X's pagination is newest-first with no `until`
      bound, so reaching an older instant means re-walking everything newer than it. That re-walk
      is not waste that a smarter cursor could avoid — it is what pagination costs — and it is why
      a pass goes STRAIGHT to the target rather than stepping there. Four 6-month steps to two
      years need 2.5x as many requests as one two-year walk, for the same history.
      `snapshot_and_hash` skips embedding anything unchanged. REQUESTS are the scarce resource,
      and they scale with depth.

    • The whole chosen window is processed so this pass can reach its oldest end.

    GitHub is the one source where the re-walk IS avoidable, and that is why the frontier goes
    back down as `before`. Its repo list arrives whole and carries every `pushed_at` before a
    single README is fetched, so the covered prefix is skipped for free and the run spends its
    ceiling (`ingest_github.REPOS_PER_RUN`) on repos it does not yet hold. A pair whose frontier
    is still NULL has no prefix to skip, so its first pass re-reads from the newest repo — one
    cycle's worth of calls, after which the frontier it reports bounds every later pass."""
    now = now or utc_now()
    if not coverage_gap(row, target):
        return {"status": "deep_enough", **_ident(row)}
    return _pull_pair(conn, embedder, row, since=target, before=st.parse_ts(row.covered_from))


def backfill_pass(conn, embedder, *,
                  now: datetime | None = None, should_stop=None) -> dict:
    """Deepen coverage toward the target window, SHALLOWEST FIRST, until the meter runs dry.

    "Until the meter runs dry" is literal since 2026-09-14, and until then it was not. A
    pre-emptive reserve used to skip an X pair whenever the bucket was below a fixed floor, so the
    pass stopped short of dry by design; a durable partial walk makes stopping short pure loss.
    What ends an X pair now is x.com refusing — `_refuse_if_spent`, or a 429 — which is what the
    `XRateLimited` handler below has always been for.

    Two source types are deepened (`DEEPENED_SOURCES`) and each has its own scarce resource, so
    what stops one never stops the other: a GitHub pair is bounded by
    `ingest_github.REPOS_PER_RUN`, not by a request meter, and it must still run behind an X pair
    that x.com has cut off.

    An X session that dies mid-pass still ends the pass, GitHub pairs included. It then needs a
    reconnect; later passes skip X until the managed session validates again.

    Shallowest-first is what keeps incompleteness uniform. Without it one prolific account eats
    the window while every other Oracle stays at its breadth pull — and "I have three of your
    fifteen and cannot tell you which" is not a sentence the store can usefully say.
    `staleness_hours` already establishes this ordering pattern for the forward job.

    It runs AFTER `refresh_all` in the same child, on the same single-flight lease and the same
    connection, and that is deliberate rather than incidental: `MAX_PAIRS_PER_RUN` is sized
    against SQLite write-lock collision with a second rail on this store, not against X. The
    worker runs at most one rail per home, so its own children no longer contend; a hand-run
    `--once` beside it still does. A pass drains what it can and REPORTS the remainder, never
    promises it.

    Never raises. `XRateLimited` ends the pass and is reported as `deferred`, not as an error:
    hitting a rate window during a job designed to span rate windows is the job working. A dead X
    session instead reports `needs_reconnect`."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.x_graphql import has_managed_x_session

    now = now or utc_now()
    target = deepen_target(now)
    x_connected = has_managed_x_session()
    rows = [r for r in st.list_sources(conn)
            if r.status != "paused" and (x_connected or r.source_type != "x")]
    gapped = [r for r in rows if coverage_gap(r, target)]
    # Shallowest first: the frontier CLOSEST to now. A NULL frontier is the shallowest thing there
    # is — nothing has reported how far back it goes — so it sorts ahead of every dated one.
    gapped.sort(key=lambda r: (r.covered_from is not None, r.covered_from or ""), reverse=True)

    agg = {"status": "ok", "target": target.isoformat(), "considered": len(gapped),
           "deepened": 0, "empty": 0, "blocked": 0, "errors": 0, "breaker_open": 0,
           "deferred": 0, "new_atoms": 0, "engagements": 0, "results": []}

    # Set once x.com has actually cut this session's timeline meter off. It skips the REMAINING X
    # pairs and nothing else — the per-source separation that the deleted pre-emptive reserve used
    # to provide, now driven by x.com's refusal instead of by a prediction about it.
    x_cut_off = False

    for row in gapped:
        if should_stop and should_stop():
            agg["status"] = "lease_lost"
            agg["deferred"] += len(gapped) - gapped.index(row)
            break
        if x_cut_off and row.source_type == "x":
            agg["deferred"] += 1
            continue
        try:
            r = backfill_pair(conn, embedder, row, target=target, now=now)
        except core.XRateLimited as e:
            # SESSION-WIDE for X, and for X only. Every remaining X pair would fail identically,
            # so none is tried again — but a GitHub pair behind them is bounded by
            # `ingest_github.REPOS_PER_RUN`, not by this meter, and breaking the whole loop here
            # would stall GitHub deepening for as long as the X meter stayed low, which is most of
            # the time. The pass continues; the X tail is counted deferred rather than left
            # looking done.
            x_cut_off = True
            agg["status"] = "rate_paused"
            agg["deferred"] += 1
            agg["resumes"] = "next-scheduled-run"
            log(f"[oracle-backfill] {row.source_type}:{row.source_key} — request window "
                f"exhausted; remaining X pairs deferred: {e}")
            continue
        except SyncAuthError as e:
            agg["status"] = "needs_reconnect"
            agg["needs_reconnect"] = "x"
            log(f"[oracle-backfill] X session needs reconnect: {e}")
            break
        except Exception as e:                  # a pair must never sink the pass
            detail = f"{type(e).__name__}: {e}"
            log(f"[oracle-backfill] {row.source_type}:{row.source_key} unhandled: {detail}")
            r = {"status": "error", **_ident(row), "error": detail}

        agg["results"].append(r)
        agg["new_atoms"] += r.get("new_atoms", 0)
        agg["engagements"] += r.get("engagements", 0)
        key = {"ingested": "deepened", "empty": "empty", "blocked": "blocked",
               "error": "errors", "breaker_open": "breaker_open"}.get(r["status"])
        if key:
            agg[key] += 1

    log(f"[oracle-backfill] {agg['considered']} shallow, {agg['deepened']} deepened, "
        f"{agg['deferred']} deferred, {agg['new_atoms']} new atoms")
    return agg


# ── Status ──────────────────────────────────────────────────────────────────────
def status_summary(conn) -> dict:
    """A read-only freshness snapshot — per-Oracle, per-source. All DERIVED: this loop writes no
    log of its own, `oracle_sources` + `circuit_breaker` ARE the log.

    A FROZEN Oracle must be visible HERE. That bug — nothing ever re-pulling an Oracle — was
    invisible for months precisely because no surface reported per-source staleness; this is
    where that stops. Never raises: a missing table degrades to an empty snapshot."""
    try:
        rows = st.list_sources(conn)
    except Exception as e:
        return {"consented": consented(), "error": f"{type(e).__name__}: {e}",
                "tracked_pairs": 0, "oracles": []}

    try:
        from pipeline.circuit_breaker import status as breaker_status
        open_breakers = {b["service"] for b in breaker_status()
                         if b["service"].startswith("oracle-refresh:") and b["state"] != "closed"}
    except Exception:
        open_breakers = set()

    # Sources waiting on a user decision, per Oracle. A needs-review source never reaches a
    # registered pair (the adapter did not run, so no entity was minted), so `rows` below is
    # structurally blind to it — and an Oracle with an unreviewed Substack has been reporting its
    # X-only coverage as complete. Fail-safe: an unreadable table contributes nothing.
    try:
        from . import oracle_reviews
        unreviewed = oracle_reviews.open_counts(conn)
    except Exception:
        unreviewed = {}

    now = utc_now()
    by_oracle: dict = {}
    overdue = 0
    for r in rows:
        lag = st.staleness_hours(r, now)
        stale = st.is_stale(r, now)
        overdue += 1 if stale else 0
        entry = by_oracle.setdefault(r.canonical_id,
                                     {"canonical_id": r.canonical_id, "name": r.name,
                                      "sources": [], "stale_sources": 0,
                                      # ⚠️ REPORTED, NEVER ESCALATED. This must not reach
                                      # `needs_attention` below: that flag drives a proactive
                                      # `oracles_stale` search notice, and the ruling is available
                                      # and honest, never nagging. The user acts on it through
                                      # `oracle(action='review')` when they choose to.
                                      "unreviewed": unreviewed.get(r.canonical_id, 0)})
        entry["sources"].append({
            "source_type": r.source_type, "source_key": r.source_key,
            "last_pulled_at": r.last_pulled_at,
            # How far BACK this source goes. `last_pulled_at` answers "how current"; this answers
            # "how deep", and it is the one a user actually asks — "do you have what they said
            # last spring". NULL means no pull has reported a lower bound.
            "covered_from": r.covered_from,
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
    #     and it is the defect that hid for months: the trigger fired, the rail refused for want of
    #     consent, and the refusal went to a log nobody reads. A rail that runs and declines looks
    #     exactly like a rail that runs and finds nothing, which is why this surfaces the state.
    #   • the cycle stretched past 2x TTL — the loop is running and losing ground.
    # Guarded on `tracked_pairs`: a store with no Oracles yet must stay silent. Telling a fresh
    # install that Oracle refresh is off is noise on the first surface a new user ever sees.
    out["needs_attention"] = bool(
        rows and (not out["consented"]
                  or out["stale_fraction"] > STALE_FRACTION_ATTENTION))

    if not out["consented"] and rows:
        out["note"] = ("Automatic Oracle refresh is OFF — nothing re-pulls your Oracles, so their "
                       "content is frozen at the day each was added. Run `onboard` to opt in; it "
                       "asks once and explains that the refresh runs in the background.")
    elif out["needs_attention"]:
        out["note"] = (f"{overdue} of {len(rows)} Oracle sources are overdue — your refresh cycle "
                       f"has stretched past twice its target. The loop drains "
                       f"{MAX_PAIRS_PER_RUN} sources per pass and the worker runs it every ten "
                       f"minutes, so it catches up on its own. To pull one person now, call "
                       f"oracle(action='ingest', canonical_ids=[...], x_lookback='since_last').")
    return out


# ── The entrypoint ──────────────────────────────────────────────────────────────
def run_oracle_refresh(*, max_pairs: int = MAX_PAIRS_PER_RUN) -> dict:
    """Load credentials, check consent, then refresh under a single-flight lease.

    Onboarding owns consent. An unconsented worker returns before starting the loop.
    """
    load_rail_env()

    if not consented():
        return {"status": "needs_consent",
                "message": ("Automatic Oracle refresh is OFF. Turning it on costs credits: it "
                            "reads connected browser sessions, then runs content-quality, "
                            "image-description and embedding steps on what it pulls. Consent "
                            "is collected inside `onboard`, and nothing refreshes until the "
                            "user gives it there.")}

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
            if lock.lost():
                return {"status": "lease_lost",
                        "message": "oracle-refresh lease was reclaimed before work began."}
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            conn = st.connect()
            try:
                out = refresh_all(conn, embedder, max_pairs=max_pairs,
                                  should_stop=lock.lost)
                if out.get("status") == "lease_lost":
                    return out
                # FRESHNESS first, then DEPTH, in that order and in this same child. Freshness is
                # TTL-gated so it usually does nothing; depth uses whatever request allowance
                # freshness left. Reversing them would let one deep walk push every
                # Oracle's freshness past its TTL. The recommendation read comes THIRD and is
                # outside that allowance entirely — it pulls no content; see `_recommendation_pass`.
                #
                # One child, not two rails: `MAX_PAIRS_PER_RUN` is sized against SQLite
                # write-lock collision with a concurrent rail (measured: 22 'database is locked'
                # errors on the probe rail alone, back when every rail launched at once), so
                # adding a second contender for this store is what that bound protects against.
                out["backfill"] = backfill_pass(conn, embedder, should_stop=lock.lost)
                if out["backfill"].get("status") == "lease_lost":
                    out["status"] = "lease_lost"
                    return out
                out["recommendations"] = _recommendation_pass(conn)
                return out
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[oracle-refresh] run_oracle_refresh errored: {detail}")
        return {"status": "error", "error": detail}


def _recommendation_pass(conn) -> dict:
    """Read each Substack Oracle's recommendations, on its OWN clock. Never raises.

    THIRD, after freshness and depth, and outside their request budget. Those two pull CONTENT and
    `MAX_PAIRS_PER_RUN` sizes them against SQLite write-lock collision; this writes signals and
    entities, makes no model call, embeds nothing, and reads a host neither of them touches.
    Letting it consume a content slot would trade an Oracle's posts for an Oracle's endorsements,
    which are not substitutes.

    Its own clock, because this rail's cadence is 600 seconds and a recommendation list changes on
    the order of months — see `substack_recommendations.FLOOR_HOURS`. `is_due` reads
    `last_attempt_at`, so a failing pass is throttled exactly like a succeeding one.

    Swallows its own failure for the reason `resolve_after_pull` states: everything the refresh
    pulled is already committed by the time this runs, and a refused public read must degrade to
    a report, never take the whole pass's result down with it.
    """
    from . import curation_state, substack_recommendations as recs

    row = curation_state.get_run(conn, recs.COLLECTOR)
    if not curation_state.is_due(row, floor_hours=recs.FLOOR_HOURS):
        return {"status": "not_due", "floor_hours": recs.FLOOR_HOURS}
    try:
        out = recs.sync_recommendation_signals(conn)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[oracle-refresh] recommendation pass errored: {detail}")
        curation_state.record_run(conn, recs.COLLECTOR, status="error", detail=detail)
        return {"status": "error", "error": detail}
    # `ok` only when every Oracle was actually read. A pass that skipped one saw a partial list,
    # and `last_ok_at` is what a later reader would take as "this walk is trustworthy".
    status = "ok" if out["skipped_oracles"] == 0 else "partial"
    curation_state.record_run(conn, recs.COLLECTOR, status=status,
                              found=out["publications"], stored_after=out["signalled"],
                              detail=None if status == "ok"
                              else f"{out['skipped_oracles']} oracle(s) unread")
    return {"status": status, **out}


def _run() -> None:
    """The rail child's entry point: report one pass in this home's log and exit.

    The worker invokes this through `-c "from pipeline.kb.oracle_refresh import _run; _run()"`
    rather than the `-m ... --once` form its seven siblings use, because this rail takes no
    options and therefore has no command line to parse — see `retired-oracle-refresh-cli`. It
    queues no successor: nothing downstream is waiting on a refreshed Oracle."""
    res = run_oracle_refresh()
    print(json.dumps(res, indent=2, default=str))
    # Backfill may pause at the rate budget while forward refresh succeeds.
    #
    # `rate_paused` JOINED THIS SET 2026-09-14, because it is a healthy pass and was recording as
    # a failure. x.com meters timeline reads in 15-minute windows, so the pass right after a first
    # ingest spends the budget and defers the rest — nothing fetched, nothing written, nothing
    # marked pulled, `errors: 0`, and the sources retried on the next pass. Measured on a fresh
    # store: the 20:11 pass reported `status: rate_paused`, `deferred: 3`, `errors: 0` and exited
    # 1. Scheduling was unharmed (`RailJobStore.finish` applies the cadence whatever the code), so
    # the cost is purely that `exit_code` lies to anyone reading it to ask whether refresh is
    # working — which is exactly the question this rail's whole repair was about.
    raise SystemExit(0 if res.get("status") in {"ok", "already_running", "needs_consent",
                                                "rate_paused", "provider_down"} else 1)
