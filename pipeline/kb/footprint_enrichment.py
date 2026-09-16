"""
pipeline/kb/footprint_enrichment.py — the metered deepening pass over Oracle X timelines.

"Enrichment" to the user, exactly as `pipeline/kb/enrichment.py` is. ONE NAME, TWO ENGINES (R6):
the person never learns there are two, because from where they sit there is one fact — more of
their writers is still arriving — and two words for it would be two things to understand.

WHY A SIBLING AND NOT A BRANCH IN `run_enrichment`. That loop is bound to bookmarks at all three
points that differ here, and folding them together would make ONE sleep serve two unrelated
meters:

    |        | bookmark Enrichment           | this                                      |
    |--------|-------------------------------|-------------------------------------------|
    | bucket | `TweetDetail`, 150/15min      | `UserTweets` + `UserRepliesTimeline`, 50  |
    | lease  | `bookmark-catchup`            | `oracle-refresh`                          |
    | corpus | saved posts owed a thread     | confirmed Oracles with an unmet window    |

R2, the single organizing rule of the flow: BACKGROUND ⟺ BLOCKED BY A RATE METER. A foreground
`oracle(action='ingest')` over four Oracles cannot cover four Oracles — `UserTweets` is 50 per 15
minutes against a measured 2-26 requests each, and nothing in this module changes that arithmetic.
The remainder is definitionally background work, and asking the user to re-trigger it is not a
smaller version of automatic: they already chose these writers when they confirmed them, and
asking again asks twice for one decision.

Until 2026-09-14 the honest answer was to ask, and the message that did so recorded why: "a user
with a rate-limited Oracle was told it was in hand while it was in fact abandoned, and stopped
waiting for a pull that could not start." Right diagnosis, wrong remedy — it fixed a false promise
by handing the user a chore, where the fix is to make the promise true.

IT MAY SLEEP; A RAIL MAY NOT. `x_graphql_core._refuse_if_spent` refuses rather than sleeps because
"a rail is a one-shot detached child holding a single-flight lease and a SQLite connection", and
sleeping inside one blocks every other rail for the window while finishing no sooner than the next
session would. Neither applies to this loop: it releases the lease between passes and closes
nothing else's path while it waits, and waiting is exactly what finishes the work.

⚠️ DURABLE PARTIAL WALKS ARE THE PRECONDITION, not a coincidence of timing. The safety property
below is "a killed process loses time, never work", and Claude Desktop DOES kill the server
(measured 2026-09-14 08:54 and 09:01). An all-or-nothing `_pull_own_timeline` could not satisfy it:
a pass killed mid-walk would lose everything it had fetched. That is why the two changes belong to
one build.

NO WALL-CLOCK DEADLINE (R1). `_MAX_PASSES` is a runaway guard on a loop that cannot otherwise
prove it terminates, not a budget; the pass that would exceed it is logged, never silently
dropped. Progress is durable at every point — `AtomSink` flushes incrementally, `record_pull`
stamps per pair as each finishes, and a partial walk keeps its atoms — so a restart resumes rather
than restarts.

Never raises.
"""

from __future__ import annotations

import threading
import time

from pipeline.ingestion.utils import log
from pipeline.kb.rail_runtime import load_rail_env, models_unroutable

RAIL = "footprint-enrichment"

# A runaway guard, not a deadline — the same role `_MAX_PASSES` plays in the bookmark sibling, and
# a different arithmetic. `backfill_pass` deepens shallowest-first and the meter allows roughly two
# deep walks per window (50 requests against a measured 12-25 per 183-day walk), so a roster of
# twenty drains in about ten windows and this leaves better than double the headroom. Past it the
# loop has almost certainly stopped converging for a reason the no-progress check could not name.
_MAX_PASSES = 24

# Wake a little AFTER the stated reset. Waking exactly on it races the server's own clock and buys
# one refused request per window for nothing. Same number and same reason as the sibling's.
_RESET_SLACK = 5.0

# ── In-process progress, for callers that must not claim a finished pull ────────
# A THREAD, not a `rail_jobs.db` row, so nothing that reads the rail table can see it. This is the
# probe `_ingest_presentation` reads before it promises the rest is filling in — see `is_running`.
_STATE_LOCK = threading.Lock()
_STATE: dict = {"running": False, "passes": 0, "owed": None, "added": 0,
                "started_at": None, "status": None}


def state() -> dict:
    """A snapshot of what this pass is doing right now. Never raises, never blocks on the work."""
    with _STATE_LOCK:
        return dict(_STATE)


def is_running() -> bool:
    return bool(state()["running"])


def _set(**kw) -> None:
    with _STATE_LOCK:
        _STATE.update(kw)


def seconds_until_window_resets(now: float | None = None) -> float:
    """How long until a footprint walk can run again — 0.0 when it already can.

    TWO BUCKETS, and the wait is the LONGER of them: a footprint pull walks `UserTweets` and
    `UserRepliesTimeline`, they meter independently at 50 per 15 minutes each, and waking while
    either is still spent means a walk that finishes one timeline and is cut off on the other —
    which `_walk_frontier` correctly refuses to claim a frontier for. Sleeping the extra seconds
    is cheaper than a pass that can only produce biased coverage.

    Read off x.com's own `x-rate-limit-reset` header via `core.rate_budget`, so nothing here
    encodes X's numbers. UNKNOWN MEANS GO, matching `_refuse_if_spent`: the meter is process-local
    and a fresh process starts blind, so treating no-evidence as spent would make a restarted
    server sleep fifteen minutes before its first request.
    """
    from pipeline.ingestion import x_graphql_core as core

    now = time.time() if now is None else now
    waits = []
    for op in (core.USERTWEETS_OP, core.USERREPLIES_OP):
        st = core.rate_budget(op)
        if st is None:
            continue
        remaining, reset = st
        if remaining > 0 or now >= reset:
            continue
        waits.append((reset - now) + _RESET_SLACK)
    return max(waits) if waits else 0.0


def owed(conn=None) -> int:
    """How many X pairs still have an unmet window — the corpus this loop drains.

    Fail-safe: unreadable is 0, because every caller uses this to decide whether to PROMISE
    something, and "nothing is owed" is the claim that costs least when wrong."""
    from . import oracle_refresh

    try:
        from . import oracle_refresh_state as st
        from pipeline.ingestion.x_graphql import has_managed_x_session

        if not has_managed_x_session():
            return 0
        own = conn is None
        conn = st.connect() if own else conn
        try:
            target = oracle_refresh.deepen_target()
            return sum(1 for r in st.list_sources(conn)
                       if r.source_type == "x" and r.status != "paused"
                       and oracle_refresh.coverage_gap(r, target))
        finally:
            if own:
                conn.close()
    except Exception:
        return 0


def run_footprint_enrichment(*, max_passes: int = _MAX_PASSES,
                             should_stop=None, sleep=time.sleep) -> dict:
    """Deepen every Oracle X timeline that is still owed its window. Never raises.

    One pass per window, sleeping between them. Each pass is `oracle_refresh.backfill_pass`,
    reused rather than reimplemented — it already walks the right pairs, SHALLOWEST FIRST, to the
    right target, and persists per pair as it goes.

    ⚠️ SHALLOWEST-FIRST IS LOAD-BEARING HERE, not a nicety inherited by accident. Without it this
    loop reproduces the exact starvation it exists to end: a stable most-vouched-first ordering
    spends every window on the same prolific accounts and the never-pulled Oracles are reached
    never. "Same shape as bookmark Enrichment" would not have given us this — bookmark Enrichment
    has no ordering problem to solve — which is why the pass is borrowed whole from the rail
    instead.

    Stops when:

      • nothing is owed — the finished case;
      • the X session needs a person (`needs_reconnect`), which no number of windows fixes;
      • the meter is NOT what is holding us back and the pass gained no ground — a dead handle, a
        breaker-open pair. Sleeping through fifteen minutes to re-learn that is the runaway this
        check exists to prevent;
      • the single-flight lease was reclaimed, `should_stop` fired, or `max_passes` tripped.

    `sleep` is injected so tests drive the whole loop without wall-clock time. `should_stop` is
    checked between passes, so a cancel never interrupts a pass mid-write.
    """
    load_rail_env()

    from . import oracle_refresh

    if not oracle_refresh.consented():
        # The same marker the rail asks for, and the same reason: this reads the user's connected
        # X session in the background and spends model credits on what it pulls. Onboarding
        # collects it. A caller must treat this as "could not start" — see `is_running`.
        return {"status": "needs_consent", "passes": 0,
                "message": "automatic Oracle refresh has not been consented to."}
    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "passes": 0, "message": reason}

    from pipeline.sync_lock import CatchupLock
    from . import oracle_refresh_state as st
    from .embed import get_kb_embedder

    _set(running=True, passes=0, owed=None, added=0, started_at=time.time(), status="running")
    passes = added = 0
    remaining = None
    status = "done"
    try:
        while passes < max_passes:
            if should_stop and should_stop():
                status = "cancelled"
                break
            # SHARES the rail's lease name on purpose, exactly as the bookmark sibling shares
            # `bookmark-catchup`. `backfill_pass` walks the SAME pairs against the SAME two
            # buckets, so a second lease name would let the two run at once and split a meter the
            # whole design is organised around. Skipping because the rail holds it is not a loss:
            # the rail's own pass does this work.
            with CatchupLock("oracle-refresh") as lock:
                if not lock.acquired:
                    status = "already_running"
                    break
                conn = st.connect()
                try:
                    out = oracle_refresh.backfill_pass(conn, get_kb_embedder(),
                                                       should_stop=lock.lost)
                    # ⚠️ RE-READ FROM THE STORE, never `out["deferred"]`. Measured live on
                    # 2026-09-14: a pass reported `2 shallow, 1 deepened, 0 deferred, 218 new
                    # atoms` and this loop stopped — with both those pairs STILL OWED.
                    #
                    # `deferred` counts pairs the pass REFUSED to try. It does not count a pair
                    # that was tried, landed atoms, and still did not meet its window — and since
                    # durable partial walks that is the COMMON case, because a one-sided walk
                    # deliberately claims no frontier (`_walk_frontier`), so `covered_from` does
                    # not move however many atoms land. The two counts agree only for an
                    # all-or-nothing walk, which is precisely what this build removed.
                    #
                    # Inherited from the bookmark sibling, where `deferred` really does mean "still
                    # owed". That is the second time "same shape as bookmark Enrichment" has been
                    # wrong here; the first is the ordering note in `run_footprint_enrichment`.
                    still_owed = owed(conn)
                finally:
                    conn.close()
            passes += 1
            gained = out.get("new_atoms", 0)
            added += gained
            previous, remaining = remaining, still_owed
            _set(passes=passes, owed=remaining, added=added)

            if out.get("status") == "lease_lost":
                status = "lease_lost"
                break
            if out.get("status") == "needs_reconnect":
                # A dead session needs a person, not a window.
                status = "needs_reconnect"
                break
            if not remaining:
                break                                  # every window is met — the finished case

            wait = seconds_until_window_resets()
            if wait <= 0 and previous is not None and remaining >= previous and not gained:
                # Not the meter, no frontier moved, and NO ATOMS EITHER. All three matter: a
                # partial walk can land real content without advancing any frontier, and calling
                # that "no progress" would abandon the exact case this pass exists to finish.
                status = "no_progress"
                break
            if wait > 0:
                log(f"[{RAIL}] {remaining} timeline(s) still owe their older window; "
                    f"x.com's buckets refill in {int(wait)}s.")
                sleep(wait)
        else:
            status = "max_passes"
            log(f"[{RAIL}] stopped after {passes} passes with {remaining} still owed — a guard "
                f"tripped, not a finished pull.")
    except Exception as e:                    # fail-safe: this never takes anything down
        status = "error"
        log(f"[{RAIL}] run_footprint_enrichment errored: {type(e).__name__}: {e}")
        return {"status": status, "passes": passes, "added": added, "owed": remaining,
                "error": f"{type(e).__name__}: {e}"}
    finally:
        _set(running=False, status=status)
    return {"status": status, "passes": passes, "added": added, "owed": remaining}


def start_background(*, should_stop=None) -> dict:
    """Run `run_footprint_enrichment` on a daemon thread and say so — the shape every background
    start in this repo uses (`enrichment.start_background`, `onboard_tools._spawn`).

    Returns WITHOUT starting when nothing is owed, so a caller can tell "finished" from "running"
    and never promises a pass that had no work. `is_running()` is the probe the message layer
    reads; this return value is what a caller holds before the thread has had a chance to set it.

    THE THREAD OPENS ITS OWN CONNECTIONS: SQLite connections are thread-bound and the caller's is
    mid-request. Fail-safe in layers — `run_footprint_enrichment` never raises, the thread body
    swallows anything above it, and a spawn failure is reported rather than thrown. A run that
    dies with the process is recovered by the `oracle_refresh` rail, which reads the same unmet
    windows out of `oracle_sources`; where no worker is installed, the next foreground ingest
    starts this again.
    """
    if is_running():
        return {"status": "running", "note": "footprint enrichment is already in flight"}
    if not owed():
        return {"status": "nothing_owed"}

    def _go():
        try:
            run_footprint_enrichment(should_stop=should_stop)
        except Exception:
            pass

    try:
        threading.Thread(target=_go, name="opyt-footprint-enrichment", daemon=True).start()
    except Exception as e:
        log(f"[{RAIL}] could not start: {type(e).__name__}: {e}")
        return {"status": "not_started", "error": f"{type(e).__name__}: {e}"}
    return {"status": "running",
            "note": ("the rest of these writers' older posts fill in over the next few of x.com's "
                     "15-minute windows. Do NOT wait for it or poll.")}
