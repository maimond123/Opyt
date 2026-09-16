"""
pipeline/kb/enrichment.py — the metered upgrade pass over X bookmarks ("Enrichment" to the user).

`ingest_x.sync_bookmarks(enrich=False)` writes every bookmark's atom from the FREE Bookmarks
payload — 11 requests of a 500/15-min bucket for 1,001 bookmarks, measured 2026-09-13. What it
cannot afford is the pair of upgrades behind a meter: the conversation (`TweetDetail`, EXACTLY
150 per 15 minutes, and 96.5% of bookmarks want one) and the image reads. This module pays for
those over as many windows as it takes, re-minting each atom in place.

R2, the single organizing rule of the flow: BACKGROUND ⟺ BLOCKED BY A RATE METER. This is the only
thing in the bookmark path that qualifies, which is why it is the only thing here that runs behind
the user instead of in front of them.

IT MAY SLEEP; A RAIL MAY NOT. `x_graphql_core._refuse_if_spent` refuses rather than sleeps, and its
reason is specific: "a rail is a one-shot detached child holding a single-flight lease and a SQLite
connection" — sleeping inside one blocks every OTHER rail for the window and finishes no sooner
than the next session would. Neither applies here. This loop releases the lease and closes nothing
else's path while it waits, and it is the only claimant of the `TweetDetail` bucket it is waiting
on, so waiting is exactly what finishes the work.

NO WALL-CLOCK DEADLINE (R1). `_MAX_PASSES` is a runaway guard on a loop that cannot otherwise
prove it will terminate, not a budget: the pass that would exceed it is logged, never silently
dropped. Progress is durable at every point — `AtomSink` flushes incrementally,
`x_convo_checked.json` and `image_descriptions.json` checkpoint, `seen` is the atom table itself —
so a killed process loses time, never work, and a restart resumes rather than restarts.
"""

from __future__ import annotations

import threading
import time

from pipeline.ingestion.utils import log
from pipeline.kb.rail_runtime import load_rail_env, models_unroutable

RAIL = "enrichment"

# A runaway guard, not a deadline. David's backlog is 966 ÷ 150 = 7 windows; a store several times
# larger still fits. Past this the loop has almost certainly stopped converging for a reason the
# no-progress check below could not name, and looping forever would hide it.
_MAX_PASSES = 24

# The 15-minute `TweetDetail` window, used only when x.com has not told us when its bucket refills
# (a fresh process has never seen the header; a 429 can arrive without one).
_WINDOW_SECONDS = 15 * 60
# Wake a little AFTER the stated reset. Waking exactly on it races the server's own clock and buys
# one refused request per window for nothing.
_RESET_SLACK = 5.0

# ── In-process progress, for callers that must not claim a finished import ──────
# Enrichment is a THREAD, not a `rail_jobs.db` row, so `atoms_tools._import_outstanding` cannot see
# it by reading the rail table — see `mcp_server/atoms_tools.py`. This is what it reads instead.
_STATE_LOCK = threading.Lock()
_STATE: dict = {"running": False, "passes": 0, "deferred": None, "added": 0,
                "started_at": None, "status": None}


def state() -> dict:
    """A snapshot of what Enrichment is doing right now. Never raises, never blocks on the work."""
    with _STATE_LOCK:
        return dict(_STATE)


def is_running() -> bool:
    return bool(state()["running"])


def _set(**kw) -> None:
    with _STATE_LOCK:
        _STATE.update(kw)


def seconds_until_window_resets(now: float | None = None) -> float:
    """How long until the `TweetDetail` bucket can answer again — 0.0 when it already can.

    Read off x.com's own `x-rate-limit-reset` header via `core.rate_budget`, so nothing here
    encodes X's numbers. UNKNOWN MEANS GO, matching `_refuse_if_spent`: the meter is
    process-local and a fresh process starts blind, so treating no-evidence as spent would make
    a restarted server sleep 15 minutes before its first request.

    (This used to cite `oracles._x_budget_spent` as the other example. That was a PREDICTIVE
    reserve and it was deleted 2026-09-14 along with its twin on the refresh rail — durable
    partial walks removed the premise both rested on. The unknown-means-go rule is unaffected:
    it belongs to reading the meter, not to reserving against it.)
    """
    from pipeline.ingestion import x_graphql_core as core

    now = time.time() if now is None else now
    st = core.rate_budget(core.TWEETDETAIL_OP)
    if st is None:
        return 0.0
    remaining, reset = st
    if remaining > 0 or now >= reset:
        return 0.0
    return (reset - now) + _RESET_SLACK


def run_enrichment(*, limit: int = 0, max_passes: int = _MAX_PASSES,
                   should_stop=None, sleep=time.sleep) -> dict:
    """Upgrade every bookmark that is still owed a conversation or an image read. Never raises.

    One pass per `TweetDetail` window, sleeping between them. Stops when:

      • nothing is owed (`deferred == 0`) — the finished case;
      • the meter is NOT what is holding us back and the pass made no progress — an article whose
        body X never ships, a deleted conversation. Another window cannot fix those, and sleeping
        through fifteen minutes to re-learn that is the runaway this check exists to prevent;
      • the walk itself failed (a dead cookie is `error`, and needs a person, not a window);
      • the single-flight lease was reclaimed, `should_stop` fired, or `max_passes` tripped.

    `sleep` is injected so tests drive the whole loop without wall-clock time. `should_stop` is
    checked between passes, so a cancel never interrupts a pass mid-write.
    """
    load_rail_env()

    from . import bookmark_catchup, ingest_common
    if not bookmark_catchup.consented():
        return {"status": "needs_consent", "passes": 0,
                "message": "bookmark import has not been consented to; nothing to enrich."}
    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "passes": 0, "message": reason}

    from pipeline.sync_lock import CatchupLock
    from . import ingest_x, schema
    from .embed import get_kb_embedder

    _set(running=True, passes=0, deferred=None, added=0, started_at=time.time(), status="running")
    passes = added = 0
    deferred = None
    status = "done"
    try:
        while passes < max_passes:
            if should_stop and should_stop():
                status = "cancelled"
                break
            # SHARES the bookmark rail's lease name on purpose. The scheduled catch-up walks the
            # SAME corpus against the SAME 150/15-min bucket, so a second lease name would let the
            # two run at once and split a meter the whole design is organized around. Skipping
            # because the rail holds it is not a loss: the rail's own pass does this work.
            with CatchupLock("bookmark-catchup") as lock:
                if not lock.acquired:
                    status = "already_running"
                    break
                conn = schema.connect()
                try:
                    out = ingest_x.sync_bookmarks(conn, get_kb_embedder(), limit=limit,
                                                  enrich=True, should_stop=lock.lost)
                finally:
                    conn.close()
            passes += 1
            added += out.get("added", 0)
            previous, deferred = deferred, out.get("deferred", 0)
            _set(passes=passes, deferred=deferred, added=added)

            if out.get("stopped") == "lease_lost":
                status = "lease_lost"
                break
            if out.get("error"):
                # `classify_run` separates the transient meter from the thing that needs a person.
                # A dead cookie reaches here, and no number of windows fixes it.
                status = ("blocked" if ingest_common.classify_run(out) == ingest_common.RUN_BLOCKED
                          else "error")
                break
            if deferred == 0:
                break

            wait = seconds_until_window_resets()
            if wait <= 0 and previous is not None and deferred >= previous:
                # Not the meter, and no ground gained. Whatever is still owed is not waiting on a
                # window — report it rather than spin.
                status = "no_progress"
                break
            if wait > 0:
                log(f"[{RAIL}] {deferred} bookmarks still owe their thread context; "
                    f"the TweetDetail window refills in {int(wait)}s.")
                sleep(wait)
        else:
            status = "max_passes"
            log(f"[{RAIL}] stopped after {passes} passes with {deferred} still owed — a guard "
                f"tripped, not a finished import.")
    except Exception as e:                     # fail-safe: Enrichment never takes anything down
        status = "error"
        log(f"[{RAIL}] run_enrichment errored: {type(e).__name__}: {e}")
        return {"status": status, "passes": passes, "added": added, "deferred": deferred,
                "error": f"{type(e).__name__}: {e}"}
    finally:
        _set(running=False, status=status)
    return {"status": status, "passes": passes, "added": added, "deferred": deferred}


def start_background(*, should_stop=None) -> dict:
    """Run `run_enrichment` on a daemon thread and say so — the shape every background start in
    this repo uses (`sitting_tools._first_pull_background`, `onboard_tools._spawn`).

    THE THREAD OPENS ITS OWN CONNECTIONS: SQLite connections are thread-bound and the caller's is
    mid-request. Fail-safe in layers — `run_enrichment` never raises, the thread body swallows
    anything above it, and a spawn failure is reported rather than thrown. A run that dies with
    the process is recovered by the `bookmark_catchup` rail, which sees exactly the same unmarked
    tweet ids as ordinary never-read ones.
    """
    if is_running():
        return {"status": "running", "note": "enrichment is already in flight"}

    def _go():
        try:
            run_enrichment(should_stop=should_stop)
        except Exception:
            pass

    try:
        threading.Thread(target=_go, name="opyt-enrichment", daemon=True).start()
    except Exception as e:
        log(f"[{RAIL}] could not start: {type(e).__name__}: {e}")
        return {"status": "not_started", "error": f"{type(e).__name__}: {e}"}
    return {"status": "running",
            "note": ("enrichment is running in the background — your saved posts are already "
                     "searchable; their thread context and image descriptions fill in over the "
                     "next few of x.com's 15-minute windows. Do NOT wait for it or poll.")}
