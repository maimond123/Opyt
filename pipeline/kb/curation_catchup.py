"""
pipeline/kb/curation_catchup.py — the automatic refresh of the Proposer's candidate list.

What it does: on session open, re-read the four people-only curation collectors — X Lists, X
following, X likes, Substack subscriptions — so a person you followed yesterday becomes a candidate
without anyone typing a command.

Four of the six collectors writing `curation_signals` (X Lists/following/likes, Substack subs)
had no automatic refresh trigger, so a stale candidate LIST looked identical to a fresh one with
nothing new — an invisible freeze, not a slow one. This rail owns its own spawner in
`mcp_server/server.py` (enforced by the `atom-rail-not-welded-to-catchup` guard).

No budget ceiling, unlike `bookmark_catchup`: these four collectors take no `embedder`, so nothing
here spends money. It IS consent-gated, though — not for money but for the browser session it
reads; see the Consent block for the cold-start behaviour that forced that. Beyond consent it
needs single-flight plus a coalesce window, so a free scrape doesn't run four times at once
against X.

Trigger rate is not pull rate: the spawn coalesces hourly, `FLOOR_HOURS` decides whether any
collector actually runs.

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb.rail_runtime import (COALESCE_DEFAULT, load_rail_env,
                                      models_unroutable, spawn_rail)
from pipeline import llm_client
from pipeline.ingestion.utils import log

# How long a collector must go unattempted before this rail re-runs it. Gates on `last_attempt_at`,
# not `last_ok_at` (see `curation_state.is_due`), so a collector with a dead X session retries every
# six hours instead of on every session open.
FLOOR_HOURS = 6.0

# The rail label this pass's spend is attributed under. No daily ceiling beside it — the four
# collectors are free, so a ceiling would conflate "spent its money" with "spends no money" in
# `budget_paused`. Still labelled so a future paid change to this rail isn't invisible.
RAIL = "curation_catchup"


# ── Consent ─────────────────────────────────────────────────────────────────────
# These four collectors spend NO money, and for a long time that was reason enough to leave this
# rail ungated. A cold-start test on 2026-08-20 showed why money is the wrong axis: on a brand-new
# install, before `onboard` ran and before the user had typed anything about OPYT, this rail read
# their Chrome cookie jar and made requests to X and Substack on their session. Back then that also
# raised an unexplained macOS credential prompt in the user's first minute; Chromium reads stopped
# doing that on 2026-08-30, and the gate does not depend on it. Free is not the same as
# unsurprising, and reading somebody's logged-in session is its own cost shape whether or not the
# OS says anything — which is exactly why `bookmark_catchup.consented()` warns that opting into one
# loop must never silently opt you into another.
#
# The gate is deliberately the same shape as the paid rails': a marker file OR an established
# store. The second half is what keeps an existing user from being re-prompted for something that
# has been running for months.
def _consent_marker() -> Path:
    """Resolved at call time so it honors `$OPYT_HOME` (Distributable: derive paths at runtime)."""
    return Path(os.environ.get("OPYT_CURATION_CATCHUP_CONSENT",
                              opyt_path("curation_catchup_consent")))


def consented() -> bool:
    """Has the user opted into the automatic cookie-scrape? Marker file OR an established store.

    `_established_store` is borrowed from `bookmark_catchup` rather than re-implemented: the
    "does this user already have content" question is one heuristic and two copies would drift.
    The MARKERS stay separate — that is the part that must never be shared.
    """
    from pipeline.kb.bookmark_catchup import _established_store
    return _consent_marker().exists() or _established_store()


def grant_consent() -> None:
    """Record consent — called by `onboard` once the browser step settles, which is the moment the
    user has actually accepted reading their browser session, and by an explicit `force` run."""
    try:
        marker = _consent_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass


@llm_client.rail(RAIL)
def run_curation_catchup(force: bool = False, floor_hours: float = FLOOR_HOURS) -> dict:
    """Refresh every collector that is past its floor. Never raises.

    Gated on `consented()` — free of money, but not free of surprise; see the Consent block.
    `force=True` grants consent rather than bypassing it, matching `run_bookmark_catchup`: a user
    who explicitly asks for this pass has, by asking, opted in.

    Must NOT call `curation_pull(tiered=True)`: that ladder's gate reads the whole store's
    signalled-entity count, not this run's own yield, so on an established store it silently skips
    following/likes after Tier 1 — the two collectors this rail exists to refresh. Calls the four
    directly instead, through the same `run_and_record` dispatch the hand-run uses.

    `force=True` ignores the floor (matching `run_bookmark_catchup`) but does NOT bypass
    single-flight — two passes at once is the one thing force must not buy.
    """
    load_rail_env()

    # Before the lock, before any collector: an unconsented pass must not even reach the cookie jar.
    if force:
        grant_consent()
    elif not consented():
        return {"status": "needs_consent",
                "message": "Keeping your candidate list current reads your logged-in X and "
                           "Substack sessions from your browser (free — no API credits). Run "
                           "`onboard` once to settle which browser profile to use; that step is "
                           "where this is granted."}

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock

    from . import curation_state, ingest_curation, schema
    try:
        with CatchupLock("curation-catchup") as lock:
            if not lock.acquired:
                # Another session's pass holds the lease — skipping is correct, not a failure, since
                # a concurrent duplicate pass would double the request count against a cookie session.
                return {"status": "already_running",
                        "message": "another curation catch-up is in flight — skipped "
                                   "(single-flight)."}
            conn = schema.connect()
            try:
                ran: dict[str, dict] = {}
                skipped: list[str] = []
                for spec in ingest_curation.COLLECTOR_SPECS:
                    row = curation_state.get_run(conn, spec.collector)
                    if not force and not curation_state.is_due(row, floor_hours=floor_hours):
                        skipped.append(spec.collector)
                        continue
                    # `run_and_record` is failure-isolated per collector and stamps the clock
                    # itself, so one dead session cannot stop the other three — the same property
                    # `curation_pull` has, from the same code.
                    ran[spec.collector] = ingest_curation.run_and_record(conn, spec)
                errors = sum(1 for r in ran.values() if "error" in r)
                # This rail bypasses `curation_pull` (see the trap note above) so it can't reach
                # `curation_pull._done`'s resolve call — this is the only other one. Skipped when
                # nothing ran: no network call means no new entity to merge.
                resolved = ingest_curation.resolve_after_pull(conn) if ran else None
                log(f"[curation-catchup] ran {len(ran)}, skipped {len(skipped)} inside the "
                    f"{floor_hours:g}h floor, {errors} errored, resolve={resolved}")
                return {"status": "ok", "ran": ran, "skipped_within_floor": skipped,
                        "errors": errors, "floor_hours": floor_hours, "resolve": resolved,
                        "freshness": curation_state.status_summary(
                            conn, ingest_curation.COLLECTORS)}
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[curation-catchup] run_curation_catchup errored: {detail}")
        return {"status": "error", "error": detail}


# ── The detached spawn ──────────────────────────────────────────────────────────
def spawn_curation_catchup(force: bool = False, coalesce_window: float = COALESCE_DEFAULT) -> bool:
    """Fire one catch-up as a detached, non-blocking child and return immediately.

    Copied from `spawn_bookmark_catchup`, which already carries both fixes the deleted
    `mcp_server.catchup.spawn_detached` lacked: the log file descriptor is closed in the PARENT
    after `Popen` (the child has already dup'd it, and the MCP server is long-lived), and the
    coalesce stamp is touched only AFTER `Popen` succeeds (stamping first means a failed fork burns
    the whole window and every session inside it declines to retry). `CatchupLock` makes a
    redundant spawn harmless, so the double-spawn race that ordering trades for is the cheaper side.
    """
    return spawn_rail("pipeline.kb.curation_catchup", slug="curation_catchup",
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Curation catch-up — refresh the Proposer's candidate list (X Lists / "
                    "following / likes, Substack subscriptions)")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--force", action="store_true", help="ignore the per-collector floor")
    ap.add_argument("--floor-hours", type=float, default=FLOOR_HOURS,
                    help=f"hours a collector must go unattempted before a re-run "
                         f"(default {FLOOR_HOURS:g})")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_curation_catchup(force=args.force, floor_hours=args.floor_hours)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "already_running"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
