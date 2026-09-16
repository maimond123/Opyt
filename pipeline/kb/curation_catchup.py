"""
pipeline/kb/curation_catchup.py — the automatic refresh of the Proposer's candidate list.

What it does: keep the Proposer's candidate list current, so a person you followed yesterday —
or whose paper you saved yesterday — becomes a candidate without anyone typing a command. Two
kinds of producer, and they are gated differently: the five people-only collectors (X Lists, X
following, X likes, Substack follows, Substack subscriptions) walk an EXTERNAL list over a cookie
session, and the two paper derivations walk atoms already on disk.

Four of the six collectors writing `curation_signals` (X Lists/following/likes, Substack subs)
had no automatic refresh trigger, so a stale candidate LIST looked identical to a fresh one with
nothing new — an invisible freeze, not a slow one. The resident worker launches this rail now;
`onboard` activates it when the user consents.

`derive_paper_signals` was missing here for the same reason and with a worse symptom: it lives in
`curation_pull`, which is hand-run, so a paper deposited through `hopper` reached the screen only
if somebody remembered to run that CLI. Measured 2026-09-08 on the live store — zero occurrences
of "paper" across the whole `curation_catchup.log`, against 11 `user-saved` paper atoms. It runs
UNCONDITIONALLY here, because new papers arrive between passes with no collector having run.

It is also a PRODUCER: a pass in which a collector actually ran makes `candidate_probe` due, since
this is the only thing that mints the candidates that rail pulls content for (see
`run_curation_catchup`, which is where that chaining lives so BOTH doors into this rail get
it — the CLI child and `onboard`'s in-process Arm A walk).

No request ceiling and NO `models_unroutable` preflight, unlike `bookmark_catchup`: nothing here
takes an `embedder`, so nothing here calls a model. The
preflight came off on 2026-09-05, by the same finding that took it off `frontier_execute` the day
before — a per-pass catalog round-trip checking models this rail never calls. Verified, not
assumed: the transitive call graph from the five collectors, `derive_paper_signals` and
`resolve_after_pull` reaches no `llm_client`, no `embed` and no `vision` — `paper_authors` imports
`json`, `sqlite3` and `schema` and nothing else. Do not re-add a preflight here without first
naming the model call it protects.

It IS consent-gated, though — not for money but for the managed X and generic Substack sessions it
reads; see the Consent block for the cold-start behaviour that forced that. Beyond consent it needs
single-flight plus a coalesce window, so a free scrape doesn't run four times at once against X.

The paper derivation sits behind that same early return and needs no exemption, because the gate
cannot block it in any reachable state: `consented()` is `marker OR _established_store()`, and
`_established_store()` is "any atom exists" — so a home holding a paper to derive from has already
cleared it. Do not hoist the derivation above the gate to "fix" this; the hoist would move a write
outside the single-flight lock and buy nothing.

Trigger rate is not pull rate: the worker's cadence is hourly, `FLOOR_HOURS` decides whether any
collector actually runs.

"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb.rail_runtime import load_rail_env
from pipeline.ingestion.utils import log

# How long a collector must go unattempted before this rail re-runs it. Gates on `last_attempt_at`,
# not `last_ok_at` (see `curation_state.is_due`), so a collector with a dead X session retries every
# six hours instead of on every pass.
FLOOR_HOURS = 6.0

# ── Consent ─────────────────────────────────────────────────────────────────────
# These five collectors make no model calls, and for a long time that was reason enough to leave
# this rail ungated. A cold-start test on 2026-08-20 showed why model use is the wrong axis: on a brand-new
# install, before `onboard` ran and before the user had typed anything about OPYT, this rail read
# their Chrome cookie jar and made requests to X and Substack on their session. Back then that also
# raised an unexplained macOS credential prompt in the user's first minute; Chromium reads stopped
# doing that on 2026-08-30, and the gate does not depend on it. Free is not the same as
# unsurprising, and reading somebody's logged-in session needs explicit consent whether or not the
# OS says anything — which is exactly why `bookmark_catchup.consented()` warns that opting into one
# loop must never silently opt you into another.
#
# The gate is deliberately the same shape as the other rails': a marker file OR an established
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


def _platform_reachable(platform: str) -> bool:
    """Can this collector's platform be read at all right now?

    X reads ONLY the OPYT-managed profile — the `retired-x-existing-browser-onboarding` guard
    says so — and with no such session every X collector fails identically with `no_viewer_id`.
    Three recorded failures per pass, forever, on a home whose user reads Substack. So the X
    collectors are gated on that session existing.

    Substack answers this differently local vs hosted, and `sources.substack.readable` owns that
    rule — `substack_saved_catchup` asks it too, and one rule with two homes is how the two rails
    would come to disagree about whether a session exists.

    Fail-safe: an unreadable probe reads as REACHABLE, so the collector attempts and records a
    real outcome. The opposite default would silently skip work over a transient error.
    """
    try:
        if platform == "x":
            from pipeline.ingestion.x_graphql import has_managed_x_session
            return has_managed_x_session()
        from pipeline.ingestion.sources.substack import readable
        return readable()
    except Exception:
        return True


def run_curation_catchup(force: bool = False, floor_hours: float = FLOOR_HOURS,
                         platforms: set[str] | None = None) -> dict:
    """Refresh every collector that is past its floor. Never raises.

    `platforms` narrows the pass to those platforms' collectors; None (the rail's own case) means
    all of them. `onboard` passes the platforms whose collectors have never run, because platforms
    are connected ONE AT A TIME and a pass triggered by connecting X has no reason to re-walk
    Substack — `force=True` would bypass that platform's floor and spend its requests again.

    Gated on `consented()` — no model call, but not free of surprise; see the Consent block.
    `force=True` grants consent rather than bypassing it: a user who explicitly asks for this
    pass has, by asking, opted in. This is the LAST rail where that is true, and it is true
    because `onboard` calls this one IN-PROCESS, holding the answer the user just gave. The
    detached rails lost their `--force` for exactly that reason — a child's argv carries no
    answer.

    Must NOT call `curation_pull(tiered=True)`: that ladder's gate reads the whole store's
    signalled-entity count, not this run's own yield, so on an established store it silently skips
    following/likes after Tier 1 — the two collectors this rail exists to refresh. Calls the four
    directly instead, through the same `run_and_record` dispatch the hand-run uses.

    `force=True` ignores the floor but does NOT bypass single-flight — two passes at once is the
    one thing force must not buy.

    ⚠️ IT IS ALSO THE PRODUCER OF `candidate_probe`, AND THAT CHAINING LIVES HERE — not in
    `main()`, where it sat until 2026-09-16. `main()` is reachable only through
    `python -m pipeline.kb.curation_catchup --once`, i.e. the rail worker's child, and
    `onboard_tools._run_curation` calls this function DIRECTLY. So on a fresh install the
    in-process Arm A walk minted the candidates and nothing ever queued the probe that pulls
    their content: no row and no `candidate_probe.log`, measured on a clean home 2026-09-16.
    A rail's scheduling side-effects belong to the rail, and every caller of the body must get
    them — which is what putting them on the public entry point, above the CLI, buys.

    Conditional on `ran`, not unconditional: a pass where every collector sat inside its floor
    minted no candidate, and re-queueing regardless is an hourly timer wearing the probe's name.
    """
    res = _run(force=force, floor_hours=floor_hours, platforms=platforms)
    # `ran` is present on the `ok` return and on both `lease_lost` returns — a lease reclaimed
    # after two collectors walked still minted their candidates, so the probe is still due.
    if res.get("ran"):
        from pipeline.kb.rail_jobs import request_now
        request_now("candidate_probe")
    return res


def _run(*, force: bool, floor_hours: float, platforms: set[str] | None) -> dict:
    """The pass itself. Private so `run_curation_catchup` is the only door, and therefore the
    only place the `candidate_probe` chain can be skipped from."""
    load_rail_env()

    # Before the lock, before any collector: an unconsented pass must not even reach the cookie jar.
    if force:
        grant_consent()
    elif not consented():
        return {"status": "needs_consent",
                "message": "Keeping your candidate list current reads your OPYT-managed X "
                           "session and your Substack session. Connect one with "
                           "`onboard(source='x')` or `onboard(source='substack')`, then call "
                           "`onboard` again to grant this refresh."}

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
                no_session: list[str] = []
                # ONE probe per PLATFORM per pass, not one per collector. The question is about a
                # platform's session and cannot change mid-pass — and on a hosted home each probe
                # is a live `/inbox` fetch, so the two Substack collectors asking separately spent
                # a duplicate request against the host whose rate limit is the whole constraint.
                # A session that dies between the two collectors is caught by the second one's own
                # refusal, which records a real outcome; the probe was never what protected that.
                reachable: dict[str, bool] = {}
                for spec in ingest_curation.COLLECTOR_SPECS:
                    if platforms is not None and spec.platform not in platforms:
                        continue          # not this pass's platform; its own clock is untouched
                    if lock.lost():
                        return {"status": "lease_lost", "ran": ran,
                                "skipped_within_floor": skipped,
                                "skipped_no_session": no_session,
                                "message": "catch-up lease was reclaimed; stopped before the next collector"}
                    if spec.platform not in reachable:
                        reachable[spec.platform] = _platform_reachable(spec.platform)
                    if not reachable[spec.platform]:
                        no_session.append(spec.collector)
                        continue
                    row = curation_state.get_run(conn, spec.collector)
                    if not force and not curation_state.is_due(row, floor_hours=floor_hours):
                        skipped.append(spec.collector)
                        continue
                    # `run_and_record` is failure-isolated per collector and stamps the clock
                    # itself, so one dead session cannot stop the other four — the same property
                    # `curation_pull` has, from the same code.
                    ran[spec.collector] = ingest_curation.run_and_record(conn, spec)
                if lock.lost():
                    return {"status": "lease_lost", "ran": ran,
                            "skipped_within_floor": skipped,
                            "skipped_no_session": no_session,
                            "message": "catch-up lease was reclaimed; stopped before resolving"}
                errors = sum(1 for r in ran.values() if "error" in r)
                # UNCONDITIONAL, unlike everything above it. The five collectors walk an external
                # list and are gated on a floor and a session; this walks atoms already on disk,
                # and new papers arrive between passes from `hopper` and `bookmark_catchup` with
                # no collector having run. Gating it on `ran` would mean a home with no X and no
                # Substack — the research reader this rail exists to serve as much as anyone —
                # never derives an author at all.
                derived = ingest_curation.derive_paper_signals(conn)
                authors = (derived["paper_authors"] or {}).get("authors") or 0
                # BEFORE resolve, and resolve now runs when EITHER producer found something: an
                # author entity minted a line above is exactly the row that needs merging with
                # the `scholar:` or `x:user:` sibling already in the store, and an unmerged person
                # is two candidates carrying one signal each — below the pre-tick bar, filtered
                # out before a human sees them.
                resolved = (ingest_curation.resolve_after_pull(conn)
                            if (ran or authors) else None)
                log(f"[curation-catchup] ran {len(ran)}, skipped {len(skipped)} inside the "
                    f"{floor_hours:g}h floor, {len(no_session)} with no session, "
                    f"{errors} errored, {authors} paper author(s) derived, resolve={resolved}")
                return {"status": "ok", "ran": ran, "skipped_within_floor": skipped,
                        "skipped_no_session": no_session, **derived,
                        "errors": errors, "floor_hours": floor_hours, "resolve": resolved,
                        "freshness": curation_state.status_summary(
                            conn, ingest_curation.COLLECTORS)}
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[curation-catchup] run_curation_catchup errored: {detail}")
        return {"status": "error", "error": detail}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Curation catch-up — refresh the Proposer's candidate list (X Lists / "
                    "following / likes, Substack follows / subscriptions)")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--floor-hours", type=float, default=FLOOR_HOURS,
                    help=f"hours a collector must go unattempted before a re-run "
                         f"(default {FLOOR_HOURS:g})")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    # NOTHING IS QUEUED HERE, deliberately. `main()` is one of two doors into this rail and the
    # other one — `onboard_tools._run_curation` — never opens it, so a `request_now` in this body
    # is a side-effect the in-process caller silently loses. The `candidate_probe` chain lives in
    # `run_curation_catchup`; `tests/kb/test_rail_activation.py` fails any rail that puts one back
    # inside a `main()`.
    res = run_curation_catchup(floor_hours=args.floor_hours)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "already_running"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
