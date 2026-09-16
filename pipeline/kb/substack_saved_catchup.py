"""
pipeline/kb/substack_saved_catchup.py — automatic Substack saved-posts catch-up, on the atom rail.

Walks the user's Substack "Saved posts" and lands the new ones as `entry_mode='user-saved'`
opinion atoms via `ingest_curation.sync_substack_saved`. The other half of the same curation act
`bookmark_catchup` covers: content the user personally kept, which becomes both an atom and a
`save` curation signal. Feeds the Proposer's candidate list and Frontier stage 1, which reads
`entry_mode='user-saved'` atoms regardless of source.

Launched by the resident worker (`pipeline/kb/rail_worker.py`), activated by backlog consent in
`onboard`.

Its own rail, not a second arm of `bookmark_catchup`, and the reason is in that module's own
consent docstring: opting into one loop must never silently opt you into another with a different
request pattern. X's bound is money over a free cookie-scrape; this one's is a Cloudflare-guarded
reader endpoint that 403s under bursty automation. Two pacing policies in one rail would be wrong
for both, and a Substack refusal would stall the X import. Not a third arm of `curation_catchup`
either: that rail is model-free by construction and this one embeds and reads images.

CADENCE, not a coalesce window: six hours, set in `rail_worker.RAILS`. `bookmark_catchup` runs
hourly because its steady-state pass is a free scrape; a steady-state pass here is a request to
`substack.com/api/v1/reader/saved`, which the transport's own docstring says to call sparingly.
The cadence also bounds the one loop this rail could otherwise create — a post whose body fetch
was BLOCKED is stored as a `pending` stub that every later pass re-fetches, by design, so that a
temporary block does not freeze into a permanent hole. Hourly that is a retry storm against the
host that just refused; four times a day it is a freshness poll. The transient case is handled
where it belongs instead, by the retry inside `_fetch_full_post`.

Consent: a dedicated `substack_saved_catchup_consent` marker, written by `onboard`. Deliberately
NO "an established store implies consent" clause — see the Consent block. Never raises: a
catch-up failure is reported, not propagated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb.rail_runtime import load_rail_env, models_unroutable
from pipeline.ingestion.utils import log
from pipeline.kb import ingest_common

RAIL = "substack_saved_catchup"

# `classify_run`'s verdict → this rail's status word, exactly as `bookmark_catchup` maps it:
# anything unlisted is `ok`, which is only RUN_INGESTED.
_RUN_STATUS = {ingest_common.RUN_BLOCKED: "blocked", ingest_common.RUN_ERROR: "error"}


# ── Consent ─────────────────────────────────────────────────────────────────────
# Importing a saved-posts backlog fetches every post at full body, cleans it, chunks it, embeds
# it, and reads its images with a VLM. Same cost shape as the X bookmark backlog, same
# irreversibility, so it gets the same treatment: an explicit answer, recorded in this rail's own
# marker.
#
# ⚠️ NO `_established_store()` FALLBACK, and that is the one place this rail differs from the
# other three. `bookmark_catchup` and `curation_catchup` accept an established store as consent
# because both were already running before their marker existed, so the clause grandfathers
# users who would otherwise be re-asked about something months old. Nothing has ever run
# `sync_substack_saved` unattended, so there is no such population — the clause would instead arm
# a brand-new metered import on every existing home, the first time a worker got to it, with the
# question never having been put. A user who answered `backlog` to a prompt that named only X
# consented to X.
def _consent_marker() -> Path:
    """Resolved at call time so it honors `$OPYT_HOME` (Distributable: derive paths at runtime)."""
    return Path(os.environ.get("OPYT_SUBSTACK_SAVED_CATCHUP_CONSENT",
                              opyt_path("substack_saved_catchup_consent")))


def consented() -> bool:
    """Has the user opted into the Substack saved-posts import? The marker, and only the marker."""
    return _consent_marker().exists()


def grant_consent() -> None:
    """Record consent — called by the surface that ASKED, once the user has said yes."""
    try:
        marker = _consent_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass


# ── The entrypoint ──────────────────────────────────────────────────────────────
def run_substack_saved_catchup() -> dict:
    """Load creds → gate on consent → check the platform is readable → single-flight → ingest.
    Never raises.

    Consent is read, never granted here — the same split `bookmark_catchup` makes. A caller that
    has just been told "yes" calls `grant_consent()` itself and then queues the rail, which is
    what `onboard` does; a `force` that granted consent on the caller's behalf would make one word
    mean both "run now" and "the user agreed".
    """
    load_rail_env()

    if not consented():
        return {"status": "needs_consent",
                "message": ("Importing your Substack saved posts fetches each one at full body "
                            "and reads the images it carries. Run `onboard` once to import them "
                            "and enable automatic catch-up — it asks for this as a ONE-TIME, "
                            "bounded backlog import, separately from the recurring Oracle "
                            "refresh.")}

    from pipeline.ingestion.sources.substack import readable
    if not readable():
        # Hosted only: no managed Substack session means Chrome would launch, be served the
        # logged-out reader page, and report zero saved posts — which reads as "you saved
        # nothing" rather than "Substack is not connected". Locally `readable()` is always True,
        # because the collector may use the user's own browser.
        return {"status": "no_session",
                "message": "Substack is not connected on this home — `onboard(source='substack')`."}

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock
    try:
        with CatchupLock("substack-saved-catchup") as lock:
            if not lock.acquired:
                return {"status": "already_running",
                        "message": "another Substack saved catch-up is in flight — skipped "
                                   "(single-flight)."}
            if lock.lost():
                return {"status": "lease_lost",
                        "message": "substack-saved-catchup lease was reclaimed before work began."}
            from . import ingest_curation, schema
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            conn = schema.connect()
            try:
                out = ingest_curation.sync_substack_saved(conn, embedder)
                # The rail's status is the adapter's own verdict, never a default of "ok" — a
                # dead session reaching here as `status: ok, added: 0` is indistinguishable from
                # a quiet week, and a rail that believes it succeeded does not retry.
                status = _RUN_STATUS.get(ingest_common.classify_run(out), "ok")
                return {"status": status, **out}
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[substack-saved-catchup] run_substack_saved_catchup errored: {detail}")
        return {"status": "error", "error": detail}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Substack saved catch-up — saved posts into user-saved atoms")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_substack_saved_catchup()
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
