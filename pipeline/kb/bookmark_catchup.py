"""
pipeline/kb/bookmark_catchup.py — automatic X-bookmark catch-up, on the atom rail.

Walks the user's X bookmarks (free cookie-scrape) and lands the new ones as
`entry_mode='user-saved'` opinion atoms via `ingest_x.sync_bookmarks`. Feeds the Proposer's
candidate list, Frontier stage 1, and the Frontier live-validation precondition.

Launched by the resident worker (`pipeline/kb/rail_worker.py`), activated by bookmark consent in
`onboard`. Trigger rate is not pull rate: the worker's hourly cadence governs how often this rail
asks; the snapshot hash governs whether any ingest work is needed.

Consent: a dedicated `bookmark_catchup_consent` marker, granted by `onboard` or implied by a
store that already holds content. Never raises — a catch-up failure is reported, not propagated.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db, opyt_path
from pipeline.kb.rail_runtime import load_rail_env, models_unroutable
from pipeline.ingestion.utils import log
from pipeline.kb import ingest_common

# ── Tuning knobs ────────────────────────────────────────────────────────────────
# How deep one pass walks the bookmark list; 0 = the whole list. 0 is load-bearing, not a
# placeholder: `iterate_bookmarks` yields newest-first with no persisted cursor, so any nonzero
# limit truncates from the newest end and silently strands older bookmarks forever (never
# retried, and a truncated run looks identical to a caught-up one). The walk itself is free;
BACKLOG_LIMIT = 0
RAIL = "bookmark_catchup"

# `classify_run`'s verdict → this rail's status word. Anything not listed is `ok`, which is only
# RUN_INGESTED. Mapped rather than passed through so the rail's vocabulary stays its own: `ok` is
# not one of `classify_run`'s three, and `ingested` is not one a caller of this rail reads.
_RUN_STATUS = {ingest_common.RUN_BLOCKED: "blocked", ingest_common.RUN_ERROR: "error"}

# ── Consent ─────────────────────────────────────────────────────────────────────
# Importing a new user's bookmark backlog invokes model-backed processing (the VLM image reads and
# embeddings in `sync_bookmarks`) and must never
# fire silently on first launch. An established store is
# auto-consented so an existing user is never re-prompted.
def _consent_marker() -> Path:
    """Resolve the marker path at call time so it honors `$OPYT_HOME` (Distributable: derive paths
    at runtime). The deleted `hot_feed` bound its marker at IMPORT, which resolves to the wrong
    home under a sandboxed `$OPYT_HOME` and, in tests, to the real one."""
    return Path(os.environ.get("OPYT_BOOKMARK_CATCHUP_CONSENT",
                              opyt_path("bookmark_catchup_consent")))


def _established_store() -> bool:
    """Stored atoms imply consent; a missing or unreadable atom store does not."""
    try:
        c = sqlite3.connect(f"file:{opyt_db()}?mode=ro", uri=True)
    except Exception:
        return False
    try:
        return c.execute("SELECT 1 FROM atoms LIMIT 1").fetchone() is not None
    except sqlite3.Error:
        return False
    finally:
        c.close()


def consented() -> bool:
    """Has the user opted into the bookmark import? Marker file OR an established store.

    Each rail owns its own marker and they must stay separate — `oracle_refresh._consent_marker`
    and `radar/refresh._consent_marker` are deliberately distinct files. Opting into one loop
    must never silently opt you into another with a different request pattern.
    """
    return _consent_marker().exists() or _established_store()


def grant_consent() -> None:
    """Record consent — called by the surface that ASKED, once the user has said yes."""
    try:
        marker = _consent_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass


# ── The entrypoint ──────────────────────────────────────────────────────────────
def run_bookmark_catchup(limit: int = BACKLOG_LIMIT) -> dict:
    """Load creds → gate on consent → single-flight → ingest. Never raises.

    The rail label goes on `run_*`, not `main()` — `main()` only wraps this for the `--once`
    child, so labelling it would miss every in-process call the MCP side makes directly.

    Consent is read, never granted here. A caller that has just been told "yes" calls
    `grant_consent()` itself and then runs — which is what `onboard` does. A `force` that granted
    it on the caller's behalf made one word mean both "run now" and "the user agreed".
    """
    load_rail_env()

    if not consented():
        return {"status": "needs_consent",
                "message": ("Importing your X bookmarks resolves threads and reads images on the "
                            "ones that carry media. Run `onboard` once to "
                            "import them and enable automatic catch-up — it asks for this as a "
                            "ONE-TIME, bounded backlog import, separately from the recurring "
                            "Oracle refresh.")}

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock
    try:
        with CatchupLock("bookmark-catchup") as lock:
            if not lock.acquired:
                # Another catch-up holds the lease — skipping is correct, not a failure. The
                # worker never runs two rails for one home at once, so the live producer of this
                # collision is a hand-run `--once` beside the worker's own child: atoms are
                # idempotent, but duplicate work still wastes the shared X session.
                return {"status": "already_running",
                        "message": "another bookmark catch-up is in flight — skipped "
                                   "(single-flight)."}
            if lock.lost():
                return {"status": "lease_lost",
                        "message": "bookmark-catchup lease was reclaimed before work began."}
            from . import ingest_x, schema
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            conn = schema.connect()
            try:
                # `enrich=True`: the rail is the recurring pass, so it does BOTH halves in one
                # call — writes the atoms of newly saved posts AND pays whatever the meter allows
                # toward the thread/image upgrades still owed. What changed on 2026-09-13 is that a
                # refusal now costs fidelity rather than the atom; `enrichment.run_enrichment` is
                # what waits out the remaining windows, and it shares this lease.
                out = ingest_x.sync_bookmarks(conn, embedder, limit=limit, enrich=True,
                                              should_stop=lock.lost)
                # The rail's status is the adapter's own verdict, not a default of "ok". A dead
                # cookie used to reach here as `status: ok, added: 0` — indistinguishable from a
                # quiet week, and a rail that believes it succeeded does not retry. The adapter
                # signals a hard stop by RETURNING `error`, so this must ask `classify_run`
                # rather than test for a raise.
                if out.get("stopped") == "lease_lost":
                    status = "lease_lost"
                else:
                    status = _RUN_STATUS.get(ingest_common.classify_run(out), "ok")
                return {"status": status, **out}
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[bookmark-catchup] run_bookmark_catchup errored: {detail}")
        return {"status": "error", "error": detail}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bookmark catch-up — X bookmarks into user-saved atoms")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--limit", type=int, default=BACKLOG_LIMIT,
                    help="max bookmarks to ingest (0 = all)")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_bookmark_catchup(limit=args.limit)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
