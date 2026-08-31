"""
pipeline/kb/bookmark_catchup.py — automatic X-bookmark catch-up, on the atom rail.

On session open, walks the user's X bookmarks (free cookie-scrape) and lands the new ones as
`entry_mode='user-saved'` opinion atoms via `ingest_x.sync_bookmarks`. Feeds the Proposer's
candidate list, Frontier stage 1, and the Frontier live-validation precondition.

Owns its own spawner (mirrors `spawn_frontier_execute`) rather than riding inside another job —
`mcp_server/server.py` calls each spawner independently in its own try/except. History for why
that matters:

Trigger rate is not pull rate: the 3600s coalesce governs how often a session asks; the snapshot
hash governs whether anything is paid for.

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
from pipeline.kb.rail_runtime import (COALESCE_DEFAULT, load_rail_env,
                                      models_unroutable, rail_budget_exhausted, spawn_rail)
from pipeline import llm_client
from pipeline.ingestion.utils import log

# ── Tuning knobs ────────────────────────────────────────────────────────────────
# How deep one pass walks the bookmark list; 0 = the whole list. 0 is load-bearing, not a
# placeholder: `iterate_bookmarks` yields newest-first with no persisted cursor, so any nonzero
# limit truncates from the newest end and silently strands older bookmarks forever (never
# retried, and a truncated run looks identical to a caught-up one). The walk itself is free;
# only genuinely-new items are paid for, and that spend is bounded by `BOOKMARK_CATCHUP_DAILY_USD`
# instead. See
# docs/Future-Investigations/2026-08-12-bookmark-catchup-ceiling-is-a-start-gate-not-a-governor.md
BACKLOG_LIMIT = 0

# Resetting daily seatbelt, in RECORDED dollars, matching `ORACLE_REFRESH_DAILY_USD`.
#
# Read ONCE before the run — this is a start gate, not a per-run governor (`sync_bookmarks` has no
# budget check of its own once started, so a run walks to completion at whatever cost). What it
# actually stops is a hash-skip regression that makes every bookmark look new on every hourly
# spawn: run 1 of the day pays a full backlog uncapped, run 2 is refused. The uncapped one-time
# backfill has its own entrypoint and bypasses this: `python -m pipeline.kb.run_ingest --x-limit 0`
# calls `sync_bookmarks` directly. Full rationale for the $1.00 figure:
BOOKMARK_CATCHUP_DAILY_USD = 1.00

# The rail label this ceiling governs — one constant, read by both the `@llm_client.rail`
# decorator on `run_bookmark_catchup` and `_daily_budget_exhausted`. Two separate literals could
# drift silently: spend fills one meter while the ceiling reads an empty one under the other name.
RAIL = "bookmark_catchup"

# ── Consent ─────────────────────────────────────────────────────────────────────
# Importing a new user's bookmark backlog costs API credits (the VLM image reads and embeds in
# `sync_bookmarks`; the walk and thread fetches themselves are free cookie-scrapes) and must never
# fire silently on first launch. An established store is
# auto-consented so an existing user is never re-prompted.
def _consent_marker() -> Path:
    """Resolve the marker path at call time so it honors `$OPYT_HOME` (Distributable: derive paths
    at runtime). The deleted `hot_feed` bound its marker at IMPORT, which resolves to the wrong
    home under a sandboxed `$OPYT_HOME` and, in tests, to the real one."""
    return Path(os.environ.get("OPYT_BOOKMARK_CATCHUP_CONSENT",
                              opyt_path("bookmark_catchup_consent")))


def _established_store() -> bool:
    """True if the user already has content — consent is then implied (a brand-new user has none).

    Tries both `atoms` and `notes`: a migrated store has one or the other but not both, so
    checking only one would re-prompt whichever user type it doesn't cover. A missing table must
    not suppress the other's count.
    """
    try:
        c = sqlite3.connect(f"file:{opyt_db()}?mode=ro", uri=True)
    except Exception:
        return False
    try:
        for table in ("atoms", "notes"):
            try:
                if c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0:
                    return True
            except Exception:
                continue          # table absent on this store — ask the other one
        return False
    finally:
        c.close()


def consented() -> bool:
    """Has the user opted into the paid bookmark import? Marker file OR an established store.

    Each rail owns its own marker and they must stay separate — `oracle_refresh._consent_marker`
    and `radar/refresh._consent_marker` are deliberately distinct files. Opting into one paid loop
    must never silently opt you into another with a different cost shape.
    """
    return _consent_marker().exists() or _established_store()


def grant_consent() -> None:
    """Record consent — called when the user explicitly runs `sync`, or forces a catch-up."""
    try:
        marker = _consent_marker()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass


# ── Spend accounting ────────────────────────────────────────────────────────────
def _daily_budget_exhausted() -> bool:
    """Has this rail's recorded spend today reached its ceiling? See
    `rail_runtime.rail_budget_exhausted` for why it is never the global total."""
    return rail_budget_exhausted(RAIL, BOOKMARK_CATCHUP_DAILY_USD)


# ── The entrypoint ──────────────────────────────────────────────────────────────
@llm_client.rail(RAIL)
def run_bookmark_catchup(force: bool = False, limit: int = BACKLOG_LIMIT) -> dict:
    """Load creds → gate on consent → check the seatbelt → single-flight → ingest. Never raises.

    The rail label goes on `run_*`, not `main()` — `main()` only wraps this for the `--once`
    child, so labelling it would miss every in-process call the MCP side makes directly.

    `force=True` grants consent and runs now (manual escape hatch, matching `run_oracle_refresh` /
    `run_people_refresh`). An un-consented, non-forced call returns `needs_consent` having spent
    nothing.

    The budget check sits outside the lock: a paused run should not take a lease another session
    could be using.
    """
    load_rail_env()

    if force:
        grant_consent()
    elif not consented():
        return {"status": "needs_consent",
                "message": ("Importing your X bookmarks uses API credits (thread resolution plus "
                            "image reading on the ones that carry media). Run `onboard` once to "
                            "import them and enable automatic catch-up — it asks for this as a "
                            "ONE-TIME, bounded backlog import, separately from the recurring "
                            "Oracle refresh.")}

    if _daily_budget_exhausted():
        return {"status": "budget_paused",
                "message": (f"today's recorded API spend reached the "
                            f"${BOOKMARK_CATCHUP_DAILY_USD:.2f} daily bookmark-catchup ceiling. "
                            f"It resets at UTC midnight; the backlog drains next run.")}

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock
    try:
        with CatchupLock("bookmark-catchup") as lock:
            if not lock.acquired:
                # Another session's catch-up holds the lease — skipping is correct, not a
                # failure. Without this, a background spawn and a manual run could walk the same
                # bookmarks at once: atoms are idempotent so the corpus survives, but the bill
                # doesn't.
                return {"status": "already_running",
                        "message": "another bookmark catch-up is in flight — skipped "
                                   "(single-flight)."}
            from . import ingest_x, schema
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            conn = schema.connect()
            try:
                # `sync_bookmarks` returns a summary with no `status` key; stamp one so every
                # return path carries the field callers key on (`main`'s exit code reads it).
                return {"status": "ok", **ingest_x.sync_bookmarks(conn, embedder, limit=limit)}
            finally:
                conn.close()
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[bookmark-catchup] run_bookmark_catchup errored: {detail}")
        return {"status": "error", "error": detail}


# ── The detached spawn ──────────────────────────────────────────────────────────
def spawn_bookmark_catchup(force: bool = False, coalesce_window: float = COALESCE_DEFAULT) -> bool:
    """Fire one catch-up as a detached, non-blocking child and return immediately.

    Copied from `spawn_frontier_execute`, not the now-deleted `mcp_server.catchup.spawn_detached`,
    which had two defects (fd leak, coalesce stamped before `Popen`). Full comparison:
    """
    return spawn_rail("pipeline.kb.bookmark_catchup", slug="bookmark_catchup",
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bookmark catch-up — X bookmarks into user-saved atoms")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--force", action="store_true", help="grant consent and run now")
    ap.add_argument("--limit", type=int, default=BACKLOG_LIMIT,
                    help="max bookmarks to ingest (0 = all)")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_bookmark_catchup(force=args.force, limit=args.limit)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
