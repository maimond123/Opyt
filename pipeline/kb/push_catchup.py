"""
pipeline/kb/push_catchup.py — keeps the SERVED copy of this knowledge base current, by itself.

R3. Freshness cannot be a user action: an owner who has to remember to run `opyt-push` serves a
stale copy, and the whole point of hosting is that a reader can read while the owner's laptop is
shut. So this is the fifth rail on `spawn_rail`'s pattern — detached, coalesced, fail-safe, fired
on session open, exactly like `bookmark_catchup` and its siblings.

Owns its own spawner in its own `try/except` in `mcp_server/server.py`, never welded to another
rail's (`atom-rail-not-welded-to-catchup`).

NOT CRON. `cron` does not fire while a Mac sleeps, and `launchd` needs a per-user plist — which
is install-time friction against the constraint this whole feature answers to.

THE LAZY GATE, and both terms are required: push when somebody has READ since the last push AND
the local store has CHANGED since it. Demand alone would ship an identical 117 MB every time
anyone reads. The gate is self-quieting — a read makes demand true, the push consumes it, and it
goes false again — which is what a token-existence gate ("does anyone hold a reader token") can
never be: grants only accumulate, so every one ever issued would become a permanent push
obligation.

⚠️THE ACCEPTED COST, stated rather than hidden: demand-triggering does not make the CURRENT read
fresh, it makes the FOLLOWING one fresh. The first reader after a change gets the previous
version.

⚠️RAILS ARE DETACHED AND CONCURRENT (`start_new_session=True`), so this cannot be sequenced after
an ingest rail — it evaluates the store while ingest is still writing. The consequence is
structural and self-correcting: a push lands one session behind the ingest that caused it. Do not
try to order them.

NO SPEND CEILING and no consent marker, unlike the paid rails. This rail calls no model and buys
nothing; its only cost is bandwidth on a connection the user already pays for, and the consent it
would ask for was already given, once and explicitly, when they shared the knowledge base at all.
The absent `OPYT_SERVICE_TOKEN` is the real gate: an install that never shared has nothing to
push to and this returns silently.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db, opyt_path
from pipeline.ingestion.utils import log
from pipeline.kb.rail_runtime import COALESCE_DEFAULT, load_rail_env, spawn_rail

RAIL = "push_catchup"


def _watermark_path() -> Path:
    """Resolved at call time so it honors `$OPYT_HOME` (Distributable: derive paths at runtime).
    A marker bound at import resolves to the real home under a sandboxed one."""
    return opyt_path("push_watermark")


def store_position() -> dict | None:
    """Where this store stands right now — `{atoms_at, chunk_id}` — or None if it cannot be read.

    TWO facts because one does not cover the other. `MAX(atoms.ingested_at)` moves when an atom
    arrives or is re-observed; `MAX(chunks.chunk_id)` is an AUTOINCREMENT and moves when anything
    is re-chunked or re-embedded without a new atom, which a re-embed run does to the whole store
    while leaving every `ingested_at` alone.

    A comparison and not a mirror: the watermark is a copy of a value this store owns, read only
    to answer "has it moved", and it is never authoritative for anything."""
    try:
        conn = sqlite3.connect(f"file:{opyt_db()}?mode=ro", uri=True)
    except Exception:
        return None
    try:
        atoms_at = conn.execute("SELECT MAX(ingested_at) FROM atoms").fetchone()[0]
        chunk_id = conn.execute("SELECT MAX(chunk_id) FROM chunks").fetchone()[0]
    except sqlite3.OperationalError:
        return None            # a store that has never ingested — nothing to publish either
    finally:
        conn.close()
    return {"atoms_at": atoms_at, "chunk_id": chunk_id}


def read_watermark() -> dict | None:
    """The store position at the last SUCCESSFUL push, or None if there has never been one.

    None means "changed", which is the safe direction: it makes the first push after this rail
    ships happen, rather than skipping forever on a store that has genuinely moved."""
    try:
        return json.loads(_watermark_path().read_text())
    except Exception:
        return None


def write_watermark(position: dict) -> None:
    """Record what was just published. Called ONLY after `publish` reports success — writing it
    before or regardless would make a failed push look like a completed one, and the store would
    then have to change AGAIN before anything retried. Fail-safe: an unwritable marker costs one
    redundant push next session, never a missed one."""
    try:
        p = _watermark_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(position))
    except OSError:
        pass


def run_push_catchup(force: bool = False) -> dict:
    """Read the gate, and push if both terms hold. Never raises.

    The order is the gate's order, and each miss is a distinct status so a log says which term
    was false:

      • no token            → `not_shared`. Silent: this install never shared anything.
      • never published     → PUBLISH. The retry net for a first push that died — without it a
                              failed first publish is permanent, because demand cannot become
                              true against an export that does not exist.
      • nobody read         → `no_demand`.
      • store has not moved → `no_change`.

    `force=True` skips both terms and publishes, for a person who just asked to.
    """
    load_rail_env()

    from opyt_core import config, push
    from pipeline.credentials import get_credential

    token = get_credential("opyt_service")
    if not token:
        return {"status": "not_shared",
                "message": "this install has not shared a knowledge base, so there is nothing "
                           "to keep current."}
    url = config.service_url().rstrip("/")

    position = store_position()
    state = push.fetch_state(token, url)
    if state["status"] != "ok":
        return state

    if not force:
        if state["last_upload_at"] is not None:
            if not state["reads_since_last_upload"]:
                return {"status": "no_demand",
                        "message": "nobody has read this knowledge base since the last push, so "
                                   "the served copy is as current as anyone needs."}
            if position is not None and position == read_watermark():
                return {"status": "no_change",
                        "message": "this store has not changed since the last push."}

    res = push.publish(token, url, owner=state["owner"])
    if res["status"] == "ok" and position is not None:
        write_watermark(position)
    if res["status"] != "ok":
        log(f"[{RAIL}] {res['status']}: {res.get('message')}")
    return res


def spawn_push_catchup(force: bool = False, coalesce_window: float = COALESCE_DEFAULT) -> bool:
    """Fire one pass as a detached, non-blocking child and return immediately."""
    return spawn_rail("pipeline.kb.push_catchup", slug=RAIL,
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Push catch-up — refresh the served copy of this knowledge base")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--force", action="store_true", help="publish now, skipping the gate")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_push_catchup(force=args.force)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "not_shared", "no_demand", "no_change"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
