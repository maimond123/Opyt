"""
pipeline/kb/probe_catchup.py — the automatic trigger for the Proposer's candidate content.

What it does: on session open, pull one shallow page of each due candidate's own posts — bounded by
a DAILY ceiling — so `oracle(action='candidates')` can answer "what does this person actually
write" from their words instead of returning an empty list.

Why this exists: `candidate_probe.probe_candidates` was complete but had no trigger — nothing
called it but a human running a CLI command, so `oracle(action='candidates')` had no content to
answer from. This module is that trigger. `onboard` cannot fill the gap either: it builds the
candidate list but doesn't probe content, so a new install ends setup with a full list and no
probed content until someone runs this by hand. Full rationale, including the "why not an oracle
action" and "why not wired into onboard" decisions and the consent-gate analysis, is in

This rail owns its spawner in its own try/except in `mcp_server/server.py`, per the
`atom-rail-not-welded-to-catchup` guard: copy the pattern, never centralize the spawners.

No consent gate: the X pull is $0 (cookie GraphQL) and the embed is bounded and cheap (~$0.003/day
at the ceiling below). A consent gate exists for the money-absent + runaway case; this is neither.
"""

from __future__ import annotations

import argparse
import json

from pipeline.kb.rail_runtime import (COALESCE_DEFAULT, load_rail_env,
                                      models_unroutable, spawn_rail)
from pipeline import llm_client
from pipeline.ingestion.utils import log

# A DAILY ceiling, not a per-run one: a per-run bound alone lets a user who opens many sessions in
# a day multiply through it (up to 24 runs inside the 1-hour coalesce window). This rail meters
# CANDIDATES, not dollars, because the scarce resource is X requests against one shared cookie
# session (same shape as `bookmark_catchup.BOOKMARK_CATCHUP_DAILY_USD`). The per-run bound IS the
# remaining daily allowance (`ceiling - probed_today()`), derived from `probe_store.record_pull`'s
# timestamps, so it needs no separate table.
#
# 120, raised from 60 on 2026-08-23. X is not the binding constraint: across every rail's
# log to date there is not one 429 — the measured 169 req/hr ceiling has never been
# touched, and `_PACE_SECONDS` alone keeps the instantaneous rate constant regardless of
# this number. What this number actually controls is how LONG the rail stays active per
# day (120 x 22s = ~44 min), and the real contention in that window is SQLite write-lock
# collision with the other session-open rails (22 'database is locked' errors here, 67
# across all rails). That is why this is a doubling and not a 5x: the collision window is
# the cost, and the lock issue is open. Raise it further once that is fixed.
PROBE_DAILY_CANDIDATES = 120

# The rail label this pass's spend is attributed under. Labelled but deliberately ungated in
# dollars: the ceiling above is denominated in candidates (the actually-scarce resource), so this
# rail is not in `rail_budgets._paid_rails()` and can never be dollar-paused. The label still
# matters — without it this rail's embed spend records as `unattributed`. The string is
# `candidate_probe`, not the module name `probe_catchup`, matching this rail's lock/log/stamp/
# spawner naming rather than its module.
RAIL = "candidate_probe"


@llm_client.rail(RAIL)
def run_candidate_probe(force: bool = False,
                        daily_ceiling: int = PROBE_DAILY_CANDIDATES) -> dict:
    """Probe as many due candidates as today's allowance still permits. Never raises.

    Labelled but deliberately ungated in dollars — see `RAIL` above.

    Reads the meter and refuses OUTSIDE the lock (matches `run_bookmark_catchup`'s ordering),
    since acquiring a lease only to report a refusal would make a free no-op look like contention.

    `force=True` hands out a FULL `daily_ceiling` allowance instead of the day's remainder, so a
    hand-run isn't refused by a ceiling the automatic rail already spent. It does not mean
    "unbounded" and does not bypass single-flight.

    `daily_ceiling <= 0` means no ceiling — probe the whole due queue (vocabulary inherited from
    `probe_candidates(max_candidates=0)`). That must be typed on purpose: it is a ~12.5 hour paced
    run, so the arithmetic below never lets a spent day fall through as a 0 budget by accident.
    """
    load_rail_env()

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock

    from . import candidate_probe, probe_store, schema
    conn = None
    try:
        ceiling = int(daily_ceiling)
        conn = schema.connect()
        # A never-probed store answers 0 here WITHOUT creating any probe tables — the meter must not
        # be the thing that brings the store into existence.
        spent = probe_store.probed_today(conn)
        if ceiling > 0:
            budget = ceiling if force else max(0, ceiling - spent)
            if budget == 0:
                return {"status": "daily_ceiling", "probed_today": spent,
                        "daily_ceiling": ceiling, "budget": 0,
                        "message": (f"today's {ceiling}-candidate probe ceiling is spent "
                                    f"({spent} probed). It resets at UTC midnight; the rest of the "
                                    f"queue drains on the following runs.")}
        else:
            budget = 0                       # explicit opt-out: walk the WHOLE due queue

        with CatchupLock("candidate-probe") as lock:
            if not lock.acquired:
                # Another session's pass holds the lease; skipping is correct, not a failure — two
                # paced walkers would otherwise double the request rate against one shared cookie
                # session.
                return {"status": "already_running",
                        "message": "another candidate probe is in flight — skipped "
                                   "(single-flight)."}
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            res = candidate_probe.probe_candidates(conn, embedder, max_candidates=budget)
            # `stopped` is NOT ok (dead X session, spent rate budget, or dead embedder can each
            # leave more in the queue than the ceiling implies) — the exit code follows it.
            status = "stopped" if res.get("stopped") else "ok"
            log(f"[candidate-probe] {status}: budget {budget or 'ALL'} "
                f"(ceiling {ceiling}, {spent} already probed today), "
                f"{res.get('requests', 0)} request(s), {res.get('atoms', 0)} atom(s), "
                f"{res.get('remaining', '?')} still due")
            return {"status": status, **res, "budget": budget, "probed_today": spent,
                    "daily_ceiling": ceiling}
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[candidate-probe] run_candidate_probe errored: {detail}")
        return {"status": "error", "error": detail}
    finally:
        if conn is not None:
            conn.close()


# ── The detached spawn ──────────────────────────────────────────────────────────
def spawn_candidate_probe(force: bool = False, coalesce_window: float = COALESCE_DEFAULT) -> bool:
    """Fire one probe pass as a detached, non-blocking child and return immediately.

    Copied from `spawn_curation_catchup`: closes the log fd in the PARENT after `Popen` (avoids an
    fd leak in the long-lived MCP server) and stamps the coalesce window only after `Popen`
    succeeds (a failed fork must not burn the retry window). `CatchupLock` makes the resulting
    double-spawn race harmless.

    Uses its own coalesce stamp and log file — sharing either with another rail would let one
    rail's spawn silently suppress the other's for an hour.
    """
    return spawn_rail("pipeline.kb.probe_catchup", slug="candidate_probe",
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Candidate probe — pull each due candidate's own posts into the CANDIDATE "
                    "store, bounded by a daily ceiling (the X pull is free; the embed is not)")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--force", action="store_true",
                    help="ignore what today already spent (a full ceiling's allowance). Still "
                         "single-flight, and still bounded.")
    ap.add_argument("--daily-ceiling", type=int, default=PROBE_DAILY_CANDIDATES,
                    help=f"candidates this rail may probe per UTC day, across ALL runs "
                         f"(default {PROBE_DAILY_CANDIDATES}). 0 or less = no ceiling, i.e. drain "
                         f"the whole due queue in one pass — hours of paced requests.")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_candidate_probe(force=args.force, daily_ceiling=args.daily_ceiling)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "already_running", "daily_ceiling"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
