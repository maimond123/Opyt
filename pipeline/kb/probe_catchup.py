"""
pipeline/kb/probe_catchup.py — the automatic trigger for the Proposer's candidate content.

What it does: pull one shallow page of each due candidate's own content — bounded by a DAILY
ceiling — so `oracle(action='candidates')` can answer "what does this person actually write" from
their words instead of returning an empty list. Two arms sharing one allowance: X posts
(`candidate_probe`) and, for a candidate the user knows only as the author of a paper they saved,
recent works (`scholar_probe`).

Why this exists: `candidate_probe.probe_candidates` was complete but had no trigger — nothing
called it but a human running a CLI command, so `oracle(action='candidates')` had no content to
answer from. This module is that trigger. `onboard` cannot fill the gap either: it builds the
candidate list but doesn't probe content, so a new install ends setup with a full list and no
probed content until someone runs this by hand. Full rationale, including the "why not an oracle
action" and "why not wired into onboard" decisions and the consent-gate analysis, is in

Launched by the resident worker, and activated by `curation_catchup`: the pass that mints a
candidate is the event that gives this rail work, so that rail queues this one (see its `main`).

No consent gate: the X pull is $0 (cookie GraphQL) and the embed is bounded and cheap (~$0.003/day
at the ceiling below). A consent gate exists for the money-absent + runaway case; this is neither.
"""

from __future__ import annotations

import argparse
import json

from pipeline.kb.rail_runtime import load_rail_env, models_unroutable
from pipeline.ingestion.utils import log

# A DAILY ceiling, not a per-run one: a per-run bound alone lets the hourly cadence multiply
# through it, up to 24 passes a day. This rail meters
# CANDIDATE SAMPLES, because the scarce resource is X requests against one shared cookie session.
# The per-run bound IS the remaining daily allowance (`ceiling - probed_today()`), derived from
# `probe_store.record_attempt`'s append-only ledger. Current pull state stays in `probe_pulls` for
# scheduling; it cannot also count retries.
#
# 120, raised from 60 on 2026-08-23. X is not the binding constraint: across every rail's
# log to date there is not one 429. `_PACE_SECONDS` keeps the instantaneous rate constant;
# this ceiling limits candidate samples, each of which may page further to characterize thin
# timelines. The shared X meter remains the hard request stop.
PROBE_DAILY_CANDIDATES = 120

RAIL = "candidate_probe"


def run_candidate_probe(daily_ceiling: int = PROBE_DAILY_CANDIDATES) -> dict:
    """Probe as many due candidates as today's sampling allowance permits. Never raises.

    The day's remainder is the ONLY allowance. A `force` that handed out a full ceiling on top of
    what the rail had already spent used to exist and had no production caller: widening a spend
    bound is what `daily_ceiling=` is for, and saying it twice let one word mean two things.

    `daily_ceiling <= 0` means no ceiling — probe the whole due queue (vocabulary inherited from
    `probe_candidates(max_candidates=0)`). That must be typed on purpose: it is a ~12.5 hour paced
    run, so the arithmetic below never lets a full day fall through as a 0 budget by accident.
    """
    load_rail_env()

    if (reason := models_unroutable(RAIL)) is not None:
        return {"status": "models_unroutable", "message": reason}

    from pipeline.sync_lock import CatchupLock

    from . import candidate_probe, probe_store, schema, scholar_probe
    conn = None
    try:
        ceiling = int(daily_ceiling)
        conn = schema.connect()
        # A never-probed store answers 0 here WITHOUT creating any probe tables — the meter must not
        # be the thing that brings the store into existence.
        probed = probe_store.probed_today(conn)
        if ceiling > 0:
            budget = max(0, ceiling - probed)
            if budget == 0:
                return {"status": "daily_ceiling", "probed_today": probed,
                        "daily_ceiling": ceiling, "budget": 0,
                        "message": (f"today's {ceiling}-candidate probe ceiling is full "
                                    f"({probed} probed). It resets at UTC midnight; the rest of the "
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
            if lock.lost():
                return {"status": "lease_lost",
                        "message": "candidate-probe lease was reclaimed before work began."}
            from .embed import get_kb_embedder
            embedder = get_kb_embedder()
            res = candidate_probe.probe_candidates(conn, embedder, max_candidates=budget,
                                                   should_stop=lock.lost)
            # `stopped` is NOT ok (dead X session, exhausted rate budget, or dead embedder can each
            # leave more in the queue than the ceiling implies) — the exit code follows it.
            status = "lease_lost" if res.get("stopped") == "lease_lost" else (
                "stopped" if res.get("stopped") else "ok")

            # The scholar arm spends what the X arm left of the SAME daily allowance, because the
            # two meter the same thing: candidate samples. Different host, though, so an X session
            # that died says nothing about OpenAlex and does not skip this — only a lost lease does.
            #
            # ⚠️ `max_candidates=0` means UNBOUNDED, so an exhausted budget must SKIP the arm
            # rather than pass the remainder through. Passing 0 here on a spent budget would turn
            # the ceiling into its opposite and drain the whole due queue.
            scholar: dict = {"skipped": "budget_spent"}
            spent = res.get("requests", 0)
            if status == "lease_lost":
                scholar = {"skipped": "lease_lost"}
            elif ceiling <= 0:
                scholar = scholar_probe.probe_scholars(conn, embedder, max_candidates=0)
            elif budget - spent > 0:
                scholar = scholar_probe.probe_scholars(conn, embedder,
                                                       max_candidates=budget - spent)
            res["scholar"] = scholar

            log(f"[candidate-probe] {status}: budget {budget or 'ALL'} "
                f"(ceiling {ceiling}, {probed} already probed today), "
                f"{spent} request(s), {res.get('atoms', 0)} atom(s), "
                f"{res.get('remaining', '?')} still due; "
                f"scholar {scholar.get('requests', 0)} request(s), "
                f"{scholar.get('atoms', 0)} atom(s)")
            return {"status": status, **res, "budget": budget, "probed_today": probed,
                    "daily_ceiling": ceiling}
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[candidate-probe] run_candidate_probe errored: {detail}")
        return {"status": "error", "error": detail}
    finally:
        if conn is not None:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Candidate probe — pull each due candidate's own posts into the CANDIDATE "
                    "store, bounded by a daily request ceiling")
    ap.add_argument("--once", action="store_true", help="run once against $OPYT_HOME")
    ap.add_argument("--daily-ceiling", type=int, default=PROBE_DAILY_CANDIDATES,
                    help=f"candidates this rail may probe per UTC day, across ALL runs "
                         f"(default {PROBE_DAILY_CANDIDATES}). 0 or less = no ceiling, i.e. drain "
                         f"the whole due queue in one pass — hours of paced requests.")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_candidate_probe(daily_ceiling=args.daily_ceiling)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "already_running", "daily_ceiling"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
