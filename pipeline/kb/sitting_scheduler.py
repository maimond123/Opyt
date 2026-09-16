"""
pipeline/kb/sitting_scheduler.py — WHICH region gets read next, and what makes the read happen.

The rail's disposer: nothing else decides, unattended, that a built region is owed a read.

Four channels propose, one scheduler disposes, at most one paid REGION read per run:

    1  pointed     an unread region whose read was ATTEMPTED and failed  -> retry it
    2  new_mass    a read region that has gained material                -> regrow, read that
    3  sub_region  an unread region a fracture produced out of a read parent -> read it
    3  remainder   a read region with material left over                 -> fracture, or continue (D15)

Consumption subscribes, construction does not — a bare `preview` or hand-run `zoom` builds a row
but queues nothing. `pointed` is a retry lane only (needs a prior failed `frontier_reader_runs`
row); `sub_region` claims only a fracture child whose parent has a `read_at`.

One scheduler ranks all four channels rather than four independent rails, because they compete
for the same paid reader and a priority order ("read what was asked for before fracture leftovers",
D16) cannot be expressed by four rails racing for one budget. Priority ranks by what rots
(`pointed`, `new_mass`) ahead of what doesn't (`sub_region`, `remainder`, oldest-first between
the two).

At most one paid REGION read per run is the bound — not one unit of bookkeeping, and (CORRECTED
2026-09-05) not one model CALL: the `claims` lens rides along, so a run costs 2 + K, where K is
notebook debt that drains (`sitting_claims._called_and_failed`).

Selection (this module) uses `first_seen`; membership uses no clock; render order uses `when_ts`
(Amendment 3) — the date filter computed here crosses into the builder as a plain id set only.

Never raises. Every outcome is a dict and a row in `frontier_reader_runs`.

This rail is ONE registry entry in `rail_worker.RAILS` covering four channels, and that is the
narrow exception worth stating. The eight rails are otherwise independent: they fail for different
reasons, spend on different things, and the worker keeps one home's failure off every other rail.
These four channels are not independent. They are four reasons to make the SAME paid call against
the same reader, and they compete for it. Four registry entries racing over one budget could not
express "read the region David asked for BEFORE the fracture leftovers", which is the entire
content of D16. One disposer that ranks them can.

It is also a PRODUCER: a read is the only thing that writes standing queries, so a read that adds
one makes `frontier_execute` due (see `main`).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pipeline.timeparse import utc_iso, utc_now

import numpy as np

from pipeline.kb.rail_runtime import load_rail_env, models_unroutable
from pipeline.ingestion.utils import log

from . import frontier_queries as fq
from . import schema
from . import sitting_claims as scl
from . import sitting_builder as sb
from . import sitting_store as sst
from . import sitting_vectors as sv
from . import sitting_zoom as sz

# Its own run label, distinct from the per-region `sitting:<slug>` the reader writes: a paid read
# leaves two rows (reader's = cost/output, this one = which claim/channel). Scoping matters because
# a rail asks "when did I last run" by filtering `frontier_reader_runs` on this label — `health()`
# below does exactly that inline.
GENERATOR = "sitting-scheduler"

# D9: `max` not `min` — reading cost tracks region size, so a bigger region needs proportionally
# more new material to justify a re-read.
NEW_MASS_FLOOR = 3
NEW_MASS_FRACTION = 0.20

# D10: three consecutive failures stop the loop; a tripped breaker re-trials once per cooldown.
# `health()` below exists because a breaker alone is blind to a job that never fired at all.
BREAKER_SERVICE = "sitting-scheduler"
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 6 * 3600      # long enough that a missing key is not retried every session

# Channel -> tier. Two channels share tier 3 and are ordered against each other by age.
_TIER = {"pointed": 1, "new_mass": 2, "sub_region": 3, "remainder": 3}


# ── Time ────────────────────────────────────────────────────────────────────────
def _parse_stamp(stamp: str | None) -> datetime | None:
    """Parse either timestamp shape this store writes.

    Never compared as raw strings: `atoms.first_seen` writes a space separator, `sittings.read_at`
    writes a `T`, and a string `>` between them would silently order every atom as older than every
    read (ord(' ') < ord('T')).
    """
    if not stamp:
        return None
    try:
        d = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


# ── The channels ────────────────────────────────────────────────────────────────
def _unread_claims(conn) -> list[dict]:
    """Channels 1 and 3 — every built-but-unread region that was actually consumed, split by
    whether it traces back to a fracture (`sub_region`) or a direct point (`pointed`).

    `pointed` is a retry lane only: it claims a region only where a `queries` read (`lens IS NULL
    OR lens = 'queries'`, excluding free host-side lens receipts) was already attempted and
    failed. `sub_region` claims a fracture child only once its parent has a `read_at`.

    Known imprecision: a region this scheduler itself regrew or continued can land here as
    `pointed` if its read failed, since the store doesn't record originating channel — harmless,
    since the promotion is still correct, just imprecisely labeled.
    """
    out = []
    for r in conn.execute(
            "SELECT sitting_id, region_key, seed_ref, built_at, atoms, tokens, parent_sitting_id "
            "  FROM sittings s WHERE read_at IS NULL AND atoms > 0 "
            "   AND ("
            "     (parent_sitting_id IS NULL AND EXISTS ("
            "       SELECT 1 FROM frontier_reader_runs r WHERE r.sitting_id = s.sitting_id "
            "         AND (r.lens IS NULL OR r.lens = 'queries'))"
            "       AND NOT EXISTS ("
            "         SELECT 1 FROM frontier_reader_runs r2 WHERE r2.sitting_id = s.sitting_id "
            "           AND r2.lens = 'queries' AND r2.status = 'ok'))"
            "     OR "
            "     (parent_sitting_id IS NOT NULL AND EXISTS ("
            "       SELECT 1 FROM sittings p WHERE p.sitting_id = s.parent_sitting_id "
            "         AND p.read_at IS NOT NULL))"
            "   )"
            " ORDER BY built_at ASC, atoms DESC, sitting_id ASC"):
        sub = r["parent_sitting_id"] is not None
        out.append({
            "channel": "sub_region" if sub else "pointed",
            "sitting_id": r["sitting_id"], "region_key": r["region_key"],
            "seed_ref": r["seed_ref"], "since": r["built_at"],
            "detail": f"{r['atoms']} atoms, ~{r['tokens']} tokens, built {r['built_at']}",
        })
    return out


def _remainder_claims(conn) -> list[dict]:
    """Channel 4 — a region that was read and still has material the budget cut off.

    `stop = 'budget'` is exact: the admission loop only stops there when the pool still held
    admissible atoms. The two NOT EXISTS clauses retire a claim once it's continued or fractured,
    so a chain's head is claimed exactly once and the claim moves to the new tail.
    """
    return [{
        "channel": "remainder", "sitting_id": r["sitting_id"], "region_key": r["region_key"],
        "seed_ref": r["seed_ref"], "since": r["read_at"],
        "detail": f"read {r['read_at']}, stopped on budget with {r['region_atoms']} atoms left",
    } for r in conn.execute(
        "SELECT sitting_id, region_key, seed_ref, read_at, region_atoms FROM sittings s "
        " WHERE s.read_at IS NOT NULL AND s.stop = 'budget' "
        "   AND NOT EXISTS (SELECT 1 FROM sittings c WHERE c.continues = s.sitting_id) "
        "   AND NOT EXISTS (SELECT 1 FROM sittings z WHERE z.parent_sitting_id = s.sitting_id) "
        " ORDER BY s.read_at ASC, s.sitting_id ASC")]


def _arrivals(conn) -> dict:
    """`{atom_id: arrival datetime}` for every human-attested atom: `promoted_at` if it was
    promoted, else `first_seen`.

    Never `ingested_at`: that column refreshes on every re-observation, so a re-scrape would make
    the whole corpus look newly arrived and trigger every region's re-read at once.

    The COALESCE is what makes promotion open the wallet at all. A promoted atom's `first_seen` is
    the date the crawler found it — typically months before the human deposit that promoted it —
    so keyed on `first_seen` alone a brand-new engagement arrives already stale and never counts as
    new mass. `promoted_at` is NULL on everything that was never promoted, which is almost every
    row, so this changes nothing else.
    """
    out = {}
    for r in conn.execute(f"SELECT atom_id, COALESCE(promoted_at, first_seen) AS arrived FROM atoms a "
                          f"WHERE {sv._human_clause()}", sv.HUMAN_ATTESTED):
        d = _parse_stamp(r["arrived"])
        if d is not None:
            out[r["atom_id"]] = d
    return out


def _new_mass_claims(conn) -> list[dict]:
    """Channel 2 — a region whose last read is now missing enough of the corpus to be worth redoing.

    New mass is re-run membership, not a corpus-wide count of recent saves: an atom counts only if
    it clears cosine against THIS region's stored anchor at THIS region's floor. One streaming pass
    scores every region's anchors together, so N regions cost one scan of the chunk table.

    The scan below stays NARROW even though `build_sitting`'s membership widened with the union
    (RULED 2026-08-24: new mass is human-lane only). Counting frontier arrivals closes the loop —
    a read mints queries, the queries pull atoms, those atoms trigger the next paid read — and the
    spend cadence becomes machine-determined instead of tracking David's engagement rate. Enforced
    twice: `restrict=fresh` comes from `_arrivals`, which is human-only, and the default
    `entry_modes` here is the narrow tuple. Pinned by
    tests/kb/test_sitting_scheduler.py::test_frontier_arrivals_never_open_the_wallet.
    """
    pending = {r["region_key"] for r in conn.execute(
        "SELECT DISTINCT region_key FROM sittings "
        " WHERE read_at IS NULL AND atoms > 0 AND region_key IS NOT NULL")}
    regions = []
    for r in conn.execute(
            "SELECT region_key, MAX(read_at) AS last_read FROM sittings "
            " WHERE read_at IS NOT NULL AND region_key IS NOT NULL GROUP BY region_key"):
        # A region with a build already waiting is already claimed by channel 1 or 3.
        if r["region_key"] in pending:
            continue
        last = _parse_stamp(r["last_read"])
        if last is None:
            continue
        row = conn.execute(
            "SELECT * FROM sittings WHERE region_key = ? "
            " ORDER BY built_at DESC, sitting_id DESC LIMIT 1", (r["region_key"],)).fetchone()
        if row is None:
            continue
        try:
            anchor = sst.ensure_seed_vector(conn, row["sitting_id"])
            # `ChainError` is a KeyError, so the clause below already covers a corrupted chain.
            chain = set(sb.chain_atom_ids(conn, row["sitting_id"]))
        except (KeyError, sb.SeedError) as e:
            # Fail-safe: a region whose anchor can't be rebuilt is skipped, never crashed on.
            log(f"[sitting-scheduler] no anchor for region {r['region_key']}: {e}")
            continue
        # region_atoms is only what was still admissible at build time; add back seeds + earlier chain parts.
        size = ((row["region_atoms"] or 0) + (row["prior_atoms"] or 0)
                + len(json.loads(row["seed_atom_ids"] or "[]")))
        regions.append({"region_key": r["region_key"], "seed_ref": row["seed_ref"],
                        "sitting_id": row["sitting_id"], "floor": row["floor"],
                        "last_read": last, "last_read_raw": r["last_read"],
                        "size": size, "anchor": anchor, "chain": chain})
    if not regions:
        return []

    arrivals = _arrivals(conn)
    oldest = min(x["last_read"] for x in regions)
    fresh = {a for a, d in arrivals.items() if d > oldest}
    if not fresh:
        return []
    scores = sv._relevance(conn, np.vstack([x["anchor"] for x in regions]), restrict=fresh)

    out = []
    for i, x in enumerate(regions):
        # `not in x["chain"]` keeps this scan agreeing with the build it triggers: `build_sitting`
        # subtracts the chain, so an already-read atom counted here buys a regrow of nothing.
        # Reached via a promoted frontier atom, whose fresh `promoted_at` reads as an arrival.
        mass = sum(1 for a, v in scores.items()
                   if a not in x["chain"] and arrivals[a] > x["last_read"]
                   and float(v[i]) >= x["floor"])
        want = max(float(NEW_MASS_FLOOR), NEW_MASS_FRACTION * x["size"])
        if mass < want:
            continue
        out.append({
            "channel": "new_mass", "sitting_id": x["sitting_id"], "region_key": x["region_key"],
            "seed_ref": x["seed_ref"], "since": x["last_read_raw"],
            "new_mass": mass, "threshold": round(want, 1), "region_size": x["size"],
            "ratio": round(mass / want, 2) if want else 0.0,
            "detail": f"{mass} new atoms clear the floor since {x['last_read_raw']} "
                      f"(needs {want:.1f} of a {x['size']}-atom region)",
        })
    # Fullest first, tie-break by age — the region with the most unread material differs most from its last read.
    out.sort(key=lambda c: (-c["ratio"], c["since"]))
    return out


def claims(conn, *, cheap_only: bool = False) -> list[dict]:
    """Every channel's proposals, best first. Reads nothing paid and writes nothing.

    `cheap_only` drops the new-mass channel (the only one that scans the chunk table): a tool call
    can't create new-mass work, so skipping it there is safe — the rail's own cadence covers it.
    """
    out = _unread_claims(conn) + _remainder_claims(conn)
    if not cheap_only:
        out += _new_mass_claims(conn)
    # Stable sort: tier-3 ordering by `since` interleaves sub-regions/remainders without disturbing
    # either channel's own order.
    out.sort(key=lambda c: (_TIER[c["channel"]], c["since"] if _TIER[c["channel"]] == 3 else ""))
    for i, c in enumerate(out):
        c["tier"] = _TIER[c["channel"]]
        c["rank"] = i
    return out


# ── Disposal ────────────────────────────────────────────────────────────────────
def _stored_seed(conn, row: dict) -> dict:
    """The region's original anchor, in `resolve_seed`'s shape. Never re-resolves the phrase.

    Re-embedding the phrase today would select a different top-k as the corpus moved, silently
    growing a different region under the same name. A stored anchor is a blob read, not a metered
    embed call.
    """
    return {"kind": row["seed_kind"], "ref": row["seed_ref"],
            "atom_ids": row["seed_atom_ids"],
            "vector": sst.ensure_seed_vector(conn, row["sitting_id"])}


def _dials(row: dict) -> dict:
    """The dials that define the region. Copied whole, because dropping one changes `region_key`."""
    return {"floor": row["floor"], "ceiling": row["ceiling"],
            "budget_tokens": row["budget_tokens"]}


def _regrow(conn, sitting_id: str, *, ref: datetime) -> dict:
    """Build the region's NEXT PART over the corpus as it is now — chained to the tail it grew from.

    `continues=sitting_id` is load-bearing and it is a REVERSAL (2026-08-24). It used to be
    deliberately absent, on the reasoning that a re-read must see the whole region again to show
    what changed. Under chained parts that reasoning is served better and cheaper: what earlier
    parts established travels forward as the claims notebook, which every later read is instructed
    to confirm, revise or refute. Re-reading their full text buys the same memory at full price.

    And detaching costs more than money. `continues` IS the notebook chain — a new-mass build with
    a NULL link starts a fresh lineage, so the preamble walk finds no ancestors and the region
    silently loses its entire history at exactly the moment new material arrived to compare it to.
    """
    row = sst.get_sitting(conn, sitting_id)
    if row is None:
        raise KeyError(f"no sitting {sitting_id!r} to regrow")
    return sb.build_sitting(conn, _stored_seed(conn, row), now=ref, continues=sitting_id,
                            **_dials(row))


def _fracture_or_continue(conn, sitting_id: str, *, ref: datetime) -> dict:
    """D15 — over budget GATES, separability DECIDES. Returns `{action, sitting_id, report}`.

        fracture yields > 1 surviving sub-region  -> FRACTURE
        otherwise                                 -> CONTINUE

    Runs `zoom(persist=False)` first to decide, then `zoom(persist=True)` only if FRACTURE, so no
    sub-sitting is ever written and then disowned. The replay is exact (fixed k-means seed, same
    `now=ref`), so the two passes land the same centroids under the same ids.
    """
    trial = sz.zoom(conn, sitting_id, persist=False, now=ref)
    survivors = [s for s in trial["sub"] if s["kept"] and s["tier"] == "standalone"]
    if len(survivors) > 1:
        rep = sz.zoom(conn, sitting_id, persist=True, now=ref)
        # `written` is `survivors` re-derived after the exact replay, so it cannot be empty here:
        # `zoom` persists on `keep and tier == "standalone"`, the predicate `survivors` just counted.
        written = [s for s in rep["sub"] if s["persisted"]]
        # Read the biggest sub-region now rather than waiting a coalesce window; siblings
        # become `sub_region` claims.
        pick = max(written, key=lambda s: s["atoms"])
        return {"action": "fracture", "sitting_id": pick["sitting_id"], "report": rep}

    row = sst.get_sitting(conn, sitting_id)
    part = sb.build_sitting(conn, _stored_seed(conn, row), continues=sitting_id, now=ref,
                            **_dials(row))
    return {"action": "continue", "sitting_id": part["sitting_id"], "report": trial}


def _act(conn, claim: dict, *, ref: datetime) -> tuple[str, dict]:
    """Turn a claim into the sitting_id that should be read, plus whatever it did on the way."""
    ch = claim["channel"]
    if ch in ("pointed", "sub_region"):
        return claim["sitting_id"], {}
    if ch == "new_mass":
        rec = _regrow(conn, claim["sitting_id"], ref=ref)
        return rec["sitting_id"], {"regrew": rec["sitting_id"], "atoms": rec["atoms"],
                                   "stop": rec["stop"]}
    if ch == "remainder":
        res = _fracture_or_continue(conn, claim["sitting_id"], ref=ref)
        rep = res["report"]
        return res["sitting_id"], {"action": res["action"], "k": rep.get("k"),
                                   "persisted": rep.get("persisted", 0),
                                   "kept": rep.get("kept")}
    raise ValueError(f"unknown channel {ch!r}")


# ── The run ─────────────────────────────────────────────────────────────────────
def _breaker():
    from pipeline.circuit_breaker import CircuitBreaker
    return CircuitBreaker(BREAKER_SERVICE, threshold=BREAKER_THRESHOLD,
                          cooldown=BREAKER_COOLDOWN_S)


def run_sitting_scheduler(conn=None, *, plan_only: bool = False,
                          now: datetime | None = None) -> dict:
    """One scheduler pass: rank the claims, take the best one, read it. Never raises.

    There is no breaker override. The breaker exists to stop a failing loop from burning money,
    and the only thing an override ever answered to was a hand-typed command line — which is not
    a caller this product has. `plan_only` ranks and returns claims without spending —
    deliberately not named `dry_run` like the reader's, since that one still makes the paid call.

    ⚠️ IT IS ALSO THE PRODUCER OF `frontier_execute`, AND THAT CHAINING LIVES HERE rather than in
    `main()`. There is no in-process caller of this one today — which is exactly why it moved:
    the other two rails in this chain each grew one and silently lost their successor, and a
    side-effect that only the CLI wrapper performs is one refactor away from the same failure.
    `tests/kb/test_rail_activation.py` now fails any rail that puts a `request_now` back inside a
    `main()`.

    `read.new` counts the queries this pass genuinely added — a refreshed query is already
    standing and already scheduled, so re-queueing on it would make every read restart stage 2.
    `plan_only` spends nothing and writes nothing, so it queues nothing.
    """
    load_rail_env()
    ref = now or utc_now()
    own = conn is None
    if own:
        conn = schema.connect()
    try:
        res = _run(conn, plan_only=plan_only, ref=ref)
        if not plan_only and (res.get("read") or {}).get("new", 0) > 0:
            from pipeline.kb.rail_jobs import request_now
            request_now("frontier_execute")
        return res
    except Exception as e:                                    # last resort: never propagate
        detail = f"{type(e).__name__}: {e}"
        log(f"[sitting-scheduler] run errored: {detail}")
        try:
            _breaker().record_failure(detail)
            fq.record_run(conn, generator=GENERATOR, status="failed", reason=detail,
                          ran_at=utc_iso(ref))
        except Exception:
            pass
        return {"status": "failed", "reason": detail}
    finally:
        if own:
            conn.close()


def _run(conn, *, plan_only: bool, ref: datetime) -> dict:
    # `plan_only` spends nothing, so the routability gate would only block a free report.
    if not plan_only and (reason := models_unroutable(GENERATOR)) is not None:
        # A ROW, unlike `plan_only` below: `health()` derives `ever_ran` from COUNT(*) here, so a
        # silent return made a routing outage report "the scheduler has never run".
        fq.record_run(conn, generator=GENERATOR, status="skipped", reason=reason,
                      ran_at=utc_iso(ref))
        return {"status": "models_unroutable", "reason": reason}

    queue = claims(conn)
    if plan_only:
        # No row, no breaker touch: recording a plan as a run would make health() report a live
        # loop when nothing has ever disposed.
        return {"status": "plan", "claims": queue, "health": health(conn)}

    if not queue:
        fq.record_run(conn, generator=GENERATOR, status="skipped",
                      reason="nothing claimable", ran_at=utc_iso(ref))
        return {"status": "skipped", "reason": "nothing claimable", "claims": []}

    breaker = _breaker()
    if not breaker.allow():
        reason = (f"breaker OPEN after {BREAKER_THRESHOLD} consecutive failures — "
                  f"retry in ~{breaker.retry_after():.0f}s")
        fq.record_run(conn, generator=GENERATOR, status="skipped", reason=reason, ran_at=utc_iso(ref))
        return {"status": "skipped", "reason": reason, "claims": queue}

    claim = queue[0]
    try:
        target, did = _act(conn, claim, ref=ref)
    except Exception as e:
        # A build or fracture failure costs nothing but still counts toward the breaker.
        detail = f"{claim['channel']}: {type(e).__name__}: {e}"
        log(f"[sitting-scheduler] {detail}")
        breaker.record_failure(detail)
        fq.record_run(conn, generator=GENERATOR, sitting_id=claim.get("sitting_id"),
                      status="failed", reason=detail, ran_at=utc_iso(ref))
        return {"status": "failed", "reason": detail, "claim": claim}

    # `read_part` owns the ORDER (notebook debt → queries read → claims receipt, RULED 2026-08-24)
    # and both doors into a read call it, so the two can never drift apart.
    res = scl.read_part(conn, target, now=ref)
    status = res.get("status")
    if status == "ok":
        # After the ritual rather than between its halves: `record_success` only resets a counter
        # on a different database, and every step of the ritual is fail-safe.
        breaker.record_success()
    elif status == "failed":
        breaker.record_failure(res.get("reason"))
    # A `skipped` read is neither success nor failure — the claim was stale, not the loop broken.

    fq.record_run(conn, generator=GENERATOR, sitting_id=target, status=status or "failed",
                  reason=f"{claim['channel']}: {claim.get('detail') or claim['seed_ref']}",
                  ran_at=utc_iso(ref))
    return {"status": status, "claim": claim, "read": res, "queued": len(queue), **did}


# ── Health ──────────────────────────────────────────────────────────────────────
def health(conn) -> dict:
    """`{ever_ran, last_run_at, runs, breaker_open, claims_waiting, needs_attention, note}`.

    `ever_ran` exists because a breaker counting failures alone can't distinguish a healthy loop
    from one that never fired. Guarded on there being work: a store with no claims stays silent.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n, MAX(ran_at) AS last FROM frontier_reader_runs WHERE generator = ?",
        (GENERATOR,)).fetchone()
    runs, last = int(row["n"] or 0), row["last"]
    waiting = len(claims(conn, cheap_only=True))
    try:
        # retry_after(), not allow(): allow() is a decision that consumes the single retry a
        # tripped breaker owes, which a status read must not spend.
        retry = _breaker().retry_after()
        open_ = retry > 0
    except Exception:
        open_, retry = False, 0.0
    out = {"ever_ran": runs > 0, "runs": runs, "last_run_at": last, "claims_waiting": waiting,
           "breaker_open": open_, "retry_after_s": round(retry, 1) if open_ else None,
           "note": None}
    out["needs_attention"] = bool(waiting and (runs == 0 or open_))
    if waiting and runs == 0:
        out["note"] = (f"{waiting} region(s) are waiting to be read and the sitting scheduler has "
                       f"never run. The resident worker launches it; if that is not happening, "
                       f"check that `opyt-worker` is running (`opyt-install-worker` installs it), "
                       f"run `python -m pipeline.kb.sitting_scheduler --once` to drain one, or "
                       f"read a specific region with sitting(action='read', sitting_id=...).")
    elif waiting and open_:
        out["note"] = (f"The sitting scheduler stopped after {BREAKER_THRESHOLD} consecutive "
                       f"failures and retries in ~{retry / 3600:.1f}h, with {waiting} region(s) "
                       f"waiting. The last reason is in frontier_reader_runs.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Decide which sitting gets read next, and read it")
    ap.add_argument("--once", action="store_true", help="run one pass against $OPYT_HOME")
    ap.add_argument("--plan", action="store_true", dest="plan_only",
                    help="rank the claims and print them; take no action, spend nothing. (Named "
                         "apart from the reader's --dry-run, which DOES make the paid call.)")
    args = ap.parse_args(argv)
    if not (args.once or args.plan_only):
        ap.print_help()
        return 2
    # NOTHING IS QUEUED HERE, deliberately. A read is the only producer of standing queries and
    # so the only event that can make Frontier stage 2 due — but that belongs to the RUN, not to
    # the CLI wrapper around it, or the next caller to reach `run_sitting_scheduler` in-process
    # loses it silently. See `run_sitting_scheduler`.
    res = run_sitting_scheduler(plan_only=args.plan_only)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "plan", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
