"""
pipeline/kb/frontier_execute.py — Frontier stage 2: run the standing queries, park what they find.

Stage 1 turned the KB into standing queries. This runs them against artifact sources on a cadence,
remembers how far each (query, source) pair got, and writes candidates to a STAGING table.

It stops before judgement. Nothing here writes to `atoms`. Stage 3 (ADMIT, fail-closed) owns that
transition, and this module's separation from `atoms` is the entire license for stage 1 generating
queries agentically in the first place: a bad query costs inbox noise, never KB pollution. A test
asserts the atom count is unchanged by a run.

This loop keys its watermark on `query_id`, which requires stable query strings across runs — see
doc for why that was once unbuildable and is now pinned by a test.

Shape borrowed from `oracle_refresh`: flat jittered TTL, `since` from the watermark minus an
overlap sliver, a window assertion before spending, bounded ATTEMPTS rather than iterations, and
— load-bearing — a failed pull never stamps.

Never raises. Every outcome is a dict and a row in `frontier_exec_runs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess  # unused directly — tests patch fe.subprocess.Popen to intercept spawn_rail()
from datetime import datetime, timedelta, timezone
from pipeline.timeparse import utc_iso, utc_now

from pipeline.kb.rail_runtime import (COALESCE_DEFAULT, load_rail_env,
                                      models_unroutable, rail_budget_exhausted, spawn_rail)
from pipeline import llm_client
from pipeline.ingestion.utils import log

from . import frontier_queries as fq
from . import schema
from .frontier_sources import SourceError, adapters

# ── Cadence ─────────────────────────────────────────────────────────────────────
# Flat per-source TTLs, deliberately coarse (adaptive TTL is a stage-5 concern with no yield data
# yet). arXiv posts daily; GitHub search is noisier so it gets a slower beat. OpenAlex gets the
# same slower beat for a different reason: its page is a RANKED SLICE of a 30-day window, not a
# stream of new items (see `OpenAlexAdapter`), so asking twice a day re-reads the same slice — and
# its ~100-request daily allowance is the one budget here that is not ours to spend.
TTL_HOURS = {"arxiv": 24.0, "github": 48.0, "openalex": 48.0}
DEFAULT_TTL_HOURS = 24.0
JITTER = 0.10               # ±10%, hash-derived and stable per pair
OVERLAP_HOURS = 6.0         # re-ask a sliver behind the cursor; dedup absorbs it
MAX_WINDOW_DAYS = 60        # the window assertion's ceiling
FIRST_PULL_DAYS = 14        # a never-pulled pair looks back this far, not to the beginning of time

MAX_REQUESTS_PER_RUN = int(os.environ.get("OPYT_FRONTIER_MAX_REQUESTS", 40))

# ── the resetting daily seatbelt ────────────────────────────────────────────────
# Daily USD ceiling for this rail (start gate, checked once before the pass, not a governor mid-run).
# Separate from MAX_REQUESTS_PER_RUN, which bounds one pass, not the day across repeated spawns.
# $1.00 matches the other rails' ceilings; see docs/plans/2026-08-16-per-rail-spend-meters.md.
FRONTIER_EXECUTE_DAILY_USD = 1.00

# Rail label shared by the `@llm_client.rail` decorator and `_daily_budget_exhausted` — one
# constant so the two never drift apart silently.
RAIL = "frontier_execute"


def _daily_budget_exhausted() -> bool:
    """Has this rail's recorded spend today reached its ceiling? See
    `rail_runtime.rail_budget_exhausted` for why it is never the global total."""
    return rail_budget_exhausted(RAIL, FRONTIER_EXECUTE_DAILY_USD)
PER_QUERY_LIMIT = 25

# Keep only candidates scoring at least this fraction of their page's own top `relevance_score`
# (RULED 2026-08-27; measured over 22 live queries / 356 candidates). Derivation and the parked
# caveats are on `_relevance_cut` and in the doc it names.
INTAKE_KEEP_FRAC = 0.5

# ── Decay tiers ─────────────────────────────────────────────────────────────────
# A query the reader drops slows down but is never removed. `miss_count` is consecutive drop
# verdicts. Stepped, not a curve, for auditability (see doc). The monthly floor is hard — TTL never
# grows past 30 days, so a demoted query can always resurface.
DECAY_TIERS = ((3, 1.0), (10, 7.0), (None, 30.0))   # (miss_count below this, TTL multiplier)


def tier_for(miss_count: int | None) -> float:
    """The TTL multiplier for a query with this many consecutive drop verdicts.

    0-2 → 1.0 (daily on arXiv) · 3-9 → 7.0 (weekly) · 10+ → 30.0 (monthly, and no slower).
    """
    n = int(miss_count or 0)
    for below, mult in DECAY_TIERS:
        if below is None or n < below:
            return mult
    return DECAY_TIERS[-1][1]                        # unreachable; the last row has no bound


# ── Watermark store ─────────────────────────────────────────────────────────────
def get_pair(conn, query_id: str, source: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM frontier_query_sources WHERE query_id=? AND source=?",
        (query_id, source)).fetchone()


def record_pull(conn, query_id: str, source: str, *, last_status: str,
                cursor_ts: str | None = None, stamp: bool = True, now: datetime | None = None):
    """Write the outcome of one pair's pull.

    `stamp=False` is the whole point of this function existing. A failed pull that advances
    `last_pulled_at` buys a full TTL of silence on that pair — one bad night and the query goes
    quiet for a day with nothing to show why.

    `cursor_ts` only ever moves FORWARD. An adapter returning an odd older date must not rewind a
    watermark and cause a re-pull of everything since.
    """
    stamp_at = utc_iso(now or utc_now()) if stamp else None
    row = get_pair(conn, query_id, source)
    if row is None:
        conn.execute(
            "INSERT INTO frontier_query_sources "
            "(query_id, source, last_pulled_at, cursor_ts, last_status, error_count) "
            "VALUES (?,?,?,?,?,?)",
            (query_id, source, stamp_at, cursor_ts, last_status,
             0 if last_status in {"ok", "empty"} else 1))
    else:
        conn.execute(
            "UPDATE frontier_query_sources SET "
            "  last_pulled_at = COALESCE(?, last_pulled_at),"
            "  cursor_ts      = MAX(COALESCE(?, ''), COALESCE(cursor_ts, '')),"
            "  last_status    = ?,"
            "  error_count    = CASE WHEN ? THEN 0 ELSE error_count + 1 END "
            "WHERE query_id=? AND source=?",
            (stamp_at, cursor_ts, last_status,
             1 if last_status in {"ok", "empty"} else 0, query_id, source))
    conn.commit()


def _jitter(query_id: str, source: str) -> float:
    """Stable ±JITTER from a hash, never random: 28 queries seeded on the same day would otherwise
    all fall due in the same second forever, turning a steady trickle into a thundering batch."""
    h = int(hashlib.sha256(f"{query_id}:{source}".encode()).hexdigest()[:8], 16)
    return 1.0 + ((h % 2001) / 1000.0 - 1.0) * JITTER


def is_due(row: sqlite3.Row | None, source: str, *, query_id: str, miss_count: int = 0,
           now: datetime | None = None) -> bool:
    """Whether this pair should be searched now.

    Three factors COMPOSE, in this order: the source's own TTL (how fast the upstream publishes),
    the decay tier (how interested the reader still is), and the per-pair jitter (so a batch seeded
    on one day does not fall due in the same second forever). The tier multiplies rather than
    replaces, so a demoted GitHub query is 30 × 48h — decay stretches the source's cadence instead
    of overriding a fact about the source.
    """
    if row is None or not row["last_pulled_at"]:
        return True                                    # never pulled → infinitely stale
    last = _parse(row["last_pulled_at"])
    if last is None:
        return True
    ttl = (TTL_HOURS.get(source, DEFAULT_TTL_HOURS)
           * tier_for(miss_count) * _jitter(query_id, source))
    return ((now or utc_now()) - last).total_seconds() / 3600.0 >= ttl


def since_for(row: sqlite3.Row | None, *, now: datetime | None = None) -> datetime:
    """Where this pair should resume, minus an overlap sliver.

    The overlap is not sloppiness: a hard boundary at the last pull time drops anything an upstream
    published while the previous request was in flight, and those gaps are invisible — nothing ever
    reports the paper you never saw. Dedup on `candidate_id` absorbs the re-ask for free.
    """
    ref = now or utc_now()
    base = _parse(row["last_pulled_at"]) if row and row["last_pulled_at"] else None
    if base is None and row and row["cursor_ts"]:
        base = _parse(row["cursor_ts"])
    if base is None:
        return ref - timedelta(days=FIRST_PULL_DAYS)   # bounded first look-back, not all of time
    return base - timedelta(hours=OVERLAP_HOURS)


def window_ok(since: datetime, now: datetime) -> bool:
    """Refuse an absurd window BEFORE spending the request.

    The realistic failure is not volume, it is a threading bug: if `since` fails to reach the
    adapter, the source silently applies its own default and every pull becomes a full-history
    re-read that looks exactly like a normal one.
    """
    delta = (now - since).total_seconds() / 86400.0
    return 0 <= delta <= MAX_WINDOW_DAYS


# ── Candidate store ─────────────────────────────────────────────────────────────
def upsert_candidate(conn, cand, query_id: str, *, now: str) -> bool:
    """Store a candidate and link it to the query that found it. True if newly seen.

    The link is a separate row rather than a column because multi-query hits are the signal: an
    artifact three independent standing queries all surfaced is the strongest thing stage 4 can
    show, and one `query_id` column would silently keep only whichever query happened to run first.
    """
    cur = conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, kind, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at)
           VALUES (?,?,?,?,?,?,?,?,'new',?,?)
           ON CONFLICT(candidate_id) DO UPDATE SET last_seen_at = excluded.last_seen_at""",
        (cand.candidate_id, cand.source, cand.kind, cand.title, cand.url, cand.published,
         cand.summary, json.dumps(cand.payload or {}), now, now))
    is_new = conn.execute(
        "SELECT first_seen_at = last_seen_at FROM frontier_candidates WHERE candidate_id=?",
        (cand.candidate_id,)).fetchone()[0]
    conn.execute(
        "INSERT OR IGNORE INTO frontier_candidate_queries (candidate_id, query_id, found_at) "
        "VALUES (?,?,?)", (cand.candidate_id, query_id, now))
    return bool(is_new) and cur.rowcount > 0


# ── The run ─────────────────────────────────────────────────────────────────────
@llm_client.rail(RAIL)
def run_frontier_execute(conn=None, *, force: bool = False, dry_run: bool = False,
                         registry: dict | None = None, now: datetime | None = None,
                         sleep=None) -> dict:
    """One execution pass. Never raises.

    The rail label goes on `run_*` and NOT on `main()` — `main()` only wraps this for the
    `--once` child, so labelling it would miss every in-process call the MCP side makes directly."""
    load_rail_env()
    ref = now or utc_now()
    own = conn is None
    if own:
        conn = schema.connect()
    try:
        # BEFORE any adapter is asked for anything. `dry_run` is gated too, deliberately: it skips
        # the WRITE but still calls `adapter.search`, so it makes the same external requests this
        # ceiling exists to bound. `force` does not bypass it either — force means "ignore the
        # freshness floor", never "ignore the money", matching `run_bookmark_catchup`.
        if _daily_budget_exhausted():
            return _record(conn, ref, "budget_paused", dry_run=dry_run,
                           reason=(f"today's recorded spend for this rail reached the "
                                   f"${FRONTIER_EXECUTE_DAILY_USD:.2f} daily frontier-execute "
                                   f"ceiling. It resets at UTC midnight."))
        if (reason := models_unroutable(RAIL)) is not None:
            return _record(conn, ref, "models_unroutable", dry_run=dry_run, reason=reason)
        return _run(conn, force=force, dry_run=dry_run,
                    registry=registry if registry is not None else adapters(), ref=ref,
                    sleep=sleep or _default_sleep)
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[frontier-exec] run errored: {detail}")
        try:
            conn.execute("INSERT INTO frontier_exec_runs (ran_at, status, reason) VALUES (?,?,?)",
                         (utc_iso(ref), "failed", detail))
            conn.commit()
        except Exception:
            pass
        return {"status": "failed", "reason": detail}
    finally:
        if own:
            conn.close()


def _run(conn, *, force: bool, dry_run: bool, registry: dict, ref: datetime, sleep) -> dict:
    queries = fq.active_queries(conn)
    if not queries:
        return _record(conn, ref, "skipped", reason="no active standing queries")

    # Build the pair list first so `pairs_due` is the honest denominator — the number of things
    # that WANTED to run, not the number that fit in the budget.
    due, no_adapter = [], 0
    for q in queries:
        for source in json.loads(q["target_sources"] or "[]"):
            if source not in registry:
                no_adapter += 1
                record_pull(conn, q["query_id"], source, last_status="no_adapter",
                            stamp=False, now=ref)
                continue
            row = get_pair(conn, q["query_id"], source)
            if force or is_due(row, source, query_id=q["query_id"],
                               miss_count=q["miss_count"], now=ref):
                due.append((q, source, row))

    # Stalest first, so the request budget delays everyone in turn rather than starving whichever
    # pairs sort last in active_queries() order forever. A never-pulled pair sorts first.
    due.sort(key=lambda d: (d[2]["last_pulled_at"] if d[2] else None) or "")

    requests = pulled = deferred = new_total = seen_total = 0
    last_call: dict[str, float] = {}     # per-source pacing clock
    for q, source, row in due:
        # Bound REQUESTS, not iterations. A pair that costs nothing (refused window, no adapter)
        # must not consume a slot, or one permanently-broken pair starves everything behind it.
        if requests >= MAX_REQUESTS_PER_RUN:
            deferred += 1
            continue
        adapter = registry[source]
        since = _lookback_floor(adapter, since_for(row, now=ref), ref)
        if not window_ok(since, ref):
            log(f"[frontier-exec] window refused for {q['text']!r}/{source}: since={since}")
            record_pull(conn, q["query_id"], source, last_status="window_refused",
                        stamp=False, now=ref)
            continue
        # Ask BEFORE pacing: an open breaker must not consume a request slot or wait out a
        # politeness delay for a call that will not leave the process.
        if not _available(adapter):
            record_pull(conn, q["query_id"], source, last_status="breaker_open",
                        stamp=False, now=ref)
            continue
        requests += 1
        _pace(adapter, source, last_call, sleep=sleep)
        try:
            found = adapter.search(q["text"], since=since, limit=PER_QUERY_LIMIT)
        except SourceError as e:
            log(f"[frontier-exec] {source} failed for {q['text']!r}: {e}")
            record_pull(conn, q["query_id"], source, last_status="error", stamp=False, now=ref)
            continue
        except Exception as e:                                   # an adapter bug is not fatal
            log(f"[frontier-exec] {source} raised for {q['text']!r}: {type(e).__name__}: {e}")
            record_pull(conn, q["query_id"], source, last_status="error", stamp=False, now=ref)
            continue

        found, cut = _relevance_cut(found)
        if cut:
            # Never a silent cap — same rule as the deferred count below.
            log(f"[frontier-exec] {source} {q['text']!r}: dropped {cut} of {cut + len(found)} "
                f"candidates under {INTAKE_KEEP_FRAC} of the page's top relevance_score")
        pulled += 1
        if dry_run:
            new_total += len(found)
            continue
        stamp = utc_iso(ref)
        for c in found:
            if upsert_candidate(conn, c, q["query_id"], now=stamp):
                new_total += 1
            else:
                seen_total += 1
        conn.commit()
        newest = max((c.published for c in found if c.published), default=None)
        record_pull(conn, q["query_id"], source, last_status="ok" if found else "empty",
                    cursor_ts=newest, stamp=True, now=ref)

    if deferred:
        # Never a silent cap: a run that quietly dropped a third of its work reads exactly like one
        # that covered everything.
        log(f"[frontier-exec] {deferred} pairs deferred (request budget {MAX_REQUESTS_PER_RUN})")
    return _record(conn, ref, "ok", pairs_due=len(due), pairs_pulled=pulled,
                   pairs_deferred=deferred, requests=requests,
                   candidates_new=new_total, candidates_seen=seen_total,
                   reason=(f"{no_adapter} pairs had no adapter" if no_adapter else None),
                   dry_run=dry_run)


def _relevance_cut(found: list) -> tuple[list, int]:
    """Keep candidates scoring at least `INTAKE_KEEP_FRAC` of their page's own top
    `relevance_score`: `(kept, dropped_count)`.

    An INGEST-time, one-way cut — a dropped work never becomes a candidate — keyed on the score
    the source itself ranked the page by, so it drops by match quality, never by timing (the
    rejected TTL instinct). Relative to the page's own top rather than absolute because the score
    is BM25-shaped: tops ranged 1.07-59.16 across the 22 measured queries, so no absolute number
    means the same thing twice.

    Candidates carrying no numeric score pass through untouched. arXiv (date-sorted by design)
    and GitHub (star-sorted) emit none, so the cut reaches exactly the source whose volume is the
    problem: OpenAlex, where any standing query finds twenty more works of roughly equal, roughly
    irrelevant match. The page's top always survives its own floor, so a non-empty page stays
    non-empty. Measured 2026-08-27: 0.5 clears the score plateau (~0.33-0.39 of top) with margin
    and cuts 60% of intake. Known limits and the re-check schedule:
    docs/Future-Investigations/2026-08-27-intake-keep-frac-rests-on-22-queries.md
    """
    scores = {c.candidate_id: s for c in found
              if isinstance(s := (c.payload or {}).get("relevance_score"), (int, float))
              and not isinstance(s, bool)}
    if not scores:
        return found, 0
    floor = INTAKE_KEEP_FRAC * max(scores.values())
    kept = [c for c in found if c.candidate_id not in scores or scores[c.candidate_id] >= floor]
    return kept, len(found) - len(kept)


def _lookback_floor(adapter, since: datetime, now: datetime) -> datetime:
    """Reach further back than the cursor when a SOURCE cannot window on its own index date.

    The cursor asks "what appeared since we last looked", and every source answers it with the
    only date it exposes. Where that date is PUBLICATION date and the source indexes late, the two
    are not the same question, and the difference is a permanent silent miss — the work is
    published before the window opens and indexed after it closes. `OVERLAP_HOURS` is the wrong
    dial for this: it is hours, and this gap is weeks.

    Declared on the adapter (like `min_interval_s`) and applied HERE rather than inside it, so
    `window_ok` still validates the window that is actually sent. An adapter that quietly widened
    the `since` it was handed is precisely the bug that assertion exists to catch.

    Optional: an adapter that does not declare one is unchanged, so the cursor stays the only
    input for every source whose window means what the loop thinks it means.
    """
    days = float(getattr(adapter, "min_lookback_days", 0.0) or 0.0)
    return since if days <= 0 else min(since, now - timedelta(days=days))


def _available(adapter) -> bool:
    """Whether a source is worth asking right now. Optional on the adapter — one that does not
    implement it is always considered available, so a new adapter stays a one-method object."""
    probe = getattr(adapter, "available", None)
    try:
        return True if probe is None else bool(probe())
    except Exception:
        return True


def _default_sleep(seconds: float) -> None:
    import time
    time.sleep(seconds)


def _pace(adapter, source: str, last_call: dict, *, sleep) -> None:
    """Wait out a source's minimum interval before calling it again. Paced in the loop, not inside
    each adapter, so the delay is visible next to the request budget and an adapter stays a dumb
    "query in, candidates out". `sleep` is injectable so tests don't actually wait.
    """
    interval = float(getattr(adapter, "min_interval_s", 0.0) or 0.0)
    if interval <= 0:
        return
    prev = last_call.get(source)
    if prev is not None:
        wait = interval - (_monotonic() - prev)
        if wait > 0:
            sleep(wait)
    last_call[source] = _monotonic()


def _monotonic() -> float:
    import time
    return time.monotonic()


def _record(conn, ref: datetime, status: str, *, dry_run: bool = False, **fields) -> dict:
    out = {"status": status, **{k: v for k, v in fields.items() if v is not None}}
    if dry_run:
        return {**out, "status": "dry-run"}
    cols = {"ran_at": utc_iso(ref), "status": status, **fields}
    conn.execute(
        f"INSERT INTO frontier_exec_runs ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})", tuple(cols.values()))
    conn.commit()
    return out


def _parse(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        d = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d


# ── The detached spawn ──────────────────────────────────────────────────────────
def spawn_frontier_execute(force: bool = False, coalesce_window: float = COALESCE_DEFAULT) -> bool:
    """Fire one execution pass as a detached, non-blocking child and return immediately.

    This rail owns its spawner (mirrors `spawn_oracle_refresh`) rather than sharing one with stage
    1, since the two fail for different reasons and need independent kill switches.

    Trigger rate is not request rate: cost is bounded by the per-pair TTL, so an hourly spawn stays
    cheap — a pass with nothing due exits after one SELECT.
    """
    return spawn_rail("pipeline.kb.frontier_execute", slug="frontier_exec",
                      force=force, coalesce=coalesce_window)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frontier stage 2 — execute the standing queries")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore per-pair TTLs")
    ap.add_argument("--dry-run", action="store_true", help="search, but write no candidates")
    args = ap.parse_args(argv)
    if not args.once:
        ap.print_help()
        return 2
    res = run_frontier_execute(force=args.force, dry_run=args.dry_run)
    print(json.dumps(res, indent=2, default=str))
    return 0 if res.get("status") in {"ok", "dry-run", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
