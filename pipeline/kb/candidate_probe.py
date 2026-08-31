"""
pipeline/kb/candidate_probe.py — the light timeline pull over the Oracle CANDIDATE list.

Piece 2 of the candidate-list Proposer (`docs/plans/2026-08-11-proposer-candidate-loop.md`): for
each person the user has already engaged with — a follow, a bookmark, a like, a List, a Substack
subscription — pull ONE shallow page of their own posts, so the question "who here works on AI and
biology?" is answered from their words instead of their bio.

The population is `curation_signals` → `screen.rank_candidates`, which is the strongest prior in
the system: membership is the user vouching for the person directly, with zero inference. This
module never scores anyone and never issues a verdict. It gathers evidence; the host judges.

Four constraints this module is built around: rate is the only (SHARED) ceiling, so the loop is
bounded, resumable, and constant-rate (see `_PACE_SECONDS`); zero tweets is a durable `empty`
fact, never re-fetched or read as "no field"; the cursor never exhausts (handled in
`fetch_user_tweets`); content must not land in `atoms` — everything writes through `probe_store`.

Filtering is the curation filter, not the substance filter: `_filter_and_stitch` drops RTs/
replies-to-others and keeps originals/self-threads with NO length gate — a Proposer asks "what
field is this person in", which a one-line aphorism answers.

No images: the OCR-VLM cascade doesn't scale to this population, so media enrichment is never
called here. Text only, and free.

"""

from __future__ import annotations

import re
import sqlite3
import time

from . import derive, probe_store, schema
from .embed import assert_model
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL, AtomSink, StageTimer,
                            body_fields, snapshot_and_hash)

_X_USER_RE = re.compile(r"^x:user:(\d+)$")

# How stale a candidate snapshot may be before it is re-pulled. 30 days: this content answers
# "what FIELD is this person in" (moves on a scale of seasons), and a full re-sweep is affordable
# monthly, not weekly. Shorten if the Proposer starts answering "what is this person working on
# right now" instead.
DEFAULT_TTL_DAYS = 30.0

# The page CAP, not the page count. Most candidates still cost one request; this is the ceiling for
# the ones a single page does not characterize (see `DEFAULT_SPAN_DAYS`).
DEFAULT_PAGES = 3

# The dial that actually matters: the walk stops once the sample covers `DEFAULT_SPAN_DAYS`
# (a representative characterization window) rather than a fixed page count, because a fixed
# count gives wildly uneven windows across accounts of different posting frequency.
DEFAULT_SPAN_DAYS = 90.0

# Seconds between requests: 3600 / 169 measured requests-per-hour, rounded up. CONSTANT-RATE, not
# burst-then-back-off, because the request budget is SHARED with every other GraphQL consumer on
# the machine — a burst here hands 429s to whichever scraper runs next.
_PACE_SECONDS = 22.0

_ATOM_PREFIX = "xprobe"
_SNAPSHOT_SOURCE = "probe"      # → kb_raw/probe/, its own directory


# ── the queue ─────────────────────────────────────────────────────────────────

def _x_user_id(members: list[str]) -> str | None:
    """The numeric X id in a resolved cluster, or None. A candidate with no X identity (a
    Substack-only subscription) has no timeline on this path — absent, not failed."""
    for m in members:
        hit = _X_USER_RE.match(m or "")
        if hit:
            return hit.group(1)
    return None


def candidate_queue(conn: sqlite3.Connection, *, min_signals: int = 1,
                    ttl_days: float = DEFAULT_TTL_DAYS) -> list[dict]:
    """The candidates due for a pull, in the SCREEN's own rank order (endorsement, then distinct
    signals, then count as a soft tiebreak — `screen.Candidate.sort_key`).

    Reusing that ordering rather than inventing one is deliberate: it is already the answer to
    "who has the user vouched for hardest", and a second ranking here would be a second opinion
    with no evidence behind it. Ordering picks who goes first under a bounded budget; it never
    picks membership, so everyone stays eventually reachable.

    Excluded: confirmed Oracles (their real footprint is already in `atoms` — probing them would
    duplicate trusted content into the untrusted store), candidates with no X identity, candidates
    below `min_signals`, and candidates whose snapshot is still fresh."""
    from . import screen

    fresh = probe_store.fresh_who_ids(conn, ttl_days=ttl_days)
    out: list[dict] = []
    for cand in screen.rank_candidates(conn):
        if cand.distinct_signals < min_signals:
            continue
        if schema.is_oracle(conn, cand.canonical_id):
            continue
        uid = _x_user_id(cand.members)
        if not uid:
            continue
        who_id = f"x:user:{uid}"
        if who_id in fresh:
            continue
        out.append({"who_id": who_id, "user_id": uid, "canonical_id": cand.canonical_id,
                    "name": cand.name, "handle": cand.handle,
                    "distinct_signals": cand.distinct_signals})
    return out


# ── one candidate ─────────────────────────────────────────────────────────────

def _render_groups(groups: list[list[dict]], *, handle: str) -> list[dict]:
    """Stitched tweet groups → write-ready probe atoms. PURE (no DB, no network, no embedder), so
    the whole rendering path is testable without a session."""
    from pipeline.ingestion import x_render as xt

    out: list[dict] = []
    for group in groups:
        root = group[0]                       # chronologically first = the atom's canonical
        root_id = str(root.get("id", ""))
        if not root_id:
            continue
        is_thread = len(group) > 1
        # An X-Article renders as its TEASER here: fetching the real body is a paid twitterapi call,
        # and this path is free by design. Recorded as `partial` rather than passed off as whole —
        # a truncated body quoted as if complete is a fabricated citation.
        is_article = any(xt._article_tweet_id(t) for t in group)
        md = xt.tweet_to_markdown(root, thread_tweets=group if is_thread else None,
                                  source="x-probe", footer_label="Candidate probe")
        meta = derive.derive_x(root)
        out.append({
            "atom_id": f"{_ATOM_PREFIX}:{root_id}",
            "source_type": "x",
            "who_id": meta["who_id"],
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "source_url": root.get("url") or f"https://x.com/{handle}/status/{root_id}",
            "description": meta["description"],
            "payload": {
                "like_count": root.get("likeCount", 0),
                "reply_count": root.get("replyCount", 0),
                "is_thread": is_thread,
                "thread_len": len(group),
                "is_quote": bool(root.get("isQuote") or root.get("quoted_tweet")),
                "is_article": is_article,
                "has_media": any((t.get("extendedEntities") or {}).get("media") for t in group),
                "source_tags": meta["source_tags"],
                **body_fields(BODY_PARTIAL if is_article else BODY_COMPLETE, BASIS_OBSERVED),
            },
            "_markdown": md,
        })
    return out


def _span_days(tweets: list[dict]) -> float:
    """Days between the oldest and newest post in hand. `0.0` when nothing is dated.

    PURE and separately named so the stop rule can be tested without a network. Unparseable or
    missing timestamps are SKIPPED rather than treated as epoch — a single bad date read as 1970
    would report a 20,000-day span and stop the walk on its first page, which is the failure mode
    that looks like success."""
    from datetime import datetime

    stamps = []
    for t in tweets:
        raw = t.get("createdAt") or t.get("created_at") or ""
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
            try:
                stamps.append(datetime.strptime(str(raw), fmt).timestamp())
                break
            except (ValueError, TypeError):
                continue
    if len(stamps) < 2:
        return 0.0
    return (max(stamps) - min(stamps)) / 86400.0


def _enough_for_characterization(span_days: float, pace_seconds: float = 0.0):
    """The Proposer's `after_page` hook: stop once the sample covers `span_days`, and PACE if not.

    A closure rather than a constant so the policy travels with the caller — `fetch_user_tweets`
    takes a hook and knows nothing about characterization windows, and never sleeps itself. The
    span check runs FIRST so a finished walk returns immediately instead of paying for a pause
    before a request it will never make."""
    def _after(tweets: list[dict]) -> bool:
        if _span_days(tweets) >= span_days:
            return True
        if pace_seconds > 0:
            time.sleep(pace_seconds)
        return False
    return _after


def probe_candidate(conn, embedder, cookies: dict, headers: dict, cand: dict, *,
                    pages: int = DEFAULT_PAGES,
                    span_days: float = DEFAULT_SPAN_DAYS,
                    pace_seconds: float = 0.0) -> dict:
    """Pull, filter, render and store ONE candidate's timeline page. Returns a per-candidate
    summary and RECORDS the outcome in `probe_pulls`.

    Raises nothing for a per-candidate problem — a fetch failure records `failed` (which is always
    due again next run) and returns. It DOES let `XRateLimited` and `SyncAuthError` propagate: those
    say the whole session is spent or dead, so every remaining candidate would fail identically and
    the loop above must stop rather than burn through the queue marking everyone failed."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.utils import log
    from .ingest_x_footprint import _filter_and_stitch

    who_id, handle = cand["who_id"], (cand.get("handle") or "")
    try:
        raw = core.fetch_user_tweets(
            cookies, headers, cand["user_id"], pages=pages,
            after_page=_enough_for_characterization(span_days, pace_seconds)
            if span_days > 0 else None)
    except core.XUserUnavailable as e:
        probe_store.record_pull(conn, who_id, probe_store.STATUS_UNAVAILABLE, detail=str(e))
        return {"who_id": who_id, "status": probe_store.STATUS_UNAVAILABLE, "atoms": 0}
    except (core.XRateLimited, core.SyncAuthError):
        raise                                   # session-wide — the LOOP decides, not this function
    except Exception as e:                      # per-candidate: SKIP, no write, always retried
        log(f"[probe] {who_id} (@{handle}) fetch failed — SKIP (retried next run): "
            f"{type(e).__name__}: {e}")
        probe_store.record_pull(conn, who_id, probe_store.STATUS_FAILED,
                                detail=f"{type(e).__name__}: {e}")
        return {"who_id": who_id, "status": probe_store.STATUS_FAILED, "atoms": 0}

    if not raw:
        # A real account that posted nothing in the window. A FACT about the candidate, recorded so
        # it is never re-fetched every run and never read as "no field".
        probe_store.record_pull(conn, who_id, probe_store.STATUS_EMPTY)
        return {"who_id": who_id, "status": probe_store.STATUS_EMPTY, "atoms": 0, "fetched": 0}

    groups = _filter_and_stitch(raw)            # curation filter only — no substance gate
    atoms = _render_groups(groups, handle=handle)
    seen = probe_store.load_probe_hashes(conn, who_id)

    written = {"n": 0}
    sink = AtomSink(conn, embedder, writer=probe_store.write_probe_atom)
    submitted = skipped = 0
    for atom in atoms:
        md = atom.pop("_markdown")
        decided = snapshot_and_hash(_SNAPSHOT_SOURCE, atom["atom_id"], md, seen)
        if decided is None:                     # unchanged → skip the (paid) embed
            skipped += 1
            continue
        atom["raw_ref"], atom["raw_hash"] = decided
        seen[atom["atom_id"]] = atom["raw_hash"]
        submitted += 1
        # `on_written` fires only for a DURABLY stored atom, so a poison chunk the sink isolates is
        # counted as neither written nor seen, and is retried.
        sink.submit(atom, md, on_written=lambda: written.__setitem__("n", written["n"] + 1))
    sink.close()

    # `ok` REQUIRES everything submitted to have actually landed — a systemic embed failure (bad
    # key, open credit breaker) would otherwise record `ok` on a silent shortfall and freeze that
    # candidate for a full TTL. `submitted == 0` is NOT a shortfall (all-replies page, or unchanged).
    if submitted and written["n"] < submitted:
        detail = f"{written['n']}/{submitted} atoms stored (embed or write failed)"
        log(f"[probe] {who_id} (@{handle}) — {detail}; recording FAILED so it retries")
        probe_store.record_pull(conn, who_id, probe_store.STATUS_FAILED,
                                atoms=written["n"], detail=detail)
        return {"who_id": who_id, "status": probe_store.STATUS_FAILED, "fetched": len(raw),
                "groups": len(groups), "submitted": submitted, "written": written["n"],
                "unchanged": skipped, "atoms": written["n"]}

    probe_store.record_pull(conn, who_id, probe_store.STATUS_OK, atoms=written["n"])
    return {"who_id": who_id, "status": probe_store.STATUS_OK, "fetched": len(raw),
            "groups": len(groups), "submitted": submitted, "written": written["n"],
            "unchanged": skipped, "atoms": written["n"]}


# ── the run ───────────────────────────────────────────────────────────────────

def probe_candidates(conn, embedder, *, min_signals: int = 1, max_candidates: int = 0,
                     ttl_days: float = DEFAULT_TTL_DAYS, pages: int = DEFAULT_PAGES,
                     span_days: float = DEFAULT_SPAN_DAYS,
                     pace_seconds: float = _PACE_SECONDS, profile: str | None = None) -> dict:
    """Walk the due candidates, newest-vouched first, one timeline page each.

    `max_candidates` is the request budget for this run (one page = one request), which is the only
    scarce resource on this path — 0 means the whole queue, which at 961 candidates is ~5.7 hours of
    paced requests. Bounded runs are the normal mode: state lives in `probe_pulls`, so stopping and
    resuming costs nothing and re-pulls nobody.

    Returns a run summary. A dead session or a spent rate budget STOPS the run with `stopped` set —
    it never marks the remaining queue failed, because nothing was observed about them."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.utils import log

    assert_model(conn, embedder)      # guard the store's embedding identity BEFORE any work
    # Preflight that the embedder can actually EMBED, before spending shared X request budget —
    # `assert_model` only checks identity, not liveness. Fail loud here rather than 25 requests
    # later as 25 skipped candidates.
    try:
        embedder.embed(["probe preflight"], role="document")
    except Exception as e:
        log(f"[probe] embedder unavailable — NOTHING pulled (no X requests spent): {e}")
        return {"source": "candidate-probe", "stopped": "embedder",
                "error": f"{type(e).__name__}: {e}", "requests": 0, "atoms": 0}
    timer = StageTimer()
    queue = candidate_queue(conn, min_signals=min_signals, ttl_days=ttl_days)
    if max_candidates:
        queue = queue[:max_candidates]
    if not queue:
        return {"source": "candidate-probe", "queued": 0, "note": "no candidate is due"}

    try:
        cookies = core.read_x_cookies(profile=profile)
    except core.SyncAuthError as e:
        # Degrade, never crash: a missing X session is a broken SOURCE, not a broken run.
        log(f"[probe] no usable X session — nothing pulled: {e}")
        return {"source": "candidate-probe", "queued": len(queue), "stopped": "auth",
                "error": str(e)}
    headers = core.auth_headers(cookies, referer="https://x.com/home")

    tally = {probe_store.STATUS_OK: 0, probe_store.STATUS_EMPTY: 0,
             probe_store.STATUS_UNAVAILABLE: 0, probe_store.STATUS_FAILED: 0}
    atoms = requests = 0
    stopped: str | None = None
    # A range, not a point: most candidates stop on page one, thin ones page up to `pages`.
    log(f"[probe] {len(queue)} candidate(s) due (min_signals={min_signals}, ttl={ttl_days}d) — "
        f"FREE, paced at {pace_seconds}s/request, span target {span_days:.0f}d, ≤{pages} pages "
        f"≈ {len(queue) * pace_seconds / 60:.0f}-{len(queue) * pages * pace_seconds / 60:.0f} min")

    for i, cand in enumerate(queue):
        if i and pace_seconds > 0:
            time.sleep(pace_seconds)          # constant-rate: see `_PACE_SECONDS`
        try:
            with timer.stage("probe"):
                # `pace_seconds` is passed down so a multi-page candidate paces its OWN requests
                # too, not just the gap between candidates.
                res = probe_candidate(conn, embedder, cookies, headers, cand, pages=pages,
                                      span_days=span_days, pace_seconds=pace_seconds)
        except core.XRateLimited as e:
            log(f"[probe] rate budget spent after {requests} request(s) — stopping. {e}")
            stopped = "rate_limited"
            break
        except core.SyncAuthError as e:
            log(f"[probe] session died after {requests} request(s) — stopping. {e}")
            stopped = "auth"
            break
        requests += 1
        tally[res["status"]] = tally.get(res["status"], 0) + 1
        atoms += res.get("atoms", 0)

    return {"source": "candidate-probe", "queued": len(queue), "requests": requests,
            "atoms": atoms, "by_status": tally, "stopped": stopped,
            # What is left AFTER this run — the number a caller schedules the next run against.
            "remaining": len(candidate_queue(conn, min_signals=min_signals, ttl_days=ttl_days)),
            "probe_total": probe_store.count_probe_atoms(conn),
            "stage_seconds": timer.totals, "stage_latency": timer.distribution()}


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    from .embed import get_kb_embedder

    ap = argparse.ArgumentParser(
        description="Light timeline pull over the Oracle candidate list (FREE; honors $OPYT_HOME).")
    ap.add_argument("--min-signals", type=int, default=1,
                    help="Only candidates with at least this many DISTINCT (type, platform) "
                         "signals. 3 ≈ 29 people, 2 ≈ 196, 1 ≈ everyone (rerun10 snapshot).")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="THE REQUEST BUDGET for this run (0 = the whole due queue). One page per "
                         "candidate = one request; the run is resumable, so bounding it is free.")
    ap.add_argument("--ttl-days", type=float, default=DEFAULT_TTL_DAYS,
                    help="How stale a snapshot may be before re-pull. 0 re-pulls everyone.")
    ap.add_argument("--pages", type=int, default=DEFAULT_PAGES,
                    help=f"CAP on timeline pages per candidate (default {DEFAULT_PAGES}). Most "
                         f"candidates stop on page 1; this bounds the thin ones.")
    ap.add_argument("--span-days", type=float, default=DEFAULT_SPAN_DAYS,
                    help=f"Stop paging once the sample covers this many days (default "
                         f"{DEFAULT_SPAN_DAYS:.0f}). 0 disables the span stop and pages to the "
                         f"cap. This is the dial that controls characterization quality; --pages "
                         f"only bounds its cost.")
    ap.add_argument("--pace-seconds", type=float, default=_PACE_SECONDS,
                    help="Constant-rate gap between requests. 0 disables pacing (will 429).")
    ap.add_argument("--x-profile", default=None, help="X cookie profile (else auto-pick).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the due queue and exit — no requests, no writes.")
    args = ap.parse_args(argv)

    conn = schema.connect()
    try:
        if args.dry_run:
            queue = candidate_queue(conn, min_signals=args.min_signals, ttl_days=args.ttl_days)
            # A range, matching the run's own log line (a candidate can page up to `--pages`).
            lo = len(queue) * args.pace_seconds / 60
            print(f"[probe] {len(queue)} candidate(s) due (≈{lo:.0f}-{lo * args.pages:.0f} min at "
                  f"{args.pace_seconds}s/request, span target {args.span_days:.0f}d, "
                  f"≤{args.pages} pages)")
            for c in queue[:50]:
                print(f"  {c['distinct_signals']}× {c['who_id']:<20} "
                      f"@{c['handle'] or '?'}  {c['name'] or ''}")
            return 0
        embedder = get_kb_embedder()
        print(f"[probe] embedder: model={embedder.model} provider={embedder.provider}")
        out = probe_candidates(conn, embedder, min_signals=args.min_signals,
                               max_candidates=args.max_candidates, ttl_days=args.ttl_days,
                               pages=args.pages, span_days=args.span_days, pace_seconds=args.pace_seconds,
                               profile=args.x_profile)
    finally:
        conn.close()
    print("[probe] summary:\n" + _json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
