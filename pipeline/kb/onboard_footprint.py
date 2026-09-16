"""
pipeline/kb/onboard_footprint.py — atom-KB footprint ingest for a confirmed Oracle.

Fills the reserved `oracle(action='ingest')` gap (.guards.py): `discover_profile`
— now surfacing blog-hub profiles (Phases A–C) — becomes the shared discovery
brain, and this router turns its TRUSTED profile sources into atoms attributed to
the Oracle (who_id via `resolve`), instead of the retiring vault.

One source → one route:

  personal blog / substack → source_adapters.gate_and_sync_website → resolve
                             (gate SKIP → record_affiliation; no atoms)
  github (repo or profile) → sync_github_source → resolve
  shape == "org"           → record_affiliation (an org fact-node, no atoms)
  scholar / orcid          → NOT BUILT (needs a profile→publication resolver;
                             the shared paper ingester already downloads PDFs)

INVARIANT (guard-enforced, .guards.py): the footprint adapters are DUMB — they
attribute a whole site to `who_id = the Oracle`, so they MUST only ever run on a
source that passed `eligibility.gate`. This router does not hold its own copy of that
order — it calls `source_adapters.gate_and_sync_website`, which is where the gate sits
next to the adapter it guards; a `skip` (multi-author/org) becomes an affiliation here,
never atoms.

Fail-safe: an adapter that raises is caught per-source (reported as "error"),
never aborting the other sources; an org url that won't canonicalize records
nothing rather than a dangling edge.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

_FOOTPRINT = ("blog", "substack")


def onboard_footprint(conn: sqlite3.Connection, embedder, oracle_id: str,
                      sources: list[dict], *, author_name: str | None = None,
                      force: bool = False, since: datetime | None = None,
                      limit: int = 0, on_source=None) -> dict:
    """Route a confirmed Oracle's discovered profile `sources` into the atom-KB.

    `sources` are `discover_profile` source dicts ({source_type, url, metadata, trust, …}).
    Only TRUSTED personal profiles ingest; org-shaped links (from a trusted hub) become
    affiliations; everything untrusted-and-non-org is left for the review step. Returns a
    per-source outcome report.

    `limit` is DISPATCH-BOUNDED for the two WEBSITE adapters — it caps posts handed to the pool,
    and a paywalled/gate-rejected post still spends one of the N (see `expand._route_source`).
    The summary reports `dispatched` alongside `atoms_added` so the gap is visible.

    `sync_github` IGNORES `limit` (repos have no post concept) and excludes forks by default, since
    a fork's atom would carry the upstream's README under this person's `who_id`. `min_stars`
    stays 0 by default — it's a lagging notability proxy that penalizes new work; a caller wanting
    a highlight reel passes it explicitly

    Outcome vocabulary: a source ends as exactly one of `ingested` / `blocked` / `error` /
    `affiliation` / `needs-review` / `not-built` / `skipped` / `unsupported`. `blocked` means
    transient (nothing written, nothing marked seen, the next run redoes the walk); `error` means
    somebody should look. Adapters signal a host-side stop by RETURNING a summary with `error`
    set rather than raising, so the try/except below can't distinguish them on its own — see
    `_record_run`.

    `on_source` is called with each outcome record the instant that source finishes, before the
    next one starts. It exists so a caller can make the run's bookkeeping durable at the same grain
    the atoms already are: `AtomSink` flushes incrementally, so a run killed part-way leaves atoms
    behind, and an end-of-run stamp leaves them with no record of what was attempted. A callback
    that raises is the caller's problem to contain — see `oracles._stamp_source`, which does.
    """
    from pipeline.ingestion.utils import log

    from . import eligibility, ingest_common, ingest_github, resolve, source_adapters

    oracle_id = (oracle_id or "").strip()
    if not oracle_id:
        return {"error": "onboard_footprint needs an oracle_id (e.g. x:user:123) to attribute to.",
                "results": []}

    results: list[dict] = []
    # ROUTER-level wall clock, one stage per routed source (`sync_github` carries no timer of its
    # own). Labelled `{source_type}`, not by url, so multiple sources of one type fold into one
    # stage. A website stage covers the eligibility gate AND the adapter, because they are one
    # call now; the gate's own share is the stage total minus the adapter's `stage_seconds`, which
    # rides along on that source's record.
    timer = ingest_common.StageTimer()

    def _record(url, stype, action, detail="", stats=None):
        rec = {"url": url, "type": stype, "action": action, "detail": detail}
        if stats:
            rec["stats"] = stats
        results.append(rec)
        if on_source:
            on_source(rec)

    def _record_run(url, stype, summary):
        """Record ONE adapter run under the outcome it actually had — without this, a blocked
        archive walk (zero atoms, nothing marked seen) is counted in the `ingested` total."""
        outcome = ingest_common.classify_run(summary)
        _record(url, stype, outcome, str(summary), stats=ingest_common.run_stats(summary))
        return outcome

    for src in sources or []:
        stype = src.get("source_type") or ""
        url = src.get("url") or ""
        meta = src.get("metadata") or {}
        shape = meta.get("shape")
        trusted = bool((src.get("trust") or {}).get("trusted"))
        if not url:
            continue

        # 1. Org-shaped (surfaced from a trusted hub) → an affiliation fact, never atoms.
        if shape == "org":
            org = eligibility.record_affiliation(conn, url, org_name=author_name)
            _record(url, stype, "affiliation" if org else "skipped",
                    f"organization retained: {org}" if org else "org url did not canonicalize")
            continue

        # 2. The trust boundary AT the ingest seam — only trusted personal profiles proceed.
        if not trusted:
            # "The review step" names a step the USER has no referent for — they never saw one,
            # and on 2026-09-15 the host passed the phrase straight through. Say what is actually
            # true of the link instead: Opyt found it, it is not confirmed as this person's, and
            # the user is the one who decides.
            _record(url, stype, "needs-review",
                    "found, but not confirmed as this person's — the user decides whether it is")
            continue

        # 3. Personal blog / Substack → the shared gate-then-sync seam. The gate order is
        #    `source_adapters`' to own (INVARIANT above); onboarding owns only what a refusal
        #    MEANS here — an org becomes an affiliation, an unknown verdict waits for review.
        if stype in _FOOTPRINT:
            try:
                with timer.stage(stype):
                    decision, summary = source_adapters.gate_and_sync_website(
                        conn, embedder, stype, url, handle=meta.get("handle"),
                        author_name=author_name, since=since, limit=limit, force=force)
                if summary is None:
                    if decision.decision == "skip":         # multi-author/org → affiliation
                        org = eligibility.record_affiliation(conn, url, org_name=author_name)
                        _record(url, stype, "affiliation" if org else "skipped",
                                f"gate skip ({decision.reason})")
                    else:                                   # needs-review (unknown / mismatch)
                        _record(url, stype, "needs-review", decision.reason)
                    continue
                # Resolve even on a blocked run: the adapter upserts the substack:/blog: entity
                # with its identity link BEFORE the archive walk, so the merge into the Oracle's
                # canonical is still owed regardless of whether any post came back.
                resolve.resolve_entities(conn)              # attribution materializes here
                _record_run(url, stype, summary)
            except Exception as e:                          # one bad source never aborts the rest
                log(f"[onboard] {stype} ingest failed for {url}: {type(e).__name__}: {e}")
                _record(url, stype, "error", f"{type(e).__name__}: {e}")
            continue

        # 4. GitHub → artifact atoms. `sync_github_source` owns the repository-vs-account
        #    split (a bio link to `acme/memory` is one repo, not an account named `memory`);
        #    `resolve` merges a crawled account into the Oracle via the identity link
        #    `_seed_owner_identity` stores.
        if stype == "github":
            try:
                with timer.stage("github"):
                    summary = ingest_github.sync_github_source(conn, embedder, url)
                if summary is None:
                    _record(url, stype, "skipped", "no github repo or account in url")
                    continue
                resolve.resolve_entities(conn)
                _record_run(url, stype, summary)
            except Exception as e:
                log(f"[onboard] github ingest failed for {url}: {type(e).__name__}: {e}")
                _record(url, stype, "error", f"{type(e).__name__}: {e}")
            continue

        # 5. Scholar / ORCID → NOT BUILT: profile-to-publication discovery has no adapter.
        # `not-built`, not `deferred`: `deferred` means bounded work something WILL resume — that is
        # what it means on `oracle(action='ingest')`'s rate-limited X pull and in `frontier_execute`
        # — and nothing resumes this one. Sharing the word would let the user-facing copy promise
        # a continuation that no rail performs.
        if stype in ("scholar", "orcid"):
            _record(url, stype, "not-built", "scholar/orcid url→author-id resolver not built yet")
            continue

        # Other source types have no atom-KB footprint adapter here.
        _record(url, stype, "unsupported", f"no atom-KB footprint adapter for {stype!r}")

    def _stat(key):
        return sum((r.get("stats") or {}).get(key, 0) for r in results)

    # This dict is what the MCP `add_oracle`/`oracle` tool hands the model, i.e. what the USER is
    # told. `ingested` counts only runs that actually ingested (a blocked host is its own count);
    # the explanatory counters (`dispatched`, `atoms_added`, `producer_failed`) are integers, not
    # substrings of a stringified summary in `detail`.
    return {
        "oracle_id": oracle_id,
        "ingested": sum(1 for r in results if r["action"] == "ingested"),
        "blocked": sum(1 for r in results if r["action"] == "blocked"),
        "errors": sum(1 for r in results if r["action"] == "error"),
        "affiliations": sum(1 for r in results if r["action"] == "affiliation"),
        "needs_review": sum(1 for r in results if r["action"] == "needs-review"),
        "not_built": sum(1 for r in results if r["action"] == "not-built"),
        "atoms_added": _stat("added"),
        "dispatched": _stat("dispatched"),
        "producer_failed": _stat("producer_failed"),
        "undetermined": _stat("undetermined"),
        # Per-source-type wall clock; `stage_latency` gives the distribution since labels can
        # legitimately repeat (an Oracle with two blogs, several gated sites).
        "stage_seconds": dict(timer.totals),
        "stage_latency": timer.distribution(),
        "results": results,
    }
