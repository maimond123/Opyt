"""
pipeline/kb/expand.py — Stage-5 footprint expansion: a confirmed Oracle -> discover + verify
their off-X sources -> route each into the atom-KB.

Roots discovery on the Oracle's X @handle via `pipeline.ingestion.discover_profile` (same probes
+ trust graph as the legacy vault path), then routes each trust-verified source to its atom-KB
adapter.

Two invariants:
  - Trusted sources are ingested; needs-review sources are RETURNED, never ingested, so a caller
    can put them in front of a human. (The "one-click confirm" consumer this was built for was
    never wired; its `ingest_confirmed_sources` was deleted 2026-08-28, unused since it was
    written. Confirming a source today goes through `add_oracle`.) Nothing is silently dropped or
    added — a JS-rendered Substack with no back-edge routinely lands in needs-review rather than
    being dropped or auto-trusted.
  - Trust is per-source, not per-person: confirming the person verifies their X identity, but a
    discovered URL (e.g. `elonmusk.substack.com`) still needs its own attested-link trust check.

Hand-rolled rather than reusing the vault rail's add-a-person path, which wrote notes instead of
`chunks.vector` and has since been deleted — this is the bridge between the atom-KB adapters
and `oracles`.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pipeline.timeparse import utc_now

from pipeline.ingestion.utils import log

from . import (ingest_common, ingest_github, ingest_x_footprint, oracles, resolve, schema,
               source_adapters)

# Source types with atom-KB adapters. scholar/youtube/podcast have none yet (youtube was
# re-filed to Radar) and are recorded skipped, not dropped. GitHub is unioned in explicitly since
# it's deliberately not a `source_adapters.WEBSITE_ADAPTERS` key (see that module's docstring).
_ROUTABLE = frozenset(source_adapters.WEBSITE_ADAPTERS) | {"github"}

_GITHUB_OWNER = re.compile(r"github\.com/([^/?#]+)", re.I)

# Lookback presets surfaced at onboarding. X is an ephemeral stream: short default, hard 2-year
# ceiling (enforced in sync_x_footprint). ⚠️ It was a PAID stream until the 2026-08-30 cutover, and
# that is no longer why the window is short — every X read now runs free on the user's own session.
# What still bounds it is wall-clock and one session's rate limits. Substack/blog are a durable
# corpus: default to the whole archive. A preset maps to days-before-now; None = 'all' (no lower
# bound), resolved to a `since` datetime once, at the entry point. Substack presets govern blog too.
X_LOOKBACK_PRESETS: dict[str, int] = {"6mo": 183, "1yr": 365, "2yr": 730}
WEB_LOOKBACK_PRESETS: dict[str, int | None] = {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
# Bookmarks get their own presets, and they are OPERATOR-ONLY: reached from
# `ingest_curation.py`'s `--bookmark-lookback` and nothing else. No MCP tool takes a bookmark
# window, and `oracle`'s `lookback_options` deliberately stops advertising one (2026-08-30) — it
# was asking users a question whose answer had nowhere to go.
#
# Default `all`, and leave it there. The walk is a free cookie-scrape; the per-bookmark thread
# fetch went free with the X cutover, leaving only the VLM read on bookmarks carrying images —
# measured $0.105 across 315 images on a ~1,080-bookmark backlog. Narrowing therefore saves cents
# and costs corpus, on an axis that misleads: this filters on when the tweet was WRITTEN, not when
# it was saved, because X exposes no bookmark timestamp. A 6-month window drops the 2019 paper
# saved yesterday.
BOOKMARK_LOOKBACK_PRESETS: dict[str, int | None] = {"6mo": 183, "1yr": 365, "2yr": 730,
                                                    "5yr": 1825, "all": None}


def _since_from_days(days: int | None) -> datetime | None:
    """A `since` datetime `days` before now, or None for 'all' (no lower bound)."""
    if not days:
        return None
    return utc_now() - timedelta(days=days)
_X_HANDLE = re.compile(r"(?:x|twitter)\.com/([^/?#]+)", re.I)


def _x_handle(conn: sqlite3.Connection, canonical_id: str) -> str | None:
    """The Oracle's X @handle — discovery is handle-rooted, and the atom-KB stores people by
    numeric `x:user:{id}`, not handle. The handle lives in the x member's `profile.handle`
    (both `ingest_x` and `ingest_curation._stamp_x_person` persist it there). Resolves a stale
    canonical_id to the current head first (post-resolve head-drift)."""
    head = schema.current_canonical(conn, canonical_id)
    row = conn.execute(
        "SELECT profile FROM entities WHERE COALESCE(canonical_id, entity_id)=? "
        "AND entity_id LIKE 'x:user:%' AND profile IS NOT NULL LIMIT 1",
        (head,),
    ).fetchone()
    if not row or not row[0]:
        return None
    return (json.loads(row[0]) or {}).get("handle")


def _github_owner(url: str | None) -> str | None:
    m = _GITHUB_OWNER.search(url or "")
    return m.group(1) if m else None


def _x_handle_from_url(url: str | None) -> str | None:
    m = _X_HANDLE.search(url or "")
    return m.group(1) if m else None


def _substack_handle(oracle: dict) -> str | None:
    """A Substack member's handle for the public_profile probe — extracted from its
    `substack:{handle}` entity_id (`derive.substack_entity_id` keys on the author handle,
    falling back to the subdomain). None when there's no usable Substack member."""
    for m in oracle.get("members", []):
        eid = m.get("entity_id", "")
        if eid.startswith("substack:"):
            h = eid.split("substack:", 1)[1].strip()
            if h and h != "unknown":
                return h
    return None


def _first_url(links) -> str | None:
    """The first http… URL in an entity's `identity_links` (a JSON string or a list). None when
    there's no usable link — the caller reconstructs a home from the `blog:{host}` id instead."""
    if not links:
        return None
    if isinstance(links, str):
        try:
            links = json.loads(links)
        except (ValueError, TypeError):
            return links if links.startswith("http") else None
    if isinstance(links, list):
        for u in links:
            if isinstance(u, str) and u.startswith("http"):
                return u
    return None


def _blog_home(oracle: dict) -> str | None:
    """A blog member's home URL for the blog root probe — the stored home from its `identity_links`,
    else reconstructed from the `blog:{host}` entity id. None when there's no usable blog member."""
    for m in oracle.get("members", []):
        eid = m.get("entity_id", "")
        if eid.startswith("blog:"):
            url = _first_url(m.get("identity_links"))
            if url:
                return url
            host = eid.split("blog:", 1)[1].strip()
            if host and host != "unknown":
                return f"https://{host}"
    return None


def _root_profile(conn: sqlite3.Connection, oracle: dict) -> dict | None:
    """The Oracle's best CONFIRMED root profile for discovery, as {seed, seed_type}.

    Preference order: X (richest probe + zero-regression for every existing X-rooted Oracle),
    then a Substack member, then a blog member. Returns None only for a github-only Oracle (no
    root probe). De-X-rooting: a Substack/blog-only person roots on that home, with no X anywhere
    — `discover_profile` fans out from there to find their other accounts."""
    x = _x_handle(conn, oracle["canonical_id"])
    if x:
        return {"seed": x, "seed_type": "x"}
    sub = _substack_handle(oracle)
    if sub:
        return {"seed": sub, "seed_type": "substack"}
    blog = _blog_home(oracle)
    if blog:
        return {"seed": blog, "seed_type": "blog"}
    return None


def _x_handle_to_pull(root: dict, profile: dict) -> str | None:
    """The Oracle's X @handle to pull the timeline for — the ROOT handle when X-rooted, else
    the handle of a DISCOVERED, trust-verified X account (a Substack-rooted Oracle whose X was
    found + graduated via Rule 5). None when there is no findable/trusted X. Every Oracle's X
    timeline is their richest channel, so it is pulled whenever it exists, regardless of which
    platform the Oracle was rooted from (`_classify_url` now tags a discovered X link `"x"`)."""
    if root["seed_type"] == "x":
        return root["seed"]
    for s in profile.get("sources") or []:
        if s.get("source_type") == "x" and (s.get("trust") or {}).get("trusted"):
            h = _x_handle_from_url(s.get("url") or "")
            if h:
                return h
    return None


def _route_source(conn, embedder, source: dict, *, author_name: str | None,
                  limit: int, github_min_stars: int = 0, web_since: datetime | None = None,
                  github_since: datetime | None = None) -> dict:
    """Route ONE discovered source (a `discover_profile` source dict) to its atom-KB adapter.
    Unadapted types are SKIPPED (recorded), never errors.

    WEBSITE sources (substack/blog) pass a single-author eligibility gate first: the adapters
    stamp `who_id`=the Oracle by inference, so an ungated multi-author site would launder other
    authors onto one trusted person. This is the single gated door for both auto-discovery and
    Pick #2 confirm; no `force` here — confirming a source's identity is orthogonal to its
    eligibility. GitHub is ungated: `sync_github` attributes to the attested repo owner, never
    the Oracle, so there's no inference to launder.

    `limit` caps posts per Substack/blog; GitHub has no post concept, so `github_min_stars`
    bounds it instead (its analog of the post cap). Two `since` knobs are NOT interchangeable:
    `web_since` bounds which posts to consider, `github_since` skips repos untouched since then.
    A shared `since` would mean different things on each adapter.

    `limit` is dispatch-bounded, not atom-bounded — it caps posts handed to the pool, so a
    paywalled or gate-rejected post still spends one of the N and `limit=20` can yield fewer
    atoms."""
    stype, url = source.get("source_type"), source.get("url")
    if stype in source_adapters.WEBSITE_ADAPTERS:
        decision, summ = source_adapters.gate_and_sync_website(
            conn, embedder, stype, url, author_name=author_name, since=web_since, limit=limit)
        if summ is None:
            return {"source_type": stype, "url": url,
                    "skipped": f"eligibility:{decision.decision}", "reason": decision.reason}
    elif stype == "github":
        owner = _github_owner(url)
        if not owner:
            return {"source_type": stype, "url": url, "skipped": "no_owner_in_url"}
        summ = ingest_github.sync_github(conn, embedder, handles=[owner],
                                         min_stars=github_min_stars, since=github_since)
    else:
        return {"source_type": stype, "url": url, "skipped": "no_adapter"}
    # Adapters report a hard stop by RETURNING an `error` summary, not raising, so the
    # try/except in `expand_oracle` never sees it. Without this branch a blocked archive walk
    # (zero atoms, nothing marked seen) would reach the user labelled `ingested`. See
    # ingest_common.classify_run.
    outcome = ingest_common.classify_run(summ)
    if outcome == ingest_common.RUN_BLOCKED:
        return {"source_type": stype, "url": url, "blocked": summ,
                "reason": str(summ.get("error"))}
    if outcome == ingest_common.RUN_ERROR:
        return {"source_type": stype, "url": url, "error": str(summ.get("error"))}
    return {"source_type": stype, "url": url, "ingested": summ}


def _review_item(source: dict) -> dict:
    """A needs-review source, with the reason it wasn't auto-trusted (from the trust verdict)."""
    reasons = (source.get("trust") or {}).get("reasons") or []
    return {"source_type": source.get("source_type"), "url": source.get("url"),
            "reason": reasons[0] if reasons else "no trust path from a confirmed root"}


def expand_oracle(conn, embedder, oracle: dict, *, limit: int = 0,
                  github_min_stars: int = 0, discover_fn=None,
                  x_since: datetime | None = None, web_since: datetime | None = None) -> dict:
    """Discover + verify + ingest ONE confirmed Oracle's footprint. Two halves: off-X sources
    (`discover_profile` -> trust-verify -> route to adapter; needs-review sources are returned,
    never ingested, for Pick #2), and the X root — the Oracle's own timeline, pulled directly.
    Seeds the cluster members as trust roots. Fail-safe per Oracle: a discovery crash is recorded
    (`discovery_error`) but does not skip the X pull, and any single-adapter error is isolated."""
    if discover_fn is None:
        from functools import partial
        from pipeline.ingestion.discover_profile import discover_profile
        # Footprint discovery skips the Semantic Scholar probe: papers aren't Oracle footprint
        # (`_ROUTABLE` discards a `scholar` source anyway), Scholar feeds no trust edges, and it's
        # a live API call per Oracle. Every other discover_profile caller keeps the probe on.
        discover_fn = partial(discover_profile, probe_scholar=False)
    cid, name = oracle["canonical_id"], oracle.get("name")
    # Root = the best CONFIRMED profile (X preferred, else Substack) — NOT assumed to be X.
    root = _root_profile(conn, oracle)
    handle = root["seed"] if root and root["seed_type"] == "x" else None   # for the report
    base = {"canonical_id": cid, "name": name, "handle": handle, "root": root,
            "ingested": [], "needs_review": [], "skipped": []}
    if not root:
        # No X or Substack member → nothing to root discovery on. Reported, never a crash.
        return {**base, "error": "no_rootable_profile"}

    # Discovery is DECOUPLED from the timeline pull: a discovery crash (network / API) records an
    # error but must NOT skip the X-footprint pull below — X is the Oracle's PRIMARY channel (for
    # an X or a Substack root whose X is found) and shouldn't be hostage to a flaky off-root probe.
    ingested, needs_review, skipped = [], [], []
    discovery_error = None
    try:
        profile = discover_fn(root["seed"], seed_type=root["seed_type"]) or {}
    except Exception as e:
        log(f"[expand] discovery FAILED for {root['seed_type']}:{root['seed']}: "
            f"{type(e).__name__}: {e}")
        profile, discovery_error = {}, f"discovery_failed: {type(e).__name__}: {e}"

    for s in profile.get("sources") or []:
        trusted = bool((s.get("trust") or {}).get("trusted"))
        if s.get("source_type") == "x":
            continue                            # the Oracle's X is pulled as a timeline (below), not routed
        if s.get("source_type") not in _ROUTABLE:
            skipped.append({"source_type": s.get("source_type"), "url": s.get("url"),
                            "skipped": "no_adapter", "trusted": trusted})
        elif trusted:
            try:
                ingested.append(_route_source(conn, embedder, s, author_name=name,
                                              limit=limit, github_min_stars=github_min_stars,
                                              web_since=web_since))
            except Exception as e:              # one bad source must not sink the Oracle's others
                log(f"[expand] ingest FAILED {s.get('source_type')} {s.get('url')}: {e}")
                ingested.append({"source_type": s.get("source_type"), "url": s.get("url"),
                                 "error": f"{type(e).__name__}: {e}"})
        else:
            needs_review.append(_review_item(s))

    # X-footprint pull: the Oracle's own X timeline, whenever findable (root handle if X-rooted,
    # else a discovered+trusted X). No eligibility gate — an X timeline is single-author by
    # construction, unlike a website. A non-X root with no findable X simply skips this; fail-safe.
    x_handle = _x_handle_to_pull(root, profile)
    if x_handle:
        try:
            x_summ = ingest_x_footprint.sync_x_footprint(conn, embedder, handle=x_handle,
                                                         author_name=name, since=x_since, limit=limit)
            # Same returned-not-raised contract as the off-X adapters — classify before labelling.
            x_outcome = ingest_common.classify_run(x_summ)
            entry = {"source_type": "x", "url": f"https://x.com/{x_handle}"}
            if x_outcome == ingest_common.RUN_BLOCKED:
                entry.update(blocked=x_summ, reason=str(x_summ.get("error")))
            elif x_outcome == ingest_common.RUN_ERROR:
                entry["error"] = str(x_summ.get("error"))
            else:
                entry["ingested"] = x_summ
            ingested.append(entry)
        except Exception as e:                  # a failed X pull must not sink the off-X sources
            log(f"[expand] x-footprint FAILED @{x_handle}: {type(e).__name__}: {e}")
            ingested.append({"source_type": "x", "url": f"https://x.com/{x_handle}",
                             "error": f"{type(e).__name__}: {e}"})

    log(f"[expand] @{handle}: {len(ingested)} ingested, {len(needs_review)} need review, "
        f"{len(skipped)} skipped" + (f"; discovery_error={discovery_error}" if discovery_error else ""))
    result = {**base, "ingested": ingested, "needs_review": needs_review, "skipped": skipped}
    if discovery_error:
        result["discovery_error"] = discovery_error
    return result


def expand_all(conn, embedder, *, limit: int = 0, github_min_stars: int = 0,
               discover_fn=None, x_since: datetime | None = None,
               web_since: datetime | None = None) -> list[dict]:
    """Expand EVERY confirmed Oracle, then re-resolve ONCE so the new substack:/blog:/github:
    rows fold into their canonical entity (the attested-link merge, same as run_ingest does)."""
    results = [expand_oracle(conn, embedder, o, limit=limit, github_min_stars=github_min_stars,
                             discover_fn=discover_fn, x_since=x_since, web_since=web_since)
               for o in oracles.confirmed_oracles(conn)]
    resolve.resolve_entities(conn)
    # Register every Oracle's sources for the background refresh loop, after the re-resolve so
    # newly folded substack:/blog:/github: members are visible. Mirrors `oracles._ingest_oracle`
    # (the live MCP path's seed) so a CLI-expanded store isn't left with an unregistered,
    # never-refreshed roster.
    try:
        from . import oracle_refresh_state
        oracle_refresh_state.seed_from_entities(conn)
    except Exception as e:
        log(f"[expand] refresh-registry seed skipped: {type(e).__name__}: {e}")
    return results


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stage-5 footprint expansion for confirmed Oracles.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap NEW posts ATTEMPTED per Substack/blog footprint (0 = full archive). "
                         "Not a cap on atoms: a paywalled or gate-rejected post spends one, so "
                         "fewer atoms than N is normal.")
    ap.add_argument("--x-lookback", choices=list(X_LOOKBACK_PRESETS), default="6mo",
                    help="How far back to pull each Oracle's X timeline (hard 2yr ceiling). "
                         "Default 6mo.")
    ap.add_argument("--substack-lookback", choices=list(WEB_LOOKBACK_PRESETS), default="all",
                    help="How far back to pull each Substack/blog archive (a durable corpus). "
                         "Default all.")
    ap.add_argument("--min-stars", type=int, default=5,
                    help="GitHub: skip repos under N stars (the star-floor bounds repo-boilerplate, "
                         "GitHub's analog of the per-post cap).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List confirmed Oracles + their resolved handles; no discovery, no ingest.")
    args = ap.parse_args(argv)

    conn = schema.connect()
    try:
        confirmed = oracles.confirmed_oracles(conn)
        if not confirmed:
            print("[expand] no confirmed Oracles — run the oracle SCREEN + confirm first.")
            return 1
        if args.dry_run:
            for o in confirmed:
                print(f"[dry-run] {o['name']}  root={_root_profile(conn, o)}  "
                      f"members={len(o['members'])}")
            return 0

        from .embed import get_kb_embedder
        embedder = get_kb_embedder()
        x_since = _since_from_days(X_LOOKBACK_PRESETS[args.x_lookback])
        web_since = _since_from_days(WEB_LOOKBACK_PRESETS[args.substack_lookback])
        print(f"[expand] embedder: model={embedder.model}  x-lookback={args.x_lookback}  "
              f"substack-lookback={args.substack_lookback}")
        for r in expand_all(conn, embedder, limit=args.limit, github_min_stars=args.min_stars,
                            x_since=x_since, web_since=web_since):
            print(f"\n[expand] {r['name']} (@{r.get('handle')})")
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
                continue
            for i in r["ingested"]:
                # The list is every ROUTED source; each entry says what actually happened to it.
                if i.get("blocked") is not None:
                    print(f"  BLOCKED   {i['source_type']}: {i.get('url')} — {i.get('reason')} "
                          f"(nothing written, retries next run)")
                elif i.get("error"):
                    print(f"  ERROR     {i['source_type']}: {i.get('url')} — {i['error']}")
                else:
                    print(f"  ingested  {i['source_type']}: {i.get('url')} -> "
                          f"{i.get('ingested', i.get('skipped'))}")
            for nr in r["needs_review"]:
                print(f"  REVIEW    {nr['source_type']}: {nr['url']}  — {nr['reason']}")
            for sk in r["skipped"]:
                print(f"  skipped   {sk['source_type']}: {sk.get('url')} ({sk['skipped']})")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_cli())
