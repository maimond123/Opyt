"""
pipeline/kb/expand.py — the shared Stage-5 footprint helpers: root an Oracle for discovery,
route ONE discovered source into its atom-KB adapter, and hold the lookback presets.

Two consumers, both of which own their own orchestration: `oracles._ingest_oracle` (the live
per-Oracle ingest engine, reached from `add_oracle` and `oracle(action='ingest')`) and
`oracle_refresh._dispatch` (the background top-up loop). This module owns no loop of its own.

Trust is per-source, not per-person: confirming the person verifies their X identity, but a
discovered URL (e.g. `elonmusk.substack.com`) still needs its own attested-link trust check.
The trust decision itself belongs to the caller; `_route_source` routes what it is handed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pipeline.timeparse import utc_now

from . import ingest_common, ingest_github, schema, source_adapters

# Lookback presets surfaced at onboarding. X is an ephemeral stream: short default, hard 2-year
# ceiling (enforced in sync_x_footprint). ⚠️ It was a PAID stream until the 2026-08-30 cutover, and
# that is no longer why the window is short — every X read now runs free on the user's own session.
# What still bounds it is wall-clock and one session's rate limits. Substack/blog are a durable
# corpus: default to the whole archive. A preset maps to days-before-now; None = 'all' (no lower
# bound), resolved to a `since` datetime once, at the entry point. Substack presets govern blog too.
X_LOOKBACK_PRESETS: dict[str, int] = {"6mo": 183, "1yr": 365, "2yr": 730}
WEB_LOOKBACK_PRESETS: dict[str, int | None] = {"1yr": 365, "2yr": 730, "5yr": 1825, "all": None}
# Papers follow the WEB shape, not the X one: a published corpus is durable, so a short default
# would truncate a body of work rather than trim a stream. No ceiling either — what bounds this
# pull is embed spend per work, and the abstract-only path makes that the only per-work cost.
# 10yr is the extra rung the web presets do not need: a researcher's career is longer than a
# blog's archive, and "the last decade" is a real answer where "the last 5 years" is a shrug.
SCHOLAR_LOOKBACK_PRESETS: dict[str, int | None] = {"2yr": 730, "5yr": 1825, "10yr": 3653,
                                                   "all": None}

_X_HANDLE = re.compile(r"(?:x|twitter)\.com/([^/?#]+)", re.I)


def _since_from_days(days: int | None) -> datetime | None:
    """A `since` datetime `days` before now, or None for 'all' (no lower bound)."""
    if not days:
        return None
    return utc_now() - timedelta(days=days)


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
    found + graduated via Rule 5). None when there is no findable/trusted X. This helper only
    identifies the handle; the ingest owner decides whether X is connected before it pulls or
    schedules that timeline (`_classify_url` now tags a discovered X link `"x"`)."""
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
                  github_since: datetime | None = None,
                  github_before: datetime | None = None) -> dict:
    """Route ONE discovered source (a `discover_profile` source dict) to its atom-KB adapter.
    Unadapted types are SKIPPED (recorded), never errors.

    WEBSITE sources (substack/blog) pass a single-author eligibility gate first: the adapters
    stamp `who_id`=the Oracle by inference, so an ungated multi-author site would launder other
    authors onto one trusted person. This is the single gated door for both auto-discovery and
    Pick #2 confirm; no `force` here — confirming a source's identity is orthogonal to its
    eligibility. GitHub is ungated: `sync_github` attributes to the attested repo owner, never
    the Oracle, so there's no inference to launder. `sync_github_source` splits the two url
    shapes: a repository link mints that one repo, an account link sweeps its archive.

    `limit` caps posts per Substack/blog; GitHub has no post concept, so `github_min_stars`
    bounds it instead (its analog of the post cap). Two `since` knobs are NOT interchangeable:
    `web_since` bounds which posts to consider, `github_since` skips repos untouched since then.
    A shared `since` would mean different things on each adapter. `github_before` is the other
    end of the same window — the coverage frontier a resume walks back from, which only GitHub
    can honor.

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
        summ = ingest_github.sync_github_source(conn, embedder, url, min_stars=github_min_stars,
                                                since=github_since, before=github_before)
        if summ is None:
            return {"source_type": stype, "url": url, "skipped": "no_owner_in_url"}
    else:
        return {"source_type": stype, "url": url, "skipped": "no_adapter"}
    # Adapters report a hard stop by RETURNING an `error` summary, not raising, so a caller's
    # try/except never sees it. Without this branch a blocked archive walk (zero atoms, nothing
    # marked seen) would reach the user labelled `ingested`. See ingest_common.classify_run.
    outcome = ingest_common.classify_run(summ)
    if outcome == ingest_common.RUN_BLOCKED:
        return {"source_type": stype, "url": url, "blocked": summ,
                "reason": str(summ.get("error"))}
    if outcome == ingest_common.RUN_ERROR:
        return {"source_type": stype, "url": url, "error": str(summ.get("error"))}
    return {"source_type": stype, "url": url, "ingested": summ}
