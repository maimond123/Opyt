"""
pipeline/kb/oracles.py — Stage-4 confirm: turn the user's picks into stored Oracles.

Two entry paths into ONE `oracles` row per confirmed canonical entity:
  • a RANKED pick — a `canonical_id` straight off the SCREEN; already resolved (Stage 3), so
    confirm just writes the row.
  • a RAW handle (free-form floor) — RESOLVE-AT-CONFIRM: mint the per-platform entity, fetch its
    `identity_links`, then recompute `canonical_id` so a pasted handle that cross-links an
    existing entity MERGES. A handle that can't resolve is reported (`unresolved`), never
    crashes confirm, and never writes a half-baked oracle.

`confirm` stops at the `oracles` table. `add_oracle` (below) chains past it: resolve reference →
confirm → ingest (footprint expand) → seed trust root → re-resolve, via the shared
`_ingest_oracle` engine (also used by `oracle(action='ingest')`).
"""

from __future__ import annotations

import re
from pipeline.timeparse import utc_now

from pipeline import llm_client

from . import derive, resolve, schema

# RAIL for this module's spend: named for the ingest activity, not the module (only
# `_ingest_oracle`'s embeddings + twitterapi calls spend). No dollar ceiling — bounded by consent
# instead, since every call is a user naming one person.
RAIL = "oracle_ingest"


# ── name resolution for a canonical ───────────────────────────────────────────

def _name_for(conn, canonical_id: str) -> str | None:
    """Best display name for a confirmed canonical: prefer a member with a non-null name, prefer
    the X row (its profile carries the richer name). Mirrors screen._best_name over the cluster."""
    from .screen import _best_name
    members = [{"entity_id": r["entity_id"], "name": r["name"], "profile": r["profile"]}
               for r in schema.entities_for_canonical(conn, canonical_id)]
    name, _handle = _best_name(members)
    return name


# ── resolve-at-confirm: a raw handle → a resolved canonical_id ─────────────────

def _fetch_x_identity(handle: str) -> dict | None:
    """One X handle → {user_id, display_name, bio, site, verified, followers, handle} via the free
    cookie-scrape `UserByScreenName`. The numeric `rest_id` is the load-bearing field — it is what
    `x:user:{id}` entities key on, and `_probe_twitter_bio` does not surface it. Returns None on any
    failure, which the caller reports as unresolved.

    This used to carry its OWN `requests.get` against twitterapi.io, which is why it was easy to
    miss when the provider was removed: nothing here imported the client module, so grepping for
    `x_render` did not find it. The t.co expansion it used to do by hand now lives in
    `fetch_user_profile`, beside the entities block it reads.

    The x.com self-link blanking below stays HERE. It is a consumer policy and the other consumer
    of the same call (`discover_profile._probe_twitter_bio`) disagrees — it skips the source
    instead — so deciding it in the transport would silently overrule one of them."""
    from pipeline.ingestion import x_graphql_core as core
    try:
        cookies = core.read_x_cookies()
        headers = core.auth_headers(cookies, f"https://x.com/{handle.lstrip('@')}")
        p = core.fetch_user_profile(cookies, headers, handle)
    except Exception:
        return None
    if not p:
        return None
    site = p["website"]
    return {
        "user_id": p["user_id"],
        "display_name": p["display_name"],
        "bio": p["bio"],
        "site": "" if ("twitter.com" in site or "x.com" in site) else site,
        "verified": p["verified"],
        "followers": p["followers"],
        "handle": p["handle"],
    }


def _resolve_handle(conn, raw: str) -> str | None:
    """A pasted handle/URL → a resolved `canonical_id`, minting + resolving the entity along the
    way. X handle → twitterapi user-info; a substack.com (or any home) URL → a substack entity
    keyed on its subdomain/host. Returns None when unresolvable (fail-safe). Runs the full Stage-3
    resolution after minting so the new entity MERGES into an existing cluster it cross-links."""
    raw = (raw or "").strip()
    if not raw:
        return None

    if raw.startswith("http") or "substack.com" in raw:
        url = raw if raw.startswith("http") else f"https://{raw}"
        # A substack.com URL keys on the Substack subdomain/host; any OTHER http… home is a
        # generic blog and keys on `blog:{host}` — the old code minted `substack:{host}` for
        # everything, which sent a personal site into the Substack cluster + adapter.
        eid = derive.substack_entity_id(None, url) if "substack.com" in url \
            else derive.blog_entity_id(url)
        schema.upsert_entity(conn, eid, identity_links=[url])
    else:
        ident = _fetch_x_identity(raw)
        if not ident:
            return None
        eid = f"x:user:{ident['user_id']}"
        prof = {k: ident[k] for k in ("bio", "verified", "followers", "handle") if ident.get(k)}
        schema.upsert_entity(conn, eid, name=ident.get("display_name"),
                             identity_links=[ident["site"]] if ident.get("site") else None,
                             profile=prof or None)

    resolve.resolve_entities(conn)                 # recompute canonical (merge if it cross-links)
    row = schema.get_entity(conn, eid)
    return (row["canonical_id"] if row and row["canonical_id"] else eid)


# ── confirm ────────────────────────────────────────────────────────────────────

def confirm(conn, canonical_ids: list[str] | None = None,
            add_handles: list[str] | None = None) -> dict:
    """Commit the user's Oracle picks. `canonical_ids` are ranked-list picks (resolved already);
    `add_handles` are free-form floor entries (resolved-at-confirm). Idempotent — re-confirming a
    canonical refreshes it without re-adding. Returns {confirmed, unresolved, unknown,
    total_oracles}:
      • confirmed  — [{canonical_id, name, source, [handle]}] written to `oracles`.
      • unresolved — raw handles a fetch couldn't resolve (report to the user; nothing written).
      • unknown    — canonical_ids with no entity row (a bad/hallucinated id; skipped, not written).
    """
    confirmed, unresolved, unknown = [], [], []

    for cid in (canonical_ids or []):
        if schema.get_entity(conn, cid) is None:
            unknown.append(cid)                    # guard: never mint an oracle for a phantom id
            continue
        name = _name_for(conn, cid)
        schema.upsert_oracle(conn, cid, name=name, source="screen")
        confirmed.append({"canonical_id": cid, "name": name, "source": "screen"})

    for raw in (add_handles or []):
        cid = _resolve_handle(conn, raw)
        if not cid:
            unresolved.append(raw)
            continue
        name = _name_for(conn, cid)
        schema.upsert_oracle(conn, cid, name=name, source="freeform")
        confirmed.append({"canonical_id": cid, "name": name, "handle": raw, "source": "freeform"})

    return {
        "confirmed": confirmed,
        "unresolved": unresolved,
        "unknown": unknown,
        "total_oracles": len(schema.list_oracles(conn)),
    }


def confirmed_oracles(conn) -> list[dict]:
    """Every confirmed Oracle + the per-platform footprints Stage 5 will expand — each member's
    identity_links, from the merged cluster. The read side of the Stage-4→5 handoff."""
    out = []
    for o in schema.list_oracles(conn):
        # The stored canonical_id can be stale after a footprint resolve shifted the cluster head;
        # re-anchor to the current head so members (and the reported id) are correct.
        cid = schema.current_canonical(conn, o["canonical_id"])
        members = []
        for r in schema.entities_for_canonical(conn, cid):
            links = r["identity_links"]
            members.append({"entity_id": r["entity_id"], "name": r["name"],
                            "identity_links": links})
        out.append({"canonical_id": cid, "name": o["name"], "source": o["source"],
                    "confirmed_at": o["confirmed_at"], "members": members})
    return out


# ── add_oracle: the single user-facing "add a person" on the atom rail ──────────
#
# "Add a person" = admit an Oracle + expand their verified footprint into atoms: resolve
# reference → confirm → ingest (footprint expand) → seed trust root → re-resolve, wired here as
# ONE two-phase entry point.

_CANONICAL_PREFIX = re.compile(
    r"^(x:user:|substack:|blog:|github:|org:|scholar:|paper-authors:)", re.I)


def _classify_reference(reference: str) -> str:
    """What KIND of reference is this: 'canonical' (an existing per-cluster id → Mode C),
    'url' (a Substack/blog/site home), or 'handle' (a bare X @handle)? Mirrors `_resolve_handle`'s
    URL test (http… or a substack.com string). Anything else is a handle the HOST resolved from a
    name — a bare blog DOMAIN (no scheme) reads as a handle, so blog refs must arrive as full URLs."""
    r = (reference or "").strip()
    if _CANONICAL_PREFIX.match(r):
        return "canonical"
    if r.startswith("http") or "substack.com" in r:
        return "url"
    return "handle"


def _url_entity_id(url: str) -> str:
    """A home URL → its per-platform entity id: `substack:{host}` for substack.com, else
    `blog:{host}`. Same split as the `_resolve_handle` blog branch — one rule, two callers."""
    u = url if url.startswith("http") else f"https://{url}"
    return derive.substack_entity_id(None, u) if "substack.com" in u else derive.blog_entity_id(u)


def _match_local_roster(conn, reference: str) -> dict | None:
    """Network-free dedup / Mode-C promote: does `reference` already map to a known entity? Returns
    {canonical_id, name, members} when it does (so we reflect the user's existing curation signals
    and never mint a duplicate Oracle), else None. STRUCTURAL match only — an X handle via the
    stored `profile.handle`, a URL via its derived `substack:`/`blog:` id, a canonical_id directly.
    NEVER fuzzy name (fuzzy resolution is deferred by design)."""
    kind = _classify_reference(reference)
    ref = reference.strip()
    if kind == "canonical":
        eid = ref
    elif kind == "url":
        eid = _url_entity_id(ref)
    else:                                   # handle — match the stored profile.handle, no network
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE entity_id LIKE 'x:user:%' "
            "AND profile IS NOT NULL "
            "AND lower(json_extract(profile, '$.handle')) = lower(?) LIMIT 1",
            (ref.lstrip("@"),),
        ).fetchone()
        eid = row[0] if row else None
    if not eid or schema.get_entity(conn, eid) is None:
        return None
    cid = schema.current_canonical(conn, eid)
    members = [{"entity_id": r["entity_id"], "name": r["name"],
                "identity_links": r["identity_links"]}
               for r in schema.entities_for_canonical(conn, cid)]
    return {"canonical_id": cid, "name": _name_for(conn, cid), "members": members}


# ── the TWO lookback windows ─────────────────────────────────────────────────────
#
# X (an ephemeral stream: free to fetch, metered per post to ingest) and the web archive (a
# durable corpus) use SEPARATE selectors — a single symmetric knob can only be wrong in one
# direction: over-pull X, or truncate the archive. Resolve each preset against its own dict; build
# the report from the datetimes that actually ran, not the requested preset. See


# The one X selector that is NOT a fixed span: it answers "since I last pulled THIS person",
# a different date per Oracle, rather than a uniform lookback. See
X_SINCE_LAST = "since_last"


def _x_since(preset: str | None):
    """An X preset → its `since`, or None for "the adapter's own ~6-month default".

    `since_last` is NOT resolvable here and raises rather than returning None: it needs a
    specific Oracle (see `x_since_last`), and None would silently fall through to the adapter's
    183-day default — the most expensive window standing in for the cheapest request."""
    from . import expand                                  # lazy: expand imports oracles at load
    if not preset:
        return None
    if preset == X_SINCE_LAST:
        raise ValueError(f"{X_SINCE_LAST!r} resolves per-Oracle — use x_since_last(conn, cid)")
    return expand._since_from_days(expand.X_LOOKBACK_PRESETS.get(preset))


def x_since_last(conn, canonical_id: str):
    """This Oracle's since-last-pull X window — the same window the automatic refresh loop uses.

    Returns `last_pulled_at - OVERLAP_HOURS` (falling back to `cursor_ts`, the newest atom held),
    or None when neither exists. Callers must treat None as a refusal, never a default — see
    `_x_since`. Seeds the registry first, since a freshly-confirmed Oracle has no `oracle_sources`
    row yet."""
    # Lazy import: `oracle_refresh` does not import this module, so this direction cannot cycle.
    from . import oracle_refresh, oracle_refresh_state as st

    st.seed_from_entities(conn, canonical_ids=[canonical_id])
    windows = [oracle_refresh.since_for(r)
               for r in st.list_sources(conn, canonical_ids=[canonical_id])
               if r.source_type == "x"]
    # Most-recently-pulled wins if a person somehow carries two X keys. Narrower is the safe
    # direction: dedup absorbs an over-ask, but nothing recovers posts an under-ask never fetched.
    return max((w for w in windows if w is not None), default=None)


def _web_since(preset: str | None):
    """A web preset → its `since`. `'all'` (and unknown) → None = the full archive, no bound."""
    from . import expand
    if not preset:
        return None
    return expand._since_from_days(expand.WEB_LOOKBACK_PRESETS.get(preset))


def _effective_x_since(x_since):
    """What the X adapter will ACTUALLY use: its default when unset, floored at the hard 2-year
    ceiling. Called through `ingest_x_footprint._resolve_since` rather than re-derived, so the
    report cannot drift from the clamp — a report that can disagree with the code IS the bug."""
    from .ingest_x_footprint import _resolve_since
    return _resolve_since(x_since, utc_now())


def _lookback_report(x_since, web_since) -> dict:
    """The human-facing windows, derived from the resolved datetimes rather than re-read off the
    preset strings — so what the user is told is what ran, including the X clamp they didn't ask
    for. This is the surface the cost-consent invariant rests on."""
    eff = _effective_x_since(x_since)
    clamped = x_since is not None and eff > x_since
    return {
        "x": f"since {eff:%Y-%m-%d}" + (" (CLAMPED to the 2-year ceiling)" if clamped else
                                        "" if x_since is not None else " (6-month default)"),
        "x_since": eff.isoformat(),
        "web": f"since {web_since:%Y-%m-%d}" if web_since else "full archive",
        "web_since": web_since.isoformat() if web_since else None,
        "note": "X is an ephemeral stream, capped at 2 years; the web archive has no cap. The "
                "two windows are chosen separately — a short X window does not truncate the "
                "archive.",
    }


def _oracle_for(conn, canonical_id: str) -> dict:
    """The confirmed-oracle record (with cluster members) for one canonical_id — re-anchored to
    the current head so the shared ingest engine gets the right members even after a resolve shift.
    Falls back to a minimal cluster read if the oracles row isn't found (shouldn't happen post-confirm)."""
    head = schema.current_canonical(conn, canonical_id)
    for o in confirmed_oracles(conn):
        if o["canonical_id"] == head:
            return o
    members = [{"entity_id": r["entity_id"], "name": r["name"],
                "identity_links": r["identity_links"]}
               for r in schema.entities_for_canonical(conn, head)]
    return {"canonical_id": head, "name": _name_for(conn, head), "source": "screen",
            "members": members}


@llm_client.rail(RAIL)
def _ingest_oracle(conn, embedder, oracle: dict, *, force: bool = False,
                   x_since=None, web_since=None, limit: int = 0,
                   extra_source_urls: list[str] | None = None) -> dict:
    """The ONE per-Oracle ingest engine, shared by `add_oracle` and `oracle(action='ingest')`.

    The rail scope lives here rather than on either caller, since this is the one function every
    path goes through.

    Runs the footprint path — `discover_profile` → trust-filter → `onboard_footprint` → the
    Oracle's own X timeline — and seeds the cluster as a tier-1.0 trust root, unconditionally and
    first: a confirmed Oracle is a root by the user's vouch regardless of whether discovery finds
    a footprint. Fail-safe: `onboard_footprint` isolates each source; a failed X pull is recorded,
    never aborts.

    `limit` means different things per adapter: substack/blog caps posts DISPATCHED, X caps
    atoms submitted, GitHub ignores it entirely (defaults to `min_stars=0`, ~2000 max). See

    Two SEPARATE windows: `x_since` bounds the X stream (clamped to a hard 2-year ceiling);
    `web_since` bounds the Substack/blog archive (no ceiling). See `_lookback_report`.

    The returned summary carries `atoms_added` vs `dispatched` so a caller can see when they
    diverge, plus `blocked` (a host stopped us, nothing written, retry) distinct from `ingested`."""
    from pipeline.ingestion.discover_profile import discover_profile

    from . import ingest_common, ingest_x_footprint
    from . import onboard_footprint as of
    from .expand import _root_profile, _x_handle_to_pull

    oracle_id = oracle["canonical_id"]
    # Time discovery and the X pull here (they're owned by this function); merge into the
    # router's own totals below so one Oracle ingest reports one per-stage table.
    timer = ingest_common.StageTimer()

    root = _root_profile(conn, oracle)
    if not root:
        return {"oracle_id": oracle_id, "name": oracle.get("name"),
                "error": "no rootable profile (no X, Substack, or blog member) — cannot discover"}

    # Discovery on this path spends NOTHING beyond the twitterapi profile pull — the cold-start
    # identity anchor, which used to be the one paid step here, was deleted 2026-08-28
    # (docs/plans/2026-08-28-delete-the-cold-start-identity-anchor.md).
    #
    # `extra_source_urls` is Probe 5's return leg: URLs the host found by web search on a previous
    # call, handed back so the trust graph judges them rather than trusting the host's say-so.
    # Empty on a first call; populated when the host acts on `followup`.
    #
    # `reverify=force`: the trust cache is keyed on the seed's display name + declared URLs only,
    # so everything else that can change a discovery outcome — a Substack or GitHub the person
    # created later, a landing page that now links their blog, a fix to our own trust rules —
    # leaves the key identical and replays the stale verdict until the 30-day TTL. `force` is the
    # only way in before that.
    with timer.stage("discover_profile"):
        profile = discover_profile(root["seed"], seed_type=root["seed_type"],
                                   reverify=force,
                                   extra_source_urls=extra_source_urls)
    srcs = profile.get("sources", [])
    # Did we actually look this time, or replay a cached profile? `add_oracle` gates its open-web
    # `followup` on this — re-searching an unchanged person can only return what's already here.
    fresh_discovery = not profile.get("from_cache")
    # Ingest trusted personal profiles + org-shaped affiliations; the rest go to the review step
    # (onboard_footprint re-checks the same boundary + gates every website through eligibility).
    eligible = [s for s in srcs
                if (s.get("trust") or {}).get("trusted")
                or (s.get("metadata") or {}).get("shape") == "org"]
    result = of.onboard_footprint(conn, embedder, oracle_id, eligible,
                                  author_name=oracle.get("name"), force=force,
                                  since=web_since, limit=limit)

    # The Oracle's OWN X timeline — the root handle when X-rooted, or a discovered+trusted X for a
    # Substack/blog-rooted Oracle (their richest channel, pulled regardless of root platform).
    xh = _x_handle_to_pull(root, profile)
    if xh:
        try:
            with timer.stage("x"):
                xs = ingest_x_footprint.sync_x_footprint(
                    conn, embedder, handle=xh, author_name=oracle.get("name"),
                    since=x_since, limit=limit)
            # The adapter reports a hard stop by returning an `error` summary, so this except
            # clause only ever sees the raising half of the contract.
            outcome = ingest_common.classify_run(xs)
            rec = {"url": f"https://x.com/{xh}", "type": "x", "action": outcome, "detail": str(xs)}
            stats = ingest_common.run_stats(xs)
            if stats:
                rec["stats"] = stats
            result.setdefault("results", []).append(rec)
            for k, agg in (("ingested", outcome == "ingested"), ("blocked", outcome == "blocked"),
                           ("errors", outcome == "error")):
                if agg:
                    result[k] = result.get(k, 0) + 1
            # Only `added` folds into the cross-source total. `dispatched` has no common unit
            # across adapters (substack/blog counts posts handed to the pool; X counts
            # referenced-link dispatches) — the X figure stays on the X record's own `stats`.
            result["atoms_added"] = result.get("atoms_added", 0) + stats.get("added", 0)
        except Exception as e:                       # a failed X pull never aborts the off-X ingest
            result.setdefault("results", []).append(
                {"url": f"https://x.com/{xh}", "type": "x", "action": "error",
                 "detail": f"{type(e).__name__}: {e}"})
            result["errors"] = result.get("errors", 0) + 1
    result["discovery_ran_fresh"] = fresh_discovery
    # Router stages and ours have disjoint keys, so this is a union, not an override.
    result["stage_seconds"] = {**result.get("stage_seconds", {}), **timer.totals}
    # The windows this run actually used, so the caller reports what ran, not what was asked.
    result["lookback"] = _lookback_report(x_since, web_since)
    # `ingest_from` is the LATER of the two effective windows (the instant both pulls are
    # complete), so a re-ingest can only under-claim coverage and re-pull, never skip a window it
    # never fetched.
    eff_x = _effective_x_since(x_since)
    covered_from = max(eff_x, web_since) if web_since else eff_x
    schema.set_oracle_window(conn, oracle_id, covered_from, utc_now())
    # Seed the refresh registry AFTER the coverage window is written — it reads `ingest_to` as
    # `last_pulled_at`, so a freshly onboarded Oracle isn't immediately re-pulled. Fail-safe:
    # registry trouble must not fail an ingest that already wrote atoms.
    try:
        from . import oracle_refresh_state
        oracle_refresh_state.seed_from_entities(conn, canonical_ids=[oracle_id])
    except Exception as e:
        from pipeline.ingestion.utils import log
        log(f"[oracles] refresh-registry seed skipped for {oracle_id}: {type(e).__name__}: {e}")
    return result


def _preview(conn, reference: str, match: dict | None,
             x_lookback: str | None, web_lookback: str | None) -> dict:
    """confirm=False → RESOLVE-ONLY preview: who is this + what confirm=True would do. NO writes.
    An unresolvable reference returns `unresolved` (nothing to confirm) — that IS the guard that
    makes silently ingesting a hallucinated handle impossible."""
    base = {"confirm_required": True, "reference": reference,
            "lookback": _lookback_report(_x_since(x_lookback), _web_since(web_lookback)),
            "on_confirm": "confirm=True confirms this Oracle, discovers + ingests their footprint "
                          "(including their X timeline), and seeds them as a trust root. The X "
                          "pull itself is free (your own browser session); embedding and the "
                          "content gate are metered."}

    if match:                                        # already in the roster (dedup / Mode C)
        already = schema.is_oracle(conn, match["canonical_id"])
        return {**base, "mode": "existing",
                "resolved": {"canonical_id": match["canonical_id"], "name": match.get("name"),
                             "members": [m["entity_id"] for m in match.get("members", [])],
                             "already_oracle": already},
                "note": ("Already one of your Oracles — confirm=True REFRESHES them "
                         "(re-pulls; atoms dedup, so nothing duplicates)." if already else
                         "Already in your roster from curation — confirm=True promotes them to an "
                         "Oracle and ingests their footprint.")}

    kind = _classify_reference(reference)
    if kind == "canonical":
        return {"error": f"no entity for canonical_id {reference!r} — it may be stale or "
                         "hallucinated. Nothing was written."}
    if kind == "url":
        eid = _url_entity_id(reference)
        return {**base, "mode": "new",
                "resolved": {"reference": reference, "root_entity": eid,
                             "platform": "substack" if eid.startswith("substack:") else "blog"}}
    # X handle — a cheap, read-only identity fetch for recognition (name / bio / followers).
    ident = _fetch_x_identity(reference)
    if not ident:
        return {"unresolved": reference,
                "note": "Could not resolve this X handle — check the spelling. Nothing was "
                        "written, so there is nothing to confirm."}
    return {**base, "mode": "new",
            "resolved": {"handle": ident.get("handle"), "name": ident.get("display_name"),
                         "bio": ident.get("bio"), "followers": ident.get("followers"),
                         "site": ident.get("site"), "root_entity": f"x:user:{ident['user_id']}"}}


def _open_web_followup(reference: str, name: str | None) -> dict:
    """Probe 5's push leg — ask the HOST to run the open-web search this rail cannot.

    Probes 1-4 are deterministic (bio, Substack convention, GitHub, Semantic Scholar). The fuzzy
    "what else does this person publish" step needs a web search the host already has; the host
    does the finding, `_compute_trust` does the judging.

    `feed_back_via` must name a real, currently-live parameter. Re-entering on the SAME `reference` with
    `extra_source_urls` cannot fragment an identity: the URLs join that person's candidate set and
    are judged there, unlike a bare `add_oracle(<found-url>)` which would mint a standalone entity.
    `test_the_followup_names_a_parameter_that_actually_exists` pins the name against drift.
    """
    who = name or reference
    return {
        "instruction": (
            f"Web-search for content platforms belonging to {who} ({reference}) that the "
            "deterministic probes miss: a personal blog or website, a Substack, a YouTube "
            "channel, and podcasts they host or regularly appear on. Exclude X/Twitter, GitHub "
            "and Semantic Scholar — those are already covered. Return home/channel pages, not "
            "individual posts or videos. You do NOT need to verify ownership; the trust graph "
            "re-checks every URL and will reject any that cannot be corroborated."
        ),
        "feed_back_via": f"add_oracle({reference!r}, confirm=True, extra_source_urls=[...])",
        # Said explicitly: a dropped URL is invisible, while a rejected one shows up in
        # needs_review — so the trust graph should judge, not the host filter beforehand.
        "note": "Send everything plausible. Unverifiable URLs are rejected, not trusted.",
    }


def add_oracle(conn, embedder, reference: str, *, confirm: bool = False,
               x_lookback: str | None = None, web_lookback: str | None = None,
               extra_source_urls: list[str] | None = None, force: bool = False) -> dict:
    """Atom-native "add a person": admit an Oracle + expand their verified footprint into atoms.

    `reference` is polymorphic — an X @handle, a Substack/blog/site URL, or a canonical_id (Mode C:
    promote a below-the-cut SCREEN pick). Two-phase, gated by `confirm`:
      • confirm=False (default) → RESOLVE-ONLY preview: verify the reference resolves + show who it
        is + which platforms will expand. Writes NOTHING (a hallucinated handle has nothing to
        confirm). Cheap — no discovery.
      • confirm=True → the full chain: resolve → confirm (write the oracles row) → ingest
        (discover → trust-filter → footprint atoms + the Oracle's X timeline) → seed trust root →
        re-resolve. WRITES.

    `x_lookback` and `web_lookback` are SEPARATE knobs on purpose — one bounds an ephemeral
    stream and the other a durable archive, and a single value can only be right for one of them.
    Both default to None = each adapter's own default (X ~6 months, web the full archive).

    Name resolution ('find Karpathy') is the HOST's job — it turns a name into a @handle/URL and
    calls this; there is no name-search endpoint (fuzzy resolution deferred by design)."""
    reference = (reference or "").strip()
    if not reference:
        return {"error": "add_oracle needs a reference — an X @handle, a Substack/blog URL, or a "
                         "canonical_id from oracle(action='screen')."}

    match = _match_local_roster(conn, reference)         # network-free dedup / Mode C

    if not confirm:                                      # ── Phase 1: preview (no writes) ──
        return _preview(conn, reference, match, x_lookback, web_lookback)

    # ── Phase 2: the chain ──────────────────────────────────────────────────────
    already = bool(match) and schema.is_oracle(conn, match["canonical_id"])

    # 1+2. RESOLVE → canonical_id, then CONFIRM (write the oracles row).
    if match:
        cid = match["canonical_id"]
        schema.upsert_oracle(conn, cid, name=match.get("name"), source="screen")
        source = "screen"
    else:
        cid = _resolve_handle(conn, reference)           # mints + resolves + the freeform vouch
        if not cid:
            return {"error": f"could not resolve {reference!r} — check the handle/URL. "
                             "Nothing was written.", "unresolved": [reference]}
        schema.upsert_oracle(conn, cid, name=_name_for(conn, cid), source="freeform")
        source = "freeform"

    # 3+4. Ingest + seed (the shared engine). Two windows, resolved against their OWN presets.
    oracle = _oracle_for(conn, cid)
    web_since = _web_since(web_lookback)
    if x_lookback == X_SINCE_LAST:
        # One Oracle here, so this resolves directly — but it refuses on a missing window for the
        # same reason the batch path does: None would mean the adapter's 183-day default, turning
        # "top them up" into a full re-onboarding. A first pull is not a top-up.
        x_since = x_since_last(conn, cid)
        if x_since is None:
            return {"error": f"{X_SINCE_LAST!r} needs a previous pull to measure from, and "
                             f"{oracle.get('name') or cid} has none. This is their first X pull — "
                             f"pass an explicit x_lookback ('6mo'/'1yr'/'2yr')."}
    else:
        x_since = _x_since(x_lookback)
    ingest = _ingest_oracle(conn, embedder, oracle, x_since=x_since, web_since=web_since,
                            extra_source_urls=extra_source_urls, force=force)

    # 5. RE-RESOLVE — fold the new substack:/blog:/github: rows into the canonical cluster.
    resolve.resolve_entities(conn)

    out = {
        "added": {"canonical_id": schema.current_canonical(conn, cid),
                  "name": oracle.get("name"), "source": source, "was_already_oracle": already},
        # What actually ran, derived from the resolved datetimes — including the X clamp nobody
        # asked for. Tell the user this; it is the consent surface for the pull.
        "lookback": _lookback_report(x_since, web_since),
        "ingest": ingest,
    }
    # Probe 5's push leg, asked only when it can pay for itself. Two gates:
    #  • `extra_source_urls` — the host just answered; asking again would loop forever.
    #  • `discovery_ran_fresh` — a cache hit means nothing changed since last look, so a search
    #    can only return what's already in the result (also covers the no-rootable-profile early
    #    return, where discovery never ran).
    if not extra_source_urls and ingest.get("discovery_ran_fresh"):
        out["followup"] = _open_web_followup(reference, oracle.get("name"))
    return out
