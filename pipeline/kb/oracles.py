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
confirm → ingest → re-resolve, via the shared `_ingest_oracle` engine (also used by
`oracle(action='ingest')`). The ingest step has two independent halves — a paper corpus and a
discovered footprint — and each runs only when the cluster carries what it needs; see
`_ingest_oracle`.
"""

from __future__ import annotations

import re
import time
from pipeline.timeparse import utc_now

from . import derive, resolve, schema


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
        session = core.x_session(f"https://x.com/{handle.lstrip('@')}")
        p = core.fetch_user_profile(session, session, handle)
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
    way. X handle → `_fetch_x_identity`; an academic identity URL → a scholar entity; a research
    venue → an `openalex:S…` entity; a substack.com (or any other home) URL → a substack or blog
    entity keyed on its subdomain/host. Returns None when unresolvable (fail-safe). Runs the full
    Stage-3 resolution after minting so the new entity MERGES into an existing cluster it
    cross-links.

    ORDER MATTERS between the OpenAlex check and the blog fallthrough, and only in that direction:
    a venue that is not recognised is still a usable blog, while a blog wrongly taken for a venue
    would pull someone else's corpus. `_openalex_root` carries the uniqueness rule that makes the
    first case the only one that happens."""
    raw = (raw or "").strip()
    if not raw:
        return None

    scholar = _scholar_root(raw) if raw.startswith("http") else None
    venue = _openalex_root(raw) if (scholar is None and raw.startswith("http")) else None
    if scholar:
        prefix, identifier = scholar
        if not _SCHOLAR_ROOTS[prefix]:
            # A registry OPYT cannot pull, REFUSED here and not only at `_unsupported_root` —
            # `confirm(add_handles=[…])` reaches this function without passing that gate. Until
            # 2026-09-09 an unpullable prefix fell through to the ORCID lookup, so a Semantic
            # Scholar id was sent to `/authors/orcid:2354728` and the refusal was a side effect of
            # that lookup failing. A refusal that depends on a network call failing is not one.
            return None
        eid = _mint_orcid_entity(conn, identifier)
        if not eid:
            return None
    elif venue:
        # A research venue mints the SAME `openalex:` entity shape a researcher does, with a
        # source id instead of an author id — so it inherits the scholar pull, DOI dedup and the
        # topic filter with no new adapter. Before this it became `blog:{host}` and a preprint
        # repository was scraped as generic articles.
        eid = f"openalex:{venue['openalex_id']}"
        schema.upsert_entity(conn, eid, name=venue.get("name"), identity_links=[raw])
    elif raw.startswith("http") or "substack.com" in raw:
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


def _mint_orcid_entity(conn, orcid: str) -> str | None:
    """An ORCID → the `openalex:` entity that has a works feed, or None when unresolvable.

    The ORCID is resolved to an OpenAlex author id BEFORE minting, because only that id has a works
    feed: `openalex:{id}` is a pullable Oracle and `orcid:{id}` would be an Oracle whose every
    refresh found nothing. The ORCID is still stored as the entity's identity link, which is what
    merges it with a `scholar:` entity for the same person.

    ORCID is the only prefix `_SCHOLAR_ROOTS` marks pullable, so this takes the identifier and not
    a `(prefix, id)` pair. A second pullable registry brings the parameter back with it.
    """
    from pipeline.ingestion.utils import log

    from .frontier_sources import OpenAlexWorksAdapter

    # One free lookup, and a failure REFUSES rather than falling back to a `blog:` entity —
    # falling back is what this whole change exists to stop.
    try:
        rec = OpenAlexWorksAdapter().author_by_orcid(orcid)
    except Exception as e:
        log(f"[oracles] ORCID {orcid} lookup failed: {type(e).__name__}: {e}")
        rec = None
    author_id = str((rec or {}).get("id") or "").rsplit("/", 1)[-1]
    if not author_id:
        return None
    eid = f"openalex:{author_id}"
    schema.upsert_entity(conn, eid, name=(rec or {}).get("display_name"),
                         identity_links=[f"https://orcid.org/{orcid}"])
    return eid


# ── confirm ────────────────────────────────────────────────────────────────────

def confirm(conn, canonical_ids: list[str] | None = None,
            add_handles: list[str] | None = None) -> dict:
    """Commit the user's Oracle picks. `canonical_ids` are ranked-list picks (resolved already);
    `add_handles` are free-form floor entries (resolved-at-confirm). Idempotent — re-confirming a
    canonical refreshes it without re-adding. Returns {confirmed, unresolved, unknown, refused,
    total_oracles}:
      • confirmed  — [{canonical_id, name, source, [handle]}] written to `oracles`.
      • unresolved — raw handles a fetch couldn't resolve (report to the user; nothing written).
      • unknown    — canonical_ids with no entity row (a bad/hallucinated id; skipped, not written).
      • refused    — resolved fine, declined on purpose: a multi-author site cannot be an Oracle
                     root (`_multi_author_refusal`). Carries a `reason` written for the user, with
                     the two alternatives in it. Nothing written. DISTINCT from `unresolved`,
                     which means we could not find the thing at all — the two need different
                     things said back, so they are different keys.
    """
    from . import ingest_scholar_footprint as isf

    confirmed, unresolved, unknown, refused = [], [], [], []

    for cid in (canonical_ids or []):
        if schema.get_entity(conn, cid) is None:
            unknown.append(cid)                    # guard: never mint an oracle for a phantom id
            continue
        name = _name_for(conn, cid)
        # Both loops, one check: the invariant is about the `oracles` ROW, not about which door
        # wrote it. A screened pick can be a multi-author publication too — the Substack
        # collectors mint `substack:` entities for whatever the user follows.
        why = _multi_author_refusal(conn, cid, name or cid)
        if why:
            refused.append({"canonical_id": cid, "name": name, "reason": why})
            continue
        schema.upsert_oracle(conn, cid, name=name, source="screen")
        confirmed.append({"canonical_id": cid, "name": name, "source": "screen"})

    for raw in (add_handles or []):
        cid = _resolve_handle(conn, raw)
        if not cid:
            unresolved.append(raw)
            continue
        name = _name_for(conn, cid)
        why = _multi_author_refusal(conn, cid, name or raw)
        if why:
            refused.append({"handle": raw, "canonical_id": cid, "name": name, "reason": why})
            continue
        schema.upsert_oracle(conn, cid, name=name, source="freeform")
        entry = {"canonical_id": cid, "name": name, "handle": raw, "source": "freeform"}
        # A VENUE IS THE ONE ROOT WHOSE DEFAULT IS WRONG. Every other root pulls in full and that
        # is the ruling — 928 works from a prolific researcher is fine. A venue is 63,565
        # (ChemRxiv, measured 2026-09-08) against `MAX_WORKS_PER_PULL` of 2,000, so "no filter"
        # does not mean "everything", it means the newest 3% with no sign that anything was cut.
        #
        # `add_oracle`'s preview already says this, loudly, with the real count. This is the OTHER
        # door: `oracle(action='confirm', add_handles=[…])` mints the same entity through
        # `_resolve_handle` and returned nothing to distinguish a preprint repository from a
        # personal blog — and it is the door `onboard`'s own `blog` root points a new user at.
        if _is_venue_id(cid) and not _has_topic_filter(conn, cid):
            entry["needs_topics"] = True
            entry["note"] = (
                f"{name or raw} is a research VENUE, not a person — it publishes far more than "
                f"one pull can hold, so ingesting it now would store the most recent "
                f"{isf.MAX_WORKS_PER_PULL} papers and silently drop the rest. Do NOT call "
                f"`oracle(action='ingest')` yet. Call `add_oracle('{raw}')` first: its preview "
                f"reports the real total and the subject list, and `scholar_topics` narrows it to "
                f"what the user actually reads.")
        confirmed.append(entry)

    return {
        "confirmed": confirmed,
        "unresolved": unresolved,
        "unknown": unknown,
        "refused": refused,
        "total_oracles": len(schema.list_oracles(conn)),
        # Only when a row was actually written. A call that refused everything minted nothing, so
        # it created no work and has nothing to schedule.
        "refresh_queued": bool(confirmed) and activate_refresh(),
    }


def activate_refresh() -> bool:
    """Make the recurring refresh due now, because an Oracle just became refreshable.

    ⚠️ THE PRODUCER LIVES AT THE MINT, NOT AT THE CONSENT ANSWER — the 2026-09-14 fix, and the
    defect it repairs is the one `tests/kb/test_rail_activation.py` was written for in the first
    place: "a rail that never runs looks exactly like a rail that runs and finds nothing."

    `onboard._apply_consent` was the ONLY caller of `request_now('oracle_refresh')`, gated on
    `_confirmed_oracles() > 0`. The consent question is asked BEFORE the roster is picked —
    `onboard` answers `next_tool='oracle'` — so on a first run that gate reads zero, the row is
    never written, and nothing else in the codebase ever writes it. The guard was not wrong; it
    was evaluated at the one moment it could never pass. Measured on a real session
    (2026-09-14): five Oracles confirmed, `oracle_refresh_consent` present, the resident worker
    installed and running, and no `oracle_refresh` row in `rail_jobs.db` at all. Every "it fills
    in on its own" the tools told the host to say was false for the life of that install — the
    owed 6-month windows, the deferred X timelines, the two-weekly blog re-check, all of it.

    Minting an Oracle is the event that CREATES the work, so it is the event that queues the
    rail. The precondition cannot go stale between the two because they are one moment. The
    consent-time call stays where it is: it is the door for a returning user who re-consents
    against a roster that already exists, and `request_now` on a live row is idempotent.

    Consent is READ, never granted — the same rule `bookmark_catchup.run_bookmark_catchup`
    states. A user who declined the recurring half gets no row here, and the rail child would
    refuse the pass anyway.

    Fail-safe: `request_now` already swallows an operator's unwritable queue into a logged False,
    and an unreadable consent marker reads as no. Neither costs the caller its confirmation.
    """
    from pipeline.kb import oracle_refresh
    try:
        if not oracle_refresh.consented():
            return False
        from pipeline.kb.rail_jobs import request_now
        return request_now("oracle_refresh")
    except Exception:
        return False


# Cluster members that are an ACCOUNT rather than a website. Their presence is what says "there
# is a person here" — the website is then one of their sources, judged by `eligibility.gate` at
# ingest like any other, and not the root's whole identity.
_ACCOUNT_PREFIXES: tuple[str, ...] = ("x:user:", "openalex:", "scholar:", "github:")


# A result row's `detail` is a rendered diagnostic summary, not a payload. The cap is generous —
# a real x-footprint summary runs ~3KB once its prefetch cache is excluded — so an ordinary row
# passes through untouched and only a leak trips it.
_DETAIL_MAX = 4000


def _detail(summary: object) -> str:
    """Render an adapter summary for a result row, BOUNDED.

    ⚠️ THE BACKSTOP, not the fix (2026-09-14). The fix is that adapters stop putting caches in
    their summaries — `ingest_x_footprint._prefetch_counters` is the first of those. This is what
    keeps the NEXT one from reaching a host: `str(summary)` renders whatever the adapter happens
    to return, so any dict added upstream lands verbatim in a tool response. One did, and
    `oracle(action='progress')` came back at 110KB — past the token limit, so the completion
    report for a finished pull could not be read at all.

    Truncation is REPORTED, never silent, for the same reason `build_screen` reports `omitted`: a
    reader has to be able to tell a short summary from a clipped one. Nothing is lost either way —
    every row carries `stats` beside this, as real JSON.
    """
    text = str(summary)
    if len(text) <= _DETAIL_MAX:
        return text
    return (f"{text[:_DETAIL_MAX]}… [truncated: {len(text)} chars. An adapter summary this long "
            f"is a cache leaking into a diagnostic — read `stats` for this row instead.]")


def _multi_author_refusal(conn, canonical_id: str, label: str) -> str | None:
    """Why this cluster cannot be an Oracle ROOT — everything OPYT knows about it is a website, and
    that website is written by a team — or None when it can.

    An Oracle is a PERSON whose judgement the user is borrowing. A team publication has no such
    person, and `eligibility.gate` already refuses to attribute one to anybody: `_route_source`
    sends every website through it with no `force`, so a `multi` verdict SKIPS forever. Confirming
    the root anyway wrote an `oracles` row that could never produce an atom — a permanent zombie,
    since `seed_from_entities` then registers an `oracle_sources` pair the refresh loop re-walks
    and re-skips every 336 hours. Refusing at the mint is the only place that state never exists.

    THE ACCOUNT TEST IS LOAD-BEARING, and it is not a convenience. `ingest_x_footprint` writes the
    X profile's website field into `identity_links`, `resolve` merges on those links, and a
    `blog:` member sorts below `x:user:` — so an ordinary person who lists their EMPLOYER on X
    ends up in a cluster whose head is the employer's site (`current_canonical` documents that
    sort). Judging by the head alone would refuse that person outright, over a source the ingest
    gate already handles correctly on its own.

    DEGRADES OPEN, the opposite of `eligibility.gate`, and for a different cost. There, an
    unclassifiable site must not be ingested — the expensive error is laundering. Here the
    expensive error is refusing a real person's blog because their home page happened not to
    fetch, so only a DEFINITIVE `multi` refuses; `unknown` mints and lets the ingest gate decide.
    """
    from . import eligibility, oracle_refresh_state as st

    members = schema.entities_for_canonical(conn, canonical_id)
    if any((m["entity_id"] or "").startswith(_ACCOUNT_PREFIXES) for m in members):
        return None

    # The canonical's own site first: it is the one the user named, and on an unmerged root it is
    # the only member there is.
    ordered = sorted(members, key=lambda m: m["entity_id"] != canonical_id)
    site = next((p[1] for p in (st.pair_from_member(m) for m in ordered)
                 if p and p[0] in ("blog", "substack")), None)
    if not site:
        return None
    if eligibility.classify_authorship(conn, site).authorship != "multi":
        return None

    return (f"{label} is a MULTI-AUTHOR site — a company, team or publication blog. An Oracle is "
            f"one PERSON whose judgement you are borrowing, so this cannot be a root: every "
            f"refresh would classify it multi-author and skip it, forever. Nothing was written. "
            f"Two things that DO work, and offer both: `hopper` saves any single post from it as "
            f"an atom, and the individual writer behind a post the user likes can be added as an "
            f"Oracle by their own X @handle, Substack or personal blog.")


def _is_venue_id(canonical_id: str) -> bool:
    """Is this cluster an OpenAlex SOURCE (a journal or preprint repository) rather than a person?

    The letter is the whole test, the same one `works_filter` routes on: OpenAlex prefixes an
    author id with `A` and a source id with `S`. No network, no store read."""
    return (canonical_id or "").startswith("openalex:S")


def _has_topic_filter(conn, canonical_id: str) -> bool:
    """Has the user already narrowed this cluster's subjects? Fail-safe: an unreadable registry
    reads as NOT narrowed, so the warning fires. Warning twice costs a sentence; staying silent
    costs 97% of a venue."""
    from . import oracle_refresh_state as st
    try:
        openalex_id = _openalex_id(conn, canonical_id)
        return bool(openalex_id and st.topic_filter_for(conn, canonical_id, openalex_id))
    except Exception:
        return False


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


def forget(conn, reference: str, *, confirm: bool = False) -> dict:
    """Preview or end one subscription, excluding the refresh pass that seeds its registry.

    The rail loads its roster before registering sources. Sharing its lease prevents a pass
    already holding that roster from re-creating registrations after removal commits.
    """
    from pipeline.sync_lock import CatchupLock

    if not confirm:
        return _forget(conn, reference, confirm=False)
    with CatchupLock("oracle-refresh") as lock:
        if not lock.acquired:
            return {"status": "busy", "scope": "oracle", "oracle": reference,
                    "message": "An Oracle refresh is running. Retry after it finishes; "
                               "the subscription has not been changed."}
        return _forget(conn, reference, confirm=True)


def _forget(conn, reference: str, *, confirm: bool) -> dict:
    """End one person's subscription, including pull registrations under old cluster heads.

    Entities, trust roots and existing atoms survive. The roster and its refresh registrations
    are removed together: the refresh loop reads the registry even without an Oracle row.
    """
    from .retrieve import resolve_who

    with conn:
        if confirm:
            conn.execute("BEGIN IMMEDIATE")
        roster = schema.list_oracles(conn)
        heads = {r["canonical_id"] for r in resolve_who(conn, reference)}
        heads.add(schema.current_canonical(conn, reference))
        matches = [o for o in roster if schema.current_canonical(conn, o["canonical_id"]) in heads
                   or (o["name"] or "").casefold() == reference.casefold()]
        groups = {}
        for o in matches:
            groups.setdefault(schema.current_canonical(conn, o["canonical_id"]), []).append(o)
        if not groups:
            return {"status": "not_found", "scope": "oracle", "oracle": reference}
        if len(groups) > 1:
            return {"status": "ambiguous", "scope": "oracle",
                    "candidates": [{"canonical_id": cid, "name": rows[0]["name"]}
                                   for cid, rows in groups.items()],
                    "message": "Choose one canonical_id and preview it again."}
        cid, matched = next(iter(groups.items()))
        members = [r["entity_id"] for r in schema.entities_for_canonical(conn, cid)]
        atoms = sum(conn.execute("SELECT COUNT(*) FROM atoms WHERE who_id=?", (eid,)
                                 ).fetchone()[0] for eid in members)
        # This registry is initialized by the refresh rail, so a never-refreshed store has none.
        has_registry = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='oracle_sources'").fetchone()
        sources = [dict(r) for r in conn.execute("SELECT * FROM oracle_sources")
                   if schema.current_canonical(conn, r["canonical_id"]) == cid] if has_registry else []
        name = matched[0]["name"] or cid
        out = {"status": "preview", "scope": "oracle", "canonical_id": cid, "name": name,
               "members": members, "atoms_kept": atoms, "sources_removed": len(sources),
               "consent": [
                   f"This stops tracking one Oracle: {name} ({cid}).",
                   f"Their subscription and {len(sources)} refresh registrations are removed.",
                   f"Their {atoms} existing atoms, identity records and trust roots remain. "
                   "Remove unwanted atoms individually with forget(atom_id=...).",
                   "Future scheduled refreshes will not select this person. If a refresh is "
                   "running, removal waits for a retry after it finishes. Bookmarks and "
                   "separately saved links can still arrive.",
                   "Adding this Oracle again starts a new subscription.",
               ]}
        if not confirm:
            return out
        # Match the entire current person, including multiple roster anchors from before a merge.
        anchors = [o["canonical_id"] for o in roster
                   if schema.current_canonical(conn, o["canonical_id"]) == cid]
        conn.executemany("DELETE FROM oracles WHERE canonical_id=?", [(a,) for a in anchors])
        from . import oracle_reviews
        oracle_reviews.delete_for_oracles(conn, [*anchors, cid])
        if has_registry:
            conn.executemany(
                "DELETE FROM oracle_sources WHERE canonical_id=? AND source_type=? AND source_key=?",
                [(s["canonical_id"], s["source_type"], s["source_key"]) for s in sources])
        out["status"] = "forgotten"
        return out


# ── add_oracle: the single user-facing "add a person" on the atom rail ──────────
#
# "Add a person" = admit an Oracle + expand what they carry into atoms: resolve reference →
# confirm → ingest (papers, and a verified footprint when they have accounts) → re-resolve, wired
# here as ONE two-phase entry point.

# `openalex:` sits here beside `scholar:` because it is the prefix every LIVE scholar candidate
# carries — `paper_authors.author_entity` mints OpenAlex ids, and `scholar:` is the Semantic Scholar
# namespace `derive_paper` writes. Missing it, `_classify_reference` called an `openalex:A…`
# canonical_id a "handle" and `add_oracle` sent it to `_fetch_x_identity` as a bare X @handle.
_CANONICAL_PREFIX = re.compile(
    r"^(x:user:|substack:|blog:|github:|org:|scholar:|openalex:|paper-authors:)", re.I)


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


# Hosts whose URLs are PROFILES on a platform, not a personal site. `_url_entity_id` keys any
# non-Substack http… reference on `blog:{host}`, which for these mints a phantom entity —
# measured: `https://github.com/karpathy` -> `blog:github.com/karpathy`, and worse,
# `https://x.com/karpathy` -> `blog:x.com/karpathy` for a person who already has an `x:user:{id}`.
# X is separated out because there IS a right answer for it: pass the bare @handle.
_PLATFORM_HOSTS: frozenset[str] = frozenset({"github.com"})
_X_HOSTS: frozenset[str] = frozenset({"x.com", "www.x.com", "twitter.com", "www.twitter.com"})


# Academic identity URLs, keyed by the prefix `canonical_identity` normalizes each host to.
# The value says whether OPYT can turn that URL into an id it can PULL work by.
#
# This replaces a two-host deny-list. Under that list every one of these was ADMITTED and minted a
# `blog:` entity — so `oracle_refresh` pointed the BLOG ADAPTER at orcid.org and scraped a
# researcher's identity record as if it were their archive. `canonical_identity` has had a
# purpose-built branch for every one of these hosts the whole time (ORCID with a format-validating
# regex, Semantic Scholar author paths, Google Scholar `user=`, DBLP `pid`/`pers`, ResearchGate,
# Academia.edu, arXiv author pages); the admission check simply never asked it.
#
# `semanticscholar.org/author/` read True until 2026-09-09 and minted `scholar:{id}` directly.
# That id has no pullable pair (`oracle_refresh_state.pair_from_member` skips it), so the URL
# confirmed an Oracle with zero sources — a fail-safe violation, measured live. Two ways out were
# measured before choosing refusal, and both lost to OpenAlex on the data:
#   • S2 author id → an OpenAlex one, by joining on the papers' DOIs. It resolves a FRAGMENT, not
#     a person: Karpathy's 19 DOIs voted 9-1 for A5009290031, but an arbitrary 8 of them voted for
#     a 2-work record, and a common surname (`W. Wang`) tied 2-2 between two different people.
#   • Pulling S2's own `/author/{id}/papers` feed, which does exist. Its records are dirtier than
#     OpenAlex's: of Karpathy's 24 keyed papers, 9 carry no Karpathy authorship in OpenAlex at all
#     ("DiplomacyAgent", "Dialect Normalization" — not his), plus 13 unkeyed entries and one SEO
#     spam paper.
# OpenAlex subsumes S2 as a WORKS index, which is what makes refusal cheap: Karpathy 24/24 keyed
# papers present, Arnold 398/408 (9 of the 10 misses are Protein Data Bank depositions, not
# papers). So the pullable academic roots are the two ids OpenAlex itself is keyed by.
_SCHOLAR_ROOTS: dict[str, bool] = {
    "orcid.org/": True,                     # → an OpenAlex author id, one free lookup
    "semanticscholar.org/author/": False,
    "scholar.google.com/": False,
    "dblp.org/": False,
    "researchgate.net/": False,
    "academia.edu/": False,
    "arxiv.org/a/": False,
}


def _scholar_root(reference: str) -> tuple[str, str] | None:
    """An academic identity URL → `(registry_prefix, identifier)`, or None if it is not one.

    Positive classification off `canonical_identity`, not a host deny-list: the canonicalizer
    already knows these are scholarly identity URLs and normalizes each to a stable prefix."""
    from pipeline.ingestion.url_canon import canonical_identity
    canon = canonical_identity((reference or "").strip())
    for prefix in _SCHOLAR_ROOTS:
        if canon.startswith(prefix):
            return prefix, canon[len(prefix):]
    return None


# An OpenAlex id URL pasted directly — `https://openalex.org/A5009290031` (an author) or
# `https://openalex.org/S4393918830` (a source). BOTH letters, because both refusal messages name
# one of these as the shape that works, and a user told to paste one needs it to be accepted.
#
# The AUTHOR form was admitted-and-broken until 2026-09-09: `canonical_identity` has no openalex
# branch, so the URL canonicalized to a bare `openalex.org`, missed `_SCHOLAR_ROOTS`, matched no
# venue name (`sources?search=openalex` returns zero), and fell through to `blog:openalex.org` —
# pointing the BLOG ADAPTER at an author record. The same defect the scholar roots were built to
# fix, on the one URL that carries a directly pullable id.
_OPENALEX_ID_RE = re.compile(r"openalex\.org/([AS]\d+)\b", re.I)


def _openalex_root(reference: str) -> dict | None:
    """A URL that names an OpenAlex id → `{openalex_id, name, works}`, or None if it is not one.

    TWO admissions with one shape, because `openalex:{id}` is one entity prefix and `works_filter`
    reads the id's own letter to pick the field it filters on. An `A…` is an author, an `S…` is a
    venue, and every stage after this one — the pair, the pull, the topic filter — is identical.

    A VENUE IS A FILTER ON OPENALEX, NEVER A NEW INGESTER (ruled 2026-09-08). Before this, a
    pasted `https://chemrxiv.org` was admitted as `blog:chemrxiv.org` and the blog adapter scraped
    a preprint repository as generic articles — no author entities, no DOI dedup, nothing reaching
    the scholar path. The same URL now resolves to `S4393918830` and pulls through
    `/works?filter=primary_location.source.id:…`, which is the author pull with one letter changed.

    Two accepted forms, and the split is the point:

      • An OpenAlex source URL, matched as a string. Exact, free, no network.
      • A host whose name resolves to EXACTLY ONE OpenAlex source. This is a NAME search, which
        the scholar path refuses for people — "Frances Arnold" returned 16 authors — and it is
        admissible here only because of that uniqueness rule. OpenAlex publishes no filterable
        homepage field (`homepage_url` is not a valid filter and is NULL even for ChemRxiv), so
        there is no exact-match alternative to reach for.

    MEASURED 2026-09-08, which is what the uniqueness rule rests on. Fifteen real personal-blog
    hosts (karpathy, simonwillison, paulgraham, stratechery, danluu, gwern, lesswrong …): fourteen
    matched ZERO sources and one matched six, so none is admitted. `chemrxiv` and `biorxiv` match
    exactly one. `medium` matches 24 (top hit: a medieval studies journal) and `nature` matches
    222 — both correctly refused, and both are the reason the rule is "exactly one" rather than
    "take the top hit".

    FAIL-SAFE: any lookup failure returns None, so the URL falls through to `blog:{host}` — which
    is precisely the behaviour that existed before this function, and the safe direction.
    """
    from urllib.parse import urlparse

    from .frontier_sources import OpenAlexWorksAdapter

    r = (reference or "").strip()
    direct = _OPENALEX_ID_RE.search(r)
    if direct:
        oid = direct.group(1).upper()
        adapter = OpenAlexWorksAdapter()
        try:
            rec = (adapter.author(oid) if oid.startswith("A") else adapter.source(oid)) or {}
        except Exception:
            rec = {}                        # the id itself is already enough to root on
        return {"openalex_id": oid, "name": rec.get("display_name"),
                "works": rec.get("works_count")}

    # The label immediately BEFORE the TLD, which is the one that names the publisher:
    # `chemrxiv.org` → chemrxiv, `journals.sagepub.com` → sagepub. Taking the FIRST label instead
    # would search `carol.substack.com` for "carol" — a person's name against a journal index,
    # which is the wrong question asked of the wrong corpus. Under this rule every blog-platform
    # host reduces to its platform name, and `substack`, `github` and `wordpress` all match zero
    # OpenAlex sources (measured 2026-09-08), so they stay blogs without a host list.
    host = (urlparse(r).hostname or "").lower().removeprefix("www.")
    parts = [x for x in host.split(".") if x]
    label = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
    if len(label) < 3:
        return None
    adapter = OpenAlexWorksAdapter()
    if not adapter.available():
        return None                         # breaker open: skip the lookup, do not pay for it
    try:
        hits = adapter.sources_by_name(label)
    except Exception:
        return None                         # fail-safe: unchanged, so it stays a blog
    if len(hits) != 1:
        # Zero (a personal blog) or several (an ambiguous name) — both stay blogs. The refusal
        # copy in `_unsupported_root` only fires for the ambiguous half.
        return None
    rec = hits[0]
    sid = str(rec.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
    if not sid.startswith("S"):
        return None
    return {"openalex_id": sid, "name": rec.get("display_name"),
            "works": rec.get("works_count")}


def _unsupported_root(reference: str) -> str | None:
    """Why this reference cannot be an Oracle ROOT, or None if it can.

    Rootable: an X @handle, a Substack URL, a personal blog/site URL, a canonical_id, an academic
    identity URL OPYT can turn into a pullable author id (an ORCID, or an OpenAlex author URL
    carrying the id itself), or a research venue. Every other platform URL is refused rather than
    minted as a personal blog.

    A VENUE NEEDS NO BRANCH HERE and deliberately does not get one. It is admitted by the same
    fallthrough that admits any site URL, and `_resolve_handle` is what decides between a venue
    and a blog — which keeps this function free of the network call that decision costs. A URL
    that turns out not to be a venue becomes a blog, exactly as before."""
    from urllib.parse import urlparse
    from pipeline.ingestion.url_canon import canonical_identity
    r = (reference or "").strip()
    if not r.startswith("http"):
        return None
    if not canonical_identity(r):
        return "that URL is unusable or belongs to an excluded platform"

    scholar = _scholar_root(r)
    if scholar:
        prefix, _ = scholar
        if _SCHOLAR_ROOTS[prefix]:
            return None                     # rootable — `_resolve_handle` mints a scholar entity
        # Refused with the shape that WOULD work, the same way the X branch below does. OPYT
        # pulls papers through OpenAlex and nothing else, so the only rootable academic URLs are
        # the two OpenAlex is keyed by. An entity minted from any other registry would be an
        # Oracle whose every refresh had nothing to pull.
        return (f"{prefix.rstrip('/')} has no author→works feed OPYT can pull — add them by their "
                "ORCID (https://orcid.org/0000-…) or their OpenAlex author page "
                "(https://openalex.org/A…), either of which resolves to a publication record "
                "OPYT can pull")

    host = (urlparse(r).hostname or "").lower()
    if host in _X_HOSTS:
        return ("that is an X profile URL — pass the bare @handle instead, so it resolves to the "
                "person's x:user id rather than minting a second, blog-shaped identity for them")
    if host in _PLATFORM_HOSTS:
        return (f"{host} is a platform profile, not a personal site — add the person by their X "
                "@handle, Substack, or blog")
    return None


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
# the report from the datetimes that actually ran, not the requested preset.


# The one X selector that is NOT a fixed span: it answers "since I last pulled THIS person",
# a different date per Oracle, rather than a uniform lookback.
X_SINCE_LAST = "since_last"


def _x_since(preset: str | None):
    """An X preset → its `since`, or None for "the adapter's own ~6-month default".

    Two inputs REFUSE rather than resolving, for one reason: on this selector None means the
    adapter's own 183-day default, so any silent fallthrough turns a request for a narrow window
    into the widest pull available. `since_last` needs a specific Oracle (see `x_since_last`),
    and an unrecognised preset — the host supplies this string, so `'1y'` for `'1yr'` is one
    typo away — is a caller bug, not a request for six months."""
    from . import expand
    if not preset:
        return None
    if preset == X_SINCE_LAST:
        raise ValueError(f"{X_SINCE_LAST!r} resolves per-Oracle — use x_since_last(conn, cid)")
    if preset not in expand.X_LOOKBACK_PRESETS:
        raise ValueError(f"unknown x_lookback {preset!r} — expected one of "
                         f"{sorted(expand.X_LOOKBACK_PRESETS)} or {X_SINCE_LAST!r}")
    return expand._since_from_days(expand.X_LOOKBACK_PRESETS[preset])


def x_since_last(conn, canonical_id: str):
    """This Oracle's since-last-pull X window — the same window the automatic refresh loop uses.

    Returns `last_pulled_at - OVERLAP_HOURS` (falling back to `cursor_ts`, the newest atom held),
    or None when neither exists. Callers must treat None as a refusal, never a default — see
    `_x_since`. Seeds the registry first only when X is connected: a fresh, unconnected Oracle
    must not acquire an X refresh row merely because a host asked for a window."""
    # Lazy for LOAD TIME, not for cycles: `pipeline.kb.oracles` is imported by the MCP tool
    # surface at request time, and `oracle_refresh` pulls in the X transport behind it. Since
    # 2026-09-04 nothing in this package imports back into `oracles`, so hoisting these is safe
    # — it just costs the transport on every tool call that never refreshes anything.
    from . import oracle_refresh, oracle_refresh_state as st

    from pipeline.ingestion.x_graphql import has_managed_x_session

    st.seed_from_entities(conn, canonical_ids=[canonical_id],
                          include_x=has_managed_x_session())
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


def _scholar_since(preset: str | None):
    """A scholar preset → its `since`. `'all'` (and unknown) → None = the whole corpus, no bound.

    LENIENT, like `_web_since` and unlike `_x_since`, and the asymmetry is the point: on this
    selector None means "no lower bound", so a fallthrough WIDENS the pull. On the X selector None
    means the adapter's own 183-day default, so a fallthrough there would silently turn a request
    for a narrow window into the widest pull available — which is why that one refuses."""
    from . import expand
    if not preset:
        return None
    return expand._since_from_days(expand.SCHOLAR_LOOKBACK_PRESETS.get(preset))


def scholar_year_counts(conn, canonical_id: str) -> dict | None:
    """The real distribution of an Oracle's papers by year, or None when there is nothing to ask.

    ONE free `group_by` call, so the lookback question can carry actual numbers — "928 papers;
    the last 2 years is 28, the last 5 is 102" — which neither the `x` nor the `web` selector can
    do. FAIL-SAFE: any failure returns None and the caller falls back to the blind presets. A
    count is what makes the question better; it must never be what makes it impossible.

    POST-FILTER, always. It counts through whatever topic filter this pair already carries, so
    once the user has narrowed, every number they are read back is a number about the pull that
    will actually run. Reporting 928 for a filter that takes 128 would break the one thing this
    function exists to provide.
    """
    from datetime import date

    from . import oracle_refresh_state as st
    from .frontier_sources import OpenAlexWorksAdapter

    openalex_id = _openalex_id(conn, canonical_id)
    if not openalex_id:
        return None
    topics = st.topic_filter_for(conn, canonical_id, openalex_id)
    try:
        counts = OpenAlexWorksAdapter().year_counts(openalex_id, topics=topics)
    except Exception:                       # fail-safe: the blind presets are the fallback
        return None
    if not counts:
        return None

    from . import expand
    this_year = date.today().year
    windows = {}
    for name, days in expand.SCHOLAR_LOOKBACK_PRESETS.items():
        if days is None:
            continue
        # Whole years, because that is the granularity `group_by=publication_year` answers at.
        # Rounding a 730-day window to 2 years over-counts by at most the current year's partial
        # — an over-count on a window the user is choosing, which errs toward showing MORE work
        # than the pull will do, never less.
        first = this_year - (days // 365) + 1
        windows[name] = sum(n for y, n in counts.items() if y >= first)
    return {"total": sum(counts.values()), "by_window": windows, "years": len(counts),
            "openalex_id": openalex_id, "topics": topics}


# How many topics the count-first ask names one by one. A DISPLAY bound and nothing else: every
# topic stays selected until the user narrows, and the ones past this are reported as a number.
# Set where a host can read the list aloud and the user can still hold it — the tail past ~25 is
# where a real distribution has thinned to single-figure counts anyway (Frances Arnold's 174
# topics, measured 2026-09-08: the 25th has 8 works, the 174th has 1).
TOPIC_SHOW = 25


def topic_counts_for_id(openalex_id: str, selected: str | None = None) -> dict | None:
    """The subject distribution behind ONE OpenAlex id — the count-first topic ask, id-keyed.

    Split out from `scholar_topic_counts` because the two callers reach it from opposite ends. An
    Oracle already in the roster arrives as a canonical_id and needs the stored selection joined
    on; a venue the user has just pasted has no cluster and no row yet, and still needs the ask —
    a first venue pull with no subjects chosen takes the whole repository.

    `topics` is capped at `TOPIC_SHOW` for DISPLAY and `more` states the remainder. Nothing is
    dropped from the pull by not being listed: everything is selected until the user narrows.

    FAIL-SAFE: a failure returns None and the ingest runs unfiltered, which is the wide direction
    and the one that loses no papers.
    """
    from .frontier_sources import OpenAlexWorksAdapter

    try:
        topics = OpenAlexWorksAdapter().topic_counts(openalex_id)
    except Exception:
        return None
    if not topics:
        return None

    shown = topics[:TOPIC_SHOW]
    total = sum(t["count"] for t in topics)
    return {
        "openalex_id": openalex_id,
        "total_works": total,
        "n_topics": len(topics),
        "topics": shown,
        "more": max(0, len(topics) - len(shown)),
        "selected": selected.split("|") if selected else None,
        "ask": (f"{total} papers across {len(topics)} subjects"
                + (f" (top {len(shown)} listed by size)" if len(topics) > len(shown) else "")
                + ". " + ("Currently narrowed to the ids under `selected` — pass "
                          "scholar_topics=[] to go back to everything."
                          if selected else
                          "EVERYTHING is selected by default; pass scholar_topics=[ids] to "
                          "narrow. The choice sticks to every future refresh, not just this "
                          "pull.")),
    }


def scholar_topic_counts(conn, canonical_id: str) -> dict | None:
    """The distribution of an Oracle's papers by SUBJECT, or None when there is nothing to ask.

    The count-first shape of `scholar_year_counts`, on the other axis, and the reason it exists is
    scope control rather than disambiguation: a clean researcher with 400 papers across 30 topics
    does not want all 30 either, so this fires for every scholar Oracle and not only for the
    contaminated ones.

    Deliberately UNFILTERED by the stored selection — this is the list the user picks FROM, so
    narrowing it to what they already chose would hide every topic they might add back. The
    selection rides along under `selected` instead.
    """
    from . import oracle_refresh_state as st

    openalex_id = _openalex_id(conn, canonical_id)
    if not openalex_id:
        return None
    return topic_counts_for_id(openalex_id,
                               st.topic_filter_for(conn, canonical_id, openalex_id))


def _openalex_id(conn, canonical_id: str) -> str | None:
    """The cluster's OpenAlex id — an author (`A…`) or a venue source (`S…`) — read off the entity
    rows. NO network, unlike `scholar_year_counts`, which is why the ingest path falls back to
    this when the count call was skipped or failed. A missing count must not also cost the pull."""
    from .oracle_refresh_state import pair_from_member

    for m in (_oracle_members(conn, canonical_id) or []):
        pair = pair_from_member(m)
        if pair and pair[0] == "openalex":
            return pair[1]
    return None


def _oracle_members(conn, canonical_id: str):
    """The cluster's entity rows, in the shape `pair_from_member` reads. Empty on any failure."""
    try:
        return conn.execute(
            "SELECT entity_id, identity_links, profile FROM entities "
            "WHERE COALESCE(canonical_id, entity_id) = ?",
            (schema.current_canonical(conn, canonical_id),)).fetchall()
    except Exception:
        return []


def _effective_x_since(x_since):
    """What the X adapter will ACTUALLY use: its default when unset, floored at the hard 2-year
    ceiling. Called through `ingest_x_footprint._resolve_since` rather than re-derived, so the
    report cannot drift from the clamp — a report that can disagree with the code IS the bug."""
    from .ingest_x_footprint import _resolve_since
    return _resolve_since(x_since, utc_now())


def _lookback_report(x_since, web_since, scholar_since=None, counts: dict | None = None) -> dict:
    """The human-facing windows, derived from the resolved datetimes rather than re-read off the
    preset strings — so what the user is told is what ran, including the X clamp they didn't ask
    for. This is the surface the consent invariant rests on.

    THREE windows, never collapsed into one. Each bounds a corpus with different physics: an
    ephemeral stream capped at 2 years, a durable web archive with no cap, and a published corpus
    bounded only by embed spend per work. One datetime can be right for at most one of them.

    `counts` is the count-first extra the other two selectors cannot offer: the real distribution
    of this person's papers by year, so the question reads "928 papers; the last 2 years is 28"
    instead of naming bare presets. Absent (a failed or impossible count) it simply does not
    appear — the presets still work blind."""
    eff = _effective_x_since(x_since)
    clamped = x_since is not None and eff > x_since
    out = {
        "x": f"since {eff:%Y-%m-%d}" + (" (CLAMPED to the 2-year ceiling)" if clamped else
                                        "" if x_since is not None else " (6-month default)"),
        "x_since": eff.isoformat(),
        "web": f"since {web_since:%Y-%m-%d}" if web_since else "full archive",
        "web_since": web_since.isoformat() if web_since else None,
        "scholar": f"since {scholar_since:%Y-%m-%d}" if scholar_since else "whole corpus",
        "scholar_since": scholar_since.isoformat() if scholar_since else None,
        "note": "X is an ephemeral stream, capped at 2 years; the web archive and the paper "
                "corpus have no cap. The three windows are chosen separately — a short X window "
                "does not truncate either archive.",
    }
    if counts:
        out["scholar_counts"] = counts
        windows = " · ".join(f"{k} is {v}" for k, v in counts["by_window"].items())
        out["scholar_ask"] = (f"{counts['total']} papers across {counts['years']} years — "
                              f"{windows}. Pick a scholar_lookback, or 'all'.")
    return out


def _oracle_for(conn, canonical_id: str) -> dict:
    """The confirmed-oracle record (with cluster members) for one canonical_id — re-anchored to
    the current head so the shared ingest engine gets the right members even after a resolve shift.

    Its only caller confirms the row four lines earlier, and `confirmed_oracles` re-anchors a
    stale stored id to the current head, so the lookup cannot miss. It raises rather than
    reconstructing a stand-in: a synthesised record here would hand the ingest engine a cluster
    with no `oracles` row behind it, and the missing row is the bug worth seeing."""
    head = schema.current_canonical(conn, canonical_id)
    for o in confirmed_oracles(conn):
        if o["canonical_id"] == head:
            return o
    raise LookupError(f"no confirmed oracle for {canonical_id!r} (head {head!r}) — confirm writes "
                      f"the row before this reads it, so this means the write did not land")


def _review_item_payload(conn, item: dict) -> dict:
    """One stored review decision in language a host can show to the user."""
    try:
        oracle = _oracle_for(conn, item["canonical_id"])
        name = oracle.get("name") or oracle["canonical_id"]
    except LookupError:
        name = item["canonical_id"]
    verified = item["status"] == "verified"
    collectable = item["source_type"] in {"blog", "substack", "github"}
    return {
        "review_id": item["review_id"],
        "oracle": name,
        "source": {"type": item["source_type"], "url": item["source_url"]},
        "status": "not_available" if not collectable else (
            "ready_to_add" if verified else "needs_confirmation"),
        "message": ("The links you supplied support this source. Add it when you are ready."
                    if verified and collectable else
                    "I could not confirm that this source belongs to this writer."
                    if collectable else
                    "OPYT cannot collect this type of source yet."),
        "actions": (["add", "verify", "dismiss", "leave_out"] if collectable else
                    ["dismiss", "leave_out"]),
    }


def review_sources(conn, embedder, *, action: str = "list", review_id: int | None = None,
                   verification_urls: list[str] | None = None, confirm: bool = False) -> dict:
    """List or carry out a user's review decision for one discovered footprint source.

    Discovery and the footprint router remain the owners of their attribution decisions. This
    coordinator only remembers a user's decision, re-checks supplied evidence, and sends exactly
    one explicitly approved source through the existing router.
    """
    from pipeline.ingestion.discover_profile import discover_profile

    from . import oracle_reviews
    from . import onboard_footprint as of
    from .expand import _root_profile

    action = (action or "list").strip().lower()
    if action == "list":
        items = oracle_reviews.list_open(conn)
        return {
            "items": [_review_item_payload(conn, item) for item in items],
            # The host gets the actual discovery context without having to turn it into onboarding
            # prose. `items` is the user-facing envelope; this is deliberately a separate field.
            "diagnostics": [{"review_id": item["review_id"], "canonical_id": item["canonical_id"],
                             "source_type": item["source_type"], "source_url": item["source_url"],
                             "reason": item["reason"], "verification_urls": item["verification_urls"]}
                            for item in items],
        }
    if action not in {"add", "verify", "dismiss"}:
        return {"error": "review_action must be 'list', 'add', 'verify', or 'dismiss'."}
    if not isinstance(review_id, int):
        return {"error": f"review_action='{action}' needs an integer review_id from review_action='list'."}

    item = oracle_reviews.get(conn, review_id)
    if not item:
        return {"error": f"no review item {review_id}. Run review_action='list' first."}
    if item["status"] not in {"pending", "verified"}:
        return {"error": f"review item {review_id} is already {item['status']}."}
    if action in {"add", "verify"} and item["source_type"] not in {"blog", "substack", "github"}:
        return {"error": f"OPYT cannot collect {item['source_type']} sources yet; dismiss it or leave it out."}
    try:
        oracle = _oracle_for(conn, item["canonical_id"])
    except LookupError:
        return {"error": "this review item belongs to an Oracle that is no longer subscribed."}

    payload = _review_item_payload(conn, item)
    if action == "dismiss":
        if not confirm:
            return {"status": "preview", "action": "dismiss", "item": payload,
                    "message": "This will keep this source out for this Oracle permanently. "
                               "Pass confirm=True to do it."}
        oracle_reviews.dismiss(conn, review_id)
        return {"status": "dismissed", "item": payload,
                "message": "This source will no longer be considered for this writer."}

    if action == "verify":
        urls = [url.strip() for url in (verification_urls or []) if isinstance(url, str) and url.strip()]
        if not urls:
            return {"error": "review_action='verify' needs verification_urls with an official profile, "
                             "site, or bio link that connects this writer to the source."}
        root = _root_profile(conn, oracle)
        if not root:
            return {"error": "this Oracle has no X, Substack, or blog profile to re-check from."}
        profile = discover_profile(root["seed"], seed_type=root["seed_type"], reverify=True,
                                   extra_source_urls=[item["source_url"], *urls])
        source = next((src for src in profile.get("sources", [])
                       if src.get("source_type") == item["source_type"]
                       and oracle_reviews.source_key(src.get("url") or "") == item["source_key"]), None)
        if source and (source.get("trust") or {}).get("trusted"):
            oracle_reviews.mark_verified(conn, review_id, urls)
            return {"status": "verified", "item": _review_item_payload(conn, oracle_reviews.get(conn, review_id)),
                    "message": "I could verify the connection. Nothing was added yet; choose Add when ready."}
        oracle_reviews.record_evidence(conn, review_id, urls)
        return {"status": "still_unverified", "item": payload,
                "message": "Those links did not establish that this source belongs to the writer. "
                           "You can add it yourself, try different official links, dismiss it, or leave it out."}

    if not confirm:
        return {"status": "preview", "action": "add", "item": payload,
                "message": "This will add only this source for this writer. Pass confirm=True to do it."}

    source = {"source_type": item["source_type"], "url": item["source_url"],
              "metadata": {}, "trust": {"trusted": True, "reasons": ["user confirmed"]}}
    result = of.onboard_footprint(conn, embedder, oracle["canonical_id"], [source],
                                  author_name=oracle.get("name"), force=True)
    if not any(row.get("action") in {"ingested", "blocked"}
               for row in result.get("results") or []):
        return {"status": "not_added", "item": payload, "result": result,
                "message": "I could not add this source yet. It remains available to retry."}
    # Approval follows a targeted source pass that established its entity. The refresh registry,
    # rather than a later discovery replay, owns its ongoing collection from here.
    oracle_reviews.approve(conn, review_id)
    try:
        from . import oracle_refresh_state
        oracle_refresh_state.seed_from_entities(conn, canonical_ids=[oracle["canonical_id"]])
        _record_coverage(conn, oracle["canonical_id"], result.get("results") or [],
                         x_since=_effective_x_since(None), web_since=None)
    except Exception as e:
        # The source decision and any content write already succeeded; registry repair must not
        # turn an explicit approval into a failed user action. Keep the failure in diagnostics so
        # the host can understand why future automatic pulls may not yet be registered.
        result["refresh_error"] = f"{type(e).__name__}: {e}"
    return {"status": "added", "item": payload, "result": result,
            "message": "This source is now included for this writer."}


def _store_scholar_topics(conn, oracle_id: str, topics) -> str | None:
    """Persist the user's topic selection onto this Oracle's OpenAlex pair, and return it.

    CALLED BEFORE THE PULL, not after, and that ordering is the feature. `oracle_refresh` re-pulls
    a scholar pair forever from `source_key` alone, so a filter applied only at ingest time lets
    the ongoing stream widen back to the full corpus within one TTL — silently, because nothing
    reports a pull that got MORE than the user asked for. Writing the row first means the first
    backlog and every later refresh read one stored decision through the same lookup.

    `upsert_source` first because the pair may not be registered yet: on a first `add_oracle` the
    entity exists but `_seed_pairs` has not run, and `set_topic_filter` updates an existing row
    rather than creating one.

    An empty list CLEARS the filter. Not calling this at all leaves it alone — which is what an
    ordinary top-up does, and why a top-up cannot quietly unfilter someone.
    """
    from . import oracle_refresh_state as st

    openalex_id = _openalex_id(conn, oracle_id)
    if not openalex_id:
        return None
    st.upsert_source(conn, st.SourceRow(oracle_id, "openalex", openalex_id))
    return st.set_topic_filter(conn, oracle_id, openalex_id, topics)


def _pull_scholar_papers(conn, embedder, oracle: dict, oracle_id: str, *, timer, since) -> dict:
    """The Oracle's OWN papers, when the cluster carries an OpenAlex id. `{}` when it does not,
    which is what the caller reads as "this Oracle has no paper corpus".

    The id is an author (`A…`) or a venue source (`S…`), and this function does not branch on
    which: `works_filter` reads the prefix. A venue Oracle is this same pull with a different
    letter and, in practice, a topic filter that makes it finite.

    NO DISCOVERY AND NO TRUST ROOT, and that is the whole reason this is its own function. There
    is nothing to discover (a paper corpus is not a set of accounts to find) and nothing for a
    trust root to verify — `atomize_paper` writes each paper's OWN first author as `who_id` with
    no override, so nothing here can attribute another person's work to the Oracle. Direct to the
    adapter, like X, with no eligibility gate to route through. Abstract-only, so the cost is one
    embed per work.

    The TOPIC FILTER is read off the `oracle_sources` row rather than taken as an argument. One
    stored home, read identically here and by the refresh rail, so the backlog and the ongoing
    stream cannot disagree about what the user chose.

    A failed pull never aborts the rest, exactly as a failed X pull does not. An open breaker
    arrives as `SourceError` and is recorded UNSTAMPED, so the pair stays infinitely stale and the
    refresh rail retries it.
    """
    openalex_id = _openalex_id(conn, oracle_id)
    if not openalex_id:
        return {}

    from . import ingest_scholar_footprint as isf
    from . import oracle_refresh_state as st
    from .frontier_sources import SourceError

    topics = st.topic_filter_for(conn, oracle_id, openalex_id)
    url = f"https://openalex.org/{openalex_id}"
    try:
        with timer.stage("openalex"):
            ss = isf.sync_scholar_footprint(conn, embedder, openalex_id=openalex_id,
                                            author_name=oracle.get("name"), since=since,
                                            topics=topics)
        # `capped` rides in `stats`, not only inside `detail`'s stringified summary. On the
        # AUTHOR path it never fires — 2,000 is an order of magnitude past the most prolific
        # researcher measured. On the VENUE path an unfiltered pull hits it every time, and a
        # truncation the user cannot see is one they will read as a complete corpus.
        entry = {"url": url, "type": "openalex", "action": "ingested", "detail": _detail(ss),
                 "stats": {"added": ss.get("atoms", 0), "dispatched": ss.get("papers", 0),
                           "capped": bool(ss.get("capped"))}}
        if ss.get("capped"):
            entry["note"] = (
                f"TRUNCATED at {isf.MAX_WORKS_PER_PULL} papers — this source publishes more than "
                f"one pull holds, and what landed is the most recent slice, not the whole of it. "
                f"Say so. Narrowing by subject (`scholar_topics`) is what makes the corpus a "
                f"selection rather than a cut-off; `add_oracle`'s preview lists the subjects.")
        return {"results": [entry], "ingested": 1, "atoms_added": ss.get("atoms", 0)}
    except Exception as e:
        action = "blocked" if isinstance(e, SourceError) else "error"
        return {"results": [{"url": url, "type": "openalex", "action": action,
                             "detail": f"{type(e).__name__}: {e}"}],
                ("blocked" if action == "blocked" else "errors"): 1}


# The counter keys a per-source partial may carry. Summed rather than overwritten, so a scholar
# pull's `errors` adds to the router's instead of replacing it.
_MERGED_COUNTERS = ("ingested", "blocked", "errors", "deferred", "atoms_added")


def _merge_source_result(result: dict, partial: dict) -> None:
    """Fold one source's partial outcome into the run's report. Appends `results`, SUMS the
    counters. A no-op for `{}`, which is what a source that does not apply returns."""
    if not partial:
        return
    result.setdefault("results", []).extend(partial.get("results") or [])
    for k in _MERGED_COUNTERS:
        if k in partial:
            result[k] = result.get(k, 0) + partial[k]


def _say_when_it_resumes(rec: dict, reset_at: float | None) -> None:
    """Turn a deferral into a time the host can say out loud — in the clock that home runs on.

    TWO CLOCKS, AND THE LATER ONE WINS. x.com's bucket refills at `reset_at`; the refresh rail
    wakes every `RAILS['oracle_refresh'].cadence` seconds. Work resumes on the first rail pass
    AFTER the bucket refills, so the honest number is the larger of the two, and quoting either
    alone under-promises in one direction and over-promises in the other.

    ⚠️ BUT A LAPTOP'S MINUTES ARE NOT WALL-CLOCK MINUTES, and that is why this splits. A local
    home is kept by a LaunchAgent, which does not run while the mac is asleep — `KeepAlive`
    restarts a process that died, it does not wake a machine — so on a lid-shut laptop the
    cadence keeps no wall-clock promise at all. Measured 2026-09-15: a user was told "about 10
    minutes", `pmset` shows the mac asleep for 39 of the next 62 (clamshell, on battery), and the
    rail's next pass landed 62 minutes later when the lid was reopened. Stating minutes there
    took a sentence that was vague-but-true and made it precise-and-false, which is worse.

    A HOSTED home has no such problem and must not be taxed for the laptop's: `opyt-worker.service`
    runs on a box that does not sleep, so the arithmetic above is exactly right there and the user
    should get the number. `queue_is_shared()` is the seam — its own docstring records the last
    time a local-mac fact was read as if it answered for hosted users, which is the mistake this
    branch exists to avoid repeating. Both branches still say SOON, because both are: the
    difference is only whether "soon" is counted in wall-clock or in awake time.

    Fail-safe: any problem here leaves `rec` exactly as it was. A deferral that cannot name a time
    is the status quo this function improves on, never a broken tool call.
    """
    from pipeline.kb.rail_jobs import queue_is_shared
    from pipeline.kb.rail_worker import RAILS
    try:
        cadence = float(RAILS["oracle_refresh"].cadence)
        wait = cadence if reset_at is None else max(cadence, float(reset_at) - time.time())
        minutes = max(1, round(wait / 60))
        rec["resumes_in_seconds"] = max(0, int(wait))
        common = ("Say it ONCE for the whole run, in plain words. Do NOT say 'the next scheduled "
                  "run', do not name a rail, a queue or a rate limit, and do not call this a "
                  "failure: the part they asked for that has not arrived yet is still coming, and "
                  "nobody has to stay in this conversation for it.")
        if queue_is_shared():
            rec["host_note"] = (f"Tell the user it resumes on its own in about {minutes} "
                                f"minute(s) from now. " + common)
        else:
            # Named so a reader of the payload sees the condition, not just the number.
            rec["resumes_requires"] = "the user's computer to be awake"
            rec["host_note"] = (
                f"Tell the user it resumes on its own SHORTLY — Opyt tries again about every "
                f"{minutes} minute(s) that their computer is awake. Put it that way round: soon, "
                f"but counted in time their machine is actually on, so if they shut the lid it "
                f"picks up shortly after they open it again. Do NOT give a bare clock time as if "
                f"it will happen whether or not the machine is running. " + common)
    except Exception:
        pass


def _stamp_source(conn, oracle_id: str, rec: dict, *, x_since, web_since, scholar_since,
                  include_x: bool) -> None:
    """Register + stamp the refresh registry for ONE source, the moment that source finishes.

    Per-source and not end-of-run because a part-way kill is the normal case, not the exception:
    an `oracle(action='ingest')` killed 62s into a 187-post archive on 2026-09-13 left atoms in the
    store with ZERO record of what had been attempted — `AtomSink` flushes incrementally while
    `_record_coverage` and `oracle_reviews.record_outcomes` both fired only after the LAST source.
    The bookkeeping now lands at the same grain the atoms do, so a killed run leaves a store that
    knows what it is missing.

    `seed_from_entities` runs here too, and must: `_record_coverage` joins through
    `pairs_by_entity`, and the substack/blog/github entities exist only after the router's
    `resolve.resolve_entities` call for THIS source. It is scoped to one Oracle and idempotent
    (`upsert_source` COALESCEs), so re-running it per source claims nothing a later one rewinds.

    Fail-safe, exactly as the end-of-run call was: registry trouble must never fail an ingest that
    has already written atoms."""
    try:
        from . import oracle_refresh_state, oracle_reviews
        oracle_refresh_state.seed_from_entities(conn, canonical_ids=[oracle_id],
                                                include_x=include_x)
        _record_coverage(conn, oracle_id, [rec], x_since=_effective_x_since(x_since),
                         web_since=web_since, scholar_since=scholar_since)
        oracle_reviews.record_outcomes(conn, oracle_id, [rec])
    except Exception as e:
        from pipeline.ingestion.utils import log
        log(f"[oracles] per-source stamp skipped for {oracle_id} "
            f"({rec.get('type')} {rec.get('url')}): {type(e).__name__}: {e}")


def _ingest_oracle(conn, embedder, oracle: dict, *, force: bool = False,
                   x_since=None, web_since=None, scholar_since=None,
                   scholar_topics=None, limit: int = 0,
                   extra_source_urls: list[str] | None = None,
                   x_only: bool = False) -> dict:
    """The ONE per-Oracle ingest engine, shared by `add_oracle` and `oracle(action='ingest')`.

    The rail scope lives here rather than on either caller, since this is the one function every
    path goes through.

    TWO INDEPENDENT HALVES, and which ones run depends on what the cluster carries:

      • The PAPER corpus (`_pull_scholar_papers`) — an OpenAlex author id. No discovery, no trust
        root. Runs FIRST, so it is reachable for an Oracle that has nothing else.
      • The FOOTPRINT path — `discover_profile` → trust-filter → `onboard_footprint` → the Oracle's
        own X timeline when X is connected. Needs a rootable profile (an X, Substack or blog
        member); skipped entirely without one.

    A scholar Oracle is therefore NEVER a trust root, which is the 2026-09-08 ruling (David) — a
    researcher is a topic-scoped feed, not a vouched voice. It needs no code to enforce: the trust
    graph exists ONLY inside `discover_profile` (`trust_graph.propagate` has exactly that one
    caller, and it computes verdicts per run rather than storing them), so an Oracle that skips
    discovery is outside it by construction. Their papers are still `oracle-footprint`, so they
    still feed the query generator — which is what a scholar Oracle is for.

    ⚠️ This docstring said "seeds the cluster as a tier-1.0 trust root, unconditionally" until
    2026-09-08. There has been no tier since 2026-08-23, when `entity_trust` was dropped for
    holding ten hand-seeded 1.0 rows and having no reader
    (docs/plans/2026-08-23-delete-edges-and-trust-tiers.md). Do not reintroduce the phrase.

    An Oracle with both halves gets both. Fail-safe throughout: `onboard_footprint` isolates each
    source, and a failed X or paper pull is recorded, never aborts.

    `limit` caps posts dispatched by the Substack/blog adapters. X processes its fetched
    window; GitHub uses its own repository selection.

    THREE SEPARATE windows: `x_since` bounds the X stream (clamped to a hard 2-year ceiling);
    `web_since` bounds the Substack/blog archive (no ceiling); `scholar_since` bounds the
    OpenAlex paper corpus (no ceiling — bounded by embed spend per work, and the pull is
    abstract-only). See `_lookback_report`.

    `scholar_topics` is a SUBJECT bound, not a fourth window, and it is stored rather than
    applied: a list of OpenAlex topic ids narrows the paper corpus permanently, `[]` clears the
    narrowing, and None leaves whatever the user chose last time alone. That last case is what
    makes a top-up safe — see `_store_scholar_topics` for why it is written before the pull.

    `x_only` is a PHASE SELECTOR, not a fourth window, and the distinction is what keeps it
    outside `.guards.py`'s `collapsed-oracle-lookback` (which bans a single `since` here). It says
    "the non-X halves already ran on this Oracle in this run" — so skip the papers, skip
    discovery, skip the whole `onboard_footprint` route, and go straight to the timeline.

    Only the X timeline can differ between two passes over one Oracle, because BREADTH IS
    EXPRESSED AS A WINDOW and the window only ever bounded X. Everything else is unbounded by
    design (R4: archives run to completion), so a second pass re-fetched an archive it already
    held: measured 2026-09-14, the depth loop over four Oracles spent 16 seconds re-running
    discovery and a sitemap crawl for zero atoms, and that cost scales with the roster.

    ⚠️ Discovery is skipped only when the ROOT already names the handle (an X-rooted Oracle,
    where `_x_handle_to_pull` returns `root["seed"]` and never reads `profile`). A Substack- or
    blog-rooted Oracle's X exists ONLY in the discovered sources, so skipping discovery there
    would silently drop their timeline from the deep pass — the one thing this pass is for.

    The returned summary carries `atoms_added` vs `dispatched` so a caller can see when they
    diverge, plus `blocked` (a host stopped us, nothing written, retry) distinct from `ingested`."""
    from pipeline.ingestion.discover_profile import discover_profile

    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.x_graphql import has_managed_x_session

    from . import ingest_common, ingest_x_footprint
    from . import onboard_footprint as of
    from .expand import _root_profile, _x_handle_to_pull

    oracle_id = oracle["canonical_id"]
    # Time discovery and the X pull here (they're owned by this function); merge into the
    # router's own totals below so one Oracle ingest reports one per-stage table.
    timer = ingest_common.StageTimer()

    # PAPERS FIRST, and ABOVE the root gate below. Hoisted 2026-09-08: this block used to sit ~100
    # lines further down, past a gate it can never pass. `_root_profile` seeds discovery from an X,
    # Substack or blog member only, so a researcher found purely through a saved paper returned
    # "no rootable profile" and their papers were never pulled — the exact Oracle the scholar rail
    # exists for. Proven against a scholar-only cluster; no test drove `_ingest_oracle` with one,
    # which is why it shipped green.
    #
    # It belongs above the gate rather than inside it because papers need NEITHER half of what the
    # gate protects: no discovery (the id already names one author or one venue) and no trust root
    # (nothing discovered, so nothing to verify). See `_pull_scholar_papers`.
    #
    # The topic selection is STORED FIRST, so the pull below reads it back off the pair rather
    # than being handed it — one home for the decision, shared with `oracle_refresh`.
    if scholar_topics is not None:
        _store_scholar_topics(conn, oracle_id, scholar_topics)

    # Hoisted above the scholar pull only so `_stamp` can close over it — the X pull below asks the
    # same question and the answer cannot change mid-run in a way that matters: it gates whether X
    # pairs are REGISTERED, and a session that appears part-way through is picked up next ingest.
    x_connected = has_managed_x_session()

    def _stamp(rec: dict) -> None:
        _stamp_source(conn, oracle_id, rec, x_since=x_since, web_since=web_since,
                      scholar_since=scholar_since, include_x=x_connected)

    # `{}` on an X-only pass, and `_merge_source_result` already treats that as "does not apply".
    scholar = {} if x_only else _pull_scholar_papers(conn, embedder, oracle, oracle_id,
                                                     timer=timer, since=scholar_since)
    # Stamped HERE, not at `_merge_source_result` below: the papers are already in the store, and
    # everything between here and there (discovery, an archive walk, a timeline) is exactly the
    # long stretch a kill lands in.
    for rec in scholar.get("results") or []:
        _stamp(rec)

    root = _root_profile(conn, oracle)
    if not root and not scholar and not x_only:
        return {"oracle_id": oracle_id, "name": oracle.get("name"),
                "error": "no rootable profile (no X, Substack, or blog member) and no OpenAlex "
                         "author id — nothing to discover and nothing to pull"}
    # `x_only` is exempt from that gate because it suppresses the scholar pull itself, so a
    # scholar-only Oracle would be reported as unrootable by the very flag that says its papers
    # already landed. With no root there is no handle either, so the pass falls through to an
    # empty — and truthful — report rather than an error the breadth pass disproves.

    # Discovery on this path makes no model call: the X reads use the user's own cookie
    # session, and the cold-start identity anchor — the one model step this used to have — was
    # deleted 2026-08-28.
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
    # A scholar-only Oracle has no root to discover FROM, so this whole stage is skipped and
    # `routable` falls to `[]` below. `onboard_footprint` is still called with that empty list
    # rather than bypassed, so the report keeps ONE shape for every Oracle — routing zero sources
    # is the honest description of a person who has none.
    profile: dict = {}
    fresh_discovery = False
    # An X-rooted Oracle's handle IS `root["seed"]`, so an X-only pass needs nothing discovery
    # returns — see the ⚠️ in the docstring for why the other two root types still discover.
    if root and not (x_only and root["seed_type"] == "x"):
        with timer.stage("discover_profile"):
            profile = discover_profile(root["seed"], seed_type=root["seed_type"],
                                       reverify=force,
                                       extra_source_urls=extra_source_urls)
        # Did we actually look this time, or replay a cached profile? `add_oracle` gates its
        # open-web `followup` on this — re-searching an unchanged person can only return what is
        # already here.
        fresh_discovery = not profile.get("from_cache")
    srcs = profile.get("sources", [])
    # Every discovered source goes to `onboard_footprint`, trusted or not, unless the user already
    # decided that exact source. The router owns the trust boundary (its step 2 records an
    # untrusted source as `needs-review` and never ingests it); a pending/verified/dismissed item
    # is a separate user permission decision, and an approved item already has its own targeted
    # source registration for ongoing refresh.
    #
    # X is the one exclusion: it is pulled as a timeline below, so routing it too recorded the
    # same handle twice — once `unsupported` by the router, once `ingested` by the pull.
    from . import oracle_reviews
    if x_only:
        # NOT `onboard_footprint([])`: the empty-list call is how a scholar-only Oracle reports
        # "routed zero sources", which is a real observation. This pass routed none because
        # another pass already routed them all, and saying nothing is the honest shape for that.
        result: dict = {}
    else:
        held_sources = oracle_reviews.held_source_keys(conn, oracle_id)
        routable = [s for s in srcs
                    if s.get("source_type") != "x"
                    and (s.get("source_type") or "",
                         oracle_reviews.source_key(s.get("url") or "")) not in held_sources]
        result = of.onboard_footprint(conn, embedder, oracle_id, routable,
                                      author_name=oracle.get("name"), force=force,
                                      since=web_since, limit=limit, on_source=_stamp)

    # The Oracle's OWN X timeline — the root handle when X-rooted, or a discovered+trusted X for a
    # Substack/blog-rooted Oracle. Discovery establishes identity, not permission to use a
    # session-backed source: X runs only after the user has connected OPYT's X session.
    xh = _x_handle_to_pull(root, profile) if root else None
    if xh and not x_connected:
        result.setdefault("available_sources", []).append(
            {"source_type": "x", "url": f"https://x.com/{xh}"})
    # NO PRE-EMPTIVE RESERVE HERE, and its absence is the point (2026-09-14). A reservation sat
    # on this branch for as long as `_pull_own_timeline` was all-or-nothing: a walk that ran dry
    # mid-way raised with nothing written, so refusing before spending anything reached the same
    # state for free and left the meter to the picks that could still finish. Once a partial walk
    # KEEPS what it walked, refusing is strictly worse — it leaves requests unspent and the Oracle
    # with nothing. Measured: two of four Oracles got 0 requests and 0 atoms, on a cold meter and
    # again on a refilled one, permanently.
    #
    # `x_graphql_core._refuse_if_spent` remains underneath and is the real bound. It refuses a
    # request x.com has already SAID it will 429 — evidence, not prediction — so the only route to
    # `deferred` below is now x.com genuinely refusing the walk.
    elif xh:
        try:
            with timer.stage("x"):
                xs = ingest_x_footprint.sync_x_footprint(
                    conn, embedder, handle=xh, author_name=oracle.get("name"),
                    since=x_since)
            # The adapter reports a hard stop by returning an `error` summary, so this except
            # clause only ever sees the raising half of the contract.
            outcome = ingest_common.classify_run(xs)
            rec = {"url": f"https://x.com/{xh}", "type": "x", "action": outcome, "detail": _detail(xs)}
            if isinstance(xs, dict) and outcome == _OBSERVED and "covered_from" in xs:
                # ⚠️ PRESENCE, NOT TRUTHINESS, at both hops — and the None is the load-bearing
                # value. A summary that carries the key has MEASURED its own reach, and a partial
                # walk's honest answer to that is None (`_walk_frontier`); one that does not
                # carry it never measured, and `_record_coverage` falls back to the window we
                # asked for, which is what every hash-bounded adapter still wants. Copy only the
                # truthy value and a one-sided walk falls through to that fallback — claiming the
                # asked window off a walk that never covered it, which is the one outcome §C of
                # the plan exists to prevent.
                rec["covered_from"] = xs["covered_from"]
            if isinstance(xs, dict) and outcome == _OBSERVED and xs.get("partial"):
                # ⚠️ AN ANNOTATION, NEVER A NEW `action` STRING. A partial walk is BOTH "atoms
                # landed" and "more is owed", and three readers key off `action == "ingested"`:
                # `_stamp_source` (`_OBSERVED`, just below), `_ingest_presentation`, and the
                # counter roll-up in `_merge_source_result`. An `action = "partial"` would make
                # every one of them drop the record SILENTLY — it would vanish from `completed`
                # and stop stamping `last_pulled_at`, which a real observation has earned.
                #
                # Additive keeps an unaware reader correct: it treats this as a complete ingest,
                # which is right for the stamp, and `covered_from` is already handled honestly
                # above by passing None. Readers that DO care opt in with `.get("partial")`.
                rec["partial"] = True
            stats = ingest_common.run_stats(xs)
            if stats:
                rec["stats"] = stats
            result.setdefault("results", []).append(rec)
            _stamp(rec)
            for k, agg in (("ingested", outcome == "ingested"), ("blocked", outcome == "blocked"),
                           ("errors", outcome == "error")):
                if agg:
                    result[k] = result.get(k, 0) + 1
            # Only `added` folds into the cross-source total. `dispatched` has no common unit
            # across adapters (substack/blog counts posts handed to the pool; X counts
            # referenced-link dispatches) — the X figure stays on the X record's own `stats`.
            result["atoms_added"] = result.get("atoms_added", 0) + stats.get("added", 0)
        except core.XRateLimited as e:
            # A rate window is temporary. The pair below remains unstamped, so the refresh rail
            # takes it on the next scheduled run.
            #
            # ⚠️ SAY WHEN, NOT "LATER". `XRateLimited` has carried `reset_at` — the unix instant
            # x.com's bucket refills, off `x-rate-limit-reset` — for exactly this, and its own
            # docstring says why: "so a caller can say WHEN rather than 'try again later'". This
            # site threw it away, so `resumes` reached the host as the bare token
            # `next-scheduled-run` and the host rendered the token as prose. Measured 2026-09-15:
            # a user who had just been told "6 months" got "the rest will fill in automatically
            # on Opyt's next scheduled run" — no time, on the one question they had. Both numbers
            # needed to answer it were in the process: this exception's `reset_at`, and the
            # refresh rail's 600s cadence in `rail_worker.RAILS`.
            rec = {"url": f"https://x.com/{xh}", "type": "x", "action": "deferred",
                   "resumes": "next-scheduled-run", "detail": f"{type(e).__name__}: {e}"}
            _say_when_it_resumes(rec, getattr(e, "reset_at", None))
            result.setdefault("results", []).append(rec)
            result["deferred"] = result.get("deferred", 0) + 1
            _stamp(rec)
        except core.SyncAuthError as e:
            # A session that was valid at the eligibility check but fails during its pull needs a
            # new login. Treating it as an automatic retry made a reconnect requirement sound
            # like background progress.
            rec = {"url": f"https://x.com/{xh}", "type": "x", "action": "needs_reconnect",
                   "detail": f"{type(e).__name__}: {e}"}
            result.setdefault("results", []).append(rec)
            result["needs_reconnect"] = result.get("needs_reconnect", 0) + 1
            _stamp(rec)
        except Exception as e:                       # a failed X pull never aborts the off-X ingest
            rec = {"url": f"https://x.com/{xh}", "type": "x", "action": "error",
                   "detail": f"{type(e).__name__}: {e}"}
            result.setdefault("results", []).append(rec)
            result["errors"] = result.get("errors", 0) + 1
            _stamp(rec)
    _merge_source_result(result, scholar)

    result["discovery_ran_fresh"] = fresh_discovery
    # Router stages and ours have disjoint keys, so this is a union, not an override.
    result["stage_seconds"] = {**result.get("stage_seconds", {}), **timer.totals}
    # The windows this run actually used, so the caller reports what ran, not what was asked.
    result["lookback"] = _lookback_report(x_since, web_since, scholar_since)
    # Coverage and the review queue were already written per source by `_stamp`, as each one
    # finished — registering the pair, THEN recording coverage for it only if it actually returned.
    # Two steps and not one: seeding claims nothing, so a transient failure remains stale and sorts
    # first in the refresh rail. A dead X session is also left unstamped, but is skipped until
    # reconnect.
    #
    # What remains here is the FINAL seed: a source whose record never reached `_stamp` (an
    # affiliation, an unsupported type, a pair `resolve` folded in only afterwards) still needs its
    # row, and a pair registered without a stamp is the retry queue, not a claim. Fail-safe, same
    # as the per-source write: registry trouble must not fail an ingest that already wrote atoms.
    try:
        from . import oracle_refresh_state
        oracle_refresh_state.seed_from_entities(conn, canonical_ids=[oracle_id],
                                                 include_x=x_connected)
        result["coverage"] = _coverage_report(conn, oracle_id)
    except Exception as e:
        from pipeline.ingestion.utils import log
        log(f"[oracles] refresh-registry seed skipped for {oracle_id}: {type(e).__name__}: {e}")
    # How many of this person's sources are waiting for a decision — including ones queued by
    # EARLIER runs, which is why it is a store read and not a tally of `result["results"]`.
    # `coverage` above cannot say this: a needs-review source never reaches a registered pair at
    # all (the adapter did not run, so no entity was minted), so an Oracle with an unreviewed
    # Substack reported its X-only coverage as complete. Fail-safe: unreadable is omitted, not
    # reported as zero — "nothing is waiting" is a claim, and a broken read must not make it.
    try:
        result["unreviewed_sources"] = oracle_reviews.open_counts(
            conn, canonical_ids=[oracle_id]).get(schema.current_canonical(conn, oracle_id), 0)
    except Exception:
        pass
    return result


# The one outcome that stamps a pair. `ingest_common.classify_run` returns `ingested` for any run
# that came back without an `error` — including one that found nothing new, which is a real
# observation of the source.
#
# Every other action is excluded, and none of them is an oversight. `blocked` (a host stopped us),
# `error` and `deferred` wrote nothing and marked nothing seen, so they must leave the pair
# unstamped: an unstamped pair is infinitely stale, sorts first in the refresh rail, and is
# re-pulled next session. That is the retry queue for a transient failure such as a rate limit.
# `needs_reconnect` also remains unstamped, but the rails skip it until X is connected again.
# `affiliation`, `needs-review`, `not-built`, `unsupported` and `skipped` never reach a registered
# pair at all — the adapter did not run, so
# no `blog:`/`substack:`/`github:` entity was minted for `pairs_by_entity` to find.
_OBSERVED = "ingested"


def _result_entity_id(rec: dict) -> str | None:
    """One `onboard_footprint`/X result row → the entity id whose pair records it, or None.

    Derived, never string-matched. The registry keys pairs off entity ids minted by
    `derive.blog_entity_id` / `derive.substack_entity_id` / the GitHub owner regex, so running a
    result's url back through those SAME functions is the only join that cannot drift from two
    spellings of one URL. X is the exception and matches on the handle instead — see
    `_record_coverage`, which has the handle but not the numeric `rest_id` the X entity keys on."""
    from . import oracle_refresh_state as st

    stype, url = rec.get("type") or "", rec.get("url") or ""
    if not url:
        return None
    if stype == "blog":
        return derive.blog_entity_id(url)
    if stype == "substack":
        return derive.substack_entity_id(publication_url=url)
    if stype == "github":
        owners = st.github_owners_from_links([url])
        return f"github:{owners[0]}" if owners else None
    return None


def _record_coverage(conn, oracle_id: str, results: list, *, x_since, web_since,
                     scholar_since=None) -> None:
    """Stamp the refresh registry for every source that ACTUALLY returned on this run.

    This is the write that used to be `schema.set_oracle_window`, moved to the grain the fact
    lives at. The Oracle-level window was written unconditionally, after a try/except that
    swallowed a failed X pull, and `seed_from_entities` copied it into every pair — so one
    rate-limited timeline produced a person with no X atoms whose row claimed a complete X pull,
    and `upsert_source`'s COALESCE made that claim permanent. A per-source stamp cannot say that:
    the pull that did not happen is the pull that does not write.

    `covered_from` is the window this run REACHED, per source — and a source that MEASURED its own
    reach says so on its record, which is what this reads first. The X timeline is the one that
    does: since durable partial walks (2026-09-14) an X pull can land atoms without covering the
    window it was handed, so the asked window stopped being a safe stand-in for the reached one.
    `_walk_frontier` is what decides the value, including when it must be None.

    Everything else still falls back to the window the CALLER asked for — the archive `since` for
    substack/blog, the corpus `since` for openalex (None for a full archive or a whole corpus,
    which record no lower bound). Those adapters are bounded by a snapshot hash rather than by a
    meter, so asked and reached cannot diverge the way a metered walk's can.

    GitHub records none HERE, and only here: a sweep's reach is bounded by a repo count rather
    than by a date, so the caller cannot know it. The sweep reports its own frontier and
    `oracle_refresh._pull_pair` is what stores it."""
    from . import oracle_refresh_state as st

    # The HEAD, not the id the caller happens to hold. Both reads below already resolve it
    # (`entities_for_canonical` → `current_canonical`), and `seed_from_entities` registers the row
    # under it — so writing with an unresolved id updates no row and the stamp vanishes. That was
    # latent while this fired once at end-of-run; per-source stamping runs it immediately after the
    # `resolve.resolve_entities` call that can MOVE the head, which is exactly when it bites.
    oracle_id = schema.current_canonical(conn, oracle_id)
    by_entity = st.pairs_by_entity(conn, oracle_id)
    _, who_ids = st.pairs_for_oracle(conn, oracle_id)
    # X pairs key on the bare handle, and nothing lowercases it on the way in, so fold here.
    x_pairs = {key.lower(): key for etype, key in by_entity.values() if etype == "x"}
    since_for_type = {"x": x_since, "blog": web_since, "substack": web_since, "github": None,
                      "openalex": scholar_since}

    for rec in results:
        if rec.get("action") != _OBSERVED:
            continue
        stype = rec.get("type") or ""
        if stype == "openalex":
            # Keyed on the bare author id, which is both the entity suffix and the pair key —
            # so the pair is found directly rather than through `_result_entity_id`, which reads
            # a url shape this record does not have.
            aid = (rec.get("url") or "").rstrip("/").rsplit("/", 1)[-1]
            pair = ("openalex", aid) if aid else None
        elif stype == "x":
            handle = (rec.get("url") or "").rstrip("/").rsplit("/", 1)[-1].lstrip("@").lower()
            key = x_pairs.get(handle)
            pair = ("x", key) if key else None
        else:
            pair = by_entity.get(_result_entity_id(rec) or "")
        if not pair:
            # No registered pair — an affiliation, an unsupported type, or an entity `resolve` has
            # not folded in yet. Recording nothing means the pair is re-pulled once it appears,
            # which is the safe direction.
            continue
        # What the pull REACHED when it measured that itself, otherwise the window it was ASKED
        # for. Presence of the key is the test, never truthiness — see the note where the X
        # record sets it. The two agreed for as long as a walk was all-or-nothing; the moment a
        # partial walk can write atoms, the asked window becomes a lie, and `covered_from` only
        # ever widens, so it is a lie that nothing revisits.
        if "covered_from" in rec:
            reached = rec["covered_from"]
        else:
            since = since_for_type.get(pair[0])
            reached = since.isoformat() if since else None
        st.record_pull(conn, st.SourceRow(oracle_id, pair[0], pair[1]),
                       last_status=rec["action"], stamp=True,
                       cursor_ts=st.latest_atom_ts(conn, pair[0], who_ids),
                       covered_from=reached)


def _coverage_report(conn, oracle_id: str) -> dict:
    """How far back we ACTUALLY hold this person, per source — the answer to the only question a
    user asks about a backfill.

    Distinct from `lookback`, which reports the window this run ASKED for. The two agree when
    everything returned and diverge exactly when something did not. A rate-limited X pull remains
    queued for the next scheduled run. A newly discovered X profile has no row here until the user
    connects X; a row from an earlier connection remains as history."""
    from . import oracle_refresh_state as st

    out = {}
    for r in st.list_sources(conn, canonical_ids=[oracle_id]):
        out[f"{r.source_type}:{r.source_key}"] = {
            "covered_from": r.covered_from,
            # NULL `last_pulled_at` after an ingest means this source did not answer. It is
            # already queued: infinitely stale, so the refresh rail takes it first next session.
            "pulled": r.last_pulled_at is not None,
            "queued": r.last_pulled_at is None,
        }
    return out


def _ask_which_one_they_meant(out: dict, reference: str, root_entity: str) -> None:
    """A URL POINTING AT ONE PAGE, about to subscribe them to the whole site — say so first.

    `url_canon.canonical_identity` collapses every URL on a host to that host, which is correct
    and is the squatter-defence primitive: `gajesh.com/blog/a-post` and `gajesh.com` name one
    trust unit, so nobody inherits a site's trust by deep-linking into it. But the same collapse
    means a user who hands over ONE article they liked gets a standing subscription to everything
    that site ever publishes, and the preview named only the root — never the difference.

    The two readings are genuinely different products, and the verb does not separate them:
    "add gajesh.com/blog/that-post" is in `hopper`'s trigger vocabulary (keep / save / add) AND
    on the roster side of `hopper`'s own bar ("a person worth keeping belongs on the roster").
    Only the user knows which they meant, and this is the last moment to ask — confirm mints the
    Oracle and starts refreshing it forever.

    Root URLs say nothing extra: handing over a bare domain is already unambiguous.
    """
    from urllib.parse import urlparse
    try:
        path = (urlparse(reference).path or "").strip("/")
    except Exception:
        return
    if not path:
        return
    site = root_entity.split(":", 1)[-1]
    out["ambiguous_scope"] = {"pointed_at": "one page", "would_follow": site}
    out["note"] = (
        f"They gave a link to ONE page, and confirming follows all of {site} from now on — every "
        f"post it has and every post it publishes later. ASK WHICH THEY MEANT BEFORE CONFIRMING, "
        f"in one line: follow {site} from now on, or just keep this one page? If they want the "
        f"one page, do NOT confirm — save it instead, which stores that page and subscribes them "
        f"to nothing.")


def _preview(conn, reference: str, match: dict | None,
             x_lookback: str | None, web_lookback: str | None,
             scholar_lookback: str | None = None) -> dict:
    """confirm=False → RESOLVE-ONLY preview: who is this + what confirm=True would do. NO writes.
    An unresolvable reference returns `unresolved` (nothing to confirm) — that IS the guard that
    makes silently ingesting a hallucinated handle impossible.

    COUNT-FIRST on BOTH axes for a person already in the roster with an OpenAlex id, each one
    free `group_by` call: by YEAR, so the lookback question reads "928 papers; the last 2 years is
    28" instead of naming bare presets, and by SUBJECT, so the user can narrow a 174-topic record
    to the subjects they actually want. Only for an existing match — a brand-new reference has no
    cluster to count against yet, and this phase writes nothing that would give it one.

    The two are asked together because they multiply: the year counts are reported THROUGH the
    stored topic filter, so a user who narrows first sees the window numbers for the pull that
    will run rather than for the whole record."""
    cid = match["canonical_id"] if match else None
    counts = scholar_year_counts(conn, cid) if cid else None
    topic_counts = scholar_topic_counts(conn, cid) if cid else None
    from pipeline.ingestion.x_graphql import has_managed_x_session

    x_connected = has_managed_x_session()
    x_since = _x_since(x_lookback) if x_connected else None
    lookback = _lookback_report(x_since, _web_since(web_lookback),
                                _scholar_since(scholar_lookback), counts)
    if not x_connected:
        lookback.pop("x", None)
        lookback.pop("x_since", None)
    base = {"confirm_required": True, "reference": reference,
            "lookback": lookback,
            "on_confirm": "confirm=True confirms this Oracle and ingests what they carry: their "
                          "papers when they have an OpenAlex record, and — when they have accounts "
                          "to discover — their verified footprint and, when X is connected, their "
                          "X timeline. The X pull itself is free (your own browser session); embedding and the content "
                          "gate are metered."}
    if topic_counts:
        base["scholar_topics"] = topic_counts

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
        # The OpenAlex lookup runs HERE too, not only at confirm, because this preview is the
        # consent surface: telling the user "blog" and then minting a venue would make the preview
        # a lie about a pull that is orders of magnitude larger.
        venue = _openalex_root(reference)
        if venue:
            from .ingest_scholar_footprint import MAX_WORKS_PER_PULL
            oid = venue["openalex_id"]
            # An author and a venue mint the same entity and run the same pull, but they need
            # DIFFERENT consent copy: 2,000 works is an order of magnitude past the most prolific
            # real researcher measured (Frances Arnold, 928) and a live truncation on a venue
            # (ChemRxiv, 63,565). Only one of the two is about to lose most of its corpus.
            author = oid.startswith("A")
            note = (f"That is a researcher's OpenAlex record — OPYT pulls their papers by that id, "
                    f"with real authors and DOIs. It carries {venue.get('works') or 'many'} works. "
                    f"Subjects narrow it and the choice PERSISTS into every later refresh."
                    if author else
                    f"That is a research venue, not a personal site — OPYT reads it "
                    f"through OpenAlex ({oid}), so papers arrive with "
                    f"their real authors and DOIs instead of being scraped as generic "
                    f"articles. It carries {venue.get('works') or 'many'} works. ASK "
                    f"WHICH SUBJECTS BEFORE CONFIRMING — confirm=True with no "
                    f"`scholar_topics` pulls the venue unfiltered and stops at "
                    f"{MAX_WORKS_PER_PULL} papers, which is a truncation, not a "
                    f"selection.")
            out = {**base, "mode": "new",
                   "resolved": {"reference": reference, "name": venue.get("name"),
                                "root_entity": f"openalex:{oid}",
                                "platform": "openalex_author" if author else "openalex_venue",
                                "works": venue.get("works")},
                   "note": note}
            # The topic ask is keyed on the OPENALEX id, not a cluster: this root has no entity and
            # no `oracle_sources` row until confirm, and asking after the first pull is too late.
            topics = topic_counts_for_id(oid)
            if topics:
                out["scholar_topics"] = topics
            return out
        eid = _url_entity_id(reference)
        out = {**base, "mode": "new",
               "resolved": {"reference": reference, "root_entity": eid,
                            "platform": "substack" if eid.startswith("substack:") else "blog"}}
        _ask_which_one_they_meant(out, reference, eid)
        return out
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
            "deterministic probes miss: a personal blog or website, or a Substack. "
            "Include a GitHub profile if it uses a different handle. "
            "Exclude LinkedIn, podcasts and YouTube. X/Twitter and Semantic Scholar "
            "are already covered by probes. Return home pages, not "
            "individual posts. You do NOT need to verify ownership; the trust graph "
            "re-checks every URL and will reject any that cannot be corroborated."
        ),
        "feed_back_via": f"add_oracle({reference!r}, confirm=True, extra_source_urls=[...])",
        # Said explicitly: a dropped URL is invisible, while a rejected one shows up in
        # needs_review — so the trust graph should judge, not the host filter beforehand.
        "note": "Send everything plausible. Unverifiable URLs are rejected, not trusted.",
    }


def add_oracle(conn, embedder, reference: str, *, confirm: bool = False,
               x_lookback: str | None = None, web_lookback: str | None = None,
               scholar_lookback: str | None = None, scholar_topics=None,
               extra_source_urls: list[str] | None = None, force: bool = False) -> dict:
    """Atom-native "add a person": admit an Oracle + expand their verified footprint into atoms.

    `reference` is polymorphic — an X @handle, a Substack/blog/site URL, or a canonical_id (Mode C:
    promote a below-the-cut SCREEN pick). Two-phase, gated by `confirm`:
      • confirm=False (default) → RESOLVE-ONLY preview: verify the reference resolves + show who it
        is + which platforms will expand. Writes NOTHING (a hallucinated handle has nothing to
        confirm). Cheap — no discovery.
      • confirm=True → the full chain: resolve → confirm (write the oracles row) → ingest →
        re-resolve. WRITES. The ingest pulls their papers when they have an OpenAlex record, and
        runs discover → trust-filter → footprint atoms + X timeline when X is connected. A
        researcher with only papers gets the first and not the second.

    When X is connected, `x_lookback`, `web_lookback` and `scholar_lookback` are separate knobs on
    purpose — they bound an ephemeral stream, a durable web archive and a published paper corpus,
    and a single value can only be right for one of them. X defaults to ~6 months, web to the full
    archive, and papers to the whole corpus. Without X, its selector is ignored rather than
    becoming a question the user did not choose. The preview reports the paper corpus with REAL
    COUNTS when it can, so the choice is made against numbers rather than presets.

    `scholar_topics` bounds the paper corpus by SUBJECT rather than by date, and it PERSISTS: a
    list of OpenAlex topic ids (from the preview's `scholar_topics`) narrows every future refresh
    too, `[]` clears it, and None leaves an existing choice alone. That is what stops a
    174-subject author record — or a 63,000-work venue — arriving whole.

    Name resolution ('find Karpathy') is the HOST's job — it turns a name into a @handle/URL and
    calls this; there is no name-search endpoint (fuzzy resolution deferred by design)."""
    reference = (reference or "").strip()
    if not reference:
        return {"error": "add_oracle needs a reference — an X @handle, a Substack/blog URL, or a "
                         "canonical_id from oracle(action='screen')."}

    match = _match_local_roster(conn, reference)         # network-free dedup / Mode C
    if not match:
        # Only when the reference is NEW: an already-known person resolves to their real cluster
        # and needs no rooting, whatever URL shape the caller happened to reach for.
        unsupported = _unsupported_root(reference)
        if unsupported:
            return {"error": f"cannot add {reference!r} — {unsupported}. Nothing was written."}

    if not confirm:                                      # ── Phase 1: preview (no writes) ──
        return _preview(conn, reference, match, x_lookback, web_lookback, scholar_lookback)

    # ── Phase 2: the chain ──────────────────────────────────────────────────────
    already = bool(match) and schema.is_oracle(conn, match["canonical_id"])

    # 1+2. RESOLVE → canonical_id, then CONFIRM (write the oracles row).
    if match:
        cid, source = match["canonical_id"], "screen"
    else:
        cid = _resolve_handle(conn, reference)           # mints + resolves + the freeform vouch
        if not cid:
            return {"error": f"could not resolve {reference!r} — check the handle/URL. "
                             "Nothing was written.", "unresolved": [reference]}
        source = "freeform"

    # BEFORE the write, and on BOTH branches. `_unsupported_root` above cannot cover this — it is a
    # pure function of the reference string, while single-vs-multi authorship needs a fetch, an LLM
    # call and the cache `conn` holds. The `match` branch needs it too: the FIRST refused attempt
    # leaves the `blog:{host}` entity `_resolve_handle` minted behind, so a second call matches the
    # local roster and would walk straight past a check placed only on the `else` branch. The
    # verdict is cached forever per site, so that second call pays nothing.
    why = _multi_author_refusal(conn, cid, (match or {}).get("name") or reference)
    if why:
        return {"error": f"cannot add {reference!r} — {why}", "refused": [reference]}
    schema.upsert_oracle(conn, cid, name=(match.get("name") if match else _name_for(conn, cid)),
                         source=source)
    # The OTHER door onto `oracles`. `confirm` is not the only mint, so it cannot be the only
    # producer — this path is how a freeform URL becomes an Oracle, and it is exactly the one
    # whose report promises the user that a deferred X timeline "resumes: next-scheduled-run".
    refresh_queued = activate_refresh()

    # 3+4. Ingest + seed (the shared engine). THREE windows, each against its OWN preset dict.
    oracle = _oracle_for(conn, cid)
    web_since = _web_since(web_lookback)
    scholar_since = _scholar_since(scholar_lookback)
    from pipeline.ingestion.x_graphql import has_managed_x_session
    x_connected = has_managed_x_session()
    if not x_connected:
        x_since = None
    elif x_lookback == X_SINCE_LAST:
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
                            scholar_since=scholar_since, scholar_topics=scholar_topics,
                            extra_source_urls=extra_source_urls, force=force)

    # 5. RE-RESOLVE — fold the new substack:/blog:/github: rows into the canonical cluster.
    resolve.resolve_entities(conn)

    out = {
        "added": {"canonical_id": schema.current_canonical(conn, cid),
                  "name": oracle.get("name"), "source": source, "was_already_oracle": already,
                  # Whether the "resumes: next-scheduled-run" this report goes on to promise has
                  # a scheduled run to resume ON. False means consent was declined or the queue
                  # refused the write — either way, do not tell the user it continues by itself.
                  "refresh_queued": refresh_queued},
        # What actually ran, derived from the resolved datetimes — including the X clamp nobody
        # asked for. Tell the user this; it is the consent surface for the pull.
        #
        # `scholar_since` was MISSING from this call until 2026-09-08, so a user who passed
        # `scholar_lookback='2yr'` was told "whole corpus" by the very report the consent
        # invariant rests on. The counts ride along too, read back AFTER the topic filter was
        # stored, so the numbers describe the pull that just ran rather than the whole record.
        "lookback": _lookback_report(x_since, web_since, scholar_since,
                                     scholar_year_counts(conn, cid)),
        "ingest": ingest,
    }
    if not x_connected:
        out["lookback"].pop("x", None)
        out["lookback"].pop("x_since", None)
    # Probe 5's push leg, asked only when it can pay for itself. Two gates:
    #  • `extra_source_urls` — the host just answered; asking again would loop forever.
    #  • `discovery_ran_fresh` — a cache hit means nothing changed since last look, so a search
    #    can only return what's already in the result (also covers the no-rootable-profile early
    #    return, where discovery never ran).
    if not extra_source_urls and ingest.get("discovery_ran_fresh"):
        out["followup"] = _open_web_followup(reference, oracle.get("name"))
    return out
