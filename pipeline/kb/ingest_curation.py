"""
pipeline/kb/ingest_curation.py — Step-2 Curation Pull: the user's own endorsement
signals into the atom-KB, feeding Stage-3 (entity resolution) and Stage-4 (candidate SCREEN).

Two flavors: CONTENT-BEARING (Substack saved-posts; X bookmarks live in ingest_x) writes a
content ATOM (`entry_mode='user-saved'`) + author ENTITY + `save` SIGNAL. PEOPLE-ONLY (X
following/Lists/likes/bookmarks; Substack follows, subscriptions and saved posts) writes only an
author ENTITY + SIGNAL (`follow`/`list`/`like`/`subscribe`/`save`) — no atom.

Both saved-content walks therefore run TWICE, by design: a signals-only pass over the list (free
of bodies, models and embeddings, so it can block the setup call and have the screen scored
before the candidate list is built) and a content pass that lands the atoms behind it. The
signals pass owns the count; the content pass may only assert presence.

Substack follows and Substack subscriptions are DIFFERENT GRAPHS read from different endpoints,
and each has its own collector. Measured 2026-09-08 on one real account: 36 subscriptions, 21
follows, overlapping by ONE.

Signals and atoms key on the same per-platform id so a person's signals unify before Stage-3
resolution: X → `x:user:{rest_id}` (matches `derive.derive_x`); Substack →
`derive.substack_entity_id(handle, publication_url)`.

This module is the ADAPTER: it funnels fetch logic (pipeline/ingestion/*) into the atom-KB
write contract (schema.add_signal / schema.upsert_entity / ingest_common.store_atom). It does
not reimplement scraping or re-derive the write path.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pipeline.timeparse import utc_now

from . import derive, schema
from .embed import assert_model
from .ingest_common import (AtomSink, BASIS_OBSERVED, BASIS_STATED, BODY_ABSENT, BODY_COMPLETE,
                            BODY_PARTIAL, BODY_PENDING, StageTimer,
                            body_fields, llm_run_marker, llm_run_stats,
                            snapshot_and_hash)

# ── People-only stampers (entity + signal, NO atom) ──────────────────────────────

def _person_profile(cand: dict) -> dict | None:
    """Extracts the Stage-4 kind-classify inputs from a normalized X user (bio, verified,
    followers, handle) for storage on the entity's `profile` blob. Returns None when nothing
    classify-worthy is present.

    ⚠️ TRUTHY keys only, so `verified=False` and `followers_count=0` are DROPPED, not stored.
    `screen._classify_prompt` therefore cannot distinguish "unverified" from "unknown" — it
    only ever learns the positive case. Changing this changes what a live LLM classifier is
    told about most accounts, which nothing here can measure, so it stays as measured rather
    than as intended."""
    prof = {k: cand[k] for k in ("bio", "verified", "followers_count", "handle") if cand.get(k)}
    if "followers_count" in prof:                       # normalize the key the classifier reads
        prof["followers"] = prof.pop("followers_count")
    return prof or None


def _stamp_x_person(conn, cand: dict, signal_type: str, *, count: int = 1,
                    extra: dict | None = None) -> str:
    """UPSERTs the X author as an entity (seeding `identity_links` with the bio site for
    Stage-3 resolution, and `profile` with Stage-4 classify inputs), then records one signal on
    `x:user:{rest_id}`. Returns the id.

    Uses `set_signal`, not `add_signal`: all three callers hand us a person's whole aggregate
    for the run, so summing would double-count — see `schema.set_signal`."""
    eid = f"x:user:{cand['user_id']}"
    site = cand.get("site")
    schema.upsert_entity(conn, eid, name=cand.get("display_name"),
                         identity_links=[site] if site else None,
                         profile=_person_profile(cand))
    schema.set_signal(conn, eid, signal_type, "x", count=count, extra=extra)
    return eid


def sync_likes_signals(conn) -> dict:
    """The authors of tweets you liked → `like` signal (count = likes earned per
    author). No atoms — a liked tweet's content never enters the KB."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.x_likes import aggregate_authors, fetch_liked_authors

    session = core.x_session("https://x.com/i/likes")
    vid = core.viewer_id(session)
    if not vid:
        return {"source": "x-likes", "skipped": "no_viewer_id"}
    authors = fetch_liked_authors(vid, session, session)
    cands = aggregate_authors(authors, vid)
    for c in cands:
        _stamp_x_person(conn, c, "like", count=c["liked_count"])
    return {"source": "x-likes", "candidates": len(cands),
            "liked_tweets_with_author": len(authors)}


def sync_lists_signals(conn) -> dict:
    """Members of your owned Lists → `list` signal (count = breadth of list
    membership; extra carries the list names). No atoms."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.x_lists import (aggregate_members, fetch_list_members,
                                            fetch_owned_lists)

    session = core.x_session("https://x.com/i/lists")
    vid = core.viewer_id(session)
    if not vid:
        return {"source": "x-lists", "skipped": "no_viewer_id"}
    owned = fetch_owned_lists(session, session, vid)
    members_by_list = {l["id"]: fetch_list_members(l["id"], session, session) for l in owned}
    cands = aggregate_members(owned, members_by_list, vid)
    for c in cands:
        _stamp_x_person(conn, c, "list", count=len(c["list_names"]),
                        extra={"list_names": c["list_names"]})
    return {"source": "x-lists", "lists": len(owned), "candidates": len(cands)}


def sync_bookmark_signals(conn) -> dict:
    """Who you bookmarked → `save` signal (count = how many of their posts you saved). No atoms.

    THE LIST ONLY, and that is the whole point. The Bookmarks walk is five requests for five
    hundred bookmarks with no sleeps and no per-item call, so it belongs beside the other four
    free collectors in the blocking setup pass. Its expensive siblings do not: the thread read is
    one serial `TweetDetail` per item against a 150/15-min bucket, and the body write cannot
    happen without an embedding round-trip (`AtomSink.flush` embeds BEFORE it writes, so there is
    no atom without vectors). Both stay in `ingest_x.sync_bookmarks`, off this path.

    ⚠️ WHY THIS EXISTS AT ALL. `save` was reachable ONLY through the content arm, which runs on
    the `bookmark_catchup` rail, which only the resident worker launches — so on an install with
    no worker the row queued at consent was never claimed and the signal never landed. That is not
    a late import, it is no import: the screen scored every candidate on one signal, nothing
    cleared `screen.CORROBORATION_MIN`, and the user was told "each of these showed up once".
    It bites Substack harder still — subscribe and follow are near-disjoint graphs (0 of 36
    overlap on the live store), so there a `save` is the only realistic second signal anybody has.
    `sync_substack_saved_signals` is that platform's twin of this function, built 2026-09-13.

    COUNTING AUTHORITY. `set_signal`, like its four peers, because this IS a full-set re-read and
    the aggregate is a person's whole count rather than a delta. An interrupted walk is NOT that,
    so it degrades to `ensure_signal`: a partial aggregate written with `set_signal` would REPLACE
    a correct count with a smaller one, which is the one way this could destroy information.
    """
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.utils import log
    from pipeline.ingestion.x_graphql import iterate_bookmarks

    me = core.viewer_id(core.x_session("https://x.com/i/bookmarks"))
    if not me:
        return {"source": "x-bookmark-signals", "skipped": "no_viewer_id"}
    by_user: dict[str, dict] = {}
    walked = 0
    complete = True
    try:
        for norm in iterate_bookmarks(limit=0):
            a = norm.get("author") or {}
            uid = str(a.get("id") or "")
            if not uid or uid == me:              # never treat yourself as a candidate
                continue
            walked += 1
            rec = by_user.get(uid)
            if rec is None:
                rec = {"user_id": uid, "handle": a.get("userName"),
                       "display_name": a.get("name") or a.get("userName"),
                       "site": a.get("site") or None, "saved_count": 0}
                by_user[uid] = rec
            rec["saved_count"] += 1
    except Exception as e:
        # What the pages that DID answer showed is real and is kept — the same rule the bookmark
        # walk's own skip paths follow. What is lost is the right to claim a total, so every
        # author below is written as presence rather than as a count.
        complete = False
        log(f"[curation] bookmark list walk stopped early ({walked} seen, counts not final): {e}")

    for cand in by_user.values():
        eid = f"x:user:{cand['user_id']}"
        site = cand.get("site")
        schema.upsert_entity(conn, eid, name=cand.get("display_name"),
                             identity_links=[site] if site else None,
                             profile=_person_profile(cand))
        write = schema.set_signal if complete else schema.ensure_signal
        write(conn, eid, "save", "x", count=cand["saved_count"])

    out = {"source": "x-bookmark-signals", "bookmarks": walked, "candidates": len(by_user)}
    if not complete:
        out["undetermined"] = 1       # `classify_run`'s BLOCKED — the extent was not established
    return out


def sync_following_signals(conn) -> dict:
    """The accounts you follow → `follow` signal. Free cookie-scrape via
    `x_graphql_core.fetch_following`. No atoms."""
    from pipeline.ingestion import x_graphql_core as core

    session = core.x_session("https://x.com/following")
    vid = core.viewer_id(session)
    if not vid:
        return {"source": "x-following", "skipped": "no_viewer_id"}
    users = core.fetch_following(session, session, vid)
    for u in users:
        _stamp_x_person(conn, u, "follow")
    return {"source": "x-following", "following": len(users)}


def _migrate_substack_follow_signals(conn) -> None:
    """ONE-SHOT rename of the misnamed Substack follow signal, 2026-09-08.

    `subscriber-lists?lists=following` returns FOLLOWS and was stamped `subscribe`, so
    `screen.reflect()` told the user "you subscribe" about 21 people they merely followed.
    Renames `('substack','subscribe')` → `('substack','follow')` and the clock row
    `substack_subs` → `substack_follows`.

    NOT a connect-hook migration, and that is the whole design. `schema.init_kb_schema` holds
    read-guarded renames that are safe to re-run forever because their old value never comes back
    with a NEW meaning. This one's does: from `sync_substack_subscriptions` onward,
    `('substack','subscribe')` is a REAL subscription, and a rename re-running on every connect
    would eat every one of them, every time. So it fires exactly once and the marker is the clock
    row it creates: after this runs, `collector_runs` holds `substack_follows`, and the guard
    below never opens again.

    It lives here rather than in `schema` because this module owns both writes — the signal type
    and the collector key are its own — and `schema` owns neither.

    `UPDATE OR REPLACE` on the signals: a store can already hold a `('substack','follow')` row for
    the same entity if an OLDER build on the primary checkout wrote one between passes, and the
    primary key is (entity_id, signal_type, platform). Both rows come from the same full-set walk
    of the same endpoint, so the later write winning loses nothing.

    KNOWN AND ACCEPTED, because it closes on merge: while an older build still runs against this
    store, its follow walk keeps writing `('substack','subscribe')` rows that this migration will
    not fire again for. Those read back as subscriptions. A publication the user actually
    subscribes to is corrected by the next subscription walk (`set_signal` replaces); a
    follow-only one is not, and stays wrong until that build is gone.
    """
    from . import curation_state
    from pipeline.ingestion.utils import log

    curation_state.init_state_schema(conn)
    if curation_state.get_run(conn, "substack_follows") is not None:
        return                                   # converged — see the docstring
    if curation_state.get_run(conn, "substack_subs") is None:
        # A store that has never run the follow collector has nothing to rename and no clock row
        # to mark convergence with. Renaming its signals anyway would be a no-op; skipping keeps
        # this a single atomic step rather than two halves that can each be half-done.
        return
    renamed = conn.execute(
        "UPDATE OR REPLACE curation_signals SET signal_type = 'follow' "
        " WHERE platform = 'substack' AND signal_type = 'subscribe'").rowcount
    conn.execute("UPDATE OR REPLACE collector_runs SET collector = 'substack_follows' "
                 " WHERE collector = 'substack_subs'")
    conn.commit()
    log(f"[curation] migrated {renamed} Substack signal(s) from 'subscribe' to 'follow' — "
        "the subscriber-lists endpoint returns follows, not subscriptions")


def sync_substack_follows(conn, *, profile: str | None = None) -> dict:
    """The people you FOLLOW on Substack → `follow` signal. The subscriber-lists endpoint
    returns only {name, url} (no handle), so the entity keys on the publication subdomain. No
    atoms.

    NOT your subscriptions. That is `sync_substack_subscriptions` below, reading a different
    endpoint, and the two graphs overlapped by ONE publication out of 36 on the measured account.
    This collector was called `sync_substack_subs` and stamped `subscribe` until 2026-09-08.

    Goes through `follow_source`, never a cookie reader: on a hosted home the same list arrives
    from Chrome's own signed-in page, and picking the transport here would put that choice in two
    places.

    A REFUSED read reports `skipped`, never `follows: 0`. `_stamp_run` derives the clock's
    status from that key, so without it a Cloudflare 403 is recorded as a SUCCESSFUL walk that
    found nobody — and `found=0, status=ok` is what `screen` and `onboard` both read as fact.
    Measured 2026-09-08: substack.com refused every reader route for over half an hour, and on a
    fresh install `onboard` responded by telling the user their sign-in had gone stale and to
    reconnect. Reconnecting cannot fix a rate limit. Same fix, same reasoning, as
    `sync_substack_saved`'s refused saved list."""
    from pipeline.ingestion.sources.substack import SubstackListingError, follow_source

    from pipeline.ingestion.utils import log

    # BEFORE the fetch, so a store whose Substack session is dead still gets its signals renamed.
    _migrate_substack_follow_signals(conn)
    try:
        people = follow_source(profile).follows()
    except SubstackListingError as e:
        log(f"[curation] substack follow-list REFUSED — recording a skip, not an empty list: {e}")
        return {"source": "substack-follows", "skipped": "refused", "detail": str(e)}
    for s in people:
        url = s.get("url") or ""
        eid = derive.substack_entity_id(None, url)   # this API drops the handle → subdomain id
        schema.upsert_entity(conn, eid, name=s.get("name"),
                             identity_links=[url] if url else None)
        # `set_signal`, not `add_signal`: this walks the whole Following list every run.
        schema.set_signal(conn, eid, "follow", "substack")
    return {"source": "substack-follows", "follows": len(people)}


def _is_paid(membership_state: str) -> bool | None:
    """`membership_state` → the money claim `screen.reflect()` renders, or None for "unknown".

    THREE-WAY on purpose, and the None arm is the important one. `reflect()` prints
    "you subscribe (paid)" only for True and falls back to a claim-free "you subscribe" for None,
    so an unrecognised state can never invent a payment. `unsubscribed` lands there too: it is
    neither a live paid relationship nor a live free one.

    ⚠️ The True arm is INFERRED, not measured. All 36 subscriptions on the account this was built
    against are `free_signup`; no `subscribed` row has ever been read. The mapping comes from
    Substack's own frontend, where `membership_state` is the paid/free discriminator. The raw
    state is stored on the signal's `extra` so the first real one can be audited without another
    request to a Cloudflare-guarded host — check it before trusting the phrase.

    `is_founding` is not consulted: founding implies `subscribed`, so it refines a fact already
    captured and no reader distinguishes the two."""
    if membership_state == "subscribed":
        return True
    if membership_state == "free_signup":
        return False
    return None


def sync_substack_subscriptions(conn, *, profile: str | None = None) -> dict:
    """The publications you SUBSCRIBE to → `subscribe` signal, carrying whether you pay. No atoms.

    The read that makes a Substack reader visible without any Notes activity. Measured 2026-09-08
    on one real account: 36 subscriptions against 21 follows, overlapping by ONE — so this is not
    a better version of `sync_substack_follows`, it is the other graph, and dropping either loses
    real endorsements.

    A fifth `CollectorSpec` rather than a second read inside the follow collector, because
    `CollectorSpec` carries exactly one `signal_type` and one `found_key`: two reads under one
    spec would make `stored_after` count half of what the collector did, and would put both
    request patterns on one clock row so a 403 on either marks both stale.

    Entity keying is unchanged and that is deliberate. The payload carries no author handle — the
    same limitation `subscriber-lists` has — so `derive.substack_entity_id(None, url)` keys on the
    subdomain, or on the HOST for a custom domain (20 of 36 measured). Resolving `author_id` to a
    handle would cost one `public_profile` request per publication against the host whose rate
    limit is the whole constraint; Stage-3 resolves the split through the publication URL instead.

    A REFUSED read reports `skipped`, never `subscriptions: 0` — the third rail to need that rule,
    for the reason `sync_substack_follows` records above."""
    from pipeline.ingestion.sources.substack import (SubstackListingError, fetch_subscription_list,
                                                     subscription_list_source)

    from pipeline.ingestion.utils import log

    try:
        subs = fetch_subscription_list(subscription_list_source(profile))
    except SubstackListingError as e:
        log(f"[curation] substack subscription list REFUSED — recording a skip, not an empty "
            f"list: {e}")
        return {"source": "substack-subscriptions", "skipped": "refused", "detail": str(e)}
    for s in subs:
        url = s.get("url") or ""
        eid = derive.substack_entity_id(None, url)
        schema.upsert_entity(conn, eid, name=s.get("name"),
                             identity_links=[url] if url else None)
        state = s.get("membership_state") or ""
        # `is_favorite` is stored and nothing reads it — 0 of 36 on the measured account, so a
        # rank weight for it would be a branch with no producer. One key on a record already
        # being written costs nothing; the consumer waits for a populated one.
        schema.set_signal(conn, eid, "subscribe", "substack",
                          extra={"is_paid": _is_paid(state), "membership_state": state,
                                 "is_favorite": s.get("is_favorite")})
    return {"source": "substack-subscriptions", "subscriptions": len(subs)}


def sync_substack_saved_signals(conn, *, profile: str | None = None) -> dict:
    """Who you saved posts FROM → `save` signal (count = how many of their posts you saved). No
    atoms. The Substack half of what `sync_bookmark_signals` is for X.

    ⚠️ IT MATTERS MORE HERE THAN ON X, and the numbers say so. X can corroborate a person three
    other ways (`list`, `follow`, `like`) so a missing `save` costs a count. Substack has three
    signal types, one of which is UNREADABLE by construction — liking is a forward write with no
    reverse index (`2026-09-08-substack-interaction-surface-map.md`; do not go looking again) —
    and the other two are near-disjoint graphs: 36 subscriptions against 20 follows overlapping
    by ZERO on the live store, 2026-09-13. So a `save` is the only realistic second signal anybody
    on Substack has, and it reached the store only through the content arm, which runs on the
    `substack_saved_catchup` rail, which only the resident worker launches. With no worker, 56
    Substack entities and not one corroborated.

    THE LIST ONLY. `derive.derive_substack` reads nothing but the list record — the body supplies
    exactly one field, `body_html` — so a signal needs no per-post fetch, no model and no
    embedding round-trip, which is the property `COLLECTOR_SPECS` selects for. It is not free the
    way X's five-request walk is: the reader endpoint is Cloudflare-guarded, so the walk sleeps 1s
    per page (~10s for a 500-post list). Still seconds, still blocking-safe — but measure before
    promising a number.

    Goes through `saved_source`, never a cookie reader, for the reason both account collectors
    above state.

    COUNTING AUTHORITY, and here it is not free: `fetch_saved_posts` has three truncation exits
    that all return normally, so `complete` is carried back explicitly (see `SavedPosts`). A full
    walk is this person's whole aggregate → `set_signal`. A truncated one is NOT, so it degrades
    to `ensure_signal`: writing a partial aggregate with `set_signal` would REPLACE a correct
    count with a smaller one, the one way this can destroy information.

    YOUR OWN posts are not dropped, unlike X's walk. A saved record's author is a PUBLICATION id,
    not a user id, so there is no equality test against `own_user_id(cookies)` to make — and the
    failure mode is one harmless self-candidate on the screen, for a thing people rarely do."""
    from pipeline.ingestion.sources.substack import (SubstackListingError, fetch_saved_posts,
                                                     saved_source)
    from pipeline.ingestion.utils import log

    try:
        recs, complete = fetch_saved_posts(saved_source(profile))
    except SubstackListingError as e:
        # A REFUSED list reports `skipped`, never `candidates: 0` — the fourth rail to need that
        # rule, for the reason `sync_substack_follows` records above.
        log(f"[curation] substack saved-list REFUSED — recording a skip, not an empty list: {e}")
        return {"source": "substack-saved-signals", "skipped": "refused", "detail": str(e)}

    by_pub: dict[str, dict] = {}
    for rec in recs:
        # The SAME derivation the content arm uses, deliberately: both sides must key a
        # publication identically or one person arrives as two candidates carrying one signal
        # each — below the >=2-signal bar, filtered out before a human sees them.
        meta = derive.derive_substack(rec)
        agg = by_pub.get(meta["who_id"])
        if agg is None:
            agg = {"name": meta.get("who_name"), "site": meta.get("who_site"), "saved_count": 0}
            by_pub[meta["who_id"]] = agg
        agg["saved_count"] += 1

    for eid, agg in by_pub.items():
        site = agg.get("site")
        schema.upsert_entity(conn, eid, name=agg.get("name"),
                             identity_links=[site] if site else None)
        write = schema.set_signal if complete else schema.ensure_signal
        write(conn, eid, "save", "substack", count=agg["saved_count"])

    out = {"source": "substack-saved-signals", "saved_posts": len(recs),
           "candidates": len(by_pub)}
    if not complete:
        out["undetermined"] = 1       # `classify_run`'s BLOCKED — the extent was not established
    return out


# ── Substack saved-posts → FULL-BODY content atoms + `save` signal ───────────────

def _clean_body_html(body_html: str) -> str:
    """Converts a Substack `body_html` fragment (already de-boilerplated) to clean markdown for
    chunking. Runs both trafilatura and html2text and keeps the longer result, since trafilatura
    can silently under-extract a bare fragment into a thin "invisible" atom. Empty in → "" out
    (caller falls back to the stub). Images are kept."""
    if not body_html:
        return ""
    # Imported here, not at module load: both are heavy and this is the only function that
    # needs them. They are hard dependencies, so the import cannot fail — only the PARSE can,
    # which is why each extractor gets its own guard and the other still gets its turn.
    import html2text
    import trafilatura
    from pipeline.ingestion.utils import log

    cands: list[str] = []
    try:
        md = trafilatura.extract(body_html, output_format="markdown",
                                 include_links=True, include_images=True, no_fallback=False)
        if md:
            cands.append(md)
    except Exception as e:
        log(f"[curation] trafilatura extract failed ({type(e).__name__}); html2text still runs")
    try:
        h = html2text.HTML2Text()
        h.ignore_links, h.ignore_images, h.body_width, h.unicode_snob = False, False, 0, True
        t = h.handle(body_html)
        if t:
            cands.append(t)
    except Exception as e:
        log(f"[curation] html2text extract failed ({type(e).__name__})")
    return max(cands, key=len).strip() if cands else ""


def _yaml_dq(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _saved_atom_markdown(rec: dict, body_md: str) -> str:
    """Builds snapshot markdown for a saved post: provenance frontmatter + the real body, or the
    stub preview when the full body couldn't be fetched. The chunker strips the frontmatter, so
    the searchable surface is the body/preview only."""
    title = (rec.get("title") or "Untitled").strip()
    subtitle = (rec.get("subtitle") or "").strip()
    url = rec.get("url") or ""
    date_str = derive._iso_day(rec.get("post_date", ""))
    handle = (rec.get("author_handle") or "").strip()
    author = f"@{handle}" if handle else (rec.get("publication_name") or "substack")
    author_name = rec.get("author_name") or rec.get("publication_name") or ""
    body = (body_md or "").strip() or (rec.get("preview") or "").strip()

    fm = (
        "---\n"
        "source: substack\n"
        f'author: "{_yaml_dq(author)}"\n'
        f'author_name: "{_yaml_dq(author_name)}"\n'
        f'publication: "{_yaml_dq(rec.get("publication_name", ""))}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"saved_at: {rec.get('saved_at', '')}\n"
        f"audience: {rec.get('audience', '')}\n"
        "type: article\n"
        "---\n\n"
    )
    out = f"# {title}\n\n"
    if subtitle:
        out += f"*{subtitle}*\n\n"
    if body:
        out += f"{body}\n\n"
    out += f"---\n*Saved from Substack · [Original post]({url})*\n"
    return fm + out


def sync_substack_saved(conn, embedder, *, profile: str | None = None) -> dict:
    """Your Substack "Saved posts" → full-body content atoms + a `save` signal.

    Per saved post: fetches the full body, cleans HTML to text, and chunks the body (a
    title+preview stub alone yields thin, invisible atoms). A paywalled/failed body falls back to
    the stub preview. Idempotent and cost-paced fail-safe: one bad post never starves the rest.

    Goes through `saved_source`, never a cookie reader: on a hosted home the same list arrives
    from Chrome's own signed-in page, and picking the transport here would put that choice in two
    places. Both Substack account collectors follow the same rule for the same reason."""
    from pipeline.ingestion.sources.substack import (SubstackFetchError, SubstackListingError,
                                                     _is_paywalled, fetch_saved_posts,
                                                     saved_source)
    from pipeline.ingestion.utils import log
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from .vision import enrich_markdown_images

    assert_model(conn, embedder)      # guard the store's embedding identity before spend
    # Marks the LLM latency baseline before the first paid call, so the summary reports this
    # run's calls, not the process-cumulative total.
    llm0 = llm_run_marker()
    source = saved_source(profile)
    # Stage-timed like the footprint adapters
    timer = StageTimer()
    try:
        with timer.stage("list_fetch"):    # one cursor-paginated walk of the saved list
            # `complete` is the signal collector's business, not this arm's: it never sets a
            # count, so a truncated walk changes nothing about what it may write.
            recs, _complete = fetch_saved_posts(source)
    except SubstackListingError as e:
        # A REFUSED list, not an empty one, and the whole pass has to say so. Reported rather
        # than raised because a raise sinks the other five sources in `curation_pull`; carrying
        # `undetermined` is what makes `classify_run` call it BLOCKED (Cloudflare lifts on its
        # own) instead of ERROR (needs a person). Measured 2026-09-07: without this the rail
        # reported `ok, added: 0` after four consecutive 403s, which is byte-identical to a week
        # in which the user saved nothing.
        log(f"[curation] substack saved-list REFUSED, nothing imported this pass: {e}")
        return {"source": "substack-saved", "added": 0, "skipped": 0, "stub_fallback": 0,
                "undetermined": 1, "failed": 0, "error": f"{type(e).__name__}: {e}",
                "total": schema.count_atoms(conn, "substack")}
    seen = schema.load_hashes(conn, "substack")
    # VLM descriptions cached by URL (immutable CDN links) → re-runs are free + hash-stable.
    img_cache = load_image_cache(opyt_home())

    # Batches the embed across saved posts, since essays are long (many chunks each). Fetch stays
    # serial — Cloudflare-rate-limited. `save` rides on_written so it's recorded only for atoms
    # that durably land; `added` counts durable writes, not submits.
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, timer=timer, flush_chunks=8 * bs)
    counts = {"added": 0}
    submitted = skipped = stub_fallback = undetermined = 0
    # Stubs whose body was BLOCKED (not absent) — the only atoms allowed past the `seen` skip.
    pending = schema.load_body_pending(conn, "substack")

    for rec in recs:
        post_id = rec.get("id")
        if not post_id:
            continue
        atom_id = f"substack:{post_id}"
        if atom_id in seen and atom_id not in pending:
            skipped += 1              # immutable saved artifact → skip before the paid fetch
            continue

        base, slug = rec.get("publication_url") or "", rec.get("slug") or ""
        body_md, got_body, blocked = "", False, False
        if base and slug:
            try:
                with timer.stage("body_fetch"):   # one GET per NEW saved post
                    full = source.full_post(base, slug)
                body_md = _clean_body_html((full or {}).get("body_html") or "")
                got_body = bool(body_md.strip())
            except SubstackFetchError as e:
                # A block, not an empty post: keep the stub but flag `body_state='pending'` so
                # the `seen` skip lets it retry next run instead of freezing a stale block.
                blocked = True
                undetermined += 1
                log(f"[curation] substack full-body BLOCKED for {slug!r} (stub kept, RETRYABLE): {e}")
            except Exception as e:     # per-post failure keeps the stub, never aborts the run
                log(f"[curation] substack full-body fetch failed for {slug!r}: {e}")
        # `paywalled` answers ONE question — did Substack mark this post paid-subscribers-only
        # (`audience == "only_paid"`)? A missing body is a DIFFERENT fact and `body_state` already
        # carries it: folding "we failed to fetch it" in here made every block and every network
        # error indistinguishable from a real paywall, in a field `export` ships to shared KBs.
        paywalled = bool(_is_paywalled(rec))
        # A paid post fetched with no Substack session comes back as Substack's own teaser, not
        # the article. We know that before looking at the bytes — the audience flag says paid and
        # `authenticated_body` says the fetch was not authenticated — so it is recorded as
        # `partial`, the state `atoms_tools` documents as "a paywall teaser". Writing `complete`
        # there would ship a claim to a shared KB that the text is the whole post.
        # PER PUBLICATION, not per transport: the local transport sends the session to every host
        # but only `*.substack.com` honors it, so a custom domain is an anonymous fetch however
        # the transport was chosen. That was a live false `complete` until 2026-09-08.
        # A paid post over a genuinely AUTHENTICATED fetch is left alone: whether the user
        # subscribes to that publication is not knowable from anything this run has, and guessing
        # from body length is a heuristic nobody has measured.
        preview_only = got_body and paywalled and not source.authenticated_body(base)
        if not got_body:
            stub_fallback += 1

        md = _saved_atom_markdown(rec, body_md if got_body else "")
        # Describe inline images before hashing, so the `*Image:* …` text is inside the hashed +
        # chunked surface. No-op on the stub path (no body, no refs).
        if got_body:
            with timer.stage("vlm"):       # per-POST (one post fans out to several describe calls)
                md, _ = enrich_markdown_images(md, img_cache, context=rec.get("title") or "")
        decided = snapshot_and_hash("substack", atom_id, md, seen)
        if decided is None:
            # Only reachable for a `pending` retry blocked again: hash unchanged, nothing to
            # rewrite, `body_state='pending'` stays set for the next retry.
            skipped += 1
            continue
        raw_ref, raw_hash = decided

        meta = derive.derive_substack(rec)
        who_id = meta["who_id"]
        site = meta.get("who_site")
        schema.upsert_entity(conn, who_id, name=meta.get("who_name"),
                             identity_links=[site] if site else None)

        atom = {
            "atom_id": atom_id,
            "source_type": "substack",
            "what_kind": "opinion",
            "who_id": who_id,
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "about_entities": meta["about_entities"],
            "source_url": rec.get("url"),
            "raw_ref": raw_ref,
            "raw_hash": raw_hash,
            "description": meta["description"],
            # The only adapter that deliberately stores an atom it couldn't fully fill, so all
            # four body states are live: `pending` (blocked, retried next run), `partial` (a paid
            # teaser we know is short), `absent` (no body at all). `paywalled` is orthogonal to
            # every one of them — it is Substack's own audience flag, not a statement about the
            # body; a `complete` paid post is a post the user's own session unlocked.
            "payload": {"word_count": rec.get("wordcount", 0), "paywalled": paywalled,
                        **(body_fields(BODY_PENDING, BASIS_OBSERVED) if blocked
                           else body_fields(BODY_PARTIAL, BASIS_STATED) if preview_only
                           else body_fields(BODY_COMPLETE, BASIS_STATED) if got_body
                           else body_fields(BODY_ABSENT, BASIS_OBSERVED))},
            "entry_mode": "user-saved",
        }

        def _mark(wid=who_id) -> None:    # post-commit: count + record the `save` only if it LANDED
            counts["added"] += 1
            # PRESENCE, not a count. This stamps once per atom against a corpus re-walked
            # forever, which was right while it was the only writer and double-counts the moment
            # `sync_substack_saved_signals` sets a true aggregate from the full-set walk. Same
            # move `ingest_x.sync_bookmarks` made on 2026-09-13, for the same reason.
            schema.ensure_signal(conn, wid, "save", "substack")

        seen[atom_id] = raw_hash          # within-run dedup on decision (in-memory, per-run)
        submitted += 1
        sink.submit(atom, md, on_written=_mark)

    sink.close()
    save_image_cache(opyt_home(), img_cache)   # persist new VLM descriptions
    # `failed` = submitted-but-never-durable (poison-chunk atoms, retried next run).
    # `undetermined` ⊆ `stub_fallback`: the subset stubbed because we were blocked, not because
    # the post has no body
    return {"source": "substack-saved", "added": counts["added"], "skipped": skipped,
            "stub_fallback": stub_fallback, "undetermined": undetermined,
            "failed": submitted - counts["added"],
            "stage_seconds": timer.totals, "stage_latency": timer.distribution(),
            **llm_run_stats(llm0),
            "total": schema.count_atoms(conn, "substack")}


# ── Reconcile: rebuild the `save` signal from the atoms that prove it ────────────

# Source types whose curation act leaves an atom, so the `save` signal is derivable rather than
# merely recorded. The other four signals write no atom by design, so there's nothing to
# reconcile them against. `source_type` and `platform` share one string.
SAVED_SOURCE_PLATFORMS: tuple[str, ...] = ("x", "substack")


def reconcile_saved_signals(conn) -> dict:
    """Stamps a `save` signal for every user-saved atom author that has none. Pure SQL, no
    network, idempotent, safe to call on every read.

    Both save-stampers fire only once per atom (on first ingest), so the signal is write-once
    against a corpus re-walked forever; this repairs drift from re-keyed entities, atoms landed
    by an unstamped path, or a future saved-content source. Uses insert-if-absent, not
    `add_signal` (which sums `count` and would inflate on every call). Requires an `entities`
    row and reports orphans rather than inventing one. Returns counts of what landed plus the
    signal-bearing entity total."""
    inserted: dict[str, int] = {}
    orphans: dict[str, int] = {}
    for src in SAVED_SOURCE_PLATFORMS:
        cur = conn.execute(
            "INSERT INTO curation_signals (entity_id, signal_type, platform, count) "
            "SELECT DISTINCT a.who_id, 'save', ?, 1 FROM atoms a "
            " WHERE a.entry_mode = 'user-saved' AND a.source_type = ? AND a.who_id IS NOT NULL "
            "   AND EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = a.who_id) "
            "   AND NOT EXISTS (SELECT 1 FROM curation_signals s "
            "                    WHERE s.entity_id = a.who_id AND s.signal_type = 'save' "
            "                      AND s.platform = ?) "
            # Belt and braces: DISTINCT + NOT EXISTS already prevent a duplicate, so this only
            # ever absorbs a concurrent writer landing the same row between the two statements.
            "ON CONFLICT(entity_id, signal_type, platform) DO NOTHING",
            (src, src, src))
        if cur.rowcount and cur.rowcount > 0:
            inserted[src] = cur.rowcount
        orphaned = conn.execute(
            "SELECT COUNT(DISTINCT a.who_id) FROM atoms a "
            " WHERE a.entry_mode = 'user-saved' AND a.source_type = ? AND a.who_id IS NOT NULL "
            "   AND NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = a.who_id)",
            (src,)).fetchone()[0]
        if orphaned:
            orphans[src] = orphaned
    conn.commit()
    return {"inserted": inserted, "orphans": orphans,
            "signal_bearing_entities": _distinct_signal_entities(conn)}


# ── Orchestration (T8) ───────────────────────────────────────────────────────────

def _distinct_signal_entities(conn) -> int:
    return conn.execute(
        "SELECT COUNT(DISTINCT entity_id) FROM curation_signals").fetchone()[0]


# ── The collector registry: what the LIST clock tracks ───────────────────────────
#
# Every PEOPLE-ONLY collector, and only those. The two content-bearing arms are absent
# on purpose: each has its OWN automatic rail — `bookmark_catchup` for X bookmarks,
# `substack_saved_catchup` for Substack saved posts — and their `save` signal is re-derivable
# from the atoms they land (`reconcile_saved_signals`). Putting either on this clock would invite
# `curation_catchup`, which is model-free by design, to re-run a paid content pipeline. The
# signal-only halves of those same two walks ARE here, and are not a counter-example: they read
# the list, write no atom and call no model, which is the property this tuple selects for.
# None of these has an automatic trigger of its own and, before `collector_runs`, no state at
# all — someone you followed yesterday stayed invisible until a human hand-ran this module.
#
# Why a spec and not normalised functions. They disagree about their own return shape:
# Lists, likes and the two saved walks report `candidates`, following reports `following`, subs
# reports `subscriptions`.
# Rewriting working collectors onto one key would also rewrite summaries other readers already
# print, for no gain the clock needs. So the spec RECORDS the disagreement — and one test asserts
# each collector really returns the key its spec names, which turns "the spec drifted" from a
# silently-NULL `found` column into a red test.
#
# `fn_name`, NOT a function object. It resolves through this module at call time, so a monkeypatch
# of `ingest_curation.sync_lists_signals` actually reaches the dispatch. A callable captured at
# import binds the original and ignores the patch — which would make every stub in the catch-up
# tests silently run the real network collector.
@dataclass(frozen=True)
class CollectorSpec:
    collector: str        # the `curation_state.collector_runs` primary key
    label: str            # the timer/log label — already the string this module logs under
    fn_name: str          # resolved against THIS module at call time (see above)
    signal_type: str      # with `platform`: what `stored_after` counts in `curation_signals`
    platform: str         # ...and which cookie profile the collector reads (x vs substack)
    found_key: str        # the key THIS collector reports its own observed count under


COLLECTOR_SPECS: tuple[CollectorSpec, ...] = (
    CollectorSpec("x_lists", "x-lists", "sync_lists_signals", "list", "x", "candidates"),
    # The two Substack reads are two specs, not one. They read different endpoints and land
    # different signal types, and `CollectorSpec` carries exactly one of each. The new one does
    # NOT reuse the retired `substack_subs` key: that would hand the follow collector's clock
    # history — its `last_ok_at`, its `prev_found` collapse baseline — to a collector that never
    # made those runs.
    CollectorSpec("substack_follows", "substack-follows", "sync_substack_follows",
                  "follow", "substack", "follows"),
    CollectorSpec("substack_subscriptions", "substack-subscriptions",
                  "sync_substack_subscriptions", "subscribe", "substack", "subscriptions"),
    # The SIGNAL half of the saved-posts walk — X's `x_bookmark_signals` on the other platform,
    # and the one this tuple needed most: Substack's two other readable signals are near-disjoint
    # graphs (0 of 36 overlapping on the live store), so without this nobody on Substack could
    # clear `screen.CORROBORATION_MIN` at all. Reads the list and nothing else: no body, no
    # model, no embedding — the property this tuple selects for. The CONTENT half
    # (`sync_substack_saved`) stays off this clock, like X's.
    CollectorSpec("substack_saved_signals", "substack-saved-signals",
                  "sync_substack_saved_signals", "save", "substack", "candidates"),
    CollectorSpec("x_following", "x-following", "sync_following_signals",
                  "follow", "x", "following"),
    CollectorSpec("x_likes", "x-likes", "sync_likes_signals", "like", "x", "candidates"),
    # The SIGNAL half of the bookmark walk, and a fifth peer rather than a sixth source: it is a
    # free full-set cookie-scrape that lands signals and no atoms, which is the exact property
    # this tuple selects for. The CONTENT half (`ingest_x.sync_bookmarks`) is deliberately still
    # not on this clock — see `_run`'s `spec is None` note, which is about that arm and not this
    # one. Registering the two together would put the paid arm on a clock `curation_catchup`
    # feels entitled to re-run.
    CollectorSpec("x_bookmark_signals", "x-bookmark-signals", "sync_bookmark_signals",
                  "save", "x", "candidates"),
)
COLLECTORS: tuple[str, ...] = tuple(s.collector for s in COLLECTOR_SPECS)
SPEC_BY_COLLECTOR: dict[str, CollectorSpec] = {s.collector: s for s in COLLECTOR_SPECS}

# Bookmark lookback presets — OPERATOR-ONLY, reached from this module's `--bookmark-lookback` and
# nothing else. No MCP tool takes a bookmark window, and `oracle`'s `lookback_options` deliberately
# stops advertising one (2026-08-30) — it was asking users a question whose answer had nowhere to go.
#
# Default `all`, and leave it there. The walk is a free cookie-scrape; the per-bookmark thread
# fetch went free with the X cutover, leaving only the VLM read on bookmarks carrying images —
# measured $0.105 across 315 images on a ~1,080-bookmark backlog. Narrowing therefore saves cents
# and costs corpus, on an axis that misleads: this filters on when the tweet was WRITTEN, not when
# it was saved, because X exposes no bookmark timestamp. A 6-month window drops the 2019 paper
# saved yesterday.
BOOKMARK_LOOKBACK_PRESETS: dict[str, int | None] = {"6mo": 183, "1yr": 365, "2yr": 730,
                                                    "5yr": 1825, "all": None}


def collector_fn(spec: CollectorSpec):
    """The collector callable this spec names, resolved NOW (see the `fn_name` note above)."""
    return getattr(sys.modules[__name__], spec.fn_name)


def run_collector(conn, spec: CollectorSpec, *, substack_profile: str | None = None) -> dict:
    """Call a collector, passing a profile only to Substack's generic cookie reader."""
    if spec.platform == "substack":
        return collector_fn(spec)(conn, profile=substack_profile)
    return collector_fn(spec)(conn)


def stored_signal_rows(conn, spec: CollectorSpec) -> int:
    """How many rows the STORE holds for this collector's signal — the other half of the pair the
    clock records. `found` is what the collector said it saw; this is what actually landed. A run
    reporting `found=468, stored_after=0` is the hot-feed failure shape: a truthful self-report
    over a write path that wrote nothing. One number alone can never show that."""
    return conn.execute(
        "SELECT COUNT(*) FROM curation_signals WHERE signal_type=? AND platform=?",
        (spec.signal_type, spec.platform)).fetchone()[0]


def _stamp_run(conn, spec: CollectorSpec | None, res: dict | None, *,
               status: str | None = None, detail: str | None = None,
               started_at: str | None = None) -> None:
    """Write ONE collector's outcome to the list clock. The only writer, and now the only call site
    is `_run` (ok / its own skip / error) — the tiered ladder's `_done` went with the ladder on
    2026-09-04, see the `retired-tiered-curation-ladder` guard.

    Status AND detail both come from the result when the caller does not force them, and they are
    two different jobs. A collector that returned `{"skipped": "no_viewer_id"}` records THAT
    string, so "we ran and the X session was dead" stays distinguishable from "we ran and saw an
    empty list" — that is the word the clock and `onboard_state` branch on. `detail` is the
    sentence underneath it, and only a skip that HAS a reason carries one: the four `no_viewer_id`
    skips are fully described by their own status word, while the three `refused` ones are not.

    FAIL-SAFE, and the direction matters. This is bookkeeping ABOUT the pull, so a state-write
    failure must never sink a pull that actually landed data. The reverse — swallowing a collector
    error — is not what this catch does; `_run` has already recorded the error by the time we get
    here."""
    if conn is None or spec is None:
        return
    from pipeline.ingestion.utils import log

    from . import curation_state
    try:
        if status is None:
            skipped = (res or {}).get("skipped")
            status = str(skipped) if skipped else curation_state.STATUS_OK
        if detail is None:
            # The collector's OWN skip reason, from the result — not just `_run`'s exception path.
            # Without this a refusal recorded `last_status='refused', last_detail=NULL`, and the
            # one word that survived could not say WHICH refusal: a Cloudflare 403, a dead session
            # and a hosted follow-list refusal all read identically. Measured 2026-09-15 on the
            # box: `mcp_child.log` carried "hosted follow-list request was refused" while that
            # home's `collector_runs` row carried nothing, so the only diagnosable copy of the
            # reason lived in a log nobody downstream reads. `status` is the word the clock
            # branches on; `detail` is the sentence a human needs, and a skip deserves both.
            detail = (res or {}).get("detail")
        found = stored = None
        if status == curation_state.STATUS_OK:
            # Counts ride ONLY on a success, which is what makes them unambiguous downstream: a
            # stored count is always "as of last_ok_at". See `curation_state.record_run`.
            found = (res or {}).get(spec.found_key)
            stored = stored_signal_rows(conn, spec)
        curation_state.record_run(conn, spec.collector, status=status, detail=detail,
                                  found=found, stored_after=stored, started_at=started_at)
    except Exception as e:
        log(f"[curation] could not record {spec.collector} run state: {type(e).__name__}: {e}")


def run_and_record(conn, spec: CollectorSpec, *, timer: StageTimer | None = None,
                   substack_profile: str | None = None) -> dict:
    """Run ONE clocked collector and stamp its outcome on the list clock.

    THE single dispatch point, shared by `curation_pull` (the hand-run, all six sources) and
    `curation_catchup` (the unattended rail, these four only). Sharing it is the point: the two
    cannot drift about failure isolation, about which cookie profile a platform reads, or about
    what gets recorded — and the rail does not have to reach into a private helper to get any of
    it. An omitted `timer` gets a throwaway one; an unread `StageTimer` costs nothing."""
    return _run(timer if timer is not None else StageTimer(), spec.label,
                lambda: run_collector(conn, spec, substack_profile=substack_profile),
                conn=conn, spec=spec)


def _run(timer: StageTimer, label: str, fn, *, conn=None, spec: CollectorSpec | None = None):
    """Run one source under `timer`, isolating its failure: a platform dying (Cloudflare on
    Substack subs, a dead X session) logs LOUD and returns an error stub — it must NOT sink the
    other five.

    The timing is INSIDE the try, so a source that dies at minute 9 still reports the nine minutes
    it burned. A failure that costs a lot of wall clock is exactly the one worth seeing in the
    profile, and timing only the successes would hide it.

    `spec` opts a source into the LIST clock. It is None for bookmarks and Substack saved — those
    two are not on this clock (see `COLLECTOR_SPECS`), and stamping them here would put a row in
    `collector_runs` that `curation_catchup` would then feel entitled to re-run."""

    from pipeline.ingestion.utils import log
    # BEFORE the collector runs. A walk confirms each person as it goes, so "was this signal seen
    # by the last walk" can only be answered against the moment the walk BEGAN — see
    # `curation_state.record_run`. Taken here rather than inside the collector so all five share it.
    started_at = utc_now().isoformat()
    try:
        with timer.stage(label):
            res = fn()
        log(f"[curation] {label}: {res}")
        _stamp_run(conn, spec, res, started_at=started_at)
        return res
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        log(f"[curation] {label} FAILED (continuing): {detail}")
        _stamp_run(conn, spec, None, status="error", detail=detail)
        return {"source": label, "error": detail}


def derive_paper_signals(conn, *, timer: StageTimer | None = None) -> dict:
    """The two derivations that turn stored PAPERS into screenable PEOPLE. No network, no model,
    no session — a local walk over atoms the user already has.

    Neither is a `_collector`: they stamp no `collector_runs` row, because there is no external
    list whose walk could be stale. That is also why they need this shared home. `curation_pull`
    spelled both calls out inline and `curation_catchup` had neither, so from the split until
    2026-09-08 a paper deposited through `hopper` reached the screen only if somebody hand-ran the
    pull. Measured on the live store that day: zero occurrences of "paper" across the whole
    `curation_catchup.log`, against 11 `user-saved` paper atoms.

    ORDER IS LOAD-BEARING, and holding it in one function is the point. Authors first: that half
    walks `user-saved` papers, the coauthor half walks `oracle-footprint` ones, and a person can
    be BOTH — an author the user saved who also co-writes with an Oracle carries two distinct
    signals, which is what corroboration means at the screen. Signals only, either way: a
    coauthor is never promoted to an Oracle without the user.

    `timer` is optional because the two callers differ honestly: `curation_pull` profiles every
    stage of one pull and wants these two in that profile, while `curation_catchup` keeps no
    profile at all. An unread timer costs nothing, so the absent case builds one rather than
    growing a second, untimed path through `_run` — which is what carries the failure isolation.
    """
    from . import paper_authors

    timer = timer or StageTimer()
    return {
        "paper_authors": _run(timer, "paper-authors",
                              lambda: paper_authors.sync_paper_author_signals(conn)),
        "paper_coauthors": _run(timer, "paper-coauthors",
                                lambda: paper_authors.sync_coauthor_signals(conn)),
    }


def resolve_after_pull(conn) -> dict:
    """Re-derive every entity's `canonical_id` and return the compact outcome. Never raises.

    This is the only moment resolution can fire before the SCREEN, and the SCREEN is where it
    pays. `screen.rank_candidates` groups on `COALESCE(canonical_id, entity_id)`, so an unresolved
    person is TWO candidates carrying one signal each instead of one carrying two — and ≥2 distinct
    signals is the pre-tick bar. They are filtered out before a human ever sees them, which is the
    silent-drop shape: nothing errors, the candidate simply never appears.

    Measured against the live store 2026-08-14: 8.5 ms over 1,020 entities, and one candidate
    (a Substack pub minting BOTH a handle-id and a subdomain-id, because subscriptions and saved
    posts read different API surfaces) newly cleared the bar because of it. That split reproduces
    for anyone who both subscribes to a publication and saves a post from it, so it is not an n=1
    artifact. Pure Python over rows already on disk — no network, no LLM — and idempotent by
    construction, so re-running costs a millisecond and can never corrupt anything.

    It swallows its own failure on purpose. Every atom, entity and signal is already committed by
    the time this runs; a resolve hiccup must degrade to an unmerged store, never take the pull's
    whole report down with it.
    """
    from . import resolve

    try:
        st = resolve.resolve_entities(conn).as_dict()
        return {k: st[k] for k in ("total_entities", "components",
                                   "duplicate_rows_collapsed", "cross_platform")}
    except Exception as e:                                   # fail-safe: report, never propagate
        from pipeline.ingestion.utils import log
        detail = f"{type(e).__name__}: {e}"
        log(f"[curation] resolve FAILED (pull still stands): {detail}")
        return {"error": detail}


def curation_pull(conn, embedder, *, substack_profile: str | None = None, x_limit: int = 0,
                  bookmark_since: datetime | None = None) -> dict:
    """Pull all six curation sources into the atom-KB, then derive the paper-author signals from
    what is now on disk. Each source is failure-isolated.

    `bookmark_since` bounds the BOOKMARK arm only — see `ingest_x.sync_bookmarks`. It is a SPEND
    filter (skip the paid per-bookmark work), and it filters on when the tweet was written, which
    is not the same question as when you saved it.

    Returns the per-source summaries plus `stage_seconds` — the ONLY clock the five signal-only
    sources and the paper-author producer have. Two of the six (`x_bookmarks`, `substack_saved`) run real content pipelines and
    carry their own internal `stage_seconds`; the other four are single pulls with no adapter-level
    timer at all, so without this they contribute nothing to a wall-clock profile except an
    unexplained gap between the sum of the parts and the length of the run."""
    from . import ingest_x

    results: dict = {}
    # ONE timer across every producer, so `stage_seconds` reads as a single profile of the pull
    # rather than a handful of unrelated numbers. No null branch: an unread timer costs nothing
    # (see StageTimer).
    timer = StageTimer()

    def _collector(name: str) -> dict:
        """Run one clocked collector through the shared dispatch, stamping `collector_runs`."""
        return run_and_record(conn, SPEC_BY_COLLECTOR[name], timer=timer,
                              substack_profile=substack_profile)

    # No `stage_latency` on the timer: each label has exactly ONE sample (one call per source),
    # so a p50/p95/max over it would be the same number printed five times.
    # Highest-intent first: bookmarks (content) + Lists + the two Substack account reads + saved
    # (content).
    # BEFORE the content arm, and worth the second walk of the same list (five requests, no
    # sleeps): it is the counting authority for `save`, so without it the content arm's
    # `ensure_signal` would leave every bookmarked author sitting at a count of 1. Running first
    # also means a content arm that dies mid-backlog still leaves the signals complete.
    results["x_bookmark_signals"] = _collector("x_bookmark_signals")
    results["x_bookmarks"] = _run(
        timer, "x-bookmarks",
        lambda: ingest_x.sync_bookmarks(conn, embedder, limit=x_limit, since=bookmark_since))
    results["x_lists"] = _collector("x_lists")
    results["substack_follows"] = _collector("substack_follows")
    results["substack_subscriptions"] = _collector("substack_subscriptions")
    # BEFORE the content arm, for the reason its X twin states two calls up: it is the counting
    # authority for `save`, so running it second would leave every saved-from publication sitting
    # at the `ensure_signal` count of 1 the content arm writes. It also means a content arm that
    # dies mid-backlog still leaves the signals complete. The second walk of the same list is ~10s
    # of paced requests here rather than X's ~0, and is still worth it for that.
    results["substack_saved_signals"] = _collector("substack_saved_signals")
    results["substack_saved"] = _run(
        timer, "substack-saved",
        lambda: sync_substack_saved(conn, embedder, profile=substack_profile))

    # Then the broader, noisier sources: following, then likes (priciest to gather → last).
    results["x_following"] = _collector("x_following")
    results["x_likes"] = _collector("x_likes")
    # Last of the producers, and the only ones that are not a network pull. They must run after
    # the six above to see this run's saves.
    results.update(derive_paper_signals(conn, timer=timer))
    # Resolution is the LAST thing the pull does — an unmerged duplicate costs a candidate
    # their pre-tick on the screen that reads this.
    results["resolve"] = resolve_after_pull(conn)
    results["stage_seconds"] = dict(timer.totals)
    return results


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    from .embed import get_kb_embedder
    from .expand import _since_from_days

    ap = argparse.ArgumentParser(description="Step-2 Curation Pull into the atom-KB "
                                             "(honors $OPYT_HOME).")
    ap.add_argument("--substack-profile", default=None,
                    help="Substack cookie profile (else auto-pick).")
    ap.add_argument("--x-limit", type=int, default=0, help="Cap bookmark ingest (0 = all).")
    ap.add_argument("--bookmark-lookback", choices=list(BOOKMARK_LOOKBACK_PRESETS), default="all",
                    help="Only ingest bookmarks of posts WRITTEN since this window (default all). "
                         "Bounds SPEND — the walk is free, but each surviving bookmark costs a "
                         "thread fetch and often a VLM read. NOTE: this is the tweet's write date, "
                         "NOT when you saved it (X exposes no bookmark timestamp), so a narrow "
                         "window drops an old post you bookmarked yesterday.")
    args = ap.parse_args(argv)

    embedder = get_kb_embedder()
    bookmark_since = _since_from_days(BOOKMARK_LOOKBACK_PRESETS[args.bookmark_lookback])
    print(f"[curation] embedder: model={embedder.model} provider={embedder.provider}  "
          f"bookmark-lookback={args.bookmark_lookback} "
          f"(posts written since {bookmark_since or 'the beginning'})")
    conn = schema.connect()
    try:
        out = curation_pull(conn, embedder, substack_profile=args.substack_profile, x_limit=args.x_limit,
                            bookmark_since=bookmark_since)
    finally:
        conn.close()
    print("[curation] summary:\n" + _json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
