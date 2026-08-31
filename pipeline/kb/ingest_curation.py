"""
pipeline/kb/ingest_curation.py — Step-2 Curation Pull: the user's own endorsement
signals into the atom-KB, feeding Stage-3 (entity resolution) and Stage-4 (candidate SCREEN).

Two flavors: CONTENT-BEARING (Substack saved-posts; X bookmarks live in ingest_x) writes a
content ATOM (`entry_mode='user-saved'`) + author ENTITY + `save` SIGNAL. PEOPLE-ONLY (X
following/Lists/likes; Substack subscriptions) writes only an author ENTITY + SIGNAL
(`follow`/`list`/`like`/`subscribe`) — no atom.

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
                            BODY_PENDING, StageTimer,
                            body_fields, llm_run_marker, llm_run_stats,
                            snapshot_and_hash)

# ── People-only stampers (entity + signal, NO atom) ──────────────────────────────

def _person_profile(cand: dict) -> dict | None:
    """Extracts the Stage-4 kind-classify inputs from a normalized X user (bio, verified,
    followers, handle) for storage on the entity's `profile` blob. Only present keys are
    written; returns None when nothing classify-worthy is present."""
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
    for the run, so summing would double-count (see `schema.set_signal` and
"""
    eid = f"x:user:{cand['user_id']}"
    site = cand.get("site")
    schema.upsert_entity(conn, eid, name=cand.get("display_name"),
                         identity_links=[site] if site else None,
                         profile=_person_profile(cand))
    schema.set_signal(conn, eid, signal_type, "x", count=count, extra=extra)
    return eid


def sync_likes_signals(conn, *, profile: str | None = None) -> dict:
    """Tier-3: the authors of tweets you liked → `like` signal (count = likes earned per
    author). No atoms — a liked tweet's content never enters the KB."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.x_likes import aggregate_authors, fetch_liked_authors

    cookies = core.read_x_cookies(profile=profile)
    vid = core.viewer_id(cookies)
    if not vid:
        return {"source": "x-likes", "skipped": "no_viewer_id"}
    headers = core.auth_headers(cookies, referer="https://x.com/i/likes")
    authors = fetch_liked_authors(vid, cookies, headers)
    cands = aggregate_authors(authors, vid)
    for c in cands:
        _stamp_x_person(conn, c, "like", count=c["liked_count"])
    return {"source": "x-likes", "candidates": len(cands),
            "liked_tweets_with_author": len(authors)}


def sync_lists_signals(conn, *, profile: str | None = None) -> dict:
    """Tier-1: members of your owned Lists → `list` signal (count = breadth of list
    membership; extra carries the list names). No atoms."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.x_lists import (aggregate_members, fetch_list_members,
                                            fetch_owned_lists)

    cookies = core.read_x_cookies(profile=profile)
    vid = core.viewer_id(cookies)
    if not vid:
        return {"source": "x-lists", "skipped": "no_viewer_id"}
    headers = core.auth_headers(cookies, referer="https://x.com/i/lists")
    owned = fetch_owned_lists(cookies, headers, vid)
    members_by_list = {l["id"]: fetch_list_members(l["id"], cookies, headers) for l in owned}
    cands = aggregate_members(owned, members_by_list, vid)
    for c in cands:
        _stamp_x_person(conn, c, "list", count=len(c["list_names"]),
                        extra={"list_names": c["list_names"]})
    return {"source": "x-lists", "lists": len(owned), "candidates": len(cands)}


def sync_following_signals(conn, *, profile: str | None = None) -> dict:
    """Tier-2: the accounts you follow → `follow` signal. Free cookie-scrape via
    `x_graphql_core.fetch_following`, not the paid radar scout. No atoms."""
    from pipeline.ingestion import x_graphql_core as core

    cookies = core.read_x_cookies(profile=profile)
    vid = core.viewer_id(cookies)
    if not vid:
        return {"source": "x-following", "skipped": "no_viewer_id"}
    headers = core.auth_headers(cookies, referer="https://x.com/following")
    users = core.fetch_following(cookies, headers, vid)
    for u in users:
        _stamp_x_person(conn, u, "follow")
    return {"source": "x-following", "following": len(users)}


def sync_substack_subs(conn, *, profile: str | None = None) -> dict:
    """Tier-1: your Substack "Following" list → `subscribe` signal. The subscriber-lists
    endpoint returns only {name, url} (no handle, no paid/free flag), so the entity keys on
    the publication subdomain and `is_paid` is recorded as unknown for later backfill. No
    atoms."""
    from pipeline.ingestion.sources.substack import (fetch_subscriptions, own_user_id,
                                                     read_substack_cookies)

    cookies = read_substack_cookies(profile=profile)
    uid = own_user_id(cookies)
    subs = fetch_subscriptions(cookies, uid)
    for s in subs:
        url = s.get("url") or ""
        eid = derive.substack_entity_id(None, url)   # subs API drops the handle → subdomain id
        schema.upsert_entity(conn, eid, name=s.get("name"),
                             identity_links=[url] if url else None)
        # is_paid isn't in this payload, so record unknown rather than fabricate a value.
        # `set_signal`, not `add_signal`: this walks the whole Following list every run.
        schema.set_signal(conn, eid, "subscribe", "substack", extra={"is_paid": None})
    return {"source": "substack-subs", "subscriptions": len(subs)}


# ── Substack saved-posts → FULL-BODY content atoms + `save` signal ───────────────

def _clean_body_html(body_html: str) -> str:
    """Converts a Substack `body_html` fragment (already de-boilerplated) to clean markdown for
    chunking. Runs both trafilatura and html2text and keeps the longer result, since trafilatura
    can silently under-extract a bare fragment into a thin "invisible" atom. Empty in → "" out
    (caller falls back to the stub). Images are kept."""
    if not body_html:
        return ""
    cands: list[str] = []
    try:
        import trafilatura
        md = trafilatura.extract(body_html, output_format="markdown",
                                 include_links=True, include_images=True, no_fallback=False)
        if md:
            cands.append(md)
    except Exception:
        pass
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links, h.ignore_images, h.body_width, h.unicode_snob = False, False, 0, True
        t = h.handle(body_html)
        if t:
            cands.append(t)
    except Exception:
        pass
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

    Per saved post: fetches the full body as a subscriber, cleans HTML to text, and chunks the
    body (a title+preview stub alone yields thin, invisible atoms). A paywalled/failed body
    falls back to the stub preview. Idempotent and cost-paced
    fail-safe: one bad post never starves the rest."""
    from pipeline.ingestion.sources.substack import (SubstackFetchError, _fetch_full_post,
                                                     _is_paywalled, fetch_saved_posts,
                                                     read_substack_cookies)
    from pipeline.ingestion.utils import log
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from .vision import enrich_markdown_images

    assert_model(conn, embedder)      # guard the store's embedding identity before spend
    # Marks the LLM latency baseline before the first paid call, so the summary reports this
    # run's calls, not the process-cumulative total.
    llm0 = llm_run_marker()
    cookies = read_substack_cookies(profile=profile)
    # Stage-timed like the footprint adapters
    timer = StageTimer()
    with timer.stage("list_fetch"):        # one cursor-paginated walk of the saved list
        recs = fetch_saved_posts(cookies)
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
                with timer.stage("body_fetch"):   # one authed GET per NEW saved post
                    full = _fetch_full_post(base, slug, cookies)
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
        paywalled = bool(_is_paywalled(rec)) or not got_body
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
            # three body states are live: `pending` (blocked, retried next run), `absent` (no
            # body at all, distinct from a shortened `partial`), `paywalled` (its own flag). See
            "payload": {"word_count": rec.get("wordcount", 0), "paywalled": paywalled,
                        **(body_fields(BODY_PENDING, BASIS_OBSERVED) if blocked
                           else body_fields(BODY_COMPLETE, BASIS_STATED) if got_body
                           else body_fields(BODY_ABSENT, BASIS_OBSERVED))},
            "entry_mode": "user-saved",
        }

        def _mark(wid=who_id) -> None:    # post-commit: count + record the `save` only if it LANDED
            counts["added"] += 1
            schema.add_signal(conn, wid, "save", "substack")

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


def reconcile_saved_signals(conn, *, sources: tuple[str, ...] = SAVED_SOURCE_PLATFORMS) -> dict:
    """Stamps a `save` signal for every user-saved atom author that has none. Pure SQL, no
    network, idempotent, safe to call on every read.

    Both save-stampers fire only once per atom (on first ingest), so the signal is write-once
    against a corpus re-walked forever; this repairs drift from re-keyed entities, atoms landed
    by an unstamped path, or a future saved-content source. Uses insert-if-absent, not
    `add_signal` (which sums `count` and would inflate on every call). Requires an `entities`
    row and reports orphans rather than inventing one. Returns counts of what landed plus the
    signal-bearing entity total. Full rationale and 2026-08-12 measurement in
    """
    inserted: dict[str, int] = {}
    orphans: dict[str, int] = {}
    for src in sources:
        cur = conn.execute(
            "INSERT INTO curation_signals (entity_id, signal_type, platform, count) "
            "SELECT DISTINCT a.who_id, 'save', ?, 1 FROM atoms a "
            " WHERE a.entry_mode = 'user-saved' AND a.source_type = ? AND a.who_id IS NOT NULL "
            "   AND EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = a.who_id) "
            "   AND NOT EXISTS (SELECT 1 FROM curation_signals s "
            "                    WHERE s.entity_id = a.who_id AND s.signal_type = 'save' "
            "                      AND s.platform = ?) "
            # Belt and braces: DISTINCT + NOT EXISTS mean this can't fire today, but it keeps a
            # future overlapping-sources caller a no-op instead of an IntegrityError.
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
# The four PEOPLE-ONLY collectors, and only those four. Bookmarks and Substack saved-posts are
# absent on purpose: bookmarks have their own automatic rail (`bookmark_catchup`) and their `save`
# signal is re-derivable from the atoms it lands (`reconcile_saved_signals`), and Substack saved
# needs a signals-only mode before it can join an unattended loop. These four have no automatic
# trigger and, before `collector_runs`, no state at all — someone you followed yesterday stayed
# invisible until a human hand-ran this module.
#
# Why a spec and not four normalised functions. The four disagree about their own return shape:
# Lists and likes report `candidates`, following reports `following`, subs reports `subscriptions`.
# Rewriting four working collectors onto one key would also rewrite summaries other readers already
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
    CollectorSpec("substack_subs", "substack-subs", "sync_substack_subs",
                  "subscribe", "substack", "subscriptions"),
    CollectorSpec("x_following", "x-following", "sync_following_signals",
                  "follow", "x", "following"),
    CollectorSpec("x_likes", "x-likes", "sync_likes_signals", "like", "x", "candidates"),
)
COLLECTORS: tuple[str, ...] = tuple(s.collector for s in COLLECTOR_SPECS)
SPEC_BY_COLLECTOR: dict[str, CollectorSpec] = {s.collector: s for s in COLLECTOR_SPECS}

# Which collectors each `--tiered` early return is about to skip. Declared once, next to the specs,
# rather than spelled out at both early-return sites — two hand-written lists is how a collector
# added to Tier 2 later gets skipped without ever being recorded as skipped.
_TIER_SKIPS: dict[str, tuple[str, ...]] = {
    "tier1": ("x_following", "x_likes"),
    "tier2": ("x_likes",),
}


def collector_fn(spec: CollectorSpec):
    """The collector callable this spec names, resolved NOW (see the `fn_name` note above)."""
    return getattr(sys.modules[__name__], spec.fn_name)


def run_collector(conn, spec: CollectorSpec, *, x_profile: str | None = None,
                  substack_profile: str | None = None) -> dict:
    """Call one collector with the cookie profile its platform reads. The single dispatch point,
    so `curation_pull` and `curation_catchup` cannot drift about which profile goes where."""
    profile = substack_profile if spec.platform == "substack" else x_profile
    return collector_fn(spec)(conn, profile=profile)


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
    """Write ONE collector's outcome to the list clock. The only writer, with two call sites: `_run`
    (ok / its own skip / error) and `curation_pull._done` (the tier skips).

    Status comes from the result when the caller does not force one: a collector that returned
    `{"skipped": "no_viewer_id"}` records THAT string, so "we ran and the X session was dead" stays
    distinguishable from "we ran and saw an empty list".

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
                   x_profile: str | None = None, substack_profile: str | None = None) -> dict:
    """Run ONE clocked collector and stamp its outcome on the list clock.

    THE single dispatch point, shared by `curation_pull` (the hand-run, all six sources) and
    `curation_catchup` (the unattended rail, these four only). Sharing it is the point: the two
    cannot drift about failure isolation, about which cookie profile a platform reads, or about
    what gets recorded — and the rail does not have to reach into a private helper to get any of
    it. An omitted `timer` gets a throwaway one; an unread `StageTimer` costs nothing."""
    return _run(timer if timer is not None else StageTimer(), spec.label,
                lambda: run_collector(conn, spec, x_profile=x_profile,
                                      substack_profile=substack_profile),
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
    from datetime import datetime, timezone

    from pipeline.ingestion.utils import log
    # BEFORE the collector runs. A walk confirms each person as it goes, so "was this signal seen
    # by the last walk" can only be answered against the moment the walk BEGAN — see
    # `curation_state.record_run`. Taken here rather than inside the collector so all four share it.
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


def curation_pull(conn, embedder, *, x_profile: str | None = None,
                  substack_profile: str | None = None, x_limit: int = 0,
                  bookmark_since: datetime | None = None,
                  tiered: bool = False, sufficient_at: int = 30) -> dict:
    """Pull all six curation sources into the atom-KB. Default (David's dogfood) runs every
    source; `--tiered` is the stop-when-sufficient ladder for thin distributable users:
    Tier-1 (bookmarks/Lists/subs/saved) → Tier-2 (following) → Tier-3 (likes), stopping once
    ≥`sufficient_at` distinct entities carry a signal. Each source is failure-isolated.

    `bookmark_since` bounds the BOOKMARK arm only — see `ingest_x.sync_bookmarks`. It is a SPEND
    filter (skip the paid per-bookmark work), and it filters on when the tweet was written, which
    is not the same question as when you saved it.

    Returns the per-source summaries plus `stage_seconds` — the ONLY clock the four signal-only
    sources have. Two of the six (`x_bookmarks`, `substack_saved`) run real content pipelines and
    carry their own internal `stage_seconds`; the other four are single pulls with no adapter-level
    timer at all, so without this they contribute nothing to a wall-clock profile except an
    unexplained gap between the sum of the parts and the length of the run."""
    from pipeline.ingestion.utils import log
    from . import ingest_x

    results: dict = {}
    # ONE timer across all six, so `stage_seconds` reads as a single profile of the pull rather
    # than six unrelated numbers. No null branch: an unread timer costs nothing (see StageTimer).
    timer = StageTimer()

    def _collector(name: str) -> dict:
        """Run one clocked collector through the shared dispatch, stamping `collector_runs`."""
        return run_and_record(conn, SPEC_BY_COLLECTOR[name], timer=timer,
                              x_profile=x_profile, substack_profile=substack_profile)

    def _done(stopped_after: str | None = None) -> dict:
        # Every exit stamps the clock — an early --tiered return is still a run someone profiles.
        # No `stage_latency` here on purpose: each label has exactly ONE sample (one call per
        # source), so a p50/p95/max over it would be the same number printed five times.
        if stopped_after:
            results["tiered_stopped_after"] = stopped_after
            # Only the orchestrator knows a skip happened. A collector the ladder skips never runs,
            # so it cannot record its own skip — and with no row at all it is indistinguishable
            # from one that has never existed, which is the state the whole clock exists to make
            # readable. `skipped_tier` says "deliberately not run", and it advances the ATTEMPT
            # mark without advancing the OK mark, so the list correctly keeps ageing.
            for name in _TIER_SKIPS.get(stopped_after, ()):
                _stamp_run(conn, SPEC_BY_COLLECTOR[name], None, status="skipped_tier",
                           detail=f"--tiered stopped after {stopped_after}")
        # Resolution is the LAST thing the pull does, and it lives in `_done` so all three exits
        # (tier-1 stop, tier-2 stop, full run) get it — a tiered run is exactly the thin store
        # where an unmerged duplicate is most likely to cost a candidate their pre-tick.
        results["resolve"] = resolve_after_pull(conn)
        results["stage_seconds"] = dict(timer.totals)
        return results

    # Tier 1 — highest-intent: bookmarks (content) + Lists + subs + saved (content).
    results["x_bookmarks"] = _run(
        timer, "x-bookmarks",
        lambda: ingest_x.sync_bookmarks(conn, embedder, limit=x_limit, profile=x_profile,
                                        since=bookmark_since))
    results["x_lists"] = _collector("x_lists")
    results["substack_subs"] = _collector("substack_subs")
    results["substack_saved"] = _run(
        timer, "substack-saved",
        lambda: sync_substack_saved(conn, embedder, profile=substack_profile))

    if tiered and _distinct_signal_entities(conn) >= sufficient_at:
        log(f"[curation] --tiered: Tier-1 yielded ≥{sufficient_at} signalled entities — "
            f"skipping following (Tier-2) + likes (Tier-3).")
        return _done("tier1")

    # Tier 2 — following (broader, noisier).
    results["x_following"] = _collector("x_following")

    if tiered and _distinct_signal_entities(conn) >= sufficient_at:
        log(f"[curation] --tiered: Tier-2 reached ≥{sufficient_at} — skipping likes (Tier-3).")
        return _done("tier2")

    # Tier 3 — likes (noisiest, priciest to gather → pulled last).
    results["x_likes"] = _collector("x_likes")
    return _done()


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json as _json

    from .embed import get_kb_embedder
    from .expand import BOOKMARK_LOOKBACK_PRESETS, _since_from_days

    ap = argparse.ArgumentParser(description="Step-2 Curation Pull into the atom-KB "
                                             "(honors $OPYT_HOME).")
    ap.add_argument("--x-profile", default=None, help="X cookie profile (else auto-pick).")
    ap.add_argument("--substack-profile", default=None,
                    help="Substack cookie profile (else auto-pick).")
    ap.add_argument("--x-limit", type=int, default=0, help="Cap bookmark ingest (0 = all).")
    ap.add_argument("--bookmark-lookback", choices=list(BOOKMARK_LOOKBACK_PRESETS), default="all",
                    help="Only ingest bookmarks of posts WRITTEN since this window (default all). "
                         "Bounds SPEND — the walk is free, but each surviving bookmark costs a "
                         "thread fetch and often a VLM read. NOTE: this is the tweet's write date, "
                         "NOT when you saved it (X exposes no bookmark timestamp), so a narrow "
                         "window drops an old post you bookmarked yesterday.")
    ap.add_argument("--tiered", action="store_true",
                    help="Stop-when-sufficient ladder for thin users; default runs all six.")
    ap.add_argument("--sufficient-at", type=int, default=30,
                    help="Distinct-signalled-entity threshold that stops the --tiered ladder.")
    args = ap.parse_args(argv)

    embedder = get_kb_embedder()
    bookmark_since = _since_from_days(BOOKMARK_LOOKBACK_PRESETS[args.bookmark_lookback])
    print(f"[curation] embedder: model={embedder.model} provider={embedder.provider}  "
          f"bookmark-lookback={args.bookmark_lookback} "
          f"(posts written since {bookmark_since or 'the beginning'})")
    conn = schema.connect()
    try:
        out = curation_pull(conn, embedder, x_profile=args.x_profile,
                            substack_profile=args.substack_profile, x_limit=args.x_limit,
                            bookmark_since=bookmark_since,
                            tiered=args.tiered, sufficient_at=args.sufficient_at)
    finally:
        conn.close()
    print("[curation] summary:\n" + _json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
