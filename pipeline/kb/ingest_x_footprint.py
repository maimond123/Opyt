"""
pipeline/kb/ingest_x_footprint.py — a confirmed Oracle's OWN X timeline → OPINION atoms.

Stage-5 footprint ingest for the ROOT channel: `discover_profile` only expands OFF-X sources, so
an Oracle's primary X channel is otherwise absent from the corpus. Pulls the Oracle's last-~6mo
timeline over x.com's internal GraphQL API on the user's own session cookies — the UNION of the
Posts and Replies tabs, since Posts alone omits standalone replies — and lands each
self-thread/original post as a full-body, chunked opinion atom. FREE: no third-party key, no
per-read billing, and nobody outside this machine sees which accounts are being read.

Distinct from `ingest_x.sync_bookmarks` (user's saves, keyed `x:{tweet_id}`) — this uses its own
atom-id namespace `xprofile:{root_id}` so a footprint thread and a bookmark of the same root never
collide and drop the richer version. `source_type` stays "x" so retrieval treats them as one family.

Curation filter: drop RTs and replies-to-others, keep originals + self-threads, no length gate.
Filtering happens BEFORE stitching by conversationId — LOAD-BEARING, since conversationId includes
the author's replies to commenters, not just the self-thread. One self-thread = one atom (chunked
for embedding). Thread images are all VLM-described.

Dedup is via content-hash (`snapshot_and_hash` on `raw_hash`), not skip-on-atom-id-presence: X's
timeline is one bulk fetch, so every group is re-rendered each run and only re-embedded when its
hash changed (which also captures thread growth).

No eligibility gate: this pull is single-author by construction — one account's own timelines,
filtered to that account — so the multi-author trust-laundering gate (`pipeline.kb.eligibility`)
doesn't apply.

Fail-safe: a group that fails to render/embed/write SKIPS (no atom, no `seen` mark). A missing key
or open credit breaker fails LOUD at the fetch.

"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from pipeline.timeparse import utc_now

from . import derive, link_router, schema
from .embed import assert_model
from .ingest_common import (AtomSink, BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL,
                            StageTimer, body_fields, llm_run_marker,
                            llm_run_stats, run_concurrent, snapshot_and_hash)

_DEFAULT_LOOKBACK_DAYS = 183       # ~6 months — cost-bounded default footprint window
_MAX_LOOKBACK_DAYS = 730           # ~2 years — hard ceiling on the window. The pull is free now,
                                   # ephemeral stream so `since` clamps rather than honors a longer ask
_FETCH_CAP = 5000                  # safety cap on tweets pulled per run
# Footprint mirrors the bookmark ingest: parallel per-group producers (OCR/VLM + article fetch) feed
# ONE serial consumer owning conn + the batching embed sink. Real fan-out is capped lower by the
# shared AIMD gates at the VLM/embed seams; this is only the producer-thread ceiling.
_INGEST_WORKERS = int(os.environ.get("OPYT_INGEST_WORKERS", "20"))
_CACHE_FLUSH_EVERY = 16           # new media reads before the consumer persists img_cache (crash bound)

# Substance filter: a post resting on the author's OWN words under this length is a fragment. Naked
# = no attachment, a decorative photo, or a bare (non-github/paper/Substack) link. A quote or a
# dispatchable link still provisionally keeps (David 2026-07-20).
_NAKED_MIN_CHARS = 200
_URL_IN_TEXT_RE = re.compile(r"https?://\S", re.I)


def _author_chars(group: list[dict]) -> int:
    """Chars of the AUTHOR's OWN words across the group (tweet text only — EXCLUDES the rendered
    media transcription, the quoted post, and link expansions). The naked-substance measure."""
    return sum(len((t.get("text") or "").strip()) for t in group)


def _has_substantive_media(group: list[dict]) -> bool:
    """Any of the AUTHOR's OWN photos the OCR cascade read as document/chart (substance=True). A
    quoted image (someone else's) renders as context but does NOT promote the atom to an artifact."""
    from .vision import _photos
    return any((m.get("media_read") or {}).get("substance")
               for t in group for m in _photos(t))


# Dispatchable = github / research paper / Substack, the set that becomes its own atom later; any
# other link counts as no retrievable substance, same as a decorative photo (David 2026-07-20).
# Re-exported from `link_router` (moved there 2026-08-13 so Hopper shares one host table) under the
# old private names — these must keep answering on the narrower artifact vocabulary only.
_PAPER_HOSTS = link_router._PAPER_HOSTS
_classify_link = link_router.classify_link
_atom_present = link_router.atom_present


def _post_urls(tweet: dict) -> list[str]:
    """The tweet's EXPANDED outbound urls (`entities.urls[].expanded_url`). The text carries only
    shortened t.co links, so classification MUST read the expanded field (populated on the profile
    fetch — live-verified)."""
    return [u.get("expanded_url", "") for u in (tweet.get("entities") or {}).get("urls") or []
            if u.get("expanded_url")]


def _dispatchable_link(urls: list[str]) -> bool:
    """True if any url is github / a research paper / Substack — the set that becomes its own atom
    (github/paper now, Substack later), so its post is KEPT (provisional) rather than judged as
    effectively naked. A SUPERSET of what Step 3 actually mints."""
    return any(_classify_link(u) for u in urls)


def _keep_group(group: list[dict], *, has_article: bool) -> tuple[bool, str]:
    """The substance decision for ONE stitched group → (keep, reason). Kept regardless of length
    if it's a thread, an X-article/quoted-article, or has substantive media; otherwise it faces the
    naked 200-char bar ("""
    if len(group) > 1:
        return True, "thread"
    if has_article:
        return True, "article"
    root = group[0]
    qt = root.get("quoted_tweet")
    if isinstance(qt, dict) and qt.get("article"):
        return True, "quoted-article"
    if _has_substantive_media(group):
        return True, "media-substance"
    # Past here media is decorative; only a quote or a dispatchable link keeps the post alive.
    urls = _post_urls(root)
    if isinstance(qt, dict) or _dispatchable_link(urls):
        return True, "provisional-keep"
    from .vision import _photos
    has_bare_link = bool(urls) or bool(_URL_IN_TEXT_RE.search(root.get("text") or ""))
    label = "bare-link" if has_bare_link else ("decorative-media" if _photos(root) else "naked")
    if _author_chars(group) >= _NAKED_MIN_CHARS:
        return True, f"{label}>=200"
    # Deep-probe LAST resort before the drop.
    if has_bare_link and any(link_router.classify_link_deep(u) for u in urls):
        return True, f"{label}-deep-paper"
    return False, f"{label}<200"


def _dedup_by_id(raw: list[dict]) -> list[dict]:
    """Order-preserving dedup by tweet id (the API repeats tweets across pages). Shared by the
    curation filter AND the engagement capture, so both always see the same tweet set."""
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for t in raw:
        tid = str(t.get("id", ""))
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            unique.append(t)
    return unique


def _majority_author(unique: list[dict]) -> tuple[str, str]:
    """(numeric id, handle) of the pull's author. The author of this pull is uniform — the walk
    filters each timeline to the account it asked for, which the Replies tab makes load-bearing
    (it ships whole conversations, 45% of them other people's tweets);
    majority vote guards a stray. ("", "") only if no tweet carries the field — then the filter
    drops any reply with a populated target (conservative: a footprint frankenstein is worse
    than a missed reply) and the engagement capture writes nothing (no attributable observer)."""
    from collections import Counter

    ids = [str((t.get("author") or {}).get("id")) for t in unique if (t.get("author") or {}).get("id")]
    handles = [(t.get("author") or {}).get("userName") for t in unique
               if (t.get("author") or {}).get("userName")]
    return (Counter(ids).most_common(1)[0][0] if ids else "",
            Counter(handles).most_common(1)[0][0] if handles else "")


def extract_engagements(raw: list[dict]) -> list[dict]:
    """PURE (no DB / network): every reply / quote / mention observation in one account's pull —
    read from the WHOLE deduped pull, BEFORE the curation filter drops replies-to-others, because
    the dropped bucket (~49% of the corpus) is exactly where the engagement signal lives.

    Rules (W0, 2026-08-07):
      • Targets key on NUMERIC ids (`x:user:{id}`) whenever the payload carries one — reply
        `inReplyToUserId` and `quoted_tweet.author.id` do. A handle-only target is stored as
        `x:@{handle}` VERBATIM (screen names are [A-Za-z0-9_]; never slugified) for later
        resolution. `inReplyToUsername` is never read: `x_graphql._normalize` does not emit it at
        all, and the twitterapi.io shape this rule was first written against left it empty 100% of
        the time. The numeric id is the only field either source populated reliably.
      • Self-acts are NOT engagements: a self-reply is a thread continuation, a self-quote/self-
        mention is thread plumbing. Excluded by id, and by handle (case-insensitive) when the
        mention carries no id.
      • Quote targets come from the `quoted_tweet` OBJECT, never the rendered markdown.
      • Mentions come from ALL thread tweets, not just roots.
      • No attributable observer (no author id anywhere) → no rows. No resolvable target → no row.
      • `observed_at` = the TWEET's own date (day precision) — when the engagement happened.
    """
    unique = _dedup_by_id(raw)
    author_id, author_handle = _majority_author(unique)
    if not author_id:
        return []
    observer = f"x:user:{author_id}"
    own_handle = (author_handle or "").lower()

    out: list[dict] = []
    seen: set[tuple] = set()

    def _add(kind: str, target: str, tid: str, when: str | None) -> None:
        key = (kind, target, tid)
        if target and key not in seen:
            seen.add(key)
            out.append({"observer_id": observer, "kind": kind, "target_id": target,
                        "src_ref": tid, "observed_at": when})

    for t in unique:
        if t.get("isRetweet"):          # a re-share's engagement surfaces aren't the author's acts
            continue
        tid = str(t.get("id", ""))
        if not tid:
            continue
        when = derive._day(t.get("createdAt", "")) or None

        reply_uid = str(t.get("inReplyToUserId") or "")
        if t.get("isReply") and reply_uid and reply_uid != author_id:
            _add("reply", f"x:user:{reply_uid}", tid, when)

        qt = t.get("quoted_tweet")
        if isinstance(qt, dict):
            qa = qt.get("author") or {}
            q_uid = str(qa.get("id") or "")
            if q_uid:
                if q_uid != author_id:
                    _add("quote", f"x:user:{q_uid}", tid, when)
            elif qa.get("userName"):
                if str(qa["userName"]).lower() != own_handle:
                    _add("quote", f"x:@{qa['userName']}", tid, when)

        for m in (t.get("entities") or {}).get("user_mentions") or []:
            m_uid = str(m.get("id_str") or m.get("id") or "")
            if m_uid:
                if m_uid != author_id:
                    _add("mention", f"x:user:{m_uid}", tid, when)
            elif m.get("screen_name"):
                if str(m["screen_name"]).lower() != own_handle:
                    _add("mention", f"x:@{m['screen_name']}", tid, when)
    return out


def _filter_and_stitch(raw: list[dict]) -> list[list[dict]]:
    """Dedup → filter (drop RTs + replies-to-OTHERS, keep originals + self-replies) → stitch by
    conversationId. Filter BEFORE stitch is LOAD-BEARING (this mirrored `x_render.sync_profile`
    814→831 before that function was deleted 2026-08-14; the ordering is now stated only here):
    conversationId is the WHOLE conversation incl. the author's replies to commenters, so
    grouping first would weld a self-thread essay to its replies-to-commenters into one Frankenstein
    atom. Returns chrono-sorted thread groups (solo posts = 1-element lists). Pure + offline — the
    unit tests hit THIS, no DB / embedder / network.

    Reply target = `inReplyToUserId` compared to the AUTHOR's own numeric id — NOT `inReplyToUsername`
    vs the handle. `x_graphql._normalize` does not emit `inReplyToUsername` at all, so a handle-string
    check would keep every reply outright. It was already the wrong field before the source changed:
    twitterapi.io's profile-fetch shape left it EMPTY (None) on every reply (LIVE-VERIFIED
    2026-07-19: 0/136 populated for @martin_casado) while `inReplyToUserId` was 100% populated — so
    the handle-string check silently kept EVERY reply and welded a person's replies-to-commenters
    into 16-tweet atoms. We KEEP a reply ONLY when it's a self-reply (target ==
    author = a thread continuation); every other reply (target is someone else, OR target unknown) is
    dropped — its engagement signal is captured by `extract_engagements` BEFORE this drop, so the
    drop costs curation noise, not data. Comparing IDs also beats a leading-`@` heuristic (a
    self-continuation like "@x @y Nevermind, I'm an idiot" carries carried-over thread-participant
    mentions but replies to the AUTHOR)."""
    from pipeline.ingestion.x_render import _stitch_threads

    unique = _dedup_by_id(raw)
    author_id, _ = _majority_author(unique)

    kept: list[dict] = []
    for t in unique:
        if t.get("isRetweet"):          # from: excludes these; belt-and-suspenders (a re-share isn't theirs)
            continue
        # KEEP originals + SELF-replies (continuations); DROP replies to anyone else (or unknown target).
        if t.get("isReply") and str(t.get("inReplyToUserId") or "") != author_id:
            continue
        kept.append(t)                   # NO length / engagement gate — short aphorisms stay

    groups = _stitch_threads(kept)       # {conv_id: [chrono tweets]} — group AFTER the filter
    return [g for g in groups.values() if g]


# ── Step-3 link dispatch: a referenced github/paper → its OWN artifact atom ──

class LinkDispatcher:
    """Each github / paper / Substack-post link in an Oracle's OWN tweets (NOT quoted nodes — a
    quoted tweet's links are the QUOTED author's references, not this Oracle's act) → its OWN
    artifact atom, minted once and entered as `author_referenced`.

    Author credibility is irrelevant here: `who_id` is always the artifact's own author, never the
    Oracle.

    State (ledgers, the in-flight set, the batching sink) is owned by ONE object because it's all
    per-run and mutated only on the serial consumer thread. `_pending` holds atom_ids submitted to
    the sink but not yet durable, and `mint_artifact` reads it as `in_flight` so a second tweet
    referencing the same artifact rides the first one's flush instead of re-fetching and
    re-embedding it. `_on_written` clears an id once it lands. `_pending` ("am I already working on
    this, marked at submit") and `seen` ("is this durably stored") answer different questions and
    must not be merged.

    It used to carry a who_id list per id, to write a `references` vouch edge once the artifact
    landed. The `edges` table was deleted 2026-08-23 for having no reader, so only the in-flight
    half survives — see docs/plans/2026-08-23-delete-edges-and-trust-tiers.md. Do NOT simplify this
    away with the vouches: dropping it silently re-mints every repeat reference in a flush window.

    """

    def __init__(self, conn, embedder, *, sink=None, img_cache: dict | None = None,
                 prefetched: dict | None = None):
        from collections import Counter

        self.conn = conn
        self.embedder = embedder
        self.sink = sink
        self.img_cache = img_cache
        self.prefetched = prefetched or {}     # url → payload from the parallel fetch phase
        self.paper_seen: dict = {}
        self.gh_seen = schema.load_hashes(conn, "github")
        self.sub_seen = schema.load_hashes(conn, "substack")
        self._pending: set[str] = set()            # atom_ids submitted but not yet durable
        self.kinds: Counter = Counter()
        self.prefetch_hits = 0

    # ── in-flight bookkeeping ───────────────────────────────────────────────────────────────
    def _on_written(self, artifact_id: str) -> None:
        """Fired by the sink when `artifact_id` is durable — it is no longer in flight."""
        self._pending.discard(artifact_id)

    def _mark_in_flight(self, artifact_id: str) -> None:
        """Track a just-minted artifact IFF it is buffered rather than durable.

        `_atom_present` is exactly the right question HERE and exactly the wrong one for "did the
        mint succeed?" — it distinguishes durable-now from buffered, which is all this needs. The
        caller must already have established that the artifact is one or the other."""
        if not _atom_present(self.conn, artifact_id):
            self._pending.add(artifact_id)

    # ── dispatch ────────────────────────────────────────────────────────────────────────────
    def dispatch(self, group: list[dict], oracle_who_id: str) -> "Counter":
        """One group → a kind→count tally of artifacts vouched (minted, already-present, or queued).

        Cheap + idempotent on a re-run: a PRESENT github/paper artifact is a DB check + an idempotent
        edge write, no network. (Substack has no url-derived id — its atom keys on the post's numeric
        id — so it re-fetches to dedup, but skips the re-embed.) Every failure is swallowed — a bad
        link never breaks the footprint pull (fail-safe)."""
        from collections import Counter

        from pipeline.ingestion.utils import log

        kinds: Counter = Counter()
        urls, seen_u = [], set()
        for t in group:                      # OWN nodes only
            for u in _post_urls(t):
                if u not in seen_u:
                    seen_u.add(u)
                    urls.append(u)

        for u in urls:
            kind = _classify_link(u)
            mint_url, content_type = u, None
            if kind is None:
                # Bounded on purpose: `dispatch()` only ever sees urls from a group that already
                # survived `_keep_group`, so this fallback runs on a small, already-filtered set —
                # not on every link of every tweet the pull touched.
                deep = link_router.classify_link_deep(u)
                if deep:
                    kind, mint_url, content_type = deep
            if kind not in ("github", "paper", "substack"):      # bare links are not dispatchable
                continue
            try:
                vouched = self._dispatch_one(mint_url, kind, oracle_who_id, content_type=content_type)
            except Exception as e:           # a single bad link never breaks the pull
                log(f"[footprint] link dispatch failed for {u}: {e}")
                vouched = False
            if vouched:
                kinds[kind] += 1
        self.kinds.update(kinds)
        return kinds

    def _dispatch_one(self, u: str, kind: str, who_id: str, *, content_type: str | None = None) -> bool:
        """One url → True if the artifact is in the store or on its way there.

        The MINT moved to `link_router.mint_artifact` (2026-08-13) so Hopper shares it; what stays
        here is this caller's own half — turning a mint OUTCOME into the in-flight bookkeeping that
        stops the NEXT tweet re-fetching the same artifact inside one flush window.

        `content_type` — the deep probe's Content-Type, when `u` came from `classify_link_deep`
        rather than the free host-list check; None on every other path."""
        pre = self.prefetched.get(u)
        res = link_router.mint_artifact(
            self.conn, self.embedder, u, kind, entry_mode="author_referenced",
            paper_seen=self.paper_seen, gh_seen=self.gh_seen, sub_seen=self.sub_seen,
            img_cache=self.img_cache, sink=self.sink, on_written=self._on_written,
            prefetched=pre, in_flight=self._pending, content_type=content_type)
        self.prefetch_hits += res["used_prefetch"]

        status, aid = res["status"], res["atom_id"]
        if status == "failed" or not aid:
            return False
        # 'present' (already durable) and 'in-flight' (buffered by an earlier tweet, already
        # tracked) both need nothing. Only a fresh mint may still be sitting in the sink's buffer.
        if status not in ("present", "in-flight"):
            self._mark_in_flight(aid)
        return True


def _resolve_since(since: datetime | None, now: datetime) -> datetime:
    """The effective lower bound for the X pull: `since` (default ~6mo), FLOORED at the hard
    2-year ceiling. However far back a caller asks, the window never exceeds
    `_MAX_LOOKBACK_DAYS` — a longer request is clamped, not honored. Pure (takes `now`) so the
    default + clamp are trivially testable."""
    since = since or (now - timedelta(days=_DEFAULT_LOOKBACK_DAYS))
    floor = now - timedelta(days=_MAX_LOOKBACK_DAYS)
    return max(since, floor)


def _prefetch_one_artifact(url: str, kind: str) -> dict | None:
    """Fetch (never write) everything needed to mint one artifact. Pure network — no conn, no
    embedder, no shared mutable state — which is what makes it safe to run on a pool thread."""
    from . import ingest_github, ingest_papers

    if kind == "github":
        gh = ingest_github._github_owner_repo(url)
        if not gh:
            return None
        repo = ingest_github._fetch_repo(*gh)
        if not repo or not repo.get("name"):
            return None
        owner = (repo.get("owner") or {}).get("login", "") or gh[0]
        from pipeline.ingestion.sources.github import _fetch_readme
        return {"repo": repo, "readme": _fetch_readme(owner, repo.get("name", ""))}
    paper = ingest_papers.paper_from_url(url)          # S2 enrich
    if not paper:
        return None
    return {"paper": paper, "fulltext": ingest_papers.resolve_fulltext(paper)}   # PDF pull


def prefetch_referenced_artifacts(groups: list, conn, *, workers: int) -> dict:
    """Fetch every referenced github repo / paper up front, one future per artifact, so the serial
    consumer's link dispatch doesn't serialize network fetches behind its writes.

    Returns `{url: payload}`; a url that fails to fetch is simply absent and dispatch fetches it
    inline instead. Does NOT skip already-present artifacts by DB lookup, since the github atom_id
    depends on API-canonical owner casing only known after the fetch.

"""
    from concurrent.futures import ThreadPoolExecutor

    from pipeline.ingestion.utils import log

    todo: dict[str, str] = {}                 # url → kind, insertion-ordered, deduped
    seen_links = 0
    for group in groups:
        for tweet in group:                   # OWN nodes only — mirrors LinkDispatcher.dispatch
            for u in _post_urls(tweet):
                kind = _classify_link(u)
                if kind not in ("github", "paper"):    # substack is not sink-routed yet
                    continue
                seen_links += 1
                todo.setdefault(u, kind)
    if not todo:
        return {}

    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(len(todo), workers),
                            thread_name_prefix="artifact-pre") as ex:
        futs = {ex.submit(_prefetch_one_artifact, u, k): u for u, k in todo.items()}
        for f in futs:
            u = futs[f]
            try:
                payload = f.result()
            except Exception as e:            # fail-safe: dispatch re-fetches this one inline
                log(f"[footprint] artifact prefetch failed for {u}: {e}")
                payload = None
            if payload:
                out[u] = payload
    return {"payloads": out, "links": seen_links, "unique": len(todo), "fetched": len(out)}


# ── stage nesting: which timed stages sit INSIDE which, so the UNtimed remainder is reportable ──
# Declared beside the stage names it refers to, deliberately: this map's whole job is to go stale
# loudly. It covers ONE adapter rather than `StageTimer` itself — nesting semantics on the shared
# class would touch five ingesters to answer a question only this one is asking.
_STAGE_NESTING = {
    # `article_fetch` left this map on 2026-08-30: an X Article's body now arrives as a field on
    # the timeline walk, so there is no fetch to time. A dead name here would contribute 0 seconds
    # and read as "measured and negligible" — the exact misreading the residual exists to prevent.
    "produce": ("vlm", "render", "snapshot"),
    "consume": ("link_dispatch", "cache_flush", "embed", "write"),
}


def _stage_residual(totals: dict[str, float]) -> dict[str, float]:
    """Seconds inside a parent stage that NO child stage measured.

    Untimed work contributes 0 to a profile and 100% to the clock, so a profile assembled only from
    timed stages silently over-attributes cost to whatever happens to be timed — and the reader
    cannot tell, because 'small' and 'unmeasured' are the same reading. Declaring the nesting turns
    that gap into arithmetic: a nonzero residual names the parent that is hiding work. Drift fails
    SAFE: an unregistered child stage inflates its parent's residual rather than hiding in silence.
"""
    return {p: round(totals[p] - sum(totals.get(c, 0.0) for c in kids), 3)
            for p, kids in _STAGE_NESTING.items() if p in totals}


# One request per page against a 50/15-min bucket, per timeline. The ceiling below the transport's
# own `_USERTWEETS_MAX_PAGES` would be a second cap on the same thing, so this passes the transport
# cap and lets it clamp — the date window is what actually ends a real walk (measured: 2 requests
# for a 340-day window on a normal account, 26 for the most prolific oracle over 183 days).
_TIMELINE_PAGES = 60


def _pull_own_timeline(cookies: dict, headers: dict, user_id: str,
                       since_ts: int, cap: int) -> list[dict]:
    """One account's OWN tweets from `since_ts` to now — the union of X's two user timelines.

    BOTH are required for parity, and this is the part that is easy to get wrong. The Posts tab
    (`UserTweets`) omits standalone replies to other accounts: measured against twitterapi.io over
    one 108-day window, 200 of the 214 tweets only twitterapi.io returned were replies. Walking
    posts alone loses that entire population silently, because a short walk of a quiet account
    looks exactly like a complete one.

    The two operations hold INDEPENDENT rate buckets (50 per 15 minutes each), so the union costs
    two walks but not two waits.

    Trimmed to the window on the way out. The walk overshoots `since_ts` by up to one page — its
    date stop fires only after a page has landed — and `since` is a bound the caller chose, not an
    approximation. `_dedupe_tweets` is the same pure helper the old paid path used; a tweet can
    legitimately appear on both timelines.

    Does NOT sleep. A walk that hits X's window budget raises `XRateLimited` out of here with
    nothing written, and the rail resumes it on the next run — which is the fail-safe direction,
    and the reason pacing is not hidden inside this call."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion import x_render as xt

    walked: list[dict] = []

    def _reached_the_window(acc: list[dict]) -> bool:
        # `cap` counts across BOTH walks, not per walk — it is a bound on the run. Reading only
        # `acc` (this walk's own accumulator) would let the second timeline start a fresh budget
        # and fetch up to 2x the cap before the trim below threw half of it away.
        if len(walked) + len(acc) >= cap:
            return True
        stamps = [xt._parse_twitter_date(t.get("createdAt", "")) for t in acc]
        oldest = min((d.timestamp() for d in stamps if d), default=None)
        return oldest is not None and oldest < since_ts

    for timeline in ("posts", "replies"):
        walked += core.fetch_user_tweets(cookies, headers, user_id, pages=_TIMELINE_PAGES,
                                         timeline=timeline, after_page=_reached_the_window)

    def _in_window(t: dict) -> bool:
        d = xt._parse_twitter_date(t.get("createdAt", ""))
        return d is not None and d.timestamp() >= since_ts

    return xt._dedupe_tweets([t for t in walked if _in_window(t)])[:cap]


def sync_x_footprint(conn: sqlite3.Connection, embedder, *, handle: str,
                     author_name: str | None = None, since: datetime | None = None,
                     limit: int = 0) -> dict:
    """Ingest a confirmed Oracle's OWN X timeline (from `since`..now) as opinion atoms.

    `handle` names the account; `since` bounds the window (default ~6mo); `limit` caps NEW/CHANGED
    atoms per run (0 = all — a resumable partial backfill, like the other footprints).

    FREE: x.com's internal GraphQL API on this machine's own X session cookies. There is no key and
    no non-browser path — a headless deployment cannot run this at all, which is the deliberate
    trade taken when twitterapi.io was removed on 2026-08-30. A dead or absent cookie raises
    `SyncAuthError`; an unreadable account returns `undetermined` rather than a silent 0.
    Returns a run summary. Does NOT run entity resolution — the caller re-resolves after footprint
    expansion so the author folds into the Oracle's canonical."""
    from collections import Counter

    from pipeline.ingestion import x_render as xt
    from pipeline.ingestion.utils import log
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from .vision import enrich_tweet_media_cascade, prefetch_group_media

    assert_model(conn, embedder)          # guard the store's embedding identity BEFORE any spend
    llm0 = llm_run_marker()               # mark before the first paid call so the summary reports
                                           # THIS run's LLM calls, not the process-cumulative total
    h = (handle or "").lstrip("@")
    if not h:
        return {"source": "x-footprint", "error": "no handle"}

    now = utc_now()
    effective = _resolve_since(since, now)
    if since is not None and effective > since:
        log(f"[footprint] x @{h}: requested window exceeds the {_MAX_LOOKBACK_DAYS}-day (2yr) cap "
            f"— clamped to {effective:%Y-%m-%d}")
    since = effective
    since_ts = int(since.timestamp())

    # A handle names the account; the timeline walk keys on the numeric `rest_id`, which is also
    # what its author filter compares against. One extra request on a 150/15-min bucket.
    from pipeline.ingestion import x_graphql_core as core
    cookies = core.read_x_cookies()
    headers = core.auth_headers(cookies, f"https://x.com/{h}")
    profile = core.fetch_user_profile(cookies, headers, h)
    if not profile:
        # Suspended, deactivated, protected, or renamed. A FACT about the account, not a transient
        # miss — but `undetermined` is still the honest verdict, because this path cannot tell
        # which, and the three have different answers.
        log(f"[footprint] x @{h}: profile unreadable (suspended / protected / renamed?)")
        return {"source": "x-footprint", "fetched": 0, "added": 0, "skipped": 0, "failed": 0,
                "undetermined": 1, "error": "x profile unreadable"}

    log(f"[footprint] x @{h}: posts + replies since {since:%Y-%m-%d} "
        f"(cap {_FETCH_CAP} tweets) — FREE, this session's own cookies")
    raw = _pull_own_timeline(cookies, headers, profile["user_id"], since_ts, _FETCH_CAP)
    log(f"[footprint] x @{h}: fetched {len(raw)} tweets")

    # Image descriptions cached by URL (immutable CDN links) → re-runs are free + keep the atom's
    # snapshot HASH stable (an un-cached VLM would vary the text → spurious re-embeds).
    img_cache = load_image_cache(opyt_home())
    seen = schema.load_hashes(conn, "x")   # SHARED "x" ledger; footprint keys (xprofile:) never
                                           # collide with bookmark keys (x:) — separate namespaces.
    groups = _filter_and_stitch(raw)
    # Record every reply/quote/mention observation from the WHOLE pull, including replies-to-others
    # the curation filter just dropped — the walk already fetched them. Fail-safe: a capture error
    # never costs the atoms.
    engagements = 0
    try:
        engagements = schema.record_engagements(conn, extract_engagements(raw))
    except Exception as e:
        log(f"[footprint] x @{h}: engagement capture failed (atoms unaffected): {e}")
    added = skipped = failed = threads = dispatched = submitted = 0   # consumer-owned (single writer)
    dispatch_kinds: Counter = Counter()
    dropped = 0                           # producer-phase tallies (many threads) → ride counts_lock
    drop_reasons: Counter = Counter()     # why a group was cut (fragment) — observability
    keep_reasons: Counter = Counter()     # why a group survived (artifact / naked>=200 / provisional)
    media_kinds: Counter = Counter()      # OCR cascade routes across the run (document/chart/photo)
    counts_lock = threading.Lock()

    home = opyt_home()
    timer = StageTimer()
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    # A BATCHING sink, not per-atom store_atom: a 6-month footprint is 100s of thread-atoms, and a
    # per-atom embed on the consumer would re-serialize the very cost the producer pool parallelizes.
    # 8×batch_size flush lets the Phase-3 embed gate probe 8-way. One consumer owns conn + this sink.
    sink = AtomSink(conn, embedder, timer=timer, flush_chunks=8 * bs)
    cache_pending = 0
    # Images read INSIDE a producer thread rather than by the prefetch. After a prefetch this should
    # be 0; anything above it means the prefetch leaked and that image was read serially — the exact
    # behavior the prefetch removes. Reported so a partial regression is a NUMBER, not a slow run
    # that looks fine.
    late_reads = 0

    def _produce(group: list[dict]):
        """PRODUCER (pool thread): the network-bound per-group work — OCR/VLM the images, read
        any X-Article body, render, hash. Touches ONLY per-group locals + thread-safe shared state (the
        AIMD-gated VLM seam, the locked img_cache, the counts_lock-guarded tallies). No conn
        writes. Returns a write-ready result — even for an UNCHANGED group, because the consumer still
        dispatches its referenced links — or None to skip (no root / dropped fragment / over-limit)."""
        nonlocal dropped
        root = group[0]                    # chronologically first tweet = the atom's canonical
        root_id = str(root.get("id", ""))
        if not root_id:
            return None
        if limit and submitted >= limit:   # best-effort early skip (STRICT cap is on the consumer below)
            return None
        atom_id = f"xprofile:{root_id}"
        is_thread = len(group) > 1

        # OCR cascade for ALL images (transcribe → route → chart-VLM), mutating each node in place with
        # `description` + `media_read` (the substance verdict _keep_group reads). Cached URLs are free.
        n_reads = 0
        with timer.stage("vlm"):
            for t in group:
                _n, node_reads = enrich_tweet_media_cascade(t, img_cache)
                n_reads += _n
                if node_reads:
                    with counts_lock:
                        media_kinds.update(mr.kind for mr in node_reads)

        # X-Article body — a FIELD READ, not a fetch. The timeline walk asks for
        # `withArticleRichContentState`, so a tweet that carries an Article arrives with the whole
        # `content_state` already attached (`x_graphql._normalize` carries the node verbatim, and
        # recurses into `quoted_status_result`, so a QUOTED article needs no separate handling
        # either). This used to be one paid `/twitter/article` call per article, per run.
        #
        # An X ARTICLE whose body did not arrive renders as its teaser tweet — a real fragment of a
        # real long-form post, and the atom must say so rather than look complete. That is still
        # possible without a fetch to fail: X returns a TEASER node (cover image, title,
        # `preview_text`, no blocks) whenever `withArticleRichContentState` is off or ignored, and
        # it is truthy, so `bool(article)` cannot be the test. Ask whether it has a BODY.
        article = None
        article_incomplete = False
        for t in group:
            if not xt._article_tweet_id(t):
                continue
            article = t.get("article")
            _, blocks = xt._article_shape(article or {})
            article_incomplete = not blocks
            if article_incomplete:
                log(f"[footprint] x @{h}: article {t.get('id')} arrived without a body "
                    f"(rendering the teaser) — check `withArticleRichContentState`.")
            break

        # Substance filter: drop a fragment (naked post under the length bar) BEFORE the paid embed.
        # Runs AFTER media-read + article fetch so the artifact signals are known.
        keep, reason = _keep_group(group, has_article=bool(article))
        if not keep:
            with counts_lock:
                dropped += 1
                drop_reasons[reason] += 1
            return None
        with counts_lock:
            keep_reasons[reason] += 1

        meta = derive.derive_x(root)
        with timer.stage("render"):
            md = xt.tweet_to_markdown(root, article=article,
                                      thread_tweets=group if is_thread else None,
                                      source="x-profile", footer_label="Profile extract")
        # Content-hash dedup: re-embed ONLY on a new/changed body (captures thread growth).
        with timer.stage("snapshot"):      # writes the snapshot file — the producer's other real I/O
            decided = snapshot_and_hash("x", atom_id, md, seen)
        return {"group": group, "root": root, "root_id": root_id, "atom_id": atom_id,
                "who_id": meta["who_id"], "meta": meta, "md": md, "is_thread": is_thread,
                "article": article, "reason": reason, "decided": decided, "n_reads": n_reads,
                "article_incomplete": article_incomplete}

    def _work(group: list[dict]):
        """`_produce` under a parent timer. The wrapper exists so the producer's children
        (vlm/render/snapshot) sum against a KNOWN total — see `stage_residual`."""
        with timer.stage("produce"):
            return _produce(group)

    def _mark_written(atom_id: str, raw_hash: str, is_thread: bool) -> None:
        """Fires AFTER an atom is durably written (consumer thread) → the summary counts DURABLE atoms."""
        nonlocal added, threads
        seen[atom_id] = raw_hash
        added += 1
        threads += int(is_thread)

    def _consume_body(res: dict) -> None:
        """CONSUMER (caller's thread, SERIAL): the sole owner of the write path. Dispatches the group's
        referenced links FIRST (always — the artifact/vouch backfill is independent of whether this body
        re-embeds), persists paid media reads on a cadence, then submits the atom to the batching sink
        when the snapshot changed."""
        nonlocal skipped, dispatched, cache_pending, submitted, late_reads
        if limit and submitted >= limit:            # STRICT cap on the serial consumer (matches the old
            return                                  # break: no dispatch, no submit past `limit` new atoms)
        group = res["group"]
        who_id = res["who_id"]
        # STEP-3 link dispatch — for EVERY kept group (even hash-unchanged) so an already-captured tweet
        # still backfills the referenced artifact + vouch. Consumer-only (writes conn + the dedup ledgers).
        with timer.stage("link_dispatch"):          # SERIAL: the fetch is prefetched, the write batches
            dk = dispatcher.dispatch(group, who_id)
        dispatched += sum(dk.values())
        dispatch_kinds.update(dk)
        cache_pending += res["n_reads"]
        late_reads += res["n_reads"]      # after a prefetch this should be 0; see `late_reads`
        if cache_pending >= _CACHE_FLUSH_EVERY:     # bound a crash's re-OCR cost (paid); save snapshots
            with timer.stage("cache_flush"):        # a whole-file JSON dump, on the serial consumer
                save_image_cache(home, img_cache)   # under the cache lock → safe vs concurrent producers
            cache_pending = 0

        decided = res["decided"]
        if decided is None:                         # snapshot unchanged → skip the (paid) embed
            skipped += 1
            return
        raw_ref, raw_hash = decided
        meta = res["meta"]
        root = res["root"]
        # Persist the author as a person with the @handle (same convention as ingest_x); idempotent.
        site = meta.get("who_site")
        schema.upsert_entity(
            conn, who_id, name=meta.get("who_name") or author_name,
            identity_links=[site] if site else None,
            profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)

        atom = {
            "atom_id": res["atom_id"],
            "source_type": "x",           # X content — one family with bookmarks for retrieval
            "what_kind": "opinion",
            "who_id": who_id,
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "about_entities": meta["about_entities"],  # @-mention slugs
            "source_url": root.get("url") or f"https://x.com/{h}/status/{res['root_id']}",
            "raw_ref": raw_ref,
            "raw_hash": raw_hash,
            "description": meta["description"],
            # Structural signals (observability + the deferred substantiveness view). This is the
            # AUTHOR's footprint, so — unlike a bookmark — there is NO user curation act to record.
            "payload": {
                "like_count": root.get("likeCount", 0),
                "reply_count": root.get("replyCount", 0),
                "is_thread": res["is_thread"],
                "thread_len": len(group),
                "is_article": bool(res["article"]),
                # Derive from the quoted OBJECT, not isQuote — the profile fetch nulls the flag on
                # real quote tweets (see tweet_to_markdown's quote-render note), so the raw bool lies.
                "is_quote": bool(root.get("isQuote") or root.get("quoted_tweet")),
                "has_media": any((t.get("extendedEntities") or {}).get("media") for t in group),
                "media_substance": _has_substantive_media(group),   # OCR found a doc/chart artifact
                "keep_reason": res["reason"],                       # substance-filter verdict (observability)
                "source_tags": meta["source_tags"],                 # hashtags — author-declared (§6)
                # PARTIAL only if this author's own ARTICLE failed to load; a failed quoted-article
                # fetch doesn't count (that's someone else's content, rendered as context only).
                **body_fields(BODY_PARTIAL if res["article_incomplete"] else BODY_COMPLETE,
                              BASIS_OBSERVED),
            },
            "entry_mode": "oracle-footprint",   # NOT user-saved (curation) / crawled (radar)
        }
        aid, it = res["atom_id"], res["is_thread"]
        submitted += 1                              # count the NEW-atom submission toward the limit
        # Bookkeeping rides on_written so added/threads/seen count DURABLE atoms, not merely submitted.
        sink.submit(atom, res["md"],
                    on_written=(lambda a=aid, rh=raw_hash, t=it: _mark_written(a, rh, t)))

    def _consume(res: dict) -> None:
        """`_consume_body` under a parent timer. This one is the load-bearing measurement: the
        consumer is SERIAL, so its total IS wall clock — directly comparable to `process`, unlike
        `produce` (thread-seconds across the pool). `consume / process` ≈ 1 means no amount of
        producer parallelism can help and the serial write path is the floor."""
        with timer.stage("consume"):
            _consume_body(res)

    # PHASE 1+2: read every image FIRST (one future per image) so every lookup in `_work` is a cache
    # hit.
    # Skipped when `limit` is set: `limit` bounds spend, and prefetching the whole window would blow
    # that bound before `_work` gets to early-skip groups past it.
    prefetch = {}
    art_prefetch = {}
    if not limit:
        with timer.stage("media_prefetch"):
            prefetch = prefetch_group_media(
                groups, img_cache, workers=_INGEST_WORKERS,
                flush_every=_CACHE_FLUSH_EVERY, on_flush=lambda: save_image_cache(home, img_cache))
        log(f"[footprint] x @{h}: media prefetch — {prefetch['read']} read, "
            f"{prefetch['failed']} failed, {prefetch['images']} refs "
            f"({prefetch['dispatched']} unique+uncached)")
        # Referenced github/paper artifacts — the SAME granularity fix, one layer down. Skipped
        # under `limit` for the same reason the media prefetch is: it walks the whole window, and a
        # bounded run must not pay for artifacts of groups it will never ingest.
        with timer.stage("artifact_prefetch"):
            art_prefetch = prefetch_referenced_artifacts(groups, conn, workers=_INGEST_WORKERS)
        if art_prefetch:
            log(f"[footprint] x @{h}: artifact prefetch — {art_prefetch['fetched']} fetched of "
                f"{art_prefetch['unique']} unique ({art_prefetch['links']} refs)")

    # The dispatcher owns every ledger the SERIAL consumer mutates, plus the batching sink, so the
    # single-writer rule those unlocked dicts depend on is a property of one object rather than a
    # comment on four separate locals.
    dispatcher = LinkDispatcher(conn, embedder, sink=sink, img_cache=img_cache,
                                prefetched=art_prefetch.get("payloads") if art_prefetch else None)

    # ── PHASE 3: produce (pooled) → consume (serial). `process` is this phase's WALL clock, the
    # denominator every other number here is read against.
    with timer.stage("process"):
        run_concurrent(groups, _work, _consume, workers=_INGEST_WORKERS)
        with timer.stage("consume"):           # the tail is consumer work too, so it stays UNDER
            sink.close()                       # `consume` — else embed/write/cache_flush leak out of
            with timer.stage("cache_flush"):   # their parent and the residual goes negative
                save_image_cache(home, img_cache)   # backstop: persist reads not yet flushed
    log(f"[footprint] x @{h}: added {added}, dropped {dropped} {dict(drop_reasons)}, "
        f"skipped {skipped}, failed {failed}; media {dict(media_kinds)}; "
        f"dispatched {dispatched} {dict(dispatch_kinds)}")
    return {"source": "x-footprint", "fetched": len(raw), "groups": len(groups),
            "engagements": engagements,
            "added": added, "dropped": dropped, "skipped": skipped, "failed": failed,
            "threads": threads, "drop_reasons": dict(drop_reasons),
            "keep_reasons": dict(keep_reasons), "media_kinds": dict(media_kinds),
            "dispatched": dispatched, "dispatch_kinds": dict(dispatch_kinds),
            # late_reads > 0 means images were read inside a producer thread despite the prefetch —
            # a partial regression, surfaced as a number instead of just latency.
            "media_prefetch": prefetch, "late_reads": late_reads,
            # A url absent from the payload map failed prefetch; dispatch re-fetches it inline.
            "artifact_prefetch": art_prefetch, "prefetch_hits": dispatcher.prefetch_hits,
            # Parent−Σ(children) per nested stage — read `consume` against `process` (consumer is
            # SERIAL, so its total is wall clock; `produce` is thread-seconds, not comparable).
            "stage_residual": _stage_residual(timer.totals),
            "stage_seconds": timer.totals, "stage_latency": timer.distribution(),
            **llm_run_stats(llm0),
            "total": schema.count_atoms(conn, "x")}
