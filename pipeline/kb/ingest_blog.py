"""
pipeline/kb/ingest_blog.py — one blog post → an opinion atom: the per-post builder the blog
archive walk uses (`build_article_atom`, `oracles/sources/archive/blog.py`), plus the
single-article path Hopper's bare links use.

who_id = `blog:{canonical_host}`; atom_id = `canon_post_url`, path-preserving. Policy B skips an
existing atom before the paid fetch; a challenge page or thin body is skipped and counted, never
stored; a failed fetch or write leaves no `seen` mark.
"""

from __future__ import annotations

from pipeline.host_access import feed_input

import sqlite3
from datetime import datetime, timezone
from functools import partial
from urllib.parse import parse_qsl, urlencode, urlparse

from . import derive, schema, vision
from .ingest_common import (BASIS_ASSUMED, BODY_COMPLETE, FETCH_OK,
                            FETCH_UNDETERMINED, body_fields, classify_fetch,
                            promote_atom, changed_hash, store_atom)

# Tracking/analytics query params, not post identity. Stripped in `canon_post_url` so an
# RSS-sourced URL (which bolts on `?utm_*`) and a clean hub-harvested URL for the same post mint
# the same atom_id; a real identity param (WordPress `?p=123`) is kept.
_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "yclid", "twclid",
    "mc_cid", "mc_eid", "igshid", "_hsenc", "_hsmi",
})


def _strip_tracking_query(query: str) -> str:
    """Drop tracking params (`utm_` prefix + `_TRACKING_PARAMS`), keep the rest sorted so param
    order never defeats dedup. Returns the cleaned query without a leading `?`, or `""`."""
    kept = [(k, v) for (k, v) in parse_qsl(query, keep_blank_values=True)
            if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)]
    return urlencode(sorted(kept))


def canon_post_url(url: str) -> str:
    """Per-post atom id: `blog:{host}{path}[?query]`, path-preserving (unlike `canonical_identity`,
    which collapses every post to the bare host). Tracking params are stripped so the same post
    from RSS and from a hub page mints one id."""
    u = (url or "").strip()
    if not u:
        return "blog:"
    if "://" not in u:
        u = "https://" + u
    p = urlparse(u)
    host = (p.netloc or "").lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/")
    clean_q = _strip_tracking_query(p.query)
    query = f"?{clean_q}" if clean_q else ""
    return f"blog:{host}{path}{query}"


def _classify_article(article: dict | None) -> str:
    """A fetched page → FETCH_OK / FETCH_ABSENT / FETCH_UNDETERMINED. Both non-OK verdicts skip
    identically (no atom, no `seen` mark); split so a blocked host reads apart from a thin stub.
    Checks the response headers' `cf-mitigated` marker before falling back to a body heuristic.
"""
    if not article:
        return FETCH_UNDETERMINED   # the fetch itself failed — we learned nothing about the page
    return classify_fetch(article.get("content"), headers=article.get("headers"),
                          title=article.get("title", "") or "")


def _lastmod_before(lastmod: str, since: datetime) -> bool:
    """True iff the sitemap `lastmod` parses to a moment strictly before `since` (→ skip it).
    Unparseable lastmod → False (don't drop a post we can't date; `since` is a best-effort listing
    bound, mirroring the Substack archive walk, not a hard per-post filter)."""
    try:
        dt = datetime.fromisoformat((lastmod or "").replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt < since


# ── Date cascade ─────────────────────────────────────────────────────────────
# `_resolve_post_date` ladders free (no-LLM) date sources by decreasing exactness;
# `when_precision` travels with the date so a coarse source is never read as a known day.

def _feed_date_map(blog_url: str) -> dict[str, str]:
    """Map ``canon_post_url(link) -> "YYYY-MM-DD"`` from the site's RSS/Atom feed, when it has one.
    Fetched once per sync and cross-referenced for real publish dates, since sitemap `lastmod` is
    only a modification time. Keyed by ``canon_post_url`` so link variants collapse. Fail-safe:
    ``{}`` on any miss — the cascade falls through to the next rung."""
    base = (blog_url or "").rstrip("/")
    if not base:
        return {}
    try:
        import feedparser
    except Exception:
        return {}
    import time as _time
    for feed_path in ("/feed.xml", "/feed", "/atom.xml", "/rss.xml", "/rss", "/index.xml"):
        try:
            feed = feedparser.parse(feed_input(f"{base}{feed_path}"))
        except Exception:
            continue
        out: dict[str, str] = {}
        for entry in getattr(feed, "entries", None) or []:
            link = (entry.get("link") or "").strip()
            parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if link and parsed:
                out[canon_post_url(link)] = _time.strftime("%Y-%m-%d", parsed)
        if out:                       # first feed that yields dated entries wins
            return out
    return {}


def _resolve_post_date(htmldate_date: str, url: str, lastmod: str,
                       feed_dates: dict[str, str]) -> tuple[str, str]:
    """Date cascade → ``(YYYY-MM-DD, when_precision)``. Cheapest+most-exact first; every rung free.

        1. htmldate (on-page / URL-mined)  → ``day``      already computed by ``fetch_article``
        2. feed ``<pubDate>``              → ``day``      genuine publish date, feed-having sites
        3. sitemap ``lastmod``             → ``day``      modified-date; pre-existing behavior
        4. (nothing)                       → ``unknown``  never a fake day

    Feed outranks lastmod: a feed pubDate is a real publish date; lastmod is a modification time.
    """
    d = (htmldate_date or "")[:10]
    if d:
        return d, "day"
    fd = feed_dates.get(canon_post_url(url), "")
    if fd:
        return fd[:10], "day"
    if lastmod:
        return lastmod[:10], "day"
    return "", "unknown"


# ── The paid per-article stages, shared by the crawl and by a single hand-dumped url ──────────
# Shared by Hopper's single-article path and the crawl so both get identical treatment. One
# ordering constraint, load-bearing for every caller: the VLM pass runs before `changed_hash`
# — a description added after the hash never reaches the indexed surface.

def _stage(timer, name):
    """`timer.stage(name)`, or a no-op when the caller keeps no timer. A single dumped article has
    no run to profile; the crawl does."""
    from contextlib import nullcontext
    return timer.stage(name) if timer is not None else nullcontext()


def build_article_atom(article: dict, *, url: str, atom_id: str, blog_url: str, author: str,
                       author_name: str | None = None, handle: str | None = None,
                       lastmod: str = "", feed_dates: dict[str, str] | None = None,
                       recover_feed_date: bool = False, img_cache: dict | None = None,
                       seen: dict[str, str] | None = None,
                       entry_mode: str = "oracle-footprint", timer=None,
                       log_tag: str = "footprint") -> dict:
    """A fetched article → a write-ready atom, without touching `conn`. Returns an outcome dict:

      • `{"outcome": "ok", atom, md, atom_id, raw_hash}` — hand it to `store_atom`.
      • `{"outcome": "unchanged"}`     — the snapshot hash matches `seen`. Skip; do not rewrite.

    Never returns None: `run_concurrent` drops a None without calling the consumer.

    `recover_feed_date` (on for a hand-dump) fetches the origin's feed here when htmldate found no
    date and no feed map was passed in.
    """
    from pipeline.ingestion.sources.blog import _article_to_markdown

    seen = {} if seen is None else seen
    feed_dates = dict(feed_dates or {})

    with _stage(timer, "date_resolve"):
        htmldate = article.get("date") or ""
        if not htmldate and recover_feed_date and not feed_dates:
            feed_dates = _feed_date_map(blog_url)      # rung-2 recovery, only once rung 1 missed
        date, precision = _resolve_post_date(htmldate, url, lastmod, feed_dates)
    article["date"] = date
    article["when_precision"] = precision

    md = _article_to_markdown(article, author=author, author_name=author_name or author)

    # Chart-read inline images so charts become searchable text (only on a page someone chose —
    # `vision._PAGE_IMAGES_READ`); `base_url=url` resolves a self-hosted blog's relative image refs.
    with _stage(timer, "vlm"):      # per-post (one post fans out to several reads)
        md, _ = vision.enrich_markdown_images(md, img_cache if img_cache is not None else {},
                                              entry_mode=entry_mode,
                                              context=article.get("title") or "", base_url=url)
    # Safe from a pool thread: a dict `.get` is atomic, and `seen` is the real ledger, which gives
    # idempotency.
    raw_hash = changed_hash(atom_id, md, seen)
    if raw_hash is None:                 # unchanged — honor the seam
        return {"outcome": "unchanged"}

    meta = derive.derive_blog(article, blog_url=blog_url, handle=handle, author_name=author_name)
    atom = {
        "atom_id": atom_id,
        "source_type": "blog",
        "what_kind": "opinion",
        "who_id": meta["who_id"],
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": url,
        "raw_hash": raw_hash,
        "description": meta["description"],
        # BASIS_ASSUMED: a truncated "read more" preview returns 200 with real prose and no
        # marker, indistinguishable from a genuinely short post, so `complete` here means "nothing
        # indicated otherwise" — weaker than Substack's stated or X's observed. Don't collapse the three.
        "payload": {"lastmod": lastmod, **body_fields(BODY_COMPLETE, BASIS_ASSUMED)},
        "entry_mode": entry_mode,                 # crawl: oracle-footprint; hand-dump: user-saved
    }
    # who_id stays the origin author for every atom, including a hub-harvested one.
    return {"outcome": "ok", "atom_id": atom_id, "raw_hash": raw_hash,
            "atom": atom, "md": md}


def prepare_article(url: str, *, entry_mode: str, seen: dict, img_cache: dict | None = None,
                    log_tag: str = "hopper", isolate: bool = True) -> dict:
    """The network half of `article_atom_from_url` — fetch, judge, build — touching no `conn`, so a
    caller can run it on a pool thread. Returns `{"status", "atom_id", "url", "res"}`:

      • "present"  — `atom_id` is in `seen`; nothing was fetched (or the redirect landed on one).
      • "ready"    — `res` is `build_article_atom`'s write-ready result; hand it to the write half.
      • "unchanged" / "blocked" / "failed" — nothing to write; see the caller.

    A redirect re-keys the atom to the page it landed on whenever that canonicalizes differently,
    so a shortener or `/p/123` → `/the-real-slug` mints the same id a direct link would.

    Never raises unless `isolate` is off: every failure is a status."""
    from pipeline.platforms.web.article import fetch_article
    from pipeline.ingestion.utils import log

    u = (url or "").strip()
    atom_id = canon_post_url(u)
    if atom_id in seen:            # policy B: present → skip before the paid fetch + gate
        return {"status": "present", "atom_id": atom_id, "url": u, "res": None}
    try:
        article = fetch_article(u)
        verdict = _classify_article(article)
        if verdict != FETCH_OK:
            # Never store a bot-check shell or a thin stub. Reported apart so "the host blocked
            # us" reads differently from "this page has no article on it".
            log(f"[{log_tag}] article not stored ({verdict}): {u}")
            return {"status": "blocked" if verdict == FETCH_UNDETERMINED else "failed",
                    "atom_id": None, "url": u, "res": None}
        landed = (article.get("final_url") or "").strip()
        if landed and canon_post_url(landed) != atom_id:
            u, atom_id = landed, canon_post_url(landed)
            article["url"] = landed
            if atom_id in seen:
                return {"status": "present", "atom_id": atom_id, "url": u, "res": None}

        origin = f"{urlparse(u).scheme}://{urlparse(u).netloc}"
        # trafilatura's byline when found, else the host. Per-atom only, never written onto the
        # entity — see `article_atom_from_url`.
        byline = (article.get("author") or "").strip()
        res = build_article_atom(
            article, url=u, atom_id=atom_id, blog_url=origin,
            author=byline or origin, author_name=byline or None,
            feed_dates=None, recover_feed_date=True, img_cache=img_cache, seen=seen,
            entry_mode=entry_mode, log_tag=log_tag)
    except Exception as e:
        if not isolate:
            raise
        log(f"[{log_tag}] article ingest failed for {u}: {e}")
        return {"status": "failed", "atom_id": None, "url": u, "res": None}
    status = {"ok": "ready"}.get(res["outcome"], res["outcome"])
    return {"status": status, "atom_id": atom_id if status in ("ready", "unchanged") else None,
            "url": u, "res": res if status == "ready" else None}


def article_atom_from_url(conn: sqlite3.Connection, url: str, *,
                          entry_mode: str = "user-saved", seen: dict | None = None,
                          img_cache: dict | None = None,
                          on_written=None, prepared: dict | None = None,
                          store=store_atom) -> tuple[str, str | None]:
    """Fetch one article by URL → a blog atom; the adapter Hopper's bare-link case routes to. Returns `(status, atom_id)` where status is
    "present" | "saved" | "blocked" | "failed"; `atom_id` is real for the first two.

    `entry_mode="oracle-footprint"` is refused: attribution is by URL host, not verified
    authorship. An unknown host gets a bare entity (no name, kind or links) so it can never merge
    into a tracked person. `prepared` is `prepare_article`'s result when a caller already ran the
    network half; absent, it runs here. A failed fetch, challenge page or thin stub writes nothing.
    `store` is the write (`store_atom`); a caller that lands the atom with its own state later
    passes one that holds it.
    """
    if entry_mode == "oracle-footprint":
        raise ValueError("article_atom_from_url may not stamp 'oracle-footprint' — see its "
                         "docstring; a URL-host attribution is not an authorship claim.")

    u = (url or "").strip()
    if not u:
        return "failed", None
    if seen is None:
        seen = schema.load_hashes(conn, "blog")
    if prepared is None:
        prepared = prepare_article(u, entry_mode=entry_mode, seen=seen, img_cache=img_cache)

    status, atom_id = prepared["status"], prepared["atom_id"]
    # Prepared off this thread, so the ledger may have moved since: two urls that land on one post
    # (a shortener and the post itself, linked in two tweets) are prepared side by side and both
    # come back "ready". The second must not write the same atom again.
    if status == "ready" and atom_id in seen:
        status = "present"
    if status in ("present", "unchanged"):
        promote_atom(conn, atom_id, entry_mode)     # human-initiated presence hit → attestation
        return "present", atom_id
    if status != "ready":
        return status, None
    res = prepared["res"]
    # `upsert_entity` COALESCEs every field, so an existing tracked blog keeps its name/kind/
    # identity-links untouched and the atom simply joins it.
    schema.upsert_entity(conn, res["atom"]["who_id"])
    store(conn, res["atom"], res["md"], on_written=on_written and partial(on_written, atom_id))
    seen[atom_id] = res["raw_hash"]
    return "saved", atom_id
