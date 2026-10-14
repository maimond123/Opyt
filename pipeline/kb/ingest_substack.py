"""
pipeline/kb/ingest_substack.py — one Substack post → an opinion atom: the per-post builder the
Substack archive walk uses (`build_footprint_post`, `oracles/sources/archive/substack.py`), and
the single-post path (`substack_atom_from_url`). The key `substack:{post_id}` is shared with the
curation path, so a post both saved and authored is one atom; Policy B skips an existing atom
before the per-post fetch. Paywalled posts are skipped and counted. A post with no body, or a
failed write, writes nothing and is not marked seen.
"""

from __future__ import annotations

from pipeline.ingestion.sources.substack import _post_to_markdown
from pipeline.ingestion.utils import log
from .vision import enrich_markdown_images

from pipeline.host_access import request as host_request

import re
import sqlite3
from functools import partial

from . import derive, schema
from .ingest_common import (store_atom, BASIS_STATED, BODY_COMPLETE,
                            body_fields, promote_atom, changed_hash)
# A Substack post url on any host, so custom-domain Substacks (reached via Hopper's `kind_hint`)
# match; the host boundary is `link_router.classify_link`'s job.
_SUBSTACK_POST_RE = re.compile(r"^(https?://[^/]+)/p/([^/?#]+)", re.I)

# The Substack platform-app share form (`substack.com/.../post/p-{id}`) carries only the global
# post id, not {pub}+{slug}. It 302-redirects to the canonical `{pub}.substack.com/p/{slug}`,
# which we resolve before parsing.
_READER_URL_RE = re.compile(r"^https?://substack\.com/(?:[^/]+/)*post/p-\d+", re.I)

# The Substack app's share form (`open.substack.com/pub/{pub}/p/{slug}`, what the iOS share sheet
# copies) does name the publication and slug — it 302s to `{pub}.substack.com/p/{slug}` — so it is
# rewritten locally, no network hop.
_APP_URL_RE = re.compile(r"^https?://open\.substack\.com/pub/([^/?#]+)/p/([^/?#]+)", re.I)


def _post_rec(post: dict, *, publication_url: str, handle: str | None,
              author_name: str | None) -> dict:
    """Map one `/api/v1/archive` post dict onto the `rec` shape `derive.derive_substack`
    expects. Publication identity comes from the Oracle source, not per-post, so `who_id`
    is stable across the whole archive."""
    return {
        "id": post.get("id"),
        "title": post.get("title") or "Untitled",
        "post_date": post.get("post_date") or "",
        "author_handle": handle or "",
        "author_name": author_name or "",
        "publication_name": author_name or "",
        "publication_url": publication_url,
        "slug": post.get("slug") or "",
        "url": post.get("canonical_url") or "",
        "audience": post.get("audience") or "",
    }


def _resolve_reader_url(url: str) -> str | None:
    """Follow a Substack reader-url's redirect (HEAD) to its canonical `{pub}.substack.com/p/{slug}`;
    None unless it lands on a `/p/` post."""
    try:
        import requests
        resp = host_request(requests.head, url, allow_redirects=True, timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
        final = resp.url or ""
        return final if _SUBSTACK_POST_RE.match(final) else None
    except Exception:
        return None


def substack_atom_from_url(conn: sqlite3.Connection, url: str, *,
                           entry_mode: str = "author_referenced", seen: dict | None = None,
                           img_cache: dict | None = None, on_written=None,
                           isolate: bool = True, store=store_atom) -> str | None:
    """Fetch one public Substack post by url → an opinion atom keyed `substack:{post_id}`.

    The Oracle referenced this post, so who_id is the post's author. Accepts the canonical,
    reader-url and app-share forms. Returns the atom id whenever the atom exists after the call
    (an existing one is vouched for without re-rendering), else None; with `isolate`, never
    raises. `img_cache` absent → loaded and saved here. `store` is the write (`store_atom`); a
    caller that lands the atom with its own state later passes one that holds it.
    """
    from pipeline.ingestion.sources.substack import _post_to_markdown
    from pipeline.platforms.substack.publication import fetch_full_post, is_paywalled
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    u = (url or "").strip()
    if _READER_URL_RE.match(u):           # reader-url carries only a post id → resolve to {pub}/p/{slug}
        u = _resolve_reader_url(u) or u   # failure → unchanged → the match below fails → skip (fail-safe)
    if (app := _APP_URL_RE.match(u)):     # app share link names pub + slug → canonical, offline
        u = f"https://{app.group(1)}.substack.com/p/{app.group(2)}"
    m = _SUBSTACK_POST_RE.match(u)
    if m is None:
        return None
    base, slug = m.group(1).rstrip("/"), m.group(2)
    try:
        full = fetch_full_post(base, slug, {})       # {} = public, cookie-less (the verified path)
        if not full:
            return None
        post_id = full.get("id")
        if not post_id:
            return None
        atom_id = f"substack:{post_id}"
        if seen is None:
            seen = schema.load_hashes(conn, "substack")
        if atom_id in seen:                           # policy B — present (any path) → vouch-only, no clobber
            promote_atom(conn, atom_id, entry_mode)   # human-initiated presence hit → attestation
            return atom_id
        if is_paywalled(full):                       # public-only v1 (never store a preview as full)
            return None
        body_html = (full.get("body_html") or "").strip()
        if not body_html:
            return None

        who_id = derive.substack_entity_id(None, base)   # the post's publication (subdomain)
        author = who_id.split(":", 1)[1]
        md = _post_to_markdown(full, author=f"@{author}", author_name=author)
        own_cache = img_cache is None
        cache = load_image_cache(opyt_home()) if own_cache else img_cache
        md, _ = enrich_markdown_images(md, cache, entry_mode=entry_mode,
                                       context=full.get("title") or "")
        if own_cache:
            save_image_cache(opyt_home(), cache)

        raw_hash = changed_hash(atom_id, md, seen)
        if raw_hash is None:                           # (unreachable: not in seen) honor the seam
            return atom_id

        meta = derive.derive_substack(_post_rec(full, publication_url=base, handle=None,
                                                author_name=author))
        schema.upsert_entity(conn, who_id, name=author, identity_links=[base])
        atom = {
            "atom_id": atom_id,
            "source_type": "substack",
            "what_kind": "opinion",                   # a substack essay is opinion content
            "who_id": who_id,
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "about_entities": meta["about_entities"],
            "source_url": full.get("canonical_url") or url,
            "raw_hash": raw_hash,
            "description": meta["description"],
            # Constant `complete`/`stated` — `is_paywalled` returns before this point.
            "payload": {"word_count": full.get("wordcount") or 0,
                        **body_fields(BODY_COMPLETE, BASIS_STATED)},
            "entry_mode": entry_mode,                 # the Oracle pointed at it, didn't write it
        }
        store(conn, atom, md, on_written=on_written and partial(on_written, atom_id))
        seen[atom_id] = raw_hash
        return atom_id
    except Exception as e:                            # foreground callers receive a failed result
        if not isolate:
            raise
        log(f"[footprint] substack atom from {url} skipped (fetch/write failed): {e}")
        return None


def build_footprint_post(job: dict, *, publication_url, handle, author_name, who_id,
                         seen, img_cache, timer) -> dict:
    """The paid stages, on a pool thread. Returns an outcome for the consumer to tally rather
    than counting anything itself. Never returns None — `run_concurrent` drops a None without
    calling the consumer, which would lose the outcome entirely."""
    atom_id, post, post_full, full = (job["atom_id"], job["post"],
                                      job["post_full"], job["full"])
    author = f"@{handle}" if handle else (author_name or "substack")
    md = _post_to_markdown(post_full, author=author, author_name=author_name or author)

    # Chart-read inline images so charts become searchable text (injected as `*Image:* …` before
    # hashing, so it's part of the indexed text).
    with timer.stage("vlm"):            # per-post (one post fans out to several reads)
        md, _ = enrich_markdown_images(md, img_cache, entry_mode="oracle-footprint",
                                       context=post_full.get("title") or "")
    # Reading `seen` from a pool thread is safe: a dict `.get` is atomic.
    raw_hash = changed_hash(atom_id, md, seen)
    if raw_hash is None:                 # unchanged — honor the seam
        return {"outcome": "unchanged"}

    meta = derive.derive_substack(_post_rec(post_full, publication_url=publication_url,
                                            handle=handle, author_name=author_name))
    atom = {
        "atom_id": atom_id,
        "source_type": "substack",
        "what_kind": "opinion",
        "who_id": who_id,
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": post.get("canonical_url") or "",
        "raw_hash": raw_hash,
        "description": meta["description"],
        # Constant `complete`/`stated`: paywalled and body-less posts are skipped before an
        # atom is built, so everything reaching here holds a full, declared-public body.
        "payload": {"word_count": post.get("wordcount") or full.get("wordcount") or 0,
                    **body_fields(BODY_COMPLETE, BASIS_STATED)},
        "entry_mode": "oracle-footprint",         # not user-saved (curation) / crawled (radar)
    }

    return {"outcome": "ok", "atom_id": atom_id, "raw_hash": raw_hash,
            "atom": atom, "md": md}
