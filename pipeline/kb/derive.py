"""
pipeline/kb/derive.py — LEAN derivation of an atom's routing metadata.

The plan's deliberate scope cut: `about_entities`/`description`/`who_id` come from the
SOURCE's own structural metadata (GitHub topics, tweet hashtags/mentions, author fields)
via simple slug normalization — NO LLM, NO NER, NO topic-clustering. Those richer
derivations are already-built, separately-scoped features; wiring them in is a documented
follow-on. Everything here is mechanical and deterministic, which keeps the `description`
SAFE to surface in `aggregate` without violating the trust invariant (it asserts nothing
the source didn't structurally state).
"""

from __future__ import annotations

import re

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """`"AI Agents"` / `"ai_agents"` / `"#AIAgents"` → `"ai-agents"`. Lowercased,
    non-alphanumerics collapsed to single hyphens, trimmed. Empty → ""."""
    return _SLUG_STRIP.sub("-", (text or "").lower()).strip("-")


def _slugs(items) -> list[str]:
    """Unique, order-preserving, empties dropped."""
    seen, out = set(), []
    for it in items or []:
        s = slugify(it if isinstance(it, str) else str(it))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── X bookmarks → opinion-atom metadata ─────────────────────────────────────────

def derive_x(norm: dict) -> dict:
    """Structural fields off a normalized bookmark (x_graphql._normalize output).

    `who_id` is the AUTHOR entity (`x:user:{id}`) — a bookmark records who SAID it, not
    who saved it (that's `entry_mode="user-saved"`). Hashtags are AUTHOR-DECLARED, so they
    land in `source_tags`. Entities = @-mentions, slugged. Description is mechanical:
    handle, name, a text prefix, day.
    """
    author = norm.get("author") or {}
    username = author.get("userName") or "unknown"
    name = author.get("name") or username
    uid = author.get("id") or username
    who_id = f"x:user:{uid}"
    site = author.get("site") or ""   # bio outbound URL → Stage-3 cross-platform seed

    entities = norm.get("entities") or {}
    hashtags = [h.get("text", "") for h in entities.get("hashtags", [])]
    mentions = [m.get("screen_name", "") for m in entities.get("user_mentions", [])]

    text = (norm.get("text") or "").replace("\n", " ").strip()
    date = _day(norm.get("createdAt", ""))
    desc = f"@{username} ({name}) · {text[:120]} · {date}".strip(" ·")

    return {
        "who_id": who_id,
        "who_name": name,
        "who_handle": author.get("userName") or None,  # the @handle — footprint discovery is handle-rooted
        "who_site": site,
        "when_ts": date,
        "when_precision": "day",
        "source_tags": _slugs(hashtags),
        "about_entities": [f"x:@{slugify(m)}" for m in mentions if m],
        "description": desc,
    }


def _day(created_at: str) -> str:
    """A twitter `createdAt` → `YYYY-MM-DD` (empty on parse failure). Reuses the
    ingester's own date parser so bookmark + profile dates agree."""
    try:
        from pipeline.ingestion.x_render import _parse_twitter_date
        dt = _parse_twitter_date(created_at or "")
        return dt.strftime("%Y-%m-%d") if dt else ""
    except Exception:
        return ""


# ── Substack pub/author identity + saved-post metadata ──────────────────────────

_SUBSTACK_SUBDOMAIN = re.compile(r"https?://([^./]+)\.substack\.com", re.I)
_URL_HOST = re.compile(r"https?://([^/]+)", re.I)


def substack_entity_id(handle: str | None = None, publication_url: str | None = None) -> str:
    """The Substack join key: `substack:{author_handle}`, falling back to
    `substack:{subdomain}` from a publication URL when the handle is absent (the
    subscriber-lists endpoint drops the handle, so subs land on the subdomain id while
    saved-posts land on the handle id). The residual handle-vs-subdomain split for a pub
    the user BOTH saved-from AND subscribes-to is resolved in Stage-3 via the publication
    URL both sides store in `identity_links` — do NOT try to unify it here."""
    h = (handle or "").strip().lstrip("@").lower()   # handles are case-insensitive → canonicalize
    if h:
        return f"substack:{h}"
    url = publication_url or ""
    m = _SUBSTACK_SUBDOMAIN.match(url)
    if m:
        return f"substack:{m.group(1).lower()}"
    m = _URL_HOST.match(url)              # custom domain → its host is a stable id
    return f"substack:{m.group(1).lower()}" if m else "substack:unknown"


def _iso_day(iso: str) -> str:
    """An ISO timestamp (`2026-06-26T11:45:17Z`) → `YYYY-MM-DD` (best-effort, "" on miss)."""
    if not iso:
        return ""
    try:
        from datetime import datetime
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return iso[:10] if len(iso) >= 10 else ""


def derive_substack(rec: dict) -> dict:
    """Structural fields off a saved-post record (`ingest_substack._saved_item_to_record`).

    `who_id` is the post's AUTHOR/publication (`substack:{handle|subdomain}`) — a saved post
    records who WROTE it, not who saved it (that's `entry_mode="user-saved"`). Description
    is mechanical."""
    handle = rec.get("author_handle") or ""
    pub_url = rec.get("publication_url") or ""
    who_id = substack_entity_id(handle, pub_url)
    name = (rec.get("author_name") or rec.get("publication_name")
            or (f"@{handle}" if handle else "substack"))
    date = _iso_day(rec.get("post_date", ""))
    title = (rec.get("title") or "Untitled").replace("\n", " ").strip()
    pub = (rec.get("publication_name") or "").strip()
    desc = f"{name} · {title}" + (f" · {pub}" if pub else "") + (f" · {date}" if date else "")

    return {
        "who_id": who_id,
        "who_name": name,
        "who_site": pub_url,
        "when_ts": date,
        "when_precision": "day" if date else "unknown",   # never claim day-precision on an empty date
        "about_entities": [],
        "description": desc.strip(" ·"),
    }


# ── GitHub repos → artifact-atom metadata ───────────────────────────────────────

def derive_github(repo: dict) -> dict:
    """Structural fields off a GitHub repo dict (ingest_github._fetch_repos output).

    `who_id` = the repo OWNER (`github:{login}`) — authoritative from the API, not the
    crawl handle (an org repo surfaced via a user's membership still belongs to the org).
    Repo topics are AUTHOR-DECLARED, so they land in `source_tags`. Entities = the primary
    language. `when_precision="push"` flags that `when_ts` is LAST-PUSH, not a publish date
    (don't read it as "new work").
    """
    name = repo.get("name", "")
    owner = (repo.get("owner") or {}).get("login", "")
    language = repo.get("language") or ""
    stars = repo.get("stargazers_count", 0)
    desc = (repo.get("description") or "").replace("\n", " ").strip()
    topics = repo.get("topics") or []
    pushed = (repo.get("pushed_at") or "")[:10]

    description = f"{owner}/{name} · {language or '—'} · {desc[:80]} · ★{stars}".strip(" ·")

    entities = _slugs([f"lang:{language}"]) if language else []
    return {
        "who_id": f"github:{owner}",
        "who_name": owner,
        "when_ts": pushed,
        "when_precision": "push",
        "source_tags": _slugs(topics),
        "about_entities": entities,
        "description": description,
    }


# ── Blog footprint identity + per-post metadata ─────────────────────────────────

def blog_entity_id(blog_url: str | None) -> str:
    """The blog join key: `blog:{canonical_host}` (via `url_canon.canonical_identity`, which
    returns the bare host for a personal site — `karpathy.github.io`, `simonwillison.net`). The
    host is the person's blog trust-unit, and it is UNIQUE per site, so resolve can treat the
    stored home as `self` without false-merging two people. Distinct from the PER-POST atom id
    (`_canon_post_url` in ingest_blog), which keeps the path so posts don't collapse to the host.
    Empty/garbage URL → `blog:unknown` (a stable, never-merging singleton)."""
    from pipeline.ingestion.url_canon import canonical_identity
    host = canonical_identity(blog_url or "")
    return f"blog:{host}" if host else "blog:unknown"


def org_entity_id(org_url: str | None) -> str:
    """The org join key: `org:{canonical_host}` (via `url_canon.canonical_identity`). Mirrors
    `blog_entity_id` but tags a MULTI-author / company site the footprint gate SKIPPED — recorded
    as an affiliation fact-node — this `org:` id PREFIX is the marker, there being no type column on
    `entities` (one was deleted 2026-08-23) — NEVER a source of opinion atoms. Unique per host,
    so it never false-merges two orgs. Empty/garbage URL → `org:unknown`."""
    from pipeline.ingestion.url_canon import canonical_identity
    host = canonical_identity(org_url or "")
    return f"org:{host}" if host else "org:unknown"


def derive_blog(article: dict, *, blog_url: str, handle: str | None = None,
                author_name: str | None = None) -> dict:
    """Structural fields off a fetched blog post (`sources/blog._fetch_article` output:
    `{url, title, date, content}`), mirroring `derive_substack`.

    `who_id` is the blog HOME (`blog:{host}`) — the same for every post of one blog, so the whole
    footprint attributes to one Oracle. `when_ts` is the post's date (day precision).
    `description` is mechanical (name · title · date)."""
    from pipeline.ingestion.url_canon import canonical_identity
    who_id = blog_entity_id(blog_url)
    name = (author_name or (f"@{handle.lstrip('@')}" if handle else "")
            or canonical_identity(blog_url or "") or "blog")
    date = (article.get("date") or "")[:10]
    title = (article.get("title") or "Untitled").replace("\n", " ").strip()
    desc = f"{name} · {title}" + (f" · {date}" if date else "")

    return {
        "who_id": who_id,
        "who_name": name,
        "who_site": blog_url,
        "when_ts": date,
        # Honor the date cascade's precision when it set one (e.g. `approx` for a Wayback upper
        # bound); else infer — never claim day-precision on an empty date.
        "when_precision": article.get("when_precision") or ("day" if date else "unknown"),
        "about_entities": [],
        "description": desc.strip(" ·"),
    }


# ── Paper (arXiv / Semantic Scholar) → artifact-atom metadata ────────────────────

def derive_paper(paper: dict) -> dict:
    """Structural fields off a normalized Paper (the S2 metadata SHAPE:
    `title/abstract/authors/externalIds/venue/year/publicationDate/…`), mirroring
    `derive_blog`/`derive_substack`.

    `who_id` is the paper's own author — NOT the Oracle. A footprint-linked paper is one the
    Oracle *pointed at*, usually not one they wrote; the Oracle's relationship is a caller-supplied
    vouch edge (`oracle → references → paper`), never an authorship claim. `who_id` =
    `scholar:{first_author_id}` when the first author carries a Semantic Scholar author id, else
    `paper-authors:{canonical_id}` (a stable per-paper placeholder for the rare id-less case — a
    raw hosted PDF with no metadata). `when_precision` is HONEST: `"day"` from a `publicationDate`,
    else `"year"` from the `year` (a paper with only a year is not dated to Jan 1 for real).
    `about_entities` is `[]` on purpose — a structural guess here would be wrong. `description`
    is mechanical (author · title · venue · year), so it stays SAFE to surface without asserting
    anything the metadata didn't state."""
    from .ingest_papers import _canonical_paper_id   # shared canonical-id rule (dedup parity)

    authors = paper.get("authors") or []
    first = authors[0] if authors else {}
    first_name = (first.get("name") or "").strip()
    first_id = first.get("authorId")

    pid = _canonical_paper_id(paper) or "unknown"
    who_id = f"scholar:{first_id}" if first_id else f"paper-authors:{pid}"
    who_name = first_name or "paper"

    pub = (paper.get("publicationDate") or "").strip()
    year = paper.get("year")
    if pub:
        when_ts, when_precision = pub[:10], "day"
    elif year:
        when_ts, when_precision = f"{year}-01-01", "year"   # honest: year-precision, not a real day
    else:
        when_ts, when_precision = "", ""

    title = (paper.get("title") or "Untitled").replace("\n", " ").strip()
    venue = (paper.get("venue") or "").replace("\n", " ").strip()
    parts = [first_name or who_name, title, venue, str(year) if year else ""]
    description = " · ".join(p for p in parts if p)

    return {
        "who_id": who_id,
        "who_name": who_name,
        "who_site": paper.get("url") or "",
        "when_ts": when_ts,
        "when_precision": when_precision,
        "about_entities": [],
        "description": description,
    }
