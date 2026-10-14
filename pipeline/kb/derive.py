"""
pipeline/kb/derive.py — derive an atom's routing metadata from the source's structural fields.

Mechanical and deterministic (no LLM, no NER), so `description` asserts nothing the source didn't
state and is safe to surface in `aggregate`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from pipeline.platforms.papers.arxiv import strip_arxiv_version
from pipeline.platforms.url_canon import canonical_identity

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """`"AI Agents"` / `"ai_agents"` → `"ai-agents"`."""
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
    """Structural fields off a normalized bookmark (x.records.normalize_tweet output).

    `who_id` is the author, not the saver. Hashtags are author-declared, so they land in
    `source_tags`; entities are the @-mentions.
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
    """A twitter `createdAt` → `YYYY-MM-DD` (empty on parse failure)."""
    try:
        from pipeline.ingestion.x_render import parse_twitter_date
        dt = parse_twitter_date(created_at or "")
        return dt.strftime("%Y-%m-%d") if dt else ""
    except Exception:
        return ""


# ── Substack pub/author identity + saved-post metadata ──────────────────────────

_SUBSTACK_SUBDOMAIN = re.compile(r"https?://([^./]+)\.substack\.com", re.I)
_URL_HOST = re.compile(r"https?://([^/]+)", re.I)


def substack_entity_id(handle: str | None = None, publication_url: str | None = None) -> str:
    """The Substack join key: `substack:{handle}`, else `substack:{subdomain}`, else
    `substack:{host}` for a custom domain.

    A handle-vs-subdomain split for the same publication is resolved later via `identity_links`;
    don't unify it here.
    """
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
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""       # a non-ISO value is not a date; never fabricate one from a prefix


def derive_substack(rec: dict) -> dict:
    """Structural fields off a saved-post record (`substack.account.saved_item_to_record`).

    `who_id` is the post's author/publication, not the saver."""
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

    `who_id` is the repo owner from the API, not the crawl handle. `when_precision="push"`
    flags that `when_ts` is the last push, not a publish date.
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
    """The blog join key: `blog:{canonical_host}`; `blog:unknown` for an empty or garbage URL.

    The host is unique per site, so resolve can treat it as `self` without false-merging people.
    """
    host = canonical_identity(blog_url or "")
    return f"blog:{host}" if host else "blog:unknown"


def derive_blog(article: dict, *, blog_url: str, handle: str | None = None,
                author_name: str | None = None) -> dict:
    """Structural fields off a fetched blog post (`platforms/web/article.fetch_article` output).

    `who_id` is the blog home, the same for every post, so the footprint attributes to one Oracle.
    """
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
        # Honor the date cascade's precision when set; never claim day precision on no date.
        "when_precision": article.get("when_precision") or ("day" if date else "unknown"),
        "about_entities": [],
        "description": desc.strip(" ·"),
    }


# ── Paper (arXiv / Semantic Scholar) → artifact-atom metadata ────────────────────

def canonical_paper_id(paper: dict) -> str | None:
    """The dedup key behind `atom_id = paper:{id}`. Prefer the explicit `paperId`/`id` a source
    supplied (paper_from_url stamps a deterministic one); else derive from external ids (arXiv id,
    version-stripped > DOI). None → the paper cannot be identified (caller returns None)."""
    pid = paper.get("paperId") or paper.get("id")
    if pid:
        return str(pid)
    ext = paper.get("externalIds") or {}
    arxiv = ext.get("ArXiv")
    if arxiv:
        return f"arXiv:{strip_arxiv_version(arxiv)}"
    doi = ext.get("DOI")
    if doi:
        return f"DOI:{str(doi).rstrip('.').lower()}"
    return None


def derive_paper(paper: dict) -> dict:
    """Structural fields off a normalized Paper (Semantic Scholar metadata shape).

    `who_id` is the paper's first author, not the Oracle: `scholar:{author_id}`, else the
    placeholder `paper-authors:{canonical_id}`. The Oracle's link is a separate vouch edge.
    `when_precision` is `"day"` from `publicationDate`, else `"year"`.
    """
    authors = paper.get("authors") or []
    first = authors[0] if authors else {}
    first_name = (first.get("name") or "").strip()
    first_id = first.get("authorId")

    pid = canonical_paper_id(paper) or "unknown"
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


def _profile_dict(value) -> dict:
    """A stored `entities.profile` → a dict; null, empty, malformed or non-dict → {}."""
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value) if value else {}
    except (ValueError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def best_name(members: list[dict]) -> tuple[str | None, str | None]:
    """(display_name, handle) for a cluster, preferring the X member. The one name-picking rule
    for the whole KB."""
    x_members = [m for m in members if (m["entity_id"] or "").startswith("x:")]
    ordered = x_members + [m for m in members if m not in x_members]
    name = next((m["name"] for m in ordered if m["name"]), None)
    handle = next((_profile_dict(m["profile"]).get("handle") for m in ordered
                   if _profile_dict(m["profile"]).get("handle")), None)
    return name, handle
