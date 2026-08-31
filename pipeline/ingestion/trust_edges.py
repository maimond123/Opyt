"""
pipeline/ingestion/trust_edges.py

Edge extractors: turn already-fetched probe data (X bio URLs, a GitHub bio/blog,
a landing-page's HTML) into canonical trust-graph `Edge`s for `trust_graph.propagate()`.

Extraction is bounded by a `relevant` set — only edges whose target is one of this
person's own candidate sources are emitted, so the graph never grows past what
discovery already found. Parsers take raw strings and return edges (fixture-testable,
no network); only `fetch_landing_edges` touches the network, wrapping `edges_from_html`.
"""

from __future__ import annotations

import re

from pipeline.ingestion.source_classify import ProfileLink, classify_source
from pipeline.ingestion.trust_types import Edge
from pipeline.ingestion.url_canon import canonical_identity

# Typed profile links worth surfacing from a trusted hub — a person's own accounts,
# not things they cite. Excludes linkedin: never an atom source, never fetched, so it
# would be a dead-end leaf.
PROFILE_SCOPE = frozenset(
    {"github", "scholar", "orcid", "substack", "blog", "x", "youtube"}
)

# Full URLs and @handles inside free text (a GitHub bio, a tweet).
_URL_RE = re.compile(r"""https?://[^\s)<>"'\]]+""")
_HANDLE_RE = re.compile(r"(?<![\w/@])@([A-Za-z0-9_]{1,15})\b")


def _handle_id(handle: str) -> str:
    return canonical_identity(f"x.com/{handle}")


def _emit(source_id: str, targets, relevant: set[str], via: str, found_by: str) -> list[Edge]:
    """Build edges source→t for each distinct t in `targets` that is relevant."""
    allow = set(relevant) | {source_id}
    out, seen = [], set()
    for t in targets:
        if not t or t == source_id or t in seen or t not in allow:
            continue
        seen.add(t)
        out.append(Edge(source_id, t, via=via, found_by=found_by))
    return out


def targets_from_text(text: str) -> set[str]:
    """Canonical ids referenced in free text: full URLs + @handles → x.com/handle."""
    ids: set[str] = set()
    for m in _URL_RE.findall(text or ""):
        cid = canonical_identity(m.rstrip(".,);]"))
        if cid:
            ids.add(cid)
    for h in _HANDLE_RE.findall(text or ""):
        ids.add(_handle_id(h))
    return ids


def edges_from_github(login: str, bio: str, blog: str, relevant: set[str]) -> list[Edge]:
    """github.com/{login} → {handles/URLs mentioned in the bio, blog field URL}."""
    src = canonical_identity(f"github.com/{login}")
    targets = targets_from_text(bio)
    blog_id = canonical_identity(blog)
    if blog_id:
        targets.add(blog_id)
    return _emit(src, sorted(targets), relevant, via="github_bio", found_by="github")


def edges_from_html(source_id: str, html: str, relevant: set[str]) -> list[Edge]:
    """source_id → each sibling platform it links via <a href> on its landing page.
    Bounded to `relevant`, so only links back to the person's own discovered sources
    count — enables bidirectional + corroboration rules without trawling the web."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    targets: set[str] = set()
    for a in soup.find_all("a", href=True):
        cid = canonical_identity(a["href"])
        if cid:
            targets.add(cid)
    return _emit(source_id, sorted(targets), relevant, via="html_link", found_by="html")


def fetch_landing_edges(source_id: str, url: str, relevant: set[str], timeout: int = 12) -> list[Edge]:
    """Fetch `url` once and extract its outbound sibling edges. Network-failure → []."""
    return edges_from_html(source_id, _fetch_page(url, timeout) or "", relevant)


# ── Unbounded profile extraction — GROW the candidate set from a trusted hub ────
# edges_from_html is bounded to `relevant` and can only corroborate nodes discovery
# already found. This extractor has no such bound, so a trusted blog/Substack can
# surface the Oracle's OTHER profiles; classify_source keeps it safe by only letting
# typed profile links in PROFILE_SCOPE through.


def _fetch_page(url: str, timeout: int = 12) -> str | None:
    """One HTTP GET → page text, or None on any non-200 / RequestException (fail-safe)."""
    import requests

    try:
        resp = requests.get(
            url, timeout=timeout, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"},
        )
        if resp.status_code != 200:
            return None
        return resp.text
    except requests.RequestException:
        return None


def profile_links_from_html(html: str) -> list[ProfileLink]:
    """`<a href>` on a page → typed PROFILE links, unbounded and de-duped. Unlike
    `edges_from_html`, not limited to a `relevant` set — it exists to grow the
    candidate set with the Oracle's own accounts, filtered to `is_profile` links
    in `PROFILE_SCOPE` so repos, papers, and unrelated sites never surface."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    out: list[ProfileLink] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        pl = classify_source(a["href"])
        if pl is None or not pl.is_profile or pl.type not in PROFILE_SCOPE:
            continue
        cid = canonical_identity(pl.url)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(pl)
    return out


def fetch_profile_links(url: str, timeout: int = 12) -> list[ProfileLink]:
    """Fetch `url` once and return its typed profile links. [] on any failure."""
    html = _fetch_page(url, timeout)
    return profile_links_from_html(html) if html is not None else []


def substack_twitter_handle(html: str) -> str | None:
    """The publication's X handle from Substack's server-rendered `window._preloads`.
    Substack renders identity/social links via JavaScript, so `<a href>` scraping
    misses them; `pub.twitter_screen_name` sits in the `_preloads` JSON instead.
    Reuses `ingest_substack.own_user_id`'s `raw_decode` trick; tolerant of the
    `pub`/`publication` key and both the `JSON.parse("…")` and raw-object shapes.
    None on any miss."""
    if not html:
        return None
    i = html.find("_preloads")
    if i < 0:
        return None

    import json

    data = None
    try:
        jp = html.find("JSON.parse(", i)
        if jp != -1 and jp - i < 400:
            # window._preloads = JSON.parse("…escaped json…")
            q = html.find('"', jp)
            if q != -1:
                inner = json.JSONDecoder().raw_decode(html, q)[0]  # decode the JSON string literal
                if isinstance(inner, str):
                    data = json.loads(inner)                       # then parse the JSON it carried
        if data is None:
            # window._preloads = {…}  (raw object literal)
            brace = html.find("{", i)
            if brace != -1:
                data = json.JSONDecoder().raw_decode(html, brace)[0]
    except (ValueError, json.JSONDecodeError):
        data = None

    if isinstance(data, dict):
        for key in ("pub", "publication"):
            pub = data.get(key)
            if isinstance(pub, dict):
                h = pub.get("twitter_screen_name")
                if isinstance(h, str) and h.strip():
                    return h.lstrip("@").strip()

    # Robust fallback: pull the field verbatim, scoped to the _preloads region, so
    # a shape drift in the JSON wrapping still recovers the handle.
    m = re.search(r'"twitter_screen_name"\s*:\s*"([^"]+)"', html[i : i + 300_000])
    if m:
        return m.group(1).lstrip("@").strip() or None
    return None


def fetch_substack_twitter_handle(url: str, timeout: int = 12) -> str | None:
    """Fetch a Substack page and pull its X handle from `_preloads`. None on failure."""
    html = _fetch_page(url, timeout)
    return substack_twitter_handle(html) if html is not None else None
