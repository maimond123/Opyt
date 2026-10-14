"""
pipeline/oracles/discovery/post_links.py — the sites an Oracle links from their own X posts,
counted. Discovery's second source of candidates; the first is the root profile.

Read from the stored raw post, not from `referenced_by`. A link becomes an atom only when the
references worker could capture it, so a homepage or product link (`executor.sh`) leaves no
`referenced_by` trace, while the raw post keeps every link: the `## Links` card and the expanded
URLs in the text.

A quoted post is rendered first (`x_render._render_quoted_tweet`): a `**Quoting** [@who](…)`
header, its text as `>` lines, then its own `## Links` and `## Media`. Those links belong to the
quoted author unless the Oracle quoted themselves, the same rule `_referenced_links` applies with
`via_quote`. An X-Article inside a quote is not recognized as part of the quote; its links count
as the Oracle's. That case is rare and only adds a candidate the back-link check then judges.

`links_in_post`, `candidate_key` and `candidates` are pure; `own_post_candidates` reads the store.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from pipeline.kb import schema
from pipeline.platforms.source_classify import classify_source
from pipeline.platforms.url_canon import canonical_identity

_URL_RE = re.compile(r"""https?://[^\s)<>\]"'`*]+""")
_QUOTING_RE = re.compile(r"^\*\*Quoting\*\* \[@([^\]]+)\]")
_AUTHOR_RE = re.compile(r'^author:\s*"?@([^"\s]+)"?\s*$', re.M)

# A rendered media line (`x_render._render_media`): the address of an image or video, not a link.
_MEDIA_LINE_PREFIXES = ("![", "[![", "[Video](")
# What a rendered quote is made of after its header (`_render_quoted_tweet`): its text, its link
# card or bare links, its media. The first line that is none of these is the Oracle's own text.
_QUOTE_LINE_PREFIXES = (">", "## Links", "## Media", "- [", "*Image:*", *_MEDIA_LINE_PREFIXES)

# Names that are not on the public web (RFC 2606 / 6761 reserved names): a link to a dev server
# or a placeholder in a code sample. Structure, not a list of sites: any other host stays a
# candidate and the back-link check judges it. Platforms that cannot be a source at all (YouTube,
# LinkedIn, podcasts) are refused by `canonical_identity`, X and paper links by `classify_source`.
_NOT_PUBLIC = ("localhost", "local", "internal", "invalid", "test", "example",
               "example.com", "example.org", "example.net")
_IP_HOST_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$")
# A public host name: dotted labels, a letter in its last one. Rejects what a code sample or a
# truncated link leaves behind (`127`, `github`, `${server.port}`), measured on Rhys's posts.
_HOST_RE = re.compile(r"^([a-z0-9-]+\.)+[a-z][a-z0-9-]*(:\d+)?$")
# Substack's share links name the publication in the path: open.substack.com/pub/{sub}/p/{slug}.
_SUBSTACK_SHARE_RE = re.compile(r"^https?://open\.substack\.com/pub/([A-Za-z0-9-]+)", re.I)


@dataclass(frozen=True)
class Candidate:
    key: str          # canonical identity: `aleximas.substack.com`, `github.com/rhys`, `rhys.dev`
    url: str          # the page to check: the profile, or the site's home
    post_count: int   # distinct own posts that link it
    example: str      # one link as it was posted


def _strip_frontmatter(md: str) -> tuple[str, str]:
    """`(frontmatter, body)`; no frontmatter → `("", md)`."""
    if md.startswith("---\n"):
        end = md.find("\n---\n", 4)
        if end != -1:
            return md[4:end], md[end + 5:]
    return "", md


def links_in_post(md: str) -> list[str]:
    """Every link in one rendered X post that is the Oracle's own, in order, without repeats.

    A quoted post's links are left out unless the quoted author is the post's author."""
    front, body = _strip_frontmatter(md or "")
    m = _AUTHOR_RE.search(front)
    author = (m.group(1) if m else "").lower()
    out: list[str] = []
    in_quote = False
    for line in body.splitlines():
        q = _QUOTING_RE.match(line)
        if q:
            in_quote = q.group(1).lower() != author
            continue
        stripped = line.strip()
        if in_quote:
            if not stripped or stripped.startswith(_QUOTE_LINE_PREFIXES):
                continue
            in_quote = False
        if stripped.startswith(_MEDIA_LINE_PREFIXES):
            continue
        for url in _URL_RE.findall(line):
            url = url.rstrip(".,;:!?")
            if url not in out:
                out.append(url)
    return out


def candidate_key(url: str) -> str | None:
    """The source a link points at, as its canonical identity, or None when it cannot be one.

    | link                                   | key                         |
    |----------------------------------------|-----------------------------|
    | an X post or profile, a paper          | None                        |
    | YouTube, LinkedIn, podcasts, unusable  | None (`canonical_identity`) |
    | localhost, `.local`, `example.com`, an IP, no dotted host name (not public) | None |
    | a Substack post or home, or a share link (`open.substack.com/pub/{sub}/…`) | `{sub}.substack.com` |
    | any other `open.substack.com` link     | None                        |
    | a GitHub repo or profile               | `github.com/{owner}`        |
    | substack.com, a GitHub feature page    | None (no owner)             |
    | anything else                          | its canonical identity (a personal site's host, `medium.com/@x`, an ORCID) |
    """
    share = _SUBSTACK_SHARE_RE.match(url)
    if share:
        return f"{share.group(1).lower()}.substack.com"
    key = canonical_identity(url)
    if not key or key == "open.substack.com":     # unusable, or a share link naming no publication
        return None
    host = key.split("/", 1)[0]
    if (not _HOST_RE.match(host) or _IP_HOST_RE.match(host)
            or any(host == h or host.endswith("." + h) for h in _NOT_PUBLIC)):
        return None
    pl = classify_source(url)
    if pl is None or pl.type in ("x", "paper"):
        return None
    if pl.type in ("github", "substack", "gitlab") and not pl.handle:
        return None
    return key


def candidates(posts: list[list[str]]) -> list[Candidate]:
    """Each post's links → the candidates they name, most-linked first, then by key.

    `post_count` counts posts, not links: a post that links the same site three times counts
    once."""
    counts: Counter[str] = Counter()
    example: dict[str, str] = {}
    for links in posts:
        seen: set[str] = set()
        for url in links:
            key = candidate_key(url)
            if key is None or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            example.setdefault(key, url)
    return [Candidate(key, f"https://{key}", n, example[key])
            for key, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def own_post_candidates(conn: sqlite3.Connection, canonical_id: str) -> list[Candidate]:
    """The candidates an Oracle's own X posts link, from the store. A store without the tables, an
    Oracle with no X, or a post with no stored body contributes nothing."""
    head = schema.current_canonical(conn, canonical_id)
    try:
        who = [r[0] for r in conn.execute(
            "SELECT entity_id FROM entities WHERE COALESCE(canonical_id, entity_id)=? "
            "AND entity_id LIKE 'x:user:%'", (head,))]
        if not who:
            return []
        marks = ", ".join("?" for _ in who)
        rows = conn.execute(
            f"SELECT atom_id FROM atoms WHERE source_type='x' "
            f"AND entry_mode='oracle-footprint' AND who_id IN ({marks})", who).fetchall()
    except sqlite3.OperationalError:
        return []
    posts = []
    for (atom_id,) in rows:
        md = schema.read_body(conn, atom_id)
        if md:
            posts.append(links_in_post(md))
    return candidates(posts)
