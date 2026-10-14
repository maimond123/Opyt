"""
pipeline/oracles/sources/archive/post_pages.py — which of a site's pages are its posts, from what
the site says about itself. Decisions over plain values; the blog walk (`blog.py`) gathers them.

A blog walk lists a site's pages from its sitemap and hub pages, and those lists hold archives,
topic, tag and company pages, team bios and sign-in pages besides posts: 1,010 of Stratechery's
3,966 stored pages were index pages. No site gets a rule of its own, and no model reads a page.
Three signals, in order:

1. A CMS sitemap files a page under a non-post type. WordPress core, Yoast and Ghost name their
   child sitemaps by type (`wp-sitemap-taxonomies-companies-1.xml`, `page-sitemap.xml`,
   `sitemap-pages.xml`), and the walk does not read those children.
2. A site with a feed lists its posts there, and only its posts. A page whose address has the shape
   of the feed's posts is a post; one that does not is kept only if it labels itself an article
   and states its publish time (a post outside the feed's section, a second part). A label alone
   is not trusted: Substack labels its archive page an article, WordPress its team bios, Ghost
   its sign-in page.
3. A site with neither says nothing about its pages, and they are kept, as before.

Measured against 506 pages from 40 sites, labeled by readers who saw only the page: 74% of non-posts dropped, 3 of 401 posts lost.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from pipeline.platforms.url_canon import host_of

# Child-sitemap name parts that file pages which are not posts. "post", "image" and custom post
# types (podcast, announcement) are posts or hold them; custom taxonomies are caught by
# WordPress core's "taxonomies" part, not by name.
_NONPOST_TYPES = frozenset({
    "page", "pages", "tag", "tags", "post_tag", "category", "categories", "taxonomy",
    "taxonomies", "author", "authors", "user", "users", "attachment", "attachments",
})
_MONTHS = frozenset("jan feb mar apr may jun jul aug sep sept oct nov dec january february march "
                    "april june july august september october november december".split())


@dataclass(frozen=True)
class Shapes:
    """The address shapes of a site's posts, learned from its feed."""
    shapes: frozenset
    sections: frozenset


def filed_as_nonpost(sitemap_url: str) -> bool:
    """True when a child sitemap's name files its pages under a non-post type."""
    name = urlsplit(sitemap_url).path.rsplit("/", 1)[-1].lower()
    return bool(set(re.split(r"[-_.]", name)) & _NONPOST_TYPES) or "post_tag" in name


def _segments(url: str) -> list[str]:
    return [s for s in urlsplit(url).path.lower().split("/") if s]


def _token(i: int, seg: str, last: bool, sections) -> str:
    if last:            # a post's own name may be digits ("/p/2050"), but not a month ("/2015/04")
        return "NUM" if re.fullmatch(r"\d{1,2}", seg) else "SLUG"
    if re.fullmatch(r"\d{4}", seg):
        return "YEAR"
    if re.fullmatch(r"\d{1,2}", seg):
        return "NUM"
    if seg in _MONTHS:  # "/2007/apr/3/…": the feed's recent posts are all one month
        return "MONTH"
    return seg if (i, seg) in sections else "SLUG"


def _shape(parts: list[str], sections) -> tuple:
    return tuple(_token(i, s, i == len(parts) - 1, sections) for i, s in enumerate(parts))


def learn(feed_links, site_url: str) -> Shapes | None:
    """The shapes of the feed's posts on the site's own host, or None with fewer than two. A path
    part shared by at least half the posts deep enough to have it is a section name ("posts/",
    "p/"); any other part is a post's name."""
    paths = [p for p in (_segments(u) for u in feed_links if host_of(u) == host_of(site_url)) if p]
    if len(paths) < 2:
        return None
    seen = collections.Counter((i, s) for p in paths for i, s in enumerate(p[:-1]))
    sections = frozenset(
        (i, s) for (i, s), n in seen.items()
        if n >= 2 and 2 * n >= sum(1 for p in paths if len(p) > i + 1)
        and s not in _MONTHS and not s.isdigit())
    return Shapes(frozenset(_shape(p, sections) for p in paths), sections)


def fits(url: str, shapes: Shapes) -> bool:
    """True when the address has the shape of one of the feed's posts, or is a numbered part of
    one ("/posts/<name>/2/")."""
    parts = _segments(url)
    if not parts:
        return False
    if _shape(parts, shapes.sections) in shapes.shapes:
        return True
    return (len(parts) > 1 and bool(re.fullmatch(r"\d{1,2}", parts[-1]))
            and _shape(parts[:-1], shapes.sections) in shapes.shapes)


def is_post(fits_feed: bool | None, declares: dict) -> bool:
    """`fits_feed` is None when the site has no feed to learn from: nothing says the page is not
    a post. A page off the feed's shapes is a post only if it labels itself a dated article."""
    if fits_feed is not False:
        return True
    return bool(declares.get("article") and declares.get("published"))
