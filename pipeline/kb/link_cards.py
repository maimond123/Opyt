"""
pipeline/kb/link_cards.py — a name and one line for a link that is never fetched as a document:
a homepage, a product, a job or event page (`link_router.is_card_link`). "Which tools are my Oracles recommending?"
needs what a link is, not its site.
A paper that earns no record gets the same card, its line: title, then authors · year · venue,
from its metadata (`read_paper`; `kb/paper_rule.py`).

The references worker reads the card (`read`: the page's title and description, or its site's
homepage when the page says almost nothing) and keeps it here by URL. A post shows it in its
`## Links` entry in X's own card format: every render attaches the stored cards (`attach`), and
the worker fills a post already written (`fill_markdown`), so both produce the same text.
Nothing becomes an item of its own.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from urllib.parse import urlparse

from pipeline.ingestion.x_render import card_block
from pipeline.platforms.web import page_head
from pipeline.platforms.url_canon import host_of
from pipeline.timeparse import utc_iso

from . import ingest_papers, link_router

# A description shorter than this says almost nothing ("Join us."), so the homepage speaks instead.
_THIN = 25
_TITLE_MAX, _DESCRIPTION_MAX = 200, 300


@dataclass(frozen=True)
class Card:
    title: str
    description: str

    @property
    def says_something(self) -> bool:
        return bool(self.title or self.description)


def homepage(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"


def wants_homepage(url: str, page: page_head.Head) -> bool:
    """Does the page say almost nothing, so its site's homepage should speak for it? Not when the
    page is the homepage, and never on a hosted platform, whose homepage describes the platform
    rather than the job or event (deciding)."""
    thin = len(page.description) < _THIN or page.description == page.title
    is_home = (urlparse(url).path or "/") == "/"
    return thin and not is_home and not link_router.is_hosted_page(url)


def card_of(url: str, head: page_head.Head) -> Card:
    """What a post shows for the link: the page's title, or its host when it has none, and its
    description unless it only repeats the title, both trimmed. An empty card when the page said
    nothing (deciding)."""
    if not (head.title or head.description):
        return Card("", "")
    title = head.title or host_of(url)
    description = "" if head.description == head.title else head.description
    return Card(title[:_TITLE_MAX], description[:_DESCRIPTION_MAX])


def paper_line(paper: dict) -> Card:
    """A paper's line: its title, then up to three authors, its year and venue (deciding)."""
    names = [(a.get("name") or "").strip() for a in paper.get("authors") or [] if isinstance(a, dict)]
    names = [n for n in names if n]
    authors = ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")
    venue = (paper.get("venue") or "").replace("\n", " ").strip()
    year = paper.get("year") or (paper.get("publicationDate") or "")[:4]
    rest = " · ".join(str(x) for x in (authors, year, venue) if x)
    title = (paper.get("title") or "").replace("\n", " ").strip()
    return Card(title[:_TITLE_MAX], rest[:_DESCRIPTION_MAX])


def read_paper(url: str) -> Card | None:
    """A paper's line, from its metadata asked by id (`ingest_papers.paper_from_url`: Semantic
    Scholar, then arXiv, OpenAlex, Zenodo, a report's landing page), never the paper itself. A
    paper none of those names (a PubMed record that declares no DOI) is named by its page head
    instead. None when the lookup was refused now, so the link stays owed."""
    paper = ingest_papers.paper_from_url(url)
    if paper and (paper.get("title") or "").strip():
        return paper_line(paper)
    if paper and paper.get(ingest_papers.S2_VERDICT) == ingest_papers.FETCH_UNDETERMINED:
        return None
    return read(url)


def read(url: str) -> Card | None:
    """The link's card, read from its page, or from its homepage when the page says almost
    nothing. None when the page could not be read now."""
    page = page_head.read_head(url)
    if page is None:
        return None
    if wants_homepage(url, page):
        home = page_head.read_head(homepage(url))
        if home is not None and (home.title or home.description):
            page = home
    return card_of(url, page)


def load(conn: sqlite3.Connection) -> dict[str, Card]:
    """Every stored card that says something, by URL, for a render to attach."""
    return {r[0]: Card(r[1], r[2]) for r in conn.execute(
        "SELECT url, title, description FROM link_cards WHERE title != '' OR description != ''")}


def stored(conn: sqlite3.Connection, url: str) -> Card | None:
    """The card already read for this URL, even an empty one; None when it was never read."""
    row = conn.execute("SELECT title, description FROM link_cards WHERE url=?", (url,)).fetchone()
    return Card(row[0], row[1]) if row else None


def save(conn: sqlite3.Connection, url: str, card: Card) -> None:
    """Keep the card read for this URL. Joins the caller's transaction."""
    conn.execute("INSERT INTO link_cards(url, title, description, read_at) VALUES(?,?,?,?) "
                 "ON CONFLICT(url) DO UPDATE SET title=excluded.title, "
                 "description=excluded.description, read_at=excluded.read_at",
                 (url, card.title, card.description, utc_iso()))


def attach(tweet: dict, cards: dict[str, Card]) -> None:
    """Put the stored cards for a tweet's links, and its quoted post's, on each node in place,
    where `x_render` reads them (`link_cards`)."""
    for node in (tweet, tweet.get("quoted_tweet")):
        if not isinstance(node, dict):
            continue
        urls = [u.get("expanded_url") for u in (node.get("entities") or {}).get("urls") or []]
        fills = {u: {"title": cards[u].title, "description": cards[u].description}
                 for u in urls if u in cards}
        if fills:
            node["link_cards"] = fills


# A blog or Substack snapshot's closing footer (`---` then `*Blog · [Original post](…)*`).
_FOOTER = re.compile(r"\n---\n\*[^\n]*\*\s*$")


def fill_markdown(markdown: str, url: str, card: Card) -> str:
    """A post's stored text with the link's card written in (pure). An X post's bare `## Links`
    entry becomes the card: the same text a render with the card attached gives. A blog or
    Substack post links inline and has no such entry, so the card goes into a `## Links` section
    above its footer. A card already there is left alone."""
    if not card.says_something:
        return markdown
    bare = re.compile(r"^- \[[^\]\n]*\]\(" + re.escape(url) + r"\)$", re.M)
    block = card_block(card.title, card.description, url)
    if bare.search(markdown):
        return bare.sub(lambda _m: block, markdown)
    if f"]({url})**" in markdown:
        return markdown
    foot = _FOOTER.search(markdown)
    head, tail = (markdown[:foot.start()], markdown[foot.start():]) if foot else (markdown, "\n")
    section = "" if "\n## Links\n" in head else "## Links\n\n"
    return f"{head.rstrip()}\n\n{section}{block}\n{tail}"
