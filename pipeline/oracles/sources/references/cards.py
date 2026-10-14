"""One link's card per turn: a homepage, product, job or event link (`link_router.is_card_link`)
is never fetched as a document, only its title and description (`kb/link_cards.py`). A paper no
referring post earns a record for (`state.line_papers`) gets a card too, its line: title, authors,
year, from the paper's metadata (`link_cards.read_paper`). The card is written into every post
that owes the link, each post rewritten, and kept by URL with the links' completion, in one
transaction.

A post's atom keeps no raw source to render again, so its stored text is filled in place
(`link_cards.fill_markdown`): an X post's `## Links` entry, the same text a render with the card
attached gives, or a `## Links` section added to a blog or Substack post. A post rewritten while
the card was read keeps its link owed, and the next turn fills it from the stored card without
reading the page again.
"""

from urllib.parse import urlsplit

from pipeline.kb import link_cards, link_router, schema
from pipeline.kb.ingest_common import changed_hash, store_atom
from pipeline.kb.writes import write_tx
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import YOURS, Changed, Outcome, PoolSource, Unit

from pipeline.oracles.sources.references import state
from pipeline.oracles.sources.web.platform import platform_key


def _platform(url: str) -> str:
    return platform_key("blog", url)


def _owed(conn) -> list[state.Link]:
    links = state.owed(conn, x=False)
    lines = state.line_papers(links)
    return [link for link in links if link_router.is_card_link(link.url) or link.url in lines]


def _read(url: str) -> link_cards.Card | None:
    """The card read now: a paper's line from its metadata, any other link's from its page."""
    if link_router.reference_kind(url) == "paper":
        return link_cards.read_paper(url)
    return link_cards.read(url)


def _filled(conn, posts: set[str], url: str,
            card: link_cards.Card) -> tuple[dict, dict[str, tuple[dict, str]]]:
    """Each post's text with the card written in. Returns the raw hash each post had when read,
    by atom id (a post that is gone is left out), and the rewrite of each post whose text
    changed, `(atom, text)` by atom id."""
    read, rewrites = {}, {}
    for post in posts:
        row = conn.execute("SELECT * FROM atoms WHERE atom_id=?", (post,)).fetchone()
        if row is None:
            continue
        read[post] = row["raw_hash"]
        text = schema.read_body(conn, post)
        filled = link_cards.fill_markdown(text or "", url, card)
        if text and filled != text:
            decided = changed_hash(post, filled, {post: row["raw_hash"]})
            if decided is not None:
                rewrites[post] = ({**dict(row), "raw_hash": decided}, filled)
    return read, rewrites


class Cards(PoolSource):
    name, quota, klass, floor = "link_cards", "web", YOURS, 0

    def __init__(self, health):
        self.health = health

    def candidates(self, conn, now):
        grouped = {}
        for link in state.ready(_owed(conn), now):
            if not self.health.paused(_platform(link.url), now):
                grouped.setdefault((link.url, link.retry), []).append(link)
        return [Unit(url, links, item=state.item(links, _platform(url)), host=urlsplit(url).hostname)
                for (url, _retry), links in grouped.items()]

    def next_unit(self, conn, now):
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit, ctx):
        """The link's card, read once (or taken from the store), written into each post that
        owes it, and the links completed. Returns what the unit came to."""
        url = unit.key
        links = [link for link in unit.payload if state.current(conn, link)]
        if not links:
            return Outcome("changed")
        card = link_cards.stored(conn, url) or _read(url)
        if card is None:
            return Outcome(retries.BLIP, failure=retries.BLIP, detail=f"page not read: {url}")
        read, rewrites = _filled(conn, {link.post for link in links}, url, card)
        with write_tx(conn, site="references.card"):
            if ctx.lease_lost():
                raise Changed("references quota lease lost")
            link_cards.save(conn, url, card)
            current = {post for post, raw_hash in read.items() if conn.execute(
                "SELECT 1 FROM atoms WHERE atom_id=? AND raw_hash=?", (post, raw_hash)).fetchone()}
            for post in current & rewrites.keys():
                atom, text = rewrites[post]        # another link may have settled meanwhile
                atom["payload"] = conn.execute("SELECT payload FROM atoms WHERE atom_id=?",
                                               (post,)).fetchone()[0]
                store_atom(conn, atom, text)
            state.finish(conn, [(link, None) for link in links if link.post in current])
        return Outcome("filled" if current & rewrites.keys() else "final")

    def record(self, conn, unit, fetched):
        return fetched

    def next_due(self, conn, now):
        times = [t for t in (state.next_due(_owed(conn), now), self.health.next_resume(now)) if t]
        return min(times) if times else None
