"""One linked web document per turn, shared by every post that references its URL."""

from pipeline.kb.schema import stored_atom_id
from urllib.parse import urlsplit
from pipeline.kb import ingest_blog, ingest_papers, link_mint, link_router, schema
from pipeline.kb.writes import write_tx
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import YOURS, Outcome, Unit

from pipeline.oracles.sources.inputs import image_cache, save_images
from pipeline.oracles.sources.references import capture
from pipeline.oracles.sources.references import state
from pipeline.oracles.sources.web.platform import platform_key
from pipeline.worker.quota_loop.source import PoolSource


def _platform(url):
    kind = link_router.reference_kind(url)
    return platform_key(kind if kind in ("substack", "github") else "blog", url)


def _owed(conn) -> list[state.Link]:
    """The links owed a document. A card link is read for its card only, and so is a paper no
    referring post earns a record for (`references/cards.py`)."""
    links = state.owed(conn, x=False)
    lines = state.line_papers(links)
    return [link for link in links if not link_router.is_card_link(link.url) and link.url not in lines]


def _fetch(conn, url, cache, held):
    kind = link_router.reference_kind(url)
    if kind is None:
        return "final", None
    seen = schema.load_hashes(conn, "blog")
    if kind == "article" and (aid := ingest_blog.canon_post_url(url)) in seen:
        return "present", aid
    content_type, prefetched = None, None
    if kind == "article" and (deep := link_router.classify_link_deep(url)):
        kind, url, content_type = deep
        aid = link_mint.predicted_atom_id(url, "paper", content_type=content_type)
        if aid and stored_atom_id(conn, aid):
            return "present", aid
        paper = ingest_papers.paper_from_url(url, **({"content_type": content_type} if content_type else {}))
        if not paper:
            return "failed", None
        prefetched = {"paper": paper}
    if kind == "article":
        prepared = ingest_blog.prepare_article(url, entry_mode="author_referenced", seen=seen,
                                               img_cache=cache, log_tag="references", isolate=False)
        return ingest_blog.article_atom_from_url(conn, url, entry_mode="author_referenced",
                                                seen=seen, img_cache=cache, prepared=prepared,
                                                store=held)
    result = link_mint.mint_artifact(conn, url, kind, img_cache=cache, store=held,
                                       content_type=content_type, prefetched=prefetched, isolate=False)
    return result["status"], result["atom_id"]


def _settle(conn, links, status, aid, url):
    """What the document's links come to once it is written: resolved to it, or final with no
    target, or still owed and charged by kind."""
    if status == "final":
        return [(link, None) for link in links], Outcome("final")
    if aid and stored_atom_id(conn, aid):
        return [(link, aid) for link in links], Outcome("captured")
    kind = (retries.ACCOUNT if status == "blocked" else
            retries.UNSAVED if aid else retries.BLIP)
    return [], Outcome(kind, failure=kind, detail=f"{status}: {url}",
                       outage=status == "blocked", answered=status != "blocked")


class Documents(PoolSource):
    name, quota, klass, floor = "linked_documents", "web", YOURS, 0

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
        """The linked document, fetched, then written with its links' completion in one
        transaction. Returns the links it resolved, and what it came to."""
        links = [link for link in unit.payload if state.current(conn, link)]
        if not links:
            return [], Outcome("changed")
        cache = image_cache(ctx)
        held = capture.Held()
        try:
            status, aid = _fetch(conn, unit.key, cache, held)
        finally:
            save_images(cache)
        with write_tx(conn, site="references.document"):
            capture.write(conn, ctx, held, lambda _aid: links)
            resolved, outcome = _settle(conn, links, status, aid, unit.key)
            state.finish(conn, resolved)
        return resolved, outcome

    def record(self, conn, unit, fetched):
        _resolved, outcome = fetched
        return outcome

    def next_due(self, conn, now):
        times = [t for t in (state.next_due(_owed(conn), now), self.health.next_resume(now)) if t]
        return min(times) if times else None
