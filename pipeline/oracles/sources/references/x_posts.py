"""One batch of linked X posts per TweetResultsByRestIds turn."""

from pipeline.kb.schema import stored_atom_id
from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import ingest_x, schema
from pipeline.kb.writes import write_tx
from pipeline.platforms import sessions
from pipeline.platforms.x import conversation as x_conversation
from pipeline.worker.quota_loop.retries import UNSAVED
from pipeline.worker.quota_loop.source import YOURS, Outcome, Unit

from pipeline.oracles.sources.inputs import image_cache, save_images
from pipeline.oracles.sources.references import capture
from pipeline.oracles.sources.references import state
from pipeline.worker.quota_loop.source import Source

QUOTA = "x:TweetResultsByRestIds"


def _resolve(conn, links, targets, returned):
    """Each link's target once the batch is written: the post, or None for a lookup that
    established nothing to capture (gone, or the referrer's own post). A target that should exist
    and does not leaves its link owed, and the batch failed."""
    resolved, failed = [], False
    for link in links:
        if link.target not in targets:
            failed = True
            continue
        target = targets[link.target]
        norm = returned.get(link.target)
        author = (f"x:user:{(norm.get('author') or {}).get('id')}" if norm else
                  conn.execute("SELECT who_id FROM atoms WHERE atom_id=?", (target,)).fetchone()[0]
                  if target else None)
        own = author == link.ref.get("by")
        if own:
            target = None
        if target and not stored_atom_id(conn, target):
            failed = True
        else:
            resolved.append((link, target))
    return resolved, failed


class XPosts(Source):
    name, quota, klass, floor = "linked_x_posts", QUOTA, YOURS, 0

    def next_unit(self, conn, now):
        links = state.ready(state.owed(conn, x=True), now)
        if not links:
            return None
        # Equal retry histories make one request's charge apply consistently to every item.
        prior = links[0].retry
        ids = list(dict.fromkeys(l.target for l in links if l.retry == prior))[:x_conversation.TWEETSBYIDS_MAX_IDS]
        batch = [l for l in links if l.target in ids and l.retry == prior]
        return Unit("|".join(ids), batch, item=state.item(batch, "x"))

    def fetch(self, conn, unit, ctx):
        links = [link for link in unit.payload if state.current(conn, link)]
        if not links:
            return [], False
        cache = image_cache(ctx)
        seen = schema.load_hashes(conn, "x")
        missing = list(dict.fromkeys(l.target for l in links if f"x:{l.target}" not in seen))
        norms = []
        if missing:
            if "session" not in ctx.cache:
                ctx.cache["session"] = sessions.x_session("https://x.com/home")
            try:
                norms = x_conversation.fetch_tweets_by_ids(ctx.cache["session"], missing)
            except SyncAuthError:
                ctx.cache.pop("session", None)
                raise
        returned = {str(n["id"]): n for n in norms}
        held = capture.Held()
        targets = {}
        for tid in dict.fromkeys(l.target for l in links):
            aid, norm = f"x:{tid}", returned.get(tid)
            if aid in seen:
                targets[tid] = aid
            elif norm is None:
                targets[tid] = None
            else:
                by = f"x:user:{(norm.get('author') or {}).get('id')}"
                if all(l.ref.get("by") == by for l in links if l.target == tid):
                    targets[tid] = None
                    continue
                md = ingest_x.render_referenced_post(norm, cache)
                if md is None:
                    targets[tid] = None
                    continue
                status, target = ingest_x.referenced_x_atom(conn, norm, md, seen=seen, store=held, isolate=False)
                if status in ("saved", "present"):
                    targets[tid] = target
        save_images(cache)
        with write_tx(conn, site="references.x_posts"):
            capture.write(conn, ctx, held,
                          lambda aid: [l for l in links if aid == f"x:{l.target}"])
            resolved, failed = _resolve(conn, links, targets, returned)
            state.finish(conn, resolved)
        return resolved, failed

    def record(self, conn, unit, fetched):
        _resolved, failed = fetched
        return Outcome(UNSAVED, failure=UNSAVED, detail="linked post did not save") if failed else Outcome("captured")

    def next_due(self, conn, now):
        return state.next_due(state.owed(conn, x=True), now)
