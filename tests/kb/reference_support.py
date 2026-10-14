"""Drive real reference sources against scripted transports, without the removed drain job."""

from pipeline.kb.schema import stored_atom_id
from collections import Counter
from functools import partial
from datetime import datetime, timezone

from pipeline.kb import ingest_x_footprint as fp, link_mint, link_router, owed_links, schema
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.quota_loop.retries import Outages, PLATFORM_PAUSES_S, Wait
from pipeline.oracles import roster
from pipeline.oracles.sources.references.documents import Documents
from pipeline.oracles.sources.references.x_posts import XPosts


class Capture:
    def __init__(self, conn, **kwargs):
        self.conn = conn
        self.waiting = False
        health = Outages(PLATFORM_PAUSES_S)
        self.web = QuotaLoop("web", [Documents(health)], executor=Serial(), outages=health, connect=schema.connect)
        self.x = QuotaLoop("x:TweetResultsByRestIds", [XPosts()], executor=Serial(), meter=lambda: None, outages=Outages(PLATFORM_PAUSES_S), connect=schema.connect)
        for loop in (self.web, self.x):
            loop.ctx.cache.update(session={})
        self.refs = []
        self._refs_flushed = 0
        self.captured = {}

    def dispatch(self, group):
        post = f"xprofile:{group[0]['id']}"
        handle = (group[0].get("author") or {}).get("userName") or ""
        who = f"x:user:{group[0]['author']['id']}"
        roster.upsert_oracle(self.conn, who, name=handle)
        entries = owed_links.owed_links(group, post, own_hosts=fp._own_site_hosts(self.conn, handle))
        schema.upsert_atom(self.conn, {"atom_id": post, "source_type": "x", "who_id": who,
                                      "entry_mode": "oracle-footprint", "payload": {"links_owed": entries}})
        self.conn.commit()
        kinds = Counter()
        for unit in self.web.sources[0].candidates(self.conn, datetime.now(timezone.utc)):
            kind = link_router.reference_kind(unit.key)
            aid = link_mint.predicted_atom_id(unit.key, kind)
            if kind != "article" or (aid and stored_atom_id(self.conn, aid)):
                kinds.update(self._run(self.web, unit))
        return kinds

    def _run(self, loop, unit):
        source = loop.sources[0]
        kinds = Counter()
        original = source.record

        def record(conn, unit, fetched):
            resolved = fetched[0]
            result = original(conn, unit, fetched)
            for link, aid in resolved:
                if aid:
                    kind = {"blog": "article"}.get(aid.split(":", 1)[0], aid.split(":", 1)[0])
                    if link.url not in self.captured:
                        kinds[kind] += 1
                        self.captured[link.url] = aid
                    self.refs.append((aid, link.ref))
            return result

        source.record = record
        try:
            self.waiting = isinstance(loop.run_unit(self.conn, source, unit), Wait)
        finally:
            source.record = original
        return kinds

    def drain(self):
        kinds = Counter()
        for loop in (self.x, self.web):
            for _ in range(500):
                now = datetime.now(timezone.utc)
                unit = loop.rotation.pick(loop.sources, partial(loop.executor.units, conn=self.conn, now=now), None, loop.strikes, now.timestamp())
                if unit is None:
                    break
                kinds.update(self._run(loop, unit[1]))
                if self.waiting:
                    break
            else:
                raise AssertionError("reference sources did not settle")
        return kinds

    def flush_references(self):
        new = len(self.refs) - self._refs_flushed
        self._refs_flushed = len(self.refs)
        return new


def drain(conn):
    roster.upsert_oracle(conn, "x:user:99", name="Carol")
    capture = Capture(conn)
    kinds = capture.drain()
    return {"kinds": dict(kinds), "referrers": capture.flush_references(),
            "captured": len(capture.captured)}
