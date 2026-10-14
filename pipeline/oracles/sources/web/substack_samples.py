"""
pipeline/oracles/sources/web/substack_samples.py — a thin sample for candidates known only by
their Substack: the titles of their latest posts, and the X handle they declared there.

One unit is one candidate and one GET of their publication's archive (`?sort=new&limit=5`). It
writes up to five probe atoms (title and subtitle, never the body) under the Substack member's id,
through `probe_store` like X scouting, so `oracle(action='candidates')` can answer from what they
write. The primary byline's `twitter_screen_name` is written to the member's `profile.x_handle`,
which puts the candidate on X scouting's queue (`timeline/samples.candidate_queue`).

A candidate is sampled once: `ok` and `empty` are final. A failed GET writes nothing and is charged
by the quota loop on the candidate's `probe_pulls` row (`scouting.PROBE_ROWS`), as an X page is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit

from pipeline.kb import schema
from pipeline.kb import writes
from pipeline.kb.ingest_common import BASIS_OBSERVED, BODY_PARTIAL, body_fields, changed_hash
from pipeline.oracles import probe_store
from pipeline.oracles import ranking
from pipeline.oracles import roster
from pipeline.oracles.sources.timeline import samples
from pipeline.oracles.sources.timeline.scouting import PROBE_ROWS
from pipeline.oracles.sources.web.platform import QUOTA, own_site
from pipeline.platforms.substack.http import public_get_json
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import SCOUTING, Changed, Item, Outcome, PoolSource, Unit

POSTS = 5                                  # titles per candidate's sample
PLATFORM = "substack"
_RANK_TTL_S = 300.0
_ATOM_PREFIX = "ssprobe"


# ── the queue ─────────────────────────────────────────────────────────────────

def substack_member(members: list[dict]) -> tuple[str, str] | None:
    """A cluster's members (`entity_id`, `identity_links`) → `(substack entity id, publication
    URL)` for the first Substack member with a home link, or None."""
    for m in members:
        links = m["identity_links"] or []
        if m["entity_id"].startswith("substack:") and links and str(links[0]).startswith("http"):
            return m["entity_id"], str(links[0]).rstrip("/")
    return None


def queue(conn) -> list[dict]:
    """Candidates with no X identity and a Substack publication, in the screen's rank order."""
    out = []
    for cand in ranking.rank_candidates(conn):
        if samples.x_user_id(cand.members):
            continue
        rows = [{"entity_id": r["entity_id"], "identity_links": ranking._loads(r["identity_links"])}
                for r in schema.entities_for_canonical(conn, cand.canonical_id)]
        hit = substack_member(rows)
        if hit:
            out.append({"who_id": hit[0], "url": hit[1], "canonical_id": cand.canonical_id,
                        "name": cand.name})
    return out


# ── one archive response ──────────────────────────────────────────────────────

def sample_atoms(posts: list[dict], who_id: str) -> list[dict]:
    """Archive posts → write-ready probe atoms, each with its `_markdown`: title and subtitle
    only, the newest `POSTS`."""
    out = []
    for p in posts[:POSTS]:
        title = (p.get("title") or "").strip()
        if not p.get("id") or not title:
            continue
        subtitle = (p.get("subtitle") or p.get("description") or "").strip()
        md = f"# {title}\n\n{subtitle}".strip()
        out.append({
            "atom_id": f"{_ATOM_PREFIX}:{p['id']}",
            "source_type": "substack",
            "who_id": who_id,
            "when_ts": p.get("post_date"),
            "when_precision": "day" if p.get("post_date") else "unknown",
            "source_url": p.get("canonical_url"),
            "description": title,
            "payload": {"subtitle": subtitle or None,
                        **body_fields(BODY_PARTIAL, BASIS_OBSERVED)},
            "_markdown": md,
        })
    return out


def primary_x_handle(posts: list[dict]) -> str | None:
    """The X handle the publication's primary author set on Substack: the byline whose
    `publicationUsers` names this publication as primary. A guest's byline never counts."""
    for p in posts:
        pub = p.get("publication_id")
        for b in p.get("publishedBylines") or []:
            if any(u.get("publication_id") == pub and u.get("is_primary")
                   for u in b.get("publicationUsers") or []):
                handle = (b.get("twitter_screen_name") or "").strip().lstrip("@")
                if handle:
                    return handle
    return None


# ── the source ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sampled:
    status: str
    written: int


class SubstackSamples(PoolSource):
    name, quota, klass, floor = "substack_samples", QUOTA, SCOUTING, 0

    def __init__(self) -> None:
        self._ranked: tuple[float, list[dict]] | None = None

    def _queue(self, conn) -> list[dict]:
        # As scouting: the ranking is a prior over who goes first, recomputed every few minutes.
        if self._ranked is None or time.monotonic() - self._ranked[0] > _RANK_TTL_S:
            self._ranked = (time.monotonic(), queue(conn))
        return self._ranked[1]

    def candidates(self, conn, now: datetime):
        if not roster.list_oracles(conn):
            return
        pulls = probe_store.pull_states(conn)
        for cand in self._queue(conn):
            row = pulls.get(cand["who_id"])
            if row is None or (row["status"] == probe_store.STATUS_FAILED
                               and retries.may_run(row.get("retry_at"), row.get("failures"), now)):
                item = Item(PROBE_ROWS, (cand["who_id"],), PROBE_ROWS.read(row), PLATFORM,
                            kind=retries.BLIP, own_site=own_site(PLATFORM, cand["url"]))
                yield Unit(f"substack_samples:{cand['who_id']}", cand, item=item,
                           host=urlsplit(cand["url"]).hostname)

    def next_unit(self, conn, now):
        return next(iter(self.candidates(conn, now)), None)

    def next_due(self, conn, now: datetime) -> datetime | None:
        retry = parse_ts(probe_store.next_retry_at(conn))
        return retry if retry is not None and retry > now else None

    def fetch(self, conn, unit: Unit, ctx) -> Sampled:
        cand = unit.payload
        probe_store.init_probe_schema(conn)         # a failed GET's charge lands on its table
        posts = public_get_json(f"{cand['url']}/api/v1/archive",
                                {"sort": "new", "offset": 0, "limit": POSTS})
        posts = posts if isinstance(posts, list) else []
        atoms = sample_atoms(posts, cand["who_id"])
        seen = probe_store.load_probe_hashes(conn, cand["who_id"])
        changed = []
        for atom in atoms:
            md = atom.pop("_markdown")
            decided = changed_hash(atom["atom_id"], md, seen)
            if decided is None:
                continue
            atom["raw_hash"] = decided
            changed.append((atom, md))
        status = probe_store.STATUS_OK if posts else probe_store.STATUS_EMPTY
        handle = primary_x_handle(posts)
        with writes.write_tx(conn, site="substack_samples"):
            if ctx.lease_lost():
                raise Changed("web quota lease lost")
            for atom, md in changed:
                probe_store.store_probe_atom(conn, atom, md)
            if handle:
                schema.upsert_entity(conn, cand["who_id"], profile={"x_handle": handle})
            probe_store.record_sample(conn, cand["who_id"], status, atoms=len(changed),
                                      pages_done=1, tail_id=None)
        return Sampled(status, len(changed))

    def record(self, conn, unit: Unit, fetched: Sampled) -> Outcome:
        return Outcome(fetched.status)
