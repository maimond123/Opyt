"""Owed references, retries and completion live on their referring posts."""
from __future__ import annotations

from pipeline.kb.schema import stored_atom_id

import json
from dataclasses import dataclass
from datetime import datetime

from pipeline.kb import link_router, paper_rule, schema
from pipeline.kb.writes import write_tx
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop.retries import may_run
from pipeline.worker.quota_loop.source import Item, Retry
from pipeline.oracles import roster
from pipeline.worker.quota_loop.source import RetryRows


@dataclass(frozen=True)
class Link:
    post: str
    url: str
    ref: dict
    retry: Retry
    # A paper linked from this post becomes a record (`paper_rule.earns_record`), read off the
    # referring post when the link is read.
    earns_record: bool = False

    @property
    def key(self) -> tuple:
        return self.post, self.url, json.dumps(self.ref, sort_keys=True)

    @property
    def target(self) -> str:
        return link_router.parse_tweet_id(self.url) or self.url


def _entries(conn, post):
    row = conn.execute("SELECT who_id, entry_mode, payload FROM atoms WHERE atom_id=?", (post,)).fetchone()
    if row is None or (row["entry_mode"] == "oracle-footprint" and not roster.on_roster(conn, row["who_id"])):
        return None
    return json.loads(row["payload"] or "{}").get("links_owed") or []


def _matches(entry, key):
    return (entry.get("url") == key[1]
            and json.dumps(entry.get("ref") or {}, sort_keys=True) == key[2])


def current(conn, link: Link) -> bool:
    return any(_matches(e, link.key) for e in _entries(conn, link.post) or [])


def owed(conn, *, x: bool) -> list[Link]:
    """Every link a post still owes, from posts of every source: X posts owe what they link, and
    blog, Substack and GitHub posts owe their paper links (`owed_links.long_form_links`)."""
    links = []
    for row in conn.execute("SELECT atom_id, source_type, entry_mode FROM atoms "
                            "WHERE json_array_length(payload, '$.links_owed') > 0 "
                            "ORDER BY when_ts DESC, rowid"):
        earns = paper_rule.earns_record(row[1], row[2])
        for e in _entries(conn, row[0]) or []:
            if (link_router.reference_kind(e["url"]) == "x") != x:
                continue
            prior = e.get("retry") or {}
            # An entry with only `tries` (no retry state) keeps its count as blip failures.
            if not prior and e.get("tries"):
                prior = {"status": "blip", "failures": e["tries"]}
            links.append(Link(row[0], e["url"], e.get("ref") or {}, Retry(**prior), earns))
    return links


def line_papers(links: list[Link]) -> set[str]:
    """The paper URLs no referring post earns a record for: each gets a line in its posts
    (`references/cards`) instead of a record (`references/documents`). Read over every owed link,
    not only the ready ones, so a URL's route does not change with its retry timing. Pure."""
    earns: dict[str, bool] = {}
    for link in links:
        if link_router.reference_kind(link.url) == "paper":
            earns[link.url] = earns.get(link.url, False) or link.earns_record
    return {url for url, e in earns.items() if not e}


def ready(links: list[Link], now: datetime) -> list[Link]:
    return [link for link in links if may_run(link.retry.retry_at, link.retry.failures, now)]


def next_due(links: list[Link], now: datetime) -> datetime | None:
    times = [at for link in links if (at := parse_ts(link.retry.retry_at)) is not None and at > now]
    return min(times) if times else None


class LinkRows(RetryRows):
    """A batch contains only equal retry states, so the shared loop charges each remaining link equally."""

    def read(self, link: Link) -> Retry:
        return link.retry

    def write(self, conn, key: tuple, retry: Retry) -> None:
        with write_tx(conn, site="references.retry"):
            for post, url, ref in key:
                entries = _entries(conn, post)
                if entries is None:
                    continue
                updated = [{**e, "retry": vars(retry)} if _matches(e, (post, url, ref)) else e
                           for e in entries]
                conn.execute("UPDATE atoms SET payload=json_set(payload, '$.links_owed', json(?)) "
                             "WHERE atom_id=?", (json.dumps(updated), post))

    def clear(self, conn, key: tuple) -> None:
        self.write(conn, key, Retry())


ROWS = LinkRows()


def item(links: list[Link], platform: str) -> Item:
    return Item(ROWS, tuple(link.key for link in links), links[0].retry, platform)


def finish(conn, resolved: list[tuple[Link, str | None]]) -> None:
    """Record provenance and remove exactly the entries we read, in one transaction.

    None means a successful lookup established no capturable target. New links or changed
    referrers survive a concurrent thread rewrite. A failed reference write rolls back completion.
    """
    with write_tx(conn, site="references.complete"):
        for link, target in resolved:
            entries = _entries(conn, link.post)
            if entries is None or not any(_matches(e, link.key) for e in entries):
                continue
            if target is not None:
                if not stored_atom_id(conn, target):
                    raise RuntimeError(f"reference target disappeared: {target}")
                schema.add_reference(conn, target, link.ref)
                payload = json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id=?",
                                                  (target,)).fetchone()[0])
                if not any((r.get("by"), r.get("post")) == (link.ref.get("by"), link.ref.get("post"))
                           for r in payload.get("referenced_by", [])):
                    raise RuntimeError(f"reference did not save: {target}")
            remaining = [e for e in entries if not _matches(e, link.key)]
            conn.execute("UPDATE atoms SET payload=json_set(payload, '$.links_owed', json(?)) "
                         "WHERE atom_id=?", (json.dumps(remaining), link.post))
