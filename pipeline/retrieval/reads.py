"""
pipeline/retrieval/reads.py — the read entry points: search, open, aggregate, feed.

Retrieval + data only, no model — the host model reasons over what these return.

All four take `kb=`: omitted (or "me") is the local store; any other name is a registered peer
(`peers.py`), either an export file opened read-only or a knowledge base served over HTTPS
(`served.ServedKB`). Both return the same envelope, so `engine.py` never learns a store can be
foreign.

Each read is four steps: check the arguments that can refuse, open the knowledge base `kb=` names
(`open_store`, which returns a `KnowledgeBase`), answer, close. A name this install cannot read
answers in the read's normal envelope with a `kb_unknown` notice instead of raising.
"""
from __future__ import annotations

from typing import Protocol

from pipeline.kb import schema
from pipeline.retrieval import notices, peers
from pipeline.retrieval.filters import date_bound
from pipeline.retrieval.local import LocalKB
from pipeline.retrieval.peers import LOCAL_KB
from pipeline.retrieval.served import ServedKB


class KnowledgeBase(Protocol):
    """What a read needs from a knowledge base, local (`local.LocalKB`) or served
    (`served.ServedKB`). The feed is not here: only the service reads one, from its own exports,
    so it opens a `LocalKB` directly (`open_local`)."""

    def search(self, query: str, **call) -> dict: ...
    def open_atom(self, atom_id: str) -> dict: ...
    def aggregate(self, scope: dict, sample: int = 0) -> dict: ...
    def close(self) -> None: ...


def _label_as(kb_name: str, as_kb: str | None) -> str:
    """What the envelope calls the store that answered. `kb=` routed; `as_kb` only relabels, so a
    served peer's opaque key comes back under the reader's own name for it.

    `as_kb == LOCAL_KB` is refused: it is untrusted input, and downstream `kb_name == LOCAL_KB`
    checks mean "my own store" (e.g. `open` reads a paper's full text live only there)."""
    return as_kb if as_kb and as_kb != LOCAL_KB else kb_name


def _known_peers() -> list[str]:
    """The registered peers' names, for the `kb_unknown` notice."""
    return [p["name"] for p in peers.list_peers()]


def _served_row(kb: str | None) -> dict | None:
    """The peer row for `kb` if it names a served (https) peer; None for the local store, a file
    peer, or an unregistered name. A local search never touches the registry."""
    if kb in (None, LOCAL_KB):
        return None
    row = peers.get(kb)
    if row and peers.is_remote(row["location"]):
        return row
    return None


def open_local(kb: str | None, as_kb: str | None = None) -> LocalKB:
    """The `LocalKB` a `kb=` argument names.

    `None`/"me" is the local store, opened writable. Any other name is a file peer, opened
    read-only by `peers.open_peer` (invariant I3); served peers are routed before this. Raises
    `PeerUnavailable`, which the reads below turn into an empty envelope."""
    if kb is None or kb == LOCAL_KB:
        conn, name, label = schema.connect(), LOCAL_KB, None
    else:
        conn, label = peers.open_peer(kb)
        name = kb
    # The store is chosen; `as_kb` is only what to call it.
    return LocalKB(conn, _label_as(name, as_kb), label)


def open_store(kb: str | None, as_kb: str | None = None) -> KnowledgeBase:
    """The knowledge base `kb=` names: a served peer over HTTP, else a SQLite store. Raises
    `PeerUnavailable` for a name this install cannot read."""
    row = _served_row(kb)
    if row is not None:
        return ServedKB(row)
    return open_local(kb, as_kb)


def run_kb_search(query: str, tags: list[str] | None = None, what_kind: str | None = None,
                  source_type: str | None = None, who_id: str | list[str] | None = None,
                  who: str | None = None, date_from: str | None = None,
                  date_to: str | None = None, entry_mode: str | list[str] | None = None,
                  k: int = 8, kb: str | None = None, as_kb: str | None = None) -> dict:
    """Keyword retrieval over atoms → `{hits, notices, insights, trace, frontier_atoms}`.

    `hits` are ranked routing cards (an excerpt of the body + pointer), never a content claim —
    the host `open()`s an atom to assert what it says. `notices` are finished sentences meant to
    be repeated verbatim (filters, resolution, truncation); `insights` are values describing the
    evidence (author/source/topic distributions, date span, filter cost); `trace` is values
    describing what the engine did (candidates, ranked, cutoff). Only `notices` is recited.

    `entry_mode` decides the shape of the answer:

      • absent (the default) → sectioned. `hits` is the full k over HUMAN_ATTESTED atoms, and
        `frontier_atoms` carries keyword-discovered ones as their own capped list. If
        that block fails, it is empty and a `frontier_block_failed` notice says so.
      • given → one list scoped to those modes; no `frontier_atoms` key.

    `who` (handle/URL, resolved via `resolve_who`) and `who_id` (exact) both restrict authorship
    and union when both are given; an unresolvable `who` matches nothing rather than widening.

    `date_from`/`date_to` bound the atom's own date, inclusive; a malformed bound raises
    `ValueError` (see `date_bound`). An undated atom is excluded and counted in
    `undated_excluded`; a year-precision atom is included if its year overlaps the window.

    `payload` is returned verbatim, never filtered on (its keys are source-specific).

    `kb` picks the knowledge base (omitted or "me" is your own); every hit carries its `kb`.
    `as_kb` relabels that store in the envelope only (see `_label_as`)."""
    # Checked before anything opens: a malformed bound fails as an argument error, on a served
    # peer too (its service would otherwise answer 400 and read as an unreachable store).
    date_bound(date_from, end=False)
    date_bound(date_to, end=True)
    try:
        store = open_store(kb, as_kb)
        try:
            return store.search(query, tags=tags, what_kind=what_kind, source_type=source_type,
                                who_id=who_id, who=who, date_from=date_from, date_to=date_to,
                                entry_mode=entry_mode, k=k)
        finally:
            store.close()
    except peers.PeerUnavailable as e:
        # The normal envelope shape, empty; a dead service answers like a missing file.
        return {"hits": [], "notices": [notices.kb_unknown(kb, e, _known_peers())],
                "insights": {},
                "trace": {"kb": kb}}


def kb_open(atom_id: str, kb: str | None = None, as_kb: str | None = None) -> dict:
    """Follow an atom's pointer. Returns the raw snapshot text + live `source_url` so the host
    asserts from the source, not a description. An unknown atom returns an error dict.
    `body_state` is the last check before a stub could be quoted as the whole article.

    `kb` must be the value the hit card carried: an atom id is scoped to its store.

    `as_kb` renames the store in the returned dict and nowhere else — see `_label_as`."""
    try:
        store = open_store(kb, as_kb)
        try:
            return store.open_atom(atom_id)
        finally:
            store.close()
    except peers.PeerUnavailable as e:
        # The same sentence `search` gives, listing the names that do resolve.
        return {"atom_id": atom_id, "kb": kb,
                "error": notices.kb_unknown(kb, e, _known_peers())["message"]}


def kb_aggregate(scope: dict | None = None, kb: str | None = None,
                 as_kb: str | None = None, sample: int = 0) -> dict:
    """A pure-SQL state-of-play skeleton for a dossier: counts by kind/source, trust coverage,
    author distribution, and recent atom descriptions. The host drafts from this, then `open()`s
    pivotal atoms to ground each claim in raw text. `scope` filters like search:
    {tags, what_kind, source_type, who_id, date_from, date_to} (same date-bound semantics).
    There is deliberately no topic list: `source_tags` are captured labels, not a taxonomy.

    No handle resolution here: pass `who_id=` from `search(who=...)`'s `insights.resolved_who`.
    `kb` picks the store (omitted or "me" is your own); `as_kb` relabels it (see `_label_as`).

    `sample` > 0 adds `corpus_sample` (see `corpus_census.py`): a breadth-first spread of that
    many descriptions for the host to name subjects from. Off by default; it is the one expensive
    key."""
    scope = scope or {}
    date_bound(scope.get("date_from"), end=False)
    date_bound(scope.get("date_to"), end=True)
    try:
        store = open_store(kb, as_kb)
        try:
            return store.aggregate(scope, sample)
        finally:
            store.close()
    except peers.PeerUnavailable as e:
        return {"scope": scope, "kb": kb, "total": 0,
                "notices": [notices.kb_unknown(kb, e, _known_peers())]}


def kb_feed(since: str | None = None, since_id: str | None = None, limit: int = 50,
            entry_mode: str | list[str] | None = None,
            date_from: str | None = None, date_to: str | None = None,
            kb: str | None = None, as_kb: str | None = None) -> dict:
    """The newest arrivals in a knowledge base, cursor-paged ("what's new since I last looked").

    The clock is `COALESCE(promoted_at, first_seen)`, surfaced as `seen_at`, so a promoted frontier
    atom is new on the day of promotion. `entry_mode` defaults to `HUMAN_ATTESTED`. `since` means
    "entered this knowledge base"; `date_from`/`date_to` mean "published", as in search.

    No cursor → the newest `limit` items and `more: False`. With a cursor (`since` = a `seen_at`,
    `since_id` = its atom_id tiebreak, required because ingests share seconds) → everything
    strictly after it. The body is always ascending, `next` is the next call's params, and an
    empty page echoes the cursor back."""
    date_bound(date_from, end=False)
    date_bound(date_to, end=True)
    try:
        store = open_local(kb, as_kb)
    except peers.PeerUnavailable as e:
        return {"kb": kb, "items": [], "next": None, "more": False,
                "notices": [notices.kb_unknown(kb, e, _known_peers())]}
    try:
        return store.feed(since=since, since_id=since_id, limit=limit, entry_mode=entry_mode,
                          date_from=date_from, date_to=date_to)
    finally:
        store.close()
