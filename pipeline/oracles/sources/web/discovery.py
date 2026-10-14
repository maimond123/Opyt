"""
pipeline/oracles/sources/web/discovery.py — finding an Oracle's sources beyond what the user
named. The `discovery` source, one Oracle per unit, in two stages:

  root   owed as soon as the Oracle is asked for. Reads their root profile and registers what it
         declares that a pull needs from the start: their X (a pair, only while X is connected)
         and their ORCID (an `openalex` pair). Stamps `oracle_asks.root_probed_at`.
  links  owed once every starting pair finished its first pull (`windows.first_pulls_done`).
         The full pass (`profile.discover_profile`), reading every profile page of theirs
         beside the root (`expand.member_profiles`): every verdict is recorded in
         `oracle_review_items`, and a source we are sure of is owed its first pull. Stamps
         `discovered_at` and clears `rediscover`.

Both stages write in one transaction, only while the Oracle is still confirmed, and never touch a
source already recorded for it (`review_items.held_source_keys`), so a dismissed one stays out.

A profile page that does not resolve, the root or another of theirs, is a failure, not an Oracle
with nothing to find: an X probe on a spent window or a dead session comes back empty, a GitHub
profile on a spent rate limit too, and stamping it would lose what that page declares for good.
Every discovery failure waits as a blip on the ask's `discovery_*` columns (`DISCOVERY_ROWS`),
shared by both stages, and none counts toward a platform's outage.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.kb.expand import _x_handle_from_url, member_profiles
from pipeline.kb import resolve
from pipeline.kb import writes
from pipeline.oracles import asks
from pipeline.oracles import coverage
from pipeline.oracles import registry
from pipeline.oracles import review_items
from pipeline.oracles import roster
from pipeline.oracles import told
from pipeline.oracles import windows
from pipeline.oracles.discovery import profile
from pipeline.oracles.discovery import verdict as v
from pipeline.oracles.sources import pair_rows
from pipeline.oracles.sources.archive import state as archive_state
from pipeline.oracles.sources.web.platform import QUOTA
from pipeline.platforms import connections
from pipeline.platforms.papers import openalex
from pipeline.timeparse import parse_ts, utc_iso, utc_now
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import YOURS, Context, Item, Outcome, Retry, Unit
from pipeline.worker.quota_loop.source import PoolSource
from pipeline.worker.quota_loop.source import RetryRows

ROOT, LINKS = "root", "links"


@dataclass(frozen=True)
class Found:
    stage: str
    root: dict | None                  # `{seed, seed_type}`; None when nothing is rootable
    profile: dict = field(default_factory=dict)
    openalex_ids: dict = field(default_factory=dict)   # ORCID verdict key → OpenAlex author id

    @property
    def resolved(self) -> bool:
        """Every profile page the stage read answered. False is a failed probe, not a person
        with no sources."""
        return self.root is None or bool(self.profile.get("display_name"))


def _orcid_ids(verdicts: list[dict]) -> dict:
    """Each sure ORCID → its OpenAlex author id, one free lookup each. A lookup that fails is
    left out, and the ORCID stays unrecorded so a later pass resolves it."""
    out = {}
    for d in verdicts:
        if d["verdict"] != v.SURE or d["source_type"] != "orcid":
            continue
        orcid = d["url"].rstrip("/").rsplit("/", 1)[-1]
        try:
            rec = openalex.OpenAlexClient().author_by_orcid(orcid)
        except Exception as e:
            log(f"  [discovery] ORCID {orcid} lookup failed: {type(e).__name__}: {e}")
            continue
        author_id = str((rec or {}).get("id") or "").rsplit("/", 1)[-1]
        if author_id:
            out[d["key"]] = author_id
    return out


def find_sources(conn: sqlite3.Connection, oracle: dict, stage: str = LINKS) -> Found:
    """One stage of discovery for one Oracle. `oracle` is a `confirmed_oracles` row (with
    `members`). The root stage reads the root alone; the links stage every profile page of
    theirs, the root first. No root, no network."""
    pages = member_profiles(conn, oracle)
    if not pages:
        return Found(stage, None)
    root = pages[0]
    if stage == ROOT:
        found = profile.discover_root(root["seed"], seed_type=root["seed_type"])
    else:
        found = profile.discover_profile(root["seed"], seed_type=root["seed_type"],
                                         others=pages[1:], conn=conn,
                                         oracle_id=oracle["canonical_id"])
    return Found(stage, root, found, _orcid_ids(found.get("verdicts") or []))


def plan(verdicts: list[dict], held: set, *, x_connected: bool, openalex_ids: dict) -> dict:
    """What `register` writes for one Oracle's verdicts. Pure.

    | verdict                         | review row        | also                              |
    |---------------------------------|-------------------|-----------------------------------|
    | held (any row for its key)      | none              | nothing                           |
    | sure X, X connected             | added             | an X pair                         |
    | sure X, X not connected         | none (next pass)  | nothing                           |
    | sure ORCID, resolved            | added             | an `openalex` pair                |
    | sure ORCID, lookup failed       | none (next pass)  | nothing                           |
    | sure, any other type            | added             | a `found_sources` first pull      |
    | merge                           | added             | nothing                           |
    | also                            | added, kind also  | nothing                           |
    | drop                            | none              | nothing                           |
    """
    rows, x_handles, openalex_pairs, first_pulls = [], [], [], []
    for d in verdicts:
        stype, url = d["source_type"], d["url"]
        if (stype, review_items.source_key(url)) in held or d["verdict"] == v.DROP:
            continue
        item = {"source_type": stype, "url": url, "reasons": d.get("reasons") or [],
                "kind": review_items.SOURCE, "status": review_items.ADDED}
        if d["verdict"] == v.SURE:
            if stype == "x":
                handle = (_x_handle_from_url(url) or "").lstrip("@").lower()
                if not (x_connected and handle):
                    continue
                x_handles.append(handle)
            elif stype == "orcid":
                if d["key"] not in openalex_ids:
                    continue
                openalex_pairs.append(openalex_ids[d["key"]])
            else:
                first_pulls.append(d)
        elif d["verdict"] == v.ALSO:
            item["kind"] = review_items.ALSO
        rows.append(item)
    return {"rows": rows, "x_handles": x_handles, "openalex": openalex_pairs,
            "first_pulls": first_pulls}


def register(conn: sqlite3.Connection, canonical_id: str, found: Found, *, x_connected: bool,
             now: datetime | None = None) -> str:
    """Record what one stage found for one Oracle (the module docstring). Returns `found`, or
    `forgotten` when the Oracle left the roster while discovery was fetching."""
    now = now or utc_now()
    registry.init_state_schema(conn)                   # DDL commits, so it stays outside the transaction
    told.init_schema(conn)
    head = schema.current_canonical(conn, canonical_id)
    verdicts = (found.profile or {}).get("verdicts") or []
    with writes.write_tx(conn, site="discovery"):
        if not roster.on_roster(conn, canonical_id):
            return "forgotten"
        held = review_items.held_source_keys(conn, canonical_id)
        todo = plan(verdicts, held, x_connected=x_connected, openalex_ids=found.openalex_ids)
        if found.stage == ROOT and found.root and found.root["seed_type"] == "x" and x_connected:
            todo["x_handles"].append(found.root["seed"].lstrip("@").lower())
        review_items.write_found(conn, canonical_id, todo["rows"])
        for handle in todo["x_handles"]:
            registry.add_x_pair(conn, head, handle)
        for author_id in todo["openalex"]:
            registry.add_openalex_pair(conn, head, author_id)
        for d in verdicts:          # the OpenAlex entity carries its ORCID, so resolve merges it
            if d["source_type"] == "orcid" and found.openalex_ids.get(d["key"]) in todo["openalex"]:
                schema.upsert_entity(conn, f"openalex:{found.openalex_ids[d['key']]}",
                                     identity_links=[d["url"]])
        merges = [d for d in verdicts if d["verdict"] == v.MERGE
                  and (d["source_type"], review_items.source_key(d["url"])) not in held]
        for d in todo["first_pulls"] + merges:
            coverage.take_over(conn, d["source_type"], d["url"])
        for d in todo["first_pulls"]:
            meta = {"metadata": {"reasons": list(d.get("reasons") or []),
                                 "post_count": d.get("post_count") or 0}}
            conn.execute(
                "INSERT INTO found_sources (canonical_id, source_type, url, metadata, found_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(canonical_id, source_type, url) DO NOTHING",
                (canonical_id, d["source_type"], d["url"], json.dumps(meta), utc_iso(now)))
        if found.stage == ROOT:
            conn.execute("UPDATE oracle_asks SET root_probed_at=? WHERE canonical_id=?",
                         (utc_iso(now), canonical_id))
        else:
            conn.execute("UPDATE oracle_asks SET discovered_at=?, rediscover=0 "
                         "WHERE canonical_id=?", (utc_iso(now), canonical_id))
        if todo["first_pulls"]:
            told.reopen(conn, canonical_id, ["archives"])   # what is beyond X grew: say it again
    if any(r["status"] == review_items.ADDED and r["kind"] == review_items.SOURCE
           for r in todo["rows"]):
        resolve.resolve_entities(conn)                 # the added sources join the Oracle
    return "found"


class _DiscoveryRows(RetryRows):
    """`oracle_asks`' `discovery_*` columns, keyed by `(canonical_id,)`. There is no status
    column: every discovery failure waits as a blip, so the count always continues."""

    def read(self, ask: asks.Ask) -> Retry:
        return Retry(retries.BLIP, ask.discovery_failures, ask.discovery_retry_at,
                     ask.discovery_error)

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        with writes.write_tx(conn, site="discovery.failure"):
            conn.execute("UPDATE oracle_asks SET discovery_failures=?, discovery_retry_at=?, "
                         "discovery_error=? WHERE canonical_id=?",
                         (retry.failures, retry.retry_at, retry.last_error, *key))

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        with writes.write_tx(conn, site="discovery"):
            conn.execute("UPDATE oracle_asks SET discovery_failures=0, discovery_retry_at=NULL, "
                         "discovery_error=NULL WHERE canonical_id=?", key)


DISCOVERY_ROWS = _DiscoveryRows()


def stage_owed(ask: asks.Ask, rows: list, now: datetime,
               exhausted: frozenset = frozenset()) -> str | None:
    """Which stage this ask owes, if any. Pure; `rows` are the Oracle's registered pairs,
    `exhausted` those whose archive has given up on all it has left (`windows.first_pulls_done`).

    | ask                                                       | owed  |
    |-----------------------------------------------------------|-------|
    | root not read                                             | root  |
    | not discovered (or a rediscovery), first pulls all done   | links |
    | otherwise                                                 | none  |
    """
    if ask.root_probed_at is None:
        return ROOT
    if ((ask.discovered_at is None or ask.rediscover)
            and windows.first_pulls_done(rows, ask, now, exhausted)):
        return LINKS
    return None


class Discovery(PoolSource):
    """YOURS: one stage of one Oracle's discovery (`stage_owed`), oldest ask first."""
    name, quota, klass, floor = "discovery", QUOTA, YOURS, 0

    def _owed(self, conn, now: datetime) -> list[tuple[str, asks.Ask]]:
        out = []
        for a in asks.list_asks(conn):
            if not roster.on_roster(conn, a.canonical_id):
                continue
            head = schema.current_canonical(conn, a.canonical_id)
            rows = [r for r in registry.list_sources(conn, [head]) if r.status != "paused"]
            exhausted = frozenset((r.source_type, r.source_key) for r in rows
                                  if archive_state.exhausted(conn, pair_rows.unit_key(r)))
            if (stage := stage_owed(a, rows, now, exhausted)) is not None:
                out.append((stage, a))
        return out

    def candidates(self, conn, now):
        for stage, ask in self._owed(conn, now):
            if retries.may_run(ask.discovery_retry_at, ask.discovery_failures, now):
                item = Item(DISCOVERY_ROWS, (ask.canonical_id,), DISCOVERY_ROWS.read(ask),
                            kind=retries.BLIP)
                yield Unit(f"{ask.canonical_id}|discovery|{stage}", (stage, ask), item=item)

    def next_unit(self, conn, now):
        return next(self.candidates(conn, now), None)

    def fetch(self, conn, unit: Unit, ctx: Context) -> Found | None:
        """One stage's findings, or None when the Oracle is no longer on the roster."""
        stage, ask = unit.payload
        head = schema.current_canonical(conn, ask.canonical_id)
        oracle = next((o for o in roster.confirmed_oracles(conn) if o["canonical_id"] == head),
                      None)
        if oracle is None:
            return None
        return find_sources(conn, oracle, stage)

    def record(self, conn, unit: Unit, found: Found | None) -> Outcome:
        _stage, ask = unit.payload
        if found is None:
            return Outcome("forgotten")
        if not found.resolved:
            page = (found.profile.get("unread")
                    or f"{found.root['seed_type']}:{found.root['seed']}")
            return Outcome(retries.BLIP, failure=retries.BLIP,
                           detail=f"{page} did not resolve to a profile")
        return Outcome(register(conn, ask.canonical_id, found,
                                x_connected=connections._connected_x(), now=utc_now()))

    def next_due(self, conn, now: datetime) -> datetime | None:
        times = [t for t in (parse_ts(a.discovery_retry_at) for _s, a in self._owed(conn, now))
                 if t is not None and t > now]
        return min(times) if times else None
