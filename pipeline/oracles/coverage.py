"""
pipeline/oracles/coverage.py — what a pull covered, stamped on the registry.

A pull reports what it reached; this module turns that report into the pair's registry row, and
only claims what was observed: `save_pull` for a web or papers pull the scrapers made,
`stamp_source` and `record_coverage` for a source pulled through the footprint router. The
connectivity rule (`windows.leaves_a_gap`) keeps a walk from claiming a range nobody read.

`take_over`: a source that became an Oracle's covers what OPYT already stored from it because the
Oracle linked it (`author_referenced`): that is the Oracle's own work (`oracle-footprint`). Its
first pull then finds those posts present, under the same atom ids, and writes no duplicates.
"""

from __future__ import annotations

from pipeline.kb import derive
from pipeline.ingestion.utils import log
from pipeline.oracles import pairs
from pipeline.oracles import registry
from pipeline.kb import schema
from pipeline.timeparse import utc_now
from pipeline.oracles import windows
from datetime import datetime
from pipeline.kb import ingest_common


def stamp_source(conn, oracle_id: str, rec: dict, *, x_since, web_since, scholar_since,
                  include_x: bool) -> None:
    """Register and stamp the refresh registry for one source, the moment that source finishes,
    so a run killed part-way leaves a store that knows what it is missing.

    Seeds pairs first: the entities `record_coverage` joins through exist only after this source's
    resolve. A failure raises to the scraper that made the pull."""
    pairs.seed_from_entities(conn, canonical_ids=[oracle_id], include_x=include_x)
    record_coverage(conn, oracle_id, [rec], x_since=windows.x_since(x_since, utc_now()),
                    web_since=web_since, scholar_since=scholar_since)


_OBSERVED = "ingested"


def result_entity_id(rec: dict) -> str | None:
    """One `onboard_footprint`/X result row → the entity id whose pair records it, or None.

    Derived through the same functions that minted the entity ids, so two spellings of one URL
    cannot drift. X matches on the handle instead, in `record_coverage`."""

    stype, url = rec.get("type") or "", rec.get("url") or ""
    if not url:
        return None
    if stype == "blog":
        return derive.blog_entity_id(url)
    if stype == "substack":
        return derive.substack_entity_id(publication_url=url)
    if stype == "github":
        owners = pairs.github_owners_from_links([url])
        return f"github:{owners[0]}" if owners else None
    return None


def record_coverage(conn, oracle_id: str, results: list, *, x_since, web_since,
                     scholar_since=None) -> None:
    """Stamp the refresh registry for every source that actually returned on this run.

    A pull that did not happen writes nothing. `covered_from` is the window the run reached: a
    source that measured its own reach (the X timeline) says so on its record; the rest fall back
    to the window the caller asked for. GitHub records none here; its sweep stores its own
    frontier through `save_pull`."""

    # The head, not the caller's id: `seed_from_entities` registers the row under it, and a
    # resolve can move it just before this runs.

    oracle_id = schema.current_canonical(conn, oracle_id)
    by_entity = pairs.pairs_by_entity(conn, oracle_id)
    _, who_ids = pairs.pairs_for_oracle(conn, oracle_id)
    # X pairs key on the bare handle, and nothing lowercases it on the way in, so fold here.
    x_pairs = {key.lower(): key for etype, key in by_entity.values() if etype == "x"}
    stored = {(r.source_type, r.source_key): r
              for r in registry.list_sources(conn, canonical_ids=[oracle_id])}
    since_for_type = {"x": x_since, "blog": web_since, "substack": web_since, "github": None,
                      "openalex": scholar_since}

    for rec in results:
        stype = rec.get("type") or ""
        if stype == "openalex":
            # Keyed on the bare author id, which is both the entity suffix and the pair key —
            # so the pair is found directly rather than through `result_entity_id`, which reads
            # a url shape this record does not have.
            aid = (rec.get("url") or "").rstrip("/").rsplit("/", 1)[-1]
            pair = ("openalex", aid) if aid else None
        elif stype == "x":
            handle = (rec.get("url") or "").rstrip("/").rsplit("/", 1)[-1].lstrip("@").lower()
            key = x_pairs.get(handle)
            pair = ("x", key) if key else None
        else:
            pair = by_entity.get(result_entity_id(rec) or "")
        if not pair:
            # No registered pair — an affiliation, an unsupported type, or an entity `resolve` has
            # not folded in yet. Recording nothing means the pair is re-pulled once it appears,
            # which is the safe direction.
            continue
        # The ask is recorded before the outcome gate, so a deferred pull's window outlives the
        # run. Ask only, no status or stamp: a deferred pair must still read as unpulled.
        asked = since_for_type.get(pair[0])
        if asked:
            registry.record_request(conn, registry.SourceRow(oracle_id, pair[0], pair[1]),
                              asked.isoformat())
        if rec.get("action") != _OBSERVED:
            # Deferred / blocked / errored: the ask above is durable, the outcome below is not
            # ours to write. Nothing marks it pulled, so it stays due.
            continue
        # What the pull reached when it measured that itself, else the window it was asked for.
        # Presence of the key is the test, never truthiness: `covered_from` only ever widens.
        if "covered_from" in rec:
            reached = rec["covered_from"]
        else:
            since = since_for_type.get(pair[0])
            reached = since.isoformat() if since else None
        # Connectivity: an X walk cut short at R on a pair last pulled before R has not read
        # `[last_pulled_at, R)`, so it claims nothing and does not stamp.
        stamp = True
        prior = stored.get(pair)
        if pair[0] == "x" and windows.leaves_a_gap(started_at_top=True, partial=bool(rec.get("partial")),
                                           reached=reached,
                                           last_pulled_at=prior.last_pulled_at if prior else None):
            stamp, reached = False, None
        # `requested_from` is not passed: `record_request` above is the one owner of the ask.
        registry.record_pull(conn, registry.SourceRow(oracle_id, pair[0], pair[1]),
                       last_status=rec["action"], stamp=stamp,
                       cursor_ts=registry.latest_atom_ts(conn, pair[0], who_ids),
                       covered_from=reached)


def save_pull(conn, row: registry.SourceRow, summary, outcome: str, note, *, since,
              stamp_at: datetime | None = None, hold_stamp: bool = False) -> dict:
    """Persist what one pull of `row` returned, for every outcome that is not a failure: a URL
    that names nothing to walk (`skipped`), and `ingested` / `empty`. Failures (`blocked`, `error`) belong to the
    caller's retry policy and never reach here, so they move no cursor, stamp or frontier.

    What each outcome persists:
      skipped        — the URL named nothing to walk: a real, cheap observation, stamped so the
                       TTL paces the re-check.
      ingested/empty — a real observation: advance the cursor, widen `covered_from` to the window
                       this pull reached, stamp `last_pulled_at`, TTL restarts. The stamp is
                       skipped for a walk that did not start at the top, and both are withheld by
                       `leaves_a_gap`.

    `stamp_at` bounds the stamp from above: a walk bounded by `before` never read the top, but if
    it finished, the forward window may move up to exactly there. A walk that did not finish
    stamps nothing. `hold_stamp` withholds the stamp whatever the rules say, for a walk its caller
    knows is not finished (a capped GitHub sweep, continued on the next turn)."""
    if outcome == "skipped":
        registry.record_pull(conn, row, last_status="skipped", stamp=True, now=registry._now())
        return {"status": "skipped", **_ident(row), "reason": note}

    added = int(summary.get("added") or 0) if isinstance(summary, dict) else 0
    status = "ingested" if added > 0 else "empty"
    _, who_ids = pairs.pairs_for_oracle(conn, row.canonical_id)
    cursor = registry.latest_atom_ts(conn, row.source_type, who_ids) or row.cursor_ts
    # An adapter that measured its own reach wins. Presence, not truthiness: `covered_from: None`
    # means it read nothing, and must not fall back to the asked window.
    if isinstance(summary, dict) and "covered_from" in summary:
        reached = summary["covered_from"]
    else:
        reached = since.isoformat() if since else None
    # Only a walk that started at the top stamps `last_pulled_at`, the forward window. Non-X
    # adapters do not say, and read as top-started.
    top = summary.get("started_at_top", True) if isinstance(summary, dict) else True
    stamp = bool(top)
    stamped_at = registry._now()
    if stamp_at is not None:
        finished = not (isinstance(summary, dict) and summary.get("partial"))
        stamp = finished
        stamped_at = min(stamp_at, utc_now()).isoformat()
    if windows.leaves_a_gap(started_at_top=top,
                    partial=bool(isinstance(summary, dict) and summary.get("partial")),
                    reached=reached, last_pulled_at=row.last_pulled_at):
        log(f"[oracle-refresh] {row.source_type}:{row.source_key}: a top-started walk stopped at "
            f"{reached}, newer than the last pull ({row.last_pulled_at}) — claiming nothing and "
            f"not stamping, so the gap between them is still owed.")
        stamp, reached = False, None
    if hold_stamp:
        stamp = False
    registry.record_pull(conn, row, last_status=status, cursor_ts=cursor, covered_from=reached,
                   stamp=stamp, now=stamped_at)
    # The adapter's own counters ride along, so avoided work (and a regression in it) is visible.
    return {"status": status, **_ident(row), "new_atoms": added,
            "cursor_ts": cursor, "stats": ingest_common.run_stats(summary)}


def _ident(row: registry.SourceRow) -> dict:
    return {"canonical_id": row.canonical_id, "name": row.name,
            "source_type": row.source_type, "source_key": row.source_key}


def take_over(conn, source_type: str, url: str) -> int:
    """Make what OPYT stored from this source, because an Oracle linked it, the Oracle's own:
    `author_referenced` atoms written by the source's entity become `oracle-footprint`. Returns
    how many. The caller holds the write transaction that marks the source added; what a user
    saved stays theirs."""
    entity = result_entity_id({"type": source_type, "url": url})
    if not entity:
        return 0
    return schema.claim_referenced(conn, who_id=entity)
