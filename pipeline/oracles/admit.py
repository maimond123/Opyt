"""
pipeline/oracles/admit.py — adding people, and asking for their work: the `confirm`, `ingest`
and `add_oracle` flows.

`confirm` admits screen picks and raw handles to the roster, refusing what cannot be an Oracle
before the first write; `add_oracle` previews or admits one free-form reference. Each flow then
records what its picks are owed (`record`, `asked`) and wakes the scrapers: nothing is pulled in
the call, which returns within a second.

`record`, for each pick:
  • registers the pairs it has now (`seed_from_entities`), so `progress` sees an X Oracle's
    timeline before the scrapers start (they seed again when they do);
  • stores the subject filter on the Oracle's OpenAlex pair, where archives reads it
    (`topics.store_scholar_topics`);
  • asks for no papers from a research venue whose subjects are not chosen: a venue publishes far more than one pull holds, and
    unfiltered it would be its newest 2,000 papers with no sign the rest was cut;
  • turns `since_last` into a check (`checks.request_check`), not an ask: the newest posts at
    your priority, with nothing else pulled again. Whatever else the call names is still asked,
    and someone never asked for and never pulled gets their first ask in full;
  • writes the ask (`record_ask`, which only widens), with the host's URLs and
    `rediscover`, and reopens the milestones the new ask makes owed again;
  then wakes the scrapers once.

`ask_again` is `ingest`: a wider window, subjects, a top-up or a rediscovery for Oracles already
confirmed. No consent is read here, or anywhere in the scrapers: OPYT never asks.

Two entry paths into ONE `oracles` row per confirmed canonical entity:
  • a RANKED pick — a `canonical_id` straight off the SCREEN; already resolved (Stage 3), so
    confirm just writes the row.
  • a RAW handle (free-form floor) — RESOLVE-AT-CONFIRM: mint the per-platform entity, fetch its
    `identity_links`, then recompute `canonical_id` so a pasted handle that cross-links an
    existing entity MERGES. A handle that can't resolve is reported (`unresolved`), never
    crashes confirm, and never writes a half-baked oracle.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.oracles import checks
from pipeline.oracles import identify
from pipeline.oracles import milestones
from pipeline.oracles import pairs
from pipeline.oracles import preview
from pipeline.oracles import roster
from pipeline.oracles import told
from pipeline.oracles import topics
from pipeline.oracles import windows
from pipeline.oracles.asks import ask_for
from pipeline.oracles.asks import ask_for_head
from pipeline.oracles.asks import record_ask
from pipeline.oracles.asks import wake
from pipeline.oracles.reference import match_local_roster
from pipeline.oracles.reference import unsupported_root
from pipeline.oracles.registry import init_state_schema
from pipeline.platforms import connections
from pipeline.timeparse import utc_now
from pipeline.worker import liveness
from pipeline.oracles.sources.timeline.walks import x_pairs
from pipeline.oracles.sources import scraper_tables


def confirm(conn, canonical_ids: list[str] | None = None,
            add_handles: list[str] | None = None) -> dict:
    """Commit the user's Oracle picks. `canonical_ids` are ranked-list picks (resolved already);
    `add_handles` are free-form floor entries (resolved-at-confirm). Idempotent — re-confirming a
    canonical refreshes it without re-adding. Writes only the roster: what the picks are owed is
    the tool's next step (`oracles.admit`). Returns {confirmed, unresolved, unknown,
    total_oracles}:
      • confirmed  — [{canonical_id, name, source, [handle]}] written to `oracles`.
      • unresolved — raw handles a fetch couldn't resolve (report to the user; nothing written).
      • unknown    — canonical_ids with no entity row (a bad/hallucinated id; skipped, not written).

    Whether a pick is one person is the user's call alone; nothing here or in the host refuses
    or questions one.
    """
    confirmed, unresolved, unknown = [], [], []

    for cid in (canonical_ids or []):
        if schema.get_entity(conn, cid) is None:
            unknown.append(cid)                    # guard: never mint an oracle for a phantom id
            continue
        name = roster.name_for(conn, cid)
        roster.upsert_oracle(conn, cid, name=name, source="screen")
        confirmed.append({"canonical_id": cid, "name": name, "source": "screen"})

    for raw in (add_handles or []):
        cid = identify.resolve_handle(conn, raw)
        if not cid:
            unresolved.append(raw)
            continue
        name = roster.name_for(conn, cid)
        roster.upsert_oracle(conn, cid, name=name, source="freeform")
        confirmed.append({"canonical_id": cid, "name": name, "handle": raw, "source": "freeform"})

    return {
        "confirmed": confirmed,
        "unresolved": unresolved,
        "unknown": unknown,
        "total_oracles": len(roster.heads(conn)),
    }


def add_oracle(conn, reference: str, *, confirm: bool = False,
               x_lookback: str | None = None, web_lookback: str | None = None,
               scholar_lookback: str | None = None) -> dict:
    """Atom-native "add a person": admit one Oracle to the roster.

    `reference` is polymorphic — an X @handle, a Substack/blog/site URL, or a canonical_id (Mode C:
    promote a below-the-cut SCREEN pick). Two-phase, gated by `confirm`:
      • confirm=False (default) → RESOLVE-ONLY preview: verify the reference resolves + show who it
        is + which platforms will expand. Writes NOTHING (a hallucinated handle has nothing to
        confirm). Cheap — no discovery. The windows reach only the preview's report, which
        counts the paper corpus with REAL COUNTS when it can, so the choice is made against
        numbers rather than presets.
      • confirm=True → resolve → write the `oracles` row. WRITES the
        roster and nothing else, and walks nothing: what the person is owed is recorded by the
        tool next (`oracles.admit`), and the background scrapers pull it.

    Name resolution ('find Karpathy') is the HOST's job — it turns a name into a @handle/URL and
    calls this; there is no name-search endpoint (fuzzy resolution deferred by design)."""
    reference = (reference or "").strip()
    if not reference:
        return {"error": "add_oracle needs a reference — an X @handle, a Substack/blog URL, or a "
                         "canonical_id from oracle(action='screen')."}

    match = match_local_roster(conn, reference)         # network-free dedup / Mode C
    if not match:
        # Only when the reference is NEW: an already-known person resolves to their real cluster
        # and needs no rooting, whatever URL shape the caller happened to reach for.
        unsupported = unsupported_root(reference)
        if unsupported:
            return {"error": f"cannot add {reference!r} — {unsupported}. Nothing was written."}

    if not confirm:                                      # ── Phase 1: preview (no writes) ──
        return preview.of(conn, reference, match, x_lookback, web_lookback, scholar_lookback)

    # ── Phase 2: the chain ──────────────────────────────────────────────────────
    already = bool(match) and roster.on_roster(conn, match["canonical_id"])

    # 1+2. RESOLVE → canonical_id, then CONFIRM (write the oracles row).
    if match:
        cid, source = match["canonical_id"], "screen"
    else:
        cid = identify.resolve_handle(conn, reference)           # mints + resolves + the freeform vouch
        if not cid:
            return {"error": f"could not resolve {reference!r} — check the handle/URL. "
                             "Nothing was written.", "unresolved": [reference]}
        source = "freeform"

    roster.upsert_oracle(conn, cid, name=(match.get("name") if match else roster.name_for(conn, cid)),
                         source=source)
    return {
        "added": {"canonical_id": schema.current_canonical(conn, cid),
                  "name": match.get("name") if match else roster.name_for(conn, cid),
                  "source": source, "was_already_oracle": already},
    }


def record(conn: sqlite3.Connection, canonical_ids, win: windows.Windows, *,
           scholar_topics: list[str] | None = None,
           rediscover: bool = False, now: datetime | None = None) -> dict:
    """Record what these picks are owed and wake the scrapers (the module docstring). Returns
    `{needs_topics, checked, woken}`: the venues given no paper ask, the picks checked rather
    than asked, and whether the wake was written."""

    init_state_schema(conn)
    scraper_tables.init_schema(conn)                # a tool may be the first writer on a fresh store
    heads = list(dict.fromkeys(schema.current_canonical(conn, c) for c in canonical_ids))
    try:
        pairs.seed_from_entities(conn, canonical_ids=heads, include_x=connections._connected_x())
    except Exception as exc:
        log(f"[asks] could not register pairs now; the scrapers do when they start: "
            f"{type(exc).__name__}: {exc}")

    needs_topics, checked, recorded = [], [], False
    for cid in heads:
        if scholar_topics is not None:
            topics.store_scholar_topics(conn, cid, scholar_topics)
        venue = topics.is_venue_id(cid) and not topics.has_topic_filter(conn, cid)
        if venue:
            needs_topics.append(cid)
        # Under the ask already on record for this Oracle, whatever anchor it was written with.
        prior = ask_for_head(conn, cid)
        ask = win
        if win.check:
            if checks.request_check(conn, cid, now=now):
                checked.append(cid)
            if prior is None and not any(r.last_pulled_at for r in x_pairs(conn, cid)):
                # a top-up of someone never pulled is their first ask
                ask = windows.parse(now=now or utc_now())
        if not (ask.asks or rediscover):
            continue
        key = prior.canonical_id if prior is not None else cid
        record_ask(conn, key, x_from=ask.x_from, web_from=ask.web_from,
                         scholar_from=None if venue else ask.scholar_from,
                         rediscover=rediscover, now=now)
        widened = prior is None or prior.x_from != ask_for(conn, key).x_from
        told.reopen(conn, key, ["archives", "stuck", *(["reach"] if widened else [])])
        recorded = True
    return {"needs_topics": needs_topics, "checked": checked,
            "woken": recorded and wake()}


def confirm_picks(conn, *, canonical_ids, add_handles, x_lookback=None, web_lookback=None,
             scholar_lookback=None, scholar_topics=None) -> dict:
    """action='confirm' — commit the picks, record what each is owed, wake the background
    scrapers. Starts nothing and returns within a second. See `oracle()`."""

    if not canonical_ids and not add_handles:
        return {"error": "confirm needs canonical_ids=[...] (kept picks) and/or "
                         "add_handles=[...] (raw handles to add)."}
    if scholar_topics is not None and len(canonical_ids or []) + len(add_handles or []) != 1:
        return {"error": _ONE_ORACLE_TOPICS}
    try:
        win = windows.parse(x_lookback, web_lookback, scholar_lookback, now=utc_now())
    except ValueError as e:
        return {"error": f"{e}. Nothing was written."}
    out = confirm(conn, canonical_ids=canonical_ids, add_handles=add_handles)
    if not out["confirmed"]:
        return out
    owed = asked(conn, [e["canonical_id"] for e in out["confirmed"]], win,
                  scholar_topics=scholar_topics)
    venues = set(owed.pop("needs_topics"))
    for entry in out["confirmed"]:
        # A VENUE IS THE ONE ROOT WHOSE DEFAULT IS WRONG. Every other root pulls in full; a venue
        # is 63,565 works (ChemRxiv, measured), so it is confirmed with no paper ask
        # until its subjects are chosen.
        if entry["canonical_id"] in venues:
            who = entry.get("name") or entry.get("handle") or entry["canonical_id"]
            entry["needs_topics"] = True
            entry["note"] = (
                f"{who} is a research VENUE, not a person — it publishes far more than one "
                f"reader follows, so none of its papers are pulled until the user picks the "
                f"subjects they read. Call "
                f"`add_oracle('{entry['canonical_id']}')`: its preview reports the real total and "
                f"the subject list. Then pass the chosen ids as `scholar_topics` to "
                f"`oracle(action='ingest', canonical_ids=['{entry['canonical_id']}'])`.")
    return {**out, **owed}


_ONE_ORACLE_TOPICS = ("scholar_topics narrows ONE Oracle's paper corpus, and this call covers "
                      "several. Pass it with exactly one pick (canonical_ids=[the person or "
                      "venue]).")


def asked(conn, ids: list[str], win, *, scholar_topics=None, rediscover: bool = False) -> dict:
    """Record what these picks are owed and say where each stands. Shared by `confirm`,
    `ingest` and `add_oracle`, which differ only in how they name the picks.

    Returns `{picks, message, needs_topics, [checking]}`. The picks' milestones are stamped as
    shown: the host is holding them, so the rider on the next call does not say them again."""

    rec = record(conn, ids, win, scholar_topics=scholar_topics,
                      rediscover=rediscover)
    picks = milestones.for_picks(conn, ids)
    told.mark_shown(conn, picks)
    out = {"picks": picks, "message": _still_to_come(picks),
           "needs_topics": rec["needs_topics"]}
    if rec["checked"]:
        out["checking"] = [p["name"] for p in picks if p["canonical_id"] in rec["checked"]]
    return out


def _still_to_come(picks: list[dict]) -> str:
    """What to tell the user about what is still to come.

    It promises background work only on the worker's own word:
    `liveness.repair_or_report_cached`, the cached `repair_or_report`, and nothing else. A user told
    their writers are on the way stops waiting for a pull nothing is running. Unknown
    is not yes: it says where to look and promises nothing."""

    owing = [p["name"] for p in picks if milestones.owes(p)]
    if not owing:
        return "Everything asked for these writers is already in."
    names = told.people(owing)
    report = liveness.repair_or_report_cached()
    if report.get("ok") is False:
        return (f"I've asked for {names}'s writing, but nothing is collecting it yet: "
                f"{(report.get('cause') or 'the background helper is not running').rstrip('.')}."
                + (f" {report['step']}" if report.get("step") else ""))
    if report.get("ok"):
        return (f"{names}'s writing is coming in on its own, and nothing waits on this "
                f"conversation. Keep your Mac open and awake while it runs: with the lid closed "
                f"on battery it pauses, and picks up again when the Mac wakes. "
                f"`oracle(action='progress')` shows where each stands.")
    return (f"I've asked for {names}'s writing. `oracle(action='progress')` shows where each "
            f"stands.")


def ask_again(conn, *, canonical_ids, force: bool, x_lookback, web_lookback,
            scholar_lookback=None, scholar_topics=None) -> dict:
    """action='ingest' — ask again for confirmed Oracles: a wider window, subjects, a top-up
    (`since_last`, a check of their newest posts) or a fresh discovery (`force`). Records the ask
    and wakes the background scrapers; starts nothing. See `oracle()`."""

    picks = roster.confirmed_oracles(conn)
    if canonical_ids:
        want = {schema.current_canonical(conn, c) for c in canonical_ids}
        picks = [o for o in picks if o["canonical_id"] in want]
    if not picks:
        return {"error": "no confirmed Oracles to ingest — run action='confirm' first "
                         "(or pass canonical_ids of confirmed Oracles)."}
    # A topic selection names ONE person's subjects, so it may not be sprayed across a batch:
    # `canonical_ids` defaults to every confirmed Oracle, and narrowing all of them to one
    # researcher's topics is a change no later call undoes for the ones it was not meant for.
    if scholar_topics is not None and len(picks) != 1:
        return {"error": _ONE_ORACLE_TOPICS}
    try:
        win = windows.parse(x_lookback, web_lookback, scholar_lookback, now=utc_now())
    except ValueError as e:
        return {"error": f"{e}. Nothing was written."}
    return asked(conn, [o["canonical_id"] for o in picks], win, scholar_topics=scholar_topics,
                  rediscover=force)


def add(conn, reference: str, win: windows.Windows | None, *, confirm: bool,
        x_lookback: str | None = None, web_lookback: str | None = None,
        scholar_lookback: str | None = None, scholar_topics: list[str] | None = None,
        force: bool = False) -> dict:
    """`add_oracle`: preview one reference, or admit it and record what it is owed. `win` is the
    call's parsed windows, read before the store was opened (None on a preview)."""
    out = add_oracle(conn, reference, confirm=confirm, x_lookback=x_lookback,
                     web_lookback=web_lookback, scholar_lookback=scholar_lookback)
    if confirm and "added" in out:
        owed = asked(conn, [out["added"]["canonical_id"]], win, scholar_topics=scholar_topics,
                     rediscover=force)
        owed.pop("needs_topics")
        out.update(owed)
    return out
