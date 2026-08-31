"""
pipeline/kb/frontier_queries.py — the store behind Frontier's standing queries.

Frontier stage 1 turns the KB into standing queries; this module is where they LIVE. It is
deliberately only a store: no LLM, no network, no policy about what a good query is. The
generator (`sitting_reader.py`) decides what to emit, and this decides what that means for the
rows already there.

Two contracts stage 2 depends on:

  • A re-emitted query lands on the SAME row. Identity is `normalized` (lowercase +
    whitespace-collapsed), hashed into `query_id`, so a re-spawned twin can't reset its own
    watermark and re-pull the same window forever.

  • The system never removes a query. A reader that explicitly says a thread is done bumps
    `miss_count`, and stage 2 reads that counter as a SPEED: 0-2 drops runs daily, 3-9 weekly,
    10+ monthly and never slower. Only David retires one, by hand, through `retire_query`.
    `miss_count` moves only on an explicit verdict, never on silence

Verdicts are scoped per generator via `frontier_query_generators`, one row per (query,
generator) CLAIM — not the `generator` column on `frontier_queries`, which freezes at insert as
ORIGIN only. `miss_count` — the SPEED — lives on the claim: "how fast does this generator want
this query run" is a fact about the pair, and a query's own speed is the MIN over its claims, so
it runs at the pace of its most engaged asker. Full history in the companion doc above.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pipeline.timeparse import utc_iso

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """The identity form: case- and whitespace-insensitive. `"Muon  Optimizer"` and
    `"muon optimizer"` are the same standing query and must not both execute."""
    return _WS.sub(" ", (text or "").strip().lower())


def query_id_for(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


# The two emission lanes. `LANE_MACHINE` means every atom the query cites was found by the
# crawler; anything with one human-attested citation is human (mixed counts as human — RULED
# 2026-08-25). Values, not the entry_mode names, because a lane is a property of the QUERY.
LANE_HUMAN = "human"
LANE_MACHINE = "machine"

# How many standing machine-lane queries ONE generator may hold at once. RULED 2026-08-25 at K=3.
# It governs question-list ownership, not money: watermarked pulls are near-free. K=0 was rejected
# precisely — it loses TRACKING of machine-found threads without buying any visibility back, since
# the region union puts those atoms in every sitting regardless. Raise it on measured promotion
# events from machine-lane pulls; drop it to 0 if those pulls are never engaged.
MACHINE_LANE_QUOTA = 3


def machine_lane_claims(conn: sqlite3.Connection, generator: str) -> set[str]:
    """The `query_id`s of every ACTIVE machine-lane query this generator claims.

    Counts CLAIMS, via `frontier_query_generators`, never the `generator` column on the query row
    — that column is ORIGIN, frozen at first insert, so a query this region emitted but a sibling
    happened to say first would be invisible here and the region would silently run over quota.
    Same join `active_queries` uses, and `status != 'retired'` for the same reason: a hand-retired
    query is gone and must not hold a slot.
    """
    return {r[0] for r in conn.execute(
        "SELECT g.query_id FROM frontier_query_generators g "
        "  JOIN frontier_queries q ON q.query_id = g.query_id "
        " WHERE g.generator = ? AND q.status != 'retired' AND q.lane = ?",
        (generator, LANE_MACHINE))}


def upsert_queries(conn: sqlite3.Connection, queries: list[dict], *, generator: str,
                   label: str | None = None, votable: bool = True,
                   now: str | None = None) -> dict:
    """Write this run's NEW queries, returning `{new, refreshed, query_ids, new_texts,
    refreshed_texts}`.

    The two text lists are the same split the two counts already report. They exist because the
    watchlist diff a user is shown after their own read has to NAME what changed — a count says
    three questions are new and leaves them unable to judge or drop any of them.

    A first sighting inserts; `ON CONFLICT` is a collision guard, not the re-emission path it
    used to be — a "new" query can still normalize onto an existing row, and must land on it
    rather than raise (confirming a known query is `apply_verdicts`' job). A collision bumps
    `emit_count` and `last_emitted_at`; `created_at` is never touched.

    It no longer writes `miss_count` directly — that write raced `_sync_speed`'s votable-only
    MIN and could silently reset a dropped query back to daily. `_sync_speed` is now the only
    writer.

    `label` and `votable` describe the GENERATOR and are forwarded once to `register_generator`.
    `status` is never touched, so re-emitting a hand-retired query's text does not resurrect it.
    The descriptive fields (`text`, `rationale`, `target_sources`, `source_atom_ids`) are
    OVERWRITTEN with the latest emission — `created_at` alone preserves the origin date.

    `lane` (from `q["lane"]`, defaulting to `LANE_HUMAN`) is the ONE column that is not
    last-writer-wins: it is sticky to `LANE_HUMAN` in the conflict arm. The classification itself
    is the caller's — this only guarantees the invariant, because a flapping lane makes the quota
    count nondeterministic and lets one query oscillate in and out of the clamp forever.
    """
    stamp = now or utc_iso()
    register_generator(conn, generator, label=label, votable=votable, now=stamp)
    new = refreshed = 0
    ids: list[str] = []
    new_texts: list[str] = []
    refreshed_texts: list[str] = []
    # Collapse intra-run duplicates FIRST — two queries differing only by case would otherwise
    # hit the upsert twice and look re-confirmed by a run that saw it once. Last occurrence wins.
    deduped: dict[str, dict] = {}
    for q in queries:
        norm = normalize(q.get("text", ""))
        if norm:
            deduped[norm] = q
    for norm, q in deduped.items():
        qid = query_id_for(norm)
        ids.append(qid)
        row = (qid, q.get("text", "").strip(), norm, generator,
               q.get("rationale"),
               json.dumps(q.get("target_sources") or []),
               json.dumps(q.get("atom_ids") or []),
               stamp, stamp, q.get("lane") or LANE_HUMAN)
        conn.execute(
            """INSERT INTO frontier_queries
                 (query_id, text, normalized, generator, rationale, target_sources,
                  source_atom_ids, created_at, last_emitted_at, lane)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(normalized) DO UPDATE SET
                 text            = excluded.text,
                 rationale       = excluded.rationale,
                 target_sources  = excluded.target_sources,
                 source_atom_ids = excluded.source_atom_ids,
                 emit_count      = frontier_queries.emit_count + 1,
                 last_emitted_at = excluded.last_emitted_at,
                 lane            = CASE WHEN frontier_queries.lane IS NULL
                                          OR frontier_queries.lane = ?
                                        THEN frontier_queries.lane ELSE excluded.lane END""",
            row + (LANE_HUMAN,))
        # This generator's CLAIM, recorded even on a collision — that's the record that a second
        # generator now also wants a query the first one owns.
        conn.execute(
            """INSERT INTO frontier_query_generators
                 (query_id, generator, first_emitted_at, last_emitted_at, miss_count)
               VALUES (?,?,?,?,0)
               ON CONFLICT(query_id, generator) DO UPDATE SET
                 last_emitted_at = excluded.last_emitted_at,
                 miss_count      = 0""",
            (qid, generator, stamp, stamp))
        # The claim moved, so the projection must follow — the only writer of the query's speed now.
        _sync_speed(conn, qid)
        # `rowcount`'s INSERT-vs-UPDATE distinction is an SQLite upsert detail, not worth leaning
        # on. Ask the row instead:
        # a first sighting is the only way to hold emit_count == 1.
        seen = conn.execute("SELECT emit_count FROM frontier_queries WHERE query_id=?",
                            (qid,)).fetchone()[0]
        if seen == 1:
            new += 1
            new_texts.append(q.get("text", "").strip())
        else:
            refreshed += 1
            refreshed_texts.append(q.get("text", "").strip())
    conn.commit()
    return {"new": new, "refreshed": refreshed, "query_ids": ids,
            "new_texts": new_texts, "refreshed_texts": refreshed_texts}


def apply_verdicts(conn: sqlite3.Connection, verdicts: list[dict], *, generator: str,
                   now: str | None = None) -> dict:
    """Move the counters the reader's verdicts ask for; return `{kept, dropped, unmatched}`.

    `keep` zeroes `miss_count` and bumps `emit_count`/`last_emitted_at`; `drop` bumps
    `miss_count`, which stage 2 reads as "run this one slower". A query with no verdict is not
    touched — silence carries no meaning here.

    Fail-safe: an uncited `keep` is still honoured (the caller only flags it) since punishing a
    thread for a missing atom_ids field would retire it on a formatting slip; `source_atom_ids`
    is refreshed only when the verdict actually cites something, so staleness stays visible
    rather than silently blanked; and anything not exactly `drop` counts as a keep. `status` is
    never written here — retirement is a human act.

    The counter moves on the claim, and the query's own `miss_count` is re-derived as the MIN
    across its claims, so one generator's `drop` can't slow a query another generator just kept
    (or vice versa).
    """
    stamp = now or utc_iso()
    kept = dropped = unmatched = 0
    for v in verdicts or []:
        norm = normalize((v or {}).get("text", ""))
        qid = query_id_for(norm) if norm else None
        # Scoped by CLAIM, not `frontier_queries.generator` — that column got overwritten by the
        # upsert, silently discarding one generator's verdicts once a second one emitted the text.
        row = conn.execute(
            "SELECT 1 FROM frontier_query_generators WHERE query_id=? AND generator=?",
            (qid, generator)).fetchone() if qid else None
        if row is None:
            unmatched += 1
            continue
        if v.get("verdict") == "drop":
            conn.execute("UPDATE frontier_query_generators SET miss_count = miss_count + 1 "
                         " WHERE query_id=? AND generator=?", (qid, generator))
            _sync_speed(conn, qid)
            dropped += 1
            continue
        conn.execute("UPDATE frontier_query_generators SET miss_count=0, last_emitted_at=? "
                     " WHERE query_id=? AND generator=?", (stamp, qid, generator))
        cited = [str(a).strip() for a in (v.get("atom_ids") or []) if str(a).strip()]
        if cited:
            conn.execute(
                "UPDATE frontier_queries SET emit_count = emit_count + 1,"
                "  last_emitted_at=?, source_atom_ids=? WHERE query_id=?",
                (stamp, json.dumps(cited), qid))
        else:
            conn.execute(
                "UPDATE frontier_queries SET emit_count = emit_count + 1,"
                "  last_emitted_at=? WHERE query_id=?", (stamp, qid))
        _sync_speed(conn, qid)
        kept += 1
    conn.commit()
    return {"kept": kept, "dropped": dropped, "unmatched": unmatched}


def _sync_speed(conn: sqlite3.Connection, query_id: str) -> None:
    """Re-derive a query's `miss_count` — its stage-2 SPEED — as the MIN across its VOTABLE claims.

    A projection, not the only copy — `frontier_execute.tier_for` and every read take the counter
    off the query row, so this must run after EVERY claim-counter write.

    Votable-only (since 2026-08-12): a claim that can never change its mind stays at
    `miss_count=0` forever, and including it in the MIN would pin a query to the fastest tier
    permanently regardless of other claims' drops. The fallback is 0 (daily) when a query has no
    votable claim at all — the fail-safe direction is over-pulling, not under-pulling. See
    """
    conn.execute(
        """UPDATE frontier_queries SET
             miss_count = COALESCE((
                 SELECT MIN(g.miss_count) FROM frontier_query_generators g
                   LEFT JOIN frontier_generators fg ON fg.generator = g.generator
                  WHERE g.query_id = frontier_queries.query_id
                    AND COALESCE(fg.votable, 1) = 1), 0)
           WHERE query_id = ?""", (query_id,))


def register_generator(conn: sqlite3.Connection, generator: str, *, label: str | None = None,
                       votable: bool = True, now: str | None = None) -> None:
    """Record a channel, idempotently. Called by `upsert_queries`, so a generator exists in the
    registry from its first emission and nothing has to remember to register.

    `votable` and `label` are declared by the READER and refreshed on re-registration, so a fixed
    flag takes effect going forward — but not retroactively: it does not re-run `_sync_speed` for
    queries already claimed, which lags rather than loses. `status` is never touched here;
    retirement is a human decision an automatic path must not undo
    """
    stamp = now or utc_iso()
    conn.execute(
        """INSERT INTO frontier_generators (generator, label, votable, created_at, last_seen_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(generator) DO UPDATE SET
             label        = COALESCE(excluded.label, frontier_generators.label),
             votable      = excluded.votable,
             last_seen_at = excluded.last_seen_at""",
        (generator, label, 1 if votable else 0, stamp, stamp))


# ── Manual retirement — the ONLY way a query stops running ──────────────────────
def retire_query(conn: sqlite3.Connection, text: str) -> bool:
    """Stop executing a query for good. True when a row changed.

    Nothing automatic may call this (a test asserts it) — the decay tiers exist so no machine
    path ever needs to, since a quiet thread just slows to a cheap monthly floor instead.
    Removal is a judgement about David's attention, so David makes it.
    """
    cur = conn.execute("UPDATE frontier_queries SET status='retired' WHERE normalized=?",
                       (normalize(text),))
    conn.commit()
    return bool(cur.rowcount)


def unretire_query(conn: sqlite3.Connection, text: str) -> bool:
    """Put a hand-retired query back into execution. True when a row changed."""
    cur = conn.execute("UPDATE frontier_queries SET status='active' WHERE normalized=?",
                       (normalize(text),))
    conn.commit()
    return bool(cur.rowcount)


def retire_generator(conn: sqlite3.Connection, generator: str) -> dict:
    """Kill a whole CHANNEL. Returns `{generator_retired, queries_retired}`.

    The per-region answer to "that turned out to be a dead end" — retiring 10-25 queries one
    exact string at a time through `retire_query` is the chore that doesn't get done. Only
    queries whose LAST live claimant this was are retired; one a second live channel still wants
    keeps running. A cascade at retire time, not a filter inside `active_queries` — that list is
    both what stage 2 executes and what a reader is shown, so retirement stays a readable write
    rather than an invisible filter. `status` is only ever set here or in `retire_query`, both
    human acts.
    """
    cur = conn.execute("UPDATE frontier_generators SET status='retired' WHERE generator=?",
                       (generator,))
    gen_hit = bool(cur.rowcount)
    q = conn.execute(
        """UPDATE frontier_queries SET status='retired'
            WHERE status != 'retired'
              AND query_id IN (SELECT g.query_id FROM frontier_query_generators g
                                WHERE g.generator = ?)
              AND query_id NOT IN (
                    SELECT g2.query_id FROM frontier_query_generators g2
                      LEFT JOIN frontier_generators fg2 ON fg2.generator = g2.generator
                     WHERE COALESCE(fg2.status, 'active') != 'retired')""",
        (generator,))
    conn.commit()
    return {"generator_retired": gen_hit, "queries_retired": q.rowcount}


def generators(conn: sqlite3.Connection, *, include_retired: bool = True) -> list[sqlite3.Row]:
    """Every channel, with its label, votability, status and how many queries it still claims.

    The answer to "which queries came from where" at channel granularity — the origin column on
    `frontier_queries` answers it for ONE query, and answers only "who emitted it first".
    """
    sql = ("SELECT fg.*, ("
           "  SELECT COUNT(*) FROM frontier_query_generators g WHERE g.generator = fg.generator"
           ") AS claims FROM frontier_generators fg")
    if not include_retired:
        sql += " WHERE fg.status != 'retired'"
    return list(conn.execute(sql + " ORDER BY fg.last_seen_at DESC"))


def record_run(conn: sqlite3.Connection, **fields) -> int:
    """Append a run row (ok | skipped | failed) and return its `run_id`.

    Every outcome is recorded, including the boring ones, because the TRIGGER reads this table:
    "new saves since the last ok run" is meaningless without a durable mark of when that was.
    """
    fields.setdefault("ran_at", utc_iso())
    cols = ",".join(fields)
    marks = ",".join("?" * len(fields))
    cur = conn.execute(f"INSERT INTO frontier_reader_runs ({cols}) VALUES ({marks})",
                       tuple(fields.values()))
    conn.commit()
    return int(cur.lastrowid)


def active_queries(conn: sqlite3.Connection, *, generator: str | None = None) -> list[sqlite3.Row]:
    """Every executable query — i.e. everything David has not hand-retired.

    `status != 'retired'` rather than `status = 'active'`: this list is both what stage 2
    executes and what the reader is shown, so anything filtered out here can never be verdicted
    or revived — a slowed-down query must stay on it; only a hand-retired one leaves. `!=` is
    also fail-safe for an unexpected/legacy status value.

    Scoped to `generator`, this reads CLAIMS (every query this generator asks for), not the
    origin column — a query a sibling region emitted later must still show up here, or the
    reader re-invents its wording and orphans stage 2's watermark
    """
    if generator:
        # `q.*`, not `*`: the join would otherwise append the claim's own `miss_count` and
        # `last_emitted_at` and shadow the query's when a row is read by name.
        sql = ("SELECT q.* FROM frontier_queries q "
               "  JOIN frontier_query_generators g ON g.query_id = q.query_id "
               " WHERE q.status != 'retired' AND g.generator=? "
               " ORDER BY q.last_emitted_at DESC")
        args: tuple = (generator,)
    else:
        sql = ("SELECT * FROM frontier_queries WHERE status != 'retired' "
               " ORDER BY last_emitted_at DESC")
        args = ()
    return list(conn.execute(sql, args))


# The generator a query the USER typed is stamped with. Its whole job is exemption: a
# user-authored query retires only by user action, never by any decay path — which is the pin
# reborn as the obvious semantics of an add button, and what stops the watchlist review from
# thrashing what the user just kept.
USER_GENERATOR = "user"

# Human words for the three decay tiers. `tier_for` returns a TTL multiplier in days; a number of
# days is not what a person asked "how often is this running" wants to hear.
_SPEED = {1.0: "daily", 7.0: "weekly", 30.0: "monthly"}


def retired_texts(conn: sqlite3.Connection, *, generator: str) -> list[str]:
    """Questions this generator claims that a HUMAN has retired. The third bucket of the watchlist
    diff, and the only one no machine path can fill: drops only SLOW a query (the decay tiers exist
    so nothing ever needs to remove one), so a query leaving the list is always a person's doing.

    Always per-generator: the diff this feeds is a region's, and a corpus-wide retired list would
    report another region's drops as this one's."""
    rows = conn.execute(
        "SELECT q.text FROM frontier_queries q "
        "  JOIN frontier_query_generators g ON g.query_id = q.query_id "
        " WHERE q.status = 'retired' AND g.generator = ? ORDER BY q.last_emitted_at DESC",
        (generator,))
    return [r[0] for r in rows]


def watchlist(conn: sqlite3.Connection, *, generator: str | None = None) -> list[dict]:
    """The standing queries as a person reads them: `[{text, speed, emitted, last_emitted_at,
    first_seen, source, searching}, ...]`, fastest first.

    A LIST SURFACE and nothing else — it writes nothing, decides nothing, and retires nothing.
    Today the user sees only a query COUNT in frontier status; the list itself has never had a
    door.

    THE `lane` COLUMN IS DELIBERATELY ABSENT, along with every word that would leak it. The quota
    is enforcement-internal bookkeeping: telling a person that three of their watched questions are
    "machine lane" requires teaching the whole entry_mode taxonomy to explain a distinction that
    changes nothing they can act on. Same boundary the promotion side keeps.

    `source` says whether the user typed the question or a read of their material proposed it —
    which they CAN act on, because only the first kind is exempt from decay.

    `searching` is the stored `target_sources`, renamed on the way out for the same reason `speed`
    and `source` are renamed: this dict already spends the word "source" on WHO PROPOSED the query,
    and a second key spelling it would read as the same fact twice. A route the user disagrees with
    is actionable, so unlike `lane` it belongs here.
    """
    from .frontier_execute import tier_for
    rows = active_queries(conn, generator=generator)
    out = []
    for r in rows:
        mult = tier_for(r["miss_count"])
        out.append({
            "text": r["text"],
            "speed": _SPEED.get(mult, f"every {int(mult)} days"),
            "emitted": r["emit_count"],
            "last_emitted_at": r["last_emitted_at"],
            "first_seen": r["created_at"],
            "source": "you" if r["generator"] == USER_GENERATOR else "a read of your material",
            "searching": json.loads(r["target_sources"] or "[]"),
            "_mult": mult,
        })
    # Sorted on the TTL MULTIPLIER, not on the position of a label in a hardcoded list: `_SPEED` is
    # a display map and a new tier added to `DECAY_TIERS` must fall into place here on its own.
    out.sort(key=lambda x: (x.pop("_mult"), x["last_emitted_at"]))
    return out


# Where a user-typed question runs. The general-purpose ones, not every adapter: the domain feeds
# (biorxiv, pubmed, clinicaltrials, sec_edgar) answer a question that was asked in their domain,
# and pointing a generic phrase at them buys noise a person then has to dismiss.
#
# openalex belongs here and is not a domain feed — it indexes published literature across every
# discipline, so it is the only entry that answers a user's question when the question leaves
# computer science. Without it, a typed question about biology or economics runs against arxiv and
# github, neither of which indexes the field it was asked about.
USER_QUERY_SOURCES: tuple[str, ...] = ("arxiv", "github", "semantic_scholar", "hackernews",
                                       "openalex")


def add_user_query(conn: sqlite3.Connection, text: str) -> str | None:
    """Put a question the USER typed onto the watchlist. Returns its `query_id`, or None if the
    text was empty.

    `votable=False` is load-bearing, not a default copied from somewhere. `_sync_speed` takes the
    MIN `miss_count` over VOTABLE claims only, and nothing ever renders a verdict on a
    user-authored query — so a votable user claim would sit at miss_count 0 forever, pin every
    query it touches to the daily tier, and erase decay for the whole set through one shared row.
    Non-votable, the user's claim records ownership and abstains from the speed vote.
    """
    text = (text or "").strip()
    if not text:
        return None
    res = upsert_queries(
        conn, [{"text": text, "rationale": "added by the user",
                "target_sources": list(USER_QUERY_SOURCES), "atom_ids": []}],
        generator=USER_GENERATOR, votable=False, label="your own watchlist")
    return res["query_ids"][0] if res["query_ids"] else None


def main(argv: list[str] | None = None) -> int:
    """The hand-retirement CLI. This is the only door out of the query set."""
    from . import schema

    ap = argparse.ArgumentParser(description="Frontier standing queries — list and hand-retire")
    ap.add_argument("--retire", metavar="TEXT", help="stop executing this query for good")
    ap.add_argument("--unretire", metavar="TEXT", help="put a retired query back into execution")
    ap.add_argument("--list", action="store_true", help="print every query with its state")
    ap.add_argument("--retire-generator", metavar="GEN",
                    help="kill a whole channel ('bookmark-reader' | 'sitting:<slug>'); retires "
                         "only the queries whose last LIVE claimant it was")
    ap.add_argument("--generators", action="store_true",
                    help="print every channel with its label, votability and claim count")
    args = ap.parse_args(argv)

    conn = schema.connect()
    try:
        if args.retire_generator:
            out = retire_generator(conn, args.retire_generator)
            print(json.dumps({"action": "retire-generator",
                              "generator": args.retire_generator, **out,
                              "note": None if out["generator_retired"]
                              else "no such generator — nothing was registered under that name"},
                             indent=2))
            return 0 if out["generator_retired"] else 1
        if args.generators:
            print(json.dumps([dict(r) for r in generators(conn)], indent=2, default=str))
            return 0
        if args.retire or args.unretire:
            text = args.retire or args.unretire
            act = retire_query if args.retire else unretire_query
            ok = act(conn, text)
            print(json.dumps({"action": "retire" if args.retire else "unretire",
                              "text": text, "changed": ok,
                              "note": None if ok else "no query with that text"}, indent=2))
            return 0 if ok else 1
        rows = conn.execute(
            "SELECT text, generator, status, miss_count, emit_count, last_emitted_at "
            "  FROM frontier_queries ORDER BY status, miss_count DESC, text").fetchall()
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
