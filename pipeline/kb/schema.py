"""
pipeline/kb/schema.py — the atom-KB substrate (the "trusted router" data layer).

The DDL for the atom-KB substrate (atoms/chunks/entities and the
Frontier + sitting tables below them) in the one machine-canonical DB (`~/.opyt/opyt.db`).
`kb_meta` (the embedding-identity guard) is owned by `pipeline/kb/embed.py`, NOT redefined
here. (Earlier versions of this store also held `radar_atoms` and `notes`; both were
deleted in prior migrations and no longer exist.)

The model, in one breath: an **atom** is a THIN routing card — a pointer to a live
source (`source_url`) + a stored raw snapshot (`raw_ref`/`raw_hash`) + a mechanical
`description`. It never carries interpretation. The searchable surface is `chunks`
(BM25 via `chunks_fts` + a per-chunk `vector`); at query time the host follows the
pointer and injects the REAL raw text. **Trust invariant: nothing asserts what a
source *says* without opening its raw.** Relations between atoms are computed at
query time by the host from the raw it opens, never stored — an `edges` table that
stored factual ones was deleted 2026-08-23, having accumulated 5,776 rows and no
reader (docs/plans/2026-08-23-delete-edges-and-trust-tiers.md).

Design invariants this file enforces:
  • IDENTITY = `atom_id` (`"x:{tweet_id}"` | `"github:{owner}/{name}"`). Re-ingest
    UPSERTs in place; `raw_hash = sha256(snapshot)` is the change-DETECTOR + idempotency
    key, not the identity.
  • Separate subspace — the KB's chunk vectors live in `chunks.vector` (hosted model,
    dim in `kb_meta`) and are comparable ONLY with vectors from that same model, guarded by
    `kb_meta`/`assert_model` (see embed.py). This was written against the 384-d bge-small
    `note_embeddings`, dropped 2026-08-05; `chunks.vector` is now the only vector store in
    the DB, which makes the invariant easier to hold and no less load-bearing.
  • Connection = WAL + busy_timeout + idempotent `CREATE TABLE IF NOT EXISTS` on every
    writable open (the repo norm — no migrations dir). Honors `$OPYT_HOME` via opyt_db(),
    so a test sandboxes the whole store with one env var.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db

# ── DDL ───────────────────────────────────────────────────────────────────────
# `when` is a SQL keyword → the temporal column is `when_ts`. JSON-array/JSON fields
# (`about_entities`/`payload`/`identity_links`) are stored as TEXT and read back with
# json.loads — a zero-migration escape hatch for anything a later tool wants to filter
# on (promote a blob field to a column + backfill, no schema break).
_DDL = """
CREATE TABLE IF NOT EXISTS atoms (
  atom_id        TEXT PRIMARY KEY,
  source_type    TEXT NOT NULL,
  what_kind      TEXT,
  who_id         TEXT,
  when_ts        TEXT,
  when_precision TEXT,
  about_entities TEXT,                 -- JSON array of entity refs
  source_url     TEXT,
  raw_ref        TEXT,                 -- path to the snapshot, RELATIVE to opyt_home()
  raw_hash       TEXT,                 -- sha256(snapshot) : idempotency + change key
  description    TEXT,                 -- MECHANICAL only (never interpretive)
  payload        TEXT,                 -- JSON of structural fields (counts, lang, …)
  entry_mode     TEXT,                 -- HOW the atom entered, NOT what it is (that's source_type).
                                       -- Live: 'user-saved' (the user personally saved it) |
                                       -- 'oracle-footprint' (a tracked person authored it) |
                                       -- 'author_referenced' (a tracked person pointed at it).
                                       -- RETIRED: 'crawled' (a standing artifact sweep). No live
                                       -- writer produces it; `_rename_crawled_to_footprint` below
                                       -- heals any row that still carries it. Documented here
                                       -- because stores written before that migration hold it.
                                       -- Free text, no CHECK — so
                                       -- readers MUST allow-list the modes they want, never
                                       -- deny-list. Frontier's anti-narrowing invariant depends on
                                       -- exactly that: the generator selects `= 'user-saved'`, so a
                                       -- new mode is excluded by DEFAULT instead of by remembering.
                                       -- The frontier mode (written ONLY by Frontier stage 3, see
                                       -- pipeline/kb/frontier_admit.py) records how an atom was
                                       -- FOUND, never whether a human blessed it — an approved
                                       -- candidate tagged 'user-saved' would re-enter the generator
                                       -- and narrow the loop onto its own output. Approval lives in
                                       -- frontier_candidates.status, never here. It is EXCLUDED
                                       -- from HUMAN_ATTESTED below, and that exclusion is what the
                                       -- anti-narrowing invariant rests on. See
                                       -- docs/plans/2026-08-12-frontier-stage3-admit.md.
  version        INTEGER NOT NULL DEFAULT 1,
  ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),  -- LAST observed at, not first. See below.
  -- When this atom FIRST arrived. Written on INSERT, NEVER on UPDATE — that is the entire point of
  -- the column. `ingested_at` cannot answer this question: `upsert_atom` refreshes it to now on
  -- every re-observation, so a re-scraped 2026-01 save reports today as its arrival date. The two
  -- columns are equal until the first re-observation and diverge permanently after it.
  -- The clock rule this serves: membership uses NO clock · render order uses `when_ts` (when the
  -- source published) · SELECTION uses `first_seen`, never `ingested_at`. Any reader that picks
  -- work by age — the sitting rail's re-read trigger above all — keys on this column.
  -- Deliberately absent from `_ATOM_UPDATABLE`, and `upsert_atom`'s SET clause is built from
  -- that tuple. Adding it there would re-create the exact bug this column exists to fix, and it
  -- would look like a consistency cleanup when someone did it.
  first_seen     TEXT DEFAULT (datetime('now')),
  -- When a machine-lane atom was PROMOTED to a human-attested mode by a human-initiated ingest
  -- touching it again (`ingest_common.promote_atom`). NULL for everything else, which is almost
  -- every row. It exists because `first_seen` cannot answer "when did a person show they cared":
  -- a frontier atom's `first_seen` is the date the crawler found it, months before the human
  -- deposit that promoted it, so the re-read trigger keyed on `first_seen` alone would treat a
  -- brand-new engagement as ancient material and never open the wallet for it. The trigger reads
  -- `COALESCE(promoted_at, first_seen)`. `first_seen` itself is untouched by promotion — the
  -- arrival date is still the arrival date.
  promoted_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_atoms_source_type ON atoms(source_type);
CREATE INDEX IF NOT EXISTS idx_atoms_what_kind   ON atoms(what_kind);
CREATE INDEX IF NOT EXISTS idx_atoms_who_id      ON atoms(who_id);
CREATE INDEX IF NOT EXISTS idx_atoms_when_ts     ON atoms(when_ts);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  atom_id    TEXT NOT NULL,
  seq        INTEGER NOT NULL,
  char_start INTEGER,
  char_end   INTEGER,
  text       TEXT NOT NULL,
  -- The text `vector` was actually built from: `text` with OPYT's OWN renderer output stripped
  -- (see embed_surface.py). SEPARATE from `text` because they answer different questions —
  -- `text` is what a reader is SHOWN and what `char_start`/`char_end` index into, so it must not
  -- move; `embed_text` is what the embedder SAW, and it is the only way to tell a vector built by
  -- an older strip from a current one. NULL means "never stripped" (every pre-2026-08-11 row).
  -- Nothing reads this column at query time — not `retrieve`, not `sitting_builder`, not the
  -- bookmark reader. Its two consumers are both offline: a human auditing what the embedder saw,
  -- and `restrip_embed_surface.py`, which detects staleness by recomputing the strip and comparing.
  -- So NULL needs no fallback handling anywhere; it is a fact about a row, not a case to handle.
  -- (This line used to claim "every reader falls back to `text`". There were no readers to do it.
  -- Say what is true today — a promise about readers that do not exist is the kind a mirror copies.)
  -- NOT in `chunks_fts` — BM25 already discounts a term that appears in 42% of documents, so the
  -- FTS arm never had this problem to fix.
  embed_text TEXT,
  vector     BLOB,                     -- L2-normalized; dim AND blob width in kb_meta
                                       -- (kb_meta.storage_dtype — do NOT assume float32)
  UNIQUE(atom_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chunks_atom ON chunks(atom_id);

-- Standalone FTS5 (NOT external-content): the BM25 router key. `atom_id`/`chunk_id`
-- are UNINDEXED payload columns so a hit maps back to its atom, and so we can delete
-- an atom's rows by atom_id when its snapshot changes.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
  USING fts5(text, atom_id UNINDEXED, chunk_id UNINDEXED);

-- What a candidate IS lives in `profile.classified_kind`, written by the screen classifier —
-- the only code that ever decides it. A `kind` column here was deleted 2026-08-23: born in the
-- founding schema as an open-ended `'person' | 'org' | …` enum, it never gained a reader, and 14
-- ingest sites filled it with the constant 'person' because the argument existed.
-- Design record: docs/plans/2026-08-23-candidate-search-atom-arm.md
CREATE TABLE IF NOT EXISTS entities (
  entity_id      TEXT PRIMARY KEY,
  name           TEXT,
  identity_links TEXT,                 -- JSON: cross-platform handles (x↔github↔…)
  canonical_id   TEXT,                 -- Stage-3: attested-resolution cluster head (self when unmerged)
  profile        TEXT                  -- Stage-4 JSON: {bio, verified, followers, classified_kind, classified_at}
);

-- Stage-4 output: the user's CONFIRMED Oracles, keyed on the CANONICAL entity (so a
-- cross-platform person is one oracle, not one-per-platform). Stage 5 reads this table
-- to know which footprints to expand.
-- What an ingest COVERED is not here: it is per (Oracle x source), in `oracle_sources`
-- (`last_pulled_at` / `covered_from`, pipeline/kb/oracle_refresh_state.py). This table answers
-- "is this person an Oracle", nothing about how much of them we hold. `paused` is reserved/inert.
CREATE TABLE IF NOT EXISTS oracles (
  canonical_id TEXT PRIMARY KEY,
  name         TEXT,
  source       TEXT,                   -- 'screen' (a ranked pick) | 'freeform' (a pasted handle)
  confirmed_at TEXT DEFAULT (datetime('now')),
  paused       INTEGER NOT NULL DEFAULT 0  -- reserved — inert v1
);

-- A discovered source that is not safe to attribute needs a durable user decision. This is
-- separate from `oracle_sources`: that table records pull coverage for sources already enabled;
-- this table records whether a possible source is allowed to become enabled at all.
CREATE TABLE IF NOT EXISTS oracle_review_items (
  review_id         INTEGER PRIMARY KEY,
  canonical_id      TEXT NOT NULL,
  source_type       TEXT NOT NULL,
  source_url        TEXT NOT NULL,
  source_key        TEXT NOT NULL,     -- platform-aware canonical identity for deduplication
  reason            TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'pending', -- pending | verified | approved | dismissed
  verification_urls TEXT,              -- JSON URLs the user supplied to re-check the identity
  created_at        TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(canonical_id, source_type, source_key)
);

-- The OTHER knowledge bases this install can read — the reader-side registry, owned by
-- pipeline/kb/peers.py. `oracles` says whose CONTENT this store collects; `peers` says whose
-- STORE this install may query. Different sentences, so different tables.
--
-- A table rather than a JSON file so it inherits $OPYT_HOME sandboxing, idempotent creation and
-- transactional writes, and so revocation is a row delete — the same shape the service's token
-- registry takes. `location` is EITHER a filesystem path or an HTTPS base URL; `peers.is_remote`
-- is the one place that tells them apart, so the two kinds cost one branch rather than two
-- tables.
--
-- This table must NEVER be carried into an export (pipeline/kb/export.py `_CARRY`): it is the
-- READER's list of who they can read, it says nothing about the corpus, and the reader's bearer
-- tokens sit in it.
CREATE TABLE IF NOT EXISTS peers (
  name     TEXT PRIMARY KEY,           -- what a caller types: kb="david"
  location TEXT NOT NULL,              -- an absolute export path, or an https:// base URL
  label    TEXT,                       -- display name, for provenance: "David's KB"
  token    TEXT,                       -- the reader bearer token, for a remote peer only
  added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The user's OWN endorsement signals, folded per (entity, signal_type, platform). This is
-- the ONLY table that records what the user did TOWARD a source (saved/followed/liked/
-- listed/subscribed) — atoms record what a source SAID, edges record what a source
-- ASSERTS; signals record the user's curation act. Stage-4 (Oracle candidate SCREEN) is
-- the sole consumer: it ranks entities by distinct-signal count. `count` is the action
-- strength (how many of your likes an author earned, how many of your lists a person is
-- in). `extra` is a JSON blob for signal-specific context (list names, is_paid) that a
-- later view may want without a schema break.
CREATE TABLE IF NOT EXISTS curation_signals (
  entity_id   TEXT NOT NULL,
  signal_type TEXT NOT NULL,           -- save | follow | like | list | subscribe
  platform    TEXT NOT NULL,           -- x | substack
  count       INTEGER NOT NULL DEFAULT 1,
  extra       TEXT,                    -- JSON: {list_names:[...]} | {is_paid:bool} | …
  first_seen  TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (entity_id, signal_type, platform)
);
CREATE INDEX IF NOT EXISTS idx_curation_signals_entity ON curation_signals(entity_id);

-- W0 substrate: one row per OBSERVED engagement act (an Oracle replying to / quoting /
-- mentioning / linking another account), captured from footprint pulls BEFORE the curation
-- filter drops replies-to-others. Deliberately NOT `edges`: the edges PK collapses repeat
-- engagements into one row and carries no `when`, and this table's rows are observations
-- of a third party's acts, not the user's curation (`curation_signals`) or a source's
-- assertions (`edges`). `target_id` is `x:user:{id}` when the payload carries the numeric
-- id (replies + quotes do), else `x:@{handle}` resolved later — never a slugified handle.
-- `observed_at` = when the engagement HAPPENED (the tweet's date; back-mine: the atom's
-- date), day precision, not wall clock. INSERT OR IGNORE on the natural key makes re-runs
-- write zero new rows; the first observation's date wins.
--
-- Capture broadly, separate at read time (2026-08-07). `kind` distinguishes the ACT, and a
-- reader filters on it; the capture side never drops an observation to encode a judgment.
-- 'quote' = one X post pointing at another (the `quoted_tweet` object, or a status link in
-- an X-source atom). 'reference' = a NON-X source (a Substack essay, a blog post) linking a
-- tweet — the same pointing act through a different, more deliberate surface. Keeping both
-- under one `kind` would make it mean two things; dropping one would destroy rows that
-- cannot be re-derived. So: two values, every row kept.
CREATE TABLE IF NOT EXISTS engagements (
  observer_id  TEXT NOT NULL,         -- the engaging Oracle's entity id ('x:user:{id}')
  kind         TEXT NOT NULL,         -- 'reply' | 'quote' | 'mention' | 'reference'
  target_id    TEXT NOT NULL,         -- 'x:user:{id}' | 'x:@{handle}' (unresolved)
  src_ref      TEXT NOT NULL,         -- WHERE it happened: a raw tweet id (live capture) or
                                      -- an atom_id (back-mine) — namespaced, so a Substack
                                      -- post id can never collide with a tweet id
  observed_at  TEXT,                  -- YYYY-MM-DD (the engagement's own date)
  PRIMARY KEY (observer_id, kind, target_id, src_ref)
);
CREATE INDEX IF NOT EXISTS idx_engagements_target ON engagements(target_id);


-- Footprint-eligibility cache (Stage-5 single-author gate). Person-INDEPENDENT: whether a
-- SITE is written by one author or by many/an org is a property of the site, not of who's
-- asking — so the verdict is keyed on the canonical site (bare host) and reused across every
-- person who links to it. Only DEFINITIVE verdicts land here (`single`|`multi`); a transient
-- `unknown` (fetch/LLM failure) is deliberately NOT cached (else one hiccup poisons the source
-- forever). `author_name` is the sole author when single — the person-specific author-match
-- check in the gate reads it. Additive, tiny at OPYT volume (few Oracles × few sources).
CREATE TABLE IF NOT EXISTS source_authorship (
  source_url    TEXT PRIMARY KEY,     -- canonical site key (bare host via url_canon.canonical_identity)
  authorship    TEXT NOT NULL,        -- 'single' | 'multi'
  author_name   TEXT,                 -- the sole author's name when single (for the mismatch check)
  classified_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Frontier stage 1: the standing queries a generator derives from the KB. NOT atoms, and the
-- separation is the whole license for generating them agentically — nothing machine-picked ever
-- enters `atoms`, so a bad query costs inbox noise and never KB pollution.
--
-- Identity is `normalized`, NOT `text`. A re-emitted query must land on the SAME row: stage 2's
-- per-query watermark and yield history key on `query_id`, so a query that spawns a twin on every
-- run silently resets its own watermark and re-pulls the same window forever. Hashing the
-- normalized form (not slug-truncating the display form) is what makes that identity stable —
-- slugs collide once two queries share a prefix, hashes do not.
--
-- Decay, not deletion. The system never removes a query. A reader that explicitly DROPS a thread
-- bumps `miss_count`, and stage 2 reads that counter as a SPEED — 0-2 daily, 3-9 weekly, 10+
-- monthly and never slower. The query keeps running at its floor forever. Only a human writes
-- `status='retired'`, through `frontier_queries.retire_query`.
--
-- `miss_count` is driven by an explicit verdict and never by absence. v1 inferred death from
-- silence — a query the reader did not re-emit accrued a miss and hid at 3 — and both halves
-- failed. The reader was capped at 10-25 queries TOTAL, so at the cap a new query could only enter
-- by displacing an old one: the miss count equalled the addition count, and three live queries were
-- retired by arithmetic. And hiding was a one-way door, since the reader's standing list came from
-- the same filter that excluded hidden rows, so nothing hidden could ever be re-emitted or revived.
CREATE TABLE IF NOT EXISTS frontier_queries (
  query_id        TEXT PRIMARY KEY,      -- sha256(normalized)[:16]
  text            TEXT NOT NULL,         -- display form, verbatim as emitted
  normalized      TEXT NOT NULL UNIQUE,  -- lowercase + whitespace-collapsed = the identity
  generator       TEXT NOT NULL,         -- ORIGIN: whoever emitted it FIRST. Frozen at insert, like
                                         -- `created_at`. NOT ownership — who still wants it is one
                                         -- row per claim in frontier_query_generators below.
  status          TEXT NOT NULL DEFAULT 'active',  -- active | retired ('retired' = a HUMAN decision)
  rationale       TEXT,
  target_sources  TEXT,                  -- JSON array of adapter slugs
  source_atom_ids TEXT,                  -- JSON array: provenance back to the atoms that motivated it
  emit_count      INTEGER NOT NULL DEFAULT 1,
  miss_count      INTEGER NOT NULL DEFAULT 0,  -- the SPEED stage 2 reads. A PROJECTION: the MIN of
                                         -- this query's per-claim counters, maintained by
                                         -- frontier_queries._sync_speed after every verdict.
  created_at      TEXT NOT NULL,
  last_emitted_at TEXT NOT NULL,
  -- Which LANE emitted this query: 'human' (some human-attested atom motivated it) | 'machine'
  -- (every atom it cites was found by the crawler). It exists to bound question-list OWNERSHIP —
  -- a region may hold at most `frontier_queries.MACHINE_LANE_QUOTA` standing machine-lane queries,
  -- so a read of a region full of machine finds cannot mint an unbounded standing watch-list off
  -- its own output. It is NOT about money: watermarked pulls are near-free.
  --
  -- ONE-WAY STICKY TO 'human', enforced in the upsert's ON CONFLICT arm. The descriptive columns
  -- above are last-writer-wins, and without the stickiness this one would flap as two regions
  -- re-emit the same text from different material — which would make the quota count
  -- nondeterministic and let a query oscillate in and out of the clamp forever.
  --
  -- The machine lane is named `machine`, never after the entry_mode it detects — the guard
  -- `human-attested-stays-human` forbids that spelling in this file, and unquoting it in prose is
  -- what its message asks for rather than an allow-list. Lanes are enforcement-internal
  -- bookkeeping and never reach a surface; the watchlist excludes this column outright.
  -- NULL means unclassified — every row written before this column existed, all of them from a
  -- human-only membership world, so NULL reads as human and consumes no quota.
  lane            TEXT
);
CREATE INDEX IF NOT EXISTS idx_frontier_queries_status ON frontier_queries(status);

-- Which generators ask for a standing query — one row per claim. A table and not the `generator`
-- column, for the same reason `frontier_candidate_queries` is a table: more than one generator
-- emits the same query, and one column can only remember the last to write.
--
-- `sitting_reader.py` calls that column the ownership key — "it decides which rows a re-seed of
-- this region refreshes, and which rail's verdicts may move a row's counters at all" — and that is
-- the right intent. The column could not deliver it: `upsert_queries` overwrote it on collision, so
-- the last writer inherited every earlier region's rights. Invisible while regions were far apart
-- and one query had one region; `zoom` fractures a region into siblings reading overlapping
-- material, and four sub-reads of one region emitted 76 queries into 75 rows (2026-08-10).
--
-- Two things broke, both silently. A verdict from the dispossessed region landed as `unmatched`,
-- so its explicit keep or drop was discarded — under a design where survival IS the verdict. And
-- `active_queries(generator=...)` filtered on the same column, so the query left that region's
-- standing list; the reader was never shown it, could not verdict it, and re-invented its wording
-- on the next read, orphaning the stage-2 watermark and leaving a duplicate row.
--
-- `miss_count` lives HERE because a verdict is one generator's opinion about how fast IT wants a
-- thread run. On the shared row, one region's `drop` slowed a query another had just kept, and one
-- region's `keep` erased another's accumulated drops. NO `status` COLUMN: a claim has no lifecycle
-- of its own. Nothing retires a claim, because nothing retires a query — a drop is a slowdown, and
-- only a human writes `frontier_queries.status='retired'`.
CREATE TABLE IF NOT EXISTS frontier_query_generators (
  query_id         TEXT NOT NULL,
  generator        TEXT NOT NULL,
  first_emitted_at TEXT NOT NULL,        -- when THIS generator first asked
  last_emitted_at  TEXT NOT NULL,
  miss_count       INTEGER NOT NULL DEFAULT 0,  -- CONSECUTIVE drop verdicts from THIS generator
  PRIMARY KEY (query_id, generator)
);
CREATE INDEX IF NOT EXISTS idx_fqg_generator ON frontier_query_generators(generator);

-- One row per GENERATOR — the channel a query came from, as an addressable thing rather than a
-- string repeated across every claim. Exactly two channels exist: `bookmark-reader` (one, fixed)
-- and `sitting:<slug>` (one per region). Added 2026-08-12; a column on the claim row was rejected
-- once this table had to carry three facts rather than one.
--
-- `votable` is the load-bearing column. A generator that can NEVER issue a keep/drop verdict must
-- not have its frozen counter counted when `_sync_speed` takes the MIN across claims: a claim that
-- cannot change its mind would otherwise pin the query to the fastest tier permanently and silence
-- every other claimant's verdicts. The DEFAULT is 1 — a claim votes unless declared otherwise,
-- matching how this rail treats silence.
--
-- The watchlist sets this to 0: `sitting_tools.py:386` calls `add_user_query(votable=False)`
-- through a production path, since a query a person typed on purpose must not decay because
-- nobody voted on it. `c046bfcf` (2026-08-25) added that and did not touch this comment, which
-- until 2026-09-06 still said "no shipping generator sets this to 0 today" — a sentence born
-- true and read as current for twelve days.
--
-- `sitting:*` also set it to 0 once, on the grounds that a sitting was read exactly once so no
-- verdict could ever arrive; that ended 2026-08-16 when `sitting_reader` was taught verdicts
-- (D11) and flipped to votable in the same commit, so retiring the bookmark reader (D13) would
-- not take decay with it.
--
-- `status` WAS the per-CHANNEL kill switch. Nothing reads it and nothing writes it any more:
-- `frontier_queries.retire_generator` (the only writer) and `generators` (the only reader outside
-- it) were deleted on 2026-09-05, because neither had ever been run and neither had a door in the
-- app. Decay already answers "that region turned out to be a dead end" without anyone acting — a
-- quiet thread slows to the monthly floor and stays visible and revivable. Every row therefore
-- holds the DEFAULT. The column is not dropped here only because dropping it is a migration; if
-- you are about to write a second reader of it, delete the column instead.
--
-- Per-QUERY retirement is untouched and is still a human act: `frontier_queries.status`, written
-- by `retire_query` and undone by `unretire_query`.
--
-- `label` exists because a slug is an id, not a name. `sitting:mlx-continuous-batching` does not
-- tell you what region that was or when it was read.
CREATE TABLE IF NOT EXISTS frontier_generators (
  generator    TEXT PRIMARY KEY,       -- 'bookmark-reader' | 'sitting:<slug>'
  label        TEXT,                   -- human-readable; the slug is an identifier, not a name
  votable      INTEGER NOT NULL DEFAULT 1,      -- can this channel ever render a verdict?
  status       TEXT NOT NULL DEFAULT 'active',  -- active | retired (a HUMAN decision)
  created_at   TEXT NOT NULL,
  last_seen_at TEXT NOT NULL           -- last time this generator emitted anything
);

-- One row per generator run, including the ones that did nothing. Skips and failures are recorded
-- rather than dropped because the TRIGGER reads this table: "new saves since the last ok run" has
-- no meaning without a durable record of when that was, and a failed run must NOT advance that
-- mark (else one outage silently swallows the window it failed to read).
--
-- `consensus` is process state, not KB content. It is what the reader understood on that date, kept
-- so a later run's queries can be explained. It is never retrieved, never embedded, and never
-- treated as a fact about the world — it is a machine opinion, and the admissibility rule keeps
-- machine opinions out of the KB proper.
-- `generator` scopes this table, and without it two reading rails corrupt each other. The bookmark
-- reader derives both its trigger ("new saves since the last ok run") and its single-flight rule
-- ("when did we last read") from these rows. A sitting read landing here unscoped would satisfy
-- both — silently suppressing the next bookmark read. NULL
-- means "written before this column existed", which was bookmark-reader-only.
CREATE TABLE IF NOT EXISTS frontier_reader_runs (
  run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  generator   TEXT,                      -- 'bookmark-reader' | 'sitting:<slug>'
  sitting_id  TEXT,                      -- set when this run read a sitting
  -- Which lens produced this row. NULL on every row written before Job L — those all predate the
  -- lens split, so NULL means 'queries' by construction, the same way a NULL `generator` means
  -- 'bookmark-reader' above. A host-side lens (briefing, trajectory, disconfirmation, gaps) writes
  -- a RECEIPT here — this column set, no consensus/queries — never a full read row; see
  -- `sitting_reader.record_lens_run`. This is what lets `sitting_scheduler`'s `pointed` channel
  -- tell "a queries read was attempted and failed" (retry it) apart from "a lens read the region
  -- and stored nothing" (not a retry signal) — see docs/plans/2026-08-16-lens-reads-subscribe-a-region.md.
  lens        TEXT,
  ran_at      TEXT NOT NULL,
  atoms_read  INTEGER,
  consensus   TEXT,
  model       TEXT, in_tokens INTEGER, out_tokens INTEGER,
  -- `emitted` is the one counter every lens shares: how many records this run produced (queries, or
  -- Job N's claims). `new`/`refreshed`/`kept`/`dropped`/`unverdicted` are QUERY-SHAPED — they answer
  -- questions ("was this seen before", "did it survive a verdict") that only make sense where a
  -- STANDING set exists to compare against. A `claims` run leaves all five NULL rather than being
  -- given a generic record count of its own: a claims read has nothing analogous to a verdict, and
  -- inventing one would look like a signal where none was measured.
  emitted INTEGER, new INTEGER, refreshed INTEGER, marked_dormant INTEGER,
  -- The verdict ledger for one run. `unverdicted` is the one to watch: it is shown-minus-decided,
  -- so a rising number means the survival signal is degrading while every counter sits still and
  -- looks perfectly healthy. `marked_dormant` is kept for rows written before verdicts existed and
  -- is never written again.
  kept INTEGER, dropped INTEGER, unverdicted INTEGER,
  -- D22, made STANDING rather than a bake-off-only measure (Job N). The fraction of cited atoms
  -- landing in the middle half of the chronological render — ~0.5 means the model read the whole
  -- thing, low means lost-in-the-middle. Was folded into free-text `reason` until now; a real column
  -- is what lets a future pass ask "which reads degraded" without parsing prose. Written by both
  -- `queries` and `claims` reads — the failure mode is about the RENDER being chronological, not
  -- about which lens is reading it.
  middle_share REAL,
  status TEXT NOT NULL,                  -- ok | skipped | failed
  reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_frontier_runs_status_time ON frontier_reader_runs(status, ran_at);

-- A SITTING is one assembled reading context: start from a seed point in embedding space, admit
-- related atoms until the qualifying pool runs dry (saturation) or the token budget binds. It is
-- the slice-picker generalized: the reader before it (`bookmark_reader.py`, deleted 2026-08-16)
-- hard-coded "the last 90 days of saves", where recency is only one seed among several (a phrase,
-- a set of atoms someone pointed at, a centroid a fracture produced).
--
-- A sitting is an event, not a name. `sitting_id` folds `built_at` in, so re-seeding the same
-- phrase next month appends a second row instead of overwriting the first. Two reasons, and the
-- second is the load-bearing one: the corpus underneath has changed, so it is genuinely a
-- different context; and overwriting would erase `read_at`, silently converting a region that was
-- read into one that never was. Building is free — only READING costs — so duplicate builds are a
-- visible non-problem, and clobbered read history would be an invisible real one.
CREATE TABLE IF NOT EXISTS sittings (
  sitting_id       TEXT PRIMARY KEY,     -- sha256(seed_kind|seed_ref|dials|built_at)[:16]
  built_at         TEXT NOT NULL,
  seed_kind        TEXT NOT NULL,        -- 'query' (FTS) | 'atoms' (explicit) | 'vector' (centroid)
  seed_ref         TEXT,                 -- the phrase / the ids / the centroid's label
  seed_atom_ids    TEXT,                 -- JSON array; empty for a raw-vector seed
  floor            REAL NOT NULL,        -- membership threshold on chunk-grain cosine
  calibrated_floor REAL,                 -- p99 of RANDOM chunk pairs + margin = the noise ceiling
  ceiling          REAL NOT NULL,        -- redundancy above this = near-duplicate, skipped outright
  budget_tokens    INTEGER NOT NULL,
  region_atoms     INTEGER,              -- still admissible at build time: at/above the floor, minus
                                         -- the seeds, minus everything prior parts already read
  region_tokens    INTEGER,
  atoms            INTEGER,              -- what actually made it into the sitting
  tokens           INTEGER,
  stop             TEXT NOT NULL,        -- saturation | budget | empty
  skipped_dupes    INTEGER NOT NULL DEFAULT 0,
  -- The skip list itself, JSON `[{atom_id, red}, ...]` — the COUNT above says how much a reader
  -- dropped, this says WHAT. The pair exists because the cross-lane ceiling ruling (2026-08-24)
  -- kept the skip and rested that choice on the skips being auditable: "an observed skip that is a
  -- response rather than a crosspost" has to be a one-query lookup, and a count cannot answer it.
  skipped          TEXT,
  -- A budget-stopped region is bigger than one sitting, so it is read in PARTS. `continues` points
  -- at the previous part; walking the links back gives the whole chain. It is a parent POINTER
  -- rather than a caller-supplied atom list because a part-4 build must inherit parts 1-3, and a
  -- caller that forgot one would produce a silently-too-small redundancy baseline — a failure that
  -- looks exactly like "there were no duplicates". One id, correct by construction.
  continues        TEXT REFERENCES sittings(sitting_id),
  prior_atoms      INTEGER NOT NULL DEFAULT 0,  -- chain size behind this part; full region is
                                                -- region_atoms + prior_atoms + |seeds|
  -- The region this one was fractured out of, set only by `sitting_zoom.zoom`. Provenance, not
  -- reading order, and deliberately NOT `continues`: a continuation is the next part of one region
  -- and must not re-read its predecessors, while a sub-sitting is a fresh region at a finer floor
  -- that is MEANT to overlap its siblings. Collapsing the two would put every sub-sitting's siblings
  -- into its redundancy baseline and gut exactly the material zoom exists to separate.
  parent_sitting_id TEXT REFERENCES sittings(sitting_id),
  -- Which region this event is a read OF. `sitting_id` answers "which BUILD" — it folds `built_at`
  -- in, deliberately, for the reason above. This answers the different question the re-read trigger
  -- asks, so last-read is `MAX(read_at) GROUP BY region_key` rather than a per-row stamp.
  --
  -- It excludes the resolved seed atom IDs, and that exclusion is the whole ruling. Any top-k
  -- membership CHURNS as the corpus grows: ask for the best 4 atoms, save something better next
  -- month, and one of the four is displaced. Hash that set into the key and the same phrase yields a
  -- different key every month — last-read becomes unfindable and the trigger fires forever on a
  -- region it just read. (A weaker drift source, BM25's corpus-relative IDF, expires once seeds are
  -- vector-resolved. It is NOT the reason. The churn is.)
  --
  -- Two region identities exist and they are not interchangeable.
  --      generator  = `sitting:<slug>`, from the seed_ref ALONE — query OWNERSHIP, coarse.
  --      region_key = seed_ref + every dial            — read state, fine.
  -- The same phrase at two floors is one generator and two regions: its queries should refresh each
  -- other, its read stamps must not.
  region_key       TEXT,
  -- The anchor itself, at `kb_meta.storage_dtype` width, encoded exactly like `chunks.vector`.
  --
  -- Stored for EVERY seed kind, including the two whose vector looks reproducible. Three things need
  -- it and only one of them is about `vector` seeds. (1) A `vector` seed has an EMPTY
  -- `seed_atom_ids` — every fracture `zoom` has ever produced is unreproducible without this column.
  -- (2) It makes the continuation rule structural rather than procedural: "part N must never
  -- re-resolve the phrase" is trivially true when there is nothing left to re-resolve. (3) The
  -- re-read trigger cannot be computed without it — "how much new mass does this region hold" means
  -- re-running membership, which is cosine against THIS vector at THIS floor.
  seed_vector      BLOB,
  -- Stamped when an agent actually READ it; NULL = unread, and `read_at IS NOT NULL` is the whole
  -- test every reader makes. A `read_status` column sat beside this until 2026-09-06 holding the
  -- literal 'ok' and nothing else: a FAILED read writes no stamp at all, so the region stays
  -- unread and the ledger keeps asking for it (`sitting_reader._fail`), and the failure is
  -- recorded in the run table. Do NOT bring it back — `b24fba39` removed the second writable
  -- value because it was a second home for a fact the run log already owns, and the only thing
  -- it added was a way to report a failure as coverage.
  read_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_sittings_read ON sittings(read_at);
-- The indexes on `continues`, `parent_sitting_id` and `region_key` are created in `init_kb_schema`,
-- NOT here. On a store written before those columns existed, `CREATE TABLE IF NOT EXISTS` is a
-- no-op, so this script would index a column that `_ensure_column` has not added yet — the whole
-- DDL then fails and the store will not open.

-- Membership, one row per atom per sitting, carrying the scores that admitted it. This IS the
-- coverage ledger of Amendment 2, and it is deliberately NOT the denormalized
-- `atom_id -> last_read_at` the plan sketched: joining to `sittings.read_at` derives the same
-- answer with no field that can go stale, and it keeps BUILT-BUT-UNREAD visible as its own state
-- instead of counting as covered. An atom no seed ever reached appears in no row at all — which is
-- the point. "Never read and never reported unread" is the silent-drop tax this table exists to pay.
CREATE TABLE IF NOT EXISTS sitting_atoms (
  sitting_id TEXT NOT NULL,
  atom_id    TEXT NOT NULL,
  rank       INTEGER NOT NULL,           -- admission order; seeds are 0..n-1
  is_seed    INTEGER NOT NULL DEFAULT 0,
  rel        REAL,                       -- MAX cosine of this atom's chunks to the SEED centroid
  red        REAL,                       -- MAX cosine to anything already admitted, at admit time
  tokens     INTEGER,
  PRIMARY KEY (sitting_id, atom_id)
);
CREATE INDEX IF NOT EXISTS idx_sitting_atoms_atom ON sitting_atoms(atom_id);

-- Job N — the `claims` lens (D7, D20). A falsifiable claim, over a set of atoms it cites, from ONE
-- read of ONE sitting. Sibling of `frontier_queries` in shape (both are structured output from a
-- reading job) and deliberately not merged with it: a claim answers "what does this material
-- establish", a query answers "what should we watch next", and D20 keeps them one API call each
-- rather than one call answering both.
CREATE TABLE IF NOT EXISTS sitting_claims (
  claim_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  sitting_id   TEXT NOT NULL,
  claim        TEXT NOT NULL,
  -- The observation that would show the claim wrong. Its reader is the HUMAN deciding whether to
  -- believe the claim, not a machine — nothing re-checks a stored claim, and none is planned
  -- (docs/plans/2026-08-16-sitting-rail-model-bakeoff.md). Do NOT read this as "so it can be
  -- verified later"; that invents a consumer that does not exist, the same failure
  -- `sitting_render`'s sprouts-digest docstrings already carried a scar for.
  falsified_by TEXT NOT NULL,
  atom_ids     TEXT NOT NULL,        -- JSON array; resolved against the window via _resolve_atom_ids
  created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sitting_claims_sitting ON sitting_claims(sitting_id);

-- Per-LENS read state, additive alongside `sittings.read_at` rather than a replacement for it.
-- `queries` keeps its existing stamp on `sittings` — `sitting_scheduler`'s
-- `pointed`/`sub_region`/new-mass SQL all read that column directly, and this table does not touch
-- any of it. What this buys is a second lens (`claims` today) getting its OWN "never re-read once
-- read" guard on the SAME sitting, independent of whether `queries` has read it — the two lenses
-- answer different questions (D20) and must not block or unblock each other.
CREATE TABLE IF NOT EXISTS sitting_reads (
  sitting_id  TEXT NOT NULL,
  lens        TEXT NOT NULL,
  read_at     TEXT NOT NULL,          -- presence IS the state; see `sittings.read_at` above
  PRIMARY KEY (sitting_id, lens)
);

-- The MAP half of a map-reduce lens (RULED 2026-08-24). One row per (part, lens): the lens's map
-- instruction applied to ONE part's rendered document. The host then receives every part's output
-- at once, plus the lens's join rule, and performs the reduce in-session.
--
-- Per-part cache. Membership is fixed at build time except when `forget` removes an atom;
-- that removal discards affected parts' outputs in the same database transaction. The next
-- lens read maps those parts again. Ordinary open-tail rebuilds get a new `sitting_id`, so they
-- naturally miss the cache. Steady-state cost is one call for the open part.
--
-- The RECONCILED output is deliberately absent from this table and from every other (ruled
-- 2026-08-25). Its input is the LIVE region, which is mutable, so storing it buys the invalidation
-- problem forever plus a second record of state-over-time that can disagree with the claims
-- notebook. Cache what is frozen; recompute what is live.
--
-- This is the same test the claims notebook passed and the deleted synthesis layer failed:
-- persistence is justified by a structural consumer, and every future reconcile is one.
CREATE TABLE IF NOT EXISTS sitting_lens_outputs (
  sitting_id TEXT NOT NULL,
  lens       TEXT NOT NULL,
  output     TEXT NOT NULL,
  model      TEXT,
  in_tokens  INTEGER,
  out_tokens INTEGER,
  created_at TEXT NOT NULL,
  PRIMARY KEY (sitting_id, lens)
);

-- ── Frontier stage 2 (EXECUTE) ────────────────────────────────────────────────
-- The watermark: how far each (query, source) pair has been searched. This is the table that makes
-- query identity load-bearing, and it was unbuildable until the generator stopped re-wording its
-- own queries every run (fixed 2026-08-10) — a `query_id` that churns nightly orphans every row
-- here, so the loop would re-pull all of history on every pass and never know it.
--
-- A FAILED pull must not advance `last_pulled_at`. Stamping on error buys a full TTL of silence on
-- that pair for one bad night — the `record_pull(stamp=False)` lesson from the Oracle rail.
CREATE TABLE IF NOT EXISTS frontier_query_sources (
  query_id       TEXT NOT NULL,
  source         TEXT NOT NULL,
  last_pulled_at TEXT,                 -- NULL = never pulled → infinitely stale
  last_status    TEXT,                 -- ok | empty | error | no_adapter | breaker_open |
                                       -- window_refused (frontier_execute.window_ok said no)
  PRIMARY KEY (query_id, source)
);

-- The staging store. NOT `atoms`, and that separation is the entire license for stage 1 generating
-- queries agentically: a bad query costs inbox noise here, never KB pollution. Stage 3 (ADMIT,
-- fail-closed) owns every `status` transition; stage 2 only ever writes 'new'.
--
-- `candidate_id` is a REAL external id ('arxiv:2501.12345', 'repo:owner/name'), never a derived
-- filename. The `pipeline/artifacts/` adapters decide "already got this?" by checking whether a
-- vault markdown file exists, and the vault is scheduled for deletion — this rail keys on an id
-- that outlives it.
CREATE TABLE IF NOT EXISTS frontier_candidates (
  candidate_id  TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  -- `source` is the FINDER, `kind` is the ATOM KIND, and they are two columns because they answer
  -- two unrelated questions. Stage 4 groups and explains by `source` (whose result is this?);
  -- stage 3 dispatches on `kind` (which minter materializes it?). The two coincided exactly while
  -- arxiv and github were the only adapters — `arxiv` naming both a finder and a minter is a
  -- coincidence of that pair, not a fact about the rail, and a second paper source ends it.
  kind          TEXT,                  -- 'paper' | 'repo'
  title         TEXT,
  url           TEXT,
  published     TEXT,                  -- the artifact's own date (ISO), for windowing + display
  summary       TEXT,
  payload       TEXT,                  -- JSON, adapter-specific
  status        TEXT NOT NULL DEFAULT 'new',   -- new | materialized | rejected
  first_seen_at TEXT NOT NULL,
  last_seen_at  TEXT NOT NULL,
  -- ── Stage 3 (ADMIT) bookkeeping ──
  -- `attempts` counts FAILED attempts ONLY. A success never touches it, so `attempts > 0` reads
  -- as "this candidate has fought us N times" rather than "it was looked at N times", and the
  -- cap in frontier_admit means what it says.
  --
  -- Why a reason slug and not a traceback. The retryable-vs-terminal rule is an open question
  -- that cannot be closed without a real failure distribution (the rail has never run). These
  -- three columns ARE that measurement: a candidate that hits the attempt cap is a row saying
  -- "the rule is wrong for me, and here is why". A traceback would be unaggregatable, would
  -- change wording between library versions, and would put arbitrary upstream text in the store.
  attempts        INTEGER NOT NULL DEFAULT 0,
  last_error      TEXT,                -- short stable slug — see frontier_admit.classify()
  last_attempt_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_frontier_candidates_status ON frontier_candidates(status);

-- Which standing queries surfaced a candidate. A separate table rather than a `query_id` column,
-- because MULTI-QUERY hits are the signal: an artifact that three independent standing queries all
-- found is the strongest thing stage 4 can put in front of the user, and collapsing this into one
-- column throws that away permanently.
CREATE TABLE IF NOT EXISTS frontier_candidate_queries (
  candidate_id TEXT NOT NULL,
  query_id     TEXT NOT NULL,
  found_at     TEXT NOT NULL,
  PRIMARY KEY (candidate_id, query_id)
);
CREATE INDEX IF NOT EXISTS idx_fcq_query ON frontier_candidate_queries(query_id);

CREATE TABLE IF NOT EXISTS frontier_exec_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ran_at TEXT NOT NULL,
  pairs_due INTEGER, pairs_pulled INTEGER, pairs_deferred INTEGER, requests INTEGER,
  candidates_new INTEGER, candidates_seen INTEGER,
  status TEXT NOT NULL,                -- ok | skipped | failed
  reason TEXT
);

-- ── Frontier stage 4 (SURFACE) ────────────────────────────────────────────────
-- What the surface DID, one row per act. Stage 4 is the only component in this rail licensed to
-- hide anything, so what it hid and when has to be checkable rather than inferred.
--
-- A table, not columns on `frontier_candidates`. A candidate is shown many times, and both the
-- COUNT (which demotes it) and the SEQUENCE (which stage 5 will read) matter. A `shown_count` +
-- `last_shown_at` pair keeps the first and throws away the second, and a `dismissed` flag cannot
-- say when. Same argument `frontier_candidate_queries` already makes one section up.
--
-- Exactly two events, because exactly two have a source that can produce them. `shown` is written
-- by the surface as it returns a row. `dismissed` is written when the user explicitly says stop.
-- There is deliberately NO `opened`: MCP hands a payload to a host and gets no callback, so
-- nothing in this process can ever observe a read. An `opened` column would be a field only a
-- guess could fill, and a guess stored next to two facts stops looking like one.
--
-- An event log, never a judgement. "Shown at 14:02, dismissed at 14:05" is a receipt — checkable
-- the way `curation_signals` rows are. No SCORE is ever written here: the ranking is recomputed
-- from these facts on every call and thrown away, so a scoring change never leaves the store
-- holding stale numbers that no longer explain the order anyone saw.
CREATE TABLE IF NOT EXISTS frontier_candidate_events (
  event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id TEXT NOT NULL,
  event        TEXT NOT NULL,          -- shown | dismissed
  at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fce_candidate ON frontier_candidate_events(candidate_id, event);
"""

# The `entry_mode` values that mean a human put this here — personally saved, authored by someone
# tracked, or pointed at by someone tracked. Lives HERE, next to the column it describes, because it
# is a fact about `entry_mode` rather than a fact about any one reader. It was previously owned by
# `sitting_builder`, which was simply the first module that needed it; a second reader could not find
# it there and would have retyped the tuple, which is how one rule becomes two spellings that drift.
#
# ALLOW-LIST, never a deny-list — see the `entry_mode` column comment above. Readers select the modes
# they want, so a newly added mode is excluded by default rather than by someone remembering.
#
# Do not add the Frontier mode to this tuple. It is the one edit that silently breaks Frontier's
# anti-narrowing invariant, and it will look like a bug fix when someone makes it: stage-3 atoms are
# real, useful artifacts, so excluding them from a "human attested" set reads like an oversight. It
# is not. Stage 1 generates standing queries from this set; admitting stage 3's own output into it
# closes the loop and the generator starts feeding on what it already found. Machine discovery is
# exactly what this tuple exists to keep OUT. Guarded by `human-attested-stays-human` in .guards.py
# and pinned by tests/kb/test_frontier_admit.py.
HUMAN_ATTESTED: tuple[str, ...] = ("user-saved", "oracle-footprint", "author_referenced")

# Atom columns the writer supplies, in DDL order, minus the three the store manages itself:
# version/ingested_at (DEFAULT-managed) and first_seen (INSERT-only, written by `upsert_atom`).
# Single source of truth so a schema change is a one-line edit.
_ATOM_COLS: tuple[str, ...] = (
    "atom_id", "source_type", "what_kind", "who_id", "when_ts", "when_precision",
    "about_entities", "source_url", "raw_ref", "raw_hash",
    "description", "payload", "entry_mode",
)
# JSON-encoded on the way in, so callers pass native Python lists/dicts.
_JSON_ATOM_FIELDS = frozenset({"about_entities", "payload"})
# Overwritten by the newer ingest; identity (atom_id) never changes on UPSERT.
_ATOM_UPDATABLE = tuple(c for c in _ATOM_COLS if c != "atom_id")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Idempotently ADD a column to an existing table (SQLite has no `ADD COLUMN IF NOT
    EXISTS`). No-op when already present — the additive, zero-migrations path this store uses
    instead of a migrations dir. Fresh DBs get the column from `_DDL`; this catches DBs created
    before it existed (e.g. a Step-2 store gaining Stage-3's `canonical_id`)."""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# The signal types a FULL-SET collector re-reads, and therefore the only ones whose absence from a
# walk means anything. `save` is deliberately absent: `sync_bookmarks` stamps once per atom and
# `reconcile_saved_signals` is insert-if-absent, so nothing ever asks "is this still bookmarked".
CONFIRMABLE_SIGNALS: tuple[str, ...] = ("follow", "list", "like", "subscribe")


def _backfill_confirmations(conn: sqlite3.Connection) -> None:
    """Stamp `last_confirmed_at = now` on pre-existing confirmable rows that have none.

    This is the hazard in the whole feature, and the direction matters enormously. A signal is
    read as RETIRED when its `last_confirmed_at` predates its collector's last successful run. So
    if the column simply stayed NULL on migration, and NULL compared as "older than everything",
    the first read after upgrading would retire EVERY follow signal in the store — 470 of them on
    the live KB — and empty the candidate list in one silent step.

    Backfilling to NOW is the fail-safe direction and it converges in exactly one collector run:
    nobody is retired today, then the next full walk re-stamps the people it still sees with a
    later time, `collector_runs.last_ok_at` advances past the migration stamp, and only the people
    the walk did NOT see are left holding an older mark. The truth arrives one run late, which is
    the correct way round for a destructive-looking read.

    Backfilling to `first_seen` was rejected for being exactly the failure above wearing a
    plausible costume: those stamps are weeks old, so every row would read as retired immediately.

    Scoped to `CONFIRMABLE_SIGNALS`, so `save` rows stay NULL forever rather than collecting a
    meaningless stamp on every connect.

    READ-GUARDED, and that probe is the whole reason this function is cheap. Without it the
    `UPDATE` ran on every `connect()` even after convergence, when its `WHERE` matched zero rows —
    and `mcp_server/server.py` forks seven detached rails within milliseconds, so seven connections
    raced for the write lock over a statement that changed nothing. That is every one of the 28
    `database is locked` events measured over six days (all `elapsed_s: 0.0`, i.e. a read->write
    upgrade refusal, which `busy_timeout` cannot help). The probe takes no write lock.

    A LIVE guarded path, not a one-shot `kb_meta` stamp. The MCP server runs from the primary
    checkout, so an older build can still insert a NULL row after this ships; a guarded path heals
    it on the next connect where a stamp would refuse forever. `_rename_crawled_to_footprint`
    states the same precedent."""
    ph = ", ".join("?" for _ in CONFIRMABLE_SIGNALS)
    probe = conn.execute(
        f"SELECT 1 FROM curation_signals WHERE last_confirmed_at IS NULL "
        f"AND signal_type IN ({ph}) LIMIT 1",
        CONFIRMABLE_SIGNALS).fetchone()
    if probe is None:
        return
    conn.execute(
        f"UPDATE curation_signals SET last_confirmed_at = datetime('now') "
        f"WHERE last_confirmed_at IS NULL AND signal_type IN ({ph})",
        CONFIRMABLE_SIGNALS)


def _backfill_first_seen(conn: sqlite3.Connection) -> None:
    """Seed `atoms.first_seen` from `ingested_at` on rows that predate the column.

    Only correct while a row is still `version = 1` — before any re-observation has refreshed
    `ingested_at` away from the true arrival date. Lossless for the near-totality of rows at the
    time this was written;
    Treat `first_seen` on a pre-column row as "no later than", not as exact.

    NULL was rejected as the migration value. A NULL arrival date has to be special-cased by every
    future selector, and the fail-safe direction for a re-read clock is a date that is too EARLY
    (the region reads sooner than it must) rather than absent (the region is skipped or crashes).

    Guarded on `IS NULL` twice over, and the two guards do different jobs. The `WHERE` makes the
    write correct — it self-heals a row that lands without a `first_seen` from a raw test-fixture
    INSERT or an atom written by an older checkout whose `upsert_atom` did not know the column,
    which is why this is a permanent path and not a one-shot. The `SELECT ... LIMIT 1` probe makes
    it cheap: a converged store takes NO write lock at all, where before it took one on every
    `connect()` to run an `UPDATE` matching zero rows. `_backfill_confirmations` above has the
    measurement of what that cost."""
    if conn.execute("SELECT 1 FROM atoms WHERE first_seen IS NULL LIMIT 1").fetchone() is None:
        return
    conn.execute("UPDATE atoms SET first_seen = ingested_at WHERE first_seen IS NULL")


def region_key(seed_kind: str, seed_ref, floor, ceiling, budget_tokens) -> str:
    """`sha256(seed_kind|seed_ref|floor|ceiling|budget)[:16]` — which region, not which build.

    Lives here rather than in `sitting_builder` because the migration below has to compute it for a
    row written by a checkout that had no such column, and the substrate must not import a consumer
    to do that. See the `sittings` DDL for what is in the key, what is deliberately out of it, and
    why `generator` is a SECOND, coarser identity that this one does not replace.

    Every input is coerced before it is formatted. The key is a string built from `repr`-shaped
    numbers, so `120000` and `120000.0` would hash apart while meaning the same budget — and the two
    call sites feed it from different places (a caller's literal at write time, a SQLite column at
    migration time). Coercion is what makes those two agree.
    """
    parts = [seed_kind, str(seed_ref), f"{float(floor)}",
             f"{float(ceiling)}", f"{int(budget_tokens)}"]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _ddl_indexes(table: str) -> list[str]:
    """Every `CREATE INDEX` statement `_DDL` declares for `table`. One declaration, read back.

    This replaced `_SITTINGS_INDEXES`, a hand-kept tuple restating four indexes that `_DDL` and
    `init_kb_schema` already declared between them — a third copy that had to be edited whenever
    either of the other two was, and that only ever covered the one table someone had noticed."""
    return [ln.rstrip(";") for ln in _DDL.splitlines()
            if ln.startswith("CREATE INDEX") and f" ON {table}(" in ln]


def _rebuild_without(conn: sqlite3.Connection, table: str, *drop_cols: str,
                     after: tuple[str, ...] = ()) -> bool:
    """Rebuild `table` from the CURRENT `_DDL` without `drop_cols`. True if it ran, False if the
    store was already converged — a read, so no write lock.

    The portable create-copy-drop-rename. NOT `ALTER TABLE ... DROP COLUMN`, which needs
    SQLite >= 3.35 (2021-03), and the distributability invariant forbids assuming a version of
    anything on the user's machine. `init_kb_schema` runs on EVERY writable `connect()`, so such a
    statement is not a wrong answer — it is a store that does not OPEN, on a machine whose owner
    did nothing but leave it closed for a fortnight.

    ONE body, because there were five and the persistence rules are the part a copy gets subtly
    wrong. Of the four below, three were wrong in all five copies; the fourth (indexes) was handled
    in exactly one of them, and only by keeping a third list by hand.

    THE COLUMN LIST IS INTERSECTED with the new table's, not taken from the old one. Every copy
    read `PRAGMA table_info` on the OLD table while slicing the replacement DDL from the CURRENT
    `_DDL`, so the moment `_DDL` dropped a SECOND column the two disagreed — `OperationalError:
    table _sittings_new has no column named read_status` — and every store last written before the
    first drop stopped opening. Reading the new table's own `table_info` after creating it is
    exact and parses nothing.

    NO `executescript`. It issues an implicit COMMIT, so the `BEGIN` above it was committed away
    and the `CREATE TABLE _<t>_new` landed durably on its own; measured, `in_transaction` went
    True then False across the call and the orphan survived a simulated crash. The slice is a
    single statement, so `execute` runs it and the transaction stays open — which is the thing
    that makes the rebuild atomic at all.

    `DROP TABLE IF EXISTS` on the temp name first, because stores that ran the old code and were
    interrupted carry that orphan right now. The old slice also replaced `CREATE TABLE IF NOT
    EXISTS <t> (` with a bare `CREATE TABLE _<t>_new (`, so the retry raised `table _<t>_new
    already exists` inside `init_kb_schema` on every connect, with no repair path. Dropping is
    better than restoring the `IF NOT EXISTS`: an orphan from an older `_DDL` has the wrong shape,
    and `IF NOT EXISTS` would silently reuse it.

    `PRAGMA legacy_alter_table=ON` for the RENAME: a copied FK clause names the original table,
    which does not exist between the DROP and the RENAME. This store never turns foreign keys on
    (see `connect`), so those clauses are inert text either way.

    THE TABLE'S OWN `_DDL` INDEXES ARE REISSUED. Dropping a table drops its indexes, and `_DDL`
    already ran at the top of `init_kb_schema`, so an index declared beside the CREATE TABLE is
    gone for the rest of that process — silently, because nothing reads an index to check it is
    there. `init_kb_schema` re-creates SOME of them further down, and only for the tables where
    somebody previously noticed. Taking them from `_DDL` needs no second list: an index there is
    consistent with the table there by construction, since removing a column removes its index in
    the same edit.

    `after` is SQL to run inside the SAME transaction, after the rename — the `region_key` reset
    `sittings` needs, and nothing else today. Running it outside the transaction leaves a window
    where the table is converged and the follow-up is not, with nothing left to notice: the guard
    reads False from then on.

    MUST NOT become `BEGIN IMMEDIATE`. `d2d30591` rejected that: correct for the measured verdict,
    but it keeps a race the system does not need to be having.
    """
    old = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if not old or not any(c in old for c in drop_cols):
        return False

    tmp = f"_{table}_new"
    head = f"CREATE TABLE IF NOT EXISTS {table} ("
    start = _DDL.index(head)
    # SLICED OUT OF `_DDL` rather than spelled again: a hand-copied replica is a second source of
    # truth for the schema, and the one thing a rebuild must never do is resurrect a column the
    # DDL has dropped.
    ddl = _DDL[start:_DDL.index("\n);", start) + 3].replace(head, f"CREATE TABLE {tmp} (", 1)

    conn.execute("PRAGMA legacy_alter_table=ON")
    try:
        conn.execute("BEGIN")
        conn.execute(f"DROP TABLE IF EXISTS {tmp}")
        conn.execute(ddl)
        new = {r[1] for r in conn.execute(f"PRAGMA table_info({tmp})")}
        keep = ",".join(c for c in old if c in new and c not in drop_cols)
        conn.execute(f"INSERT INTO {tmp} ({keep}) SELECT {keep} FROM {table}")
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {tmp} RENAME TO {table}")
        for stmt in _ddl_indexes(table):
            conn.execute(stmt)
        for stmt in after:
            conn.execute(stmt)
        conn.commit()
    except Exception:
        # This function opened the transaction, so it closes it. Relying on the connection being
        # discarded would work in `connect()`'s own failure path and nowhere else, and it makes
        # "the rebuild runs entirely or not at all" a property of the caller instead of this unit.
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table=OFF")
    return True


def _drop_sittings_lam(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-08-25: rebuild `sittings` without the `lam` column.

    `lam` was the MMR mix. Chronological admission took MMR's ordering job when the budget cut
    became a time cut (2026-08-24), and the two lines where `lam` still did work went with that
    commit — it survived only in storage, in two hashes and on one display line. Removed rather
    than left inert: a stored dial with a plausible name is what a future reader tunes.

    Every surviving row's `region_key` is restamped, because the key's recipe just lost an input.
    `_backfill_region_keys` does the stamping; nulling the column here is what makes those rows
    visible to it, and it rides in `after=` so it cannot land without the rebuild. `sitting_id` is
    deliberately NOT restamped: it is an opaque event id, and `continues`/`parent_sitting_id`
    reference it as a string.

    Mechanics — portability, the DDL slice, the transaction — live in `_rebuild_without`.
    """
    _rebuild_without(conn, "sittings", "lam",
                     after=("UPDATE sittings SET region_key = NULL",))


def _drop_oracle_ingest_window(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-09-02: rebuild `oracles` without `ingest_from` / `ingest_to`.

    They recorded the window an ingest covered, at the wrong grain: ONE window for a person whose
    sources are pulled independently and fail independently. `set_oracle_window` was called
    unconditionally, outside the try/except that swallowed a failed X pull, so a run whose X
    timeline hit a rate limit still stamped the full requested window as covered — and
    `oracle_refresh_state.seed_from_entities` then copied that stamp into every per-source row,
    where `upsert_source`'s COALESCE made it permanent. The Oracle looked complete and no rail
    would ever pick it up again.

    Coverage now lives per (Oracle x source) in `oracle_sources`, written by whoever performed the
    pull. Removed rather than left inert: a stored column with a plausible name and no writer is
    what a future reader believes.

    Mechanics live in `_rebuild_without`, which is guarded on a read so this is a no-op, and
    takes no write lock, on a converged store.
    """
    _rebuild_without(conn, "oracles", "ingest_from", "ingest_to")


def _drop_frontier_pair_counters(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-09-04: rebuild `frontier_query_sources` without `cursor_ts` /
    `error_count`.

    `cursor_ts`'s only reader was `since_for`'s fallback, and that branch was measured
    UNREACHABLE: the five `record_pull` sites passing `stamp=False` never pass `cursor_ts`, the one
    passing `cursor_ts` also passes `stamp=True`, and the SQL couples them — so no row could exist
    with `cursor_ts` set and `last_pulled_at` NULL. The branch went in b0d1bda2 and left the column
    with no reader at all. `error_count` was read by one test assertion and no production code; the
    escalating back-off it was written for was never built.

    Dropped rather than reconciled. The write was `MAX(COALESCE(?, ''), COALESCE(cursor_ts, ''))`,
    which turns NULL into '', while the sibling implementing the same idea
    (`oracle_refresh_state.record_pull`) writes `COALESCE(?, cursor_ts)` and does not — two copies
    of one concept, already diverged. Deleting the column removes the divergence; fixing the write
    would only re-align two copies that drifted once and would drift again. `last_status` stays: it
    is an operator-readable diagnostic and costs one string.

    Mechanics live in `_rebuild_without`.
    Design record: docs/plans/2026-09-04-nine-open-deletion-decisions.md
    """
    _rebuild_without(conn, "frontier_query_sources", "cursor_ts", "error_count")


def _drop_entities_kind(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-08-23, made PORTABLE 2026-09-05: rebuild `entities` without
    `kind`.

    `kind` held 1140 'person' (a hardcoded constant at 14 ingest sites), 2 'org' and 1 NULL, and
    nothing read it — what a candidate IS is decided by the screen classifier and stored in
    `profile.classified_kind`. The guard `no-entity-kind-column` states the full case.

    WHY THIS IS NOT STILL `ALTER TABLE entities DROP COLUMN kind`. That statement needs
    SQLite >= 3.35 (2021-03), it was the ONLY line in the repository imposing that floor (swept
    2026-09-05; the highest version-gated feature anywhere else is `ON CONFLICT ... DO UPDATE`,
    3.24, 2018, plus FTS5, 3.9, 2015), and it sat two hundred lines below the rule it broke. One
    line was costing three years of SQLite compatibility for a column nothing reads.

    ORDERING IS LOAD-BEARING, and it applies to every caller of `_rebuild_without`: they must run
    in `init_kb_schema`'s REBUILD group, not among the read-guarded drops further down. Those sit
    after an `INSERT OR IGNORE`, which opens an implicit transaction, and `BEGIN` inside one
    raises "cannot start a transaction within a transaction". Measured 2026-09-05: it is the
    INSERT that opens it, not the drops themselves — `DROP TABLE IF EXISTS` leaves
    `in_transaction` False.

    `idx_entities_canonical` is NOT recreated here. `init_kb_schema` creates it further down with
    `IF NOT EXISTS`, which runs after this and so lands on the rebuilt table — which is why the
    `after=` argument is empty for every table but `sittings`.
    """
    _rebuild_without(conn, "entities", "kind")


def _drop_reader_cost_columns(conn: sqlite3.Connection) -> None:
    """Rebuild the two response-receipt tables without their retired dollar field.

    `cost_usd` was a locally derived provider-price estimate, not a product fact. It has no
    reader after the dollar-accounting deletion, while the neighboring model and token metadata
    remains useful operational evidence.

    ONE TRANSACTION PER TABLE now, not one across both. A crash between them leaves the first
    converged and the second not, and the next `connect()` finishes the second — which is a
    better failure than an all-or-nothing that must redo work it already did correctly.
    """
    for table in ("frontier_reader_runs", "sitting_lens_outputs"):
        _rebuild_without(conn, table, "cost_usd")


def _drop_reader_run_windows(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-09-06: rebuild `frontier_reader_runs` without
    `window_from` / `window_to`.

    NEVER WRITTEN. `grep -rn "window_from="` returns nothing across the repository and always
    has; the columns carried no DDL comment either, so no reader could learn what they were for.
    They were substrate for a windowed reader that reads by SITTING instead — `sitting_id` and
    `lens` are how a run says what it covered.

    Removed rather than left inert: a stored column with a plausible name and no writer is what a
    future reader believes, and two NULL date columns beside `ran_at` read as a coverage claim.
    """
    _rebuild_without(conn, "frontier_reader_runs", "window_from", "window_to")


def _drop_read_status(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-09-06: rebuild both read-stamp tables without `read_status`.

    Nothing reads the VALUE. Both callers of `sitting_store.lens_read_state` test the ROW, and
    `_READ_SITTING_IDS` tests presence on both halves of its UNION. The writers only ever wrote
    the literal `'ok'`, and `mark_lens_read`'s `ON CONFLICT` updates only `read_at` — so a
    non-`ok` value, had one ever appeared, would have been PERMANENT with nothing able to correct
    it.

    DO NOT ADD A SECOND WRITABLE VALUE in its place. `b24fba39` removed exactly that: a
    `status="failed"` on the read stamp was a second home for a fact the run log already owns, and
    the only thing it added was a way to report a failure as coverage. The absence of a stamp is
    the failure signal, and it is the one the ledger already reads.

    `sitting_reads.read_status` was NOT NULL, so this rebuild is not optional for that table: the
    column had to go from `_DDL` and the writer together or one of them would refuse the other.
    """
    for table in ("sittings", "sitting_reads"):
        _rebuild_without(conn, table, "read_status")


def _drop_candidate_event_surface(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-09-06: rebuild `frontier_candidate_events` without `surface`.

    WRITE-ONLY, and with one writer passing one constant. Its DDL comment justified it "for stage
    5 attribution", and stage 5 is ruled do-not-build —
    `docs/plans/2026-08-13-…-context-brief.md:204`, under "what NOT to do next", says *"Do not
    build stage 5 (LEARN)."* The column landed one day before that ruling.

    The parameter went with it, down all three of `record_event` / `record_shown` /
    `record_dismissed`, along with the `_SURFACE` constant in `frontier_tools` that its only
    caller passed — a single value naming the tool doing the passing. A parameter with exactly one
    value in the entire codebase does not record a distinction; it records that someone expected
    one.

    (The constant's spelling is deliberately not quoted here. `.guards.py`'s
    `human-attested-stays-human` greps this file line by line for that word in quotes, because
    that is the only signal an AST rule cannot give for tuple membership — and a docstring
    quoting it is the false positive that gets a good rule disabled.)
    """
    _rebuild_without(conn, "frontier_candidate_events", "surface")


def _drop_oracle_follows(conn: sqlite3.Connection) -> None:
    """SUBTRACTIVE migration, 2026-08-26: drop the write-only `oracle_follows` table.

    It was W0 substrate for a follow-graph reader that was never built. Its writer had no caller,
    the CLI the `oracle` tool told users to run had no `__main__` block, nothing ever SELECTed the
    table, and it held 0 rows on the live
    store — the feature never ran outside its own test. Same disposition as the `edges` table
    (deleted 2026-08-23, 5,776 rows and no reader); this one did not even accumulate rows.

    Guarded on a READ so it converges to a no-op and takes no write lock on a converged store —
    see `_rename_crawled_to_footprint` below for why that shape matters here. The X call it fed
    on, `x_graphql_core.fetch_following`, is untouched and still reachable.
    """
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='oracle_follows'"
                    ).fetchone() is None:
        return
    conn.execute("DROP TABLE oracle_follows")


def _rename_crawled_to_footprint(conn: sqlite3.Connection) -> None:
    """Rename the retired `crawled` entry_mode to `oracle-footprint`, 2026-08-25.

    `ingest_github.sync_github` was the only writer. It swept a TRACKED handle's own archive —
    the same act the X and Substack footprint sweeps perform — but stamped a mode outside
    HUMAN_ATTESTED, so a confirmed Oracle's repos were unreachable to every sitting. The
    exclusion was justified by a v1 artifacts sweep that no longer exists; the anti-narrowing
    invariant it claimed to protect rests on `frontier`, not on this mode. Measured before the
    change: folding the affected chunks into the calibration pool moves the floor by +0.001 at
    the production seed, inside the sampler's own seed-to-seed spread. See
    docs/plans/2026-08-25-rename-github-crawled-to-oracle-footprint.md.

    GUARDED ON A READ, deliberately. `_backfill_first_seen` below is a bare UPDATE on every
    connect, and that shape — a converged one-shot backfill still taking a write lock every time
    the store opens — is what
    docs/Future-Investigations/2026-08-25-lock-contention-is-a-migration-backfill-on-every-connect.md
    identifies as the source of this store's `database is locked` errors. The probe here is a
    scan (`entry_mode` is not indexed) but it scans a few thousand rows and takes NO write lock.
    Do not add an index for it.

    A LIVE guarded path, not a one-shot. The MCP server runs from the primary checkout, so an
    older build can still write a `crawled` row after this ships; it heals on the next connect.
    `_backfill_region_keys` states the same precedent.
    """
    if conn.execute("SELECT 1 FROM atoms WHERE entry_mode = 'crawled' LIMIT 1").fetchone() is None:
        return
    conn.execute("UPDATE atoms SET entry_mode = 'oracle-footprint' WHERE entry_mode = 'crawled'")


def _backfill_region_keys(conn: sqlite3.Connection) -> None:
    """Stamp `sittings.region_key` on rows that predate the column. EXACT, not approximate.

    Unlike `_backfill_first_seen` above, this loses nothing: every input to the key is a stored
    column of the row itself, so a recomputed key is bit-identical to one written at build time.

    Guarded on `IS NULL`, and that guard is a LIVE path rather than a one-shot migration. The MCP
    server runs from the primary checkout, so a sitting can be written by an older build after this
    column exists here — it lands with a NULL key and is healed on the next connect from a build
    that knows about it. `_backfill_confirmations` sets the same precedent.

    The other new column, `seed_vector`, is deliberately not backfilled here. For a `vector` seed
    it is unrecomputable (the centroid was never stored); for `query`/`atoms` seeds it needs numpy
    and `sitting_store` (a consumer of this module) to re-centroid chunk vectors, which would
    invert this layer's dependency direction. It heals on demand instead, in the module that owns
    the store: `sitting_store.ensure_seed_vector`.
    """
    # Read POSITIONALLY, like `_ensure_column` above and for the same reason: `init_kb_schema` is
    # public and a caller may hand it a bare connection with no `sqlite3.Row` factory. Named access
    # would work on every path that exists today and break on the first one that does not.
    rows = conn.execute(
        "SELECT sitting_id, seed_kind, seed_ref, floor, ceiling, budget_tokens "
        "  FROM sittings WHERE region_key IS NULL").fetchall()
    conn.executemany(
        "UPDATE sittings SET region_key = ? WHERE sitting_id = ?",
        [(region_key(kind, ref, floor, ceil, budget), sid)
         for sid, kind, ref, floor, ceil, budget in rows])


def init_kb_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe on every connect (CREATE ... IF NOT EXISTS)."""
    conn.executescript(_DDL)
    _ensure_column(conn, "entities", "canonical_id", "TEXT")   # additive for pre-Stage-3 stores
    _ensure_column(conn, "entities", "profile", "TEXT")        # additive for pre-Stage-4 stores
    # Additive for stores whose vectors were built before the embedder got its own surface. NULL
    # reads as "this vector came from `text` verbatim", which is exactly what those rows are.
    _ensure_column(conn, "chunks", "embed_text", "TEXT")
    # Additive for stores written before Frontier grew a second reading rail. Existing rows keep
    # NULL, and every row written before this column existed was a bookmark read — which is why
    # A generator-scoped read counts NULL as a match for that label. See the DDL comment above.
    _ensure_column(conn, "frontier_reader_runs", "generator", "TEXT")
    _ensure_column(conn, "frontier_reader_runs", "sitting_id", "TEXT")
    # Additive for stores written before the lens split (Job L). NULL means 'queries' by
    # construction — every row written before this column existed WAS a queries read, so
    # `sitting_scheduler`'s retry check treats NULL and 'queries' as the same evidence.
    _ensure_column(conn, "frontier_reader_runs", "lens", "TEXT")
    # Additive for stores written before D22's positional-coverage metric got its own column (Job N).
    # NULL on every pre-existing row means "never measured", not "measured as zero" — there is no
    # backfill, because the render each of those runs actually saw is gone.
    _ensure_column(conn, "frontier_reader_runs", "middle_share", "REAL")
    # Additive for stores written before survival became an explicit verdict. No data migration is
    # needed: `status` gained a value ('retired') and lost one ('dormant'), and the live store holds
    # zero dormant rows — `active_queries()` keys on `!= 'retired'`, so any pre-existing status
    # keeps executing rather than being silently dropped.
    for col in ("kept", "dropped", "unverdicted"):
        _ensure_column(conn, "frontier_reader_runs", col, "INTEGER")
    # Additive for stores written before Frontier stage 3 (ADMIT) existed. Every pre-existing row
    # is `status='new'` with no attempt history — which is exactly what the defaults describe, so
    # there is no data migration here and never was one to skip.
    _ensure_column(conn, "frontier_candidates", "attempts", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "frontier_candidates", "last_error", "TEXT")
    _ensure_column(conn, "frontier_candidates", "last_attempt_at", "TEXT")
    # Additive for stores written before the finder/minter split. NO BACKFILL, and that is a
    # decision rather than an omission: `frontier_admit` reads `kind` only under
    # `WHERE status = 'new'`, and every pre-existing row is `materialized` (verified 2026-08-26 on
    # the live store — 111 of 111), so a NULL `kind` on those rows is never read. A backfill here
    # would also be the exact shape measured as the cause of the live lock contention in
    # docs/Future-Investigations/2026-08-25-lock-contention-is-a-migration-backfill-on-every-connect.md
    # — one converged migration re-running on every connect, for rows nothing reads.
    _ensure_column(conn, "frontier_candidates", "kind", "TEXT")
    # Additive for stores written before budget-stopped regions were read in parts. NULL `continues`
    # is a first part, which is what every pre-existing row is.
    _ensure_column(conn, "sittings", "continues", "TEXT")
    _ensure_column(conn, "sittings", "prior_atoms", "INTEGER NOT NULL DEFAULT 0")
    # Additive for stores written before regions could be fractured. NULL = not a sub-sitting, which
    # is what every pre-existing row is.
    _ensure_column(conn, "sittings", "parent_sitting_id", "TEXT")
    # Additive: the two columns that make a region — as opposed to one BUILD of it — addressable.
    # `region_key` is recomputed for every pre-existing row; `seed_vector` is not, and the docstring
    # on the backfill says why that asymmetry is deliberate rather than half-finished.
    _ensure_column(conn, "sittings", "region_key", "TEXT")
    _ensure_column(conn, "sittings", "seed_vector", "BLOB")
    # Additive: the skip list behind `skipped_dupes`. Pre-existing rows kept only the count, and
    # their atom-level detail is unrecoverable — NULL, not `[]`, so "built before the column" stays
    # distinguishable from "skipped nothing".
    _ensure_column(conn, "sittings", "skipped", "TEXT")
    # Subtractive, and it must run BEFORE the backfill below: the rebuild nulls every `region_key`
    # so the backfill restamps them under the recipe that no longer takes `lam`.
    _drop_sittings_lam(conn)
    _drop_oracle_ingest_window(conn)
    # Grouped with the two rebuilds above, not with the read-guarded drops further down:
    # a create-copy-drop-rename opens its OWN transaction, so it must run while no implicit
    # one is already open.
    _drop_frontier_pair_counters(conn)
    # Grouped here for the same reason, and it was NOT here until 2026-09-05: this migration used
    # to be a bare `ALTER TABLE entities DROP COLUMN kind` among the drops at the bottom of this
    # function. See the docstring for why that line was a portability bug and why a rebuild cannot
    # live down there.
    _drop_entities_kind(conn)
    _drop_reader_cost_columns(conn)
    # Three more rebuilds, 2026-09-06, in the same group and for the same reason: each opens its
    # own transaction, so all of them must run before the first `INSERT OR IGNORE` below opens an
    # implicit one.
    _drop_reader_run_windows(conn)
    _drop_read_status(conn)
    _drop_candidate_event_surface(conn)
    _backfill_region_keys(conn)
    # Additive: when a FULL-SET collector last re-saw this signal. Only the full-set collectors'
    # ever carry it — `save` stays permanently NULL by construction, because no path re-reads your
    # bookmark set to check a bookmark is still there. (The fuller investigation doc was
    # deleted 2026-08-16 in the open-source cleanup; git history has it.)
    _ensure_column(conn, "curation_signals", "last_confirmed_at", "TEXT")
    _backfill_confirmations(conn)
    # Additive: the arrival date `ingested_at` stopped being able to answer. No default is attached
    # here and that is not an oversight — SQLite forbids a non-constant default in ADD COLUMN, so
    # `datetime('now')` is illegal on this path while it is legal in the `_DDL` above. That leaves
    # a migrated store's column defaultless and a fresh store's not, so `upsert_atom` supplies the
    # value EXPLICITLY rather than leaning on either. Both store shapes then behave identically.
    _ensure_column(conn, "atoms", "first_seen", "TEXT")
    _backfill_first_seen(conn)
    # A RENAME, not a column change: GitHub's footprint sweep used to stamp a mode outside
    # HUMAN_ATTESTED. Read-guarded so it costs a scan and no write lock once converged.
    _rename_crawled_to_footprint(conn)
    # Subtractive: a W0 substrate table whose reader was never built. Read-guarded like the
    # rename above, so a converged store pays one catalogue lookup and no write lock.
    _drop_oracle_follows(conn)
    # Additive: the promotion stamp. No backfill and none possible — nothing promoted before the
    # primitive existed, so NULL is the true value on every pre-existing row.
    _ensure_column(conn, "atoms", "promoted_at", "TEXT")
    # Additive: the emission lane. NULL on every pre-existing row and that is the true value —
    # they were all emitted when regions held human-attested atoms only.
    _ensure_column(conn, "frontier_queries", "lane", "TEXT")
    # Additive for registries written before a peer could be served over HTTPS. NULL is the true
    # value on every pre-existing row: they are all local files, and a file needs no bearer token.
    _ensure_column(conn, "peers", "token", "TEXT")
    # Additive for stores written before a query could be claimed by more than one generator. Every
    # pre-existing row had exactly one claimant by construction, so its `generator` column IS that
    # claim and its counter carries over. Earlier co-claimants are unrecoverable — the column had
    # already overwritten them, which is the bug the table exists to end. `INSERT OR IGNORE` makes
    # this idempotent on every connect and unable to disturb a claim already recorded.
    #
    # READ-GUARDED for the reason `_backfill_confirmations` records: `INSERT OR IGNORE` is
    # idempotent but not free — it takes a write lock even when it inserts nothing, and seven rails
    # connect within milliseconds of each other. The probe rides the (query_id, generator) primary
    # key, and every source column is NOT NULL, so a row that fails the probe is always insertable
    # and this converges instead of re-taking the lock forever.
    unclaimed = conn.execute(
        """SELECT 1 FROM frontier_queries q
            WHERE NOT EXISTS (SELECT 1 FROM frontier_query_generators g
                               WHERE g.query_id = q.query_id AND g.generator = q.generator)
            LIMIT 1""").fetchone()
    if unclaimed is not None:
        conn.execute(
            """INSERT OR IGNORE INTO frontier_query_generators
                 (query_id, generator, first_emitted_at, last_emitted_at, miss_count)
               SELECT query_id, generator, created_at, last_emitted_at, miss_count
                 FROM frontier_queries""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frontier_runs_status_time "
                 "ON frontier_reader_runs(status, ran_at)")
    # AFTER `_ensure_column`, for the reason spelled out next to the `sittings` DDL: on an older
    # store the column does not exist until the lines above run.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sittings_continues ON sittings(continues)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sittings_parent ON sittings(parent_sitting_id)")
    # `MAX(read_at) GROUP BY region_key` is the re-read trigger's hot query, and it is a GROUP BY
    # over the whole table rather than a lookup of one row.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sittings_region ON sittings(region_key)")
    # Subtractive migration, 2026-08-23: `entity_trust` held ten hand-seeded 1.0 rows and had no
    # reader — `trust_weight` was never called from production and `propagate()` was never built.
    # Dropped rather than left inert: an unread table with a plausible name is what a future reader
    # trusts. Design record: docs/plans/2026-08-23-delete-edges-and-trust-tiers.md
    conn.execute("DROP TABLE IF EXISTS entity_trust")
    # Same migration, same reason: `edges` accumulated 5,776 rows across six relation types and no
    # code ever SELECTed from it. Its two intended consumers — trust propagation and an
    # Oracle-referenced-person candidate channel — were ruled dead the same day.
    conn.execute("DROP TABLE IF EXISTS edges")
    # Subtractive migration, 2026-09-04: `sync_health` (created by `dedup_store`, written on every
    # Oracle refresh) recorded per-source failure detail under the key
    # `oracle-refresh:<canonical_id>:<source_type>` — byte-identical to the CircuitBreaker key
    # written on the adjacent line with the same string. `circuit_breaker.last_failure` already
    # holds it and clears on success identically; `oracle_sources.last_status` holds the per-pair
    # outcome and `oracle_refresh.status_summary` already ships it to the `screen` tool. Dropped
    # here rather than in `dedup_store._connect` so the retired-table list has one home and the
    # dedup hot path pays nothing. Design record: docs/plans/2026-09-04-nine-open-deletion-decisions.md
    conn.execute("DROP TABLE IF EXISTS sync_health")
    # `entities.kind` was dropped here by a bare `ALTER TABLE ... DROP COLUMN` until 2026-09-05.
    # It is now `_drop_entities_kind`, in the rebuild group above — that statement needs
    # SQLite >= 3.35 and the distributability invariant forbids assuming one. Design record:
    # docs/plans/2026-08-23-candidate-search-atom-arm.md
    conn.commit()


def connect(db_path: Path | str | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the atom-KB store — the machine-canonical `~/.opyt/opyt.db` by default
    (honoring `$OPYT_HOME`); pass `db_path` to sandbox a test.

    Writable opens run `init_kb_schema`, so a caller never has to create a table itself. That
    is NOT the same as "callers never race a missing table", which this said until 2026-09-06 and
    which stopped being true on 2026-08-25 18:12, when `153c7d21` put the first non-idempotent
    REBUILD on this path: a rebuild DROPs its table and re-creates it, so a second writable open
    arriving inside that window sees the table gone. (A stale-prose sweep re-canonised the old
    sentence five hours later the same night. The sweep read it, and it read as current.)

    Read-only opens do NOT run DDL — the retrieval path is read-only and must degrade to
    "no such table" (caught upstream → empty result), never create.
    """
    p = Path(db_path) if db_path else opyt_db()
    if read_only:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    # busy_timeout FIRST, and the order is load-bearing. `journal_mode=WAL` takes a brief
    # exclusive lock, so with the default timeout of 0 it errors IMMEDIATELY rather than waiting
    # when another process holds one — the one statement most likely to collide was the one
    # statement running unprotected. Found while serving two concurrent uploads.
    conn.execute("PRAGMA busy_timeout=5000")    # wait out a concurrent writer, don't error
    conn.execute("PRAGMA journal_mode=WAL")     # readers don't block the writer
    conn.row_factory = sqlite3.Row
    init_kb_schema(conn)
    return conn


def _encode_atom(atom: dict) -> list:
    """One atom dict → positional values for `_ATOM_COLS`, JSON-encoding list/dict fields."""
    out = []
    for c in _ATOM_COLS:
        v = atom.get(c)
        if c in _JSON_ATOM_FIELDS and v is not None and not isinstance(v, str):
            v = json.dumps(v)
        out.append(v)
    return out


def upsert_atom(conn: sqlite3.Connection, atom: dict) -> None:
    """Insert one atom, overwriting in place on `atom_id` conflict and BUMPING `version`.

    Idempotent by identity: a changed source (new `raw_hash`) replaces the row and
    increments `version` (an audit trail of how many times this atom was re-observed).
    `ingested_at` refreshes to now. Missing keys default to NULL. Note the CALLER is
    responsible for the hash-skip (don't call this for an unchanged atom) — this always
    writes and always bumps.

    `first_seen` is the one column that does NOT refresh: it is written in the INSERT arm and is
    absent from the SET arm, so the arrival date survives every re-observation. It is passed as a
    SQL literal rather than left to the column default because a store migrated by
    `_ensure_column` has no default to fall back on (SQLite forbids one there) — see the note at
    the ALTER site. Keep it out of `_ATOM_UPDATABLE`.
    """
    values = _encode_atom(atom)
    placeholders = ", ".join("?" for _ in _ATOM_COLS)
    set_clause = ", ".join(f"{c}=excluded.{c}" for c in _ATOM_UPDATABLE)
    conn.execute(
        f"INSERT INTO atoms ({', '.join(_ATOM_COLS)}, first_seen) "
        f"VALUES ({placeholders}, datetime('now')) "
        f"ON CONFLICT(atom_id) DO UPDATE SET {set_clause}, "
        f"version=atoms.version+1, ingested_at=datetime('now')",
        values,
    )
    conn.commit()


def replace_chunks(conn: sqlite3.Connection, atom_id: str, chunks: list[dict]) -> None:
    """Replace ALL chunks for one atom, keeping `chunks_fts` in sync. The caller commits.

    Full replace, not per-seq upsert: when a snapshot changes, chunk BOUNDARIES shift, so
    a stale seq-3 from the old text would otherwise linger. Each chunk dict:
    {seq, char_start, char_end, text, embed_text(str|None), vector(bytes|None)}. The FTS row
    carries the same text + the freshly-minted chunk_id so the semantic-arm argmax can name its span.

    `embed_text` is OPTIONAL on the dict — a caller that omits it (any pre-embed_surface writer,
    and every test that builds chunks by hand) stores NULL, which every reader treats as "the
    vector came from `text`". The FTS row deliberately keeps `text`, not `embed_text`: BM25
    discounts a term that appears in half the corpus on its own, so the boilerplate that skews the
    vector arm was never skewing this one.
    """
    conn.execute("DELETE FROM chunks_fts WHERE atom_id = ?", (atom_id,))
    conn.execute("DELETE FROM chunks WHERE atom_id = ?", (atom_id,))
    for ch in chunks:
        cur = conn.execute(
            "INSERT INTO chunks (atom_id, seq, char_start, char_end, text, embed_text, vector) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (atom_id, ch["seq"], ch.get("char_start"), ch.get("char_end"),
             ch["text"], ch.get("embed_text"), ch.get("vector")),
        )
        conn.execute(
            "INSERT INTO chunks_fts (text, atom_id, chunk_id) VALUES (?, ?, ?)",
            (ch["text"], atom_id, cur.lastrowid),
        )


def upsert_entity(conn: sqlite3.Connection, entity_id: str, name: str | None = None,
                  identity_links: list | None = None,
                  profile: dict | None = None) -> None:
    """UPSERT an entity. name/identity_links COALESCE (a non-null new value wins, else
    the stored value survives). `profile` is MERGED, not overwritten, via json_patch: the
    ingest-time write ({bio, verified, followers}) and the later classifier write
    ({classified_kind, classified_at}) each land without clobbering the other's keys. A null
    profile is a no-op patch (keeps whatever is stored).

    `identity_links` took `list | dict` until 2026-09-06 and no caller ever passed a dict —
    `git log -S'identity_links={' --all` is empty across all history, and all 17 live call sites
    pass a one-string list or None. The annotation was not merely unused: a dict would have
    serialized to a JSON OBJECT, and all four readers of this column
    (`screen._loads`, `resolve._loads`, `oracle_refresh_state._links`, `expand._first_url`)
    decode an ARRAY. The column has exactly one writer and holds only NULL or '[…]', which is
    why those four agree today; the wider annotation was an invitation to write the one shape
    they do not."""
    links = json.dumps(identity_links) if identity_links is not None else None
    prof = json.dumps(profile) if profile is not None else None
    conn.execute(
        "INSERT INTO entities (entity_id, name, identity_links, profile) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(entity_id) DO UPDATE SET "
        "name=COALESCE(excluded.name, entities.name), "
        "identity_links=COALESCE(excluded.identity_links, entities.identity_links), "
        # json_patch(a, b) = RFC-7386 merge of b INTO a; a null b keeps a unchanged.
        "profile=json_patch(COALESCE(entities.profile, '{}'), COALESCE(excluded.profile, '{}'))",
        (entity_id, name, links, prof),
    )
    conn.commit()


def set_entity_profile(conn: sqlite3.Connection, entity_id: str, patch: dict) -> None:
    """Merge `patch` into an entity's `profile` JSON (json_patch) WITHOUT touching name/
    links. The classifier uses this to cache its verdict ({classified_kind, classified_at}) on
    the canonical entity, so a re-screen skips already-classified candidates. No-op-safe: an
    entity_id with no row is silently ignored (UPDATE affects 0 rows)."""
    conn.execute(
        "UPDATE entities SET profile=json_patch(COALESCE(profile, '{}'), ?) WHERE entity_id=?",
        (json.dumps(patch), entity_id),
    )
    conn.commit()


def signals_with_canonical(conn: sqlite3.Connection) -> list:
    """Every curation signal joined to its entity, carrying the resolution head. The Stage-4
    ranking input: one row per (entity, signal_type, platform) with the entity's canonical_id
    (COALESCE to entity_id when Stage-3 hasn't run / a singleton), name, identity_links,
    and profile — so the ranker can GROUP BY canonical and reflect the user's own signals back."""
    return conn.execute(
        "SELECT s.entity_id, s.signal_type, s.platform, s.count, s.extra, "
        "       s.last_confirmed_at, "
        "       COALESCE(e.canonical_id, s.entity_id) AS canonical_id, "
        "       e.name, e.identity_links, e.profile "
        "FROM curation_signals s "
        "LEFT JOIN entities e ON e.entity_id = s.entity_id"
    ).fetchall()


def get_entity(conn: sqlite3.Connection, entity_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT entity_id, name, identity_links, canonical_id, profile "
        "FROM entities WHERE entity_id=?", (entity_id,)
    ).fetchone()


def current_canonical(conn: sqlite3.Connection, entity_id: str) -> str:
    """Follow a possibly-STALE canonical_id/anchor to its CURRENT cluster head. `resolve`
    recomputes each entity's `canonical_id` (= min entity_id of its component), so a value stored
    earlier — e.g. an oracle's confirm-time `canonical_id` — goes stale when a later footprint
    merge changes the head (a `blog:`/`substack:` member can sort below the `x:user:` head). The
    stored value is always still a real member entity, so its own `canonical_id` column lands on
    the current head. Fail-safe: an unknown id resolves to itself."""
    row = conn.execute(
        "SELECT COALESCE(canonical_id, entity_id) FROM entities WHERE entity_id=?", (entity_id,)
    ).fetchone()
    return row[0] if row else entity_id


def entities_for_canonical(conn: sqlite3.Connection, canonical_id: str) -> list:
    """Every member entity of one resolved cluster (its per-platform rows). Stage-5 reads a
    confirmed oracle's members' `identity_links` to know which footprints to expand. Resolves a
    stale/anchor `canonical_id` to the current head first, so a post-resolve head-drift doesn't
    orphan a confirmed oracle (see `current_canonical`)."""
    head = current_canonical(conn, canonical_id)
    return conn.execute(
        "SELECT entity_id, name, identity_links, profile "
        "FROM entities WHERE COALESCE(canonical_id, entity_id)=?", (head,)
    ).fetchall()


# ── Oracles (Stage-4 output) ──────────────────────────────────────────────────
def upsert_oracle(conn: sqlite3.Connection, canonical_id: str, name: str | None = None,
                  source: str = "screen") -> None:
    """Confirm one Oracle (idempotent on canonical_id). Re-confirming refreshes name/source but
    NEVER resets confirmed_at or the reserved `paused` (a re-confirm isn't a re-add)."""
    conn.execute(
        "INSERT INTO oracles (canonical_id, name, source) VALUES (?, ?, ?) "
        "ON CONFLICT(canonical_id) DO UPDATE SET "
        "name=COALESCE(excluded.name, oracles.name), source=excluded.source",
        (canonical_id, name, source),
    )
    conn.commit()


def reanchor_oracles(conn: sqlite3.Connection) -> int:
    """Re-point every `oracles` row at its cluster's CURRENT head. Returns how many moved.

    ⚠️ THE STALE HEAD WAS A SILENT WRONG ANSWER IN THREE PLACES, measured 2026-09-14 on a live
    store. `oracles.canonical_id` is written at confirm time; a later footprint merge can move the
    head (a `blog:`/`substack:` member sorts below an `x:user:` one), and nothing rewrote the row.
    `current_canonical` exists to absorb that on READ, and `confirmed_oracles` remembers to call
    it — but every OTHER join on `oracles.canonical_id` silently misses:

      • `kb.kb_aggregate`'s `trusted_atoms` joined `oracles` directly and reported 80 trusted
        atoms against a true 884. Three of five Oracles counted as zero.
      • `oracle_refresh_state`'s source listing LEFT JOINs `oracles` for the display name, so
        every merged Oracle's pairs dispatched with `author_name=None` — which is why 182 of
        Dwarkesh Patel's posts are attributed to "substack" rather than to him.
      • The same stale id got a SECOND `oracle_sources` registration, because `seed_from_entities`
        registers under the current head while pre-merge rows kept the old one.

    Fixing the reads one at a time would have left the next join to discover it again. The row is
    what is wrong, so the row is what this repairs.

    A MERGE CAN COLLIDE: two separately-confirmed Oracles that turn out to be one person now share
    a head, and `canonical_id` is the primary key. The oldest confirmation wins the row (it is the
    one whose `confirmed_at` the user would recognise) and the other is deleted, not renamed onto
    a duplicate key. `INSERT OR REPLACE` would silently pick by insertion order instead.
    """
    moved = 0
    for row in conn.execute("SELECT canonical_id, name, confirmed_at FROM oracles").fetchall():
        stale = row["canonical_id"]
        head = current_canonical(conn, stale)
        if head == stale:
            continue
        existing = conn.execute(
            "SELECT confirmed_at FROM oracles WHERE canonical_id=?", (head,)).fetchone()
        if existing is None:
            conn.execute("UPDATE oracles SET canonical_id=? WHERE canonical_id=?", (head, stale))
        elif (row["confirmed_at"] or "") < (existing["confirmed_at"] or ""):
            conn.execute("DELETE FROM oracles WHERE canonical_id=?", (head,))
            conn.execute("UPDATE oracles SET canonical_id=? WHERE canonical_id=?", (head, stale))
        else:
            # The head row is the older confirmation; keep it, and take the name if it has none.
            conn.execute("UPDATE oracles SET name=COALESCE(name, ?) WHERE canonical_id=?",
                         (row["name"], head))
            conn.execute("DELETE FROM oracles WHERE canonical_id=?", (stale,))
        moved += 1
    if moved:
        conn.commit()
    return moved


def list_oracles(conn: sqlite3.Connection) -> list:
    """Every confirmed Oracle, newest first — what Stage 5 consumes."""
    return conn.execute(
        "SELECT canonical_id, name, source, confirmed_at, paused "
        "FROM oracles ORDER BY confirmed_at DESC, canonical_id"
    ).fetchall()


def is_oracle(conn: sqlite3.Connection, canonical_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM oracles WHERE canonical_id=?", (canonical_id,)
    ).fetchone() is not None


def all_entities(conn: sqlite3.Connection) -> list:
    """Every entity row (entity_id, name, identity_links, canonical_id) — the input
    Stage-3 resolution consumes to compute `canonical_id`, and Stage-4 to rank candidates."""
    return conn.execute(
        "SELECT entity_id, name, identity_links, canonical_id FROM entities"
    ).fetchall()


def set_canonical_ids(conn: sqlite3.Connection, mapping: dict) -> None:
    """Bulk-write each entity's resolved `canonical_id` in ONE transaction. Idempotent:
    Stage-3 recomputes the whole mapping each run and overwrites — `canonical_id` is a
    materialized VIEW of the identity_links graph, never a source of truth (atoms key on the
    stable per-platform `who_id`, so re-pointing it breaks nothing downstream)."""
    conn.executemany(
        "UPDATE entities SET canonical_id=? WHERE entity_id=?",
        [(cid, eid) for eid, cid in mapping.items()],
    )
    conn.commit()


def add_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str,
               platform: str, count: int = 1, extra: dict | str | None = None) -> None:
    """Record one curation signal, SUMMING `count` into the (entity, signal_type, platform)
    row on conflict. Two separate `like` writes on one entity → count=2; a `like` and a
    `follow` on the same entity coexist as two rows (signal_type is in the key). `extra` is
    JSON-encoded; the newer non-null `extra` wins on conflict (COALESCE), so a re-observation
    can refresh the context without a null clobbering it.

    SUM (not overwrite) is deliberate for an event: `count` is cumulative action strength, and a
    caller streaming one write per atomic action is exactly the shape SUM is right for. Its two
    live callers are both that shape — `ingest_x.sync_bookmarks` and `ingest_curation.
    sync_substack_saved` each stamp `save` ONCE per atom, on the run that first ingests it.

    Use `set_signal` for a full-set re-read. A caller that walked the whole list already holds
    the person's TOTAL, and summing a total into a total is what inflated the live store's
    `follow/x` from 468 to 886 in one pass. See `set_signal` for the full story.
    """
    _write_signal(conn, entity_id, signal_type, platform, count, extra,
                  "count = curation_signals.count + excluded.count")


def set_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str,
               platform: str, count: int = 1, extra: dict | str | None = None) -> None:
    """Record one curation signal, REPLACING `count` on conflict. Idempotent by construction:
    writing the same observation twice leaves the same row.

    Exists as a second function (not a flag on `add_signal`) because the five people-only
    collectors (`x_lists`, `x_following`, `x_likes`, `substack_follows`, `substack_subscriptions`)
    are FULL-SET re-reads: each
    hands in a person's whole aggregate, not a delta, so SUMming it (as `add_signal` does) double-
    counts on every re-run. That double-counting actually happened once `curation_catchup` made
    these collectors run automatically. Self-heals what it sees on the next full-set pass;
    a person absent from a walk (e.g. unfollowed) keeps whatever count they last received.

    Two functions, not a `mode=` argument, following the same rule `AtomSink` follows by taking a
    `writer=` function rather than a `table=` string: there is no argument a caller can get wrong.
    """
    _write_signal(conn, entity_id, signal_type, platform, count, extra,
                  "count = excluded.count", confirm=signal_type in CONFIRMABLE_SIGNALS)


def ensure_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str,
                  platform: str, count: int = 1, extra: dict | str | None = None) -> None:
    """Record one curation signal, LEAVING an existing row untouched. Presence, not counting.

    The third operator, and the one for a caller that knows the signal is TRUE but is not the
    authority on how strong it is. `save` is the type that needs it: the count means "how many of
    their posts you saved", and only a caller that walked the whole saved list can say that — so a
    per-atom writer summing 1 into the row (`add_signal`) inflates the very number the screen
    shows the user as "bookmarked 12x", while `set_signal` would flatten it to 1.

    This is what `reconcile_saved_signals` open-coded in SQL for exactly this reason ("Uses
    insert-if-absent, not `add_signal`, which sums `count` and would inflate on every call"). It is
    a function now because a second caller needed it: `ingest_x.sync_bookmarks` stamps once per
    atom against a corpus re-walked forever, and `sync_bookmark_signals` — the full-set walk — is
    the counting authority it must not fight.

    Never CONFIRMS (`set_signal`'s job), because a caller that cannot count the set cannot say
    what is absent from it either.
    """
    _write_signal(conn, entity_id, signal_type, platform, count, extra,
                  "count = curation_signals.count")


def _write_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str, platform: str,
                  count: int, extra: dict | str | None, count_clause: str,
                  confirm: bool = False) -> None:
    """The shared body of `add_signal` / `set_signal` — identical but for the count operator and
    whether the write counts as a CONFIRMATION.

    `extra` is JSON-encoded; the newer non-null `extra` wins on conflict (COALESCE), so a
    re-observation can refresh the context without a null clobbering it. That half is the same
    under both operators: `extra` was never additive.

    `confirm` rides on `set_signal` alone, because the two go together by definition — only a
    caller that walked the WHOLE list can say "this signal is still true", and that is exactly the
    caller `set_signal` exists for."""
    extra_json = (json.dumps(extra) if extra is not None and not isinstance(extra, str)
                  else extra)
    stamp = "datetime('now')" if confirm else "curation_signals.last_confirmed_at"
    # The INSERT arm's value for the same column. Named for the column it fills — it was called
    # `first_seen` until 2026-09-06, collateral of a mechanical py3.12 compat hoist, and
    # `curation_signals` has no such column: anyone auditing "who writes `atoms.first_seen`" got
    # a hit in the one function that does not.
    #
    # Hoisted for the same reason `stamp` above it is, plus one more: inlined, the conditional
    # needs a backslash-escaped quote INSIDE an f-string expression, which is PEP 701 syntax and
    # a hard SyntaxError before 3.12. pyproject declares `requires-python = ">=3.10"`, so the
    # inline form made this module unimportable on two of the three versions it claims.
    confirmed_now = "datetime('now')" if confirm else "NULL"
    conn.execute(
        "INSERT INTO curation_signals (entity_id, signal_type, platform, count, extra, "
        "                              last_confirmed_at) "
        f"VALUES (?, ?, ?, ?, ?, {confirmed_now}) "
        "ON CONFLICT(entity_id, signal_type, platform) DO UPDATE SET "
        f"{count_clause}, "
        f"last_confirmed_at = {stamp}, "
        "extra = COALESCE(excluded.extra, curation_signals.extra)",
        (entity_id, signal_type, platform, int(count), extra_json),
    )
    conn.commit()


def record_engagements(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Write engagement observations, idempotent on the natural key (observer, kind, target,
    src_ref). Returns how many rows were actually NEW — a re-run over the same pull
    returns 0. INSERT OR IGNORE (never upsert): the first observation's `observed_at` wins,
    so a back-mine can't overwrite a live capture's date or vice versa."""
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO engagements "
        "(observer_id, kind, target_id, src_ref, observed_at) VALUES (?, ?, ?, ?, ?)",
        [(r["observer_id"], r["kind"], r["target_id"], r["src_ref"], r.get("observed_at"))
         for r in rows],
    )
    conn.commit()
    return conn.total_changes - before


def load_hashes(conn: sqlite3.Connection, source_type: str) -> dict[str, str]:
    """`{atom_id: raw_hash}` for one source — the idempotency ledger an ingester consults
    to skip unchanged sources (embed iff new OR changed, never re-pay for identical raw)."""
    return {
        row["atom_id"]: row["raw_hash"]
        for row in conn.execute(
            "SELECT atom_id, raw_hash FROM atoms WHERE source_type=?", (source_type,)
        )
    }


def load_body_pending(conn: sqlite3.Connection, source_type: str) -> set[str]:
    """Atom ids stored WITHOUT their body because a fetch was BLOCKED, not because the source
    has no body — the narrow exception to `load_hashes`'s "already seen → skip" rule.

    A stub whose body is merely absent (podcast, link post) will never gain one, so re-fetching
    it forever is waste. A stub caused by a block WILL succeed once the block clears, and the
    plain `atom_id in seen` check freezes that temporary failure into a permanent hole. Writers
    set `payload.body_state = 'pending'` only on an UNDETERMINED verdict, which is the whole
    reason the three-verdict fetch contract distinguishes "stopped" from "empty".

    A legacy `body_pending` boolean spelling (written before `body_state` existed, pre-754a6440
    2026-08-03) was read here too until 2026-08-29. Its stated removal condition — zero
    legacy-key atoms — was met and measured: the live store held 0, and no other store ever ran
    the old writer (distribution postdates it).

    Self-clearing: a successful re-fetch rewrites the atom as `complete`, so it drops out of
    this set without anyone having to remember to remove it."""
    return {
        row["atom_id"]
        for row in conn.execute(
            "SELECT atom_id FROM atoms WHERE source_type=? AND "
            "json_extract(payload, '$.body_state') = 'pending'",
            (source_type,),
        )
    }


def count_atoms(conn: sqlite3.Connection, source_type: str | None = None) -> int:
    if source_type is None:
        return conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE source_type=?", (source_type,)
    ).fetchone()[0]


# ── Footprint-eligibility cache (Stage-5 single-author gate) ────────────────────
def get_authorship(conn: sqlite3.Connection, source_url: str) -> sqlite3.Row | None:
    """The cached authorship verdict for a canonical site key, or None on a miss. The gate
    consults this BEFORE any fetch/classify, so a source is classified once and reused forever."""
    return conn.execute(
        "SELECT source_url, authorship, author_name, classified_at "
        "FROM source_authorship WHERE source_url=?", (source_url,)
    ).fetchone()


def put_authorship(conn: sqlite3.Connection, source_url: str, authorship: str,
                   author_name: str | None = None) -> None:
    """Cache a DEFINITIVE authorship verdict (idempotent on the site key; a re-classify refreshes
    the verdict + `classified_at`). Callers must NEVER write an `unknown` here — a transient
    fetch/LLM failure has no place in a permanent cache (the fail-safe invariant)."""
    conn.execute(
        "INSERT INTO source_authorship (source_url, authorship, author_name) VALUES (?, ?, ?) "
        "ON CONFLICT(source_url) DO UPDATE SET authorship=excluded.authorship, "
        "author_name=excluded.author_name, classified_at=datetime('now')",
        (source_url, authorship, author_name),
    )
    conn.commit()
