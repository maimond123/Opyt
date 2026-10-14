"""
pipeline/kb/schema.py — the atom-KB store: DDL and the writers every ingest shares.

Atoms, their bodies, entities, and the Frontier and topic tables, in the one machine-canonical DB
(`~/.opyt/opyt.db`, honoring `$OPYT_HOME`).

An atom is a thin routing card: a pointer to a live source (`source_url`), its stored snapshot
(one `bodies` row, hashed as `raw_hash`) and a mechanical `description`, never interpretation.
Trust invariant: nothing asserts what a source says without opening its body.

  • Identity is `atom_id`; `raw_hash = sha256(snapshot)` is the change detector, not the identity.
  • The DDL describes the current shape only: no migration steps; an older store is re-onboarded.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from opyt_core.paths import opyt_db, owner_only
from pipeline.kb.writes import write_tx

# ── DDL ───────────────────────────────────────────────────────────────────────
# The search index over `bodies` and the triggers that keep it in step. Its own constant so an
# export, which copies `bodies` as rows, builds the same index from the same text.
BODIES_INDEX_DDL = """
-- The BM25 index over each whole body, external-content: FTS5 reads the text from `bodies`, so it
-- is stored once. The three triggers keep the index in step with every write to `bodies` (the
-- ingest, `forget`, a copy's patch, a hand edit), so no writer needs to know the index exists.
-- Never `INSERT OR REPLACE INTO bodies`: REPLACE deletes the old row without firing `bodies_ad`,
-- and its text would stay indexed (`.guards.py` `bodies-index-by-trigger`). Porter-stemmed, so
-- "accountable" finds "accountability".
CREATE VIRTUAL TABLE IF NOT EXISTS bodies_fts
  USING fts5(text, content='bodies', content_rowid='body_id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS bodies_ai AFTER INSERT ON bodies BEGIN
  INSERT INTO bodies_fts(rowid, text) VALUES (new.body_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS bodies_ad AFTER DELETE ON bodies BEGIN
  INSERT INTO bodies_fts(bodies_fts, rowid, text) VALUES ('delete', old.body_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS bodies_au AFTER UPDATE ON bodies BEGIN
  INSERT INTO bodies_fts(bodies_fts, rowid, text) VALUES ('delete', old.body_id, old.text);
  INSERT INTO bodies_fts(rowid, text) VALUES (new.body_id, new.text);
END;
"""

# `when` is a SQL keyword → the temporal column is `when_ts`. JSON fields
# (`about_entities`/`payload`/`identity_links`) are stored as TEXT and read back with json.loads.
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
  raw_hash       TEXT,                 -- sha256(the `bodies` text) : idempotency + change key
  description    TEXT,                 -- MECHANICAL only (never interpretive)
  payload        TEXT,                 -- JSON of structural fields (counts, lang, …)
  entry_mode     TEXT,                 -- how the atom entered, not what it is (that's source_type).
                                       -- Live: 'user-saved' (the user personally saved it) |
                                       -- 'oracle-footprint' (a tracked person authored it) |
                                       -- 'author_referenced' (a tracked person pointed at it).
                                       -- Free text, no CHECK — so
                                       -- readers must allow-list the modes they want, never
                                       -- deny-list, and a new mode is excluded by default instead
                                       -- of by remembering.
                                       -- The frontier mode (written only by Frontier stage 3, see
                                       -- pipeline/frontier/admit.py) records how an atom was
                                       -- found, never whether a human blessed it: a machine find
                                       -- tagged 'user-saved' would read as something the user
                                       -- chose. Approval lives in frontier_candidates.status,
                                       -- never here. It is excluded from HUMAN_ATTESTED below.
  version        INTEGER NOT NULL DEFAULT 1,
  ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),  -- last observed at, not first. See below.
  -- When this atom first arrived. Written on INSERT, never on UPDATE — that is the entire point of
  -- the column. `ingested_at` cannot answer this question: `upsert_atom` refreshes it to now on
  -- every re-observation, so a re-scraped 2026-01 save reports today as its arrival date. The two
  -- columns are equal until the first re-observation and diverge permanently after it.
  -- The clock rule this serves: membership uses no clock · render order uses `when_ts` (when the
  -- source published) · SELECTION uses `first_seen`, never `ingested_at`. Any reader that picks
  -- atoms by when they arrived — `latest` (pipeline/arrivals/window.py) and search's `since` —
  -- keys on this column.
  -- Deliberately absent from `_ATOM_UPDATABLE`, and `upsert_atom`'s SET clause is built from
  -- that tuple. Adding it there would re-create the exact bug this column exists to fix, and it
  -- would look like a consistency cleanup when someone did it.
  first_seen     TEXT DEFAULT (datetime('now')),
  -- When a machine-lane atom was promoted to a human-attested mode by a human-initiated ingest
  -- touching it again (`ingest_common.promote_atom`). NULL for everything else, which is almost
  -- every row. It exists because `first_seen` cannot answer "when did a person show they cared":
  -- a frontier atom's `first_seen` is the date the crawler found it, months before the human
  -- deposit that promoted it, so a reader keyed on `first_seen` alone would treat a brand-new
  -- engagement as old material. `latest` and search's `since` read
  -- `COALESCE(promoted_at, first_seen)`. `first_seen` itself is untouched by promotion — the
  -- arrival date is still the arrival date.
  promoted_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_atoms_source_type ON atoms(source_type);
CREATE INDEX IF NOT EXISTS idx_atoms_what_kind   ON atoms(what_kind);
CREATE INDEX IF NOT EXISTS idx_atoms_who_id      ON atoms(who_id);
CREATE INDEX IF NOT EXISTS idx_atoms_when_ts     ON atoms(when_ts);

-- An atom's snapshot (frontmatter + body, as `open` returns it): the one home of an item's text,
-- written in the same transaction as its atom row (`put_body`). `body_id` is the index's docid,
-- declared so `VACUUM` never renumbers it under `bodies_fts`.
CREATE TABLE IF NOT EXISTS bodies (
  body_id INTEGER PRIMARY KEY,
  atom_id TEXT NOT NULL UNIQUE,
  text    TEXT NOT NULL
);
-- Its search index and triggers are `BODIES_INDEX_DDL`, run right after this script.

-- What a candidate is (person or organisation) is not stored: the host judges it when it prints
-- the screen (`screen._NOTE`). There is no `kind` column: an open-ended
-- `'person' | 'org' | …` enum never gained a reader, and every ingest site filled it with the
-- constant 'person'.
CREATE TABLE IF NOT EXISTS entities (
  entity_id      TEXT PRIMARY KEY,
  name           TEXT,
  identity_links TEXT,                 -- JSON: cross-platform handles (x↔github↔…)
  canonical_id   TEXT,                 -- Stage-3: attested-resolution cluster head (self when unmerged)
  profile        TEXT                  -- Stage-4 JSON: {bio, verified, followers, handle}
);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_id);

-- Stage-4 output: the user's CONFIRMED Oracles, keyed on the CANONICAL entity (so a
-- cross-platform person is one oracle, not one-per-platform). Stage 5 reads this table
-- to know which footprints to expand.
-- What an ingest COVERED is not here: it is per (Oracle x source), in `oracle_sources`
-- (`last_pulled_at` / `covered_from`, pipeline/oracles/registry.py). This table answers
-- "is this person an Oracle", nothing about how much of them we hold. `paused` is reserved/inert.
CREATE TABLE IF NOT EXISTS oracles (
  canonical_id TEXT PRIMARY KEY,
  name         TEXT,
  source       TEXT,                   -- 'screen' (a ranked pick) | 'freeform' (a pasted handle)
  confirmed_at TEXT DEFAULT (datetime('now')),
  paused       INTEGER NOT NULL DEFAULT 0  -- reserved — inert v1
);

-- Every source discovery found for an Oracle, and what became of it. Separate from `oracle_sources`: that table
-- records the pulls of sources already enabled; this one records each finding and the decision
-- on it, so a dismissed source is never found again.
CREATE TABLE IF NOT EXISTS oracle_review_items (
  review_id    INTEGER PRIMARY KEY,
  canonical_id TEXT NOT NULL,
  source_type  TEXT NOT NULL,                  -- x | substack | blog | github | orcid | …
  source_url   TEXT NOT NULL,
  source_key   TEXT NOT NULL,                  -- platform-aware canonical identity, for dedup
  kind         TEXT NOT NULL DEFAULT 'source', -- source | also (a project or org: shown, never pulled)
  reasons      TEXT NOT NULL DEFAULT '[]',     -- JSON: at most two plain phrases, for the user
  status       TEXT NOT NULL DEFAULT 'pending', -- added | dismissed (older stores: pending, unread); `also` rows are added
  created_at   TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(canonical_id, source_type, source_key)
);

-- The OTHER knowledge bases this install can read — the reader-side registry, owned by
-- pipeline/retrieval/peers.py. `oracles` says whose CONTENT this store collects; `peers` says whose
-- STORE this install may query. Different sentences, so different tables.
--
-- A table rather than a JSON file so it inherits $OPYT_HOME sandboxing, idempotent creation and
-- transactional writes, and so revocation is a row delete — the same shape the service's token
-- registry takes. `location` is EITHER a filesystem path or an HTTPS base URL; `peers.is_remote`
-- is the one place that tells them apart, so the two kinds cost one branch rather than two
-- tables.
--
-- This table must never be carried into an export (pipeline/sharing/export.py `_CARRY`): it is the
-- READER's list of who they can read, it says nothing about the corpus, and the reader's bearer
-- tokens sit in it.
-- A name and one line for a link that is never fetched as a document (a homepage, a product, a
-- job or event page), read from its page by the references worker and shown in every post that
-- links it (`kb/link_cards.py`). Empty title and description: the page said nothing, so it is
-- not read again.
CREATE TABLE IF NOT EXISTS link_cards (
  url         TEXT PRIMARY KEY,
  title       TEXT NOT NULL,
  description TEXT NOT NULL,
  read_at     TEXT NOT NULL
);

-- A thread being read: the posts fetched so far and the ones already asked for
-- (`oracles/sources/threads/state.py`). It has its own table and is not in `atoms.payload`,
-- because search and open return the payload whole and a walk runs to tens of thousands of
-- characters. The row goes when the thread is finished.
CREATE TABLE IF NOT EXISTS thread_walks (
  atom_id TEXT PRIMARY KEY,          -- the post that owes the thread
  walk    TEXT NOT NULL              -- JSON: `threads.walk.Walk.saved()`
);

CREATE TABLE IF NOT EXISTS peers (
  name     TEXT PRIMARY KEY,           -- what a caller types: kb="david"
  location TEXT NOT NULL,              -- an absolute export path, or an https:// base URL
  label    TEXT,                       -- display name, for provenance: "Alice's KB"
  token    TEXT,                       -- the reader bearer token, for a remote peer only
  added_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The user's OWN endorsement signals, folded per (entity, signal_type, platform). This is
-- the only table that records what the user did toward a source (saved/followed/liked/
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
  -- When a FULL-SET collector last re-saw this signal (`CONFIRMABLE_SIGNALS`). `save` stays
  -- permanently NULL by construction: no path re-reads your bookmark set to check a bookmark is
  -- still there.
  last_confirmed_at TEXT,
  PRIMARY KEY (entity_id, signal_type, platform)
);
CREATE INDEX IF NOT EXISTS idx_curation_signals_entity ON curation_signals(entity_id);

-- There is no `engagements` table (one row per observed reply / quote / mention act): it had one
-- writer and never acquired a reader. If peers or curation want that signal, re-derive it from the
-- tweet pull, with the reader in the same change; a table ahead of its consumer fills with
-- write-only rows.


-- The standing queries the crawler searches: topics' watches, one row per spelling. Not atoms: a
-- query is what to
-- look for, and what it finds reaches `atoms` only through stage 3's admission.
--
-- Identity is `normalized`, not `text`. A query asked again must land on the same row: stage 2's
-- per-pair watermark keys on `query_id`, so a query that spawned a twin would reset its own
-- watermark and re-pull the same window forever. Hashing the normalized form (not slug-truncating
-- the display form) is what keeps that identity stable: slugs collide once two queries share a
-- prefix, hashes do not.
--
-- Which topic owns a query is `topic_watches`; a query runs while an active topic owns it.
CREATE TABLE IF NOT EXISTS frontier_queries (
  query_id       TEXT PRIMARY KEY,       -- sha256(normalized)[:16]
  text           TEXT NOT NULL,          -- display form, as first asked
  normalized     TEXT NOT NULL UNIQUE,   -- lowercase + whitespace-collapsed = the identity
  target_sources TEXT,                   -- JSON array of adapter slugs
  created_at     TEXT NOT NULL
);

-- A watch's context words, per standing query: what narrows an ambiguous name on X ("Chai-2" +
-- antibody). Only the X source reads them. Its own table rather than a `frontier_queries` column,
-- so a store made before it needs no re-onboard.
CREATE TABLE IF NOT EXISTS watch_context (
  query_id TEXT PRIMARY KEY,
  words    TEXT NOT NULL DEFAULT '[]'      -- JSON array of words or short phrases
);

-- ── Frontier stage 2 (EXECUTE) ────────────────────────────────────────────────
-- The watermark: how far each (query, source) pair has been searched. This is the table that makes
-- query identity load-bearing: a `query_id` that churned would orphan every row here, so the loop
-- would re-pull all of history on every pass and never know it.
--
-- A failed pull must not advance `last_pulled_at`. Stamping on error buys a full TTL of silence on
-- that pair for one bad night — the `record_pull(stamp=False)` lesson from the Oracle rail.
CREATE TABLE IF NOT EXISTS frontier_query_sources (
  query_id       TEXT NOT NULL,
  source         TEXT NOT NULL,
  last_pulled_at TEXT,                 -- NULL = never pulled → infinitely stale
  last_status    TEXT,                 -- ok | empty | error | no_adapter |
                                       -- window_refused (frontier.cadence.window_ok said no)
  -- When the pair was last tried, whatever came of it. A pair tried since its last pull waits an
  -- hour, and the tries of the last hour that sent a request are the searches' hourly allowance
  -- (frontier.cadence, frontier.allowance).
  tried_at       TEXT,
  PRIMARY KEY (query_id, source)
);

-- The staging store, not `atoms`: a bad query costs rows here, and only stage 3 (ADMIT,
-- fail-closed) puts anything in the store, and only what the host's check kept. Stage 2 writes
-- 'new'; the host's check writes 'kept' or 'passed'; stage 3 writes the rest
-- (pipeline/frontier/candidates.py has the table).
--
-- `candidate_id` is a real external id ('arxiv:2501.12345', 'repo:owner/name'), never a derived
-- filename. The `pipeline/artifacts/` adapters decide "already got this?" by checking whether a
-- vault markdown file exists, and the vault is scheduled for deletion — this rail keys on an id
-- that outlives it.
CREATE TABLE IF NOT EXISTS frontier_candidates (
  candidate_id  TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  -- `source` is the FINDER, `kind` is the ATOM KIND, and they are two columns because they answer
  -- two unrelated questions. `source` says whose result this is; stage 3 dispatches on `kind`
  -- (which minter materializes it?). The two coincided exactly while
  -- arxiv and github were the only adapters — `arxiv` naming both a finder and a minter is a
  -- coincidence of that pair, not a fact about the rail, and a second paper source ends it.
  kind          TEXT,                  -- 'paper' | 'repo' | 'x' | 'substack'
  title         TEXT,
  url           TEXT,
  published     TEXT,                  -- the artifact's own date (ISO), for windowing + display
  summary       TEXT,
  payload       TEXT,                  -- JSON, adapter-specific
  status        TEXT NOT NULL DEFAULT 'new',   -- new | kept | passed | materialized | rejected
  first_seen_at TEXT NOT NULL,
  last_seen_at  TEXT NOT NULL,
  -- ── Stage 3 (ADMIT) bookkeeping ──
  -- `attempts` counts failed attempts only. A success never touches it, so `attempts > 0` reads
  -- as "this candidate has fought us N times" rather than "it was looked at N times", and the
  -- cap in frontier/admitting.py means what it says.
  --
  -- Why a reason slug and not a traceback. The retryable-vs-terminal rule is an open question
  -- that cannot be closed without a real failure distribution (the rail has never run). These
  -- three columns are that measurement: a candidate that hits the attempt cap is a row saying
  -- "the rule is wrong for me, and here is why". A traceback would be unaggregatable, would
  -- change wording between library versions, and would put arbitrary upstream text in the store.
  attempts        INTEGER NOT NULL DEFAULT 0,
  last_error      TEXT,                -- short stable slug — see frontier/admit.py
  last_attempt_at TEXT,
  -- The atom this candidate became, written when admission settles it `materialized`. It is how
  -- `latest` says which watch found an atom. `forget` sets it NULL and keeps the row, whose
  -- `materialized` status is what stops the candidate being admitted again.
  atom_id         TEXT
);
CREATE INDEX IF NOT EXISTS idx_frontier_candidates_status ON frontier_candidates(status);

-- Which standing queries surfaced a candidate. A separate table rather than a `query_id` column,
-- because one artifact is often found by several queries, and `latest` names every watch that
-- found an atom.
CREATE TABLE IF NOT EXISTS frontier_candidate_queries (
  candidate_id TEXT NOT NULL,
  query_id     TEXT NOT NULL,
  found_at     TEXT NOT NULL,
  PRIMARY KEY (candidate_id, query_id)
);
CREATE INDEX IF NOT EXISTS idx_fcq_query ON frontier_candidate_queries(query_id);

-- ── Topics ────────────────────────────────────────────────────────────────────
-- A subject the user follows: its name, the words and people that pick out the user's own people's
-- posts on it, and its watches. Opyt runs no model for it: the host checks what `latest` hands over.
CREATE TABLE IF NOT EXISTS user_topics (
  topic_id   TEXT PRIMARY KEY,
  name       TEXT NOT NULL,                     -- the subject in the user's words
  words      TEXT NOT NULL DEFAULT '[]',        -- JSON array of phrases
  people     TEXT NOT NULL DEFAULT '[]',        -- JSON array of {name, who_ids}
  status     TEXT NOT NULL DEFAULT 'active',    -- active | dropped
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
-- Which topic owns which standing query, under the name of the watch it spells.
CREATE TABLE IF NOT EXISTS topic_watches (
  topic_id TEXT NOT NULL,
  query_id TEXT NOT NULL,
  name     TEXT NOT NULL,
  added_at TEXT NOT NULL,
  PRIMARY KEY (topic_id, query_id)
);
CREATE INDEX IF NOT EXISTS idx_topic_watches_query ON topic_watches(query_id);
-- When the user last looked at what arrived (`latest`). One row.
CREATE TABLE IF NOT EXISTS arrivals_seen (
  id      INTEGER PRIMARY KEY CHECK (id = 1),
  seen_at TEXT NOT NULL
);
"""

# The `entry_mode` values that mean a human put this here. An allow-list: a new mode is excluded
# by default. Do not add the Frontier mode: it records what the crawler found, never what a person
# chose (`.guards.py`'s `human-attested-stays-human`).
HUMAN_ATTESTED: tuple[str, ...] = ("user-saved", "oracle-footprint", "author_referenced")

# Atom columns the writer supplies, in DDL order, minus the three the store manages itself:
# version/ingested_at (DEFAULT-managed) and first_seen (INSERT-only, written by `upsert_atom`).
_ATOM_COLS: tuple[str, ...] = (
    "atom_id", "source_type", "what_kind", "who_id", "when_ts", "when_precision",
    "about_entities", "source_url", "raw_hash",
    "description", "payload", "entry_mode",
)
# JSON-encoded on the way in, so callers pass native Python lists/dicts.
_JSON_ATOM_FIELDS = frozenset({"about_entities", "payload"})
# Overwritten by the newer ingest; identity (atom_id) never changes on UPSERT.
_ATOM_UPDATABLE = tuple(c for c in _ATOM_COLS if c != "atom_id")


# The signal types a full-set collector re-reads, and therefore the only ones whose absence from a
# walk means anything. `save` is deliberately absent: the saved-post sources stamp once per atom and
# `reconcile_saved_signals` is insert-if-absent, so nothing ever asks "is this still bookmarked".
CONFIRMABLE_SIGNALS: tuple[str, ...] = ("follow", "list", "like", "subscribe")


def init_kb_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe on every connect (CREATE ... IF NOT EXISTS)."""
    conn.executescript(_DDL + BODIES_INDEX_DDL)
    conn.commit()


def rebuild_body_index(conn: sqlite3.Connection) -> None:
    """Rebuild `bodies_fts` from every `bodies` row: for a copy filled in bulk before its
    triggers existed. The caller commits."""
    conn.execute("INSERT INTO bodies_fts(bodies_fts) VALUES ('rebuild')")


def connect(db_path: Path | str | None = None, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the atom-KB store (`opyt_db()` by default). Writable opens run `init_kb_schema`;
    read-only opens run no DDL, so a missing table surfaces as an error the caller turns into
    an empty result."""
    p = Path(db_path) if db_path else opyt_db()
    if read_only:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    owner_only(p)  # user content is not world-readable
    # busy_timeout first: `journal_mode=WAL` takes a brief exclusive lock and would otherwise
    # error immediately when another process holds one.
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


# The one payload key a rewrite carries across: `referenced_by` (`add_reference`), which no adapter
# knows. Every other key is replaced wholesale. An incoming payload with its own `referenced_by` wins.
_PAYLOAD_SET = (
    "payload=CASE WHEN json_extract(atoms.payload, '$.referenced_by') IS NOT NULL "
    "AND json_extract(excluded.payload, '$.referenced_by') IS NULL "
    "THEN json_set(COALESCE(excluded.payload, '{}'), '$.referenced_by', "
    "json(json_extract(atoms.payload, '$.referenced_by'))) "
    "ELSE excluded.payload END")


def add_reference(conn: sqlite3.Connection, atom_id: str, ref: dict) -> bool:
    """Record that a post linked this atom (an Oracle's own, or one the user saved): `ref` joins
    `payload.referenced_by`. Returns True when it was new. The caller commits. `ref["by"]` is the
    referrer's `who_id`, `ref["post"]` the post atom it came from.

    One entry per (`by`, `post`), since a re-walk re-dispatches the same links. Touches payload
    only, so the body is not rewritten. A missing atom is a no-op: a reference never creates a
    row."""
    row = conn.execute("SELECT payload FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()
    if row is None:
        return False
    try:
        payload = json.loads(row[0] or "{}")
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    refs = payload.get("referenced_by")
    refs = refs if isinstance(refs, list) else []
    key = (ref.get("by"), ref.get("post"))
    if any(isinstance(r, dict) and (r.get("by"), r.get("post")) == key for r in refs):
        return False
    payload["referenced_by"] = [*refs, ref]
    conn.execute("UPDATE atoms SET payload=? WHERE atom_id=?", (json.dumps(payload), atom_id))
    return True


def upsert_atom(conn: sqlite3.Connection, atom: dict) -> None:
    """Insert one atom, overwriting in place on `atom_id` conflict and bumping `version`. The
    caller commits (atom and body in one `write_tx`) and owns the hash-skip: this always writes.

    `first_seen` is written only on INSERT, so keep it out of `_ATOM_UPDATABLE`. `payload` is
    replaced wholesale except for `referenced_by` (`_PAYLOAD_SET`).
    """
    values = _encode_atom(atom)
    placeholders = ", ".join("?" for _ in _ATOM_COLS)
    set_clause = ", ".join(_PAYLOAD_SET if c == "payload" else f"{c}=excluded.{c}"
                           for c in _ATOM_UPDATABLE)
    conn.execute(
        f"INSERT INTO atoms ({', '.join(_ATOM_COLS)}, first_seen) "
        f"VALUES ({placeholders}, datetime('now')) "
        f"ON CONFLICT(atom_id) DO UPDATE SET {set_clause}, "
        f"version=atoms.version+1, ingested_at=datetime('now')",
        values,
    )


def put_body(conn: sqlite3.Connection, atom_id: str, text: str) -> None:
    """Store an atom's snapshot text, replacing the old one in place. The caller commits.

    An upsert that keeps `body_id`, so `bodies_au` re-indexes the text under the same docid."""
    conn.execute("INSERT INTO bodies (atom_id, text) VALUES (?, ?) "
                 "ON CONFLICT(atom_id) DO UPDATE SET text = excluded.text", (atom_id, text))


def read_body(conn: sqlite3.Connection, atom_id: str) -> str | None:
    """An atom's stored snapshot text, or None when it has none. The one reader, whether `conn`
    is the local store or a copy of someone else's."""
    row = conn.execute("SELECT text FROM bodies WHERE atom_id = ?", (atom_id,)).fetchone()
    return row[0] if row else None


def upsert_entity(conn: sqlite3.Connection, entity_id: str, name: str | None = None,
                  identity_links: list | None = None,
                  profile: dict | None = None) -> None:
    """UPSERT an entity. name/identity_links COALESCE (a non-null new value wins). `profile` is
    merged via json_patch, so the ingest write and the classifier write keep each other's keys.
    `identity_links` must be a list: every reader decodes a JSON array."""
    links = json.dumps(identity_links) if identity_links is not None else None
    prof = json.dumps(profile) if profile is not None else None
    with write_tx(conn, site="schema.upsert_entity"):
        conn.execute(
            "INSERT INTO entities (entity_id, name, identity_links, profile) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(entity_id) DO UPDATE SET "
            "name=COALESCE(excluded.name, entities.name), "
            "identity_links=COALESCE(excluded.identity_links, entities.identity_links), "
            # json_patch(a, b) = RFC-7386 merge of b INTO a; a null b keeps a unchanged.
            "profile=json_patch(COALESCE(entities.profile, '{}'), "
            "COALESCE(excluded.profile, '{}'))",
            (entity_id, name, links, prof),
        )


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


def claim_referenced(conn: sqlite3.Connection, *, who_id: str | None = None,
                     atom_id: str | None = None) -> int:
    """The one rule for making what an Oracle linked their own: `author_referenced` atoms (by
    author or by id) become `oracle-footprint`. A saved or footprint atom is left as it is.
    Returns how many changed. The caller holds the transaction."""
    col, val = ("who_id", who_id) if who_id is not None else ("atom_id", atom_id)
    return conn.execute(
        f"UPDATE atoms SET entry_mode='oracle-footprint' "
        f"WHERE {col}=? AND entry_mode='author_referenced'", (val,)).rowcount


def cluster_members(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """`{head: [entity_id, …]}` for every resolved cluster, the same head
    `signals_with_canonical` reports. One read, for callers that need every member of many
    clusters, including members that carry no signal."""
    out: dict[str, list[str]] = {}
    for r in conn.execute("SELECT entity_id, COALESCE(canonical_id, entity_id) AS head "
                          "FROM entities ORDER BY entity_id"):
        out.setdefault(r["head"], []).append(r["entity_id"])
    return out


def get_entity(conn: sqlite3.Connection, entity_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT entity_id, name, identity_links, canonical_id, profile "
        "FROM entities WHERE entity_id=?", (entity_id,)
    ).fetchone()


def current_canonical(conn: sqlite3.Connection, entity_id: str) -> str:
    """Follow a possibly-stale canonical_id/anchor to its current cluster head. `resolve`
    recomputes each entity's `canonical_id` (= min entity_id of its component), so a value stored
    earlier — e.g. an oracle's confirm-time `canonical_id` — goes stale when a later footprint
    merge changes the head (a `blog:`/`substack:` member can sort below the `x:user:` head). The
    stored value is always still a real member entity, so its own `canonical_id` column lands on
    the current head. Fail-safe: an unknown id resolves to itself."""
    row = conn.execute(
        "SELECT COALESCE(canonical_id, entity_id) FROM entities WHERE entity_id=?", (entity_id,)
    ).fetchone()
    return row[0] if row else entity_id


# `current_canonical` for every `oracles` row, in SQL: one row per confirmation with the head it
# folds to now (`head`), its `name` and `confirmed_at`. Roster rows keep the id each Oracle was
# confirmed under (a merge rewrites none of them), so every join on the roster joins on `head`.
ROSTER_HEADS_SQL = ("SELECT COALESCE(e.canonical_id, e.entity_id, o.canonical_id) AS head, "
                    "o.canonical_id AS anchor, o.name AS name, o.confirmed_at AS confirmed_at "
                    "FROM oracles o LEFT JOIN entities e ON e.entity_id = o.canonical_id")


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


def all_entities(conn: sqlite3.Connection) -> list:
    """Every entity row (entity_id, name, identity_links, canonical_id, profile) — the input
    Stage-3 resolution consumes to compute `canonical_id`, and Stage-4 to rank candidates."""
    return conn.execute(
        "SELECT entity_id, name, identity_links, canonical_id, profile FROM entities"
    ).fetchall()


def set_canonical_ids(conn: sqlite3.Connection, mapping: dict) -> None:
    """Bulk-write each entity's resolved `canonical_id` in one transaction. `canonical_id` is a
    materialized view of the identity_links graph; atoms key on `who_id`, so re-pointing it
    breaks nothing."""
    with write_tx(conn, site="schema.canonical_ids"):
        conn.executemany(
            "UPDATE entities SET canonical_id=? WHERE entity_id=?",
            [(cid, eid) for eid, cid in mapping.items()],
        )


def set_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str,
               platform: str, count: int = 1, extra: dict | str | None = None) -> None:
    """Record one curation signal, replacing `count` on conflict. For full-set collectors, which
    hand in a person's whole aggregate rather than a delta; confirms the signal when its type is
    in `CONFIRMABLE_SIGNALS`. A person absent from a walk keeps their last count.
    """
    _write_signal(conn, entity_id, signal_type, platform, count, extra,
                  "count = excluded.count", confirm=signal_type in CONFIRMABLE_SIGNALS)


def ensure_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str,
                  platform: str, count: int = 1, extra: dict | str | None = None) -> None:
    """Record one curation signal, leaving an existing row untouched: presence, not counting.

    For a caller that knows the signal is true but cannot count it, such as every per-atom writer
    of `save`; the full-set walk owns the count. Never confirms.
    """
    _write_signal(conn, entity_id, signal_type, platform, count, extra,
                  "count = curation_signals.count")


def _write_signal(conn: sqlite3.Connection, entity_id: str, signal_type: str, platform: str,
                  count: int, extra: dict | str | None, count_clause: str,
                  confirm: bool = False) -> None:
    """The shared body of `set_signal` / `ensure_signal`. `extra` is JSON-encoded and the newer
    non-null value wins on conflict."""
    extra_json = (json.dumps(extra) if extra is not None and not isinstance(extra, str)
                  else extra)
    stamp = "datetime('now')" if confirm else "curation_signals.last_confirmed_at"
    # Hoisted: inlined, the quote inside the f-string expression is a SyntaxError before 3.12.
    confirmed_now = "datetime('now')" if confirm else "NULL"
    with write_tx(conn, site="schema.signal"):
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


def load_hashes(conn: sqlite3.Connection, source_type: str) -> dict[str, str]:
    """`{atom_id: raw_hash}` for one source — the idempotency ledger an ingester consults
    to skip unchanged sources (write iff new OR changed)."""
    return {
        row["atom_id"]: row["raw_hash"]
        for row in conn.execute(
            "SELECT atom_id, raw_hash FROM atoms WHERE source_type=?", (source_type,)
        )
    }


def load_body_pending(conn: sqlite3.Connection, source_type: str) -> set[str]:
    """Atom ids stored without their body because a fetch was blocked (`payload.body_state =
    'pending'`) — the exception to `load_hashes`'s "already seen → skip" rule. A successful
    re-fetch rewrites the atom as `complete`, so it drops out of this set."""
    return {
        row["atom_id"]
        for row in conn.execute(
            "SELECT atom_id FROM atoms WHERE source_type=? AND "
            "json_extract(payload, '$.body_state') = 'pending'",
            (source_type,),
        )
    }


def load_thread_tails(conn: sqlite3.Connection) -> dict[str, dict]:
    """`{atom_id: {"thread_tail", "thread_ids"}}` for every X atom that carries a thread-tail
    state — `owed`, `fetched` or `gone` (`ingest_x_footprint`).

    The footprint walk reads this before it rewrites a thread from Originals, which ships only a
    thread's root and one self-reply: a stored atom whose tail was already fetched holds more than
    the walk can see, and overwriting it would flip it back to owed and rewrite its body."""
    out: dict[str, dict] = {}
    for row in conn.execute(
            "SELECT atom_id, json_extract(payload, '$.thread_tail') AS tail, "
            "json_extract(payload, '$.thread_ids') AS ids FROM atoms "
            "WHERE source_type = 'x' AND json_extract(payload, '$.thread_tail') IS NOT NULL"):
        try:
            ids = json.loads(row["ids"]) if row["ids"] else []
        except (TypeError, ValueError):
            ids = []
        out[row["atom_id"]] = {"thread_tail": row["tail"], "thread_ids": ids}
    return out


def stored_atom_id(conn: sqlite3.Connection, atom_id: str, *, nocase: bool = False) -> str | None:
    """The stored atom_id if this artifact is already an atom, else None. `nocase` costs a table
    scan instead of a primary-key lookup."""
    sql = ("SELECT atom_id FROM atoms WHERE atom_id = ? COLLATE NOCASE" if nocase
           else "SELECT atom_id FROM atoms WHERE atom_id = ?")
    row = conn.execute(sql, (atom_id,)).fetchone()
    return row[0] if row else None


def count_atoms(conn: sqlite3.Connection, source_type: str | None = None) -> int:
    if source_type is None:
        return conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM atoms WHERE source_type=?", (source_type,)
    ).fetchone()[0]
