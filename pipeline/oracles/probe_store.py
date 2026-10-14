"""
pipeline/oracles/probe_store.py — the candidate content store, held outside the trusted KB.

The candidate-list Proposer pulls one shallow page of each candidate's own content — X posts, or
recent papers for someone the user knows only as an author — so "who here works on AI and
biology?" can be answered from their words rather than their bio. A candidate is someone the user
engaged with once; that is a real prior, not the vouch an Oracle has.

**The trust boundary is a separate table, not a flag.** `retrieve.py` searches every row in
`atoms` with no filter, so a candidate row there would be indistinguishable from Oracle knowledge.
Table names are baked into this module's SQL, and a probe atom has its own writer
(`store_probe_atom`, beside the KB's `ingest_common.store_atom`) rather than a `table=` argument,
so no caller argument can leak a probe row into `atoms`. The table is the entry mode, so there is
no `entry_mode` column.

Promotion is a copy, not a migration: `expand_oracle` re-pulls an Oracle's footprint into `atoms`
through the trusted path, and nothing moves across the boundary.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline.kb import writes
from pipeline.retrieval import excerpt

# ── DDL ───────────────────────────────────────────────────────────────────────
# Mirrors the `atoms`/`bodies`/`bodies_fts` shape closely enough that the keyword search is the
# same query, and no further.
_DDL = """
CREATE TABLE IF NOT EXISTS probe_atoms (
  atom_id        TEXT PRIMARY KEY,     -- 'xprobe:{root tweet id}' / 'oaprobe:{openalex work id}'
                                       -- — its own namespace per probe source, so an id that
                                       -- surfaces in a log is never mistaken for a trusted
                                       -- 'x:'/'xprofile:'/'paper:' atom
  source_type    TEXT NOT NULL,        -- 'x' | 'paper'. The column anticipated the second source
                                       -- and got it in 2026-09 (`scholar_probe`), so adding one
                                       -- was additive rather than a schema break
  who_id         TEXT NOT NULL,        -- the CANDIDATE: 'x:user:{id}' or 'openalex:{author id}'.
                                       -- NOT NULL because this whole store answers "which
                                       -- candidate said this"; an unattributed row is unscopeable
                                       -- and therefore useless
  when_ts        TEXT,
  when_precision TEXT,
  source_url     TEXT,
  raw_hash       TEXT,                 -- sha256(snapshot) : idempotency + change key
  description    TEXT,                 -- MECHANICAL only (never interpretive)
  payload        TEXT,                 -- JSON of structural fields
  version        INTEGER NOT NULL DEFAULT 1,
  pulled_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_probe_atoms_who ON probe_atoms(who_id);

-- A probe atom's snapshot text, the one home of it, as `bodies` is an atom's. Whole-item, not
-- windowed: the index matches an item on all its words.
CREATE TABLE IF NOT EXISTS probe_bodies (
  body_id INTEGER PRIMARY KEY,         -- the index's docid; explicit, so VACUUM never renumbers it
  atom_id TEXT NOT NULL UNIQUE,
  text    TEXT NOT NULL
);

-- External-content and stemmed like `bodies_fts`, so candidate search by keyword matches word
-- forms the same way and stores the text once. The triggers keep it in step with every write to
-- `probe_bodies`; never `INSERT OR REPLACE` into it (REPLACE fires no delete trigger).
CREATE VIRTUAL TABLE IF NOT EXISTS probe_bodies_fts
  USING fts5(text, content='probe_bodies', content_rowid='body_id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS probe_bodies_ai AFTER INSERT ON probe_bodies BEGIN
  INSERT INTO probe_bodies_fts(rowid, text) VALUES (new.body_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS probe_bodies_ad AFTER DELETE ON probe_bodies BEGIN
  INSERT INTO probe_bodies_fts(probe_bodies_fts, rowid, text) VALUES ('delete', old.body_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS probe_bodies_au AFTER UPDATE ON probe_bodies BEGIN
  INSERT INTO probe_bodies_fts(probe_bodies_fts, rowid, text) VALUES ('delete', old.body_id, old.text);
  INSERT INTO probe_bodies_fts(rowid, text) VALUES (new.body_id, new.text);
END;

-- One row per candidate the pull has ATTEMPTED, whatever came back. Deliberately NOT derived from
-- `MAX(pulled_at)` over `probe_atoms`, and the reason is the plan's first named failure mode: some
-- accounts return zero tweets. Derived state cannot record that, so a candidate with no content
-- would look identical to a candidate never tried — and would be re-fetched on every run forever,
-- out of the scarcest budget this path has (~169 requests/hour, shared with every other scraper).
--
-- `status` also carries the retry rule, which is why it is four values and not a boolean:
--   ok / empty / unavailable  — a real observation. Due again only once the TTL expires.
--   failed                    — we were STOPPED (fetch error). ALWAYS due, never waits out a TTL.
-- That last line is the fail-safe invariant made concrete: a failed external call records what
-- happened but must never mark unfinished work done.
CREATE TABLE IF NOT EXISTS probe_pulls (
  who_id    TEXT PRIMARY KEY,            -- 'x:user:{id}' — the candidate
  pulled_at TEXT NOT NULL,
  status    TEXT NOT NULL,               -- ok | empty | unavailable | failed
  atoms     INTEGER NOT NULL DEFAULT 0,  -- atoms durably written by that attempt
  detail    TEXT,                        -- the failure/unavailability reason, verbatim
  -- Scouting's progress through one candidate: each page is its
  -- own unit, so a sample cut short keeps the pages it read and resumes after them.
  pages_done INTEGER,                    -- pages of this candidate's sample read so far
  tail_id    TEXT,                       -- id of the last post read, where the next page resumes
  retry_at   TEXT,                       -- a `failed` attempt is not retried before this instant
  failures   INTEGER                     -- failed attempts in a row (`pipeline.worker.quota_loop.retries`)
);

-- A pull row answers where a candidate stands. This append-only ledger answers a different
-- question: how many candidate samples consumed today's allowance. Repeated failures overwrite
-- the former but must each consume the latter.
CREATE TABLE IF NOT EXISTS probe_attempts (
  attempt_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  who_id       TEXT NOT NULL,
  attempted_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_probe_attempts_at ON probe_attempts(attempted_at);
"""

# Retry semantics, keyed by what the last attempt observed. See the DDL comment.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
# A sample with pages still owed (the scouting scraper reads one page per unit). Not an
# observation either: it is due until the sample finishes.
STATUS_SAMPLING = "sampling"
_PROBE_COLS: tuple[str, ...] = (
    "atom_id", "source_type", "who_id", "when_ts", "when_precision",
    "source_url", "raw_hash", "description", "payload",
)
_JSON_PROBE_FIELDS = frozenset({"payload"})
_PROBE_UPDATABLE = tuple(c for c in _PROBE_COLS if c != "atom_id")


def init_probe_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, safe on every call.

    Owned here, not in `schema.init_kb_schema`: the module that names these tables is the only
    one that names them. A probe path calls this before its first read or write; a store that has never probed simply
    has no probe tables, and no trusted path is any the wiser."""
    conn.executescript(_DDL)
    conn.commit()


def probe_tables_exist(conn: sqlite3.Connection) -> bool:
    """Does this store hold a probe store at all?

    The maintenance hooks ask this rather than calling `init_probe_schema`, so a store that has
    never probed comes out of a maintenance run byte-identical."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='probe_bodies'"
    ).fetchone() is not None


# ── writes ────────────────────────────────────────────────────────────────────

def _encode(atom: dict) -> list:
    out = []
    for c in _PROBE_COLS:
        v = atom.get(c)
        if c in _JSON_PROBE_FIELDS and v is not None and not isinstance(v, str):
            v = json.dumps(v)
        out.append(v)
    return out


def upsert_probe_atom(conn: sqlite3.Connection, atom: dict) -> None:
    """Insert one probe atom, overwriting in place on `atom_id` and bumping `version`.

    Same contract as `schema.upsert_atom`: the caller owns the hash-skip (never call this for an
    unchanged atom) — this always writes and always bumps. The caller commits: `store_probe_atom`
    writes the atom and its body in one `write_tx`."""
    placeholders = ", ".join("?" for _ in _PROBE_COLS)
    set_clause = ", ".join(f"{c}=excluded.{c}" for c in _PROBE_UPDATABLE)
    conn.execute(
        f"INSERT INTO probe_atoms ({', '.join(_PROBE_COLS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(atom_id) DO UPDATE SET {set_clause}, "
        f"version=probe_atoms.version+1, pulled_at=datetime('now')",
        _encode(atom),
    )


def put_probe_body(conn: sqlite3.Connection, atom_id: str, text: str) -> None:
    """Store a probe atom's snapshot text, replacing the old one in place, so the triggers
    re-index it under the same docid. The caller commits."""
    conn.execute("INSERT INTO probe_bodies (atom_id, text) VALUES (?, ?) "
                 "ON CONFLICT(atom_id) DO UPDATE SET text = excluded.text", (atom_id, text))


def store_probe_atom(conn: sqlite3.Connection, atom: dict, text: str) -> None:
    """Write one probe atom and its body `text` into the probe store, never `atoms`: the probe
    store's counterpart of `ingest_common.store_atom`, kept a separate function so the trust
    boundary is which function a caller names, never an argument it passes.

    One transaction: the atom row carries the `raw_hash` the pull's hash-skip reads as done, so
    it never lands without its body. Inside a caller's `write_tx` it joins that transaction.
    Creating the tables commits (`executescript`), so inside one the caller has created them
    already."""
    if not writes.in_write_tx(conn):
        init_probe_schema(conn)
    with writes.write_tx(conn, site="probe_atom"):
        upsert_probe_atom(conn, atom)
        put_probe_body(conn, atom["atom_id"], text)


# ── reads ─────────────────────────────────────────────────────────────────────
#
# A read never creates the store: every function below checks `probe_tables_exist` and returns the
# empty answer rather than calling `init_probe_schema`: a never-probed store asked a question must
# not gain a schema as a side effect. Only the write paths may bring the store into existence.

def load_probe_hashes(conn: sqlite3.Connection, who_id: str | None = None) -> dict[str, str]:
    """`{atom_id: raw_hash}` — the idempotency ledger the pull hash-skips against, so a re-run
    rewrites only what actually changed. Scoped to one candidate when `who_id` is given, which is
    what keeps a per-candidate pull from loading the whole store.

    Fail-safe: a store with no probe tables yet returns `{}` rather than raising, so a first-ever
    pull and a read-only caller both degrade to "nothing seen"."""
    if not probe_tables_exist(conn):
        return {}
    sql = "SELECT atom_id, raw_hash FROM probe_atoms"
    params: tuple = ()
    if who_id:
        sql += " WHERE who_id = ?"
        params = (who_id,)
    return {r["atom_id"]: r["raw_hash"] for r in conn.execute(sql, params)}


def count_probe_atoms(conn: sqlite3.Connection, who_id: str | None = None) -> int:
    if not probe_tables_exist(conn):
        return 0
    if who_id:
        return conn.execute("SELECT COUNT(*) FROM probe_atoms WHERE who_id=?",
                            (who_id,)).fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM probe_atoms").fetchone()[0]


def probed_who_ids(conn: sqlite3.Connection) -> set[str]:
    """Every candidate this store holds content for. Cheap (indexed), and it is how a pull knows
    who it can skip without loading a hash ledger per person."""
    if not probe_tables_exist(conn):
        return set()
    return {r[0] for r in conn.execute("SELECT DISTINCT who_id FROM probe_atoms")}


# ── pull state ────────────────────────────────────────────────────────────────


def record_sample(conn: sqlite3.Connection, who_id: str, status: str, *, atoms: int,
                  pages_done: int, tail_id: str | None, retry_at: str | None = None,
                  detail: str | None = None, failures: int | None = None) -> None:
    """Record where one candidate's sample stands after a page: `sampling` while pages are owed,
    then `ok` / `empty` / `unavailable`, or `failed` with the instant it may be retried.

    The scouting scraper's writer (`oracles.sources.timeline.scouting`). It neither creates the
    tables nor commits: it runs inside the caller's write transaction, after the page's atoms are
    written."""
    conn.execute(
        "INSERT INTO probe_pulls (who_id, pulled_at, status, atoms, detail, pages_done, tail_id, "
        "  retry_at, failures) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(who_id) DO UPDATE SET pulled_at=excluded.pulled_at, status=excluded.status, "
        "atoms=excluded.atoms, detail=excluded.detail, pages_done=excluded.pages_done, "
        "tail_id=excluded.tail_id, retry_at=excluded.retry_at, failures=excluded.failures",
        (who_id, status, int(atoms), detail, int(pages_done), tail_id, retry_at, failures))


def record_retry(conn: sqlite3.Connection, who_id: str, status: str, *, retry_at: str | None,
                 failures: int | None, detail: str | None) -> None:
    """Record a candidate's retry state, a failed page charged or a charge given back, inside the
    caller's write transaction. A candidate never sampled gets its row here, with nothing read
    yet. The quota loop's writer, through `scouting.PROBE_ROWS`."""
    conn.execute(
        "INSERT INTO probe_pulls (who_id, pulled_at, status, atoms, detail, pages_done, tail_id, "
        "  retry_at, failures) VALUES (?, datetime('now'), ?, 0, ?, 0, NULL, ?, ?) "
        "ON CONFLICT(who_id) DO UPDATE SET pulled_at=excluded.pulled_at, status=excluded.status, "
        "detail=excluded.detail, retry_at=excluded.retry_at, failures=excluded.failures",
        (who_id, status, detail, retry_at, failures))


def clear_retry(conn: sqlite3.Connection, who_id: str) -> None:
    """Clear a candidate's retry state after a page that came back, inside the caller's write
    transaction. `detail` is the sample's, and stays."""
    conn.execute("UPDATE probe_pulls SET failures=NULL, retry_at=NULL WHERE who_id=?", (who_id,))


def next_retry_at(conn: sqlite3.Connection) -> str | None:
    """The earliest `retry_at` among failed samples — when scouting next has a retry owed."""
    if not probe_tables_exist(conn):
        return None
    row = conn.execute("SELECT MIN(retry_at) FROM probe_pulls "
                       "WHERE status = ? AND retry_at IS NOT NULL", (STATUS_FAILED,)).fetchone()
    return row[0] if row else None


def pull_states(conn: sqlite3.Connection) -> dict[str, dict]:
    """`{who_id: {pulled_at, status, atoms, detail}}` for every candidate ever attempted."""
    if not probe_tables_exist(conn):
        return {}
    return {r["who_id"]: dict(r)
            for r in conn.execute("SELECT * FROM probe_pulls")}


# ── query: the SQL half of candidate search ───────────────────────────────────
#
# SQL lives here, ranking lives in `probe_search` — this module is the only one that names these
# tables, so `probe_search` needs no `.guards.py` allowlist entry.
#
# No `entry_mode` filter below: unlike `retrieve`, every row here is one candidate's own timeline,
# pulled because a human curated that person — no second population to exclude.

def probe_fts_rows(conn: sqlite3.Connection, match: str, *,
                   limit: int, who_ids: set[str] | None = None) -> list[sqlite3.Row]:
    """Probe atoms matching an FTS5 query, strongest first, one row per atom. Each row's `marked`
    is its body with every match wrapped in `excerpt`'s markers, so a caller cuts its excerpt
    where the index matched (`excerpt.window`).

    `match` arrives already escaped by `pipeline.retrieval.rank.fts_query` — this function does not build
    query syntax from user text, so a caller cannot smuggle MATCH operators through it."""
    if not probe_tables_exist(conn):
        return []
    sql = ("SELECT a.atom_id, a.who_id, a.when_ts, a.source_url, "
           f"highlight(probe_bodies_fts, 0, char({excerpt.OPEN}), char({excerpt.CLOSE})) "
           "AS marked, bm25(probe_bodies_fts) AS rank "
           "FROM probe_bodies_fts "
           "JOIN probe_bodies b ON b.body_id = probe_bodies_fts.rowid "
           "JOIN probe_atoms a ON a.atom_id = b.atom_id "
           "WHERE probe_bodies_fts MATCH ?")
    params: list = [match]
    if who_ids is not None:
        if not who_ids:
            return []
        sql += f" AND a.who_id IN ({','.join('?' * len(who_ids))})"
        params += sorted(who_ids)
    return conn.execute(sql + " ORDER BY rank LIMIT ?", [*params, int(limit)]).fetchall()


def count_people_matching(conn: sqlite3.Connection, match: str) -> int:
    """How many sampled people have a post matching an FTS5 query, already escaped like
    `probe_fts_rows`' `match`."""
    if not probe_tables_exist(conn):
        return 0
    return conn.execute(
        "SELECT COUNT(DISTINCT a.who_id) FROM probe_bodies_fts "
        "JOIN probe_bodies b ON b.body_id = probe_bodies_fts.rowid "
        "JOIN probe_atoms a ON a.atom_id = b.atom_id "
        "WHERE probe_bodies_fts MATCH ?", (match,)).fetchone()[0]


def texts_by_person(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Every sampled person's item texts, by `who_id`. `{}` when nothing was ever sampled."""
    if not probe_tables_exist(conn):
        return {}
    out: dict[str, list[str]] = {}
    for who_id, text in conn.execute(
            "SELECT a.who_id, b.text FROM probe_bodies b "
            "JOIN probe_atoms a ON a.atom_id = b.atom_id"):
        out.setdefault(who_id, []).append(text or "")
    return out


def probe_author_rollup(conn: sqlite3.Connection) -> dict[str, dict]:
    """`who_id -> {atoms, first_ts, last_ts}` — the per-candidate totals a summary needs.

    One grouped query rather than a `count_probe_atoms` per candidate: the caller is rendering a
    list, and a per-row count is the shape that turns a 900-candidate screen into 900 queries."""
    if not probe_tables_exist(conn):
        return {}
    rows = conn.execute(
        "SELECT who_id, COUNT(*) AS atoms, MIN(when_ts) AS first_ts, MAX(when_ts) AS last_ts "
        "FROM probe_atoms GROUP BY who_id")
    return {r["who_id"]: {"atoms": r["atoms"],
                          "first_ts": r["first_ts"], "last_ts": r["last_ts"]} for r in rows}

