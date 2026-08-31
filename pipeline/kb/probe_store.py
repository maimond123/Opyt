"""
pipeline/kb/probe_store.py — the CANDIDATE content store, held outside the trusted KB.

The candidate-list Proposer pulls one shallow page of each candidate's own posts so the question
"who here works on AI and biology?" can be answered from their words rather than their bio
(`docs/plans/2026-08-11-proposer-candidate-loop.md`). A candidate is someone the user engaged with
once — a follow, a bookmark, a like. That is a real prior, and it is NOT the vouch an Oracle has.

**The trust boundary, and why it is a separate table rather than a flag.**
`retrieve.py` searches every row in `atoms` with no filter, so a candidate row there would be
indistinguishable from Oracle knowledge. Table names are baked into this module's SQL, not passed
as a parameter — `AtomSink` takes a `writer=` function, not a `table=` string, so no caller
argument can leak a probe row into `atoms`. A default-exclude flag on a shared table was the
rejected alternative.

**What this store deliberately does NOT have:**
  • `entry_mode` — every row here entered the same way. The TABLE is the entry mode; a column whose
    value is constant is a lie waiting to be read as a distinction.

**Promotion is a COPY, not a migration.** When a candidate becomes an Oracle, `expand_oracle`
re-pulls their real footprint into `atoms` through the trusted path. Nothing is ever moved across
the boundary. (A `drop_probe_atoms` helper for clearing the candidate's probe rows afterwards was
deleted 2026-08-28 — nothing ever called it, so promotion has always simply left them.)
"""

from __future__ import annotations

import sqlite3

from pipeline import jitter
from pipeline.timeparse import parse_ts, utc_now

# ── DDL ───────────────────────────────────────────────────────────────────────
# Mirrors the `atoms`/`chunks`/`chunks_fts` shape closely enough that the search arms are the same
# two queries, and no further — see the docstring for the three columns deliberately absent.
_DDL = """
CREATE TABLE IF NOT EXISTS probe_atoms (
  atom_id        TEXT PRIMARY KEY,     -- 'xprobe:{root tweet id}' — its own namespace, so an id
                                       -- that surfaces in a log is never mistaken for a trusted
                                       -- 'x:'/'xprofile:' atom
  source_type    TEXT NOT NULL,        -- 'x' today; the column exists so a second probe source is
                                       -- additive rather than a schema break
  who_id         TEXT NOT NULL,        -- 'x:user:{id}' — the CANDIDATE. NOT NULL because this whole
                                       -- store answers "which candidate said this"; an
                                       -- unattributed row is unscopeable and therefore useless
  when_ts        TEXT,
  when_precision TEXT,
  source_url     TEXT,
  raw_ref        TEXT,                 -- path to the snapshot, RELATIVE to opyt_home()
  raw_hash       TEXT,                 -- sha256(snapshot) : idempotency + change key
  description    TEXT,                 -- MECHANICAL only (never interpretive)
  payload        TEXT,                 -- JSON of structural fields
  version        INTEGER NOT NULL DEFAULT 1,
  pulled_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_probe_atoms_who ON probe_atoms(who_id);

CREATE TABLE IF NOT EXISTS probe_chunks (
  chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  atom_id    TEXT NOT NULL,
  seq        INTEGER NOT NULL,
  char_start INTEGER,
  char_end   INTEGER,
  text       TEXT NOT NULL,
  -- The text `vector` was built from: `text` with OPYT's own renderer output stripped
  -- (`embed_surface`), same as `chunks.embed_text` and for the same reason — it distinguishes a
  -- vector built by an older strip from a current one. NULL means the vector came from `text`
  -- verbatim; `probe_search._snippet` falls back to `text` on NULL.
  -- NOT in `probe_chunks_fts` — BM25 already discounts terms common to every document.
  embed_text TEXT,
  vector     BLOB,                     -- L2-normalized; dim AND blob width in kb_meta, SHARED with
                                       -- `chunks` because one embedder writes both. Same subspace
                                       -- by construction; `assert_model` guards it identically,
                                       -- and `assert_strip_version` guards the surface identically.
  UNIQUE(atom_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_probe_chunks_atom ON probe_chunks(atom_id);

CREATE VIRTUAL TABLE IF NOT EXISTS probe_chunks_fts
  USING fts5(text, atom_id UNINDEXED, chunk_id UNINDEXED);

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
  detail    TEXT                         -- the failure/unavailability reason, verbatim
);
"""

# Retry semantics, keyed by what the last attempt observed. See the DDL comment.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
# A real observation whose freshness a TTL can reason about. `failed` is absent on purpose — it is
# not an observation about the candidate, it is an observation about us.
_OBSERVED = (STATUS_OK, STATUS_EMPTY, STATUS_UNAVAILABLE)

# ±25% per-candidate spread on the probe TTL (derived from `who_id`), so a batch probed together
# doesn't fall due together and re-form the same request burst every cycle. Wider than the trusted
# path's ±10%: at the measured 923-candidate backlog, ±10% would outrun the 60/day drain ceiling
# (~154/day due) while ±25% doesn't (~62/day).
PROBE_TTL_JITTER = 0.25

_PROBE_COLS: tuple[str, ...] = (
    "atom_id", "source_type", "who_id", "when_ts", "when_precision",
    "source_url", "raw_ref", "raw_hash", "description", "payload",
)
_JSON_PROBE_FIELDS = frozenset({"payload"})
_PROBE_UPDATABLE = tuple(c for c in _PROBE_COLS if c != "atom_id")


def init_probe_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, safe on every call.

    Owned HERE and not folded into `schema.init_kb_schema`, which is the same decision as
    `oracle_refresh_state`: the module that names these tables is the only module that names them.
    A probe path calls this before its first read or write; a store that has never probed simply
    has no probe tables, and no trusted path is any the wiser."""
    from .schema import _ensure_column   # ONE additive-column helper for the package, not a second
                                         # copy (the `retrieve` → `screen._best_name` precedent)

    conn.executescript(_DDL)
    # Additive for probe stores written before the embedder got its own surface. `CREATE TABLE IF
    # NOT EXISTS` is a no-op on an existing table, so a column added to `_DDL` never reaches one —
    # the same trap `schema.init_kb_schema` documents for `chunks.embed_text`. NULL reads as "this
    # vector came from `text` verbatim", which is exactly what those rows are.
    _ensure_column(conn, "probe_chunks", "embed_text", "TEXT")
    conn.commit()


def probe_tables_exist(conn: sqlite3.Connection) -> bool:
    """Does this store hold a probe store at all?

    The maintenance hooks below ASK before they act, and never call `init_probe_schema` to find
    out. A store that has never probed must come out of a maintenance run byte-identical — creating
    three empty tables as a side effect of a dtype conversion would be a schema change nobody asked
    for, on the one path whose whole job is to leave the store consistent."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='probe_chunks'"
    ).fetchone() is not None


# ── maintenance hooks (the store-wide tools call THESE, never probe SQL) ──────
#
# `kb_meta` governs dim/strip/blob-width for probe vectors too, so a tool that restripes/converts
# only `chunks` and then stamps `kb_meta` is making a claim about rows it never touched — a dtype
# mismatch there is silent garbage, not just staleness. These hooks live here rather than in the
# tools because probe-table SQL belongs to this module (`.guards.py` trust boundary).

def restrip_probe_rows(conn: sqlite3.Connection, embedder, *, profile: str,
                       apply: bool = False, group: int = 128) -> dict:
    """Re-embed every probe chunk whose `embed_text` disagrees with `profile`. Returns counts.

    Mirrors `scripts/restrip_embed_surface.py::restrip` deliberately, including its fail-safe
    granularity: a group whose embed call fails is skipped whole — no write, no partial vectors —
    so those rows stay stale and the next run retries them. A partially-written group is the one
    outcome worse than doing nothing, because those rows would look current while holding vectors
    built from the old surface.

    `{"stale": 0, ...}` on a store with no probe tables, so the caller's gate reads "nothing to do"
    rather than needing to know whether probes exist."""
    from .embed_surface import strip_for_embedding

    s = {"stale": 0, "embedded": 0, "failed": 0, "groups": 0}
    if not probe_tables_exist(conn):
        return s
    # Tables already exist here, so this only adds `embed_text` if the store predates it — never a
    # creation. The probe-less early-return above is what keeps a never-probed store untouched.
    init_probe_schema(conn)
    rows = conn.execute(
        "SELECT chunk_id, text, embed_text, source_type "
        "FROM probe_chunks pc JOIN probe_atoms pa USING(atom_id) "
        "WHERE pc.vector IS NOT NULL"
    ).fetchall()
    stale = []
    for r in rows:
        want = strip_for_embedding(r["text"] or "", r["source_type"] or "", profile)
        if (r["embed_text"] or "") != want:
            stale.append((r["chunk_id"], want))
    s["stale"] = len(stale)
    if not apply or not stale:
        return s

    import numpy as np

    from .embed import CHUNK_STORAGE_DTYPE, EmbedError

    dt = np.dtype(CHUNK_STORAGE_DTYPE)
    for i in range(0, len(stale), group):
        batch = stale[i:i + group]
        try:
            vecs = embedder.embed([t for _cid, t in batch], role="document")
        except EmbedError as e:
            from pipeline.ingestion.utils import log
            log(f"[probe-restrip] group {s['groups']} embed FAILED — "
                f"{len(batch)} chunks left stale: {e}")
            s["failed"] += len(batch)
            s["groups"] += 1
            continue
        for (cid, t), v in zip(batch, vecs):
            conn.execute("UPDATE probe_chunks SET embed_text = ?, vector = ? WHERE chunk_id = ?",
                         (t, np.asarray(v, dtype=dt).tobytes(), cid))
        conn.commit()                   # per group: bounds what an interrupt can cost
        s["embedded"] += len(batch)
        s["groups"] += 1
    return s


def convert_probe_chunk_dtype(conn: sqlite3.Connection, src, dst) -> int:
    """Rewrite every probe vector blob from width `src` to `dst`. Returns rows converted.

    Does NOT commit and does NOT touch `kb_meta` — the caller owns both, because the whole point is
    that the two tables move inside ONE transaction. Splitting the commit would reintroduce the
    exact window this closes: a crash between them leaves half the store at each width under a
    single kb_meta claim."""
    import numpy as np

    if not probe_tables_exist(conn):
        return 0
    rows = conn.execute(
        "SELECT chunk_id, vector FROM probe_chunks WHERE vector IS NOT NULL").fetchall()
    n = 0
    for r in rows:
        cid, blob = (r["chunk_id"], r["vector"]) if hasattr(r, "keys") else (r[0], r[1])
        v = np.frombuffer(blob, dtype=src).astype(dst)
        conn.execute("UPDATE probe_chunks SET vector = ? WHERE chunk_id = ?", (v.tobytes(), cid))
        n += 1
    return n


# ── writes ────────────────────────────────────────────────────────────────────

def _encode(atom: dict) -> list:
    import json

    out = []
    for c in _PROBE_COLS:
        v = atom.get(c)
        if c in _JSON_PROBE_FIELDS and v is not None and not isinstance(v, str):
            v = json.dumps(v)
        out.append(v)
    return out


def upsert_probe_atom(conn: sqlite3.Connection, atom: dict) -> None:
    """Insert one probe atom, overwriting in place on `atom_id` and bumping `version`.

    Same contract as `schema.upsert_atom`: the CALLER owns the hash-skip (never call this for an
    unchanged atom) — this always writes and always bumps."""
    placeholders = ", ".join("?" for _ in _PROBE_COLS)
    set_clause = ", ".join(f"{c}=excluded.{c}" for c in _PROBE_UPDATABLE)
    conn.execute(
        f"INSERT INTO probe_atoms ({', '.join(_PROBE_COLS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(atom_id) DO UPDATE SET {set_clause}, "
        f"version=probe_atoms.version+1, pulled_at=datetime('now')",
        _encode(atom),
    )
    conn.commit()


def replace_probe_chunks(conn: sqlite3.Connection, atom_id: str, chunks: list[dict]) -> None:
    """Replace ALL chunks for one probe atom, keeping `probe_chunks_fts` in sync. Full replace, not
    per-seq upsert: a changed snapshot shifts chunk BOUNDARIES, so a stale seq would linger."""
    conn.execute("DELETE FROM probe_chunks_fts WHERE atom_id = ?", (atom_id,))
    conn.execute("DELETE FROM probe_chunks WHERE atom_id = ?", (atom_id,))
    for ch in chunks:
        cur = conn.execute(
            "INSERT INTO probe_chunks "
            "(atom_id, seq, char_start, char_end, text, embed_text, vector) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (atom_id, ch["seq"], ch.get("char_start"), ch.get("char_end"),
             ch["text"], ch.get("embed_text"), ch.get("vector")),
        )
        conn.execute(
            "INSERT INTO probe_chunks_fts (text, atom_id, chunk_id) VALUES (?, ?, ?)",
            (ch["text"], atom_id, cur.lastrowid),
        )
    conn.commit()


def write_probe_atom(conn, embedder, atom: dict, chunks: list[dict]) -> None:
    """The `writer=` an `AtomSink` is handed to write into the PROBE store instead of `atoms`.

    Signature matches `ingest_common._write_atom` exactly, which is the whole seam — the sink keeps
    its batching, its positional-alignment assert, and its poison-chunk isolation, and the only
    thing that differs is which tables get named. No `table=` string is threaded anywhere, so there
    is no argument a future caller can get wrong. **Keep the two signatures identical**: this used
    to take (and loudly refuse) an `edges` list, and both halves went together when the `edges`
    table was deleted 2026-08-23."""
    from .embed import ensure_kb_meta

    # The probe vectors share `chunks`' subspace (one embedder for the whole store), so the identity
    # guard is the SAME guard — not a parallel one. Locking it here too means a probe-first store
    # still records what model wrote its vectors.
    ensure_kb_meta(conn, embedder.model, int(embedder.dim), embedder.provider,
                   getattr(embedder, "query_instruction", "") or "")
    init_probe_schema(conn)
    upsert_probe_atom(conn, atom)
    replace_probe_chunks(conn, atom["atom_id"], chunks)


# ── reads ─────────────────────────────────────────────────────────────────────
#
# A read never creates the store: every function below checks `probe_tables_exist` and returns the
# empty answer rather than calling `init_probe_schema`. A never-probed store asked a question must
# not gain a schema as a side effect (measured: one bare `count_probe_atoms()` call created nine
# tables). Only the WRITE paths may bring the store into existence.

def load_probe_hashes(conn: sqlite3.Connection, who_id: str | None = None) -> dict[str, str]:
    """`{atom_id: raw_hash}` — the idempotency ledger the pull hash-skips against, so a re-run
    re-embeds only what actually changed. Scoped to one candidate when `who_id` is given, which is
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

def record_pull(conn: sqlite3.Connection, who_id: str, status: str, *,
                atoms: int = 0, detail: str | None = None) -> None:
    """Record what one candidate's pull attempt observed. Overwrites in place — this is a current
    state row ("where does this candidate stand"), not an event log."""
    init_probe_schema(conn)
    conn.execute(
        "INSERT INTO probe_pulls (who_id, pulled_at, status, atoms, detail) "
        "VALUES (?, datetime('now'), ?, ?, ?) "
        "ON CONFLICT(who_id) DO UPDATE SET pulled_at=excluded.pulled_at, status=excluded.status, "
        "atoms=excluded.atoms, detail=excluded.detail",
        (who_id, status, int(atoms), detail))
    conn.commit()


def probed_today(conn: sqlite3.Connection) -> int:
    """Candidates whose pull was ATTEMPTED today (UTC) — the meter `probe_catchup`'s daily ceiling
    reads. Counts attempts, not successes: a `failed` candidate still spent the X request the
    ceiling rations. Derived from `probe_pulls.pulled_at` rather than a separate counter, so it
    can't drift from what actually ran. Answers 0 without creating tables on a never-probed store —
"""
    if not probe_tables_exist(conn):
        return 0
    return conn.execute(
        "SELECT COUNT(*) FROM probe_pulls WHERE date(pulled_at) = date('now')").fetchone()[0]


def pull_states(conn: sqlite3.Connection) -> dict[str, dict]:
    """`{who_id: {pulled_at, status, atoms, detail}}` for every candidate ever attempted."""
    if not probe_tables_exist(conn):
        return {}
    return {r["who_id"]: dict(r)
            for r in conn.execute("SELECT * FROM probe_pulls")}


def candidate_ttl_days(who_id: str, ttl_days: float) -> float:
    """This candidate's effective TTL: the flat base, spread by its own stable jitter.

    Pure — same `who_id` and same base give the same answer in every process, forever. That is what
    keeps `fresh_who_ids` a function of stored state — what the repeat-run harness checks."""
    return float(ttl_days) * jitter.stable_factor(who_id, PROBE_TTL_JITTER)


def fresh_who_ids(conn: sqlite3.Connection, *, ttl_days: float) -> set[str]:
    """Candidates whose snapshot is CURRENT — the set a pull skips this run.

    Only a real observation (`ok`/`empty`/`unavailable`) can be fresh. A `failed` row is never in
    here however recent it is, so a transient error retries on the next run instead of buying the
    candidate a TTL's worth of silence. `ttl_days <= 0` means "nothing is fresh" — a full re-pull.

    The cutoff is per-candidate, so this no longer compares against one flat `datetime('now',
    '-N days')` in SQL. It cannot: the effective TTL is a sha256 of the `who_id` and SQLite has no
    such function. So the observed rows come back unfiltered and the comparison happens here. That
    is a full scan of `probe_pulls` — one row per candidate ever attempted, so bounded by the
    candidate population itself (923 due on the live store, 2026-08-16) — which is nothing next to
    the paced X request the result gates.

    An unparseable `pulled_at` reads as DUE rather than fresh: fail-safe here means re-observing a
    candidate we cannot date, never marking them permanently current."""
    if ttl_days is None or ttl_days <= 0 or not probe_tables_exist(conn):
        return set()
    ph = ",".join("?" for _ in _OBSERVED)
    now = utc_now()
    fresh: set[str] = set()
    for r in conn.execute(
            f"SELECT who_id, pulled_at FROM probe_pulls WHERE status IN ({ph})", _OBSERVED):
        pulled = parse_ts(r["pulled_at"])
        if pulled is None:
            continue
        age_days = (now - pulled).total_seconds() / 86400.0
        if age_days < candidate_ttl_days(r["who_id"], ttl_days):
            fresh.add(r["who_id"])
    return fresh


# ── query: the SQL half of candidate search ───────────────────────────────────
#
# SQL lives here, ranking lives in `probe_search` — this module is the only one that names these
# tables, so `probe_search` needs no `.guards.py` allowlist entry.
#
# No `entry_mode` filter below: unlike `sitting_builder`/`retrieve`, every row here is one
# candidate's own timeline, pulled because a human curated that person — no second population to
# exclude.

def probe_vector_rows(conn: sqlite3.Connection,
                      who_ids: set[str] | None = None) -> list[sqlite3.Row]:
    """Every embedded probe chunk, with the atom fields a hit needs to name itself.

    Loads whole — cheap at the current scale but will need `sitting_vectors.VEC_BATCH`-style
    streaming at full candidate-population scope. Passing `who_ids` bounds it meanwhile. The
    trusted side's arm used to load whole too and no longer does (2026-08-26); this is now the
    last unbatched vector scan in the read path.
"""
    if not probe_tables_exist(conn):
        return []
    sql = ("SELECT a.atom_id, a.who_id, a.when_ts, a.source_url, "
           "c.seq AS seq, c.text AS text, c.embed_text AS embed_text, c.vector AS vector "
           "FROM probe_chunks c JOIN probe_atoms a ON a.atom_id = c.atom_id "
           "WHERE c.vector IS NOT NULL")
    params: list = []
    if who_ids is not None:
        if not who_ids:
            return []
        sql += f" AND a.who_id IN ({','.join('?' * len(who_ids))})"
        params = sorted(who_ids)
    return conn.execute(sql, params).fetchall()


def probe_fts_rows(conn: sqlite3.Connection, match: str, *,
                   limit: int, who_ids: set[str] | None = None) -> list[sqlite3.Row]:
    """Probe chunks matching an FTS5 query, strongest first.

    `match` arrives already escaped by `pipeline.rank._fts_query` — this function does NOT build
    query syntax from user text, so a caller cannot smuggle MATCH operators through it."""
    if not probe_tables_exist(conn):
        return []
    sql = ("SELECT a.atom_id, a.who_id, a.when_ts, a.source_url, "
           "c.seq AS seq, c.text AS text, c.embed_text AS embed_text, "
           "bm25(probe_chunks_fts) AS rank "
           "FROM probe_chunks_fts "
           "JOIN probe_chunks c ON c.chunk_id = probe_chunks_fts.chunk_id "
           "JOIN probe_atoms a ON a.atom_id = c.atom_id "
           "WHERE probe_chunks_fts MATCH ?")
    params: list = [match]
    if who_ids is not None:
        if not who_ids:
            return []
        sql += f" AND a.who_id IN ({','.join('?' * len(who_ids))})"
        params += sorted(who_ids)
    return conn.execute(sql + " ORDER BY rank LIMIT ?", [*params, int(limit)]).fetchall()


def probe_author_rollup(conn: sqlite3.Connection) -> dict[str, dict]:
    """`who_id -> {atoms, chunks, first_ts, last_ts}` — the per-candidate totals a summary needs.

    One grouped query rather than a `count_probe_atoms` per candidate: the caller is rendering a
    list, and a per-row count is the shape that turns a 900-candidate screen into 900 queries."""
    if not probe_tables_exist(conn):
        return {}
    rows = conn.execute(
        "SELECT a.who_id AS who_id, COUNT(DISTINCT a.atom_id) AS atoms, "
        "COUNT(c.chunk_id) AS chunks, MIN(a.when_ts) AS first_ts, MAX(a.when_ts) AS last_ts "
        "FROM probe_atoms a LEFT JOIN probe_chunks c ON c.atom_id = a.atom_id "
        "GROUP BY a.who_id")
    return {r["who_id"]: {"atoms": r["atoms"], "chunks": r["chunks"],
                          "first_ts": r["first_ts"], "last_ts": r["last_ts"]} for r in rows}


