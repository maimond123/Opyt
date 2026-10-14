"""
pipeline/oracles/registry.py — the `oracle_sources` table: one row per (Oracle × source), and
its only writer.

One row per (Oracle × source), recording when we last pulled that pair and how much of the person
we hold — forward to `cursor_ts` (the newest atom) and back to `covered_from` (the oldest instant
a pull has reached). Nothing reads this on the query path; it lets a background loop answer "who
has gone stale" and "who is thinnest" without re-deriving either from the corpus.

This row is the only record of what a pull covered. It is written by whoever performed the pull,
never inferred from an Oracle-level column. Pairs are seeded from `entities`, not from
`atoms.source_url`, which holds permalinks rather than feeds.

Design invariants:
  • Derivable, with one exception. `cursor_ts` and the pair's identity rebuild from `atoms` +
    `entities`, so a dropped row self-heals on the next `seed_from_entities`. `last_pulled_at`
    and `covered_from` do not: they record an attempt, so a dropped row self-heals into a
    re-pull, the safe direction.
  • Breaker state is not here — it lives in the `circuit_breaker` table, keyed by service string.
  • `source_key` is adapter-ready, never a permalink — the exact identifier the source's adapter
    takes (`x` → bare handle, `substack`/`blog` → the home URL, `github` → the owner login).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from pipeline.kb import schema
from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_now

SUPPORTED_SOURCES: tuple[str, ...] = ("x", "substack", "blog", "github", "openalex")

# The columns the background scrapers (`pipeline/oracles/sources/`) keep per pair. Additive
# only: a server and a worker on different versions share one store for hours.
SCRAPER_COLUMNS: tuple[str, ...] = ("retry_at", "last_error", "check_requested_at", "back_tail_id",
                                    "fwd_started_at", "fwd_reached", "fwd_tail_id", "failures")

_DDL = """
CREATE TABLE IF NOT EXISTS oracle_sources (
  canonical_id   TEXT NOT NULL,
  source_type    TEXT NOT NULL,   -- 'x' | 'substack' | 'blog' | 'github' | 'openalex'
  source_key     TEXT NOT NULL,   -- handle / pub url / blog url / gh owner / openalex author or
                                  -- source id — NOT a permalink
  status         TEXT NOT NULL DEFAULT 'trusted',
  added_at       TEXT NOT NULL DEFAULT (datetime('now')),
  last_pulled_at TEXT,            -- NULL = never refreshed → infinitely stale
  cursor_ts      TEXT,            -- MAX(when_ts) over this pair's atoms
  covered_from   TEXT,            -- oldest instant this pair has been pulled BACK to (widen-only)
  requested_from TEXT,            -- oldest instant a USER asked this pair for (widen-only)
  topic_filter   TEXT,            -- openalex only: '|'-joined topic ids; NULL = unfiltered
  last_status    TEXT,
  -- The background scrapers' per-pair state. All nullable; NULL = nothing recorded.
  retry_at       TEXT,            -- a failed unit is not retried before this instant
  last_error     TEXT,            -- what the last failed unit saw, verbatim
  check_requested_at TEXT,        -- someone asked for this pair's newest posts (check-on-ask)
  back_tail_id   TEXT,            -- id of the oldest post the backward walk has read
  fwd_started_at TEXT,            -- when the forward walk in progress read the top of the timeline
  fwd_reached    TEXT,            -- how far down that forward walk has got
  fwd_tail_id    TEXT,            -- id of the last post that forward walk read
  failures       INTEGER,         -- failed units in a row, of the kind `last_status` names
  PRIMARY KEY (canonical_id, source_type, source_key)
);
CREATE INDEX IF NOT EXISTS idx_oracle_sources_type ON oracle_sources(source_type);
"""


# ── time helpers ────────────────────────────────────────────────────────────────
def _now() -> str:
    # Full precision: stored stamps are microsecond-width, and narrowing would make new rows sort
    # against old ones on a shared prefix. `utc_iso()` (seconds) is the format for new stamps.
    return utc_now().isoformat()


# ── the row ─────────────────────────────────────────────────────────────────────
@dataclass
class SourceRow:
    canonical_id: str
    source_type: str
    source_key: str
    status: str = "trusted"
    added_at: str | None = None
    last_pulled_at: str | None = None
    cursor_ts: str | None = None
    covered_from: str | None = None
    # The oldest instant a user asked this pair for. Widen-only, like `covered_from`; it keeps a
    # deferred pull's ask in the background deepen target.
    requested_from: str | None = None
    # OpenAlex only. The '|'-joined topic ids the ongoing stream is narrowed to, NULL = all of it.
    # The one home of that decision, so every refresh pull reads the same filter.
    topic_filter: str | None = None
    last_status: str | None = None
    name: str | None = None          # display name, joined from `oracles` — not stored here
    # The background scrapers' state; see `SCRAPER_COLUMNS` and the DDL comments.
    retry_at: str | None = None
    last_error: str | None = None
    check_requested_at: str | None = None
    back_tail_id: str | None = None
    fwd_started_at: str | None = None
    fwd_reached: str | None = None
    fwd_tail_id: str | None = None
    failures: int | None = None


def _row_to_source(row: sqlite3.Row) -> SourceRow:
    return SourceRow(
        canonical_id=row["canonical_id"],
        source_type=row["source_type"],
        source_key=row["source_key"],
        status=row["status"],
        added_at=row["added_at"],
        last_pulled_at=row["last_pulled_at"],
        cursor_ts=row["cursor_ts"],
        covered_from=row["covered_from"],
        requested_from=row["requested_from"],
        topic_filter=row["topic_filter"],
        last_status=row["last_status"],
        name=row["name"],
        **{col: row[col] for col in SCRAPER_COLUMNS},
    )


# ── connection + schema ─────────────────────────────────────────────────────────
def init_state_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe on every writable open, and called by every public writer here — a
    caller may hand us a plain `schema.connect()` that has never seen this table."""
    conn.executescript(_DDL)
    conn.commit()


# ── cursor ──────────────────────────────────────────────────────────────────────
def latest_atom_ts(conn: sqlite3.Connection, source_type: str, who_ids) -> str | None:
    """`MAX(when_ts)` over this pair's atoms — the corpus-derived cursor bookmark.

    A MAX over the whole resolved cluster, not over one identifier: the atom rail keys people on
    `who_id`, and one person is several per-platform ids."""
    ids = [i for i in (who_ids or []) if i]
    if not ids:
        return None
    placeholders = ", ".join("?" for _ in ids)
    row = conn.execute(
        f"SELECT MAX(when_ts) AS m FROM atoms WHERE source_type=? "
        f"AND who_id IN ({placeholders}) AND when_ts IS NOT NULL AND when_ts != ''",
        [source_type, *ids],
    ).fetchone()
    return row["m"] if row else None


# ── persistence ─────────────────────────────────────────────────────────────────
def upsert_source(conn: sqlite3.Connection, row: SourceRow) -> None:
    """Register a pair, preserving any freshness already recorded for it.

    `last_pulled_at` / `cursor_ts` / `covered_from` / `topic_filter` COALESCE onto the stored
    value, so `seed_from_entities` is idempotent and safe to re-run after every ingest: re-seeding
    never rewinds what the loop recorded, and never erases the user's topic choice (only
    `set_topic_filter` changes it)."""
    init_state_schema(conn)
    with write_tx(conn, site="registry.upsert_source"):
        conn.execute(
            "INSERT INTO oracle_sources "
            "(canonical_id, source_type, source_key, status, added_at, last_pulled_at, "
            " cursor_ts, covered_from, topic_filter, last_status) "
            "VALUES (?, ?, ?, ?, COALESCE(?, datetime('now')), ?, ?, ?, ?, ?) "
            "ON CONFLICT(canonical_id, source_type, source_key) DO UPDATE SET "
            "  status=excluded.status, "
            "  last_pulled_at=COALESCE(oracle_sources.last_pulled_at, excluded.last_pulled_at), "
            "  cursor_ts=COALESCE(oracle_sources.cursor_ts, excluded.cursor_ts), "
            "  covered_from=COALESCE(oracle_sources.covered_from, excluded.covered_from), "
            "  topic_filter=COALESCE(oracle_sources.topic_filter, excluded.topic_filter)",
            (row.canonical_id, row.source_type, row.source_key, row.status, row.added_at,
             row.last_pulled_at, row.cursor_ts, row.covered_from, row.topic_filter,
             row.last_status),
        )


def record_pull(conn: sqlite3.Connection, row: SourceRow, *, last_status: str,
                cursor_ts: str | None = None, covered_from: str | None = None,
                stamp: bool = True, now: str | None = None) -> None:
    """Persist one pair's outcome. Does not write `requested_from`; `record_request` owns it.

    `stamp` is false for a blocked pull: it wrote nothing and marked nothing seen, so it must not
    restart the TTL. A pull that succeeded, even with nothing new, stamps `last_pulled_at`.

    `covered_from` is the backward frontier — the oldest instant this pull reached — and it only
    ever widens (takes the MIN). `last_pulled_at`/`cursor_ts` answer "how current are we"; this
    answers "how far back do we go", which is the question `windows.deepen_target` walks
    and which no other column records. Passing None means "this pull had no lower bound to
    report" and leaves the stored value alone — so a stored NULL means exactly one thing, no
    lower bound recorded, and never doubles as "unbounded".

    ⚠️ It mutates `row`: the object passed in is updated in place to match what was written, so a
    caller comparing "before" against that same object after the call compares a value with
    itself — capture first."""
    init_state_schema(conn)
    with write_tx(conn, site="registry.record_pull"):
        conn.execute(
            "UPDATE oracle_sources SET last_status=?, "
            "  cursor_ts=COALESCE(?, cursor_ts), "
            # Positional `?` throughout, so `covered_from` is bound three times rather than
            # named `?1`. Mixing numbered and unnumbered placeholders in one statement re-numbers
            # every later `?` off the highest index used so far, which silently shifts the whole
            # binding.
            "  covered_from=CASE WHEN ? IS NULL THEN covered_from "
            "                    ELSE MIN(COALESCE(covered_from, ?), ?) END, "
            "  last_pulled_at=CASE WHEN ? THEN ? ELSE last_pulled_at END "
            "WHERE canonical_id=? AND source_type=? AND source_key=?",
            (last_status, cursor_ts,
             covered_from, covered_from, covered_from,
             1 if stamp else 0, now or _now(),
             row.canonical_id, row.source_type, row.source_key),
        )
    row.last_status = last_status
    if cursor_ts:
        row.cursor_ts = cursor_ts
    if covered_from and (row.covered_from is None or covered_from < row.covered_from):
        row.covered_from = covered_from
    if stamp:
        row.last_pulled_at = now or _now()


def record_request(conn: sqlite3.Connection, row: SourceRow,
                   requested_from: str | None) -> None:
    """Persist the window a user asked for, independent of whether the pull returned.

    Separate from `record_pull` because the ask is known before the pull starts and must survive
    a pull that never returns (a deferred pull is the case that needs it). Writes
    `requested_from` and nothing else: asking for a window is not observing it.

    Widen-only (MIN): a later narrower ask never retracts an earlier wider one. None is "no window
    to record" and leaves the stored value alone."""
    if not requested_from:
        return
    init_state_schema(conn)
    with write_tx(conn, site="registry.record_request"):
        conn.execute(
            "UPDATE oracle_sources SET "
            "  requested_from=MIN(COALESCE(requested_from, ?), ?) "
            "WHERE canonical_id=? AND source_type=? AND source_key=?",
            (requested_from, requested_from,
             row.canonical_id, row.source_type, row.source_key),
        )
    if row.requested_from is None or requested_from < row.requested_from:
        row.requested_from = requested_from


# An OpenAlex topic id. The shape is fixed (`T` + digits) and this regex is the one place a
# caller-supplied id is checked before it reaches a URL — see `set_topic_filter`.
_TOPIC_ID_RE = re.compile(r"^T\d{1,9}$")



# ── the scrapers' writes, inside the caller's transaction ─────────────────────────
# Each runs in a `writes.write_tx` the caller holds, so a pair's state commits with the page or
# the pull that produced it. `key` is `(canonical_id, source_type, source_key)`.
_WHERE = "WHERE canonical_id=? AND source_type=? AND source_key=?"


def save_back(conn: sqlite3.Connection, key: tuple, *, status: str, cursor: str | None,
              covered: str | None, tail_id: str | None, read_at: str, from_top: bool) -> None:
    """One page further back on an X pair: its status, cursor, how far back it is covered
    (widen-only), where the next page resumes, and its first pull's stamp when the page was read
    from the top."""
    conn.execute(
        "UPDATE oracle_sources SET last_status=?, cursor_ts=COALESCE(?, cursor_ts), "
        "  covered_from=CASE WHEN ? IS NULL THEN covered_from "
        "                    ELSE MIN(COALESCE(covered_from, ?), ?) END, "
        "  back_tail_id=COALESCE(?, back_tail_id), "
        "  last_pulled_at=CASE WHEN ? THEN COALESCE(last_pulled_at, ?) "
        "                      ELSE last_pulled_at END " + _WHERE,
        (status, cursor, covered, covered, covered, tail_id, 1 if from_top else 0, read_at,
         *key))


def save_forward(conn: sqlite3.Connection, key: tuple, *, status: str, cursor: str | None,
                 started: str, reached: str | None = None, tail_id: str | None = None,
                 done: bool) -> None:
    """One page of a pair's forward walk: still walking (where it got), or done (the pull it
    started is the pair's last pull, and the walk is cleared)."""
    if done:
        conn.execute(
            "UPDATE oracle_sources SET last_status=?, cursor_ts=COALESCE(?, cursor_ts), "
            "  last_pulled_at=?, fwd_started_at=NULL, fwd_reached=NULL, fwd_tail_id=NULL "
            + _WHERE, (status, cursor, started, *key))
    else:
        conn.execute(
            "UPDATE oracle_sources SET last_status=?, cursor_ts=COALESCE(?, cursor_ts), "
            "  fwd_started_at=?, fwd_reached=?, fwd_tail_id=? " + _WHERE,
            (status, cursor, started, reached, tail_id, *key))


def end_forward(conn: sqlite3.Connection, key: tuple) -> None:
    """A web pair's archive pull finished: no forward walk is in progress."""
    conn.execute("UPDATE oracle_sources SET fwd_started_at=NULL, fwd_reached=NULL " + _WHERE, key)


def cap_last_pulled(conn: sqlite3.Connection, key: tuple, at: str) -> None:
    """A long archive's first pull stamps its observed top, never later, so what was published
    during it stays owed."""
    conn.execute("UPDATE oracle_sources SET last_pulled_at=MIN(last_pulled_at, ?) " + _WHERE,
                 (at, *key))


def respell_x(conn: sqlite3.Connection, key: tuple, spelled: str) -> None:
    """Align an X pair's key to the handle as X spells it, unless that pair already exists."""
    conn.execute(
        "UPDATE oracle_sources SET source_key=? " + _WHERE + " AND NOT EXISTS ("
        "  SELECT 1 FROM oracle_sources WHERE canonical_id=? AND source_type='x' "
        "  AND source_key=?)", (spelled, *key, key[0], spelled))


def add_x_pair(conn: sqlite3.Connection, canonical_id: str, handle: str) -> None:
    """Register an X pair discovery found, unless the Oracle has it under any spelling."""
    if not conn.execute("SELECT 1 FROM oracle_sources WHERE canonical_id=? AND source_type='x' "
                        "AND lower(source_key)=?", (canonical_id, handle)).fetchone():
        conn.execute("INSERT INTO oracle_sources (canonical_id, source_type, source_key, "
                     "status) VALUES (?, 'x', ?, 'trusted')", (canonical_id, handle))


def add_openalex_pair(conn: sqlite3.Connection, canonical_id: str, author_id: str) -> None:
    """Register the OpenAlex author pair a declared ORCID resolved to, unless it is there."""
    conn.execute("INSERT INTO oracle_sources (canonical_id, source_type, source_key, status) "
                 "VALUES (?, 'openalex', ?, 'trusted') ON CONFLICT DO NOTHING",
                 (canonical_id, author_id))



def rehome(conn: sqlite3.Connection, key: tuple, canonical_id: str) -> None:
    """Move a pair registered under an old cluster id to the Oracle's current head."""
    conn.execute("UPDATE oracle_sources SET canonical_id=? " + _WHERE, (canonical_id, *key))


def take_pull(conn: sqlite3.Connection, key: tuple, row) -> None:
    """Give a pair another row's pull history (`last_pulled_at`, cursor, status)."""
    conn.execute("UPDATE oracle_sources SET last_pulled_at=?, cursor_ts=?, last_status=? "
                 + _WHERE, (row["last_pulled_at"], row["cursor_ts"], row["last_status"], *key))


def keep_coverage(conn: sqlite3.Connection, key: tuple, row) -> None:
    """Fold another row of the same pair into this one without losing what either holds: the
    older `covered_from` (widen-only) and this pair's `topic_filter`, else the other's."""
    conn.execute("UPDATE oracle_sources SET "
                 "covered_from=CASE WHEN covered_from IS NULL OR ? < covered_from "
                 "THEN COALESCE(?, covered_from) ELSE covered_from END, "
                 "topic_filter=COALESCE(topic_filter, ?) " + _WHERE,
                 (row["covered_from"], row["covered_from"], row["topic_filter"], *key))


def delete_pair(conn: sqlite3.Connection, key: tuple) -> None:
    conn.execute("DELETE FROM oracle_sources " + _WHERE, key)


def charge(conn: sqlite3.Connection, key: tuple, failures: int | None, status: str | None,
           retry_at: str | None, last_error: str | None) -> None:
    """A pair's retry columns after a failure, or a charge given back (worker's `RetryRows`)."""
    conn.execute("UPDATE oracle_sources SET failures=?, last_status=?, retry_at=?, last_error=? "
                 + _WHERE, (failures, status, retry_at, last_error, *key))


def clear_retry(conn: sqlite3.Connection, key: tuple) -> None:
    """A pair's retry columns after a success."""
    conn.execute("UPDATE oracle_sources SET failures=0, retry_at=NULL, last_error=NULL "
                 + _WHERE, key)


def rearm(conn: sqlite3.Connection, canonical_ids) -> None:
    """A new ask: these Oracles' pairs are tried again, given up or not."""
    ids = tuple(canonical_ids)
    conn.execute(f"UPDATE oracle_sources SET failures=0, retry_at=NULL "
                 f"WHERE canonical_id IN ({', '.join('?' for _ in ids)})", ids)

def mark_check(conn: sqlite3.Connection, keys: list[tuple], requested_at: str) -> None:
    """Someone asked for these pairs' newest posts (check-on-ask)."""
    conn.executemany("UPDATE oracle_sources SET check_requested_at=? " + _WHERE,
                     [(requested_at, *k) for k in keys])

def set_topic_filter(conn: sqlite3.Connection, canonical_id: str, source_key: str,
                     topics) -> str | None:
    """Narrow one OpenAlex pair to a set of topics, and return what is now stored.

    `topics` is a list of OpenAlex topic ids. An empty list clears the filter; `None` is not a
    value this function takes, because "leave it alone" is expressed by not calling it. That split
    is load-bearing: an ordinary top-up (`oracle(action='ingest')` with no topics named) must not
    touch a stored selection, and the user needs a way back to the whole corpus.

    The trust boundary for topic ids: downstream code concatenates the stored string into a URL,
    so it is validated here, once. An id that is not `T…` is dropped rather than refused.

    Raises nothing and writes nothing for a pair that does not exist — a caller registers the pair
    first (`upsert_source`), the same order `record_pull` requires.
    """
    init_state_schema(conn)
    clean = [t for t in (str(x).strip() for x in (topics or [])) if _TOPIC_ID_RE.match(t)]
    stored = "|".join(dict.fromkeys(clean)) or None
    with write_tx(conn, site="registry.set_topic_filter"):
        conn.execute(
            "UPDATE oracle_sources SET topic_filter=? "
            "WHERE canonical_id=? AND source_type='openalex' AND source_key=?",
            (stored, canonical_id, source_key))
    return stored


def topic_filter_for(conn: sqlite3.Connection, canonical_id: str,
                     source_key: str) -> str | None:
    """The stored topic filter for one OpenAlex pair, or None when it is unfiltered or unknown.

    The single read behind both pulls — the first backlog and every later refresh. Neither is
    handed a topic list by its caller; both look it up here, so the backlog and the ongoing stream
    cannot disagree about what the user chose."""
    init_state_schema(conn)
    row = conn.execute(
        "SELECT topic_filter FROM oracle_sources "
        "WHERE canonical_id=? AND source_type='openalex' AND source_key=?",
        (canonical_id, source_key)).fetchone()
    return (row["topic_filter"] or None) if row else None


def list_sources(conn: sqlite3.Connection, canonical_ids=None) -> list[SourceRow]:
    """Every registered pair (optionally scoped to some Oracles), with the Oracle's display name
    joined on so a report can name a person without a second query."""
    init_state_schema(conn)
    sql = ("SELECT s.canonical_id, s.source_type, s.source_key, s.status, s.added_at, "
           "       s.last_pulled_at, s.cursor_ts, s.covered_from, s.requested_from, "
           "       s.topic_filter, s.last_status, o.name AS name, "
           + ", ".join(f"s.{col}" for col in SCRAPER_COLUMNS) + " "
           "FROM oracle_sources s LEFT JOIN (SELECT head, name, MIN(confirmed_at) FROM ("
           + schema.ROSTER_HEADS_SQL + ") GROUP BY head) o ON o.head = s.canonical_id")
    params: list = []
    if canonical_ids:
        ids = list(canonical_ids)
        sql += f" WHERE s.canonical_id IN ({', '.join('?' for _ in ids)})"
        params = ids
    sql += " ORDER BY s.canonical_id, s.source_type, s.source_key"
    return [_row_to_source(r) for r in conn.execute(sql, params)]
