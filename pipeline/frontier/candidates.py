"""
pipeline/frontier/candidates.py — the staged-candidate store and its shape.

`Candidate` is what an adapter returns; `frontier_candidates` is where stage 2 stages it, the
host's check decides it, and stage 3 settles it. A find stays here whatever comes of it: persistent,
never an atom until it is kept, so `search()` never sees one the host did not keep.

  | status         | written by                         | means                                  |
  |----------------|------------------------------------|----------------------------------------|
  | `new`          | `stage` (stage 2)                  | found, waiting for the host's check    |
  | `kept`         | `check` (the host)                 | worth the user's time: admission owed  |
  | `passed`       | `check` (the host)                 | not worth it; never shown again        |
  | `materialized` | `set_status` (stage 3)             | admitted: `atom_id` is the atom        |
  | `rejected`     | `set_status` (stage 3)             | admission gave up; `requeue_rejected`  |
  |                |                                    | (operator only) makes it `kept` again  |
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from pipeline.kb.id_batches import rows_for
from pipeline.kb.writes import write_tx

NEW, KEPT, PASSED = "new", "kept", "passed"

# How much of a source's own description a candidate carries as its summary. The sources return
# the whole text; each adapter cuts it to this when it builds the `Candidate`.
MAX_SUMMARY_CHARS = 2000


def kind_of(value: str | None) -> str:
    """The normalized artifact kind used for identity, admission, and ranking."""
    return (value or "").strip().lower()


def payload_of(row) -> dict:
    """Parse a staged row's payload, using an empty object for absent or malformed data."""
    try:
        payload = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def payload_authors(payload: dict) -> list[dict]:
    """Current author objects; schema migration upgrades legacy string-author payloads."""
    return [author for author in payload.get("authors") or []
            if isinstance(author, dict) and (author.get("name") or "").strip()]


@dataclass
class Candidate:
    """One artifact found by one query. `candidate_id` is the dedup identity: a real external id,
    stable across runs.

    `source` is the finder; `kind` is the atom kind (stage 3 dispatches on it). `kind` is
    per-candidate and has no default, so an adapter that forgets it fails at construction.
    """
    candidate_id: str
    source: str
    kind: str                          # atom kind: 'paper' | 'repo' | 'x' | 'substack' — see above
    title: str
    url: str
    published: str | None = None       # ISO date, stored on the candidate
    summary: str = ""
    payload: dict = field(default_factory=dict)


# ── Staging (stage 2) ───────────────────────────────────────────────────────────
def stage(conn, page, query_id: str, *, now: str) -> tuple[int, int]:
    """Stage one page of candidates under the query that found them: `(new, seen)`.

    Does not commit: the caller stages the page and stamps its pair in one write transaction.
    """
    new = seen = 0
    for cand in page:
        if _stage_one(conn, cand, query_id, now=now):
            new += 1
        else:
            seen += 1
    return new, seen


def _stage_one(conn, cand, query_id: str, *, now: str) -> bool:
    """Store a candidate and link it to the query that found it. True if newly seen.

    The link is a separate row because one artifact is often found by several queries.
    """
    # Newness is whether this insert inserted, so a second query's sighting is not counted new.
    cur = conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, kind, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at)
           VALUES (?,?,?,?,?,?,?,?,'new',?,?)
           ON CONFLICT(candidate_id) DO NOTHING""",
        (cand.candidate_id, cand.source, cand.kind, cand.title, cand.url, cand.published,
         cand.summary, json.dumps(cand.payload or {}), now, now))
    is_new = cur.rowcount > 0
    if not is_new:
        conn.execute("UPDATE frontier_candidates SET last_seen_at = ? WHERE candidate_id = ?",
                     (now, cand.candidate_id))
    conn.execute(
        "INSERT OR IGNORE INTO frontier_candidate_queries (candidate_id, query_id, found_at) "
        "VALUES (?,?,?)", (cand.candidate_id, query_id, now))
    return is_new


# ── The host's check ────────────────────────────────────────────────────────────
def waiting(conn, query_ids) -> list[sqlite3.Row]:
    """The finds of these standing queries still waiting for the host's check, oldest first, each
    with the text of the first query that found it."""
    return list(rows_for(
        conn, "SELECT c.candidate_id, c.source, c.kind, c.title, c.url, c.published, c.summary, "
              "MIN(q.text) AS found_by FROM frontier_candidates c "
              "JOIN frontier_candidate_queries cq ON cq.candidate_id = c.candidate_id "
              "JOIN frontier_queries q ON q.query_id = cq.query_id "
              f"WHERE c.status = '{NEW}' AND cq.query_id IN ({{ids}}) "
              "GROUP BY c.candidate_id ORDER BY c.first_seen_at, c.candidate_id",
        list(query_ids)))


def check(conn, *, kept, passed) -> dict[str, int]:
    """The host's verdicts on waiting finds: `{kept, passed}` counts of the rows that changed.
    Only a waiting find moves, so a verdict sent twice, or on a find already settled, changes
    nothing. Joins the caller's transaction."""
    out = {}
    for status, ids in ((KEPT, kept), (PASSED, passed)):
        out[status] = sum(conn.execute(
            f"UPDATE frontier_candidates SET status = ? WHERE status = '{NEW}' AND candidate_id = ?",
            (status, cid)).rowcount for cid in dict.fromkeys(ids))
    return out


# ── Admission status (stage 3) ──────────────────────────────────────────────────
def backlog(conn) -> int:
    """How many kept candidates are still owed admission."""
    return conn.execute(
        f"SELECT COUNT(*) FROM frontier_candidates WHERE status = '{KEPT}'").fetchone()[0]


def due_for_admission(conn, *, tried_before: str, limit: int) -> list[sqlite3.Row]:
    """The oldest-first `kept` candidates not tried since `tried_before`: a failed attempt waits.

    Selects the descriptive columns too, not just the identifying ones: `known_metadata` mints
    from what stage 2 already collected rather than re-fetching it.
    """
    return list(conn.execute(
        "SELECT candidate_id, source, kind, url, title, summary, payload, published, attempts, "
        "last_attempt_at FROM frontier_candidates "
        f"WHERE status = '{KEPT}' AND (last_attempt_at IS NULL OR last_attempt_at <= ?) "
        "ORDER BY first_seen_at ASC, candidate_id ASC LIMIT ?", (tried_before, limit)))


def attempts_since(conn, since: str) -> list[str]:
    """When each admission attempt since `since` was made: one per candidate, since a candidate
    is tried at most once an hour."""
    return [r[0] for r in conn.execute(
        "SELECT last_attempt_at FROM frontier_candidates WHERE last_attempt_at > ?", (since,))]


def next_retry(conn, since: str) -> str | None:
    """The oldest attempt since `since` among `kept` candidates: the next one to be due again."""
    return conn.execute("SELECT MIN(last_attempt_at) FROM frontier_candidates "
                        f"WHERE status = '{KEPT}' AND last_attempt_at > ?", (since,)).fetchone()[0]


def set_status(conn, candidate_id: str, *, status: str, attempts: int, last_error: str | None,
               atom_id: str | None, at: str) -> None:
    """Write one admission attempt's outcome onto its candidate, in its own write transaction:
    with `atom_id`, the atom a materialized candidate became."""
    with write_tx(conn, site="frontier_candidates.status"):
        conn.execute(
            "UPDATE frontier_candidates SET status=?, attempts=?, last_error=?, last_attempt_at=?, "
            "atom_id=? WHERE candidate_id=?",
            (status, attempts, last_error, at, atom_id, candidate_id))


def requeue_rejected(conn, *, last_error: str | None = None) -> int:
    """Reset `rejected` rows to `kept` with a cleared attempt count. Returns the row count.

    Operator-only: an automatic reset would be a higher attempt cap in disguise. `last_error`
    narrows the reset to rows with that failure reason.
    """
    sql = f"UPDATE frontier_candidates SET status='{KEPT}', attempts=0, last_error=NULL " \
          "WHERE status='rejected'"
    params: tuple = ()
    if last_error is not None:
        sql += " AND last_error = ?"
        params = (last_error,)
    with write_tx(conn, site="frontier.requeue_rejected"):
        return conn.execute(sql, params).rowcount
