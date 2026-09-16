"""Durable user decisions for discovered Oracle sources that need attribution review.

Discovery and the footprint router decide whether a source can be attributed automatically. This
module owns the different question left to the user: whether a specific unverified source may be
enabled, should be re-checked with supplied evidence, or must stay excluded.
"""

from __future__ import annotations

import json
import sqlite3

from pipeline.ingestion.url_canon import canonical_identity


_OPEN = ("pending", "verified")
_HELD = ("pending", "verified", "approved", "dismissed")


def source_key(source_url: str) -> str:
    """The identity unit used to keep one review choice per actual source."""
    return canonical_identity(source_url) or source_url.strip().rstrip("/").casefold()


def record_outcomes(conn: sqlite3.Connection, canonical_id: str, results: list[dict]) -> None:
    """Persist sources the router deliberately left un-attributed.

    Terminal user decisions win over a later discovery replay. In particular, a dismissed source
    must never become eligible simply because a later cache refresh judges it differently.
    """
    rows = [r for r in results if r.get("action") == "needs-review" and r.get("url")]
    if not rows:
        return
    from . import schema

    oracle_head = schema.current_canonical(conn, canonical_id)
    with conn:
        for row in rows:
            url = row["url"]
            stype, key = row.get("type") or "", source_key(url)
            candidates = conn.execute(
                "SELECT review_id, canonical_id FROM oracle_review_items "
                "WHERE source_type=? AND source_key=?", (stype, key),
            ).fetchall()
            existing = next((candidate for candidate in candidates
                             if schema.current_canonical(conn, candidate["canonical_id"]) == oracle_head), None)
            if existing:
                conn.execute(
                    "UPDATE oracle_review_items SET canonical_id=?, source_url=?, reason=?, "
                    "updated_at=datetime('now'), status=CASE "
                    "WHEN status IN ('verified', 'approved', 'dismissed') THEN status ELSE 'pending' END "
                    "WHERE review_id=?",
                    (canonical_id, url, row.get("detail") or "", existing["review_id"]),
                )
                continue
            conn.execute(
                "INSERT INTO oracle_review_items "
                "(canonical_id, source_type, source_url, source_key, reason) VALUES (?, ?, ?, ?, ?)",
                (canonical_id, stype, url, key, row.get("detail") or ""),
            )


def held_source_keys(conn: sqlite3.Connection, canonical_id: str) -> set[tuple[str, str]]:
    """Sources a normal discovery pass must not route again.

    Pending/verified/dismissed sources await or reject a user decision. An approved source already
    took its one targeted routing pass and is refreshed from its registered source row; re-routing
    it through discovery would re-open the same trust decision and duplicate a source pull.
    """
    from . import schema

    marks = ", ".join("?" for _ in _HELD)
    rows = conn.execute(
        f"SELECT canonical_id, source_type, source_key FROM oracle_review_items "
        f"WHERE status IN ({marks})", _HELD,
    ).fetchall()
    oracle_head = schema.current_canonical(conn, canonical_id)
    # A footprint resolve may change a cluster head after an item was created. Read every held
    # row rather than relying on the stored anchor, otherwise a dismissal would disappear on the
    # next ingest exactly when the newly resolved source makes that person easier to identify.
    return {(row["source_type"], row["source_key"]) for row in rows
            if schema.current_canonical(conn, row["canonical_id"]) == oracle_head}


def list_open(conn: sqlite3.Connection) -> list[dict]:
    """All sources still available for a user's review, oldest first."""
    marks = ", ".join("?" for _ in _OPEN)
    rows = conn.execute(
        f"SELECT * FROM oracle_review_items WHERE status IN ({marks}) "
        f"ORDER BY created_at, review_id", _OPEN,
    ).fetchall()
    return [_row(row) for row in rows]


def open_counts(conn: sqlite3.Connection, canonical_ids=None) -> dict[str, int]:
    """`canonical_id -> how many sources are still waiting for a user's decision`.

    ⚠️ THIS QUEUE HAS BEEN A SILENT DROP. `list_open` has exactly ONE caller in the repo, no rail
    reads the table, and `oracles._coverage_report` / `oracle_refresh.status_summary` both iterate
    `list_sources` only — so a needs-review source is invisible to every surface AND absent from
    the coverage report, because the adapter never ran and no `blog:`/`substack:`/`github:` entity
    was minted for it. An Oracle with an unreviewed Substack reported X-only coverage as COMPLETE.

    Measured (docs/plans/2026-07-19-oracle-trust-propagation-audit.md): 21 Oracles → 57 off-X
    sources → 25 trusted, 21 needs-review, 16 no adapter — about one per Oracle, and the audit's
    own per-row reason for nearly all of them is "no trusted source links this", on names like
    Karpathy's GitHub and Taleb's Substack. Overwhelmingly real sources that merely lacked a
    back-link, so "nothing is lost by leaving them out" is measurably false.

    A COUNT, not a list, and deliberately: the ruling is available and honest, never nagging. It
    must not feed `needs_attention`, which drives a proactive search notice. Report the number;
    `oracle(action='review')` remains where a user acts on it.

    Counts `_OPEN` — pending + verified. `approved` has had its decision and `dismissed` is a
    terminal no; neither is waiting on anybody. Clusters are folded to their current head, so a
    resolve that moved the head does not split one Oracle's queue across two keys.
    """
    from . import schema

    marks = ", ".join("?" for _ in _OPEN)
    rows = conn.execute(
        f"SELECT canonical_id FROM oracle_review_items WHERE status IN ({marks})", _OPEN,
    ).fetchall()
    want = ({schema.current_canonical(conn, c) for c in canonical_ids}
            if canonical_ids is not None else None)
    out: dict[str, int] = {}
    for row in rows:
        head = schema.current_canonical(conn, row["canonical_id"])
        if want is not None and head not in want:
            continue
        out[head] = out.get(head, 0) + 1
    return out


def get(conn: sqlite3.Connection, review_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM oracle_review_items WHERE review_id=?", (review_id,)).fetchone()
    return _row(row) if row else None


def mark_verified(conn: sqlite3.Connection, review_id: int, verification_urls: list[str]) -> None:
    _set_status(conn, review_id, "verified", verification_urls=verification_urls)


def record_evidence(conn: sqlite3.Connection, review_id: int, verification_urls: list[str]) -> None:
    """Keep the URLs a user supplied even when they did not establish the source yet."""
    item = get(conn, review_id)
    if item:
        _set_status(conn, review_id, item["status"], verification_urls=verification_urls)


def approve(conn: sqlite3.Connection, review_id: int) -> None:
    _set_status(conn, review_id, "approved")


def dismiss(conn: sqlite3.Connection, review_id: int) -> None:
    _set_status(conn, review_id, "dismissed")


def _set_status(conn: sqlite3.Connection, review_id: int, status: str,
                *, verification_urls: list[str] | None = None) -> None:
    with conn:
        if verification_urls is None:
            conn.execute(
                "UPDATE oracle_review_items SET status=?, updated_at=datetime('now') WHERE review_id=?",
                (status, review_id),
            )
        else:
            conn.execute(
                "UPDATE oracle_review_items SET status=?, verification_urls=?, "
                "updated_at=datetime('now') WHERE review_id=?",
                (status, json.dumps(verification_urls), review_id),
            )


def delete_for_oracles(conn: sqlite3.Connection, canonical_ids: list[str]) -> None:
    """Remove review choices when their Oracle subscription is forgotten.

    The subscription deletion already owns the surrounding transaction, so this deliberately does
    not open a nested connection context that could commit its broader roster change early.
    """
    if not canonical_ids:
        return
    from . import schema

    heads = {schema.current_canonical(conn, canonical_id) for canonical_id in canonical_ids}
    rows = conn.execute("SELECT review_id, canonical_id FROM oracle_review_items").fetchall()
    conn.executemany("DELETE FROM oracle_review_items WHERE review_id=?",
                     [(row["review_id"],) for row in rows
                      if schema.current_canonical(conn, row["canonical_id"]) in heads])


def _row(row: sqlite3.Row) -> dict:
    item = dict(row)
    raw_urls = item.get("verification_urls")
    item["verification_urls"] = json.loads(raw_urls) if raw_urls else []
    return item
