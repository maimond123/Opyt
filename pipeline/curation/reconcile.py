"""Repairs the `save` presence from the user-saved atoms that prove it.

Both save-stampers fire once per atom, on the run that lands it, so the signal is write-once
against a corpus re-walked forever. This repairs the drift that leaves behind: re-keyed entities,
atoms landed by an unstamped path, a future saved-content source. It runs on every
`oracle(action='candidates')` read, so on a store with nothing missing it only reads: the insert,
and the write lock it takes, happen only for a platform that lacks something.
"""

import sqlite3

from pipeline.kb.writes import write_tx

# Source types whose curation act leaves an atom, so the `save` signal is derivable rather than
# merely recorded. The other four signals write no atom by design, so there's nothing to
# reconcile them against. `source_type` and `platform` share one string.
SAVED_SOURCE_PLATFORMS: tuple[str, ...] = ("x", "substack")

# The user-saved atoms of one platform that name their author.
_SAVED = ("FROM atoms a "
          " WHERE a.entry_mode = 'user-saved' AND a.source_type = ? AND a.who_id IS NOT NULL ")

# Of those, the ones whose author has an entity row but no `save` signal.
_MISSING = (
    _SAVED +
    "   AND EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = a.who_id) "
    "   AND NOT EXISTS (SELECT 1 FROM curation_signals s "
    "                    WHERE s.entity_id = a.who_id AND s.signal_type = 'save' "
    "                      AND s.platform = ?)")


def _orphans(conn: sqlite3.Connection, src: str) -> int:
    return conn.execute(
        f"SELECT COUNT(DISTINCT a.who_id) {_SAVED}"
        "   AND NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = a.who_id)",
        (src,)).fetchone()[0]


def reconcile_saved_signals(conn: sqlite3.Connection) -> dict:
    """Stamps a `save` signal for every user-saved atom author that has none. Pure SQL, no
    network, idempotent, safe to call on every read.

    Insert-if-absent, never a sum, so it cannot inflate a count. Requires an `entities` row and
    reports orphans rather than inventing one. Returns counts of what landed plus the
    signal-bearing entity total."""
    missing = [src for src in SAVED_SOURCE_PLATFORMS
               if conn.execute(f"SELECT 1 {_MISSING} LIMIT 1", (src, src)).fetchone()]
    inserted: dict[str, int] = {}
    if missing:
        with write_tx(conn, site="curation.reconcile"):
            for src in missing:
                cur = conn.execute(
                    "INSERT INTO curation_signals (entity_id, signal_type, platform, count) "
                    f"SELECT DISTINCT a.who_id, 'save', ?, 1 {_MISSING} "
                    # Belt and braces: DISTINCT + NOT EXISTS already prevent a duplicate, so this
                    # only ever absorbs a concurrent writer landing the same row between the two.
                    "ON CONFLICT(entity_id, signal_type, platform) DO NOTHING",
                    (src, src, src))
                if cur.rowcount and cur.rowcount > 0:
                    inserted[src] = cur.rowcount
    orphans = {src: n for src in SAVED_SOURCE_PLATFORMS if (n := _orphans(conn, src))}
    entities = conn.execute("SELECT COUNT(DISTINCT entity_id) FROM curation_signals").fetchone()[0]
    return {"inserted": inserted, "orphans": orphans, "signal_bearing_entities": entities}
