"""
pipeline/dedup_store.py

SQLite-backed dedup state. Replaces the per-source ``state/*_synced.json`` blobs, whose
whole-file read-modify-write let two concurrent sessions clobber each other's dedup IDs
(costing a re-pull from a paid source). ``mark(ns, id)`` is a single-row ``INSERT OR
IGNORE`` keyed by ``(namespace, item_id)`` instead — commutative and idempotent under
concurrent writers; WAL lets a reader not block the writer.

``namespace`` is the old filename stem (``synced_ids``, ``blog_synced``, ...), so
existing call sites keep working unchanged via ``load_state``/``save_state``.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pipeline.timeparse import utc_now
from pathlib import Path

from pipeline.sqlite_db import default_db_path

# Table name is deliberately distinct from every other table in the shared ~/.opyt/opyt.db.
# The two it was originally written to avoid — `note_vectors` and `note_embeddings` — have
# both since been retired (2026-06 and 2026-08-05), but the rule stands and the names stay
# burned: never reuse a retired table name, or a stale reader silently joins the wrong data.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_dedup (
    namespace TEXT NOT NULL,
    item_id   TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (namespace, item_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS sync_dedup_seeded (
    namespace TEXT PRIMARY KEY,
    seeded_at TEXT NOT NULL
);

-- Per-source sync health, so a credential/auth failure is visible: "0 new" is only
-- trustworthy if last_ok_at is recent. See health_status() for reader status.
CREATE TABLE IF NOT EXISTS sync_health (
    source        TEXT PRIMARY KEY,
    last_ok_at    TEXT,
    last_error    TEXT,
    last_error_at TEXT
);
"""


def _now() -> str:
    # Full precision, deliberately unchanged: this stamp is already stored in
    # `collector_runs` / `oracle_sources` / `sync_dedup` at microsecond width, and
    # narrowing it would make new rows sort against old ones on a shared prefix.
    # `utc_iso()` (seconds) is the format for NEW stamps. See the audit's open
    # question on unifying stored-stamp precision.
    return utc_now().isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")     # readers don't block the writer
    conn.execute("PRAGMA busy_timeout=5000")    # wait up to 5s for a concurrent writer, don't error
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


class SyncSet:
    """A set-like dedup store backed by one ``namespace`` of ``sync_dedup``.

    Quacks like the plain ``set`` that ``load_state`` used to return — ``id in
    seen``, ``seen.add(id)``, ``len(seen)``, iteration — so call sites need no
    changes. But every ``.add()`` is a durable single-row ``INSERT OR IGNORE``:
    no whole-collection rewrite, and concurrent adds of disjoint IDs never
    conflict.

    Add-only by design. Dedup sets only ever grow; there is no ``remove`` because
    dropping a synced ID would silently re-pull (and re-bill) that item.
    """

    def __init__(
        self,
        namespace: str,
        legacy_json: Path | None = None,
        db_path: Path | None = None,
    ):
        self.namespace = namespace
        self._db_path = Path(db_path) if db_path else default_db_path()
        self._conn = _connect(self._db_path)
        self._maybe_seed(legacy_json)
        # Pull this namespace's IDs into RAM once so __contains__ stays O(1) and
        # the hot ingest loop doesn't round-trip to SQLite per membership test.
        self._cache: set[str] = {
            row[0]
            for row in self._conn.execute(
                "SELECT item_id FROM sync_dedup WHERE namespace=?", (self.namespace,)
            )
        }

    def _maybe_seed(self, legacy_json: Path | None) -> None:
        """One-time import of the legacy JSON blob — then never read it again.

        Guarded by the ``sync_dedup_seeded`` marker so it runs exactly once per
        namespace. This is what makes the hard cutover safe: if the table were
        empty and we skipped this, the first sync would think nothing is synced
        and re-pull the entire archive.
        """
        seeded = self._conn.execute(
            "SELECT 1 FROM sync_dedup_seeded WHERE namespace=?", (self.namespace,)
        ).fetchone()
        if seeded:
            return
        ids = []
        if legacy_json and legacy_json.exists():
            try:
                ids = json.loads(legacy_json.read_text())
            except (json.JSONDecodeError, OSError):
                ids = []
        now = _now()
        self._conn.executemany(
            "INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) VALUES (?,?,?)",
            [(self.namespace, str(i), now) for i in ids],
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO sync_dedup_seeded(namespace, seeded_at) VALUES (?,?)",
            (self.namespace, now),
        )
        self._conn.commit()

    def __contains__(self, item_id) -> bool:
        return str(item_id) in self._cache

    def __iter__(self):
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def add(self, item_id) -> None:
        sid = str(item_id)
        if sid in self._cache:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) VALUES (?,?,?)",
            (self.namespace, sid, _now()),
        )
        self._conn.commit()  # checkpoint per-add — crash-safe, replaces the old per-bookmark file rewrite
        self._cache.add(sid)

    def update(self, ids) -> None:
        new = [str(i) for i in ids if str(i) not in self._cache]
        if not new:
            return
        now = _now()
        self._conn.executemany(
            "INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) VALUES (?,?,?)",
            [(self.namespace, i, now) for i in new],
        )
        self._conn.commit()
        self._cache.update(new)

    def close(self) -> None:
        self._conn.close()


def record_health(source: str, ok: bool, detail: str | None = None,
                  db_path: Path | None = None) -> None:
    """Stamp a source's sync health. ``ok=True`` clears any prior error and sets
    last_ok_at; ``ok=False`` records the error + when. Distinguishes "caught up" from
    "silently broken auth". No reader since 2026-08-07 — see health_status()."""
    conn = _connect(Path(db_path) if db_path else default_db_path())
    now = _now()
    if ok:
        conn.execute(
            "INSERT INTO sync_health(source, last_ok_at, last_error, last_error_at) "
            "VALUES (?,?,NULL,NULL) "
            "ON CONFLICT(source) DO UPDATE SET last_ok_at=excluded.last_ok_at, "
            "last_error=NULL, last_error_at=NULL",
            (source, now),
        )
    else:
        conn.execute(
            "INSERT INTO sync_health(source, last_ok_at, last_error, last_error_at) "
            "VALUES (?,NULL,?,?) "
            "ON CONFLICT(source) DO UPDATE SET last_error=excluded.last_error, "
            "last_error_at=excluded.last_error_at",
            (source, detail, now),
        )
    conn.commit()
    conn.close()


def health_status(db_path: Path | None = None) -> list[dict]:
    """Per-source sync health. A source with a recent last_error but a stale last_ok_at is
    broken, not caught up.

    `sync_health` currently has no writer and no reader — kept intentionally, not dead code, as
    the future observability surface for "is a source caught up or is its auth silently broken".
"""
    conn = _connect(Path(db_path) if db_path else default_db_path())
    rows = conn.execute(
        "SELECT source, last_ok_at, last_error, last_error_at FROM sync_health ORDER BY source"
    ).fetchall()
    conn.close()
    return [
        {"source": s, "last_ok_at": ok, "last_error": err, "last_error_at": eat}
        for (s, ok, err, eat) in rows
    ]


# Namespaces that are genuine dedup ID-sets (NOT value caches like
# discovered_profiles.json / api_stats.json).
def migrate_all(state_dir: Path) -> dict[str, int]:
    """Force the one-time seed for every dedup JSON in ``state_dir``.

    Idempotent — safe to re-run. Returns ``{namespace: row_count}``.
    """
    paths = sorted(state_dir.glob("*_synced.json"))
    for extra in ("synced_ids.json", "processed_ids.json"):
        p = state_dir / extra
        if p.exists():
            paths.append(p)

    out: dict[str, int] = {}
    for p in paths:
        s = SyncSet(namespace=p.stem, legacy_json=p)
        out[p.stem] = len(s)
        s.close()
    return out


if __name__ == "__main__":
    # Default to the repo-root state/ dir (pipeline/ -> repo root -> state/).
    repo_state = Path(__file__).resolve().parent.parent / "state"
    print(f"Migrating dedup JSON → sync_dedup table in {default_db_path()}")
    print(f"Source dir: {repo_state}\n")
    for ns, n in migrate_all(repo_state).items():
        print(f"  {ns:24s} {n:>7,d} ids")
