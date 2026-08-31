"""schema — the substrate contracts (upsert identity, FTS sync, the curation-signal
operators, and the migration that dropped the two write-only relation stores). Uses an
explicit db_path so no $OPYT_HOME needed."""
from __future__ import annotations

import json

import pytest

from pipeline.kb import schema


@pytest.fixture()
def conn(tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _atom(atom_id="x:1", **over):
    base = dict(
        atom_id=atom_id, source_type="x", what_kind="opinion", who_id="x:user:1",
        when_ts="2024-01-01", when_precision="day",
        about_entities=["x:@ylecun"], source_url="https://x.com/a/status/1",
        raw_ref="kb_raw/x/x_1.md", raw_hash="h1", description="a card",
        payload={"like_count": 5}, entry_mode="user-saved",
    )
    base.update(over)
    return base


def test_upsert_atom_roundtrips_json_and_bumps_version(conn):
    schema.upsert_atom(conn, _atom())
    row = conn.execute("SELECT * FROM atoms WHERE atom_id='x:1'").fetchone()
    assert json.loads(row["about_entities"]) == ["x:@ylecun"]
    assert json.loads(row["payload"]) == {"like_count": 5}
    assert row["version"] == 1

    # Re-ingest the same identity with a new hash → in-place overwrite + version bump.
    schema.upsert_atom(conn, _atom(raw_hash="h2", description="changed"))
    row = conn.execute("SELECT version, raw_hash, description FROM atoms WHERE atom_id='x:1'").fetchone()
    assert row["version"] == 2 and row["raw_hash"] == "h2" and row["description"] == "changed"
    assert schema.count_atoms(conn, "x") == 1   # still one row (identity, not append)


def test_first_seen_survives_a_re_observation_while_ingested_at_moves(conn):
    """THE reason the column exists. `ingested_at` is refreshed by every upsert, so it answers
    "last observed at"; `first_seen` must keep answering "arrived at" forever.

    Both stamps are backdated before the re-upsert on purpose. `datetime('now')` has one-second
    resolution, so a test that inserted and re-upserted in the same second would see the two
    columns agree and pass without proving anything."""
    schema.upsert_atom(conn, _atom())
    conn.execute("UPDATE atoms SET ingested_at='2026-01-01 00:00:00', "
                 "first_seen='2026-01-01 00:00:00' WHERE atom_id='x:1'")
    conn.commit()

    schema.upsert_atom(conn, _atom(raw_hash="h2"))       # re-observed: version 1 -> 2
    row = conn.execute("SELECT version, ingested_at, first_seen FROM atoms "
                       "WHERE atom_id='x:1'").fetchone()
    assert row["version"] == 2
    assert row["first_seen"] == "2026-01-01 00:00:00"    # untouched by the UPDATE arm
    assert row["ingested_at"] > "2026-01-01 00:00:00"    # refreshed, as it always was


def test_first_seen_is_written_on_insert_without_relying_on_a_column_default(conn):
    """A migrated store's `first_seen` has NO default — SQLite forbids `datetime('now')` in ADD
    COLUMN — so `upsert_atom` supplies it as a literal. Pin that the writer, not the DDL, is what
    populates it, or the migrated path silently writes NULL and every arrival date is lost."""
    assert "first_seen" not in schema._ATOM_COLS        # not caller-supplied
    assert "first_seen" not in schema._ATOM_UPDATABLE   # ⚠️ adding it re-creates the bug
    schema.upsert_atom(conn, _atom("x:9"))
    assert conn.execute(
        "SELECT first_seen FROM atoms WHERE atom_id='x:9'").fetchone()[0] is not None


def test_migration_backfills_first_seen_from_ingested_at(tmp_path):
    """A store written before the column existed gets it on the next connect, seeded from
    `ingested_at`. That copy is exact only while `version = 1`, which is why it ran the day the
    column landed — see `_backfill_first_seen`."""
    import sqlite3 as _sq
    p = tmp_path / "old.db"
    raw = _sq.connect(str(p))
    # The pre-column shape, DERIVED from the writer's own column tuple rather than retyped, so
    # this fixture cannot drift out of date the next time a column is added to `atoms`.
    cols = ", ".join(f"{c} TEXT" for c in schema._ATOM_COLS)
    raw.execute(f"CREATE TABLE atoms ({cols}, version INTEGER NOT NULL DEFAULT 1, "
                "ingested_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY (atom_id))")
    raw.execute("INSERT INTO atoms (atom_id, source_type, ingested_at) "
                "VALUES ('x:old','x','2026-03-04 05:06:07')")
    raw.commit()
    raw.close()

    c = schema.connect(p)                                # runs init_kb_schema -> ALTER + backfill
    assert c.execute("SELECT first_seen FROM atoms WHERE atom_id='x:old'"
                     ).fetchone()[0] == "2026-03-04 05:06:07"

    # Idempotent: a second connect must not re-stamp a row that already carries an arrival date.
    c.close()
    c = schema.connect(p)
    assert c.execute("SELECT first_seen FROM atoms WHERE atom_id='x:old'"
                     ).fetchone()[0] == "2026-03-04 05:06:07"
    c.close()


def test_migration_renames_crawled_to_oracle_footprint(tmp_path):
    """A store holding the retired mode heals on connect, and ONLY that mode moves.

    `crawled` was GitHub's footprint sweep — the same act X and Substack write as
    'oracle-footprint', but stamped outside HUMAN_ATTESTED, so those atoms were unreachable to
    every sitting. See docs/plans/2026-08-25-rename-github-crawled-to-oracle-footprint.md.
    """
    p = tmp_path / "old.db"
    c = schema.connect(p)
    schema.upsert_atom(c, _atom("gh:swept", entry_mode="crawled"))
    schema.upsert_atom(c, _atom("gh:pointed", entry_mode="author_referenced"))
    schema.upsert_atom(c, _atom("x:saved", entry_mode="user-saved"))
    # Write BEHIND the migration: `connect` already ran it on the empty store above, so the row
    # has to be planted after that to be there for the next connect to find.
    c.commit()
    c.close()

    c = schema.connect(p)
    modes = dict(c.execute("SELECT atom_id, entry_mode FROM atoms"))
    assert modes == {"gh:swept": "oracle-footprint", "gh:pointed": "author_referenced",
                     "x:saved": "user-saved"}
    c.close()


def test_the_crawled_rename_converges_to_a_read_with_no_write(tmp_path):
    """The guard is a READ that no-ops once converged — NOT a bare UPDATE on every connect.

    That shape is the point: an unguarded backfill takes a write lock every time the store opens,
    which is what
    docs/Future-Investigations/2026-08-25-lock-contention-is-a-migration-backfill-on-every-connect.md
    identifies as this store's source of `database is locked`. Asserting the no-op is what keeps
    someone from "simplifying" the guard away.
    """
    c = schema.connect(tmp_path / "clean.db")
    schema.upsert_atom(c, _atom("x:saved", entry_mode="user-saved"))

    # `set_trace_callback` sees every statement, which is what distinguishes "took no write lock"
    # from "ran an UPDATE that matched nothing" — the second still locks, and `total_changes`
    # cannot tell them apart.
    seen: list[str] = []
    c.set_trace_callback(seen.append)
    schema._rename_crawled_to_footprint(c)
    c.set_trace_callback(None)
    assert not any("UPDATE" in sql.upper() for sql in seen), seen
    c.close()


def test_migration_drops_the_write_only_follows_table(tmp_path):
    """A store carrying `oracle_follows` loses it on connect, and the drop converges to a no-op.

    The table was W0 substrate for a reader nobody built: no SELECT, no caller on its writer, a
    CLI with no entry point, and 0 rows on the live store. The no-op half is the load-bearing
    half — an unguarded DROP would take a write lock on every connect forever.
    """
    p = tmp_path / "old.db"
    c = schema.connect(p)
    c.execute("CREATE TABLE oracle_follows (oracle_id TEXT, target_id TEXT)")
    c.execute("INSERT INTO oracle_follows VALUES ('x:user:1', 'x:user:2')")
    c.commit()
    c.close()

    c = schema.connect(p)
    assert c.execute("SELECT name FROM sqlite_master WHERE name='oracle_follows'").fetchone() is None

    seen: list[str] = []
    c.set_trace_callback(seen.append)
    schema._drop_oracle_follows(c)
    c.set_trace_callback(None)
    assert not any("DROP" in sql.upper() for sql in seen), seen
    c.close()


def test_replace_chunks_keeps_fts_in_sync(conn):
    schema.upsert_atom(conn, _atom())
    schema.replace_chunks(conn, "x:1", [
        {"seq": 0, "char_start": 0, "char_end": 20, "text": "autonomous agent framework", "vector": None},
        {"seq": 1, "char_start": 18, "char_end": 40, "text": "with tool use", "vector": None},
    ])
    hits = conn.execute(
        "SELECT atom_id FROM chunks_fts WHERE chunks_fts MATCH 'agent'").fetchall()
    assert [h["atom_id"] for h in hits] == ["x:1"]

    # A changed snapshot with FEWER chunks must not leave stale FTS rows behind.
    schema.replace_chunks(conn, "x:1", [
        {"seq": 0, "char_start": 0, "char_end": 10, "text": "totally different crypto text", "vector": None},
    ])
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE atom_id='x:1'").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'agent'").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'crypto'").fetchone()[0] == 1


def test_load_hashes_is_the_idempotency_ledger(conn):
    schema.upsert_atom(conn, _atom("x:1", raw_hash="aaa"))
    schema.upsert_atom(conn, _atom("x:2", raw_hash="bbb"))
    schema.upsert_atom(conn, _atom("github:o/r", source_type="github", raw_hash="ccc"))
    assert schema.load_hashes(conn, "x") == {"x:1": "aaa", "x:2": "bbb"}
    assert schema.load_hashes(conn, "github") == {"github:o/r": "ccc"}


def test_no_relation_store_survives_the_migration(conn):
    """`edges` and `entity_trust` are both DROPPED on connect. Pinned so neither regrows: each was
    write-only for months, and an unread table with a plausible name is what a future reader
    trusts. Relations are the host's job at query time, from the raw it opens."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "edges" not in tables and "entity_trust" not in tables
    assert not hasattr(schema, "upsert_edge") and not hasattr(schema, "set_trust")


def _signal(conn, entity_id, signal_type, platform):
    return conn.execute(
        "SELECT count, extra FROM curation_signals WHERE entity_id=? AND signal_type=? "
        "AND platform=?", (entity_id, signal_type, platform)).fetchone()


def test_add_signal_sums_count_on_conflict(conn):
    # Two 'like' writes on one entity fold into one row with summed count.
    schema.add_signal(conn, "x:user:1", "like", "x")
    schema.add_signal(conn, "x:user:1", "like", "x")
    row = _signal(conn, "x:user:1", "like", "x")
    assert row["count"] == 2
    # An aggregated write sums too (count is cumulative action strength).
    schema.add_signal(conn, "x:user:1", "like", "x", count=5)
    assert _signal(conn, "x:user:1", "like", "x")["count"] == 7


def test_distinct_signal_types_coexist(conn):
    # signal_type is part of the PK → a like and a follow on one entity are two rows.
    schema.add_signal(conn, "x:user:1", "like", "x", count=3)
    schema.add_signal(conn, "x:user:1", "follow", "x")
    assert _signal(conn, "x:user:1", "like", "x")["count"] == 3
    assert _signal(conn, "x:user:1", "follow", "x")["count"] == 1
    rows = conn.execute(
        "SELECT COUNT(*) FROM curation_signals WHERE entity_id='x:user:1'").fetchone()[0]
    assert rows == 2


def test_add_signal_extra_json_and_coalesce(conn):
    # extra is JSON-encoded; a later non-null extra refreshes it, a null does not clobber.
    schema.add_signal(conn, "x:user:9", "list", "x", extra={"list_names": ["AI"]})
    assert json.loads(_signal(conn, "x:user:9", "list", "x")["extra"]) == {"list_names": ["AI"]}
    schema.add_signal(conn, "x:user:9", "list", "x")  # extra=None → keep prior
    assert json.loads(_signal(conn, "x:user:9", "list", "x")["extra"]) == {"list_names": ["AI"]}
    schema.add_signal(conn, "x:user:9", "list", "x", extra={"list_names": ["AI", "Crypto"]})
    assert json.loads(_signal(conn, "x:user:9", "list", "x")["extra"]) == {
        "list_names": ["AI", "Crypto"]}


# ── set_signal: the FULL-SET re-read operator ───────────────────────────────────
#
# `add_signal` sums, which is right for an EVENT (one write per atomic action). The four
# people-only collectors are full-set re-reads that hand over a person's whole aggregate, so
# summing added a total to a total. Automatic at a 6h floor, that inflated the live store's
# `follow/x` from 468 to 886 in one pass.

def test_set_signal_replaces_instead_of_summing(conn):
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    row = conn.execute("SELECT count FROM curation_signals WHERE entity_id='x:user:1'").fetchone()
    assert row["count"] == 15          # add_signal would have made this 45


def test_set_signal_tracks_a_real_change(conn):
    """Replacement is not "ignore the new value" — a count that genuinely moved must land."""
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    schema.set_signal(conn, "x:user:1", "like", "x", count=18)
    assert conn.execute(
        "SELECT count FROM curation_signals WHERE entity_id='x:user:1'").fetchone()["count"] == 18
    schema.set_signal(conn, "x:user:1", "like", "x", count=2)      # ...downward too
    assert conn.execute(
        "SELECT count FROM curation_signals WHERE entity_id='x:user:1'").fetchone()["count"] == 2


def test_add_signal_still_sums_for_the_event_shape(conn):
    """PINNED. `sync_bookmarks` and `sync_substack_saved` stamp `save` ONCE per atom, on the run
    that first ingests it. That is a real event stream and SUM is correct for it — this fix must
    not quietly convert every signal writer to replacement."""
    schema.add_signal(conn, "x:user:2", "save", "x")
    schema.add_signal(conn, "x:user:2", "save", "x")
    assert conn.execute(
        "SELECT count FROM curation_signals WHERE entity_id='x:user:2'").fetchone()["count"] == 2


def test_the_two_operators_share_everything_but_the_count(conn):
    """`extra` was never additive, so both must handle it identically: a newer non-null value
    wins, and a null never clobbers what is stored."""
    for fn, eid in ((schema.add_signal, "x:user:3"), (schema.set_signal, "x:user:4")):
        fn(conn, eid, "list", "x", extra={"list_names": ["AI"]})
        fn(conn, eid, "list", "x")                                  # extra=None → keep prior
        row = conn.execute("SELECT extra FROM curation_signals WHERE entity_id=?", (eid,)).fetchone()
        assert json.loads(row["extra"]) == {"list_names": ["AI"]}, eid
        fn(conn, eid, "list", "x", extra={"list_names": ["AI", "Crypto"]})
        row = conn.execute("SELECT extra FROM curation_signals WHERE entity_id=?", (eid,)).fetchone()
        assert json.loads(row["extra"]) == {"list_names": ["AI", "Crypto"]}, eid
