"""schema — the substrate contracts (upsert identity, FTS sync, the curation-signal
operators, and the migration that dropped the two write-only relation stores). Uses an
explicit db_path so no $OPYT_HOME needed."""
from __future__ import annotations

import json
import sqlite3

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
        raw_hash="h1", description="a card",
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
    """The reason the column exists. `ingested_at` is refreshed by every upsert, so it answers
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
    assert row["ingested_at"] > "2026-01-01 00:00:00"    # refreshed


def test_first_seen_is_written_on_insert_without_relying_on_a_column_default(conn):
    """A migrated store's `first_seen` has no default — SQLite forbids `datetime('now')` in ADD
    COLUMN — so `upsert_atom` supplies it as a literal. Pin that the writer, not the DDL, is what
    populates it, or the migrated path silently writes NULL and every arrival date is lost."""
    assert "first_seen" not in schema._ATOM_COLS        # not caller-supplied
    assert "first_seen" not in schema._ATOM_UPDATABLE   # adding it would re-create the bug
    schema.upsert_atom(conn, _atom("x:9"))
    assert conn.execute(
        "SELECT first_seen FROM atoms WHERE atom_id='x:9'").fetchone()[0] is not None


def _index_matches_bodies(conn) -> None:
    """FTS5's own check of an external-content index against its content table: raises when the
    index holds a term the `bodies` rows do not, or misses one they do."""
    conn.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")


def _matching_atoms(conn, word: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT b.atom_id FROM bodies_fts JOIN bodies b ON b.body_id = bodies_fts.rowid "
        "WHERE bodies_fts MATCH ? ORDER BY b.atom_id", (word,))]


def test_put_body_keeps_the_index_in_step_with_bodies(conn):
    """Every way a body is written (first write, a rewrite with less text, a rewrite with more,
    a delete as `forget` makes) leaves the index matching `bodies` exactly."""
    schema.upsert_atom(conn, _atom())
    schema.upsert_atom(conn, _atom("x:2"))
    schema.put_body(conn, "x:1", "autonomous agent framework with tool use")
    schema.put_body(conn, "x:2", "agent memory")
    _index_matches_bodies(conn)
    assert _matching_atoms(conn, "agent") == ["x:1", "x:2"]

    # A changed snapshot with less text must not leave stale index terms behind.
    schema.put_body(conn, "x:1", "totally different crypto text")
    _index_matches_bodies(conn)
    assert conn.execute("SELECT COUNT(*) FROM bodies WHERE atom_id='x:1'").fetchone()[0] == 1
    assert _matching_atoms(conn, "agent") == ["x:2"]
    assert _matching_atoms(conn, "crypto") == ["x:1"]

    schema.put_body(conn, "x:1", " ".join(f"part{i} agents" for i in range(3)))
    conn.execute("DELETE FROM bodies WHERE atom_id = 'x:2'")
    _index_matches_bodies(conn)
    assert _matching_atoms(conn, "agent") == ["x:1"]       # one row per atom, never per part
    assert _matching_atoms(conn, "memory") == []


def test_the_index_check_catches_a_body_changed_behind_its_back(conn):
    """The check above has teeth: an index edited outside the triggers is drift, and drift fails
    it. This is the failure an external-content index invites."""
    schema.upsert_atom(conn, _atom())
    schema.put_body(conn, "x:1", "agent memory")
    bid = conn.execute("SELECT body_id FROM bodies WHERE atom_id = 'x:1'").fetchone()[0]
    conn.execute("INSERT INTO bodies_fts(bodies_fts, rowid, text) VALUES ('delete', ?, ?)",
                 (bid, "agent memory"))
    with pytest.raises(sqlite3.DatabaseError):
        _index_matches_bodies(conn)


def test_a_writer_that_names_atom_id_cannot_index_a_body(conn):
    """A writer that predates the external-content index inserts `(text, atom_id)` with no docid.
    The index has no `atom_id` column so that insert fails, instead of indexing the body under a
    docid that is not its `body_id`. Do not add the column back."""
    with pytest.raises(sqlite3.OperationalError, match="atom_id"):
        conn.execute("INSERT INTO bodies_fts (text, atom_id) VALUES ('a', 'x:1')")


def test_load_hashes_is_the_idempotency_ledger(conn):
    schema.upsert_atom(conn, _atom("x:1", raw_hash="aaa"))
    schema.upsert_atom(conn, _atom("x:2", raw_hash="bbb"))
    schema.upsert_atom(conn, _atom("github:o/r", source_type="github", raw_hash="ccc"))
    assert schema.load_hashes(conn, "x") == {"x:1": "aaa", "x:2": "bbb"}
    assert schema.load_hashes(conn, "github") == {"github:o/r": "ccc"}


def test_no_relation_store_is_created(conn):
    """`edges` and `entity_trust` do not exist. Pinned so neither regrows: each was
    write-only, and an unread table with a plausible name is what a future reader
    trusts. Relations are the host's job at query time, from the raw it opens."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "edges" not in tables and "entity_trust" not in tables
    assert not hasattr(schema, "upsert_edge") and not hasattr(schema, "set_trust")


def _signal(conn, entity_id, signal_type, platform):
    return conn.execute(
        "SELECT count, extra FROM curation_signals WHERE entity_id=? AND signal_type=? "
        "AND platform=?", (entity_id, signal_type, platform)).fetchone()


def test_distinct_signal_types_coexist(conn):
    # signal_type is part of the PK → a like and a follow on one entity are two rows.
    schema.ensure_signal(conn, "x:user:1", "like", "x", count=3)
    schema.ensure_signal(conn, "x:user:1", "follow", "x")
    assert _signal(conn, "x:user:1", "like", "x")["count"] == 3
    assert _signal(conn, "x:user:1", "follow", "x")["count"] == 1
    rows = conn.execute(
        "SELECT COUNT(*) FROM curation_signals WHERE entity_id='x:user:1'").fetchone()[0]
    assert rows == 2


def test_ensure_signal_extra_json_and_coalesce(conn):
    # extra is JSON-encoded; a later non-null extra refreshes it, a null does not clobber.
    schema.ensure_signal(conn, "x:user:9", "list", "x", extra={"list_names": ["AI"]})
    assert json.loads(_signal(conn, "x:user:9", "list", "x")["extra"]) == {"list_names": ["AI"]}
    schema.ensure_signal(conn, "x:user:9", "list", "x")  # extra=None → keep prior
    assert json.loads(_signal(conn, "x:user:9", "list", "x")["extra"]) == {"list_names": ["AI"]}
    schema.ensure_signal(conn, "x:user:9", "list", "x", extra={"list_names": ["AI", "Crypto"]})
    assert json.loads(_signal(conn, "x:user:9", "list", "x")["extra"]) == {
        "list_names": ["AI", "Crypto"]}


# ── set_signal: the full-set re-read operator ───────────────────────────────────
#
# Summing suits an event stream (one write per atomic action). The people-only collectors are
# full-set re-reads that hand over a person's whole aggregate, so summing would add a total to a
# total and inflate counts on every automatic re-read.

def test_set_signal_replaces_instead_of_summing(conn):
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    row = conn.execute("SELECT count FROM curation_signals WHERE entity_id='x:user:1'").fetchone()
    assert row["count"] == 15          # summing would have made this 45


def test_set_signal_tracks_a_real_change(conn):
    """Replacement is not "ignore the new value" — a count that genuinely moved must land."""
    schema.set_signal(conn, "x:user:1", "like", "x", count=15)
    schema.set_signal(conn, "x:user:1", "like", "x", count=18)
    assert conn.execute(
        "SELECT count FROM curation_signals WHERE entity_id='x:user:1'").fetchone()["count"] == 18
    schema.set_signal(conn, "x:user:1", "like", "x", count=2)      # ...downward too
    assert conn.execute(
        "SELECT count FROM curation_signals WHERE entity_id='x:user:1'").fetchone()["count"] == 2


def test_the_two_operators_share_everything_but_the_count(conn):
    """`extra` was never additive, so both must handle it identically: a newer non-null value
    wins, and a null never clobbers what is stored."""
    for fn, eid in ((schema.ensure_signal, "x:user:3"), (schema.set_signal, "x:user:4")):
        fn(conn, eid, "list", "x", extra={"list_names": ["AI"]})
        fn(conn, eid, "list", "x")                                  # extra=None → keep prior
        row = conn.execute("SELECT extra FROM curation_signals WHERE entity_id=?", (eid,)).fetchone()
        assert json.loads(row["extra"]) == {"list_names": ["AI"]}, eid
        fn(conn, eid, "list", "x", extra={"list_names": ["AI", "Crypto"]})
        row = conn.execute("SELECT extra FROM curation_signals WHERE entity_id=?", (eid,)).fetchone()
        assert json.loads(row["extra"]) == {"list_names": ["AI", "Crypto"]}, eid


# ── referenced_by: who pointed at an atom, carried across rewrites ─────────

def _ref(post="xprofile:1", by="x:user:99"):
    return {"by": by, "handle": "carol", "post": post, "url": "u", "date": "2026-01-06"}


def test_a_reference_lands_once_per_post(tmp_path):
    import json
    conn = schema.connect(tmp_path / "opyt.db")
    schema.upsert_atom(conn, {"atom_id": "github:a/b", "source_type": "github",
                              "payload": {"stars": 1}})
    assert schema.add_reference(conn, "github:a/b", _ref()) is True
    assert schema.add_reference(conn, "github:a/b", _ref()) is False         # same post, again
    assert schema.add_reference(conn, "github:a/b", _ref(post="xprofile:2")) is True
    assert schema.add_reference(conn, "nope", _ref()) is False               # never creates a row
    payload = json.loads(conn.execute("SELECT payload FROM atoms").fetchone()[0])
    assert payload["stars"] == 1 and [r["post"] for r in payload["referenced_by"]] == \
        ["xprofile:1", "xprofile:2"]
    assert conn.execute("SELECT version FROM atoms").fetchone()[0] == 1, "no re-observation"


def test_a_rewrite_keeps_the_referrers(tmp_path):
    """A repo re-synced through its owner's own GitHub source, or an X post later bookmarked,
    rebuilds its payload from the source — which knows nothing of who referenced it."""
    import json
    conn = schema.connect(tmp_path / "opyt.db")
    atom = {"atom_id": "x:5", "source_type": "x", "payload": {"like_count": 1}}
    schema.upsert_atom(conn, atom)
    schema.add_reference(conn, "x:5", _ref())
    schema.upsert_atom(conn, {**atom, "payload": {"like_count": 9}, "entry_mode": "user-saved"})
    payload = json.loads(conn.execute("SELECT payload FROM atoms").fetchone()[0])
    assert payload == {"like_count": 9, "referenced_by": [_ref()]}
    # An incoming payload that names its own list wins — the carry-over is a default, not a lock.
    schema.upsert_atom(conn, {**atom, "payload": {"like_count": 9, "referenced_by": []}})
    assert json.loads(conn.execute("SELECT payload FROM atoms").fetchone()[0])["referenced_by"] == []
