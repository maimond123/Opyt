"""The `save` presence repaired from the user-saved atoms that prove it (`curation/reconcile`)."""

import sqlite3

import pytest

from pipeline.curation import reconcile
from pipeline.kb import schema


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _sig(conn, entity_id, signal_type, platform="x"):
    return conn.execute(
        "SELECT count, extra FROM curation_signals WHERE entity_id=? AND signal_type=? "
        "AND platform=?", (entity_id, signal_type, platform)).fetchone()


# The stampers fire once per atom, on the run that ingests it. These prove the reconcile
# repairs what that write-once property can lose, WITHOUT the two ways a naive fix breaks:
# inflating `count` (add_signal SUMS) and inventing nameless candidates (no entities row).

def _saved_atom(conn, *, atom_id, who_id, source_type, entry_mode="user-saved"):
    schema.upsert_atom(conn, {
        "atom_id": atom_id, "source_type": source_type, "what_kind": "opinion",
        "who_id": who_id, "when_ts": "2026-08-01T00:00:00+00:00", "when_precision": "day",
        "about_entities": None, "source_url": None,
        "raw_hash": "h", "description": None, "payload": None,
        "entry_mode": entry_mode, "basis": "observed", "body_state": "complete",
    })


def test_reconcile_stamps_a_save_for_an_unsignalled_bookmark_author(conn):
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    assert _sig(conn, "x:user:7", "save") is None          # the drift this repairs

    out = reconcile.reconcile_saved_signals(conn)

    assert out["inserted"] == {"x": 1}
    assert _sig(conn, "x:user:7", "save")["count"] == 1
    assert out["signal_bearing_entities"] == 1


def test_reconcile_covers_substack_saves_on_the_same_pass(conn):
    schema.upsert_entity(conn, "substack:acme", name="Acme")
    _saved_atom(conn, atom_id="substack:9", who_id="substack:acme", source_type="substack")

    out = reconcile.reconcile_saved_signals(conn)

    assert out["inserted"] == {"substack": 1}
    assert _sig(conn, "substack:acme", "save", platform="substack")["count"] == 1


def test_reconcile_is_idempotent_and_never_inflates_count(conn):
    """The whole reason this is insert-if-absent and not `add_signal`: it runs on every read."""
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    _saved_atom(conn, atom_id="x:2", who_id="x:user:7", source_type="x")   # 2 atoms, 1 author

    first = reconcile.reconcile_saved_signals(conn)
    for _ in range(5):
        again = reconcile.reconcile_saved_signals(conn)
        assert again["inserted"] == {}                     # nothing left to do

    assert first["inserted"] == {"x": 1}                   # DISTINCT: one row for two atoms
    assert _sig(conn, "x:user:7", "save")["count"] == 1     # never accumulated


def test_reconcile_leaves_a_stamped_signal_and_its_count_alone(conn):
    """A real like/save history must survive the reconcile untouched — it repairs absence only."""
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    schema.set_signal(conn, "x:user:7", "save", "x", count=4)
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")

    out = reconcile.reconcile_saved_signals(conn)

    assert out["inserted"] == {}
    assert _sig(conn, "x:user:7", "save")["count"] == 4


def test_reconcile_reports_an_orphan_author_instead_of_inventing_one(conn):
    """No entities row → a signal would make a NAMELESS candidate. Report, never fabricate."""
    _saved_atom(conn, atom_id="x:1", who_id="x:user:404", source_type="x")

    out = reconcile.reconcile_saved_signals(conn)

    assert out["inserted"] == {}
    assert out["orphans"] == {"x": 1}
    assert _sig(conn, "x:user:404", "save") is None


def test_reconcile_ignores_atoms_the_user_did_not_save(conn):
    """`entry_mode` is the curation act. Oracle footprint is corpus, not a curation signal —
    stamping it would make every account an Oracle ever quoted into a candidate."""
    schema.upsert_entity(conn, "x:user:8", name="Bob")
    _saved_atom(conn, atom_id="x:3", who_id="x:user:8", source_type="x",
                entry_mode="oracle-footprint")

    out = reconcile.reconcile_saved_signals(conn)

    assert out["inserted"] == {}
    assert _sig(conn, "x:user:8", "save") is None




def test_reconciled_author_becomes_a_ranked_candidate(conn):
    """The consumer's view: the reconcile's job is not a row, it is a candidate."""
    from pipeline.oracles import ranking

    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    assert ranking.rank_candidates(conn) == []

    reconcile.reconcile_saved_signals(conn)

    cands = ranking.rank_candidates(conn)
    assert [c.name for c in cands] == ["Ada"]


def test_a_reconciled_store_is_only_read_while_another_writer_holds_the_lock(conn, tmp_path):
    """B-C1: it runs on every candidates read, so with nothing missing it must not ask for the
    write lock. Before the fix it ran its INSERT anyway and failed with `database is locked`."""
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    reconcile.reconcile_saved_signals(conn)
    conn.commit()

    writer = sqlite3.connect(tmp_path / "opyt.db")
    writer.execute("BEGIN IMMEDIATE")
    conn.execute("PRAGMA busy_timeout=100")
    try:
        out = reconcile.reconcile_saved_signals(conn)
    finally:
        writer.rollback()
        writer.close()

    assert out["inserted"] == {}
    assert _sig(conn, "x:user:7", "save")["count"] == 1


def test_a_repair_commits_through_one_write_transaction(conn, tmp_path):
    """What it inserts is committed when it returns: another connection sees it."""
    schema.upsert_entity(conn, "x:user:7", name="Ada")
    _saved_atom(conn, atom_id="x:1", who_id="x:user:7", source_type="x")
    conn.commit()

    assert reconcile.reconcile_saved_signals(conn)["inserted"] == {"x": 1}
    assert not conn.in_transaction
    reader = sqlite3.connect(tmp_path / "opyt.db")
    try:
        assert reader.execute("SELECT count FROM curation_signals WHERE entity_id='x:user:7' "
                              "AND signal_type='save' AND platform='x'").fetchone() == (1,)
    finally:
        reader.close()
