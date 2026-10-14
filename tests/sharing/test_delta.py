"""Changes-only uploads: a served copy patched with the rows that changed holds exactly what a
fresh export of the same store holds (`pipeline/sharing/delta.py`, `fingerprint.py`).

That equality is the subject, so the tests end the same way: the patched copy compared with a
fresh build table by table, rowids included, read with plain SQL rather than through the
fingerprint under test; its search index checked against its bodies by FTS5 itself; and then
searched like one. The refusals are the other half: a delta that does not fit the copy it meets
changes nothing, so the push can go whole instead.
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from pipeline.forget import atoms as forget
from pipeline.kb import schema
from pipeline.kb.ingest_common import promote_atom
from pipeline.oracles import roster
from pipeline.retrieval.engine import search_atoms
from pipeline.sharing import delta, export, fingerprint
from tests.kb.test_export import _QUERIES, A, B, C, _add, _corpus, _hit_tuple

NEW = "github:arrived/after-the-upload"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A live store holding the export tests' corpus, and a place to put copies of it."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path / "live"))
    conn = schema.connect()
    _corpus(conn)
    conn.close()
    return SimpleNamespace(tmp=tmp_path, builds=0)


def _export(store):
    store.builds += 1
    out = store.tmp / f"export-{store.builds}.db"
    export.build_export(out)
    return out


def _serve(store, export_path):
    """What a full upload leaves on the service: the export, stamped with its digest."""
    served = store.tmp / "served.db"
    shutil.copyfile(export_path, served)
    fingerprint.stamp(served, fingerprint.of_copy(export_path).digest)
    return served


def _cut(store, old_export, new_export, changes=None):
    old, new = fingerprint.of_copy(old_export), fingerprint.of_copy(new_export)
    out = store.tmp / "push.delta"
    delta.build(new_export, changes or fingerprint.diff(old, new), old.digest, new.digest, out)
    return out


def _push(store, served, old_export, new_export, **kw) -> dict:
    return delta.apply(served, _cut(store, old_export, new_export), **kw)


def _rows(path) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"SELECT rowid, * FROM {t} ORDER BY rowid").fetchall()
                for t in export.TABLES}
    finally:
        conn.close()


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_MATCH = ("SELECT b.atom_id FROM bodies_fts JOIN bodies b ON b.body_id = bodies_fts.rowid "
          "WHERE bodies_fts MATCH ?")


def _assert_same(patched, fresh):
    """Every row the same, rowids included, the search index in step with the bodies, and every
    query of the export tests' sweep answering the same hits in the same order."""
    assert _rows(patched) == _rows(fresh)
    conn = sqlite3.connect(patched)
    try:
        conn.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")
    finally:
        conn.close()
    a = schema.connect(patched, read_only=True)
    b = schema.connect(fresh, read_only=True)
    try:
        for spec in _QUERIES:
            kwargs = {k: v for k, v in spec.items() if k != "query"}
            hits_a = search_atoms(a, spec["query"], **kwargs).hits
            hits_b = search_atoms(b, spec["query"], **kwargs).hits
            assert [_hit_tuple(h) for h in hits_a] == [_hit_tuple(h) for h in hits_b], spec
    finally:
        a.close()
        b.close()
    conn = sqlite3.connect(f"file:{patched}?mode=ro", uri=True)
    try:
        assert fingerprint.stamped(conn) == fingerprint.of_copy(fresh).digest
    finally:
        conn.close()


def _delta_rows(path) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in (*export.TABLES, delta.DELETES)}
    finally:
        conn.close()


# ── each change the store makes ──────────────────────────────────────────────────

def test_a_new_atom_crosses_as_its_own_rows_only(store):
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    _add(conn, NEW, "github", "artifact", "github:arrived", ["ai-agents"],
         "an autonomous agent library that arrived after the first upload")
    conn.close()
    second = _export(store)

    cut = _delta_rows(_cut(store, first, second))
    assert cut == {"atoms": 1, "bodies": 1, "entities": 0, "oracles": 0, delta.DELETES: 0}

    res = _push(store, served, first, second)
    assert res == {"digest": fingerprint.of_copy(second).digest, "written": 2, "deleted": 0}
    _assert_same(served, second)
    conn = sqlite3.connect(served)
    try:
        assert conn.execute(_MATCH, ("arrived",)).fetchall() == [(NEW,)]
    finally:
        conn.close()


def test_a_promoted_atom_rewrites_its_row_and_nothing_else(store):
    """`promote_atom` changes `entry_mode` and `promoted_at` on the atom row; its body is
    untouched, so it stays off the wire."""
    conn = schema.connect()
    _add(conn, NEW, "github", "artifact", "github:arrived", ["ai-agents"],
         "a machine find a person later saved", entry_mode="frontier")
    conn.close()
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    promote_atom(conn, NEW, "user-saved")
    conn.close()
    second = _export(store)

    assert _delta_rows(_cut(store, first, second))["atoms"] == 1
    assert _push(store, served, first, second)["written"] == 1
    _assert_same(served, second)


def test_a_body_changed_in_place_keeps_its_key_and_reindexes(store):
    """A new snapshot rewrites the body under the same `body_id`, so the delta carries that one
    row, and the served copy's triggers un-index the old text as the row is replaced: a search
    finds the new words and not the old."""
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    body_id = conn.execute("SELECT body_id FROM bodies WHERE atom_id = ?", (A,)).fetchone()[0]
    _add(conn, A, "github", "artifact", "github:root", ["ai-agents"],
         "a rewritten readme about a dashboard for react", entry_mode="user-saved")
    conn.close()
    second = _export(store)

    changes = fingerprint.diff(fingerprint.of_copy(first), fingerprint.of_copy(second))
    assert changes.write["bodies"] == [body_id]
    _push(store, served, first, second)
    _assert_same(served, second)
    conn = sqlite3.connect(served)
    try:
        assert conn.execute("SELECT body_id FROM bodies WHERE atom_id = ?",
                            (A,)).fetchone()[0] == body_id
        assert conn.execute(_MATCH, ("rewritten",)).fetchall() == [(A,)]
        assert (A,) not in conn.execute(_MATCH, ("autonomous",)).fetchall()
    finally:
        conn.close()


def test_a_forgotten_atom_leaves_every_table(store):
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    assert forget.atom(conn, B, confirm=True)["status"] == "forgotten"
    conn.close()
    second = _export(store)

    _push(store, served, first, second)
    _assert_same(served, second)
    conn = sqlite3.connect(served)
    try:
        for table in ("atoms", "bodies"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE atom_id = ?",
                                (B,)).fetchone()[0] == 0, table
        assert (B,) not in conn.execute(_MATCH, ("agent",)).fetchall()
    finally:
        conn.close()


def test_entities_and_oracles_follow(store):
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    schema.upsert_entity(conn, "github:root", name="Root Developer")
    roster.upsert_oracle(conn, "github:root", name="Root Developer")
    roster.end_subscription(conn, "x:user:1")
    conn.close()
    second = _export(store)

    _push(store, served, first, second)
    _assert_same(served, second)


def test_a_scraped_bio_never_reaches_the_wire(store):
    """The export keeps only `profile.handle`, so a bio change moves nothing a reader can see,
    and the delta is empty."""
    first = _export(store)
    conn = schema.connect()
    schema.upsert_entity(conn, "x:user:1", profile={"bio": "a new scraped bio"})
    conn.close()
    second = _export(store)

    assert fingerprint.of_copy(first).digest == fingerprint.of_copy(second).digest
    assert not any(_delta_rows(_cut(store, first, second)).values())


def test_a_day_of_changes_at_once_and_then_another(store):
    """Several pushes in a row, each patching the copy the previous one left, still equal a
    fresh build at every step."""
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    _add(conn, NEW, "github", "artifact", "github:arrived", ["crypto"],
         "a rollup proof library")
    _add(conn, C, "x", "opinion", "x:user:1", ["crypto"],
         "second thoughts on rollup and proof systems", entry_mode="user-saved")
    forget.atom(conn, B, confirm=True)
    schema.upsert_entity(conn, "github:arrived", name="Arrived")
    conn.close()
    second = _export(store)
    _push(store, served, first, second)
    _assert_same(served, second)

    conn = schema.connect()
    forget.atom(conn, NEW, confirm=True)
    _add(conn, "x:3", "x", "opinion", "x:user:2", ["web-dev"], "web tools for react")
    conn.close()
    third = _export(store)
    _push(store, served, second, third)
    _assert_same(served, third)


# ── what is refused, and that a refusal writes nothing ───────────────────────────

def _changed(store):
    first = _export(store)
    conn = schema.connect()
    _add(conn, NEW, "github", "artifact", "github:arrived", ["ai-agents"],
         "an agent framework that arrived later")
    conn.close()
    return first, _export(store)


def test_a_delta_cut_against_another_copy_is_refused(store):
    """The service is ahead of the owner (a lost lease after the upload): the base does not match,
    and applying anyway could leave rows nobody deletes."""
    first, second = _changed(store)
    conn = schema.connect()
    _add(conn, "x:3", "x", "opinion", "x:user:2", ["web-dev"], "web tools")
    conn.close()
    third = _export(store)
    served = _serve(store, third)
    before = _sha(served)

    with pytest.raises(delta.Stale, match="not the one this delta was cut against"):
        _push(store, served, first, second)
    assert _sha(served) == before


def test_a_delta_that_misses_a_change_fails_its_digest(store):
    first, second = _changed(store)
    served = _serve(store, first)
    before = _sha(served)
    full = fingerprint.diff(fingerprint.of_copy(first), fingerprint.of_copy(second))
    short = fingerprint.Changes({**full.write, "bodies": []}, full.delete)

    with pytest.raises(delta.Stale, match="did not come out as the owner's export"):
        delta.apply(served, _cut(store, first, second, short))
    assert _sha(served) == before


@pytest.mark.parametrize("change", [
    "ALTER TABLE atoms ADD COLUMN arrived_from TEXT DEFAULT 'somewhere'",
    # Not the last column: SQLite cannot drop one that trails comments in the stored DDL.
    "ALTER TABLE atoms DROP COLUMN first_seen",
])
def test_an_upgrade_that_changes_a_column_is_refused(store, change):
    """The owner's newer install exports a column the served copy lacks, or lacks one it has."""
    first, second = _changed(store)
    served = _serve(store, first)
    before = _sha(served)
    conn = sqlite3.connect(second)
    conn.execute(change)
    conn.commit()
    conn.close()

    with pytest.raises(delta.Stale):
        _push(store, served, first, second)
    assert _sha(served) == before


def test_a_copy_with_no_stamp_is_refused(store):
    """Uploaded whole by an install older than deltas."""
    first, second = _changed(store)
    served = store.tmp / "served.db"
    shutil.copyfile(first, served)

    with pytest.raises(delta.Stale, match="records no digest"):
        _push(store, served, first, second)


def test_no_copy_is_refused(store):
    first, second = _changed(store)
    with pytest.raises(delta.Stale, match="holds no copy"):
        _push(store, store.tmp / "nothing-here.db", first, second)
    assert not (store.tmp / "nothing-here.db").exists()


def test_a_damaged_delta_is_refused(store):
    first, second = _changed(store)
    served = _serve(store, first)
    junk = store.tmp / "junk.delta"
    junk.write_bytes(b"not a database" * 100)

    with pytest.raises(delta.Stale, match="could not be patched"):
        delta.apply(served, junk)


def test_a_patch_past_the_cap_is_refused(store):
    first, second = _changed(store)
    served = _serve(store, first)
    before = _sha(served)

    with pytest.raises(delta.TooLarge):
        _push(store, served, first, second, max_bytes=1000)
    assert _sha(served) == before


# ── a copy from before whole-item search, and readers ─────────────────────────────

def _older_copy(served, sql: list[str]) -> None:
    conn = sqlite3.connect(served)
    for statement in sql:
        conn.execute(statement)
    conn.commit()
    conn.close()


@pytest.mark.parametrize("older", [
    # Uploaded before whole-item search: no bodies index at all.
    ["DROP TABLE bodies_fts"],
    # An index without the triggers: a patch would leave it describing the old text.
    ["DROP TRIGGER bodies_au"],
], ids=["no-index", "no-trigger"])
def test_a_copy_from_before_whole_item_search_is_refused(store, older):
    """Only a copy whose `bodies_fts` is kept by all three triggers can be patched; anything
    else goes whole, and nothing is written."""
    first, second = _changed(store)
    served = _serve(store, first)
    _older_copy(served, older)
    before = _sha(served)

    with pytest.raises(delta.Stale, match="made by an older Opyt"):
        _push(store, served, first, second)
    assert _sha(served) == before


def test_readers_see_the_copy_before_or_after_never_between(store):
    first = _export(store)
    served = _serve(store, first)
    conn = schema.connect()
    for i in range(30):
        _add(conn, f"github:batch/{i}", "github", "artifact", "github:batch",
             ["ai-agents"], f"agent library number {i}")
    forget.atom(conn, B, confirm=True)
    conn.close()
    second = _export(store)
    cut = _cut(store, first, second)
    counts = {path: len(_rows(path)["atoms"]) for path in (first, second)}

    seen, errors, done = set(), [], threading.Event()

    def read():
        while not done.is_set():
            try:
                r = sqlite3.connect(f"file:{served}?mode=ro", uri=True)
                try:
                    seen.add(r.execute("SELECT COUNT(*) FROM atoms").fetchone()[0])
                finally:
                    r.close()
            except sqlite3.Error as e:
                errors.append(e)

    readers = [threading.Thread(target=read) for _ in range(3)]
    for t in readers:
        t.start()
    try:
        delta.apply(served, cut)
    finally:
        done.set()
        for t in readers:
            t.join()
    assert errors == []
    assert seen <= {counts[first], counts[second]}
    _assert_same(served, second)
