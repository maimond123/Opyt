"""The notice that ends the silence measured on 2026-09-15 — 1,004 atoms in the store and
"anything else you want to add?" as the last thing OPYT said."""
import sqlite3

import pytest

from pipeline.kb import new_material


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A store with an `atoms` table and nothing else. The notice must work off atoms alone —
    the tour is optional and the rest of the schema is not its business."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atoms (atom_id TEXT PRIMARY KEY, source_type TEXT, "
                 "entry_mode TEXT)")
    return conn


def _add(conn, source_type, entry_mode, n, *, start=0):
    conn.executemany("INSERT INTO atoms VALUES (?,?,?)",
                     [(f"{source_type}-{entry_mode}-{i + start}", source_type, entry_mode)
                      for i in range(n)])


def test_an_empty_store_says_nothing(store):
    """Nothing has landed, so there is nothing to lead with. The notice must not become a
    greeting."""
    assert new_material.new_material_notice(store) is None


def test_the_first_call_after_material_lands_leads_with_it(store):
    """THE MEASURED FAILURE, inverted. 1,001 of the 1,004 atoms were bookmarks, which never pass
    through an ingest presentation — so the tour that existed could not have described them."""
    _add(store, "x", "user-saved", 1001)

    notice = new_material.new_material_notice(store)

    assert notice["in_your_library"] == [{"material": "the X posts you saved yourself",
                                          "items": 1001}]
    assert "LEAD WITH THIS" in notice["host_note"]


def test_it_rides_once_and_then_never_again(store):
    """Un-stamped it would ride every call forever and become furniture, and the reader would
    learn to skip the one field that says something happened — `completion_notice`'s own rule."""
    _add(store, "x", "user-saved", 12)
    assert new_material.new_material_notice(store) is not None
    assert new_material.new_material_notice(store) is None


def test_growth_within_a_kind_stays_silent(store):
    """A background pull streams atoms in over minutes. If growth fired the notice, it would fire
    on every call for as long as the pull ran, which is the furniture this is built to avoid."""
    _add(store, "x", "oracle-footprint", 9)
    assert new_material.new_material_notice(store) is not None
    _add(store, "x", "oracle-footprint", 400, start=9)
    assert new_material.new_material_notice(store) is None


def test_a_genuinely_new_KIND_speaks_again(store):
    """Bounded is not silent. Posts the user saved themselves and the archives of people they
    chose to track are different material and different news, even when both are X."""
    _add(store, "x", "user-saved", 1001)
    assert new_material.new_material_notice(store) is not None

    _add(store, "substack", "oracle-footprint", 40)
    second = new_material.new_material_notice(store)

    assert second["in_your_library"] == [
        {"material": "the Substack archives of the people you track", "items": 40}]


def test_an_unlabelled_source_is_still_announced(store):
    """A silent drop is how a new `source_type` would go unannounced forever with nothing saying
    so. Fall back to a plain phrase rather than losing the material."""
    _add(store, "podcast", "oracle-footprint", 3)

    notice = new_material.new_material_notice(store)

    assert notice["in_your_library"] == [
        {"material": "podcast material from the people you track", "items": 3}]


def test_a_store_with_no_atoms_table_is_not_an_error(store, tmp_path):
    """A first call on a home too young to have a schema is the normal case, not a failure. This
    rides on somebody else's answer and must never be able to break it."""
    bare = sqlite3.connect(":memory:")
    assert new_material.new_material_notice(bare) is None


def test_an_unreadable_stamp_repeats_rather_than_swallows(store, tmp_path):
    """The direction is deliberate: a corrupt stamp costs a repeated sentence, not a user who
    never learns what is in their own library."""
    _add(store, "x", "user-saved", 5)
    assert new_material.new_material_notice(store) is not None
    (tmp_path / "material_announced.json").write_text("{not json")

    assert new_material.new_material_notice(store) is not None
