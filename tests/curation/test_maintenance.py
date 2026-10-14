"""Paper-author derivation and identity resolution run as curation's local work: when what they
read changed, hourly otherwise, and with no platform session at all."""

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.curation import maintenance
from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.worker.quota_loop.source import Context

NOW = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    # A paper-only home: neither platform is connected, and nothing here may ask.
    monkeypatch.setattr(connections, "_connected_x", lambda: pytest.fail("asked X"))
    monkeypatch.setattr(connections, "_connected_substack",
                        lambda: pytest.fail("asked Substack"))
    c = schema.connect(tmp_path / "opyt.db")
    maintenance.init_schema(c)
    yield c
    c.close()


def paper(conn, atom_id, authors):
    schema.upsert_atom(conn, {
        "atom_id": atom_id, "source_type": "paper", "what_kind": "artifact",
        "who_id": "paper-authors:" + atom_id, "when_ts": "2026-01-01", "when_precision": "day",
        "about_entities": [], "source_url": "", "raw_hash": atom_id,
        "description": "", "entry_mode": "user-saved", "payload": {"authors": authors}})
    conn.commit()


def drain(conn, now=NOW):
    source, out = maintenance.Maintenance(), []
    while (unit := source.next_unit(conn, now)) is not None:
        out.append(source.record(conn, unit, source.fetch(conn, unit, Context(source.quota))).status)
        assert len(out) < 10, out
    return out


def save_count(conn, entity_id):
    row = conn.execute("SELECT count FROM curation_signals WHERE entity_id=? "
                       "AND signal_type='save'", (entity_id,)).fetchone()
    return row[0] if row else None


def test_a_paper_only_home_derives_its_authors_and_resolves_them(conn):
    paper(conn, "paper:1", [{"name": "Ada", "scholar_id": "111",
                             "orcid": "0000-0001-0000-0001"}])

    assert drain(conn) == ["papers", "resolve"]
    assert save_count(conn, "scholar:111") == 1
    assert conn.execute("SELECT canonical_id FROM entities WHERE entity_id='scholar:111'"
                        ).fetchone()[0] is not None


def test_nothing_is_owed_again_until_the_inputs_change(conn):
    paper(conn, "paper:1", [{"name": "Ada", "scholar_id": "111"}])
    drain(conn)

    assert drain(conn, NOW + timedelta(minutes=30)) == []
    paper(conn, "paper:2", [{"name": "Ada", "scholar_id": "111"}])
    assert drain(conn, NOW + timedelta(minutes=31)) == ["papers"]
    assert save_count(conn, "scholar:111") == 2


def test_a_new_entity_from_elsewhere_makes_resolve_owed(conn):
    drain(conn)
    schema.upsert_entity(conn, "x:user:7", name="Ada", identity_links=["https://ada.example"])

    assert drain(conn, NOW + timedelta(minutes=1)) == ["resolve"]


def test_the_hourly_pass_runs_both_even_when_nothing_arrived(conn):
    """An edit inside a row (a backfilled author list) changes no fingerprint."""
    drain(conn)
    source = maintenance.Maintenance()

    assert source.next_due(conn, NOW) == NOW + maintenance.SAFETY
    assert drain(conn, NOW + maintenance.SAFETY) == ["papers", "resolve"]


def test_a_failed_pass_records_nothing_so_it_stays_owed(conn, monkeypatch):
    def broken(conn):
        raise RuntimeError("bad row")

    monkeypatch.setattr(maintenance.papers, "sync_paper_author_signals", broken)
    source = maintenance.Maintenance()
    unit = source.next_unit(conn, NOW)
    with pytest.raises(RuntimeError):
        source.fetch(conn, unit, Context(source.quota))

    assert source.next_unit(conn, NOW).key == "maintenance:papers"


def test_an_uninitialized_store_owes_both_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    c = schema.connect(tmp_path / "fresh.db")
    try:
        assert [u.key for u in maintenance.Maintenance().candidates(c, NOW)] == \
            ["maintenance:papers", "maintenance:resolve"]
    finally:
        c.close()
