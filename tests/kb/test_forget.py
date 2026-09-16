"""Singular removal: no dangling references, no invisible re-subscription, no preview writes."""
import json
import re
import sqlite3
from pathlib import Path

import pytest

from pipeline.kb import forget, oracle_refresh, oracle_refresh_state as state, oracles, schema
from pipeline.kb import raw_store, sitting_store
from tests.kb.test_export import _add

A = "x:1"
B = "x:10"


@pytest.fixture()
def conn(kb_home, fake_embedder):
    c = schema.connect()
    _add(c, fake_embedder, A, "x", "opinion", "x:user:1", [],
         "autonomous agents uniqueatomword", entry_mode="user-saved")
    _add(c, fake_embedder, B, "x", "opinion", "x:user:2", [], "crypto proof systems")
    yield c
    c.close()


def _sitting(conn, sid, ids, *, skipped=()):
    conn.execute(
        "INSERT INTO sittings (sitting_id,built_at,seed_kind,seed_ref,seed_atom_ids,"
        "floor,ceiling,budget_tokens,atoms,tokens,stop,skipped,skipped_dupes,read_at,region_key) "
        "VALUES (?, '2026-09-01', 'atoms', ?, ?, .3, .9, 1000, ?, ?, 'saturation', ?, ?, "
        "'2026-09-02', ?)",
        (sid, ','.join(ids), json.dumps(ids), len(ids), 10 * len(ids),
         json.dumps(list(skipped)), len(skipped), f"region-{sid}"))
    for rank, aid in enumerate(ids):
        conn.execute("INSERT INTO sitting_atoms (sitting_id,atom_id,rank,tokens) VALUES (?,?,?,10)",
                     (sid, aid, rank))
    conn.commit()
    sitting_store.record_lens_output(conn, sid, "connections", f"cached {ids}")


def test_removal_inventory_equals_atom_reference_tables_in_schema():
    ddl = re.sub(r"--[^\n]*", "", schema._DDL)
    tables = re.findall(r"CREATE (?:VIRTUAL )?TABLE IF NOT EXISTS (\w+)", ddl)
    with sqlite3.connect(":memory:") as c:
        c.executescript(schema._DDL)
        holders = {t for t in tables if any(
            re.search(r"(?:^|_)atom_ids?$", col[1]) for col in c.execute(f"PRAGMA table_info({t})"))}
    assert set(forget._ATOM_REFERENCES) == holders


def test_atom_preview_and_complete_removal(conn):
    _sitting(conn, "mixed", [A, B])
    _sitting(conn, "untouched", [B])
    _sitting(conn, "skipped", [B], skipped=[{"atom_id": A, "red": .99}])
    sitting_store.record_claims(conn, "mixed", [
        {"claim": "mixed evidence", "falsified_by": "experiment", "atom_ids": [A, B]},
        {"claim": "remaining evidence", "falsified_by": "experiment", "atom_ids": [B]}])
    conn.execute("INSERT INTO frontier_queries (query_id,text,normalized,generator,source_atom_ids,"
                 "created_at,last_emitted_at) VALUES ('q','agents','agents','sitting:mixed',?, 'a','b')",
                 (json.dumps([A, B]),))
    conn.commit()
    snapshot = raw_store.resolve_ref(conn.execute(
        "SELECT raw_ref FROM atoms WHERE atom_id=?", (A,)).fetchone()[0])
    before = list(conn.iterdump())
    preview = forget.atom(conn, A, shared=True)
    assert preview["status"] == "preview"
    assert preview["description"] == f"{A} card"
    assert preview["who_id"] == "x:user:1"
    assert preview["when_ts"] == "2024-05-01"
    assert preview["source_url"] == f"https://example/{A}"
    assert preview["references"]["sitting_claims"] == 1
    assert preview["sittings_affected"] == 2
    assert preview["lens_outputs"] == 2
    assert preview["snapshot"] and preview["shared"]
    assert list(conn.iterdump()) == before
    assert snapshot.exists()

    assert forget.atom(conn, A, confirm=True)["status"] == "forgotten"
    assert not snapshot.exists()
    for table, where in forget._ATOM_REFERENCES.items():
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                            {"atom_id": A}).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'uniqueatomword'"
                        ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_id=?", (B,)).fetchone()[0] == 1
    s = sitting_store.get_sitting(conn, "mixed")
    assert s["seed_atom_ids"] == [B] and s["seed_ref"] == B
    assert s["atoms"] == 1 and s["tokens"] == 10
    assert s["read_at"] == "2026-09-02" and s["region_key"] == "region-mixed"
    assert len(sitting_store.get_claims(conn, "mixed")) == 1
    q = conn.execute("SELECT * FROM frontier_queries WHERE query_id='q'").fetchone()
    assert json.loads(q["source_atom_ids"]) == [B] and q["status"] == "active"
    assert sitting_store.get_lens_output(conn, "mixed", "connections") is None
    assert sitting_store.get_lens_output(conn, "untouched", "connections") is not None
    skipped = sitting_store.get_sitting(conn, "skipped")
    assert json.loads(skipped["skipped"]) == [] and skipped["skipped_dupes"] == 0
    assert forget.atom(conn, A, confirm=True)["status"] == "not_found"


def test_unlink_failure_rolls_back_all_database_removal(conn, monkeypatch):
    _sitting(conn, "s", [A])
    before = list(conn.iterdump())

    def denied(*args, **kwargs):
        raise PermissionError("snapshot directory is read-only")

    monkeypatch.setattr(Path, "unlink", denied)
    with pytest.raises(PermissionError):
        forget.atom(conn, A, confirm=True)
    assert list(conn.iterdump()) == before


def test_invalid_snapshot_pointer_cannot_delete_other_files(conn, kb_home):
    outside = kb_home / "settings.json"
    outside.write_text("keep")
    conn.execute("UPDATE atoms SET raw_ref=? WHERE atom_id=?", (str(outside), A))
    conn.commit()
    with pytest.raises(ValueError):
        forget.atom(conn, A, confirm=True)
    assert outside.read_text() == "keep"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2


def test_missing_snapshot_does_not_strand_an_atom(conn):
    ref = conn.execute("SELECT raw_ref FROM atoms WHERE atom_id=?", (A,)).fetchone()[0]
    raw_store.resolve_ref(ref).unlink()
    assert forget.atom(conn, A, confirm=True)["status"] == "forgotten"


def test_materialized_frontier_record_survives(conn):
    conn.execute("UPDATE atoms SET entry_mode='frontier' WHERE atom_id=?", (A,))
    conn.execute("INSERT INTO frontier_candidates (candidate_id,source,kind,title,url,status,"
                 "first_seen_at,last_seen_at) VALUES ('c','arxiv','paper','a','https://example',"
                 "'materialized','a','a')")
    conn.commit()
    forget.atom(conn, A, confirm=True)
    assert conn.execute("SELECT status FROM frontier_candidates WHERE candidate_id='c'"
                        ).fetchone()[0] == "materialized"


def test_forgetting_oracle_stops_refresh_even_after_cluster_head_drift(conn, fake_embedder):
    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    schema.upsert_entity(conn, "blog:alice.test", name="Alice")
    schema.upsert_oracle(conn, "x:user:1", name="Alice")
    state.seed_from_entities(conn)
    conn.execute("UPDATE entities SET canonical_id='blog:alice.test'")
    conn.commit()
    schema.upsert_oracle(conn, "blog:alice.test", name="Alice")
    state.seed_from_entities(conn)
    before = list(conn.iterdump())
    preview = oracles.forget(conn, "@alice")
    assert preview["status"] == "preview" and preview["canonical_id"] == "blog:alice.test"
    assert preview["sources_removed"] >= 2
    assert preview["atoms_kept"] == 1
    assert list(conn.iterdump()) == before
    assert oracles.forget(conn, "@alice", confirm=True)["status"] == "forgotten"
    assert schema.list_oracles(conn) == []
    assert state.list_sources(conn) == []
    result = oracle_refresh.refresh_all(conn, fake_embedder)
    assert result["registered"] == 0 and result["refreshed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert schema.get_entity(conn, "x:user:1") is not None
    assert oracles.forget(conn, "@alice", confirm=True)["status"] == "not_found"


def test_ambiguous_oracle_requires_singular_choice_and_keeps_other_subscription(conn):
    for i in (1, 2):
        schema.upsert_entity(conn, f"x:user:{i}", name="Alex", profile={"handle": f"alex{i}"})
        schema.upsert_oracle(conn, f"x:user:{i}", name="Alex")
    state.seed_from_entities(conn)
    assert oracles.forget(conn, "Alex", confirm=True)["status"] == "ambiguous"
    assert len(schema.list_oracles(conn)) == 2
    assert oracles.forget(conn, "x:user:1", confirm=True)["status"] == "forgotten"
    assert [o["canonical_id"] for o in schema.list_oracles(conn)] == ["x:user:2"]
    assert {s.canonical_id for s in state.list_sources(conn)} == {"x:user:2"}


def test_oracle_without_refresh_registry_can_be_forgotten(conn):
    schema.upsert_entity(conn, "x:user:1", name="Alice")
    schema.upsert_oracle(conn, "x:user:1", name="Alice")
    assert oracles.forget(conn, "Alice", confirm=True)["status"] == "forgotten"


def test_oracle_removal_cannot_race_a_refresh_registration(conn):
    from pipeline.sync_lock import CatchupLock

    schema.upsert_entity(conn, "x:user:1", name="Alice", profile={"handle": "alice"})
    schema.upsert_oracle(conn, "x:user:1", name="Alice")
    with CatchupLock("oracle-refresh") as refresh:
        assert refresh.acquired
        assert oracles.forget(conn, "Alice")["status"] == "preview"
        assert oracles.forget(conn, "Alice", confirm=True)["status"] == "busy"
        state.seed_from_entities(conn)
        assert len(schema.list_oracles(conn)) == 1
        assert state.list_sources(conn)
    assert oracles.forget(conn, "Alice", confirm=True)["status"] == "forgotten"
    assert state.seed_from_entities(conn)["pairs"] == 0
    assert state.list_sources(conn) == []


def test_tool_is_singular_and_previews_before_removing(conn, monkeypatch):
    from mcp_server.forget_tools import register_forget_tools
    got = {}

    class MCP:
        def tool(self):
            def register(fn):
                got[fn.__name__] = fn
                return fn
            return register

    monkeypatch.setattr("pipeline.credentials.get_credential", lambda service: None)
    register_forget_tools(MCP())
    tool = got["forget"]
    assert tool()["status"] == "error"
    assert tool(atom_id=A, oracle="Alice", confirm=True)["status"] == "error"
    assert tool(atom_id=" ")["status"] == "error"
    assert tool(atom_id=A)["status"] == "preview"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 2
    assert tool(atom_id=A, confirm=True)["status"] == "forgotten"
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
